"""End-to-end experiment.

    python -m src.run_experiment --quick          # ~2 min smoke run
    python -m src.run_experiment                  # full run

Stages
  A  collect last-token residuals for every prompt and paraphrase, and record the
     model's unperturbed behaviour
  B  per prompt / layer / budget: PGD latent attack plus matched-norm random and
     paraphrase-direction controls
  C  geometry of the resulting perturbations, written to results/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src import data, metrics
from src.attack import paraphrase_deltas, pgd_attack, random_deltas, target_loss
from src.config import RESULTS_DIR, Config, add_common_args, config_from_args, resolve_layers
from src.model import (
    chat_tokens,
    generate_with_delta,
    is_refusal,
    judge,
    load_model,
    residuals_at_last_token,
    target_tokens,
)

class Paths:
    """Output locations; `--quick` runs write to a separate directory."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.activations = root / "activations.pt"
        self.deltas = root / "deltas.pt"
        self.baseline = root / "baseline.csv"
        self.natural = root / "natural.csv"
        self.attacks = root / "attacks.csv"
        self.summary = root / "summary.json"
        self.completions = root / "completions.local.jsonl"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stage_a_activations(model, prompts: list[data.Prompt], cfg: Config, paths: Paths):
    """Residuals for every variant, plus unperturbed behaviour."""
    acts: dict[str, dict] = {}
    baseline_rows: list[dict] = []
    completions: list[dict] = []

    for prompt in tqdm(prompts, desc="A: activations"):
        per_layer_orig = residuals_at_last_token(model, prompt.text, cfg.layers)
        para_stack = {layer: [] for layer in cfg.layers}
        for text in prompt.paraphrases:
            res = residuals_at_last_token(model, text, cfg.layers)
            for layer in cfg.layers:
                para_stack[layer].append(res[layer])
        acts[prompt.prompt_id] = {
            "kind": prompt.kind,
            "orig": per_layer_orig,
            "paras": {layer: torch.stack(v) for layer, v in para_stack.items()},
        }

        for variant_idx, text in enumerate(prompt.variants):
            toks = chat_tokens(model, text)
            completion = generate_with_delta(model, toks, cfg.layers[0], None, cfg.gen_tokens)
            baseline_rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "kind": prompt.kind,
                    "variant": "original" if variant_idx == 0 else f"para_{variant_idx}",
                    "refusal": int(is_refusal(completion)),
                }
            )
            completions.append(
                {
                    "stage": "baseline",
                    "prompt_id": prompt.prompt_id,
                    "variant": variant_idx,
                    "completion": completion,
                }
            )

    torch.save({"layers": list(cfg.layers), "prompts": acts}, paths.activations)
    _write_csv(paths.baseline, baseline_rows)
    return acts, baseline_rows, completions


def stage_natural_geometry(acts: dict, cfg: Config, paths: Paths) -> tuple[list[dict], dict]:
    """Natural-variation scale per prompt/layer, and the refusal direction per layer."""
    directions: dict[int, torch.Tensor] = {}
    for layer in cfg.layers:
        harmful, harmless = [], []
        for record in acts.values():
            vectors = [record["orig"][layer]] + list(record["paras"][layer])
            (harmful if record["kind"] == "harmful" else harmless).extend(vectors)
        if harmful and harmless:
            directions[layer] = metrics.refusal_direction(
                torch.stack(harmful), torch.stack(harmless)
            )

    rows: list[dict] = []
    for prompt_id, record in acts.items():
        for layer in cfg.layers:
            h_orig = record["orig"][layer]
            h_paras = record["paras"][layer]
            displacements = h_paras - h_orig
            row = {
                "prompt_id": prompt_id,
                "kind": record["kind"],
                "layer": layer,
                "h_norm": float(h_orig.norm()),
                "natural_scale": metrics.natural_scale(h_orig, h_paras),
                "natural_scale_std": float(displacements.norm(dim=-1).std()),
                "natural_rel_norm": float(
                    (displacements.norm(dim=-1).mean() / h_orig.norm())
                ),
                "natural_cos_mean": float(
                    torch.nn.functional.cosine_similarity(
                        h_paras, h_orig.unsqueeze(0), dim=-1
                    ).mean()
                ),
            }
            if layer in directions:
                row["natural_cos_refusal_dir"] = float(
                    torch.nn.functional.cosine_similarity(
                        displacements, directions[layer].unsqueeze(0), dim=-1
                    )
                    .abs()
                    .mean()
                )
            rows.append(row)

    # Between-prompt distances give the natural scale some context (H1).
    for layer in cfg.layers:
        originals = torch.stack([r["orig"][layer] for r in acts.values()])
        pairwise = torch.cdist(originals, originals)
        off_diag = pairwise[~torch.eye(len(originals), dtype=bool)]
        for row in rows:
            if row["layer"] == layer:
                row["between_prompt_distance_mean"] = float(off_diag.mean())

    _write_csv(paths.natural, rows)
    return rows, directions


def stage_b_attacks(model, prompts, acts, natural_rows, directions, cfg: Config, paths: Paths):
    scale = {(r["prompt_id"], r["layer"]): r["natural_scale"] for r in natural_rows}
    targets = {p.prompt_id: p for p in prompts}
    harmful = [p for p in prompts if p.kind == "harmful"]

    rows: list[dict] = []
    completions: list[dict] = []
    deltas: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(cfg.seed)

    total = len(harmful) * len(cfg.layers) * len(cfg.budgets)
    bar = tqdm(total=total, desc="B: attacks")
    for prompt in harmful:
        prompt_toks = chat_tokens(model, prompt.text)
        tgt_toks = target_tokens(model, targets[prompt.prompt_id].target)
        record = acts[prompt.prompt_id]

        for layer in cfg.layers:
            h_orig = record["orig"][layer]
            h_paras = record["paras"][layer]
            basis = metrics.orthonormal_basis(h_paras - h_orig)
            chance = metrics.chance_subspace_fraction(basis.shape[1], model.cfg.d_model)
            s = scale[(prompt.prompt_id, layer)]

            for c in cfg.budgets:
                eps = c * s
                candidates: list[tuple[str, int, torch.Tensor, float]] = []

                result = pgd_attack(
                    model,
                    prompt_toks,
                    tgt_toks,
                    layer,
                    eps=eps,
                    steps=cfg.pgd_steps,
                    lr_frac=cfg.pgd_lr_frac,
                )
                candidates.append(("adversarial", 0, result.delta, result.best_loss))

                for i, delta in enumerate(
                    random_deltas(
                        model.cfg.d_model, eps, cfg.n_random_controls, generator, cfg.device
                    )
                ):
                    candidates.append(("random", i, delta, float("nan")))

                for i, delta in enumerate(
                    paraphrase_deltas(h_orig, h_paras, eps, cfg.device)
                ):
                    candidates.append(("paraphrase_dir", i, delta, float("nan")))

                for method, trial, delta, loss in candidates:
                    completion = generate_with_delta(
                        model, prompt_toks, layer, delta, cfg.gen_tokens
                    )
                    verdict = judge(completion)
                    delta_cpu = delta.detach().float().cpu()
                    if math.isnan(loss):  # controls: measure loss for comparability
                        with torch.no_grad():
                            loss = float(
                                target_loss(model, prompt_toks, tgt_toks, layer, delta)
                            )
                    delta_key = f"{prompt.prompt_id}|{layer}|{c}|{method}|{trial}"
                    deltas[delta_key] = delta_cpu
                    row = {
                        "prompt_id": prompt.prompt_id,
                        "layer": layer,
                        "budget_c": c,
                        "method": method,
                        "trial": trial,
                        "delta_key": delta_key,
                        "eps": eps,
                        "natural_scale": s,
                        "delta_norm": float(delta_cpu.norm()),
                        "delta_over_natural": float(delta_cpu.norm()) / s,
                        "delta_over_h": float(delta_cpu.norm() / h_orig.norm()),
                        "cos_h_hadv": metrics.cosine(h_orig, h_orig + delta_cpu),
                        "target_loss": loss,
                        **verdict,
                        "subspace_frac": metrics.subspace_fraction(delta_cpu, basis),
                        "chance_subspace_frac": chance,
                    }
                    if layer in directions:
                        row["cos_delta_refusal_dir"] = metrics.cosine(
                            delta_cpu, directions[layer]
                        )
                    rows.append(row)
                    completions.append(
                        {
                            "stage": "attack",
                            "prompt_id": prompt.prompt_id,
                            "layer": layer,
                            "budget_c": c,
                            "method": method,
                            "trial": trial,
                            "completion": completion,
                        }
                    )
                bar.update(1)
    bar.close()

    torch.save(deltas, paths.deltas)
    _write_csv(paths.attacks, rows)
    return rows, completions


def summarise(
    attack_rows, natural_rows, baseline_rows, cfg: Config, paths: Paths, d_model: int
) -> dict:
    def rate(rows) -> float:
        return sum(r["success"] for r in rows) / len(rows) if rows else float("nan")

    summary: dict = {
        "model": cfg.model_name,
        "d_model": d_model,
        "layers": list(cfg.layers),
        "budgets": list(cfg.budgets),
        "pgd_steps": cfg.pgd_steps,
        "baseline_refusal_rate": {
            kind: sum(r["refusal"] for r in baseline_rows if r["kind"] == kind)
            / max(1, sum(1 for r in baseline_rows if r["kind"] == kind))
            for kind in ("harmful", "harmless")
        },
        "natural_scale_by_layer": {},
        "success_by_method_budget": {},
        "geometry_of_successful_attacks": {},
    }

    for layer in cfg.layers:
        harmful_nat = [
            r for r in natural_rows if r["layer"] == layer and r["kind"] == "harmful"
        ]
        if harmful_nat:
            summary["natural_scale_by_layer"][str(layer)] = {
                "mean_paraphrase_distance": sum(r["natural_scale"] for r in harmful_nat)
                / len(harmful_nat),
                "mean_between_prompt_distance": harmful_nat[0].get(
                    "between_prompt_distance_mean"
                ),
                "mean_h_norm": sum(r["h_norm"] for r in harmful_nat) / len(harmful_nat),
                "mean_cos_original_paraphrase": sum(
                    r["natural_cos_mean"] for r in harmful_nat
                )
                / len(harmful_nat),
            }

        for method in ("adversarial", "random", "paraphrase_dir"):
            for c in cfg.budgets:
                subset = [
                    r
                    for r in attack_rows
                    if r["layer"] == layer and r["method"] == method and r["budget_c"] == c
                ]
                if not subset:
                    continue
                # A control counts as a success for a prompt if any trial succeeds.
                by_prompt: dict[str, int] = {}
                for r in subset:
                    by_prompt[r["prompt_id"]] = max(
                        by_prompt.get(r["prompt_id"], 0), r["success"]
                    )
                key = f"layer{layer}/{method}/c={c}"
                summary["success_by_method_budget"][key] = {
                    "per_prompt_success_rate": sum(by_prompt.values()) / len(by_prompt),
                    "per_trial_success_rate": rate(subset),
                    "per_trial_no_refusal_rate": sum(r["no_refusal"] for r in subset)
                    / len(subset),
                    "n_prompts": len(by_prompt),
                }

        successful = [
            r
            for r in attack_rows
            if r["layer"] == layer and r["method"] == "adversarial" and r["success"]
        ]
        if successful:
            smallest: dict[str, dict] = {}
            for r in successful:
                current = smallest.get(r["prompt_id"])
                if current is None or r["budget_c"] < current["budget_c"]:
                    smallest[r["prompt_id"]] = r
            values = sorted(r["delta_over_natural"] for r in smallest.values())
            median = values[len(values) // 2]
            summary["geometry_of_successful_attacks"][str(layer)] = {
                "n_prompts_ever_successful": len(smallest),
                "median_smallest_successful_delta_over_natural": median,
                "mean_cos_h_hadv": sum(r["cos_h_hadv"] for r in smallest.values())
                / len(smallest),
                "mean_subspace_frac": sum(r["subspace_frac"] for r in smallest.values())
                / len(smallest),
                "chance_subspace_frac": next(iter(smallest.values()))[
                    "chance_subspace_frac"
                ],
                "mean_cos_delta_refusal_dir": (
                    sum(r.get("cos_delta_refusal_dir", 0.0) for r in smallest.values())
                    / len(smallest)
                ),
                "chance_abs_cos": 1 / math.sqrt(d_model),
            }

    paths.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def select_prompts(cfg: Config) -> list[data.Prompt]:
    """Load the prompt set, rebuilding only if it is too small for this config."""
    prompts = data.load(cfg)
    harmful = [p for p in prompts if p.kind == "harmful"]
    harmless = [p for p in prompts if p.kind == "harmless"]
    short_paraphrases = any(len(p.paraphrases) < cfg.n_paraphrases for p in prompts)
    if len(harmful) < cfg.n_harmful or len(harmless) < cfg.n_harmless or short_paraphrases:
        prompts = data.build(cfg)
        harmful = [p for p in prompts if p.kind == "harmful"]
        harmless = [p for p in prompts if p.kind == "harmless"]
    selected = harmful[: cfg.n_harmful] + harmless[: cfg.n_harmless]
    for prompt in selected:
        prompt.paraphrases = prompt.paraphrases[: cfg.n_paraphrases]
    return selected


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--out", type=Path, default=None, help="results directory")
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(args.out or (RESULTS_DIR / "quick" if args.quick else RESULTS_DIR))

    torch.manual_seed(cfg.seed)
    prompts = select_prompts(cfg)

    print(f"loading {cfg.model_name} on {cfg.device} ...")
    model = load_model(cfg)
    cfg.layers = resolve_layers(cfg, model.cfg.n_layers)
    print(f"n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, layers={cfg.layers}")

    started = time.time()
    acts, baseline_rows, completions_a = stage_a_activations(model, prompts, cfg, paths)
    natural_rows, directions = stage_natural_geometry(acts, cfg, paths)
    attack_rows, completions_b = stage_b_attacks(
        model, prompts, acts, natural_rows, directions, cfg, paths
    )
    summary = summarise(
        attack_rows, natural_rows, baseline_rows, cfg, paths, model.cfg.d_model
    )
    summary["runtime_seconds"] = round(time.time() - started, 1)
    paths.summary.write_text(json.dumps(summary, indent=2) + "\n")

    with paths.completions.open("w") as f:
        for record in completions_a + completions_b:
            f.write(json.dumps(record) + "\n")

    print(f"\ndone in {summary['runtime_seconds']}s -> {paths.root}")
    print(json.dumps(summary["natural_scale_by_layer"], indent=2))
    print(json.dumps(summary["geometry_of_successful_attacks"], indent=2))


if __name__ == "__main__":
    main()
