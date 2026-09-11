"""Figures from the saved results.

    python -m src.figures
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import FIGURES_DIR, RESULTS_DIR

METHOD_STYLE = {
    "adversarial": ("tab:red", "o", "optimised (adversarial)"),
    "paraphrase_dir": ("tab:blue", "s", "paraphrase direction"),
    "random": ("tab:grey", "^", "random direction"),
}


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key in {"prompt_id", "kind", "method", "variant", "delta_key"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def fig_success_vs_budget(attacks: list[dict]) -> None:
    """Headline figure: behaviour (top) and the judge-free loss (bottom) vs budget."""
    layers = sorted({int(r["layer"]) for r in attacks})
    fig, axes = plt.subplots(
        2, len(layers), figsize=(5.2 * len(layers), 7.4), sharex=True, squeeze=False
    )

    for col, layer in enumerate(layers):
        top, bottom = axes[0][col], axes[1][col]
        for method, (colour, marker, label) in METHOD_STYLE.items():
            subset = [r for r in attacks if int(r["layer"]) == layer and r["method"] == method]
            budgets = sorted({r["budget_c"] for r in subset})
            rates, losses = [], []
            for c in budgets:
                at_c = [r for r in subset if r["budget_c"] == c]
                per_prompt: dict[str, float] = defaultdict(float)
                for r in at_c:
                    per_prompt[r["prompt_id"]] = max(per_prompt[r["prompt_id"]], r["success"])
                rates.append(100 * sum(per_prompt.values()) / len(per_prompt))
                losses.append(float(np.mean([r["target_loss"] for r in at_c])))
            top.plot(budgets, rates, marker=marker, color=colour, label=label, lw=2)
            bottom.plot(budgets, losses, marker=marker, color=colour, label=label, lw=2)

        for ax in (top, bottom):
            ax.axvline(1.0, color="black", ls=":", lw=1)
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.3)
        top.text(1.05, 6, "typical paraphrase shift", rotation=90, fontsize=8, va="bottom")
        top.set_title(f"layer {layer}")
        top.set_ylim(-4, 104)
        bottom.set_xlabel("perturbation budget  ‖δ‖ / natural paraphrase shift")

    axes[0][0].set_ylabel("prompts jailbroken (%)")
    axes[1][0].set_ylabel("cross-entropy of compliance prefix")
    axes[0][-1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "At the same displacement, only the optimised direction changes behaviour",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "attack_success_vs_budget.png", dpi=180)
    plt.close(fig)


def fig_distance_context(attacks: list[dict], natural: list[dict]) -> None:
    """Where successful attacks sit relative to paraphrase and between-prompt distance."""
    layers = sorted({int(r["layer"]) for r in attacks})
    fig, axes = plt.subplots(1, len(layers), figsize=(5.2 * len(layers), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, layer in zip(axes, layers):
        nat = {r["prompt_id"]: r for r in natural if int(r["layer"]) == layer}
        successes = [
            r
            for r in attacks
            if int(r["layer"]) == layer and r["method"] == "adversarial" and r["success"]
        ]
        smallest: dict[str, dict] = {}
        for r in successes:
            if r["prompt_id"] not in smallest or r["budget_c"] < smallest[r["prompt_id"]]["budget_c"]:
                smallest[r["prompt_id"]] = r

        ids = sorted(smallest)
        x = np.arange(len(ids))
        ax.bar(
            x,
            [smallest[i]["delta_norm"] for i in ids],
            color="tab:red",
            alpha=0.75,
            label="smallest successful ‖δ‖",
        )
        ax.scatter(
            x,
            [nat[i]["natural_scale"] for i in ids],
            color="tab:blue",
            zorder=3,
            label="mean paraphrase shift",
        )
        between = np.mean([nat[i]["between_prompt_distance_mean"] for i in ids])
        ax.axhline(between, color="black", ls="--", lw=1, label="distance to another prompt")
        ax.set_xticks(x)
        ax.set_xticklabels([i.replace("harmful_", "") for i in ids], fontsize=8)
        ax.set_xlabel("prompt")
        ax.set_title(f"layer {layer}")
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_ylabel("L2 distance in residual stream")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Size of a successful attack against natural distances", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distance_context.png", dpi=180)
    plt.close(fig)


def fig_direction_geometry(attacks: list[dict], natural: list[dict]) -> None:
    """Is the attack direction one that rewording explores? Does it touch refusal?"""
    layers = sorted({int(r["layer"]) for r in attacks})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    width = 0.35
    x = np.arange(len(layers))
    for offset, (key, colour, label) in [
        (-width / 2, ("subspace_frac", "tab:red", "optimised δ")),
        (width / 2, ("chance_subspace_frac", "tab:grey", "random direction (chance)")),
    ]:
        values = []
        for layer in layers:
            subset = [
                r
                for r in attacks
                if int(r["layer"]) == layer and r["method"] == "adversarial" and r["success"]
            ]
            values.append(np.mean([r[key] for r in subset]) if subset else np.nan)
        axes[0].bar(x + offset, values, width, color=colour, label=label)
    axes[0].axhline(1.0, color="tab:blue", ls="--", lw=1, label="paraphrase direction (=1)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"layer {l}" for l in layers])
    axes[0].set_ylabel("fraction of ‖δ‖ inside the paraphrase subspace")
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Does the attack move where rewording moves?")
    axes[0].grid(alpha=0.3, axis="y")

    adv, nat_vals = [], []
    for layer in layers:
        subset = [
            r
            for r in attacks
            if int(r["layer"]) == layer and r["method"] == "adversarial" and r["success"]
        ]
        adv.append(np.mean([abs(r.get("cos_delta_refusal_dir", np.nan)) for r in subset]) if subset else np.nan)
        nat_subset = [
            r for r in natural if int(r["layer"]) == layer and r["kind"] == "harmful"
        ]
        nat_vals.append(np.mean([r.get("natural_cos_refusal_dir", np.nan) for r in nat_subset]))
    axes[1].bar(x - width / 2, adv, width, color="tab:red", label="optimised δ")
    axes[1].bar(x + width / 2, nat_vals, width, color="tab:blue", label="paraphrase displacement")
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    axes[1].axhline(
        1 / np.sqrt(summary["d_model"]),
        color="black",
        ls=":",
        lw=1,
        label="chance (random direction)",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"layer {l}" for l in layers])
    axes[1].set_ylabel("|cos(displacement, refusal direction)|")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Does it push on the refusal feature?")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "direction_geometry.png", dpi=180)
    plt.close(fig)


def fig_pca(attacks: list[dict], prompt_id: str | None = None) -> None:
    acts = torch.load(RESULTS_DIR / "activations.pt", weights_only=False)
    deltas = torch.load(RESULTS_DIR / "deltas.pt", weights_only=False)
    layers = acts["layers"]
    layer = layers[0]

    if prompt_id is None:
        successful = [
            r for r in attacks
            if r["method"] == "adversarial" and r["success"] and int(r["layer"]) == layer
        ]
        prompt_id = successful[0]["prompt_id"] if successful else "harmful_01"

    record = acts["prompts"][prompt_id]
    h = record["orig"][layer].float()
    paras = record["paras"][layer].float()
    adv_rows = sorted(
        (
            r
            for r in attacks
            if r["prompt_id"] == prompt_id
            and int(r["layer"]) == layer
            and r["method"] == "adversarial"
        ),
        key=lambda r: r["budget_c"],
    )
    adv = torch.stack([h + deltas[r["delta_key"]].float() for r in adv_rows])

    points = torch.cat([h.unsqueeze(0), paras, adv])
    centred = points - points.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(centred, full_matrices=False)
    proj = (centred @ v[:2].T).numpy()

    n_para = paras.shape[0]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(*proj[0], s=160, marker="*", color="black", label="original prompt", zorder=4)
    ax.scatter(
        proj[1 : 1 + n_para, 0],
        proj[1 : 1 + n_para, 1],
        s=60,
        color="tab:blue",
        label="paraphrases (reachable)",
    )
    ax.scatter(
        proj[1 + n_para :, 0],
        proj[1 + n_para :, 1],
        s=60,
        color="tab:red",
        label="adversarial h + δ",
    )
    for point, row in zip(proj[1 + n_para :], adv_rows):
        if row["budget_c"] < 0.5:
            continue
        marker = "✓" if row["success"] else "✗"
        ax.annotate(
            f"{marker} {row['budget_c']:g}×",
            point,
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="tab:red",
        )

    natural = float((paras - h).norm(dim=-1).mean())
    ax.add_patch(
        plt.Circle(
            proj[0],
            natural,
            fill=False,
            ls="--",
            lw=1,
            color="tab:blue",
            label="mean paraphrase distance",
        )
    )
    ax.set_aspect("equal")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(
        f"{prompt_id}, layer {layer}\n"
        "2D projection only; distances shrink under projection"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_example.png", dpi=180)
    plt.close(fig)


def main_from_dirs(
    results: Path, figures_dir: Path, pca_prompt: str | None = None
) -> None:
    global RESULTS_DIR, FIGURES_DIR  # rebound so the figure helpers stay argument-light

    RESULTS_DIR, FIGURES_DIR = Path(results), Path(figures_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    attacks = load_csv(RESULTS_DIR / "attacks.csv")
    natural = load_csv(RESULTS_DIR / "natural.csv")
    fig_success_vs_budget(attacks)
    fig_distance_context(attacks, natural)
    fig_direction_geometry(attacks, natural)
    fig_pca(attacks, pca_prompt)
    print(f"figures written to {FIGURES_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    parser.add_argument("--pca-prompt", default=None, help="prompt_id for the PCA figure")
    args = parser.parse_args()
    main_from_dirs(args.results, args.figures, args.pca_prompt)


if __name__ == "__main__":
    main()
