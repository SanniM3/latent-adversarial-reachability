"""Paths, defaults and CLI parsing shared by every script."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = field(default_factory=pick_device)

    # Layers to study, as fractions of model depth; resolved once the model is loaded.
    layer_fractions: tuple[float, ...] = (0.5, 0.75)
    layers: tuple[int, ...] = ()

    n_harmful: int = 10
    n_harmless: int = 5
    n_paraphrases: int = 5
    seed: int = 0

    # Attack budgets, in units of the per-prompt natural paraphrase displacement.
    budgets: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    pgd_steps: int = 100
    pgd_lr_frac: float = 0.1  # Adam lr = pgd_lr_frac * eps
    n_random_controls: int = 3

    gen_tokens: int = 48

    @property
    def paraphrase_file(self) -> Path | None:
        return self._paraphrase_file

    _paraphrase_file: Path | None = None


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    d = Config()
    parser.add_argument("--model", default=d.model_name, help="HuggingFace model id")
    parser.add_argument("--device", default=d.device, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--n-harmful", type=int, default=d.n_harmful)
    parser.add_argument("--n-harmless", type=int, default=d.n_harmless)
    parser.add_argument("--n-paraphrases", type=int, default=d.n_paraphrases)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="explicit layer indices; default is 50%% and 75%% of depth",
    )
    parser.add_argument("--budgets", type=float, nargs="*", default=list(d.budgets))
    parser.add_argument("--pgd-steps", type=int, default=d.pgd_steps)
    parser.add_argument("--gen-tokens", type=int, default=d.gen_tokens)
    parser.add_argument(
        "--paraphrase-file",
        type=Path,
        default=None,
        help="JSON {prompt_id: [paraphrase, ...]} overriding the template paraphrases",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke run: 2 harmful prompts, 1 layer, 3 budgets, 20 PGD steps",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(
        model_name=args.model,
        device=args.device,
        seed=args.seed,
        n_harmful=args.n_harmful,
        n_harmless=args.n_harmless,
        n_paraphrases=args.n_paraphrases,
        budgets=tuple(args.budgets),
        pgd_steps=args.pgd_steps,
        gen_tokens=args.gen_tokens,
    )
    cfg._paraphrase_file = args.paraphrase_file
    if args.layers:
        cfg.layers = tuple(args.layers)
    if getattr(args, "quick", False):
        cfg.n_harmful = 2
        cfg.n_harmless = 2
        cfg.n_paraphrases = 3
        cfg.layer_fractions = (0.5,)
        cfg.budgets = (1.0, 4.0)
        cfg.pgd_steps = 20
        cfg.n_random_controls = 2
        cfg.gen_tokens = 24
        cfg.layers = ()
    return cfg


def resolve_layers(cfg: Config, n_layers: int) -> tuple[int, ...]:
    """Turn layer fractions into concrete indices, unless indices were given."""
    if cfg.layers:
        return cfg.layers
    return tuple(sorted({int(f * n_layers) for f in cfg.layer_fractions}))
