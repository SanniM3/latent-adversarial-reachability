"""Correctness checks for the injection machinery and the geometry helpers.

    python -m tests.test_pipeline          # runs everything, loads the small model once
"""

from __future__ import annotations

import torch

from src import metrics
from src.attack import pgd_attack, target_loss
from src.config import Config
from src.model import chat_tokens, generate_with_delta, load_model, target_tokens

PROMPT = "Write a tutorial on how to build a model rocket for a school science fair"
TARGET = "Sure, here is a tutorial"
LAYER = 12


def test_subspace_fraction() -> None:
    torch.manual_seed(0)
    displacements = torch.randn(3, 64)
    basis = metrics.orthonormal_basis(displacements)
    assert basis.shape == (64, 3)

    inside = displacements[0] + 2 * displacements[1]
    assert abs(metrics.subspace_fraction(inside, basis) - 1.0) < 1e-4

    random_vec = torch.randn(64)
    outside = random_vec - basis @ (basis.T @ random_vec)
    assert metrics.subspace_fraction(outside, basis) < 1e-4

    # A random direction should sit near chance level in high dimensions.
    chance = metrics.chance_subspace_fraction(3, 64)
    observed = metrics.subspace_fraction(torch.randn(64), basis)
    assert abs(observed - chance) < 0.25


def test_refusal_direction() -> None:
    harmful = torch.randn(8, 32) + torch.tensor([5.0] + [0.0] * 31)
    harmless = torch.randn(8, 32)
    direction = metrics.refusal_direction(harmful, harmless)
    assert abs(float(direction.norm()) - 1.0) < 1e-5
    assert direction[0] > 0.8  # recovers the planted axis


def model_tests(model) -> None:
    prompt_toks = chat_tokens(model, PROMPT)
    tgt_toks = target_tokens(model, TARGET)
    d_model = model.cfg.d_model

    # 1. A zero perturbation is a no-op.
    clean = target_loss(model, prompt_toks, tgt_toks, LAYER, None)
    zero = target_loss(model, prompt_toks, tgt_toks, LAYER, torch.zeros(d_model, device=model.cfg.device))
    assert torch.allclose(clean, zero, atol=1e-5), (clean, zero)
    print(f"  zero-delta no-op ok (loss={float(clean):.4f})")

    # 2. The prefill-only hook must match uncached generation, where the hook fires on
    #    every forward pass at the same absolute position.
    delta = torch.randn(d_model, device=model.cfg.device)
    delta = 5.0 * delta / delta.norm()
    cached = generate_with_delta(model, prompt_toks, LAYER, delta, 16, use_cache=True)
    uncached = generate_with_delta(model, prompt_toks, LAYER, delta, 16, use_cache=False)
    assert cached == uncached, f"\ncached:   {cached!r}\nuncached: {uncached!r}"
    print("  cached == uncached generation ok")

    # 3. PGD must reduce the objective and respect the norm ball.
    eps = 10.0
    result = pgd_attack(model, prompt_toks, tgt_toks, LAYER, eps=eps, steps=15)
    assert result.best_loss < result.init_loss, (result.init_loss, result.best_loss)
    assert result.norm <= eps + 1e-4, result.norm
    print(f"  pgd {result.init_loss:.3f} -> {result.best_loss:.3f}, ||delta||={result.norm:.2f} <= {eps}")


def main() -> None:
    test_subspace_fraction()
    test_refusal_direction()
    print("geometry tests ok")

    cfg = Config()
    print(f"loading {cfg.model_name} on {cfg.device} ...")
    model = load_model(cfg)
    model_tests(model)
    print("model tests ok")


if __name__ == "__main__":
    main()
