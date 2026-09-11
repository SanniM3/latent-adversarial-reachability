"""Geometry of the residual stream: distances, subspaces and the refusal direction."""

from __future__ import annotations

import math

import torch


def l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))


def natural_scale(h_orig: torch.Tensor, h_paras: torch.Tensor) -> float:
    """Mean L2 displacement from the original to its paraphrases.

    This is the unit every perturbation budget is expressed in.
    """
    return float((h_paras - h_orig).norm(dim=-1).mean())


def orthonormal_basis(displacements: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis (d, k) of the span of the rows of `displacements`."""
    matrix = displacements.T.float()  # (d, k)
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    keep = s > tol * s.max()
    return u[:, keep]


def subspace_fraction(delta: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of `delta`'s norm lying inside the span of `basis`."""
    norm = delta.norm()
    if norm == 0:
        return float("nan")
    return float((basis.T @ delta.float()).norm() / norm)


def chance_subspace_fraction(k: int, d: int) -> float:
    """Expected `subspace_fraction` for a direction drawn uniformly at random."""
    return math.sqrt(k / d)


def refusal_direction(
    harmful: torch.Tensor, harmless: torch.Tensor
) -> torch.Tensor:
    """Normalised difference in means between harmful and harmless representations.

    The standard 'refusal direction' estimator (Arditi et al., 2024). Positive cosine
    with this direction means 'more refusal-like'.
    """
    direction = harmful.float().mean(dim=0) - harmless.float().mean(dim=0)
    return direction / direction.norm()
