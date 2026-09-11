"""Latent adversarial attack: PGD on a single residual-stream vector.

The model is frozen. The only optimised parameter is `delta`, added to the residual
stream at one layer and one token position (the last prompt token). The objective is
the teacher-forced cross-entropy of a compliance prefix such as
"Sure, here is a ...".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

from src.model import make_delta_hook, resid_hook_name


@dataclass
class AttackResult:
    delta: torch.Tensor
    init_loss: float
    best_loss: float
    loss_curve: list[float]

    @property
    def norm(self) -> float:
        return float(self.delta.norm())


def target_loss(
    model: HookedTransformer,
    prompt_toks: torch.Tensor,
    tgt_toks: torch.Tensor,
    layer: int,
    delta: torch.Tensor | None,
) -> torch.Tensor:
    """Mean cross-entropy of the target tokens under teacher forcing."""
    prompt_len = prompt_toks.shape[1]
    n_target = tgt_toks.shape[1]
    full = torch.cat([prompt_toks, tgt_toks], dim=1)

    hooks = []
    if delta is not None:
        hooks.append((resid_hook_name(layer), make_delta_hook(delta, prompt_len)))
    logits = model.run_with_hooks(full, fwd_hooks=hooks, return_type="logits")

    # Position prompt_len - 1 predicts the first target token.
    pred = logits[0, prompt_len - 1 : prompt_len - 1 + n_target]
    logprobs = pred.log_softmax(dim=-1)
    chosen = logprobs.gather(-1, tgt_toks[0].unsqueeze(-1)).squeeze(-1)
    return -chosen.mean()


def pgd_attack(
    model: HookedTransformer,
    prompt_toks: torch.Tensor,
    tgt_toks: torch.Tensor,
    layer: int,
    eps: float,
    steps: int,
    lr_frac: float = 0.1,
) -> AttackResult:
    """Projected gradient descent on delta inside the L2 ball of radius `eps`."""
    device = prompt_toks.device
    delta = torch.zeros(model.cfg.d_model, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([delta], lr=max(eps * lr_frac, 1e-6))

    best_delta = delta.detach().clone()
    best_loss = float("inf")
    init_loss = None
    curve: list[float] = []

    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss = target_loss(model, prompt_toks, tgt_toks, layer, delta)
        value = loss.detach().item()
        curve.append(value)
        if init_loss is None:
            init_loss = value
        if value < best_loss:
            best_loss = value
            best_delta = delta.detach().clone()

        loss.backward()
        optimiser.step()
        with torch.no_grad():
            norm = delta.norm()
            if norm > eps:
                delta.mul_(eps / norm)

    with torch.no_grad():
        final_loss = float(target_loss(model, prompt_toks, tgt_toks, layer, delta))
    curve.append(final_loss)
    if final_loss < best_loss:
        best_loss = final_loss
        best_delta = delta.detach().clone()

    return AttackResult(
        delta=best_delta,
        init_loss=float(init_loss if init_loss is not None else final_loss),
        best_loss=best_loss,
        loss_curve=curve,
    )


def random_deltas(
    d_model: int, eps: float, n: int, generator: torch.Generator, device: str
) -> list[torch.Tensor]:
    """`n` perturbations of norm exactly `eps`, uniform on the sphere."""
    raw = torch.randn(n, d_model, generator=generator)
    unit = raw / raw.norm(dim=-1, keepdim=True)
    return [(eps * u).to(device) for u in unit]


def paraphrase_deltas(
    h_orig: torch.Tensor, h_paras: torch.Tensor, eps: float, device: str
) -> list[torch.Tensor]:
    """Displacements toward each paraphrase, rescaled to norm `eps`.

    These directions are reachable by construction: each one points from the original
    representation to one a real token sequence actually produced.
    """
    out = []
    for h in h_paras:
        direction = h - h_orig
        norm = direction.norm()
        if norm > 0:
            out.append((eps * direction / norm).to(device))
    return out
