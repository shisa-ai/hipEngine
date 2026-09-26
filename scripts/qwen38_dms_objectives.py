"""Frozen L0-L3 DMS training objectives (training-only, not runtime code)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def train_only_log_mass_stats(mass: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.log1p(torch.clamp(mass, min=0.0))
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(float(eps))
    return mean.detach(), std.detach()


def normalize_log_mass(mass: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (torch.log1p(torch.clamp(mass, min=0.0)) - mean) / std


def importance_weights(mass: torch.Tensor, p95_positive: torch.Tensor) -> torch.Tensor:
    denominator = torch.where(p95_positive > 0, p95_positive, torch.ones_like(p95_positive))
    weights = 1.0 + torch.minimum(torch.clamp(mass, min=0.0) / denominator, torch.full_like(mass, 4.0))
    return torch.where(p95_positive > 0, weights, torch.ones_like(weights))


def sample_same_head_pairs(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair one retained and one discarded row per head, deterministically."""
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("pairwise scores and labels must have matching [rows,heads] shape")
    generator = torch.Generator(device=scores.device).manual_seed(int(seed))
    left: list[torch.Tensor] = []
    right: list[torch.Tensor] = []
    for head in range(scores.shape[1]):
        keep = torch.nonzero(~labels[:, head], as_tuple=False).flatten()
        discard = torch.nonzero(labels[:, head], as_tuple=False).flatten()
        if not keep.numel() or not discard.numel():
            continue
        count = min(int(keep.numel()), int(discard.numel()))
        keep_order = torch.randperm(keep.numel(), generator=generator, device=scores.device)[:count]
        discard_order = torch.randperm(discard.numel(), generator=generator, device=scores.device)[:count]
        # Importance polarity: retained score should be greater than discarded.
        left.append(scores[keep[keep_order], head])
        right.append(scores[discard[discard_order], head])
    if not left:
        return scores.new_empty((0,)), scores.new_empty((0,))
    return torch.cat(left), torch.cat(right)


def objective_loss(
    objective: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    budget_weight: float = 0.1,
    mass: torch.Tensor | None = None,
    log_mass_mean: torch.Tensor | None = None,
    log_mass_std: torch.Tensor | None = None,
    p95_positive: torch.Tensor | None = None,
    pair_seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute one frozen objective; logits are eviction polarity (high=evict)."""
    if logits.shape != labels.shape:
        raise ValueError("objective logits and labels must have matching shape")
    target = labels.to(dtype=logits.dtype)
    key = str(objective).upper()
    if key == "L0":
        bce = F.binary_cross_entropy_with_logits(logits, target)
        budget = (torch.sigmoid(logits).mean(dim=0) - target.mean(dim=0)).square().mean()
        return {"loss": bce + float(budget_weight) * budget, "bce": bce, "budget": budget}
    if key == "L2":
        importance = -logits
        kept, discarded = sample_same_head_pairs(importance, labels, seed=pair_seed)
        if not kept.numel():
            zero = logits.sum() * 0.0
            return {"loss": zero, "pairwise": zero, "pairs": torch.zeros((), device=logits.device)}
        pairwise = F.softplus(-(kept - discarded)).mean()
        return {"loss": pairwise, "pairwise": pairwise, "pairs": torch.tensor(kept.numel(), device=logits.device)}
    if mass is None:
        raise ValueError(f"{key} requires continuous mass")
    if key == "L1":
        if log_mass_mean is None or log_mass_std is None:
            raise ValueError("L1 requires train-only log-mass statistics")
        target_importance = normalize_log_mass(mass, log_mass_mean, log_mass_std)
        regression = F.mse_loss(-logits, target_importance)
        return {"loss": regression, "regression": regression}
    if key == "L3":
        if p95_positive is None:
            raise ValueError("L3 requires train-only positive-mass p95 statistics")
        weights = importance_weights(mass, p95_positive)
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights)
        budget = (torch.sigmoid(logits).mean(dim=0) - target.mean(dim=0)).square().mean()
        return {"loss": bce + float(budget_weight) * budget, "weighted_bce": bce, "budget": budget}
    raise ValueError(f"unknown DMS objective {objective!r}; expected L0, L1, L2, or L3")


def fold_affine_importance_into_eviction(weight: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold importance = scale * (hidden·weight+bias)+offset into eviction polarity."""
    return -scale[..., None] * weight, -(scale * bias + offset)
