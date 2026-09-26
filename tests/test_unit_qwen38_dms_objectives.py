from __future__ import annotations

import pytest
import torch

from scripts.qwen38_dms_objectives import (
    fold_affine_importance_into_eviction,
    importance_weights,
    normalize_log_mass,
    objective_loss,
    sample_same_head_pairs,
    train_only_log_mass_stats,
)


def fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.7], [0.8, -0.4]], dtype=torch.float32)
    labels = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.bool)
    mass = torch.tensor([[4.0, 0.0], [1.0, 2.0], [3.0, 1.0], [0.0, 5.0]])
    return logits, labels, mass


def test_l0_and_l3_have_finite_budgeted_losses() -> None:
    logits, labels, mass = fixture()
    l0 = objective_loss("L0", logits, labels)
    assert torch.isfinite(l0["loss"]) and "bce" in l0 and "budget" in l0
    p95 = torch.tensor([3.0, 5.0])
    l3 = objective_loss("L3", logits, labels, mass=mass, p95_positive=p95)
    assert torch.isfinite(l3["loss"])
    assert torch.all(importance_weights(mass, torch.zeros(2)) == 1)


def test_l1_train_only_stats_and_polarity() -> None:
    logits, labels, mass = fixture()
    mean, std = train_only_log_mass_stats(mass)
    normalized = normalize_log_mass(mass, mean, std)
    result = objective_loss("L1", logits, labels, mass=mass, log_mass_mean=mean, log_mass_std=std)
    assert normalized.shape == mass.shape
    assert torch.isfinite(result["loss"])
    with pytest.raises(ValueError, match="statistics"):
        objective_loss("L1", logits, labels, mass=mass)


def test_l2_pairing_is_seeded_and_zero_pair_is_safe() -> None:
    logits, labels, _ = fixture()
    left0, right0 = sample_same_head_pairs(-logits, labels, seed=0)
    left1, right1 = sample_same_head_pairs(-logits, labels, seed=0)
    assert torch.equal(left0, left1) and torch.equal(right0, right1)
    result = objective_loss("L2", logits, labels, pair_seed=0)
    assert int(result["pairs"]) == left0.numel()
    no_pairs = objective_loss("L2", logits, torch.zeros_like(labels), pair_seed=0)
    assert no_pairs["pairs"].item() == 0 and no_pairs["loss"].item() == 0


def test_affine_importance_export_preserves_eviction_polarity() -> None:
    weight = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    bias = torch.tensor([0.2, -0.4])
    scale = torch.tensor([2.0, 0.5])
    offset = torch.tensor([0.1, -0.2])
    new_weight, new_bias = fold_affine_importance_into_eviction(weight, bias, scale, offset)
    hidden = torch.tensor([[0.3, 0.7]])
    old_importance = (hidden @ weight.T + bias) * scale + offset
    new_eviction = hidden @ new_weight.T + new_bias
    assert torch.allclose(new_eviction, -old_importance)
