from __future__ import annotations

import pytest
import torch

from scripts.qwen38_dms_nonlinear import DMSBottleneckMLP, bf16_export, export_score_equivalence, parameter_count, rank_agreement


def test_mlp_geometry_forward_and_parameter_count() -> None:
    model = DMSBottleneckMLP(hidden_size=8, bottleneck=3, heads=4)
    output = model(torch.randn(5, 8))
    assert output.shape == (5, 4)
    assert parameter_count(model) == (8 * 3 + 3 + 3 * 4 + 4)
    assert set(bf16_export(model)) == {"input.weight", "input.bias", "output.weight", "output.bias"}


def test_mlp_rejects_wrong_geometry() -> None:
    model = DMSBottleneckMLP(hidden_size=8, bottleneck=3, heads=4)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(5, 7))


def test_bf16_export_is_finite_and_rank_metric_is_bounded() -> None:
    model = DMSBottleneckMLP(hidden_size=8, bottleneck=3, heads=4)
    hidden = torch.randn(12, 8)
    reference, rounded = export_score_equivalence(model, hidden)
    assert torch.isfinite(rounded).all()
    assert 0.0 <= rank_agreement(reference, rounded) <= 1.0
    assert rank_agreement(reference, reference) == 1.0
    with pytest.raises(ValueError, match="matching"):
        rank_agreement(reference, rounded[:, :2])
