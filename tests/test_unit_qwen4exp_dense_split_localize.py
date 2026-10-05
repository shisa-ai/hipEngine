"""CPU contracts for non-intervening dense-split projection attribution."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.qwen4exp_dense_split_localize import (
    eligible_projection, observe_parent_launch, projection_metrics, sample_rows,
)


def test_metrics_preserve_f64_reference_and_count_bf16_boundary_crossings():
    reference = np.array([[1.00390624, -2.0, 0.5]], np.float64)
    parent = reference.astype(np.float32)
    candidate = np.array([[1.0039064, -2.0, 0.5]], np.float32)
    metrics = projection_metrics(reference, candidate)
    assert metrics["finite"]
    assert metrics["max_abs"] == float(np.max(np.abs(candidate.astype(np.float64) - reference)))
    assert metrics["bf16_flips"] == 1
    assert metrics["elements"] == 3
    assert projection_metrics(parent, parent)["relative_l2"] == 0.0
    assert metrics["kl_mean"] >= 0.0


def test_metrics_reject_nonfinite_and_mismatched_inputs():
    with pytest.raises(ValueError, match="finite"):
        projection_metrics(np.ones((1, 2)), np.array([[1.0, np.inf]]))
    with pytest.raises(ValueError, match="shape"):
        projection_metrics(np.ones((1, 2)), np.ones((2, 2)))
    assert projection_metrics(np.zeros((2, 3)), np.zeros((2, 3)))["relative_l2"] == 0.0
    assert projection_metrics(np.zeros((2, 3)), np.ones((2, 3)))["relative_l2"] is None


def test_sample_rows_include_endpoints_without_duplicates():
    assert sample_rows(2, 8) == [0, 1]
    assert sample_rows(512, 8) == [0, 73, 146, 219, 292, 365, 438, 511]
    with pytest.raises(ValueError):
        sample_rows(0, 8)


def test_only_scoped_raw_f32_parent_roles_are_observed():
    parent, other = object(), object()
    weight = SimpleNamespace(spec=SimpleNamespace(slot_path="layers.3.attn_q", quant_key="gguf_q8_0"))
    kw = dict(rows=512, in_features=2560, out_features=12288, scope={3}, parents={parent})
    assert eligible_projection(parent, weight, **kw)
    assert not eligible_projection(other, weight, **kw)
    assert not eligible_projection(parent, weight, **{**kw, "rows": 256})
    assert not eligible_projection(parent, weight, **{**kw, "in_features": 65})
    assert not eligible_projection(parent, weight, **{**kw, "scope": {4}})
    weight.spec.quant_key = "gguf_q5_k"
    assert not eligible_projection(parent, weight, **kw)


def test_observer_calls_parent_first_without_replacing_its_output():
    events = []
    output = np.zeros(2)

    def parent(value):
        events.append("parent")
        value[:] = [3, 4]
        return "parent-result"

    def measure(value):
        events.append("measure")
        np.testing.assert_array_equal(value, [3, 4])
        diagnostic = value.copy()
        diagnostic[:] = 100  # never publish diagnostic outputs to the model

    assert observe_parent_launch(parent, measure, output) == "parent-result"
    np.testing.assert_array_equal(output, [3, 4])
    assert events == ["parent", "measure"]
