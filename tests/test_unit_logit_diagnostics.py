from __future__ import annotations

import json

import numpy as np
import pytest

from hipengine.benchmark.execution_profiles import RowDescriptor, compare_profile_logits
from scripts.quant_quality.metrics import compare_logits, per_row_metrics


def test_distribution_diagnostics_are_shift_invariant_and_signed():
    reference = np.array([[2., 0., -2.], [2., 0., -2.]])
    identical = compare_logits(reference, reference + 100., np.array([0, 0]), top_k=2)
    diagnostics = identical["diagnostics"]
    assert diagnostics["gating"] is False
    for key in ("total_variation", "entropy_delta_nats", "max_probability_delta"):
        assert diagnostics[key]["max"] == pytest.approx(0., abs=1e-14)
    vectors = per_row_metrics(reference, reference * np.array([[.5], [2.]]), np.array([0, 0]), top_k=2)
    assert vectors["entropy_delta_nats"][0] > 0 > vectors["entropy_delta_nats"][1]
    assert vectors["max_probability_delta"][0] < 0 < vectors["max_probability_delta"][1]
    p = np.exp(reference[0]) / np.exp(reference[0]).sum()
    q = np.exp(reference[0] / 2) / np.exp(reference[0] / 2).sum()
    assert vectors["total_variation"][0] == pytest.approx(np.abs(p - q).sum() / 2)
    assert vectors["entropy_delta_nats"][0] == pytest.approx(-sum(q * np.log(q)) + sum(p * np.log(p)))


def test_margin_buckets_counts_boundaries_and_empty_buckets():
    margins = np.array([0., .01, .05, .3, .5, 1., 2., 3.])
    reference = np.column_stack((margins, np.zeros(8)))
    candidate = reference.copy()
    candidate[[1, 3, 5, 7]] = candidate[[1, 3, 5, 7], ::-1]
    result = compare_logits(reference, candidate, np.zeros(8, dtype=int), top_k=1,
                            groups=["a"] * 4 + ["b"] * 4)
    diagnostics = result["diagnostics"]
    assert diagnostics["rows"] == 8
    assert diagnostics["sequence_count"] is None
    assert diagnostics["top1_flips"]["flips"] == 4
    assert diagnostics["top1_flips"]["rate"] == .5
    assert diagnostics["top1_flips"]["wilson95_iid_rows"]["low"] == pytest.approx(.2152160622)
    assert diagnostics["top1_flips"]["wilson95_iid_rows"]["high"] == pytest.approx(.7847839378)
    for bucket in diagnostics["margin_buckets"]:
        assert (bucket["rows"], bucket["flips"], bucket["rate"]) == (2, 1, .5)
    empty = result["by_group"]["a"]["diagnostics"]["margin_buckets"][-1]
    assert empty["rows"] == empty["flips"] == 0
    assert empty["rate"] is None and empty["wilson95_iid_rows"] is None
    json.dumps(result, allow_nan=False)


def test_extreme_finite_logits_and_single_token_diagnostics():
    reference = np.array([[10000., -10000.], [-10000., 10000.]])
    candidate = reference[:, ::-1]
    rows = per_row_metrics(reference, candidate, np.array([0, 1]), top_k=1)
    np.testing.assert_allclose(rows["total_variation"], 1.)
    np.testing.assert_allclose(rows["entropy_delta_nats"], 0.)
    assert all(np.isfinite(values).all() for values in rows.values())
    result = compare_logits(np.ones((2, 1)), np.zeros((2, 1)), np.zeros(2, dtype=int), top_k=1)
    assert result["diagnostics"]["margin_buckets"][0]["rows"] == 2
    assert result["diagnostics"]["top1_flips"]["rate"] == 0
    assert result["diagnostics"]["top1_flips"]["wilson95_iid_rows"]["high"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("flip", [False, True])
def test_profile_diagnostics_are_report_only_and_scoped(monkeypatch, flip):
    import hipengine.benchmark.execution_profiles as profiles

    reference = np.tile([3., 1., 0.], (4, 1))
    candidate = reference.copy()
    if flip:
        candidate[0] = [0., 1., 3.]
    rows = [RowDescriptor(scenario_id="s", scenario_step=i, request_id=str(i // 2),
                          teacher_step=i % 2, category="a" if i < 2 else "b",
                          shape="c1", transition="steady", teacher_token_id=0) for i in range(4)]
    result = compare_profile_logits(reference, candidate, rows)
    assert result["summary"]["diagnostics"]["sequence_count"] == 2
    assert result["by_scope"]["category"]["a"]["diagnostics"]["sequence_count"] == 1
    assert result["summary"]["diagnostics"]["top1_flips"]["flips"] == int(flip)
    original = profiles.summarize_diagnostics
    monkeypatch.setattr(profiles, "summarize_diagnostics", lambda *a, **kw: {"gating": False, "sentinel": "arbitrary"})
    changed = compare_profile_logits(reference, candidate, rows)
    for key in ("hard_gates_passed", "eligible_for_automatic_admission", "requires_outlier_review", "scope_failures", "thresholds"):
        assert changed[key] == result[key]
    monkeypatch.setattr(profiles, "summarize_diagnostics", original)
    assert result["hard_gates_passed"] is not flip
    json.dumps(result, allow_nan=False)


def test_prefix_reuse_report_inherits_shared_diagnostics():
    from scripts.gguf_prefix_reuse_gate import _profile_metrics

    reference = np.array([[2., 0.], [0., 2.]])
    result = _profile_metrics(reference, reference + 5., category="test", lifecycle="active")
    assert result["hard_gates_passed"]
    assert result["summary"]["diagnostics"]["sequence_count"] == 1
    assert result["summary"]["diagnostics"]["total_variation"]["max"] == 0
