"""CPU tests for the prefill-schedule failure probe's analysis helpers.

No GPU, no model, no capture: these cover the parts that decide what the probe
reports about a divergence — the distribution-shape classifier that separates a
near-tie artifact from a real distributional difference, the envelope summary,
and the identity-diff split between host-process fields (which this environment
re-nices mid-run) and everything else (which must block a comparison).
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.tp2_prefill_schedule_failure_probe as probe


def test_distribution_shape_reports_flat_and_sharp_tops() -> None:
    flat = np.full(8, -10.0, dtype=np.float32)
    flat[3] = 0.0
    flat[5] = -0.5  # a near tie: the top candidate holds well under half the mass

    shape = probe._distribution_shape(flat)
    assert shape["top1"] == 3
    assert shape["top2_logit_gap"] == pytest.approx(0.5, abs=1e-6)
    assert 0.6 < shape["top1_prob"] < 0.65
    assert shape["entropy"] > 0.5

    sharp = np.full(8, -20.0, dtype=np.float32)
    sharp[6] = 0.0
    sharp[1] = -12.0

    shape = probe._distribution_shape(sharp)
    assert shape["top1"] == 6
    assert shape["top2_logit_gap"] == pytest.approx(12.0, abs=1e-6)
    assert shape["top1_prob"] > 0.999
    assert shape["entropy"] < 1e-3


def test_distribution_shape_is_invariant_to_a_constant_logit_offset() -> None:
    row = np.linspace(-3.0, 3.0, 16, dtype=np.float32)
    base = probe._distribution_shape(row)
    # A constant offset must not change the shape; float32 rounding of the
    # offset itself bounds how exactly that can hold.
    shifted = probe._distribution_shape(row + 20.0)
    for key in ("top1_prob", "top2_logit_gap", "entropy"):
        assert base[key] == pytest.approx(shifted[key], abs=1e-5)
    assert base["top1"] == shifted["top1"]


def test_envelope_summary_counts_breaches_of_the_declared_ceiling() -> None:
    ceiling = probe.PRODUCTION_GATE["max_kl"]
    kl = np.array([0.0, 1e-6, 1e-3, 0.02, ceiling, ceiling * 4.0], dtype=np.float64)
    top1 = np.array([True, True, True, True, True, False])

    summary = probe._envelope_summary(kl, top1)
    assert summary["rows"] == 6
    assert summary["max_kl"] == pytest.approx(ceiling * 4.0)
    assert summary["top1_agreement"] == pytest.approx(5 / 6)
    assert summary["positions_over_0.01"] == 3
    # The ceiling itself is not a breach; only the row strictly above it is.
    assert summary["positions_over_max_kl"] == 1
    assert summary["p95_kl"] <= summary["p99_kl"] <= summary["max_kl"]


def test_identity_diff_separates_host_process_fields_from_arithmetic_ones() -> None:
    teacher = {
        "model_sha256": "a" * 64,
        "source_revision": "deadbee",
        "source_sha256": {"hipengine/core/memory.py": "b" * 64},
        "host": {"node": "w7900", "nice": 16},
    }
    same = {**teacher, "host": {"node": "w7900", "nice": -4}}
    diff = probe._identity_diff(teacher, same)
    assert set(diff) == {"host"}
    assert set(diff) - set(probe._HOST_PROCESS_FIELDS) == set()

    drift = {**same, "source_sha256": {"hipengine/core/memory.py": "c" * 64}}
    diff = probe._identity_diff(teacher, drift)
    assert set(diff) == {"host", "source_sha256"}
    assert set(diff) - set(probe._HOST_PROCESS_FIELDS) == {"source_sha256"}

    missing = probe._identity_diff(teacher, {"model_sha256": teacher["model_sha256"]})
    assert set(missing) == {"source_revision", "source_sha256", "host"}


def _summary(**overrides):
    base = {
        "rows": 128,
        "mean_kl": 1e-3,
        "p95_kl": 5e-4,
        "p99_kl": 2e-2,
        "max_kl": 0.04,
        "top1_agreement": 1.0,
        "positions_over_0.01": 1,
        "positions_over_max_kl": 0,
    }
    return {**base, **overrides}


def test_aggregate_counts_prompts_breaching_the_ceiling() -> None:
    rollup = probe._aggregate(
        {
            "inside": _summary(),
            "breaching": _summary(
                mean_kl=4e-3,
                p95_kl=9e-4,
                max_kl=0.6,
                top1_agreement=0.99,
                positions_over_max_kl=2,
            ),
        }
    )
    assert rollup["prompts"] == 2
    assert rollup["prompts_breaching_ceiling"] == 1
    assert rollup["positions_over_max_kl"] == 2
    assert rollup["positions_total"] == 256
    assert rollup["mean_kl_over_prompts"] == pytest.approx(2.5e-3)
    assert rollup["worst_prompt_mean_kl"] == pytest.approx(4e-3)
    assert rollup["worst_p95_kl"] == pytest.approx(9e-4)
    assert rollup["worst_max_kl"] == pytest.approx(0.6)
    assert rollup["worst_max_kl_prompt"] == "breaching"
    assert rollup["min_top1_agreement"] == pytest.approx(0.99)


def test_aggregate_is_empty_without_prompts() -> None:
    assert probe._aggregate({}) == {}


def test_compare_reports_curves_and_optional_logit_detail() -> None:
    reference = np.zeros((3, 4), dtype=np.float32)
    candidate = np.zeros((3, 4), dtype=np.float32)
    candidate[1, 0] = 1.0

    plain = probe._compare(reference, candidate, with_shape=False)
    assert plain["rows"] == 3
    assert plain["max_kl_index"] == 1
    assert "max_abs_logit_diff_at_worst" not in plain
    assert len(plain["kl_curve"]) == 3

    detailed = probe._compare(reference, candidate, with_shape=True)
    assert detailed["max_abs_logit_diff_at_worst"] == pytest.approx(1.0)
    assert detailed["teacher_shape_at_worst"]["top2_logit_gap"] == pytest.approx(0.0)
    assert detailed["arm_shape_at_worst"]["top1"] == 0
