"""CPU-only checks for the AR-versus-verification numerics gate.

The gate itself needs a GPU; these cover the metric and screen arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.ud_mtp_ar_verify_numerics_gate import (
    ENVELOPE,
    _pool,
    _row_kl,
    _screen,
)


def test_identical_logits_have_zero_kl_and_full_top1() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(4, 64)).astype(np.float32)
    kl = _row_kl(logits, logits)
    assert np.allclose(kl, 0.0, atol=1e-12)


def test_kl_is_non_negative_and_grows_with_the_shift() -> None:
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(6, 128)).astype(np.float32)
    near = reference + 1e-3
    far = reference + 1.0
    small = _row_kl(reference, near)
    large = _row_kl(reference, far)
    assert np.all(small >= 0.0) and np.all(large >= 0.0)
    assert float(large.mean()) > float(small.mean())


def test_row_kl_matches_a_hand_computed_two_class_case() -> None:
    reference = np.array([[0.0, 0.0]], dtype=np.float64)
    # shift the second class up: p = [0.5, 0.5], q = sigmoid-shifted
    candidate = np.array([[0.0, np.log(3.0)]], dtype=np.float64)
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    expected = float(np.sum(p * (np.log(p) - np.log(q))))
    assert _row_kl(reference, candidate)[0] == pytest.approx(expected, rel=1e-12)


def test_row_kl_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        _row_kl(np.zeros((2, 4)), np.zeros((3, 4)))


def _run(rows, *, budget=1, category="code", kl=None, top1=None):
    kl = kl if kl is not None else [0.0] * len(rows)
    top1 = top1 if top1 is not None else [True] * len(rows)
    return {
        "id": "p",
        "category": category,
        "prompt_tokens": 64,
        "ar_tokens": [],
        "cases": [{
            "budget": budget,
            "rows": len(rows),
            "kl": list(kl),
            "top1": list(top1),
            "root_position": 64,
            "native_graph": True,
            "fallback_reason": None,
            "target_top1": [],
            "ar_top1": [],
            "target_top1_matches_ar": True,
            "max_abs_diff": 0.0,
            "finite": True,
        }],
    }


def test_pool_aggregates_scope_and_budget_and_flags_p99_outliers() -> None:
    results = [
        _run([0, 0], budget=1, category="code", kl=[1e-9, 1e-9]),
        _run([0, 0, 0], budget=2, category="general_ja",
             kl=[1e-9, 1e-9, 1.0], top1=[True, True, False]),
    ]
    pooled = _pool(results)
    assert pooled["rows"] == 5
    assert pooled["kl_max"] == pytest.approx(1.0)
    assert pooled["top1_agreement"] == pytest.approx(4 / 5)
    assert pooled["top1_by_scope"]["code"]["agreement"] == 1.0
    assert pooled["top1_by_scope"]["general_ja"]["agreement"] == pytest.approx(2 / 3)
    assert set(pooled["top1_by_budget"]) == {"1", "2"}
    assert pooled["rows_above_p99"] and pooled["rows_above_p99"][0]["kl"] == 1.0


def test_screen_binds_every_section_61_threshold() -> None:
    clean = _pool([_run([0, 0], kl=[0.0, 0.0])])
    assert _screen(clean)["passed"] is True

    for name, value, key in (
        ("mean", ENVELOPE["mean"] * 2, "kl_mean"),
        ("p95", ENVELOPE["p95"] * 2, "kl_p95"),
        ("p99", ENVELOPE["p99"] * 2, "kl_p99"),
        ("max", ENVELOPE["max"] * 2, "kl_max"),
    ):
        broken = _pool([_run([0, 0], kl=[value, value])])
        assert broken[key] > ENVELOPE[name]
        assert _screen(broken)["passed"] is False

    bad_top1 = _pool([_run([0, 0, 0, 0], top1=[False, False, True, True])])
    assert _screen(bad_top1)["passed"] is False
