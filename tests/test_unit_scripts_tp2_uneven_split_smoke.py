"""Unit tier: the TP2 uneven-split smoke's pure comparison logic.

The smoke is the cheap gate that runs before the full teacher-coverage suite,
so its pass/fail rule has to be exactly what it claims: identical generated
tokens, a matching logits shape, and top-1 agreement of 1.0 on every
teacher-forced row. Logit *values* are deliberately not part of the rule - the
uneven split regroups ``ffn_down``'s summation, so a difference is expected and
is reported as context for the numerical gate instead.

These tests exercise that rule and the per-rank variant normalization without
touching a device.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "scripts" / "tp2_uneven_split_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("tp2_uneven_split_smoke", SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke()


def _arm(*, tokens, logits, label="arm"):
    return {
        "label": label,
        "tokens": list(tokens),
        "teacher_logits": np.asarray(logits, dtype=np.float64).tolist(),
        "teacher_logits_shape": list(np.asarray(logits).shape),
    }


# --- the pass rule -------------------------------------------------------


def test_identical_tokens_and_top1_pass() -> None:
    logits = [[0.0, 3.0, 1.0], [2.0, 0.0, 1.0]]
    report = smoke.compare(
        _arm(tokens=[7, 8], logits=logits),
        _arm(tokens=[7, 8], logits=logits),
    )
    assert report["passed"] is True
    assert report["tokens_identical"] is True
    assert report["top1_agreement"] == 1.0
    assert report["max_abs_logit_diff"] == 0.0


def test_a_small_logit_difference_that_keeps_top1_still_passes() -> None:
    """The split regroups ffn_down's sum: values may move, top-1 may not."""

    even = _arm(tokens=[7, 8], logits=[[0.0, 3.0, 1.0], [2.0, 0.0, 1.0]])
    uneven = _arm(tokens=[7, 8], logits=[[0.05, 2.9, 1.02], [1.98, 0.01, 1.0]])
    report = smoke.compare(even, uneven)
    assert report["passed"] is True
    assert report["max_abs_logit_diff"] == pytest.approx(0.1, abs=1e-9)
    assert report["top1_agreement"] == 1.0


def test_a_flipped_top1_fails_even_with_close_values() -> None:
    even = _arm(tokens=[7], logits=[[3.0, 3.1]])
    uneven = _arm(tokens=[7], logits=[[3.05, 3.0]])
    report = smoke.compare(even, uneven)
    assert report["passed"] is False
    assert report["top1_agreement"] == 0.0


def test_different_generated_tokens_fail() -> None:
    logits = [[0.0, 3.0, 1.0]]
    report = smoke.compare(
        _arm(tokens=[7], logits=logits),
        _arm(tokens=[9], logits=logits),
    )
    assert report["passed"] is False
    assert report["tokens_identical"] is False


def test_a_shape_mismatch_fails_and_reports_no_metrics() -> None:
    report = smoke.compare(
        _arm(tokens=[7], logits=[[0.0, 3.0]]),
        _arm(tokens=[7], logits=[[0.0, 3.0, 1.0]]),
    )
    assert report["passed"] is False
    assert report["teacher_logits_shape_match"] is False
    assert "max_abs_logit_diff" not in report


def test_empty_logits_do_not_divide_by_zero() -> None:
    report = smoke.compare(
        _arm(tokens=[7], logits=np.zeros((0, 4))),
        _arm(tokens=[7], logits=np.zeros((0, 4))),
    )
    assert report["teacher_logits_shape_match"] is True
    assert report["passed"] is False


# --- per-rank variant normalization --------------------------------------


def test_a_scalar_variant_is_expanded_to_every_rank() -> None:
    assert smoke._variant_by_rank("dense_dual_local32_bf16_bf16_out", 2) == {
        0: "dense_dual_local32_bf16_bf16_out",
        1: "dense_dual_local32_bf16_bf16_out",
    }


def test_a_per_rank_mapping_is_preserved() -> None:
    mapping = {0: "a", 1: None}
    assert smoke._variant_by_rank(mapping, 2) == {0: "a", 1: None}


def test_none_variant_reports_none_per_rank() -> None:
    """An unadmitted width falls back to the unfused chain and shows up here."""

    assert smoke._variant_by_rank(None, 2) == {0: None, 1: None}


# --- fractions parsing ---------------------------------------------------


def test_fractions_parse() -> None:
    assert smoke._parse_fractions("0.417145/0.582855") == pytest.approx(
        (0.417145, 0.582855)
    )


def test_fractions_reject_one_share() -> None:
    with pytest.raises(SystemExit):
        smoke._parse_fractions("0.5")


def test_fractions_reject_non_positive() -> None:
    with pytest.raises(SystemExit):
        smoke._parse_fractions("0.0/1.0")
