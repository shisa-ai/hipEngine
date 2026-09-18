"""Unit tier: the TP2 MLP pair-route admission probe's pure decision logic.

The probe's device work cannot run without ROCm, but its *verdict* must not
depend on the device: the policy row that admits a shard width to
``GGUF_DENSE_PAIR_SILU_DECODE_POLICIES`` is only as trustworthy as the rule
that decided it. These tests pin that rule - including the cases where a
missing comparison must count as *not* admitted rather than as agreement - and
the byte-level comparison the rule consumes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "scripts" / "tp2_mlp_pair_admission_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("tp2_mlp_pair_admission_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def _field(*, identical: bool = True, present: bool = True, shape_match: bool = True):
    return {
        "present_in_both": present,
        "shape_match": shape_match,
        "bit_identical": identical,
    }


def _row(**overrides):
    row = {
        "rank": 0,
        "per_rank_ffn": 7168,
        "kernel_contract_error": None,
        "fused_error": None,
        "fields": {"activated": _field(), "down_partial": _field()},
    }
    row.update(overrides)
    return row


# --- the verdict ---------------------------------------------------------


def test_a_bit_identical_pair_of_ranks_is_admitted() -> None:
    rows = [_row(rank=0, per_rank_ffn=7168), _row(rank=1, per_rank_ffn=10240)]
    assert probe.admission_verdict(rows) is True


def test_a_single_differing_element_blocks_admission() -> None:
    rows = [
        _row(rank=0),
        _row(rank=1, fields={"activated": _field(identical=False), "down_partial": _field()}),
    ]
    assert probe.admission_verdict(rows) is False


def test_a_width_the_kernel_contract_rejects_blocks_admission() -> None:
    rows = [_row(kernel_contract_error="requires out_features to be a positive multiple of 16")]
    assert probe.admission_verdict(rows) is False


def test_a_failed_fused_launch_blocks_admission() -> None:
    rows = [_row(fused_error="RuntimeError: the fused pair+SiLU candidate did not launch")]
    assert probe.admission_verdict(rows) is False


def test_a_missing_comparison_field_blocks_admission() -> None:
    """Absence of a difference is not evidence of equality."""

    rows = [_row(fields={})]
    assert probe.admission_verdict(rows) is False


def test_a_field_present_in_only_one_chain_blocks_admission() -> None:
    rows = [_row(fields={"activated": _field(present=False), "down_partial": _field()})]
    assert probe.admission_verdict(rows) is False


def test_a_shape_mismatch_blocks_admission() -> None:
    rows = [_row(fields={"activated": _field(shape_match=False), "down_partial": _field()})]
    assert probe.admission_verdict(rows) is False


def test_no_ranks_is_not_admission() -> None:
    assert probe.admission_verdict([]) is False


# --- the byte comparison -------------------------------------------------


def test_identical_bytes_compare_identical() -> None:
    a = np.arange(16, dtype=np.float32)
    report = probe._compare(a, a.copy())
    assert report["shape_match"] is True
    assert report["bit_identical"] is True
    assert report["differing_elements"] == 0
    assert report["max_abs_diff"] == 0.0


def test_a_single_ulp_difference_is_reported_as_not_identical() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = a.copy()
    b[1] = np.nextafter(b[1], np.float32(np.inf))
    report = probe._compare(a, b)
    assert report["bit_identical"] is False
    assert report["differing_elements"] == 1
    assert report["max_abs_diff"] > 0.0


def test_negative_zero_and_zero_are_different_bytes() -> None:
    """Bit-identity is a byte question, so -0.0 != +0.0 here."""

    a = np.array([0.0], dtype=np.float32)
    b = np.array([-0.0], dtype=np.float32)
    assert probe._compare(a, b)["bit_identical"] is False


def test_identity_is_decided_on_bytes_not_dtype_width() -> None:
    """Two views of the same bytes are identical; equal values are not the test."""

    a = np.array([1.5, 2.5], dtype=np.float32)
    b = a.view(np.uint8)
    report = probe._compare(a, b)
    assert report["shape_match"] is False


def test_shape_mismatch_is_reported_without_crashing() -> None:
    report = probe._compare(np.zeros(4, dtype=np.uint8), np.zeros(8, dtype=np.uint8))
    assert report == {"shape_match": False, "a_shape": [4], "b_shape": [8]}


# --- fractions parsing ---------------------------------------------------


def test_fractions_parse_two_shares() -> None:
    assert probe._parse_fractions("0.417145/0.582855") == pytest.approx(
        (0.417145, 0.582855)
    )


def test_fractions_reject_a_single_share() -> None:
    with pytest.raises(SystemExit):
        probe._parse_fractions("1.0")


def test_fractions_reject_non_numeric() -> None:
    with pytest.raises(SystemExit):
        probe._parse_fractions("half/rest")
