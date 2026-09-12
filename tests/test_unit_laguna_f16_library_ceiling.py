from __future__ import annotations

import argparse
import ctypes

import pytest

from scripts.laguna_f16_library_ceiling import (
    DEFAULT_ROWS,
    _HipblasLtAlgo,
    _HipblasLtHeuristicResult,
    _MODE_ORDER,
    _SHAPES,
    _offset,
    _parse_rows,
    _summarize,
)


def test_laguna_f16_ceiling_declares_exact_production_shapes() -> None:
    assert DEFAULT_ROWS == (16, 32, 64, 128, 256, 512)
    assert _SHAPES == {
        "full_q": (3072, 6144),
        "swa_q": (3072, 9216),
        "kv": (3072, 1024),
        "full_gate": (3072, 48),
        "swa_gate": (3072, 72),
        "full_o": (6144, 3072),
        "swa_o": (9216, 3072),
    }
    assert _offset(1000, 2, 3) == 1024


def test_laguna_f16_ceiling_rows_are_sorted_and_distinct() -> None:
    assert _parse_rows("512,16,128,16") == (16, 128, 512)
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _parse_rows("16,0")


def test_laguna_f16_ceiling_hipblaslt_ctypes_abi_sizes_match_headers() -> None:
    assert ctypes.sizeof(_HipblasLtAlgo) == 24
    assert ctypes.sizeof(_HipblasLtHeuristicResult) == 56


def test_laguna_f16_ceiling_summary_requires_inclusive_gain_everywhere() -> None:
    samples = {
        16: {
            family: {
                mode: ([2.0, 2.1, 1.9] if mode == "exact" else [1.0, 1.1, 0.9])
                for mode in _MODE_ORDER
            }
            for family in ("full", "swa")
        }
    }
    wall = {
        16: {
            family: {mode: [value + 0.1 for value in values] for mode, values in modes.items()}
            for family, modes in samples[16].items()
        }
    }

    result = _summarize((16,), samples, wall)

    assert result["pass"] is True
    assert result["shapes"]["16"]["families"]["full"]["hipblaslt_inclusive"][
        "speedup_vs_exact"
    ] == pytest.approx(2.0)

    samples[16]["swa"]["hipblaslt_inclusive"] = [2.2, 2.1, 2.3]
    rejected = _summarize((16,), samples, wall)
    assert rejected["pass"] is False
    assert rejected["failed_checks"] == [
        "rows_16_swa_hipblaslt_inclusive_not_faster"
    ]
