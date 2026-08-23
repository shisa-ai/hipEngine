from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mtp_iu4_roofline",
    ROOT / "scripts" / "mtp_iu4_roofline.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_percentile_uses_linear_interpolation() -> None:
    assert MODULE._percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)


def test_summarize_selects_best_median_chain_count() -> None:
    raw = {
        "device": {"arch": "gfx1151"},
        "measurements": [
            {
                "lane": "u4s4_wmma",
                "chains": 2,
                "instruction_ops": 8192,
                "total_ops": 1_000_000_000,
                "milliseconds": [0.02, 0.01, 0.03],
            },
            {
                "lane": "u4s4_wmma",
                "chains": 8,
                "instruction_ops": 8192,
                "total_ops": 4_000_000_000,
                "milliseconds": [0.02, 0.02, 0.02],
            },
        ],
    }

    rows, best = MODULE._summarize(raw)

    assert rows[0]["throughput"]["median_tops"] == pytest.approx(50.0)
    assert rows[1]["throughput"]["median_tops"] == pytest.approx(200.0)
    assert rows[1]["throughput"]["theoretical_tops"] == pytest.approx(118.784)
    assert best["u4s4_wmma"]["chains"] == 8


def test_source_covers_current_and_candidate_instruction_lanes() -> None:
    source = (ROOT / "scripts" / "microbench" / "mtp_iu4_roofline.hip").read_text()
    for builtin in (
        "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32",
        "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32",
        "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32",
        "__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32",
        "__builtin_amdgcn_sudot4",
        "__builtin_amdgcn_sudot8",
    ):
        assert builtin in source
