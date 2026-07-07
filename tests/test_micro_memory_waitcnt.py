from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "runners"
        / "memory_waitcnt.py"
    )
    spec = importlib.util.spec_from_file_location("micro_memory_waitcnt", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_memory_waitcnt_variants() -> None:
    module = _load_runner_module()

    variants = module.parse_variants("coalesced:4,strided:8,gather:1,interleave:16")

    assert variants == [
        {"mode": "coalesced", "mode_id": 0, "param": 4},
        {"mode": "strided", "mode_id": 1, "param": 8},
        {"mode": "gather", "mode_id": 2, "param": 1},
        {"mode": "interleave", "mode_id": 3, "param": 16},
    ]


def test_memory_waitcnt_wavefront_flags() -> None:
    module = _load_runner_module()

    assert module._hip_wavefront_flags("default") == []
    assert module._hip_wavefront_flags("32") == ["-mno-wavefrontsize64"]
    assert module._hip_wavefront_flags("64") == ["-mwavefrontsize64"]


def test_build_memory_waitcnt_comparison() -> None:
    module = _load_runner_module()
    hip = {
        "source": {"commit": "c" * 40},
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {
            "rows": [
                {
                    "mode": "coalesced",
                    "param": 4,
                    "median_us": 10.0,
                    "bandwidth_gbps": 100.0,
                    "correctness_pass": True,
                    "instruction_count": 40,
                    "waitcnt_count": 4,
                    "waitcnt_per_load_instruction": 2.0,
                    "load_instruction_count": 2,
                    "wave_size": 32,
                    "vgpr": 12,
                    "sgpr": 18,
                    "scratch_bytes": 0,
                    "vopd_count": 1,
                }
            ]
        },
    }
    vulkan = {
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {
            "rows": [
                {
                    "mode": "coalesced",
                    "param": 4,
                    "median_us": 5.0,
                    "bandwidth_gbps": 200.0,
                    "correctness_pass": True,
                    "instruction_count": 30,
                    "waitcnt_count": 1,
                    "waitcnt_per_load_instruction": 1.0,
                    "load_instruction_count": 1,
                    "wave_size": 64,
                    "estimated_vgpr_span": 10,
                    "estimated_sgpr_span": 16,
                    "vopd_count": 0,
                }
            ]
        },
    }

    comparison = module.build_comparison(
        hip,
        vulkan,
        command=["python3", "memory_waitcnt.py", "--compare", "hip.json", "vulkan.json"],
    )

    assert comparison["kind"] == "hipengine_micro_comparison"
    assert comparison["bench"] == "memory_waitcnt_scheduling"
    assert comparison["classification"] == "diagnostic_unclassified"
    assert len(comparison["matched_rows"]) == 1
    row = comparison["matched_rows"][0]
    assert row["mode"] == "coalesced"
    assert row["param"] == 4
    assert row["vulkan_vs_hip_speedup"] == 2.0
    assert row["hip_waitcnt_count"] == 4
    assert row["vulkan_waitcnt_count"] == 1
    json.dumps(comparison, allow_nan=False)
