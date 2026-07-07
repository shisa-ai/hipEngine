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
        / "vopd_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("micro_vopd_sweep", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_vopd_variants() -> None:
    module = _load_runner_module()

    variants = module.parse_variants("independent_fma:2,dependent_fma:4,dequant_like:4")

    assert variants == [
        {"mode": "independent_fma", "mode_id": 0, "accums": 2},
        {"mode": "dependent_fma", "mode_id": 1, "accums": 4},
        {"mode": "dequant_like", "mode_id": 3, "accums": 4},
    ]


def test_build_vopd_comparison() -> None:
    module = _load_runner_module()
    hip = {
        "source": {"commit": "c" * 40},
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {
            "rows": [
                {
                    "mode": "independent_fma",
                    "accums": 4,
                    "median_us": 10.0,
                    "gops": 100.0,
                    "correctness_pass": True,
                    "vopd_count": 2,
                    "vopd_op_count": 4,
                    "instruction_count": 40,
                    "waitcnt_count": 1,
                    "wave_size": 32,
                    "vgpr": 12,
                    "sgpr": 18,
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
                    "mode": "independent_fma",
                    "accums": 4,
                    "median_us": 5.0,
                    "gops": 200.0,
                    "correctness_pass": True,
                    "vopd_count": 4,
                    "vopd_op_count": 8,
                    "instruction_count": 30,
                    "waitcnt_count": 0,
                    "wave_size": 64,
                    "estimated_vgpr_span": 10,
                    "estimated_sgpr_span": 16,
                }
            ]
        },
    }

    comparison = module.build_comparison(
        hip,
        vulkan,
        command=["python3", "vopd_sweep.py", "--compare", "hip.json", "vulkan.json"],
    )

    assert comparison["kind"] == "hipengine_micro_comparison"
    assert comparison["bench"] == "f32_vopd_scheduling"
    assert comparison["classification"] == "diagnostic_unclassified"
    assert len(comparison["matched_rows"]) == 1
    row = comparison["matched_rows"][0]
    assert row["mode"] == "independent_fma"
    assert row["vulkan_vs_hip_speedup"] == 2.0
    assert row["hip_vopd_count"] == 2
    assert row["vulkan_vopd_count"] == 4
    json.dumps(comparison, allow_nan=False)
