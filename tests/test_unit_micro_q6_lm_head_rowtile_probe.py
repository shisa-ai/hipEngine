from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "runners"
        / "q6_lm_head_rowtile_probe.py"
    )
    spec = importlib.util.spec_from_file_location("micro_q6_lm_head_probe", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unmatched_q6_lm_head_join_emits_no_backend_speed_ratio() -> None:
    module = _load_runner_module()
    source = {"repo": "/repo", "commit": "c" * 40, "dirty": False}
    hip = {
        "source": source,
        "hardware": {"gpu_name": "GPU", "gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {
            "rows": [
                {
                    "in_features": 2048,
                    "out_features": 32768,
                    "rows": 1,
                    "variant": "q6_t16_rowtile",
                    "q6_lm_head_median_us": 100.0,
                    "correctness_pass": True,
                }
            ]
        },
    }
    vulkan = {
        "hardware": {"gpu_name": "GPU", "gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {
            "rows": [
                {
                    "in_features": 2048,
                    "out_features": 32768,
                    "rows": 1,
                    "local_size": 128,
                    "variant": "q6_x8_q8_1",
                    "q6_x8_dot_median_us": 80.0,
                    "q6_x8_quantize_plus_dot_median_us": 85.0,
                    "correctness_pass": True,
                }
            ]
        },
    }

    result = module.build_comparison(hip, vulkan, command=["probe", "--compare"])

    assert result["schema_version"] == 2
    assert result["performance_claim"] is False
    assert result["classification"] == "not_reproducible"
    assert result["comparisons"] == []
    assert result["matched_rows"][0]["comparison_status"] == (
        "blocked_unmatched_math_layout"
    )
    assert not [key for key in result["matched_rows"][0] if "speedup" in key]
