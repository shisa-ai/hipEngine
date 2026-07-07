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
        / "dot_path.py"
    )
    spec = importlib.util.spec_from_file_location("micro_dot_path", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_dot_path_variants() -> None:
    module = _load_runner_module()

    variants = module.parse_variants("q8_signed:8,q4_unsigned:16,q6_zero:4,scalar_dequant:2")

    assert variants == [
        {"mode": "q8_signed", "mode_id": 0, "groups": 8},
        {"mode": "q4_unsigned", "mode_id": 1, "groups": 16},
        {"mode": "q6_zero", "mode_id": 2, "groups": 4},
        {"mode": "scalar_dequant", "mode_id": 3, "groups": 2},
    ]


def test_dot_path_wavefront_flags() -> None:
    module = _load_runner_module()

    assert module._hip_wavefront_flags("default") == []
    assert module._hip_wavefront_flags("32") == ["-mno-wavefrontsize64"]
    assert module._hip_wavefront_flags("64") == ["-mwavefrontsize64"]


def test_dot_path_fixed_block_arg_is_hip_only() -> None:
    module = _load_runner_module()

    args = module.parse_args(["--backend", "hip", "--hip-fixed-block-index"])
    assert args.hip_fixed_block_index is True

    try:
        module.parse_args(["--backend", "vulkan", "--hip-fixed-block-index"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit for Vulkan fixed HIP block flag")


def test_build_dot_path_comparison() -> None:
    module = _load_runner_module()
    hip = {
        "source": {"commit": "c" * 40},
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {
            "rows": [
                {
                    "mode": "q4_unsigned",
                    "groups": 16,
                    "median_us": 10.0,
                    "gops": 100.0,
                    "correctness_pass": True,
                    "instruction_count": 40,
                    "dot4_count": 8,
                    "waitcnt_count": 4,
                    "load_instruction_count": 2,
                    "wave_size": 32,
                    "vgpr": 12,
                    "sgpr": 18,
                    "scratch_bytes": 0,
                    "sgpr_spill_count": 0,
                    "vgpr_spill_count": 0,
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
                    "mode": "q4_unsigned",
                    "groups": 16,
                    "median_us": 5.0,
                    "gops": 200.0,
                    "correctness_pass": True,
                    "instruction_count": 30,
                    "dot4_count": 8,
                    "spirv_sdot_count": 0,
                    "spirv_sudot_count": 1,
                    "spirv_udot_count": 0,
                    "spirv_dot_op_count": 1,
                    "waitcnt_count": 1,
                    "load_instruction_count": 1,
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
        command=["python3", "dot_path.py", "--compare", "hip.json", "vulkan.json"],
    )

    assert comparison["kind"] == "hipengine_micro_comparison"
    assert comparison["bench"] == "packed_dot_path"
    assert comparison["classification"] == "diagnostic_unclassified"
    assert len(comparison["matched_rows"]) == 1
    row = comparison["matched_rows"][0]
    assert row["mode"] == "q4_unsigned"
    assert row["groups"] == 16
    assert row["vulkan_vs_hip_speedup"] == 2.0
    assert row["hip_dot4_count"] == 8
    assert row["vulkan_dot4_count"] == 8
    assert row["vulkan_spirv_sudot_count"] == 1
    json.dumps(comparison, allow_nan=False)
