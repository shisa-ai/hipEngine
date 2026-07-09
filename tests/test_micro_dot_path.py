from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


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


def _timed_row(module, *, backend: str, median_us: float, timing_mode: str = "serial_latency"):
    repetitions = 8
    row = module._row_from_raw(
        {
            "rows": [
                {
                    "mode": "q4_unsigned",
                    "groups": 16,
                    "block_size": 128,
                    "median_us": median_us,
                    "gops": 1000.0 / median_us,
                    "correctness_pass": True,
                    "timed_sequence_correctness_pass": True,
                    "synchronization_pass": True,
                    "timing_mode": timing_mode,
                    "queue_or_stream_count": 4 if backend == "hip" and timing_mode == "independent_throughput" else 1,
                    "gpu_timestamps_supported": True,
                    "barrier_count": repetitions - 1 if backend == "vulkan" and timing_mode == "serial_latency" else 0,
                    "timing_raw": {
                        "single": {
                            "logical_iterations": 1,
                            "dispatches_per_iteration": 1,
                            "gpu_samples_us": [median_us] * 3,
                            "host_samples_us": [median_us + 4.0] * 3,
                        },
                        "burst": {
                            "logical_iterations": repetitions,
                            "dispatches_per_iteration": 1,
                            "gpu_samples_us": [median_us * repetitions] * 3,
                            "host_samples_us": [(median_us + 2.0) * repetitions] * 3,
                        },
                    },
                }
            ]
        },
        backend=backend,
    )
    return row


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


def test_dot_path_timing_and_workgroup_args() -> None:
    module = _load_runner_module()

    args = module.parse_args(
        [
            "--backend",
            "hip",
            "--timing-mode",
            "independent_throughput",
            "--workgroups",
            "64,128,256",
        ]
    )

    assert args.timing_mode == "independent_throughput"
    assert args.workgroup_sizes == [64, 128, 256]


def test_build_dot_path_comparison() -> None:
    module = _load_runner_module()
    hip_row = _timed_row(module, backend="hip", median_us=10.0)
    hip_row.update(
        {
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
    )
    hip = {
        "source": {"commit": "c" * 40},
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {"rows": [hip_row]},
    }
    vulkan_row = _timed_row(module, backend="vulkan", median_us=5.0)
    vulkan_row.update(
        {
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
    )
    vulkan = {
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {"rows": [vulkan_row]},
    }

    comparison = module.build_comparison(
        hip,
        vulkan,
        command=["python3", "dot_path.py", "--compare", "hip.json", "vulkan.json"],
    )

    assert comparison["kind"] == "hipengine_micro_comparison"
    assert comparison["bench"] == "packed_dot_path"
    assert comparison["classification"] == "diagnostic_unclassified"
    assert comparison["schema_version"] == 2
    assert len(comparison["comparisons"]) == 2
    row = comparison["comparisons"][1]
    assert row["mode"] == "q4_unsigned"
    assert row["groups"] == 16
    assert row["workgroup_size"] == 128
    assert row["gpu_elapsed"]["status"] == "ok"
    assert row["gpu_elapsed"]["vulkan_vs_hip_speedup"] == 2.0
    assert row["host_wall"]["status"] == "not_comparable_submission_contract"
    assert row["hip_dot4_count"] == 8
    assert row["vulkan_dot4_count"] == 8
    assert row["vulkan_spirv_sudot_count"] == 1
    json.dumps(comparison, allow_nan=False)


def test_dot_path_comparison_rejects_cross_mode_rows() -> None:
    module = _load_runner_module()
    hip = {"measurements": {"rows": [_timed_row(module, backend="hip", median_us=10.0)]}}
    vulkan = {
        "measurements": {
            "rows": [
                _timed_row(
                    module,
                    backend="vulkan",
                    median_us=5.0,
                    timing_mode="independent_throughput",
                )
            ]
        }
    }

    with pytest.raises(ValueError, match="timing modes"):
        module.build_comparison(hip, vulkan, command=["compare"])
