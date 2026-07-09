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
        / "memory_waitcnt.py"
    )
    spec = importlib.util.spec_from_file_location("micro_memory_waitcnt", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timed_row(
    module, *, backend: str, median_us: float, timing_mode: str = "serial_latency"
):
    repetitions = 8
    return module._row_from_raw(
        {
            "rows": [
                {
                    "mode": "coalesced",
                    "param": 4,
                    "block_size": 128,
                    "median_us": median_us,
                    "bandwidth_gbps": 1000.0 / median_us,
                    "correctness_pass": True,
                    "timed_sequence_correctness_pass": True,
                    "synchronization_pass": True,
                    "timing_mode": timing_mode,
                    "queue_or_stream_count": 1,
                    "gpu_timestamps_supported": True,
                    "barrier_count": (
                        repetitions - 1
                        if backend == "vulkan" and timing_mode == "serial_latency"
                        else 0
                    ),
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


def test_memory_waitcnt_fixed_block_arg_is_hip_only() -> None:
    module = _load_runner_module()

    args = module.parse_args(["--backend", "hip", "--hip-fixed-block-index"])
    assert args.hip_fixed_block_index is True

    try:
        module.parse_args(["--backend", "vulkan", "--hip-fixed-block-index"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit for Vulkan fixed HIP block flag")


def test_memory_waitcnt_timing_and_workgroup_args() -> None:
    module = _load_runner_module()
    args = module.parse_args(
        ["--backend", "hip", "--timing-mode", "independent_throughput", "--workgroups", "64,256"]
    )
    assert args.timing_mode == "independent_throughput"
    assert args.workgroup_sizes == [64, 256]


def test_memory_waitcnt_serial_kernels_accumulate_sequence_state() -> None:
    root = Path(__file__).resolve().parents[1]
    hip_source = (root / "benchmarks/micro/runners/hip_memory_waitcnt.hip").read_text()
    vulkan_source = (
        root / "benchmarks/micro/kernels/vulkan/memory_waitcnt.comp"
    ).read_text()
    assert "out[idx] = out[idx] + run_value" in hip_source
    assert "out_values[output_index] = out_values[output_index] + run_value" in vulkan_source


def test_memory_waitcnt_independent_storage_covers_warmup() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "benchmarks/micro/runners/hip_memory_waitcnt.hip",
        "benchmarks/micro/runners/vulkan_memory_waitcnt.cpp",
    ):
        source = (root / relative).read_text()
        assert "std::max(args.reps, args.warmup)" in source


def test_build_memory_waitcnt_comparison() -> None:
    module = _load_runner_module()
    hip_row = _timed_row(module, backend="hip", median_us=10.0)
    hip_row.update(
        instruction_count=40,
        waitcnt_count=4,
        waitcnt_per_load_instruction=2.0,
        load_instruction_count=2,
        wave_size=32,
        vgpr=12,
        sgpr=18,
        scratch_bytes=0,
        vopd_count=1,
    )
    hip = {
        "source": {"commit": "c" * 40},
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {"rows": [hip_row]},
    }
    vulkan_row = _timed_row(module, backend="vulkan", median_us=5.0)
    vulkan_row.update(
        instruction_count=30,
        waitcnt_count=1,
        waitcnt_per_load_instruction=1.0,
        load_instruction_count=1,
        wave_size=64,
        estimated_vgpr_span=10,
        estimated_sgpr_span=16,
        vopd_count=0,
    )
    vulkan = {
        "hardware": {"gfx_arch": "gfx1151"},
        "correctness": {"status": "pass"},
        "measurements": {"rows": [vulkan_row]},
    }

    comparison = module.build_comparison(
        hip,
        vulkan,
        command=["python3", "memory_waitcnt.py", "--compare", "hip.json", "vulkan.json"],
    )

    assert comparison["kind"] == "hipengine_micro_comparison"
    assert comparison["bench"] == "memory_waitcnt_scheduling"
    assert comparison["classification"] == "diagnostic_unclassified"
    assert comparison["schema_version"] == 2
    assert len(comparison["comparisons"]) == 2
    row = comparison["comparisons"][1]
    assert row["mode"] == "coalesced"
    assert row["param"] == 4
    assert row["gpu_elapsed"]["status"] == "ok"
    assert row["gpu_elapsed"]["vulkan_vs_hip_speedup"] == 2.0
    assert row["host_wall"]["status"] == "not_comparable_submission_contract"
    assert row["hip_waitcnt_count"] == 4
    assert row["vulkan_waitcnt_count"] == 1
    json.dumps(comparison, allow_nan=False)


def test_memory_waitcnt_comparison_rejects_cross_mode_rows() -> None:
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
