from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "runners"
        / "two_stage_reduction.py"
    )
    spec = importlib.util.spec_from_file_location("micro_two_stage_reduction", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stats(value: float) -> dict:
    return {
        "samples": 3,
        "n": 3,
        "median": value,
        "p05": value,
        "p95": value,
        "min": value,
        "max": value,
        "stdev": 0.0,
    }


def _metric(clock: str, value: float, iterations: int) -> dict:
    return {
        "status": "ok",
        "clock": clock,
        "sequence_us": _stats(value * iterations),
        "per_iteration_us": _stats(value),
    }


def _row(backend: str, mode: str, gpu_us: float, reps: int = 4) -> dict:
    independent = mode == "independent_throughput"
    ordering = "none" if independent else (
        "hip_stream_order" if backend == "hip" else "vulkan_compute_barrier"
    )
    submission = "multi_stream" if independent and backend == "hip" else (
        "direct" if backend == "hip" else "vulkan_command_buffer"
    )
    clock = "hip_event" if backend == "hip" else "vulkan_timestamp"
    return {
        "k": 512,
        "rows": 1,
        "workgroup_size": 64,
        "split_count": 4,
        "body_repeats": 8,
        "timing_mode": mode,
        "backend": backend,
        "correctness_pass": True,
        "dependency_contract": {
            "work_dependency": "independent" if independent else "chained",
            "inter_dispatch_ordering": ordering,
            "output_partitioning": "disjoint" if independent else "chained_shared",
            "validation_status": "pass",
        },
        "submission": {
            "strategy": submission,
            "recording_in_timed_region": False,
            "submit_in_host_wall": True,
            "completion_in_host_wall": True,
            "queue_or_stream_count": 2 if independent and backend == "hip" else 1,
        },
        "timing": {
            "single": {
                "logical_iterations": 1,
                "dispatches_per_iteration": 2,
                "gpu_elapsed": _metric(clock, gpu_us, 1),
                "host_wall": _metric("steady_clock", gpu_us + 2.0, 1),
            },
            "burst": {
                "logical_iterations": reps,
                "dispatches_per_iteration": 2,
                "gpu_elapsed": _metric(clock, gpu_us, reps),
                "host_wall": _metric("steady_clock", gpu_us + 1.0, reps),
            },
        },
        "correctness": {
            "single_dispatch": {"status": "pass", "oracle": "CPU reference"},
            "timed_sequence": {
                "status": "pass",
                "oracle": "CPU reference with sequence tag",
                "logical_iterations": reps,
                "coverage": "all_dispatches" if independent else "chained_final_state",
            },
            "synchronization": {
                "status": "pass",
                "method": ordering,
                "barrier_count": reps if backend == "vulkan" else 0,
            },
        },
    }


@pytest.mark.parametrize("mode", ["serial_latency", "independent_throughput"])
def test_two_stage_comparison_uses_dependency_correct_gpu_ratio(mode: str) -> None:
    module = _load_runner_module()
    matched = module._matched_rows(
        [_row("hip", mode, 10.0), _row("vulkan", mode, 5.0)]
    )

    assert len(matched) == 1
    row = matched[0]
    assert row["timing_mode"] == mode
    assert row["ratios"]["burst"]["gpu_elapsed"]["status"] == "ok"
    assert row["vulkan_vs_hip_gpu_burst_speedup"] == 2.0
    assert (
        row["ratios"]["burst"]["host_wall"]["status"]
        == "not_comparable_submission_contract"
    )


def test_two_stage_comparison_rejects_duplicate_rows() -> None:
    module = _load_runner_module()
    row = _row("hip", "serial_latency", 10.0)
    with pytest.raises(ValueError, match="duplicate two-stage result row"):
        module._matched_rows([row, dict(row)])


def test_two_stage_cli_exposes_explicit_timing_modes() -> None:
    module = _load_runner_module()
    args = module.parse_args(
        ["--out", "/tmp/result.json", "--timing-mode", "independent_throughput"]
    )
    assert args.timing_mode == "independent_throughput"
    assert args.independent_streams == 4
