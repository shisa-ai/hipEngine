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
        / "two_stage_reduction.py"
    )
    spec = importlib.util.spec_from_file_location("micro_two_stage_reduction", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_v2_comparison_schema_shape(artifact: dict) -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "schemas"
        / "result.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    definition = schema["$defs"]["v2Comparison"]
    assert set(definition["required"]) <= set(artifact)
    assert artifact["schema_version"] == definition["properties"]["schema_version"]["const"]
    assert artifact["kind"] == definition["properties"]["kind"]["const"]
    assert artifact["classification"] in schema["$defs"]["classification"]["enum"]
    assert isinstance(artifact["command"], list)
    assert all(isinstance(item, str) for item in artifact["command"])
    assert isinstance(artifact["inputs"], dict)
    assert isinstance(artifact["correctness"], dict)

    hardware_definition = schema["$defs"]["hardware"]
    for backend in ("hip", "vulkan"):
        hardware = artifact["hardware"][backend]
        assert set(hardware_definition["required"]) <= set(hardware)
    source_definition = schema["$defs"]["source"]
    assert set(source_definition["required"]) <= set(artifact["source"])

    comparison_definition = definition["properties"]["comparisons"]["items"]
    assert isinstance(artifact["comparisons"], list)
    for comparison in artifact["comparisons"]:
        assert set(comparison_definition["required"]) <= set(comparison)
        assert comparison["timing_mode"] in comparison_definition["properties"][
            "timing_mode"
        ]["enum"]
        assert comparison["control"] in comparison_definition["properties"]["control"][
            "enum"
        ]
        assert isinstance(comparison["gpu_elapsed"], dict)
        assert isinstance(comparison["host_wall"], dict)


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
    ordering = (
        "hip_round_robin_stream_order"
        if independent and backend == "hip"
        else "vulkan_round_robin_queue_order"
        if independent
        else "hip_stream_order"
        if backend == "hip"
        else "vulkan_compute_barrier"
    )
    submission = (
        "multi_stream"
        if independent and backend == "hip"
        else "vulkan_multi_queue"
        if independent
        else "direct"
        if backend == "hip"
        else "vulkan_command_buffer"
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
            "queue_or_stream_count": 2 if independent else 1,
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


def test_two_stage_comparison_rejects_worker_lane_mismatch() -> None:
    module = _load_runner_module()
    hip = _row("hip", "independent_throughput", 10.0)
    vulkan = _row("vulkan", "independent_throughput", 5.0)
    vulkan["submission"]["queue_or_stream_count"] = 1
    with pytest.raises(ValueError, match="worker lane counts"):
        module._matched_rows([hip, vulkan])


def test_two_stage_cli_exposes_explicit_timing_modes() -> None:
    module = _load_runner_module()
    args = module.parse_args(
        ["--out", "/tmp/result.json", "--timing-mode", "independent_throughput"]
    )
    assert args.timing_mode == "independent_throughput"
    assert args.independent_streams == 4


def test_two_stage_vulkan_uses_calibrated_queue_lanes_not_events() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "runners"
        / "vulkan_two_stage_reduction.cpp"
    ).read_text(encoding="utf-8")
    assert "VulkanMultiQueueTimer" in source
    assert "calibrated_timestamps_extension" in source
    assert "vulkan_round_robin_queue_order" in source
    assert "vkCmdSetEvent" not in source
    assert "vkCmdWaitEvents" not in source


@pytest.mark.parametrize("mode", ["serial_latency", "independent_throughput"])
def test_two_stage_joint_artifact_matches_v2_comparison_schema(mode: str) -> None:
    module = _load_runner_module()
    args = module.parse_args(
        [
            "--out",
            "/tmp/two-stage-comparison.json",
            "--k-list",
            "512",
            "--rows-list",
            "1",
            "--workgroups",
            "64",
            "--split-counts",
            "4",
            "--timing-mode",
            mode,
            "--hardware-gpu",
            "test-gpu",
            "--gfx-arch",
            "gfx-test",
        ]
    )
    rows = [_row("hip", mode, 10.0), _row("vulkan", mode, 5.0)]
    matched = module._matched_rows(rows)
    artifact = module._build_comparison_artifact(
        args=args,
        environment={
            "repo": {
                "root": "/repo",
                "branch": "main",
                "commit": "abc123",
                "dirty": False,
            }
        },
        source_hash="sha256:test",
        commands=[{"kind": "test"}],
        raw_results={"hip": {}, "vulkan": {}},
        rows=rows,
        matched=matched,
        wrapper_command=["python3", "two_stage_reduction.py"],
    )

    _assert_v2_comparison_schema_shape(artifact)
    assert len(artifact["comparisons"]) == 2
    assert {row["control"] for row in artifact["comparisons"]} == {"single", "burst"}
    assert all(row["timing_mode"] == mode for row in artifact["comparisons"])
    assert artifact["matched_rows"] == matched
    assert artifact["raw_results"] == {"hip": {}, "vulkan": {}}
    assert artifact["correctness"]["status"] == "pass"
