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
        "variant": "two_stage",
        "workgroup_specialization": (
            "fixed" if backend == "hip" else "specialization_constant"
        ),
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


def _raw_result(
    backend: str,
    *,
    device_name: str | None = None,
    source_hash: str | None = None,
) -> dict:
    hardware = {
        "device_name": device_name
        or (
            "Radeon 8060S Graphics"
            if backend == "hip"
            else "AMD Radeon 8060S Graphics (RADV STRIX_HALO)"
        )
    }
    if backend == "hip":
        hardware["gcn_arch_name"] = "gfx1151"
    raw = {
        "run_tag": f"{backend}-two-stage-reduction",
        "status": "diagnostic",
        "backend": backend,
        "hardware": hardware,
    }
    if source_hash:
        raw["source"] = {"source_hash": source_hash}
    return raw


def _joint_artifact(
    module,
    mode: str,
    *,
    dirty: bool = False,
    commit: str = "abc123",
    vulkan_device: str | None = None,
    drop_vulkan_raw: bool = False,
    fail_correctness: bool = False,
    include_source_hashes: bool = True,
) -> dict:
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
            "--body-repeats",
            "8",
            "--reps",
            "4",
            "--independent-streams",
            "2",
            "--timing-mode",
            mode,
            "--hardware-gpu",
            "AMD Radeon 8060S Graphics",
            "--gfx-arch",
            "gfx1151",
        ]
    )
    rows = [_row("hip", mode, 10.0), _row("vulkan", mode, 5.0)]
    if fail_correctness:
        rows[0]["correctness_pass"] = False
    matched = module._matched_rows(rows)
    raw_results = {
        "hip:wg64": _raw_result(
            "hip",
            source_hash="sha256:raw-hip" if include_source_hashes else None,
        ),
        "vulkan": _raw_result(
            "vulkan",
            device_name=vulkan_device,
            source_hash=("sha256:raw-vulkan" if include_source_hashes else None),
        ),
    }
    if drop_vulkan_raw:
        del raw_results["vulkan"]
    return module._build_comparison_artifact(
        args=args,
        environment={
            "repo": {
                "root": "/repo",
                "branch": "main",
                "commit": commit,
                "dirty": dirty,
            }
        },
        source_hash="sha256:joint",
        commands=[{"kind": "test"}],
        raw_results=raw_results,
        rows=rows,
        matched=matched,
        wrapper_command=["python3", "two_stage_reduction.py"],
    )


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
    artifact = _joint_artifact(module, mode)

    _assert_v2_comparison_schema_shape(artifact)
    assert len(artifact["comparisons"]) == 2
    assert {row["control"] for row in artifact["comparisons"]} == {"single", "burst"}
    assert all(row["timing_mode"] == mode for row in artifact["comparisons"])
    assert len(artifact["matched_rows"]) == 1
    assert artifact["correctness"]["status"] == "pass"
    assert artifact["performance_claim"] is True
    assert artifact["claim_gate"]["status"] == "pass"
    assert artifact["claim_gate"]["same_commit"] is True
    assert artifact["claim_gate"]["device_match"] is True
    assert artifact["claim_gate"]["matrix_complete"] is True
    assert artifact["claim_gate"]["blocking_reasons"] == []
    assert artifact["sources"] == {"shared": artifact["source"]}
    assert artifact["source_coverage"]["backend_source_hashes"] == {
        "hip": ["sha256:raw-hip"],
        "vulkan": ["sha256:raw-vulkan"],
    }


def test_two_stage_joint_artifact_blocks_dirty_device_and_raw_identity_claims() -> None:
    module = _load_runner_module()

    dirty = _joint_artifact(module, "serial_latency", dirty=True)
    assert dirty["performance_claim"] is False
    assert dirty["claim_gate"]["blocking_reasons"] == ["dirty_source"]

    mismatched_device = _joint_artifact(
        module,
        "serial_latency",
        vulkan_device="Different GPU",
    )
    assert mismatched_device["performance_claim"] is False
    assert "device_identity_mismatch_or_missing" in mismatched_device["claim_gate"][
        "blocking_reasons"
    ]

    missing_raw = _joint_artifact(
        module,
        "serial_latency",
        drop_vulkan_raw=True,
    )
    assert missing_raw["performance_claim"] is False
    assert missing_raw["claim_gate"]["raw_result_identity_pass"] is False
    assert "raw_result_identity_mismatch_or_missing" in missing_raw["claim_gate"][
        "blocking_reasons"
    ]

    incorrect = _joint_artifact(
        module,
        "serial_latency",
        fail_correctness=True,
    )
    assert incorrect["performance_claim"] is False
    assert incorrect["claim_gate"]["correctness_pass"] is False
    assert "correctness_not_passed" in incorrect["claim_gate"]["blocking_reasons"]

    missing_commit = _joint_artifact(module, "serial_latency", commit="")
    assert missing_commit["performance_claim"] is False
    assert missing_commit["claim_gate"]["same_commit"] is False
    assert missing_commit["claim_gate"]["clean_source"] is True
    assert missing_commit["claim_gate"]["blocking_reasons"] == ["commit_missing"]


def test_two_stage_joint_artifact_marks_combined_hash_backend_coverage() -> None:
    module = _load_runner_module()
    artifact = _joint_artifact(
        module,
        "serial_latency",
        include_source_hashes=False,
    )

    assert artifact["performance_claim"] is True
    assert artifact["source_coverage"]["backend_source_hashes"] == {
        "hip": [],
        "vulkan": [],
    }
    assert artifact["source_coverage"]["backend_hash_status"] == {
        "hip": "covered_by_combined_source_hash",
        "vulkan": "covered_by_combined_source_hash",
    }
