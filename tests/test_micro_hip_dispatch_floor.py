from __future__ import annotations

import copy
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
        / "hip_dispatch_floor.py"
    )
    spec = importlib.util.spec_from_file_location("micro_hip_dispatch_floor", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_row(
    count: int,
    *,
    grid_blocks: int = 1,
    timing_mode: str = "serial_latency",
) -> dict:
    scale = float(count)
    return {
        "node_count": count,
        "grid_blocks": grid_blocks,
        "timing_mode": timing_mode,
        "submission_strategy": (
            "hip_graph" if timing_mode == "serial_latency" else "multi_stream"
        ),
        "stream_count": 1 if timing_mode == "serial_latency" else min(4, count),
        "single_gpu_samples_us": [10.0, 11.0, 12.0],
        "single_host_samples_us": [14.0, 15.0, 16.0],
        "burst_gpu_samples_us": [10.0 * scale, 11.0 * scale, 12.0 * scale],
        "burst_host_samples_us": [14.0 * scale, 15.0 * scale, 16.0 * scale],
        "single_correctness_pass": True,
        "burst_correctness_pass": True,
        "correctness_mismatches": 0,
    }


def _legacy_artifact(timing_mode: str = "serial_latency") -> dict:
    rows = [
        _raw_row(1, timing_mode=timing_mode),
        _raw_row(4, timing_mode=timing_mode),
    ]
    return {
        "run_tag": "hip-dispatch-floor-v2-raw",
        "status": "diagnostic",
        "hardware": {"gpu": "AMD Radeon Pro W7900", "arch": "gfx1100"},
        "software": {"python": "3.12.0", "platform": "Linux", "hipcc_version": "HIP clang"},
        "config": {
            "counts": [1, 4],
            "n_elements": 256,
            "local_size_x": 256,
            "reps": 3,
            "warmup": 2,
            "kernels": ["tiny", "wide"],
            "timing_mode": timing_mode,
            "independent_streams": 4,
            "grid_sweep": [128],
            "grid_sweep_count": 4,
            "target_nodes_per_batch": 5000,
            "method": "fixture",
        },
        "rows": rows,
        "rows_by_kernel": {"tiny": rows},
        "grid_sweep": [
            _raw_row(4, grid_blocks=128, timing_mode=timing_mode)
        ],
        "hip_only_wide_diagnostics": [{"node_count": 4, "diagnostic": True}],
    }


def _environment() -> dict:
    return {
        "schema_version": 1,
        "kind": "hipengine_micro_environment",
        "repo": {
            "root": "/repo",
            "branch": "benchmarks-micro",
            "commit": "a" * 40,
            "dirty": False,
            "status_short": [],
        },
        "devices": {"rocminfo_name_gfx_lines": ["Name: gfx1100"]},
        "commands": {},
        "host": {},
    }


def _normalize(module, timing_mode: str = "serial_latency") -> dict:
    return module.normalize_legacy_dispatch_result(
        _legacy_artifact(timing_mode),
        environment=_environment(),
        wrapper_command=["python3", "benchmarks/micro/runners/hip_dispatch_floor.py"],
        legacy_command=["python3", "scripts/graph_node_microbench.py"],
        source_hash="sha256:test",
    )


def test_default_timing_mode_is_serial_latency() -> None:
    module = _load_runner_module()

    assert module.parse_args([]).timing_mode == "serial_latency"


def _vulkan_result_from_hip(module, hip: dict) -> dict:
    result = copy.deepcopy(hip)
    result["backend"] = "vulkan"
    result["hardware"] = {"gpu_name": "RADV fixture", "gfx_arch": "gfx1100"}
    for row in result["measurements"]["rows"]:
        row["dispatch_count"] = row.pop("node_count")
        row["dependency_contract"] = module.timing_contract.make_dependency_contract(
            timing_mode=row["timing_mode"],
            backend="vulkan",
            validation_status="pass",
        )
        row["submission"] = module.timing_contract.make_submission(
            strategy="vulkan_command_buffer",
            queue_or_stream_count=1,
            recording_in_timed_region=False,
        )
    return result


def test_normalize_dispatch_result_shapes_v2_serial_contract() -> None:
    module = _load_runner_module()
    result = _normalize(module)

    assert result["schema_version"] == 2
    assert result["kind"] == "hipengine_micro_result"
    assert result["bench"] == "dispatch_grid_floor"
    assert result["backend"] == "hip"
    assert result["classification"] == "runtime_dispatch"
    assert result["hardware"] == {"gpu_name": "AMD Radeon Pro W7900", "gfx_arch": "gfx1100"}
    assert result["source"]["commit"] == "a" * 40
    assert result["correctness"]["status"] == "pass"
    assert result["parameters"]["timing_mode"] == "serial_latency"
    row = result["measurements"]["count_rows"][1]
    module.timing_contract.validate_timed_row(row, expected_repetitions=4)
    assert row["sweep"] == "count"
    assert row["submission"]["strategy"] == "hip_graph"
    assert row["timing"]["burst"]["logical_iterations"] == 4
    assert row["timing"]["burst"]["dispatches_per_iteration"] == 1
    assert row["correctness"]["synchronization"]["barrier_count"] == 0
    assert row["timing"]["burst"]["gpu_elapsed"]["sequence_us"]["median"] == 44.0
    assert result["measurements"]["grid_sweep_rows"][0]["grid_blocks"] == 128
    assert result["measurements"]["hip_only_wide_diagnostics"][0]["diagnostic"] is True
    json.dumps(result, allow_nan=False)


def test_normalize_independent_contract_validates_all_outputs() -> None:
    module = _load_runner_module()
    result = _normalize(module, "independent_throughput")

    row = result["measurements"]["count_rows"][1]
    module.timing_contract.validate_timed_row(row, expected_repetitions=4)
    assert row["submission"]["strategy"] == "multi_stream"
    assert row["submission"]["queue_or_stream_count"] == 4
    assert row["timing"]["burst"]["logical_iterations"] == 4
    assert row["timing"]["burst"]["dispatches_per_iteration"] == 1
    assert row["correctness"]["timed_sequence"]["coverage"] == "all_dispatches"
    assert row["timing"]["burst"]["gpu_elapsed"]["per_iteration_us"]["median"] == 11.0


def test_serial_comparison_allows_pre_recorded_host_wall() -> None:
    module = _load_runner_module()
    hip = _normalize(module)
    vulkan = _vulkan_result_from_hip(module, hip)

    comparison = module.build_comparison(hip, vulkan, command=["compare"])

    assert comparison["schema_version"] == 2
    assert comparison["comparisons"]
    assert all(
        row["host_wall"]["status"] == "ok"
        for row in comparison["comparisons"]
    )
    assert comparison["performance_claim"] is True
    assert comparison["sources"]["hip"]["source_hash"] == "sha256:test"
    assert comparison["sources"]["vulkan"]["source_hash"] == "sha256:test"


def test_independent_comparison_rejects_unmatched_host_submission() -> None:
    module = _load_runner_module()
    hip = _normalize(module, "independent_throughput")
    vulkan = _vulkan_result_from_hip(module, hip)

    comparison = module.build_comparison(hip, vulkan, command=["compare"])

    assert all(
        row["gpu_elapsed"]["status"] == "ok"
        for row in comparison["comparisons"]
    )
    assert all(
        row["host_wall"]["status"] == "not_comparable_submission_contract"
        for row in comparison["comparisons"]
    )


def test_native_dispatch_warmup_is_exact_and_storage_covers_it() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "microbench" / "graph_node_microbench.hip"
    ).read_text()
    wrapper = (root / "scripts" / "graph_node_microbench.py").read_text()

    assert "timer.run_and_wait(static_cast<uint32_t>(warmup), launch);" in source
    assert "args.grid_sweep_count if grid_sweep else 1, args.warmup, 1" in wrapper


def test_comparison_rejects_unmatched_row_shapes() -> None:
    module = _load_runner_module()
    hip = _normalize(module)
    vulkan = _vulkan_result_from_hip(module, hip)
    vulkan["measurements"]["rows"].pop()

    with pytest.raises(ValueError, match="row shapes"):
        module.build_comparison(hip, vulkan, command=["compare"])


def test_comparison_rejects_identically_truncated_requested_matrix() -> None:
    module = _load_runner_module()
    hip = _normalize(module)
    vulkan = _vulkan_result_from_hip(module, hip)
    hip["measurements"]["rows"].pop()
    vulkan["measurements"]["rows"].pop()

    with pytest.raises(ValueError, match="requested matrix"):
        module.build_comparison(hip, vulkan, command=["compare"])


def test_comparison_rejects_duplicate_rows() -> None:
    module = _load_runner_module()
    hip = _normalize(module)
    vulkan = _vulkan_result_from_hip(module, hip)
    hip["measurements"]["rows"].append(
        copy.deepcopy(hip["measurements"]["rows"][0])
    )

    with pytest.raises(ValueError, match="duplicate hip"):
        module.build_comparison(hip, vulkan, command=["compare"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "legacy_result", "result kind"),
        ("bench", "other", "benchmark identity"),
        ("classification", "compiler_aco", "classification"),
    ],
)
def test_comparison_rejects_wrong_result_identity(
    field: str, value: str, message: str
) -> None:
    module = _load_runner_module()
    hip = _normalize(module)
    vulkan = _vulkan_result_from_hip(module, hip)
    vulkan[field] = value

    with pytest.raises(ValueError, match=message):
        module.build_comparison(hip, vulkan, command=["compare"])


def test_comparison_rejects_unmatched_workload_parameters() -> None:
    module = _load_runner_module()
    hip = _normalize(module)
    vulkan = _vulkan_result_from_hip(module, hip)
    vulkan["parameters"]["counts"] = [1]

    with pytest.raises(ValueError, match="counts"):
        module.build_comparison(hip, vulkan, command=["compare"])


def test_comparison_records_nonclaiming_dirty_and_commit_mismatch() -> None:
    module = _load_runner_module()
    hip = _normalize(module)
    vulkan = _vulkan_result_from_hip(module, hip)
    hip["source"]["dirty"] = True
    vulkan["source"]["commit"] = "b" * 40

    comparison = module.build_comparison(hip, vulkan, command=["compare"])

    assert comparison["performance_claim"] is False
    assert comparison["claim_gate"] == {
        "commit_match": False,
        "clean_sources": False,
        "correctness_pass": True,
    }


def test_normalize_rejects_stale_artifact_without_timing_mode() -> None:
    module = _load_runner_module()
    legacy = _legacy_artifact()
    legacy["config"].pop("timing_mode")

    with pytest.raises(ValueError, match="timing mode"):
        module.normalize_legacy_dispatch_result(
            legacy,
            environment=_environment(),
            wrapper_command=["wrapper"],
            legacy_command=None,
            source_hash="sha256:test",
        )


def test_normalize_can_reference_external_environment() -> None:
    module = _load_runner_module()
    result = module.normalize_legacy_dispatch_result(
        _legacy_artifact(),
        environment=_environment(),
        wrapper_command=["wrapper"],
        legacy_command=None,
        source_hash="sha256:test",
        gfx_arch="gfx1151",
        hardware_gpu="AMD Radeon 8060S",
        environment_ref="benchmarks/micro/results/gfx1151/env.json",
    )

    assert result["hardware"] == {"gpu_name": "AMD Radeon 8060S", "gfx_arch": "gfx1151"}
    assert result["environment_ref"] == "benchmarks/micro/results/gfx1151/env.json"
    assert "environment" not in result
