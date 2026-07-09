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
        / "geometry_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("micro_geometry_sweep", path)
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


def _metric(clock: str, sequence_us: float, logical_iterations: int) -> dict:
    return {
        "status": "ok",
        "clock": clock,
        "sequence_us": _stats(sequence_us),
        "per_iteration_us": _stats(sequence_us / logical_iterations),
    }


def _timed_contract(backend: str, median_us: float, reps: int = 5) -> dict:
    gpu_clock = "hip_event" if backend == "hip" else "vulkan_timestamp"
    ordering = "hip_stream_order" if backend == "hip" else "vulkan_compute_barrier"
    return {
        "timing_mode": "serial_latency",
        "dependency_contract": {
            "work_dependency": "chained",
            "inter_dispatch_ordering": ordering,
            "output_partitioning": "chained_shared",
            "validation_status": "pass",
        },
        "submission": {
            "strategy": "direct" if backend == "hip" else "vulkan_command_buffer",
            "recording_in_timed_region": False,
            "submit_in_host_wall": True,
            "completion_in_host_wall": True,
            "queue_or_stream_count": 1,
        },
        "timing": {
            "single": {
                "logical_iterations": 1,
                "dispatches_per_iteration": 1,
                "gpu_elapsed": _metric(gpu_clock, median_us + 1.0, 1),
                "host_wall": _metric("steady_clock", median_us + 4.0, 1),
            },
            "burst": {
                "logical_iterations": reps,
                "dispatches_per_iteration": 1,
                "gpu_elapsed": _metric(gpu_clock, median_us * reps, reps),
                "host_wall": _metric("steady_clock", (median_us + 2.0) * reps, reps),
            },
        },
        "correctness": {
            "single_dispatch": {"status": "pass", "oracle": "CPU reference"},
            "timed_sequence": {
                "status": "pass",
                "oracle": "CPU reference with sequence tag",
                "logical_iterations": reps,
                "coverage": "chained_final_state",
            },
            "synchronization": {
                "status": "pass",
                "method": ordering,
                "barrier_count": 0 if backend == "hip" else reps - 1,
            },
        },
    }


def _raw_artifact(backend: str) -> dict:
    return {
        "run_tag": f"{backend}-geometry-sweep",
        "status": "diagnostic",
        "backend": backend,
        "hardware": {
            "device_name": "AMD Radeon Test",
            "gcn_arch_name": "gfx1100" if backend == "hip" else "",
            "device_id": 0x744C,
        },
        "config": {
            "k_list": [512, 2048],
            "rows_list": [1],
            "workgroups": [32, 64],
            "body_repeats": 8,
            "reps": 5,
            "warmup": 2,
            "samples": 3,
            "timing_mode": "serial_latency",
            "independent_streams": 4 if backend == "hip" else None,
            "hip_workgroup_specialization": "fixed" if backend == "hip" else None,
        },
        "rows": [
            {
                "k": 512,
                "rows": 1,
                "workgroup_size": 32,
                "body_repeats": 8,
                "median_us": 10.0 if backend == "hip" else 8.0,
                "p05_us": 9.0,
                "p95_us": 11.0,
                "gflops": 1.0,
                "max_abs": 0.0,
                "max_rel": 0.0,
                "correctness_pass": True,
                **_timed_contract(backend, 10.0 if backend == "hip" else 8.0),
            },
            {
                "k": 512,
                "rows": 1,
                "workgroup_size": 64,
                "body_repeats": 8,
                "median_us": 7.0 if backend == "hip" else 6.0,
                "p05_us": 6.5,
                "p95_us": 7.5,
                "gflops": 2.0,
                "max_abs": 0.0,
                "max_rel": 0.0,
                "correctness_pass": True,
                **_timed_contract(backend, 7.0 if backend == "hip" else 6.0),
            },
            {
                "k": 2048,
                "rows": 1,
                "workgroup_size": 64,
                "body_repeats": 8,
                "median_us": 20.0 if backend == "hip" else 15.0,
                "p05_us": 19.0,
                "p95_us": 21.0,
                "gflops": 3.0,
                "max_abs": 0.001,
                "max_rel": 0.0,
                "correctness_pass": True,
                **_timed_contract(backend, 20.0 if backend == "hip" else 15.0),
            },
        ],
    }


def _environment() -> dict:
    return {
        "schema_version": 1,
        "kind": "hipengine_micro_environment",
        "repo": {
            "root": "/repo",
            "branch": "benchmarks-micro",
            "commit": "c" * 40,
            "dirty": False,
            "status_short": [],
        },
        "devices": {"rocminfo_name_gfx_lines": ["Name: gfx1100"]},
        "commands": {},
        "host": {},
    }


def test_normalize_geometry_result_shapes_schema() -> None:
    module = _load_runner_module()

    result = module.normalize_raw_result(
        _raw_artifact("hip"),
        backend="hip",
        environment=_environment(),
        wrapper_command=["python3", "geometry_sweep.py"],
        harness_command=["/tmp/hip_geometry_sweep"],
        build_command=["hipcc", "hip_geometry_sweep.hip"],
        shader_command=None,
        source_hash="sha256:test",
    )

    assert result["schema_version"] == 2
    assert result["kind"] == "hipengine_micro_result"
    assert result["bench"] == "f32_gemv_geometry_sweep"
    assert result["backend"] == "hip"
    assert result["classification"] == "geometry"
    assert result["hardware"]["gfx_arch"] == "gfx1100"
    assert result["source"]["commit"] == "c" * 40
    assert result["correctness"]["status"] == "pass"
    assert result["correctness"]["max_abs"] == 0.001
    assert result["timing"]["unit"] == "us_per_dispatch"
    assert result["timing"]["primary_domain"] == "gpu_elapsed"
    assert result["timing"]["median"] == 20.0
    assert result["timing"]["primary"]["k"] == 2048
    assert result["timing"]["primary"]["workgroup_size"] == 64
    assert result["isa"]["workgroup_size"] == 64
    assert result["isa"]["lds_bytes"] == 256
    assert result["parameters"]["build_command"] == ["hipcc", "hip_geometry_sweep.hip"]
    assert "environment" in result
    json.dumps(result, allow_nan=False)


def test_geometry_fixed_workgroup_merge_and_schema() -> None:
    module = _load_runner_module()
    raw32 = _raw_artifact("hip")
    raw32["config"]["workgroups"] = [32]
    raw32["rows"] = [row for row in raw32["rows"] if row["workgroup_size"] == 32]
    raw64 = _raw_artifact("hip")
    raw64["config"]["workgroups"] = [64]
    raw64["rows"] = [row for row in raw64["rows"] if row["workgroup_size"] == 64]

    merged = module._merge_fixed_hip_raw_results([raw32, raw64], workgroups=[32, 64])
    assert merged["config"]["hip_workgroup_specialization"] == "fixed"
    assert merged["config"]["hip_fixed_workgroup_sizes"] == [32, 64]
    assert [row["workgroup_size"] for row in merged["rows"]] == [32, 64, 64]

    result = module.normalize_raw_result(
        merged,
        backend="hip",
        environment=_environment(),
        wrapper_command=["python3", "geometry_sweep.py"],
        harness_command=[["/tmp/fixed32"], ["/tmp/fixed64"]],
        build_command=[["hipcc", "fixed32"], ["hipcc", "fixed64"]],
        shader_command=None,
        source_hash="sha256:test",
    )
    assert result["parameters"]["hip_workgroup_specialization"] == "fixed"
    assert result["parameters"]["hip_fixed_workgroup_sizes"] == [32, 64]
    json.dumps(result, allow_nan=False)


def test_geometry_runtime_workgroups_are_not_comparable_to_vulkan() -> None:
    module = _load_runner_module()

    args = module.parse_args(["--backend", "hip", "--hip-workgroup-specialization", "fixed"])
    assert args.hip_workgroup_specialization == "fixed"

    try:
        module.parse_args(["--backend", "vulkan", "--hip-workgroup-specialization", "runtime"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit for Vulkan comparison against runtime HIP workgroups")


def test_geometry_comparison_matches_rows() -> None:
    module = _load_runner_module()
    env = _environment()
    hip = module.normalize_raw_result(
        _raw_artifact("hip"),
        backend="hip",
        environment=env,
        wrapper_command=["python3", "geometry_sweep.py"],
        harness_command=None,
        build_command=None,
        shader_command=None,
        source_hash="sha256:hip",
        environment_ref="env.json",
    )
    vulkan = module.normalize_raw_result(
        _raw_artifact("vulkan"),
        backend="vulkan",
        environment=env,
        wrapper_command=["python3", "geometry_sweep.py"],
        harness_command=None,
        build_command=None,
        shader_command=["glslc", "geometry_sweep.comp"],
        source_hash="sha256:vulkan",
        environment_ref="env.json",
    )

    comparison = module.build_comparison(
        hip,
        vulkan,
        command=["python3", "geometry_sweep.py", "--compare", "hip.json", "vulkan.json"],
    )

    assert comparison["kind"] == "hipengine_micro_comparison"
    assert comparison["classification"] == "diagnostic_unclassified"
    assert len(comparison["matched_rows"]) == 3
    reference = [
        row
        for row in comparison["matched_rows"]
        if row["k"] == 2048 and row["workgroup_size"] == 64
    ][0]
    assert reference["vulkan_vs_hip_gpu_burst_speedup"] == 20.0 / 15.0
    summary = [row for row in comparison["shape_summary"] if row["k"] == 512][0]
    assert summary["best_hip_workgroup"] == 64
    assert summary["best_vulkan_workgroup"] == 64
    burst = [
        row
        for row in comparison["comparisons"]
        if row["k"] == 2048 and row["control"] == "burst"
    ][0]
    assert burst["gpu_elapsed"]["status"] == "ok"
    assert burst["host_wall"]["status"] == "not_comparable_submission_contract"
    json.dumps(comparison, allow_nan=False)


def test_geometry_comparison_rejects_cross_mode_rows() -> None:
    module = _load_runner_module()
    env = _environment()
    hip = module.normalize_raw_result(
        _raw_artifact("hip"),
        backend="hip",
        environment=env,
        wrapper_command=["geometry_sweep.py"],
        harness_command=None,
        build_command=None,
        shader_command=None,
        source_hash="sha256:hip",
    )
    raw_vulkan = _raw_artifact("vulkan")
    raw_vulkan["config"]["timing_mode"] = "independent_throughput"
    for row in raw_vulkan["rows"]:
        row["timing_mode"] = "independent_throughput"
        row["dependency_contract"].update(
            {
                "work_dependency": "independent",
                "inter_dispatch_ordering": "none",
                "output_partitioning": "disjoint",
            }
        )
        row["correctness"]["timed_sequence"]["coverage"] = "all_dispatches"
    vulkan = module.normalize_raw_result(
        raw_vulkan,
        backend="vulkan",
        environment=env,
        wrapper_command=["geometry_sweep.py"],
        harness_command=None,
        build_command=None,
        shader_command=None,
        source_hash="sha256:vulkan",
    )

    try:
        module.build_comparison(hip, vulkan, command=["geometry_sweep.py", "--compare"])
    except ValueError as exc:
        assert "timing modes do not match" in str(exc)
    else:
        raise AssertionError("expected cross-mode comparison rejection")
