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

    assert result["schema_version"] == 1
    assert result["kind"] == "hipengine_micro_result"
    assert result["bench"] == "f32_gemv_geometry_sweep"
    assert result["backend"] == "hip"
    assert result["classification"] == "geometry"
    assert result["hardware"]["gfx_arch"] == "gfx1100"
    assert result["source"]["commit"] == "c" * 40
    assert result["correctness"]["status"] == "pass"
    assert result["correctness"]["max_abs"] == 0.001
    assert result["timing"]["unit"] == "us_per_dispatch"
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


def test_geometry_fixed_workgroup_arg_is_hip_only() -> None:
    module = _load_runner_module()

    args = module.parse_args(["--backend", "hip", "--hip-workgroup-specialization", "fixed"])
    assert args.hip_workgroup_specialization == "fixed"

    try:
        module.parse_args(["--backend", "vulkan", "--hip-workgroup-specialization", "fixed"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit for Vulkan fixed HIP workgroup flag")


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
    assert reference["vulkan_vs_hip_speedup"] == 20.0 / 15.0
    summary = [row for row in comparison["shape_summary"] if row["k"] == 512][0]
    assert summary["best_hip_workgroup"] == 64
    assert summary["best_vulkan_workgroup"] == 64
    json.dumps(comparison, allow_nan=False)
