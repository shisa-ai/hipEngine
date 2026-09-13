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
        / "vulkan_dispatch_floor.py"
    )
    spec = importlib.util.spec_from_file_location("micro_vulkan_dispatch_floor", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_row(count: int, *, grid_blocks: int = 1) -> dict:
    scale = float(count)
    return {
        "dispatch_count": count,
        "grid_blocks": grid_blocks,
        "burst_us_median": 8.0 * scale,
        "us_per_dispatch": 8.0,
        "burst_us_min": 7.0 * scale,
        "reps": 3,
        "single_gpu_samples_us": [7.0, 8.0, 9.0],
        "single_host_samples_us": [11.0, 12.0, 13.0],
        "burst_gpu_samples_us": [7.0 * scale, 8.0 * scale, 9.0 * scale],
        "burst_host_samples_us": [11.0 * scale, 12.0 * scale, 13.0 * scale],
        "single_correctness_pass": True,
        "burst_correctness_pass": True,
        "correctness_mismatches": 0,
        "gpu_timestamps_supported": True,
    }


def _legacy_artifact(timing_mode: str = "serial_latency") -> dict:
    return {
        "run_tag": "vulkan-dispatch-floor",
        "status": "diagnostic",
        "hardware": {
            "device_name": "AMD Radeon Pro W7900 (RADV NAVI31)",
            "vendor_id": 4098,
            "device_id": 29772,
            "device_type": 2,
            "api_version": "1.3.0",
            "driver_version_raw": 100663296,
            "queue_family": 0,
            "output_device_local": True,
            "output_memory_type": 0,
            "readback_memory_type": 2,
        },
        "config": {
            "counts": [1, 4],
            "grid_sweep": [128],
            "grid_sweep_count": 4,
            "n_elements": 256,
            "reps": 3,
            "warmup": 2,
            "timing_mode": timing_mode,
            "local_size_x": 256,
            "method": "fixture",
        },
        "rows": [_raw_row(1), _raw_row(4)],
        "grid_sweep_rows": [_raw_row(4, grid_blocks=128)],
    }


def _environment() -> dict:
    return {
        "schema_version": 1,
        "kind": "hipengine_micro_environment",
        "repo": {
            "root": "/repo",
            "branch": "benchmarks-micro",
            "commit": "b" * 40,
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
        wrapper_command=["python3", "benchmarks/micro/runners/vulkan_dispatch_floor.py"],
        legacy_command=["/tmp/vulkan_dispatch_floor", "--json", "/tmp/raw.json"],
        shader_command=["glslc", "dispatch_floor.comp"],
        harness_build_command=["c++", "vulkan_dispatch_floor.cpp", "-lvulkan"],
        source_hash="sha256:test",
    )


def test_default_timing_mode_is_serial_latency() -> None:
    module = _load_runner_module()

    assert module.parse_args([]).timing_mode == "serial_latency"


def test_normalize_vulkan_dispatch_result_shapes_v2_serial_contract() -> None:
    module = _load_runner_module()
    result = _normalize(module)

    assert result["schema_version"] == 2
    assert result["kind"] == "hipengine_micro_result"
    assert result["bench"] == "dispatch_grid_floor"
    assert result["backend"] == "vulkan"
    assert result["classification"] == "runtime_dispatch"
    assert result["hardware"]["gpu_name"] == "AMD Radeon Pro W7900 (RADV NAVI31)"
    assert result["hardware"]["gfx_arch"] == "gfx1100"
    assert result["hardware"]["device_id"] == "0x744c"
    assert result["parameters"]["vulkan_hardware"]["output_device_local"] is True
    assert result["source"]["commit"] == "b" * 40
    assert result["correctness"]["status"] == "pass"
    row = result["measurements"]["count_rows"][1]
    module.timing_contract.validate_timed_row(row, expected_repetitions=4)
    assert row["sweep"] == "count"
    assert row["dependency_contract"]["inter_dispatch_ordering"] == "vulkan_compute_barrier"
    assert row["correctness"]["synchronization"]["barrier_count"] == 3
    assert row["timing"]["burst"]["logical_iterations"] == 4
    assert row["timing"]["burst"]["dispatches_per_iteration"] == 1
    assert row["timing"]["burst"]["gpu_elapsed"]["sequence_us"]["median"] == 32.0
    assert result["measurements"]["grid_sweep_rows"][0]["grid_blocks"] == 128
    json.dumps(result, allow_nan=False)


def test_normalize_vulkan_independent_contract_uses_disjoint_outputs() -> None:
    module = _load_runner_module()
    result = _normalize(module, "independent_throughput")

    row = result["measurements"]["count_rows"][1]
    module.timing_contract.validate_timed_row(row, expected_repetitions=4)
    assert row["dependency_contract"]["work_dependency"] == "independent"
    assert row["dependency_contract"]["output_partitioning"] == "disjoint"
    assert row["correctness"]["timed_sequence"]["coverage"] == "all_dispatches"
    assert row["correctness"]["synchronization"]["barrier_count"] == 0
    assert row["timing"]["burst"]["gpu_elapsed"]["per_iteration_us"]["median"] == 8.0


def test_normalize_rejects_missing_exact_burst_samples() -> None:
    module = _load_runner_module()
    legacy = _legacy_artifact()
    legacy["rows"][0]["burst_gpu_samples_us"] = [1.0]

    with pytest.raises(ValueError, match="exactly 3"):
        module.normalize_legacy_dispatch_result(
            legacy,
            environment=_environment(),
            wrapper_command=["wrapper"],
            legacy_command=None,
            shader_command=None,
            harness_build_command=None,
            source_hash="sha256:test",
        )


def test_vulkan_dispatch_warmup_is_one_sequence_and_storage_covers_it() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "runners"
        / "vulkan_dispatch_floor.cpp"
    ).read_text()

    assert "args.warmup,\n        grid_blocks" in source
    assert "std::max(max_dispatch_count, std::max(args.warmup, 1u))" in source


def test_normalize_vulkan_dispatch_result_can_reference_environment() -> None:
    module = _load_runner_module()
    result = module.normalize_legacy_dispatch_result(
        _legacy_artifact(),
        environment=_environment(),
        wrapper_command=["wrapper"],
        legacy_command=None,
        shader_command=None,
        harness_build_command=None,
        source_hash="sha256:test",
        gfx_arch="gfx1151",
        hardware_gpu="Radeon 8060S Graphics",
        environment_ref="benchmarks/micro/results/gfx1151/strix-halo/environment.json",
    )

    assert result["hardware"]["gpu_name"] == "Radeon 8060S Graphics"
    assert result["hardware"]["gfx_arch"] == "gfx1151"
    assert result["environment_ref"] == "benchmarks/micro/results/gfx1151/strix-halo/environment.json"
    assert "environment" not in result
