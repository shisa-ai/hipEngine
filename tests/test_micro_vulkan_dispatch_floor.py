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
        / "vulkan_dispatch_floor.py"
    )
    spec = importlib.util.spec_from_file_location("micro_vulkan_dispatch_floor", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_artifact() -> dict:
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
        },
        "config": {
            "counts": [1, 941],
            "grid_sweep": [1, 128],
            "grid_sweep_count": 941,
            "n_elements": 256,
            "reps": 9,
            "warmup": 3,
            "local_size_x": 256,
            "method": "fixture",
        },
        "rows": [
            {
                "dispatch_count": 1,
                "grid_blocks": 1,
                "burst_us_median": 20.0,
                "us_per_dispatch": 20.0,
                "burst_us_min": 19.5,
                "reps": 9,
            },
            {
                "dispatch_count": 941,
                "grid_blocks": 1,
                "burst_us_median": 1882.0,
                "us_per_dispatch": 2.0,
                "burst_us_min": 1800.0,
                "reps": 9,
            },
        ],
        "grid_sweep_rows": [
            {
                "dispatch_count": 941,
                "grid_blocks": 128,
                "burst_us_median": 2823.0,
                "us_per_dispatch": 3.0,
                "burst_us_min": 2800.0,
                "reps": 9,
            }
        ],
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


def test_normalize_vulkan_dispatch_result_shapes_micro_schema() -> None:
    module = _load_runner_module()

    result = module.normalize_legacy_dispatch_result(
        _legacy_artifact(),
        environment=_environment(),
        wrapper_command=["python3", "benchmarks/micro/runners/vulkan_dispatch_floor.py"],
        legacy_command=["/tmp/vulkan_dispatch_floor", "--json", "/tmp/raw.json"],
        shader_command=["glslc", "dispatch_floor.comp"],
        harness_build_command=["c++", "vulkan_dispatch_floor.cpp", "-lvulkan"],
        source_hash="sha256:test",
    )

    assert result["schema_version"] == 1
    assert result["kind"] == "hipengine_micro_result"
    assert result["bench"] == "dispatch_grid_floor"
    assert result["backend"] == "vulkan"
    assert result["classification"] == "runtime_dispatch"
    assert result["hardware"]["gpu_name"] == "AMD Radeon Pro W7900 (RADV NAVI31)"
    assert result["hardware"]["gfx_arch"] == "gfx1100"
    assert result["hardware"]["device_id"] == "0x744c"
    assert result["source"]["commit"] == "b" * 40
    assert result["correctness"]["status"] == "not_applicable"
    assert result["timing"]["unit"] == "us_per_dispatch"
    assert result["timing"]["median"] == 2.0
    assert result["timing"]["warmup_iters"] == 3
    assert result["timing"]["measured_iters"] == 9
    assert result["timing"]["primary"]["dispatch_count"] == 941
    assert result["parameters"]["shader_command"] == ["glslc", "dispatch_floor.comp"]
    assert result["measurements"]["grid_sweep_rows"][0]["grid_blocks"] == 128
    assert "environment" in result
    assert "environment_ref" not in result
    json.dumps(result, allow_nan=False)


def test_normalize_vulkan_dispatch_result_can_reference_environment() -> None:
    module = _load_runner_module()

    result = module.normalize_legacy_dispatch_result(
        _legacy_artifact(),
        environment=_environment(),
        wrapper_command=["python3", "benchmarks/micro/runners/vulkan_dispatch_floor.py"],
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
    json.dumps(result, allow_nan=False)
