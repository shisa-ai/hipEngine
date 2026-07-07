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
        / "hip_dispatch_floor.py"
    )
    spec = importlib.util.spec_from_file_location("micro_hip_dispatch_floor", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_artifact() -> dict:
    tiny_rows = [
        {
            "node_count": 1,
            "direct": {"us_per_node": 11.0, "burst_us_median": 11.0, "burst_us_min": 10.0},
            "graph": {
                "steady_us_per_node": 12.0,
                "batch_us_median": 60.0,
                "replay_latency_us_median": 12.0,
            },
            "graph_speedup_vs_direct": 0.9166666667,
        },
        {
            "node_count": 941,
            "direct": {"us_per_node": 5.7, "burst_us_median": 5363.7, "burst_us_min": 5320.0},
            "graph": {
                "steady_us_per_node": 5.6,
                "batch_us_median": 5270.0,
                "replay_latency_us_median": 5300.0,
            },
            "graph_speedup_vs_direct": 1.0178571429,
        },
    ]
    wide_rows = [
        {
            "node_count": 941,
            "direct": {"us_per_node": 5.8, "burst_us_median": 5457.8, "burst_us_min": 5400.0},
            "graph": {
                "steady_us_per_node": 5.9,
                "batch_us_median": 5551.9,
                "replay_latency_us_median": 5560.0,
            },
            "graph_speedup_vs_direct": 0.9830508475,
        }
    ]
    return {
        "run_tag": "m16.2-arg-scaling-graph-vs-native",
        "status": "diagnostic",
        "hardware": {"gpu": "AMD Radeon Pro W7900", "arch": "gfx1100"},
        "software": {"python": "3.12.0", "platform": "Linux", "hipcc_version": "HIP clang"},
        "config": {
            "counts": [1, 941],
            "n_elements": 256,
            "reps": 7,
            "warmup": 2,
            "kernels": ["tiny", "wide"],
            "target_nodes_per_batch": 5000,
            "method": "fixture",
        },
        "rows": tiny_rows,
        "rows_by_kernel": {"tiny": tiny_rows, "wide": wide_rows},
        "grid_sweep": [
            {
                "grid_blocks": 128,
                "node_count": 941,
                "direct_us_per_node": 6.1,
                "graph_us_per_node": 6.0,
                "graph_speedup_vs_direct": 1.0166666667,
            }
        ],
        "verdict": {
            "kernel": "tiny",
            "reference_node_count": 941,
            "hip_floor_us_per_node": 5.6,
            "program": "fewer-larger-kernels",
        },
        "arg_scaling_verdict": {
            "reference_node_count": 941,
            "arg_marshal_delta_us": 0.1,
            "program": "per-launch cost ~arg-count-independent",
        },
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


def test_normalize_legacy_dispatch_result_shapes_micro_schema() -> None:
    module = _load_runner_module()

    result = module.normalize_legacy_dispatch_result(
        _legacy_artifact(),
        environment=_environment(),
        wrapper_command=["python3", "benchmarks/micro/runners/hip_dispatch_floor.py"],
        legacy_command=["python3", "scripts/graph_node_microbench.py"],
        source_hash="sha256:test",
    )

    assert result["schema_version"] == 1
    assert result["kind"] == "hipengine_micro_result"
    assert result["bench"] == "dispatch_grid_floor"
    assert result["backend"] == "hip"
    assert result["classification"] == "runtime_dispatch"
    assert result["hardware"] == {"gpu_name": "AMD Radeon Pro W7900", "gfx_arch": "gfx1100"}
    assert result["source"]["commit"] == "a" * 40
    assert result["source"]["dirty"] is False
    assert result["correctness"]["status"] == "not_applicable"
    assert result["timing"]["unit"] == "us_per_launch"
    assert result["timing"]["median"] == 5.7
    assert result["timing"]["warmup_iters"] == 2
    assert result["timing"]["measured_iters"] == 7
    assert result["timing"]["primary"]["node_count"] == 941
    assert result["timing"]["primary"]["graph_steady_us_per_node_median"] == 5.6
    assert result["measurements"]["grid_sweep"][0]["grid_blocks"] == 128
    assert result["parameters"]["legacy_command"] == ["python3", "scripts/graph_node_microbench.py"]
    assert "environment" in result
    assert "environment_ref" not in result
    json.dumps(result, allow_nan=False)


def test_normalize_can_reference_external_environment() -> None:
    module = _load_runner_module()

    result = module.normalize_legacy_dispatch_result(
        _legacy_artifact(),
        environment=_environment(),
        wrapper_command=["python3", "benchmarks/micro/runners/hip_dispatch_floor.py"],
        legacy_command=None,
        source_hash="sha256:test",
        gfx_arch="gfx1151",
        hardware_gpu="AMD Radeon 8060S",
        environment_ref="benchmarks/micro/results/gfx1151/env.json",
    )

    assert result["hardware"] == {"gpu_name": "AMD Radeon 8060S", "gfx_arch": "gfx1151"}
    assert result["environment_ref"] == "benchmarks/micro/results/gfx1151/env.json"
    assert "environment" not in result
    json.dumps(result, allow_nan=False)
