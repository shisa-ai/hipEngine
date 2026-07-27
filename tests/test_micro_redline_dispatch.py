from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "benchmarks" / "micro" / "redline_dispatch.py"
    spec = importlib.util.spec_from_file_location("micro_redline_dispatch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pack_kernarg_populates_explicit_and_hidden_geometry() -> None:
    module = _load_module()
    spec = {
        "kernarg_size": 96,
        "args": [
            {"offset": 0, "size": 8, "value_kind": "global_buffer"},
            {"offset": 8, "size": 4, "value_kind": "by_value"},
            {"offset": 16, "size": 4, "value_kind": "hidden_block_count_x"},
            {"offset": 28, "size": 2, "value_kind": "hidden_group_size_x"},
            {"offset": 80, "size": 2, "value_kind": "hidden_grid_dims"},
        ],
    }

    packed = module.pack_gmb_kernarg(
        spec,
        output_pointer=0x1234_5678_9ABC_DEF0,
        n=256,
        grid_blocks=128,
        block_size=256,
    )

    assert len(packed) == 96
    assert int.from_bytes(packed[0:8], "little") == 0x1234_5678_9ABC_DEF0
    assert int.from_bytes(packed[8:12], "little") == 256
    assert int.from_bytes(packed[16:20], "little") == 128
    assert int.from_bytes(packed[28:30], "little") == 256
    assert int.from_bytes(packed[80:82], "little") == 1


def test_dispatch_result_has_v2_pm4_contract_and_full_correctness() -> None:
    module = _load_module()
    rows = [
        module.make_dispatch_row(
            mode="serial_latency",
            sweep="count",
            count=50,
            grid_blocks=1,
            lane_count=1,
            warmup=5,
            gpu_single_us=[5.0, 5.2, 5.1],
            host_single_us=[8.0, 8.2, 8.1],
            gpu_burst_us=[60.0, 61.0, 59.0],
            host_burst_us=[70.0, 72.0, 71.0],
            single_correct=True,
            burst_correct=True,
        ),
        module.make_dispatch_row(
            mode="independent_throughput",
            sweep="grid",
            count=50,
            grid_blocks=128,
            lane_count=2,
            warmup=5,
            gpu_single_us=[5.0, 5.2, 5.1],
            host_single_us=[8.0, 8.2, 8.1],
            gpu_burst_us=[40.0, 41.0, 39.0],
            host_burst_us=[50.0, 52.0, 51.0],
            single_correct=True,
            burst_correct=True,
        ),
    ]
    result = module.build_dispatch_result(
        rows=rows,
        hardware={"gpu_name": "AMD Radeon Pro W7900", "gfx_arch": "gfx1100"},
        source={
            "repo": str(REPO_ROOT),
            "branch": "redline-integration-spike",
            "commit": "a" * 40,
            "dirty": False,
            "source_hash": "sha256:test",
        },
        command=["python3", "redline_dispatch.py"],
        environment_ref="/tmp/environment.json",
        redline_provenance={
            "checkout": {"commit": module.PINNED_REDLINE_COMMIT, "dirty": False},
            "execution_proof": {
                "api": "redline-capi",
                "native_hip_fallback_available": False,
                "profiled_retained_pm4_required": True,
                "radiowave_manifest_verified": True,
            },
        },
        parameters={"reps": 3, "warmup": 5},
    )

    assert result["schema_version"] == 2
    assert result["kind"] == "hipengine_micro_result"
    assert result["backend"] == "redline"
    assert result["correctness"] == {
        "status": "pass",
        "oracle": "exact every-element output after single and measured burst replay",
        "rows": 2,
    }
    serial, independent = result["measurements"]["rows"]
    assert serial["dependency_contract"]["inter_dispatch_ordering"] == "redline_rmw"
    assert independent["dependency_contract"]["inter_dispatch_ordering"] == "none"
    assert independent["submission"]["queue_or_stream_count"] == 2
    assert independent["timing"]["single"]["retained_lane_count"] == 1
    assert independent["timing"]["burst"]["retained_lane_count"] == 2
    assert independent["correctness"]["timed_sequence"]["coverage"] == "all_dispatches"
    sequence = independent["timing"]["burst"]["gpu_elapsed"]["sequence_us"]
    assert sequence["samples"] == 3
    assert sequence["median"] == 40.0
    assert sequence["min"] == 39.0
    assert sequence["max"] == 41.0
    json.dumps(result, allow_nan=False)


def test_dispatch_row_key_accepts_native_dispatch_count_alias() -> None:
    module = _load_module()

    assert module.dispatch_row_key(
        {
            "sweep": "grid",
            "dispatch_count": 941,
            "grid_blocks": 8192,
            "timing_mode": "serial_latency",
        }
    ) == ("grid", 941, 8192, "serial_latency")
