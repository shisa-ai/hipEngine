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
        / "sampler_argmax.py"
    )
    spec = importlib.util.spec_from_file_location("micro_sampler_argmax", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment() -> dict:
    return {
        "repo": {
            "root": "/repo",
            "branch": "main",
            "commit": "a" * 40,
            "dirty": False,
        },
        "devices": {"rocminfo_name_gfx_lines": ["Name: gfx1151"]},
    }


def _raw_row(*, backend: str, timing_mode: str) -> dict:
    return {
        "rows": 1,
        "vocab": 256,
        "workgroup_size": 64,
        "top_k": 1,
        "bytes_per_dispatch": 1024.0,
        "comparisons_per_dispatch": 256.0,
        "timing_mode": timing_mode,
        "stream_count": 2 if timing_mode == "independent_throughput" else 1,
        "single_gpu_samples_us": [10.0, 12.0],
        "single_host_samples_us": [20.0, 22.0],
        "burst_gpu_samples_us": [32.0, 36.0],
        "burst_host_samples_us": [44.0, 48.0],
        "single_mismatches": 0,
        "burst_mismatches": 0,
        "mismatches": 0,
        "max_abs": 0.0,
        "correctness_pass": True,
        "gpu_timestamps_supported": backend == "vulkan",
        "raw_config": {"reps": 4, "warmup": 2, "timing_mode": timing_mode},
        "hardware": {"device_name": "Radeon 8060S Graphics", "gcn_arch_name": "gfx1151"},
    }


def _result(module, *, backend: str, timing_mode: str) -> dict:
    return module._normalize_result(
        backend=backend,
        raw_rows=[_raw_row(backend=backend, timing_mode=timing_mode)],
        isa_by_variant={(64, 1): {"instruction_count": 10, "waitcnt_count": 2}},
        environment=_environment(),
        source_hash="sha256:test",
        wrapper_command=["sampler_argmax.py", "--backend", backend],
        commands=[],
        hardware_gpu="Radeon 8060S Graphics",
        gfx_arch="gfx1151",
        environment_ref=None,
    )


@pytest.mark.parametrize("timing_mode", ["serial_latency", "independent_throughput"])
def test_normalize_emits_valid_v2_timing_contract(timing_mode: str) -> None:
    module = _load_runner_module()
    result = _result(module, backend="hip", timing_mode=timing_mode)
    row = result["measurements"]["rows"][0]

    assert result["schema_version"] == 2
    assert result["correctness"]["status"] == "pass"
    module.timing_contract.validate_timed_row(row, expected_repetitions=4)
    assert row["timing"]["single"]["logical_iterations"] == 1
    assert row["timing"]["burst"]["logical_iterations"] == 4
    assert row["timing"]["burst"]["gpu_elapsed"]["per_iteration_us"]["median"] == 8.5
    expected_partition = "disjoint" if timing_mode == "independent_throughput" else "chained_shared"
    assert row["dependency_contract"]["output_partitioning"] == expected_partition


def test_comparison_separates_gpu_ratio_and_unmatched_host_wall() -> None:
    module = _load_runner_module()
    hip = _result(module, backend="hip", timing_mode="serial_latency")
    vulkan = _result(module, backend="vulkan", timing_mode="serial_latency")

    comparison = module.build_comparison(hip, vulkan, command=["compare"])

    assert comparison["schema_version"] == 2
    assert len(comparison["comparisons"]) == 2
    assert comparison["comparisons"][1]["gpu_elapsed"]["status"] == "ok"
    assert comparison["comparisons"][1]["host_wall"]["status"] == (
        "not_comparable_submission_contract"
    )


def test_serial_barrier_count_is_backend_specific() -> None:
    module = _load_runner_module()
    hip = _result(module, backend="hip", timing_mode="serial_latency")
    vulkan = _result(module, backend="vulkan", timing_mode="serial_latency")

    assert hip["measurements"]["rows"][0]["correctness"]["synchronization"][
        "barrier_count"
    ] == 0
    assert vulkan["measurements"]["rows"][0]["correctness"]["synchronization"][
        "barrier_count"
    ] == 3


def test_comparison_rejects_cross_mode_results() -> None:
    module = _load_runner_module()
    hip = _result(module, backend="hip", timing_mode="serial_latency")
    vulkan = _result(module, backend="vulkan", timing_mode="independent_throughput")

    with pytest.raises(ValueError, match="timing modes do not match"):
        module.build_comparison(hip, vulkan, command=["compare"])


def test_default_timing_mode_is_serial_latency() -> None:
    module = _load_runner_module()
    args = module.parse_args(["--backend", "hip"])

    assert args.timing_mode == "serial_latency"
    assert args.independent_streams == 4
