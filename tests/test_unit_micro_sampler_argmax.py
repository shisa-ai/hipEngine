from __future__ import annotations

import importlib.util
from copy import deepcopy
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
        "raw_config": {
            "rows": 1,
            "vocab": 256,
            "workgroup_size": 64,
            "top_k": 1,
            "reps": 4,
            "warmup": 2,
            "samples": 2,
            "timing_mode": timing_mode,
        },
        "hardware": {"device_name": "Radeon 8060S Graphics", "gcn_arch_name": "gfx1151"},
    }


def _result(module, *, backend: str, timing_mode: str) -> dict:
    return module._normalize_result(
        backend=backend,
        raw_rows=[_raw_row(backend=backend, timing_mode=timing_mode)],
        isa_by_variant={(64, 1): {"instruction_count": 10, "waitcnt_count": 2}},
        environment=_environment(),
        source_hash=f"sha256:{backend}",
        wrapper_command=["sampler_argmax.py", "--backend", backend],
        commands=[{"rows": 1, "workgroup_size": 64, "top_k": 1}],
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
    assert comparison["source"]["source_hash"] == "sha256:hip"
    assert comparison["sources"]["hip"]["source_hash"] == "sha256:hip"
    assert comparison["sources"]["vulkan"]["source_hash"] == "sha256:vulkan"
    assert comparison["performance_claim"] is True


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: result.update(schema_version=1), "v2"),
        (lambda result: result.update(kind="wrong"), "micro result"),
        (lambda result: result.update(bench="wrong"), "bench"),
        (lambda result: result.update(backend="hip"), "HIP then Vulkan"),
        (lambda result: result.update(classification="wrong"), "classification"),
        (lambda result: result["hardware"].update(gfx_arch="gfx1100"), "architectures"),
        (lambda result: result["hardware"].update(gpu_name="different"), "device identities"),
        (lambda result: result["source"].update(commit="b" * 40), "commit"),
        (lambda result: result["source"].update(dirty=True), "dirty"),
        (lambda result: result["source"].pop("source_hash"), "source hash"),
    ],
)
def test_sampler_comparison_rejects_identity_or_provenance_mismatch(
    mutation, message: str
) -> None:
    module = _load_runner_module()
    hip = _result(module, backend="hip", timing_mode="serial_latency")
    vulkan = _result(module, backend="vulkan", timing_mode="serial_latency")
    mutation(vulkan)

    with pytest.raises(ValueError, match=message):
        module.build_comparison(hip, vulkan, command=["compare"])


def test_sampler_comparison_requires_exact_unique_vocab_rows() -> None:
    module = _load_runner_module()
    hip = _result(module, backend="hip", timing_mode="serial_latency")
    vulkan = _result(module, backend="vulkan", timing_mode="serial_latency")

    duplicate = deepcopy(vulkan)
    duplicate["measurements"]["rows"].append(
        deepcopy(duplicate["measurements"]["rows"][0])
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.build_comparison(hip, duplicate, command=["compare"])

    different_vocab = deepcopy(vulkan)
    different_vocab["measurements"]["rows"][0]["vocab"] = 512
    with pytest.raises(ValueError, match="exact requested"):
        module.build_comparison(hip, different_vocab, command=["compare"])

    bad_expected_count = deepcopy(vulkan)
    bad_expected_count["parameters"]["expected_row_count"] = 2
    with pytest.raises(ValueError, match="expected_row_count"):
        module.build_comparison(hip, bad_expected_count, command=["compare"])

    duplicate_parameters_hip = deepcopy(hip)
    duplicate_parameters_vulkan = deepcopy(vulkan)
    duplicate_parameters_hip["parameters"]["rows_list"].append(1)
    duplicate_parameters_vulkan["parameters"]["rows_list"].append(1)
    with pytest.raises(ValueError, match="duplicate requested"):
        module.build_comparison(
            duplicate_parameters_hip,
            duplicate_parameters_vulkan,
            command=["compare"],
        )


def test_sampler_comparison_marks_dirty_or_incorrect_inputs_non_claiming() -> None:
    module = _load_runner_module()
    hip = _result(module, backend="hip", timing_mode="serial_latency")
    vulkan = _result(module, backend="vulkan", timing_mode="serial_latency")
    hip["source"]["dirty"] = True
    vulkan["source"]["dirty"] = True
    dirty = module.build_comparison(hip, vulkan, command=["compare"])
    assert dirty["performance_claim"] is False
    assert dirty["provenance"]["blocking_reasons"] == ["dirty_source"]

    hip["source"]["dirty"] = False
    vulkan["source"]["dirty"] = False
    vulkan["correctness"]["status"] = "fail"
    incorrect = module.build_comparison(hip, vulkan, command=["compare"])
    assert incorrect["performance_claim"] is False
    assert incorrect["provenance"]["blocking_reasons"] == [
        "correctness_not_passed"
    ]
