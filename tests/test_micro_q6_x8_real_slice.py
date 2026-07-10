from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "benchmarks" / "micro" / "runners" / "q6_x8_real_slice.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("micro_q6_x8_real_slice", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(module, timing_mode: str = "serial_latency") -> argparse.Namespace:
    return argparse.Namespace(
        rows=2,
        experts=4,
        in_features=256,
        out_features=64,
        input_scale=0.1,
        local_size=64,
        timing_mode=timing_mode,
        reps=3,
        warmup=5,
        samples=2,
        independent_streams=4,
    )


def _result(module, backend: str, timing_mode: str = "serial_latency") -> dict:
    args = _args(module, timing_mode)
    samples = module.hip_timing.HipTimingSamples([10.0, 12.0], [20.0, 22.0])
    burst_samples = module.hip_timing.HipTimingSamples([30.0, 36.0], [60.0, 66.0])
    numeric = {
        "oracle": "test",
        "outputs_checked": 3 if timing_mode == "independent_throughput" else 1,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "kl_divergence": 0.0,
        "top1": 1.0,
        "exact_bf16_mismatches": 0,
        "pass": True,
        "expected_repetitions": (
            [0, 1, 2] if timing_mode == "independent_throughput" else [2]
        ),
    }
    rows = []
    effective_lanes = module._effective_independent_lanes(args)
    for operation, dispatches in (
        ("q8_1_quantize", 1),
        ("x8_selected_dp4a_dot_prequantized", 1),
        ("x8_selected_dp4a_quantize_plus_dot", 2),
    ):
        rows.append(
            module._make_operation_row(
                backend=backend,
                timing_mode=timing_mode,
                operation=operation,
                repetitions=args.reps,
                dispatches_per_iteration=dispatches,
                stream_count=(
                    effective_lanes
                    if timing_mode == "independent_throughput"
                    and (
                        backend == "hip"
                        or operation == "x8_selected_dp4a_quantize_plus_dot"
                    )
                    else 1
                ),
                single_samples=samples,
                burst_samples=burst_samples,
                single_correctness={**numeric, "outputs_checked": 1, "expected_repetitions": [0]},
                burst_correctness=numeric,
                barrier_count=0 if backend == "hip" else 2,
                shape_fields=module._shape_fields(args, operation=operation),
                calibrated_gpu_timing=bool(
                    backend == "vulkan"
                    and timing_mode == "independent_throughput"
                    and operation == "x8_selected_dp4a_quantize_plus_dot"
                ),
            )
        )
    hardware = {"gpu_name": "test", "gfx_arch": "gfx1151"}
    if backend == "vulkan":
        hardware["device"] = {
            "active_queue_count": effective_lanes,
            "calibrated_timestamps_extension": (
                "VK_KHR_calibrated_timestamps" if effective_lanes > 1 else None
            ),
            "cross_queue_gpu_timing_calibrated": effective_lanes > 1,
        }
    return {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": module.BENCH_NAME,
        "backend": backend,
        "classification": "real_slice_probe",
        "hardware": hardware,
        "source": {
            "repo": str(REPO_ROOT),
            "branch": "main",
            "commit": "abc123",
            "dirty": False,
            "source_hash": f"sha256:{backend}",
        },
        "command": ["q6-test"],
        "parameters": module._common_parameters(args),
        "correctness": {"status": "pass", "all_pass": True, "rows": 3},
        "measurements": {"rows": rows},
        "environment": {},
    }


def test_q6_defaults_and_shared_quantizer_abi() -> None:
    module = _load_runner()
    args = module.parse_args(["--out", "/tmp/q6.json"])

    assert args.backend is None
    assert args.timing_mode == "serial_latency"
    assert args.independent_streams == 4
    quantizer = (
        REPO_ROOT / "benchmarks" / "micro" / "kernels" / "vulkan" / "q8_1_quantize.comp"
    ).read_text(encoding="utf-8")
    for field in ("rep", "xq_slice", "output_slice"):
        assert f"uint {field};" in quantizer


def test_q6_independent_vulkan_uses_multi_queue_lanes_and_max_allocation() -> None:
    module_source = RUNNER.read_text(encoding="utf-8")
    harness = (
        REPO_ROOT
        / "benchmarks"
        / "micro"
        / "runners"
        / "vulkan_q6_x8_selected_down.cpp"
    ).read_text(encoding="utf-8")

    assert "min(args.independent_streams, args.reps)" in module_source
    assert "std::max({args.reps, args.warmup, 1u})" in harness
    assert "std::min(args.independent_lanes, args.reps)" in harness
    assert "vkCmdSetEvent" not in harness
    assert "vkCmdWaitEvents" not in harness
    assert "VulkanMultiQueueTimer" in harness
    assert "calibrated_timestamps_extension" in harness
    assert "rep += lane_count" in harness
    assert "requested_queue_count" in harness
    assert "static_cast<VkDeviceSize>(rep)" in harness
    assert "kl_divergence <= 0.05" in harness
    assert "top1 >= 0.90" in harness


def test_q6_independent_lane_count_is_capped_by_timed_repetitions() -> None:
    module = _load_runner()
    args = _args(module, "independent_throughput")
    args.reps = 2
    args.warmup = 9

    assert module._effective_independent_lanes(args) == 2


@pytest.mark.parametrize("mode", ["serial_latency", "independent_throughput"])
def test_q6_strict_comparison_emits_gpu_ratios_and_rejects_host_wall(mode: str) -> None:
    module = _load_runner()
    comparison = module.build_comparison(
        _result(module, "hip", mode),
        _result(module, "vulkan", mode),
        command=["q6-compare"],
    )

    assert comparison["schema_version"] == 2
    assert comparison["sources"]["hip"]["source_hash"] == "sha256:hip"
    assert comparison["sources"]["vulkan"]["source_hash"] == "sha256:vulkan"
    assert comparison["performance_claim"] is True
    assert comparison["provenance"]["hip_source_hash"] == "sha256:hip"
    assert comparison["provenance"]["vulkan_source_hash"] == "sha256:vulkan"
    assert comparison["provenance"]["blocking_reasons"] == []
    assert comparison["summary"]["matched_rows"] == 3
    assert len(comparison["comparisons"]) == 6
    for row in comparison["matched_rows"]:
        assert row["ratios"]["burst"]["gpu_elapsed"]["status"] == "ok"
        assert (
            row["ratios"]["burst"]["host_wall"]["status"]
            == "not_comparable_submission_contract"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: result["hardware"].update(gfx_arch="gfx1100"), "architectures"),
        (lambda result: result["source"].update(commit="different"), "source commit"),
        (lambda result: result["parameters"].update(kv_type="bf16"), "parameters"),
        (
            lambda result: result["measurements"]["rows"].pop(),
            "exact quantize/dot/combined row triplet",
        ),
        (lambda result: result["hardware"].update(gpu_name="different"), "device identities"),
        (lambda result: result.update(classification="geometry"), "real-slice probes"),
        (
            lambda result: result["correctness"].update(all_pass=False),
            "correctness gate",
        ),
        (
            lambda result: result["correctness"].update(rows=2),
            "correctness gate",
        ),
    ],
)
def test_q6_strict_comparison_rejects_identity_mismatches(mutation, message: str) -> None:
    module = _load_runner()
    hip = _result(module, "hip")
    vulkan = copy.deepcopy(_result(module, "vulkan"))
    mutation(vulkan)

    with pytest.raises(ValueError, match=message):
        module.build_comparison(hip, vulkan, command=["q6-compare"])


def test_q6_comparison_requires_exact_operation_triplet_and_rejects_duplicates() -> None:
    module = _load_runner()
    hip = _result(module, "hip")
    vulkan = _result(module, "vulkan")
    duplicate = copy.deepcopy(vulkan["measurements"]["rows"][0])
    vulkan["measurements"]["rows"].append(duplicate)

    with pytest.raises(ValueError, match="duplicate"):
        module.build_comparison(hip, vulkan, command=["q6-compare"])

    vulkan = _result(module, "vulkan")
    vulkan["measurements"]["rows"][0]["operation"] = "unexpected_operation"
    with pytest.raises(ValueError, match="exact quantize/dot/combined row triplet"):
        module.build_comparison(hip, vulkan, command=["q6-compare"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "hip", "backend metadata"),
        ("workgroup_match", "unmatched", "workgroup metadata"),
        ("variant", "wrong", "variant metadata"),
        ("q8_blocks_per_row", 999, "q8_blocks_per_row metadata"),
    ],
)
def test_q6_comparison_rejects_row_metadata_mismatches(
    field: str, value, message: str
) -> None:
    module = _load_runner()
    hip = _result(module, "hip")
    vulkan = _result(module, "vulkan")
    vulkan["measurements"]["rows"][0][field] = value

    with pytest.raises(ValueError, match=message):
        module.build_comparison(hip, vulkan, command=["q6-compare"])


def test_q6_comparison_marks_cleanliness_without_losing_backend_sources() -> None:
    module = _load_runner()
    hip = _result(module, "hip")
    vulkan = _result(module, "vulkan")
    hip["source"]["dirty"] = True
    vulkan["source"]["dirty"] = True

    comparison = module.build_comparison(hip, vulkan, command=["q6-compare"])

    assert comparison["performance_claim"] is False
    assert comparison["provenance"]["dirty"] is True
    assert comparison["provenance"]["blocking_reasons"] == ["dirty_source"]
    assert comparison["sources"]["hip"]["source_hash"] == "sha256:hip"
    assert comparison["sources"]["vulkan"]["source_hash"] == "sha256:vulkan"


def test_q6_comparison_rejects_missing_source_hash_and_dirty_mismatch() -> None:
    module = _load_runner()
    hip = _result(module, "hip")
    vulkan = _result(module, "vulkan")
    vulkan["source"]["source_hash"] = ""

    with pytest.raises(ValueError, match="source source_hash is missing"):
        module.build_comparison(hip, vulkan, command=["q6-compare"])

    vulkan = _result(module, "vulkan")
    del vulkan["source"]["dirty"]
    with pytest.raises(ValueError, match="source dirty is missing"):
        module.build_comparison(hip, vulkan, command=["q6-compare"])

    vulkan = _result(module, "vulkan")
    vulkan["source"]["dirty"] = True
    with pytest.raises(ValueError, match="source dirty values do not match"):
        module.build_comparison(hip, vulkan, command=["q6-compare"])


def test_q6_independent_comparison_rejects_lane_count_mismatch() -> None:
    module = _load_runner()
    hip = _result(module, "hip", "independent_throughput")
    vulkan = _result(module, "vulkan", "independent_throughput")
    combined = next(
        row
        for row in vulkan["measurements"]["rows"]
        if row["operation"] == "x8_selected_dp4a_quantize_plus_dot"
    )
    combined["submission"]["queue_or_stream_count"] = 2

    with pytest.raises(ValueError, match="lane counts"):
        module.build_comparison(hip, vulkan, command=["q6-compare"])


def test_q6_independent_comparison_requires_calibrated_queue_metadata() -> None:
    module = _load_runner()
    hip = _result(module, "hip", "independent_throughput")
    vulkan = _result(module, "vulkan", "independent_throughput")
    vulkan["hardware"]["device"]["cross_queue_gpu_timing_calibrated"] = False

    with pytest.raises(ValueError, match="uncalibrated"):
        module.build_comparison(hip, vulkan, command=["q6-compare"])


def test_q6_repetition_salt_changes_input() -> None:
    module = _load_runner()
    first = module._make_x_bf16(2, 256, 0.1, 0)
    second = module._make_x_bf16(2, 256, 0.1, 1)

    assert first.shape == second.shape
    assert not (first == second).all()
