from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "benchmarks" / "micro" / "runners" / "q4_selected_dual_real_slice.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("micro_q4_selected_dual_v2", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    module,
    *,
    backend: str,
    mode: str,
    gpu_us: float,
    operation: str = "selected_dual_dp4a_quantize_plus_dot",
    local_size: int = 64,
):
    repetitions = 4
    dispatches = 2 if operation == "selected_dual_dp4a_quantize_plus_dot" else 1
    samples = module.hip_timing.HipTimingSamples(
        [gpu_us, gpu_us, gpu_us],
        [gpu_us + 5.0, gpu_us + 5.0, gpu_us + 5.0],
    )
    burst = module.hip_timing.HipTimingSamples(
        [gpu_us * repetitions] * 3,
        [(gpu_us + 2.0) * repetitions] * 3,
    )
    correctness = {
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "kl_divergence": 0.0,
        "top1": 1.0,
        "exact_bf16_mismatches": 0,
        "outputs_checked": repetitions if mode == "independent_throughput" else 1,
        "pass": True,
    }
    row = module._make_operation_row(
        backend=backend,
        timing_mode=mode,
        operation=operation,
        repetitions=repetitions,
        dispatches_per_iteration=dispatches,
        stream_count=(
            2
            if mode == "independent_throughput"
            and (
                backend == "hip"
                or operation == "selected_dual_dp4a_quantize_plus_dot"
            )
            else 1
        ),
        single_samples=samples,
        burst_samples=burst,
        single_correctness=correctness,
        burst_correctness=correctness,
        barrier_count=(
            3 * repetitions - 2
            if backend == "vulkan"
            and mode == "serial_latency"
            and dispatches == 2
            else repetitions - 1
            if backend == "vulkan" and mode == "serial_latency"
            else repetitions
            if backend == "vulkan" and dispatches == 2
            else 0
        ),
        submission_strategy=(
            "vulkan_multi_queue"
            if backend == "vulkan"
            and mode == "independent_throughput"
            and operation == "selected_dual_dp4a_quantize_plus_dot"
            else None
        ),
        gpu_clock_override=(
            "vulkan_calibrated_cross_queue_timestamp"
            if backend == "vulkan"
            and mode == "independent_throughput"
            and operation == "selected_dual_dp4a_quantize_plus_dot"
            else None
        ),
        shape_fields={
            "quant": "q4_k",
            "buffer_abi": (
                "hip_raw_device_pointer_q8_1_q4_k"
                if backend == "hip"
                else "vulkan_storage_buffer_q8_1_q4_k"
            ),
            "input_scale": 0.1,
            "x_rows": 4,
            "rows": 32,
            "experts": 256,
            "in_features": 2048,
            "out_features": 512,
            "local_size": local_size,
            "workgroup_match": "exact",
        },
    )
    if (
        backend == "vulkan"
        and mode == "independent_throughput"
        and operation == "selected_dual_dp4a_quantize_plus_dot"
    ):
        row["calibrated_timestamp_domain"] = True
        row["calibrated_timestamps_extension"] = "VK_KHR_calibrated_timestamps"
    return row


def _result(module, *, backend: str, mode: str = "serial_latency"):
    rows = [
        _row(
            module,
            backend=backend,
            mode=mode,
            gpu_us=10.0,
            operation="q8_1_quantize",
            local_size=32,
        ),
        _row(
            module,
            backend=backend,
            mode=mode,
            gpu_us=10.0,
            operation="selected_dual_dp4a_dot_prequantized",
        ),
        _row(module, backend=backend, mode=mode, gpu_us=10.0),
    ]
    return {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": module.BENCH_NAME,
        "backend": backend,
        "classification": "real_slice_probe",
        "source": {
            "repo": str(REPO_ROOT),
            "branch": "main",
            "commit": "a" * 40,
            "dirty": False,
            "source_hash": f"sha256:{backend}",
        },
        "environment_source": {"commit": "b" * 40, "dirty": False},
        "hardware": {"gpu_name": "fixture", "gfx_arch": "gfx1151"},
        "parameters": {
            "input_scale": 0.1,
            "x_rows": 4,
            "rows": 32,
            "experts": 256,
            "in_features": 2048,
            "out_features": 512,
            "workgroups": [64],
            "timing_mode": mode,
            "repetitions": 4,
            "warmup_logical_iterations": 3,
            "samples": 3,
            "requested_independent_lanes": 2,
            "actual_independent_lanes": 2 if mode == "independent_throughput" else 1,
            "buffer_abi": (
                "hip_raw_device_pointer_q8_1_q4_k"
                if backend == "hip"
                else "vulkan_storage_buffer_q8_1_q4_k"
            ),
        },
        "correctness": {"status": "pass"},
        "measurements": {"rows": rows},
    }


def test_q4_parse_timing_and_workgroups() -> None:
    module = _load_runner()
    args = module.parse_args(
        [
            "--backend",
            "hip",
            "--timing-mode",
            "independent_throughput",
            "--workgroups",
            "64,128,256",
            "--reps",
            "2",
            "--warmup",
            "3",
        ]
    )
    assert args.timing_mode == "independent_throughput"
    assert module._parse_workgroups(args.workgroups) == [64, 128, 256]
    assert max(args.reps, args.warmup, 1) == 3
    assert module._effective_independent_lanes(args) == 2


def test_q4_vulkan_shaders_share_push_layout_and_native_bf16_output() -> None:
    fields = [
        "rows",
        "in_features",
        "out_features",
        "experts",
        "q8_blocks_per_row",
        "out_packed",
        "blocks_per_row",
        "rep",
        "xq_slice",
        "output_slice",
    ]
    for relative in (
        "benchmarks/micro/kernels/vulkan/q8_1_quantize.comp",
        "benchmarks/micro/kernels/vulkan/q4_selected_dual.comp",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        positions = [source.index(f"uint {field};") for field in fields]
        assert positions == sorted(positions)
    dot_source = (
        REPO_ROOT / "benchmarks/micro/kernels/vulkan/q4_selected_dual.comp"
    ).read_text(encoding="utf-8")
    assert "uint16_t out_bf16[]" in dot_source
    assert "pc.xq_slice * xq_slice_words" in dot_source
    assert "pc.output_slice * 2u * tensor_stride" in dot_source
    assert "This Vulkan probe's push layout" in dot_source
    assert "shared ABI" not in dot_source


def test_vulkan_q8_quantizer_has_explicit_rounding_contract() -> None:
    source = (
        REPO_ROOT / "benchmarks/micro/kernels/vulkan/q8_1_quantize.comp"
    ).read_text(encoding="utf-8")

    assert "uint f32_to_f16_rne(float value)" in source
    assert "uint pack_half2_rne(vec2 values)" in source
    assert "int round_away_from_zero(float value)" in source
    assert "packHalf2x16(" not in source
    assert "int(round(" not in source


def test_q4_vulkan_harness_records_calibrated_multi_queue_combined_path() -> None:
    source = (
        REPO_ROOT / "benchmarks/micro/runners/vulkan_q4_selected_dual.cpp"
    ).read_text(encoding="utf-8")
    assert "std::max(args.reps, std::max(args.warmup, 1u))" in source
    assert "record_operation(warmup_cmd, operation, args.warmup)" in source
    assert "VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT" in source
    assert "storageBuffer16BitAccess = VK_TRUE" in source
    assert "VK_API_VERSION_1_2" in source
    assert "select_compute_queue_family" in source
    assert "calibrated_timestamps_extension" in source
    assert "VkPhysicalDeviceTimelineSemaphoreFeatures" in source
    assert "VulkanMultiQueueTimer" in source
    assert "record_independent_quantize_dot_lane" in source
    assert "rep += rep_stride" in source
    assert "rep += lane_count" in source
    assert "vulkan_multi_queue" not in source
    assert "vkCmdResetEvent" not in source
    assert "vkCmdSetEvent" not in source
    assert "vkCmdWaitEvents" not in source
    assert "VK_ACCESS_SHADER_WRITE_BIT,\n            VK_ACCESS_SHADER_READ_BIT" in source


def test_q4_vulkan_harness_isolates_q8_and_cpu_prequantized_dot() -> None:
    source = (
        REPO_ROOT / "benchmarks/micro/runners/vulkan_q4_selected_dual.cpp"
    ).read_text(encoding="utf-8")

    assert "q8_cpu_vs_vulkan" in source
    assert "dot_with_cpu_prequantized_q8" in source
    assert "VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT" in source
    assert "compare_q8_slices" in source
    assert "q8_dot_isolation_by_workgroup" in RUNNER.read_text(encoding="utf-8")


def test_q4_comparison_emits_gpu_controls_and_rejects_host_wall() -> None:
    module = _load_runner()
    hip = _result(module, backend="hip")
    vulkan = _result(module, backend="vulkan")
    vulkan["measurements"]["rows"] = [
        _row(
            module,
            backend="vulkan",
            mode="serial_latency",
            gpu_us=5.0,
            operation="q8_1_quantize",
            local_size=32,
        ),
        _row(
            module,
            backend="vulkan",
            mode="serial_latency",
            gpu_us=5.0,
            operation="selected_dual_dp4a_dot_prequantized",
        ),
        _row(module, backend="vulkan", mode="serial_latency", gpu_us=5.0),
    ]
    comparison = module.build_comparison(hip, vulkan, command=["compare"])
    assert comparison["schema_version"] == 2
    assert len(comparison["comparisons"]) == 6
    burst = next(
        row
        for row in comparison["comparisons"]
        if row["operation"] == "selected_dual_dp4a_quantize_plus_dot"
        and row["control"] == "burst"
    )
    assert burst["control"] == "burst"
    assert burst["gpu_elapsed"]["status"] == "ok"
    assert burst["gpu_elapsed"]["vulkan_vs_hip_speedup"] == 2.0
    assert burst["host_wall"]["status"] == "not_comparable_submission_contract"
    assert comparison["summary"]["matched_rows"] == 3
    assert comparison["performance_claim"] is True
    assert comparison["provenance"]["commit_match"] is True
    assert comparison["source"]["commit"] == "a" * 40
    assert comparison["sources"]["hip"]["source_hash"] == "sha256:hip"
    assert comparison["sources"]["vulkan"]["source_hash"] == "sha256:vulkan"
    assert comparison["environment_source"]["hip"]["commit"] == "b" * 40
    json.dumps(comparison, allow_nan=False)


def test_q4_comparison_rejects_cross_mode_rows() -> None:
    module = _load_runner()
    hip = _result(module, backend="hip", mode="serial_latency")
    vulkan = _result(module, backend="vulkan", mode="independent_throughput")
    with pytest.raises(ValueError, match="timing_mode"):
        module.build_comparison(hip, vulkan, command=["compare"])


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("input_scale", 0.2),
        ("x_rows", 8),
        ("rows", 64),
        ("experts", 128),
        ("in_features", 1024),
        ("out_features", 256),
        ("workgroups", [128]),
        ("timing_mode", "independent_throughput"),
        ("repetitions", 5),
        ("warmup_logical_iterations", 2),
        ("samples", 5),
        ("requested_independent_lanes", 3),
        ("actual_independent_lanes", 3),
    ],
)
def test_q4_comparison_rejects_parameter_mismatch(field: str, mismatch) -> None:
    module = _load_runner()
    hip = _result(module, backend="hip")
    vulkan = _result(module, backend="vulkan")
    vulkan["parameters"][field] = mismatch
    with pytest.raises(ValueError, match=field):
        module.build_comparison(hip, vulkan, command=["compare"])


def test_q4_comparison_rejects_schema_backend_arch_and_row_set_mismatches() -> None:
    module = _load_runner()
    hip = _result(module, backend="hip")
    vulkan = _result(module, backend="vulkan")

    bad = deepcopy(vulkan)
    bad["schema_version"] = 1
    with pytest.raises(ValueError, match="v2 timing-contract"):
        module.build_comparison(hip, bad, command=["compare"])

    bad = deepcopy(vulkan)
    bad["backend"] = "hip"
    with pytest.raises(ValueError, match="must be HIP then Vulkan"):
        module.build_comparison(hip, bad, command=["compare"])

    bad = deepcopy(vulkan)
    bad["hardware"]["gfx_arch"] = "gfx1100"
    with pytest.raises(ValueError, match="architectures do not match"):
        module.build_comparison(hip, bad, command=["compare"])

    bad = deepcopy(vulkan)
    bad["measurements"]["rows"][0]["local_size"] = 128
    with pytest.raises(ValueError, match="exact expected"):
        module.build_comparison(hip, bad, command=["compare"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "other", "kind"),
        ("bench", "other", "bench"),
        ("classification", "diagnostic", "classification"),
    ],
)
def test_q4_comparison_rejects_result_identity(field: str, value: str, message: str) -> None:
    module = _load_runner()
    hip = _result(module, backend="hip")
    vulkan = _result(module, backend="vulkan")
    vulkan[field] = value
    with pytest.raises(ValueError, match=message):
        module.build_comparison(hip, vulkan, command=["compare"])


def test_q4_comparison_requires_quant_workgroup_and_complete_expected_rows() -> None:
    module = _load_runner()
    hip = _result(module, backend="hip")
    vulkan = _result(module, backend="vulkan")

    bad = deepcopy(vulkan)
    bad["measurements"]["rows"][0]["quant"] = "q5_k"
    with pytest.raises(ValueError, match="quant must be q4_k"):
        module.build_comparison(hip, bad, command=["compare"])

    bad = deepcopy(vulkan)
    bad["measurements"]["rows"][1]["workgroup_match"] = "diagnostic"
    with pytest.raises(ValueError, match="workgroup_match must be exact"):
        module.build_comparison(hip, bad, command=["compare"])

    bad = deepcopy(vulkan)
    bad["measurements"]["rows"].pop()
    with pytest.raises(ValueError, match="exact expected 3-row set"):
        module.build_comparison(hip, bad, command=["compare"])


def test_q4_comparison_marks_dirty_or_mismatched_provenance_non_claiming() -> None:
    module = _load_runner()
    hip = _result(module, backend="hip")
    vulkan = _result(module, backend="vulkan")
    vulkan["source"]["dirty"] = True
    comparison = module.build_comparison(hip, vulkan, command=["compare"])
    assert comparison["performance_claim"] is False
    assert "dirty_source" in comparison["provenance"]["blocking_reasons"]


def test_q4_independent_row_requires_all_dispatch_coverage() -> None:
    module = _load_runner()
    row = _row(
        module,
        backend="vulkan",
        mode="independent_throughput",
        gpu_us=5.0,
    )
    assert row["dependency_contract"]["output_partitioning"] == "disjoint"
    assert row["submission"]["strategy"] == "vulkan_multi_queue"
    assert row["submission"]["queue_or_stream_count"] == 2
    assert (
        row["timing"]["burst"]["gpu_elapsed"]["clock"]
        == "vulkan_calibrated_cross_queue_timestamp"
    )
    assert row["correctness"]["timed_sequence"]["coverage"] == "all_dispatches"
    assert row["correctness"]["synchronization"]["barrier_count"] == 4
    assert row["numeric_correctness"]["burst"]["outputs_checked"] == 4


def test_q4_rows_label_backend_abi_and_quantize_downstream_oracle() -> None:
    module = _load_runner()
    hip_quant = _row(
        module,
        backend="hip",
        mode="serial_latency",
        gpu_us=5.0,
        operation="q8_1_quantize",
        local_size=32,
    )
    vulkan_quant = _row(
        module,
        backend="vulkan",
        mode="serial_latency",
        gpu_us=5.0,
        operation="q8_1_quantize",
        local_size=32,
    )
    assert hip_quant["buffer_abi"] == "hip_raw_device_pointer_q8_1_q4_k"
    assert vulkan_quant["buffer_abi"] == "vulkan_storage_buffer_q8_1_q4_k"
    assert "downstream Q4_K" in hip_quant["correctness"]["single_dispatch"]["oracle"]
    assert "downstream Q4_K" in vulkan_quant["correctness"]["single_dispatch"]["oracle"]
