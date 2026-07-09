from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "benchmarks" / "micro" / "runners" / "q8_0_dense_real_slice.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("micro_q8_real_slice", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(module, backend: str, timing_mode: str = "serial_latency"):
    actual_lanes = 2 if timing_mode == "independent_throughput" else 1
    samples = module.hip_timing.HipTimingSamples([10.0, 12.0], [20.0, 22.0])
    correctness = {
        "oracle": "test",
        "outputs_checked": 1,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "kl_divergence": 0.0,
        "top1": 1.0,
        "pass": True,
    }
    rows = []
    for operation, variant, row_tile, dispatches in (
        ("q8_1_quantize", "quantize", 0, 1),
        ("q8_0_dense_dp4a_dot_prequantized", "single", 1, 1),
        ("q8_0_dense_dp4a_quantize_plus_dot", "single", 1, 2),
    ):
        multi_queue = (
            backend == "vulkan"
            and timing_mode == "independent_throughput"
            and dispatches == 2
        )
        rows.append(
            module._make_operation_row(
                backend=backend,
                timing_mode=timing_mode,
                operation=operation,
                repetitions=2,
                dispatches_per_iteration=dispatches,
                stream_count=(actual_lanes if backend == "hip" or multi_queue else 1),
                single_samples=samples,
                burst_samples=module.hip_timing.HipTimingSamples(
                    [20.0, 24.0], [40.0, 44.0]
                ),
                single_correctness=correctness,
                burst_correctness=correctness,
                barrier_count=(2 if multi_queue else 4 if backend == "vulkan" else 0),
                submission_strategy="vulkan_multi_queue" if multi_queue else None,
                gpu_clock_override=(
                    "vulkan_calibrated_multi_queue_timestamp_span"
                    if multi_queue
                    else None
                ),
                synchronization_method=(
                    "timeline_coordinated_multi_queue_disjoint_slices"
                    if multi_queue
                    else None
                ),
                execution_metadata={
                    "requested_parallel_lanes": 2,
                    "calibrated_timestamp_domain": multi_queue,
                    "calibrated_timestamp_extension": (
                        "VK_KHR_calibrated_timestamps" if multi_queue else ""
                    ),
                },
                shape_fields={
                    "variant": variant,
                    "row_tile": row_tile,
                    "rows": 1,
                    "in_features": 64,
                    "out_features": 64,
                    "local_size": 32,
                    "workgroup_match": "exact_hip_wave32",
                },
            )
        )
    parameters = {
        "shapes": "64x64",
        "rows_list": "1",
        "row_tiles": "1",
        "exact_local_size": 32,
        "timing_mode": timing_mode,
        "input_scale": 0.1,
        "repetitions": 2,
        "warmup_sequences": 1,
        "samples": 2,
        "independent_streams": 2,
        "actual_parallel_lanes": actual_lanes,
    }
    if backend == "vulkan":
        parameters["local_sizes"] = "32"
    return {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": module.BENCH_NAME,
        "backend": backend,
        "classification": "real_slice_probe",
        "hardware": {"gpu_name": "fixture", "gfx_arch": "gfx1151"},
        "source": {
            "repo": str(REPO_ROOT),
            "branch": "main",
            "commit": "abc123",
            "dirty": True,
            "source_hash": f"sha256:{backend}",
        },
        "parameters": parameters,
        "measurements": {"rows": rows},
        "correctness": {"status": "pass", "all_pass": True},
        "environment": {},
    }


def test_q8_defaults_to_serial_and_includes_matched_wave32() -> None:
    module = _load_runner()

    args = module.parse_args(["--backend", "vulkan"])

    assert args.timing_mode == "serial_latency"
    assert args.independent_streams == 4
    assert 32 in module._parse_csv_u32(args.local_sizes)


def test_q8_shader_push_layout_and_event_free_multi_queue_path_are_explicit() -> None:
    quant_shader = (
        REPO_ROOT / "benchmarks" / "micro" / "kernels" / "vulkan" / "q8_1_quantize.comp"
    ).read_text(encoding="utf-8")
    dot_shader = (
        REPO_ROOT / "benchmarks" / "micro" / "kernels" / "vulkan" / "q8_0_dense.comp"
    ).read_text(encoding="utf-8")
    harness = (
        REPO_ROOT / "benchmarks" / "micro" / "runners" / "vulkan_q8_0_dense.cpp"
    ).read_text(encoding="utf-8")

    for source in (quant_shader, dot_shader):
        assert "uint rep;" in source
        assert "uint xq_slice;" in source
        assert "uint output_slice;" in source
    assert "uint load_bf16(uint element_index)" in quant_shader
    assert "uint16_t out_bf16[]" in dot_shader
    assert "storageBuffer16BitAccess" in harness
    assert "VK_API_VERSION_1_2" in harness
    assert "select_compute_queue_family" in harness
    assert "VulkanMultiQueueTimer" in harness
    assert "record_quantize_dot_lane" in harness
    assert "vulkan_multi_queue" in harness
    assert "VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES" in harness
    assert "calibrated_timestamps_extension" in harness
    assert "vkCmdSetEvent" not in harness
    assert "vkCmdWaitEvents" not in harness
    assert "VkEvent" not in harness
    assert "VK_ACCESS_SHADER_WRITE_BIT,\n            VK_ACCESS_SHADER_READ_BIT" in harness
    assert "VK_ACCESS_SHADER_READ_BIT,\n                  VK_ACCESS_SHADER_WRITE_BIT" in harness
    assert harness.count("VK_ACCESS_SHADER_WRITE_BIT,\n                  VK_ACCESS_SHADER_WRITE_BIT") >= 1


def test_q8_comparison_uses_gpu_ratios_and_marks_host_wall_not_comparable() -> None:
    module = _load_runner()
    comparison = module.build_comparison(
        _result(module, "hip"),
        _result(module, "vulkan"),
        command=["q8-test"],
    )

    assert comparison["schema_version"] == 2
    assert comparison["summary"]["matched_rows"] == 3
    row = next(
        item
        for item in comparison["matched_rows"]
        if item["operation"] == "q8_0_dense_dp4a_quantize_plus_dot"
    )
    assert row["local_size"] == 32
    assert row["ratios"]["single"]["gpu_elapsed"]["status"] == "ok"
    assert (
        row["ratios"]["burst"]["host_wall"]["status"]
        == "not_comparable_submission_contract"
    )


def test_q8_warmup_is_one_logical_sequence() -> None:
    module = _load_runner()

    class FakeTimer:
        def __init__(self):
            self.calls = []

        def run_and_wait(self, iterations, launch):
            self.calls.append(("warmup", iterations, launch))

        def measure(self, iterations, samples, launch):
            self.calls.append(("measure", iterations, samples, launch))
            return (iterations, samples)

    launch = object()
    timer = FakeTimer()
    single, burst = module._measure_hip_operation(
        timer,
        launch,
        repetitions=4,
        warmup=7,
        samples=3,
    )

    assert single == (1, 3)
    assert burst == (4, 3)
    assert timer.calls == [
        ("warmup", 7, launch),
        ("measure", 1, 3, launch),
        ("measure", 4, 3, launch),
    ]


def test_q8_comparison_rejects_mismatched_provenance_and_rows() -> None:
    module = _load_runner()
    hip = _result(module, "hip")
    vulkan = _result(module, "vulkan")

    vulkan["parameters"]["input_scale"] = 0.2
    try:
        module.build_comparison(hip, vulkan, command=["q8-test"])
    except ValueError as exc:
        assert "input_scale" in str(exc)
    else:
        raise AssertionError("expected an input-scale mismatch rejection")

    vulkan = _result(module, "vulkan")
    vulkan["measurements"]["rows"].pop()
    try:
        module.build_comparison(hip, vulkan, command=["q8-test"])
    except ValueError as exc:
        assert "row sets" in str(exc)
    else:
        raise AssertionError("expected a comparable-row mismatch rejection")

    vulkan = _result(module, "vulkan")
    vulkan["source"]["commit"] = "different"
    try:
        module.build_comparison(hip, vulkan, command=["q8-test"])
    except ValueError as exc:
        assert "source commit" in str(exc)
    else:
        raise AssertionError("expected a source-provenance mismatch rejection")


def test_q8_independent_comparison_requires_matched_calibrated_lanes() -> None:
    module = _load_runner()
    hip = _result(module, "hip", "independent_throughput")
    vulkan = _result(module, "vulkan", "independent_throughput")

    comparison = module.build_comparison(hip, vulkan, command=["q8-test"])
    row = next(
        item
        for item in comparison["matched_rows"]
        if item["operation"] == "q8_0_dense_dp4a_quantize_plus_dot"
    )
    assert row["hip_execution"]["actual_parallel_lanes"] == 2
    assert row["vulkan_execution"]["actual_parallel_lanes"] == 2
    assert row["vulkan_execution"]["calibrated_timestamp_domain"] is True

    combined = next(
        item
        for item in vulkan["measurements"]["rows"]
        if item["operation"] == "q8_0_dense_dp4a_quantize_plus_dot"
    )
    combined["execution"]["actual_parallel_lanes"] = 1
    try:
        module.build_comparison(hip, vulkan, command=["q8-test"])
    except ValueError as exc:
        assert "actual lane counts" in str(exc)
    else:
        raise AssertionError("expected a combined lane-count mismatch rejection")

    vulkan = _result(module, "vulkan", "independent_throughput")
    combined = next(
        item
        for item in vulkan["measurements"]["rows"]
        if item["operation"] == "q8_0_dense_dp4a_quantize_plus_dot"
    )
    combined["execution"]["calibrated_timestamp_domain"] = False
    try:
        module.build_comparison(hip, vulkan, command=["q8-test"])
    except ValueError as exc:
        assert "calibrated timestamps" in str(exc)
    else:
        raise AssertionError("expected an uncalibrated multi-queue rejection")


def test_q8_independent_storage_covers_warmup() -> None:
    python_source = RUNNER.read_text(encoding="utf-8")
    vulkan_source = (
        REPO_ROOT / "benchmarks" / "micro" / "runners" / "vulkan_q8_0_dense.cpp"
    ).read_text(encoding="utf-8")

    assert "work_repetitions = max(args.reps, args.warmup, 1)" in python_source
    assert "std::max(args.reps, std::max(args.warmup, 1u))" in vulkan_source
    assert "kl_divergence <= 0.05" in vulkan_source
    assert "top1 >= 0.90" in vulkan_source
