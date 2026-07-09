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
    samples = module.hip_timing.HipTimingSamples([10.0, 12.0], [20.0, 22.0])
    correctness = {
        "oracle": "test",
        "outputs_checked": 1,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "top1": 1.0,
        "pass": True,
    }
    row = module._make_operation_row(
        backend=backend,
        timing_mode=timing_mode,
        operation="q8_0_dense_dp4a_quantize_plus_dot",
        repetitions=2,
        dispatches_per_iteration=2,
        stream_count=2 if timing_mode == "independent_throughput" else 1,
        single_samples=samples,
        burst_samples=module.hip_timing.HipTimingSamples([20.0, 24.0], [40.0, 44.0]),
        single_correctness=correctness,
        burst_correctness=correctness,
        barrier_count=4 if backend == "vulkan" else 0,
        shape_fields={
            "variant": "single",
            "row_tile": 1,
            "rows": 1,
            "in_features": 64,
            "out_features": 64,
            "local_size": 32,
            "workgroup_match": "exact_hip_wave32",
        },
    )
    return {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": module.BENCH_NAME,
        "backend": backend,
        "measurements": {"rows": [row]},
    }


def test_q8_defaults_to_serial_and_includes_matched_wave32() -> None:
    module = _load_runner()

    args = module.parse_args(["--backend", "vulkan"])

    assert args.timing_mode == "serial_latency"
    assert args.independent_streams == 4
    assert 32 in module._parse_csv_u32(args.local_sizes)


def test_q8_shader_push_layout_and_serial_barriers_are_explicit() -> None:
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
    assert comparison["summary"]["matched_rows"] == 1
    row = comparison["matched_rows"][0]
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


def test_q8_independent_storage_covers_warmup() -> None:
    python_source = RUNNER.read_text(encoding="utf-8")
    vulkan_source = (
        REPO_ROOT / "benchmarks" / "micro" / "runners" / "vulkan_q8_0_dense.cpp"
    ).read_text(encoding="utf-8")

    assert "work_repetitions = max(args.reps, args.warmup, 1)" in python_source
    assert "std::max(args.reps, std::max(args.warmup, 1u))" in vulkan_source
