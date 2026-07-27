from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_contract_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "timing_contract.py"
    )
    spec = importlib.util.spec_from_file_location("micro_timing_contract", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timed_row(
    module,
    *,
    backend: str,
    timing_mode: str,
    repetitions: int = 8,
    strategy: str | None = None,
):
    gpu_clock = {
        "hip": "hip_event",
        "vulkan": "vulkan_timestamp",
        "redline": "redline_pm4_timestamp",
    }[backend]
    single = module.make_timing_control(
        logical_iterations=1,
        dispatches_per_iteration=1,
        gpu_samples_us=[10.0, 12.0, 11.0],
        host_samples_us=[14.0, 16.0, 15.0],
        gpu_clock=gpu_clock,
    )
    burst = module.make_timing_control(
        logical_iterations=repetitions,
        dispatches_per_iteration=1,
        gpu_samples_us=[80.0, 88.0, 84.0],
        host_samples_us=[96.0, 104.0, 100.0],
        gpu_clock=gpu_clock,
    )
    return module.make_timed_row_contract(
        timing_mode=timing_mode,
        backend=backend,
        repetitions=repetitions,
        dispatches_per_iteration=1,
        dependency_validation_status="pass",
        submission=module.make_submission(
            strategy=strategy
            or ("multi_stream" if timing_mode == "independent_throughput" else "direct"),
            queue_or_stream_count=4 if timing_mode == "independent_throughput" else 1,
            recording_in_timed_region=False,
        ),
        single_timing=single,
        burst_timing=burst,
        correctness=module.make_correctness(
            status="pass",
            oracle="deterministic CPU reference",
            logical_iterations=repetitions,
            coverage=(
                "all_dispatches"
                if timing_mode == "independent_throughput"
                else "chained_final_state"
            ),
            synchronization_method=(
                "disjoint_outputs" if timing_mode == "independent_throughput" else "ordered_stream"
            ),
            barrier_count=repetitions - 1 if timing_mode == "serial_latency" else 0,
        ),
    )


@pytest.mark.parametrize("mode", ["serial_latency", "independent_throughput"])
def test_valid_timing_modes_require_single_and_burst_controls(mode: str) -> None:
    module = _load_contract_module()
    row = _timed_row(module, backend="hip", timing_mode=mode)

    module.validate_timed_row(row, expected_repetitions=8)
    assert row["timing"]["single"]["logical_iterations"] == 1
    assert row["timing"]["burst"]["logical_iterations"] == 8
    assert row["timing"]["burst"]["gpu_elapsed"]["per_iteration_us"]["median"] == 10.5
    assert row["timing"]["burst"]["host_wall"]["sequence_us"]["samples"] == 3


def test_independent_mode_rejects_shared_outputs() -> None:
    module = _load_contract_module()
    row = _timed_row(module, backend="hip", timing_mode="independent_throughput")
    row["dependency_contract"]["output_partitioning"] = "chained_shared"

    with pytest.raises(ValueError, match="output partitioning"):
        module.validate_timed_row(row)


def test_independent_mode_requires_validation_of_every_output() -> None:
    module = _load_contract_module()
    row = _timed_row(module, backend="vulkan", timing_mode="independent_throughput")
    row["correctness"]["timed_sequence"]["coverage"] = "chained_final_state"

    with pytest.raises(ValueError, match="every disjoint output"):
        module.validate_timed_row(row)


def test_comparison_emits_separate_gpu_and_host_ratios() -> None:
    module = _load_contract_module()
    hip = _timed_row(
        module,
        backend="hip",
        timing_mode="serial_latency",
        strategy="hip_graph",
    )
    vulkan = _timed_row(
        module,
        backend="vulkan",
        timing_mode="serial_latency",
        strategy="vulkan_command_buffer",
    )
    vulkan["timing"]["burst"]["gpu_elapsed"]["per_iteration_us"]["median"] = 7.0
    vulkan["timing"]["burst"]["host_wall"]["per_iteration_us"]["median"] = 8.0

    gpu = module.comparison_ratio(hip, vulkan, control="burst", domain="gpu_elapsed")
    host = module.comparison_ratio(hip, vulkan, control="burst", domain="host_wall")

    assert gpu["vulkan_vs_hip_speedup"] == 1.5
    assert host["vulkan_vs_hip_speedup"] == 12.5 / 8.0


def test_redline_retained_pm4_is_a_pre_recorded_submission() -> None:
    module = _load_contract_module()
    redline = _timed_row(
        module,
        backend="redline",
        timing_mode="serial_latency",
        strategy="retained_pm4_ib",
    )
    vulkan = _timed_row(
        module,
        backend="vulkan",
        timing_mode="serial_latency",
        strategy="vulkan_command_buffer",
    )

    assert redline["dependency_contract"]["inter_dispatch_ordering"] == "redline_rmw"
    host = module.backend_pair_ratio(
        redline,
        vulkan,
        lhs_backend="redline",
        rhs_backend="vulkan",
        control="burst",
        domain="host_wall",
    )
    assert host["rhs_vs_lhs_speedup"] == 1.0


def test_host_wall_comparison_rejects_enqueue_vs_command_buffer() -> None:
    module = _load_contract_module()
    hip = _timed_row(
        module,
        backend="hip",
        timing_mode="serial_latency",
        strategy="direct",
    )
    vulkan = _timed_row(
        module,
        backend="vulkan",
        timing_mode="serial_latency",
        strategy="vulkan_command_buffer",
    )

    gpu = module.comparison_ratio(hip, vulkan, control="burst", domain="gpu_elapsed")
    assert gpu["vulkan_vs_hip_speedup"] == 1.0
    with pytest.raises(ValueError, match="host-wall submission contracts"):
        module.comparison_ratio(hip, vulkan, control="burst", domain="host_wall")


def test_comparison_rejects_cross_mode_rows() -> None:
    module = _load_contract_module()
    hip = _timed_row(module, backend="hip", timing_mode="serial_latency")
    vulkan = _timed_row(module, backend="vulkan", timing_mode="independent_throughput")

    with pytest.raises(ValueError, match="not comparable"):
        module.comparison_ratio(hip, vulkan, control="burst", domain="gpu_elapsed")


def test_unavailable_gpu_timestamp_is_explicit() -> None:
    module = _load_contract_module()
    metric = module.make_metric(
        clock="vulkan_timestamp",
        logical_iterations=8,
        status="unsupported",
    )

    assert metric == {"status": "unsupported", "clock": "vulkan_timestamp"}


def test_result_schema_contains_v2_result_and_comparison_contracts() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "schemas"
        / "result.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$defs"]["v2Result"]["properties"]["schema_version"] == {"const": 2}
    assert "redline" in schema["$defs"]["v2Result"]["properties"]["backend"]["enum"]
    assert schema["$defs"]["v2Comparison"]["properties"]["kind"] == {
        "const": "hipengine_micro_comparison"
    }
    timed_row = schema["$defs"]["timedRow"]
    assert set(timed_row["required"]) == {
        "timing_mode",
        "dependency_contract",
        "submission",
        "timing",
        "correctness",
    }
