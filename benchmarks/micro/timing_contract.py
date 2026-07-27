#!/usr/bin/env python3
"""Shared timing and comparison contract for HIP/Vulkan microbenchmarks."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


TIMING_MODES = ("serial_latency", "independent_throughput")
TIMING_CONTROLS = ("single", "burst")
TIMING_DOMAINS = ("gpu_elapsed", "host_wall")
METRIC_STATUSES = ("ok", "unsupported", "below_resolution", "not_run")
TIMED_BACKENDS = ("hip", "vulkan", "redline")
_PRE_RECORDED_STRATEGIES = {
    "hip_graph",
    "vulkan_command_buffer",
    "retained_pm4_ib",
}
_HOST_ENQUEUED_STRATEGIES = {"direct", "multi_stream"}


def parse_timing_mode(value: str) -> str:
    if value not in TIMING_MODES:
        choices = ", ".join(TIMING_MODES)
        raise ValueError(f"timing mode must be one of: {choices}")
    return value


def _finite_samples(values: Iterable[float]) -> list[float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("timing samples must not be empty")
    if not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise ValueError("timing samples must be finite and non-negative")
    return samples


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(values: Iterable[float]) -> dict[str, float | int]:
    samples = _finite_samples(values)
    return {
        "samples": len(samples),
        "n": len(samples),
        "median": statistics.median(samples),
        "p05": _percentile(samples, 0.05),
        "p95": _percentile(samples, 0.95),
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.pstdev(samples),
    }


def make_metric(
    *,
    clock: str,
    logical_iterations: int,
    sequence_samples_us: Iterable[float] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    if status not in METRIC_STATUSES:
        raise ValueError(f"unknown metric status: {status}")
    if logical_iterations <= 0:
        raise ValueError("logical_iterations must be positive")
    metric: dict[str, Any] = {"status": status, "clock": clock}
    if status != "ok":
        if sequence_samples_us is not None:
            raise ValueError("non-ok metrics cannot contain timing samples")
        return metric
    if sequence_samples_us is None:
        raise ValueError("ok metrics require timing samples")
    sequence = _finite_samples(sequence_samples_us)
    metric["sequence_us"] = summarize_samples(sequence)
    metric["per_iteration_us"] = summarize_samples(
        value / logical_iterations for value in sequence
    )
    return metric


def make_timing_control(
    *,
    logical_iterations: int,
    dispatches_per_iteration: int,
    gpu_samples_us: Iterable[float] | None,
    host_samples_us: Iterable[float] | None,
    gpu_clock: str,
    host_clock: str = "steady_clock",
    gpu_status: str = "ok",
    host_status: str = "ok",
) -> dict[str, Any]:
    if logical_iterations <= 0 or dispatches_per_iteration <= 0:
        raise ValueError("logical_iterations and dispatches_per_iteration must be positive")
    return {
        "logical_iterations": logical_iterations,
        "dispatches_per_iteration": dispatches_per_iteration,
        "gpu_elapsed": make_metric(
            clock=gpu_clock,
            logical_iterations=logical_iterations,
            sequence_samples_us=gpu_samples_us,
            status=gpu_status,
        ),
        "host_wall": make_metric(
            clock=host_clock,
            logical_iterations=logical_iterations,
            sequence_samples_us=host_samples_us,
            status=host_status,
        ),
    }


def make_dependency_contract(
    *,
    timing_mode: str,
    backend: str,
    validation_status: str,
) -> dict[str, str]:
    mode = parse_timing_mode(timing_mode)
    if backend not in TIMED_BACKENDS:
        choices = ", ".join(TIMED_BACKENDS)
        raise ValueError(f"timed cross-backend rows require one of: {choices}")
    if validation_status not in {"pass", "fail", "not_run"}:
        raise ValueError("invalid dependency validation status")
    if mode == "serial_latency":
        return {
            "work_dependency": "chained",
            "inter_dispatch_ordering": {
                "hip": "hip_stream_order",
                "vulkan": "vulkan_compute_barrier",
                "redline": "redline_rmw",
            }[backend],
            "output_partitioning": "chained_shared",
            "validation_status": validation_status,
        }
    return {
        "work_dependency": "independent",
        "inter_dispatch_ordering": "none",
        "output_partitioning": "disjoint",
        "validation_status": validation_status,
    }


def make_submission(
    *,
    strategy: str,
    queue_or_stream_count: int,
    recording_in_timed_region: bool,
    submit_in_host_wall: bool = True,
    completion_in_host_wall: bool = True,
) -> dict[str, Any]:
    if queue_or_stream_count <= 0:
        raise ValueError("queue_or_stream_count must be positive")
    return {
        "strategy": strategy,
        "recording_in_timed_region": recording_in_timed_region,
        "submit_in_host_wall": submit_in_host_wall,
        "completion_in_host_wall": completion_in_host_wall,
        "queue_or_stream_count": queue_or_stream_count,
    }


def make_correctness(
    *,
    status: str,
    oracle: str,
    logical_iterations: int,
    coverage: str,
    synchronization_method: str,
    barrier_count: int,
) -> dict[str, Any]:
    if status not in {"pass", "fail", "not_run"}:
        raise ValueError("invalid correctness status")
    if logical_iterations <= 0 or barrier_count < 0:
        raise ValueError("invalid correctness iteration or barrier count")
    if coverage not in {"all_dispatches", "chained_final_state"}:
        raise ValueError("invalid timed-sequence correctness coverage")
    return {
        "single_dispatch": {"status": status, "oracle": oracle},
        "timed_sequence": {
            "status": status,
            "oracle": oracle,
            "logical_iterations": logical_iterations,
            "coverage": coverage,
        },
        "synchronization": {
            "status": status,
            "method": synchronization_method,
            "barrier_count": barrier_count,
        },
    }


def make_timed_row_contract(
    *,
    timing_mode: str,
    backend: str,
    repetitions: int,
    dispatches_per_iteration: int,
    dependency_validation_status: str,
    submission: dict[str, Any],
    single_timing: dict[str, Any],
    burst_timing: dict[str, Any],
    correctness: dict[str, Any],
) -> dict[str, Any]:
    contract = {
        "timing_mode": parse_timing_mode(timing_mode),
        "dependency_contract": make_dependency_contract(
            timing_mode=timing_mode,
            backend=backend,
            validation_status=dependency_validation_status,
        ),
        "submission": submission,
        "timing": {"single": single_timing, "burst": burst_timing},
        "correctness": correctness,
    }
    validate_timed_row(contract, expected_repetitions=repetitions)
    if burst_timing["dispatches_per_iteration"] != dispatches_per_iteration:
        raise ValueError("burst dispatches_per_iteration does not match row contract")
    return contract


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_metric(metric: dict[str, Any], name: str) -> None:
    status = metric.get("status")
    if status not in METRIC_STATUSES:
        raise ValueError(f"{name}.status is invalid")
    if not isinstance(metric.get("clock"), str) or not metric["clock"]:
        raise ValueError(f"{name}.clock is required")
    if status != "ok":
        return
    for stats_name in ("sequence_us", "per_iteration_us"):
        stats = _require_mapping(metric.get(stats_name), f"{name}.{stats_name}")
        for field in ("samples", "n", "median", "p95", "min", "max", "stdev"):
            if field not in stats:
                raise ValueError(f"{name}.{stats_name}.{field} is required")


def validate_timed_row(row: dict[str, Any], *, expected_repetitions: int | None = None) -> None:
    mode = parse_timing_mode(str(row.get("timing_mode")))
    dependency = _require_mapping(row.get("dependency_contract"), "dependency_contract")
    expected_dependency = "chained" if mode == "serial_latency" else "independent"
    expected_partition = "chained_shared" if mode == "serial_latency" else "disjoint"
    if dependency.get("work_dependency") != expected_dependency:
        raise ValueError("timing mode and work dependency disagree")
    if dependency.get("output_partitioning") != expected_partition:
        raise ValueError("timing mode and output partitioning disagree")
    if dependency.get("validation_status") != "pass":
        raise ValueError("timed dependency contract is not validated")

    submission = _require_mapping(row.get("submission"), "submission")
    for field in (
        "strategy",
        "recording_in_timed_region",
        "submit_in_host_wall",
        "completion_in_host_wall",
        "queue_or_stream_count",
    ):
        if field not in submission:
            raise ValueError(f"submission.{field} is required")

    timing = _require_mapping(row.get("timing"), "timing")
    for control_name in TIMING_CONTROLS:
        control = _require_mapping(timing.get(control_name), f"timing.{control_name}")
        logical_iterations = control.get("logical_iterations")
        if not isinstance(logical_iterations, int) or logical_iterations <= 0:
            raise ValueError(f"timing.{control_name}.logical_iterations must be positive")
        if control_name == "single" and logical_iterations != 1:
            raise ValueError("single timing control must contain one logical iteration")
        if not isinstance(control.get("dispatches_per_iteration"), int):
            raise ValueError(f"timing.{control_name}.dispatches_per_iteration is required")
        for domain in TIMING_DOMAINS:
            _validate_metric(
                _require_mapping(control.get(domain), f"timing.{control_name}.{domain}"),
                f"timing.{control_name}.{domain}",
            )
    burst_iterations = timing["burst"]["logical_iterations"]
    if expected_repetitions is not None and burst_iterations != expected_repetitions:
        raise ValueError("burst logical_iterations does not match configured repetitions")

    correctness = _require_mapping(row.get("correctness"), "correctness")
    for field in ("single_dispatch", "timed_sequence", "synchronization"):
        section = _require_mapping(correctness.get(field), f"correctness.{field}")
        if section.get("status") != "pass":
            raise ValueError(f"correctness.{field} did not pass")
    if correctness["timed_sequence"].get("logical_iterations") != burst_iterations:
        raise ValueError("timed-sequence correctness did not validate the measured burst")
    coverage = correctness["timed_sequence"].get("coverage")
    if mode == "independent_throughput" and coverage != "all_dispatches":
        raise ValueError("independent throughput must validate every disjoint output")


def dependency_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    validate_timed_row(row)
    dependency = row["dependency_contract"]
    return (
        row["timing_mode"],
        dependency["work_dependency"],
        dependency["output_partitioning"],
    )


def submission_class(row: dict[str, Any]) -> tuple[Any, ...]:
    validate_timed_row(row)
    submission = row["submission"]
    strategy = submission["strategy"]
    if strategy in _PRE_RECORDED_STRATEGIES:
        strategy_class = "pre_recorded_replay"
    elif strategy in _HOST_ENQUEUED_STRATEGIES:
        strategy_class = "host_enqueued"
    else:
        strategy_class = strategy
    return (
        strategy_class,
        submission["recording_in_timed_region"],
        submission["submit_in_host_wall"],
        submission["completion_in_host_wall"],
    )


def backend_pair_ratio(
    lhs_row: dict[str, Any],
    rhs_row: dict[str, Any],
    *,
    lhs_backend: str,
    rhs_backend: str,
    control: str,
    domain: str,
) -> dict[str, Any]:
    """Compare two timing-contract rows without assuming a HIP/Vulkan pair.

    ``lhs_vs_rhs_speedup`` is ``rhs_time / lhs_time`` and therefore exceeds one
    when the left-hand backend is faster. ``rhs_vs_lhs_speedup`` is its inverse.
    """

    if lhs_backend not in TIMED_BACKENDS or rhs_backend not in TIMED_BACKENDS:
        raise ValueError("comparison backends must be timed backends")
    if lhs_backend == rhs_backend:
        raise ValueError("comparison backends must differ")
    if control not in TIMING_CONTROLS or domain not in TIMING_DOMAINS:
        raise ValueError("invalid timing control or domain")
    if dependency_signature(lhs_row) != dependency_signature(rhs_row):
        raise ValueError(f"{lhs_backend} and {rhs_backend} rows are not comparable")
    if domain == "host_wall" and submission_class(lhs_row) != submission_class(rhs_row):
        raise ValueError(
            f"{lhs_backend} and {rhs_backend} host-wall submission contracts are not comparable"
        )
    lhs_control = lhs_row["timing"][control]
    rhs_control = rhs_row["timing"][control]
    for field in ("logical_iterations", "dispatches_per_iteration"):
        if lhs_control[field] != rhs_control[field]:
            raise ValueError(
                f"{lhs_backend} and {rhs_backend} {control} {field} values differ"
            )
    lhs_metric = lhs_control[domain]
    rhs_metric = rhs_control[domain]
    if lhs_metric["status"] != "ok" or rhs_metric["status"] != "ok":
        raise ValueError(f"{domain} is unavailable for one or both backends")
    lhs_us = float(lhs_metric["per_iteration_us"]["median"])
    rhs_us = float(rhs_metric["per_iteration_us"]["median"])
    if lhs_us <= 0.0 or rhs_us <= 0.0:
        raise ValueError("timing medians must be positive for ratio calculation")
    return {
        "lhs_backend": lhs_backend,
        "rhs_backend": rhs_backend,
        "lhs_us_per_iteration": lhs_us,
        "rhs_us_per_iteration": rhs_us,
        "lhs_vs_rhs_speedup": rhs_us / lhs_us,
        "rhs_vs_lhs_speedup": lhs_us / rhs_us,
    }


def comparison_ratio(
    hip_row: dict[str, Any],
    vulkan_row: dict[str, Any],
    *,
    control: str,
    domain: str,
) -> dict[str, float]:
    """Backward-compatible HIP/Vulkan ratio used by existing paired runners."""

    pair = backend_pair_ratio(
        hip_row,
        vulkan_row,
        lhs_backend="hip",
        rhs_backend="vulkan",
        control=control,
        domain=domain,
    )
    return {
        "hip_us_per_iteration": float(pair["lhs_us_per_iteration"]),
        "vulkan_us_per_iteration": float(pair["rhs_us_per_iteration"]),
        "vulkan_vs_hip_speedup": float(pair["rhs_vs_lhs_speedup"]),
    }
