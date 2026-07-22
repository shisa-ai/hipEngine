#!/usr/bin/env python3
"""Counterbalanced IQ2_XS tuning benchmark on synthetic Laguna geometry.

This harness measures raw IQ2_XS selected decode and compact grouped prefill.
It uses deterministic valid blocks, rotates selected experts to distinguish
cold/distinct-weight behavior from cache-hot routes, and checks the production
shape against the existing exact GPU fallbacks before timing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

import numpy as np

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import (
    build_gguf_iq_gemv,
    gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_iq2_xs_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill import (
    build_gguf_iq_selected_prefill,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
)


IQ2_XS_BLOCK_BYTES = 74
QK_K = 256


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "decode", "prefill"), default="all")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=3072)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--decode-patterns", default="rotating,hot,repeated")
    parser.add_argument("--decode-threads", default="256")
    parser.add_argument("--route-sets", type=int, default=32)
    parser.add_argument("--prefill-tokens", default="16,32,64,128,512")
    parser.add_argument("--distributions", default="balanced,hot,zipf")
    parser.add_argument("--warmup-ms", type=float, default=250.0)
    parser.add_argument("--decode-iters", type=int, default=200)
    parser.add_argument("--prefill-iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0x12A5)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--hardware-gpu", default="AMD Radeon RX 7900 XTX")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"expected comma-separated integers, got {value!r}") from exc
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _csv_names(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated name")
    return values


def parse_decode_threads(value: str) -> tuple[int, ...]:
    threads = _csv_ints(value)
    if any(item not in {64, 128, 256} for item in threads):
        raise ValueError("decode threads must be selected from 64, 128, and 256")
    if len(set(threads)) != len(threads):
        raise ValueError("decode threads must not contain duplicates")
    return threads


def validate_args(args: argparse.Namespace) -> None:
    if args.experts <= 0:
        raise ValueError("experts must be positive")
    if args.in_features <= 0 or args.in_features % QK_K:
        raise ValueError("in_features must be positive and divisible by 256")
    if args.out_features <= 0:
        raise ValueError("out_features must be positive")
    if args.top_k <= 0 or args.top_k > args.experts:
        raise ValueError("top_k must be in [1, experts]")
    parse_decode_threads(args.decode_threads)
    if args.route_sets <= 0:
        raise ValueError("route_sets must be positive")
    if args.warmup_ms < 0:
        raise ValueError("warmup_ms must be non-negative")
    if args.decode_iters <= 0 or args.prefill_iters <= 0 or args.repeats <= 0:
        raise ValueError("iteration and repeat counts must be positive")
    if any(value <= 0 for value in _csv_ints(args.prefill_tokens)):
        raise ValueError("prefill token counts must be positive")
    allowed_distributions = {"balanced", "hot", "zipf"}
    distributions = set(_csv_names(args.distributions))
    if not distributions <= allowed_distributions:
        raise ValueError(f"distribution must be one of {sorted(allowed_distributions)}")
    allowed_patterns = {"rotating", "hot", "repeated"}
    patterns = set(_csv_names(args.decode_patterns))
    if not patterns <= allowed_patterns:
        raise ValueError(f"decode pattern must be one of {sorted(allowed_patterns)}")


def raw_weight_bytes_per_dispatch(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    matrices: int,
) -> int:
    if in_features <= 0 or in_features % QK_K:
        raise ValueError("in_features must be positive and divisible by 256")
    if rows < 0 or out_features <= 0 or matrices <= 0:
        raise ValueError("rows must be non-negative and output/matrix counts positive")
    row_bytes = (in_features // QK_K) * IQ2_XS_BLOCK_BYTES
    return rows * out_features * row_bytes * matrices


def grouped_raw_weight_bytes_per_dispatch(
    counts: np.ndarray,
    *,
    in_features: int,
    out_features: int,
    matrices: int,
    row_batch: int,
) -> int:
    counts = np.asarray(counts, dtype=np.int64)
    if row_batch <= 0:
        raise ValueError("row_batch must be positive")
    weight_visits = int(np.sum((counts + row_batch - 1) // row_batch))
    return raw_weight_bytes_per_dispatch(
        rows=weight_visits,
        in_features=in_features,
        out_features=out_features,
        matrices=matrices,
    )


def build_expert_counts(
    *,
    num_experts: int,
    assignments: int,
    distribution: str,
    seed: int,
) -> np.ndarray:
    del seed  # Distributions are deterministic; seed is reserved for held-out variants.
    if num_experts <= 0 or assignments < 0:
        raise ValueError("num_experts must be positive and assignments non-negative")
    if distribution not in {"balanced", "hot", "zipf"}:
        raise ValueError("distribution must be balanced, hot, or zipf")
    counts = np.zeros(num_experts, dtype=np.int64)
    if assignments == 0:
        return counts
    if distribution == "balanced":
        quotient, remainder = divmod(assignments, num_experts)
        counts.fill(quotient)
        counts[:remainder] += 1
        return counts
    if distribution == "hot":
        hot_experts = min(8, num_experts)
        hot_assignments = (assignments + 1) // 2
        hot_q, hot_r = divmod(hot_assignments, hot_experts)
        counts[:hot_experts] = hot_q
        counts[:hot_r] += 1
        remaining = assignments - hot_assignments
        if num_experts == hot_experts:
            q, r = divmod(remaining, hot_experts)
            counts[:hot_experts] += q
            counts[:r] += 1
        else:
            q, r = divmod(remaining, num_experts - hot_experts)
            counts[hot_experts:] = q
            counts[hot_experts : hot_experts + r] += 1
        return counts

    weights = 1.0 / np.arange(1, num_experts + 1, dtype=np.float64)
    exact = weights * (float(assignments) / float(weights.sum()))
    counts = np.floor(exact).astype(np.int64)
    remainder = assignments - int(counts.sum())
    if remainder:
        fractions = exact - counts
        order = np.argsort(-fractions, kind="stable")
        counts[order[:remainder]] += 1
    return counts


def build_decode_route_sets(
    *,
    num_experts: int,
    top_k: int,
    route_sets: int,
    pattern: str,
    seed: int,
) -> tuple[np.ndarray, ...]:
    if num_experts <= 0 or top_k <= 0 or top_k > num_experts or route_sets <= 0:
        raise ValueError("invalid expert/top_k/route_sets contract")
    if pattern not in {"rotating", "hot", "repeated"}:
        raise ValueError("pattern must be rotating, hot, or repeated")
    if pattern == "hot":
        row = np.arange(top_k, dtype=np.int64)
        return tuple(row.copy() for _ in range(route_sets))
    if pattern == "repeated":
        row = np.full(top_k, seed % num_experts, dtype=np.int64)
        return tuple(row.copy() for _ in range(route_sets))
    rows = []
    for set_index in range(route_sets):
        start = (set_index * top_k) % num_experts
        rows.append((start + np.arange(top_k, dtype=np.int64)) % num_experts)
    return tuple(rows)


def counterbalanced_order(names: Sequence[str], repeat: int) -> tuple[str, ...]:
    if not names:
        raise ValueError("names must be non-empty")
    order = list(names if repeat % 2 == 0 else reversed(names))
    shift = (repeat // 2) % len(order)
    return tuple(order[shift:] + order[:shift])


def _f32_to_bf16_u16(array: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(array, dtype=np.float32)
    bits = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    out = ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)
    out[nan_mask] = np.uint16(0x7FC0)
    return out.reshape(f32.shape)


def _bf16_u16_to_f32(array: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(array, dtype=np.uint16)
    return (
        (bits.astype(np.uint32) << np.uint32(16))
        .view(np.float32)
        .reshape(bits.shape)
        .copy()
    )


def _make_x(rows: int, features: int) -> np.ndarray:
    row_ids = np.arange(rows, dtype=np.int32)[:, None]
    feature_ids = np.arange(features, dtype=np.int32)[None, :]
    values = (row_ids * np.int32(17) + feature_ids) % np.int32(29)
    return (values - np.int32(14)).astype(np.float32) / np.float32(32.0)


def _make_iq2_weight(
    num_experts: int,
    out_features: int,
    in_features: int,
    *,
    seed: int,
) -> np.ndarray:
    blocks = in_features // QK_K
    rng = np.random.default_rng(seed)
    shaped = rng.integers(
        0,
        256,
        size=(num_experts, out_features, blocks, IQ2_XS_BLOCK_BYTES),
        dtype=np.uint8,
    )
    scale_ids = np.arange(num_experts * out_features * blocks, dtype=np.uint32)
    scales = (
        np.float32(0.001953125)
        * (np.float32(1.0) + (scale_ids % np.uint32(5)).astype(np.float32))
    ).astype(np.float16)
    scale_bytes = scales.view(np.uint8).reshape(num_experts, out_features, blocks, 2)
    shaped[..., :2] = scale_bytes
    return shaped.reshape(num_experts, out_features, blocks * IQ2_XS_BLOCK_BYTES)


def _copy(array: np.ndarray):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
    return buffer


def _read(buffer, *, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes)
    return out


def _warm_launches(
    runtime: HipRuntime,
    launches: Sequence[Callable[[], None]],
    *,
    warmup_ms: float,
) -> int:
    if warmup_ms <= 0:
        return 0
    deadline = time.perf_counter() + warmup_ms / 1000.0
    count = 0
    while True:
        for launch in launches:
            launch()
            count += 1
        runtime.device_synchronize()
        if time.perf_counter() >= deadline:
            return count


def _time_launch(
    runtime: HipRuntime,
    launch: Callable[[], None],
    *,
    iterations: int,
) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(iterations):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return runtime.event_elapsed_time_ms(start, stop) / iterations
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)


def _measure_counterbalanced(
    runtime: HipRuntime,
    launches: dict[str, Callable[[], None]],
    *,
    warmup_ms: float,
    iterations: int,
    repeats: int,
) -> tuple[int, dict[str, list[float]]]:
    warmup_launches = _warm_launches(
        runtime,
        tuple(launches.values()),
        warmup_ms=warmup_ms,
    )
    samples = {name: [] for name in launches}
    for repeat in range(repeats):
        for name in counterbalanced_order(tuple(launches), repeat):
            samples[name].append(
                _time_launch(runtime, launches[name], iterations=iterations)
            )
    return warmup_launches, samples


def _sample_summary(samples_ms: Sequence[float], *, raw_bytes: int) -> dict[str, object]:
    med = float(median(samples_ms))
    return {
        "samples_ms": [float(value) for value in samples_ms],
        "median_ms": med,
        "raw_weight_bytes_per_dispatch": raw_bytes,
        "approx_effective_gb_s": float(raw_bytes / (med * 1.0e6)),
    }


def _selected_launch(
    *,
    library,
    runtime: HipRuntime,
    x_ptr: int,
    selected_ptrs: Sequence[int],
    weight_ptr: int,
    out_ptr: int,
    threads: int,
    args: argparse.Namespace,
) -> Callable[[], None]:
    index = [0]

    def launch() -> None:
        selected_ptr = selected_ptrs[index[0] % len(selected_ptrs)]
        index[0] += 1
        gguf_iq2_xs_selected_gemv_bf16_bf16_out(
            x_ptr,
            selected_ptr,
            weight_ptr,
            out_ptr,
            x_rows=1,
            rows=args.top_k,
            num_experts=args.experts,
            in_features=args.in_features,
            out_features=args.out_features,
            threads=threads,
            library=library,
            runtime=runtime,
        )

    return launch


def _dual_launch(
    *,
    library,
    runtime: HipRuntime,
    x_ptr: int,
    selected_ptrs: Sequence[int],
    gate_ptr: int,
    up_ptr: int,
    out_ptr: int,
    threads: int,
    args: argparse.Namespace,
) -> Callable[[], None]:
    index = [0]

    def launch() -> None:
        selected_ptr = selected_ptrs[index[0] % len(selected_ptrs)]
        index[0] += 1
        gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out(
            x_ptr,
            selected_ptr,
            gate_ptr,
            up_ptr,
            out_ptr,
            x_rows=1,
            rows=args.top_k,
            num_experts=args.experts,
            in_features=args.in_features,
            out_features=args.out_features,
            threads=threads,
            library=library,
            runtime=runtime,
        )

    return launch


def _decode_correctness(
    *,
    library,
    runtime: HipRuntime,
    x_ptr: int,
    selected_ptr: int,
    gate_ptr: int,
    up_ptr: int,
    out_buf,
    args: argparse.Namespace,
) -> dict[str, object]:
    out_ptr = out_buf.ptr
    shape = (args.top_k, args.out_features)
    common = dict(
        x_rows=1,
        rows=args.top_k,
        num_experts=args.experts,
        in_features=args.in_features,
        out_features=args.out_features,
        library=library,
        runtime=runtime,
    )

    def run_single(weight_ptr: int, threads: int) -> np.ndarray:
        gguf_iq2_xs_selected_gemv_bf16_bf16_out(
            x_ptr,
            selected_ptr,
            weight_ptr,
            out_ptr,
            threads=threads,
            **common,
        )
        return _read(out_buf, shape=shape, dtype=np.dtype(np.uint16))

    def run_dual(threads: int) -> np.ndarray:
        gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out(
            x_ptr,
            selected_ptr,
            gate_ptr,
            up_ptr,
            out_ptr,
            threads=threads,
            **common,
        )
        return _read(out_buf, shape=shape, dtype=np.dtype(np.uint16))

    gate_bits = run_single(gate_ptr, 256)
    up_bits = run_single(up_ptr, 256)
    gate = _bf16_u16_to_f32(gate_bits)
    up = _bf16_u16_to_f32(up_bits)
    expected = _f32_to_bf16_u16(
        gate
        * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate)))
        * up
    )
    actual = run_dual(256)
    mismatches = int(np.count_nonzero(actual != expected))
    geometry = {}
    for threads in parse_decode_threads(args.decode_threads):
        candidate_gate = run_single(gate_ptr, threads)
        candidate_up = run_single(up_ptr, threads)
        candidate_dual = run_dual(threads)
        projection_reference = np.concatenate((gate_bits, up_bits), axis=1)
        projection_candidate = np.concatenate((candidate_gate, candidate_up), axis=1)
        projection_result = evaluate_logits(
            _bf16_u16_to_f32(projection_reference),
            _bf16_u16_to_f32(projection_candidate),
        )
        dual_result = evaluate_logits(
            _bf16_u16_to_f32(expected),
            _bf16_u16_to_f32(candidate_dual),
        )
        geometry[str(threads)] = {
            "passed": projection_result.passed and dual_result.passed,
            "projection_bf16_bit_mismatches": int(
                np.count_nonzero(projection_reference != projection_candidate)
            ),
            "dual_bf16_bit_mismatches": int(
                np.count_nonzero(expected != candidate_dual)
            ),
            "projection_kl_max": projection_result.kl_max,
            "projection_top1": projection_result.top1_agreement,
            "dual_kl_max": dual_result.kl_max,
            "dual_top1": dual_result.top1_agreement,
        }
    return {
        "passed": mismatches == 0
        and all(bool(result["passed"]) for result in geometry.values()),
        "bf16_bit_mismatches": mismatches,
        "elements": int(actual.size),
        "oracle": "selected-single gate/up plus BF16-boundary SiLU",
        "geometry_vs_threads256": geometry,
    }


def _run_decode(
    *,
    library,
    runtime: HipRuntime,
    gate_buf,
    up_buf,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    x = _f32_to_bf16_u16(_make_x(1, args.in_features))
    x_buf = _copy(x)
    out_buf = malloc(args.top_k * args.out_features * np.dtype(np.uint16).itemsize)
    selected_buffers = []
    try:
        boundary = np.arange(args.top_k, dtype=np.int64)
        if args.top_k >= 2:
            boundary[-1] = args.experts - 1
        boundary_buf = _copy(boundary)
        selected_buffers.append(boundary_buf)
        correctness = (
            {"skipped": True}
            if args.skip_correctness
            else _decode_correctness(
                library=library,
                runtime=runtime,
                x_ptr=x_buf.ptr,
                selected_ptr=boundary_buf.ptr,
                gate_ptr=gate_buf.ptr,
                up_ptr=up_buf.ptr,
                out_buf=out_buf,
                args=args,
            )
        )
        rows = []
        for pattern in _csv_names(args.decode_patterns):
            host_sets = build_decode_route_sets(
                num_experts=args.experts,
                top_k=args.top_k,
                route_sets=args.route_sets,
                pattern=pattern,
                seed=args.seed,
            )
            pattern_buffers = [_copy(row) for row in host_sets]
            selected_buffers.extend(pattern_buffers)
            selected_ptrs = tuple(buffer.ptr for buffer in pattern_buffers)
            thread_counts = parse_decode_threads(args.decode_threads)
            use_legacy_names = thread_counts == (256,)
            launches = {}
            launch_meta = {}
            for threads in thread_counts:
                suffix = "" if use_legacy_names else f"_t{threads}"
                single_name = f"selected_single{suffix}"
                dual_name = f"selected_dual_silu{suffix}"
                launches[single_name] = _selected_launch(
                    library=library,
                    runtime=runtime,
                    x_ptr=x_buf.ptr,
                    selected_ptrs=selected_ptrs,
                    weight_ptr=gate_buf.ptr,
                    out_ptr=out_buf.ptr,
                    threads=threads,
                    args=args,
                )
                launches[dual_name] = _dual_launch(
                    library=library,
                    runtime=runtime,
                    x_ptr=x_buf.ptr,
                    selected_ptrs=selected_ptrs,
                    gate_ptr=gate_buf.ptr,
                    up_ptr=up_buf.ptr,
                    out_ptr=out_buf.ptr,
                    threads=threads,
                    args=args,
                )
                launch_meta[single_name] = (threads, 1)
                launch_meta[dual_name] = (threads, 2)
            warmup_count, samples = _measure_counterbalanced(
                runtime,
                launches,
                warmup_ms=args.warmup_ms,
                iterations=args.decode_iters,
                repeats=args.repeats,
            )
            row = {
                "pattern": pattern,
                "route_sets": args.route_sets,
                "warmup_launches": warmup_count,
            }
            for name, (threads, matrices) in launch_meta.items():
                row[name] = {
                    "threads": threads,
                    **_sample_summary(
                        samples[name],
                        raw_bytes=raw_weight_bytes_per_dispatch(
                            rows=args.top_k,
                            in_features=args.in_features,
                            out_features=args.out_features,
                            matrices=matrices,
                        ),
                    ),
                }
            rows.append(row)
        return correctness, {"cases": rows}
    finally:
        for buffer in reversed(selected_buffers):
            free(buffer)
        free(out_buf)
        free(x_buf)


def _prefill_launch(
    wrapper,
    *,
    library,
    runtime: HipRuntime,
    x_ptr: int,
    starts_ptr: int,
    gate_ptr: int,
    up_ptr: int,
    out_ptr: int,
    compact_rows: int,
    args: argparse.Namespace,
) -> Callable[[], None]:
    def launch() -> None:
        wrapper(
            x_ptr,
            starts_ptr,
            gate_ptr,
            up_ptr,
            out_ptr,
            compact_rows=compact_rows,
            in_features=args.in_features,
            out_features=args.out_features,
            num_experts=args.experts,
            library=library,
            runtime=runtime,
        )

    return launch


def _run_prefill_case(
    *,
    library,
    runtime: HipRuntime,
    gate_buf,
    up_buf,
    args: argparse.Namespace,
    tokens: int,
    distribution: str,
    check_correctness: bool,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    compact_rows = tokens * args.top_k
    counts = build_expert_counts(
        num_experts=args.experts,
        assignments=compact_rows,
        distribution=distribution,
        seed=args.seed,
    )
    starts = np.zeros(args.experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    x = _f32_to_bf16_u16(_make_x(compact_rows, args.in_features))
    out_shape = (compact_rows, 2 * args.out_features)
    x_buf = _copy(x)
    starts_buf = _copy(starts)
    out_buf = malloc(int(np.prod(out_shape)) * np.dtype(np.uint16).itemsize)
    try:
        base = _prefill_launch(
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
            library=library,
            runtime=runtime,
            x_ptr=x_buf.ptr,
            starts_ptr=starts_buf.ptr,
            gate_ptr=gate_buf.ptr,
            up_ptr=up_buf.ptr,
            out_ptr=out_buf.ptr,
            compact_rows=compact_rows,
            args=args,
        )
        rowbatch4 = _prefill_launch(
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
            library=library,
            runtime=runtime,
            x_ptr=x_buf.ptr,
            starts_ptr=starts_buf.ptr,
            gate_ptr=gate_buf.ptr,
            up_ptr=up_buf.ptr,
            out_ptr=out_buf.ptr,
            compact_rows=compact_rows,
            args=args,
        )
        correctness = None
        if check_correctness:
            base()
            expected = _read(out_buf, shape=out_shape, dtype=np.dtype(np.uint16))
            rowbatch4()
            actual = _read(out_buf, shape=out_shape, dtype=np.dtype(np.uint16))
            mismatches = int(np.count_nonzero(actual != expected))
            correctness = {
                "passed": mismatches == 0,
                "bf16_bit_mismatches": mismatches,
                "elements": int(actual.size),
                "oracle": "grouped scalar",
            }
        launches = {"base": base, "rowbatch4": rowbatch4}
        warmup_count, samples = _measure_counterbalanced(
            runtime,
            launches,
            warmup_ms=args.warmup_ms,
            iterations=args.prefill_iters,
            repeats=args.repeats,
        )
        active = int(np.count_nonzero(counts))
        case = {
            "tokens": tokens,
            "compact_rows": compact_rows,
            "distribution": distribution,
            "active_experts": active,
            "count_min": int(counts.min()),
            "count_max": int(counts.max()),
            "count_mean_active": float(compact_rows / active),
            "warmup_launches": warmup_count,
            "base": _sample_summary(
                samples["base"],
                raw_bytes=grouped_raw_weight_bytes_per_dispatch(
                    counts,
                    in_features=args.in_features,
                    out_features=args.out_features,
                    matrices=2,
                    row_batch=1,
                ),
            ),
            "rowbatch4": _sample_summary(
                samples["rowbatch4"],
                raw_bytes=grouped_raw_weight_bytes_per_dispatch(
                    counts,
                    in_features=args.in_features,
                    out_features=args.out_features,
                    matrices=2,
                    row_batch=4,
                ),
            ),
        }
        case["rowbatch4_vs_base_percent"] = float(
            100.0
            * (
                case["rowbatch4"]["median_ms"] / case["base"]["median_ms"]
                - 1.0
            )
        )
        return correctness, case
    finally:
        free(out_buf)
        free(starts_buf)
        free(x_buf)


def _run_prefill(
    *,
    library,
    runtime: HipRuntime,
    gate_buf,
    up_buf,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    correctness_rows = []
    cases = []
    first = True
    for tokens in _csv_ints(args.prefill_tokens):
        for distribution in _csv_names(args.distributions):
            correctness, case = _run_prefill_case(
                library=library,
                runtime=runtime,
                gate_buf=gate_buf,
                up_buf=up_buf,
                args=args,
                tokens=tokens,
                distribution=distribution,
                check_correctness=first and not args.skip_correctness,
            )
            first = False
            if correctness is not None:
                correctness_rows.append(
                    {"tokens": tokens, "distribution": distribution, **correctness}
                )
            cases.append(case)
    correctness_payload: dict[str, object]
    if args.skip_correctness:
        correctness_payload = {"skipped": True}
    else:
        correctness_payload = {
            "passed": all(bool(row["passed"]) for row in correctness_rows),
            "cases": correctness_rows,
        }
    return correctness_payload, {"cases": cases}


def main(argv: Sequence[str] | None = None) -> None:
    invocation_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    compiler_version = (
        args.compiler_version_file.read_text()
        if args.compiler_version_file is not None
        else None
    )
    runtime = get_hip_runtime()
    direct_library = (
        build_gguf_iq_gemv(
            load=True,
            compiler_version=compiler_version,
            require_cached=args.require_cached_build,
        )
        if args.mode in {"all", "decode"}
        else None
    )
    prefill_library = (
        build_gguf_iq_selected_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=args.require_cached_build,
        )
        if args.mode in {"all", "prefill"}
        else None
    )

    gate = _make_iq2_weight(
        args.experts,
        args.out_features,
        args.in_features,
        seed=args.seed,
    )
    gate_buf = _copy(gate)
    del gate
    up = _make_iq2_weight(
        args.experts,
        args.out_features,
        args.in_features,
        seed=args.seed + 1,
    )
    up_buf = _copy(up)
    del up
    try:
        correctness = {}
        payload: dict[str, object] = {
            "schema": "hipengine.iq2_xs_tuning_bench.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "synthetic Laguna-shaped raw IQ2_XS primitives; no model throughput",
            "command": "PYTHONPATH=. python3 scripts/iq2_xs_tuning_bench.py "
            + shlex.join(invocation_args),
            "hardware": {
                "gpu_index": os.environ.get("HIP_VISIBLE_DEVICES"),
                "name": args.hardware_gpu,
                "arch": os.environ.get("HIPENGINE_HIP_ARCH", "gfx1100"),
            },
            "shape": {
                "experts": args.experts,
                "in_features": args.in_features,
                "out_features": args.out_features,
                "top_k": args.top_k,
            },
            "timing": {
                "warmup_ms_per_case": args.warmup_ms,
                "decode_iterations": args.decode_iters,
                "prefill_iterations": args.prefill_iters,
                "repeats": args.repeats,
                "counterbalanced": True,
            },
            "source_lineage": {
                "llama_cpp": "1ebf790cda38d827559548f67b0469189690cc8c",
                "qwen_kernel": "52e240f9c6d91750d0e5e692976cfb67fd9bc603",
            },
        }
        if args.mode in {"all", "decode"}:
            decode_correctness, decode = _run_decode(
                library=direct_library,
                runtime=runtime,
                gate_buf=gate_buf,
                up_buf=up_buf,
                args=args,
            )
            correctness["decode"] = decode_correctness
            payload["decode"] = decode
        if args.mode in {"all", "prefill"}:
            prefill_correctness, prefill = _run_prefill(
                library=prefill_library,
                runtime=runtime,
                gate_buf=gate_buf,
                up_buf=up_buf,
                args=args,
            )
            correctness["prefill"] = prefill_correctness
            payload["prefill"] = prefill
        payload["correctness"] = correctness
        print(json.dumps(payload, indent=2))
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        free(up_buf)
        free(gate_buf)


if __name__ == "__main__":
    main()
