#!/usr/bin/env python3
"""Gate exact natural-shape Laguna global-attention reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import numpy as np

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
    build_laguna_kv_attention,
    laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed40_local1024_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    laguna_global_attention_decode_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_bf16_spans,
    laguna_global_attention_decode_wmma_gqa6_k64_three_term_raw_numerator_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_dim32_vstage64_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_deferrednorm_dim32_vstage64_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage64_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch8_dense_prefix_nontemporal_key_value_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch16_dense_prefix_nontemporal_key_value_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_dim32_vstage64_t256_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_tile4_dim32_vstage64_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_dim64_vstage64_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_gqa6_dim64_vstage32_bf16_spans,
    laguna_global_attention_decode_split_exact_gated_qhead_dim32_direct_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim32_vstage64_ctx4096_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim32_vstage64_ctx4096_compensated_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_compensated_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_compensated_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_nontemporal_key_value_bf16_spans,
    laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_nontemporal_key_value_bf16_spans,
)
from hipengine.loading.laguna_gguf import FULL_ATTENTION
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache


CAPACITY = 4096
Q_HEADS = 48
KV_HEADS = 8
HEAD_DIM = 128
LIVE_COUNTS = (513, 576, 639)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--burst", type=int, default=50)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument(
        "--live-counts",
        default=",".join(str(value) for value in LIVE_COUNTS),
    )
    parser.add_argument(
        "--candidate",
        choices=(
            "fixedshape",
            "split-gqa6-dim32-vstage64",
            "split-gqa6-deferrednorm-dim32-vstage64",
            "split-gqa6-tokenloop4-deferrednorm-dim32-vstage64",
            "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80",
            "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix",
            "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix-nontemporal",
            "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix-nontemporal-key-value",
            "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-prefetch8-dense-prefix-nontemporal-key-value",
            "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-prefetch16-dense-prefix-nontemporal-key-value",
            "split-gqa6-dim32-vstage64-t256",
            "split-gqa6-tile4-dim32-vstage64",
            "split-gqa6-dim64-vstage64",
            "split-gqa6-dim64-vstage32",
            "split-qhead-dim32-direct",
            "split-gqa6-dim32-vstage64-ctx4096",
            "split-gqa6-dim64-vstage64-ctx4096",
            "split-gqa6-dim64-vstage64-ctx4096-compensated",
            "split-gqa6-dim64-vstage64-ctx4096-tokenloop4",
            "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-compensated",
            "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm",
            "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated",
            "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-dense-prefix",
            "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated-dense-prefix",
            "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-dense-prefix-nontemporal-key-value",
            "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated-dense-prefix-nontemporal-key-value",
            "fused-gqa1",
            "fused-gqa2-vstage64",
            "fused-gqa2-vstage64-vec16",
            "fused-gqa2-vstage64-vec16-direct",
            "fused-gqa2-vstage64-vec16-direct-assume-exp",
            "fused-gqa2-exp32-vstage64-vec16-direct-assume-exp",
            "fused-mixed32-exp32-vstage64-vec16-direct-assume-exp",
            "fused-mixed32-exp32-producer-max-vstage64-vec16-direct-assume-exp",
            "fused-mixed32-exp32-producer-max-dpp-qk-vstage64-vec16-direct-assume-exp",
            "fused-mixed32-exp32-producer-max-dpp-qk-probability-vec4-vstage64-vec16-direct-assume-exp",
            "fused-mixed32-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
            "fused-mixed32-local512-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
            "fused-mixed40-local512-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
            "fused-mixed40-local512-exp32-producer-max-dpp-qk-dense-prefix-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
            "fused-mixed40-local512-exp32-producer-max-dpp-qk-dense-prefix-idle-double-buffer-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
            "fused-mixed40-local1024-exp32-producer-max-dpp-qk-dense-prefix-idle-double-buffer-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
            "wmma-qk-three-term-mixed32-exact-pv",
            "wmma-gqa6-k64-three-term-raw-numerator",
        ),
        default="fixedshape",
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _tracked_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _time_ms(runtime, fn, burst: int) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            fn()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return float(runtime.event_elapsed_time_ms(start, stop)) / burst
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _download(runtime, buffer, dtype) -> np.ndarray:
    host = np.empty(Q_HEADS * HEAD_DIM, dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host),
        buffer,
        host.nbytes,
        runtime=runtime,
    )
    return host


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.samples <= 0 or args.warmups < 0 or args.burst <= 0:
        raise ValueError("samples/burst must be positive and warmups non-negative")
    tracked = _tracked_status()
    if tracked and not args.allow_dirty:
        raise RuntimeError("tracked worktree must be clean; use --allow-dirty")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )
    capacity = int(args.capacity)
    live_counts = tuple(
        int(value.strip())
        for value in str(args.live_counts).split(",")
        if value.strip()
    )
    if capacity <= 0 or not live_counts:
        raise ValueError("capacity and live counts must be positive")
    if min(live_counts) <= 0 or max(live_counts) > capacity:
        raise ValueError("live counts must be within [1, capacity]")

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=args.require_cached_build,
    )
    config = SimpleNamespace(
        block_count=1,
        layer_types=(FULL_ATTENTION,),
        head_counts=(Q_HEADS,),
        head_count_kv=KV_HEADS,
        key_length=HEAD_DIM,
        value_length=HEAD_DIM,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=capacity,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    max_live = max(live_counts)
    rng = np.random.default_rng(20260728)
    keys = rng.normal(
        0.0, 0.12, size=(max_live, KV_HEADS, HEAD_DIM)
    ).astype(np.float32)
    values = rng.normal(
        0.0, 0.12, size=(max_live, KV_HEADS, HEAD_DIM)
    ).astype(np.float32)
    query = rng.normal(0.0, 0.12, size=(Q_HEADS, HEAD_DIM)).astype(np.float32)
    gate = rng.normal(0.0, 0.4, size=Q_HEADS).astype(np.float32)
    allocations = []
    try:
        key_device = malloc(keys.nbytes, runtime=runtime)
        value_device = malloc(values.nbytes, runtime=runtime)
        query_device = malloc(query.nbytes, runtime=runtime)
        gate_device = malloc(gate.nbytes, runtime=runtime)
        control_context = malloc(query.nbytes, runtime=runtime)
        candidate_context = malloc(query.nbytes, runtime=runtime)
        control_gated = malloc(query.size * 2, runtime=runtime)
        candidate_gated = malloc(query.size * 2, runtime=runtime)
        score_scratch = malloc(Q_HEADS * capacity * 4, runtime=runtime)
        physical_scratch = malloc(Q_HEADS * capacity * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                gate_device,
                control_context,
                candidate_context,
                control_gated,
                candidate_gated,
                score_scratch,
                physical_scratch,
            )
        )
        for device, host in (
            (key_device, keys),
            (value_device, values),
            (query_device, query),
            (gate_device, gate),
        ):
            copy_host_to_device(
                device,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )
        cache.prepare_rows(tuple(range(max_live)))
        cache.append_rows(
            0,
            key_device.ptr,
            value_device.ptr,
            max_live,
            library=library,
        )
        cache.commit_rows()
        cache.prepare_position(max_live)
        state = cache.layer(0)
        common = (
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
        )
        results = []
        for live_count in live_counts:
            tail = (
                score_scratch.ptr,
                physical_scratch.ptr,
                state.spans,
                live_count,
                capacity,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                HEAD_DIM**-0.5,
            )
            if args.candidate in (
                "fused-gqa2-vstage64-vec16-direct-assume-exp",
                "fused-gqa2-exp32-vstage64-vec16-direct-assume-exp",
                "fused-mixed32-exp32-vstage64-vec16-direct-assume-exp",
                "fused-mixed32-exp32-producer-max-vstage64-vec16-direct-assume-exp",
                "fused-mixed32-exp32-producer-max-dpp-qk-vstage64-vec16-direct-assume-exp",
                "fused-mixed32-exp32-producer-max-dpp-qk-probability-vec4-vstage64-vec16-direct-assume-exp",
                "fused-mixed32-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
                "fused-mixed32-local512-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
                "fused-mixed40-local512-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
                "fused-mixed40-local512-exp32-producer-max-dpp-qk-dense-prefix-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
                "fused-mixed40-local512-exp32-producer-max-dpp-qk-dense-prefix-idle-double-buffer-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
                "fused-mixed40-local1024-exp32-producer-max-dpp-qk-dense-prefix-idle-double-buffer-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp",
                "wmma-qk-three-term-mixed32-exact-pv",
                "wmma-gqa6-k64-three-term-raw-numerator",
            ):
                control_kernel = laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans
                if (
                    args.candidate
                    == "fused-gqa2-exp32-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed32-exp32-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed32-exp32-producer-max-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed32-exp32-producer-max-dpp-qk-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed32-exp32-producer-max-dpp-qk-probability-vec4-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed32-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed32-local512-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed40-local512-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed40-local512-exp32-producer-max-dpp-qk-dense-prefix-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed40-local512-exp32-producer-max-dpp-qk-dense-prefix-idle-double-buffer-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "fused-mixed40-local1024-exp32-producer-max-dpp-qk-dense-prefix-idle-double-buffer-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "wmma-qk-three-term-mixed32-exact-pv"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
                elif (
                    args.candidate
                    == "wmma-gqa6-k64-three-term-raw-numerator"
                ):
                    control_kernel = laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans
            else:
                control_kernel = (
                    laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans
                    if args.candidate == "fused-gqa2-vstage64-vec16-direct"
                    else laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans
                    if args.candidate == "fused-gqa2-vstage64-vec16"
                    else laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans
                    if args.candidate == "fused-gqa2-vstage64"
                    else laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans
                    if args.candidate.startswith("fused-")
                    else laguna_global_attention_decode_split_exact_gated_bf16_spans
                )
            if args.candidate in (
                "split-gqa6-dim32-vstage64-t256",
                "split-gqa6-deferrednorm-dim32-vstage64",
                "split-gqa6-tile4-dim32-vstage64",
                "split-gqa6-dim64-vstage64",
                "split-gqa6-dim64-vstage32",
                "split-qhead-dim32-direct",
            ):
                control_kernel = laguna_global_attention_decode_split_exact_gated_gqa6_dim32_vstage64_bf16_spans
            if args.candidate in (
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage64",
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80",
            ):
                control_kernel = laguna_global_attention_decode_split_exact_gated_gqa6_deferrednorm_dim32_vstage64_bf16_spans
            if (
                args.candidate
                == "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80"
            ):
                control_kernel = laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage64_bf16_spans
            if (
                args.candidate
                == "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix"
            ):
                control_kernel = laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_bf16_spans
            if (
                args.candidate
                == "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix-nontemporal"
            ):
                control_kernel = laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_bf16_spans
            if (
                args.candidate
                == "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix-nontemporal-key-value"
            ):
                control_kernel = laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_bf16_spans
            if (
                args.candidate
                == "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-prefetch8-dense-prefix-nontemporal-key-value"
            ):
                control_kernel = laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value_bf16_spans
            if (
                args.candidate
                == "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-prefetch16-dense-prefix-nontemporal-key-value"
            ):
                control_kernel = laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch8_dense_prefix_nontemporal_key_value_bf16_spans
            if args.candidate == "split-gqa6-dim64-vstage64-ctx4096":
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim32_vstage64_ctx4096_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-compensated"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim32_vstage64_ctx4096_compensated_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-tokenloop4"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-compensated"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_compensated_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_compensated_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-dense-prefix"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated-dense-prefix"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-dense-prefix-nontemporal-key-value"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_bf16_spans
            elif (
                args.candidate
                == "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated-dense-prefix-nontemporal-key-value"
            ):
                control_kernel = laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_bf16_spans
            candidate_kernel = {
                "fixedshape": laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans,
                "split-gqa6-dim32-vstage64": laguna_global_attention_decode_split_exact_gated_gqa6_dim32_vstage64_bf16_spans,
                "split-gqa6-deferrednorm-dim32-vstage64": laguna_global_attention_decode_split_exact_gated_gqa6_deferrednorm_dim32_vstage64_bf16_spans,
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage64": laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage64_bf16_spans,
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80": laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_bf16_spans,
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix": laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_bf16_spans,
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix-nontemporal": laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_bf16_spans,
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-dense-prefix-nontemporal-key-value": laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value_bf16_spans,
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-prefetch8-dense-prefix-nontemporal-key-value": laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch8_dense_prefix_nontemporal_key_value_bf16_spans,
                "split-gqa6-tokenloop4-deferrednorm-dim32-vstage80-prefetch16-dense-prefix-nontemporal-key-value": laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch16_dense_prefix_nontemporal_key_value_bf16_spans,
                "split-gqa6-dim32-vstage64-t256": laguna_global_attention_decode_split_exact_gated_gqa6_dim32_vstage64_t256_bf16_spans,
                "split-gqa6-tile4-dim32-vstage64": laguna_global_attention_decode_split_exact_gated_gqa6_tile4_dim32_vstage64_bf16_spans,
                "split-gqa6-dim64-vstage64": laguna_global_attention_decode_split_exact_gated_gqa6_dim64_vstage64_bf16_spans,
                "split-gqa6-dim64-vstage32": laguna_global_attention_decode_split_exact_gated_gqa6_dim64_vstage32_bf16_spans,
                "split-qhead-dim32-direct": laguna_global_attention_decode_split_exact_gated_qhead_dim32_direct_bf16_spans,
                "split-gqa6-dim32-vstage64-ctx4096": laguna_global_attention_decode_split_gated_gqa6_dim32_vstage64_ctx4096_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-compensated": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_compensated_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-tokenloop4": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-compensated": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_compensated_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-dense-prefix": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated-dense-prefix": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-dense-prefix-nontemporal-key-value": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_nontemporal_key_value_bf16_spans,
                "split-gqa6-dim64-vstage64-ctx4096-tokenloop4-deferrednorm-compensated-dense-prefix-nontemporal-key-value": laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_nontemporal_key_value_bf16_spans,
                "fused-gqa1": laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans,
                "fused-gqa2-vstage64": laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans,
                "fused-gqa2-vstage64-vec16": laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans,
                "fused-gqa2-vstage64-vec16-direct": laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans,
                "fused-gqa2-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-gqa2-exp32-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed32-exp32-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed32-exp32-producer-max-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed32-exp32-producer-max-dpp-qk-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed32-exp32-producer-max-dpp-qk-probability-vec4-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed32-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed32-local512-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed40-local512-exp32-producer-max-dpp-qk-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed40-local512-exp32-producer-max-dpp-qk-dense-prefix-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed40-local512-exp32-producer-max-dpp-qk-dense-prefix-idle-double-buffer-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "fused-mixed40-local1024-exp32-producer-max-dpp-qk-dense-prefix-idle-double-buffer-probability-vec4-prenorm-vstage64-vec16-direct-assume-exp": laguna_global_attention_decode_fused_exact_gated_mixed40_local1024_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
                "wmma-qk-three-term-mixed32-exact-pv": laguna_global_attention_decode_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_bf16_spans,
                "wmma-gqa6-k64-three-term-raw-numerator": laguna_global_attention_decode_wmma_gqa6_k64_three_term_raw_numerator_bf16_spans,
            }[args.candidate]

            def control() -> None:
                control_kernel(
                    *common,
                    control_context.ptr,
                    gate_device.ptr,
                    control_gated.ptr,
                    *tail,
                    library=library,
                    runtime=runtime,
                )

            def candidate() -> None:
                candidate_kernel(
                    *common,
                    candidate_context.ptr,
                    gate_device.ptr,
                    candidate_gated.ptr,
                    *tail,
                    library=library,
                    runtime=runtime,
                )

            control()
            candidate()
            runtime.device_synchronize()
            control_context_host = _download(runtime, control_context, np.float32)
            candidate_context_host = _download(
                runtime, candidate_context, np.float32
            )
            control_gated_host = _download(runtime, control_gated, np.uint16)
            candidate_gated_host = _download(
                runtime, candidate_gated, np.uint16
            )
            repair_mask_host = np.empty(Q_HEADS * HEAD_DIM, dtype=np.int32)
            if args.candidate == "split-gqa6-dim32-vstage64-ctx4096":
                runtime.memcpy(
                    host_array_ptr(repair_mask_host),
                    physical_scratch.ptr + args.capacity * 4,
                    repair_mask_host.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
            else:
                repair_mask_host.fill(0)
            repair_mask_2d = repair_mask_host.reshape(Q_HEADS, HEAD_DIM) != 0
            repair_tile_mask = repair_mask_2d.reshape(
                KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM // 32, 32
            ).any(axis=(1, 3))
            context_exact = np.array_equal(
                control_context_host, candidate_context_host
            )
            gated_exact = np.array_equal(control_gated_host, candidate_gated_host)
            approximate_candidate = args.candidate in (
                "split-gqa6-dim32-vstage64-ctx4096",
                "wmma-qk-three-term-mixed32-exact-pv",
                "wmma-gqa6-k64-three-term-raw-numerator",
            )
            if (not context_exact or not gated_exact) and not approximate_candidate:
                raise AssertionError(
                    f"{args.candidate} is not byte-exact at {live_count=}"
                )
            context_max_abs = float(
                np.max(
                    np.abs(
                        candidate_context_host.astype(np.float64)
                        - control_context_host.astype(np.float64)
                    )
                )
            )
            gated_mismatches = int(
                np.count_nonzero(candidate_gated_host != control_gated_host)
            )
            mismatch_mask = candidate_gated_host != control_gated_host
            gate_softplus = (
                np.log1p(np.exp(-np.abs(gate), dtype=np.float32), dtype=np.float32)
                + np.maximum(gate, np.float32(0.0))
            ).astype(np.float32)
            candidate_gated_f32 = (
                candidate_context_host.reshape(Q_HEADS, HEAD_DIM)
                * gate_softplus[:, None]
            ).astype(np.float32).reshape(-1)
            control_gated_f32 = (
                control_context_host.reshape(Q_HEADS, HEAD_DIM)
                * gate_softplus[:, None]
            ).astype(np.float32).reshape(-1)
            if gated_mismatches:
                candidate_rounded = bf16_to_float32(candidate_gated_host)
                control_rounded = bf16_to_float32(control_gated_host)
                mismatch_midpoint = (
                    candidate_rounded[mismatch_mask].astype(np.float64)
                    + control_rounded[mismatch_mask].astype(np.float64)
                ) * 0.5
                mismatch_boundary_distance = np.abs(
                    candidate_gated_f32[mismatch_mask].astype(np.float64)
                    - mismatch_midpoint
                )
                mismatch_relative_boundary_distance = (
                    mismatch_boundary_distance
                    / np.maximum(np.abs(mismatch_midpoint), np.finfo(np.float32).tiny)
                )
                mismatch_gated_delta = np.abs(
                    candidate_gated_f32[mismatch_mask].astype(np.float64)
                    - control_gated_f32[mismatch_mask].astype(np.float64)
                )
                rounding_diagnostics = {
                    "max_mismatch_boundary_distance": float(
                        np.max(mismatch_boundary_distance)
                    ),
                    "max_mismatch_relative_boundary_distance": float(
                        np.max(mismatch_relative_boundary_distance)
                    ),
                    "max_mismatch_gated_f32_delta": float(
                        np.max(mismatch_gated_delta)
                    ),
                }
            else:
                rounding_diagnostics = {
                    "max_mismatch_boundary_distance": 0.0,
                    "max_mismatch_relative_boundary_distance": 0.0,
                    "max_mismatch_gated_f32_delta": 0.0,
                }

            for _ in range(args.warmups):
                control()
                candidate()
            runtime.device_synchronize()
            control_ms = []
            candidate_ms = []
            for sample in range(args.samples):
                order = (
                    (("control", control), ("candidate", candidate))
                    if sample % 2 == 0
                    else (("candidate", candidate), ("control", control))
                )
                row = {}
                for name, fn in order:
                    row[name] = _time_ms(runtime, fn, args.burst)
                control_ms.append(row["control"])
                candidate_ms.append(row["candidate"])
            control_median = statistics.median(control_ms)
            candidate_median = statistics.median(candidate_ms)
            results.append(
                {
                    "live_count": live_count,
                    "correctness": {
                        "context_f32_byte_exact": context_exact,
                        "gated_bf16_byte_exact": gated_exact,
                        "context_max_abs": context_max_abs,
                        "gated_bf16_mismatches": gated_mismatches,
                        "repair_outputs": int(np.count_nonzero(repair_mask_host)),
                        "repair_query_heads": int(
                            np.count_nonzero(repair_mask_2d.any(axis=1))
                        ),
                        "repair_gqa_dim32_tiles": int(
                            np.count_nonzero(repair_tile_mask)
                        ),
                        "repair_false_negative_outputs": int(
                            np.count_nonzero(mismatch_mask & ~repair_mask_2d.reshape(-1))
                        ),
                        "rounding_diagnostics": rounding_diagnostics,
                        "context_sha256": hashlib.sha256(
                            candidate_context_host.tobytes()
                        ).hexdigest(),
                        "gated_sha256": hashlib.sha256(
                            candidate_gated_host.tobytes()
                        ).hexdigest(),
                    },
                    "control": {
                        "samples_ms": control_ms,
                        "median_ms": control_median,
                    },
                    "candidate": {
                        "samples_ms": candidate_ms,
                        "median_ms": candidate_median,
                        "latency_change_percent": (
                            candidate_median / control_median - 1.0
                        )
                        * 100.0,
                    },
                }
            )
        return {
            "schema": 1,
            "kind": "hipengine_laguna_global_fixedshape_reduce_leaf",
            "candidate_kind": args.candidate,
            "status": "directional_candidate",
            "performance_claim": False,
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "tracked_changes": tracked,
            "hardware": {
                "device": "AMD Radeon 8060S Graphics",
                "arch": os.environ.get("HIPENGINE_HIP_ARCH"),
            },
            "shape": {
                "capacity": capacity,
                "query_heads": Q_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "live_counts": live_counts,
            },
            "protocol": {
                "samples": args.samples,
                "warmups": args.warmups,
                "burst": args.burst,
            },
            "rows": results,
        }
    finally:
        cache.free()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
