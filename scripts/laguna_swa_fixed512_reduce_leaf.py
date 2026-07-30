#!/usr/bin/env python3
"""Gate the exact saturated-512 Laguna SWA reducer specialization."""

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

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
    build_laguna_kv_attention,
    laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_producer_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_denom_prefetch4_idle_vec4_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_gated_only_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
    laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans,
    laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans,
)
from hipengine.loading.laguna_gguf import SLIDING_ATTENTION
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache


CAPACITY = 512
Q_HEADS = 72
KV_HEADS = 8
HEAD_DIM = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--burst", type=int, default=50)
    parser.add_argument(
        "--candidate",
        choices=(
            "fixed512",
            "fused-gqa2",
            "fused-gqa3-local384",
            "fused-gqa3-vstage64",
            "fused-gqa3-vstage64-vec16",
            "fused-gqa3-vstage64-vec16-direct",
            "fused-gqa3-vstage64-vec16-direct-assume-exp",
            "mixed32-vstage64-vec16-direct-assume-exp",
            "mixed32-exp4-vstage64-vec16-direct-assume-exp",
            "mixed32-exp8-vstage64-vec16-direct-assume-exp",
            "mixed32-exp16-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-stage-pcache-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-stage-pcache-idle-producer-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-stage-pcache-vec4-denom-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-stage-pcache-vec4-denom-probability-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-stage-pcache-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp",
            "mixed40-exp32-producer-max-gate-stage-pcache-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp",
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp",
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp",
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp",
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp",
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-dual-tail-producer-value-tail-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp",
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-output-sharded-probability-allwave-value-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp",
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-output-sharded-probability-dpp-qk-allwave-value-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp",
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-denom-prefetch4-idle-vec4-probability-vstage64-vec16-direct-assume-exp",
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-stage-pcache-dpp-qk-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-dpp-qk-vstage64-vec16-direct-assume-exp",
            "mixed32-exp32-producer-max-gate-gated-only-vstage64-vec16-direct-assume-exp",
        ),
        default="fixed512",
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
    gated_only = (
        args.candidate
        == "mixed32-exp32-producer-max-gate-gated-only-vstage64-vec16-direct-assume-exp"
    )

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=args.require_cached_build,
    )
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(Q_HEADS,),
        head_count_kv=KV_HEADS,
        key_length=HEAD_DIM,
        value_length=HEAD_DIM,
        sliding_window=CAPACITY,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=CAPACITY + 1,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(20260728)
    keys = rng.normal(
        0.0, 0.12, size=(CAPACITY + 1, KV_HEADS, HEAD_DIM)
    ).astype(np.float32)
    values = rng.normal(
        0.0, 0.12, size=(CAPACITY + 1, KV_HEADS, HEAD_DIM)
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
        score_scratch = malloc(Q_HEADS * CAPACITY * 4, runtime=runtime)
        physical_scratch = malloc(Q_HEADS * CAPACITY * 4, runtime=runtime)
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
        cache.prepare_rows(tuple(range(CAPACITY)))
        cache.append_rows(
            0,
            key_device.ptr,
            value_device.ptr,
            CAPACITY,
            library=library,
        )
        cache.commit_rows()
        cache.prepare_position(CAPACITY)
        row_nbytes = KV_HEADS * HEAD_DIM * np.dtype(np.float32).itemsize
        cache.append(
            0,
            key_device.ptr + CAPACITY * row_nbytes,
            value_device.ptr + CAPACITY * row_nbytes,
            library=library,
        )
        state = cache.layer(0)
        common = (
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
        )
        tail = (
            score_scratch.ptr,
            physical_scratch.ptr,
            state.spans,
            CAPACITY,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            HEAD_DIM**-0.5,
        )

        control_kernel = {
            "fixed512": laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans,
            "fused-gqa2": laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans,
            "fused-gqa3-local384": laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans,
            "fused-gqa3-vstage64": laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans,
            "fused-gqa3-vstage64-vec16": laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans,
            "fused-gqa3-vstage64-vec16-direct": laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans,
            "fused-gqa3-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans,
            "mixed32-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp4-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp8-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp16-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-idle-producer-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-vec4-denom-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-exp32-producer-max-gate-stage-pcache-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-dual-tail-producer-value-tail-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-output-sharded-probability-allwave-value-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-output-sharded-probability-dpp-qk-allwave-value-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-denom-prefetch4-idle-vec4-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-dpp-qk-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-dpp-qk-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-gated-only-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        }[args.candidate]
        candidate_kernel = {
            "fixed512": laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans,
            "fused-gqa2": laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans,
            "fused-gqa3-local384": laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans,
            "fused-gqa3-vstage64": laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans,
            "fused-gqa3-vstage64-vec16": laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans,
            "fused-gqa3-vstage64-vec16-direct": laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans,
            "fused-gqa3-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp4-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp8-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp16-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-idle-producer-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_producer_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-vec4-denom-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-exp32-producer-max-gate-stage-pcache-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-dual-tail-producer-value-tail-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-output-sharded-probability-allwave-value-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-local512-exp32-producer-max-gate-stage-pcache-output-sharded-probability-dpp-qk-allwave-value-idle-vec4-denom-probability-vstage128-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-denom-prefetch4-idle-vec4-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_denom_prefetch4_idle_vec4_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed40-exp32-producer-max-gate-stage-pcache-tail-producer-value-tail-idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-stage-pcache-dpp-qk-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-dpp-qk-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
            "mixed32-exp32-producer-max-gate-gated-only-vstage64-vec16-direct-assume-exp": laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_gated_only_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        }[args.candidate]
        context_sentinel = np.full(
            Q_HEADS * HEAD_DIM,
            np.float32(-123.5),
            dtype=np.float32,
        )
        if gated_only:
            copy_host_to_device(
                candidate_context,
                host_array_ptr(context_sentinel),
                context_sentinel.nbytes,
                runtime=runtime,
            )

        def control() -> None:
            control_kernel(
                *common,
                control_context.ptr,
                gate_device.ptr,
                control_gated.ptr,
                *tail,
                sliding_window=CAPACITY,
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
                sliding_window=CAPACITY,
                library=library,
                runtime=runtime,
            )

        control()
        candidate()
        runtime.device_synchronize()
        control_context_host = _download(runtime, control_context, np.float32)
        candidate_context_host = _download(runtime, candidate_context, np.float32)
        control_gated_host = _download(runtime, control_gated, np.uint16)
        candidate_gated_host = _download(runtime, candidate_gated, np.uint16)
        context_exact = np.array_equal(control_context_host, candidate_context_host)
        gated_exact = np.array_equal(control_gated_host, candidate_gated_host)
        context_store_omitted = bool(
            gated_only
            and np.array_equal(candidate_context_host, context_sentinel)
        )
        if (
            args.candidate != "fused-gqa3-vstage64-vec16-direct-fast-exp"
            and (
                not gated_exact
                or (gated_only and not context_store_omitted)
                or (not gated_only and not context_exact)
            )
        ):
            raise AssertionError(f"{args.candidate} reducer is not byte-exact")

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
        result = {
            "schema": 1,
            "kind": "hipengine_laguna_swa_fixed512_reduce_leaf",
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
                "capacity": CAPACITY,
                "sliding_window": CAPACITY,
                "scan_slots": CAPACITY,
                "query_heads": Q_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
            },
            "protocol": {
                "samples": args.samples,
                "warmups": args.warmups,
                "burst": args.burst,
            },
            "correctness": {
                "context_f32_byte_exact": (
                    None if gated_only else context_exact
                ),
                "context_f32_store_omitted": context_store_omitted,
                "gated_bf16_byte_exact": gated_exact,
                "gated_bf16_mismatches": int(
                    np.count_nonzero(
                        candidate_gated_host != control_gated_host
                    )
                ),
                "context_max_abs_error": (
                    None
                    if gated_only
                    else float(
                        np.max(
                            np.abs(
                                candidate_context_host - control_context_host
                            )
                        )
                    )
                ),
                "gated_max_abs_error": float(
                    np.max(
                        np.abs(
                            bf16_to_float32(candidate_gated_host)
                            - bf16_to_float32(control_gated_host)
                        )
                    )
                ),
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
        return result
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
