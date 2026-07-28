"""Exact grouped-GQA score-producer gates for Laguna SWA decode."""

from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.laguna_gguf import SLIDING_ATTENTION


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_laguna_swa_split_gqa3_score_variants_are_registered_on_gfx1151() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_laguna_kv_attention_kernels()
    load_backend_kernel_package("hip_gfx1151")
    variants = (
        "swa_context_split_exact_gated_gqa3_scores_spans",
        "swa_context_split_tile16_exact_gated_gqa3_scores_spans",
        "swa_context_split_tile16_exact_gated_gqa3_scores_fixed512_spans",
        "swa_context_fused_exact_gated_gqa2_fixed512_spans",
    )
    assert all(
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_decode",
            quant="bf16",
            variant=variant,
            missing="none",
        )
        is not None
        for variant in variants
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_swa_split_gqa3_scores_match_exact_wave_local_at_ring_boundaries() -> None:
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
        laguna_swa_attention_decode_split_exact_gated_gqa3_scores_bf16_spans,
        laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    heads = 72
    head_dim = 128
    kv_heads = 8
    capacity = 512
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(heads,),
        head_count_kv=kv_heads,
        key_length=head_dim,
        value_length=head_dim,
        sliding_window=capacity,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=capacity + 1,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(250728)
    width = heads * head_dim
    allocations = []
    try:
        key_device = malloc(kv_heads * head_dim * 4, runtime=runtime)
        value_device = malloc(kv_heads * head_dim * 4, runtime=runtime)
        query_device = malloc(width * 4, runtime=runtime)
        gate_device = malloc(heads * 4, runtime=runtime)
        retained_context_device = malloc(width * 4, runtime=runtime)
        candidate_context_device = malloc(width * 4, runtime=runtime)
        fixed512_context_device = malloc(width * 4, runtime=runtime)
        fused_gqa2_context_device = malloc(width * 4, runtime=runtime)
        retained_gated_device = malloc(width * 2, runtime=runtime)
        candidate_gated_device = malloc(width * 2, runtime=runtime)
        fixed512_gated_device = malloc(width * 2, runtime=runtime)
        fused_gqa2_gated_device = malloc(width * 2, runtime=runtime)
        score_scratch_device = malloc(heads * capacity * 4, runtime=runtime)
        physical_scratch_device = malloc(heads * capacity * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                gate_device,
                retained_context_device,
                candidate_context_device,
                fixed512_context_device,
                fused_gqa2_context_device,
                retained_gated_device,
                candidate_gated_device,
                fixed512_gated_device,
                fused_gqa2_gated_device,
                score_scratch_device,
                physical_scratch_device,
            )
        )

        query = rng.normal(0.0, 0.12, size=(heads, head_dim)).astype(np.float32)
        query[0] = 0.0
        query[1] *= 1.0e4
        gate = np.resize(
            np.array([-40.0, -1.0, -0.0, 0.0, 1.0, 40.0], dtype=np.float32),
            heads,
        )
        copy_host_to_device(query_device, host_array_ptr(query), runtime=runtime)
        copy_host_to_device(gate_device, host_array_ptr(gate), runtime=runtime)
        state = cache.layer(0)

        boundaries = {0, 64, 126, 127, 255, 256, 510, 511, 512}
        for position in range(capacity + 1):
            key = rng.normal(0.0, 0.12, size=(kv_heads, head_dim)).astype(np.float32)
            value = rng.normal(0.0, 0.12, size=(kv_heads, head_dim)).astype(np.float32)
            copy_host_to_device(key_device, host_array_ptr(key), runtime=runtime)
            copy_host_to_device(value_device, host_array_ptr(value), runtime=runtime)
            cache.prepare_position(position)
            cache.append(0, key_device.ptr, value_device.ptr, library=library)
            if position == capacity:
                cache.evict_swa_position(0, 100)
            if position not in boundaries:
                continue

            retained = (
                laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans
            )
            candidate = (
                laguna_swa_attention_decode_split_exact_gated_gqa3_scores_bf16_spans
            )
            if position >= 256:
                retained = (
                    laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans
                )
                candidate = (
                    laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans
                )
            common = (
                query_device.ptr,
                state.key_cache.ptr,
                state.value_cache.ptr,
            )
            tail = (
                score_scratch_device.ptr,
                physical_scratch_device.ptr,
                state.spans,
                min(position + 1, capacity),
                heads,
                kv_heads,
                head_dim,
                head_dim**-0.5,
            )
            retained(
                *common,
                retained_context_device.ptr,
                gate_device.ptr,
                retained_gated_device.ptr,
                *tail,
                sliding_window=capacity,
                library=library,
                runtime=runtime,
            )
            candidate(
                *common,
                candidate_context_device.ptr,
                gate_device.ptr,
                candidate_gated_device.ptr,
                *tail,
                sliding_window=capacity,
                library=library,
                runtime=runtime,
            )
            if position >= 511:
                laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans(
                    *common,
                    fixed512_context_device.ptr,
                    gate_device.ptr,
                    fixed512_gated_device.ptr,
                    *tail,
                    sliding_window=capacity,
                    library=library,
                    runtime=runtime,
                )
                laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans(
                    *common,
                    fused_gqa2_context_device.ptr,
                    gate_device.ptr,
                    fused_gqa2_gated_device.ptr,
                    *tail,
                    sliding_window=capacity,
                    library=library,
                    runtime=runtime,
                )
            runtime.device_synchronize()

            retained_context = np.empty(width, dtype=np.float32)
            candidate_context = np.empty_like(retained_context)
            fixed512_context = np.empty_like(retained_context)
            fused_gqa2_context = np.empty_like(retained_context)
            retained_gated = np.empty(width, dtype=np.uint16)
            candidate_gated = np.empty_like(retained_gated)
            fixed512_gated = np.empty_like(retained_gated)
            fused_gqa2_gated = np.empty_like(retained_gated)
            copy_device_to_host(
                host_array_ptr(retained_context),
                retained_context_device,
                nbytes=retained_context.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(candidate_context),
                candidate_context_device,
                nbytes=candidate_context.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(retained_gated),
                retained_gated_device,
                nbytes=retained_gated.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(candidate_gated),
                candidate_gated_device,
                nbytes=candidate_gated.nbytes,
                runtime=runtime,
            )
            if position >= 511:
                copy_device_to_host(
                    host_array_ptr(fixed512_context),
                    fixed512_context_device,
                    nbytes=fixed512_context.nbytes,
                    runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(fixed512_gated),
                    fixed512_gated_device,
                    nbytes=fixed512_gated.nbytes,
                    runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(fused_gqa2_context),
                    fused_gqa2_context_device,
                    nbytes=fused_gqa2_context.nbytes,
                    runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(fused_gqa2_gated),
                    fused_gqa2_gated_device,
                    nbytes=fused_gqa2_gated.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_array_equal(
                candidate_context,
                retained_context,
                err_msg=f"context mismatch at {position=}",
            )
            np.testing.assert_array_equal(
                candidate_gated,
                retained_gated,
                err_msg=f"gated mismatch at {position=}",
            )
            if position >= 511:
                np.testing.assert_array_equal(
                    fixed512_context,
                    retained_context,
                    err_msg=f"fixed512 context mismatch at {position=}",
                )
                np.testing.assert_array_equal(
                    fixed512_gated,
                    retained_gated,
                    err_msg=f"fixed512 gated mismatch at {position=}",
                )
                np.testing.assert_array_equal(
                    fused_gqa2_context,
                    retained_context,
                    err_msg=f"fused GQA2 context mismatch at {position=}",
                )
                np.testing.assert_array_equal(
                    fused_gqa2_gated,
                    retained_gated,
                    err_msg=f"fused GQA2 gated mismatch at {position=}",
                )
    finally:
        cache.free()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
