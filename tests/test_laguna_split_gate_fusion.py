from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION


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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_split_exact_gated_reducers_match_unfused_chain() -> None:
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
        laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans,
        laguna_global_attention_decode_split_exact_bf16_spans,
        laguna_global_attention_decode_split_exact_gated_bf16_spans,
        laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans,
    )
    from hipengine.kernels.hip_gfx1100.fused.laguna_attention import (
        build_laguna_attention,
        laguna_softplus_head_gate_f32_bf16_out,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    kv_library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    gate_library = build_laguna_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=2,
        layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(48, 72),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    rng = np.random.default_rng(404)
    max_heads = 72
    width = max_heads * 128
    allocations = []
    try:
        key_device = malloc(8 * 128 * 4, runtime=runtime)
        value_device = malloc(8 * 128 * 4, runtime=runtime)
        query_device = malloc(width * 4, runtime=runtime)
        gate_device = malloc(max_heads * 4, runtime=runtime)
        unfused_context_device = malloc(width * 4, runtime=runtime)
        fused_context_device = malloc(width * 4, runtime=runtime)
        fixedshape_context_device = malloc(width * 4, runtime=runtime)
        fused_gqa1_context_device = malloc(width * 4, runtime=runtime)
        unfused_gated_device = malloc(width * 2, runtime=runtime)
        fused_gated_device = malloc(width * 2, runtime=runtime)
        fixedshape_gated_device = malloc(width * 2, runtime=runtime)
        fused_gqa1_gated_device = malloc(width * 2, runtime=runtime)
        score_scratch_device = malloc(max_heads * 4096 * 4, runtime=runtime)
        physical_scratch_device = malloc(max_heads * 4096 * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                gate_device,
                unfused_context_device,
                fused_context_device,
                fixedshape_context_device,
                fused_gqa1_context_device,
                unfused_gated_device,
                fused_gated_device,
                fixedshape_gated_device,
                fused_gqa1_gated_device,
                score_scratch_device,
                physical_scratch_device,
            )
        )

        for position in range(257):
            key = rng.normal(0.0, 0.12, size=(8, 128)).astype(np.float32)
            value = rng.normal(0.0, 0.12, size=(8, 128)).astype(np.float32)
            copy_host_to_device(key_device, host_array_ptr(key), runtime=runtime)
            copy_host_to_device(value_device, host_array_ptr(value), runtime=runtime)
            cache.prepare_position(position)
            cache.append(
                0,
                key_device.ptr,
                value_device.ptr,
                library=kv_library,
            )
            cache.append(
                1,
                key_device.ptr,
                value_device.ptr,
                library=kv_library,
            )

        for layer_id, heads in ((0, 48), (1, 72)):
            query = rng.normal(0.0, 0.12, size=(heads, 128)).astype(np.float32)
            gate = rng.normal(0.0, 0.5, size=(heads,)).astype(np.float32)
            copy_host_to_device(
                query_device,
                host_array_ptr(query),
                nbytes=query.nbytes,
                runtime=runtime,
            )
            copy_host_to_device(
                gate_device,
                host_array_ptr(gate),
                nbytes=gate.nbytes,
                runtime=runtime,
            )
            state = cache.layer(layer_id)
            common = (
                query_device.ptr,
                state.key_cache.ptr,
                state.value_cache.ptr,
            )
            split_tail = (
                score_scratch_device.ptr,
                physical_scratch_device.ptr,
                state.spans,
                257,
            )
            if layer_id == 0:
                laguna_global_attention_decode_split_exact_bf16_spans(
                    *common,
                    unfused_context_device.ptr,
                    *split_tail,
                    state.capacity,
                    heads,
                    8,
                    128,
                    128**-0.5,
                    library=kv_library,
                    runtime=runtime,
                )
                laguna_global_attention_decode_split_exact_gated_bf16_spans(
                    *common,
                    fused_context_device.ptr,
                    gate_device.ptr,
                    fused_gated_device.ptr,
                    *split_tail,
                    state.capacity,
                    heads,
                    8,
                    128,
                    128**-0.5,
                    library=kv_library,
                    runtime=runtime,
                )
                laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans(
                    *common,
                    fixedshape_context_device.ptr,
                    gate_device.ptr,
                    fixedshape_gated_device.ptr,
                    *split_tail,
                    state.capacity,
                    heads,
                    8,
                    128,
                    128**-0.5,
                    library=kv_library,
                    runtime=runtime,
                )
                laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans(
                    *common,
                    fused_gqa1_context_device.ptr,
                    gate_device.ptr,
                    fused_gqa1_gated_device.ptr,
                    *split_tail,
                    state.capacity,
                    heads,
                    8,
                    128,
                    128**-0.5,
                    library=kv_library,
                    runtime=runtime,
                )
            else:
                laguna_swa_attention_decode_split_tile16_exact_bf16_spans(
                    *common,
                    unfused_context_device.ptr,
                    *split_tail,
                    heads,
                    8,
                    128,
                    128**-0.5,
                    sliding_window=512,
                    library=kv_library,
                    runtime=runtime,
                )
                laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans(
                    *common,
                    fused_context_device.ptr,
                    gate_device.ptr,
                    fused_gated_device.ptr,
                    *split_tail,
                    heads,
                    8,
                    128,
                    128**-0.5,
                    sliding_window=512,
                    library=kv_library,
                    runtime=runtime,
                )
            laguna_softplus_head_gate_f32_bf16_out(
                unfused_context_device.ptr,
                gate_device.ptr,
                unfused_gated_device.ptr,
                1,
                heads,
                128,
                library=gate_library,
                runtime=runtime,
            )
            runtime.device_synchronize()

            context_elements = heads * 128
            unfused_context = np.empty(context_elements, dtype=np.float32)
            fused_context = np.empty_like(unfused_context)
            fixedshape_context = np.empty_like(unfused_context)
            fused_gqa1_context = np.empty_like(unfused_context)
            unfused_gated = np.empty(context_elements, dtype=np.uint16)
            fused_gated = np.empty_like(unfused_gated)
            fixedshape_gated = np.empty_like(unfused_gated)
            fused_gqa1_gated = np.empty_like(unfused_gated)
            copy_device_to_host(
                host_array_ptr(unfused_context),
                unfused_context_device,
                nbytes=unfused_context.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(fused_context),
                fused_context_device,
                nbytes=fused_context.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(unfused_gated),
                unfused_gated_device,
                nbytes=unfused_gated.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(fused_gated),
                fused_gated_device,
                nbytes=fused_gated.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(fused_context, unfused_context)
            np.testing.assert_array_equal(fused_gated, unfused_gated)
            if layer_id == 0:
                copy_device_to_host(
                    host_array_ptr(fixedshape_context),
                    fixedshape_context_device,
                    nbytes=fixedshape_context.nbytes,
                    runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(fused_gqa1_context),
                    fused_gqa1_context_device,
                    nbytes=fused_gqa1_context.nbytes,
                    runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(fused_gqa1_gated),
                    fused_gqa1_gated_device,
                    nbytes=fused_gqa1_gated.nbytes,
                    runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(fixedshape_gated),
                    fixedshape_gated_device,
                    nbytes=fixedshape_gated.nbytes,
                    runtime=runtime,
                )
                np.testing.assert_array_equal(
                    fixedshape_context,
                    fused_context,
                )
                np.testing.assert_array_equal(
                    fixedshape_gated,
                    fused_gated,
                )
                np.testing.assert_array_equal(
                    fused_gqa1_context,
                    fused_context,
                )
                np.testing.assert_array_equal(
                    fused_gqa1_gated,
                    fused_gated,
                )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
