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


def test_laguna_swa_split_wave_local_variants_are_registered() -> None:
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans,
        laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans,
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_laguna_kv_attention_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_split_exact_gated_wave_local_spans",
        )
        is laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_split_tile16_exact_gated_wave_local_spans",
        )
        is laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_split_exact_gated_wave_local_dim2_spans",
        )
        is laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_split_tile16_exact_gated_wave_local_dim2_spans",
        )
        is laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans
    )


def test_laguna_split_attention_bundle_is_screenable_on_gfx1151() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import resolve

    load_backend_kernel_package("hip_gfx1151")
    variants = (
        "global_context_split_exact_spans",
        "global_context_split_exact_gated_spans",
        "swa_context_split_exact_spans",
        "swa_context_split_exact_gated_spans",
        "swa_context_split_exact_gated_wave_local_spans",
        "swa_context_split_exact_gated_wave_local_dim2_spans",
        "swa_context_split_tile16_exact_spans",
        "swa_context_split_tile16_exact_gated_spans",
        "swa_context_split_tile16_exact_gated_wave_local_spans",
        "swa_context_split_tile16_exact_gated_wave_local_dim2_spans",
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
def test_laguna_swa_split_wave_local_matches_retained_reducer() -> None:
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
        laguna_swa_attention_decode_split_exact_gated_bf16_spans,
        laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans,
        laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans,
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
        backend="hip_gfx1100",
        runtime=runtime,
    )
    rng = np.random.default_rng(250725)
    width = heads * head_dim
    allocations = []
    try:
        key_device = malloc(kv_heads * head_dim * 4, runtime=runtime)
        value_device = malloc(kv_heads * head_dim * 4, runtime=runtime)
        query_device = malloc(width * 4, runtime=runtime)
        gate_device = malloc(heads * 4, runtime=runtime)
        retained_context_device = malloc(width * 4, runtime=runtime)
        wave_local_context_device = malloc(width * 4, runtime=runtime)
        candidate_context_device = malloc(width * 4, runtime=runtime)
        retained_gated_device = malloc(width * 2, runtime=runtime)
        wave_local_gated_device = malloc(width * 2, runtime=runtime)
        candidate_gated_device = malloc(width * 2, runtime=runtime)
        score_scratch_device = malloc(heads * capacity * 4, runtime=runtime)
        physical_scratch_device = malloc(heads * capacity * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                gate_device,
                retained_context_device,
                wave_local_context_device,
                candidate_context_device,
                retained_gated_device,
                wave_local_gated_device,
                candidate_gated_device,
                score_scratch_device,
                physical_scratch_device,
            )
        )

        query = rng.normal(0.0, 0.12, size=(heads, head_dim)).astype(np.float32)
        query[0] = 0.0  # tied scores
        query[1] *= 1.0e4  # finite extremes and softmax underflow
        gate = np.resize(
            np.array([-40.0, -1.0, -0.0, 0.0, 1.0, 40.0], dtype=np.float32),
            heads,
        )
        copy_host_to_device(query_device, host_array_ptr(query), runtime=runtime)
        copy_host_to_device(gate_device, host_array_ptr(gate), runtime=runtime)
        state = cache.layer(0)

        empty_common = (
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
        )
        empty_tail = (
            score_scratch_device.ptr,
            physical_scratch_device.ptr,
            state.spans,
            1,
            heads,
            kv_heads,
            head_dim,
            head_dim**-0.5,
        )
        laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans(
            *empty_common,
            wave_local_context_device.ptr,
            gate_device.ptr,
            wave_local_gated_device.ptr,
            *empty_tail,
            sliding_window=capacity,
            library=library,
            runtime=runtime,
        )
        laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans(
            *empty_common,
            candidate_context_device.ptr,
            gate_device.ptr,
            candidate_gated_device.ptr,
            *empty_tail,
            sliding_window=capacity,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        empty_context = np.empty(width, dtype=np.float32)
        empty_candidate_context = np.empty_like(empty_context)
        empty_gated = np.empty(width, dtype=np.uint16)
        empty_candidate_gated = np.empty_like(empty_gated)
        copy_device_to_host(
            host_array_ptr(empty_context),
            wave_local_context_device,
            nbytes=empty_context.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(empty_candidate_context),
            candidate_context_device,
            nbytes=empty_candidate_context.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(empty_gated),
            wave_local_gated_device,
            nbytes=empty_gated.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(empty_candidate_gated),
            candidate_gated_device,
            nbytes=empty_candidate_gated.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(empty_candidate_context, empty_context)
        np.testing.assert_array_equal(empty_candidate_gated, empty_gated)
        np.testing.assert_array_equal(empty_context, np.zeros_like(empty_context))
        np.testing.assert_array_equal(empty_gated, np.zeros_like(empty_gated))

        boundary_positions = {64, 69, 126, 127, 255, 256, 510, 511, 512}
        for position in range(513):
            key = rng.normal(0.0, 0.12, size=(kv_heads, head_dim)).astype(np.float32)
            value = rng.normal(0.0, 0.12, size=(kv_heads, head_dim)).astype(np.float32)
            copy_host_to_device(key_device, host_array_ptr(key), runtime=runtime)
            copy_host_to_device(value_device, host_array_ptr(value), runtime=runtime)
            cache.prepare_position(position)
            cache.append(0, key_device.ptr, value_device.ptr, library=library)
            if position == 512:
                cache.evict_swa_position(0, 100)
            if position not in boundary_positions:
                continue

            retained = laguna_swa_attention_decode_split_exact_gated_bf16_spans
            wave_local = laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans
            candidate = (
                laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans
            )
            if position >= 256:
                retained = laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans
                wave_local = (
                    laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans
                )
                candidate = (
                    laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans
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
            wave_local(
                *common,
                wave_local_context_device.ptr,
                gate_device.ptr,
                wave_local_gated_device.ptr,
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
            runtime.device_synchronize()

            retained_context = np.empty(width, dtype=np.float32)
            wave_local_context = np.empty_like(retained_context)
            candidate_context = np.empty_like(retained_context)
            retained_gated = np.empty(width, dtype=np.uint16)
            wave_local_gated = np.empty_like(retained_gated)
            candidate_gated = np.empty_like(retained_gated)
            copy_device_to_host(
                host_array_ptr(retained_context),
                retained_context_device,
                nbytes=retained_context.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(wave_local_context),
                wave_local_context_device,
                nbytes=wave_local_context.nbytes,
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
                host_array_ptr(wave_local_gated),
                wave_local_gated_device,
                nbytes=wave_local_gated.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(candidate_gated),
                candidate_gated_device,
                nbytes=candidate_gated.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(wave_local_context, retained_context)
            np.testing.assert_array_equal(wave_local_gated, retained_gated)
            np.testing.assert_array_equal(candidate_context, retained_context)
            np.testing.assert_array_equal(candidate_gated, retained_gated)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
