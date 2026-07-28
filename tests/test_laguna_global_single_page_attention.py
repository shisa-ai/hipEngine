from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path("hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip")
_VARIANT = "global_context_single_page_spans"


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


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _fake_global_spans(capacity: int = 512) -> KVLiveSpans:
    pages = (capacity + 255) // 256
    return KVLiveSpans.paged_dense(
        block_table=_tensor(0x1000, (pages,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        token_positions=_tensor(0x3000, (capacity,), "int64"),
        evict_mask=_tensor(0x4000, (capacity,), "bool"),
        row_positions=_tensor(0x5000, (1,), "int64"),
        capacity=capacity,
        block_size=256,
        storage_dtype="bf16",
    )


def _function_block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {marker}")


def _ordered_attention_reference(
    query: np.ndarray,
    key_bits: np.ndarray,
    value_bits: np.ndarray,
    *,
    token_positions: np.ndarray,
    evict_mask: np.ndarray,
    query_position: int,
    num_kv_heads: int,
) -> np.ndarray:
    from hipengine.quant.gguf import bf16_to_float32

    keys = bf16_to_float32(key_bits)
    values = bf16_to_float32(value_bits)
    q_heads, head_dim = query.shape
    group = q_heads // num_kv_heads
    result = np.empty_like(query)
    scale = np.float32(head_dim**-0.5)
    visible = np.flatnonzero(
        (~evict_mask)
        & (token_positions >= 0)
        & (token_positions <= query_position)
    )
    for q_head in range(q_heads):
        kv_head = q_head // group
        scores = np.empty(visible.size, dtype=np.float32)
        for index, token in enumerate(visible):
            acc = np.float32(0.0)
            for dim in range(head_dim):
                acc = np.float32(
                    acc + np.float32(query[q_head, dim] * keys[token, kv_head, dim])
                )
            scores[index] = np.float32(acc * scale)
        max_score = np.max(scores)
        weights = np.exp(scores - max_score, dtype=np.float32)
        inv_denom = np.float32(1.0) / np.sum(weights, dtype=np.float32)
        for dim in range(head_dim):
            acc = np.float32(0.0)
            for index, token in enumerate(visible):
                weight = np.float32(weights[index] * inv_denom)
                if weight > np.float32(0.0):
                    acc = np.float32(
                        acc + np.float32(weight * values[token, kv_head, dim])
                    )
            result[q_head, dim] = acc
    return result


def _quality(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    left = expected.astype(np.float64)
    right = actual.astype(np.float64)
    left -= np.max(left, axis=1, keepdims=True)
    right -= np.max(right, axis=1, keepdims=True)
    left_prob = np.exp(left)
    right_prob = np.exp(right)
    left_prob /= np.sum(left_prob, axis=1, keepdims=True)
    right_prob /= np.sum(right_prob, axis=1, keepdims=True)
    kl = np.sum(
        left_prob
        * (
            np.log(np.maximum(left_prob, 1.0e-300))
            - np.log(np.maximum(right_prob, 1.0e-300))
        ),
        axis=1,
    )
    agreement = np.mean(np.argmax(left, axis=1) == np.argmax(right, axis=1))
    return float(np.max(kl)), float(agreement)


def test_single_page_source_is_an_exact_address_only_sibling() -> None:
    source = _SOURCE.read_text(encoding="utf-8")
    baseline = _function_block(
        source,
        "__global__ void laguna_global_attention_decode_bf16_kernel(",
    )
    candidate = _function_block(
        source,
        "__global__ void laguna_global_attention_decode_single_page_bf16_kernel(",
    )
    expected = baseline.replace(
        "laguna_global_attention_decode_bf16_kernel",
        "laguna_global_attention_decode_single_page_bf16_kernel",
        1,
    ).replace(
        "laguna_global_physical_slot(base_offsets, token, block_size)",
        "static_cast<int64_t>(base_offsets[0]) * block_size + token",
    ).replace(
        "  if (context_len <= 0 || query_position < 0) {",
        "  if (context_len <= 0 || context_len > block_size || query_position < 0) {",
        1,
    )
    assert candidate == expected
    assert baseline.count(
        "laguna_global_physical_slot(base_offsets, token, block_size)"
    ) == 2
    assert candidate.count(
        "static_cast<int64_t>(base_offsets[0]) * block_size + token"
    ) == 2
    assert source.count(
        'extern "C" int hipengine_laguna_global_attention_decode_single_page_bf16_spans('
    ) == 1


def test_single_page_wrapper_validates_before_build(monkeypatch) -> None:
    import hipengine.kernels.hip_gfx1100.attention.laguna_kv as module

    def fail_build(**_kwargs):
        raise AssertionError("build reached")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    launch = module.laguna_global_attention_decode_single_page_bf16_spans
    spans = _fake_global_spans()
    valid = (0x6000, 0x7000, 0x8000, 0x9000)

    for index in range(4):
        pointers = list(valid)
        pointers[index] = 0
        with pytest.raises(ValueError, match="non-zero"):
            launch(
                *pointers,
                spans,
                512,
                48,
                8,
                128,
                128**-0.5,
            )
    with pytest.raises(ValueError, match="max_context_len"):
        launch(*valid, spans, 256, 48, 8, 128, 128**-0.5)
    with pytest.raises(ValueError, match="num_q_heads"):
        launch(*valid, spans, 512, 0, 8, 128, 128**-0.5)
    with pytest.raises(ValueError, match="head_dim"):
        launch(*valid, spans, 512, 48, 8, 64, 64**-0.5)


def test_single_page_package_registry_backend_scope() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100 import attention
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_global_attention_decode_single_page_bf16_spans,
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    assert (
        attention.laguna_global_attention_decode_single_page_bf16_spans
        is laguna_global_attention_decode_single_page_bf16_spans
    )
    assert "laguna_global_attention_decode_single_page_bf16_spans" in attention.__all__
    register_laguna_kv_attention_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant=_VARIANT,
        )
        is laguna_global_attention_decode_single_page_bf16_spans
    )

    load_backend_kernel_package("hip_gfx1151")
    for backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert not is_registered(
            KernelKey(backend, "laguna_attention_decode", "bf16", _VARIANT)
        )



@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_single_page_attention_matches_scalar_and_cpu_with_complete_spans() -> None:
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
        laguna_global_attention_decode_bf16_spans,
        laguna_global_attention_decode_single_page_bf16_spans,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    capacity = 512
    block_size = 256
    num_q_heads = 48
    num_kv_heads = 8
    head_dim = 128
    storage_slots = 3 * block_size
    rng = np.random.default_rng(1729)

    query = rng.normal(0.0, 0.12, size=(num_q_heads, head_dim)).astype(np.float32)
    edge_bits = np.array(
        [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x3F00, 0xBF00],
        dtype=np.uint16,
    )
    query[:, : edge_bits.size] = bf16_to_float32(edge_bits)[None, :]
    logical_keys = float_array_to_bf16_bits(
        rng.normal(0.0, 0.12, size=(capacity, num_kv_heads, head_dim)).astype(np.float32)
    )
    logical_values = float_array_to_bf16_bits(
        rng.normal(0.0, 0.12, size=(capacity, num_kv_heads, head_dim)).astype(np.float32)
    )
    logical_keys[0, 0, : edge_bits.size] = edge_bits
    logical_values[0, 0, : edge_bits.size] = edge_bits[::-1]

    base_offsets = np.array([2, 0], dtype=np.int32)
    physical_keys = np.full(
        (storage_slots, num_kv_heads, head_dim),
        np.uint16(0x3F40),
        dtype=np.uint16,
    )
    physical_values = np.full_like(physical_keys, np.uint16(0xBF40))
    for page, physical_page in enumerate(base_offsets):
        logical_slice = slice(page * block_size, (page + 1) * block_size)
        physical_slice = slice(
            int(physical_page) * block_size,
            (int(physical_page) + 1) * block_size,
        )
        physical_keys[physical_slice] = logical_keys[logical_slice]
        physical_values[physical_slice] = logical_values[logical_slice]

    live_counts = np.array([0], dtype=np.int64)
    token_positions = np.arange(capacity, dtype=np.int64)
    evict_mask = np.zeros(capacity, dtype=np.bool_)
    row_positions = np.array([0], dtype=np.int64)
    allocations = []
    try:
        query_device = malloc(query.nbytes, runtime=runtime)
        key_device = malloc(physical_keys.nbytes, runtime=runtime)
        value_device = malloc(physical_values.nbytes, runtime=runtime)
        baseline_device = malloc(query.nbytes, runtime=runtime)
        candidate_device = malloc(query.nbytes, runtime=runtime)
        offsets_device = malloc(base_offsets.nbytes, runtime=runtime)
        live_device = malloc(live_counts.nbytes, runtime=runtime)
        positions_device = malloc(token_positions.nbytes, runtime=runtime)
        evict_device = malloc(evict_mask.nbytes, runtime=runtime)
        row_device = malloc(row_positions.nbytes, runtime=runtime)
        allocations.extend(
            (
                query_device,
                key_device,
                value_device,
                baseline_device,
                candidate_device,
                offsets_device,
                live_device,
                positions_device,
                evict_device,
                row_device,
            )
        )
        for buffer, array in (
            (query_device, query),
            (key_device, physical_keys),
            (value_device, physical_values),
            (offsets_device, base_offsets),
            (positions_device, token_positions),
            (evict_device, evict_mask),
            (row_device, row_positions),
        ):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)

        spans = KVLiveSpans.paged_dense(
            block_table=_tensor(offsets_device.ptr, base_offsets.shape, "int32"),
            live_counts=_tensor(live_device.ptr, live_counts.shape, "int64"),
            token_positions=_tensor(
                positions_device.ptr,
                token_positions.shape,
                "int64",
            ),
            evict_mask=_tensor(evict_device.ptr, evict_mask.shape, "bool"),
            row_positions=_tensor(row_device.ptr, row_positions.shape, "int64"),
            capacity=capacity,
            block_size=block_size,
            storage_dtype="bf16",
        )

        for live_count in (1, 70, 126, 256):
            live_counts[0] = live_count
            row_positions[0] = max(0, live_count - 5)
            evict_mask.fill(False)
            if live_count > 2:
                evict_mask[1] = True
            if live_count > 17:
                evict_mask[17] = True
            runtime.memcpy(
                live_device.ptr,
                host_array_ptr(live_counts),
                live_counts.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            runtime.memcpy(
                evict_device.ptr,
                host_array_ptr(evict_mask),
                evict_mask.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            runtime.memcpy(
                row_device.ptr,
                host_array_ptr(row_positions),
                row_positions.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )

            laguna_global_attention_decode_bf16_spans(
                query_device.ptr,
                key_device.ptr,
                value_device.ptr,
                baseline_device.ptr,
                spans,
                capacity,
                num_q_heads,
                num_kv_heads,
                head_dim,
                head_dim**-0.5,
                library=library,
                runtime=runtime,
            )
            laguna_global_attention_decode_single_page_bf16_spans(
                query_device.ptr,
                key_device.ptr,
                value_device.ptr,
                candidate_device.ptr,
                spans,
                capacity,
                num_q_heads,
                num_kv_heads,
                head_dim,
                head_dim**-0.5,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            baseline = np.empty_like(query)
            candidate = np.empty_like(query)
            copy_device_to_host(
                host_array_ptr(baseline),
                baseline_device,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(candidate),
                candidate_device,
                runtime=runtime,
            )
            np.testing.assert_array_equal(candidate, baseline)

            expected = _ordered_attention_reference(
                query,
                logical_keys[:live_count],
                logical_values[:live_count],
                token_positions=token_positions[:live_count],
                evict_mask=evict_mask[:live_count],
                query_position=int(row_positions[0]),
                num_kv_heads=num_kv_heads,
            )
            kl, agreement = _quality(candidate, expected)
            assert np.isfinite(candidate).all()
            assert kl <= 0.05
            assert agreement >= 0.90

        sentinel = np.full_like(query, np.float32(-123.25))
        copy_host_to_device(
            candidate_device,
            host_array_ptr(sentinel),
            sentinel.nbytes,
            runtime=runtime,
        )
        live_counts[0] = 257
        runtime.memcpy(
            live_device.ptr,
            host_array_ptr(live_counts),
            live_counts.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
        )
        laguna_global_attention_decode_single_page_bf16_spans(
            query_device.ptr,
            key_device.ptr,
            value_device.ptr,
            candidate_device.ptr,
            spans,
            capacity,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim**-0.5,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        untouched = np.empty_like(sentinel)
        copy_device_to_host(
            host_array_ptr(untouched),
            candidate_device,
            runtime=runtime,
        )
        np.testing.assert_array_equal(untouched, sentinel)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
