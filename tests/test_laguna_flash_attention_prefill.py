"""Source-shaped gfx1100 Laguna F16-WMMA FlashAttention contracts.

The candidate follows llama.cpp HIP ``c0bc8591e`` D128/V128, eight-query by
eight-GQA-head geometry while consuming hipEngine's complete ``KVLiveSpans``
ABI and BF16 resident cache.  Existing exact global/SWA kernels remain the
fallback and correctness oracle.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import laguna_flash_attention_prefill as fa
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.kvcache import KVLiveSpans

_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "attention"
    / "laguna_flash_attention_prefill.hip"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def test_laguna_source_flash_attention_registry_build_and_scope_contract() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    fa.register_laguna_flash_attention_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    key = KernelKey(
        "hip_gfx1100",
        "laguna_attention_prefill",
        "bf16",
        "source_f16_wmma_q8_gqa8_spans",
    )
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is fa.laguna_flash_attention_prefill_f16_wmma_bf16_spans
    assert not is_registered(
        KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
    )

    artifact = fa.plan_laguna_flash_attention_prefill_build(
        compiler_version="test"
    )
    assert artifact.output_path.name == "laguna_flash_attention_prefill.so"
    assert any(
        path.name == "laguna_flash_attention_prefill.hip"
        for path in artifact.sources
    )
    source = _SOURCE.read_text()
    assert "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32" in source
    assert "base_offsets" in source
    assert "live_counts" in source
    assert "token_positions" in source
    assert "evict_mask" in source
    assert "row_positions" in source
    assert "torch::Tensor" not in source


def test_laguna_source_flash_attention_rejects_unsupported_shapes_before_build() -> None:
    spans = KVLiveSpans.sliding_ring(
        base_offsets=_tensor(0x1000, (32,), DType.INT32),
        live_counts=_tensor(0x2000, (1,), DType.INT64),
        token_positions=_tensor(0x3000, (32,), DType.INT64),
        evict_mask=_tensor(0x4000, (32,), DType.BOOL),
        row_positions=_tensor(0x5000, (1,), DType.INT64),
        capacity=32,
        storage_dtype=DType.BF16,
    )
    with pytest.raises(ValueError, match="rows"):
        fa.laguna_flash_attention_prefill_f16_wmma_bf16_spans(
            1, 2, 3, 4, spans, 0, 72, 8, 128, 128**-0.5
        )
    with pytest.raises(ValueError, match="num_q_heads"):
        fa.laguna_flash_attention_prefill_f16_wmma_bf16_spans(
            1, 2, 3, 4, spans, 17, 64, 8, 128, 128**-0.5
        )
    with pytest.raises(ValueError, match="head_dim"):
        fa.laguna_flash_attention_prefill_f16_wmma_bf16_spans(
            1, 2, 3, 4, spans, 17, 72, 8, 64, 128**-0.5
        )


def _attention_reference(
    query: np.ndarray,
    logical_key: np.ndarray,
    logical_value: np.ndarray,
    token_positions: np.ndarray,
    evict_mask: np.ndarray,
    *,
    num_q_heads: int,
    sliding_window: int | None,
) -> np.ndarray:
    rows, _, head_dim = query.shape
    num_kv_heads = logical_key.shape[1]
    group = num_q_heads // num_kv_heads
    out = np.empty_like(query)
    for row in range(rows):
        query_position = row
        visible = np.flatnonzero(
            (token_positions >= 0)
            & (token_positions <= query_position)
            & (~evict_mask)
        )
        if sliding_window is not None:
            visible = visible[
                token_positions[visible] > query_position - sliding_window
            ]
        for head in range(num_q_heads):
            kv_head = head // group
            scores = np.einsum(
                "d,td->t",
                query[row, head],
                logical_key[visible, kv_head],
                dtype=np.float32,
            )
            scores *= np.float32(head_dim**-0.5)
            scores -= np.max(scores)
            weights = np.exp(scores, dtype=np.float32)
            weights /= np.sum(weights, dtype=np.float32)
            out[row, head] = np.einsum(
                "t,td->d",
                weights,
                logical_value[visible, kv_head],
                dtype=np.float32,
            )
    return out


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("mode", "num_q_heads", "capacity", "sliding_window", "rows"),
    [
        ("global", 48, 256, None, 17),
        ("swa", 72, 32, 512, 17),
        ("global", 48, 256, None, 65),
        ("swa", 72, 96, 512, 65),
    ],
)
def test_laguna_source_flash_attention_tails_and_spans_pass_quality(
    mode: str,
    num_q_heads: int,
    capacity: int,
    sliding_window: int | None,
    rows: int,
) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    before = memory_stats()
    rng = np.random.default_rng(0xFA22 + num_q_heads + rows)
    query = rng.normal(0.0, 0.16, size=(rows, num_q_heads, 128)).astype(
        np.float32
    )
    logical_key_bits = _bf16_bits(
        rng.normal(0.0, 0.16, size=(capacity, 8, 128)).astype(np.float32)
    )
    logical_value_bits = _bf16_bits(
        rng.normal(0.0, 0.16, size=(capacity, 8, 128)).astype(np.float32)
    )
    token_positions = np.full((capacity,), -1, dtype=np.int64)
    token_positions[:rows] = np.arange(rows, dtype=np.int64)
    evict_mask = np.ones((capacity,), dtype=np.uint8)
    evict_mask[:rows] = 0
    evict_mask[3] = 1
    live_counts = np.asarray([rows], dtype=np.int64)
    row_positions = np.asarray([0], dtype=np.int64)

    if mode == "global":
        base_offsets = np.asarray([0], dtype=np.int32)
        physical_key_bits = logical_key_bits.copy()
        physical_value_bits = logical_value_bits.copy()
    else:
        base_offsets = np.arange(capacity - 1, -1, -1, dtype=np.int32)
        physical_key_bits = np.empty_like(logical_key_bits)
        physical_value_bits = np.empty_like(logical_value_bits)
        physical_key_bits[base_offsets] = logical_key_bits
        physical_value_bits[base_offsets] = logical_value_bits

    expected = _attention_reference(
        query,
        _bf16_to_f32(logical_key_bits),
        _bf16_to_f32(logical_value_bits),
        token_positions,
        evict_mask.astype(bool),
        num_q_heads=num_q_heads,
        sliding_window=sliding_window,
    )
    actual = np.empty_like(expected)
    allocations = []

    def upload(values: np.ndarray):
        buffer = malloc(values.nbytes, runtime=runtime)
        allocations.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(values), runtime=runtime)
        return buffer

    try:
        query_dev = upload(query)
        key_dev = upload(physical_key_bits)
        value_dev = upload(physical_value_bits)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        allocations.append(out_dev)
        base_dev = upload(base_offsets)
        live_dev = upload(live_counts)
        positions_dev = upload(token_positions)
        evict_dev = upload(evict_mask)
        row_positions_dev = upload(row_positions)
        if mode == "global":
            spans = KVLiveSpans.paged_dense(
                block_table=_tensor(
                    base_dev.ptr, base_offsets.shape, DType.INT32
                ),
                live_counts=_tensor(
                    live_dev.ptr, live_counts.shape, DType.INT64
                ),
                token_positions=_tensor(
                    positions_dev.ptr, token_positions.shape, DType.INT64
                ),
                evict_mask=_tensor(
                    evict_dev.ptr, evict_mask.shape, DType.BOOL
                ),
                row_positions=_tensor(
                    row_positions_dev.ptr, row_positions.shape, DType.INT64
                ),
                capacity=capacity,
                block_size=256,
                storage_dtype=DType.BF16,
            )
        else:
            spans = KVLiveSpans.sliding_ring(
                base_offsets=_tensor(
                    base_dev.ptr, base_offsets.shape, DType.INT32
                ),
                live_counts=_tensor(
                    live_dev.ptr, live_counts.shape, DType.INT64
                ),
                token_positions=_tensor(
                    positions_dev.ptr, token_positions.shape, DType.INT64
                ),
                evict_mask=_tensor(
                    evict_dev.ptr, evict_mask.shape, DType.BOOL
                ),
                row_positions=_tensor(
                    row_positions_dev.ptr, row_positions.shape, DType.INT64
                ),
                capacity=capacity,
                storage_dtype=DType.BF16,
            )
        fa.laguna_flash_attention_prefill_f16_wmma_bf16_spans(
            query_dev.ptr,
            key_dev.ptr,
            value_dev.ptr,
            out_dev.ptr,
            spans,
            rows,
            num_q_heads,
            8,
            128,
            128**-0.5,
            sliding_window=0 if sliding_window is None else sliding_window,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), out_dev, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
    assert np.all(np.isfinite(actual)), {
        "nonfinite": int(np.count_nonzero(~np.isfinite(actual))),
        "indices": np.argwhere(~np.isfinite(actual))[:16].tolist(),
    }
    result = evaluate_logits(
        expected.reshape(rows, -1), actual.reshape(rows, -1)
    )
    if result.top1_agreement < 0.90:
        print(
            "nearest",
            np.argmin(
                np.abs(actual[0, 0, :, None] - expected[0, 0, None, :]),
                axis=1,
            ).tolist(),
        )
    assert result.kl_mean <= 0.05, result
    assert result.top1_agreement >= 0.90, {
        "result": result,
        "row0_nearest_expected_dim": np.argmin(
            np.abs(
                actual[0, 0, :, None] - expected[0, 0, None, :]
            ),
            axis=1,
        )[:64].tolist(),
        "max_abs": float(np.max(np.abs(actual - expected))),
    }
    assert result.passed, result
