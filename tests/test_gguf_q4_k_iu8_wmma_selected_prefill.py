"""Correctness test for the iu8-WMMA (llama mul_mat_q class) Q4_K selected
dual gate/up prefill kernel.

The kernel requantizes activations and Q4_K weights per 32-k subblock to
signed int8 and rescales in f32, so the contract is dp4a-class tolerance
against a NumPy dequant reference (not exact parity with the f32 owner)."""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
    gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out,
)
from tests.test_gguf_q4_k_selected_wmma_prefill import (
    _bf16_bits_to_float32,
    _hip_available,
)

_TOLERANCE_IU8 = 5e-2  # dp4a-class: per-output relative tolerance


def _decode_output_bf16(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return _bf16_bits_to_float32(raw).reshape(rows, cols)


def _float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def _make_q4_k_weights(*, experts: int, out_features: int, in_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = in_features // 256
    w = np.zeros((experts, out_features, blocks, 144), dtype=np.uint8)
    d16 = rng.uniform(0.005, 0.02, size=(experts, out_features, blocks)).astype(np.float16).view(np.uint16)
    m16 = rng.uniform(0.001, 0.008, size=(experts, out_features, blocks)).astype(np.float16).view(np.uint16)
    w[..., 0] = d16 & 0xFF
    w[..., 1] = d16 >> 8
    w[..., 2] = m16 & 0xFF
    w[..., 3] = m16 >> 8
    w[..., 4:16] = rng.integers(0, 64, size=(experts, out_features, blocks, 12), dtype=np.uint8)
    w[..., 16:144] = rng.integers(0, 256, size=(experts, out_features, blocks, 128), dtype=np.uint8)
    return w.reshape(experts, out_features, blocks * 144)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "experts,in_features,out_features,rows",
    [
        (1, 256, 128, 16),
        (4, 512, 128, 32),
        (8, 1024, 128, 77),
        (16, 2048, 640, 160),
    ],
)
def test_iu8_wmma_selected_dual_matches_numpy_reference(
    experts: int, in_features: int, out_features: int, rows: int
) -> None:
    rng = np.random.default_rng(rows)
    counts = rng.multinomial(rows, [1.0 / experts] * experts).astype(np.int64)
    compact_rows = int(counts.sum())
    starts = np.zeros(experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    padded16 = ((counts + 15) // 16) * 16
    starts16 = np.zeros(experts + 1, dtype=np.int64)
    starts16[1:] = np.cumsum(padded16)
    tile_expert = np.repeat(np.arange(experts, dtype=np.int64), padded16 // 16)
    wmma_total_rows = int(starts16[-1])

    activations_f = (rng.standard_normal((compact_rows, in_features)) * 0.5).astype(np.float32)
    activations = _float32_to_bf16_bits(activations_f).reshape(-1)
    weight_shape = (experts, out_features, (in_features // 256) * 144)
    weights_a = _make_q4_k_weights(experts=experts, out_features=out_features, in_features=in_features, seed=rows)
    weights_b = _make_q4_k_weights(experts=experts, out_features=out_features, in_features=in_features, seed=rows + 1)

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    allocs = []
    try:
        rows_dev = malloc(activations.nbytes, runtime=runtime); allocs.append(rows_dev)
        starts_dev = malloc(starts.nbytes, runtime=runtime); allocs.append(starts_dev)
        starts16_dev = malloc(starts16.nbytes, runtime=runtime); allocs.append(starts16_dev)
        tiles_dev = malloc(tile_expert.nbytes, runtime=runtime); allocs.append(tiles_dev)
        wa_dev = malloc(weights_a.nbytes, runtime=runtime); allocs.append(wa_dev)
        wb_dev = malloc(weights_b.nbytes, runtime=runtime); allocs.append(wb_dev)
        out_cols = 2 * out_features
        out_dev = malloc(compact_rows * out_cols * 2, runtime=runtime); allocs.append(out_dev)
        copy_host_to_device(rows_dev, host_array_ptr(activations), runtime=runtime)
        copy_host_to_device(starts_dev, host_array_ptr(starts), runtime=runtime)
        copy_host_to_device(starts16_dev, host_array_ptr(starts16), runtime=runtime)
        copy_host_to_device(tiles_dev, host_array_ptr(tile_expert), runtime=runtime)
        copy_host_to_device(wa_dev, host_array_ptr(weights_a), runtime=runtime)
        copy_host_to_device(wb_dev, host_array_ptr(weights_b), runtime=runtime)

        gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out(
            rows_dev.ptr,
            starts_dev.ptr,
            starts16_dev.ptr,
            tiles_dev.ptr,
            wa_dev.ptr,
            wb_dev.ptr,
            out_dev.ptr,
            compact_rows,
            in_features,
            out_features,
            out_features,
            experts,
            wmma_total_rows,
            runtime=runtime,
        )
        runtime.device_synchronize()
        raw = np.empty(compact_rows * out_cols, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(raw), out_dev, raw.nbytes, runtime=runtime)
    finally:
        for allocation in reversed(allocs):
            free(allocation, runtime=runtime)

    got = _decode_output_bf16(raw, compact_rows, out_cols)

    def dequant(matrix: np.ndarray) -> np.ndarray:
        blocks = in_features // 256
        decoded = np.zeros((out_features, in_features), dtype=np.float64)
        flat = matrix.reshape(out_features, blocks, 144)
        d = flat[:, :, 0:2].copy().view(np.float16).astype(np.float32)
        dmin = flat[:, :, 2:4].copy().view(np.float16).astype(np.float32)
        scales = flat[:, :, 4:16].astype(np.int64)
        qs = flat[:, :, 16:144].astype(np.int64)
        for blk in range(blocks):
            for sub in range(8):
                if sub < 4:
                    sc = scales[:, blk, sub] & 0x3F
                    mn = scales[:, blk, 4 + sub] & 0x3F
                else:
                    idx = sub - 4
                    sc = (scales[:, blk, 8 + idx] & 0x0F) | ((scales[:, blk, idx] >> 2) & 0x30)
                    mn = (scales[:, blk, 8 + idx] >> 4) | ((scales[:, blk, 4 + idx] >> 2) & 0x30)
                nibbles = qs[:, blk, (sub >> 1) * 32:(sub >> 1) * 32 + 32]
                vals = (nibbles & 0x0F) if (sub & 1) == 0 else (nibbles >> 4)
                col = blk * 256 + sub * 32
                decoded[:, col:col + 32] = (
                    d[:, blk, 0][:, None] * sc[:, None] * vals
                    - dmin[:, blk, 0][:, None] * mn[:, None]
                )
        return decoded

    weights_a_flat = weights_a.reshape(experts, out_features, -1)
    weights_b_flat = weights_b.reshape(experts, out_features, -1)

    cursor = 0
    for expert in range(experts):
        count = int(counts[expert])
        if count == 0:
            continue
        act = _bf16_bits_to_float32(activations[cursor * in_features:(cursor + count) * in_features])
        act = act.reshape(count, in_features).astype(np.float64)
        ref_a = act @ dequant(weights_a_flat[expert]).T
        ref_b = act @ dequant(weights_b_flat[expert]).T
        got_a = got[cursor:cursor + count, :out_features]
        got_b = got[cursor:cursor + count, out_features:]
        tol_a = _TOLERANCE_IU8 * np.maximum(np.abs(ref_a).max(), 1.0)
        tol_b = _TOLERANCE_IU8 * np.maximum(np.abs(ref_b).max(), 1.0)
        assert np.abs(got_a - ref_a).max() < tol_a, (
            f"expert {expert} gate mismatch: {np.abs(got_a - ref_a).max()} vs {tol_a}"
        )
        assert np.abs(got_b - ref_b).max() < tol_b, (
            f"expert {expert} up mismatch: {np.abs(got_b - ref_b).max()} vs {tol_b}"
        )
        cursor += count
