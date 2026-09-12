"""#19 R9: Q8_0 selected grouped WMMA down kernel vs dequant reference.

Pins the kernel's arithmetic against a NumPy dequant reference at the
production geometry (in=512, out=2560, 4 experts, compact rows with
ragged per-expert row counts) through the same tile-map contract the
runner uses (qwen35_moe_wmma_tile_map). Tolerance is the f16-WMMA class
(same as the promoted Q5_1 grouped WMMA down), not bit-exactness.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
)
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _quantize_q8_0(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a [out, in] F32 matrix to raw Q8_0 bytes; return (bytes, d)."""
    out, in_f = w.shape
    blocks = in_f // 32
    qs = np.empty((out, blocks, 32), dtype=np.int8)
    d = np.empty((out, blocks), dtype=np.float32)
    for b in range(blocks):
        chunk = w[:, b * 32:(b + 1) * 32]
        amax = np.max(np.abs(chunk), axis=1)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        d[:, b] = scale
        q = np.rint(chunk / scale[:, None]).astype(np.int8)
        q = np.clip(q, -127, 127)
        qs[:, b] = q
    raw = np.empty((out, blocks * 34), dtype=np.uint8)
    for b in range(blocks):
        d16 = d[:, b].astype(np.float16).view(np.uint16)
        raw[:, b * 34:b * 34 + 2] = d16[:, None]
        raw[:, b * 34 + 2:b * 34 + 34] = qs[:, b].view(np.uint8).reshape(out, 32)
    return raw.reshape(-1), d


def _tile_map(expert_rows: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Host re-implementation of qwen35_moe_wmma_tile_map (16-row tiles)."""
    starts = np.zeros(len(expert_rows) + 1, dtype=np.int64)
    for e, n in enumerate(expert_rows):
        starts[e + 1] = starts[e] + n
    wmma_start = np.zeros(len(expert_rows) + 1, dtype=np.int64)
    tiles: list[int] = []
    for e, n in enumerate(expert_rows):
        wmma_start[e] = len(tiles) * 16
        padded = (n + 15) // 16
        tiles.extend([e] * padded)
        wmma_start[e + 1] = len(tiles) * 16
    return starts, wmma_start, len(tiles)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")
def test_q8_0_selected_grouped_wmma_matches_reference():
    rng = np.random.default_rng(7409)
    experts, in_f, out_f = 4, 512, 320
    expert_rows = [17, 33, 1, 48]
    compact = sum(expert_rows)

    # Weights: [experts, out, in]
    w = rng.normal(0, 0.5, size=(experts, out_f, in_f)).astype(np.float32)
    raw_all = np.empty((experts, out_f, in_f // 32 * 34), dtype=np.uint8)
    d_all = np.empty((experts, out_f, in_f // 32), dtype=np.float32)
    for e in range(experts):
        raw_all[e], d_all[e] = _quantize_q8_0(w[e])
    x = rng.normal(0, 1.0, size=(compact, in_f)).astype(np.float32)
    x_bf16 = _f32_to_bf16_bits(x)

    starts, wmma_start, total_tiles = _tile_map(expert_rows)
    wmma_total_rows = total_tiles * 16

    runtime = get_hip_runtime()
    allocations = []
    try:
        dx = _upload(x_bf16.reshape(compact, in_f), runtime, allocations)
        d_starts = _upload(starts, runtime, allocations)
        d_wmma = _upload(wmma_start, runtime, allocations)
        d_tiles = _upload(np.asarray(tiles, dtype=np.int64), runtime, allocations)
        dw = _upload(raw_all.reshape(-1), runtime, allocations)
        out = _alloc(compact * out_f, np.float32, runtime, allocations)

        gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
            dx.ptr, d_starts.ptr, d_wmma.ptr, d_tiles.ptr, dw.ptr, out.ptr,
            compact, experts, in_f, out_f, wmma_total_rows,
            runtime=runtime, library=None,
        )
        runtime.device_synchronize()
        got = _download(out, (compact, out_f), np.float32, runtime)
    finally:
        while allocations:
            free(allocations.pop())

    # Reference: dequant to bf16 weights, bf16 activations, exact f32 dot.
    def dequant(e):
        blocks = in_f // 32
        wq = np.empty((out_f, in_f), dtype=np.float32)
        raw = raw_all[e]
        for c in range(out_f):
            for b in range(blocks):
                d16 = np.frombuffer(
                    raw[c, b * 34:b * 34 + 2].tobytes(), dtype=np.float16)[0]
                q = raw[c, b * 34 + 2:b * 34 + 34].view(np.int8)
                wq[c, b * 32:(b + 1) * 32] = np.float32(d16) * q.astype(np.float32)
        return wq

    row = 0
    worst = 0.0
    for e, n in enumerate(expert_rows):
        we = dequant(e)
        # The kernel converts activations and dequantized weights to f16
        # before the WMMA dot; emulate both roundings.
        we16 = we.astype(np.float16).astype(np.float32)
        x16 = x[row:row + n].astype(np.float16).astype(np.float32)
        ref = x16 @ we16.T
        rel = np.abs(got[row:row + n] - ref) / np.maximum(np.abs(ref), 1e-6)
        worst = max(worst, float(np.quantile(rel, 0.999)))
        row += n
    assert worst < 5e-3, f"f16-WMMA reference drift too large: {worst}"


def _f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u32 = x.astype(np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + lsb
    return (u32 >> 16).astype(np.uint16)
