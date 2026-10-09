"""GPU correctness tests for the Q5_1T16 selected grouped WMMA prefill leaf.

The tile kernel must reproduce the raw-block grouped WMMA kernel's bf16 output
byte-for-byte at the same geometry: the tile layout is a pure permutation whose
value bytes carry exactly the integer the raw decoder derives, and the kernel
keeps the raw kernel's accumulation order.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, free, host_array_ptr
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
    build_qwen4_exp_q5_1,
    qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
    qwen4_exp_q5_1_t16_selected_grouped_prefill_bf16_bf16_out,
)
from hipengine.quant.gguf_t16 import repack_gguf_q5_1_tile16


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _metadata(counts: np.ndarray):
    counts = np.asarray(counts, dtype=np.int64)
    experts = counts.size
    start_compact = np.zeros(experts + 1, dtype=np.int64)
    start_compact[1:] = np.cumsum(counts)
    padded = ((counts + 15) // 16) * 16
    start_wmma = np.zeros(experts + 1, dtype=np.int64)
    start_wmma[1:] = np.cumsum(padded)
    tile_expert = np.asarray(
        [e for e, rows in enumerate(padded) for _ in range(int(rows) // 16)],
        dtype=np.int64,
    )
    return start_compact, start_wmma, tile_expert, int(start_compact[-1]), int(start_wmma[-1])


def _raw_q5_1_blocks(*, experts: int, out_features: int, blocks_per_row: int) -> np.ndarray:
    rng = np.random.default_rng(90210 + experts + out_features + blocks_per_row)
    raw = rng.integers(
        0, 256, size=(experts, out_features, blocks_per_row * 24), dtype=np.uint8
    )
    blocks = raw.reshape(experts, out_features, blocks_per_row, 24)
    d = np.full((experts, out_features, blocks_per_row), 0.03125, dtype=np.float16)
    m = np.full((experts, out_features, blocks_per_row), -0.25, dtype=np.float16)
    blocks[..., 0:2] = d.view(np.uint8).reshape(experts, out_features, blocks_per_row, 2)
    blocks[..., 2:4] = m.view(np.uint8).reshape(experts, out_features, blocks_per_row, 2)
    return blocks.reshape(experts, out_features, blocks_per_row * 24)


def _bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _to_device(arr: np.ndarray, runtime):
    from hipengine.core.memory import malloc

    contiguous = np.ascontiguousarray(arr)
    dev = malloc(contiguous.nbytes, runtime=runtime)
    from hipengine.core.memory import copy_host_to_device

    copy_host_to_device(dev, host_array_ptr(contiguous), runtime=runtime)
    return dev


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "counts",
    [
        np.array([16, 16, 16], dtype=np.int64),
        np.array([5, 33, 1, 20], dtype=np.int64),
    ],
    ids=["uniform", "ragged"],
)
def test_q5_1_t16_grouped_prefill_matches_raw_wmma_bytes(counts: np.ndarray) -> None:
    experts = counts.size
    in_features = 96   # three 32-wide K blocks
    out_features = 64  # two 32-column output tiles
    raw = _raw_q5_1_blocks(experts=experts, out_features=out_features, blocks_per_row=in_features // 32)
    tiles = repack_gguf_q5_1_tile16(raw).tiles

    start_compact, start_wmma, tile_expert, compact_rows, wmma_total_rows = _metadata(counts)
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((compact_rows, in_features)) * 0.05).astype(np.float32)
    x_host = _bf16_bits(x)

    runtime = get_hip_runtime()
    library = build_qwen4_exp_q5_1(load=True)
    buffers = []
    x_dev = _to_device(x_host, runtime)
    sc = _to_device(start_compact, runtime)
    sw = _to_device(start_wmma, runtime)
    te = _to_device(tile_expert, runtime)
    w = _to_device(raw, runtime)
    tl = _to_device(tiles, runtime)
    out_dev = _to_device(np.zeros((compact_rows, out_features), dtype=np.uint16), runtime)
    buffers += [x_dev, sc, sw, te, w, tl, out_dev]
    try:
        outs = {}
        for name in ("raw", "t16"):
            if name == "raw":
                qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
                    x_dev.ptr, sc.ptr, sw.ptr, te.ptr, w.ptr, out_dev.ptr,
                    compact_rows, experts, in_features, out_features,
                    wmma_total_rows, library=library, runtime=runtime,
                )
            else:
                qwen4_exp_q5_1_t16_selected_grouped_prefill_bf16_bf16_out(
                    x_dev.ptr, sc.ptr, sw.ptr, te.ptr, tl.ptr, out_dev.ptr,
                    compact_rows, experts, in_features, out_features,
                    wmma_total_rows, library=library, runtime=runtime,
                )
            runtime.device_synchronize()
            host = np.zeros((compact_rows, out_features), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(host), out_dev, runtime=runtime)
            outs[name] = host.copy()
        assert np.array_equal(outs["raw"], outs["t16"])
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)
