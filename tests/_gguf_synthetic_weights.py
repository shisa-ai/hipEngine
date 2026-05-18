"""Synthetic GGUF block-quantised weight helpers for P9 decode-GEMV tests.

The legacy ``tests/test_gguf_*_gemv.py::make_*_weight`` helpers compute
indices using uint8 arithmetic, which overflows for ``out_features > 127``.
The P9.B decode-GEMV fixtures (tasks #20-#24) need realistic shapes like
``out_features=2048/4096``, so this module re-exports the same helpers with
their indexing math done in ``int64`` before downcasting to ``uint8``.

These helpers produce byte-identical output to the originals for
``out_features <= 127`` (we re-use the same dequant math). They are not
intended to replace the originals; the originals remain the canonical source
in ``tests/test_gguf_q4_k_gemv.py`` / ``tests/test_gguf_k_gemv.py`` for the
existing P8 fixtures.
"""

from __future__ import annotations

import numpy as np

QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210
Q8_0_BLOCK_BYTES = 34


def _pack_q4_k_scales(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    scales = np.asarray(scales, dtype=np.uint8)
    mins = np.asarray(mins, dtype=np.uint8)
    out = np.zeros(12, dtype=np.uint8)
    out[:4] = (scales[:4] & 0x3F) | ((scales[4:] & 0x30) << 2)
    out[4:8] = (mins[:4] & 0x3F) | ((mins[4:] & 0x30) << 2)
    out[8:12] = (scales[4:] & 0x0F) | ((mins[4:] & 0x0F) << 4)
    return out


def _make_q4_k_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.015625 * (1 + (out_idx % 5)))
    dmin = np.float16(0.0078125 * (1 + (block_idx % 3)))
    base = np.arange(8, dtype=np.int64)
    scales = ((base * 3 + out_idx + block_idx) % 63 + 1).astype(np.uint8)
    mins = ((base * 5 + 2 * out_idx + block_idx) % 17).astype(np.uint8)
    q = ((np.arange(QK_K, dtype=np.int64) + out_idx * 7 + block_idx * 11) % 16).astype(np.uint8)

    packed_scales = _pack_q4_k_scales(scales, mins)
    q_groups = q.reshape(8, 32)
    packed_q = np.empty(128, dtype=np.uint8)
    for pair in range(4):
        packed_q[pair * 32 : (pair + 1) * 32] = q_groups[2 * pair] | (q_groups[2 * pair + 1] << 4)
    return np.concatenate(
        [
            np.asarray([d], dtype=np.float16).view(np.uint8),
            np.asarray([dmin], dtype=np.float16).view(np.uint8),
            packed_scales,
            packed_q,
        ]
    )


def make_q4_k_weight(out_features: int, in_features: int) -> np.ndarray:
    """Build raw Q4_K bytes ``[out_features, blocks * Q4_K_BLOCK_BYTES]``.

    Same math as ``tests/test_gguf_q4_k_gemv.py::make_q4_k_weight`` but the
    index arithmetic uses int64 to avoid uint8 overflow for large
    ``out_features``.
    """

    if in_features % QK_K:
        raise ValueError("in_features must be a multiple of 256")
    blocks_per_row = in_features // QK_K
    data = np.empty((out_features, blocks_per_row * Q4_K_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            block = _make_q4_k_block(out_idx, block_idx)
            start = block_idx * Q4_K_BLOCK_BYTES
            data[out_idx, start : start + Q4_K_BLOCK_BYTES] = block
    return data


def _make_q5_k_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.015625 * (1 + (out_idx % 5)))
    dmin = np.float16(0.0078125 * (1 + (block_idx % 3)))
    base = np.arange(8, dtype=np.int64)
    scales = ((base * 3 + out_idx + block_idx) % 63 + 1).astype(np.uint8)
    mins = ((base * 5 + 2 * out_idx + block_idx) % 17).astype(np.uint8)
    packed_scales = _pack_q4_k_scales(scales, mins)
    q = ((np.arange(QK_K, dtype=np.int64) + out_idx * 13 + block_idx * 17) % 32).astype(np.uint8)
    qh = np.zeros(32, dtype=np.uint8)
    for sb in range(8):
        for lane in range(32):
            idx = sb * 32 + lane
            high = (q[idx] >> 4) & 0x01
            qh[lane] |= high << sb
    ql = q & 0x0F
    ql_groups = ql.reshape(8, 32)
    packed_ql = np.empty(128, dtype=np.uint8)
    for pair in range(4):
        packed_ql[pair * 32 : (pair + 1) * 32] = ql_groups[2 * pair] | (ql_groups[2 * pair + 1] << 4)
    return np.concatenate(
        [
            np.asarray([d], dtype=np.float16).view(np.uint8),
            np.asarray([dmin], dtype=np.float16).view(np.uint8),
            packed_scales,
            qh,
            packed_ql,
        ]
    )


def make_q5_k_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be a multiple of 256")
    blocks_per_row = in_features // QK_K
    data = np.empty((out_features, blocks_per_row * Q5_K_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * Q5_K_BLOCK_BYTES
            data[out_idx, start : start + Q5_K_BLOCK_BYTES] = _make_q5_k_block(out_idx, block_idx)
    return data


def _make_q6_k_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.0078125 * (1 + (out_idx % 7)))
    base = np.arange(16, dtype=np.int64)
    scales = (((base * 3 + out_idx + block_idx) % 33) - 16).astype(np.int8)
    q = (((np.arange(QK_K, dtype=np.int64) + out_idx * 11 + block_idx * 19) % 64) - 32).astype(np.int8)
    q_unsigned = (q + 32).astype(np.uint8)
    # Q6_K layout: 128 ql + 64 qh + 16 scales (int8) + fp16 d, total 210 bytes.
    # ql[128]: 4-bit lower nibbles for 256 K's, packed in the quartet
    # interleaving expected by the dequant helper.
    ql = np.zeros(128, dtype=np.uint8)
    qh = np.zeros(64, dtype=np.uint8)
    for k in range(QK_K):
        group32 = k >> 5
        lane = k & 31
        base64 = 64 if group32 >= 4 else 0
        ql_group = group32 & 1
        ql_idx = base64 + ql_group * 32 + lane
        low_nibble = (group32 & 2) == 0
        qh_base = 32 if group32 >= 4 else 0
        qh_idx = qh_base + lane
        qh_shift = 2 * (group32 & 3)
        low = q_unsigned[k] & 0x0F
        high = (q_unsigned[k] >> 4) & 0x03
        if low_nibble:
            ql[ql_idx] |= low
        else:
            ql[ql_idx] |= low << 4
        qh[qh_idx] |= high << qh_shift
    return np.concatenate(
        [
            ql,
            qh,
            scales.view(np.uint8),
            np.asarray([d], dtype=np.float16).view(np.uint8),
        ]
    )


def make_q6_k_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be a multiple of 256")
    blocks_per_row = in_features // QK_K
    data = np.empty((out_features, blocks_per_row * Q6_K_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * Q6_K_BLOCK_BYTES
            data[out_idx, start : start + Q6_K_BLOCK_BYTES] = _make_q6_k_block(out_idx, block_idx)
    return data


def _make_q8_0_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.03125 * (1 + (out_idx % 5)))
    q = (((np.arange(32, dtype=np.int64) + out_idx * 7 + block_idx * 3) % 31) - 15).astype(np.int8)
    return np.concatenate([np.asarray([d], dtype=np.float16).view(np.uint8), q.view(np.uint8)])


def make_q8_0_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % 32:
        raise ValueError("in_features must be a multiple of 32")
    blocks_per_row = in_features // 32
    data = np.empty((out_features, blocks_per_row * Q8_0_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * Q8_0_BLOCK_BYTES
            data[out_idx, start : start + Q8_0_BLOCK_BYTES] = _make_q8_0_block(out_idx, block_idx)
    return data
