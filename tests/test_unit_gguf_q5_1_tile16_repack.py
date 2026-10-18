"""Unit tests for the Q5_1T16 tile repack and its inverse."""

from __future__ import annotations

import numpy as np

from hipengine.quant.gguf_t16 import (
    GGUF_Q5_1_BLOCK_BYTES,
    GGUF_Q5_1_T16_BLOCK_BYTES,
    GGUF_Q5_1_T16_D_OFFSET,
    GGUF_Q5_1_T16_M_OFFSET,
    GGUF_Q5_1_T16_V_OFFSET,
    repack_gguf_q5_1_tile16,
    unpack_gguf_q5_1_tile16,
)


def _raw_q5_1_bytes(*, experts: int, out_features: int, blocks_per_row: int) -> np.ndarray:
    rng = np.random.default_rng(1234 + experts * 17 + out_features + blocks_per_row)
    raw = rng.integers(
        0,
        256,
        size=(experts, out_features, blocks_per_row * GGUF_Q5_1_BLOCK_BYTES),
        dtype=np.uint8,
    )
    # Force d/m to finite ordinary fp16 values so decode tests do not depend
    # on NaN payload behavior; the bit-exact roundtrip holds for any bytes.
    blocks = raw.reshape(experts, out_features, blocks_per_row, GGUF_Q5_1_BLOCK_BYTES)
    d = np.full((experts, out_features, blocks_per_row), 0.125, dtype=np.float16).view(np.uint8).reshape(experts, out_features, blocks_per_row, 2)
    m = np.full((experts, out_features, blocks_per_row), -0.5, dtype=np.float16).view(np.uint8).reshape(experts, out_features, blocks_per_row, 2)
    blocks[..., 0:2] = d
    blocks[..., 2:4] = m
    return blocks.reshape(experts, out_features, blocks_per_row * GGUF_Q5_1_BLOCK_BYTES)


def test_q5_1_t16_repack_roundtrip_exact() -> None:
    raw = _raw_q5_1_bytes(experts=3, out_features=32, blocks_per_row=5)
    packed = repack_gguf_q5_1_tile16(raw)
    assert packed.tiles.shape == (3, 2, 5, GGUF_Q5_1_T16_BLOCK_BYTES)
    assert packed.in_features == 5 * 32
    back = unpack_gguf_q5_1_tile16(packed)
    assert back.shape == raw.shape
    assert np.array_equal(back, raw)


def test_q5_1_t16_roundtrip_via_array() -> None:
    raw = _raw_q5_1_bytes(experts=1, out_features=16, blocks_per_row=2)
    packed = repack_gguf_q5_1_tile16(raw)
    back = unpack_gguf_q5_1_tile16(packed.tiles, out_features=packed.out_features)
    assert np.array_equal(back, raw)


def test_q5_1_t16_value_bytes_match_raw_decode() -> None:
    """The tile's value byte must equal the raw decoder's ``low | (high << 4)``."""

    raw = _raw_q5_1_bytes(experts=2, out_features=16, blocks_per_row=3)
    packed = repack_gguf_q5_1_tile16(raw)
    blocks = raw.reshape(2, 16, 3, GGUF_Q5_1_BLOCK_BYTES)
    tiles = packed.tiles
    for expert in range(2):
        for blk in range(3):
            for col in range(16):
                block = blocks[expert, col, blk]
                qh = int(block[4]) | (int(block[5]) << 8) | (int(block[6]) << 16) | (int(block[7]) << 24)
                for k in range(32):
                    packed_byte = int(block[8 + (k & 15)])
                    low = packed_byte & 0x0F if k < 16 else packed_byte >> 4
                    high = (qh >> k) & 1
                    expect = low | (high << 4)
                    got = int(tiles[expert, 0, blk, GGUF_Q5_1_T16_V_OFFSET + col * 32 + (k // 16) * 16 + (k % 16)])
                    assert got == expect, (expert, blk, col, k)


def test_q5_1_t16_headers_match_raw_scales() -> None:
    raw = _raw_q5_1_bytes(experts=2, out_features=16, blocks_per_row=3)
    packed = repack_gguf_q5_1_tile16(raw)
    blocks = raw.reshape(2, 16, 3, GGUF_Q5_1_BLOCK_BYTES)
    for expert in range(2):
        for blk in range(3):
            for col in range(16):
                assert (
                    packed.tiles[expert, 0, blk, GGUF_Q5_1_T16_D_OFFSET + col * 2 :
                                 GGUF_Q5_1_T16_D_OFFSET + col * 2 + 2]
                    == blocks[expert, col, blk, 0:2]
                ).all()
                assert (
                    packed.tiles[expert, 0, blk, GGUF_Q5_1_T16_M_OFFSET + col * 2 :
                                 GGUF_Q5_1_T16_M_OFFSET + col * 2 + 2]
                    == blocks[expert, col, blk, 2:4]
                ).all()


def test_q5_1_t16_repack_rejects_bad_geometry() -> None:
    raw = _raw_q5_1_bytes(experts=1, out_features=16, blocks_per_row=2)
    with __import__("pytest").raises(ValueError):
        repack_gguf_q5_1_tile16(raw.reshape(1, 16, 24 * 2 + 1))
    with __import__("pytest").raises(ValueError):
        repack_gguf_q5_1_tile16(raw.reshape(16, 1 * 2 * 24))
