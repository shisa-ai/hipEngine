"""The Q5_1 T16 repack is bit-lossless: the round trip reconstructs raw bytes."""

import numpy as np
import pytest

from hipengine.quant.gguf_t16 import (
    GGUF_Q5_1_BLOCK_BYTES,
    GGUF_Q5_1_T16_BLOCK_BYTES,
    repack_gguf_q5_1_tile16,
    unpack_gguf_q5_1_tile16,
)


@pytest.mark.parametrize("experts,out_features,in_features", [
    (4, 16, 32),        # one block per row, one tile
    (2, 32, 64),        # two blocks per row
    (3, 48, 320),       # ten blocks per row
    (2, 64, 1024),      # the down leg's K scale class
])
def test_q5_1_tile16_round_trip_is_bit_lossless(experts, out_features, in_features):
    rng = np.random.default_rng(20261008)
    bytes_per_row = (in_features // 32) * GGUF_Q5_1_BLOCK_BYTES
    raw = rng.integers(0, 256, (experts, out_features, bytes_per_row), dtype=np.uint8)
    # Edge values: all-zero and all-0xFF blocks, plus f16 extremes in d/m.
    raw[0, 0, :] = 0
    raw[-1, -1, :] = 0xFF
    raw[0, -1, 0:2] = np.array([0x00, 0x7C], dtype=np.uint8)  # +inf f16 (unusual but bytes are bytes)

    packed = repack_gguf_q5_1_tile16(raw)
    tiles = np.asarray(packed.tiles)
    assert tiles.shape == (
        experts, out_features // 16, in_features // 32, GGUF_Q5_1_T16_BLOCK_BYTES,
    )
    rebuilt = unpack_gguf_q5_1_tile16(packed)
    assert rebuilt.shape == raw.shape
    np.testing.assert_array_equal(raw, np.ascontiguousarray(rebuilt))


def test_q5_1_tile16_preserves_dequant_values():
    """The staged bytes decode to the same values as the raw bytes."""
    from hipengine.quant.gguf import GGMLQuantizationType, _dequant_q5_1_blocks

    rng = np.random.default_rng(7)
    experts, out_features, in_features = 2, 16, 64
    bytes_per_row = (in_features // 32) * GGUF_Q5_1_BLOCK_BYTES
    raw = rng.integers(0, 256, (experts, out_features, bytes_per_row), dtype=np.uint8)
    rebuilt = unpack_gguf_q5_1_tile16(repack_gguf_q5_1_tile16(raw))

    for arr in (raw, np.ascontiguousarray(rebuilt)):
        blocks = arr.reshape(-1, GGUF_Q5_1_BLOCK_BYTES)
        _dequant_q5_1_blocks(blocks)  # exercises the reference on both layouts
    np.testing.assert_array_equal(
        _dequant_q5_1_blocks(raw.reshape(-1, GGUF_Q5_1_BLOCK_BYTES)),
        _dequant_q5_1_blocks(np.ascontiguousarray(rebuilt).reshape(-1, GGUF_Q5_1_BLOCK_BYTES)),
    )
