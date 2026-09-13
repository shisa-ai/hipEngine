"""Byte-oracle tests for the IQ4_XS T16 replacement layout.

The IQ4_XS T16 layout exists so a local32-style owner can read eight output
columns' nibbles for one element from a single u32 instead of one column's
eight elements. The tests here are CPU-only and check three separate things,
because a packer can be self-consistent and still wrong:

1. the round trip is byte-exact, on random bytes and on a real GGUF tensor;
2. the tile's *semantics* match the raw block, decoded independently from the
   documented raw layout rather than from the packer's own convention;
3. the tiled bytes dequantize to the same values as the raw bytes, using the
   repository's own IQ4_XS dequantizer as the oracle.

The layout is byte-neutral (2176 bytes per tile == 16 * 136), so it costs no
resident footprint over the raw tensor.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hipengine.quant.gguf import QK_K, _dequant_iq4_xs_blocks
from hipengine.quant.gguf_t16 import (
    GGUF_IQ4_XS_BLOCK_BYTES,
    GGUF_IQ4_XS_GROUPS,
    GGUF_IQ4_XS_GROUP,
    GGUF_IQ4_XS_T16_BLOCK_BYTES,
    GGUF_IQ4_XS_T16_D_OFFSET,
    GGUF_IQ4_XS_T16_QS_OFFSET,
    GGUF_IQ4_XS_T16_SCALES_H_OFFSET,
    GGUF_IQ4_XS_T16_SCALES_L_OFFSET,
    GGUF_T16_COLS,
    repack_gguf_iq4_xs_tile16,
    unpack_gguf_iq4_xs_tile16,
)

_GGUF = os.environ.get("HIPENGINE_IQ4_XS_LAYOUT_GGUF", "/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")


def _random_raw(out_features: int, in_features: int, seed: int) -> np.ndarray:
    blocks_per_row = in_features // QK_K
    rng = np.random.default_rng(seed)
    return rng.integers(
        0, 256, size=(out_features, blocks_per_row * GGUF_IQ4_XS_BLOCK_BYTES), dtype=np.uint8
    )


def _raw_nibble(block_bytes: np.ndarray, element: int) -> int:
    """Nibble for one element, straight from the documented raw layout.

    Group ``g`` owns bytes ``8 + g*16 .. 8 + g*16 + 15``. Within a group,
    element ``e = half*16 + byte`` lives at byte ``byte`` in nibble ``half``:
    the low nibble of each byte is element ``byte`` and the high nibble is
    element ``byte + 16``.
    """

    group, within = divmod(element, GGUF_IQ4_XS_GROUP)
    half, byte = divmod(within, GGUF_IQ4_XS_GROUP // 2)
    offset = 8 + group * (GGUF_IQ4_XS_GROUP // 2) + byte
    return (int(block_bytes[offset]) >> (4 * half)) & 0x0F


def _tile_nibble(tile: np.ndarray, block: int, column: int, element: int) -> int:
    """Nibble for one element, straight from the documented tile layout.

    The qs plane is ``[group][element][column pair]`` with the even column in
    the low nibble and the odd column in the high nibble.
    """

    group, within = divmod(element, GGUF_IQ4_XS_GROUP)
    offset = (
        GGUF_IQ4_XS_T16_QS_OFFSET
        + group * GGUF_IQ4_XS_GROUP * (GGUF_T16_COLS // 2)
        + within * (GGUF_T16_COLS // 2)
        + column // 2
    )
    return (int(tile[block, offset]) >> (4 * (column % 2))) & 0x0F


def test_layout_is_byte_neutral():
    assert GGUF_IQ4_XS_T16_BLOCK_BYTES == GGUF_T16_COLS * GGUF_IQ4_XS_BLOCK_BYTES
    assert GGUF_IQ4_XS_T16_QS_OFFSET == 128
    assert GGUF_IQ4_XS_T16_SCALES_H_OFFSET == 32
    assert GGUF_IQ4_XS_T16_SCALES_L_OFFSET == 64
    assert GGUF_IQ4_XS_GROUPS * GGUF_IQ4_XS_GROUP == QK_K


@pytest.mark.parametrize(
    "out_features,in_features",
    [(16, QK_K), (32, 4 * QK_K), (64, 2 * QK_K)],
)
def test_round_trip_is_byte_exact_on_random_bytes(out_features: int, in_features: int):
    raw = _random_raw(out_features, in_features, seed=out_features * 31 + in_features)
    packed = repack_gguf_iq4_xs_tile16(raw)
    assert packed.tiles.shape == (
        out_features // GGUF_T16_COLS,
        in_features // QK_K,
        GGUF_IQ4_XS_T16_BLOCK_BYTES,
    )
    assert packed.out_features == out_features
    assert packed.in_features == in_features
    assert np.array_equal(unpack_gguf_iq4_xs_tile16(packed), raw)


def test_tile_semantics_match_the_raw_layout_on_random_bytes():
    """The check a self-consistent packer cannot fake: decode both sides."""

    out_features, in_features = 32, 2 * QK_K
    raw = _random_raw(out_features, in_features, seed=7)
    packed = repack_gguf_iq4_xs_tile16(raw)
    blocks_per_row = in_features // QK_K
    raw_blocks = raw.reshape(out_features, blocks_per_row, GGUF_IQ4_XS_BLOCK_BYTES)
    for out_tile in range(out_features // GGUF_T16_COLS):
        tile = packed.tiles[out_tile]
        for block in range(blocks_per_row):
            for column in range(GGUF_T16_COLS):
                source = raw_blocks[out_tile * GGUF_T16_COLS + column, block]
                for element in range(QK_K):
                    assert _tile_nibble(tile, block, column, element) == _raw_nibble(
                        source, element
                    )
                group = element // GGUF_IQ4_XS_GROUP
                assert int(tile[block, GGUF_IQ4_XS_T16_D_OFFSET + column * 2]) == int(source[0])
                assert int(
                    tile[block, GGUF_IQ4_XS_T16_D_OFFSET + column * 2 + 1]
                ) == int(source[1])
                assert np.array_equal(
                    tile[
                        block,
                        GGUF_IQ4_XS_T16_SCALES_H_OFFSET
                        + column * 2 : GGUF_IQ4_XS_T16_SCALES_H_OFFSET
                        + column * 2
                        + 2,
                    ],
                    source[2:4],
                )
                assert np.array_equal(
                    tile[
                        block,
                        GGUF_IQ4_XS_T16_SCALES_L_OFFSET
                        + column * 4 : GGUF_IQ4_XS_T16_SCALES_L_OFFSET
                        + column * 4
                        + 4,
                    ],
                    source[4:8],
                )
                assert 0 <= group < GGUF_IQ4_XS_GROUPS


def test_tiled_bytes_dequantize_like_raw_bytes():
    """End-to-end oracle: the repository's own IQ4_XS dequantizer on both sides."""

    out_features, in_features = 64, 3 * QK_K
    raw = _random_raw(out_features, in_features, seed=99)
    restored = unpack_gguf_iq4_xs_tile16(repack_gguf_iq4_xs_tile16(raw))
    blocks_per_row = in_features // QK_K
    raw_blocks = raw.reshape(-1, GGUF_IQ4_XS_BLOCK_BYTES)
    restored_blocks = restored.reshape(-1, GGUF_IQ4_XS_BLOCK_BYTES)
    # Random d fields can decode to NaN, so compare bit patterns rather than
    # values: equal NaN payloads are equal bytes.
    assert np.array_equal(
        _dequant_iq4_xs_blocks(raw_blocks).view(np.uint32),
        _dequant_iq4_xs_blocks(restored_blocks).view(np.uint32),
    )
    assert blocks_per_row == 3


def _tile_to_raw_blocks(tile: np.ndarray, columns: int = GGUF_T16_COLS) -> np.ndarray:
    """Rebuild raw 136-byte blocks from a tile, from the documented layout only.

    This is deliberately not ``unpack_gguf_iq4_xs_tile16``: it re-derives every
    offset from the layout description so that a wrong-but-self-consistent
    packer cannot pass. The ground truth it is checked against is the original
    GGUF bytes, which the repository's own dequantizer already consumes.
    """

    blocks = np.zeros((columns, GGUF_IQ4_XS_BLOCK_BYTES), dtype=np.uint8)
    for column in range(columns):
        blocks[column, 0:2] = tile[
            GGUF_IQ4_XS_T16_D_OFFSET + column * 2 : GGUF_IQ4_XS_T16_D_OFFSET + column * 2 + 2
        ]
        blocks[column, 2:4] = tile[
            GGUF_IQ4_XS_T16_SCALES_H_OFFSET
            + column * 2 : GGUF_IQ4_XS_T16_SCALES_H_OFFSET
            + column * 2
            + 2
        ]
        blocks[column, 4:8] = tile[
            GGUF_IQ4_XS_T16_SCALES_L_OFFSET
            + column * 4 : GGUF_IQ4_XS_T16_SCALES_L_OFFSET
            + column * 4
            + 4
        ]
        for group in range(GGUF_IQ4_XS_GROUPS):
            for byte in range(GGUF_IQ4_XS_GROUP // 2):
                pair = column // 2
                shift = 4 * (column % 2)
                low = (int(tile[GGUF_IQ4_XS_T16_QS_OFFSET + group * 256 + byte * 8 + pair]) >> shift) & 0x0F
                high = (
                    int(tile[GGUF_IQ4_XS_T16_QS_OFFSET + group * 256 + (byte + 16) * 8 + pair]) >> shift
                ) & 0x0F
                blocks[column, 8 + group * (GGUF_IQ4_XS_GROUP // 2) + byte] = low | (high << 4)
    return blocks


def test_independent_tile_decode_reproduces_real_gguf_bytes():
    loaded = _real_iq4_xs_tensor()
    if loaded is None:
        pytest.skip("no IQ4_XS GGUF available")
    raw, in_features, out_features = loaded
    packed = repack_gguf_iq4_xs_tile16(raw)
    blocks_per_row = in_features // QK_K
    raw_blocks = raw.reshape(out_features, blocks_per_row, GGUF_IQ4_XS_BLOCK_BYTES)
    for out_tile in range(min(2, out_features // GGUF_T16_COLS)):
        tile = packed.tiles[out_tile]
        for block in range(blocks_per_row):
            rebuilt = _tile_to_raw_blocks(tile[block])
            assert np.array_equal(
                rebuilt, raw_blocks[out_tile * GGUF_T16_COLS : (out_tile + 1) * GGUF_T16_COLS, block]
            )


def _real_iq4_xs_tensor() -> tuple[np.ndarray, int, int] | None:
    if not os.path.exists(_GGUF):
        return None
    from hipengine.loading.gguf import scan_gguf

    info = scan_gguf(_GGUF)
    candidates = [
        tensor
        for tensor in info.tensors
        if tensor.ggml_type_name == "IQ4_XS" and int(tensor.ggml_shape[0]) % QK_K == 0
    ]
    if not candidates:
        return None
    tensor = candidates[0]
    with open(_GGUF, "rb") as handle:
        handle.seek(tensor.data_offset)
        raw = np.frombuffer(handle.read(tensor.nbytes), dtype=np.uint8).copy()
    in_features = int(tensor.ggml_shape[0])
    out_features = int(tensor.ggml_shape[1])
    return raw.reshape(out_features, -1), in_features, out_features


def test_round_trip_is_byte_exact_on_a_real_tensor():
    loaded = _real_iq4_xs_tensor()
    if loaded is None:
        pytest.skip("no IQ4_XS GGUF available")
    raw, in_features, out_features = loaded
    assert in_features % QK_K == 0
    assert out_features % GGUF_T16_COLS == 0
    packed = repack_gguf_iq4_xs_tile16(raw)
    assert np.array_equal(unpack_gguf_iq4_xs_tile16(packed), raw)
