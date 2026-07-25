from __future__ import annotations

import numpy as np
import pytest

from hipengine.quant.gguf_q4_k import (
    GGUF_Q4_K_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_COLS,
    GGUF_Q4_K_TILE16_LITE_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_LITE_META_OFFSET,
    GGUF_Q4_K_TILE16_LITE_Q_OFFSET,
    repack_gguf_q4_k_tile16,
    repack_gguf_q4_k_tile16_lite,
    unpack_gguf_q4_k_tile16,
    unpack_gguf_q4_k_tile16_lite,
)


def _raw_q4_k_bytes(*, experts: int, out_features: int, blocks_per_row: int) -> np.ndarray:
    rng = np.random.default_rng(1234 + experts * 17 + out_features + blocks_per_row)
    raw = rng.integers(
        0,
        256,
        size=(experts, out_features, blocks_per_row * GGUF_Q4_K_BLOCK_BYTES),
        dtype=np.uint8,
    )
    # Force d/dmin to finite ordinary fp16 values so future decode tests do not
    # accidentally depend on NaN payload behavior. The bit-exact roundtrip would
    # work with arbitrary bytes, but finite metadata is a better fixture.
    blocks = raw.reshape(experts, out_features, blocks_per_row, GGUF_Q4_K_BLOCK_BYTES)
    d = np.full((experts, out_features, blocks_per_row), 0.125, dtype=np.float16).view(np.uint8)
    dmin = np.full((experts, out_features, blocks_per_row), 0.25, dtype=np.float16).view(np.uint8)
    blocks[..., 0:2] = d.reshape(experts, out_features, blocks_per_row, 2)
    blocks[..., 2:4] = dmin.reshape(experts, out_features, blocks_per_row, 2)
    return raw


def test_q4_k_tile16_repack_roundtrips_raw_bytes_exactly() -> None:
    raw = _raw_q4_k_bytes(experts=3, out_features=32, blocks_per_row=2)

    packed = repack_gguf_q4_k_tile16(raw)
    restored = unpack_gguf_q4_k_tile16(packed)

    assert packed.tiles.shape == (3, 2, 2, GGUF_Q4_K_TILE16_BLOCK_BYTES)
    assert packed.experts == 3
    assert packed.out_features == 32
    assert packed.in_features == 512
    np.testing.assert_array_equal(restored, raw)


def test_q4_k_tile16_repack_has_expected_near_raw_storage_overhead() -> None:
    raw = _raw_q4_k_bytes(experts=2, out_features=16, blocks_per_row=1)
    packed = repack_gguf_q4_k_tile16(raw)

    raw_tile_bytes = GGUF_Q4_K_TILE16_COLS * GGUF_Q4_K_BLOCK_BYTES
    assert raw_tile_bytes == 2304
    assert GGUF_Q4_K_TILE16_BLOCK_BYTES == 2368
    assert packed.tiles.nbytes == 2 * GGUF_Q4_K_TILE16_BLOCK_BYTES
    assert packed.tiles.nbytes / raw.nbytes == pytest.approx(2368 / 2304)


def test_q4_k_tile16_lite_repack_roundtrips_raw_bytes_exactly() -> None:
    raw = _raw_q4_k_bytes(experts=3, out_features=32, blocks_per_row=2)

    packed = repack_gguf_q4_k_tile16_lite(raw)
    restored = unpack_gguf_q4_k_tile16_lite(packed)

    assert packed.tiles.shape == (
        3,
        2,
        2,
        GGUF_Q4_K_TILE16_LITE_BLOCK_BYTES,
    )
    assert packed.experts == 3
    assert packed.out_features == 32
    assert packed.in_features == 512
    np.testing.assert_array_equal(restored, raw)


def test_q4_k_tile16_lite_is_byte_neutral_with_raw_q4_k() -> None:
    raw = _raw_q4_k_bytes(experts=2, out_features=16, blocks_per_row=3)
    packed = repack_gguf_q4_k_tile16_lite(raw)

    raw_tile_bytes = GGUF_Q4_K_TILE16_COLS * GGUF_Q4_K_BLOCK_BYTES
    assert raw_tile_bytes == 2304
    assert GGUF_Q4_K_TILE16_LITE_BLOCK_BYTES == raw_tile_bytes
    assert packed.tiles.nbytes == raw.nbytes


def test_q4_k_tile16_lite_keeps_packed_metadata_byte_planes_coalesced() -> None:
    raw = _raw_q4_k_bytes(experts=1, out_features=16, blocks_per_row=2)
    packed = repack_gguf_q4_k_tile16_lite(raw)

    raw_meta = raw.reshape(1, 16, 2, 144)[..., 4:16]
    expected = raw_meta.transpose(0, 2, 3, 1)
    actual = packed.tiles[
        ...,
        GGUF_Q4_K_TILE16_LITE_META_OFFSET:
        GGUF_Q4_K_TILE16_LITE_Q_OFFSET,
    ].reshape(1, 1, 2, 12, 16)
    np.testing.assert_array_equal(actual[:, 0], expected)


def test_q4_k_tile16_unpack_rejects_out_feature_mismatch() -> None:
    raw = _raw_q4_k_bytes(experts=1, out_features=16, blocks_per_row=1)
    packed = repack_gguf_q4_k_tile16(raw)

    with pytest.raises(ValueError, match="out_features mismatch"):
        unpack_gguf_q4_k_tile16(packed.tiles, out_features=32)


def test_q4_k_tile16_lite_unpack_rejects_out_feature_mismatch() -> None:
    raw = _raw_q4_k_bytes(experts=1, out_features=16, blocks_per_row=1)
    packed = repack_gguf_q4_k_tile16_lite(raw)

    with pytest.raises(ValueError, match="out_features mismatch"):
        unpack_gguf_q4_k_tile16_lite(packed.tiles, out_features=32)


def test_q4_k_tile16_repack_validates_shape() -> None:
    with pytest.raises(ValueError, match=r"\[experts, out_features, bytes_per_row\]"):
        repack_gguf_q4_k_tile16(np.zeros((16, GGUF_Q4_K_BLOCK_BYTES), dtype=np.uint8))
    with pytest.raises(ValueError, match="divisible by 16"):
        repack_gguf_q4_k_tile16(
            np.zeros((1, 15, GGUF_Q4_K_BLOCK_BYTES), dtype=np.uint8)
        )
    with pytest.raises(ValueError, match="multiple of 144"):
        repack_gguf_q4_k_tile16(np.zeros((1, 16, 145), dtype=np.uint8))
