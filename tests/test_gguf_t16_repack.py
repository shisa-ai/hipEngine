from __future__ import annotations

import numpy as np
import pytest

from hipengine.quant import resolve_quant
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.quant.gguf_t16 import (
    GGUF_Q5_K_BLOCK_BYTES,
    GGUF_Q5_K_T16_BLOCK_BYTES,
    GGUF_Q6_K_BLOCK_BYTES,
    GGUF_Q6_K_T16_BLOCK_BYTES,
    GGUF_Q8_0_BLOCK_BYTES,
    GGUF_Q8_0_T16_BLOCK_BYTES,
    GGUF_T16_COLS,
    repack_gguf_q5_k_tile16,
    repack_gguf_q6_k_tile16,
    repack_gguf_q8_0_tile16,
    unpack_gguf_q5_k_tile16,
    unpack_gguf_q6_k_tile16,
    unpack_gguf_q8_0_tile16,
)


def _finite_fp16_bytes(value: float, shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, value, dtype=np.float16).view(np.uint8).reshape(*shape, 2)


def _raw_q5_k_bytes(*, experts: int, out_features: int, blocks_per_row: int) -> np.ndarray:
    rng = np.random.default_rng(5000 + experts * 13 + out_features * 7 + blocks_per_row)
    raw = rng.integers(
        0,
        256,
        size=(experts, out_features, blocks_per_row * GGUF_Q5_K_BLOCK_BYTES),
        dtype=np.uint8,
    )
    blocks = raw.reshape(experts, out_features, blocks_per_row, GGUF_Q5_K_BLOCK_BYTES)
    blocks[..., 0:2] = _finite_fp16_bytes(0.125, (experts, out_features, blocks_per_row))
    blocks[..., 2:4] = _finite_fp16_bytes(0.25, (experts, out_features, blocks_per_row))
    return raw


def _raw_q6_k_bytes(*, experts: int, out_features: int, blocks_per_row: int) -> np.ndarray:
    rng = np.random.default_rng(6000 + experts * 17 + out_features * 5 + blocks_per_row)
    raw = rng.integers(
        0,
        256,
        size=(experts, out_features, blocks_per_row * GGUF_Q6_K_BLOCK_BYTES),
        dtype=np.uint8,
    )
    blocks = raw.reshape(experts, out_features, blocks_per_row, GGUF_Q6_K_BLOCK_BYTES)
    blocks[..., 208:210] = _finite_fp16_bytes(0.125, (experts, out_features, blocks_per_row))
    return raw


def _raw_q8_0_bytes(*, out_features: int, blocks_per_row: int) -> np.ndarray:
    rng = np.random.default_rng(8000 + out_features * 3 + blocks_per_row)
    raw = rng.integers(
        0,
        256,
        size=(out_features, blocks_per_row * GGUF_Q8_0_BLOCK_BYTES),
        dtype=np.uint8,
    )
    blocks = raw.reshape(out_features, blocks_per_row, GGUF_Q8_0_BLOCK_BYTES)
    blocks[..., 0:2] = _finite_fp16_bytes(0.125, (out_features, blocks_per_row))
    return raw


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("gguf_q4_k_t16_v1", "gguf_t16_gemv"),
        ("gguf_q5_k_t16_v1", "gguf_t16_gemv"),
        ("gguf_q6_k_t16_v1", "gguf_t16_gemv"),
        ("gguf_q8_0_t16_v1", "gguf_t16_gemv"),
    ],
)
def test_t16_quant_keys_are_registered(name: str, family: str) -> None:
    plugin = resolve_quant(name)

    assert plugin.name == name
    assert plugin.kernel_family == family
    assert plugin.weight_storage.endswith("t16_v1")


def test_q5_k_tile16_roundtrips_raw_bytes_and_dequantizes_equivalently() -> None:
    raw = _raw_q5_k_bytes(experts=3, out_features=32, blocks_per_row=2)

    packed = repack_gguf_q5_k_tile16(raw)
    restored = unpack_gguf_q5_k_tile16(packed)

    assert GGUF_Q5_K_T16_BLOCK_BYTES == 2880
    assert packed.tiles.shape == (3, 2, 2, GGUF_Q5_K_T16_BLOCK_BYTES)
    assert packed.in_features == 512
    np.testing.assert_array_equal(restored, raw)
    np.testing.assert_allclose(
        dequantize_gguf_data(restored.reshape(-1, 2 * GGUF_Q5_K_BLOCK_BYTES), GGMLQuantizationType.Q5_K),
        dequantize_gguf_data(raw.reshape(-1, 2 * GGUF_Q5_K_BLOCK_BYTES), GGMLQuantizationType.Q5_K),
        rtol=0,
        atol=0,
    )


def test_q6_k_tile16_roundtrips_raw_bytes_and_dequantizes_equivalently() -> None:
    raw = _raw_q6_k_bytes(experts=2, out_features=32, blocks_per_row=2)

    packed = repack_gguf_q6_k_tile16(raw)
    restored = unpack_gguf_q6_k_tile16(packed)

    assert GGUF_Q6_K_T16_BLOCK_BYTES == 3360
    assert packed.tiles.shape == (2, 2, 2, GGUF_Q6_K_T16_BLOCK_BYTES)
    assert packed.in_features == 512
    np.testing.assert_array_equal(restored, raw)
    np.testing.assert_allclose(
        dequantize_gguf_data(restored.reshape(-1, 2 * GGUF_Q6_K_BLOCK_BYTES), GGMLQuantizationType.Q6_K),
        dequantize_gguf_data(raw.reshape(-1, 2 * GGUF_Q6_K_BLOCK_BYTES), GGMLQuantizationType.Q6_K),
        rtol=0,
        atol=0,
    )


def test_q8_0_tile16_roundtrips_raw_bytes_and_dequantizes_equivalently() -> None:
    raw = _raw_q8_0_bytes(out_features=32, blocks_per_row=3)

    packed = repack_gguf_q8_0_tile16(raw)
    restored = unpack_gguf_q8_0_tile16(packed)

    assert GGUF_Q8_0_T16_BLOCK_BYTES == 544
    assert packed.tiles.shape == (2, 3, GGUF_Q8_0_T16_BLOCK_BYTES)
    assert packed.in_features == 96
    np.testing.assert_array_equal(restored, raw)
    np.testing.assert_allclose(
        dequantize_gguf_data(restored, GGMLQuantizationType.Q8_0),
        dequantize_gguf_data(raw, GGMLQuantizationType.Q8_0),
        rtol=0,
        atol=0,
    )


def test_t16_storage_overheads_match_design_doc() -> None:
    q5_raw = _raw_q5_k_bytes(experts=1, out_features=GGUF_T16_COLS, blocks_per_row=1)
    q6_raw = _raw_q6_k_bytes(experts=1, out_features=GGUF_T16_COLS, blocks_per_row=1)
    q8_raw = _raw_q8_0_bytes(out_features=GGUF_T16_COLS, blocks_per_row=1)

    assert repack_gguf_q5_k_tile16(q5_raw).tiles.nbytes / q5_raw.nbytes == pytest.approx(2880 / 2816)
    assert repack_gguf_q6_k_tile16(q6_raw).tiles.nbytes / q6_raw.nbytes == pytest.approx(1.0)
    assert repack_gguf_q8_0_tile16(q8_raw).tiles.nbytes / q8_raw.nbytes == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("repack", "bad_shape", "match"),
    [
        (repack_gguf_q5_k_tile16, np.zeros((16, GGUF_Q5_K_BLOCK_BYTES), dtype=np.uint8), "expert byte shape"),
        (repack_gguf_q5_k_tile16, np.zeros((1, 15, GGUF_Q5_K_BLOCK_BYTES), dtype=np.uint8), "divisible by 16"),
        (repack_gguf_q5_k_tile16, np.zeros((1, 16, GGUF_Q5_K_BLOCK_BYTES + 1), dtype=np.uint8), "multiple of 176"),
        (repack_gguf_q6_k_tile16, np.zeros((1, 16, GGUF_Q6_K_BLOCK_BYTES + 1), dtype=np.uint8), "multiple of 210"),
        (repack_gguf_q8_0_tile16, np.zeros((1, 16, GGUF_Q8_0_BLOCK_BYTES), dtype=np.uint8), "dense byte shape"),
        (repack_gguf_q8_0_tile16, np.zeros((15, GGUF_Q8_0_BLOCK_BYTES), dtype=np.uint8), "divisible by 16"),
        (repack_gguf_q8_0_tile16, np.zeros((16, GGUF_Q8_0_BLOCK_BYTES + 1), dtype=np.uint8), "multiple of 34"),
    ],
)
def test_t16_repack_validates_shapes(repack, bad_shape: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        repack(bad_shape)


@pytest.mark.parametrize(
    ("raw", "repack", "unpack"),
    [
        (_raw_q5_k_bytes(experts=1, out_features=16, blocks_per_row=1), repack_gguf_q5_k_tile16, unpack_gguf_q5_k_tile16),
        (_raw_q6_k_bytes(experts=1, out_features=16, blocks_per_row=1), repack_gguf_q6_k_tile16, unpack_gguf_q6_k_tile16),
        (_raw_q8_0_bytes(out_features=16, blocks_per_row=1), repack_gguf_q8_0_tile16, unpack_gguf_q8_0_tile16),
    ],
)
def test_t16_unpack_rejects_out_feature_mismatch(raw: np.ndarray, repack, unpack) -> None:
    packed = repack(raw)

    with pytest.raises(ValueError, match="out_features mismatch"):
        unpack(packed.tiles, out_features=32)
