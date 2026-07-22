"""IQ3_XXS / Q3_K CPU dequantizer tests (RED-first for UD-Q3_K_M support).

Covers the two new GGUF block formats needed by the
``Qwen3.6-35B-A3B-UD-Q3_K_M`` expert tensors:

* ``IQ3_XXS`` (98-byte blocks): 2-byte fp16 super scale + 64 grid indices +
  32 sign/scale bytes (8 u32 aux words). Reference: llama.cpp
  ``dequantize_row_iq3_xxs``.
* ``Q3_K`` (110-byte blocks): 32-byte hmask + 64-byte 2-bit quants + 12-byte
  packed 6-bit scales + 2-byte fp16 super scale. Reference: llama.cpp
  ``dequantize_row_q3_K``.

The synthetic cases below are hand-computed from the ggml block contracts so
they pin field order and sign/scale mechanics without re-implementing the
dequantizer. ``test_gguf_iq_dequant_oracle.py`` additionally validates real
model rows against a committed llama.cpp-generated fixture.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

import hipengine.quant.gguf as gguf_quant
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    dequantization_supported,
    dequantize_gguf_data,
    nbytes_for_shape,
    quant_layout,
)


def _f16_bytes(value: float) -> np.ndarray:
    return np.asarray([value], dtype=np.float16).view(np.uint8)


def test_iq2_xs_layout_matches_ggml_block_contract() -> None:
    layout = quant_layout(GGMLQuantizationType.IQ2_XS)
    assert layout.block_size == 256
    assert layout.type_size == 74
    assert nbytes_for_shape((3, 512), GGMLQuantizationType.IQ2_XS) == 3 * 2 * 74
    assert dequantization_supported(GGMLQuantizationType.IQ2_XS)


def test_iq2_xs_compressed_grid_matches_llamacpp_1ebf790cd() -> None:
    grid_bytes = np.asarray(gguf_quant._IQ2_XS_GRID_PACKED, dtype="<u2").tobytes()
    assert hashlib.sha256(grid_bytes).hexdigest() == (
        "f66435966ca8d64e9e4a6e3c91e64afcf9015e700ec85962f527283ebae207b9"
    )


def test_iq2_xs_matches_pinned_llamacpp_gguf_py_oracle() -> None:
    fixture = Path(__file__).parent / "fixtures" / "gguf" / "iq2_xs_dequant_oracle.json"
    payload = json.loads(fixture.read_text())
    raw = np.frombuffer(base64.b64decode(payload["block_bytes_b64"]), dtype=np.uint8)
    expected_bytes = base64.b64decode(payload["expected_f32_b64"])
    expected = np.frombuffer(expected_bytes, dtype="<f4")
    assert hashlib.sha256(raw).hexdigest() == payload["block_sha256"]
    assert hashlib.sha256(expected_bytes).hexdigest() == payload["expected_sha256"]
    actual = dequantize_gguf_data(raw.reshape(1, 74), GGMLQuantizationType.IQ2_XS)
    np.testing.assert_array_equal(actual.reshape(-1), expected)


def _iq2_xs_block(d: float, qs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    qs_bytes = np.asarray(qs, dtype="<u2").reshape(32).view(np.uint8)
    scales = np.asarray(scales, dtype=np.uint8).reshape(8)
    return np.concatenate([_f16_bytes(d), qs_bytes, scales]).reshape(1, 74)


def test_iq2_xs_dequantizes_base_grid() -> None:
    block = _iq2_xs_block(
        2.0,
        np.zeros(32, dtype=np.uint16),
        np.zeros(8, dtype=np.uint8),
    )
    out = dequantize_gguf_data(block, GGMLQuantizationType.IQ2_XS)
    assert out.shape == (1, 256)
    # grid[0] is eight 0x08 magnitudes and db = 2 * 0.5 * 0.25.
    np.testing.assert_array_equal(out[0], np.full(256, 2.0, dtype=np.float32))


def test_iq2_xs_grid_and_sign_selectors() -> None:
    qs = np.zeros(32, dtype=np.uint16)
    qs[0] = np.uint16(1)  # grid[1] starts with 0x2b, then seven 0x08 values.
    qs[1] = np.uint16(1 << 9)  # sign selector 1 -> sign bits 0 and 7.
    block = _iq2_xs_block(2.0, qs, np.zeros(8, dtype=np.uint8))
    out = dequantize_gguf_data(block, GGMLQuantizationType.IQ2_XS)
    expected = np.full(256, 2.0, dtype=np.float32)
    expected[0] = 10.75
    expected[8] = -2.0
    expected[15] = -2.0
    np.testing.assert_array_equal(out[0], expected)


def test_iq2_xs_scale_nibbles_and_later_groups_are_independent() -> None:
    scales = np.zeros(8, dtype=np.uint8)
    scales[0] = np.uint8(0x53)
    scales[7] = np.uint8(0x21)
    block = _iq2_xs_block(2.0, np.zeros(32, dtype=np.uint16), scales)
    out = dequantize_gguf_data(block, GGMLQuantizationType.IQ2_XS)
    expected = np.full(256, 2.0, dtype=np.float32)
    expected[0:16] = 14.0  # low nibble 3: db = 1.75; grid magnitude 8.
    expected[16:32] = 22.0  # high nibble 5: db = 2.75.
    expected[224:240] = 6.0  # low nibble 1: db = 0.75.
    expected[240:256] = 10.0  # high nibble 2: db = 1.25.
    np.testing.assert_array_equal(out[0], expected)


def test_iq3_xxs_layout_matches_ggml_block_contract() -> None:
    layout = quant_layout(GGMLQuantizationType.IQ3_XXS)
    assert layout.block_size == 256
    assert layout.type_size == 98
    assert nbytes_for_shape((3, 512), GGMLQuantizationType.IQ3_XXS) == 3 * 2 * 98


def test_q3_k_layout_matches_ggml_block_contract() -> None:
    layout = quant_layout(GGMLQuantizationType.Q3_K)
    assert layout.block_size == 256
    assert layout.type_size == 110
    assert nbytes_for_shape((3, 512), GGMLQuantizationType.Q3_K) == 3 * 2 * 110


def test_ud_q3_k_m_expert_types_have_dequant_support() -> None:
    for qtype in (
        GGMLQuantizationType.IQ3_XXS,
        GGMLQuantizationType.IQ4_XS,
        GGMLQuantizationType.Q3_K,
    ):
        assert dequantization_supported(qtype), qtype.name


def test_iq3_xxs_tables_match_llamacpp_1ebf790cd() -> None:
    grid_bytes = np.asarray(gguf_quant._IQ3_XXS_GRID, dtype="<u4").tobytes()
    sign_bytes = np.asarray(gguf_quant._KSIGNS_IQ2XS, dtype=np.uint8).tobytes()

    assert hashlib.sha256(grid_bytes).hexdigest() == (
        "46e35f5a997efdee6c99ce57854c8a0d4f0ff8ca57e5e8a60c0793ea580acf5d"
    )
    assert hashlib.sha256(sign_bytes).hexdigest() == (
        "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31"
    )


def _iq3_xxs_block(d: float, grid_idx: np.ndarray, aux32: np.ndarray) -> np.ndarray:
    """Assemble one 98-byte IQ3_XXS block from parts.

    ``grid_idx`` is 64 bytes (8 ib32 groups x 4 l-pairs x 2 indices) and
    ``aux32`` is 8 little-endian u32 words (sign nibbles + 4-bit scale).
    """

    grid_idx = np.asarray(grid_idx, dtype=np.uint8).reshape(64)
    aux = np.asarray(aux32, dtype=np.uint32).reshape(8).view(np.uint8)
    return np.concatenate([_f16_bytes(d), grid_idx, aux]).reshape(1, 98)


def test_iq3_xxs_dequantizes_zero_block() -> None:
    # grid index 0 -> iq3xxs_grid[0] = 0x04040404 -> four +4 magnitudes,
    # aux32 = 0 -> signs all positive, sub-scale 0 -> db = d * 0.5 * 0.5.
    block = _iq3_xxs_block(2.0, np.zeros(64, dtype=np.uint8), np.zeros(8, dtype=np.uint32))
    out = dequantize_gguf_data(block, GGMLQuantizationType.IQ3_XXS)
    assert out.shape == (1, 256)
    np.testing.assert_allclose(out[0], np.full(256, 4.0 * 0.5, dtype=np.float32))


def test_iq3_xxs_grid_index_selects_second_grid_row() -> None:
    grid_idx = np.zeros(64, dtype=np.uint8)
    grid_idx[1] = 1  # l=0 second grid: 0x04040414 -> bytes (0x14, 4, 4, 4)
    block = _iq3_xxs_block(2.0, grid_idx, np.zeros(8, dtype=np.uint32))
    out = dequantize_gguf_data(block, GGMLQuantizationType.IQ3_XXS)
    expected = np.full(256, 2.0, dtype=np.float32)
    expected[4] = 0x14 * 0.5  # 20 * db
    np.testing.assert_allclose(out[0], expected)


def test_iq3_xxs_sign_byte_negates_group() -> None:
    aux32 = np.zeros(8, dtype=np.uint32)
    aux32[0] = 0x7F  # l=0 sign selector 127 -> ksigns 0xFF -> all 8 negative
    block = _iq3_xxs_block(2.0, np.zeros(64, dtype=np.uint8), aux32)
    out = dequantize_gguf_data(block, GGMLQuantizationType.IQ3_XXS)
    expected = np.full(256, 2.0, dtype=np.float32)
    expected[:8] = -2.0
    np.testing.assert_allclose(out[0], expected)


def test_iq3_xxs_subscale_nibble_scales_group() -> None:
    aux32 = np.zeros(8, dtype=np.uint32)
    aux32[0] = 0x30000000  # top nibble 3 -> db = d * (0.5 + 3) * 0.5
    block = _iq3_xxs_block(2.0, np.zeros(64, dtype=np.uint8), aux32)
    out = dequantize_gguf_data(block, GGMLQuantizationType.IQ3_XXS)
    expected = np.full(256, 2.0, dtype=np.float32)
    expected[:32] = 4.0 * 3.5
    np.testing.assert_allclose(out[0], expected)


def test_iq3_xxs_later_ib32_groups_are_independent() -> None:
    # Only ib32=7 (bytes 56..63 of grid area, aux word 7) differs.
    grid_idx = np.zeros(64, dtype=np.uint8)
    grid_idx[56] = 2  # l=0 first grid of last ib32: 0x04040424 -> (0x24,4,4,4)
    aux32 = np.zeros(8, dtype=np.uint32)
    aux32[7] = 0x10000000
    block = _iq3_xxs_block(2.0, grid_idx, aux32)
    out = dequantize_gguf_data(block, GGMLQuantizationType.IQ3_XXS)
    expected = np.full(256, 2.0, dtype=np.float32)
    db7 = 2.0 * (0.5 + 1) * 0.5
    expected[224] = 0x24 * db7
    expected[225:256] = 4.0 * db7
    np.testing.assert_allclose(out[0], expected)


def _q3_k_block(hmask: np.ndarray, qs: np.ndarray, scales: np.ndarray, d: float) -> np.ndarray:
    hmask = np.asarray(hmask, dtype=np.uint8).reshape(32)
    qs = np.asarray(qs, dtype=np.uint8).reshape(64)
    scales = np.asarray(scales, dtype=np.uint8).reshape(12)
    return np.concatenate([hmask, qs, scales, _f16_bytes(d)]).reshape(1, 110)


def test_q3_k_dequantizes_zero_block() -> None:
    block = _q3_k_block(
        np.full(32, 0xFF, dtype=np.uint8),
        np.zeros(64, dtype=np.uint8),
        np.zeros(12, dtype=np.uint8),
        4.0,
    )
    out = dequantize_gguf_data(block, GGMLQuantizationType.Q3_K)
    assert out.shape == (1, 256)
    np.testing.assert_allclose(out[0], np.zeros(256, dtype=np.float32))


def test_q3_k_scale_unpack_and_first_subblock() -> None:
    # scale0 = (scales[0] & 0x0F) | ((scales[8] & 0x03) << 4) = 1 | 32 = 33
    # dl0 = d * (33 - 32) = 4.0; every other scale is 0 -> dl = 4.0 * -32.
    scales = np.zeros(12, dtype=np.uint8)
    scales[0] = 1
    scales[8] = 2
    qs = np.zeros(64, dtype=np.uint8)
    qs[0] = 1  # (j=0, shift 0) first half, seg 0, lane 0 -> y[0]
    qs[1] = 2  # y[1]
    qs[16] = 1  # seg 1 lane 0 -> y[16], scale1 = 0 -> dl = -128
    block = _q3_k_block(np.full(32, 0xFF, dtype=np.uint8), qs, scales, 4.0)
    out = dequantize_gguf_data(block, GGMLQuantizationType.Q3_K)
    expected = np.zeros(256, dtype=np.float32)
    expected[0] = 4.0
    expected[1] = 8.0
    expected[16] = -128.0
    np.testing.assert_allclose(out[0], expected)


def test_q3_k_hmask_zero_subtracts_four() -> None:
    scales = np.zeros(12, dtype=np.uint8)
    scales[0] = 1
    scales[8] = 2  # scale0 = 33 -> dl0 = 4.0
    hmask = np.full(32, 0xFF, dtype=np.uint8)
    hmask[2] = 0xFE  # bit 0 clear -> j=0 lane 2 loses the high bit -> -4
    block = _q3_k_block(hmask, np.zeros(64, dtype=np.uint8), scales, 4.0)
    out = dequantize_gguf_data(block, GGMLQuantizationType.Q3_K)
    expected = np.zeros(256, dtype=np.float32)
    expected[2] = 4.0 * (0 - 4)
    np.testing.assert_allclose(out[0], expected)


def test_q3_k_two_bit_shift_groups() -> None:
    # qs[0] = 0b10000 -> j=2 (shift 4) sees value 1; scale4 = 0 -> dl = -128.
    qs = np.zeros(64, dtype=np.uint8)
    qs[0] = 0x10
    block = _q3_k_block(np.full(32, 0xFF, dtype=np.uint8), qs, np.zeros(12, dtype=np.uint8), 4.0)
    out = dequantize_gguf_data(block, GGMLQuantizationType.Q3_K)
    expected = np.zeros(256, dtype=np.float32)
    expected[64] = -128.0
    np.testing.assert_allclose(out[0], expected)


def test_q3_k_second_half_uses_scales_8_to_15() -> None:
    # scale8 = ((scales[0] >> 4) & 0x0F) | ((scales[8] >> 4 & 0x03) << 4) = 1 -> dl8 = 4.0*-31
    scales = np.zeros(12, dtype=np.uint8)
    scales[0] = 0x10
    qs = np.zeros(64, dtype=np.uint8)
    qs[32] = 1  # second half, j=0, seg 0, lane 0 -> y[128]
    block = _q3_k_block(np.full(32, 0xFF, dtype=np.uint8), qs, scales, 4.0)
    out = dequantize_gguf_data(block, GGMLQuantizationType.Q3_K)
    expected = np.zeros(256, dtype=np.float32)
    expected[128] = 4.0 * (1 - 32)
    np.testing.assert_allclose(out[0], expected)


def test_q3_k_second_half_hmask_uses_high_nibble_bits() -> None:
    # h=1, j=0 -> mask bit 16 on hm[0]; clear it -> -4 with dl8.
    scales = np.zeros(12, dtype=np.uint8)
    scales[0] = 0x10  # scale8 = 1 -> dl8 = -124
    hmask = np.full(32, 0xFF, dtype=np.uint8)
    hmask[0] = 0xEF  # bit 4 clear
    block = _q3_k_block(hmask, np.zeros(64, dtype=np.uint8), scales, 4.0)
    out = dequantize_gguf_data(block, GGMLQuantizationType.Q3_K)
    expected = np.zeros(256, dtype=np.float32)
    expected[128] = 4.0 * (1 - 32) * (0 - 4)
    np.testing.assert_allclose(out[0], expected)
