from __future__ import annotations

import numpy as np
import pytest

from hipengine.quant.gguf import (
    GGMLQuantizationType,
    bf16_to_float32,
    dequantization_supported,
    dequantize_gguf_data,
    nbytes_for_shape,
    quant_layout,
    quant_shape_from_byte_shape,
    quant_shape_to_byte_shape,
)


def _f16_bytes(value: float) -> np.ndarray:
    return np.asarray([value], dtype=np.float16).view(np.uint8)


def test_gguf_quant_layout_sizes_match_ggml_block_contracts() -> None:
    assert quant_layout(GGMLQuantizationType.BF16).type_size == 2
    assert quant_layout(GGMLQuantizationType.Q8_0).block_size == 32
    assert quant_layout(GGMLQuantizationType.Q8_0).type_size == 34
    assert quant_layout(GGMLQuantizationType.Q4_1).type_size == 20
    assert quant_layout(GGMLQuantizationType.Q4_K).block_size == 256
    assert quant_layout(GGMLQuantizationType.Q4_K).type_size == 144
    assert quant_layout(GGMLQuantizationType.Q5_K).type_size == 176
    assert quant_layout(GGMLQuantizationType.Q6_K).type_size == 210
    assert quant_layout(GGMLQuantizationType.IQ4_XS).type_size == 136
    assert quant_layout(GGMLQuantizationType.MXFP4).type_size == 17

    assert nbytes_for_shape((3, 64), GGMLQuantizationType.Q8_0) == 3 * 2 * 34
    assert quant_shape_to_byte_shape((3, 64), GGMLQuantizationType.Q8_0) == (3, 68)
    assert quant_shape_from_byte_shape((3, 68), GGMLQuantizationType.Q8_0) == (3, 64)

    with pytest.raises(ValueError, match="row size 63"):
        quant_shape_to_byte_shape((3, 63), GGMLQuantizationType.Q8_0)


def test_bf16_and_q8_0_dequantize_to_float32() -> None:
    bits = np.asarray([0x3F80, 0xC020, 0x3F00], dtype=np.uint16)
    np.testing.assert_allclose(
        bf16_to_float32(bits), np.asarray([1.0, -2.5, 0.5], dtype=np.float32)
    )

    block = np.concatenate([_f16_bytes(2.0), np.arange(-16, 16, dtype=np.int8).view(np.uint8)])
    out = dequantize_gguf_data(block.reshape(1, 34), GGMLQuantizationType.Q8_0)

    assert out.dtype == np.float32
    assert out.shape == (1, 32)
    np.testing.assert_allclose(out[0], np.arange(-16, 16, dtype=np.float32) * 2.0)


def test_q4_1_dequantizes_one_block() -> None:
    q = (np.arange(32, dtype=np.uint8) % np.uint8(16)).reshape(2, 16)
    packed = q[0] | (q[1] << np.uint8(4))
    block = np.concatenate([_f16_bytes(0.5), _f16_bytes(-1.0), packed]).reshape(1, 20)

    out = dequantize_gguf_data(block, GGMLQuantizationType.Q4_1)

    assert out.shape == (1, 32)
    expected = (np.arange(32, dtype=np.float32) % 16) * 0.5 - 1.0
    np.testing.assert_allclose(out[0], expected)


def test_q4_k_dequantizes_one_superblock_scale_group() -> None:
    scales = np.zeros(12, dtype=np.uint8)
    scales[0] = 1  # first 32-value group uses scale 1, min 0
    qs = np.zeros(128, dtype=np.uint8)
    qs[:32] = np.arange(32, dtype=np.uint8) & np.uint8(0x0F)
    block = np.concatenate([_f16_bytes(2.0), _f16_bytes(0.0), scales, qs]).reshape(1, 144)

    out = dequantize_gguf_data(block, GGMLQuantizationType.Q4_K)

    assert out.shape == (1, 256)
    expected = (np.arange(32, dtype=np.uint8) & np.uint8(0x0F)).astype(np.float32) * 2.0
    np.testing.assert_allclose(out[0, :32], expected)
    np.testing.assert_allclose(out[0, 32:], 0.0)


def test_target_local_model_tensor_types_have_fallback_dequant_support() -> None:
    for qtype in (
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q4_1,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        GGMLQuantizationType.IQ4_XS,
        GGMLQuantizationType.MXFP4,
    ):
        assert dequantization_supported(qtype), qtype.name
