from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipengine.generation.qwen35_gguf import make_qwen35_gguf_bringup_generator_gfx1151
from hipengine.generation.registry import resolve_text_generator
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_F32,
    LAYOUT_ROCMFP4_DENSE_BF16,
    plan_qwen35_gguf_weight_spec,
    planned_qwen35_gguf_weight_allocation_nbytes,
)
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    LlamaFileType,
    bf16_to_float32,
    dequantization_supported,
    dequantize_gguf_data,
    llama_file_type_name,
    quant_layout,
)
from hipengine.quant.registry import resolve_quant
from hipengine.runtime.gguf_linear import (
    _dense_bf16_wmma_prefill_is_qualified,
    resolve_gguf_linear_dispatch,
)


def _pack_fp6(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint8).reshape(-1, 4)
    packed = np.empty((values.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = (values[:, 0] & 0x3F) | ((values[:, 1] & 0x03) << 6)
    packed[:, 1] = ((values[:, 1] >> 2) & 0x0F) | ((values[:, 2] & 0x0F) << 4)
    packed[:, 2] = ((values[:, 2] >> 4) & 0x03) | ((values[:, 3] & 0x3F) << 2)
    return packed.reshape(-1)


def _tensor(qtype: GGMLQuantizationType) -> GGUFTensorInfo:
    shape = (128, 64)
    layout = quant_layout(qtype)
    n_elements = int(np.prod(shape))
    nbytes = n_elements // layout.block_size * layout.type_size
    return GGUFTensorInfo(
        name="blk.0.attn_output.weight",
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=n_elements,
        nbytes=nbytes,
        offset=0,
        data_offset=4096,
        byte_shape=(shape[0], shape[1] // layout.block_size * layout.type_size),
    )


def test_rocmfpx_custom_gguf_ids_and_product_file_type_are_stable() -> None:
    assert int(GGMLQuantizationType.Q4_0_ROCMFP4) == 100
    assert int(GGMLQuantizationType.Q6_0_ROCMFPX) == 102
    assert quant_layout(GGMLQuantizationType.Q4_0_ROCMFP4).block_size == 32
    assert quant_layout(GGMLQuantizationType.Q4_0_ROCMFP4).type_size == 18
    assert quant_layout(GGMLQuantizationType.Q6_0_ROCMFPX).block_size == 32
    assert quant_layout(GGMLQuantizationType.Q6_0_ROCMFPX).type_size == 26
    assert int(LlamaFileType.MOSTLY_Q4_0_ROCMFP4_COHERENT) == 102
    assert llama_file_type_name(102) == "MOSTLY_Q4_0_ROCMFP4_COHERENT"


def test_rocmfp4_dual_ue4m3_half_scale_dequant_matches_release_reference() -> None:
    low_codes = np.arange(16, dtype=np.uint8)
    high_codes = np.arange(15, -1, -1, dtype=np.uint8)
    packed = low_codes | (high_codes << np.uint8(4))
    block = np.concatenate([packed, np.asarray([0x38, 0x40], dtype=np.uint8)])

    actual = dequantize_gguf_data(
        block.reshape(1, 18), GGMLQuantizationType.Q4_0_ROCMFP4
    )
    codebook = np.asarray(
        [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10],
        dtype=np.float32,
    )
    expected = np.concatenate([codebook[low_codes] * 0.5, codebook[high_codes]])

    assert dequantization_supported(GGMLQuantizationType.Q4_0_ROCMFP4)
    np.testing.assert_array_equal(actual, expected.reshape(1, 32))


def test_rocmfpx_fp6_bitstream_and_signed_magnitude_dequant_match_release_reference() -> None:
    codes = np.arange(64, dtype=np.uint8).reshape(2, 32)
    blocks = np.concatenate(
        [
            np.stack([_pack_fp6(row) for row in codes]),
            np.tile(np.asarray([0x38, 0x40], dtype=np.uint8), (2, 1)),
        ],
        axis=1,
    )

    actual = dequantize_gguf_data(blocks, GGMLQuantizationType.Q6_0_ROCMFPX)
    decoded = (codes & np.uint8(31)).astype(np.int16)
    negative = (codes & np.uint8(32)) != 0
    decoded[negative] *= -1
    decoded[codes == np.uint8(32)] = -32
    expected = decoded.astype(np.float32)
    expected[:, :16] *= 0.5

    assert dequantization_supported(GGMLQuantizationType.Q6_0_ROCMFPX)
    np.testing.assert_array_equal(actual, expected)


def test_every_finite_rocmfp4_product_is_exactly_representable_in_bf16() -> None:
    packed = np.tile(
        np.arange(16, dtype=np.uint8)
        | (np.arange(15, -1, -1, dtype=np.uint8) << np.uint8(4)),
        (127, 1),
    )
    scales = np.repeat(np.arange(127, dtype=np.uint8)[:, None], 2, axis=1)
    values = dequantize_gguf_data(
        np.concatenate([packed, scales], axis=1),
        GGMLQuantizationType.Q4_0_ROCMFP4,
    )

    np.testing.assert_array_equal(
        bf16_to_float32(float_array_to_bf16_bits(values)),
        values,
    )


def test_rocmfpx_authority_materialization_is_lossless_for_q4_and_f32_for_q6() -> None:
    q4 = plan_qwen35_gguf_weight_spec(
        "layers.0.attn_output",
        _tensor(GGMLQuantizationType.Q4_0_ROCMFP4),
        decode_repack=True,
    )
    q6 = plan_qwen35_gguf_weight_spec(
        "layers.0.attn_output",
        _tensor(GGMLQuantizationType.Q6_0_ROCMFPX),
        decode_repack=True,
    )

    assert q4.quant_key == "gguf_q4_0_rocmfp4"
    assert q4.layout == LAYOUT_ROCMFP4_DENSE_BF16
    assert planned_qwen35_gguf_weight_allocation_nbytes(q4) == (("raw", 128 * 64 * 2),)
    assert q6.quant_key == "gguf_q6_0_rocmfpx"
    assert q6.layout == LAYOUT_DENSE_F32
    assert planned_qwen35_gguf_weight_allocation_nbytes(q6) == (("raw", 128 * 64 * 4),)

    q4_plugin = resolve_quant(q4.quant_key)
    q6_plugin = resolve_quant(q6.quant_key)
    assert q4_plugin.kernel_family == "dense_bf16_authority_fallback"
    assert q6_plugin.kernel_family == "dense_f32_authority_fallback"

    q4_dispatch = resolve_gguf_linear_dispatch(
        SimpleNamespace(spec=q4, backend="hip_gfx1151"), rows=1
    )
    q6_dispatch = resolve_gguf_linear_dispatch(
        SimpleNamespace(spec=q6, backend="hip_gfx1151"), rows=1
    )
    assert (
        q4_dispatch.key.backend,
        q4_dispatch.key.layer,
        q4_dispatch.key.quant,
        q4_dispatch.key.variant,
    ) == ("hip_gfx1151", "dense_gemv", "bf16", "out")
    assert _dense_bf16_wmma_prefill_is_qualified(
        SimpleNamespace(spec=q4, backend="hip_gfx1151"),
        backend="hip_gfx1151",
        rows=512,
        in_features=5_120,
        out_features=17_408,
    )
    assert not _dense_bf16_wmma_prefill_is_qualified(
        SimpleNamespace(spec=q4, backend="hip_gfx1151"),
        backend="hip_gfx1151",
        rows=96,
        in_features=6_144,
        out_features=5_120,
    )
    assert (
        q6_dispatch.key.backend,
        q6_dispatch.key.layer,
        q6_dispatch.key.quant,
        q6_dispatch.key.variant,
    ) == (
        "hip_gfx1151",
        "dense_gemv",
        "f32",
        "bf16_hidden_bf16_out",
    )


def test_rocmfp4_coherent_product_registers_through_the_existing_qwen_generator() -> None:
    assert resolve_text_generator(
        model="qwen3_5_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_0_rocmfp4_coherent",
    ) is make_qwen35_gguf_bringup_generator_gfx1151
