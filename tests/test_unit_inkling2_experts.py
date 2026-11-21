import numpy as np
import pytest

from hipengine.quant.inkling2_experts import decode_mxfp8_expert, decode_nvfp4_expert


def test_nvfp4_reads_low_nibble_first_and_scales_each_block():
    packed = np.array([[0x21] * 8 + [0xFA] * 8, [0x87] * 16], dtype=np.uint8)
    scales = np.array([[0x38, 0x40], [0x30, 0x38]], dtype=np.uint8)
    got = decode_nvfp4_expert(packed, scales, amax=3, denominator=6)
    np.testing.assert_array_equal(got, [
        [0.25, 0.5] * 8 + [-1.0, -6.0] * 8,
        [1.5, -0.0] * 8 + [3.0, -0.0] * 8,
    ])
    assert np.signbit(got[1, 1])
    assert got.dtype == np.float32


def test_mxfp8_decodes_subnormals_signs_and_ue8m0_block_scale():
    weights = np.array([[0x01, 0xFE] * 16 + [0xB8, 0x38] * 16], dtype=np.uint8)
    scales = np.array([[126, 129]], dtype=np.uint8)
    got = decode_mxfp8_expert(weights, scales)
    np.testing.assert_array_equal(got, [[2**-10, -224.0] * 16 + [-4.0, 4.0] * 16])
    assert got.dtype == np.float32


def test_mxfp8_ue8m0_lowest_power_of_two_and_nan_encoding():
    weights = np.full((1, 64), 0x38, dtype=np.uint8)
    got = decode_mxfp8_expert(weights, np.array([[0, 255]], dtype=np.uint8))
    np.testing.assert_array_equal(got[0, :32], np.full(32, np.float32(2**-127)))
    assert np.isnan(got[0, 32:]).all()


@pytest.mark.parametrize("decode, options", [
    (decode_nvfp4_expert, {"amax": 1, "denominator": 1}),
    (decode_mxfp8_expert, {}),
])
def test_expert_decoder_rejects_incompatible_raw_layout(decode, options):
    with pytest.raises(ValueError, match="layout"):
        decode(np.zeros((1, 64), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), **options)
    with pytest.raises(ValueError, match="uint8"):
        decode(np.zeros((1, 64), dtype=np.int8), np.zeros((1, 1), dtype=np.uint8), **options)


def test_nvfp4_requires_valid_outer_scale():
    with pytest.raises(ValueError, match="outer scale"):
        decode_nvfp4_expert(np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8),
                            amax=1, denominator=0)
