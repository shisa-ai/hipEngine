from __future__ import annotations

import numpy as np
import pytest

from hipengine.loading import (
    paro_marlin_k_pack8_decode_view,
    repack_paro_awq_to_marlin_k_host,
)


def test_repack_paro_awq_to_marlin_k_host_matches_v0_layout() -> None:
    group_size = 128
    groups = 2
    out_packed = 3
    qweight = np.arange(groups * group_size * out_packed, dtype=np.int32).reshape(groups * group_size, out_packed)
    qzeros = (1000 + np.arange(groups * out_packed, dtype=np.int32)).reshape(groups, out_packed)
    scales = (np.arange(groups * out_packed * 8, dtype=np.float16).reshape(groups, out_packed * 8) / np.float16(16.0))

    qweight_mk, qzeros_mk, scales_mk = repack_paro_awq_to_marlin_k_host(
        qweight,
        qzeros,
        scales,
        bits=4,
        group_size=group_size,
    )

    assert qweight_mk.shape == (out_packed, groups, group_size)
    assert qzeros_mk.shape == (out_packed, groups)
    assert scales_mk.shape == (out_packed, groups, 8)
    assert qweight_mk.dtype == np.int32
    assert qzeros_mk.dtype == np.int32
    assert scales_mk.dtype == np.float16
    assert qweight_mk.flags.c_contiguous
    assert qzeros_mk.flags.c_contiguous
    assert scales_mk.flags.c_contiguous
    np.testing.assert_array_equal(qweight_mk, qweight.reshape(groups, group_size, out_packed).transpose(2, 0, 1))
    np.testing.assert_array_equal(qzeros_mk, qzeros.T)
    np.testing.assert_array_equal(scales_mk, scales.reshape(groups, out_packed, 8).transpose(1, 0, 2))


def test_paro_marlin_k_pack8_decode_view_aliases_qweight_mk() -> None:
    qweight = np.arange(2 * 128 * 3, dtype=np.int32).reshape(2 * 128, 3)
    qzeros = np.zeros((2, 3), dtype=np.int32)
    scales = np.zeros((2, 24), dtype=np.float16)
    qweight_mk, _, _ = repack_paro_awq_to_marlin_k_host(qweight, qzeros, scales)

    pack8_view = paro_marlin_k_pack8_decode_view(qweight_mk)

    assert pack8_view.shape == (3, 256)
    assert pack8_view.dtype == np.int32
    assert pack8_view.flags.c_contiguous
    assert np.shares_memory(pack8_view, qweight_mk)
    np.testing.assert_array_equal(pack8_view, qweight_mk.reshape(3, 256))


def test_repack_paro_awq_to_marlin_k_host_rejects_shape_mismatches() -> None:
    qweight = np.zeros((127, 1), dtype=np.int32)
    qzeros = np.zeros((1, 1), dtype=np.int32)
    scales = np.zeros((1, 8), dtype=np.float16)

    with pytest.raises(ValueError, match="multiple of group_size"):
        repack_paro_awq_to_marlin_k_host(qweight, qzeros, scales)

    qweight = np.zeros((128, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="qzeros shape"):
        repack_paro_awq_to_marlin_k_host(qweight, qzeros, np.zeros((1, 16), dtype=np.float16))

    with pytest.raises(ValueError, match="scales shape"):
        repack_paro_awq_to_marlin_k_host(qweight, np.zeros((1, 2), dtype=np.int32), scales)

    with pytest.raises(ValueError, match="bits=4"):
        repack_paro_awq_to_marlin_k_host(qweight, np.zeros((1, 2), dtype=np.int32), np.zeros((1, 16), dtype=np.float16), bits=8)
