from __future__ import annotations

import numpy as np
import pytest

from scripts.moonshine_decoder_smoke import (
    _certified_encoder_bucket,
    _pad_cross_reference,
    _pad_encoder_inputs,
)


def test_certified_encoder_bucket_uses_smallest_fitting_bucket() -> None:
    assert [_certified_encoder_bucket(value) for value in (24, 40, 42, 207, 208, 1248)] == [
        40,
        40,
        207,
        207,
        1248,
        1248,
    ]
    with pytest.raises(ValueError, match="exceeds certified buckets"):
        _certified_encoder_bucket(1249)


def test_pad_encoder_inputs_preserves_source_and_zero_masks_padding() -> None:
    hidden = np.arange(1 * 2 * 3, dtype=np.float16).reshape(1, 2, 3)
    mask = np.array([[1, 1]], dtype=np.int32)
    padded_hidden, padded_mask = _pad_encoder_inputs(hidden, mask, 4)

    assert padded_hidden.shape == (1, 4, 3)
    assert padded_hidden.dtype == np.float16
    assert np.array_equal(padded_hidden[:, :2], hidden)
    assert np.array_equal(padded_hidden[:, 2:], np.zeros((1, 2, 3), dtype=np.float16))
    assert np.array_equal(padded_mask, np.array([[1, 1, 0, 0]], dtype=np.int32))


def test_pad_cross_reference_preserves_head_major_prefix() -> None:
    expected = np.arange(1 * 2 * 3 * 4, dtype=np.float16).reshape(1, 2, 3, 4)
    padded = _pad_cross_reference(expected, 5)
    assert padded.shape == (1, 2, 5, 4)
    assert np.array_equal(padded[:, :, :3], expected)
    assert np.array_equal(padded[:, :, 3:], np.zeros((1, 2, 2, 4), dtype=np.float16))
