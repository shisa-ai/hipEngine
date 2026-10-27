"""Dense Llama prefill attention: mask contract and argument guards, no device.

The mask is the only part of this module that is pure host code, and it is the
part a decode-shaped window depends on: a one-row query over a long key axis is
the same ``key <= query`` relation, evaluated at an offset. The launch guards run
before any HIP call, so a bad geometry is a ``ValueError`` on a CPU-only host
rather than a launch with an invalid shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.attention.dense_prefill_attention import (
    causal_keep_mask,
    dense_causal_prefill_bf16,
)


def test_mask_keeps_the_diagonal_and_below():
    mask = causal_keep_mask(3)
    assert mask.shape == (3, 3)
    assert mask.dtype == np.uint8
    assert np.array_equal(mask, np.tril(np.ones((3, 3), dtype=np.uint8)))


def test_mask_windows_over_a_longer_key_axis():
    # A decode-shaped window: one query row attending every cached key.
    assert np.array_equal(causal_keep_mask(1, keys=5), np.ones((1, 5), dtype=np.uint8))
    # A two-row window ending at key 5 sees keys 3 and 4 causally.
    assert np.array_equal(
        causal_keep_mask(2, keys=5),
        np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=np.uint8),
    )


def test_mask_rejects_empty_geometry():
    with pytest.raises(ValueError, match="positive"):
        causal_keep_mask(0)
    with pytest.raises(ValueError, match="positive"):
        causal_keep_mask(2, keys=0)


_VALID = {
    "tokens": 2,
    "num_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 8,
    "scale": 8.0**-0.5,
}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tokens": 0}, "tokens must be positive"),
        ({"num_heads": 0}, "num_heads and num_kv_heads must be positive"),
        ({"num_kv_heads": 0}, "num_heads and num_kv_heads must be positive"),
        ({"num_heads": 3}, "divisible by num_kv_heads"),
        ({"head_dim": 0}, "head_dim must be positive"),
        ({"scale": 0.0}, "scale must be finite and positive"),
        ({"scale": float("nan")}, "scale must be finite and positive"),
    ],
)
def test_launch_rejects_an_impossible_geometry(overrides, message):
    kwargs = dict(_VALID)
    kwargs.update(overrides)
    # Every guard precedes the HIP call, so null pointers never reach a launch.
    with pytest.raises(ValueError, match=message):
        dense_causal_prefill_bf16(0, 0, 0, 0, 0, **kwargs)
