"""The decode read range of a sliding layer.

A sliding layer's keep-mask zeroes every key outside its window, so the kernel
walks keys whose weight is exactly zero. The walk is not free: the decode
kernel's cost tracks the key count, not the number of live keys, and 25 of
Gemma 4's 30 layers are sliding. Passing a shorter range is bit-exact - the
dropped terms contribute exp(-inf) = 0 and removing them does not reorder the
surviving terms - but only when the dropped keys are *exactly* the masked ones,
which is what these tests pin.
"""

from __future__ import annotations

import pytest

from hipengine.runtime.gemma4 import _sliding_read_range


class _Attention:
    def __init__(self, sliding_window):
        self.sliding_window = sliding_window


def test_dense_layer_never_skips():
    assert _sliding_read_range(_Attention(None), start=5000, rows=1) == 0


def test_window_wider_than_context_never_skips():
    assert _sliding_read_range(_Attention(1024), start=500, rows=1) == 0


def test_exactly_at_the_window_never_skips():
    # live = 1024, window = 1024: every live key is in the window.
    assert _sliding_read_range(_Attention(1024), start=1023, rows=1) == 0


def test_one_key_past_the_window_skips_exactly_one():
    # live = 1025: position 0 is 1025 back from the query at 1024, outside the
    # window, so the first live key is 1.
    assert _sliding_read_range(_Attention(1024), start=1024, rows=1) == 1


def test_long_context_skips_down_to_the_window():
    # live = 4097, window = 1024: keys 0..3072 are masked, the first live key is
    # 3073, leaving exactly 1024 keys to walk.
    key_begin = _sliding_read_range(_Attention(1024), start=4096, rows=1)
    assert key_begin == 4097 - 1024
    assert 4097 - key_begin == 1024


def test_the_skipped_keys_are_exactly_the_masked_ones():
    """The range must agree with the mask, not merely be plausible.

    This is the contract that makes the change bit-exact: for every context
    length, the first key the range keeps is the first key the mask keeps.
    """

    for window in (1, 4, 1024):
        for live in range(1, 40):
            key_begin = _sliding_read_range(_Attention(window), start=live - 1, rows=1)
            kept = [p for p in range(live) if live - 1 - p < window]
            assert key_begin == (kept[0] if kept else 0), (window, live)


def test_multi_row_blocks_never_skip():
    """A prefill block's rows have different windows, and the mask rows are
    strided, so the single-pointer offset the decode path uses is not valid."""

    assert _sliding_read_range(_Attention(1024), start=4096, rows=8) == 0


def test_zero_window_is_rejected_rather_than_silently_wrong():
    with pytest.raises(ValueError):
        _sliding_read_range(_Attention(0), start=10, rows=1)
