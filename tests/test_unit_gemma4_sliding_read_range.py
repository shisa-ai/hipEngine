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

from hipengine.runtime.gemma4 import _keep_mask, _sliding_read_range


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


def test_multi_row_blocks_skip_to_the_earliest_key_their_first_row_can_read():
    """A prefill block narrows to the first key any of its rows can read.

    Row 0 sits at ``start`` and a sliding layer keeps key ``k`` for query ``q``
    only while ``q - k < window``, so the earliest key the whole block can need
    is ``start - window + 1``. Rows further into the block have later windows
    and cannot need anything earlier, so this is the exact block bound.
    """

    # Rows at 4096..4103 with window 1024: row 4096 keeps keys from 3073 on.
    assert _sliding_read_range(_Attention(1024), start=4096, rows=8) == 4096 - 1024 + 1


def test_a_multi_row_block_wider_than_its_window_still_skips_nothing():
    # Rows at 0..7 with window 1024: every key is inside row 0's window.
    assert _sliding_read_range(_Attention(1024), start=0, rows=8) == 0


def test_the_block_range_agrees_with_the_block_mask():
    """The range must be the first key the mask keeps for any row of the block.

    This is the contract that makes a narrowed prefill block bit-exact, and it
    has to hold for every block shape, not only for one-row blocks.
    """

    for window in (1, 4, 1024):
        for start in (0, 1, 5, 1024, 4096):
            for rows in (1, 2, 8, 64):
                key_begin = _sliding_read_range(_Attention(window), start=start, rows=rows)
                kept = [
                    key
                    for key in range(start + rows)
                    if any(0 <= query - key < window for query in range(start, start + rows))
                ]
                assert key_begin == (kept[0] if kept else 0), (window, start, rows)


def test_the_block_mask_columns_start_at_the_block_range():
    """The mask has to be built over the same range the kernel walks.

    ``key_begin`` offsets the K/V pointers and the key count, and the kernel
    indexes the mask as ``keep_mask + token * keys``, so column 0 of the mask
    must be key ``key_begin`` for every row. A full-width mask would address
    the wrong row from the second row on, which is why the range and the mask
    have to move together.
    """

    attention = _Attention(4)
    start, rows = 10, 3
    key_begin = _sliding_read_range(attention, start, rows)
    assert key_begin == 7

    mask = _keep_mask(attention, start, rows)
    assert mask.shape == (rows, start + rows - key_begin)
    for token in range(rows):
        for column in range(mask.shape[1]):
            key = key_begin + column
            query = start + token
            expected = key <= query and (query - key) < 4
            assert bool(mask[token, column]) == expected, (token, column)


def test_a_dense_block_mask_covers_every_key_from_the_range():
    attention = _Attention(None)
    mask = _keep_mask(attention, start=6, rows=3)
    assert mask.shape == (3, 9)
    for token in range(3):
        for key in range(9):
            assert bool(mask[token, key]) == (key <= 6 + token)


def test_zero_window_is_rejected_rather_than_silently_wrong():
    with pytest.raises(ValueError):
        _sliding_read_range(_Attention(0), start=10, rows=1)


def test_every_column_below_the_per_row_bound_is_masked():
    """The kernel may start a row's walk at its own first kept column.

    Row ``t`` of a block sits at ``start + t`` and the mask's column 0 is key
    ``key_begin``, so the row's own bound is ``max(0, (start - key_begin) + t -
    window + 1)``. Every column below it must be zero, or starting the walk
    there would drop a key the mask keeps and the output would be wrong rather
    than merely different. The bound must also be tight, since a bound that
    lags the mask would silently give back the win it is there to take.

    This is the contract the prefill kernel's per-row skip relies on, and it
    has to hold for every block shape, not only for one-row blocks.
    """

    for window in (1, 4, 1024):
        for start in (0, 1, 5, 1024, 4096):
            for rows in (1, 2, 8, 64):
                attention = _Attention(window)
                key_begin = _sliding_read_range(attention, start, rows)
                mask = _keep_mask(attention, start, rows)
                for token in range(rows):
                    bound = max(0, (start - key_begin) + token - window + 1)
                    assert not mask[token, :bound].any(), (window, start, rows, token)
                    # A row always keeps its own position, so the bound names a
                    # column that exists and is kept.
                    assert mask[token, bound], (window, start, rows, token)
