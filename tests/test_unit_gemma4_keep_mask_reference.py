"""V2: bind the production keep-mask to the independently validated reference.

Punchlist V2 exists because the campaign gate compares two hipEngine arms
against a captured baseline, so it "cannot detect a wrong mask that both arms
share". Iteration 30 measured exactly that shape of agreement -- tokenwise and
production-shaped bulk agree to 6.8e-06 mean KL -- which is evidence of
consistency, not correctness.

The shared mask is built in one place: ``_keep_mask`` in
``hipengine/runtime/gemma4.py``, called for every block of every arm. Nothing
previously compared it to anything outside the runtime. This test binds it to
``gemma4_attention_mask`` in ``hipengine/kernels/cpu_reference/gemma4.py``, a
separate implementation that is itself gated against the HuggingFace
``Gemma4ForCausalLM`` oracle in ``test_unit_gemma4_cpu_reference.py``.

The two are written differently on purpose -- production accumulates
``(queries - key_positions) < W`` while the reference intersects
``key > query - W`` -- so agreeing is a real comparison rather than a shared
expression. If either drifts, these cases fail.

Coverage follows the punchlist's own boundary instinct: window widths at and
around the keep-region edge, and query/key pairs one step either side of it,
because an off-by-one in the window bound is invisible at interior points.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    gemma4_attention_mask,
)
from hipengine.runtime.gemma4 import _keep_mask


def _geometry(sliding_window: int | None) -> Gemma4AttentionGeometry:
    return Gemma4AttentionGeometry(
        layer_type=SLIDING_ATTENTION if sliding_window is not None else FULL_ATTENTION,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        rope=Gemma4RopeConfig(
            rope_theta=10000.0,
            head_dim=8,
            rope_angles=4,
            rope_type=1,
        ),
        sliding_window=sliding_window,
        k_eq_v=False,
    )


def _reference_mask(sliding_window: int | None, start: int, rows: int) -> np.ndarray:
    """The same (rows, start + rows) window the production mask must match."""
    keys = start + rows
    query_positions = np.arange(start, start + rows, dtype=np.int64)
    key_positions = np.arange(keys, dtype=np.int64)
    keep = gemma4_attention_mask(_geometry(sliding_window), query_positions, key_positions)
    return np.ascontiguousarray(keep.astype(np.uint8))


@pytest.mark.parametrize("sliding_window", [None, 1, 2, 3, 7, 8, 16, 512, 1024])
@pytest.mark.parametrize("start", [0, 1, 4, 127, 511])
@pytest.mark.parametrize("rows", [1, 2, 3, 8, 17])
def test_keep_mask_matches_the_reference(
    sliding_window: int | None, start: int, rows: int
) -> None:
    """Production and reference agree on every window, offset and block width."""
    actual = _keep_mask(_geometry(sliding_window), start, rows)
    expected = _reference_mask(sliding_window, start, rows)

    assert actual.dtype == np.uint8
    assert actual.shape == (rows, start + rows)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            f"keep-mask disagrees with the reference at window={sliding_window} "
            f"start={start} rows={rows}"
        ),
    )


@pytest.mark.parametrize("sliding_window", [1, 2, 3, 8, 16, 512])
def test_window_bound_is_exclusive_at_exactly_one_past(sliding_window: int) -> None:
    """Pin the edge itself: ``q - k == W`` is masked, ``q - k == W - 1`` is kept.

    An interior point cannot distinguish ``< W`` from ``<= W``, so assert the
    boundary directly rather than trusting a sweep that happens to pass.
    """
    start = sliding_window + 4
    rows = 1
    keep = _keep_mask(_geometry(sliding_window), start, rows)[0]
    query = start

    inside = query - (sliding_window - 1)   # distance W-1: must be kept
    outside = query - sliding_window        # distance W:   must be masked
    assert 0 <= outside < inside < keep.size
    assert keep[inside] == 1, f"key at distance {sliding_window - 1} must stay visible"
    assert keep[outside] == 0, f"key at distance {sliding_window} must be evicted"


def test_sliding_mask_stays_causal_below_the_window() -> None:
    """While ``keys <= sliding_window`` the window cannot bind, so the mask is
    pure causality -- the vacuous region the admission logic relies on."""
    window = 512
    for keys in (1, 2, 511, 512):
        keep = _keep_mask(_geometry(window), 0, keys)
        expected = np.tril(np.ones((keys, keys), dtype=np.uint8))
        np.testing.assert_array_equal(keep, expected, err_msg=f"keys={keys}")


def test_full_layer_is_pure_causality() -> None:
    """A non-sliding layer keeps the window branch out entirely."""
    keep = _keep_mask(_geometry(None), 0, 64)
    np.testing.assert_array_equal(keep, np.tril(np.ones((64, 64), dtype=np.uint8)))


def test_decode_block_reads_only_the_cached_range() -> None:
    """A decode block at ``start`` must not mask against keys past its own
    write range: the mask is exactly ``(rows, start + rows)``."""
    keep = _keep_mask(_geometry(8), start=40, rows=1)
    assert keep.shape == (1, 41)
    # position 40 may see itself and the 8 keys before it
    assert keep[0, 40] == 1
    assert keep[0, 33] == 1, "key at distance 7 is inside a window of 8"
    assert keep[0, 32] == 0, "key at distance 8 is outside a window of 8"


def test_every_key_row_is_produced_for_a_block() -> None:
    """No column may be dropped: the kernel reads the mask as (rows, keys)."""
    for start, rows in ((0, 1), (7, 3), (100, 16)):
        assert _keep_mask(_geometry(4), start, rows).shape == (rows, start + rows)