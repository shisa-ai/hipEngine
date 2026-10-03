"""Bind shortened and graph-bucket keep masks to the independent CPU oracle."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    FULL_ATTENTION, SLIDING_ATTENTION,
    Gemma4AttentionGeometry, Gemma4RopeConfig, gemma4_attention_mask,
)
from hipengine.runtime import gemma4 as module
from hipengine.runtime.gemma4 import Gemma4Runner, _keep_mask


def _geometry(sliding_window: int | None) -> Gemma4AttentionGeometry:
    return Gemma4AttentionGeometry(
        layer_type=SLIDING_ATTENTION if sliding_window is not None else FULL_ATTENTION,
        num_heads=2, num_kv_heads=1, head_dim=8,
        rope=Gemma4RopeConfig(rope_theta=10000.0, head_dim=8, rope_angles=4, rope_type=1),
        sliding_window=sliding_window, k_eq_v=False,
    )


def _reference_mask(window, start, rows, *, extent=None, begin=0):
    query_positions = np.arange(start, start + rows, dtype=np.int64)
    key_positions = np.arange(begin, start + rows if extent is None else extent, dtype=np.int64)
    keep = gemma4_attention_mask(_geometry(window), query_positions, key_positions)
    return np.ascontiguousarray(keep.astype(np.uint8))


@pytest.mark.parametrize("sliding_window", [None, 1, 2, 3, 7, 8, 16, 512, 1024])
@pytest.mark.parametrize("start", [0, 1, 4, 127, 511])
@pytest.mark.parametrize("rows", [1, 2, 3, 8, 17])
def test_keep_mask_matches_the_reference(sliding_window, start, rows) -> None:
    actual = _keep_mask(_geometry(sliding_window), start, rows)
    full = _reference_mask(sliding_window, start, rows)
    # Derive the dropped prefix from the oracle's own visible columns, not
    # from the runtime helper under test. Every dropped column must be zero.
    visible = np.flatnonzero(np.any(full, axis=0))
    begin = int(visible[0])
    assert not np.any(full[:, :begin])
    assert actual.dtype == np.uint8
    assert actual.shape == (rows, start + rows - begin)
    np.testing.assert_array_equal(actual, full[:, begin:])


@pytest.mark.parametrize("sliding_window", [1, 2, 3, 8, 16, 512])
def test_window_bound_is_exclusive_at_exactly_one_past(sliding_window) -> None:
    start = sliding_window + 4
    # A frozen range may include older zero columns; exercise both sides of
    # the window edge without changing their absolute key coordinates.
    keep = _keep_mask(_geometry(sliding_window), start, 1, key_begin=0)[0]
    assert keep[start - sliding_window + 1] == 1
    assert keep[start - sliding_window] == 0


def test_sliding_mask_stays_causal_below_the_window() -> None:
    for keys in (1, 2, 511, 512):
        keep = _keep_mask(_geometry(512), 0, keys)
        np.testing.assert_array_equal(keep, np.tril(np.ones((keys, keys), dtype=np.uint8)))


def test_full_layer_is_pure_causality() -> None:
    keep = _keep_mask(_geometry(None), 0, 64)
    np.testing.assert_array_equal(keep, np.tril(np.ones((64, 64), dtype=np.uint8)))


def test_decode_block_reads_only_the_cached_range() -> None:
    keep = _keep_mask(_geometry(8), start=40, rows=1)
    assert keep.shape == (1, 8)
    # Columns now correspond to absolute keys 33..40, all inside the window.
    assert np.all(keep == 1)


def test_every_key_row_is_produced_for_a_block() -> None:
    for start, rows in ((0, 1), (7, 3), (100, 16)):
        expected = _reference_mask(4, start, rows)
        begin = int(np.flatnonzero(np.any(expected, axis=0))[0])
        np.testing.assert_array_equal(_keep_mask(_geometry(4), start, rows), expected[:, begin:])


@pytest.mark.parametrize("start", [128, 131, 191])
def test_graph_mask_matches_the_frozen_column_origin_and_extent(start) -> None:
    keep = _keep_mask(_geometry(8), start, 1, 192, key_begin=120)
    assert keep.shape == (1, 72)
    np.testing.assert_array_equal(keep, _reference_mask(8, start, 1, extent=192, begin=120))


def test_staging_threads_the_graphs_frozen_first_key(monkeypatch) -> None:
    attention = _geometry(8)
    runner = object.__new__(Gemma4Runner)
    runner.weights = SimpleNamespace(layers=[object()], config=SimpleNamespace(geometry=lambda _: attention))
    runner._position = 131
    runner._token_ids = object()
    runner._stage_upload = lambda name, values, **kwargs: values
    monkeypatch.setattr(module, "enqueue_host_to_device", lambda *args, **kwargs: None)
    _, masks = runner._stage_block_content([1], stream=0, keys_extent=192, key_begin_at=lambda _: 120)
    assert masks[8].shape == (1, 72)
    np.testing.assert_array_equal(masks[8], _reference_mask(8, 131, 1, extent=192, begin=120))


@pytest.mark.parametrize("begin", [-1, 125])
def test_mask_refuses_an_origin_that_is_negative_or_drops_live_keys(begin) -> None:
    with pytest.raises(ValueError, match="key_begin"):
        _keep_mask(_geometry(8), 131, 1, key_begin=begin)
