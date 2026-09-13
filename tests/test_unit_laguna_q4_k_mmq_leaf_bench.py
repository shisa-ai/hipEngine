from __future__ import annotations

import numpy as np

from scripts.laguna_q4_k_mmq_leaf_bench import _mixed_metadata


def test_mixed_metadata_partitions_active_experts_at_global_threshold() -> None:
    counts = np.asarray([0, 1, 3, 4, 33], dtype=np.int64)

    metadata = _mixed_metadata(counts, min_mmq_rows=4)

    np.testing.assert_array_equal(metadata["mmq_experts"], [3, 4])
    np.testing.assert_array_equal(metadata["exact_experts"], [1, 2])
    np.testing.assert_array_equal(metadata["starts32"], [0, 0, 0, 0, 32, 96])
    np.testing.assert_array_equal(metadata["tile_expert32"], [3, 4, 4])
    assert metadata["total32"] == 96
    assert metadata["mmq_compact_rows"] == 37
    assert metadata["exact_compact_rows"] == 4


def test_mixed_metadata_supports_all_exact_and_all_mmq_edges() -> None:
    counts = np.asarray([0, 1, 9, 32], dtype=np.int64)

    all_exact = _mixed_metadata(counts, min_mmq_rows=33)
    assert all_exact["total32"] == 0
    assert all_exact["mmq_compact_rows"] == 0
    np.testing.assert_array_equal(all_exact["exact_experts"], [1, 2, 3])

    all_mmq = _mixed_metadata(counts, min_mmq_rows=1)
    assert all_mmq["exact_compact_rows"] == 0
    assert all_mmq["mmq_compact_rows"] == 42
    np.testing.assert_array_equal(all_mmq["tile_expert32"], [1, 2, 3])


def test_mixed_metadata_rejects_nonpositive_threshold() -> None:
    counts = np.asarray([1, 2], dtype=np.int64)

    try:
        _mixed_metadata(counts, min_mmq_rows=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected a nonpositive threshold to fail")
