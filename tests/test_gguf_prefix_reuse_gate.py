from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.gguf_prefix_reuse_gate import _compare_states, _logical_page_segments


def test_logical_page_segments_follow_noncontiguous_block_table_order() -> None:
    allocation = SimpleNamespace(
        block_ids=(8, 9, 11),
        chunk_start_block_id=8,
    )

    assert _logical_page_segments(
        allocation,
        position=513,
        row_nbytes=4,
    ) == (
        (0, 1024),
        (1024, 1024),
        (3072, 4),
    )

    with pytest.raises(ValueError, match="does not cover"):
        _logical_page_segments(allocation, position=769, row_nbytes=4)


def test_compare_states_reports_exact_component_and_layer() -> None:
    reference = {
        "position": 257,
        "linear": [{"layer": 0, "conv": "a", "recurrent": "b"}],
        "kv": [{"layer": 3, "key": "c", "value": "d", "checked_nbytes": 8}],
    }
    exact = {
        "position": 257,
        "linear": [{"layer": 0, "conv": "a", "recurrent": "b"}],
        "kv": [{"layer": 3, "key": "c", "value": "d", "checked_nbytes": 8}],
    }
    changed = {
        "position": 257,
        "linear": [{"layer": 0, "conv": "a", "recurrent": "changed"}],
        "kv": [{"layer": 3, "key": "c", "value": "d", "checked_nbytes": 8}],
    }

    assert _compare_states(exact, reference) == []
    assert _compare_states(changed, reference) == [
        {
            "component": "linear",
            "layer": 0,
            "part": "recurrent",
            "candidate": "changed",
            "reference": "b",
        }
    ]
