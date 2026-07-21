from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.gguf_prefix_reuse_gate import (
    _compare_states,
    _lifecycle_exact,
    _logical_page_segments,
    _production_metadata_exact,
    build_parser,
)


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


def test_completed_source_lifecycle_and_metadata_fail_closed() -> None:
    assert _lifecycle_exact(
        "completed",
        source_refcount_before_release=1,
        source_refcount_after_release=1,
        shared_refcount_after_admission=2,
        shared_refcount_after_continuation_release=1,
        final_refcounted_pages=0,
        source_session_reset=True,
        snapshot_evicted=True,
    )
    assert not _lifecycle_exact(
        "completed",
        source_refcount_before_release=1,
        source_refcount_after_release=0,
        shared_refcount_after_admission=1,
        shared_refcount_after_continuation_release=0,
        final_refcounted_pages=0,
        source_session_reset=True,
        snapshot_evicted=True,
    )
    assert _production_metadata_exact(
        "completed",
        boundary=256,
        reused_tokens=256,
        source_request_id=None,
        source_id=1001,
        clone_bytes=384,
        snapshot_hit=True,
    )
    assert not _production_metadata_exact(
        "completed",
        boundary=256,
        reused_tokens=256,
        source_request_id=1001,
        source_id=1001,
        clone_bytes=384,
        snapshot_hit=False,
    )

    args = build_parser().parse_args(
        [
            "--source-lifecycle",
            "completed",
            "--sampler-mode",
            "processed_argmax",
            "--forced-token-id",
            "811",
        ]
    )
    assert args.source_lifecycle == "completed"
    assert args.sampler_mode == "processed_argmax"
    assert args.forced_token_id == 811
