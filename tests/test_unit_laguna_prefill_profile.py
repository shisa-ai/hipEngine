from __future__ import annotations

import base64
import struct

import pytest

from scripts.laguna_prefill_profile import (
    _profile_token_stream,
    _summarize_routing_replay,
    _summarize_timing_samples,
)


def test_profile_token_stream_uses_longest_canonical_prompt_and_explicit_extension() -> None:
    prompts = [
        {"id": "short", "category": "code", "token_ids": (1, 2, 3)},
        {"id": "long", "category": "general_en", "token_ids": (9, 8, 7, 6, 5)},
    ]

    tokens, source = _profile_token_stream(prompts, 8)

    assert tokens == (9, 8, 7, 6, 5, 8, 7, 6)
    assert source == {
        "prompt_id": "long",
        "category": "general_en",
        "source_tokens": 5,
        "requested_tokens": 8,
        "extension": "repeat_without_leading_bos",
    }


def test_profile_token_stream_rejects_empty_or_singleton_extension_source() -> None:
    with pytest.raises(ValueError, match="canonical prompts"):
        _profile_token_stream([], 8)
    with pytest.raises(ValueError, match="at least two tokens"):
        _profile_token_stream(
            [{"id": "one", "category": "code", "token_ids": (1,)}],
            8,
        )


def test_routing_replay_summary_records_per_expert_counts_and_padding() -> None:
    replay = {
        1: (0, 0, 1, 2, 2, 2),
        2: (1, 1, 1, 1, 3, 3),
    }

    summary = _summarize_routing_replay(
        replay,
        rows=2,
        top_k=3,
        expert_count=4,
        tile_rows=4,
    )

    assert summary["actual_lanes"] == 12
    assert summary["active_expert_groups"] == 5
    assert summary["group_size_histogram"] == {"1": 1, "2": 2, "3": 1, "4": 1}
    assert summary["compact_tile_rows"] == 4
    assert summary["compact_padded_lanes"] == 20
    assert summary["compact_padding_lanes"] == 8
    assert summary["compact_padding_overhead_ratio"] == pytest.approx(2 / 3)
    assert summary["max_expert_lanes"] == 4
    assert summary["layers"]["1"]["per_expert_counts_encoding"] == (
        "uint16_le_dense_expert_id_order_base64"
    )
    assert struct.unpack(
        "<4H",
        base64.b64decode(summary["layers"]["1"]["per_expert_counts_u16_le_base64"]),
    ) == (2, 1, 3, 0)
    assert struct.unpack(
        "<4H",
        base64.b64decode(summary["layers"]["2"]["per_expert_counts_u16_le_base64"]),
    ) == (0, 4, 0, 2)
    assert len(summary["selected_ids_sha256"]) == 64


def test_routing_replay_summary_fails_closed_on_bad_lane_count_or_id() -> None:
    with pytest.raises(ValueError, match="selected lanes"):
        _summarize_routing_replay(
            {1: (0, 1)}, rows=2, top_k=2, expert_count=4, tile_rows=16
        )
    with pytest.raises(ValueError, match="expert IDs"):
        _summarize_routing_replay(
            {1: (0, 1, 2, 4)}, rows=2, top_k=2, expert_count=4, tile_rows=16
        )


def test_timing_summary_uses_complete_samples_and_median() -> None:
    summary = _summarize_timing_samples([2.0, 1.0, 3.0], rows=12)

    assert summary == {
        "rows": 12,
        "samples_seconds": [2.0, 1.0, 3.0],
        "median_seconds": 2.0,
        "median_tok_s": 6.0,
        "min_seconds": 1.0,
        "max_seconds": 3.0,
    }

    with pytest.raises(ValueError, match="timing sample"):
        _summarize_timing_samples([], rows=12)
