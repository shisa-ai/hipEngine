"""The pressure gate's decision logic, without a server or a GPU.

The gate itself needs a live endpoint, but what it *decides* is pure: which
reuse depth each conversation should reach, whether an evicted conversation
missed cleanly, whether reuse changed the output, and whether ``/ready``
accounts for the hits and misses the requests reported.  Those are the checks
that must not silently weaken.
"""

from __future__ import annotations

import copy

from scripts.prefix_pressure_gate import (
    captured_boundary,
    conversation_prompt,
    evaluate,
    reuse_expectation,
)


def _conversation(
    index: int,
    *,
    kind: str,
    expected: int | None,
    cached: int,
    parity: bool = True,
    text: str = "answer",
) -> dict:
    return {
        "index": index,
        "kind": kind,
        "prompt_tokens": 900,
        "expected_reuse": expected,
        "cold": {"cached_tokens": 0, "text": text, "ttft_s": 4.0, "wall_s": 5.0},
        "warm": {
            "cached_tokens": cached,
            "text": text if parity else "different",
            "ttft_s": 1.0,
            "wall_s": 2.0,
            "decode_tps": 200.0,
        },
    }


def _record(*, warm_hits: int = 4, misses: int = 2, reported_hits: int = 4,
            reported_misses: int = 2) -> dict:
    conversations = [
        _conversation(index, kind="warm", expected=768, cached=768)
        for index in range(warm_hits)
    ] + [
        _conversation(index + warm_hits, kind="evicted", expected=None, cached=0)
        for index in range(misses)
    ]
    return {
        "routes": {
            "on": {
                "conversations": conversations,
                "ready_after_cold": {"usable_hits": 0, "misses": 6, "stats": {"live_requests": 0}},
                "ready_after_warm": {
                    "usable_hits": reported_hits,
                    "misses": 6 + reported_misses,
                    "stats": {"live_requests": 0},
                },
            }
        }
    }


def test_captured_boundary_is_the_deepest_reusable_block_boundary() -> None:
    assert captured_boundary(919) == 768
    assert captured_boundary(768) == 768
    assert captured_boundary(900) == 768


def test_reuse_expectation_excludes_a_prompt_that_ends_on_a_boundary() -> None:
    """A whole-prompt boundary cannot serve a same-length resend."""

    expectation = reuse_expectation([919, 1024, 512, 200])
    assert expectation[0] == 768
    assert expectation[1] is None
    assert expectation[2] is None
    assert expectation[3] is None


def test_conversation_prompts_are_distinct_and_long_enough() -> None:
    first = conversation_prompt("mtp on", 0, nonce="a")
    second = conversation_prompt("mtp on", 1, nonce="a")
    other_route = conversation_prompt("mtp off", 0, nonce="a")
    assert len({first, second, other_route}) == 3
    assert len(first.split()) > 700


def test_gate_accepts_a_record_that_meets_every_acceptance() -> None:
    assert evaluate(_record()) == []


def test_gate_flags_a_shallower_reuse_than_the_captured_boundary() -> None:
    record = _record()
    record["routes"]["on"]["conversations"][0]["warm"]["cached_tokens"] = 512
    failures = evaluate(record)
    assert any("reused 512 tokens, expected 768" in failure for failure in failures)


def test_gate_flags_a_stale_hit_on_an_evicted_conversation() -> None:
    record = _record()
    record["routes"]["on"]["conversations"][4]["warm"]["cached_tokens"] = 256
    failures = evaluate(record)
    assert any("expected a clean miss" in failure for failure in failures)


def test_gate_flags_output_that_changed_with_prefix_reuse() -> None:
    record = _record()
    record["routes"]["on"]["conversations"][1]["warm"]["text"] = "corrupted"
    failures = evaluate(record)
    assert any("output changed with prefix reuse" in failure for failure in failures)


def test_gate_flags_a_served_surface_that_stops_reporting_hits() -> None:
    failures = evaluate(_record(reported_hits=0))
    assert any("/ready reported 0 hits for 4 reusing requests" in f for f in failures)
    failures = evaluate(_record(reported_misses=0))
    assert any("/ready reported 0 misses for 2" in f for f in failures)


def test_gate_flags_an_empty_completion_and_a_live_counter_sample() -> None:
    record = _record()
    record["routes"]["on"]["conversations"][2]["warm"]["text"] = ""
    record["routes"]["on"]["ready_after_warm"]["stats"]["live_requests"] = 1
    failures = evaluate(record)
    assert any("produced no text" in failure for failure in failures)
    assert any("with requests live" in failure for failure in failures)


def test_gate_flags_a_route_with_no_conversations() -> None:
    record = _record()
    record["routes"]["off"] = {"conversations": []}
    failures = evaluate(record)
    assert any("off: no conversations recorded" in failure for failure in failures)


def test_gate_record_round_trips_through_json() -> None:
    import json

    record = _record()
    restored = json.loads(json.dumps(copy.deepcopy(record)))
    assert evaluate(restored) == []
