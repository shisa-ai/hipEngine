"""Unit tests for the prefix-cache multi-turn benchmark harness.

Covers the serving-transport contract: per-turn prefix telemetry differenced
from the server's aggregate counters, TTFT attribution over SSE, warmup
separation, alternating A/B arm order, and multi-arm subprocess orchestration
with merge and failure propagation. The GPU path is exercised by the script
itself; these tests pin the measurement plumbing with a stub engine and a stub
arm subprocess, so no model is loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.prefix_cache_multiturn_bench import (
    _arm_plan,
    _lane_order,
    _prefix_counter_delta,
    _run_turn_server,
    _run_warmup,
    _summarize_mode,
    LaneResult,
)


class _StubLLM:
    """Minimal engine surface the server transport touches."""

    def __init__(self) -> None:
        self.count_tokens_calls: list[str] = []

    def count_tokens(self, text: str) -> int:
        self.count_tokens_calls.append(str(text))
        return len(str(text).split())


def _build_server_client(
    prefix_block: dict[str, Any],
    *,
    on_request: dict[str, Any] | None = None,
    prompt_tokens: int = 768,
    generated_tokens: int = 2,
) -> tuple[Any, Any]:
    """Create a TestClient whose ``/health`` reports ``prefix_block``.

    The chat endpoint is stubbed at the routing layer so the test measures the
    harness's own SSE parsing and counter differencing, not a model.
    """

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    state = {"prefix": dict(prefix_block)}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"object": "hipengine.health", "status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"object": "hipengine.readiness", "prefix_cache": dict(state["prefix"])}

    @app.post("/v1/chat/completions")
    def chat() -> Any:
        def stream() -> Any:
            yield (
                'data: {"choices":[{"delta":{"content":"he"},'
                '"hipengine":{"timing":{"elapsed_ms":1.5,"ttft_ms":140.0}}}]}\n\n'
            )
            # The prefix lookup happens while the turn is in flight.
            if on_request is not None:
                state["prefix"] = {**state["prefix"], **on_request}
            yield (
                'data: {"choices":[{"delta":{"content":"llo"},'
                '"hipengine":{"timing":{"elapsed_ms":3.5,"ttft_ms":140.0}}}]}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"length",'
                '"hipengine":{"decode_state":{"prompt_tokens":%d,'
                '"generated_tokens":%d,"prefill_ms":12.5}}}]}\n\n'
                % (int(prompt_tokens), int(generated_tokens))
            )
            yield (
                'data: {"choices":[],"usage":{"prompt_tokens":%d,'
                '"completion_tokens":%d}}\n\n'
                % (int(prompt_tokens), int(generated_tokens))
            )
            yield "data: [DONE]\n\n"

        from fastapi.responses import StreamingResponse

        return StreamingResponse(stream(), media_type="text/event-stream")

    return TestClient(app), state


def test_prefix_counter_delta_differences_nested_counters() -> None:
    before = {
        "mode": "radix",
        "reused_tokens": 512,
        "stats": {"hits": 2, "misses": 1},
        "snapshot_entries": 4,
    }
    after = {
        "mode": "radix",
        "reused_tokens": 1536,
        "stats": {"hits": 3, "misses": 2},
        "snapshot_entries": 5,
    }

    delta = _prefix_counter_delta(before, after)

    assert delta["reused_tokens"] == 1024
    assert delta["stats"] == {"hits": 1, "misses": 1}
    # snapshot_entries is a gauge: the turn reports the level, not the step.
    assert delta["snapshot_entries"] == 5
    # Non-numeric fields report the current value, not a difference.
    assert delta["mode"] == "radix"


def test_prefix_counter_delta_treats_missing_keys_as_zero() -> None:
    delta = _prefix_counter_delta({}, {"stats": {"hits": 1}, "resident_bytes": 8})

    assert delta["stats"]["hits"] == 1
    assert delta["resident_bytes"] == 8


def test_run_turn_server_records_ttft_usage_and_prefix_delta() -> None:
    client, _state = _build_server_client(
        {
            "mode": "radix",
            "block_size_tokens": 256,
            "stats": {"hits": 0, "misses": 0},
            "reused_tokens": 0,
            "snapshot_entries": 0,
            "resident_bytes": 0,
        },
        on_request={
            "stats": {"hits": 1, "misses": 0},
            "reused_tokens": 512,
            "snapshot_entries": 3,
            "resident_bytes": 4096,
        },
    )
    llm = _StubLLM()

    record = _run_turn_server(
        client,
        llm=llm,  # type: ignore[arg-type]
        engine=llm,
        messages=[{"role": "user", "content": "hello there"}],
        tools=None,
        max_tokens=2,
    )

    assert record["http_status"] == 200
    assert record["error"] is None
    assert record["finish_reason"] == "length"
    assert record["text"] == "hello"
    assert record["transport"] == "server"
    assert record["ttft_ms"] == 140.0
    assert record["ttft_source"] == "server"
    # A stub fabricates the server clock, so only presence is asserted here.
    assert record["client_observed_ttft_ms"] is not None
    assert record["prompt_tokens"] == 768
    assert record["generator_prompt_tokens"] == 768
    assert record["output_tokens"] == 2
    prefix = record["prefix"]
    assert prefix["mode"] == "radix"
    assert prefix["lookup"] is True
    assert prefix["hit"] is True
    assert prefix["reused_tokens"] == 512
    assert prefix["matched_tokens"] == 512
    assert prefix["snapshot_entries"] == 3
    assert prefix["fallback_reason"] is None
    # The server's own usage block is authoritative; the local renderer count is
    # kept only as a cross-check.
    assert llm.count_tokens_calls
    assert record["local_prompt_tokens"] == 4


def test_run_turn_server_reports_miss_without_reuse() -> None:
    client, _state = _build_server_client(
        {
            "mode": "radix",
            "block_size_tokens": 256,
            "stats": {"hits": 0, "misses": 0},
            "reused_tokens": 0,
            "snapshot_entries": 0,
        },
        on_request={"stats": {"hits": 0, "misses": 1}, "snapshot_entries": 1},
    )

    record = _run_turn_server(
        client,
        llm=_StubLLM(),  # type: ignore[arg-type]
        engine=None,
        messages=[{"role": "user", "content": "short prompt"}],
        tools=None,
        max_tokens=4,
    )

    assert record["prefix"]["lookup"] is True
    assert record["prefix"]["hit"] is False
    assert record["prefix"]["reused_tokens"] == 0
    assert record["prefix"]["fallback_reason"] == "miss"


def test_run_turn_server_without_a_lookup_is_not_a_lookup() -> None:
    """A turn the cache never consulted must not report a lookup."""

    client, _state = _build_server_client(
        {"mode": "radix", "stats": {"hits": 3, "misses": 5}, "reused_tokens": 4096}
    )

    record = _run_turn_server(
        client,
        llm=_StubLLM(),  # type: ignore[arg-type]
        engine=None,
        messages=[{"role": "user", "content": "short prompt"}],
        tools=None,
        max_tokens=4,
    )

    assert record["prefix"]["lookup"] is False
    assert record["prefix"]["hit"] is False
    assert record["prefix"]["reused_tokens"] == 0
    assert record["prefix"]["fallback_reason"] == "no_lookup"


def test_run_turn_server_labels_a_below_granularity_prompt() -> None:
    client, _state = _build_server_client(
        {"mode": "radix", "stats": {"hits": 0, "misses": 0}},
        prompt_tokens=64,
    )

    record = _run_turn_server(
        client,
        llm=_StubLLM(),  # type: ignore[arg-type]
        engine=None,
        messages=[{"role": "user", "content": "short prompt"}],
        tools=None,
        max_tokens=4,
    )

    assert record["prefix"]["lookup"] is False
    assert record["prefix"]["fallback_reason"] == "prompt_too_short"


def test_run_warmup_is_separate_from_measured_summaries() -> None:
    calls: list[dict[str, Any]] = []

    def run_turn(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "render_ms": 1.0,
            "wall_ms": 10.0,
            "ttft_ms": 5.0,
            "prefill_ms": 2.0,
            "decode_ms": 3.0,
            "prompt_tokens": 32,
            "generator_prompt_tokens": 32,
            "output_tokens": 4,
            "gtt_bytes_before": None,
            "gtt_bytes_after": None,
            "text": "warm",
            "prefix": {},
        }

    warmup = _run_warmup(run_turn, turns=2)

    # A short ready turn plus the warmup conversation turns.
    assert len(warmup) == 3
    assert all(record["lane_kind"] == "warmup" for record in warmup[1:])
    assert all(record["lane_id"] == "warmup" for record in warmup[1:])

    measured = _summarize_mode(
        [LaneResult(lane_id="lane-a", lane_kind="sharegpt", turns=[dict(warmup[0])])]
    )
    assert measured["turns"] == 1
    assert measured["ttft_ms"]["count"] == 1


def test_run_warmup_can_be_disabled() -> None:
    assert _run_warmup(lambda **_: {}, turns=0) == []


def test_arm_plan_alternates_mode_order_across_repetitions() -> None:
    sequential = _arm_plan(["off", "radix"], repetitions=2, arm_order="sequential")
    alternating = _arm_plan(["off", "radix"], repetitions=2, arm_order="alternating")

    assert sequential == [(0, "off"), (0, "radix"), (1, "off"), (1, "radix")]
    assert alternating == [(0, "off"), (0, "radix"), (1, "radix"), (1, "off")]


def test_arm_plan_single_mode_is_unaffected_by_order() -> None:
    assert _arm_plan(["off"], repetitions=3, arm_order="alternating") == [
        (0, "off"),
        (1, "off"),
        (2, "off"),
    ]


def test_lane_order_reverses_on_odd_repetitions() -> None:
    lane_ids = ["a", "b", "c"]

    assert _lane_order(lane_ids, 0) == ["a", "b", "c"]
    assert _lane_order(lane_ids, 1) == ["c", "b", "a"]


def test_server_transport_parser_defaults_to_inprocess() -> None:
    from scripts.prefix_cache_multiturn_bench import build_parser

    parser = build_parser()
    args = parser.parse_args([])

    assert args.transport == "inprocess"
    assert args.arm_order == "sequential"
    assert args.warmup_turns == 1
    server = parser.parse_args(["--transport", "server", "--arm-order", "alternating"])
    assert server.transport == "server"
    assert server.arm_order == "alternating"
    with pytest.raises(SystemExit):
        parser.parse_args(["--transport", "bogus"])
    # The one-process multi-arm escape hatch is gone: it faulted the GPU.
    assert not any(
        "--allow-multi-mode" in action.option_strings for action in parser._actions
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--allow-multi-mode"])


def test_server_prefix_counters_survive_a_missing_health_block() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from scripts.prefix_cache_multiturn_bench import _server_prefix_counters

    app = FastAPI()

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"object": "hipengine.readiness"}

    with TestClient(app) as client:
        assert _server_prefix_counters(client) == {}


def test_ttft_is_taken_from_a_reasoning_only_stream() -> None:
    """A thinking model can spend its whole budget in the reasoning channel.

    TTFT must therefore fire on the first delta of any kind; requiring answer
    content silently reports ``ttft_ms = None`` for every measured turn.
    """

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"prefix_cache": {"mode": "off", "stats": {"hits": 0, "misses": 0}}}

    @app.post("/v1/chat/completions")
    def chat() -> Any:
        def stream() -> Any:
            yield (
                'data: {"choices":[{"delta":{"reasoning_content":"hmm"},'
                '"hipengine":{"timing":{"elapsed_ms":2.0,"ttft_ms":95.0}}}]}\n\n'
            )
            yield 'data: {"choices":[{"delta":{"reasoning_content":"more"}}]}\n\n'
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"length",'
                '"hipengine":{"decode_state":{"prompt_tokens":300,'
                '"generated_tokens":2,"reasoning_tokens":2,"answer_tokens":0}}}]}\n\n'
            )
            yield 'data: {"choices":[],"usage":{"prompt_tokens":300,"completion_tokens":2}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    with TestClient(app) as client:
        record = _run_turn_server(
            client,
            llm=_StubLLM(),  # type: ignore[arg-type]
            engine=None,
            messages=[{"role": "user", "content": "think hard"}],
            tools=None,
            max_tokens=2,
        )

    assert record["ttft_ms"] == 95.0
    assert record["text"] == ""
    assert record["reasoning_chars"] == 7
    assert record["reasoning_tokens"] == 2
    assert record["answer_tokens"] == 0
    assert record["output_tokens"] == 2
    assert record["prefix"]["fallback_reason"] is None


def test_merge_pairs_modes_by_sequence_position(tmp_path: Any) -> None:
    """A mirrored run (off, radix, radix, off) merges into two comparable pairs."""

    from scripts.prefix_cache_multiturn_bench import _merge_arm_artifacts

    def write(name: str, mode: str, wall_ms: float) -> Any:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "created_at": name,
                    "hardware": {"label": "test"},
                    "model": {"path": "m"},
                    "repo": {"revision": "r"},
                    "protocol": {"transport": "server"},
                    "modes": {
                        mode: {
                            "repetitions": [
                                {
                                    "repetition": 0,
                                    "mode": mode,
                                    "summary": {
                                        "wall_ms": wall_ms,
                                        "hits": 0,
                                        "lookups": 0,
                                        "reused_tokens": 0,
                                        "output_tokens_per_second": 10.0,
                                        "fallback_reasons": {},
                                        "by_kind": {},
                                        "ttft_ms": {"median": wall_ms / 10.0},
                                    },
                                }
                            ]
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    off_first = write("a.json", "off", 100.0)
    radix_first = write("b.json", "radix", 110.0)
    radix_last = write("c.json", "radix", 120.0)
    off_last = write("d.json", "off", 90.0)

    merged = _merge_arm_artifacts(
        [off_first, radix_first, radix_last, off_last], output={}
    )

    assert [arm["repetition"] for arm in merged["modes"]["off"]["repetitions"]] == [0, 1]
    assert [arm["repetition"] for arm in merged["modes"]["radix"]["repetitions"]] == [0, 1]
    assert merged["modes"]["off"]["repetitions"][0]["source_artifact"] == str(off_first)
    assert merged["modes"]["off"]["repetitions"][1]["source_artifact"] == str(off_last)
    assert merged["protocol"]["transport"] == "server"
    assert merged["protocol"]["repetitions"] == 2
    assert merged["protocol"]["modes"] == ["off", "radix"]
    # Pair 0 is first-arm vs first-arm, pair 1 is last-arm vs last-arm.
    assert merged["comparison"]["repetition_0"]["wall_delta_percent"] == 10.0
    assert merged["comparison"]["repetition_1"]["wall_delta_percent"] == pytest.approx(
        100.0 * (120.0 - 90.0) / 90.0
    )
    assert len(merged["protocol"]["merged_from"]) == 4


def test_ttft_is_taken_from_a_tool_call_stream() -> None:
    """Agentic lanes answer with tool calls, which are token events too."""

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"prefix_cache": {"mode": "off", "stats": {"hits": 0, "misses": 0}}}

    @app.post("/v1/chat/completions")
    def chat() -> Any:
        def stream() -> Any:
            yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"id":"call_1","type":"function","function":{"name":"read_file",'
                '"arguments":"{"}}]},'
                '"hipengine":{"timing":{"elapsed_ms":9.0,"ttft_ms":880.0}}}]}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"\\"path\\": \\"a.py\\"}"}}]}}]}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            )
            yield 'data: {"choices":[],"usage":{"prompt_tokens":900,"completion_tokens":7}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    with TestClient(app) as client:
        record = _run_turn_server(
            client,
            llm=_StubLLM(),  # type: ignore[arg-type]
            engine=None,
            messages=[{"role": "user", "content": "read a.py"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
            max_tokens=7,
        )

    assert record["ttft_ms"] == 880.0
    assert record["tool_call_chunks"] == 2
    assert record["text"] == ""
    assert record["finish_reason"] == "tool_calls"
    assert record["output_tokens"] == 7


def test_ttft_comes_from_server_timing_not_client_arrival() -> None:
    """TestClient buffers ASGI body chunks, so client arrival is not TTFT.

    The regression this pins: with a buffered client every turn reported
    ``ttft_ms`` equal to its own wall time (ratio 1.000 on all 164 measured
    turns of the 2026-09-18 server A/B), which is not a first-token latency.
    """

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"prefix_cache": {"mode": "off", "stats": {"hits": 0, "misses": 0}}}

    @app.post("/v1/chat/completions")
    def chat() -> Any:
        def stream() -> Any:
            time.sleep(0.05)  # the whole body is produced, then delivered at once
            yield (
                'data: {"choices":[{"delta":{"content":"a"},'
                '"hipengine":{"timing":{"elapsed_ms":50.0,"ttft_ms":7.0}}}]}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{"content":"b"},'
                '"hipengine":{"timing":{"elapsed_ms":50.0,"ttft_ms":7.0}}}]}\n\n'
            )
            yield 'data: {"choices":[],"usage":{"prompt_tokens":300,"completion_tokens":2}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    with TestClient(app) as client:
        record = _run_turn_server(
            client,
            llm=_StubLLM(),  # type: ignore[arg-type]
            engine=None,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=2,
        )

    assert record["ttft_ms"] == 7.0
    assert record["ttft_source"] == "server"
    # The client-observed value is kept only as a buffering diagnostic.
    assert record["client_observed_ttft_ms"] > record["ttft_ms"]
    assert record["wall_ms"] >= record["client_observed_ttft_ms"]


def test_finish_details_explain_a_suppressed_stream() -> None:
    """The server discards an invalid tool call and emits no tokens.

    Such a turn has no TTFT at all, so the artifact must carry the server's
    reason instead of leaving ``ttft_ms = None`` unexplained.
    """

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"prefix_cache": {"mode": "off", "stats": {"hits": 0, "misses": 0}}}

    @app.post("/v1/chat/completions")
    def chat() -> Any:
        def stream() -> Any:
            yield (
                'data: {"choices":[{"delta":{"role":"assistant"},'
                '"finish_reason":null}],'
                '"hipengine":{"timing":{"elapsed_ms":1.0}}}\n\n'
            )
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop",'
                '"finish_details":{"reason":"invalid_tool_call",'
                '"cache_action":"append_none"}}],'
                '"hipengine":{"timing":{"elapsed_ms":5000.0}}}\n\n'
            )
            yield 'data: {"choices":[],"usage":{"prompt_tokens":2552,"completion_tokens":24}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    with TestClient(app) as client:
        record = _run_turn_server(
            client,
            llm=_StubLLM(),  # type: ignore[arg-type]
            engine=None,
            messages=[{"role": "user", "content": "read a file"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
            max_tokens=24,
        )

    assert record["ttft_ms"] is None
    assert record["server_elapsed_ms"] == 5000.0
    assert record["finish_reason"] == "stop"
    assert record["finish_details"] == {
        "reason": "invalid_tool_call",
        "cache_action": "append_none",
    }
    assert record["output_tokens"] == 24


def test_prefix_counter_delta_differences_counters_and_copies_gauges() -> None:
    from scripts.prefix_cache_multiturn_bench import _prefix_counter_delta

    before = {
        "mode": "radix",
        "snapshot_entries": 1,
        "snapshot_limit": 1,
        "retained_snapshot_entries": 1,
        "snapshot_bytes": 66846720,
        "resident_bytes": 87818240,
        "snapshot_captures": 7,
        "state_clone_bytes": 0,
        "stats": {"hits": 2, "misses": 5},
    }
    after = {
        "mode": "radix",
        "snapshot_entries": 1,
        "snapshot_limit": 1,
        "retained_snapshot_entries": 0,
        "snapshot_bytes": 66846720,
        "resident_bytes": 66846720,
        "snapshot_captures": 9,
        "state_clone_bytes": 66846720,
        "stats": {"hits": 3, "misses": 5},
    }

    delta = _prefix_counter_delta(before, after)

    # Counters are per-turn work.
    assert delta["snapshot_captures"] == 2
    assert delta["state_clone_bytes"] == 66846720
    assert delta["stats"] == {"hits": 1, "misses": 0}
    # Gauges are levels: an unchanged level must not read as zero.
    assert delta["snapshot_entries"] == 1
    assert delta["snapshot_limit"] == 1
    assert delta["snapshot_bytes"] == 66846720
    assert delta["resident_bytes"] == 66846720
    assert delta["retained_snapshot_entries"] == 0


def test_common_prefix_len_counts_shared_tokens_only() -> None:
    from scripts.prefix_cache_multiturn_bench import _common_prefix_len

    assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert _common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert _common_prefix_len([1, 2, 3], [9, 2, 3]) == 0
    assert _common_prefix_len([], [1, 2]) == 0
    # A cumulative client resends the previous prompt verbatim: the LCP is the
    # whole previous prompt, not the new one.
    assert _common_prefix_len([1, 2, 3], [1, 2, 3, 4, 5]) == 3


def test_lane_records_prompt_lcp_against_the_previous_turn(monkeypatch: Any) -> None:
    """The lane loop records how much of the previous prompt is resent."""

    from scripts import prefix_cache_multiturn_bench as bench

    prompts = {"one": [1, 2, 3], "one plus": [1, 2, 3, 4], "diverged": [1, 9, 3, 4]}
    monkeypatch.setattr(
        bench,
        "_render_messages",
        lambda messages, **kwargs: str(messages[-1]["content"]),
    )

    class _TokenizingLLM:
        def tokenize(self, text: str) -> list[int]:
            return list(prompts[text])

    def run_turn(**kwargs: Any) -> dict[str, Any]:
        return {
            "render_ms": 1.0,
            "wall_ms": 10.0,
            "ttft_ms": 5.0,
            "prefill_ms": 1.0,
            "decode_ms": 1.0,
            "prompt_tokens": 3,
            "generator_prompt_tokens": 3,
            "output_tokens": 1,
            "gtt_bytes_before": None,
            "gtt_bytes_after": None,
            "text": "reply",
            "prefix": {},
        }

    lane = {
        "kind": "sharegpt",
        "system": "sys",
        "user_turns": ["one", "one plus", "diverged"],
        "max_tokens": 4,
    }
    result = bench._run_lane(
        _TokenizingLLM(),  # type: ignore[arg-type]
        engine=None,
        lane_id="lane",
        lane=lane,
        turn_limit=None,
        run_turn=run_turn,
    )

    turns = result.turns
    assert "prompt_lcp_tokens" not in turns[0]
    assert turns[1]["prompt_lcp_tokens"] == 3
    assert turns[1]["prompt_lcp_reusable_tokens"] == 0
    assert turns[2]["prompt_lcp_tokens"] == 1
    assert turns[2]["previous_prompt_tokens"] == 4


def test_artifact_prefix_telemetry_is_json_serializable() -> None:
    """The recorded per-turn block must survive ``json.dump`` for the artifact."""

    client, _state = _build_server_client({"mode": "radix", "stats": {"hits": 0, "misses": 0}})
    record = _run_turn_server(
        client,
        llm=_StubLLM(),  # type: ignore[arg-type]
        engine=None,
        messages=[{"role": "user", "content": "serialize me"}],
        tools=None,
        max_tokens=2,
    )

    payload = json.dumps({key: value for key, value in record.items() if key != "text"})
    assert json.loads(payload)["prefix"]["mode"] == "radix"


# ---------------------------------------------------------------------------
# Multi-arm subprocess orchestration
# ---------------------------------------------------------------------------
#
# A second prefix-cache engine in one process faulted the GPU on 2026-09-18
# (docs/REFACTOR.md), so a run with more than one arm must execute one arm per
# subprocess and merge the artifacts. These tests replace the child process with
# a writer so orchestration, merge, and failure propagation are deterministic
# and load no model.


def _write_arm_artifact(
    path: Any,
    *,
    mode: str,
    wall_ms: float,
    transport: str = "server",
) -> None:
    """Write the one-arm artifact shape ``_merge_arm_artifacts`` consumes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "created_at": target.name,
                "hardware": {"label": "test"},
                "model": {"path": "m"},
                "repo": {"revision": "r"},
                "protocol": {
                    "transport": transport,
                    "modes": [mode],
                    "repetitions": 1,
                    "arm_order": "sequential",
                    "arms": [{"repetition": 0, "mode": mode}],
                },
                "modes": {
                    mode: {
                        "repetitions": [
                            {
                                "repetition": 0,
                                "mode": mode,
                                "summary": {
                                    "wall_ms": wall_ms,
                                    "hits": 0,
                                    "lookups": 0,
                                    "reused_tokens": 0,
                                    "output_tokens_per_second": 10.0,
                                    "fallback_reasons": {},
                                    "by_kind": {},
                                    "ttft_ms": {"median": wall_ms / 10.0},
                                },
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _fake_arm_spawn(
    calls: list[list[str]],
    *,
    wall_ms: dict[str, float],
    returncode: int = 0,
    write_artifact: bool = True,
) -> Any:
    """A stand-in child that records its command and writes its arm artifact."""

    def spawn(command: Any) -> Any:
        argv = [str(token) for token in command]
        calls.append(argv)
        from scripts.prefix_cache_multiturn_bench import build_parser

        child = build_parser().parse_args(argv[2:])
        if write_artifact:
            mode = child.modes[0]
            _write_arm_artifact(
                child.json, mode=mode, wall_ms=wall_ms[mode], transport=child.transport
            )
        return subprocess.CompletedProcess(args=argv, returncode=returncode)

    return spawn


def _multi_arm_args(
    tmp_path: Any,
    *,
    repetitions: int = 2,
    arm_order: str = "alternating",
    extra: tuple[str, ...] = (),
) -> Any:
    from scripts.prefix_cache_multiturn_bench import build_parser

    return build_parser().parse_args(
        [
            "--transport",
            "server",
            "--modes",
            "off",
            "radix",
            "--repetitions",
            str(repetitions),
            "--arm-order",
            arm_order,
            "--json",
            str(Path(tmp_path) / "ab.json"),
            *extra,
        ]
    )


def test_multi_arm_run_executes_one_subprocess_per_arm(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from scripts.prefix_cache_multiturn_bench import (
        _arm_plan,
        _run_arm_subprocesses,
        build_parser,
    )

    args = _multi_arm_args(tmp_path)
    arms = _arm_plan(args.modes, repetitions=args.repetitions, arm_order=args.arm_order)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "scripts.prefix_cache_multiturn_bench._spawn_arm",
        _fake_arm_spawn(calls, wall_ms={"off": 100.0, "radix": 110.0}),
    )

    merged = _run_arm_subprocesses(args, output={}, arms=arms)

    # A mirrored plan: off, radix, radix, off, one child each.
    assert arms == [(0, "off"), (0, "radix"), (1, "radix"), (1, "off")]
    assert len(calls) == 4
    child_repetitions: list[int] = []
    for command in calls:
        child = build_parser().parse_args(command[2:])
        assert len(child.modes) == 1
        assert child.repetitions == 1
        assert child.arm_order == "sequential"
        assert child.transport == "server"
        child_repetitions.append(child.arm_repetition)
    # Each arm keeps its plan repetition, so the lane order a repetition calls
    # for is preserved across process boundaries.
    assert child_repetitions == [repetition for repetition, _mode in arms]
    artifacts = [Path(command[-1]) for command in calls]
    assert len(set(artifacts)) == 4
    assert all(path.parent == Path(tmp_path) / "ab-arms" for path in artifacts)
    assert artifacts[0].name == "arm00-off.json"
    assert artifacts[1].name == "arm01-radix.json"

    # The merge keeps one repetition per mode and pairs by plan position.
    assert len(merged["modes"]["off"]["repetitions"]) == 2
    assert len(merged["modes"]["radix"]["repetitions"]) == 2
    assert merged["comparison"]["repetition_0"]["wall_delta_percent"] == pytest.approx(
        10.0
    )
    assert merged["comparison"]["repetition_1"]["wall_delta_percent"] == pytest.approx(
        10.0
    )
    assert merged["protocol"]["repetitions"] == 2
    assert merged["protocol"]["arm_isolation"] == "one subprocess per arm"
    assert merged["protocol"]["arms"] == [
        {"repetition": repetition, "mode": mode} for repetition, mode in arms
    ]
    assert len(merged["protocol"]["merged_from"]) == 4
    assert merged["protocol"]["merged_from"][0]["path"] == str(artifacts[0])


def test_arm_child_command_forwards_measured_options(tmp_path: Any) -> None:
    from scripts.prefix_cache_multiturn_bench import _arm_child_command, build_parser

    parent = build_parser().parse_args(
        [
            "--model",
            "/models/x.gguf",
            "--backend",
            "hip_gfx1100",
            "--quant",
            "gguf_q4_k_m",
            "--transport",
            "server",
            "--lane-kinds",
            "code",
            "--turn-limit",
            "3",
            "--warmup-turns",
            "0",
            "--agentic-workloads",
            "small_repo",
            "growing_history",
            "--modes",
            "off",
            "radix",
            "--repetitions",
            "2",
            "--arm-order",
            "alternating",
            "--merge",
            "a.json",
            "b.json",
            "--baseline-artifact",
            "base.json",
            "--dump-sse",
            "d.sse",
            "--json",
            "out.json",
        ]
    )
    artifact = Path(tmp_path) / "arm.json"
    dump = Path(tmp_path) / "arm.sse"

    command = _arm_child_command(
        parent, mode="radix", repetition=1, artifact_path=artifact, dump_path=dump
    )

    assert command[0] == sys.executable
    assert command[1].endswith("prefix_cache_multiturn_bench.py")
    assert "--allow-multi-mode" not in command
    child = build_parser().parse_args(command[2:])
    # Arm selection is the parent's; everything measured is forwarded.
    assert child.modes == ["radix"]
    assert child.repetitions == 1
    assert child.arm_order == "sequential"
    assert child.arm_repetition == 1
    assert child.json == artifact
    assert child.dump_sse == dump
    assert child.merge is None
    assert child.baseline_artifact is None
    assert child.transport == "server"
    assert child.model == Path("/models/x.gguf")
    assert child.backend == "hip_gfx1100"
    assert child.quant == "gguf_q4_k_m"
    assert child.lane_kinds == "code"
    assert child.turn_limit == 3
    assert child.warmup_turns == 0
    assert child.agentic_workloads == ["small_repo", "growing_history"]


def test_multi_arm_run_requires_an_output_path(tmp_path: Any) -> None:
    from scripts.prefix_cache_multiturn_bench import _arm_plan, build_parser, _run_arm_subprocesses

    args = build_parser().parse_args(
        ["--transport", "server", "--modes", "off", "radix"]
    )
    arms = _arm_plan(args.modes, repetitions=args.repetitions, arm_order=args.arm_order)

    with pytest.raises(SystemExit, match="--json"):
        _run_arm_subprocesses(args, output={}, arms=arms)


def test_failed_arm_propagates_the_child_exit_status(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    from scripts.prefix_cache_multiturn_bench import _arm_plan, _run_arm_subprocesses

    args = _multi_arm_args(tmp_path, repetitions=1, arm_order="sequential")
    arms = _arm_plan(["off", "radix"], repetitions=1, arm_order="sequential")
    calls: list[list[str]] = []

    def spawn(command: Any) -> Any:
        argv = [str(token) for token in command]
        calls.append(argv)
        from scripts.prefix_cache_multiturn_bench import build_parser

        child = build_parser().parse_args(argv[2:])
        if child.modes == ["off"]:
            _write_arm_artifact(child.json, mode="off", wall_ms=100.0)
            return subprocess.CompletedProcess(args=argv, returncode=0)
        return subprocess.CompletedProcess(args=argv, returncode=7)

    monkeypatch.setattr("scripts.prefix_cache_multiturn_bench._spawn_arm", spawn)

    with pytest.raises(SystemExit) as exit_info:
        _run_arm_subprocesses(args, output={}, arms=arms)

    assert exit_info.value.code == 7
    assert len(calls) == 2
    stderr = capsys.readouterr().err
    assert "mode=radix" in stderr
    assert "exit status 7" in stderr
    assert "stopping before the merge" in stderr
    assert not (Path(tmp_path) / "ab.json").exists()


def test_arm_that_writes_no_artifact_is_an_error(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from scripts.prefix_cache_multiturn_bench import _arm_plan, _run_arm_subprocesses

    args = _multi_arm_args(tmp_path, repetitions=1, arm_order="sequential")
    arms = _arm_plan(["off", "radix"], repetitions=1, arm_order="sequential")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "scripts.prefix_cache_multiturn_bench._spawn_arm",
        _fake_arm_spawn(calls, wall_ms={"off": 100.0}, write_artifact=False),
    )

    with pytest.raises(SystemExit, match="wrote no artifact"):
        _run_arm_subprocesses(args, output={}, arms=arms)

    assert len(calls) == 1


def test_dump_sse_is_concatenated_in_arm_order(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from scripts.prefix_cache_multiturn_bench import _arm_plan, _run_arm_subprocesses

    dump = Path(tmp_path) / "dump.sse"
    args = _multi_arm_args(
        tmp_path,
        repetitions=1,
        arm_order="sequential",
        extra=("--dump-sse", str(dump)),
    )
    arms = _arm_plan(["off", "radix"], repetitions=1, arm_order="sequential")
    calls: list[list[str]] = []

    def spawn(command: Any) -> Any:
        argv = [str(token) for token in command]
        calls.append(argv)
        from scripts.prefix_cache_multiturn_bench import build_parser

        child = build_parser().parse_args(argv[2:])
        mode = child.modes[0]
        _write_arm_artifact(child.json, mode=mode, wall_ms=100.0)
        assert child.dump_sse is not None
        Path(child.dump_sse).write_text(f"data: {mode}\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=argv, returncode=0)

    monkeypatch.setattr("scripts.prefix_cache_multiturn_bench._spawn_arm", spawn)

    _run_arm_subprocesses(args, output={}, arms=arms)

    assert len(calls) == 2
    text = dump.read_text(encoding="utf-8")
    assert text.index("### arm r0 mode=off") < text.index("### arm r0 mode=radix")
    assert "data: off" in text
    assert "data: radix" in text


def test_isolation_is_required_for_every_additional_arm_load() -> None:
    from scripts.prefix_cache_multiturn_bench import _arm_plan, _needs_isolated_arms

    def plan(modes: list[str], repetitions: int, arm_order: str = "sequential") -> Any:
        return _arm_plan(modes, repetitions=repetitions, arm_order=arm_order)

    # The server transport loads once per arm, so a second repetition is a
    # second engine load in the same process.
    assert _needs_isolated_arms(
        transport="server", arms=plan(["radix"], 2), modes=["radix"]
    )
    # One arm in one process is the unchanged single-arm path.
    assert not _needs_isolated_arms(
        transport="server", arms=plan(["off"], 1), modes=["off"]
    )
    # The in-process transport loads once per mode and repeats inside that load.
    assert not _needs_isolated_arms(
        transport="inprocess", arms=plan(["radix"], 3), modes=["radix"]
    )
    assert _needs_isolated_arms(
        transport="inprocess", arms=plan(["off", "radix"], 1), modes=["off", "radix"]
    )


def test_run_dispatches_a_multi_arm_plan_to_subprocesses(
    tmp_path: Any, monkeypatch: Any
) -> None:
    import scripts.prefix_cache_multiturn_bench as harness

    model = Path(tmp_path) / "m.gguf"
    model.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(harness, "_repo_state", lambda: {"revision": "r"})
    monkeypatch.setattr(
        harness, "collect_artifact_provenance", lambda **_kwargs: {}
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        harness,
        "_spawn_arm",
        _fake_arm_spawn(calls, wall_ms={"off": 100.0, "radix": 110.0}),
    )

    args = harness.build_parser().parse_args(
        [
            "--model",
            str(model),
            "--transport",
            "server",
            "--modes",
            "off",
            "radix",
            "--json",
            str(Path(tmp_path) / "ab.json"),
        ]
    )
    result = harness.run(args)

    assert len(calls) == 2
    assert result["protocol"]["arm_isolation"] == "one subprocess per arm"
    assert sorted(result["modes"]) == ["off", "radix"]
    assert result["comparison"]["repetition_0"]["wall_delta_percent"] == pytest.approx(
        10.0
    )


def test_run_keeps_a_single_arm_in_this_process(
    tmp_path: Any, monkeypatch: Any
) -> None:
    import scripts.prefix_cache_multiturn_bench as harness

    model = Path(tmp_path) / "m.gguf"
    model.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(harness, "_repo_state", lambda: {"revision": "r"})
    monkeypatch.setattr(
        harness, "collect_artifact_provenance", lambda **_kwargs: {}
    )
    recorded: dict[str, Any] = {}

    def fake_arm(
        args: Any,
        *,
        output: Any,
        model_path: Any,
        quant: str,
        mode: str,
        repetition: int,
    ) -> None:
        recorded.update(
            {"mode": mode, "repetition": repetition, "quant": quant, "output": output}
        )

    def no_spawn(command: Any) -> Any:  # pragma: no cover - must not be reached
        raise AssertionError("a single arm must not spawn a subprocess")

    monkeypatch.setattr(harness, "_run_server_arm", fake_arm)
    monkeypatch.setattr(harness, "_spawn_arm", no_spawn)

    args = harness.build_parser().parse_args(
        ["--model", str(model), "--transport", "server", "--modes", "radix"]
    )
    result = harness.run(args)

    assert recorded["mode"] == "radix"
    assert recorded["repetition"] == 0
    assert result["comparison"] == {}


def test_retired_allow_multi_mode_flag_is_ignored_with_a_note(capsys: Any) -> None:
    from scripts.prefix_cache_multiturn_bench import _strip_retired_flags

    argv = ["--transport", "server", "--modes", "off", "radix"]
    assert _strip_retired_flags([*argv, "--allow-multi-mode"]) == argv
    assert "retired" in capsys.readouterr().err


def test_plan_repetitions_pins_one_arm_slot() -> None:
    from scripts.prefix_cache_multiturn_bench import _plan_repetitions, build_parser

    parser = build_parser()
    assert _plan_repetitions(parser.parse_args(["--repetitions", "3"])) == [0, 1, 2]
    assert _plan_repetitions(
        parser.parse_args(["--repetitions", "1", "--arm-repetition", "1"])
    ) == [1]


def test_arm_repetition_reaches_the_single_arm_measurement(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The child's plan slot decides the lane order, not its process-local 0."""

    import scripts.prefix_cache_multiturn_bench as harness

    model = Path(tmp_path) / "m.gguf"
    model.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(harness, "_repo_state", lambda: {"revision": "r"})
    monkeypatch.setattr(harness, "collect_artifact_provenance", lambda **_kwargs: {})
    recorded: dict[str, Any] = {}

    def fake_arm(
        args: Any,
        *,
        output: Any,
        model_path: Any,
        quant: str,
        mode: str,
        repetition: int,
    ) -> None:
        recorded.update({"mode": mode, "repetition": repetition})

    monkeypatch.setattr(harness, "_run_server_arm", fake_arm)
    args = harness.build_parser().parse_args(
        [
            "--model",
            str(model),
            "--transport",
            "server",
            "--modes",
            "radix",
            "--repetitions",
            "1",
            "--arm-repetition",
            "1",
        ]
    )

    harness.run(args)

    assert recorded == {"mode": "radix", "repetition": 1}


_STUB_ARM_SCRIPT_TEMPLATE = '''
import argparse
import json
import os
from pathlib import Path

LOG = __LOG__
EXIT_CODE = __EXIT_CODE__

parser = argparse.ArgumentParser()
parser.add_argument("--json", type=Path)
parser.add_argument("--modes", nargs="*", default=[])
parser.add_argument("--dump-sse", type=Path, default=None)
args, _unknown = parser.parse_known_args()
mode = args.modes[0] if args.modes else "?"

if LOG:
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(os.getcwd() + "\\t" + mode + "\\n")
if args.dump_sse is not None:
    args.dump_sse.parent.mkdir(parents=True, exist_ok=True)
    args.dump_sse.write_text("data: " + mode + "\\n", encoding="utf-8")
if EXIT_CODE:
    raise SystemExit(EXIT_CODE)
args.json.parent.mkdir(parents=True, exist_ok=True)
args.json.write_text(json.dumps({
    "created_at": args.json.name,
    "hardware": {"label": "test"},
    "model": {"path": "m"},
    "repo": {"revision": "r"},
    "protocol": {"transport": "server", "modes": [mode]},
    "modes": {mode: {"repetitions": [{
        "repetition": 0,
        "mode": mode,
        "summary": {
            "wall_ms": 100.0,
            "hits": 0,
            "lookups": 0,
            "reused_tokens": 0,
            "output_tokens_per_second": 10.0,
            "fallback_reasons": {},
            "by_kind": {},
            "ttft_ms": {"median": 10.0},
        },
    }]}},
}), encoding="utf-8")
'''


def _stub_arm_script(
    tmp_path: Any, *, log: Any = None, exit_code: int = 0
) -> Path:
    """Write a stand-in child with this test's behaviour baked into the source.

    Each test gets its own copy, so the stub needs no environment variables and
    no arguments beyond the ones the harness forwards.
    """

    source = _STUB_ARM_SCRIPT_TEMPLATE.replace(
        "__LOG__", repr(None if log is None else str(log))
    ).replace("__EXIT_CODE__", str(int(exit_code)))
    script = Path(tmp_path) / "stub_arm_child.py"
    script.write_text(source, encoding="utf-8")
    return script


def test_real_subprocess_arms_are_launched_and_merged(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """Exercise ``_spawn_arm`` itself, not a replaced runner."""

    import scripts.prefix_cache_multiturn_bench as harness

    log = Path(tmp_path) / "arms.log"
    monkeypatch.setattr(harness, "_SCRIPT_PATH", _stub_arm_script(tmp_path, log=log))

    args = _multi_arm_args(tmp_path, repetitions=1, arm_order="sequential")
    arms = harness._arm_plan(["off", "radix"], repetitions=1, arm_order="sequential")

    merged = harness._run_arm_subprocesses(args, output={}, arms=arms)

    logged = [
        line.split("\t")[1]
        for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert logged == ["off", "radix"]
    assert sorted(merged["modes"]) == ["off", "radix"]
    assert merged["protocol"]["arm_isolation"] == "one subprocess per arm"
    assert "repetition_0" in merged["comparison"]


def test_real_subprocess_arm_failure_propagates_its_status(
    tmp_path: Any, monkeypatch: Any
) -> None:
    import scripts.prefix_cache_multiturn_bench as harness

    monkeypatch.setattr(
        harness, "_SCRIPT_PATH", _stub_arm_script(tmp_path, exit_code=3)
    )

    args = _multi_arm_args(tmp_path, repetitions=1, arm_order="sequential")
    arms = harness._arm_plan(["off", "radix"], repetitions=1, arm_order="sequential")

    with pytest.raises(SystemExit) as exit_info:
        harness._run_arm_subprocesses(args, output={}, arms=arms)

    assert exit_info.value.code == 3


def test_preexisting_arm_artifact_is_not_merged_when_the_child_writes_nothing(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """A leftover from an earlier run must not satisfy the exists() check."""

    from scripts.prefix_cache_multiturn_bench import (
        _arm_artifact_path,
        _arm_plan,
        _run_arm_subprocesses,
    )

    args = _multi_arm_args(tmp_path, repetitions=1, arm_order="sequential")
    arms = _arm_plan(["off", "radix"], repetitions=1, arm_order="sequential")
    stale = _arm_artifact_path(Path(tmp_path) / "ab.json", index=0, mode="off")
    _write_arm_artifact(stale, mode="off", wall_ms=999.0)

    calls: list[list[str]] = []
    monkeypatch.setattr(
        "scripts.prefix_cache_multiturn_bench._spawn_arm",
        _fake_arm_spawn(calls, wall_ms={"off": 100.0}, write_artifact=False),
    )

    with pytest.raises(SystemExit, match="wrote no artifact"):
        _run_arm_subprocesses(args, output={}, arms=arms)

    assert len(calls) == 1
    assert not stale.exists()


def test_preexisting_arm_artifact_is_replaced_by_this_runs_output(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from scripts.prefix_cache_multiturn_bench import (
        _arm_artifact_path,
        _arm_plan,
        _run_arm_subprocesses,
    )

    args = _multi_arm_args(tmp_path, repetitions=1, arm_order="sequential")
    arms = _arm_plan(["off", "radix"], repetitions=1, arm_order="sequential")
    stale = _arm_artifact_path(Path(tmp_path) / "ab.json", index=0, mode="off")
    _write_arm_artifact(stale, mode="off", wall_ms=999.0)

    calls: list[list[str]] = []
    monkeypatch.setattr(
        "scripts.prefix_cache_multiturn_bench._spawn_arm",
        _fake_arm_spawn(calls, wall_ms={"off": 100.0, "radix": 110.0}),
    )

    merged = _run_arm_subprocesses(args, output={}, arms=arms)

    assert merged["modes"]["off"]["repetitions"][0]["summary"]["wall_ms"] == 100.0
    assert merged["comparison"]["repetition_0"]["wall_delta_percent"] == pytest.approx(
        10.0
    )


def test_arms_run_in_the_invocation_directory(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """Relative paths must resolve where the caller stood, not at the repo root."""

    import scripts.prefix_cache_multiturn_bench as harness

    log = Path(tmp_path) / "arms.log"
    monkeypatch.setattr(harness, "_SCRIPT_PATH", _stub_arm_script(tmp_path, log=log))
    monkeypatch.chdir(tmp_path)

    args = harness.build_parser().parse_args(
        [
            "--transport",
            "server",
            "--modes",
            "off",
            "radix",
            "--repetitions",
            "1",
            "--arm-order",
            "sequential",
            "--json",
            "ab.json",
        ]
    )
    arms = harness._arm_plan(["off", "radix"], repetitions=1, arm_order="sequential")

    merged = harness._run_arm_subprocesses(args, output={}, arms=arms)

    child_cwds = [
        Path(line.split("\t")[0]).resolve()
        for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert child_cwds == [Path(tmp_path).resolve(), Path(tmp_path).resolve()]
    assert (tmp_path / "ab-arms" / "arm00-off.json").exists()
    assert (tmp_path / "ab-arms" / "arm01-radix.json").exists()
    assert merged["protocol"]["merged_from"][0]["path"] == "ab-arms/arm00-off.json"
    assert sorted(merged["modes"]) == ["off", "radix"]


def test_invalid_repetition_counts_are_rejected(tmp_path: Any) -> None:
    import scripts.prefix_cache_multiturn_bench as harness

    model = Path(tmp_path) / "m.gguf"
    model.write_text("stub", encoding="utf-8")
    parser = harness.build_parser()

    for count in ("0", "-2"):
        args = parser.parse_args(["--model", str(model), "--repetitions", count])
        with pytest.raises(SystemExit, match="at least 1"):
            harness.run(args)

    args = parser.parse_args(
        ["--model", str(model), "--repetitions", "1", "--arm-repetition", "-1"]
    )
    with pytest.raises(SystemExit, match="zero or greater"):
        harness.run(args)
