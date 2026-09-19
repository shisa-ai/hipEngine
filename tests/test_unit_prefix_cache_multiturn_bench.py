"""Unit tests for the prefix-cache multi-turn benchmark harness.

Covers the serving-transport contract: per-turn prefix telemetry differenced
from the server's aggregate counters, TTFT attribution over SSE, warmup
separation, and alternating A/B arm order. The GPU path is exercised by the
script itself; these tests pin the measurement plumbing with a stub engine.
"""

from __future__ import annotations

import json
import time
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
    assert args.allow_multi_mode is False
    server = parser.parse_args(
        ["--transport", "server", "--arm-order", "alternating", "--allow-multi-mode"]
    )
    assert server.transport == "server"
    assert server.arm_order == "alternating"
    assert server.allow_multi_mode is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--transport", "bogus"])


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
