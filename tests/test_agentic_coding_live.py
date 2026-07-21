from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import pytest

from hipengine.benchmark.agentic import (
    AgenticBenchmarkError,
    build_agentic_benchmark_artifact,
    load_agentic_workload_suite,
)
from hipengine.benchmark.agentic_live import (
    build_canonical_turn_messages,
    build_openai_tools,
    final_ownership_from_server,
    normalize_chat_oracle,
    normalize_chat_sse_turn,
    render_workload_prefix,
)
from scripts.agentic_coding_live import collect_live_records


WORKLOADS = Path("benchmarks/prompts/agentic-coding-v1.json")


def _byte_tokenize(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _byte_detokenize(token_ids) -> str:
    return bytes(int(token) for token in token_ids).decode("utf-8")


def _oracle_payload(suite, *, turn_index: int = 0) -> dict[str, object]:
    turn = suite.workloads["small_repo"]["turns"][turn_index]
    arguments = json.dumps(turn["expected_arguments"], separators=(",", ":"))
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "oracle-call",
                            "type": "function",
                            "function": {
                                "name": turn["expected_tool"],
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                "hipengine": {
                    "generated_token_ids": [11, 12, 13],
                    "generated_tokens": 3,
                },
            }
        ],
        "usage": {"prompt_tokens": 2048, "completion_tokens": 3, "total_tokens": 2051},
    }


def _sse_events(suite, *, turn_index: int = 0, response_ids: bool = False):
    turn = suite.workloads["small_repo"]["turns"][turn_index]
    arguments = json.dumps(turn["expected_arguments"], separators=(",", ":"))
    midpoint = len(arguments) // 2
    telemetry = {
        "phase": "tool_call",
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 2048,
            "generated_tokens": 3,
            "phase": "tool_call",
            "continuation_eligible": False,
            "sampler_mode": "processed_argmax",
            "logits_d2h_bytes": 0,
            "serial_decode_fallback": False,
        },
        "timing": {"prefill_ms": 20.0, "decode_ms": 6.0},
        "timing_scope": "batch",
        "batch_id": "batch-7",
        "group_rows": 2,
        "timing_owner": True,
    }
    events = [
        (10.01, {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
        (
            10.20,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "live-call",
                                    "type": "function",
                                    "function": {
                                        "name": turn["expected_tool"],
                                        "arguments": arguments[:midpoint],
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                        "hipengine": copy.deepcopy(telemetry),
                    }
                ],
                "hipengine": {"event": "tool_call", "timing": {"elapsed_ms": 200.0}},
            },
        ),
        (
            10.25,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "live-call",
                                    "type": "function",
                                    "function": {"arguments": arguments[midpoint:]},
                                }
                            ]
                        },
                        "finish_reason": None,
                        "hipengine": copy.deepcopy(telemetry),
                    }
                ],
                "hipengine": {"event": "tool_call", "timing": {"elapsed_ms": 250.0}},
            },
        ),
        (
            10.30,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                        "finish_details": {"reason": "tool_calls", "cache_action": "append_none"},
                        "hipengine": copy.deepcopy(telemetry),
                    }
                ],
                "hipengine": {
                    "event": "done",
                    "timing": {"elapsed_ms": 300.0},
                    "kv_pool": {"refcounted_pages": 0, "pinned_pages": 0},
                },
            },
        ),
        (
            10.31,
            {
                "choices": [],
                "usage": {"prompt_tokens": 2048, "completion_tokens": 3, "total_tokens": 2051},
                "hipengine": {"event": "usage"},
            },
        ),
        (10.32, "[DONE]"),
    ]
    if response_ids:
        events[3][1]["choices"][0]["hipengine"]["generated_token_ids"] = [11, 12, 13]
        events[3][1]["choices"][0]["hipengine"]["generated_tokens"] = 3
    return events


def test_prefix_renderer_hits_exact_target_and_roundtrips() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)

    rendered = render_workload_prefix(
        suite,
        "small_repo",
        tokenize=_byte_tokenize,
        detokenize=_byte_detokenize,
    )

    assert rendered.target_tokens == 2048
    assert len(rendered.token_ids) == 2048
    assert _byte_tokenize(rendered.text) == list(rendered.token_ids)
    assert len(rendered.token_ids_sha256) == 64
    assert "acme-router" in rendered.text


def test_canonical_messages_replay_prior_tool_turns_with_stable_ids() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    prefix = "synthetic repository prefix"

    messages = build_canonical_turn_messages(
        suite,
        "small_repo",
        turn_index=2,
        agent_id="agent-3",
        prefix_text=prefix,
    )

    assert messages[0]["role"] == "system"
    assert prefix in messages[0]["content"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    first_call = messages[2]["tool_calls"][0]
    assert first_call["id"] == "call_agent-3_0"
    assert (
        json.loads(first_call["function"]["arguments"])
        == suite.workloads["small_repo"]["turns"][0]["expected_arguments"]
    )
    assert messages[3]["tool_call_id"] == first_call["id"]
    assert [tool["function"]["name"] for tool in build_openai_tools(suite)] == [
        "read",
        "grep",
        "run",
    ]


def test_oracle_and_buffered_tool_sse_normalize_to_exact_a0_record() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    oracle = normalize_chat_oracle(suite, "small_repo", 0, _oracle_payload(suite))

    record = normalize_chat_sse_turn(
        suite,
        workload_id="small_repo",
        turn_index=0,
        run_id="run-0",
        agent_id="agent-0",
        session_id="session-0",
        request_id="request-0",
        prompt_token_ids=[1] * 2048,
        submitted_at_s=10.0,
        tool_result_submitted_at_s=10.33,
        oracle=oracle,
        events=_sse_events(suite),
        cache_mode="off",
    )

    assert record["output"]["generated_token_ids"] == [11, 12, 13]
    assert record["output"]["generated_token_ids_source"] == "matched_nonstreaming_oracle"
    assert record["output"]["sse_exact_ids_observed"] is False
    assert record["tool"]["name"] == "read"
    assert record["tool"]["arguments"] == {"path": "pyproject.toml", "mode": "summary"}
    assert record["timing"] == {
        "submitted_at_s": 10.0,
        "first_token_at_s": 10.2,
        "token_observed_at_s": [10.2],
        "token_event_token_counts": [3],
        "token_timing_mode": "buffered_public",
        "tool_call_ready_at_s": 10.25,
        "response_done_at_s": 10.3,
        "tool_result_submitted_at_s": 10.33,
    }
    assert record["backend"] == {
        "batch_id": "batch-7",
        "timing_scope": "batch",
        "timing_owner": True,
        "sampler_mode": "processed_argmax",
        "logits_d2h_bytes": 0,
        "physical_width": 2,
        "serial_fallback": False,
    }
    assert record["prefix"] == {
        "lookup": False,
        "hit": False,
        "reused_tokens": 0,
        "cache_bytes": 0,
    }
    assert record["finish"] == {"reason": "tool_calls", "retry_count": 0}

    content_oracle = _oracle_payload(suite)
    content_oracle["choices"][0]["message"]["content"] = "unexpected"
    with pytest.raises(
        AgenticBenchmarkError,
        match="oracle emitted content alongside a tool-only response",
    ):
        normalize_chat_oracle(suite, "small_repo", 0, content_oracle)


def test_sse_normalizer_uses_exact_response_ids_and_rejects_oracle_drift() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    oracle = normalize_chat_oracle(suite, "small_repo", 0, _oracle_payload(suite))

    record = normalize_chat_sse_turn(
        suite,
        workload_id="small_repo",
        turn_index=0,
        run_id="run-0",
        agent_id="agent-0",
        session_id="session-0",
        request_id="request-0",
        prompt_token_ids=[1] * 2048,
        submitted_at_s=10.0,
        tool_result_submitted_at_s=10.33,
        oracle=oracle,
        events=_sse_events(suite, response_ids=True),
        cache_mode="off",
    )

    assert record["output"]["generated_token_ids"] == [11, 12, 13]
    assert record["output"]["generated_token_ids_source"] == "response"
    assert record["output"]["sse_exact_ids_observed"] is True
    assert record["timing"]["token_timing_mode"] == "buffered_public"

    drifted = _sse_events(suite, response_ids=True)
    drifted[3][1]["choices"][0]["hipengine"]["generated_token_ids"][-1] = 99
    with pytest.raises(AgenticBenchmarkError, match="SSE generated token IDs differ from oracle"):
        normalize_chat_sse_turn(
            suite,
            workload_id="small_repo",
            turn_index=0,
            run_id="run-0",
            agent_id="agent-0",
            session_id="session-0",
            request_id="request-0",
            prompt_token_ids=[1] * 2048,
            submitted_at_s=10.0,
            tool_result_submitted_at_s=10.33,
            oracle=oracle,
            events=drifted,
            cache_mode="off",
        )


def test_sse_normalizer_rejects_incomplete_or_ambiguous_tool_streams() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    oracle = normalize_chat_oracle(suite, "small_repo", 0, _oracle_payload(suite))

    cases = []
    no_done = _sse_events(suite)[:-1]
    cases.append((no_done, r"SSE stream is missing \[DONE\]"))

    no_batch = _sse_events(suite)
    for _, event in no_batch:
        if isinstance(event, dict):
            for choice in event.get("choices", []):
                choice.get("hipengine", {}).pop("batch_id", None)
    cases.append((no_batch, "SSE tool turn is missing backend batch_id"))

    wrong_args = _sse_events(suite)
    wrong_args[1][1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = (
        '{"path":"WRONG"'
    )
    cases.append((wrong_args, "SSE tool arguments are not valid JSON"))

    extra_content = _sse_events(suite)
    extra_content[0][1]["choices"][0]["delta"]["content"] = "unexpected"
    cases.append((extra_content, "SSE emitted content alongside a tool-only response"))

    for events, message in cases:
        with pytest.raises(AgenticBenchmarkError, match=message):
            normalize_chat_sse_turn(
                suite,
                workload_id="small_repo",
                turn_index=0,
                run_id="run-0",
                agent_id="agent-0",
                session_id="session-0",
                request_id="request-0",
                prompt_token_ids=[1] * 2048,
                submitted_at_s=10.0,
                tool_result_submitted_at_s=10.33,
                oracle=oracle,
                events=events,
                cache_mode="off",
            )


class _FakeLiveTransport:
    def __init__(self, suite) -> None:
        self.suite = suite
        self.oracle_calls = 0
        self.lock = threading.Lock()

    def capabilities(self):
        return {"cache": {"prefix_cache": "off"}}

    def tokenize(self, text):
        return _byte_tokenize(text)

    def detokenize(self, token_ids):
        return _byte_detokenize(token_ids)

    def rendered_prompt_ids(self, **_kwargs):
        return [1] * 128

    def chat_json(self, _payload):
        with self.lock:
            turn_index = (self.oracle_calls // 2) % 4
            self.oracle_calls += 1
        return _oracle_payload(self.suite, turn_index=turn_index)

    def chat_sse(self, payload, *, release=None):
        if release is not None:
            release.wait()
        current_user = payload["messages"][-1]["content"]
        turns = self.suite.workloads["small_repo"]["turns"]
        turn_index = next(index for index, turn in enumerate(turns) if turn["user"] == current_user)
        events = _sse_events(self.suite, turn_index=turn_index, response_ids=True)
        for _, event in events:
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices", []):
                telemetry = choice.get("hipengine", {})
                telemetry["timing_scope"] = "choice"
                telemetry.pop("batch_id", None)
                telemetry["group_rows"] = 1
                telemetry["timing_owner"] = True
        return 10.0, 10.32, events

    def ready(self):
        return {
            "ready": True,
            "queue": {"depth": 0, "worker_active": False, "active_requests": 0},
            "kv_capacity": {"pool": {"refcounted_pages": 0, "pinned_pages": 0}},
            "graph_cache": {"entries": 1},
        }

    def sessions(self):
        return {"sessions": [], "continuations": {"active": 0}}


def test_live_collector_builds_complete_multirun_records_with_fake_transport() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    transport = _FakeLiveTransport(suite)

    loaded_suite, records = collect_live_records(
        transport,
        workloads_path=WORKLOADS,
        workload_id="small_repo",
        model="fake-model",
        backend="fake",
        concurrency=2,
        runs=2,
        max_tokens=32,
        cache_mode="off",
        idle_timeout_s=1.0,
    )
    artifact = build_agentic_benchmark_artifact(loaded_suite, records)

    assert artifact["validation"]["passed"] is True
    assert artifact["coverage"] == {
        "workloads": ["small_repo"],
        "runs": 2,
        "concurrency": 2,
        "agents": 4,
        "turns": 16,
        "tool_calls": 16,
        "generated_tokens": 48,
        "batches": 16,
    }
    assert artifact["rollup"]["latency_ms"]["inter_token"]["count"] == 0
    assert artifact["rollup"]["backend"]["token_timing_mode_turns"] == {"buffered_public": 16}
    assert artifact["rollup"]["backend"]["generated_token_id_source_turns"] == {
        "response": 16
    }


def test_final_ownership_maps_ready_and_session_surfaces() -> None:
    ownership = final_ownership_from_server(
        {
            "ready": True,
            "queue": {"depth": 0, "worker_active": False, "active_requests": 0},
            "kv_capacity": {
                "pool": {
                    "refcounted_pages": 0,
                    "pinned_pages": 0,
                }
            },
            "graph_cache": {"entries": 4},
        },
        {"sessions": [], "continuations": {"active": 0}},
        cache_mode="off",
    )

    assert ownership == {
        "pending_requests": 0,
        "active_requests": 0,
        "stream_producers": 0,
        "model_active_requests": 0,
        "session_count": 0,
        "kv_refcounted_pages": 0,
        "kv_pinned_pages": 0,
        "graph_owners": 0,
        "workspace_owners": 0,
        "cache_resident_bytes": 0,
        "allowed_cache_bytes": 0,
    }

    with pytest.raises(AgenticBenchmarkError, match="server is not idle after agentic run"):
        final_ownership_from_server(
            {
                "ready": True,
                "queue": {"depth": 0, "worker_active": True, "active_requests": 1},
                "kv_capacity": {"pool": {"refcounted_pages": 1, "pinned_pages": 0}},
                "graph_cache": {"entries": 4},
            },
            {"sessions": []},
            cache_mode="off",
        )
