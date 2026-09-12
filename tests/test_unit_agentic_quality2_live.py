from __future__ import annotations

import copy
import json
from pathlib import Path

from hipengine.benchmark.agentic_quality2_sandbox import AgenticQuality2Sandbox
from scripts.agentic_quality2_live import collect_live_quality2_records

SUITE = Path("benchmarks/prompts/agentic-quality2-v2.json")


def _response(*, calls=(), content="", finish_reason=None):
    tool_calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": call["tool"],
                "arguments": json.dumps(
                    call["arguments"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            },
        }
        for index, call in enumerate(calls)
    ]
    raw = content or json.dumps(tool_calls, sort_keys=True, ensure_ascii=False)
    ids = [ord(char) for char in raw] or [32]
    reason = finish_reason or ("tool_calls" if tool_calls else "stop")
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": reason,
                "finish_details": {"reason": reason, "cache_action": "append_none"},
                "message": message,
                "hipengine": {"generated_token_ids": ids},
            }
        ],
    }


class _ExpandedTransport:
    def __init__(self, response_by_user):
        self.response_by_user = response_by_user
        self.requests = []

    def capabilities(self):
        return {
            "model": {"id": "fake-model", "backend": "fake"},
            "cache": {"prefix_cache": "off"},
            "tokenizer": {"tokenize": True, "detokenize": True},
            "features": {"tools": {"enabled": True, "strict_result_validation": True}},
        }

    def tokenize(self, text):
        return [ord(char) for char in str(text)]

    def detokenize(self, token_ids):
        return "".join(chr(int(token)) for token in token_ids)

    def rendered_prompt_ids(self, *, messages, tools, tool_choice):
        del tools, tool_choice
        return self.tokenize(json.dumps(messages, sort_keys=True, ensure_ascii=False))

    def chat_json(self, payload):
        self.requests.append(copy.deepcopy(dict(payload)))
        user = payload["messages"][-1]["content"]
        return copy.deepcopy(self.response_by_user[user])

    def ready(self):
        return {
            "ready": True,
            "queue": {"depth": 0, "worker_active": False, "active_requests": 0},
            "kv_capacity": {"pool": {"refcounted_pages": 7, "pinned_pages": 7}},
        }

    def sessions(self):
        return {"sessions": [], "continuations": {"active": 0}}


class _PassingSandbox:
    def run_code_case(self, **kwargs):
        return {
            "status": "passed",
            "tests_attempted": len(kwargs["hidden_tests"]),
            "tests_passed": len(kwargs["hidden_tests"]),
            "stdout": "",
            "stderr": "",
            "hidden_expected_exposed": False,
        }


def test_expanded_live_collector_handles_single_multiple_and_no_tool_rows() -> None:
    nested_user = (
        "Normalize station west-17 with high priority and tags night, edge; include metadata."
    )
    multiple_user = (
        "Retrieve the owner of west-17 and independently calculate 18 times 7. "
        "Use both appropriate tools."
    )
    irrelevant_user = (
        "Write a friendly two-line greeting. No repository or operations lookup is needed."
    )
    users = {
        nested_user: _response(
            calls=[
                {
                    "tool": "transform_record",
                    "arguments": {
                        "record": {
                            "station": "west-17",
                            "priority": "high",
                            "tags": ["night", "edge"],
                        },
                        "mode": "normalize",
                        "include_metadata": True,
                    },
                }
            ]
        ),
        multiple_user: _response(
            calls=[
                {"tool": "calculate", "arguments": {"expression": "18 * 7"}},
                {
                    "tool": "lookup_record",
                    "arguments": {"key": "station_owner", "locale": "en"},
                },
            ]
        ),
        irrelevant_user: _response(
            content="Hello there.\nThanks for visiting.",
            finish_reason="stop",
        ),
    }
    transport = _ExpandedTransport(users)
    checkpoints = []

    _suite, records, summary = collect_live_quality2_records(
        transport,
        suite_path=SUITE,
        workload_ids=(
            "aq2_dev_tool_nested_alert_en",
            "aq2_dev_tool_multiple_en",
            "aq2_dev_tool_irrelevant_en",
        ),
        model="fake-model",
        backend="fake",
        repetitions=2,
        max_tokens=192,
        cache_mode="off",
        idle_timeout_s=1.0,
        sandbox=AgenticQuality2Sandbox(),
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(dict(payload))),
    )

    assert len(records["records"]) == 6
    assert summary["aggregation"]["overall"]["passed"] == 6
    assert summary["aggregation"]["outcomes"] == {"passed": 6}
    assert summary["aggregation"]["determinism"] == {
        "basis": "normalized_response_v1",
        "evaluated": True,
        "mismatches": [],
        "passed": True,
    }
    assert summary["fail_safe_controls"]["passed"] is True
    assert summary["fail_safe_controls"]["passed_count"] == 10
    assert summary["fail_safe_controls"]["total"] == 10
    assert summary["quality_metrics"]["observations"] == {
        "attempted": 6,
        "terminal": 6,
    }
    assert summary["quality_metrics"]["tool_selection"]["correct"] == 6
    assert summary["quality_metrics"]["external_oracle"]["passed"] == 6
    assert summary["final_ownership"]["kv_refcounted_pages"] == 0
    assert summary["configuration"]["persistent_ownership_baseline"] == {
        "kv_refcounted_pages": 7,
        "kv_pinned_pages": 7,
    }
    assert len(checkpoints) == 8
    assert checkpoints[-2]["status"] == "controls_complete"
    assert checkpoints[-2]["fail_safe_controls"]["passed_count"] == 10
    assert checkpoints[-1]["status"] == "complete"
    assert checkpoints[-1]["progress"] == {"completed": 6, "total": 6}
    assert all(request["tool_choice"] == "auto" for request in transport.requests)
    assert [request["parallel_tool_calls"] for request in transport.requests[:3]] == [
        False,
        True,
        False,
    ]


def test_expanded_live_collector_seals_heldout_details_but_preserves_local_raw() -> None:
    user = "docs 内で「rollback snapshot」を検索してください。大文字小文字の指定は省略します。"
    transport = _ExpandedTransport(
        {
            user: _response(
                calls=[
                    {
                        "tool": "search_repo",
                        "arguments": {"path": "docs", "query": "wrong query"},
                    }
                ]
            )
        }
    )

    _suite, records, summary = collect_live_quality2_records(
        transport,
        suite_path=SUITE,
        workload_ids=("aq2_held_tool_optional_search_ja",),
        model="fake-model",
        backend="fake",
        repetitions=2,
        max_tokens=192,
        cache_mode="off",
        idle_timeout_s=1.0,
        sandbox=AgenticQuality2Sandbox(),
    )

    assert summary["aggregation"]["overall"]["failed"] == 2
    assert summary["aggregation"]["heldout_details"] == []
    assert summary["aggregation"]["heldout_details_sealed"] is True
    assert records["records"][0]["calls"][0]["arguments"]["query"] == "wrong query"
    assert "wrong query" not in json.dumps(summary, ensure_ascii=False)


def test_expanded_live_collector_detects_distinct_passing_responses() -> None:
    class AlternatingTransport(_ExpandedTransport):
        def __init__(self):
            super().__init__({})
            self.index = 0

        def chat_json(self, payload):
            self.requests.append(copy.deepcopy(dict(payload)))
            responses = (
                _response(content="Hello there.\nThanks for visiting.", finish_reason="stop"),
                _response(content="Welcome friend.\nHave a good day.", finish_reason="stop"),
            )
            response = responses[self.index]
            self.index += 1
            return copy.deepcopy(response)

    _suite, records, summary = collect_live_quality2_records(
        AlternatingTransport(),
        suite_path=SUITE,
        workload_ids=("aq2_dev_tool_irrelevant_en",),
        model="fake-model",
        backend="fake",
        repetitions=2,
        max_tokens=192,
        cache_mode="off",
        idle_timeout_s=1.0,
        sandbox=AgenticQuality2Sandbox(),
    )

    assert summary["aggregation"]["overall"]["passed"] == 2
    assert records["records"][0]["normalized_response_sha256"] != records["records"][1][
        "normalized_response_sha256"
    ]
    assert summary["aggregation"]["determinism"]["basis"] == "normalized_response_v1"
    assert summary["aggregation"]["determinism"]["passed"] is False
    assert summary["aggregation"]["determinism"]["mismatches"] == [
        {
            "workload_id": "aq2_dev_tool_irrelevant_en",
            "fingerprints": sorted(
                record["normalized_response_sha256"] for record in records["records"]
            ),
        }
    ]


def test_expanded_live_collector_routes_code_through_sandbox() -> None:
    user = (
        "Submit Python source defining clamp_readings(values, low, high), returning a new "
        "list with every numeric value clamped inclusively."
    )
    transport = _ExpandedTransport(
        {
            user: _response(
                calls=[
                    {
                        "tool": "submit_code",
                        "arguments": {
                            "entry_point": "clamp_readings",
                            "source": "def clamp_readings(values, low, high): return []",
                        },
                    }
                ]
            )
        }
    )

    _suite, records, summary = collect_live_quality2_records(
        transport,
        suite_path=SUITE,
        workload_ids=("aq2_dev_code_clamp_en",),
        model="fake-model",
        backend="fake",
        repetitions=2,
        max_tokens=192,
        cache_mode="off",
        idle_timeout_s=1.0,
        sandbox=_PassingSandbox(),
    )

    assert summary["aggregation"]["overall"]["passed"] == 2
    assert records["records"][0]["quality"]["sandbox"]["tests_passed"] == 4
