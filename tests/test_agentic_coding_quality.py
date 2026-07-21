from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import pytest

from hipengine.benchmark.agentic import AgenticBenchmarkError, load_agentic_workload_suite
from hipengine.benchmark.agentic_quality import (
    AGENTIC_QUALITY_RECORDS_KIND,
    build_agentic_quality_artifact,
    normalize_chat_quality_turn,
    validate_agentic_quality_artifact,
)
from scripts.agentic_coding_quality import collect_live_quality_records


WORKLOADS = Path("benchmarks/prompts/agentic-coding-v1.json")


def _quality_response(
    suite,
    *,
    turn_index: int,
    outcome: str = "passed",
) -> dict[str, object]:
    turn = suite.workloads["small_repo"]["turns"][turn_index]
    selected_tool = str(turn["expected_tool"])
    arguments = copy.deepcopy(turn["expected_arguments"])
    finish_reason = "tool_calls"
    detail_reason = "tool_calls"
    content = ""
    tool_calls: list[dict[str, object]] = [
        {
            "id": f"call-{turn_index}",
            "type": "function",
            "function": {
                "name": selected_tool,
                "arguments": json.dumps(arguments, separators=(",", ":")),
            },
        }
    ]
    if outcome == "invalid_tool_call":
        finish_reason = "length"
        detail_reason = "invalid_tool_call"
        tool_calls = []
    elif outcome == "wrong_tool":
        selected_tool = "read" if turn["expected_tool"] != "read" else "grep"
        tool_calls[0]["function"]["name"] = selected_tool
        if selected_tool == "read":
            arguments = {"path": "pyproject.toml", "mode": "summary"}
        else:
            arguments = {"pattern": "def admit", "path": "src"}
        tool_calls[0]["function"]["arguments"] = json.dumps(
            arguments, separators=(",", ":")
        )
    elif outcome == "content_leak":
        content = "unexpected assistant content"

    generated = [1000 + turn_index, 1100 + turn_index]
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "finish_details": {"reason": detail_reason},
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "hipengine": {
                    "generated_token_ids": generated,
                    "generated_tokens": len(generated),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 2048 + turn_index,
            "completion_tokens": len(generated),
            "total_tokens": 2050 + turn_index,
        },
    }


def _record(suite, *, turn_index: int, outcome: str = "passed") -> dict[str, object]:
    return normalize_chat_quality_turn(
        suite,
        workload_id="small_repo",
        turn_index=turn_index,
        run_id="run-0",
        agent_id="agent-0",
        session_id="session-0",
        request_id=f"request-{turn_index}",
        prompt_token_ids=[1] * (2048 + turn_index),
        payload=_quality_response(suite, turn_index=turn_index, outcome=outcome),
    )


def _records_payload(suite) -> dict[str, object]:
    return {
        "kind": AGENTIC_QUALITY_RECORDS_KIND,
        "schema_version": 1,
        "configuration": {
            "id": "quality-test",
            "lane": "auto_tool",
            "concurrency": 1,
            "cache_mode": "off",
            "backend": "fake",
            "model": "fake-model",
            "require_complete_workloads": True,
            "performance_claim": False,
            "tool_choice": "auto",
        },
        "turn_records": [
            _record(suite, turn_index=0),
            _record(suite, turn_index=1, outcome="invalid_tool_call"),
            _record(suite, turn_index=2, outcome="content_leak"),
            _record(suite, turn_index=3),
        ],
        "final_ownership": {
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
        },
    }


def test_quality_normalizer_preserves_natural_failures_instead_of_aborting() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)

    passed = _record(suite, turn_index=0)
    assert passed["quality"] == {
        "expected_tool": "read",
        "selected_tool": "read",
        "tool_call_count": 1,
        "call_id": "call-0",
        "arguments": {"path": "pyproject.toml", "mode": "summary"},
        "arguments_json_valid": True,
        "schema_valid": True,
        "correct_tool": True,
        "exact_arguments": True,
        "valid_call": True,
        "success": True,
        "outcome": "passed",
        "repair_count": 0,
    }
    assert passed["output"]["generated_token_ids_source"] == "response"

    failed = _record(suite, turn_index=1, outcome="invalid_tool_call")
    assert failed["quality"]["success"] is False
    assert failed["quality"]["valid_call"] is False
    assert failed["quality"]["outcome"] == "invalid_tool_call"
    assert failed["finish"] == {"reason": "length", "detail_reason": "invalid_tool_call"}
    assert failed["output"]["generated_token_ids"] == [1001, 1101]

    leaked = _record(suite, turn_index=2, outcome="content_leak")
    assert leaked["quality"]["outcome"] == "content_alongside_tool_call"
    assert leaked["quality"]["success"] is False


def test_quality_artifact_reports_rates_without_performance_rollups() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    artifact = build_agentic_quality_artifact(suite, _records_payload(suite))

    assert validate_agentic_quality_artifact(artifact) == {
        "passed": True,
        "failure_reasons": [],
    }
    assert artifact["performance_claim"] is False
    assert artifact["coverage"] == {
        "workloads": ["small_repo"],
        "runs": 1,
        "concurrency": 1,
        "agents": 1,
        "turns": 4,
        "generated_tokens": 8,
    }
    assert artifact["quality"] == {
        "attempts": 4,
        "valid_calls": 3,
        "correct_tools": 3,
        "exact_arguments": 3,
        "successes": 2,
        "valid_call_rate": pytest.approx(0.75),
        "correct_tool_rate": pytest.approx(0.75),
        "exact_arguments_rate": pytest.approx(0.75),
        "success_rate": pytest.approx(0.5),
        "repair_attempts": 0,
        "outcomes": {
            "content_alongside_tool_call": 1,
            "invalid_tool_call": 1,
            "passed": 2,
        },
    }
    assert "rollup" not in artifact


def test_quality_json_schemas_pin_separate_non_performance_kinds() -> None:
    schemas = {
        "agentic-coding-quality-records.schema.json": AGENTIC_QUALITY_RECORDS_KIND,
        "agentic-coding-quality-benchmark.schema.json": (
            "hipengine_agentic_coding_quality_benchmark"
        ),
    }
    for filename, kind in schemas.items():
        payload = json.loads((Path("benchmarks/schemas") / filename).read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["properties"]["kind"]["const"] == kind
        if filename.endswith("records.schema.json"):
            assert payload["$defs"]["configuration"]["properties"]["performance_claim"]["const"] is False
        else:
            assert payload["properties"]["performance_claim"]["const"] is False


def test_quality_artifact_rejects_false_claims_and_tampering() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)

    false_claim = _records_payload(suite)
    false_claim["configuration"]["performance_claim"] = True
    with pytest.raises(AgenticBenchmarkError, match="quality artifacts cannot make performance claims"):
        build_agentic_quality_artifact(suite, false_claim)

    artifact = build_agentic_quality_artifact(suite, _records_payload(suite))
    artifact["turn_records"][0]["quality"]["success"] = False
    with pytest.raises(AgenticBenchmarkError, match="turn_records_sha256 does not match turn_records"):
        validate_agentic_quality_artifact(artifact)


class _FakeQualityTransport:
    def __init__(self, suite) -> None:
        self.suite = suite
        self.lock = threading.Lock()
        self.tool_choices: list[object] = []

    def capabilities(self):
        return {"cache": {"prefix_cache": "off"}}

    def tokenize(self, text):
        return list(text.encode("utf-8"))

    def detokenize(self, token_ids):
        return bytes(int(token) for token in token_ids).decode("utf-8")

    def rendered_prompt_ids(self, *, tool_choice, **_kwargs):
        with self.lock:
            self.tool_choices.append(copy.deepcopy(tool_choice))
        return [1] * 128

    def chat_json(self, payload):
        assert payload["tool_choice"] == "auto"
        current_user = payload["messages"][-1]["content"]
        turns = self.suite.workloads["small_repo"]["turns"]
        turn_index = next(index for index, turn in enumerate(turns) if turn["user"] == current_user)
        outcome = "invalid_tool_call" if turn_index == 1 else "passed"
        return _quality_response(self.suite, turn_index=turn_index, outcome=outcome)

    def ready(self):
        return {
            "ready": True,
            "queue": {"depth": 0, "worker_active": False, "active_requests": 0},
            "kv_capacity": {"pool": {"refcounted_pages": 0, "pinned_pages": 0}},
        }

    def sessions(self):
        return {"sessions": [], "continuations": {"active": 0}}


def test_live_quality_collector_keeps_failed_turn_and_uses_auto_tool_choice() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    transport = _FakeQualityTransport(suite)

    loaded_suite, records = collect_live_quality_records(
        transport,
        workloads_path=WORKLOADS,
        workload_id="small_repo",
        model="fake-model",
        backend="fake",
        concurrency=1,
        runs=1,
        max_tokens=32,
        cache_mode="off",
        idle_timeout_s=1.0,
    )
    artifact = build_agentic_quality_artifact(loaded_suite, records)

    assert artifact["coverage"]["turns"] == 4
    assert artifact["quality"]["successes"] == 3
    assert artifact["quality"]["outcomes"] == {"invalid_tool_call": 1, "passed": 3}
    assert transport.tool_choices == ["auto"] * 4
