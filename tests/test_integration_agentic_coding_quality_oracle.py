from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

from hipengine.benchmark.agentic import load_agentic_workload_suite
from hipengine.benchmark.agentic_quality import (
    build_agentic_quality_artifact,
    normalize_chat_quality_turn,
)
from hipengine.benchmark.agentic_quality_oracle import evaluate_quality_oracle
from scripts.agentic_coding_quality import collect_live_quality_records

WORKLOADS = Path("benchmarks/prompts/agentic-quality-v2.json")
ORACLE = Path("benchmarks/oracles/agentic-quality-v2.json")
FAMILIES = {"repository", "general_en", "general_ja", "mixed_ja_en"}


def _quality_response(suite, workload_id: str, turn_index: int) -> dict[str, object]:
    turn = suite.workloads[workload_id]["turns"][turn_index]
    generated = [1000 + turn_index, 2000 + turn_index]
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "finish_details": {"reason": "tool_calls"},
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{workload_id}-{turn_index}",
                            "type": "function",
                            "function": {
                                "name": turn["expected_tool"],
                                "arguments": json.dumps(
                                    turn["expected_arguments"],
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    ],
                },
                "hipengine": {
                    "generated_token_ids": generated,
                    "generated_tokens": len(generated),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1024,
            "completion_tokens": len(generated),
            "total_tokens": 1024 + len(generated),
        },
    }


def test_broad_quality_suite_binds_external_oracle_and_families() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)

    assert suite.quality_oracle_path == ORACLE.resolve()
    assert suite.quality_oracle is not None
    assert suite.quality_oracle["kind"] == "hipengine.agentic_quality_oracles"
    assert suite.identity()["quality_oracle"]["file_sha256"] == suite.quality_oracle_file_sha256
    assert len(suite.workloads) == 6
    assert sum(len(workload["turns"]) for workload in suite.workloads.values()) == 24
    assert {workload["family"] for workload in suite.workloads.values()} == FAMILIES
    cases = suite.quality_oracle["cases"]
    referenced = {
        turn["oracle_case"]
        for workload in suite.workloads.values()
        for turn in workload["turns"]
    }
    assert len(referenced) == 24
    assert referenced <= set(cases)


def test_broad_quality_schemas_pin_external_oracle_and_no_performance_contract() -> None:
    oracle_schema = json.loads(
        Path("benchmarks/schemas/agentic-coding-quality-oracles.schema.json").read_text(
            encoding="utf-8"
        )
    )
    records_schema = json.loads(
        Path("benchmarks/schemas/agentic-coding-quality-records.schema.json").read_text(
            encoding="utf-8"
        )
    )
    benchmark_schema = json.loads(
        Path("benchmarks/schemas/agentic-coding-quality-benchmark.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert oracle_schema["properties"]["kind"]["const"] == (
        "hipengine.agentic_quality_oracles"
    )
    assert set(oracle_schema["$defs"]["case"]["properties"]["kind"]["enum"]) == {
        "read",
        "grep",
        "lookup",
        "calculate",
        "patch",
        "test",
    }
    assert records_schema["$defs"]["externalOracle"]["properties"]["evaluated"] == {
        "const": True
    }
    assert "oracle_failed" in records_schema["$defs"]["turnRecord"]["properties"][
        "quality"
    ]["properties"]["outcome"]["enum"]
    assert benchmark_schema["properties"]["performance_claim"] == {"const": False}
    assert "external_oracle" in benchmark_schema["properties"]["quality"]["properties"]
    assert "families" in benchmark_schema["properties"]["quality"]["properties"]


def test_external_oracle_executes_every_tool_family_and_equivalent_results() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    by_kind: dict[str, tuple[str, dict[str, object]]] = {}
    for workload in suite.workloads.values():
        for turn in workload["turns"]:
            case = suite.quality_oracle["cases"][turn["oracle_case"]]
            by_kind.setdefault(case["kind"], (turn["oracle_case"], turn))

    assert set(by_kind) == {"read", "grep", "lookup", "calculate", "patch", "test"}
    for case_id, turn in by_kind.values():
        result = evaluate_quality_oracle(
            suite,
            case_id=case_id,
            selected_tool=turn["expected_tool"],
            arguments=turn["expected_arguments"],
        )
        assert result["evaluated"] is True
        assert result["passed"] is True
        assert result["error"] is None
        assert len(result["result_sha256"]) == 64
        assert result["result_sha256"] == result["expected_result_sha256"]

    calculate_case, calculate_turn = by_kind["calculate"]
    equivalent = evaluate_quality_oracle(
        suite,
        case_id=calculate_case,
        selected_tool=calculate_turn["expected_tool"],
        arguments={"expression": "19 * 37"},
    )
    assert calculate_turn["expected_arguments"] == {"expression": "37 * 19"}
    assert equivalent["passed"] is True

    wrong = evaluate_quality_oracle(
        suite,
        case_id=calculate_case,
        selected_tool=calculate_turn["expected_tool"],
        arguments={"expression": "19 * 38"},
    )
    assert wrong["passed"] is False

    patch_case, patch_turn = by_kind["patch"]
    patch = evaluate_quality_oracle(
        suite,
        case_id=patch_case,
        selected_tool=patch_turn["expected_tool"],
        arguments=patch_turn["expected_arguments"],
    )
    assert patch["patch_applied"] is True
    assert patch["tests_passed"] is True

    test_case, test_turn = by_kind["test"]
    test_result = evaluate_quality_oracle(
        suite,
        case_id=test_case,
        selected_tool=test_turn["expected_tool"],
        arguments=test_turn["expected_arguments"],
    )
    assert test_result["tests_passed"] is True


def test_external_oracle_success_does_not_require_exact_argument_text() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    workload_id = "general_en_operations"
    turn_index = 1
    payload = _quality_response(suite, workload_id, turn_index)
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        json.dumps({"expression": "19 * 37"})
    )

    record = normalize_chat_quality_turn(
        suite,
        workload_id=workload_id,
        turn_index=turn_index,
        run_id="run-0",
        agent_id="agent-0",
        session_id="session-0",
        request_id="equivalent-calculation",
        prompt_token_ids=[1] * 128,
        payload=payload,
    )

    assert record["quality"]["exact_arguments"] is False
    assert record["quality"]["external_oracle"]["passed"] is True
    assert record["quality"]["success"] is True
    assert record["quality"]["outcome"] == "passed"


def test_extended_quality_record_and_rollup_report_oracle_patch_test_and_families() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    records = []
    for workload_id, workload in suite.workloads.items():
        for turn_index, _turn in enumerate(workload["turns"]):
            records.append(
                normalize_chat_quality_turn(
                    suite,
                    workload_id=workload_id,
                    turn_index=turn_index,
                    run_id=f"run-0-{workload_id}",
                    agent_id="agent-0",
                    session_id="session-0",
                    request_id=f"run-0-{workload_id}-turn-{turn_index}",
                    prompt_token_ids=[1] * 128,
                    payload=_quality_response(suite, workload_id, turn_index),
                )
            )
    payload = {
        "kind": "hipengine_agentic_coding_quality_records",
        "schema_version": 1,
        "configuration": {
            "id": "broad-quality-test",
            "lane": "auto_tool",
            "concurrency": 1,
            "cache_mode": "off",
            "backend": "fake",
            "model": "fake-model",
            "require_complete_workloads": True,
            "performance_claim": False,
            "tool_choice": "auto",
        },
        "turn_records": records,
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
    artifact = build_agentic_quality_artifact(suite, payload)

    assert artifact["performance_claim"] is False
    assert artifact["coverage"]["families"] == sorted(FAMILIES)
    assert artifact["coverage"]["turns"] == 24
    assert artifact["quality"]["external_oracle"] == {
        "attempts": 24,
        "passes": 24,
        "pass_rate": 1.0,
        "patch_attempts": 3,
        "patch_successes": 3,
        "patch_success_rate": 1.0,
        "test_attempts": 4,
        "test_successes": 4,
        "test_success_rate": 1.0,
    }
    assert set(artifact["quality"]["families"]) == FAMILIES
    assert sum(
        row["attempts"] for row in artifact["quality"]["families"].values()
    ) == 24
    assert all(record["quality"]["external_oracle"]["passed"] for record in records)
    assert "latency" not in artifact
    assert "goodput" not in artifact
    assert "tok_per_s" not in artifact


class _BroadQualityTransport:
    def __init__(self, suite) -> None:
        self.suite = suite
        self.lock = threading.Lock()
        self.tool_choices: list[object] = []
        self.system_messages: list[str] = []

    def capabilities(self):
        return {
            "object": "hipengine.capabilities",
            "model": {"id": "fake-model", "backend": "fake"},
            "tokenizer": {"tokenize": True, "detokenize": True},
            "features": {
                "tools": {"enabled": True, "strict_result_validation": True},
            },
            "cache": {"prefix_cache": "off"},
        }

    def tokenize(self, text):
        return list(text.encode("utf-8"))

    def detokenize(self, token_ids):
        return bytes(int(token) for token in token_ids).decode("utf-8")

    def rendered_prompt_ids(self, *, tool_choice, messages, **_kwargs):
        with self.lock:
            self.tool_choices.append(copy.deepcopy(tool_choice))
            self.system_messages.append(messages[0]["content"])
        return [1] * 128

    def chat_json(self, payload):
        current_user = payload["messages"][-1]["content"]
        for workload_id, workload in self.suite.workloads.items():
            for turn_index, turn in enumerate(workload["turns"]):
                if turn["user"] == current_user:
                    return _quality_response(self.suite, workload_id, turn_index)
        raise AssertionError(f"unknown user prompt: {current_user}")

    def ready(self):
        return {
            "ready": True,
            "queue": {"depth": 0, "worker_active": False, "active_requests": 0},
            "kv_capacity": {"pool": {"refcounted_pages": 0, "pinned_pages": 0}},
        }

    def sessions(self):
        return {"sessions": [], "continuations": {"active": 0}}


def test_live_quality_collector_runs_all_workloads_without_tool_name_hinting() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    transport = _BroadQualityTransport(suite)

    loaded_suite, records = collect_live_quality_records(
        transport,
        workloads_path=WORKLOADS,
        workload_ids=tuple(suite.workloads),
        model="fake-model",
        backend="fake",
        concurrency=1,
        runs=1,
        max_tokens=32,
        cache_mode="off",
        idle_timeout_s=1.0,
    )
    artifact = build_agentic_quality_artifact(loaded_suite, records)

    assert artifact["coverage"]["workloads"] == sorted(suite.workloads)
    assert artifact["coverage"]["turns"] == 24
    assert artifact["quality"]["successes"] == 24
    assert transport.tool_choices == ["auto"] * 24
    assert all("Choose the appropriate tool" in message for message in transport.system_messages)
    assert all(
        "specifically requested tool" not in message
        for message in transport.system_messages
    )
