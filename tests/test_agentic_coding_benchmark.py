from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hipengine.benchmark.agentic import (
    AGENTIC_RECORDS_KIND,
    AgenticBenchmarkError,
    build_agentic_benchmark_artifact,
    load_agentic_workload_suite,
    percentile,
    validate_agentic_benchmark_artifact,
)
from scripts.agentic_coding_bench import build_artifact_from_paths


WORKLOADS = Path("benchmarks/prompts/agentic-coding-v1.json")


def _record(suite, turn_index: int, *, batch_id: str, timing_owner: bool) -> dict[str, object]:
    generated = [100 + turn_index, 200 + turn_index]
    submitted = float(turn_index)
    expected = suite.workloads["small_repo"]["turns"][turn_index]
    return {
        "workload_id": "small_repo",
        "workload_sha256": suite.workload_sha256("small_repo"),
        "run_id": "run-0",
        "agent_id": "agent-0",
        "session_id": "session-0",
        "turn_index": turn_index,
        "request_id": f"request-{turn_index}",
        "prompt": {
            "token_count": 2048 + turn_index * 32,
            "token_ids_sha256": f"{turn_index + 1:064x}",
        },
        "output": {
            "generated_token_ids": generated,
            "generated_token_ids_sha256": suite.token_ids_sha256(generated),
            "generated_token_ids_source": "response",
            "sse_exact_ids_observed": True,
            "raw_markup_leaked": False,
        },
        "tool": {
            "expected_name": expected["expected_tool"],
            "name": expected["expected_tool"],
            "declared_schema_sha256": suite.tool_schema_sha256(expected["expected_tool"]),
            "call_id": f"call-{turn_index}",
            "arguments": copy.deepcopy(expected["expected_arguments"]),
            "arguments_json_valid": True,
            "schema_valid": True,
            "result_linked": True,
        },
        "timing": {
            "submitted_at_s": submitted,
            "first_token_at_s": submitted + 0.1,
            "token_observed_at_s": [submitted + 0.1, submitted + 0.2],
            "token_event_token_counts": [1, 1],
            "token_timing_mode": "live_exact",
            "tool_call_ready_at_s": submitted + 0.3,
            "response_done_at_s": submitted + 0.4,
            "tool_result_submitted_at_s": submitted + 0.5,
        },
        "backend": {
            "batch_id": batch_id,
            "timing_scope": "batch",
            "timing_owner": timing_owner,
            "sampler_mode": "greedy_fast",
            "logits_d2h_bytes": 0,
            "physical_width": 2 if batch_id == "batch-0" else 1,
            "serial_fallback": False,
        },
        "prefix": {
            "lookup": turn_index > 0,
            "hit": turn_index == 1,
            "reused_tokens": 256 if turn_index == 1 else 0,
            "cache_bytes": 4096 if turn_index == 1 else 0,
        },
        "finish": {"reason": "tool_calls", "retry_count": 0},
    }


def _records_payload(suite) -> dict[str, object]:
    return {
        "kind": AGENTIC_RECORDS_KIND,
        "schema_version": 1,
        "configuration": {
            "id": "a0-test",
            "lane": "deterministic",
            "concurrency": 1,
            "cache_mode": "radix",
            "backend": "fake",
            "model": "fake-model",
            "require_complete_workloads": True,
        },
        "turn_records": [
            _record(suite, 0, batch_id="batch-0", timing_owner=True),
            _record(suite, 1, batch_id="batch-0", timing_owner=False),
            _record(suite, 2, batch_id="batch-1", timing_owner=True),
            _record(suite, 3, batch_id="batch-2", timing_owner=True),
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


def test_workload_suite_is_stable_and_covers_initial_families() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)

    assert suite.kind == "hipengine.agentic_coding_workloads"
    assert suite.schema_version == 1
    assert set(suite.workloads) == {"small_repo", "medium_repo", "growing_history"}
    assert suite.workloads["small_repo"]["target_prefix_tokens"] == 2048
    assert suite.workloads["medium_repo"]["target_prefix_tokens"] == 8192
    assert [len(suite.workloads[name]["turns"]) for name in suite.workloads] == [4, 6, 8]
    assert len(suite.file_sha256) == 64
    assert len(suite.canonical_sha256) == 64
    assert suite.file_sha256 == suite.canonical_sha256
    assert all(len(suite.workload_sha256(name)) == 64 for name in suite.workloads)


def test_agentic_artifact_rolls_up_exact_turn_latency_and_goodput() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    artifact = build_agentic_benchmark_artifact(suite, _records_payload(suite))

    assert validate_agentic_benchmark_artifact(artifact) == {
        "passed": True,
        "failure_reasons": [],
    }
    assert artifact["validation"] == {"passed": True, "failure_reasons": []}
    assert artifact["coverage"] == {
        "workloads": ["small_repo"],
        "runs": 1,
        "concurrency": 1,
        "agents": 1,
        "turns": 4,
        "tool_calls": 4,
        "generated_tokens": 8,
        "batches": 3,
    }
    latency = artifact["rollup"]["latency_ms"]
    assert latency["ttft"]["p50"] == pytest.approx(100.0)
    assert latency["tool_call_ready"]["p95"] == pytest.approx(300.0)
    assert latency["inter_token"]["p99"] == pytest.approx(100.0)
    assert latency["complete_turn"]["p50"] == pytest.approx(500.0)
    assert artifact["rollup"]["workload_wall_scope"] == (
        "first_submit_to_last_tool_result_submit_including_inter_turn_control"
    )
    assert artifact["rollup"]["workload_wall_s"] == pytest.approx(3.5)
    assert artifact["rollup"]["exact_generated_tok_s"] == pytest.approx(8 / 3.5)
    assert artifact["rollup"]["validated_tool_calls_s"] == pytest.approx(4 / 3.5)
    assert artifact["rollup"]["active_sse_wave_count"] == 4
    assert artifact["rollup"]["active_sse_wave_wall_s"] == pytest.approx(1.6)
    assert artifact["rollup"]["active_sse_exact_generated_tok_s"] == pytest.approx(8 / 1.6)
    assert artifact["rollup"]["active_sse_validated_tool_calls_s"] == pytest.approx(4 / 1.6)
    assert artifact["rollup"]["prefix"] == {
        "lookups": 3,
        "hits": 1,
        "hit_rate": pytest.approx(1 / 3),
        "reused_tokens": 256,
        "max_cache_bytes": 4096,
    }
    assert artifact["rollup"]["backend"]["physical_width_turns"] == {"1": 2, "2": 2}
    assert artifact["rollup"]["backend"]["full_vocab_logits_d2h_bytes"] == 0
    assert artifact["rollup"]["backend"]["token_timing_mode_turns"] == {"live_exact": 4}
    assert artifact["rollup"]["backend"]["generated_token_id_source_turns"] == {"response": 4}

    tampered = copy.deepcopy(artifact)
    tampered["turn_records"][0]["output"]["generated_token_ids"].append(999)
    with pytest.raises(
        AgenticBenchmarkError,
        match="turn_records_sha256 does not match turn_records",
    ):
        validate_agentic_benchmark_artifact(tampered)


def test_agentic_active_sse_wall_groups_concurrent_rows_once_per_turn() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    records = _records_payload(suite)
    records["configuration"]["concurrency"] = 2
    second_agent = []
    for raw_record in records["turn_records"]:
        record = copy.deepcopy(raw_record)
        record["agent_id"] = "agent-1"
        record["session_id"] = "session-1"
        record["request_id"] = f"{record['request_id']}-agent-1"
        record["backend"]["timing_owner"] = False
        for key in (
            "submitted_at_s",
            "first_token_at_s",
            "tool_call_ready_at_s",
            "response_done_at_s",
            "tool_result_submitted_at_s",
        ):
            record["timing"][key] += 0.05
        record["timing"]["token_observed_at_s"] = [
            value + 0.05 for value in record["timing"]["token_observed_at_s"]
        ]
        second_agent.append(record)
    records["turn_records"].extend(second_agent)

    artifact = build_agentic_benchmark_artifact(suite, records)

    assert artifact["coverage"]["concurrency"] == 2
    assert artifact["rollup"]["active_sse_wave_count"] == 4
    assert artifact["rollup"]["active_sse_wave_wall_s"] == pytest.approx(1.8)
    assert artifact["rollup"]["active_sse_exact_generated_tok_s"] == pytest.approx(16 / 1.8)


def test_agentic_artifact_rejects_tool_timing_token_owner_and_resource_failures() -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    cases: list[tuple[str, object, str]] = []

    wrong_tool = _records_payload(suite)
    wrong_tool["turn_records"][0]["tool"]["name"] = "undeclared"
    cases.append(("wrong tool", wrong_tool, "record[0].tool.name does not match expected tool"))

    missing_timestamp = _records_payload(suite)
    del missing_timestamp["turn_records"][0]["timing"]["tool_call_ready_at_s"]
    cases.append(
        ("timestamp", missing_timestamp, "record[0].timing.tool_call_ready_at_s is required")
    )

    bad_denominator = _records_payload(suite)
    bad_denominator["turn_records"][0]["output"]["generated_token_ids_sha256"] = "0" * 64
    cases.append(("token hash", bad_denominator, "record[0] generated token hash mismatch"))

    leaked_markup = _records_payload(suite)
    leaked_markup["turn_records"][0]["output"]["raw_markup_leaked"] = True
    cases.append(("markup", leaked_markup, "record[0] leaked raw model markup"))

    false_sse_ids = _records_payload(suite)
    false_sse_ids["turn_records"][0]["output"]["sse_exact_ids_observed"] = False
    cases.append(
        (
            "id source",
            false_sse_ids,
            "record[0] exact-ID source/observation metadata is inconsistent",
        )
    )

    ambiguous_owner = _records_payload(suite)
    ambiguous_owner["turn_records"][1]["backend"]["timing_owner"] = True
    cases.append(("owner", ambiguous_owner, "batch run-0/batch-0 has 2 timing owners; expected 1"))

    leaked_resource = _records_payload(suite)
    leaked_resource["final_ownership"]["active_requests"] = 1
    cases.append(("ownership", leaked_resource, "final_ownership.active_requests must be zero"))

    false_claim = _records_payload(suite)
    false_claim["configuration"]["performance_claim"] = True
    cases.append(
        (
            "claim",
            false_claim,
            "A0 normalized-record artifacts cannot set performance_claim=true",
        )
    )

    for label, payload, expected in cases:
        with pytest.raises(
            AgenticBenchmarkError, match=expected.replace("[", r"\[").replace("]", r"\]")
        ):
            build_agentic_benchmark_artifact(suite, payload)


def test_percentile_is_deterministic_and_rejects_invalid_values() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 95.0) == pytest.approx(3.85)
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 50.0)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], 101.0)


def test_agentic_json_schemas_pin_workload_records_and_artifact_kinds() -> None:
    schemas = {
        "agentic-coding-workloads.schema.json": "hipengine.agentic_coding_workloads",
        "agentic-coding-records.schema.json": "hipengine_agentic_coding_records",
        "agentic-coding-benchmark.schema.json": "hipengine_agentic_coding_benchmark",
    }
    for filename, kind in schemas.items():
        payload = json.loads((Path("benchmarks/schemas") / filename).read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["properties"]["kind"]["const"] == kind
        assert payload["properties"]["schema_version"]["const"] == 1


def test_agentic_benchmark_cli_builds_json_and_fails_closed(tmp_path: Path) -> None:
    suite = load_agentic_workload_suite(WORKLOADS)
    records_path = tmp_path / "records.json"
    output_path = tmp_path / "artifact.json"
    records_path.write_text(json.dumps(_records_payload(suite)), encoding="utf-8")

    artifact = build_artifact_from_paths(WORKLOADS, records_path, output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["validation"]["passed"] is True

    invalid = _records_payload(suite)
    invalid["final_ownership"]["stream_producers"] = 2
    records_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(AgenticBenchmarkError, match="stream_producers must be zero"):
        build_artifact_from_paths(WORKLOADS, records_path, output_path)
