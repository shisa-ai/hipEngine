from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hipengine.benchmark.agentic import load_agentic_workload_suite
from hipengine.benchmark.agentic_quality_taxonomy import (
    classify_agentic_quality_observation,
    parse_independent_tool_envelope,
)

WORKLOADS = Path("benchmarks/prompts/agentic-quality-v2.json")


def _record() -> dict[str, object]:
    return {
        "workload_id": "repository_scheduler_en",
        "turn_index": 0,
        "quality": {
            "selected_tool": "grep",
            "tool_call_count": 1,
            "arguments": {"path": "src/scheduler.py", "pattern": "ValueError"},
            "arguments_json_valid": True,
            "schema_valid": True,
            "repair_count": 0,
            "external_oracle": {"evaluated": True, "passed": True},
        },
        "output": {
            "public_content": "",
            "raw_markup_leaked": False,
            "generated_token_ids_sha256": "a" * 64,
        },
        "finish": {"reason": "tool_calls", "detail_reason": "tool_calls"},
    }


def _raw_text(name: str, arguments: object) -> str:
    return (
        "<tool_call>"
        + json.dumps(
            {"name": name, "arguments": arguments},
            separators=(",", ":"),
        )
        + "</tool_call>"
    )


def _classify(record, *, raw_text=None, raw_response=None, prompt_roundtrip=True):
    suite = load_agentic_workload_suite(WORKLOADS)
    quality = record["quality"]
    text = raw_text or _raw_text(
        quality["selected_tool"],
        quality["arguments"],
    )
    return classify_agentic_quality_observation(
        suite,
        record,
        raw_response=raw_response or {"choices": [{}]},
        raw_text=text,
        prompt_token_roundtrip=prompt_roundtrip,
    )


def test_independent_parser_accepts_canonical_envelope_without_server_parser() -> None:
    parsed = parse_independent_tool_envelope(
        _raw_text("grep", {"path": "src/scheduler.py", "pattern": "ValueError"})
    )

    assert parsed["accepted"] is True
    assert parsed["content"] == ""
    assert parsed["tool_calls"][0]["name"] == "grep"
    assert parsed["tool_calls"][0]["arguments"] == {
        "path": "src/scheduler.py",
        "pattern": "ValueError",
    }


@pytest.mark.parametrize(
    ("mutation", "primary", "owner", "boundary"),
    [
        ({}, "passed", "none", "none"),
        (
            {
                "selected_tool": "read",
                "arguments": {"path": "src/scheduler.py", "mode": "raw"},
                "external_oracle": {"evaluated": True, "passed": False},
            },
            "wrong_tool",
            "model_quality",
            "model_tool_selection",
        ),
        (
            {
                "arguments": {
                    "path": "src/scheduler.py",
                    "pattern": "full pending queue",
                },
                "external_oracle": {"evaluated": True, "passed": False},
            },
            "wrong_arguments",
            "model_quality",
            "model_generated_arguments",
        ),
        (
            {
                "arguments": {
                    "path": "src/scheduler.py",
                    "pattern": "semantically equivalent",
                },
                "external_oracle": {"evaluated": True, "passed": True},
            },
            "passed",
            "none",
            "none",
        ),
        (
            {"external_oracle": {"evaluated": True, "passed": False}},
            "external_oracle_failure",
            "model_quality",
            "tool_execution_or_oracle",
        ),
    ],
)
def test_classifier_separates_model_quality_outcomes(
    mutation: dict[str, object],
    primary: str,
    owner: str,
    boundary: str,
) -> None:
    record = _record()
    record["quality"].update(copy.deepcopy(mutation))

    taxonomy = _classify(record)

    assert taxonomy["primary_outcome"] == primary
    assert taxonomy["owner"] == owner
    assert taxonomy["earliest_bad_boundary"] == boundary
    assert taxonomy["independent_parser"]["projection_matches_public"] is True


def test_classifier_detects_parser_projection_mismatch_before_model_quality() -> None:
    record = _record()
    raw = _raw_text("grep", {"path": "src/scheduler.py", "pattern": "OTHER"})

    taxonomy = _classify(record, raw_text=raw)

    assert taxonomy["primary_outcome"] == "parser_mismatch"
    assert taxonomy["owner"] == "runtime_implementation"
    assert taxonomy["earliest_bad_boundary"] == "raw_envelope_to_public_projection"


def test_classifier_detects_template_roundtrip_and_runtime_failures_first() -> None:
    record = _record()
    template = _classify(record, prompt_roundtrip=False)
    runtime = _classify(record, raw_response={"error": {"message": "generation failed"}})

    assert template["primary_outcome"] == "template_or_tokenizer_mismatch"
    assert template["owner"] == "runtime_implementation"
    assert runtime["primary_outcome"] == "runtime_error"
    assert runtime["owner"] == "runtime_implementation"


def test_classifier_distinguishes_length_malformed_and_no_tool_call() -> None:
    length = _record()
    length["quality"].update(
        {"selected_tool": None, "arguments": None, "tool_call_count": 0}
    )
    length["finish"] = {"reason": "length", "detail_reason": "length"}
    malformed = copy.deepcopy(length)
    malformed["finish"] = {"reason": "stop", "detail_reason": "invalid_tool_call"}
    no_call = copy.deepcopy(malformed)

    length_taxonomy = _classify(length, raw_text='<tool_call>{"name":"grep"')
    malformed_taxonomy = _classify(malformed, raw_text="<tool_call>not json</tool_call>")
    no_call_taxonomy = _classify(no_call, raw_text="ordinary answer")

    assert length_taxonomy["primary_outcome"] == "length_exhausted"
    assert malformed_taxonomy["primary_outcome"] == "invalid_json"
    assert no_call_taxonomy["primary_outcome"] == "no_tool_call"


def test_classifier_detects_public_content_and_raw_marker_leaks() -> None:
    content = _record()
    content["output"]["public_content"] = "extra answer"
    raw_marker = copy.deepcopy(content)
    raw_marker["output"]["public_content"] = "<think>hidden</think>"
    raw_marker["output"]["raw_markup_leaked"] = True

    assert _classify(content)["primary_outcome"] == "content_alongside_tool_call"
    assert _classify(raw_marker)["primary_outcome"] == "raw_markup_leak"


def test_classifier_detects_undeclared_and_schema_invalid_calls() -> None:
    undeclared = _record()
    undeclared["quality"].update(
        {
            "selected_tool": "ghost",
            "arguments": {"value": "x"},
            "external_oracle": {"evaluated": True, "passed": False},
        }
    )
    schema = _record()
    schema["quality"].update(
        {
            "arguments": {"pattern": "ValueError"},
            "schema_valid": False,
            "external_oracle": {"evaluated": True, "passed": False},
        }
    )

    assert _classify(undeclared)["primary_outcome"] == "undeclared_tool"
    assert _classify(schema)["primary_outcome"] == "schema_violation"
