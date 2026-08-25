from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

SUITE = Path("benchmarks/prompts/agentic-quality2-v2.json")
ORACLE = Path("benchmarks/oracles/agentic-quality2-v2.json")
SOURCES = Path("benchmarks/sources/agentic-quality2-v2-sources.json")
SCHEMAS = {
    "suite": Path("benchmarks/schemas/agentic-quality2-v2-suite.schema.json"),
    "oracle": Path("benchmarks/schemas/agentic-quality2-v2-oracles.schema.json"),
    "sources": Path("benchmarks/schemas/agentic-quality2-v2-sources.schema.json"),
}
FROZEN_SHA256 = {
    SUITE: "dbe4668667ba3ca57649408f4dc9a5004ee771ce61dc95f7816cf6799b62cbdd",
    ORACLE: "c6fd180a2fe7156307995b9567a149f8c50f7448da003bdbe4cc0abe41f0706a",
    SOURCES: "b5d7ed2573b78ca05b14e34616c562fce4dba154fb938a414b5b96ed1ad1fdf8",
    SCHEMAS["suite"]: "6a2a36e81e82a64430bd5bcdeb62d7e45e11a81cba65809093677547da6060f1",
    SCHEMAS["oracle"]: "c7deefcf776cf053046cc19e41d951de55ea1bb1dd3698df213abbb866e19a97",
    SCHEMAS["sources"]: "db1df4e2ac0e25bd9df494503def97ae4a0c612637a7762fc4af5f508a323cfb",
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_quality2_v1_exact_files_are_frozen_before_candidate_code() -> None:
    assert {path: _sha256(path) for path in FROZEN_SHA256} == FROZEN_SHA256

    suite = _load(SUITE)
    oracle = _load(ORACLE)
    assert suite["split_policy"]["frozen_before_candidate_code"] is True
    assert suite["source_manifest"]["file_sha256"] == FROZEN_SHA256[SOURCES]
    assert suite["quality_oracle"]["file_sha256"] == FROZEN_SHA256[ORACLE]
    assert oracle["source_manifest"]["file_sha256"] == FROZEN_SHA256[SOURCES]


def test_agentic_quality2_v1_split_and_minimum_coverage_are_explicit() -> None:
    suite = _load(SUITE)
    oracle = _load(ORACLE)
    workloads = suite["workloads"]
    workload_ids = [row["id"] for row in workloads]
    development = set(suite["split_policy"]["development_ids"])
    heldout = set(suite["split_policy"]["heldout_ids"])

    assert len(workload_ids) == len(set(workload_ids)) == 34
    assert development.isdisjoint(heldout)
    assert development | heldout == set(workload_ids)
    assert len(development) == len(heldout) == 17
    assert development == {row["id"] for row in workloads if row["split"] == "development"}
    assert heldout == {row["id"] for row in workloads if row["split"] == "heldout"}
    assert Counter((row["family"], row["split"]) for row in workloads) == Counter(
        {
            ("tool_selection", "development"): 5,
            ("tool_selection", "heldout"): 5,
            ("repository", "development"): 4,
            ("repository", "heldout"): 4,
            ("code", "development"): 4,
            ("code", "heldout"): 4,
            ("instruction", "development"): 4,
            ("instruction", "heldout"): 4,
        }
    )
    assert sum(row["split"] == "heldout" and row["language"] != "en" for row in workloads) == 12
    assert (
        sum(
            row["split"] == "heldout" and row["task_kind"] in {"patch", "code"} for row in workloads
        )
        == 5
    )
    assert {row["task_kind"] for row in workloads if row["family"] == "tool_selection"} == {
        "nested",
        "optional",
        "enum",
        "multiple",
        "irrelevant",
    }
    assert len(oracle["code_cases"]) == 8
    assert all(len(case["hidden_tests"]) >= 4 for case in oracle["code_cases"].values())
    assert len(oracle["instruction_cases"]) == 8
    assert len(oracle["fail_safe_controls"]) == 10


def test_agentic_quality2_v1_cases_are_one_to_one_and_same_split() -> None:
    suite = _load(SUITE)
    oracle = _load(ORACLE)
    cases = oracle["cases"]
    workloads = {row["id"]: row for row in suite["workloads"]}

    assert set(workloads) == set(cases)
    for workload_id, workload in workloads.items():
        turn = workload["turns"][0]
        case = cases[turn["oracle_case"]]
        assert case["workload_id"] == workload_id
        assert case["split"] == workload["split"]


def test_agentic_quality2_v1_has_no_reference_code_or_prose() -> None:
    suite = _load(SUITE)
    oracle = _load(ORACLE)
    sources = _load(SOURCES)
    oracle_text = json.dumps(oracle, sort_keys=True, ensure_ascii=False)
    visible = "\n".join(
        [
            suite["repository_context"]["base"],
            *suite["repository_context"]["expansion_blocks"],
            *(tool["description"] for tool in suite["tools"]),
            *(row["turns"][0]["user"] for row in suite["workloads"]),
        ]
    )

    assert sources["selection"] == {
        "mode": "project_original_bounded_style_tasks",
        "official_score_claimed": False,
        "reason": sources["selection"]["reason"],
        "upstream_bytes_imported": False,
    }
    assert "reference_source" not in oracle_text
    assert "reference_response" not in oracle_text
    assert not any(case["expected_result_sha256"] in visible for case in oracle["cases"].values())
    assert not any(
        patch["old"] in visible or patch["new"] in visible for patch in oracle["patches"].values()
    )


def test_agentic_quality2_draft_schemas_reject_missing_language_and_license() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    suite_schema = _load(SCHEMAS["suite"])
    source_schema = _load(SCHEMAS["sources"])
    suite = _load(SUITE)
    sources = _load(SOURCES)

    missing_language = copy.deepcopy(suite)
    del missing_language["workloads"][0]["language"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(suite_schema).validate(missing_language)

    missing_license = copy.deepcopy(sources)
    del missing_license["public_source_audit"][0]["license"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(source_schema).validate(missing_license)
