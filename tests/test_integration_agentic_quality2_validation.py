from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hipengine.benchmark.agentic_quality2 import (
    AgenticQuality2Error,
    aggregate_quality2_results,
    evaluate_quality2_fail_safe_control,
    evaluate_quality2_oracle,
    execute_reference_case,
    load_agentic_quality2_suite,
)
from hipengine.benchmark.agentic_quality2_sandbox import (
    AgenticQuality2Sandbox,
    SandboxLimits,
)

SUITE = Path("benchmarks/prompts/agentic-quality2-v2.json")


def _payloads() -> tuple[dict, dict, dict]:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    oracle_path = SUITE.parent / suite["quality_oracle"]["path"]
    source_path = SUITE.parent / suite["source_manifest"]["path"]
    return (
        suite,
        json.loads(oracle_path.read_text(encoding="utf-8")),
        json.loads(source_path.read_text(encoding="utf-8")),
    )


def _write_fixture(
    tmp_path: Path,
    suite: dict,
    oracle: dict,
    sources: dict,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    oracle_path = tmp_path / "oracle.json"
    source_path = tmp_path / "sources.json"
    oracle_path.write_text(json.dumps(oracle, sort_keys=True), encoding="utf-8")
    source_path.write_text(json.dumps(sources, sort_keys=True), encoding="utf-8")
    suite = copy.deepcopy(suite)
    suite["quality_oracle"] = {
        "path": "oracle.json",
        "file_sha256": __import__("hashlib").sha256(oracle_path.read_bytes()).hexdigest(),
    }
    suite["source_manifest"] = {
        "path": "sources.json",
        "file_sha256": __import__("hashlib").sha256(source_path.read_bytes()).hexdigest(),
    }
    oracle["source_manifest"] = copy.deepcopy(suite["source_manifest"])
    oracle_path.write_text(json.dumps(oracle, sort_keys=True), encoding="utf-8")
    suite["quality_oracle"]["file_sha256"] = (
        __import__("hashlib").sha256(oracle_path.read_bytes()).hexdigest()
    )
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite, sort_keys=True), encoding="utf-8")
    return suite_path


def test_loader_accepts_frozen_suite_and_proves_all_reference_cases() -> None:
    loaded = load_agentic_quality2_suite(SUITE)

    assert len(loaded.workloads) == 34
    assert len(loaded.development_ids) == len(loaded.heldout_ids) == 17
    results = [execute_reference_case(loaded, case_id) for case_id in loaded.workloads]
    assert all(result["passed"] is True for result in results)
    assert {result["kind"] for result in results} == {
        "read",
        "search",
        "lookup",
        "transform",
        "patch",
        "test",
        "code",
        "instruction",
        "multiple",
        "no_tool",
    }


def test_all_fail_safe_control_oracles_pass_independently() -> None:
    loaded = load_agentic_quality2_suite(SUITE)

    results = [
        evaluate_quality2_fail_safe_control(loaded, row["id"])
        for row in loaded.oracle["fail_safe_controls"]
    ]

    assert len(results) == 10
    assert all(result["passed"] is True for result in results)
    assert {result["class"] for result in results} == {
        "malformed",
        "truncated",
        "duplicate",
        "undeclared",
        "schema_invalid",
        "content_leak",
        "reasoning_leak",
        "required_missing",
        "ambiguous",
        "tool_none_violation",
    }


def test_loader_rejects_duplicate_ids_and_split_overlap(tmp_path: Path) -> None:
    suite, oracle, sources = _payloads()
    duplicate = copy.deepcopy(suite)
    duplicate["workloads"].append(copy.deepcopy(duplicate["workloads"][0]))
    with pytest.raises(AgenticQuality2Error, match="duplicate workload"):
        load_agentic_quality2_suite(
            _write_fixture(tmp_path / "duplicate", duplicate, oracle, sources)
        )

    overlap = copy.deepcopy(suite)
    overlap["split_policy"]["heldout_ids"].append(overlap["split_policy"]["development_ids"][0])
    with pytest.raises(AgenticQuality2Error, match="split overlap"):
        load_agentic_quality2_suite(_write_fixture(tmp_path / "overlap", overlap, oracle, sources))


def test_loader_rejects_missing_language_and_malformed_counts(tmp_path: Path) -> None:
    suite, oracle, sources = _payloads()
    missing = copy.deepcopy(suite)
    del missing["workloads"][0]["language"]
    with pytest.raises(AgenticQuality2Error, match="language"):
        load_agentic_quality2_suite(_write_fixture(tmp_path / "language", missing, oracle, sources))

    malformed = copy.deepcopy(suite)
    malformed["split_policy"]["heldout_ids"].pop()
    with pytest.raises(AgenticQuality2Error, match="split manifest"):
        load_agentic_quality2_suite(_write_fixture(tmp_path / "counts", malformed, oracle, sources))

    reduced = copy.deepcopy(suite)
    reduced_oracle = copy.deepcopy(oracle)
    removed = reduced["workloads"].pop()
    reduced["split_policy"]["heldout_ids"].remove(removed["id"])
    del reduced_oracle["cases"][removed["id"]]
    with pytest.raises(AgenticQuality2Error, match="expanded split count"):
        load_agentic_quality2_suite(
            _write_fixture(tmp_path / "family-count", reduced, reduced_oracle, sources)
        )


def test_loader_rejects_oracle_mismatch_and_unknown_case(tmp_path: Path) -> None:
    suite, oracle, sources = _payloads()
    mismatch = copy.deepcopy(oracle)
    case_id = next(iter(mismatch["cases"]))
    mismatch["cases"][case_id]["split"] = "heldout"
    with pytest.raises(AgenticQuality2Error, match="oracle split"):
        load_agentic_quality2_suite(_write_fixture(tmp_path / "mismatch", suite, mismatch, sources))

    unknown = copy.deepcopy(suite)
    unknown["workloads"][0]["turns"][0]["oracle_case"] = "unknown-case"
    with pytest.raises(AgenticQuality2Error, match="unknown oracle case"):
        load_agentic_quality2_suite(_write_fixture(tmp_path / "unknown", unknown, oracle, sources))


def test_loader_rejects_expected_answer_leakage(tmp_path: Path) -> None:
    suite, oracle, sources = _payloads()
    leaked_hash = copy.deepcopy(suite)
    leaked_hash["workloads"][0]["turns"][0]["user"] += (
        " " + next(iter(oracle["cases"].values()))["expected_result_sha256"]
    )
    with pytest.raises(AgenticQuality2Error, match="expected-answer leakage"):
        load_agentic_quality2_suite(_write_fixture(tmp_path / "hash", leaked_hash, oracle, sources))

    leaked_test = copy.deepcopy(suite)
    hidden = next(iter(oracle["code_cases"].values()))["hidden_tests"][0]
    leaked_test["workloads"][0]["turns"][0]["user"] += " " + json.dumps(
        hidden,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(AgenticQuality2Error, match="expected-answer leakage"):
        load_agentic_quality2_suite(
            _write_fixture(tmp_path / "hidden", leaked_test, oracle, sources)
        )


def test_expanded_oracle_evaluates_single_multiple_no_tool_and_instruction() -> None:
    loaded = load_agentic_quality2_suite(SUITE)

    nested = evaluate_quality2_oracle(
        loaded,
        workload_id="aq2_dev_tool_nested_alert_en",
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
        ],
    )
    multiple = evaluate_quality2_oracle(
        loaded,
        workload_id="aq2_dev_tool_multiple_en",
        calls=[
            {"tool": "calculate", "arguments": {"expression": "18 * 7"}},
            {
                "tool": "lookup_record",
                "arguments": {"key": "station_owner", "locale": "en"},
            },
        ],
    )
    no_tool = evaluate_quality2_oracle(
        loaded,
        workload_id="aq2_dev_tool_irrelevant_en",
        calls=[],
        public_text="Hello. Thank you for checking in.",
    )
    instruction = evaluate_quality2_oracle(
        loaded,
        workload_id="aq2_dev_instruction_status_en",
        calls=[
            {
                "tool": "submit_response",
                "arguments": {"text": "STATUS: green cache ready"},
            }
        ],
    )
    wrong = evaluate_quality2_oracle(
        loaded,
        workload_id="aq2_dev_tool_nested_alert_en",
        calls=[
            {
                "tool": "transform_record",
                "arguments": {
                    "record": {
                        "station": "north-2",
                        "priority": "high",
                        "tags": ["night", "edge"],
                    },
                    "mode": "normalize",
                    "include_metadata": True,
                },
            }
        ],
    )

    assert all(row["status"] == "passed" for row in (nested, multiple, no_tool, instruction))
    assert wrong["status"] == "failed"


def test_expanded_oracle_runs_code_only_through_qualified_sandbox(tmp_path: Path) -> None:
    loaded = load_agentic_quality2_suite(SUITE)
    calls = [
        {
            "tool": "submit_code",
            "arguments": {
                "entry_point": "clamp_readings",
                "source": (
                    "def clamp_readings(values, low, high):\n"
                    "    return [max(low, min(high, value)) for value in values]\n"
                ),
            },
        }
    ]
    passed = evaluate_quality2_oracle(
        loaded,
        workload_id="aq2_dev_code_clamp_en",
        calls=calls,
        sandbox=AgenticQuality2Sandbox(),
    )
    blocked = evaluate_quality2_oracle(
        loaded,
        workload_id="aq2_dev_code_clamp_en",
        calls=calls,
        sandbox=AgenticQuality2Sandbox(bwrap_path=tmp_path / "missing"),
    )

    assert passed["status"] == "passed"
    assert passed["sandbox"]["tests_passed"] == 4
    assert blocked["status"] == "blocked_sandbox"
    assert blocked["success"] is False


def test_quality2_aggregation_is_deterministic_and_seals_heldout_detail() -> None:
    loaded = load_agentic_quality2_suite(SUITE)
    rows = []
    for workload_id, workload in loaded.workloads.items():
        for repetition in range(2):
            status = "blocked_sandbox" if workload_id == "aq2_held_code_rolling_ja" else "passed"
            rows.append(
                {
                    "workload_id": workload_id,
                    "split": workload["split"],
                    "family": workload["family"],
                    "language": workload["language"],
                    "kind": loaded.oracle["cases"][workload["turns"][0]["oracle_case"]]["kind"],
                    "status": status,
                    "success": status == "passed",
                    "result_sha256": None if status != "passed" else "a" * 64,
                    "error": None,
                    "repetition": repetition,
                }
            )

    first = aggregate_quality2_results(loaded, rows, expected_repetitions=2)
    second = aggregate_quality2_results(
        loaded,
        list(reversed(rows)),
        expected_repetitions=2,
    )

    assert first == second
    assert first["overall"] == {
        "observations": 68,
        "passed": 66,
        "failed": 0,
        "blocked_sandbox": 2,
        "unscorable": 0,
        "scored_denominator": 66,
        "success_rate": 1.0,
    }
    assert first["heldout_details"] == []
    assert first["heldout_details_sealed"] is True
    assert first["determinism"]["passed"] is True
    assert "generated_token_ids" not in json.dumps(first, sort_keys=True)


def test_quality2_aggregation_rejects_incomplete_and_nondeterministic_rows() -> None:
    loaded = load_agentic_quality2_suite(SUITE)
    rows = []
    for workload_id, workload in loaded.workloads.items():
        for repetition in range(2):
            rows.append(
                {
                    "workload_id": workload_id,
                    "split": workload["split"],
                    "family": workload["family"],
                    "language": workload["language"],
                    "kind": loaded.oracle["cases"][workload["turns"][0]["oracle_case"]]["kind"],
                    "status": "passed",
                    "success": True,
                    "result_sha256": "a" * 64,
                    "error": None,
                    "repetition": repetition,
                }
            )
    with pytest.raises(AgenticQuality2Error, match="repetitions are incomplete"):
        aggregate_quality2_results(loaded, rows[:-1], expected_repetitions=2)

    split_tamper = copy.deepcopy(rows)
    heldout_index = next(
        index for index, row in enumerate(split_tamper) if row["split"] == "heldout"
    )
    split_tamper[heldout_index]["split"] = "development"
    with pytest.raises(AgenticQuality2Error, match="split/family"):
        aggregate_quality2_results(loaded, split_tamper, expected_repetitions=2)

    duplicate_repetition = copy.deepcopy(rows)
    duplicate_repetition[1]["repetition"] = 0
    with pytest.raises(AgenticQuality2Error, match="repetition identities"):
        aggregate_quality2_results(loaded, duplicate_repetition, expected_repetitions=2)

    rows[1]["result_sha256"] = "b" * 64
    artifact = aggregate_quality2_results(loaded, rows, expected_repetitions=2)
    reversed_artifact = aggregate_quality2_results(
        loaded,
        list(reversed(rows)),
        expected_repetitions=2,
    )
    assert artifact == reversed_artifact
    assert artifact["determinism"]["passed"] is False
    assert artifact["determinism"]["mismatches"][0]["workload_id"] in loaded.workloads


def test_sandbox_executes_valid_function_and_reports_bounded_result(tmp_path: Path) -> None:
    sandbox = AgenticQuality2Sandbox(
        limits=SandboxLimits(wall_seconds=2.0, cpu_seconds=1, memory_bytes=128 << 20)
    )
    result = sandbox.run_code_case(
        source=(
            "def clamp_readings(values, low, high):\n"
            "    return [max(low, min(high, x)) for x in values]\n"
        ),
        entry_point="clamp_readings",
        hidden_tests=[{"args": [[-2, 4, 11], 0, 8], "kwargs": {}, "expected": [0, 4, 8]}],
        scratch_root=tmp_path,
    )

    assert result["status"] == "passed"
    assert result["tests_passed"] == 1
    assert result["tests_attempted"] == 1
    assert result["network_isolated"] is True
    assert result["filesystem_isolated"] is True
    assert result["device_isolated"] is True


@pytest.mark.parametrize(
    ("source", "failure", "allowed_imports"),
    [
        (
            "def probe(*args):\n import socket\n socket.create_connection(('127.0.0.1', 9), 0.1)\n",
            "network",
            ("socket",),
        ),
        (
            "def probe(*args):\n return open('/etc/passwd').read()\n",
            "filesystem",
            (),
        ),
        (
            "def probe(*args):\n return open('/home/lhl/hipEngine/README.md').read()\n",
            "repository",
            (),
        ),
        (
            "def probe(*args):\n return open('/dev/kfd', 'rb').read(1)\n",
            "device",
            (),
        ),
        (
            "def probe(*args):\n import os\n return os.getenv('AQ2_SANDBOX_SECRET')\n",
            "secret",
            ("os",),
        ),
        (
            (
                "def probe(*args):\n"
                " import subprocess\n"
                " subprocess.Popen(['/usr/bin/sleep','30'])\n"
                " return 1\n"
            ),
            "process",
            ("subprocess",),
        ),
    ],
)
def test_sandbox_blocks_host_escape_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    failure: str,
    allowed_imports: tuple[str, ...],
) -> None:
    monkeypatch.setenv("AQ2_SANDBOX_SECRET", "must-not-leak")
    sandbox = AgenticQuality2Sandbox(
        limits=SandboxLimits(wall_seconds=1.0, cpu_seconds=1, memory_bytes=96 << 20)
    )
    result = sandbox.run_code_case(
        source=source,
        entry_point="probe",
        hidden_tests=[{"args": [], "kwargs": {}, "expected": "never"}],
        scratch_root=tmp_path / failure,
        allowed_imports=allowed_imports,
    )

    assert result["status"] == "failed"
    if failure == "secret":
        assert result["failure"]["kind"] == "wrong_result"
    else:
        assert result["failure"]["kind"] == "exception"
    assert "must-not-leak" not in result["stdout"]
    assert "must-not-leak" not in result["stderr"]
    assert result["network_isolated"] is True
    assert result["filesystem_isolated"] is True
    assert result["device_isolated"] is True


def test_sandbox_never_mounts_hidden_expected_values(tmp_path: Path) -> None:
    marker = "hidden-expected-must-never-enter-sandbox"
    sandbox = AgenticQuality2Sandbox()
    result = sandbox.run_code_case(
        source=(
            "def inspect_inputs():\n"
            "    import os\n"
            "    return '|'.join(open('/input/' + name).read() "
            "for name in sorted(os.listdir('/input')) "
            "if os.path.isfile('/input/' + name))\n"
        ),
        entry_point="inspect_inputs",
        hidden_tests=[{"args": [], "kwargs": {}, "expected": marker}],
        scratch_root=tmp_path,
        allowed_imports=("os",),
    )

    assert result["status"] == "failed"
    assert result["failure"]["kind"] == "wrong_result"
    assert result["hidden_expected_exposed"] is False
    assert marker not in result["stdout"]
    assert marker not in result["stderr"]


def test_sandbox_enforces_wall_memory_file_and_output_limits(tmp_path: Path) -> None:
    sandbox = AgenticQuality2Sandbox(
        limits=SandboxLimits(
            wall_seconds=0.25,
            cpu_seconds=1,
            memory_bytes=64 << 20,
            file_bytes=4096,
            output_bytes=1024,
        )
    )
    timeout = sandbox.run_code_case(
        source="def spin():\n    while True: pass\n",
        entry_point="spin",
        hidden_tests=[{"args": [], "kwargs": {}, "expected": 1}],
        scratch_root=tmp_path / "timeout",
    )
    memory = sandbox.run_code_case(
        source="def allocate():\n    return bytearray(200_000_000)\n",
        entry_point="allocate",
        hidden_tests=[{"args": [], "kwargs": {}, "expected": None}],
        scratch_root=tmp_path / "memory",
    )
    file_size = sandbox.run_code_case(
        source=(
            "def write_large():\n"
            "    with open('/work/payload', 'wb') as f:\n"
            "        f.write(b'x' * 20000)\n"
            "    return 1\n"
        ),
        entry_point="write_large",
        hidden_tests=[{"args": [], "kwargs": {}, "expected": 1}],
        scratch_root=tmp_path / "file-size",
    )
    output = sandbox.run_code_case(
        source="def noisy():\n    print('x' * 20000)\n    return 1\n",
        entry_point="noisy",
        hidden_tests=[{"args": [], "kwargs": {}, "expected": 1}],
        scratch_root=tmp_path / "output",
    )

    assert timeout["status"] == "timeout"
    assert timeout["process_group_killed"] is True
    assert memory["status"] == "failed"
    assert file_size["status"] == "failed"
    assert output["status"] == "failed"
    assert output["output_truncated"] is True
    assert len(output["stdout"].encode()) <= 1024


def test_sandbox_fails_closed_when_bwrap_is_unavailable(tmp_path: Path) -> None:
    sandbox = AgenticQuality2Sandbox(bwrap_path=tmp_path / "missing-bwrap")

    result = sandbox.run_code_case(
        source="def f(): return 1",
        entry_point="f",
        hidden_tests=[{"args": [], "kwargs": {}, "expected": 1}],
        scratch_root=tmp_path,
    )

    assert result["status"] == "blocked_sandbox"
    assert result["tests_attempted"] == 0
