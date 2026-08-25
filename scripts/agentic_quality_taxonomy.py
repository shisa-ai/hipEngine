#!/usr/bin/env python3
"""Build a compact independent failure taxonomy from agentic quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.agentic import (  # noqa: E402
    AgenticBenchmarkError,
    load_agentic_workload_suite,
)
from hipengine.benchmark.agentic_quality import (  # noqa: E402
    AGENTIC_QUALITY_RECORDS_KIND,
)
from hipengine.benchmark.agentic_quality_taxonomy import (  # noqa: E402
    classify_agentic_quality_observation,
)
from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from hipengine.tokenization.identity import token_ids_sha256  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AgenticBenchmarkError(f"{path} must contain a JSON object")
    return payload


def _git_head() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def _generated_ids(raw_response: dict[str, Any]) -> tuple[int, ...]:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AgenticBenchmarkError("taxonomy raw response must contain one choice")
    hipengine = choices[0].get("hipengine")
    token_ids = hipengine.get("generated_token_ids") if isinstance(hipengine, dict) else None
    if not isinstance(token_ids, list) or not token_ids:
        raise AgenticBenchmarkError("taxonomy raw response omits generated token ids")
    return tuple(int(token) for token in token_ids)


def _task_group(
    classifications: list[dict[str, Any]],
    *,
    expected_repetitions: int,
) -> dict[str, Any]:
    first = classifications[0]
    if len(classifications) != expected_repetitions:
        raise AgenticBenchmarkError("taxonomy task group has incomplete repetitions")
    stable_fields = (
        "primary_outcome",
        "owner",
        "earliest_bad_boundary",
        "contributing_causes",
        "expected",
        "observed",
    )
    for row in classifications[1:]:
        if any(row[field] != first[field] for field in stable_fields):
            raise AgenticBenchmarkError("taxonomy task group is not repeat-stable")
        if row["independent_parser"] != first["independent_parser"]:
            raise AgenticBenchmarkError("taxonomy parser evidence is not repeat-stable")
    return {
        "workload_id": first["workload_id"],
        "turn_index": first["turn_index"],
        "agent_id": first["agent_id"],
        "observation_request_ids": [row["request_id"] for row in classifications],
        "repeat_stable": True,
        "primary_outcome": first["primary_outcome"],
        "owner": first["owner"],
        "earliest_bad_boundary": first["earliest_bad_boundary"],
        "contributing_causes": first["contributing_causes"],
        "expected": first["expected"],
        "observed": first["observed"],
        "independent_parser": first["independent_parser"],
        "prompt_token_roundtrip": first["prompt_token_roundtrip"],
    }


def build_taxonomy(
    *,
    records_path: Path,
    checkpoint_path: Path,
    workloads_path: Path,
    model_path: Path,
    baseline_path: Path,
) -> dict[str, Any]:
    records = _load_json(records_path)
    checkpoint = _load_json(checkpoint_path)
    baseline = _load_json(baseline_path)
    if records.get("kind") != AGENTIC_QUALITY_RECORDS_KIND:
        raise AgenticBenchmarkError("taxonomy records kind is not agentic quality records")
    if checkpoint.get("kind") != "hipengine_agentic_coding_quality_checkpoint":
        raise AgenticBenchmarkError("taxonomy checkpoint kind is invalid")
    if checkpoint.get("status") != "complete":
        raise AgenticBenchmarkError("taxonomy checkpoint is incomplete")
    if baseline.get("kind") != "hipengine_agentic_quality_campaign_baseline":
        raise AgenticBenchmarkError("taxonomy baseline kind is invalid")
    evidence_files = baseline.get("raw_evidence", {}).get("files", {})
    expected_records_hash = evidence_files.get(records_path.name, {}).get("sha256")
    expected_checkpoint_hash = evidence_files.get(checkpoint_path.name, {}).get("sha256")
    if _sha256(records_path) != expected_records_hash:
        raise AgenticBenchmarkError("taxonomy records hash differs from AQ2 baseline")
    if _sha256(checkpoint_path) != expected_checkpoint_hash:
        raise AgenticBenchmarkError("taxonomy checkpoint hash differs from AQ2 baseline")

    suite = load_agentic_workload_suite(workloads_path)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(model_path).info)
    raw_turns = checkpoint.get("raw_turns")
    normalized = records.get("turn_records")
    if not isinstance(raw_turns, list) or not isinstance(normalized, list):
        raise AgenticBenchmarkError("taxonomy evidence omits turn arrays")
    raw_by_request = {
        str(row["request_id"]): row
        for row in raw_turns
        if isinstance(row, dict) and "request_id" in row
    }
    if len(raw_by_request) != len(raw_turns):
        raise AgenticBenchmarkError("taxonomy raw request ids are missing or duplicated")

    classified: list[dict[str, Any]] = []
    for record in normalized:
        if not isinstance(record, dict):
            raise AgenticBenchmarkError("taxonomy normalized row must be an object")
        request_id = str(record.get("request_id", ""))
        raw = raw_by_request.get(request_id)
        if raw is None:
            raise AgenticBenchmarkError(f"taxonomy raw response missing for {request_id}")
        response = raw.get("response")
        prompt_ids = raw.get("prompt_token_ids")
        if not isinstance(response, dict) or not isinstance(prompt_ids, list):
            raise AgenticBenchmarkError("taxonomy raw response/prompt ids are invalid")
        generated_ids = _generated_ids(response)
        if token_ids_sha256(prompt_ids) != record["prompt"]["token_ids_sha256"]:
            raise AgenticBenchmarkError(
                "taxonomy prompt ids differ from normalized row"
            )
        if token_ids_sha256(generated_ids) != record["output"]["generated_token_ids_sha256"]:
            raise AgenticBenchmarkError("taxonomy generated ids differ from normalized row")
        if list(generated_ids) != record["output"]["generated_token_ids"]:
            raise AgenticBenchmarkError(
                "taxonomy generated id sequence differs from normalized row"
            )
        raw_text = tokenizer.decode(generated_ids)
        prompt_text = tokenizer.decode(tuple(int(token) for token in prompt_ids))
        prompt_roundtrip = tokenizer.encode(prompt_text) == [int(token) for token in prompt_ids]
        result = classify_agentic_quality_observation(
            suite,
            record,
            raw_response=response,
            raw_text=raw_text,
            prompt_token_roundtrip=prompt_roundtrip,
        )
        result["independent_parser"]["raw_control_token_ids"] = {
            "first": generated_ids[0],
            "last": generated_ids[-1],
        }
        result.update(
            {
                "request_id": request_id,
                "run_id": str(record["run_id"]),
                "workload_id": str(record["workload_id"]),
                "turn_index": int(record["turn_index"]),
                "agent_id": str(record["agent_id"]),
            }
        )
        classified.append(result)
    if len(classified) != len(normalized) or set(raw_by_request) != {
        row["request_id"] for row in classified
    }:
        raise AgenticBenchmarkError("taxonomy evidence is not one-to-one")

    expected_tool_start = tokenizer.token_to_id.get("<tool_call>")
    expected_tool_end = tokenizer.token_to_id.get("</tool_call>")
    if expected_tool_start is None or expected_tool_end is None:
        raise AgenticBenchmarkError("taxonomy tokenizer omits Qwen tool controls")
    canonical_control_rows = sum(
        row["independent_parser"]["raw_control_token_ids"]
        == {"first": expected_tool_start, "last": expected_tool_end}
        for row in classified
    )
    expected_repetitions = int(records["configuration"].get("repetitions", 1))
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in classified:
        key = (row["workload_id"], row["turn_index"], row["agent_id"])
        grouped.setdefault(key, []).append(row)
    task_groups = [
        _task_group(rows, expected_repetitions=expected_repetitions)
        for _key, rows in sorted(grouped.items())
    ]
    primary_observations = Counter(row["primary_outcome"] for row in classified)
    primary_tasks = Counter(row["primary_outcome"] for row in task_groups)
    owners = Counter(row["owner"] for row in task_groups)
    boundaries = Counter(row["earliest_bad_boundary"] for row in task_groups)
    runtime_failures = sum(
        count for owner, count in owners.items() if owner == "runtime_implementation"
    )
    unresolved = int(owners.get("unresolved", 0))
    if runtime_failures or unresolved:
        decision = "blocked: runtime or unresolved AQ2 failures remain"
    else:
        decision = "all AQ2 answer failures are model-quality-owned; freeze expanded suite next"

    return {
        "kind": "hipengine_agentic_quality_failure_taxonomy",
        "schema_version": 1,
        "status": "complete" if not runtime_failures and not unresolved else "blocked",
        "date": "2026-08-26",
        "performance_claim": False,
        "campaign": {
            "name": "AGENTIC-QUALITY2",
            "phase": "AQ3",
            "task_id": 43,
            "baseline": str(baseline_path),
        },
        "source": {
            "classifier": "agentic_quality_taxonomy_v1",
            "classifier_base_commit": _git_head(),
            "classifier_sha256": _sha256(
                REPO_ROOT / "hipengine/benchmark/agentic_quality_taxonomy.py"
            ),
            "collector_sha256": _sha256(
                REPO_ROOT / "scripts/agentic_quality_taxonomy.py"
            ),
            "baseline_source_commit": baseline["source"]["hipengine_commit"],
            "records_path": str(records_path),
            "records_sha256": _sha256(records_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "workloads_path": str(workloads_path),
            "workloads_sha256": suite.identity()["file_sha256"],
            "model_path": str(model_path),
            "model_sha256": baseline["model"]["sha256"],
            "template_tokenizer_controls": {
                "chat_template_json_sha256": baseline["model"][
                    "chat_template_json_sha256"
                ],
                "tokenizer_vocab_json_sha256": baseline["model"][
                    "tokenizer_vocab_json_sha256"
                ],
                "tool_call_start_token_id": expected_tool_start,
                "tool_call_end_token_id": expected_tool_end,
            },
        },
        "coverage": {
            "observations": len(classified),
            "task_blocks": len(task_groups),
            "expected_repetitions": expected_repetitions,
            "raw_response_matches": len(classified),
            "response_owned_id_matches": len(classified),
            "prompt_token_roundtrips": sum(
                row["prompt_token_roundtrip"] for row in classified
            ),
            "canonical_qwen_tool_control_rows": canonical_control_rows,
            "independent_parser_accepts": sum(
                row["independent_parser"]["accepted"] for row in classified
            ),
            "independent_public_projection_matches": sum(
                row["independent_parser"]["projection_matches_public"]
                for row in classified
            ),
            "schema_valid_calls": sum(
                row["observed"]["schema_valid"] for row in classified
            ),
            "terminal_tool_call_finishes": sum(
                row["observed"]["finish_reason"] == "tool_calls" for row in classified
            ),
            "empty_public_content": sum(
                row["observed"]["public_content_sha256"]
                == hashlib.sha256(b"").hexdigest()
                for row in classified
            ),
            "zero_repair_observations": sum(
                row["observed"]["repair_count"] == 0 for row in classified
            ),
            "external_oracle_evaluated": sum(
                isinstance(row["observed"]["external_oracle"], dict)
                and row["observed"]["external_oracle"].get("evaluated") is True
                for row in classified
            ),
            "repeat_stable_task_blocks": sum(row["repeat_stable"] for row in task_groups),
        },
        "rollup": {
            "primary_outcomes_observations": dict(sorted(primary_observations.items())),
            "primary_outcomes_task_blocks": dict(sorted(primary_tasks.items())),
            "owners_task_blocks": dict(sorted(owners.items())),
            "earliest_bad_boundaries_task_blocks": dict(sorted(boundaries.items())),
            "runtime_failure_task_blocks": runtime_failures,
            "unresolved_task_blocks": unresolved,
        },
        "task_blocks": task_groups,
        "repaired_pre_evidence_blockers": [
            {
                "owner": "runtime_implementation",
                "retained_in_quality_denominator": False,
                **row,
            }
            for row in baseline.get("excluded_attempts", [])
        ],
        "candidate_class_nominations": [
            {
                "class": "model_tool_selection",
                "task_blocks": int(primary_tasks.get("wrong_tool", 0)),
                "implementation_admitted": False,
            },
            {
                "class": "model_argument_grounding",
                "task_blocks": int(primary_tasks.get("wrong_arguments", 0)),
                "implementation_admitted": False,
            },
        ],
        "validation": {
            "one_to_one_raw_normalized_join": True,
            "all_prompt_token_roundtrips": all(
                row["prompt_token_roundtrip"] for row in classified
            ),
            "all_qwen_tool_controls_canonical": (
                canonical_control_rows == len(classified)
            ),
            "all_raw_envelopes_independently_parsed": all(
                row["independent_parser"]["accepted"] for row in classified
            ),
            "all_independent_public_projections_match": all(
                row["independent_parser"]["projection_matches_public"]
                for row in classified
            ),
            "all_repeats_stable": all(row["repeat_stable"] for row in task_groups),
            "runtime_failure_task_blocks": runtime_failures,
            "unresolved_task_blocks": unresolved,
            "stateless_session_commit_scope": "none",
            "final_request_ownership": baseline["ownership"][
                "final_request_session_deltas"
            ],
        },
        "decision": {
            "result": decision,
            "implementation_selected": False,
            "next": "AQ4 freezes development and heldout evaluation before candidate code",
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        taxonomy = build_taxonomy(
            records_path=args.records,
            checkpoint_path=args.checkpoint,
            workloads_path=args.workloads,
            model_path=args.model_path,
            baseline_path=args.baseline,
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(taxonomy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (AgenticBenchmarkError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"agentic quality taxonomy rejected: {exc}", file=sys.stderr)
        return 2
    print(
        f"Agentic quality taxonomy: {taxonomy['coverage']['task_blocks']} task blocks, "
        f"{taxonomy['rollup']['runtime_failure_task_blocks']} runtime, "
        f"{taxonomy['rollup']['unresolved_task_blocks']} unresolved -> {args.json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
