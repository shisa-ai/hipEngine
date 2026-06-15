#!/usr/bin/env python3
"""Emit a StepFun oracle-parity source-map artifact.

The source map turns the retained oracle next-action handoff into concrete
in-tree owners: prompt/logit smoke generation, runtime logits probes, llama.cpp
oracle execution, and blocker diagnosis artifacts. It is navigation evidence for
the next logits/backend parity investigation and makes no oracle, KV, e2e, or
performance claim.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_host_logit_margin as host_margin
from scripts import stepfun_oracle_blocker_diagnosis as diagnosis_mod
from scripts import stepfun_oracle_evidence_consistency_check as consistency_mod
from scripts import stepfun_oracle_next_action_manifest as next_action_mod
from scripts import stepfun_oracle_rank_check as rank_check
from scripts import stepfun_oracle_token_mismatch as token_mismatch

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-source-map.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to resolve source paths.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the source-map artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write JSON output atomically to this path instead of stdout. Use "
            "--default-output for the canonical StepFun artifact path."
        ),
    )
    parser.add_argument(
        "--default-output",
        action="store_true",
        help=f"Write to the canonical artifact path: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--status-only", action="store_true", help="Emit only status.")
    parser.add_argument(
        "--missing-symbols-only",
        action="store_true",
        help="Emit only missing symbol records.",
    )
    parser.add_argument(
        "--entry-keys-only",
        action="store_true",
        help="Emit only source-map entry keys.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    parser.add_argument(
        "--verify-source-map",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted oracle source-map artifact with current "
            f"in-tree symbol and artifact metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-source-map, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-source-map, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-source-map, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def collect_python_symbols(path: Path) -> dict[str, int]:
    """Return top-level and class-qualified symbol line numbers for a Python file."""

    tree = ast.parse(path.read_text())
    symbols: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[node.name] = int(node.lineno)
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols[f"{node.name}.{child.name}"] = int(child.lineno)
    return symbols


def _source_record(
    *,
    repo_root: Path,
    path: str,
    required_symbols: Sequence[str],
    role: str,
) -> dict[str, object]:
    source_path = repo_root / path
    if not source_path.exists():
        return {
            "path": path,
            "role": role,
            "exists": False,
            "required_symbols": list(required_symbols),
            "present_symbols": [],
            "missing_symbols": list(required_symbols),
            "all_symbols_present": False,
        }
    symbols = collect_python_symbols(source_path)
    present = [symbol for symbol in required_symbols if symbol in symbols]
    missing = [symbol for symbol in required_symbols if symbol not in symbols]
    return {
        "path": path,
        "role": role,
        "exists": True,
        "required_symbols": list(required_symbols),
        "present_symbols": [
            {"symbol": symbol, "line": symbols[symbol]} for symbol in present
        ],
        "missing_symbols": missing,
        "all_symbols_present": not missing,
    }


def _entry(
    *,
    key: str,
    role: str,
    sources: Sequence[dict[str, object]],
    artifacts: Sequence[Path],
    commands: Sequence[str],
    next_focus: str,
) -> dict[str, object]:
    missing_symbols = [
        {"path": source["path"], "symbol": symbol}
        for source in sources
        for symbol in source.get("missing_symbols", [])
    ]
    return {
        "key": key,
        "role": role,
        "sources": list(sources),
        "artifacts": [str(path) for path in artifacts],
        "commands": list(commands),
        "next_focus": next_focus,
        "all_symbols_present": not missing_symbols,
        "missing_symbols": missing_symbols,
    }


def _artifact_record(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None}
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "exists": True,
        "artifact_kind": payload.get("artifact_kind") if isinstance(payload, dict) else None,
        "status": payload.get("status") if isinstance(payload, dict) else None,
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _flatten_missing(entries: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"entry_key": entry["key"], **missing}
        for entry in entries
        for missing in entry.get("missing_symbols", [])
    ]


def build_oracle_source_map(
    *,
    repo_root: Path = REPO_ROOT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the in-tree source map for the oracle parity investigation."""

    repo_root = repo_root.resolve()
    prompt_smoke = _entry(
        key="host_prompt_logit_smoke",
        role="Owns retained host prompt, input IDs, top-token logits, and top-k visible margin.",
        sources=[
            _source_record(
                repo_root=repo_root,
                path="scripts/stepfun_layer_prefix_smoke.py",
                required_symbols=("main", "_run_chunked_prefix", "_emit_json"),
                role="CLI that produced the retained all-45 prompt smoke artifact.",
            ),
            _source_record(
                repo_root=repo_root,
                path="hipengine/runtime/stepfun_gguf_runner.py",
                required_symbols=(
                    "StepFunLayerPrefixLogitsProbe",
                    "StepFunResidentSession.embed_chat_prompt_bf16",
                    "StepFunResidentSession.layer_prefix_prompt_logits_probe_bf16",
                    "StepFunResidentSession.layer_prefill_probe_bf16",
                    "StepFunResidentSession.final_logits_probe_bf16",
                ),
                role="Torch-free resident runtime methods that produce host prompt logits.",
            ),
        ],
        artifacts=(
            status_mod.DEFAULT_PROMPT_ARTIFACT,
            host_margin.DEFAULT_OUTPUT,
        ),
        commands=(
            "python3 scripts/stepfun_layer_prefix_smoke.py --layer-count 45 --message hello --output benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json --pretty",
            "python3 scripts/stepfun_host_logit_margin.py --default-output --pretty",
        ),
        next_focus="Compare the retained host prompt/logit path with llama.cpp logits for the same prompt before touching KV or e2e decode.",
    )
    oracle_runner = _entry(
        key="llamacpp_oracle_runner",
        role="Owns canonical llama.cpp one-token oracle command construction and output comparison.",
        sources=[
            _source_record(
                repo_root=repo_root,
                path="scripts/stepfun_llamacpp_oracle.py",
                required_symbols=(
                    "main",
                    "_run_with_timeout",
                    "_comparison_fields",
                    "_partial_execution_result",
                ),
                role="Builds/runs the llama.cpp one-token oracle and records timeout/executed artifacts.",
            ),
            _source_record(
                repo_root=repo_root,
                path="scripts/stepfun_oracle_token_mismatch.py",
                required_symbols=("main", "build_oracle_token_mismatch_summary"),
                role="Summarizes canonical oracle generated text and tokenizer diagnostics.",
            ),
            _source_record(
                repo_root=repo_root,
                path="scripts/stepfun_oracle_rank_check.py",
                required_symbols=("main", "build_oracle_rank_check"),
                role="Compares llama.cpp generated token with retained host top-token IDs.",
            ),
        ],
        artifacts=(
            status_mod.DEFAULT_ORACLE_ARTIFACT,
            token_mismatch.DEFAULT_OUTPUT,
            rank_check.DEFAULT_OUTPUT,
        ),
        commands=(
            "python3 scripts/stepfun_llamacpp_oracle.py --execute --timeout-s 900 --output benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-step35-timeout.json --pretty",
            "python3 scripts/stepfun_oracle_token_mismatch.py --default-output --pretty",
            "python3 scripts/stepfun_oracle_rank_check.py --default-output --pretty",
        ),
        next_focus="Re-run or instrument llama.cpp only after confirming the prompt/logit source artifacts are current.",
    )
    blocker_handoff = _entry(
        key="oracle_blocker_handoff",
        role="Owns consolidated blocker diagnosis, consistency gate, and next-action handoff.",
        sources=[
            _source_record(
                repo_root=repo_root,
                path="scripts/stepfun_oracle_blocker_diagnosis.py",
                required_symbols=("main", "build_oracle_blocker_diagnosis"),
                role="Consolidates ruled-out causes and active oracle findings.",
            ),
            _source_record(
                repo_root=repo_root,
                path="scripts/stepfun_oracle_evidence_consistency_check.py",
                required_symbols=("main", "build_oracle_evidence_consistency_check"),
                role="Checks retained oracle evidence for stale or contradictory artifacts.",
            ),
            _source_record(
                repo_root=repo_root,
                path="scripts/stepfun_oracle_next_action_manifest.py",
                required_symbols=("main", "build_oracle_next_action_manifest"),
                role="Pins the next investigation to canonical Vulkan logits/backend parity.",
            ),
        ],
        artifacts=(
            diagnosis_mod.DEFAULT_OUTPUT,
            consistency_mod.DEFAULT_OUTPUT,
            next_action_mod.DEFAULT_OUTPUT,
        ),
        commands=(
            "python3 scripts/stepfun_oracle_blocker_diagnosis.py --default-output --pretty",
            "python3 scripts/stepfun_oracle_evidence_consistency_check.py --default-output --pretty",
            "python3 scripts/stepfun_oracle_next_action_manifest.py --default-output --pretty",
        ),
        next_focus="Keep these handoff artifacts matching while investigating generated_text_matches_target.",
    )
    entries = [prompt_smoke, oracle_runner, blocker_handoff]
    missing_symbols = _flatten_missing(entries)
    artifacts = [
        status_mod.DEFAULT_PROMPT_ARTIFACT,
        status_mod.DEFAULT_ORACLE_ARTIFACT,
        host_margin.DEFAULT_OUTPUT,
        token_mismatch.DEFAULT_OUTPUT,
        rank_check.DEFAULT_OUTPUT,
        diagnosis_mod.DEFAULT_OUTPUT,
        consistency_mod.DEFAULT_OUTPUT,
        next_action_mod.DEFAULT_OUTPUT,
    ]
    missing_artifacts = [str(path) for path in artifacts if not path.exists()]
    ready = not missing_symbols and not missing_artifacts
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_oracle_source_map",
        "date": artifact_date,
        "status": "mapped" if ready else "incomplete",
        "ready": ready,
        "repo_root": str(repo_root),
        "next_action_kind": "logits_backend_parity_investigation",
        "unresolved_evidence_gap": "generated_text_matches_target",
        "entry_count": len(entries),
        "entries": entries,
        "all_symbols_present": not missing_symbols,
        "missing_symbols": missing_symbols,
        "artifact_records": [_artifact_record(path) for path in artifacts],
        "missing_artifacts": missing_artifacts,
        "source_map_summary": {
            "host_prompt_logits_owner": "scripts/stepfun_layer_prefix_smoke.py + StepFunResidentSession logits probes",
            "canonical_oracle_owner": "scripts/stepfun_llamacpp_oracle.py + token/rank diagnostics",
            "handoff_owner": "oracle diagnosis, consistency, and next-action manifest scripts",
        },
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This source map only identifies in-tree owners for the next "
                "logits/backend parity investigation; generated_text_matches_target remains unresolved."
            ),
        },
    }


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def verify_oracle_source_map(
    source_map_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted oracle source-map artifact with current metadata."""

    persisted = _load_json_object(source_map_artifact)
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "oracle_source_map_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted oracle source-map artifact differs from current "
                    "in-tree symbol or retained-artifact metadata."
                ),
            }
        )
    all_match = not failures
    persisted_entries = persisted.get("entries")
    current_entries = current_report.get("entries")
    return {
        "schema_version": 1,
        "artifact_path": str(source_map_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status"),
        "current_status": current_report.get("status"),
        "persisted_ready": persisted.get("ready"),
        "current_ready": current_report.get("ready"),
        "persisted_entry_keys": [
            entry.get("key")
            for entry in persisted_entries
            if isinstance(entry, dict)
        ]
        if isinstance(persisted_entries, list)
        else [],
        "current_entry_keys": [
            entry.get("key")
            for entry in current_entries
            if isinstance(entry, dict)
        ]
        if isinstance(current_entries, list)
        else [],
        "persisted_missing_symbol_count": len(persisted.get("missing_symbols", []))
        if isinstance(persisted.get("missing_symbols"), list)
        else None,
        "current_missing_symbol_count": len(current_report.get("missing_symbols", []))
        if isinstance(current_report.get("missing_symbols"), list)
        else None,
        "persisted_missing_artifact_count": len(persisted.get("missing_artifacts", []))
        if isinstance(persisted.get("missing_artifacts"), list)
        else None,
        "current_missing_artifact_count": len(current_report.get("missing_artifacts", []))
        if isinstance(current_report.get("missing_artifacts"), list)
        else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_oracle_source_map(repo_root=args.repo_root, artifact_date=args.artifact_date)
    if args.verify_source_map is not None:
        verification = verify_oracle_source_map(
            args.verify_source_map,
            current_report=report,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        elif args.verification_sha_only:
            payload = status_mod._stable_json_sha256(verification)
        else:
            payload = verification
        output = DEFAULT_OUTPUT if args.default_output else args.output
        status_mod._emit_json(payload, pretty=args.pretty, output=output)
        return (
            0
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    if args.status_only:
        payload: object = report["status"]
    elif args.missing_symbols_only:
        payload = report["missing_symbols"]
    elif args.entry_keys_only:
        payload = [entry["key"] for entry in report["entries"]]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
