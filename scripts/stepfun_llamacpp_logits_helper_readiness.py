#!/usr/bin/env python3
"""Check StepFun retained-token llama-debug helper post-apply readiness.

This verifier is read-only: it inspects the external llama.cpp checkout and built
llama-debug binary to determine whether the retained-token patch has been applied,
whether the rebuilt helper advertises --token-ids / --parse-special support, and
whether the same-prompt logits artifact has been captured. It makes no oracle,
KV, e2e, or performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_llamacpp_logits_helper_contract as contract_mod
from scripts import stepfun_llamacpp_logits_helper_patch_dry_run as dry_run_mod
from scripts import stepfun_llamacpp_logits_helper_patch_plan as patch_plan_mod
from scripts import stepfun_llamacpp_logits_probe as probe_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper-readiness.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"
DEFAULT_LLAMA_DEBUG = probe_mod.DEFAULT_LLAMA_DEBUG
DEFAULT_LLAMA_LOGITS_ARTIFACT = probe_mod.DEFAULT_OUTPUT
DEFAULT_PATCH_ARTIFACT = dry_run_mod.DEFAULT_PATCH_OUTPUT
DEFAULT_PATCH_DRY_RUN_ARTIFACT = dry_run_mod.DEFAULT_OUTPUT

PATCH_APPLIED_MARKERS = (
    {
        "key": "include_sstream",
        "marker": "#include <sstream>",
        "role": "CSV token-id parser include",
    },
    {
        "key": "extra_params_struct",
        "marker": "struct stepfun_debug_extra_params",
        "role": "debug-local helper argument state",
    },
    {
        "key": "token_ids_csv_parser",
        "marker": "stepfun_parse_token_ids_csv",
        "role": "comma-separated retained token-id parser",
    },
    {
        "key": "token_ids_arg",
        "marker": 'arg == "--token-ids"',
        "role": "debug-local --token-ids argv handling",
    },
    {
        "key": "parse_special_arg",
        "marker": 'arg == "--parse-special"',
        "role": "debug-local --parse-special argv handling",
    },
    {
        "key": "retained_decode_selection",
        "marker": "common_tokenize(ctx, params.prompt, add_bos, extra.parse_special)",
        "role": "parse-special text tokenization fallback for non-token-id mode",
    },
    {
        "key": "exact_prompt_tokens_saved",
        "marker": "output_data output {ctx, model, params, tokens};",
        "role": "saved prompt metadata uses decoded retained tokens",
    },
    {
        "key": "main_threads_extra",
        "marker": "if (!run(ctx, params, stepfun_extra))",
        "role": "main threads debug-local helper options into run",
    },
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp-root", type=Path, default=contract_mod.DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--llama-debug", type=Path, default=DEFAULT_LLAMA_DEBUG)
    parser.add_argument("--patch-artifact", type=Path, default=DEFAULT_PATCH_ARTIFACT)
    parser.add_argument("--patch-dry-run-artifact", type=Path, default=DEFAULT_PATCH_DRY_RUN_ARTIFACT)
    parser.add_argument("--llama-logits-artifact", type=Path, default=DEFAULT_LLAMA_LOGITS_ARTIFACT)
    parser.add_argument("--artifact-date", default=DEFAULT_ARTIFACT_DATE)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output atomically to this path.")
    parser.add_argument("--default-output", action="store_true", help=f"Write to {DEFAULT_OUTPUT}.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--status-only", action="store_true", help="Emit only status.")
    parser.add_argument("--patch-applied-only", action="store_true", help="Emit only patch-applied boolean.")
    parser.add_argument("--helper-capable-only", action="store_true", help="Emit only helper retained-token capability boolean.")
    parser.add_argument("--ready-only", action="store_true", help="Emit only end-to-end readiness boolean.")
    parser.add_argument("--missing-evidence-only", action="store_true", help="Emit only missing evidence.")
    parser.add_argument("--sha-only", action="store_true", help="Emit stable SHA-256 of the selected payload.")
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _patch_artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _file_sha256(path),
    }


def _patch_dry_run_record(path: Path, *, patch_sha256: str | None) -> dict[str, object]:
    payload = _load_json_object(path)
    dry_run_patch_artifact = payload.get("patch_artifact") if payload else None
    dry_run_patch_sha256 = None
    dry_run_apply_command = None
    if isinstance(dry_run_patch_artifact, dict):
        dry_run_patch_sha256 = dry_run_patch_artifact.get("sha256")
        dry_run_apply_command = dry_run_patch_artifact.get("apply_command")
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _file_sha256(path),
        "status": payload.get("status") if payload else None,
        "patch_ready": payload.get("patch_ready") if payload else None,
        "git_apply_check_status": (
            payload.get("git_apply_check", {}).get("status") if payload else None
        ),
        "patch_artifact_sha256": dry_run_patch_sha256,
        "patch_artifact_sha256_matches": (
            patch_sha256 is not None and dry_run_patch_sha256 == patch_sha256
        ),
        "patch_apply_command": dry_run_apply_command,
    }


def _git_command(llama_cpp_root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=llama_cpp_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git_worktree_record(llama_cpp_root: Path) -> dict[str, object]:
    inside = _git_command(llama_cpp_root, ["rev-parse", "--is-inside-work-tree"])
    is_repo = inside.returncode == 0 and inside.stdout.strip() == "true"
    if not is_repo:
        return {
            "path": str(llama_cpp_root),
            "is_git_worktree": False,
            "head_sha": None,
            "branch": None,
            "tracked_dirty": None,
            "tracked_dirty_entries": [],
            "tracked_dirty_entry_count": None,
            "clean_for_patch_apply": False,
            "error": inside.stderr.strip() or inside.stdout.strip(),
        }
    head = _git_command(llama_cpp_root, ["rev-parse", "HEAD"])
    branch = _git_command(llama_cpp_root, ["branch", "--show-current"])
    status = _git_command(llama_cpp_root, ["status", "--porcelain", "--untracked-files=no"])
    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()]
    clean = status.returncode == 0 and not dirty_entries
    return {
        "path": str(llama_cpp_root),
        "is_git_worktree": True,
        "head_sha": head.stdout.strip() if head.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "tracked_dirty": not clean,
        "tracked_dirty_entries": dirty_entries,
        "tracked_dirty_entry_count": len(dirty_entries),
        "clean_for_patch_apply": clean,
        "error": None if clean else status.stderr.strip(),
    }


def _source_patch_markers(llama_cpp_root: Path) -> dict[str, object]:
    source_path = llama_cpp_root / dry_run_mod.DEBUG_RELATIVE_PATH
    source_text = source_path.read_text(errors="replace") if source_path.exists() else ""
    marker_records = []
    for marker in PATCH_APPLIED_MARKERS:
        marker_records.append(
            {
                "key": marker["key"],
                "marker": marker["marker"],
                "role": marker["role"],
                "present": marker["marker"] in source_text,
            }
        )
    missing = [record["key"] for record in marker_records if not record["present"]]
    return {
        "path": str(source_path),
        "exists": source_path.exists(),
        "patch_applied": source_path.exists() and not missing,
        "markers": marker_records,
        "missing_markers": missing,
    }


def _llama_debug_record(llama_debug: Path) -> dict[str, object]:
    capability = probe_mod._llama_help_capability(llama_debug)
    return {
        "path": str(llama_debug),
        "exists": llama_debug.exists(),
        "executable": os.access(llama_debug, os.X_OK),
        "version": probe_mod._llama_version(llama_debug),
        "capability": capability,
        "retained_token_ids_capable": bool(
            capability.get("same_prompt_retained_token_ids_capable") is True
        ),
        "parse_special_capable": bool(
            capability.get("same_prompt_text_tokenization_capable") is True
        ),
    }


def _llama_logits_record(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _file_sha256(path),
        "status": payload.get("status") if payload else None,
        "ready": payload.get("ready") if payload else None,
        "same_prompt_tokens_match": payload.get("same_prompt_tokens_match") if payload else None,
        "prompt_token_source": payload.get("prompt_token_source") if payload else None,
    }


def build_llamacpp_logits_helper_readiness(
    *,
    llama_cpp_root: Path = contract_mod.DEFAULT_LLAMA_CPP_ROOT,
    llama_debug: Path = DEFAULT_LLAMA_DEBUG,
    patch_artifact: Path = DEFAULT_PATCH_ARTIFACT,
    patch_dry_run_artifact: Path = DEFAULT_PATCH_DRY_RUN_ARTIFACT,
    llama_logits_artifact: Path = DEFAULT_LLAMA_LOGITS_ARTIFACT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return a read-only post-apply readiness report."""

    git_record = _git_worktree_record(llama_cpp_root)
    source_record = _source_patch_markers(llama_cpp_root)
    helper_record = _llama_debug_record(llama_debug)
    logits_record = _llama_logits_record(llama_logits_artifact)
    patch_artifact_record = _patch_artifact_record(patch_artifact)
    patch_dry_run_record = _patch_dry_run_record(
        patch_dry_run_artifact,
        patch_sha256=patch_artifact_record["sha256"],
    )

    missing_evidence: list[str] = []
    if not patch_artifact_record["exists"]:
        missing_evidence.append("llama_cpp_token_ids_helper_patch_artifact_present")
    if not patch_dry_run_record["exists"]:
        missing_evidence.append("llama_cpp_token_ids_helper_patch_dry_run_present")
    if patch_dry_run_record["patch_ready"] is not True:
        missing_evidence.append("llama_cpp_token_ids_helper_patch_dry_run_ready")
    if patch_dry_run_record["git_apply_check_status"] != "passed":
        missing_evidence.append("llama_cpp_token_ids_helper_patch_git_apply_check_passes")
    if patch_dry_run_record["patch_artifact_sha256_matches"] is not True:
        missing_evidence.append("llama_cpp_token_ids_helper_patch_artifact_matches_dry_run")
    if not git_record["clean_for_patch_apply"]:
        missing_evidence.append("llama_cpp_worktree_clean_for_patch_apply")
    if not source_record["patch_applied"]:
        missing_evidence.append("llama_cpp_token_ids_helper_patch_applied")
    if not (helper_record["exists"] and helper_record["executable"]):
        missing_evidence.append("same_prompt_logits_helper_built")
    if not helper_record["retained_token_ids_capable"]:
        missing_evidence.append("llama_debug_retained_token_ids_input_present")
    if not (
        logits_record["exists"]
        and logits_record["status"] == "captured"
        and logits_record["ready"] is True
        and logits_record["same_prompt_tokens_match"] is True
    ):
        missing_evidence.append("llama_cpp_same_prompt_logits_artifact_present")

    ready = not missing_evidence
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_logits_helper_readiness",
        "date": artifact_date,
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "llama_cpp_root": str(llama_cpp_root),
        "git_worktree": git_record,
        "patch_artifact": patch_artifact_record,
        "patch_dry_run_artifact": patch_dry_run_record,
        "source_patch": source_record,
        "llama_debug": helper_record,
        "llama_logits_artifact": logits_record,
        "required_next_commands": [
            f"git -C {llama_cpp_root} apply --unidiff-zero {(patch_artifact if patch_artifact.is_absolute() else REPO_ROOT / patch_artifact)}",
            f"cmake --build {patch_plan_mod.DEFAULT_BUILD_DIR} --target llama-debug -j",
            "python3 scripts/stepfun_llamacpp_logits_probe.py --prompt-token-source retained-input-ids --execute --default-output --pretty",
        ],
        "missing_evidence": missing_evidence,
        "blocked_reason": None if ready else "retained-token helper patch/build/capture evidence is incomplete",
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": "Readiness metadata is not same-prompt logits parity evidence.",
        },
    }


def _select_payload(report: dict[str, object], args: argparse.Namespace) -> object:
    if args.status_only:
        return report["status"]
    if args.patch_applied_only:
        return report["source_patch"]["patch_applied"]
    if args.helper_capable_only:
        return report["llama_debug"]["retained_token_ids_capable"]
    if args.ready_only:
        return report["ready"]
    if args.missing_evidence_only:
        return report["missing_evidence"]
    return report


def _write_json(payload: object, *, output: Path | None, pretty: bool) -> None:
    text = json.dumps(payload, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = DEFAULT_OUTPUT if args.default_output else args.output
    report = build_llamacpp_logits_helper_readiness(
        llama_cpp_root=args.llama_cpp_root,
        llama_debug=args.llama_debug,
        patch_artifact=args.patch_artifact,
        patch_dry_run_artifact=args.patch_dry_run_artifact,
        llama_logits_artifact=args.llama_logits_artifact,
        artifact_date=args.artifact_date,
    )
    payload = _select_payload(report, args)
    if args.sha_only:
        payload = status_mod._stable_json_sha256(payload)
    _write_json(payload, output=output, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
