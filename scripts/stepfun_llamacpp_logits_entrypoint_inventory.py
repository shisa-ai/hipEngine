#!/usr/bin/env python3
"""Inventory llama.cpp logits-dump entrypoint candidates for StepFun.

The inventory scans the local llama.cpp build bin directory and selected source
files for logits capture flags/API calls. It records whether a ready-to-run
same-prompt logits-dump binary exists and which source examples/APIs should be
used to add or build one. It is blocker evidence only and makes no oracle, KV,
e2e, or performance claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_llamacpp_logits_preflight as preflight_mod
from scripts import stepfun_llamacpp_logits_plan as plan_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-entrypoint-inventory.json"
)
DEFAULT_LLAMA_CPP_ROOT = Path("/home/lhl/llama.cpp/llama.cpp-vulkan")
DEFAULT_LLAMA_BIN_DIR = DEFAULT_LLAMA_CPP_ROOT / "build-vulkan-release/bin"
DEFAULT_ARTIFACT_DATE = "2026-06-15"
LOGITS_DUMP_HELP_MARKERS = (
    "--save-logits",
    "--logits-output-dir",
    "--save-all-logits",
    "--kl-divergence-base",
    "--logits",
    "--logits-file",
    "--dump-logits",
    "--logprobs",
    "--top-logprobs",
)
SOURCE_CANDIDATES = (
    {
        "path": "examples/debug/debug.cpp",
        "role": "existing llama-debug example that can save final logits via common debug args",
        "markers": ("params.save_logits", "llama_get_logits_ith", "save_output_data"),
        "expected_binary": "llama-debug",
    },
    {
        "path": "common/arg.cpp",
        "role": "common argument definitions for logits-related debug/perplexity flags",
        "markers": ("--save-logits", "--logits-output-dir", "--save-all-logits", "--kl-divergence-base"),
        "expected_binary": None,
    },
    {
        "path": "include/llama.h",
        "role": "public C API for retrieving decoded logits",
        "markers": ("llama_get_logits", "llama_get_logits_ith", "llama_get_sampled_logits_ith"),
        "expected_binary": None,
    },
    {
        "path": "examples/batched/batched.cpp",
        "role": "minimal generation example showing batch.logits on the prompt tail",
        "markers": ("batch.logits", "llama_decode", "llama_sampler_sample"),
        "expected_binary": "llama-batched",
    },
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-cpp-root",
        type=Path,
        default=DEFAULT_LLAMA_CPP_ROOT,
        help="Local llama.cpp source checkout to scan read-only.",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=DEFAULT_LLAMA_BIN_DIR,
        help="Local llama.cpp build binary directory to inspect.",
    )
    parser.add_argument(
        "--preflight-artifact",
        type=Path,
        default=preflight_mod.DEFAULT_OUTPUT,
        help="Retained llama.cpp logits preflight artifact.",
    )
    parser.add_argument(
        "--plan-artifact",
        type=Path,
        default=plan_mod.DEFAULT_OUTPUT,
        help="Retained llama.cpp logits plan artifact.",
    )
    parser.add_argument(
        "--help-timeout-s",
        type=float,
        default=5.0,
        help="Per-binary timeout for --help inspection.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the artifact.",
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
        "--built-logits-binary-only",
        action="store_true",
        help="Emit only whether a built binary exposes a logits dump marker.",
    )
    parser.add_argument(
        "--source-candidates-only",
        action="store_true",
        help="Emit only source candidate records.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _artifact_ref(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    if payload is None:
        return {"path": str(path), "exists": False, "sha256": None}
    return {
        "path": str(path),
        "exists": True,
        "artifact_kind": payload.get("artifact_kind"),
        "status": payload.get("status"),
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _is_executable(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    return bool(mode & stat.S_IXUSR) or os.access(path, os.X_OK)


def _marker_in_help(text: str, marker: str) -> bool:
    return re.search(rf"(?<!\S){re.escape(marker)}(?![\w-])", text) is not None


def _binary_help_markers(binary: Path, timeout_s: float) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(binary),
        "name": binary.name,
        "exists": binary.exists(),
        "executable": _is_executable(binary),
        "help_status": "not_run",
        "help_returncode": None,
        "logits_dump_markers": [],
        "logit_bias_only": False,
    }
    if not record["exists"] or not record["executable"]:
        record["help_status"] = "unavailable"
        return record
    try:
        completed = subprocess.run(
            [str(binary), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        record["help_status"] = "timeout"
        return record
    except OSError as exc:
        record["help_status"] = f"error:{type(exc).__name__}"
        record["help_error"] = str(exc)
        return record
    text = (completed.stdout or "") + (completed.stderr or "")
    markers = [marker for marker in LOGITS_DUMP_HELP_MARKERS if _marker_in_help(text, marker)]
    record.update(
        {
            "help_status": "executed",
            "help_returncode": completed.returncode,
            "logits_dump_markers": markers,
            "logit_bias_only": "--logit-bias" in text and not markers,
        }
    )
    return record


def _iter_binaries(bin_dir: Path) -> list[Path]:
    if not bin_dir.exists():
        return []
    return sorted(
        path
        for path in bin_dir.iterdir()
        if path.is_file() and path.name.startswith("llama-") and _is_executable(path)
    )


def _source_candidate(root: Path, candidate: dict[str, object], bin_dir: Path) -> dict[str, object]:
    rel_path = str(candidate["path"])
    path = root / rel_path
    markers = list(candidate["markers"])
    expected_binary = candidate.get("expected_binary")
    text = path.read_text() if path.exists() else ""
    present = [marker for marker in markers if marker in text]
    missing = [marker for marker in markers if marker not in text]
    built_binary = bin_dir / expected_binary if isinstance(expected_binary, str) else None
    return {
        "path": rel_path,
        "role": candidate["role"],
        "exists": path.exists(),
        "markers_checked": markers,
        "markers_present": present,
        "markers_missing": missing,
        "all_markers_present": not missing,
        "expected_binary": expected_binary,
        "expected_binary_built": built_binary.exists() if built_binary is not None else None,
        "candidate_kind": (
            "ready_built_binary" if built_binary is not None and built_binary.exists() else "source_only_or_api"
        ),
    }


def build_llamacpp_logits_entrypoint_inventory(
    *,
    llama_cpp_root: Path = DEFAULT_LLAMA_CPP_ROOT,
    bin_dir: Path = DEFAULT_LLAMA_BIN_DIR,
    preflight_artifact: Path = preflight_mod.DEFAULT_OUTPUT,
    plan_artifact: Path = plan_mod.DEFAULT_OUTPUT,
    help_timeout_s: float = 5.0,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the llama.cpp logits entrypoint inventory."""

    binary_records = [
        _binary_help_markers(path, timeout_s=help_timeout_s) for path in _iter_binaries(bin_dir)
    ]
    binaries_with_logits_dump = [
        record for record in binary_records if record.get("logits_dump_markers")
    ]
    source_candidates = [
        _source_candidate(llama_cpp_root, candidate, bin_dir) for candidate in SOURCE_CANDIDATES
    ]
    source_logits_candidates = [
        record for record in source_candidates if record.get("all_markers_present") is True
    ]
    debug_candidate = next(
        (record for record in source_candidates if record.get("path") == "examples/debug/debug.cpp"),
        {},
    )
    missing_evidence: list[str] = []
    if not binaries_with_logits_dump:
        missing_evidence.append("built_llamacpp_logits_dump_binary_present")
    if not source_logits_candidates:
        missing_evidence.append("llamacpp_source_logits_api_candidate_present")
    if debug_candidate.get("expected_binary_built") is not True:
        missing_evidence.append("llama_debug_binary_built")
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_logits_entrypoint_inventory",
        "date": artifact_date,
        "status": "blocked" if missing_evidence else "ready",
        "ready": not missing_evidence,
        "llama_cpp_root": str(llama_cpp_root),
        "bin_dir": str(bin_dir),
        "preflight_artifact": _artifact_ref(preflight_artifact),
        "plan_artifact": _artifact_ref(plan_artifact),
        "help_markers_checked": list(LOGITS_DUMP_HELP_MARKERS),
        "built_binary_count": len(binary_records),
        "built_binary_records": binary_records,
        "built_logits_dump_binary_present": bool(binaries_with_logits_dump),
        "built_logits_dump_binaries": binaries_with_logits_dump,
        "source_candidate_count": len(source_candidates),
        "source_candidates": source_candidates,
        "source_logits_candidate_count": len(source_logits_candidates),
        "source_logits_candidates": source_logits_candidates,
        "missing_evidence": missing_evidence,
        "next_action": (
            "build or add a llama.cpp logits-dump entrypoint, preferably from examples/debug/debug.cpp "
            "or a minimal helper using llama_get_logits_ith, then refresh the logits preflight"
        ),
        "blocked_reason": (
            "source-level logits APIs/examples exist but no ready built logits-dump binary is present"
            if source_logits_candidates and not binaries_with_logits_dump
            else "llama.cpp logits entrypoint inventory is incomplete"
        ),
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This inventory only records logits entrypoint availability; "
                "generated_text_matches_target remains unresolved."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_llamacpp_logits_entrypoint_inventory(
        llama_cpp_root=args.llama_cpp_root,
        bin_dir=args.bin_dir,
        preflight_artifact=args.preflight_artifact,
        plan_artifact=args.plan_artifact,
        help_timeout_s=args.help_timeout_s,
        artifact_date=args.artifact_date,
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.built_logits_binary_only:
        payload = report["built_logits_dump_binary_present"]
    elif args.source_candidates_only:
        payload = report["source_logits_candidates"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
