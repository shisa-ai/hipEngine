#!/usr/bin/env python3
"""Emit a metadata-only StepFun KV streaming session-contract artifact.

The artifact is produced from the current StepFun GGUF planner and a resident
session shell, but it deliberately does not materialize weights, launch kernels,
or generate a token. It is blocker/implementation-contract evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.runtime.stepfun_gguf_runner import (
    DEFAULT_STEPFUN_SHORT_CONTEXT,
    StepFunKVCacheAllocation,
    StepFunResidentSession,
    StepFunShortContextDecodePlanner,
    stepfun_kv_cache_layer_nbytes,
)
from scripts import stepfun_correctness_status as status_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-kv-session-contract.json"
)
DEFAULT_GGUF_DIR = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", "/models/gguf"))
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gguf-dir",
        type=Path,
        default=DEFAULT_GGUF_DIR,
        help="Directory containing Step-3.7-flash-Q3_K_L GGUF shards.",
    )
    parser.add_argument(
        "--backend",
        default="hip_gfx1151",
        help="Backend key used by the planner/session metadata contract.",
    )
    parser.add_argument(
        "--prompt",
        default="hello",
        help="User prompt for the short-context one-token KV decode plan.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=("low", "medium", "high"),
        help="StepFun reasoning-effort prefix for the rendered chat prompt.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_STEPFUN_SHORT_CONTEXT,
        help="Planner max-context token budget.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Planner max-new-token budget; must remain one for this evidence path.",
    )
    parser.add_argument(
        "--context-pages",
        type=int,
        default=1,
        help="Number of KV context pages for the short-context plan.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_STEPFUN_SHORT_CONTEXT,
        help="KV page size for the short-context plan.",
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
        "--blocked-by-only",
        action="store_true",
        help="Emit only contract blocked_by value.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    parser.add_argument(
        "--verify-session-contract",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted KV session-contract artifact with current "
            f"planner/session metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-session-contract, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-session-contract, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-session-contract, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def stepfun_gguf_paths(gguf_dir: Path) -> tuple[Path, ...]:
    """Return sorted StepFun Q3_K_L GGUF shard paths or raise a helpful error."""

    paths = tuple(sorted(gguf_dir.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        raise FileNotFoundError(
            "expected 3 Step-3.7-flash-Q3_K_L GGUF shards under "
            f"{gguf_dir}; set --gguf-dir or HIPENGINE_STEPFUN_GGUF_DIR"
        )
    return paths


def build_kv_session_contract_artifact(
    *,
    gguf_dir: Path = DEFAULT_GGUF_DIR,
    backend: str = "hip_gfx1151",
    prompt: str = "hello",
    reasoning_effort: str = "low",
    max_context: int = DEFAULT_STEPFUN_SHORT_CONTEXT,
    max_new_tokens: int = 1,
    context_pages: int = 1,
    page_size: int = DEFAULT_STEPFUN_SHORT_CONTEXT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return a metadata-only KV streaming session contract artifact."""

    paths = stepfun_gguf_paths(gguf_dir)
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        paths,
        backend=backend,
        max_context=max_context,
        max_new_tokens=max_new_tokens,
    )
    run_plan = planner.plan_kv_decode_chat(
        [{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
        context_pages=context_pages,
        page_size=page_size,
    )
    session = StepFunResidentSession(
        info=planner.info,
        model_map=planner.model_map,
        tokenizer=planner.tokenizer,
        weights=object(),
        backend=backend,
    )
    contract = session.kv_streaming_decode_contract(run_plan)
    kv_cache = StepFunKVCacheAllocation(
        buffers=(),
        context_pages=context_pages,
        page_size=page_size,
        layer_nbytes=stepfun_kv_cache_layer_nbytes(
            planner.model_map.config,
            context_pages=context_pages,
            page_size=page_size,
        ),
    )
    decode_entrypoint = session.decode_one_token_kv_bf16(
        run_plan,
        kv_cache=kv_cache,
    )
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_kv_session_streaming_decode_contract",
        "date": artifact_date,
        "status": "blocked",
        "gguf_dir": str(gguf_dir),
        "gguf_paths": [str(path) for path in paths],
        "backend": backend,
        "prompt": prompt,
        "reasoning_effort": reasoning_effort,
        "max_context": max_context,
        "max_new_tokens": max_new_tokens,
        "context_pages": context_pages,
        "page_size": page_size,
        "contract": contract,
        "decode_entrypoint_blocker": decode_entrypoint,
        "no_claim_policy": {
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This artifact captures the resident-session streaming decode contract "
                "but does not materialize weights, launch kernels, or generate a token."
            ),
        },
    }


def verify_kv_session_contract_artifact(
    session_artifact: Path,
    *,
    current_artifact: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted KV session-contract artifact with current metadata."""

    persisted = json.loads(session_artifact.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_artifact)
    failures: list[dict[str, object]] = []
    if persisted != current_artifact:
        failures.append(
            {
                "name": "kv_session_contract_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted KV session-contract artifact differs from current "
                    "StepFun planner/session metadata."
                ),
            }
        )
    all_match = not failures
    persisted_contract = persisted.get("contract") if isinstance(persisted, dict) else None
    persisted_contract = persisted_contract if isinstance(persisted_contract, dict) else {}
    current_contract = current_artifact.get("contract")
    current_contract = current_contract if isinstance(current_contract, dict) else {}
    return {
        "schema_version": 1,
        "artifact_path": str(session_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status") if isinstance(persisted, dict) else None,
        "current_status": current_artifact.get("status"),
        "persisted_blocked_by": persisted_contract.get("blocked_by"),
        "current_blocked_by": current_contract.get("blocked_by"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    artifact = build_kv_session_contract_artifact(
        gguf_dir=args.gguf_dir,
        backend=args.backend,
        prompt=args.prompt,
        reasoning_effort=args.reasoning_effort,
        max_context=args.max_context,
        max_new_tokens=args.max_new_tokens,
        context_pages=args.context_pages,
        page_size=args.page_size,
        artifact_date=args.artifact_date,
    )
    if args.verify_session_contract is not None:
        verification = verify_kv_session_contract_artifact(
            args.verify_session_contract,
            current_artifact=artifact,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        elif args.verification_sha_only:
            payload = status_mod._stable_json_sha256(verification)
        else:
            payload = verification
        status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
        return (
            status_mod.READY_EXIT_CODE
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    if args.status_only:
        payload: object = artifact["status"]
    elif args.blocked_by_only:
        contract = artifact.get("contract")
        if not isinstance(contract, dict):
            raise ValueError("artifact contract is not a JSON object")
        payload = contract["blocked_by"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(artifact)
    else:
        payload = artifact
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
