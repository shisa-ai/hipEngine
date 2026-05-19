#!/usr/bin/env python3
"""MTP chain speculative benchmark entrypoint / readiness diagnostic.

Plain chain MTP should reuse the DFlash-built target verifier.  This script is
therefore intentionally thin: it validates target-attached ``mtp.*`` tensors and
records the exact DraftBatch/TargetVerifyBatch contract that the native MTP
provider will feed into ``verify_chain_bulk_and_commit``.  On the current shisa
packed PARO target the validation is expected to report missing MTP tensors and
exit successfully with a retained diagnostic JSON rather than forking a fake
proposal path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading import DFLASH_PACKED_TARGET_MODEL, load_weight_index, validate_qwen35_mtp_metadata
from hipengine.speculative import MTP_CHAIN_CANDIDATE_BUDGETS, MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain


def _demo_contract_rows(candidate_budget: int) -> dict[str, Any]:
    draft = compile_mtp_chain(
        [MtpDraftRequest(request_id=0, root_position=100, candidate_tokens=tuple(range(1001, 1001 + candidate_budget)))],
        candidate_budget=candidate_budget,
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(1000,), root_positions=(100,))
    return {
        "draft_batch": {
            "request_ids": list(draft.request_ids),
            "candidate_tokens": list(draft.candidate_tokens),
            "parent_positions": list(draft.parent_positions),
            "draft_depths": list(draft.draft_depths),
            "row_to_request": list(draft.row_to_request),
            "active_mask": list(draft.active_mask),
            "mode": draft.mode,
        },
        "target_verify_batch": {
            "tokens": list(target.tokens),
            "positions": list(target.positions),
            "parent_rows": list(target.parent_rows),
            "root_rows": list(target.root_rows),
            "candidate_rows": list(target.candidate_rows),
            "draft_depths": list(target.draft_depths),
            "active_mask": list(target.active_mask),
            "mode": target.mode,
            "tree_shape": list(target.tree_shape),
        },
    }


def _git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    dirty = bool(run(["git", "status", "--porcelain"]))
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "dirty": dirty,
    }


def build_mtp_readiness_artifact(target_model: str | Path, *, draft_budgets: tuple[int, ...]) -> dict[str, Any]:
    index = load_weight_index(target_model)
    validation = validate_qwen35_mtp_metadata(index)
    status = "ready_for_native_proposal_port" if validation.passed else "blocked_missing_mtp_tensors"
    if validation.passed:
        status = "blocked_native_mtp_proposal_kernels_not_ported"
    return {
        "schema_version": 1,
        "date": datetime.now(UTC).date().isoformat(),
        "title": "MTP chain verifier readiness diagnostic",
        "status": status,
        "performance_claim": False,
        "target_model": str(target_model),
        "mtp_validation": validation.to_json_dict(),
        "draft_budgets": list(draft_budgets),
        "shared_contract": {
            "provider": "mtp",
            "proposal_mode": "chain",
            "verify_mode": "verify_chain",
            "verifier_entrypoint": "Qwen35ParoResidentSession.verify_chain_bulk_and_commit",
            "accept_summary": "dflash_accept_chain_i32 / TargetVerifyBatch.accept_from_top1 semantics",
            "candidate_rows_exclude_root": True,
            "transactional_commit_reuse": True,
        },
        "contract_examples": {str(budget): _demo_contract_rows(int(budget)) for budget in draft_budgets},
        "decision_reason": (
            "MTP can reuse the DFlash native chain verifier/accept/commit ABI. "
            "The current target checkpoint must carry BF16 target-attached `mtp.*` tensors before a real "
            "MtpDraftProvider can produce proposals.  The shisa packed PARO snapshot currently has no `mtp.*` tensors, "
            "so this run is retained as a readiness/blocked diagnostic rather than a speed row."
        ),
        "software": {"hipengine": _git_state()},
    }


def _parse_budgets(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("at least one draft budget is required")
    invalid = [value for value in values if value not in MTP_CHAIN_CANDIDATE_BUDGETS]
    if invalid:
        raise ValueError(f"MTP draft budgets must be in {MTP_CHAIN_CANDIDATE_BUDGETS}, got {invalid}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DFLASH_PACKED_TARGET_MODEL, help="Target checkpoint path or HF-cache id")
    parser.add_argument("--draft-budgets", default="1,2,3", help="Comma-separated MTP chain budgets; default 1,2,3")
    parser.add_argument("--json", type=Path, help="Optional output JSON path")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return non-zero if mtp.* tensors are missing (default writes blocked diagnostic with exit 0)",
    )
    args = parser.parse_args()

    budgets = _parse_budgets(args.draft_budgets)
    artifact = build_mtp_readiness_artifact(args.target_model, draft_budgets=budgets)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "passed": artifact["mtp_validation"]["passed"], "missing": len(artifact["mtp_validation"]["missing"])}, sort_keys=True))
    if args.fail_on_missing and artifact["status"] == "blocked_missing_mtp_tensors":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
