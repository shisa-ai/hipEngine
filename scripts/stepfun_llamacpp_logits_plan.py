#!/usr/bin/env python3
"""Emit a StepFun llama.cpp logits-probe implementation plan artifact.

The plan converts the retained llama.cpp logits preflight into concrete follow-up
steps for capturing same-prompt llama.cpp logits and comparing expected token 369
against generated token 671. It is planning/handoff evidence only and does not
claim oracle parity, KV readiness, e2e readiness, or performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_llamacpp_logits_preflight as preflight_mod
from scripts import stepfun_oracle_next_action_manifest as next_action_mod
from scripts import stepfun_oracle_source_map as source_map_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-plan.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-artifact",
        type=Path,
        default=preflight_mod.DEFAULT_OUTPUT,
        help="Retained llama.cpp logits preflight artifact.",
    )
    parser.add_argument(
        "--source-map-artifact",
        type=Path,
        default=source_map_mod.DEFAULT_OUTPUT,
        help="Retained oracle source-map artifact.",
    )
    parser.add_argument(
        "--next-action-manifest",
        type=Path,
        default=next_action_mod.DEFAULT_OUTPUT,
        help="Retained oracle next-action manifest artifact.",
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
        "--implementation-ready-only",
        action="store_true",
        help="Emit only whether the plan is ready for implementation work.",
    )
    parser.add_argument(
        "--step-keys-only",
        action="store_true",
        help="Emit only ordered implementation step keys.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _artifact_ref(path: Path, payload: dict[str, object]) -> dict[str, object]:
    return {
        "path": str(path),
        "artifact_kind": payload.get("artifact_kind"),
        "status": payload.get("status"),
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _source_entry(source_map: dict[str, object], key: str) -> dict[str, object]:
    entries = source_map.get("entries")
    if not isinstance(entries, list):
        return {}
    for item in entries:
        if isinstance(item, dict) and item.get("key") == key:
            return item
    return {}


def _precondition_passed(preflight: dict[str, object], name: str) -> object:
    prerequisites = preflight.get("prerequisites")
    if not isinstance(prerequisites, list):
        return None
    for item in prerequisites:
        if isinstance(item, dict) and item.get("name") == name:
            return item.get("passed")
    return None


def _missing_only_expected_logits_evidence(missing: object) -> bool:
    expected = {
        "llama_cpp_same_prompt_logits_artifact_present",
        "llama_cpp_logits_dump_entrypoint_identified",
        "llama_cpp_same_prompt_special_token_support_present",
    }
    return (
        isinstance(missing, list)
        and bool(missing)
        and set(missing).issubset(expected)
        and "llama_cpp_same_prompt_logits_artifact_present" in set(missing)
    )


def build_llamacpp_logits_plan(
    *,
    preflight_artifact: Path = preflight_mod.DEFAULT_OUTPUT,
    source_map_artifact: Path = source_map_mod.DEFAULT_OUTPUT,
    next_action_manifest: Path = next_action_mod.DEFAULT_OUTPUT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the llama.cpp logits-probe implementation plan."""

    preflight = _load_json_object(preflight_artifact)
    source_map = _load_json_object(source_map_artifact)
    manifest = _load_json_object(next_action_manifest)
    target = preflight.get("target") if isinstance(preflight.get("target"), dict) else {}
    missing_evidence = preflight.get("missing_evidence")
    implementation_ready = (
        preflight.get("status") == "blocked"
        and _missing_only_expected_logits_evidence(missing_evidence)
        and _precondition_passed(preflight, "next_action_manifest_investigation_ready") is True
        and _precondition_passed(preflight, "source_map_mapped") is True
        and _precondition_passed(preflight, "host_logit_margin_passed") is True
        and _precondition_passed(preflight, "llama_cli_available") is True
    )
    expected_logits = preflight.get("expected_llama_logits_artifact")
    expected_logits_path = (
        expected_logits.get("path") if isinstance(expected_logits, dict) else None
    )
    llama_probe = preflight.get("llama_cli_probe")
    llama_cli = llama_probe.get("path") if isinstance(llama_probe, dict) else None
    missing_set = set(missing_evidence) if isinstance(missing_evidence, list) else set()
    entrypoint_ready = (
        "llama_cpp_logits_dump_entrypoint_identified" not in missing_set
        and "llama_cpp_same_prompt_special_token_support_present" not in missing_set
    )
    prompt_entry = _source_entry(source_map, "host_prompt_logit_smoke")
    oracle_entry = _source_entry(source_map, "llamacpp_oracle_runner")
    handoff_entry = _source_entry(source_map, "oracle_blocker_handoff")
    steps = [
        {
            "key": "identify_or_add_logits_dump_entrypoint",
            "status": "completed" if entrypoint_ready else "pending",
            "owner_sources": oracle_entry.get("sources"),
            "acceptance": [
                "a llama.cpp command or helper can emit same-prompt logits/top-k for the retained StepFun prompt",
                "scripts/stepfun_llamacpp_logits_preflight.py no longer reports llama_cpp_logits_dump_entrypoint_identified",
            ],
            "notes": (
                [
                    "current preflight found a llama.cpp probe binary with logits dump flags",
                    "do not use --logit-bias as a substitute for logits capture",
                ]
                if entrypoint_ready
                else [
                    (
                        "current preflight found a logits dump binary, but it cannot parse the retained "
                        "prompt's special tokens"
                        if "llama_cpp_same_prompt_special_token_support_present" in missing_set
                        else "current preflight found llama-cli executable but no obvious logits dump flag"
                    ),
                    "do not use --logit-bias as a substitute for logits capture",
                ]
            ),
        },
        {
            "key": "capture_same_prompt_llamacpp_logits",
            "status": "pending",
            "owner_sources": [oracle_entry, prompt_entry],
            "expected_output_artifact": expected_logits_path,
            "acceptance": [
                "retained llama.cpp logits artifact exists for the exact prompt/input IDs used by the host prompt-smoke artifact",
                "artifact records logits or ranks for expected token 369 and generated token 671",
                "artifact records command, backend, model path, and prompt artifact SHA",
            ],
        },
        {
            "key": "compare_host_vs_llamacpp_logits",
            "status": "pending",
            "owner_sources": [prompt_entry, handoff_entry],
            "acceptance": [
                "comparison explains whether token 671 outranks token 369 in llama.cpp logits for the same prompt",
                "comparison preserves existing host evidence that token 369 is host top-1 with margin 0.8150444030761719",
                "no oracle parity claim is made unless generated_text_matches_target passes",
            ],
        },
        {
            "key": "refresh_oracle_handoff_artifacts",
            "status": "pending",
            "owner_sources": [handoff_entry],
            "acceptance": [
                "oracle diagnosis, consistency check, next-action manifest, logits preflight, remaining-blockers rollup, status, final-blocker manifest, and handoff check are regenerated",
                "full StepFun guard passes",
                "WORKLOG and docs/STEPFUN.md record the evidence and no-claim policy",
            ],
        },
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_logits_plan",
        "date": artifact_date,
        "status": "blocked",
        "ready": False,
        "implementation_ready": implementation_ready,
        "preflight_artifact": _artifact_ref(preflight_artifact, preflight),
        "source_map_artifact": _artifact_ref(source_map_artifact, source_map),
        "next_action_manifest": _artifact_ref(next_action_manifest, manifest),
        "missing_evidence": missing_evidence,
        "target": {
            "canonical_backend": target.get("canonical_backend", "vulkan"),
            "readiness_gate": "oracle_parity",
            "unresolved_evidence_gap": "generated_text_matches_target",
            "expected_next_token_id": target.get("expected_next_token_id"),
            "expected_next_token_text": target.get("expected_next_token_text"),
            "generated_first_token_id": target.get("generated_first_token_id"),
            "generated_text": target.get("generated_text"),
            "host_top_token_ids": target.get("host_top_token_ids"),
            "host_top1_to_top2_margin": target.get("host_top1_to_top2_margin"),
        },
        "llama_cli": llama_cli,
        "expected_llamacpp_logits_artifact": expected_logits,
        "implementation_steps": steps,
        "step_keys": [step["key"] for step in steps],
        "required_preflight_to_clear_plan_blocker": [
            "scripts/stepfun_llamacpp_logits_preflight.py --status-only returns \"ready\"",
            "scripts/stepfun_llamacpp_logits_preflight.py --missing-evidence-only returns []",
            "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-probe.json exists and is retained",
        ],
        "blocked_reason": (
            "same-prompt llama.cpp logits capture is not implemented or retained yet"
        ),
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This plan only describes the llama.cpp logits evidence work needed next; "
                "generated_text_matches_target remains unresolved."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_llamacpp_logits_plan(
        preflight_artifact=args.preflight_artifact,
        source_map_artifact=args.source_map_artifact,
        next_action_manifest=args.next_action_manifest,
        artifact_date=args.artifact_date,
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.implementation_ready_only:
        payload = report["implementation_ready"]
    elif args.step_keys_only:
        payload = report["step_keys"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
