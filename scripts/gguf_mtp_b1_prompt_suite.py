#!/usr/bin/env python3
"""hipEngine GGUF MTP B1 prompt-suite child harness.

The native GGUF MTP execution path is still under construction.  This harness is
therefore a correctness-first preflight child: it verifies the model exposes a
validated MTP block, verifies prompt-token and sampling parity inputs, and emits a
compact blocked artifact instead of silently comparing accepted/output metrics
before the runtime exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.loading.qwen35_gguf import (  # noqa: E402
    build_qwen35_gguf_mtp_draft_tensor_plans,
    validate_qwen35_gguf_mtp_blocks,
)
from scripts.gguf_mtp_oracle_gate import (  # noqa: E402
    DEFAULT_FIXTURE as DEFAULT_ORACLE_FIXTURE,
    run_oracle_gate,
)
from scripts.gguf_mtp_parity_precheck import (  # noqa: E402
    build_parity_precheck,
    load_json,
    load_sampling_settings,
)
from scripts.gguf_prompt_token_inventory import load_prompt_suite  # noqa: E402


DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPTS = Path("benchmarks/fixtures/llamacpp_mtp_bench_prompts.json")
DEFAULT_HIPENGINE_TOKENS = Path(
    "benchmarks/fixtures/hipengine_gguf_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)
DEFAULT_LLAMACPP_TOKENS = Path(
    "benchmarks/fixtures/llamacpp_hip_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)
DEFAULT_SAMPLING = Path("benchmarks/fixtures/gguf_mtp_b1_sampling_greedy_seed12345.json")
DEFAULT_LLAMACPP_TRACE_FIXTURE = Path("benchmarks/fixtures/llamacpp_mtp_explain_concept_draft_trace.json")


class B1PromptSuitePreflightError(RuntimeError):
    """Raised when the B1 harness cannot build a preflight artifact."""


def _sampling_draft_budget(settings: dict[str, Any]) -> dict[str, Any]:
    draft = settings.get("draft") if isinstance(settings.get("draft"), dict) else {}
    return {
        "budget": draft.get("budget"),
        "draft_max": draft.get("draft_max"),
    }


def _sampling_draft_contract(settings: dict[str, Any]) -> dict[str, Any]:
    draft = settings.get("draft") if isinstance(settings.get("draft"), dict) else {}
    return {
        "top_k": draft.get("top_k"),
        "selection": draft.get("selection"),
        "selected_index": draft.get("selected_index"),
    }


def _build_draft_sampling_contract_precheck(
    *,
    hipengine_sampling: dict[str, Any],
    llamacpp_sampling: dict[str, Any],
    draft_topk: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "top_k": draft_topk.get("top_k"),
        "selection": draft_topk.get("selection"),
        "selected_index": draft_topk.get("selected_index"),
    }
    observed = {
        "hipengine": _sampling_draft_contract(hipengine_sampling),
        "llamacpp": _sampling_draft_contract(llamacpp_sampling),
    }
    mismatches: list[dict[str, Any]] = []
    for engine, values in observed.items():
        for field, expected_value in expected.items():
            if values.get(field) != expected_value:
                mismatches.append(
                    {
                        "engine": engine,
                        "field": field,
                        "expected": expected_value,
                        "actual": values.get(field),
                    }
                )
    return {
        "checked": True,
        "passed": not mismatches,
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
    }


def _build_draft_budget_precheck(
    *,
    hipengine_sampling: dict[str, Any],
    llamacpp_sampling: dict[str, Any],
    draft_max: int,
) -> dict[str, Any]:
    expected = {"budget": f"B{draft_max}", "draft_max": int(draft_max)}
    observed = {
        "hipengine": _sampling_draft_budget(hipengine_sampling),
        "llamacpp": _sampling_draft_budget(llamacpp_sampling),
    }
    mismatches: list[dict[str, Any]] = []
    for engine, values in observed.items():
        for field, expected_value in expected.items():
            if values.get(field) != expected_value:
                mismatches.append(
                    {
                        "engine": engine,
                        "field": field,
                        "expected": expected_value,
                        "actual": values.get(field),
                    }
                )
    return {
        "checked": True,
        "passed": not mismatches,
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
    }


def _validate_llamacpp_trace_oracle(trace_fixture: Path, *, draft_max: int) -> dict[str, Any]:
    trace = load_json(trace_fixture)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    calls = trace.get("calls")
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    timing = trace.get("llamacpp_timing_summary") if isinstance(trace.get("llamacpp_timing_summary"), dict) else {}
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    server_command = metadata.get("server_command") if isinstance(metadata.get("server_command"), list) else []
    request = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}

    check(
        "kind",
        trace.get("kind") == "llamacpp_mtp_draft_candidate_trace",
        "fixture kind must be llamacpp_mtp_draft_candidate_trace",
    )
    check("calls_present", isinstance(calls, list) and bool(calls), "fixture must contain at least one draft call")
    call_items = calls if isinstance(calls, list) else []
    candidate_count = 0
    selected_token_ids: list[int] = []
    max_candidates = 0
    calls_valid = True
    for index, call in enumerate(call_items):
        candidates = call.get("candidates") if isinstance(call, dict) else None
        if not isinstance(candidates, list) or not candidates:
            calls_valid = False
            continue
        max_candidates = max(max_candidates, len(candidates))
        candidate_count += len(candidates)
        selected = candidates[0]
        if not isinstance(selected, dict):
            calls_valid = False
            continue
        token_id = selected.get("token_id")
        if selected.get("rank") != 0 or not isinstance(token_id, int):
            calls_valid = False
            continue
        selected_token_ids.append(token_id)
        generated = call.get("generated") if isinstance(call, dict) else None
        if isinstance(generated, int) and generated > draft_max:
            calls_valid = False
            check(
                f"call_{index}_generated_budget",
                False,
                f"generated={generated} exceeds requested draft_max={draft_max}",
            )
    check("candidate_rows", calls_valid, "each draft call must expose a rank-0 token and stay within draft_max")
    check(
        "candidate_count",
        summary.get("candidate_count") == candidate_count,
        "summary candidate_count must equal summed candidate rows",
    )
    check(
        "draft_call_count",
        summary.get("draft_call_count") == len(call_items),
        "summary draft_call_count must equal calls length",
    )
    check(
        "observed_top_k",
        isinstance(summary.get("observed_top_k"), int) and summary.get("observed_top_k") >= 1,
        "summary observed_top_k must be a positive integer",
    )
    check(
        "debug_trace_not_benchmark",
        "--no-spec-draft-backend-sampling" in [str(item) for item in server_command],
        "trace fixture must be marked as debug/provenance, not a backend-sampling benchmark",
    )

    passed = all(item["passed"] for item in checks)
    return {
        "fixture": str(trace_fixture),
        "passed": passed,
        "kind": trace.get("kind"),
        "prompt_name": request.get("prompt_name"),
        "prompt_tokens": trace.get("prompt_tokens"),
        "draft_call_count": len(call_items),
        "candidate_count": candidate_count,
        "observed_top_k": summary.get("observed_top_k", max_candidates),
        "selected_token_ids": selected_token_ids,
        "draft_acceptance": timing.get("draft_acceptance", summary.get("draft_acceptance")),
        "draft_n": timing.get("draft_n", summary.get("draft_n")),
        "draft_n_accepted": timing.get("draft_n_accepted", summary.get("draft_n_accepted")),
        "checks": checks,
    }


def build_b1_prompt_suite_artifact(
    *,
    model: Path,
    prompts_file: Path,
    hipengine_token_inventory: Path,
    llamacpp_token_inventory: Path,
    hipengine_sampling: Path,
    llamacpp_sampling: Path,
    oracle_fixture: Path = DEFAULT_ORACLE_FIXTURE,
    llamacpp_trace_fixture: Path = DEFAULT_LLAMACPP_TRACE_FIXTURE,
    prompt_limit: int | None = None,
    draft_max: int = 1,
) -> dict[str, Any]:
    requested_draft_max = int(draft_max)
    if requested_draft_max < 1 or requested_draft_max > 4:
        raise B1PromptSuitePreflightError("draft_max must be in 1..4 for B1-B4 preflight")
    requested_budget = f"B{requested_draft_max}"

    prompts_payload = load_prompt_suite(prompts_file)
    prompts = list(prompts_payload.get("prompts", []))
    if prompt_limit is not None:
        prompts = prompts[: int(prompt_limit)]
    if not prompts:
        raise B1PromptSuitePreflightError("prompt suite is empty after filtering")

    model_info = scan_gguf(model)
    mtp_blocks = validate_qwen35_gguf_mtp_blocks(model_info)
    if not mtp_blocks:
        raise B1PromptSuitePreflightError(f"{model}: no validated MTP blocks found")
    mtp_draft_tensor_plans = build_qwen35_gguf_mtp_draft_tensor_plans(model_info, strict=True)
    if not mtp_draft_tensor_plans:
        raise B1PromptSuitePreflightError(f"{model}: no MTP draft tensor plans found")
    mtp_draft_tensor_plan_dicts = [plan.as_dict() for plan in mtp_draft_tensor_plans]
    draft_topk_contract = mtp_draft_tensor_plan_dicts[0].get("draft_topk")
    if not isinstance(draft_topk_contract, dict):
        raise B1PromptSuitePreflightError(f"{model}: MTP draft plan did not expose draft_topk")

    hipengine_sampling_settings = load_sampling_settings(hipengine_sampling)
    llamacpp_sampling_settings = load_sampling_settings(llamacpp_sampling)
    parity = build_parity_precheck(
        hipengine_token_inventory=load_json(hipengine_token_inventory),
        llamacpp_token_inventory=load_json(llamacpp_token_inventory),
        hipengine_sampling=hipengine_sampling_settings,
        llamacpp_sampling=llamacpp_sampling_settings,
        require_sampling=True,
    )
    draft_budget_precheck = _build_draft_budget_precheck(
        hipengine_sampling=hipengine_sampling_settings,
        llamacpp_sampling=llamacpp_sampling_settings,
        draft_max=requested_draft_max,
    )
    draft_sampling_contract_precheck = _build_draft_sampling_contract_precheck(
        hipengine_sampling=hipengine_sampling_settings,
        llamacpp_sampling=llamacpp_sampling_settings,
        draft_topk=draft_topk_contract,
    )
    oracle_gate = run_oracle_gate(oracle_fixture)
    llamacpp_trace_oracle = _validate_llamacpp_trace_oracle(
        llamacpp_trace_fixture,
        draft_max=requested_draft_max,
    )
    blockers: list[dict[str, Any]] = []
    if not parity["all_pass"]:
        blockers.append(
            {
                "code": "parity_precheck_failed",
                "detail": "token-id and sampling parity must pass before B1 accepted/output metrics are comparable",
                "token_match": bool(parity["token_ids"]["all_match"]),
                "sampling_match": bool(parity["sampling"]["passed"]),
            }
        )
    if not draft_budget_precheck["passed"]:
        blockers.append(
            {
                "code": "draft_budget_mismatch",
                "detail": "requested GGUF MTP draft budget must match sampling settings before metrics are comparable",
                "expected": draft_budget_precheck["expected"],
                "mismatches": draft_budget_precheck["mismatches"],
            }
        )
    if not draft_sampling_contract_precheck["passed"]:
        blockers.append(
            {
                "code": "draft_sampling_contract_mismatch",
                "detail": "sampling fixtures must match the GGUF MTP draft top-k contract before metrics are comparable",
                "expected": draft_sampling_contract_precheck["expected"],
                "mismatches": draft_sampling_contract_precheck["mismatches"],
            }
        )
    if not oracle_gate["passed"]:
        blockers.append(
            {
                "code": "oracle_gate_failed",
                "detail": "CPU-reference GGUF MTP oracle KL/top-1 gate must pass before B1 metrics are comparable",
                "max_kl": float(oracle_gate["metrics"]["max_kl"]),
                "top1_agreement": float(oracle_gate["metrics"]["top1_agreement"]),
            }
        )
    if not llamacpp_trace_oracle["passed"]:
        blockers.append(
            {
                "code": "llamacpp_trace_oracle_failed",
                "detail": "captured llama.cpp GGUF MTP draft trace must validate before accepted/output metrics are comparable",
                "failed_checks": [
                    item for item in llamacpp_trace_oracle["checks"] if not item["passed"]
                ],
            }
        )
    if (
        parity["all_pass"]
        and draft_budget_precheck["passed"]
        and draft_sampling_contract_precheck["passed"]
        and oracle_gate["passed"]
        and llamacpp_trace_oracle["passed"]
    ):
        blockers.append(
            {
                "code": "native_gguf_mtp_runtime_missing",
                "detail": (
                    "Native GGUF MTP draft execution is not implemented yet; this harness "
                    "stops after metadata/token/sampling preflight instead of reporting metrics."
                ),
            }
        )

    return {
        "schema": 1,
        "kind": "hipengine_gguf_mtp_b1_prompt_suite",
        "mode": "preflight",
        "status": "blocked" if blockers else "ready",
        "model": str(model),
        "model_metadata": {
            "architecture": model_info.architecture,
            "file_type_name": model_info.file_type_name,
            "tensor_count": model_info.tensor_count,
        },
        "prompts_file": str(prompts_file),
        "prompt_count": len(prompts),
        "prompt_names": [str(prompt.get("name", index)) for index, prompt in enumerate(prompts)],
        "budget": requested_budget,
        "draft_max": requested_draft_max,
        "validated_mtp_blocks": [
            {
                "layer_id": int(block.layer_id),
                "tensor_count": len(block.tensor_names),
                "nextn_tensor_count": len(block.nextn_tensor_names),
                "optional_fallback_tensor_names": dict(block.optional_fallback_tensor_names),
            }
            for block in mtp_blocks
        ],
        "mtp_draft_tensor_plans": mtp_draft_tensor_plan_dicts,
        "mtp_draft_call_specs": [
            plan.cpu_reference_call_spec.as_dict() for plan in mtp_draft_tensor_plans
        ],
        "parity_precheck": parity,
        "draft_budget_precheck": draft_budget_precheck,
        "draft_sampling_contract_precheck": draft_sampling_contract_precheck,
        "oracle_gate": oracle_gate,
        "llamacpp_trace_oracle": llamacpp_trace_oracle,
        "execution": {
            "implemented": False,
            "exactness_gate": "passed" if oracle_gate["passed"] and llamacpp_trace_oracle["passed"] else "failed",
            "accepted_output_metrics": "not_run",
            "next_action": f"implement native GGUF MTP draft execution and re-run this harness for {requested_budget}",
        },
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--hipengine-token-inventory", type=Path, default=DEFAULT_HIPENGINE_TOKENS)
    parser.add_argument("--llamacpp-token-inventory", type=Path, default=DEFAULT_LLAMACPP_TOKENS)
    parser.add_argument("--hipengine-sampling", type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument("--llamacpp-sampling", type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument("--oracle-fixture", type=Path, default=DEFAULT_ORACLE_FIXTURE)
    parser.add_argument("--llamacpp-trace-fixture", type=Path, default=DEFAULT_LLAMACPP_TRACE_FIXTURE)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument(
        "--draft-max",
        type=int,
        default=1,
        choices=range(1, 5),
        metavar="{1,2,3,4}",
        help="requested GGUF MTP draft budget cap to preflight (default: 1)",
    )
    parser.add_argument("--out", type=Path, help="write JSON artifact to this path")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="return exit code 2 when the artifact status is blocked",
    )
    args = parser.parse_args(argv)

    artifact = build_b1_prompt_suite_artifact(
        model=args.model,
        prompts_file=args.prompts_file,
        hipengine_token_inventory=args.hipengine_token_inventory,
        llamacpp_token_inventory=args.llamacpp_token_inventory,
        hipengine_sampling=args.hipengine_sampling,
        llamacpp_sampling=args.llamacpp_sampling,
        oracle_fixture=args.oracle_fixture,
        llamacpp_trace_fixture=args.llamacpp_trace_fixture,
        prompt_limit=args.prompt_limit,
        draft_max=args.draft_max,
    )
    payload = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.fail_on_blocked and artifact["status"] == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
