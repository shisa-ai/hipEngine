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
from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    qwen35_gguf_current_hidden_seed_contract,
    qwen35_gguf_fp32_hidden_seed_contract,
)
from hipengine.speculative import (  # noqa: E402
    GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE,
    GGUF_MTP_ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE,
    GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE,
    GGUF_MTP_ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE,
    GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE,
    GGUF_MTP_METRICS_CONTRACT_READY,
    GGUF_MTP_PARTIAL_TRACE_BUDGET_COVERAGE,
    Qwen35GGUFMTPPerformanceReadiness,
    Qwen35GGUFMTPRuntimeKernelPlan,
    Qwen35GGUFMTPVerificationMetrics,
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
DEFAULT_BACKEND = "hip_gfx1100"
DEFAULT_PROMPTS = Path("benchmarks/fixtures/llamacpp_mtp_bench_prompts.json")
DEFAULT_HIPENGINE_TOKENS = Path(
    "benchmarks/fixtures/hipengine_gguf_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)
DEFAULT_LLAMACPP_TOKENS = Path(
    "benchmarks/fixtures/llamacpp_hip_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)
DEFAULT_SAMPLING = Path("benchmarks/fixtures/gguf_mtp_b1_sampling_greedy_seed12345.json")
DEFAULT_SAMPLING_BY_DRAFT_MAX = {
    1: DEFAULT_SAMPLING,
    2: Path("benchmarks/fixtures/gguf_mtp_b2_sampling_greedy_seed12345.json"),
    3: Path("benchmarks/fixtures/gguf_mtp_b3_sampling_greedy_seed12345.json"),
    4: Path("benchmarks/fixtures/gguf_mtp_b4_sampling_greedy_seed12345.json"),
}
DEFAULT_LLAMACPP_TRACE_FIXTURE = Path("benchmarks/fixtures/llamacpp_mtp_explain_concept_draft_trace.json")
FULL_TRACE_BUDGET_COVERAGE = GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE
PARTIAL_TRACE_BUDGET_COVERAGE = GGUF_MTP_PARTIAL_TRACE_BUDGET_COVERAGE
ACCEPTED_DRAFT_COMPARABLE = GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE
ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE = GGUF_MTP_ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE
ACCEPTED_OUTPUT_COMPARABLE = GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE
ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE = GGUF_MTP_ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE
METRICS_CONTRACT_READY = GGUF_MTP_METRICS_CONTRACT_READY
CLI_GATE_EXIT_CODES = {
    "blocked": 2,
    "partial_trace_budget": 3,
    "noncomparable_accepted_output": 4,
    "performance_unready": 5,
    "noncomparable_accepted_draft": 6,
    "native_runtime_missing": 7,
    "optimization_missing": 8,
    "kvlivespans_smoke_fail": 9,
}


class B1PromptSuitePreflightError(RuntimeError):
    """Raised when the B1 harness cannot build a preflight artifact."""


def default_sampling_fixture(draft_max: int) -> Path:
    try:
        return DEFAULT_SAMPLING_BY_DRAFT_MAX[int(draft_max)]
    except KeyError as exc:
        raise B1PromptSuitePreflightError("draft_max must be in 1..4 for B1-B4 preflight") from exc


def _hidden_seed_dynamic_input(call_spec: dict[str, Any]) -> dict[str, Any] | None:
    dynamic_inputs = call_spec.get("dynamic_inputs")
    if not isinstance(dynamic_inputs, list):
        return None
    for item in dynamic_inputs:
        if isinstance(item, dict) and item.get("argument") == "hidden_seed":
            return item
    return None


def _hidden_size_from_plan(plan: dict[str, Any], call_spec: dict[str, Any]) -> int:
    hidden_size = plan.get("hidden_size")
    if isinstance(hidden_size, int):
        return int(hidden_size)
    hidden_input = _hidden_seed_dynamic_input(call_spec) or {}
    shape = hidden_input.get("shape")
    if isinstance(shape, list | tuple) and len(shape) == 2 and isinstance(shape[1], int):
        return int(shape[1])
    raise B1PromptSuitePreflightError("MTP draft plan did not expose hidden_size or hidden_seed shape")


def _build_hidden_seed_contract_precheck(
    *,
    hidden_size: int,
    call_spec: dict[str, Any],
) -> dict[str, Any]:
    hidden_input = _hidden_seed_dynamic_input(call_spec)
    required_contract = qwen35_gguf_fp32_hidden_seed_contract(
        hidden_size,
        rows=1,
        populated_by_decode=True,
    ).as_dict()
    default_ar_contract = qwen35_gguf_current_hidden_seed_contract(hidden_size, rows=1).as_dict()
    dynamic_shape = None if hidden_input is None else hidden_input.get("shape")
    checks = [
        {
            "name": "required_fp32_post_output_norm",
            "passed": bool(
                required_contract["provenance"] == "post_output_norm"
                and required_contract["dtype"] == "FP32"
                and required_contract["ready_for_mtp"]
            ),
            "detail": "MTP seed must be a populated fp32 post-output_norm row",
        },
        {
            "name": "default_ar_tap_not_used_for_mtp",
            "passed": bool(
                default_ar_contract["dtype"] == "BF16" and not default_ar_contract["ready_for_mtp"]
            ),
            "detail": "default GGUF AR generation tap is BF16 and must not be consumed as an MTP seed",
        },
        {
            "name": "hidden_seed_dynamic_input_shape",
            "passed": dynamic_shape == ["tokens", hidden_size],
            "detail": "MTP call spec must expose hidden_seed with shape [tokens, hidden_size]",
        },
    ]
    return {
        "checked": True,
        "passed": all(item["passed"] for item in checks),
        "hidden_size": hidden_size,
        "required_contract": required_contract,
        "default_ar_contract": default_ar_contract,
        "dynamic_input": hidden_input,
        "checks": checks,
    }


def _build_runtime_kernel_precheck(*, backend: str, draft_topk: dict[str, Any]) -> dict[str, Any]:
    draft_topk_kernel = draft_topk.get("kernel")
    if not isinstance(draft_topk_kernel, list | tuple) or len(draft_topk_kernel) != 4:
        raise B1PromptSuitePreflightError("draft_topk.kernel must be a four-axis registry key")
    return Qwen35GGUFMTPRuntimeKernelPlan.from_registry(
        backend=backend,
        draft_topk_kernel=tuple(str(part) for part in draft_topk_kernel),  # type: ignore[arg-type]
    ).as_dict()


def _build_hipengine_metrics_contract(*, draft_max: int) -> dict[str, Any]:
    return {
        "status": "not_run",
        "blocked_until": "native_gguf_mtp_runtime",
        "source": "Qwen35GGUFMTPVerificationMetrics",
        "result_source": "Qwen35GGUFMTPVerificationResult",
        "draft_max": int(draft_max),
        "required_fields": [
            "cycle_count",
            "draft_token_count",
            "accepted_token_count",
            "output_token_count",
            "accepted_per_draft",
            "accepted_per_output",
        ],
        "denominators": Qwen35GGUFMTPVerificationMetrics.denominator_labels(),
    }


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
    generated_draft_tokens = 0
    accepted_draft_tokens = 0
    max_generated_per_call = 0
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
        accepted = call.get("accepted") if isinstance(call, dict) else None
        if not isinstance(generated, int) or generated < 0:
            calls_valid = False
            continue
        generated_draft_tokens += generated
        max_generated_per_call = max(max_generated_per_call, generated)
        if generated > draft_max:
            calls_valid = False
            check(
                f"call_{index}_generated_budget",
                False,
                f"generated={generated} exceeds requested draft_max={draft_max}",
            )
        if not isinstance(accepted, int) or accepted < 0 or accepted > generated:
            calls_valid = False
            continue
        accepted_draft_tokens += accepted
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
    reported_draft_n = timing.get("draft_n", summary.get("draft_n"))
    reported_draft_n_accepted = timing.get("draft_n_accepted", summary.get("draft_n_accepted"))
    reported_draft_acceptance = timing.get("draft_acceptance", summary.get("draft_acceptance"))
    expected_draft_acceptance = (
        float(accepted_draft_tokens) / float(generated_draft_tokens)
        if generated_draft_tokens
        else None
    )
    check(
        "draft_n",
        reported_draft_n == generated_draft_tokens,
        "reported draft_n must equal summed generated draft tokens",
    )
    check(
        "draft_n_accepted",
        reported_draft_n_accepted == accepted_draft_tokens,
        "reported draft_n_accepted must equal summed accepted draft tokens",
    )
    check(
        "draft_acceptance",
        expected_draft_acceptance is not None
        and isinstance(reported_draft_acceptance, int | float)
        and abs(float(reported_draft_acceptance) - expected_draft_acceptance) <= 1.0e-9,
        "reported draft_acceptance must equal draft_n_accepted / draft_n",
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

    visible_output_token_count = trace.get("visible_output_token_count")
    if not isinstance(visible_output_token_count, int) or visible_output_token_count <= 0:
        visible_output_token_count = None
    budget_coverage = (
        FULL_TRACE_BUDGET_COVERAGE
        if max_generated_per_call >= draft_max
        else PARTIAL_TRACE_BUDGET_COVERAGE
    )
    denominator_metrics = {
        "accepted_draft_tokens": accepted_draft_tokens,
        "generated_draft_tokens": generated_draft_tokens,
        "accepted_per_draft": expected_draft_acceptance,
        "accepted_per_draft_status": ACCEPTED_DRAFT_COMPARABLE
        if expected_draft_acceptance is not None
        else ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE,
        "visible_output_token_count": visible_output_token_count,
        "accepted_per_output": None
        if visible_output_token_count is None
        else float(accepted_draft_tokens) / float(visible_output_token_count),
        "accepted_per_output_status": ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE
        if visible_output_token_count is None
        else ACCEPTED_OUTPUT_COMPARABLE,
        "denominators": {
            "accepted_per_draft": "accepted_draft_tokens / generated_draft_tokens",
            "accepted_per_output": "accepted_draft_tokens / visible_output_token_count",
        },
    }

    passed = all(item["passed"] for item in checks)
    return {
        "fixture": str(trace_fixture),
        "passed": passed,
        "kind": trace.get("kind"),
        "requested_draft_max": int(draft_max),
        "max_generated_per_call": max_generated_per_call,
        "budget_coverage": budget_coverage,
        "prompt_name": request.get("prompt_name"),
        "prompt_tokens": trace.get("prompt_tokens"),
        "draft_call_count": len(call_items),
        "candidate_count": candidate_count,
        "observed_top_k": summary.get("observed_top_k", max_candidates),
        "selected_token_ids": selected_token_ids,
        "denominator_metrics": denominator_metrics,
        "draft_acceptance": reported_draft_acceptance,
        "draft_n": reported_draft_n,
        "draft_n_accepted": reported_draft_n_accepted,
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
    backend: str = DEFAULT_BACKEND,
) -> dict[str, Any]:
    requested_draft_max = int(draft_max)
    if requested_draft_max < 1 or requested_draft_max > 4:
        raise B1PromptSuitePreflightError("draft_max must be in 1..4 for B1-B4 preflight")
    requested_budget = f"B{requested_draft_max}"
    target = str(backend)
    if not target:
        raise B1PromptSuitePreflightError("backend must be non-empty")

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
    first_call_spec = mtp_draft_tensor_plans[0].cpu_reference_call_spec.as_dict()
    hidden_seed_contract_precheck = _build_hidden_seed_contract_precheck(
        hidden_size=_hidden_size_from_plan(mtp_draft_tensor_plan_dicts[0], first_call_spec),
        call_spec=first_call_spec,
    )

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
    runtime_kernel_precheck = _build_runtime_kernel_precheck(
        backend=target,
        draft_topk=draft_topk_contract,
    )
    oracle_gate = run_oracle_gate(oracle_fixture)
    llamacpp_trace_oracle = _validate_llamacpp_trace_oracle(
        llamacpp_trace_fixture,
        draft_max=requested_draft_max,
    )
    hipengine_metrics_contract = _build_hipengine_metrics_contract(draft_max=requested_draft_max)
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
    if not hidden_seed_contract_precheck["passed"]:
        blockers.append(
            {
                "code": "hidden_seed_contract_mismatch",
                "detail": "GGUF MTP hidden seed must be fp32 post-output_norm and match the call-spec hidden_seed input",
                "failed_checks": [
                    item for item in hidden_seed_contract_precheck["checks"] if not item["passed"]
                ],
            }
        )
    if not runtime_kernel_precheck["exactness_oracles_ready"]:
        blockers.append(
            {
                "code": "runtime_kernel_precheck_failed",
                "detail": "required GGUF MTP CPU-reference oracle registry keys must be present before metrics are comparable",
                "missing_exactness_oracle_keys": runtime_kernel_precheck["missing_exactness_oracle_keys"],
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
        and hidden_seed_contract_precheck["passed"]
        and runtime_kernel_precheck["exactness_oracles_ready"]
        and oracle_gate["passed"]
        and llamacpp_trace_oracle["passed"]
    ):
        blockers.append(
            {
                "code": "native_gguf_mtp_runtime_missing",
                "detail": (
                    "Native GGUF MTP draft execution is not implemented yet; this harness "
                    "stops after metadata/token/sampling/runtime-kernel preflight instead of reporting metrics."
                ),
                "missing_native_runtime_keys": runtime_kernel_precheck["missing_native_runtime_keys"],
                "missing_optimization_keys": runtime_kernel_precheck["missing_optimization_keys"],
            }
        )

    return {
        "schema": 1,
        "kind": "hipengine_gguf_mtp_b1_prompt_suite",
        "mode": "preflight",
        "status": "blocked" if blockers else "ready",
        "cli_gate_exit_codes": dict(CLI_GATE_EXIT_CODES),
        "model": str(model),
        "backend": target,
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
        "hidden_seed_contract_precheck": hidden_seed_contract_precheck,
        "runtime_kernel_precheck": runtime_kernel_precheck,
        "oracle_gate": oracle_gate,
        "llamacpp_trace_oracle": llamacpp_trace_oracle,
        "hipengine_metrics_contract": hipengine_metrics_contract,
        "execution": {
            "implemented": False,
            "exactness_gate": "passed" if oracle_gate["passed"] and llamacpp_trace_oracle["passed"] else "failed",
            "accepted_output_metrics": "not_run",
            "next_action": f"implement native GGUF MTP draft execution and re-run this harness for {requested_budget}",
        },
        "blockers": blockers,
    }


def _performance_comparison_blockers(readiness: dict[str, Any]) -> list[str]:
    return list(
        Qwen35GGUFMTPPerformanceReadiness.from_gate_inputs(
            parity_precheck=bool(readiness["parity_precheck"]),
            draft_budget_precheck=bool(readiness["draft_budget_precheck"]),
            draft_sampling_contract_precheck=bool(readiness["draft_sampling_contract_precheck"]),
            hidden_seed_contract_precheck=bool(readiness["hidden_seed_contract_precheck"]),
            exactness_gate=str(readiness["exactness_gate"]),
            kvlivespans_paged_cache_smoke=bool(readiness["kvlivespans_paged_cache_smoke"]),
            llamacpp_trace_budget_coverage=str(readiness["llamacpp_trace_budget_coverage"]),
            accepted_per_draft_status=str(readiness["accepted_per_draft_status"]),
            accepted_per_output_status=str(readiness["accepted_per_output_status"]),
            native_runtime_kernels_ready=bool(readiness["native_runtime_kernels_ready"]),
            optimization_kernels_ready=bool(readiness["optimization_kernels_ready"]),
            metrics_contract_status=str(readiness["metrics_contract_status"]),
        ).blockers
    )


def _matrix_budget_readiness(artifact: dict[str, Any]) -> dict[str, Any]:
    readiness = {
        "status": artifact["status"],
        "draft_max": artifact["draft_max"],
        "parity_precheck": artifact["parity_precheck"]["all_pass"],
        "draft_budget_precheck": artifact["draft_budget_precheck"]["passed"],
        "draft_sampling_contract_precheck": artifact["draft_sampling_contract_precheck"]["passed"],
        "hidden_seed_contract_precheck": artifact["hidden_seed_contract_precheck"]["passed"],
        "exactness_gate": artifact["execution"]["exactness_gate"],
        "kvlivespans_paged_cache_smoke": artifact["oracle_gate"]["kvlivespans_paged_cache_smoke"][
            "passed"
        ],
        "kvlivespans_paged_cache_max_abs_diff": artifact["oracle_gate"][
            "kvlivespans_paged_cache_smoke"
        ]["max_abs_diff"],
        "llamacpp_trace_budget_coverage": artifact["llamacpp_trace_oracle"]["budget_coverage"],
        "accepted_per_draft_status": artifact["llamacpp_trace_oracle"]["denominator_metrics"][
            "accepted_per_draft_status"
        ],
        "accepted_per_output_status": artifact["llamacpp_trace_oracle"]["denominator_metrics"][
            "accepted_per_output_status"
        ],
        "native_runtime_kernels_ready": artifact["runtime_kernel_precheck"]["native_runtime_kernels_ready"],
        "optimization_kernels_ready": artifact["runtime_kernel_precheck"]["optimization_kernels_ready"],
        "missing_native_runtime_keys": artifact["runtime_kernel_precheck"]["missing_native_runtime_keys"],
        "missing_optimization_keys": artifact["runtime_kernel_precheck"]["missing_optimization_keys"],
        "metrics_contract_status": artifact["hipengine_metrics_contract"]["status"],
        "blocker_codes": [blocker["code"] for blocker in artifact["blockers"]],
    }
    readiness["performance_comparison_blockers"] = _performance_comparison_blockers(readiness)
    readiness["performance_comparison_ready"] = not readiness["performance_comparison_blockers"]
    return readiness


def build_b1_b4_prompt_suite_matrix(
    *,
    model: Path,
    prompts_file: Path,
    hipengine_token_inventory: Path,
    llamacpp_token_inventory: Path,
    oracle_fixture: Path = DEFAULT_ORACLE_FIXTURE,
    llamacpp_trace_fixture: Path = DEFAULT_LLAMACPP_TRACE_FIXTURE,
    prompt_limit: int | None = None,
    backend: str = DEFAULT_BACKEND,
    include_artifacts: bool = True,
) -> dict[str, Any]:
    artifacts = [
        build_b1_prompt_suite_artifact(
            model=model,
            prompts_file=prompts_file,
            hipengine_token_inventory=hipengine_token_inventory,
            llamacpp_token_inventory=llamacpp_token_inventory,
            hipengine_sampling=default_sampling_fixture(draft_max),
            llamacpp_sampling=default_sampling_fixture(draft_max),
            oracle_fixture=oracle_fixture,
            llamacpp_trace_fixture=llamacpp_trace_fixture,
            prompt_limit=prompt_limit,
            draft_max=draft_max,
            backend=backend,
        )
        for draft_max in (1, 2, 3, 4)
    ]
    readiness_by_budget = {item["budget"]: _matrix_budget_readiness(item) for item in artifacts}
    trace_budget_coverage_by_budget = {
        budget: readiness["llamacpp_trace_budget_coverage"]
        for budget, readiness in readiness_by_budget.items()
    }
    partial_trace_budget_budgets = [
        budget
        for budget, coverage in trace_budget_coverage_by_budget.items()
        if coverage != FULL_TRACE_BUDGET_COVERAGE
    ]
    accepted_per_draft_status_by_budget = {
        budget: readiness["accepted_per_draft_status"]
        for budget, readiness in readiness_by_budget.items()
    }
    noncomparable_accepted_per_draft_budgets = [
        budget
        for budget, status in accepted_per_draft_status_by_budget.items()
        if status != ACCEPTED_DRAFT_COMPARABLE
    ]
    accepted_per_output_status_by_budget = {
        budget: readiness["accepted_per_output_status"]
        for budget, readiness in readiness_by_budget.items()
    }
    noncomparable_accepted_per_output_budgets = [
        budget
        for budget, status in accepted_per_output_status_by_budget.items()
        if status != ACCEPTED_OUTPUT_COMPARABLE
    ]
    performance_comparison_ready_by_budget = {
        budget: readiness["performance_comparison_ready"]
        for budget, readiness in readiness_by_budget.items()
    }
    performance_comparison_blockers_by_budget = {
        budget: readiness["performance_comparison_blockers"]
        for budget, readiness in readiness_by_budget.items()
    }
    performance_unready_budgets = [
        budget
        for budget, ready in performance_comparison_ready_by_budget.items()
        if not ready
    ]
    kvlivespans_paged_cache_smoke_by_budget = {
        budget: readiness["kvlivespans_paged_cache_smoke"]
        for budget, readiness in readiness_by_budget.items()
    }
    kvlivespans_paged_cache_max_abs_diff_by_budget = {
        budget: readiness["kvlivespans_paged_cache_max_abs_diff"]
        for budget, readiness in readiness_by_budget.items()
    }
    matrix = {
        "schema": 1,
        "kind": "hipengine_gguf_mtp_b1_b4_prompt_suite_matrix",
        "mode": "preflight",
        "status": "blocked" if any(item["status"] == "blocked" for item in artifacts) else "ready",
        "cli_gate_exit_codes": dict(CLI_GATE_EXIT_CODES),
        "model": str(model),
        "backend": str(backend),
        "budgets": [item["budget"] for item in artifacts],
        "draft_max_values": [item["draft_max"] for item in artifacts],
        "artifact_count": len(artifacts),
        "artifacts_included": bool(include_artifacts),
        "all_parity_prechecks_pass": all(item["parity_precheck"]["all_pass"] for item in artifacts),
        "all_budget_prechecks_pass": all(item["draft_budget_precheck"]["passed"] for item in artifacts),
        "all_sampling_contract_prechecks_pass": all(
            item["draft_sampling_contract_precheck"]["passed"] for item in artifacts
        ),
        "all_hidden_seed_contract_prechecks_pass": all(
            item["hidden_seed_contract_precheck"]["passed"] for item in artifacts
        ),
        "all_exactness_gates_pass": all(item["execution"]["exactness_gate"] == "passed" for item in artifacts),
        "all_kvlivespans_paged_cache_smokes_pass": all(
            kvlivespans_paged_cache_smoke_by_budget.values()
        ),
        "kvlivespans_paged_cache_smoke_by_budget": kvlivespans_paged_cache_smoke_by_budget,
        "kvlivespans_paged_cache_max_abs_diff_by_budget": kvlivespans_paged_cache_max_abs_diff_by_budget,
        "all_llamacpp_trace_budgets_full": not partial_trace_budget_budgets,
        "llamacpp_trace_budget_coverage_by_budget": trace_budget_coverage_by_budget,
        "partial_llamacpp_trace_budget_budgets": partial_trace_budget_budgets,
        "all_accepted_per_draft_metrics_comparable": not noncomparable_accepted_per_draft_budgets,
        "accepted_per_draft_status_by_budget": accepted_per_draft_status_by_budget,
        "noncomparable_accepted_per_draft_budgets": noncomparable_accepted_per_draft_budgets,
        "all_accepted_per_output_metrics_comparable": not noncomparable_accepted_per_output_budgets,
        "accepted_per_output_status_by_budget": accepted_per_output_status_by_budget,
        "noncomparable_accepted_per_output_budgets": noncomparable_accepted_per_output_budgets,
        "all_native_runtime_kernels_ready": all(
            item["runtime_kernel_precheck"]["native_runtime_kernels_ready"] for item in artifacts
        ),
        "all_optimization_kernels_ready": all(
            item["runtime_kernel_precheck"]["optimization_kernels_ready"] for item in artifacts
        ),
        "all_performance_comparisons_ready": not performance_unready_budgets,
        "performance_comparison_ready_by_budget": performance_comparison_ready_by_budget,
        "performance_comparison_blockers_by_budget": performance_comparison_blockers_by_budget,
        "performance_unready_budgets": performance_unready_budgets,
        "readiness_by_budget": readiness_by_budget,
        "blocker_codes_by_budget": {
            budget: readiness["blocker_codes"] for budget, readiness in readiness_by_budget.items()
        },
    }
    if include_artifacts:
        matrix["artifacts"] = artifacts
    return matrix


def _has_partial_llamacpp_trace_budget_coverage(artifact: dict[str, Any]) -> bool:
    partial_budgets = artifact.get("partial_llamacpp_trace_budget_budgets")
    if isinstance(partial_budgets, list):
        return bool(partial_budgets)
    trace_oracle = artifact.get("llamacpp_trace_oracle")
    if not isinstance(trace_oracle, dict):
        return False
    coverage = trace_oracle.get("budget_coverage")
    return isinstance(coverage, str) and coverage != FULL_TRACE_BUDGET_COVERAGE


def _has_failed_kvlivespans_paged_cache_smoke(artifact: dict[str, Any]) -> bool:
    all_pass = artifact.get("all_kvlivespans_paged_cache_smokes_pass")
    if isinstance(all_pass, bool):
        return not all_pass
    oracle_gate = artifact.get("oracle_gate")
    if not isinstance(oracle_gate, dict):
        return False
    smoke = oracle_gate.get("kvlivespans_paged_cache_smoke")
    if not isinstance(smoke, dict):
        return False
    passed = smoke.get("passed")
    return isinstance(passed, bool) and not passed


def _has_noncomparable_accepted_draft_metrics(artifact: dict[str, Any]) -> bool:
    noncomparable_budgets = artifact.get("noncomparable_accepted_per_draft_budgets")
    if isinstance(noncomparable_budgets, list):
        return bool(noncomparable_budgets)
    trace_oracle = artifact.get("llamacpp_trace_oracle")
    if not isinstance(trace_oracle, dict):
        return False
    denominator_metrics = trace_oracle.get("denominator_metrics")
    if not isinstance(denominator_metrics, dict):
        return False
    status = denominator_metrics.get("accepted_per_draft_status")
    return isinstance(status, str) and status != ACCEPTED_DRAFT_COMPARABLE


def _has_noncomparable_accepted_output_metrics(artifact: dict[str, Any]) -> bool:
    noncomparable_budgets = artifact.get("noncomparable_accepted_per_output_budgets")
    if isinstance(noncomparable_budgets, list):
        return bool(noncomparable_budgets)
    trace_oracle = artifact.get("llamacpp_trace_oracle")
    if not isinstance(trace_oracle, dict):
        return False
    denominator_metrics = trace_oracle.get("denominator_metrics")
    if not isinstance(denominator_metrics, dict):
        return False
    status = denominator_metrics.get("accepted_per_output_status")
    return isinstance(status, str) and status != ACCEPTED_OUTPUT_COMPARABLE


def _has_missing_native_runtime_kernels(artifact: dict[str, Any]) -> bool:
    all_ready = artifact.get("all_native_runtime_kernels_ready")
    if isinstance(all_ready, bool):
        return not all_ready
    runtime_precheck = artifact.get("runtime_kernel_precheck")
    if not isinstance(runtime_precheck, dict):
        return False
    ready = runtime_precheck.get("native_runtime_kernels_ready")
    return isinstance(ready, bool) and not ready


def _has_missing_optimization_kernels(artifact: dict[str, Any]) -> bool:
    all_ready = artifact.get("all_optimization_kernels_ready")
    if isinstance(all_ready, bool):
        return not all_ready
    runtime_precheck = artifact.get("runtime_kernel_precheck")
    if not isinstance(runtime_precheck, dict):
        return False
    ready = runtime_precheck.get("optimization_kernels_ready")
    return isinstance(ready, bool) and not ready


def _has_unready_performance_comparisons(artifact: dict[str, Any]) -> bool:
    unready_budgets = artifact.get("performance_unready_budgets")
    if isinstance(unready_budgets, list):
        return bool(unready_budgets)
    if "llamacpp_trace_oracle" not in artifact:
        return False
    return bool(_matrix_budget_readiness(artifact)["performance_comparison_blockers"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--hipengine-token-inventory", type=Path, default=DEFAULT_HIPENGINE_TOKENS)
    parser.add_argument("--llamacpp-token-inventory", type=Path, default=DEFAULT_LLAMACPP_TOKENS)
    parser.add_argument(
        "--hipengine-sampling",
        type=Path,
        help="hipEngine sampling fixture (default: matching gguf_mtp_bN fixture for --draft-max)",
    )
    parser.add_argument(
        "--llamacpp-sampling",
        type=Path,
        help="llama.cpp sampling fixture (default: matching gguf_mtp_bN fixture for --draft-max)",
    )
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
    parser.add_argument(
        "--all-budgets",
        action="store_true",
        help="emit a B1-B4 preflight matrix using budget-matched default sampling fixtures",
    )
    parser.add_argument(
        "--compact-matrix",
        action="store_true",
        help="with --all-budgets, omit full child artifacts and keep only readiness summaries",
    )
    parser.add_argument("--out", type=Path, help="write JSON artifact to this path")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="return exit code 2 when the artifact status is blocked",
    )
    parser.add_argument(
        "--fail-on-partial-trace-budget",
        action="store_true",
        help="return exit code 3 when the llama.cpp trace did not exercise the requested draft budget",
    )
    parser.add_argument(
        "--fail-on-kvlivespans-smoke-fail",
        action="store_true",
        help="return exit code 9 when the KVLiveSpans paged-cache smoke fails",
    )
    parser.add_argument(
        "--fail-on-noncomparable-accepted-output",
        action="store_true",
        help="return exit code 4 when accepted_per_output denominators are not comparable",
    )
    parser.add_argument(
        "--fail-on-noncomparable-accepted-draft",
        action="store_true",
        help="return exit code 6 when accepted_per_draft denominators are not comparable",
    )
    parser.add_argument(
        "--fail-on-performance-unready",
        action="store_true",
        help="return exit code 5 when M6 performance comparison readiness is incomplete",
    )
    parser.add_argument(
        "--fail-on-native-runtime-missing",
        action="store_true",
        help="return exit code 7 when native GGUF MTP runtime kernel keys are missing",
    )
    parser.add_argument(
        "--fail-on-optimization-missing",
        action="store_true",
        help="return exit code 8 when GGUF MTP optimization kernel keys are missing",
    )
    args = parser.parse_args(argv)

    if args.compact_matrix and not args.all_budgets:
        parser.error("--compact-matrix requires --all-budgets")
    if args.all_budgets:
        if args.hipengine_sampling or args.llamacpp_sampling:
            parser.error("--all-budgets uses budget-matched default sampling fixtures; omit sampling overrides")
        artifact = build_b1_b4_prompt_suite_matrix(
            model=args.model,
            prompts_file=args.prompts_file,
            hipengine_token_inventory=args.hipengine_token_inventory,
            llamacpp_token_inventory=args.llamacpp_token_inventory,
            oracle_fixture=args.oracle_fixture,
            llamacpp_trace_fixture=args.llamacpp_trace_fixture,
            prompt_limit=args.prompt_limit,
            backend=args.backend,
            include_artifacts=not args.compact_matrix,
        )
    else:
        default_sampling = default_sampling_fixture(args.draft_max)
        artifact = build_b1_prompt_suite_artifact(
            model=args.model,
            prompts_file=args.prompts_file,
            hipengine_token_inventory=args.hipengine_token_inventory,
            llamacpp_token_inventory=args.llamacpp_token_inventory,
            hipengine_sampling=args.hipengine_sampling or default_sampling,
            llamacpp_sampling=args.llamacpp_sampling or default_sampling,
            oracle_fixture=args.oracle_fixture,
            llamacpp_trace_fixture=args.llamacpp_trace_fixture,
            prompt_limit=args.prompt_limit,
            draft_max=args.draft_max,
            backend=args.backend,
        )
    payload = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.fail_on_partial_trace_budget and _has_partial_llamacpp_trace_budget_coverage(artifact):
        return 3
    if args.fail_on_kvlivespans_smoke_fail and _has_failed_kvlivespans_paged_cache_smoke(
        artifact
    ):
        return 9
    if args.fail_on_noncomparable_accepted_output and _has_noncomparable_accepted_output_metrics(
        artifact
    ):
        return 4
    if args.fail_on_noncomparable_accepted_draft and _has_noncomparable_accepted_draft_metrics(
        artifact
    ):
        return 6
    if args.fail_on_native_runtime_missing and _has_missing_native_runtime_kernels(artifact):
        return 7
    if args.fail_on_optimization_missing and _has_missing_optimization_kernels(artifact):
        return 8
    if args.fail_on_performance_unready and _has_unready_performance_comparisons(artifact):
        return 5
    if args.fail_on_blocked and artifact["status"] == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
