#!/usr/bin/env python3
"""Select the layer-0 dtype oracle policy after the attn_norm formula audit."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_AUDIT = Path(
    "benchmarks/results/mtp-gguf-iter321-layer0-attn-norm-formula-audit.json"
)
DEFAULT_ATTN_COMPARE = Path(
    "benchmarks/results/mtp-gguf-iter320-layer0-attn-norm-compare.json"
)
DEFAULT_RUNNER = Path("hipengine/runtime/qwen35_gguf_runner.py")
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter322-layer0-dtype-oracle-policy.json"
)

BF16_CONTRACTION_CONCLUSION = (
    "attn_norm_mismatch_explained_by_input_activation_bf16_contraction"
)
EXPECTED_LLAMA_CANDIDATE = "input_f32_weight_f32_eps_model_f32_out"
EXPECTED_LLAMA_BF16_CANDIDATE = "input_f32_weight_f32_eps_model_bf16_out"
EXPECTED_HIP_CANDIDATE = "input_bf16_weight_f32_eps_model_bf16_out"

BOUNDARY_FIELD_ORDER = [
    "attn_norm_f32",
    "linear_qkv_f32",
    "linear_z_f32",
    "ssm_alpha_f32",
    "ssm_beta_f32",
    "conv_out_f32",
    "recurrent_out_f32",
    "recurrent_bf16_f32",
    "attn_out_f32",
]

FIELD_PROVENANCE = {
    "attn_norm_f32": {
        "source_buffer": "scratch.norm.ptr",
        "host_copy": "BF16_to_F32",
        "semantic_stage": "RMSNorm(hidden, attn_norm.weight)",
    },
    "linear_qkv_f32": {
        "source_buffer": "scratch.linear_qkv.ptr",
        "host_copy": "BF16_to_F32",
        "semantic_stage": "attn_qkv projection output",
    },
    "linear_z_f32": {
        "source_buffer": "scratch.linear_z.ptr",
        "host_copy": "BF16_to_F32",
        "semantic_stage": "attn_gate projection output",
    },
    "ssm_alpha_f32": {
        "source_buffer": "scratch.linear_alpha[_beta].ptr",
        "host_copy": "BF16_to_F32",
        "semantic_stage": "SSM alpha projection output",
    },
    "ssm_beta_f32": {
        "source_buffer": "scratch.linear_beta or alpha_beta offset",
        "host_copy": "BF16_to_F32",
        "semantic_stage": "SSM beta projection output",
    },
    "conv_out_f32": {
        "source_buffer": "scratch.conv_out.ptr",
        "host_copy": "F32",
        "semantic_stage": "linear-attention convolution output",
    },
    "recurrent_out_f32": {
        "source_buffer": "scratch.recurrent_out.ptr",
        "host_copy": "F32",
        "semantic_stage": "GDN recurrent output before BF16 cast",
    },
    "recurrent_bf16_f32": {
        "source_buffer": "scratch.recurrent_bf16.ptr",
        "host_copy": "BF16_to_F32",
        "semantic_stage": "GDN recurrent output after BF16 cast",
    },
    "attn_out_f32": {
        "source_buffer": "scratch.attn_out.ptr",
        "host_copy": "BF16_to_F32",
        "semantic_stage": "ssm_out projection output",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--attn-compare", type=Path, default=DEFAULT_ATTN_COMPARE)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=322)
    args = parser.parse_args()

    artifact = build_layer0_dtype_oracle_policy(
        audit_path=args.audit,
        attn_compare_path=args.attn_compare,
        runner_path=args.runner,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "decision": artifact["decision"]["selected_policy"],
                "scope": artifact["decision"]["scope"],
                "first_next_probe": artifact["next_probe_plan"]["ordered_probes"][0][
                    "field"
                ],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_layer0_dtype_oracle_policy(
    *,
    audit_path: Path,
    attn_compare_path: Path,
    runner_path: Path,
    iteration: int = 322,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text())
    attn_compare = json.loads(attn_compare_path.read_text())
    runner_text = runner_path.read_text()
    readiness = audit_readiness(audit)
    capture_fields = extract_boundary_capture_fields(runner_text, audit)
    status = "ready" if readiness["ready"] and capture_fields["ready"] else "blocked"
    decision = build_decision(readiness, audit, attn_compare)
    return {
        "schema": 1,
        "kind": "layer0_dtype_oracle_policy",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "audit_path": str(audit_path),
        "attn_compare_path": str(attn_compare_path),
        "runner_path": str(runner_path),
        "model": audit.get("model"),
        "layer_id": audit.get("layer_id"),
        "position": audit.get("position"),
        "token_id": audit.get("token_id"),
        "readiness": readiness,
        "decision": decision,
        "boundary_capture_fields": capture_fields,
        "next_probe_plan": build_next_probe_plan(capture_fields),
        "constraints": build_constraints(audit),
        "external_checkout_modified": False,
        "next_action": next_action(status, decision),
    }


def audit_readiness(audit: Mapping[str, Any]) -> dict[str, Any]:
    best = audit.get("best_candidates") or {}
    llama = best.get("vs_llamacpp_attn_norm") or {}
    llama_bf16 = best.get("vs_llamacpp_attn_norm_bf16") or {}
    hip = best.get("vs_hipengine_attn_norm") or {}
    facts = {
        "status_ready": audit.get("status") == "ready",
        "bf16_contraction_conclusion": audit.get("conclusion")
        == BF16_CONTRACTION_CONCLUSION,
        "llama_f32_candidate_exact": is_exact_candidate(llama, EXPECTED_LLAMA_CANDIDATE),
        "llama_bf16_candidate_exact": is_exact_candidate(
            llama_bf16,
            EXPECTED_LLAMA_BF16_CANDIDATE,
        ),
        "hip_bf16_candidate_exact": is_exact_candidate(hip, EXPECTED_HIP_CANDIDATE),
        "weight_direct_f32": audit.get("weight", {}).get("materialization_layout")
        == "dense_f32",
        "weight_quant_key_f32": audit.get("weight", {}).get("materialization_quant_key")
        == "f32",
        "model_eps_present": isinstance(audit.get("formula", {}).get("model_eps"), float),
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "evidence": {
            "llamacpp_input_sha256": audit.get("input_capture", {}).get("sha256"),
            "llamacpp_input_bf16_sha256": audit.get("input_bf16_roundtrip", {}).get(
                "sha256"
            ),
            "llamacpp_attn_norm_sha256": audit.get("llamacpp_attn_norm", {}).get(
                "sha256"
            ),
            "hipengine_attn_norm_sha256": audit.get("hipengine_attn_norm", {})
            .get("summary", {})
            .get("sha256"),
            "model_eps": audit.get("formula", {}).get("model_eps"),
        },
    }


def is_exact_candidate(candidate: Mapping[str, Any], expected_name: str) -> bool:
    delta = candidate.get("delta") or {}
    return candidate.get("name") == expected_name and delta.get("exact_match") is True


def build_decision(
    readiness: Mapping[str, Any],
    audit: Mapping[str, Any],
    attn_compare: Mapping[str, Any],
) -> dict[str, Any]:
    if readiness["ready"]:
        selected = "bf16_contracted_llamacpp_or_cpu_oracle"
        reason = (
            "Iteration 321 proved hipEngine exactly matches a BF16(input) + "
            "F32-weight + model-eps + BF16-output RMSNorm candidate, while "
            "llama.cpp exactly matches the F32-input candidate. Continue "
            "bisection under hipEngine's resident BF16 activation contract instead "
            "of adding a temporary F32 runtime branch."
        )
    else:
        selected = "undecided"
        reason = "Required exact-candidate facts are missing from the formula audit."
    return {
        "selected_policy": selected,
        "scope": "layer0_ar_boundary_bisection_only",
        "reason": reason,
        "direct_llamacpp_f32_comparison_status": attn_compare.get("status"),
        "direct_llamacpp_f32_classification": attn_compare.get("classification"),
        "not_selected": [
            {
                "policy": "temporary_hipengine_f32_activation_rmsnorm_path",
                "reason": (
                    "Not needed for the next diagnostic because the existing "
                    "BF16 resident path is exactly explained; adding a runtime "
                    "F32 branch would increase cleanup/refactor burden before a "
                    "deeper unexplained mismatch is found."
                ),
            },
            {
                "policy": "raw_direct_llamacpp_f32_deeper_boundary_compare",
                "reason": (
                    "Direct deeper comparisons would inherit the known attn_norm "
                    "dtype split and are not semantic parity checks unless the "
                    "oracle is contracted to hipEngine's BF16 activation path."
                ),
            },
        ],
        "long_term_note": (
            "This diagnostic policy does not replace the MTP seed contract: "
            "docs/MTP-gguf.md still requires a post-output_norm fp32 hidden seed."
        ),
    }


def extract_boundary_capture_fields(runner_text: str, audit: Mapping[str, Any]) -> dict[str, Any]:
    declared = extract_dataclass_fields(runner_text)
    summary = audit.get("hipengine_attn_norm", {}).get("capture_summary") or {}
    records = []
    for field in BOUNDARY_FIELD_ORDER:
        provenance = dict(FIELD_PROVENANCE[field])
        shape_key = field.replace("_f32", "_shape")
        if field == "recurrent_bf16_f32":
            shape_key = "recurrent_bf16_shape"
        records.append(
            {
                "field": field,
                "declared": field in declared,
                "shape": summary.get(shape_key),
                "source_buffer": provenance["source_buffer"],
                "host_copy": provenance["host_copy"],
                "semantic_stage": provenance["semantic_stage"],
            }
        )
    return {
        "ready": all(record["declared"] for record in records)
        and bool(summary.get("finite", True)),
        "capture_method": "Qwen35GGUFResidentSession.capture_linear_attention_boundary",
        "fields": records,
        "source_order": BOUNDARY_FIELD_ORDER,
        "finite": summary.get("finite"),
        "shape_context": {
            "hidden_size": summary.get("hidden_size"),
            "linear_qkv_width": summary.get("linear_qkv_width"),
            "ssm_inner_size": summary.get("ssm_inner_size"),
            "ssm_time_step_rank": summary.get("ssm_time_step_rank"),
        },
    }


def extract_dataclass_fields(runner_text: str) -> set[str]:
    pattern = (
        r"class Qwen35GGUFLinearAttentionBoundaryCapture:\n"
        r"(?P<body>.*?)(?:\n\n@dataclass|\n\nclass )"
    )
    match = re.search(pattern, runner_text, re.S)
    if not match:
        return set()
    fields: set[str] = set()
    for line in match.group("body").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('"""'):
            continue
        if ":" not in stripped or stripped.startswith("def "):
            continue
        fields.add(stripped.split(":", 1)[0])
    return fields


def build_next_probe_plan(capture_fields: Mapping[str, Any]) -> dict[str, Any]:
    available = {record["field"] for record in capture_fields.get("fields", [])}
    ordered = []
    for field in ["linear_qkv_f32", "linear_z_f32", "ssm_alpha_f32", "ssm_beta_f32"]:
        if field in available:
            ordered.append(
                {
                    "field": field,
                    "oracle_input": "attn_norm_bf16_contracted_exact",
                    "expected_dtype": FIELD_PROVENANCE[field]["host_copy"],
                    "reason": (
                        "This is an immediate consumer of the contracted attn_norm "
                        "buffer before convolution/recurrent state effects."
                    ),
                }
            )
    return {
        "ordered_probes": ordered,
        "first_probe_goal": (
            "separate projection/quantization mismatch from later GDN state effects"
        ),
        "oracle_requirement": (
            "Generate or compute the oracle from BF16(input_embed_f32) through "
            "the exact GGUF weights using model epsilon and the same BF16 output "
            "points as hipEngine; do not compare raw llama.cpp F32 activations "
            "past attn_norm without contraction."
        ),
        "blocked_until": [] if ordered else ["linear_attention_boundary_fields_missing"],
    }


def build_constraints(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "must_not_change_hot_path": True,
        "no_torch_hot_path": True,
        "no_backend_or_quant_dispatch_branch": True,
        "no_performance_claim": True,
        "oracle_exactness_before_deeper_bisection": True,
        "diagnostic_seed_contract": "BF16 resident activation after token embedding",
        "mtp_seed_contract_unchanged": "post_output_norm fp32 hidden seed",
        "model_eps": audit.get("formula", {}).get("model_eps"),
    }


def next_action(status: str, decision: Mapping[str, Any]) -> str:
    if status != "ready":
        return "inspect_layer0_dtype_oracle_policy_blockers"
    if decision.get("selected_policy") == "bf16_contracted_llamacpp_or_cpu_oracle":
        return "build_layer0_bf16_contracted_projection_oracle"
    return "decide_layer0_dtype_oracle_policy_manually"


if __name__ == "__main__":
    main()
