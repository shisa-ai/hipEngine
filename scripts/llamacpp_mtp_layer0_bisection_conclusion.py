#!/usr/bin/env python3
"""Summarize the layer-0 bisection after the MoE/FFN oracle chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LAYER_COMPARE = Path("benchmarks/results/mtp-gguf-iter316-layer0-compare.json")
DEFAULT_INPUT_COMPARE = Path("benchmarks/results/mtp-gguf-iter318-input-embed-compare.json")
DEFAULT_ATTN_NORM = Path(
    "benchmarks/results/mtp-gguf-iter321-layer0-attn-norm-formula-audit.json"
)
DEFAULT_PROJECTION = Path(
    "benchmarks/results/mtp-gguf-iter323-layer0-bf16-projection-oracle.json"
)
DEFAULT_CONV_GDN = Path(
    "benchmarks/results/mtp-gguf-iter326-layer0-warm-conv-gdn-oracle.json"
)
DEFAULT_POST_ATTN = Path(
    "benchmarks/results/mtp-gguf-iter327-layer0-post-attn-residual-oracle.json"
)
DEFAULT_ROUTER = Path("benchmarks/results/mtp-gguf-iter328-layer0-moe-router-oracle.json")
DEFAULT_EXPERTS = Path(
    "benchmarks/results/mtp-gguf-iter329-layer0-moe-expert-outputs-oracle.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter330-layer0-bisection-conclusion.json"
)

PREREQUISITE_ORDER = (
    "layer_compare",
    "input_compare",
    "attn_norm",
    "projection",
    "conv_gdn",
    "post_attn",
    "router",
    "experts",
)

EXPECTED_STATES = {
    "layer_compare": {
        "kind": "llamacpp_vs_hipengine_layer_boundary_compare",
        "status": "mismatched",
        "classification": "layer_boundary_mismatch",
    },
    "input_compare": {
        "kind": "llamacpp_vs_hipengine_input_embed_compare",
        "status": "mismatched",
        "classification": "input_embed_matches_after_bf16_roundtrip",
    },
    "attn_norm": {
        "kind": "layer0_attn_norm_formula_audit",
        "status": "ready",
        "conclusion": "attn_norm_mismatch_explained_by_input_activation_bf16_contraction",
    },
    "projection": {
        "kind": "layer0_bf16_contracted_projection_oracle",
        "status": "ready",
        "classification": "layer0_projections_match_bf16_oracle_within_rounding",
    },
    "conv_gdn": {
        "kind": "layer0_warm_bf16_contracted_conv_gdn_oracle",
        "status": "ready",
        "classification": "layer0_warm_conv_gdn_matches_oracle_within_tolerance",
    },
    "post_attn": {
        "kind": "layer0_post_attn_residual_oracle",
        "status": "ready",
        "classification": "layer0_post_attn_residual_matches_oracle_exactly",
    },
    "router": {
        "kind": "layer0_moe_router_oracle",
        "status": "ready",
        "classification_prefix": "layer0_moe_router_matches_oracle",
    },
    "experts": {
        "kind": "layer0_moe_expert_outputs_oracle",
        "status": "ready",
        "classification": "layer0_moe_expert_outputs_match_oracle_exactly",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-compare", type=Path, default=DEFAULT_LAYER_COMPARE)
    parser.add_argument("--input-compare", type=Path, default=DEFAULT_INPUT_COMPARE)
    parser.add_argument("--attn-norm", type=Path, default=DEFAULT_ATTN_NORM)
    parser.add_argument("--projection", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--conv-gdn", type=Path, default=DEFAULT_CONV_GDN)
    parser.add_argument("--post-attn", type=Path, default=DEFAULT_POST_ATTN)
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--experts", type=Path, default=DEFAULT_EXPERTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=330)
    args = parser.parse_args()

    artifact = build_layer0_bisection_conclusion(
        artifact_paths={
            "layer_compare": args.layer_compare,
            "input_compare": args.input_compare,
            "attn_norm": args.attn_norm,
            "projection": args.projection,
            "conv_gdn": args.conv_gdn,
            "post_attn": args.post_attn,
            "router": args.router,
            "experts": args.experts,
        },
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact["classification"],
                "layer_compare_rmse": artifact["layer_out_llamacpp_vs_hipengine"]["rmse"],
                "internal_layer_out_max_abs": artifact["internal_bf16_oracle_chain"][
                    "final_layer_out_delta"
                ]["max_abs_diff"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_layer0_bisection_conclusion(
    *,
    artifact_paths: Mapping[str, Path],
    iteration: int = 330,
) -> dict[str, Any]:
    docs = {name: json.loads(Path(path).read_text()) for name, path in artifact_paths.items()}
    prerequisite_results = validate_prerequisites(docs)
    ready = all(item["ready"] for item in prerequisite_results.values())
    consistency = validate_common_boundary(docs)
    ready = bool(ready and consistency["ready"])
    layer_delta = docs["layer_compare"].get("numeric_delta") or {}
    expert_layer_delta = (
        docs["experts"].get("oracle_results", {}).get("layer_out_f32", {}).get(
            "delta_oracle_vs_hip",
            {},
        )
    )
    classification = classify_conclusion(
        ready=ready,
        layer_delta=layer_delta,
        expert_layer_delta=expert_layer_delta,
        prerequisite_results=prerequisite_results,
        consistency=consistency,
    )
    return {
        "schema": 1,
        "kind": "layer0_bisection_conclusion",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "model": docs["experts"].get("model") or docs["layer_compare"].get("model"),
        "layer_id": int(
            docs["experts"].get("layer_id", docs["layer_compare"].get("layer_id", 0))
        ),
        "position": int(
            docs["experts"].get(
                "target_position",
                docs["layer_compare"].get("position", 0),
            )
        ),
        "token_id": int(
            docs["experts"].get("token_id", docs["layer_compare"].get("token_id", -1))
        ),
        "prompt_tokens": docs["experts"].get("prompt_tokens")
        or docs["layer_compare"].get("prompt_tokens"),
        "prerequisites": prerequisite_results,
        "boundary_consistency": consistency,
        "layer_out_llamacpp_vs_hipengine": summarize_layer_delta(docs["layer_compare"]),
        "input_and_attn_norm_dtype_finding": summarize_dtype_finding(
            input_compare=docs["input_compare"],
            attn_norm=docs["attn_norm"],
        ),
        "internal_bf16_oracle_chain": summarize_internal_chain(docs),
        "conclusion": conclusion_text(classification),
        "limitations": [
            "The direct llama.cpp layer_out comparison remains an F32 llama.cpp boundary "
            "versus hipEngine resident-BF16 boundary comparison.",
            "This artifact summarizes correctness evidence only; it is not a performance claim.",
        ],
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_prerequisites(docs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name in PREREQUISITE_ORDER:
        doc = docs[name]
        expected = EXPECTED_STATES[name]
        facts: dict[str, bool] = {
            "kind": doc.get("kind") == expected["kind"],
            "status": doc.get("status") == expected["status"],
        }
        if "classification" in expected:
            facts["classification"] = doc.get("classification") == expected["classification"]
        if "classification_prefix" in expected:
            facts["classification"] = str(doc.get("classification", "")).startswith(
                str(expected["classification_prefix"])
            )
        if "conclusion" in expected:
            facts["conclusion"] = doc.get("conclusion") == expected["conclusion"]
        results[name] = {
            "ready": all(facts.values()),
            "facts": facts,
            "kind": doc.get("kind"),
            "status": doc.get("status"),
            "classification": doc.get("classification"),
            "conclusion": doc.get("conclusion"),
            "iteration": doc.get("iteration"),
            "next_action": doc.get("next_action"),
        }
    return results


def validate_common_boundary(docs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reference = boundary_tuple(docs["experts"])
    per_doc = {name: boundary_tuple(doc) for name, doc in docs.items()}
    same_boundary = {name: item == reference for name, item in per_doc.items()}
    prompt_ref = tuple(int(value) for value in docs["experts"].get("prompt_tokens", []))
    prompt_checks = {}
    for name, doc in docs.items():
        prompt = tuple(int(value) for value in doc.get("prompt_tokens", []))
        prompt_checks[name] = bool(prompt and prompt == prompt_ref)
    return {
        "ready": all(same_boundary.values()) and all(prompt_checks.values()),
        "reference": {
            "layer_id": reference[0],
            "position": reference[1],
            "token_id": reference[2],
        },
        "per_doc": {
            name: {"layer_id": item[0], "position": item[1], "token_id": item[2]}
            for name, item in per_doc.items()
        },
        "same_layer_position_token": same_boundary,
        "same_prompt_tokens": prompt_checks,
    }


def boundary_tuple(doc: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(doc.get("layer_id", 0)),
        int(doc.get("position", doc.get("target_position", 0))),
        int(doc.get("token_id", -1)),
    )


def classify_conclusion(
    *,
    ready: bool,
    layer_delta: Mapping[str, Any],
    expert_layer_delta: Mapping[str, Any],
    prerequisite_results: Mapping[str, Mapping[str, Any]],
    consistency: Mapping[str, Any],
) -> str:
    if not consistency.get("ready"):
        return "layer0_bisection_conclusion_blocked_by_boundary_mismatch"
    if not all(item.get("ready") for item in prerequisite_results.values()):
        return "layer0_bisection_conclusion_blocked_by_prerequisite"
    if not ready:
        return "layer0_bisection_conclusion_blocked"
    if not layer_delta.get("shape_match") or layer_delta.get("exact_match"):
        return "layer0_bisection_conclusion_not_applicable"
    if not expert_layer_delta.get("exact_match"):
        return "layer0_bisection_internal_oracle_chain_incomplete"
    return "layer0_runtime_matches_bf16_oracle_chain_after_llamacpp_f32_split"


def status_from_classification(classification: str) -> str:
    if "blocked" in classification:
        return "blocked"
    if classification.endswith("not_applicable"):
        return "not_applicable"
    if classification.endswith("incomplete"):
        return "incomplete"
    return "ready"


def summarize_layer_delta(layer_compare: Mapping[str, Any]) -> dict[str, Any]:
    delta = layer_compare.get("numeric_delta") or {}
    capture = layer_compare.get("llamacpp_capture") or {}
    hip = layer_compare.get("hipengine_capture") or {}
    all_rows = delta.get("all_rows_scan") or {}
    return {
        "source_artifact_iteration": layer_compare.get("iteration"),
        "status": layer_compare.get("status"),
        "classification": layer_compare.get("classification"),
        "llamacpp_effective_tap": capture.get("effective_tap"),
        "hipengine_provenance": hip.get("provenance"),
        "shape_match": delta.get("shape_match"),
        "exact_match": delta.get("exact_match"),
        "count": delta.get("count"),
        "max_abs_diff": delta.get("max_abs_diff"),
        "mean_abs_diff": delta.get("mean_abs_diff"),
        "rmse": delta.get("rmse"),
        "llamacpp_sha256": delta.get("llamacpp_sha256"),
        "hipengine_sha256": delta.get("hipengine_sha256"),
        "best_all_rows_rmse_row": (all_rows.get("best_by_rmse") or {}).get("row"),
        "best_all_rows_rmse": (all_rows.get("best_by_rmse") or {}).get("rmse"),
    }


def summarize_dtype_finding(
    *,
    input_compare: Mapping[str, Any],
    attn_norm: Mapping[str, Any],
) -> dict[str, Any]:
    input_delta = input_compare.get("numeric_delta") or {}
    bf16_delta = input_compare.get("bf16_rounded_delta") or {}
    best = attn_norm.get("best_candidates") or {}
    hip = best.get("vs_hipengine_attn_norm") or {}
    llama = best.get("vs_llamacpp_attn_norm") or {}
    return {
        "input_classification": input_compare.get("classification"),
        "input_exact_rmse": input_delta.get("rmse"),
        "input_bf16_exact_match": bf16_delta.get("exact_match"),
        "attn_norm_conclusion": attn_norm.get("conclusion"),
        "llamacpp_attn_norm_best_candidate": llama.get("name"),
        "hipengine_attn_norm_best_candidate": hip.get("name"),
        "hipengine_attn_norm_delta_exact": (hip.get("delta") or {}).get("exact_match"),
    }


def summarize_internal_chain(docs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expert_layer = docs["experts"].get("oracle_results", {}).get("layer_out_f32", {})
    router_selected = docs["router"].get("oracle_results", {}).get("selected_experts_i64", {})
    return {
        "projection_classification": docs["projection"].get("classification"),
        "conv_gdn_classification": docs["conv_gdn"].get("classification"),
        "post_attn_classification": docs["post_attn"].get("classification"),
        "router_classification": docs["router"].get("classification"),
        "router_selected_experts": router_selected.get("oracle_values"),
        "experts_classification": docs["experts"].get("classification"),
        "final_layer_out_classification": expert_layer.get("classification"),
        "final_layer_out_delta": expert_layer.get("delta_oracle_vs_hip") or {},
        "expert_layer_out_sha256": (
            docs["experts"].get("expert_oracle", {}).get("layer_out_sha256")
        ),
    }


def conclusion_text(classification: str) -> str:
    if classification == "layer0_runtime_matches_bf16_oracle_chain_after_llamacpp_f32_split":
        return (
            "The direct layer-0 llama.cpp layer_out capture still mismatches hipEngine, "
            "but every hipEngine sub-boundary from BF16-contracted input through MoE "
            "combine now matches a torch-free CPU/raw-GGUF oracle. The retained "
            "layer-0 mismatch is therefore the known llama.cpp F32 activation versus "
            "hipEngine resident-BF16 activation split, not an uncovered layer-0 "
            "runtime arithmetic error in the audited chain."
        )
    if "boundary_mismatch" in classification:
        return "Artifacts do not refer to the same layer/position/token/prompt boundary."
    if "prerequisite" in classification:
        return "One or more prerequisite artifacts is not in the expected state."
    return "Layer-0 bisection conclusion is not ready."


def next_action(classification: str) -> str:
    if classification == "layer0_runtime_matches_bf16_oracle_chain_after_llamacpp_f32_split":
        return "advance_bisection_to_layer1_or_next_layer_boundary_under_bf16_contract"
    if "boundary_mismatch" in classification:
        return "repair_layer0_bisection_artifact_boundary_selection"
    if "prerequisite" in classification:
        return "inspect_layer0_bisection_prerequisite_artifacts"
    return "inspect_layer0_bisection_conclusion_blocker"


if __name__ == "__main__":
    main()
