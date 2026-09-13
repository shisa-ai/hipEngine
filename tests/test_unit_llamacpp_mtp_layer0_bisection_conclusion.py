from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_layer0_bisection_conclusion import (
    build_layer0_bisection_conclusion,
    classify_conclusion,
    validate_common_boundary,
    validate_prerequisites,
)


def test_validate_prerequisites_accepts_expected_chain() -> None:
    docs = _docs()

    results = validate_prerequisites(docs)

    assert all(item["ready"] for item in results.values())
    assert results["layer_compare"]["classification"] == "layer_boundary_mismatch"
    assert results["attn_norm"]["conclusion"] == (
        "attn_norm_mismatch_explained_by_input_activation_bf16_contraction"
    )


def test_validate_prerequisites_rejects_unexpected_expert_state() -> None:
    docs = _docs()
    docs["experts"]["classification"] = "layer0_moe_expert_outputs_mismatch_after_oracle"

    results = validate_prerequisites(docs)

    assert results["experts"]["ready"] is False
    assert results["experts"]["facts"]["classification"] is False


def test_validate_common_boundary_checks_position_and_prompt() -> None:
    docs = _docs()

    consistency = validate_common_boundary(docs)

    assert consistency["ready"] is True
    assert consistency["reference"] == {"layer_id": 0, "position": 16, "token_id": 271}
    assert all(consistency["same_prompt_tokens"].values())


def test_validate_common_boundary_rejects_prompt_mismatch() -> None:
    docs = _docs()
    docs["router"]["prompt_tokens"] = [1, 2, 99]

    consistency = validate_common_boundary(docs)

    assert consistency["ready"] is False
    assert consistency["same_prompt_tokens"]["router"] is False


def test_classify_conclusion_ready_when_direct_boundary_mismatch_has_exact_oracle_chain() -> None:
    docs = _docs()
    prerequisites = validate_prerequisites(docs)
    consistency = validate_common_boundary(docs)

    classification = classify_conclusion(
        ready=True,
        layer_delta=docs["layer_compare"]["numeric_delta"],
        expert_layer_delta=docs["experts"]["oracle_results"]["layer_out_f32"][
            "delta_oracle_vs_hip"
        ],
        prerequisite_results=prerequisites,
        consistency=consistency,
    )

    assert classification == (
        "layer0_runtime_matches_bf16_oracle_chain_after_llamacpp_f32_split"
    )


def test_build_layer0_bisection_conclusion_with_injected_artifacts(tmp_path: Path) -> None:
    paths = {}
    for name, doc in _docs().items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc))
        paths[name] = path

    artifact = build_layer0_bisection_conclusion(
        artifact_paths=paths,
        iteration=330,
    )

    assert artifact["status"] == "ready"
    assert artifact["classification"] == (
        "layer0_runtime_matches_bf16_oracle_chain_after_llamacpp_f32_split"
    )
    assert artifact["layer_out_llamacpp_vs_hipengine"]["rmse"] == 0.012
    assert artifact["internal_bf16_oracle_chain"]["final_layer_out_delta"]["max_abs_diff"] == 0.0
    assert artifact["next_action"] == (
        "advance_bisection_to_layer1_or_next_layer_boundary_under_bf16_contract"
    )
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def _docs() -> dict[str, dict]:
    prompt = [10, 11, 271]
    base = {
        "layer_id": 0,
        "position": 16,
        "target_position": 16,
        "token_id": 271,
        "prompt_tokens": prompt,
        "model": "/tmp/model.gguf",
    }
    return {
        "layer_compare": {
            **base,
            "kind": "llamacpp_vs_hipengine_layer_boundary_compare",
            "iteration": 316,
            "status": "mismatched",
            "classification": "layer_boundary_mismatch",
            "next_action": "inspect_initial_embedding_or_token_capture",
            "llamacpp_capture": {"effective_tap": "h_nextn_layer_out"},
            "hipengine_capture": {"provenance": "capture_attention_layer.layer_out_f32"},
            "numeric_delta": {
                "shape_match": True,
                "exact_match": False,
                "count": 2048,
                "rmse": 0.012,
                "max_abs_diff": 0.04,
                "mean_abs_diff": 0.009,
                "llamacpp_sha256": "llama",
                "hipengine_sha256": "hip",
                "all_rows_scan": {"best_by_rmse": {"row": 16, "rmse": 0.012}},
            },
        },
        "input_compare": {
            **base,
            "kind": "llamacpp_vs_hipengine_input_embed_compare",
            "iteration": 318,
            "status": "mismatched",
            "classification": "input_embed_matches_after_bf16_roundtrip",
            "next_action": "investigate_layer0_implementation_after_embedding",
            "numeric_delta": {"rmse": 1.0e-6},
            "bf16_rounded_delta": {"exact_match": True},
        },
        "attn_norm": {
            **base,
            "kind": "layer0_attn_norm_formula_audit",
            "iteration": 321,
            "status": "ready",
            "conclusion": "attn_norm_mismatch_explained_by_input_activation_bf16_contraction",
            "next_action": (
                "decide_whether_to_add_f32_activation_path_or_adjust_llamacpp_oracle_dtype"
            ),
            "best_candidates": {
                "vs_llamacpp_attn_norm": {"name": "input_f32_weight_f32_eps_model_f32_out"},
                "vs_hipengine_attn_norm": {
                    "name": "input_bf16_weight_f32_eps_model_bf16_out",
                    "delta": {"exact_match": True},
                },
            },
        },
        "projection": {
            **base,
            "kind": "layer0_bf16_contracted_projection_oracle",
            "iteration": 323,
            "status": "ready",
            "classification": "layer0_projections_match_bf16_oracle_within_rounding",
            "next_action": "continue_layer0_bf16_bisection_at_conv_or_gdn_state",
        },
        "conv_gdn": {
            **base,
            "kind": "layer0_warm_bf16_contracted_conv_gdn_oracle",
            "iteration": 326,
            "status": "ready",
            "classification": "layer0_warm_conv_gdn_matches_oracle_within_tolerance",
            "next_action": "continue_layer0_bisection_after_attn_out_or_residual",
        },
        "post_attn": {
            **base,
            "kind": "layer0_post_attn_residual_oracle",
            "iteration": 327,
            "status": "ready",
            "classification": "layer0_post_attn_residual_matches_oracle_exactly",
            "next_action": "audit_layer0_moe_router_from_post_norm",
        },
        "router": {
            **base,
            "kind": "layer0_moe_router_oracle",
            "iteration": 328,
            "status": "ready",
            "classification": "layer0_moe_router_matches_oracle_within_tolerance",
            "next_action": "audit_layer0_moe_selected_and_shared_expert_outputs",
            "oracle_results": {
                "selected_experts_i64": {"oracle_values": [200, 140]},
            },
        },
        "experts": {
            **base,
            "kind": "layer0_moe_expert_outputs_oracle",
            "iteration": 329,
            "status": "ready",
            "classification": "layer0_moe_expert_outputs_match_oracle_exactly",
            "next_action": "continue_layer0_bisection_after_moe_ffn_or_compare_layer_out",
            "expert_oracle": {"layer_out_sha256": "oracle-layer"},
            "oracle_results": {
                "layer_out_f32": {
                    "classification": "expert_output_matches_oracle_exactly",
                    "delta_oracle_vs_hip": {
                        "available": True,
                        "shape_match": True,
                        "exact_match": True,
                        "max_abs_diff": 0.0,
                        "rmse": 0.0,
                    },
                },
            },
        },
    }
