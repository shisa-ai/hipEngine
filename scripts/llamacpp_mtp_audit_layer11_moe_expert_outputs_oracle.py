#!/usr/bin/env python3
"""Audit layer-11 MoE selected/shared expert outputs from verified router state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_moe_expert_outputs_oracle import (  # noqa: E402
    EXPERT_COMPARE_FIELDS,
    compare_expert_output_oracle,
    compute_moe_expert_outputs_oracle,
    load_expert_output_weights,
    summarize_expert_oracle,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer1_post_attn_residual_oracle import (  # noqa: E402
    capture_layer1_full_layer,
)
from scripts.llamacpp_mtp_audit_layer11_moe_router_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER11_ROUTER,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter414-layer11-moe-expert-outputs-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
ExpertWeightLoader = Callable[
    [Path, int, np.ndarray],
    tuple[dict[str, np.ndarray], dict[str, Any]],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-artifact",
        type=Path,
        default=DEFAULT_LAYER11_ROUTER,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=11)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--selected-atol", type=float, default=2.5e-4)
    parser.add_argument("--shared-atol", type=float, default=5.0e-7)
    parser.add_argument("--layer-out-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=414)
    args = parser.parse_args()

    artifact = audit_layer11_moe_expert_outputs_oracle(
        router_artifact_path=args.router_artifact,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        selected_atol=args.selected_atol,
        shared_atol=args.shared_atol,
        layer_out_atol=args.layer_out_atol,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact["classification"],
                "position": artifact["position"],
                "router_input_classification": artifact.get(
                    "router_input_classification"
                ),
                "field_classifications": field_classifications(artifact),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer11_moe_expert_outputs_oracle(
    *,
    router_artifact_path: Path,
    model_path: Path | None = None,
    layer_id: int = 11,
    max_sequence_length: int | None = None,
    selected_atol: float = 2.5e-4,
    shared_atol: float = 5.0e-7,
    layer_out_atol: float = 2.5e-4,
    iteration: int = 414,
    layer_capture_fn: LayerCaptureFn | None = None,
    expert_weight_loader: ExpertWeightLoader | None = None,
) -> dict[str, Any]:
    router_artifact = json.loads(router_artifact_path.read_text())
    validate_layer11_router_artifact(router_artifact, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or router_artifact["model"])
    position = int(router_artifact["position"])
    prompt_tokens = tuple(int(token) for token in router_artifact["prompt_tokens"])
    token_id = int(prompt_tokens[position])
    capture_fn = layer_capture_fn or capture_layer1_full_layer
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            router_artifact_path=router_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    summary = capture.get("summary") or {}
    if int(summary.get("layer_id", -1)) != int(layer_id):
        raise ValueError("layer-11 expert capture layer_id mismatch")
    if not summary.get("is_moe", False):
        raise ValueError("layer capture is not MoE-enabled")
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        raise ValueError("layer-11 expert capture must run preceding layers")
    router_inputs = compare_router_inputs(
        capture=capture,
        router_artifact=router_artifact,
    )
    router_input_classification = classify_router_inputs(router_inputs)
    selected = np.asarray(
        capture["fields"]["moe_selected_experts_i64"],
        dtype=np.int64,
    )
    loader = expert_weight_loader or load_expert_output_weights
    weights, weight_metadata = loader(resolved_model, int(layer_id), selected)
    oracle = compute_moe_expert_outputs_oracle(
        post_norm=np.asarray(capture["fields"]["post_norm_f32"], dtype=np.float32),
        residual=np.asarray(capture["fields"]["residual_f32"], dtype=np.float32),
        selected_experts=selected,
        routing_weights=np.asarray(
            capture["fields"]["moe_routing_weights_f32"],
            dtype=np.float32,
        ),
        shared_gate_logit=np.asarray(
            capture["fields"]["moe_shared_gate_f32"],
            dtype=np.float32,
        ),
        weights=weights,
    )
    oracle_results = compare_expert_output_oracle(
        oracle=oracle,
        capture=capture,
        selected_atol=float(selected_atol),
        shared_atol=float(shared_atol),
        layer_out_atol=float(layer_out_atol),
    )
    classification = classify_layer11_expert_outputs(
        router_input_classification,
        oracle_results,
    )
    return {
        "schema": 1,
        "kind": "layer11_moe_expert_outputs_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "router_artifact_path": str(router_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "verified layer11 post_norm/router -> raw GGUF expert branches",
            "router_source": router_artifact.get("classification"),
            "selected_experts": [int(value) for value in selected.tolist()],
            "selected_branch": (
                "Q4_K gate/up BF16 GEMV -> BF16 SiLU*up -> Q5_K down BF16 GEMV"
            ),
            "shared_branch": "Q8_0 gate/up/down BF16 GEMV chain",
            "gemv_threads": 128,
            "combine": (
                "weighted selected branch is BF16-rounded before residual + "
                "sigmoid(shared_gate_logit) * shared_out"
            ),
        },
        "router_inputs": router_inputs,
        "router_input_classification": router_input_classification,
        "weights": weight_metadata,
        "hipengine_capture": summarize_capture(capture),
        "expert_oracle": summarize_expert_oracle(oracle),
        "oracle_results": oracle_results,
        "tolerances": {
            "ffn_or_moe_down_f32": float(selected_atol),
            "moe_shared_out_f32": float(shared_atol),
            "layer_out_f32": float(layer_out_atol),
        },
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer11_router_artifact(
    router: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if router.get("status") != "ready":
        raise ValueError("layer11 router artifact must be ready")
    if not str(router.get("classification", "")).startswith(
        "layer11_moe_router_matches"
    ):
        raise ValueError("layer11 router artifact must have matched")
    if int(router.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer11 router layer_id does not match requested layer")
    if router.get("next_action") != (
        "audit_layer11_moe_selected_and_shared_expert_outputs"
    ):
        raise ValueError("layer11 router artifact must point to expert-output audit")
    if (router.get("post_norm_input") or {}).get("exact_hash_match") is not True:
        raise ValueError("layer11 router post_norm input must match prior artifact")


def compare_router_inputs(
    *,
    capture: Mapping[str, Any],
    router_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    selected = np.asarray(
        capture["fields"]["moe_selected_experts_i64"],
        dtype=np.int64,
    )
    expected_selected = np.asarray(
        router_artifact["oracle_results"]["selected_experts_i64"]["oracle_values"],
        dtype=np.int64,
    )
    post_norm = np.asarray(capture["fields"]["post_norm_f32"], dtype=np.float32)
    post_norm_sha = sha256_float32(post_norm)
    expected_post_norm_sha = str(
        router_artifact["post_norm_input"]["expected_sha256"]
    )
    selected_match = bool(
        selected.shape == expected_selected.shape
        and np.array_equal(selected, expected_selected)
    )
    return {
        "post_norm_f32": {
            "field": "post_norm_f32",
            "expected_sha256": expected_post_norm_sha,
            "actual_sha256": post_norm_sha,
            "exact_hash_match": post_norm_sha == expected_post_norm_sha,
            "summary": summarize_array(post_norm),
            "classification": (
                "expert_input_post_norm_matches_router_artifact"
                if post_norm_sha == expected_post_norm_sha
                else "expert_input_post_norm_mismatch_before_experts"
            ),
        },
        "selected_experts_i64": {
            "field": "moe_selected_experts_i64",
            "expected_values": [int(value) for value in expected_selected.tolist()],
            "actual_values": [int(value) for value in selected.tolist()],
            "exact_match": selected_match,
            "classification": (
                "expert_input_selected_experts_match_router_artifact"
                if selected_match
                else "expert_input_selected_experts_mismatch_before_experts"
            ),
        },
    }


def classify_router_inputs(router_inputs: Mapping[str, Any]) -> str:
    if (
        router_inputs["post_norm_f32"].get("exact_hash_match")
        and router_inputs["selected_experts_i64"].get("exact_match")
    ):
        return "layer11_moe_expert_inputs_match_router_artifact"
    return "layer11_moe_expert_inputs_mismatch_before_experts"


def classify_layer11_expert_outputs(
    router_input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if router_input_classification != (
        "layer11_moe_expert_inputs_match_router_artifact"
    ):
        return "layer11_moe_expert_outputs_blocked_router_input_mismatch"
    classes = [
        oracle_results[name]["classification"]
        for name in EXPERT_COMPARE_FIELDS
    ]
    if all(item == "expert_output_matches_oracle_exactly" for item in classes):
        return "layer11_moe_expert_outputs_match_oracle_exactly"
    if all(item.startswith("expert_output_matches_oracle") for item in classes):
        return "layer11_moe_expert_outputs_match_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer11_moe_expert_outputs_mismatch_after_oracle"
    return "layer11_moe_expert_outputs_oracle_unavailable"


def status_from_classification(classification: str) -> str:
    if classification.endswith("unavailable"):
        return "unavailable"
    if "blocked" in classification:
        return "blocked"
    if "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer11_moe_expert_outputs_match_oracle_exactly",
        "layer11_moe_expert_outputs_match_oracle_within_tolerance",
    }:
        return "audit_layer12_bf16_handoff_or_mtp_next_boundary"
    if classification == "layer11_moe_expert_outputs_blocked_router_input_mismatch":
        return "reconcile_layer11_moe_router_inputs_before_expert_outputs"
    if classification == "layer11_moe_expert_outputs_mismatch_after_oracle":
        return "inspect_layer11_moe_expert_output_dtype_or_dequant_semantics"
    return "rerun_layer11_moe_expert_outputs_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: artifact["oracle_results"][name]["classification"]
        for name in EXPERT_COMPARE_FIELDS
        if name in artifact.get("oracle_results", {})
    }


def unavailable_artifact(
    *,
    router_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer11_moe_expert_outputs_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer11_moe_expert_outputs_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "router_artifact_path": str(router_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
