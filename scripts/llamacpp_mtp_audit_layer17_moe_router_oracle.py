#!/usr/bin/env python3
"""Audit layer-17 MoE router/top-k/shared-gate from verified post_norm."""

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
from scripts.llamacpp_mtp_audit_layer0_moe_router_oracle import (  # noqa: E402
    ROUTER_COMPARE_FIELDS,
    compare_router_oracle,
    compute_router_oracle,
    load_router_weights,
    summarize_router,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer1_post_attn_residual_oracle import (  # noqa: E402
    capture_layer1_full_layer,
)
from scripts.llamacpp_mtp_audit_layer17_post_attn_residual_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER17_POST_ATTN,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter458-layer17-moe-router-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
RouterWeightLoader = Callable[
    [Path, int],
    tuple[dict[str, np.ndarray], dict[str, Any]],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-attn-artifact",
        type=Path,
        default=DEFAULT_LAYER17_POST_ATTN,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=17)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--routing-atol", type=float, default=1.0e-6)
    parser.add_argument("--shared-gate-atol", type=float, default=1.0e-6)
    parser.add_argument("--iteration", type=int, default=458)
    args = parser.parse_args()

    artifact = audit_layer17_moe_router_oracle(
        post_attn_artifact_path=args.post_attn_artifact,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        routing_atol=args.routing_atol,
        shared_gate_atol=args.shared_gate_atol,
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
                "post_norm_input_classification": artifact.get(
                    "post_norm_input_classification"
                ),
                "field_classifications": field_classifications(artifact),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer17_moe_router_oracle(
    *,
    post_attn_artifact_path: Path,
    model_path: Path | None = None,
    layer_id: int = 17,
    max_sequence_length: int | None = None,
    routing_atol: float = 1.0e-6,
    shared_gate_atol: float = 1.0e-6,
    iteration: int = 458,
    layer_capture_fn: LayerCaptureFn | None = None,
    router_weight_loader: RouterWeightLoader | None = None,
) -> dict[str, Any]:
    post_attn = json.loads(post_attn_artifact_path.read_text())
    validate_layer17_post_attn_artifact(post_attn, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or post_attn["model"])
    position = int(post_attn.get("position", post_attn.get("target_position")))
    prompt_tokens = tuple(int(token) for token in post_attn["prompt_tokens"])
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
            post_attn_artifact_path=post_attn_artifact_path,
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
        raise ValueError("layer-17 MoE router capture layer_id mismatch")
    if not summary.get("is_moe", False):
        raise ValueError("layer capture is not MoE-enabled")
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        raise ValueError("layer-17 MoE router capture must run preceding layers")
    post_norm_input = compare_post_norm_hash(capture=capture, post_attn=post_attn)
    selected_loader = router_weight_loader or load_router_weights
    weights, weight_metadata = selected_loader(resolved_model, int(layer_id))
    top_k = int(summary["top_k"])
    router = compute_router_oracle(
        np.asarray(capture["fields"]["post_norm_f32"], dtype=np.float32),
        weights["router_weight_bf16_f32"],
        weights["shared_gate_weight_bf16_f32"],
        top_k=top_k,
    )
    oracle_results = compare_router_oracle(
        router=router,
        capture=capture,
        routing_atol=float(routing_atol),
        shared_gate_atol=float(shared_gate_atol),
    )
    post_norm_input_classification = classify_post_norm_input(post_norm_input)
    classification = classify_layer17_moe_router(
        post_norm_input_classification,
        oracle_results,
    )
    return {
        "schema": 1,
        "kind": "layer17_moe_router_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "post_attn_artifact_path": str(post_attn_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "verified layer17 post_norm_f32 -> BF16-contracted MoE router",
            "post_attn_source": post_attn.get("classification"),
            "router_weight_contract": "GGUF F32 rounded to resident BF16 before dot product",
            "top_k": top_k,
            "selection": "descending logits, softmax over top-k logits",
            "shared_gate_capture": "raw shared-gate logit; combine kernel applies sigmoid later",
        },
        "post_norm_input": post_norm_input,
        "post_norm_input_classification": post_norm_input_classification,
        "weights": weight_metadata,
        "hipengine_capture": summarize_capture(capture),
        "router_oracle": summarize_router(router),
        "oracle_results": oracle_results,
        "tolerances": {
            "routing_weights_f32": float(routing_atol),
            "shared_gate_logit_f32": float(shared_gate_atol),
        },
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer17_post_attn_artifact(
    post_attn: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if post_attn.get("status") != "ready":
        raise ValueError("layer17 post-attn residual artifact must be ready")
    if post_attn.get("classification") != (
        "layer17_post_attn_residual_matches_oracle_exactly"
    ):
        raise ValueError("layer17 post-attn residual artifact must be exact")
    if int(post_attn.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer17 post-attn layer_id does not match requested layer")
    if post_attn.get("next_action") != "audit_layer17_moe_router_from_post_norm":
        raise ValueError("layer17 post-attn artifact must point to MoE router audit")
    delta = (post_attn.get("oracle_results") or {}).get("post_norm_f32", {}).get(
        "delta_oracle_vs_hip",
        {},
    )
    if delta.get("exact_match") is not True:
        raise ValueError("layer17 post-attn post_norm delta must be exact")
    fields = (post_attn.get("hipengine_capture") or {}).get("field_summaries") or {}
    if "post_norm_f32" not in fields:
        raise ValueError("layer17 post-attn artifact missing post_norm summary")


def compare_post_norm_hash(
    *,
    capture: Mapping[str, Any],
    post_attn: Mapping[str, Any],
) -> dict[str, Any]:
    post_norm = np.asarray(capture["fields"]["post_norm_f32"], dtype=np.float32)
    actual_sha = sha256_float32(post_norm)
    expected_sha = str(
        post_attn["hipengine_capture"]["field_summaries"]["post_norm_f32"]["sha256"]
    )
    exact = actual_sha == expected_sha
    return {
        "field": "post_norm_f32",
        "reference_source": "layer17_post_attn_residual_oracle",
        "reference_classification": post_attn.get("classification"),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "exact_hash_match": exact,
        "hipengine_summary": summarize_array(post_norm),
        "classification": (
            "router_input_post_norm_matches_reference"
            if exact
            else "router_input_post_norm_mismatch_before_router"
        ),
    }


def classify_post_norm_input(post_norm_input: Mapping[str, Any]) -> str:
    if post_norm_input.get("exact_hash_match"):
        return "layer17_moe_router_input_matches_post_attn_artifact"
    return "layer17_moe_router_input_mismatch_before_router"


def classify_layer17_moe_router(
    post_norm_input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if post_norm_input_classification != (
        "layer17_moe_router_input_matches_post_attn_artifact"
    ):
        return "layer17_moe_router_blocked_post_norm_input_mismatch"
    classes = [
        oracle_results[name]["classification"]
        for name in ROUTER_COMPARE_FIELDS
    ]
    if all(item == "router_field_matches_oracle_exactly" for item in classes):
        return "layer17_moe_router_matches_oracle_exactly"
    if all(item.startswith("router_field_matches_oracle") for item in classes):
        return "layer17_moe_router_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer17_moe_router_mismatch_after_oracle"
    return "layer17_moe_router_oracle_unavailable"


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
        "layer17_moe_router_matches_oracle_exactly",
        "layer17_moe_router_matches_oracle_within_tolerance",
    }:
        return "audit_layer17_moe_selected_and_shared_expert_outputs"
    if classification == "layer17_moe_router_blocked_post_norm_input_mismatch":
        return "reconcile_layer17_post_norm_before_moe_router"
    if classification == "layer17_moe_router_mismatch_after_oracle":
        return "inspect_layer17_moe_router_dtype_or_topk_semantics"
    return "rerun_layer17_moe_router_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: artifact["oracle_results"][name]["classification"]
        for name in ROUTER_COMPARE_FIELDS
        if name in artifact.get("oracle_results", {})
    }


def unavailable_artifact(
    *,
    post_attn_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer17_moe_router_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer17_moe_router_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "post_attn_artifact_path": str(post_attn_artifact_path),
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
