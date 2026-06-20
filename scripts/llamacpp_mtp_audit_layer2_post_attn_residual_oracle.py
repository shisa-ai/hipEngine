#!/usr/bin/env python3
"""Audit layer-2 post-attention residual/post-norm under the BF16 contract."""

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

from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle import (  # noqa: E402
    load_post_attention_norm_weight,
)
from scripts.llamacpp_mtp_audit_layer1_post_attn_residual_oracle import (  # noqa: E402
    POST_ATTN_FIELDS,
    build_layer1_post_attn_results,
    capture_layer1_full_layer,
    compare_input_hashes,
)
from scripts.llamacpp_mtp_audit_layer2_conv_gdn_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER2_CONV_GDN,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter342-layer2-post-attn-residual-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
NormWeightLoader = Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]]
InputReferenceLoader = Callable[[Path, Mapping[str, Any]], dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conv-gdn-artifact", type=Path, default=DEFAULT_LAYER2_CONV_GDN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--residual-atol", type=float, default=0.0)
    parser.add_argument("--post-norm-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=342)
    args = parser.parse_args()

    artifact = audit_layer2_post_attn_residual_oracle(
        conv_gdn_artifact_path=args.conv_gdn_artifact,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        residual_atol=args.residual_atol,
        post_norm_atol=args.post_norm_atol,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact["classification"],
                "target_position": artifact["target_position"],
                "input_classification": artifact.get("input_classification"),
                "field_classifications": field_classifications(artifact),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer2_post_attn_residual_oracle(
    *,
    conv_gdn_artifact_path: Path,
    model_path: Path | None = None,
    layer_id: int = 2,
    max_sequence_length: int | None = None,
    residual_atol: float = 0.0,
    post_norm_atol: float = 2.5e-4,
    iteration: int = 342,
    layer_capture_fn: LayerCaptureFn | None = None,
    norm_weight_loader: NormWeightLoader | None = None,
    input_reference_loader: InputReferenceLoader | None = None,
) -> dict[str, Any]:
    conv_gdn = json.loads(conv_gdn_artifact_path.read_text())
    validate_layer2_conv_gdn(conv_gdn, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or conv_gdn["model"])
    target_position = int(conv_gdn["target_position"])
    prompt_tokens = tuple(int(token) for token in conv_gdn["prompt_tokens"])
    token_id = int(prompt_tokens[target_position])
    capture_fn = layer_capture_fn or capture_layer1_full_layer
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        target_position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            conv_gdn_artifact_path=conv_gdn_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            target_position=target_position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    summary = capture.get("summary") or {}
    if int(summary.get("layer_id", -1)) != int(layer_id):
        raise ValueError("layer-2 post-attn capture layer_id mismatch")
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        raise ValueError("layer-2 post-attn capture must run preceding layers")
    reference_loader = input_reference_loader or load_layer2_input_references
    input_references = reference_loader(conv_gdn_artifact_path, conv_gdn)
    input_results = compare_input_hashes(
        capture=capture,
        input_references=input_references,
    )
    selected_norm_loader = norm_weight_loader or load_post_attention_norm_weight
    post_norm_weight, eps, norm_metadata = selected_norm_loader(resolved_model, int(layer_id))
    oracle_results = build_layer1_post_attn_results(
        capture=capture,
        post_norm_weight=post_norm_weight,
        eps=float(eps),
        residual_atol=float(residual_atol),
        post_norm_atol=float(post_norm_atol),
    )
    input_classification = classify_layer2_inputs(input_results)
    classification = classify_layer2_post_attn(input_classification, oracle_results)
    return {
        "schema": 1,
        "kind": "layer2_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "conv_gdn_artifact_path": str(conv_gdn_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "target_position": target_position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "verified layer2 hidden_in + verified layer2 attn_out",
            "conv_gdn_source": conv_gdn.get("classification"),
            "capture_mode": "full layer capture with run_preceding_layers=True",
            "hidden_reference": input_references["hidden_in_f32"]["source"],
            "attn_out_reference": input_references["attn_out_f32"]["source"],
            "post_norm_tensor": norm_metadata["tensor_name"],
            "eps": float(eps),
        },
        "input_references": input_references,
        "post_attention_norm_weight": norm_metadata,
        "hipengine_capture": summarize_capture(capture),
        "input_results": input_results,
        "input_classification": input_classification,
        "oracle_results": oracle_results,
        "tolerances": {
            "residual_f32": float(residual_atol),
            "post_norm_f32": float(post_norm_atol),
        },
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer2_conv_gdn(conv_gdn: Mapping[str, Any], *, expected_layer_id: int) -> None:
    if conv_gdn.get("status") != "ready":
        raise ValueError("layer2 conv/GDN artifact must be ready")
    if not str(conv_gdn.get("classification", "")).startswith(
        "layer2_warm_conv_gdn_matches_oracle"
    ):
        raise ValueError("layer2 conv/GDN artifact must have matched")
    if int(conv_gdn.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer2 conv/GDN layer_id does not match requested layer")
    if conv_gdn.get("next_action") != "audit_layer2_post_attn_residual_or_moe_boundary":
        raise ValueError("layer2 conv/GDN artifact must point to post-attn audit")
    attn_result = (conv_gdn.get("oracle_results") or {}).get("attn_out_f32") or {}
    if not str(attn_result.get("classification", "")).startswith("warm_field_matches_oracle"):
        raise ValueError("layer2 conv/GDN artifact must establish attn_out")


def load_layer2_input_references(
    conv_gdn_artifact_path: Path,
    conv_gdn: Mapping[str, Any],
) -> dict[str, Any]:
    projection_path = resolve_artifact_path(
        conv_gdn["layer2_projection_path"],
        relative_to=conv_gdn_artifact_path,
    )
    projection = json.loads(projection_path.read_text())
    attn_norm_path = resolve_artifact_path(
        projection["layer2_attn_norm_path"],
        relative_to=projection_path,
    )
    attn_norm = json.loads(attn_norm_path.read_text())
    handoff_path = resolve_artifact_path(
        attn_norm["layer2_handoff_path"],
        relative_to=attn_norm_path,
    )
    handoff = json.loads(handoff_path.read_text())
    return {
        "hidden_in_f32": {
            "source": str(handoff_path),
            "source_kind": handoff.get("kind"),
            "source_classification": handoff.get("classification"),
            "expected_sha256": handoff["target_capture"]["fields"]["hidden_in_f32"][
                "sha256"
            ],
        },
        "attn_out_f32": {
            "source": str(conv_gdn_artifact_path),
            "source_kind": conv_gdn.get("kind"),
            "source_classification": conv_gdn.get("classification"),
            "expected_sha256": conv_gdn["oracle_results"]["attn_out_f32"][
                "hipengine_summary"
            ]["sha256"],
        },
    }


def resolve_artifact_path(path_value: str | Path, *, relative_to: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [path, relative_to.parent / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def classify_layer2_inputs(input_results: Mapping[str, Any]) -> str:
    classes = [
        input_results[field]["classification"]
        for field in ("hidden_in_f32", "attn_out_f32")
    ]
    if all(item == "post_attn_input_hash_matches_reference" for item in classes):
        return "layer2_post_attn_inputs_match_prior_artifacts"
    if any("mismatch" in item for item in classes):
        return "layer2_post_attn_inputs_mismatch_before_residual"
    return "layer2_post_attn_inputs_unavailable"


def classify_layer2_post_attn(
    input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if input_classification != "layer2_post_attn_inputs_match_prior_artifacts":
        return "layer2_post_attn_residual_blocked_input_mismatch"
    classes = [oracle_results[name]["classification"] for name in POST_ATTN_FIELDS]
    if all(item == "post_attn_field_matches_oracle_exactly" for item in classes):
        return "layer2_post_attn_residual_matches_oracle_exactly"
    if all(item.startswith("post_attn_field_matches_oracle") for item in classes):
        return "layer2_post_attn_residual_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer2_post_attn_residual_mismatch_after_oracle"
    return "layer2_post_attn_residual_oracle_unavailable"


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
        "layer2_post_attn_residual_matches_oracle_exactly",
        "layer2_post_attn_residual_matches_oracle_within_tolerance",
    }:
        return "audit_layer2_moe_router_from_post_norm"
    if classification == "layer2_post_attn_residual_blocked_input_mismatch":
        return "reconcile_layer2_hidden_or_attn_out_before_post_attn"
    if classification == "layer2_post_attn_residual_mismatch_after_oracle":
        return "inspect_layer2_post_attn_residual_or_norm_kernel"
    return "rerun_layer2_post_attn_residual_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: artifact["oracle_results"][name]["classification"]
        for name in POST_ATTN_FIELDS
        if name in artifact.get("oracle_results", {})
    }


def unavailable_artifact(
    *,
    conv_gdn_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    target_position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer2_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer2_post_attn_residual_oracle_unavailable",
        "conv_gdn_artifact_path": str(conv_gdn_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "target_position": int(target_position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_layer2_post_attn_residual_oracle_on_rocm_host",
    }


if __name__ == "__main__":
    main()
