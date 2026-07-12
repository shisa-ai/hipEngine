#!/usr/bin/env python3
"""Audit layer-10 post-attention residual/post-norm under the BF16 contract."""

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

from scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle import (  # noqa: E402
    load_post_attention_norm_weight,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer1_post_attn_residual_oracle import (  # noqa: E402
    POST_ATTN_FIELDS,
    build_layer1_post_attn_results,
    capture_layer1_full_layer,
    compare_input_hashes,
)
from scripts.llamacpp_mtp_audit_layer10_conv_gdn_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER10_CONV_GDN,
)
from scripts.llamacpp_mtp_layer10_bf16_handoff_audit import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER10_HANDOFF,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter403-layer10-post-attn-residual-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
NormWeightLoader = Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]]
InputReferenceLoader = Callable[
    [Path, Mapping[str, Any], Path, Mapping[str, Any]], dict[str, Any]
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conv-gdn-artifact",
        type=Path,
        default=DEFAULT_LAYER10_CONV_GDN,
    )
    parser.add_argument(
        "--handoff-artifact",
        type=Path,
        default=DEFAULT_LAYER10_HANDOFF,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--residual-atol", type=float, default=0.0)
    parser.add_argument("--post-norm-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=403)
    args = parser.parse_args()

    artifact = audit_layer10_post_attn_residual_oracle(
        conv_gdn_artifact_path=args.conv_gdn_artifact,
        handoff_artifact_path=args.handoff_artifact,
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
                "position": artifact["position"],
                "input_classification": artifact.get("input_classification"),
                "field_classifications": field_classifications(artifact),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer10_post_attn_residual_oracle(
    *,
    conv_gdn_artifact_path: Path,
    handoff_artifact_path: Path = DEFAULT_LAYER10_HANDOFF,
    model_path: Path | None = None,
    layer_id: int = 10,
    max_sequence_length: int | None = None,
    residual_atol: float = 0.0,
    post_norm_atol: float = 2.5e-4,
    iteration: int = 403,
    layer_capture_fn: LayerCaptureFn | None = None,
    norm_weight_loader: NormWeightLoader | None = None,
    input_reference_loader: InputReferenceLoader | None = None,
) -> dict[str, Any]:
    conv_gdn = json.loads(conv_gdn_artifact_path.read_text())
    handoff = json.loads(handoff_artifact_path.read_text())
    validate_layer10_conv_gdn(conv_gdn, expected_layer_id=int(layer_id))
    validate_layer10_handoff(handoff, expected_layer_id=int(layer_id))
    validate_source_alignment(conv_gdn=conv_gdn, handoff=handoff)
    resolved_model = Path(model_path or conv_gdn["model"])
    position = int(conv_gdn["target_position"])
    prompt_tokens = tuple(int(token) for token in conv_gdn["prompt_tokens"])
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
            conv_gdn_artifact_path=conv_gdn_artifact_path,
            handoff_artifact_path=handoff_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    metadata_classification = classify_capture_metadata(
        capture,
        layer_id=int(layer_id),
    )
    reference_loader = input_reference_loader or load_layer10_input_references
    input_references = reference_loader(
        conv_gdn_artifact_path,
        conv_gdn,
        handoff_artifact_path,
        handoff,
    )
    if metadata_classification is not None:
        return blocked_artifact(
            conv_gdn_artifact_path=conv_gdn_artifact_path,
            handoff_artifact_path=handoff_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            input_references=input_references,
            input_results={},
            input_classification="layer10_post_attn_inputs_unavailable",
            classification=metadata_classification,
            iteration=iteration,
        )
    input_results = compare_input_hashes(
        capture=capture,
        input_references=input_references,
    )
    selected_norm_loader = norm_weight_loader or load_post_attention_norm_weight
    post_norm_weight, eps, norm_metadata = selected_norm_loader(
        resolved_model,
        int(layer_id),
    )
    oracle_results = build_layer1_post_attn_results(
        capture=capture,
        post_norm_weight=post_norm_weight,
        eps=float(eps),
        residual_atol=float(residual_atol),
        post_norm_atol=float(post_norm_atol),
    )
    input_classification = classify_layer10_inputs(input_results)
    classification = classify_layer10_post_attn(input_classification, oracle_results)
    return {
        "schema": 1,
        "kind": "layer10_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "conv_gdn_artifact_path": str(conv_gdn_artifact_path),
        "handoff_artifact_path": str(handoff_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "verified layer10 hidden_in + verified layer10 attn_out",
            "handoff_source": handoff.get("classification"),
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


def validate_layer10_conv_gdn(
    conv_gdn: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if conv_gdn.get("status") != "ready":
        raise ValueError("layer10 conv/GDN artifact must be ready")
    if not str(conv_gdn.get("classification", "")).startswith(
        "layer10_warm_conv_gdn_matches_oracle"
    ):
        raise ValueError("layer10 conv/GDN artifact must have matched")
    if int(conv_gdn.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer10 conv/GDN layer_id does not match requested layer")
    if conv_gdn.get("next_action") != (
        "audit_layer10_post_attn_residual_or_moe_boundary"
    ):
        raise ValueError("layer10 conv/GDN artifact must point to post-attn audit")
    if conv_gdn.get("target_input_classification") != (
        "target_inputs_match_replay_exactly"
    ):
        raise ValueError("layer10 conv/GDN target inputs must match replay exactly")
    attn_result = (conv_gdn.get("oracle_results") or {}).get("attn_out_f32") or {}
    if not str(attn_result.get("classification", "")).startswith(
        "warm_field_matches_oracle"
    ):
        raise ValueError("layer10 conv/GDN artifact must establish attn_out")


def validate_layer10_handoff(
    handoff: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if handoff.get("status") != "ready":
        raise ValueError("layer10 handoff artifact must be ready")
    if handoff.get("classification") != (
        "layer10_hidden_in_matches_layer9_layer_out_exactly"
    ):
        raise ValueError("layer10 handoff artifact must be exact")
    if int(handoff.get("target_layer", -1)) != int(expected_layer_id):
        raise ValueError(
            "layer10 handoff target_layer does not match requested layer"
        )
    fields = ((handoff.get("target_capture") or {}).get("fields") or {})
    if "hidden_in_f32" not in fields:
        raise ValueError("layer10 handoff artifact missing hidden_in summary")


def validate_source_alignment(
    *,
    conv_gdn: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> None:
    comparisons = {
        "model": conv_gdn.get("model") == handoff.get("model"),
        "position": int(conv_gdn.get("target_position", -1))
        == int(handoff.get("position", -2)),
        "token_id": conv_gdn.get("token_id") == handoff.get("token_id"),
        "prompt_tokens": conv_gdn.get("prompt_tokens")
        == handoff.get("prompt_tokens"),
    }
    for key, matches in comparisons.items():
        if not matches:
            raise ValueError(f"layer10 source artifact {key} mismatch")


def load_layer10_input_references(
    conv_gdn_artifact_path: Path,
    conv_gdn: Mapping[str, Any],
    handoff_artifact_path: Path,
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "hidden_in_f32": {
            "source": str(handoff_artifact_path),
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


def classify_capture_metadata(capture: Mapping[str, Any], *, layer_id: int) -> str | None:
    summary = capture.get("summary") or {}
    if int(summary.get("layer_id", -1)) != int(layer_id):
        return "layer10_post_attn_wrong_layer_capture"
    if str(summary.get("layer_type")) != "linear_attention":
        return "layer10_post_attn_wrong_layer_type"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer10_post_attn_wrong_preceding_layer_count"
    return None


def classify_layer10_inputs(input_results: Mapping[str, Any]) -> str:
    classes = [
        input_results[field]["classification"]
        for field in ("hidden_in_f32", "attn_out_f32")
    ]
    if all(item == "post_attn_input_hash_matches_reference" for item in classes):
        return "layer10_post_attn_inputs_match_prior_artifacts"
    if any("mismatch" in item for item in classes):
        return "layer10_post_attn_inputs_mismatch_before_residual"
    return "layer10_post_attn_inputs_unavailable"


def classify_layer10_post_attn(
    input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if input_classification != "layer10_post_attn_inputs_match_prior_artifacts":
        return "layer10_post_attn_residual_blocked_input_mismatch"
    classes = [oracle_results[name]["classification"] for name in POST_ATTN_FIELDS]
    if all(item == "post_attn_field_matches_oracle_exactly" for item in classes):
        return "layer10_post_attn_residual_matches_oracle_exactly"
    if all(item.startswith("post_attn_field_matches_oracle") for item in classes):
        return "layer10_post_attn_residual_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer10_post_attn_residual_mismatch_after_oracle"
    return "layer10_post_attn_residual_oracle_unavailable"


def status_from_classification(classification: str) -> str:
    if classification.endswith("unavailable"):
        return "unavailable"
    if "blocked" in classification or "wrong" in classification:
        return "blocked"
    if "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer10_post_attn_residual_matches_oracle_exactly",
        "layer10_post_attn_residual_matches_oracle_within_tolerance",
    }:
        return "audit_layer10_moe_router_from_post_norm"
    if classification == "layer10_post_attn_residual_blocked_input_mismatch":
        return "reconcile_layer10_hidden_or_attn_out_before_post_attn"
    if classification in {
        "layer10_post_attn_wrong_layer_capture",
        "layer10_post_attn_wrong_layer_type",
        "layer10_post_attn_wrong_preceding_layer_count",
    }:
        return "inspect_layer10_post_attn_capture_metadata"
    if classification == "layer10_post_attn_residual_mismatch_after_oracle":
        return "inspect_layer10_post_attn_residual_or_norm_kernel"
    return "rerun_layer10_post_attn_residual_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: artifact["oracle_results"][name]["classification"]
        for name in POST_ATTN_FIELDS
        if name in artifact.get("oracle_results", {})
    }


def unavailable_artifact(
    *,
    conv_gdn_artifact_path: Path,
    handoff_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer10_post_attn_residual_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer10_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "conv_gdn_artifact_path": str(conv_gdn_artifact_path),
        "handoff_artifact_path": str(handoff_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def blocked_artifact(
    *,
    conv_gdn_artifact_path: Path,
    handoff_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    input_references: Mapping[str, Any],
    input_results: Mapping[str, Any],
    input_classification: str,
    classification: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer10_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "conv_gdn_artifact_path": str(conv_gdn_artifact_path),
        "handoff_artifact_path": str(handoff_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "input_references": dict(input_references),
        "input_results": dict(input_results),
        "input_classification": str(input_classification),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
