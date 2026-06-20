#!/usr/bin/env python3
"""Audit layer-7 post-attention residual/post-norm under the BF16 contract."""

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
from scripts.llamacpp_mtp_audit_layer7_attn_output_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER7_ATTN_OUTPUT,
)

DEFAULT_HANDOFF = Path(
    "benchmarks/results/mtp-gguf-iter375-layer7-bf16-handoff.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter382-layer7-post-attn-residual-oracle.json"
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
        "--attn-output-artifact",
        type=Path,
        default=DEFAULT_LAYER7_ATTN_OUTPUT,
    )
    parser.add_argument("--handoff-artifact", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=7)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--residual-atol", type=float, default=0.0)
    parser.add_argument("--post-norm-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=382)
    args = parser.parse_args()

    artifact = audit_layer7_post_attn_residual_oracle(
        attn_output_artifact_path=args.attn_output_artifact,
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


def audit_layer7_post_attn_residual_oracle(
    *,
    attn_output_artifact_path: Path,
    handoff_artifact_path: Path = DEFAULT_HANDOFF,
    model_path: Path | None = None,
    layer_id: int = 7,
    max_sequence_length: int | None = None,
    residual_atol: float = 0.0,
    post_norm_atol: float = 2.5e-4,
    iteration: int = 382,
    layer_capture_fn: LayerCaptureFn | None = None,
    norm_weight_loader: NormWeightLoader | None = None,
    input_reference_loader: InputReferenceLoader | None = None,
) -> dict[str, Any]:
    attn_output = json.loads(attn_output_artifact_path.read_text())
    handoff = json.loads(handoff_artifact_path.read_text())
    validate_layer7_attn_output_artifact(attn_output, expected_layer_id=int(layer_id))
    validate_layer7_handoff_artifact(handoff, expected_layer_id=int(layer_id))
    validate_source_alignment(attn_output=attn_output, handoff=handoff)
    resolved_model = Path(model_path or attn_output["model"])
    position = int(attn_output["position"])
    prompt_tokens = tuple(int(token) for token in attn_output["prompt_tokens"])
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
            attn_output_artifact_path=attn_output_artifact_path,
            handoff_artifact_path=handoff_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    metadata_classification = classify_capture_metadata(capture, layer_id=int(layer_id))
    reference_loader = input_reference_loader or load_layer7_input_references
    input_references = reference_loader(
        attn_output_artifact_path,
        attn_output,
        handoff_artifact_path,
        handoff,
    )
    if metadata_classification is not None:
        return blocked_artifact(
            attn_output_artifact_path=attn_output_artifact_path,
            handoff_artifact_path=handoff_artifact_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            input_references=input_references,
            input_results={},
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
    input_classification = classify_layer7_inputs(input_results)
    classification = classify_layer7_post_attn(input_classification, oracle_results)
    return {
        "schema": 1,
        "kind": "layer7_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "attn_output_artifact_path": str(attn_output_artifact_path),
        "handoff_artifact_path": str(handoff_artifact_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "verified layer7 hidden_in + verified layer7 attn_out",
            "handoff_source": handoff.get("classification"),
            "attn_output_source": attn_output.get("classification"),
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


def validate_layer7_attn_output_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if artifact.get("status") != "ready":
        raise ValueError("layer7 attn-output artifact must be ready")
    if artifact.get("classification") not in {
        "layer7_attn_output_projection_matches_bf16_oracle_exactly",
        "layer7_attn_output_projection_matches_bf16_oracle_within_rounding",
    }:
        raise ValueError("layer7 attn-output artifact must be validated")
    if int(artifact.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer7 attn-output layer_id does not match requested layer")
    if artifact.get("next_action") != (
        "audit_layer7_post_attn_residual_or_moe_boundary"
    ):
        raise ValueError("layer7 attn-output artifact must point to post-attn audit")
    result = artifact.get("projection_result") or {}
    if not str(result.get("classification", "")).startswith(
        "full_attention_attn_output_matches_bf16_oracle"
    ):
        raise ValueError("layer7 attn-output artifact must establish attn_out")
    fields = (artifact.get("hipengine_capture") or {}).get("field_summaries") or {}
    if "attn_out_f32" not in fields:
        raise ValueError("layer7 attn-output artifact missing attn_out summary")


def validate_layer7_handoff_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if artifact.get("status") != "ready":
        raise ValueError("layer7 handoff artifact must be ready")
    if artifact.get("classification") != (
        "layer7_hidden_in_matches_layer6_layer_out_exactly"
    ):
        raise ValueError("layer7 handoff artifact must be exact")
    if int(artifact.get("target_layer", -1)) != int(expected_layer_id):
        raise ValueError("layer7 handoff target_layer does not match requested layer")
    fields = ((artifact.get("target_capture") or {}).get("fields") or {})
    if "hidden_in_f32" not in fields:
        raise ValueError("layer7 handoff artifact missing hidden_in summary")


def validate_source_alignment(
    *,
    attn_output: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> None:
    for key in ("model", "position", "token_id", "prompt_tokens"):
        if attn_output.get(key) != handoff.get(key):
            raise ValueError(f"layer7 source artifact {key} mismatch")


def load_layer7_input_references(
    attn_output_artifact_path: Path,
    attn_output: Mapping[str, Any],
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
            "source": str(attn_output_artifact_path),
            "source_kind": attn_output.get("kind"),
            "source_classification": attn_output.get("classification"),
            "expected_sha256": attn_output["hipengine_capture"]["field_summaries"][
                "attn_out_f32"
            ]["sha256"],
        },
    }


def classify_capture_metadata(capture: Mapping[str, Any], *, layer_id: int) -> str | None:
    summary = capture.get("summary") or {}
    if int(summary.get("layer_id", -1)) != int(layer_id):
        return "layer7_post_attn_wrong_layer_capture"
    if str(summary.get("layer_type")) != "full_attention":
        return "layer7_post_attn_wrong_layer_type"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer7_post_attn_wrong_preceding_layer_count"
    return None


def classify_layer7_inputs(input_results: Mapping[str, Any]) -> str:
    classes = [
        input_results[field]["classification"]
        for field in ("hidden_in_f32", "attn_out_f32")
    ]
    if all(item == "post_attn_input_hash_matches_reference" for item in classes):
        return "layer7_post_attn_inputs_match_prior_artifacts"
    if any("mismatch" in item for item in classes):
        return "layer7_post_attn_inputs_mismatch_before_residual"
    return "layer7_post_attn_inputs_unavailable"


def classify_layer7_post_attn(
    input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if input_classification != "layer7_post_attn_inputs_match_prior_artifacts":
        return "layer7_post_attn_residual_blocked_input_mismatch"
    classes = [oracle_results[name]["classification"] for name in POST_ATTN_FIELDS]
    if all(item == "post_attn_field_matches_oracle_exactly" for item in classes):
        return "layer7_post_attn_residual_matches_oracle_exactly"
    if all(item.startswith("post_attn_field_matches_oracle") for item in classes):
        return "layer7_post_attn_residual_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer7_post_attn_residual_mismatch_after_oracle"
    return "layer7_post_attn_residual_oracle_unavailable"


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
        "layer7_post_attn_residual_matches_oracle_exactly",
        "layer7_post_attn_residual_matches_oracle_within_tolerance",
    }:
        return "audit_layer7_moe_router_from_post_norm"
    if classification == "layer7_post_attn_residual_blocked_input_mismatch":
        return "reconcile_layer7_hidden_or_attn_out_before_post_attn"
    if classification in {
        "layer7_post_attn_wrong_layer_capture",
        "layer7_post_attn_wrong_layer_type",
        "layer7_post_attn_wrong_preceding_layer_count",
    }:
        return "inspect_layer7_post_attn_capture_metadata"
    if classification == "layer7_post_attn_residual_mismatch_after_oracle":
        return "inspect_layer7_post_attn_residual_or_norm_kernel"
    return "rerun_layer7_post_attn_residual_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: artifact["oracle_results"][name]["classification"]
        for name in POST_ATTN_FIELDS
        if name in artifact.get("oracle_results", {})
    }


def unavailable_artifact(
    *,
    attn_output_artifact_path: Path,
    handoff_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer7_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer7_post_attn_residual_oracle_unavailable",
        "attn_output_artifact_path": str(attn_output_artifact_path),
        "handoff_artifact_path": str(handoff_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_layer7_post_attn_residual_oracle_on_rocm_host",
    }


def blocked_artifact(
    *,
    attn_output_artifact_path: Path,
    handoff_artifact_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    input_references: Mapping[str, Any],
    input_results: Mapping[str, Any],
    classification: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer7_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "attn_output_artifact_path": str(attn_output_artifact_path),
        "handoff_artifact_path": str(handoff_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "input_references": dict(input_references),
        "input_results": dict(input_results),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
