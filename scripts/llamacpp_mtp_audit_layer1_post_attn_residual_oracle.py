#!/usr/bin/env python3
"""Audit layer-1 post-attention residual/post-norm under the BF16 contract."""

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

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle import (  # noqa: E402
    add_rmsnorm_bf16_oracle,
    classify_field_delta,
    load_post_attention_norm_weight,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer1_conv_gdn_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER1_CONV_GDN,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter335-layer1-post-attn-residual-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
NormWeightLoader = Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]]
InputReferenceLoader = Callable[[Path, Mapping[str, Any]], dict[str, Any]]

POST_ATTN_FIELDS = ("residual_f32", "post_norm_f32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conv-gdn-artifact", type=Path, default=DEFAULT_LAYER1_CONV_GDN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--residual-atol", type=float, default=0.0)
    parser.add_argument("--post-norm-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=335)
    args = parser.parse_args()

    artifact = audit_layer1_post_attn_residual_oracle(
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
                "input_classification": artifact["input_classification"],
                "field_classifications": {
                    name: artifact["oracle_results"][name]["classification"]
                    for name in POST_ATTN_FIELDS
                    if name in artifact.get("oracle_results", {})
                },
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer1_post_attn_residual_oracle(
    *,
    conv_gdn_artifact_path: Path,
    model_path: Path | None = None,
    layer_id: int = 1,
    max_sequence_length: int | None = None,
    residual_atol: float = 0.0,
    post_norm_atol: float = 2.5e-4,
    iteration: int = 335,
    layer_capture_fn: LayerCaptureFn | None = None,
    norm_weight_loader: NormWeightLoader | None = None,
    input_reference_loader: InputReferenceLoader | None = None,
) -> dict[str, Any]:
    conv_gdn = json.loads(conv_gdn_artifact_path.read_text())
    validate_layer1_conv_gdn(conv_gdn, expected_layer_id=int(layer_id))
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
    if int((capture.get("summary") or {}).get("preceding_layer_count", -1)) != int(layer_id):
        raise ValueError("layer-1 post-attn capture must run preceding layers")
    reference_loader = input_reference_loader or load_input_references
    input_references = reference_loader(conv_gdn_artifact_path, conv_gdn)
    input_results = compare_input_hashes(capture=capture, input_references=input_references)
    selected_norm_loader = norm_weight_loader or load_post_attention_norm_weight
    post_norm_weight, eps, norm_metadata = selected_norm_loader(resolved_model, int(layer_id))
    oracle_results = build_layer1_post_attn_results(
        capture=capture,
        post_norm_weight=post_norm_weight,
        eps=float(eps),
        residual_atol=float(residual_atol),
        post_norm_atol=float(post_norm_atol),
    )
    input_classification = classify_inputs(input_results)
    classification = classify_layer1_post_attn(input_classification, oracle_results)
    return {
        "schema": 1,
        "kind": "layer1_post_attn_residual_oracle",
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
            "source": "verified layer1 hidden_in + verified layer1 attn_out",
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


def validate_layer1_conv_gdn(conv_gdn: Mapping[str, Any], *, expected_layer_id: int) -> None:
    if conv_gdn.get("status") != "ready":
        raise ValueError("layer1 conv/GDN artifact must be ready")
    if not str(conv_gdn.get("classification", "")).startswith(
        "layer1_warm_conv_gdn_matches_oracle"
    ):
        raise ValueError("layer1 conv/GDN artifact must have matched")
    if int(conv_gdn.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer1 conv/GDN layer_id does not match requested layer")
    if conv_gdn.get("next_action") != "audit_layer1_post_attn_residual_or_moe_boundary":
        raise ValueError("layer1 conv/GDN artifact must point to post-attn audit")
    attn_result = (conv_gdn.get("oracle_results") or {}).get("attn_out_f32") or {}
    if not str(attn_result.get("classification", "")).startswith("warm_field_matches_oracle"):
        raise ValueError("layer1 conv/GDN artifact must establish attn_out")


def load_input_references(
    conv_gdn_artifact_path: Path,
    conv_gdn: Mapping[str, Any],
) -> dict[str, Any]:
    projection_path = Path(conv_gdn["layer1_projection_path"])
    if not projection_path.is_absolute():
        projection_path = conv_gdn_artifact_path.parent.parent / projection_path
        if not projection_path.exists():
            projection_path = Path(conv_gdn["layer1_projection_path"])
    projection = json.loads(projection_path.read_text())
    attn_norm_path = Path(projection["layer1_attn_norm_path"])
    if not attn_norm_path.is_absolute():
        attn_norm_path = projection_path.parent.parent / attn_norm_path
        if not attn_norm_path.exists():
            attn_norm_path = Path(projection["layer1_attn_norm_path"])
    attn_norm = json.loads(attn_norm_path.read_text())
    handoff_path = Path(attn_norm["layer1_handoff_path"])
    if not handoff_path.is_absolute():
        handoff_path = attn_norm_path.parent.parent / handoff_path
        if not handoff_path.exists():
            handoff_path = Path(attn_norm["layer1_handoff_path"])
    handoff = json.loads(handoff_path.read_text())
    return {
        "hidden_in_f32": {
            "source": str(handoff_path),
            "source_kind": handoff.get("kind"),
            "source_classification": handoff.get("classification"),
            "expected_sha256": handoff["target_capture"]["fields"]["hidden_in_f32"]["sha256"],
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


def compare_input_hashes(
    *,
    capture: Mapping[str, Any],
    input_references: Mapping[str, Any],
) -> dict[str, Any]:
    fields = capture["fields"]
    results: dict[str, Any] = {}
    for field in ("hidden_in_f32", "attn_out_f32"):
        values = np.asarray(fields[field], dtype=np.float32)
        actual_sha = sha256_float32(values)
        expected_sha = str(input_references[field]["expected_sha256"])
        exact = actual_sha == expected_sha
        results[field] = {
            "field": field,
            "reference_source": input_references[field]["source"],
            "reference_classification": input_references[field]["source_classification"],
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "exact_hash_match": exact,
            "hipengine_summary": summarize_array(values),
            "classification": (
                "post_attn_input_hash_matches_reference"
                if exact
                else "post_attn_input_hash_mismatch_before_residual"
            ),
        }
    return results


def build_layer1_post_attn_results(
    *,
    capture: Mapping[str, Any],
    post_norm_weight: np.ndarray,
    eps: float,
    residual_atol: float,
    post_norm_atol: float,
) -> dict[str, Any]:
    fields = capture["fields"]
    residual, post_norm = add_rmsnorm_bf16_oracle(
        np.asarray(fields["hidden_in_f32"], dtype=np.float32),
        np.asarray(fields["attn_out_f32"], dtype=np.float32),
        np.asarray(post_norm_weight, dtype=np.float32),
        eps=float(eps),
    )
    residual_delta = delta_summary(
        residual,
        np.asarray(fields["residual_f32"], dtype=np.float32),
    )
    post_norm_delta = delta_summary(
        post_norm,
        np.asarray(fields["post_norm_f32"], dtype=np.float32),
    )
    return {
        "residual_f32": {
            "field": "residual_f32",
            "oracle": "BF16(hidden_in + attn_out)",
            "oracle_summary": summarize_array(residual),
            "hipengine_summary": summarize_array(fields["residual_f32"]),
            "delta_oracle_vs_hip": residual_delta,
            "near_atol": float(residual_atol),
            "classification": classify_field_delta(
                residual_delta,
                near_atol=float(residual_atol),
            ),
        },
        "post_norm_f32": {
            "field": "post_norm_f32",
            "oracle": "BF16(RMSNorm(hidden_in + attn_out, post_attention_norm.weight))",
            "oracle_summary": summarize_array(post_norm),
            "hipengine_summary": summarize_array(fields["post_norm_f32"]),
            "delta_oracle_vs_hip": post_norm_delta,
            "near_atol": float(post_norm_atol),
            "classification": classify_field_delta(
                post_norm_delta,
                near_atol=float(post_norm_atol),
            ),
        },
    }


def capture_layer1_full_layer(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    layer_id: int,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        for index, token_id in enumerate(prompt_tokens[:position]):
            session.step(int(token_id), position=index, return_logits=False)
        capture = session.capture_attention_layer(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=int(layer_id),
            run_preceding_layers=True,
        )
    fields = {
        "hidden_in_f32": np.asarray(capture.hidden_in_f32, dtype=np.float32),
        "attn_out_f32": np.asarray(capture.attn_out_f32, dtype=np.float32),
        "residual_f32": np.asarray(capture.residual_f32, dtype=np.float32),
        "post_norm_f32": np.asarray(capture.post_norm_f32, dtype=np.float32),
        "ffn_or_moe_down_f32": np.asarray(capture.ffn_or_moe_down_f32, dtype=np.float32),
        "layer_out_f32": np.asarray(capture.layer_out_f32, dtype=np.float32),
    }
    if capture.moe_shared_out_f32 is not None:
        fields["moe_shared_out_f32"] = np.asarray(
            capture.moe_shared_out_f32,
            dtype=np.float32,
        )
    if capture.moe_routing_weights_f32 is not None:
        fields["moe_routing_weights_f32"] = np.asarray(
            capture.moe_routing_weights_f32,
            dtype=np.float32,
        )
    if capture.moe_shared_gate_f32 is not None:
        fields["moe_shared_gate_f32"] = np.asarray(capture.moe_shared_gate_f32, dtype=np.float32)
    if capture.moe_selected_experts_i64 is not None:
        fields["moe_selected_experts_i64"] = np.asarray(
            capture.moe_selected_experts_i64,
            dtype=np.int64,
        )
    return {"status": "captured", "summary": capture.as_summary_dict(), "fields": fields}


def classify_inputs(input_results: Mapping[str, Any]) -> str:
    classes = [
        input_results[field]["classification"]
        for field in ("hidden_in_f32", "attn_out_f32")
    ]
    if all(item == "post_attn_input_hash_matches_reference" for item in classes):
        return "layer1_post_attn_inputs_match_prior_artifacts"
    if any("mismatch" in item for item in classes):
        return "layer1_post_attn_inputs_mismatch_before_residual"
    return "layer1_post_attn_inputs_unavailable"


def classify_layer1_post_attn(
    input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if input_classification != "layer1_post_attn_inputs_match_prior_artifacts":
        return "layer1_post_attn_residual_blocked_input_mismatch"
    classes = [oracle_results[name]["classification"] for name in POST_ATTN_FIELDS]
    if all(item == "post_attn_field_matches_oracle_exactly" for item in classes):
        return "layer1_post_attn_residual_matches_oracle_exactly"
    if all(item.startswith("post_attn_field_matches_oracle") for item in classes):
        return "layer1_post_attn_residual_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer1_post_attn_residual_mismatch_after_oracle"
    return "layer1_post_attn_residual_oracle_unavailable"


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
        "layer1_post_attn_residual_matches_oracle_exactly",
        "layer1_post_attn_residual_matches_oracle_within_tolerance",
    }:
        return "audit_layer1_moe_router_from_post_norm"
    if classification == "layer1_post_attn_residual_blocked_input_mismatch":
        return "reconcile_layer1_hidden_or_attn_out_before_post_attn"
    if classification == "layer1_post_attn_residual_mismatch_after_oracle":
        return "inspect_layer1_post_attn_residual_or_norm_kernel"
    return "rerun_layer1_post_attn_residual_oracle_on_rocm_host"


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
        "kind": "layer1_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer1_post_attn_residual_oracle_unavailable",
        "conv_gdn_artifact_path": str(conv_gdn_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "target_position": int(target_position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_layer1_post_attn_residual_oracle_on_rocm_host",
    }


if __name__ == "__main__":
    main()
