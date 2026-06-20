#!/usr/bin/env python3
"""Audit layer-10 attention RMSNorm under the resident-BF16 contract."""

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
    bf16_roundtrip_array,
    delta_summary,
    load_layer0_attn_norm_weight,
    rmsnorm_f32,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer1_attn_norm_oracle import (  # noqa: E402
    capture_layer_attn_norm,
    summarize_capture,
)
from scripts.llamacpp_mtp_layer10_bf16_handoff_audit import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER10_HANDOFF,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter400-layer10-attn-norm-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
WeightLoaderFn = Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer10-handoff",
        type=Path,
        default=DEFAULT_LAYER10_HANDOFF,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--attn-norm-atol", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=400)
    args = parser.parse_args()

    artifact = audit_layer10_attn_norm_oracle(
        layer10_handoff_path=args.layer10_handoff,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        attn_norm_atol=args.attn_norm_atol,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact["classification"],
                "layer_id": artifact["layer_id"],
                "input_classification": artifact.get("input_classification"),
                "attn_norm_max_abs": artifact.get("attn_norm_delta", {}).get(
                    "max_abs_diff"
                ),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer10_attn_norm_oracle(
    *,
    layer10_handoff_path: Path,
    model_path: Path | None = None,
    layer_id: int = 10,
    max_sequence_length: int | None = None,
    attn_norm_atol: float = 0.0,
    iteration: int = 400,
    layer_capture_fn: LayerCaptureFn | None = None,
    weight_loader: WeightLoaderFn | None = None,
) -> dict[str, Any]:
    handoff = json.loads(layer10_handoff_path.read_text())
    validate_layer10_handoff(handoff, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or handoff["model"])
    prompt_tokens = tuple(int(token) for token in handoff["prompt_tokens"])
    position = int(handoff["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = layer_capture_fn or capture_layer_attn_norm
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        classification = "layer10_attn_norm_oracle_unavailable"
        return {
            "schema": 1,
            "kind": "layer10_attn_norm_oracle",
            "date": "2026-06-20",
            "loop": "mtp-gguf/run-20260615-103738",
            "iteration": int(iteration),
            "status": str(capture.get("status", "unavailable")),
            "classification": classification,
            "layer10_handoff_path": str(layer10_handoff_path),
            "model": str(resolved_model),
            "layer_id": int(layer_id),
            "position": position,
            "token_id": token_id,
            "prompt_tokens": list(prompt_tokens),
            "hipengine_capture": summarize_capture(capture),
            "external_checkout_modified": False,
            "next_action": next_action(classification),
        }
    input_result = compare_hidden_input(capture=capture, handoff=handoff)
    selected_loader = weight_loader or load_layer0_attn_norm_weight
    weight, eps, weight_metadata = selected_loader(resolved_model, int(layer_id))
    hidden_in = np.asarray(capture["fields"]["hidden_in_f32"], dtype=np.float32)
    expected = bf16_roundtrip_array(rmsnorm_f32(hidden_in, weight, float(eps)))
    actual = np.asarray(capture["fields"]["attn_norm_f32"], dtype=np.float32)
    attn_norm_delta = delta_summary(expected, actual)
    input_classification = classify_input(input_result)
    classification = classify_layer10_attn_norm(
        input_classification,
        attn_norm_delta,
        capture=capture,
        layer_id=int(layer_id),
        attn_norm_atol=float(attn_norm_atol),
    )
    return {
        "schema": 1,
        "kind": "layer10_attn_norm_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer10_handoff_path": str(layer10_handoff_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer10 hidden_in_f32 from exact BF16 handoff artifact",
            "formula": "BF16(RMSNorm(hidden_in_f32, attn_norm.weight_f32, eps_model))",
            "handoff_classification": handoff.get("classification"),
            "layer_type": "linear_attention",
            "expectation": (
                "exact resident-BF16 attn_norm before layer10 "
                "projection audits"
            ),
        },
        "input_result": input_result,
        "input_classification": input_classification,
        "weight": weight_metadata,
        "hipengine_capture": summarize_capture(capture),
        "oracle_attn_norm": {
            "summary": summarize_array(expected),
            "sha256": sha256_float32(expected),
        },
        "attn_norm_delta": attn_norm_delta,
        "attn_norm_atol": float(attn_norm_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer10_handoff(
    handoff: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if handoff.get("status") != "ready":
        raise ValueError("layer10 handoff artifact must be ready")
    if (
        handoff.get("classification")
        != "layer10_hidden_in_matches_layer9_layer_out_exactly"
    ):
        raise ValueError("layer10 handoff artifact must be exact")
    if int(handoff.get("target_layer", -1)) != int(expected_layer_id):
        raise ValueError(
            "layer10 handoff target_layer does not match requested layer"
        )
    if (handoff.get("source_reference") or {}).get("exact_hash_match") is not True:
        raise ValueError("layer10 handoff source reference must be exact")
    target_summary = (handoff.get("target_capture") or {}).get("summary") or {}
    if str(target_summary.get("layer_type")) != "linear_attention":
        raise ValueError("layer10 handoff target must be a linear_attention layer")
    delta = handoff.get("handoff_delta") or {}
    if (
        delta.get("exact_match") is not True
        or float(delta.get("max_abs_diff", 1.0)) != 0.0
    ):
        raise ValueError("layer10 handoff delta must be exact")
    expected_next = "audit_layer10_attn_norm_under_bf16_contract_or_mtp_boundary"
    if handoff.get("next_action") != expected_next:
        raise ValueError("layer10 handoff artifact must point to attn_norm audit")


def compare_hidden_input(
    *,
    capture: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    hidden = np.asarray(capture["fields"]["hidden_in_f32"], dtype=np.float32)
    actual_sha = sha256_float32(hidden)
    expected_sha = str(
        handoff["target_capture"]["fields"]["hidden_in_f32"]["sha256"]
    )
    exact = actual_sha == expected_sha
    return {
        "field": "hidden_in_f32",
        "reference_source": "layer10_bf16_handoff_audit",
        "reference_classification": handoff.get("classification"),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "exact_hash_match": exact,
        "summary": summarize_array(hidden),
        "classification": (
            "layer10_attn_norm_input_matches_handoff_artifact"
            if exact
            else "layer10_attn_norm_input_mismatch_before_norm"
        ),
    }


def classify_input(input_result: Mapping[str, Any]) -> str:
    if input_result.get("exact_hash_match"):
        return "layer10_attn_norm_input_matches_handoff_artifact"
    return "layer10_attn_norm_input_mismatch_before_norm"


def classify_layer10_attn_norm(
    input_classification: str,
    delta: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    layer_id: int,
    attn_norm_atol: float,
) -> str:
    summary = capture.get("summary") or {}
    if input_classification != "layer10_attn_norm_input_matches_handoff_artifact":
        return "layer10_attn_norm_blocked_input_mismatch"
    if int(summary.get("layer_id", -1)) != int(layer_id):
        return "layer10_attn_norm_wrong_layer_capture"
    if str(summary.get("layer_type")) != "linear_attention":
        return "layer10_attn_norm_wrong_layer_type"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer10_attn_norm_wrong_preceding_layer_count"
    if not delta.get("available") or not delta.get("shape_match"):
        return "layer10_attn_norm_oracle_unavailable"
    if delta.get("exact_match"):
        return "layer10_attn_norm_matches_bf16_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(attn_norm_atol):
        return "layer10_attn_norm_matches_bf16_oracle_within_tolerance"
    return "layer10_attn_norm_mismatch_after_bf16_oracle"


def status_from_classification(classification: str) -> str:
    if "unavailable" in classification:
        return "unavailable"
    if "blocked" in classification:
        return "blocked"
    if "wrong" in classification or "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer10_attn_norm_matches_bf16_oracle_exactly",
        "layer10_attn_norm_matches_bf16_oracle_within_tolerance",
    }:
        return "audit_layer10_projection_or_conv_gdn_under_bf16_contract"
    if classification == "layer10_attn_norm_blocked_input_mismatch":
        return "reconcile_layer10_hidden_in_before_attn_norm"
    if classification == "layer10_attn_norm_wrong_preceding_layer_count":
        return "inspect_layer10_attn_norm_capture_preceding_layers"
    if classification in {
        "layer10_attn_norm_wrong_layer_capture",
        "layer10_attn_norm_wrong_layer_type",
    }:
        return "inspect_layer10_attn_norm_capture_layer_metadata"
    if classification == "layer10_attn_norm_mismatch_after_bf16_oracle":
        return "inspect_layer10_attn_norm_weight_or_rmsnorm_semantics"
    return "rerun_layer10_attn_norm_oracle_on_rocm_host"


if __name__ == "__main__":
    main()
