#!/usr/bin/env python3
"""Audit layer-12 linear-attention projection outputs under the BF16 contract."""

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
from scripts.llamacpp_mtp_audit_layer1_projection_oracle import (  # noqa: E402
    PROJECTION_SPECS,
    ProjectionWeights,
    build_projection_results,
    capture_layer1_projection_boundary,
    load_layer1_projection_weights,
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer12_attn_norm_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER12_ATTN_NORM,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter417-layer12-projection-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

BoundaryCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
ProjectionWeightLoader = Callable[[Path, int], ProjectionWeights]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer12-attn-norm",
        type=Path,
        default=DEFAULT_LAYER12_ATTN_NORM,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=12)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--near-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=417)
    args = parser.parse_args()

    artifact = audit_layer12_projection_oracle(
        layer12_attn_norm_path=args.layer12_attn_norm,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        near_atol=args.near_atol,
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
                "field_classifications": field_classifications(artifact),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer12_projection_oracle(
    *,
    layer12_attn_norm_path: Path,
    model_path: Path | None = None,
    layer_id: int = 12,
    max_sequence_length: int | None = None,
    near_atol: float = 2.5e-4,
    iteration: int = 417,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
    projection_weight_loader: ProjectionWeightLoader | None = None,
) -> dict[str, Any]:
    attn_norm = json.loads(layer12_attn_norm_path.read_text())
    validate_layer12_attn_norm(attn_norm, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or attn_norm["model"])
    prompt_tokens = tuple(int(token) for token in attn_norm["prompt_tokens"])
    position = int(attn_norm["position"])
    token_id = int(prompt_tokens[position])
    capture_fn = boundary_capture_fn or capture_layer1_projection_boundary
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(layer_id),
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            layer12_attn_norm_path=layer12_attn_norm_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    input_result = compare_attn_norm_input(capture=capture, attn_norm=attn_norm)
    input_classification = classify_input(input_result)
    preflight = classify_projection_preflight(
        input_classification,
        capture=capture,
        layer_id=int(layer_id),
    )
    if preflight is not None:
        return blocked_artifact(
            layer12_attn_norm_path=layer12_attn_norm_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            input_result=input_result,
            input_classification=input_classification,
            classification=preflight,
            iteration=iteration,
        )
    selected_loader = projection_weight_loader or load_layer1_projection_weights
    projection_weights = selected_loader(resolved_model, int(layer_id))
    projection_results = build_projection_results(
        attn_norm_values=np.asarray(
            capture["fields"]["attn_norm_f32"],
            dtype=np.float32,
        ),
        capture=capture,
        projection_weights=projection_weights,
        near_atol=float(near_atol),
    )
    classification = classify_layer12_projection_audit(
        input_classification,
        projection_results,
        capture=capture,
        layer_id=int(layer_id),
    )
    return {
        "schema": 1,
        "kind": "layer12_bf16_projection_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer12_attn_norm_path": str(layer12_attn_norm_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer12 attn_norm_f32 from exact resident-BF16 RMSNorm oracle",
            "formula": "BF16(project_f32(attn_norm_f32, GGUF projection weight))",
            "attn_norm_classification": attn_norm.get("classification"),
            "layer_type": "linear_attention",
            "captured_with_run_preceding_layers": True,
            "input_hash_checked": True,
        },
        "input_result": input_result,
        "input_classification": input_classification,
        "weights": {
            name: metadata for name, (_value, metadata) in projection_weights.items()
        },
        "hipengine_capture": summarize_capture(capture),
        "projection_results": projection_results,
        "near_atol": float(near_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer12_attn_norm(
    artifact: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if artifact.get("status") != "ready":
        raise ValueError("layer12 attn_norm artifact must be ready")
    if artifact.get("classification") != (
        "layer12_attn_norm_matches_bf16_oracle_exactly"
    ):
        raise ValueError("layer12 attn_norm artifact must be exact")
    if int(artifact.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer12 attn_norm layer_id does not match requested layer")
    delta = artifact.get("attn_norm_delta") or {}
    if (
        delta.get("exact_match") is not True
        or float(delta.get("max_abs_diff", 1.0)) != 0.0
    ):
        raise ValueError("layer12 attn_norm delta must be exact")
    expected_next = "audit_layer12_projection_or_conv_gdn_under_bf16_contract"
    if artifact.get("next_action") != expected_next:
        raise ValueError("layer12 attn_norm artifact must point to projection audit")


def compare_attn_norm_input(
    *,
    capture: Mapping[str, Any],
    attn_norm: Mapping[str, Any],
) -> dict[str, Any]:
    values = np.asarray(capture["fields"]["attn_norm_f32"], dtype=np.float32)
    actual_sha = sha256_float32(values)
    expected_sha = str(
        attn_norm["hipengine_capture"]["fields"]["attn_norm_f32"]["sha256"]
    )
    exact = actual_sha == expected_sha
    return {
        "field": "attn_norm_f32",
        "reference_source": "layer12_attn_norm_oracle",
        "reference_classification": attn_norm.get("classification"),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "exact_hash_match": exact,
        "summary": summarize_array(values),
        "classification": (
            "layer12_projection_input_matches_attn_norm_artifact"
            if exact
            else "layer12_projection_input_mismatch_before_projection"
        ),
    }


def classify_input(input_result: Mapping[str, Any]) -> str:
    if input_result.get("exact_hash_match"):
        return "layer12_projection_input_matches_attn_norm_artifact"
    return "layer12_projection_input_mismatch_before_projection"


def classify_projection_preflight(
    input_classification: str,
    *,
    capture: Mapping[str, Any],
    layer_id: int,
) -> str | None:
    summary = capture.get("summary") or {}
    if input_classification != (
        "layer12_projection_input_matches_attn_norm_artifact"
    ):
        return "layer12_projection_blocked_attn_norm_input_mismatch"
    if int(summary.get("layer_id", -1)) != int(layer_id):
        return "layer12_projection_wrong_layer_capture"
    if str(summary.get("layer_type")) != "linear_attention":
        return "layer12_projection_wrong_layer_type"
    if int(summary.get("preceding_layer_count", -1)) != int(layer_id):
        return "layer12_projection_wrong_preceding_layer_count"
    return None


def classify_layer12_projection_audit(
    input_classification: str,
    projection_results: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    layer_id: int,
) -> str:
    preflight = classify_projection_preflight(
        input_classification,
        capture=capture,
        layer_id=layer_id,
    )
    if preflight is not None:
        return preflight
    classes = [
        projection_results[name]["classification"]
        for name in PROJECTION_SPECS
    ]
    if all(item == "projection_matches_bf16_oracle_exactly" for item in classes):
        return "layer12_projections_match_bf16_oracle_exactly"
    if all(item.startswith("projection_matches_bf16_oracle") for item in classes):
        return "layer12_projections_match_bf16_oracle_within_rounding"
    if any("mismatch" in item for item in classes):
        return "layer12_projection_mismatch_after_bf16_oracle"
    return "layer12_projection_oracle_unavailable"


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
        "layer12_projections_match_bf16_oracle_exactly",
        "layer12_projections_match_bf16_oracle_within_rounding",
    }:
        return "audit_layer12_conv_gdn_under_bf16_contract"
    if classification == "layer12_projection_blocked_attn_norm_input_mismatch":
        return "reconcile_layer12_attn_norm_before_projection"
    if classification == "layer12_projection_wrong_preceding_layer_count":
        return "inspect_layer12_projection_capture_preceding_layers"
    if classification in {
        "layer12_projection_wrong_layer_capture",
        "layer12_projection_wrong_layer_type",
    }:
        return "inspect_layer12_projection_capture_layer_metadata"
    if classification == "layer12_projection_mismatch_after_bf16_oracle":
        return "inspect_layer12_projection_weight_or_kernel_quantization"
    return "rerun_layer12_projection_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: artifact["projection_results"][name]["classification"]
        for name in PROJECTION_SPECS
        if name in artifact.get("projection_results", {})
    }


def unavailable_artifact(
    *,
    layer12_attn_norm_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer12_projection_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer12_bf16_projection_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": classification,
        "layer12_attn_norm_path": str(layer12_attn_norm_path),
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
    layer12_attn_norm_path: Path,
    model_path: Path,
    layer_id: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    capture: Mapping[str, Any],
    input_result: Mapping[str, Any],
    input_classification: str,
    classification: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer12_bf16_projection_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer12_attn_norm_path": str(layer12_attn_norm_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "input_result": dict(input_result),
        "input_classification": str(input_classification),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
