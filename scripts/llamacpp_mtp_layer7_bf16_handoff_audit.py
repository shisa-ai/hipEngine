#!/usr/bin/env python3
"""Audit the BF16 handoff from layer-6 output into layer-7 hidden input."""

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

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import delta_summary  # noqa: E402
from scripts.llamacpp_mtp_layer1_bf16_handoff_audit import (  # noqa: E402
    capture_hipengine_attention_layer,
    summarize_capture,
)

DEFAULT_LAYER6_EXPERTS = Path(
    "benchmarks/results/mtp-gguf-iter374-layer6-moe-expert-outputs-oracle.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter375-layer7-bf16-handoff.json")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, bool, int | None],
    dict[str, Any],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer6-experts", type=Path, default=DEFAULT_LAYER6_EXPERTS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-layer", type=int, default=6)
    parser.add_argument("--target-layer", type=int, default=7)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--handoff-atol", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=375)
    args = parser.parse_args()

    artifact = audit_layer7_bf16_handoff(
        layer6_experts_path=args.layer6_experts,
        model_path=args.model,
        source_layer=args.source_layer,
        target_layer=args.target_layer,
        max_sequence_length=args.max_sequence_length,
        handoff_atol=args.handoff_atol,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact["classification"],
                "source_layer": artifact["source_layer"],
                "target_layer": artifact["target_layer"],
                "handoff_max_abs": artifact.get("handoff_delta", {}).get(
                    "max_abs_diff"
                ),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer7_bf16_handoff(
    *,
    layer6_experts_path: Path,
    model_path: Path | None = None,
    source_layer: int = 6,
    target_layer: int = 7,
    max_sequence_length: int | None = None,
    handoff_atol: float = 0.0,
    iteration: int = 375,
    layer_capture_fn: LayerCaptureFn | None = None,
) -> dict[str, Any]:
    layer6 = json.loads(layer6_experts_path.read_text())
    validate_layer6_expert_outputs(layer6, expected_layer_id=int(source_layer))
    if target_layer != source_layer + 1:
        raise ValueError("target_layer must be the immediate successor of source_layer")
    resolved_model = Path(model_path or layer6["model"])
    prompt_tokens = tuple(int(token) for token in layer6["prompt_tokens"])
    position = int(layer6.get("position", layer6.get("target_position")))
    token_id = int(prompt_tokens[position])
    capture_fn = layer_capture_fn or capture_hipengine_attention_layer
    source_capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(source_layer),
        True,
        max_sequence_length,
    )
    target_capture = capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        int(target_layer),
        True,
        max_sequence_length,
    )
    if (
        source_capture.get("status") != "captured"
        or target_capture.get("status") != "captured"
    ):
        classification = "layer7_bf16_handoff_capture_unavailable"
        return unavailable_artifact(
            classification=classification,
            layer6_experts_path=layer6_experts_path,
            model_path=resolved_model,
            source_layer=int(source_layer),
            target_layer=int(target_layer),
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            source_capture=source_capture,
            target_capture=target_capture,
            iteration=iteration,
        )
    source_reference = compare_source_layer_out(
        source_capture=source_capture,
        layer6=layer6,
    )
    handoff_delta = delta_summary(
        np.asarray(source_capture["fields"]["layer_out_f32"], dtype=np.float32),
        np.asarray(target_capture["fields"]["hidden_in_f32"], dtype=np.float32),
    )
    classification = classify_handoff(
        handoff_delta,
        source_capture=source_capture,
        target_capture=target_capture,
        source_layer=int(source_layer),
        target_layer=int(target_layer),
        source_reference=source_reference,
        handoff_atol=float(handoff_atol),
    )
    return {
        "schema": 1,
        "kind": "layer7_bf16_handoff_audit",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer6_experts_path": str(layer6_experts_path),
        "model": str(resolved_model),
        "source_layer": int(source_layer),
        "target_layer": int(target_layer),
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "layer6 layer_out_f32 under resident-BF16 contract",
            "target": "layer7 hidden_in_f32 captured after run_preceding_layers=True",
            "expectation": "exact BF16 handoff before auditing layer7 sub-boundaries",
            "layer6_experts_classification": layer6.get("classification"),
        },
        "source_reference": source_reference,
        "source_capture": summarize_capture(source_capture),
        "target_capture": summarize_capture(target_capture),
        "handoff_delta": handoff_delta,
        "handoff_atol": float(handoff_atol),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer6_expert_outputs(
    layer6: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if layer6.get("status") != "ready":
        raise ValueError("layer6 expert-output artifact must be ready")
    if not str(layer6.get("classification", "")).startswith(
        "layer6_moe_expert_outputs_match_oracle"
    ):
        raise ValueError("layer6 expert-output artifact must have matched")
    if int(layer6.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError(
            "layer6 expert-output layer_id does not match requested source layer"
        )
    if layer6.get("next_action") != "audit_layer7_bf16_handoff_or_mtp_next_boundary":
        raise ValueError("layer6 expert-output artifact must point to layer7 handoff")
    delta = (layer6.get("oracle_results") or {}).get("layer_out_f32", {}).get(
        "delta_oracle_vs_hip",
        {},
    )
    if delta.get("exact_match") is not True:
        raise ValueError("layer6 final layer_out oracle must be exact")


def compare_source_layer_out(
    *,
    source_capture: Mapping[str, Any],
    layer6: Mapping[str, Any],
) -> dict[str, Any]:
    source_sha = summarize_capture(source_capture)["fields"]["layer_out_f32"][
        "sha256"
    ]
    expected_sha = str(
        layer6["hipengine_capture"]["field_summaries"]["layer_out_f32"]["sha256"]
    )
    exact = source_sha == expected_sha
    return {
        "field": "layer_out_f32",
        "reference_source": "layer6_moe_expert_outputs_oracle",
        "reference_classification": layer6.get("classification"),
        "expected_sha256": expected_sha,
        "actual_sha256": source_sha,
        "exact_hash_match": exact,
        "classification": (
            "layer7_handoff_source_matches_layer6_artifact"
            if exact
            else "layer7_handoff_source_mismatch_before_handoff"
        ),
    }


def classify_handoff(
    delta: Mapping[str, Any],
    *,
    source_capture: Mapping[str, Any],
    target_capture: Mapping[str, Any],
    source_layer: int,
    target_layer: int,
    source_reference: Mapping[str, Any],
    handoff_atol: float,
) -> str:
    source_summary = source_capture.get("summary") or {}
    target_summary = target_capture.get("summary") or {}
    if not source_reference.get("exact_hash_match"):
        return "layer7_bf16_handoff_blocked_source_mismatch"
    if int(source_summary.get("preceding_layer_count", -1)) != int(source_layer):
        return "layer7_bf16_handoff_wrong_source_preceding_layer_count"
    if int(target_summary.get("preceding_layer_count", -1)) != int(target_layer):
        return "layer7_bf16_handoff_wrong_target_preceding_layer_count"
    if not delta.get("available") or not delta.get("shape_match"):
        return "layer7_bf16_handoff_unavailable"
    if delta.get("exact_match"):
        return "layer7_hidden_in_matches_layer6_layer_out_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(handoff_atol):
        return "layer7_hidden_in_matches_layer6_layer_out_within_tolerance"
    return "layer7_hidden_in_mismatch_after_layer6_bf16_handoff"


def status_from_classification(classification: str) -> str:
    if "unavailable" in classification:
        return "unavailable"
    if "wrong" in classification or "mismatch" in classification:
        return "mismatched"
    if "blocked" in classification:
        return "blocked"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer7_hidden_in_matches_layer6_layer_out_exactly",
        "layer7_hidden_in_matches_layer6_layer_out_within_tolerance",
    }:
        return "audit_layer7_attn_norm_under_bf16_contract_or_mtp_boundary"
    if classification == "layer7_bf16_handoff_blocked_source_mismatch":
        return "reconcile_layer6_layer_out_before_layer7_handoff"
    if "wrong" in classification:
        return "inspect_layer7_handoff_capture_preceding_layers"
    if classification == "layer7_hidden_in_mismatch_after_layer6_bf16_handoff":
        return "inspect_layer6_to_layer7_hidden_buffer_handoff"
    return "rerun_layer7_bf16_handoff_on_rocm_host"


def unavailable_artifact(
    *,
    classification: str,
    layer6_experts_path: Path,
    model_path: Path,
    source_layer: int,
    target_layer: int,
    position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    source_capture: Mapping[str, Any],
    target_capture: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "layer7_bf16_handoff_audit",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "unavailable",
        "classification": classification,
        "layer6_experts_path": str(layer6_experts_path),
        "model": str(model_path),
        "source_layer": int(source_layer),
        "target_layer": int(target_layer),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "source_capture": summarize_capture(source_capture),
        "target_capture": summarize_capture(target_capture),
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
