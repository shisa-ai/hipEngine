#!/usr/bin/env python3
"""Audit layer-12 conv/GDN outputs under the resident-BF16 contract."""

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

from scripts.llamacpp_mtp_audit_layer0_position0_conv_gdn_oracle import (  # noqa: E402
    FIELD_ORDER,
    load_conv_gdn_weights,
    summarize_capture,
    summarize_weights,
)
from scripts.llamacpp_mtp_audit_layer0_warm_conv_gdn_oracle import (  # noqa: E402
    INPUT_FIELDS,
    build_warm_oracle_results,
    classify_target_inputs,
    compare_target_inputs,
)
from scripts.llamacpp_mtp_audit_layer1_conv_gdn_oracle import (  # noqa: E402
    capture_layer1_projection_sequence,
)
from scripts.llamacpp_mtp_audit_layer12_projection_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER12_PROJECTION,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter418-layer12-conv-gdn-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

SequenceCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
ConvGDNWeightLoader = Callable[
    [Path, int],
    dict[str, tuple[np.ndarray, dict[str, Any]]],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer12-projection",
        type=Path,
        default=DEFAULT_LAYER12_PROJECTION,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=12)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--input-atol", type=float, default=0.0)
    parser.add_argument("--conv-atol", type=float, default=2.5e-4)
    parser.add_argument("--recurrent-atol", type=float, default=1.0e-3)
    parser.add_argument("--bf16-atol", type=float, default=2.5e-4)
    parser.add_argument("--attn-out-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=418)
    args = parser.parse_args()

    artifact = audit_layer12_conv_gdn_oracle(
        layer12_projection_path=args.layer12_projection,
        model_path=args.model,
        layer_id=args.layer_id,
        max_sequence_length=args.max_sequence_length,
        input_atol=args.input_atol,
        conv_atol=args.conv_atol,
        recurrent_atol=args.recurrent_atol,
        bf16_atol=args.bf16_atol,
        attn_out_atol=args.attn_out_atol,
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
                "target_input_classification": artifact.get(
                    "target_input_classification"
                ),
                "field_classifications": field_classifications(artifact),
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer12_conv_gdn_oracle(
    *,
    layer12_projection_path: Path,
    model_path: Path | None = None,
    layer_id: int = 12,
    max_sequence_length: int | None = None,
    input_atol: float = 0.0,
    conv_atol: float = 2.5e-4,
    recurrent_atol: float = 1.0e-3,
    bf16_atol: float = 2.5e-4,
    attn_out_atol: float = 2.5e-4,
    iteration: int = 418,
    sequence_capture_fn: SequenceCaptureFn | None = None,
    conv_gdn_weight_loader: ConvGDNWeightLoader | None = None,
) -> dict[str, Any]:
    projection = json.loads(layer12_projection_path.read_text())
    validate_layer12_projection(projection, expected_layer_id=int(layer_id))
    resolved_model = Path(model_path or projection["model"])
    prompt_tokens = tuple(int(token) for token in projection["prompt_tokens"])
    target_position = int(projection["position"])
    token_id = int(prompt_tokens[target_position])
    capture_fn = sequence_capture_fn or capture_layer1_projection_sequence
    sequence = capture_fn(
        resolved_model,
        prompt_tokens[: target_position + 1],
        target_position,
        int(layer_id),
        max_sequence_length,
    )
    if sequence.get("status") != "captured":
        return unavailable_artifact(
            layer12_projection_path=layer12_projection_path,
            model_path=resolved_model,
            layer_id=int(layer_id),
            target_position=target_position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            sequence=sequence,
            iteration=iteration,
        )
    replay_inputs = sequence["replay_inputs"]
    target_capture = sequence["target_capture"]
    dimensions = sequence["dimensions"]
    eps = float(sequence["rms_norm_eps"])
    weights = (conv_gdn_weight_loader or load_conv_gdn_weights)(
        resolved_model,
        int(layer_id),
    )
    target_input_results = compare_target_inputs(
        replay_inputs[target_position],
        target_capture,
        near_atol=float(input_atol),
    )
    target_input_classification = classify_target_inputs(target_input_results)
    tolerances = {
        "conv_out_f32": float(conv_atol),
        "recurrent_out_f32": float(recurrent_atol),
        "recurrent_bf16_f32": float(bf16_atol),
        "attn_out_f32": float(attn_out_atol),
    }
    oracle_results, replay_summary = build_warm_oracle_results(
        target_capture=target_capture,
        replay_inputs=replay_inputs,
        weights=weights,
        dimensions=dimensions,
        eps=eps,
        target_position=target_position,
        tolerances=tolerances,
    )
    classification = classify_layer12_conv_gdn(
        target_input_classification,
        oracle_results,
    )
    return {
        "schema": 1,
        "kind": "layer12_warm_bf16_contracted_conv_gdn_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer12_projection_path": str(layer12_projection_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "target_position": target_position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "replay_contract": {
            "source": "captured layer12 projection inputs after verified attn_norm",
            "projection_source": projection.get("classification"),
            "replayed_positions": list(range(target_position + 1)),
            "target_capture_mode": (
                "sequential layer12 full-layer captures with preceding layers 0-11"
            ),
            "starts_from_zero_state": True,
            "conv_state_floats": dimensions["conv_state_floats"],
            "recurrent_state_floats": dimensions["recurrent_state_floats"],
        },
        "model_dimensions": dimensions,
        "sequence_capture": sequence["metadata"],
        "weights": summarize_weights(weights),
        "target_input_results": target_input_results,
        "target_input_classification": target_input_classification,
        "hipengine_capture": summarize_capture(target_capture),
        "replay_summary": replay_summary,
        "oracle_results": oracle_results,
        "tolerances": {"input": float(input_atol), **tolerances},
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_layer12_projection(
    projection: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if projection.get("status") != "ready":
        raise ValueError("layer12 projection artifact must be ready")
    if not str(projection.get("classification", "")).startswith(
        "layer12_projections_match_bf16_oracle"
    ):
        raise ValueError("layer12 projection artifact must have matched")
    if int(projection.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError(
            "layer12 projection layer_id does not match requested layer"
        )
    if projection.get("next_action") != "audit_layer12_conv_gdn_under_bf16_contract":
        raise ValueError("layer12 projection artifact must point to conv/GDN audit")
    if (projection.get("input_result") or {}).get("exact_hash_match") is not True:
        raise ValueError(
            "layer12 projection input hash must match attn_norm artifact"
        )
    results = projection.get("projection_results") or {}
    for field in INPUT_FIELDS[1:]:
        if field not in results:
            raise ValueError(f"projection artifact missing {field}")
        classification = str(results[field].get("classification", ""))
        if not classification.startswith("projection_matches_bf16_oracle"):
            raise ValueError(f"projection artifact field {field} did not match")


def classify_layer12_conv_gdn(
    target_input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if target_input_classification in {
        "target_inputs_mismatch_before_conv_gdn_replay",
        "target_inputs_unavailable",
    }:
        return "layer12_warm_conv_gdn_blocked_target_input_mismatch"
    classes = [oracle_results[name]["classification"] for name in FIELD_ORDER]
    if all(item == "warm_field_matches_oracle_exactly" for item in classes):
        return "layer12_warm_conv_gdn_matches_oracle_exactly"
    if all(item.startswith("warm_field_matches_oracle") for item in classes):
        return "layer12_warm_conv_gdn_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer12_warm_conv_gdn_mismatch_after_replay_oracle"
    return "layer12_warm_conv_gdn_oracle_unavailable"


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
        "layer12_warm_conv_gdn_matches_oracle_exactly",
        "layer12_warm_conv_gdn_matches_oracle_within_tolerance",
    }:
        return "audit_layer12_post_attn_residual_or_moe_boundary"
    if classification == "layer12_warm_conv_gdn_blocked_target_input_mismatch":
        return "inspect_layer12_projection_input_capture_sequence"
    if classification == "layer12_warm_conv_gdn_mismatch_after_replay_oracle":
        return "inspect_layer12_first_warm_conv_gdn_replay_mismatch"
    return "rerun_layer12_conv_gdn_oracle_on_rocm_host"


def field_classifications(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: artifact["oracle_results"][name]["classification"]
        for name in FIELD_ORDER
        if name in artifact.get("oracle_results", {})
    }


def unavailable_artifact(
    *,
    layer12_projection_path: Path,
    model_path: Path,
    layer_id: int,
    target_position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    sequence: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer12_warm_conv_gdn_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer12_warm_bf16_contracted_conv_gdn_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(sequence.get("status", "unavailable")),
        "classification": classification,
        "layer12_projection_path": str(layer12_projection_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "target_position": int(target_position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "sequence_capture": dict(sequence),
        "target_input_classification": "target_inputs_unavailable",
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


if __name__ == "__main__":
    main()
