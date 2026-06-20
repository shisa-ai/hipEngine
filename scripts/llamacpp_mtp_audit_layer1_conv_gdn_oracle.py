#!/usr/bin/env python3
"""Audit layer-1 conv/GDN outputs under the resident-BF16 contract."""

from __future__ import annotations

import argparse
import ctypes
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
from scripts.llamacpp_mtp_audit_layer1_projection_oracle import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LAYER1_PROJECTION,
)

DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter334-layer1-conv-gdn-oracle.json")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

SequenceCaptureFn = Callable[
    [Path, tuple[int, ...], int, int, int | None],
    dict[str, Any],
]
ConvGDNWeightLoader = Callable[[Path, int], dict[str, tuple[np.ndarray, dict[str, Any]]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer1-projection", type=Path, default=DEFAULT_LAYER1_PROJECTION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-id", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--input-atol", type=float, default=0.0)
    parser.add_argument("--conv-atol", type=float, default=2.5e-4)
    parser.add_argument("--recurrent-atol", type=float, default=1.0e-3)
    parser.add_argument("--bf16-atol", type=float, default=2.5e-4)
    parser.add_argument("--attn-out-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=334)
    args = parser.parse_args()

    artifact = audit_layer1_conv_gdn_oracle(
        layer1_projection_path=args.layer1_projection,
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
                "target_input_classification": artifact.get("target_input_classification"),
                "field_classifications": {
                    name: artifact["oracle_results"][name]["classification"]
                    for name in FIELD_ORDER
                    if name in artifact.get("oracle_results", {})
                },
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer1_conv_gdn_oracle(
    *,
    layer1_projection_path: Path,
    model_path: Path | None = None,
    layer_id: int = 1,
    max_sequence_length: int | None = None,
    input_atol: float = 0.0,
    conv_atol: float = 2.5e-4,
    recurrent_atol: float = 1.0e-3,
    bf16_atol: float = 2.5e-4,
    attn_out_atol: float = 2.5e-4,
    iteration: int = 334,
    sequence_capture_fn: SequenceCaptureFn | None = None,
    conv_gdn_weight_loader: ConvGDNWeightLoader | None = None,
) -> dict[str, Any]:
    projection = json.loads(layer1_projection_path.read_text())
    validate_layer1_projection(projection, expected_layer_id=int(layer_id))
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
            layer1_projection_path=layer1_projection_path,
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
    weights = (conv_gdn_weight_loader or load_conv_gdn_weights)(resolved_model, int(layer_id))
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
    classification = classify_layer1_conv_gdn(target_input_classification, oracle_results)
    return {
        "schema": 1,
        "kind": "layer1_warm_bf16_contracted_conv_gdn_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "layer1_projection_path": str(layer1_projection_path),
        "model": str(resolved_model),
        "layer_id": int(layer_id),
        "target_position": target_position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "replay_contract": {
            "source": "captured layer1 projection inputs after verified attn_norm",
            "projection_source": projection.get("classification"),
            "replayed_positions": list(range(target_position + 1)),
            "target_capture_mode": "sequential layer1 full-layer captures with preceding layer0",
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


def validate_layer1_projection(
    projection: Mapping[str, Any],
    *,
    expected_layer_id: int,
) -> None:
    if projection.get("status") != "ready":
        raise ValueError("layer1 projection artifact must be ready")
    if not str(projection.get("classification", "")).startswith(
        "layer1_projections_match_bf16_oracle"
    ):
        raise ValueError("layer1 projection artifact must have matched")
    if int(projection.get("layer_id", -1)) != int(expected_layer_id):
        raise ValueError("layer1 projection layer_id does not match requested layer")
    if projection.get("next_action") != "audit_layer1_conv_gdn_under_bf16_contract":
        raise ValueError("layer1 projection artifact must point to conv/GDN audit")
    results = projection.get("projection_results") or {}
    for field in INPUT_FIELDS[1:]:
        if field not in results:
            raise ValueError(f"projection artifact missing {field}")
        classification = str(results[field].get("classification", ""))
        if not classification.startswith("projection_matches_bf16_oracle"):
            raise ValueError(f"projection artifact field {field} did not match")


def capture_layer1_projection_sequence(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    target_position: int,
    layer_id: int,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    if not hip_available():
        return {"status": "skipped_no_hip_runtime", "replay_inputs": []}
    from hipengine.runtime.qwen35_gguf_runner import (
        DType,
        Qwen35GGUFResidentSession,
        _copy_bf16_ptr_to_host_f32,
        _copy_f32_ptr_to_host,
    )

    max_seq = int(max_sequence_length or max(len(prompt_tokens) + 8, 32))
    replay_inputs: list[dict[str, np.ndarray]] = []
    captures: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(model_path, max_sequence_length=max_seq) as session:
        cfg = session.runner.weights.config
        dimensions = layer_dimensions(session)
        for position, token_id in enumerate(prompt_tokens):
            layer_capture = session.capture_attention_layer(
                int(token_id),
                position=int(position),
                layer_id=int(layer_id),
                run_preceding_layers=True,
            )
            runtime = session.runtime
            rank = int(cfg.ssm_time_step_rank)
            alpha_ptr = int(session.scratch.linear_alpha.ptr)
            beta_ptr = int(session.scratch.linear_beta.ptr)
            if cfg.is_moe:
                alpha_ptr = int(session.scratch.linear_alpha_beta.ptr)
                beta_ptr = alpha_ptr + rank * DType.BF16.itemsize
            fields = {
                "attn_norm_f32": _copy_bf16_ptr_to_host_f32(
                    int(session.scratch.norm.ptr),
                    dimensions["hidden_size"],
                    runtime=runtime,
                ),
                "linear_qkv_f32": _copy_bf16_ptr_to_host_f32(
                    int(session.scratch.linear_qkv.ptr),
                    dimensions["linear_qkv_width"],
                    runtime=runtime,
                ),
                "linear_z_f32": _copy_bf16_ptr_to_host_f32(
                    int(session.scratch.linear_z.ptr),
                    dimensions["ssm_inner_size"],
                    runtime=runtime,
                ),
                "ssm_alpha_f32": _copy_bf16_ptr_to_host_f32(
                    alpha_ptr,
                    rank,
                    runtime=runtime,
                ),
                "ssm_beta_f32": _copy_bf16_ptr_to_host_f32(
                    beta_ptr,
                    rank,
                    runtime=runtime,
                ),
                "conv_out_f32": _copy_f32_ptr_to_host(
                    int(session.scratch.conv_out.ptr),
                    dimensions["linear_qkv_width"],
                    runtime=runtime,
                ),
                "recurrent_out_f32": _copy_f32_ptr_to_host(
                    int(session.scratch.recurrent_out.ptr),
                    dimensions["ssm_inner_size"],
                    runtime=runtime,
                ),
                "recurrent_bf16_f32": _copy_bf16_ptr_to_host_f32(
                    int(session.scratch.recurrent_bf16.ptr),
                    dimensions["ssm_inner_size"],
                    runtime=runtime,
                ),
                "attn_out_f32": np.asarray(layer_capture.attn_out_f32, dtype=np.float32),
            }
            replay_inputs.append({field: fields[field] for field in INPUT_FIELDS})
            captures.append(
                {
                    "status": "captured",
                    "summary": {
                        **layer_capture.as_summary_dict(),
                        "attn_norm_shape": list(fields["attn_norm_f32"].shape),
                        "linear_qkv_shape": list(fields["linear_qkv_f32"].shape),
                        "linear_z_shape": list(fields["linear_z_f32"].shape),
                        "ssm_alpha_shape": list(fields["ssm_alpha_f32"].shape),
                        "ssm_beta_shape": list(fields["ssm_beta_f32"].shape),
                        "conv_out_shape": list(fields["conv_out_f32"].shape),
                        "recurrent_out_shape": list(fields["recurrent_out_f32"].shape),
                        "recurrent_bf16_shape": list(fields["recurrent_bf16_f32"].shape),
                    },
                    "fields": fields,
                }
            )
        eps = float(cfg.rms_norm_eps)
    target_capture = captures[int(target_position)]
    return {
        "status": "captured",
        "replay_inputs": replay_inputs,
        "target_capture": target_capture,
        "dimensions": dimensions,
        "rms_norm_eps": eps,
        "metadata": summarize_sequence_capture(replay_inputs, captures, int(target_position)),
    }


def layer_dimensions(session: Any) -> dict[str, int]:
    cfg = session.runner.weights.config
    conv_state_floats = int(session.runner.linear_qkv_width) * int(cfg.ssm_conv_kernel)
    recurrent_state_floats = (
        int(cfg.ssm_time_step_rank) * int(cfg.ssm_state_size) * int(session.runner.ssm_value_dim)
    )
    return {
        "hidden_size": int(session.runner.hidden_size),
        "ssm_group_count": int(cfg.ssm_group_count),
        "ssm_time_step_rank": int(cfg.ssm_time_step_rank),
        "ssm_state_size": int(cfg.ssm_state_size),
        "ssm_value_dim": int(session.runner.ssm_value_dim),
        "ssm_inner_size": int(cfg.ssm_inner_size),
        "linear_qkv_width": int(session.runner.linear_qkv_width),
        "ssm_conv_kernel": int(cfg.ssm_conv_kernel),
        "conv_state_floats": conv_state_floats,
        "recurrent_state_floats": recurrent_state_floats,
    }


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def summarize_sequence_capture(
    replay_inputs: list[dict[str, np.ndarray]],
    captures: list[dict[str, Any]],
    target_position: int,
) -> dict[str, Any]:
    return {
        "position_count": int(len(replay_inputs)),
        "captured_positions": [int(item["summary"]["position"]) for item in captures],
        "target_position": int(target_position),
        "target_capture_summary": captures[int(target_position)]["summary"],
        "first_position_input_summary": summarize_input_record(0, replay_inputs[0]),
        "target_position_input_summary": summarize_input_record(
            int(target_position),
            replay_inputs[int(target_position)],
        ),
        "stack_sha256": {
            field: sha256_float32(
                np.asarray([record[field] for record in replay_inputs], dtype=np.float32)
                .reshape(-1)
            )
            for field in INPUT_FIELDS
        },
    }


def summarize_input_record(position: int, record: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "position": int(position),
        "field_summaries": {
            field: summarize_array(np.asarray(record[field], dtype=np.float32))
            for field in INPUT_FIELDS
        },
    }


def classify_layer1_conv_gdn(
    target_input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if target_input_classification in {
        "target_inputs_mismatch_before_conv_gdn_replay",
        "target_inputs_unavailable",
    }:
        return "layer1_warm_conv_gdn_blocked_target_input_mismatch"
    classes = [oracle_results[name]["classification"] for name in FIELD_ORDER]
    if all(item == "warm_field_matches_oracle_exactly" for item in classes):
        return "layer1_warm_conv_gdn_matches_oracle_exactly"
    if all(item.startswith("warm_field_matches_oracle") for item in classes):
        return "layer1_warm_conv_gdn_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer1_warm_conv_gdn_mismatch_after_replay_oracle"
    return "layer1_warm_conv_gdn_oracle_unavailable"


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
        "layer1_warm_conv_gdn_matches_oracle_exactly",
        "layer1_warm_conv_gdn_matches_oracle_within_tolerance",
    }:
        return "audit_layer1_post_attn_residual_or_moe_boundary"
    if classification == "layer1_warm_conv_gdn_blocked_target_input_mismatch":
        return "inspect_layer1_projection_input_capture_sequence"
    if classification == "layer1_warm_conv_gdn_mismatch_after_replay_oracle":
        return "inspect_layer1_first_warm_conv_gdn_replay_mismatch"
    return "rerun_layer1_conv_gdn_oracle_on_rocm_host"


def unavailable_artifact(
    *,
    layer1_projection_path: Path,
    model_path: Path,
    layer_id: int,
    target_position: int,
    token_id: int,
    prompt_tokens: tuple[int, ...],
    sequence: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    classification = "layer1_warm_conv_gdn_oracle_unavailable"
    return {
        "schema": 1,
        "kind": "layer1_warm_bf16_contracted_conv_gdn_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(sequence.get("status", "unavailable")),
        "classification": classification,
        "layer1_projection_path": str(layer1_projection_path),
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
