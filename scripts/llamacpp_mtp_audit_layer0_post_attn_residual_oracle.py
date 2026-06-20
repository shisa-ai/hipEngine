#!/usr/bin/env python3
"""Audit layer-0 post-attention residual and post-norm after verified attn_out."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    bf16_roundtrip_array,
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)
from scripts.llamacpp_mtp_audit_layer0_warm_conv_gdn_oracle import (  # noqa: E402
    load_token_embedding_rows,
)

DEFAULT_WARM_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter326-layer0-warm-conv-gdn-oracle.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter327-layer0-post-attn-residual-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
NormWeightLoader = Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]]
TokenHiddenLoader = Callable[[Path, tuple[int, ...], int], tuple[np.ndarray, dict[str, Any]]]

POST_ATTN_FIELDS = ("residual_f32", "post_norm_f32")
INPUT_FIELDS = ("hidden_in_f32", "attn_out_f32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-artifact", type=Path, default=DEFAULT_WARM_ARTIFACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--hidden-atol", type=float, default=0.0)
    parser.add_argument("--residual-atol", type=float, default=0.0)
    parser.add_argument("--post-norm-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=327)
    args = parser.parse_args()

    artifact = audit_layer0_post_attn_residual_oracle(
        warm_artifact_path=args.warm_artifact,
        model_path=args.model,
        max_sequence_length=args.max_sequence_length,
        hidden_atol=args.hidden_atol,
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


def audit_layer0_post_attn_residual_oracle(
    *,
    warm_artifact_path: Path,
    model_path: Path | None = None,
    max_sequence_length: int | None = None,
    hidden_atol: float = 0.0,
    residual_atol: float = 0.0,
    post_norm_atol: float = 2.5e-4,
    iteration: int = 327,
    layer_capture_fn: LayerCaptureFn | None = None,
    norm_weight_loader: NormWeightLoader | None = None,
    token_hidden_loader: TokenHiddenLoader | None = None,
) -> dict[str, Any]:
    warm = json.loads(warm_artifact_path.read_text())
    validate_warm_artifact(warm)
    resolved_model = Path(model_path or warm["model"])
    layer_id = int(warm["layer_id"])
    target_position = int(warm["target_position"])
    prompt_tokens = tuple(int(token) for token in warm["prompt_tokens"])
    token_id = int(prompt_tokens[target_position])
    capture_fn = layer_capture_fn or capture_hipengine_linear_layer
    capture = capture_fn(
        resolved_model,
        prompt_tokens,
        target_position,
        layer_id,
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            warm_artifact_path=warm_artifact_path,
            model_path=resolved_model,
            layer_id=layer_id,
            target_position=target_position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    selected_norm_loader = norm_weight_loader or load_post_attention_norm_weight
    post_norm_weight, eps, norm_metadata = selected_norm_loader(resolved_model, layer_id)
    selected_token_loader = token_hidden_loader or load_target_token_hidden
    token_hidden, token_metadata = selected_token_loader(
        resolved_model,
        prompt_tokens,
        target_position,
    )
    input_results = compare_inputs(
        capture=capture,
        token_hidden=token_hidden,
        near_atol=float(hidden_atol),
    )
    oracle_results = build_post_attn_oracle_results(
        capture=capture,
        post_norm_weight=post_norm_weight,
        eps=eps,
        residual_atol=float(residual_atol),
        post_norm_atol=float(post_norm_atol),
    )
    input_classification = classify_inputs(input_results)
    classification = classify_post_attn_oracle(input_classification, oracle_results)
    return {
        "schema": 1,
        "kind": "layer0_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "warm_artifact_path": str(warm_artifact_path),
        "model": str(resolved_model),
        "layer_id": layer_id,
        "target_position": target_position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": (
                "verified warm attn_out + layer input -> BF16 residual "
                "and post_attention_norm"
            ),
            "attention_source": warm.get("classification"),
            "target_capture_mode": (
                "hipEngine resident full layer capture after stepping prior tokens"
            ),
            "post_norm_tensor": norm_metadata["tensor_name"],
            "eps": float(eps),
        },
        "token_hidden_metadata": token_metadata,
        "post_attention_norm_weight": norm_metadata,
        "hipengine_capture": summarize_capture(capture),
        "input_results": input_results,
        "input_classification": input_classification,
        "oracle_results": oracle_results,
        "tolerances": {
            "hidden_in_f32": float(hidden_atol),
            "residual_f32": float(residual_atol),
            "post_norm_f32": float(post_norm_atol),
        },
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_warm_artifact(warm: Mapping[str, Any]) -> None:
    if warm.get("status") != "ready":
        raise ValueError("warm conv/GDN artifact must be ready")
    if not str(warm.get("classification", "")).startswith("layer0_warm_conv_gdn_matches"):
        raise ValueError("warm conv/GDN artifact must have matched")
    if warm.get("next_action") != "continue_layer0_bisection_after_attn_out_or_residual":
        raise ValueError("warm artifact must point to post-attn/residual bisection")


def capture_hipengine_linear_layer(
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
        capture = session.capture_linear_attention_layer(
            int(prompt_tokens[position]),
            position=int(position),
            layer_id=int(layer_id),
        )
        fields = {
            "hidden_in_f32": np.asarray(capture.hidden_in_f32, dtype=np.float32),
            "attn_out_f32": np.asarray(capture.attn_out_f32, dtype=np.float32),
            "residual_f32": np.asarray(capture.residual_f32, dtype=np.float32),
            "post_norm_f32": np.asarray(capture.post_norm_f32, dtype=np.float32),
            "ffn_or_moe_down_f32": np.asarray(
                capture.ffn_or_moe_down_f32,
                dtype=np.float32,
            ),
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
            fields["moe_shared_gate_f32"] = np.asarray(
                capture.moe_shared_gate_f32,
                dtype=np.float32,
            )
        if capture.moe_selected_experts_i64 is not None:
            fields["moe_selected_experts_i64"] = np.asarray(
                capture.moe_selected_experts_i64,
                dtype=np.int64,
            )
        return {"status": "captured", "summary": capture.as_summary_dict(), "fields": fields}


def load_post_attention_norm_weight(
    model_path: Path,
    layer_id: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    reader = GGUFReader(model_path)
    tensor_name = f"blk.{layer_id}.post_attention_norm.weight"
    tensor = reader.tensor_info(tensor_name)
    values = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
    eps = float(reader.info.metadata["qwen35moe.attention.layer_norm_rms_epsilon"])
    return values, eps, {
        "tensor_name": tensor_name,
        "ggml_type": tensor.ggml_type_name,
        "shape": list(values.shape),
        "summary": summarize_array(values),
        "sha256": sha256_float32(values),
    }


def load_target_token_hidden(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    target_position: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    reader = GGUFReader(model_path)
    rows, metadata = load_token_embedding_rows(reader, (int(prompt_tokens[target_position]),))
    hidden = bf16_roundtrip_array(rows[0])
    metadata = dict(metadata)
    metadata["target_position"] = int(target_position)
    metadata["target_hidden_bf16_summary"] = summarize_array(hidden)
    metadata["target_hidden_bf16_sha256"] = sha256_float32(hidden)
    return hidden, metadata


def compare_inputs(
    *,
    capture: Mapping[str, Any],
    token_hidden: np.ndarray,
    near_atol: float,
) -> dict[str, Any]:
    fields = capture["fields"]
    hidden_delta = delta_summary(
        token_hidden,
        np.asarray(fields["hidden_in_f32"], dtype=np.float32),
    )
    return {
        "hidden_in_f32": {
            "field": "hidden_in_f32",
            "oracle": "BF16(token_embd.weight[token_id])",
            "oracle_summary": summarize_array(token_hidden),
            "hipengine_summary": summarize_array(fields["hidden_in_f32"]),
            "delta_oracle_vs_hip": hidden_delta,
            "near_atol": float(near_atol),
            "classification": classify_field_delta(hidden_delta, near_atol=float(near_atol)),
        },
        "attn_out_f32": {
            "field": "attn_out_f32",
            "oracle": "previous warm conv/GDN artifact establishes attn_out correctness",
            "hipengine_summary": summarize_array(fields["attn_out_f32"]),
            "classification": "input_covered_by_warm_conv_gdn_oracle",
        },
    }


def build_post_attn_oracle_results(
    *,
    capture: Mapping[str, Any],
    post_norm_weight: np.ndarray,
    eps: float,
    residual_atol: float,
    post_norm_atol: float,
) -> dict[str, Any]:
    fields = capture["fields"]
    hidden = np.asarray(fields["hidden_in_f32"], dtype=np.float32)
    attn_out = np.asarray(fields["attn_out_f32"], dtype=np.float32)
    residual, post_norm = add_rmsnorm_bf16_oracle(
        hidden,
        attn_out,
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
            "oracle": "BF16((hidden_in + attn_out) * rsqrt(mean(square(sum)) + eps) * weight)",
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


def add_rmsnorm_bf16_oracle(
    hidden: np.ndarray,
    add: np.ndarray,
    weight: np.ndarray,
    *,
    eps: float,
    threads: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    hidden32 = np.asarray(hidden, dtype=np.float32).reshape(-1)
    add32 = np.asarray(add, dtype=np.float32).reshape(-1)
    weight32 = np.asarray(weight, dtype=np.float32).reshape(-1)
    if hidden32.shape != add32.shape or hidden32.shape != weight32.shape:
        raise ValueError("hidden/add/weight shapes must match")
    summed = np.asarray(hidden32 + add32, dtype=np.float32)
    sumsq = kernel_strided_sum(np.asarray(summed * summed, dtype=np.float32), threads=threads)
    variance = np.float32(sumsq / np.float32(summed.shape[0]) + np.float32(eps))
    inv_rms = np.float32(1.0 / math.sqrt(float(variance)))
    residual = bf16_roundtrip_array(summed)
    post_norm = bf16_roundtrip_array(np.asarray(summed * inv_rms * weight32, dtype=np.float32))
    return residual, post_norm


def kernel_strided_sum(values: np.ndarray, *, threads: int = 256) -> np.float32:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    partial = np.zeros((int(threads),), dtype=np.float32)
    for tid in range(int(threads)):
        local = np.float32(0.0)
        for idx in range(tid, arr.shape[0], int(threads)):
            local = np.float32(local + arr[idx])
        partial[tid] = local
    stride = int(threads) // 2
    while stride > 0:
        for tid in range(stride):
            partial[tid] = np.float32(partial[tid] + partial[tid + stride])
        stride //= 2
    return np.float32(partial[0])


def classify_field_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "post_attn_field_oracle_unavailable"
    if delta.get("exact_match"):
        return "post_attn_field_matches_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "post_attn_field_matches_oracle_within_tolerance"
    return "post_attn_field_mismatch_after_oracle"


def classify_inputs(input_results: Mapping[str, Any]) -> str:
    hidden_class = input_results["hidden_in_f32"]["classification"]
    attn_class = input_results["attn_out_f32"]["classification"]
    if hidden_class == "post_attn_field_matches_oracle_exactly" and attn_class.startswith(
        "input_covered"
    ):
        return "post_attn_inputs_match_oracle"
    if "mismatch" in hidden_class:
        return "post_attn_inputs_mismatch_before_residual"
    return "post_attn_inputs_unavailable"


def classify_post_attn_oracle(
    input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if input_classification != "post_attn_inputs_match_oracle":
        return "layer0_post_attn_residual_blocked_input_mismatch"
    classes = [oracle_results[name]["classification"] for name in POST_ATTN_FIELDS]
    if all(item == "post_attn_field_matches_oracle_exactly" for item in classes):
        return "layer0_post_attn_residual_matches_oracle_exactly"
    if all(item.startswith("post_attn_field_matches_oracle") for item in classes):
        return "layer0_post_attn_residual_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer0_post_attn_residual_mismatch_after_oracle"
    return "layer0_post_attn_residual_oracle_unavailable"


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
        "layer0_post_attn_residual_matches_oracle_exactly",
        "layer0_post_attn_residual_matches_oracle_within_tolerance",
    }:
        return "audit_layer0_moe_router_from_post_norm"
    if classification == "layer0_post_attn_residual_blocked_input_mismatch":
        return "reconcile_layer0_hidden_or_attn_out_before_post_attn"
    if classification == "layer0_post_attn_residual_mismatch_after_oracle":
        return "inspect_post_attn_residual_or_norm_kernel"
    return "rerun_post_attn_residual_oracle_on_rocm_host"


def unavailable_artifact(
    *,
    warm_artifact_path: Path,
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
        "kind": "layer0_post_attn_residual_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer0_post_attn_residual_oracle_unavailable",
        "warm_artifact_path": str(warm_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "target_position": int(target_position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_post_attn_residual_oracle_on_rocm_host",
    }


if __name__ == "__main__":
    main()
