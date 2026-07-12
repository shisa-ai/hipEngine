#!/usr/bin/env python3
"""Audit position-0 layer-0 conv/GDN outputs with a BF16-contracted CPU oracle."""

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
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    _gguf_ssm_a_to_kernel_a_log,
)
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    bf16_roundtrip_array,
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    capture_hipengine_linear_boundary,
    project_f32,
)

DEFAULT_PROJECTION_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter323-layer0-bf16-projection-oracle.json"
)
DEFAULT_PLAN = Path("benchmarks/results/mtp-gguf-iter324-layer0-conv-gdn-plan.json")
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter325-layer0-position0-conv-gdn-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

BoundaryCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
ConvGDNWeightLoader = Callable[[Path, int], dict[str, tuple[np.ndarray, dict[str, Any]]]]

FIELD_ORDER = (
    "conv_out_f32",
    "recurrent_out_f32",
    "recurrent_bf16_f32",
    "attn_out_f32",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-artifact", type=Path, default=DEFAULT_PROJECTION_ARTIFACT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--conv-atol", type=float, default=1.0e-5)
    parser.add_argument("--recurrent-atol", type=float, default=5.0e-5)
    parser.add_argument("--bf16-atol", type=float, default=2.5e-4)
    parser.add_argument("--attn-out-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=325)
    args = parser.parse_args()

    artifact = audit_layer0_position0_conv_gdn_oracle(
        projection_artifact_path=args.projection_artifact,
        plan_path=args.plan,
        model_path=args.model,
        max_sequence_length=args.max_sequence_length,
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


def audit_layer0_position0_conv_gdn_oracle(
    *,
    projection_artifact_path: Path,
    plan_path: Path,
    model_path: Path | None = None,
    max_sequence_length: int | None = None,
    conv_atol: float = 1.0e-5,
    recurrent_atol: float = 5.0e-5,
    bf16_atol: float = 2.5e-4,
    attn_out_atol: float = 2.5e-4,
    iteration: int = 325,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
    weight_loader: ConvGDNWeightLoader | None = None,
) -> dict[str, Any]:
    projection = json.loads(projection_artifact_path.read_text())
    plan = json.loads(plan_path.read_text())
    validate_inputs(projection, plan)
    resolved_model = Path(model_path or projection["model"])
    layer_id = int(projection["layer_id"])
    position = 0
    prompt_tokens = tuple(int(token) for token in projection["prompt_tokens"])
    token_id = int(prompt_tokens[position])
    selected_capture_fn = boundary_capture_fn or capture_hipengine_linear_boundary
    capture = selected_capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        layer_id,
        max_sequence_length,
    )
    if capture.get("status") != "captured":
        return unavailable_artifact(
            projection_artifact_path=projection_artifact_path,
            plan_path=plan_path,
            model_path=resolved_model,
            layer_id=layer_id,
            position=position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    selected_weight_loader = weight_loader or load_conv_gdn_weights
    weights = selected_weight_loader(resolved_model, layer_id)
    tolerances = {
        "conv_out_f32": float(conv_atol),
        "recurrent_out_f32": float(recurrent_atol),
        "recurrent_bf16_f32": float(bf16_atol),
        "attn_out_f32": float(attn_out_atol),
    }
    oracle_results = build_position0_oracle_results(
        capture=capture,
        weights=weights,
        dimensions=plan["model_metadata"]["dimensions"],
        eps=float(plan["model_metadata"].get("rms_norm_eps", 1.0e-6)),
        tolerances=tolerances,
    )
    classification = classify_position0_oracle(oracle_results)
    return {
        "schema": 1,
        "kind": "layer0_position0_bf16_contracted_conv_gdn_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "projection_artifact_path": str(projection_artifact_path),
        "plan_path": str(plan_path),
        "model": str(resolved_model),
        "layer_id": layer_id,
        "position": position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "input_contract": {
            "source": "hipEngine resident BF16 boundary capture at position 0",
            "uses_captured_projection_inputs": [
                "linear_qkv_f32",
                "linear_z_f32",
                "ssm_alpha_f32",
                "ssm_beta_f32",
            ],
            "zero_state": True,
            "conv_state_floats": plan["model_metadata"]["dimensions"]["conv_state_floats"],
            "recurrent_state_floats": plan["model_metadata"]["dimensions"][
                "recurrent_state_floats"
            ],
        },
        "model_dimensions": plan["model_metadata"]["dimensions"],
        "weights": summarize_weights(weights),
        "hipengine_capture": summarize_capture(capture),
        "oracle_results": oracle_results,
        "tolerances": tolerances,
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_inputs(projection: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if projection.get("status") != "ready":
        raise ValueError("projection artifact must be ready")
    if not str(projection.get("classification", "")).startswith("layer0_projections_match"):
        raise ValueError("projection artifact must have matched layer-0 projections")
    if plan.get("status") != "ready":
        raise ValueError("conv/GDN plan must be ready")
    decision = plan.get("decision") or {}
    if decision.get("selected_strategy") != "position0_stateless_conv_gdn_oracle_first":
        raise ValueError("conv/GDN plan must select the position-0 strategy")


def build_position0_oracle_results(
    *,
    capture: Mapping[str, Any],
    weights: Mapping[str, tuple[np.ndarray, dict[str, Any]]],
    dimensions: Mapping[str, Any],
    eps: float,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    fields = capture["fields"]
    linear_qkv = np.asarray(fields["linear_qkv_f32"], dtype=np.float32)
    linear_z = np.asarray(fields["linear_z_f32"], dtype=np.float32)
    alpha = np.asarray(fields["ssm_alpha_f32"], dtype=np.float32)
    beta = np.asarray(fields["ssm_beta_f32"], dtype=np.float32)
    conv_weight = weights["ssm_conv1d"][0]
    dt_bias = weights["ssm_dt_bias"][0]
    a_log = weights["ssm_a_log"][0]
    norm_weight = weights["ssm_norm"][0]
    ssm_out_weight = weights["ssm_out"][0]

    conv_out = conv_decode_zero_state(linear_qkv, conv_weight)
    recurrent_out = gdn_recurrent_zero_state(
        conv_out=conv_out,
        gate=linear_z,
        alpha=alpha,
        beta=beta,
        dt_bias=dt_bias,
        a_log=a_log,
        norm_weight=norm_weight,
        eps=float(eps),
        num_k_heads=int(dimensions["ssm_group_count"]),
        num_v_heads=int(dimensions["ssm_time_step_rank"]),
        head_k_dim=int(dimensions["ssm_state_size"]),
        head_v_dim=int(dimensions["ssm_value_dim"]),
    )
    recurrent_bf16 = bf16_roundtrip_array(recurrent_out)
    attn_out = bf16_roundtrip_array(project_f32(recurrent_bf16, ssm_out_weight))
    oracles = {
        "conv_out_f32": {
            "oracle": "silu(BF16(linear_qkv) * ssm_conv1d[:, -1]) from zero conv state",
            "values": conv_out,
        },
        "recurrent_out_f32": {
            "oracle": "zero-state GDN recurrent RMSNorm+gate in F32 before BF16 cast",
            "values": recurrent_out,
        },
        "recurrent_bf16_f32": {
            "oracle": "BF16(recurrent_out_f32) copied back to F32",
            "values": recurrent_bf16,
        },
        "attn_out_f32": {
            "oracle": "BF16(ssm_out(Q8_0) @ recurrent_bf16)",
            "values": attn_out,
        },
    }
    results: dict[str, Any] = {}
    for field, record in oracles.items():
        oracle = np.asarray(record["values"], dtype=np.float32)
        hip = np.asarray(fields[field], dtype=np.float32)
        delta = delta_summary(oracle, hip)
        results[field] = {
            "field": field,
            "oracle": record["oracle"],
            "oracle_summary": summarize_array(oracle),
            "hipengine_summary": summarize_array(hip),
            "delta_oracle_vs_hip": delta,
            "near_atol": float(tolerances[field]),
            "classification": classify_field_delta(delta, near_atol=float(tolerances[field])),
        }
    return results


def conv_decode_zero_state(linear_qkv: np.ndarray, conv_weight: np.ndarray) -> np.ndarray:
    x = np.asarray(linear_qkv, dtype=np.float32).reshape(-1)
    w = np.asarray(conv_weight, dtype=np.float32)
    if w.ndim != 2 or w.shape[0] != x.shape[0]:
        raise ValueError(f"conv shape mismatch: input={x.shape}, weight={w.shape}")
    newest_weight = w[:, -1]
    acc = np.asarray(x * newest_weight, dtype=np.float32)
    return silu_array_f32(acc)


def gdn_recurrent_zero_state(
    *,
    conv_out: np.ndarray,
    gate: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    norm_weight: np.ndarray,
    eps: float,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> np.ndarray:
    conv = np.asarray(conv_out, dtype=np.float32).reshape(-1)
    gate32 = np.asarray(gate, dtype=np.float32).reshape(-1)
    alpha32 = np.asarray(alpha, dtype=np.float32).reshape(-1)
    beta32 = np.asarray(beta, dtype=np.float32).reshape(-1)
    dt_bias32 = np.asarray(dt_bias, dtype=np.float32).reshape(-1)
    a_log32 = np.asarray(a_log, dtype=np.float32).reshape(-1)
    norm32 = np.asarray(norm_weight, dtype=np.float32).reshape(-1)
    key_dim = int(num_k_heads) * int(head_k_dim)
    value_dim = int(num_v_heads) * int(head_v_dim)
    if conv.shape[0] != 2 * key_dim + value_dim:
        raise ValueError("conv_out width does not match Q/K/V dimensions")
    if gate32.shape[0] != value_dim:
        raise ValueError("gate width does not match value dimensions")
    if norm32.shape[0] != head_v_dim:
        raise ValueError("norm weight width does not match head_v_dim")
    if alpha32.shape[0] != num_v_heads or beta32.shape[0] != num_v_heads:
        raise ValueError("alpha/beta width does not match num_v_heads")
    if dt_bias32.shape[0] != num_v_heads or a_log32.shape[0] != num_v_heads:
        raise ValueError("dt_bias/a_log width does not match num_v_heads")

    out = np.empty((value_dim,), dtype=np.float32)
    inv_sqrt_head_k = np.float32(1.0 / math.sqrt(float(head_k_dim)))
    for v_head in range(int(num_v_heads)):
        # llama.cpp maps Qwen3.5 linear-attention K groups interleaved across
        # V heads: iv1 % neq1 in GGML_OP_GATED_DELTA_NET.
        k_head = v_head % int(num_k_heads)
        q_base = k_head * int(head_k_dim)
        k_base = key_dim + q_base
        value_offset = 2 * key_dim + v_head * int(head_v_dim)
        q = conv[q_base : q_base + int(head_k_dim)]
        k = conv[k_base : k_base + int(head_k_dim)]
        value = conv[value_offset : value_offset + int(head_v_dim)]
        q_sum = tree_reduce_sum(square_f32(q), threads=128, group=8)
        k_sum = tree_reduce_sum(square_f32(k), threads=128, group=8)
        q_scale = np.float32(
            np.float32(1.0 / max(math.sqrt(float(q_sum)), 1.0e-6))
            * inv_sqrt_head_k
        )
        k_scale = np.float32(1.0 / max(math.sqrt(float(k_sum)), 1.0e-6))
        beta_value = sigmoid_f32(beta32[v_head])
        _decay = decay_f32(alpha32[v_head], dt_bias32[v_head], a_log32[v_head])
        head_raw = np.empty((int(head_v_dim),), dtype=np.float32)
        for value_idx in range(int(head_v_dim)):
            delta = np.float32(value[value_idx] * beta_value)
            out_acc = np.float32(0.0)
            tail_start = int(head_k_dim) & ~7
            for idx in range(0, tail_start, 8):
                group_acc = np.float32(0.0)
                for offset in range(8):
                    q_norm = np.float32(q[idx + offset] * q_scale)
                    k_norm = np.float32(k[idx + offset] * k_scale)
                    new_state = np.float32(k_norm * delta)
                    group_acc = np.float32(group_acc + np.float32(q_norm * new_state))
                out_acc = np.float32(out_acc + group_acc)
            for idx in range(tail_start, int(head_k_dim)):
                q_norm = np.float32(q[idx] * q_scale)
                k_norm = np.float32(k[idx] * k_scale)
                new_state = np.float32(k_norm * delta)
                out_acc = np.float32(out_acc + np.float32(q_norm * new_state))
            head_raw[value_idx] = out_acc
        norm_sum = tree_reduce_sum(square_f32(head_raw), threads=128, group=1)
        inv_rms = np.float32(
            1.0 / math.sqrt(float(norm_sum / np.float32(head_v_dim) + np.float32(eps)))
        )
        head_base = v_head * int(head_v_dim)
        for value_idx in range(int(head_v_dim)):
            gated = np.float32(
                head_raw[value_idx]
                * inv_rms
                * norm32[value_idx]
                * silu_f32(gate32[head_base + value_idx])
            )
            out[head_base + value_idx] = gated
    return out


def square_f32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.asarray(arr * arr, dtype=np.float32)


def tree_reduce_sum(values: np.ndarray, *, threads: int, group: int) -> np.float32:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    partial = np.zeros((threads,), dtype=np.float32)
    for tid in range(threads):
        local = np.float32(0.0)
        start = tid * group
        while start < arr.shape[0]:
            stop = min(start + group, arr.shape[0])
            for idx in range(start, stop):
                local = np.float32(local + arr[idx])
            start += threads * group
        partial[tid] = local
    stride = threads // 2
    while stride > 0:
        for tid in range(stride):
            partial[tid] = np.float32(partial[tid] + partial[tid + stride])
        stride //= 2
    return np.float32(partial[0])


def silu_array_f32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    rounded = np.asarray(
        [silu_f32(value) for value in arr.reshape(-1)],
        dtype=np.float32,
    )
    return rounded.reshape(arr.shape)


def silu_f32(value: float | np.float32) -> np.float32:
    value32 = np.float32(value)
    return np.float32(value32 / np.float32(1.0 + np.exp(np.float32(-value32))))


def sigmoid_f32(value: float | np.float32) -> np.float32:
    value32 = np.float32(value)
    return np.float32(1.0 / np.float32(1.0 + np.exp(np.float32(-value32))))


def softplus_f32(value: float | np.float32) -> np.float32:
    value32 = np.float32(value)
    if value32 > np.float32(20.0):
        return value32
    return np.float32(np.log1p(np.exp(value32)))


def decay_f32(alpha: np.float32, dt_bias: np.float32, a_log: np.float32) -> np.float32:
    return np.float32(
        np.exp(
            np.float32(
                -np.exp(np.float32(a_log)) * softplus_f32(np.float32(alpha + dt_bias))
            )
        )
    )


def load_conv_gdn_weights(
    model_path: Path,
    layer_id: int,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    reader = GGUFReader(model_path)
    specs = {
        "ssm_conv1d": "ssm_conv1d.weight",
        "ssm_dt_bias": "ssm_dt.bias",
        "ssm_norm": "ssm_norm.weight",
        "ssm_out": "ssm_out.weight",
    }
    weights: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for slot, suffix in specs.items():
        tensor_name = f"blk.{layer_id}.{suffix}"
        tensor = reader.tensor_info(tensor_name)
        values = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
        weights[slot] = (values, tensor_metadata(tensor_name, tensor, values))
    raw_a_name = f"blk.{layer_id}.ssm_a"
    raw_a_tensor = reader.tensor_info(raw_a_name)
    raw_a = np.asarray(reader.dequantize_tensor(raw_a_name), dtype=np.float32)
    a_log = np.asarray(_gguf_ssm_a_to_kernel_a_log(raw_a), dtype=np.float32)
    weights["ssm_a_log"] = (
        a_log,
        tensor_metadata(raw_a_name, raw_a_tensor, a_log)
        | {
            "source_transform": "log(-GGUF blk.*.ssm_a)",
            "source_summary": summarize_array(raw_a),
            "source_sha256": sha256_float32(raw_a),
        },
    )
    return weights


def tensor_metadata(tensor_name: str, tensor: Any, values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "tensor_name": tensor_name,
        "ggml_type": tensor.ggml_type_name,
        "shape": list(arr.shape),
        "summary": summarize_array(arr.reshape(-1)),
        "sha256": sha256_float32(arr.reshape(-1)),
    }


def classify_field_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "field_oracle_unavailable"
    if delta.get("exact_match"):
        return "field_matches_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "field_matches_oracle_within_tolerance"
    return "field_mismatch_after_position0_oracle"


def classify_position0_oracle(oracle_results: Mapping[str, Any]) -> str:
    classes = [oracle_results[name]["classification"] for name in FIELD_ORDER]
    if all(item == "field_matches_oracle_exactly" for item in classes):
        return "layer0_position0_conv_gdn_matches_oracle_exactly"
    if all(item.startswith("field_matches_oracle") for item in classes):
        return "layer0_position0_conv_gdn_matches_oracle_within_tolerance"
    mismatched = [
        name for name in FIELD_ORDER if "mismatch" in oracle_results[name]["classification"]
    ]
    if mismatched:
        return "layer0_position0_conv_gdn_mismatch_after_oracle"
    return "layer0_position0_conv_gdn_oracle_unavailable"


def status_from_classification(classification: str) -> str:
    if classification.endswith("unavailable"):
        return "unavailable"
    if "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer0_position0_conv_gdn_matches_oracle_exactly",
        "layer0_position0_conv_gdn_matches_oracle_within_tolerance",
    }:
        return "extend_conv_gdn_oracle_to_warm_position16_replay_or_state_capture"
    if classification == "layer0_position0_conv_gdn_mismatch_after_oracle":
        return "inspect_first_mismatched_position0_conv_gdn_field"
    return "rerun_position0_conv_gdn_oracle_on_rocm_host"


def summarize_weights(weights: Mapping[str, tuple[np.ndarray, dict[str, Any]]]) -> dict[str, Any]:
    return {name: metadata for name, (_values, metadata) in weights.items()}


def summarize_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    if capture.get("status") != "captured":
        return {"status": capture.get("status"), "reason": capture.get("reason")}
    return {
        "status": "captured",
        "summary": capture.get("summary"),
        "field_summaries": {
            name: summarize_array(np.asarray(values, dtype=np.float32))
            for name, values in capture.get("fields", {}).items()
        },
    }


def unavailable_artifact(
    *,
    projection_artifact_path: Path,
    plan_path: Path,
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
        "kind": "layer0_position0_bf16_contracted_conv_gdn_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer0_position0_conv_gdn_oracle_unavailable",
        "projection_artifact_path": str(projection_artifact_path),
        "plan_path": str(plan_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "position": int(position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_position0_conv_gdn_oracle_on_rocm_host",
    }


if __name__ == "__main__":
    main()
