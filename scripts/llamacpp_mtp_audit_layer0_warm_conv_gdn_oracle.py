#!/usr/bin/env python3
"""Replay warm layer-0 conv/GDN state to the original position-16 boundary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.quant.gguf import dequantize_gguf_data  # noqa: E402
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    bf16_roundtrip_array,
    delta_summary,
    rmsnorm_f32,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_position0_conv_gdn_oracle import (  # noqa: E402
    FIELD_ORDER,
    decay_f32,
    load_conv_gdn_weights,
    sigmoid_f32,
    silu_array_f32,
    silu_f32,
    square_f32,
    summarize_capture,
    summarize_weights,
    tree_reduce_sum,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    capture_hipengine_linear_boundary,
    load_attn_norm_weight,
    load_projection_weights,
    project_f32,
)

DEFAULT_PROJECTION_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter323-layer0-bf16-projection-oracle.json"
)
DEFAULT_PLAN = Path("benchmarks/results/mtp-gguf-iter324-layer0-conv-gdn-plan.json")
DEFAULT_POSITION0 = Path(
    "benchmarks/results/mtp-gguf-iter325-layer0-position0-conv-gdn-oracle.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter326-layer0-warm-conv-gdn-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

BoundaryCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
ReplayInputBuilder = Callable[
    [Path, tuple[int, ...], int, int],
    tuple[list[dict[str, np.ndarray]], dict[str, Any]],
]
ConvGDNWeightLoader = Callable[[Path, int], dict[str, tuple[np.ndarray, dict[str, Any]]]]

INPUT_FIELDS = (
    "attn_norm_f32",
    "linear_qkv_f32",
    "linear_z_f32",
    "ssm_alpha_f32",
    "ssm_beta_f32",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-artifact", type=Path, default=DEFAULT_PROJECTION_ARTIFACT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--position0-artifact", type=Path, default=DEFAULT_POSITION0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--input-atol", type=float, default=2.5e-4)
    parser.add_argument("--conv-atol", type=float, default=2.5e-4)
    parser.add_argument("--recurrent-atol", type=float, default=1.0e-3)
    parser.add_argument("--bf16-atol", type=float, default=2.5e-4)
    parser.add_argument("--attn-out-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=326)
    args = parser.parse_args()

    artifact = audit_layer0_warm_conv_gdn_oracle(
        projection_artifact_path=args.projection_artifact,
        plan_path=args.plan,
        position0_artifact_path=args.position0_artifact,
        model_path=args.model,
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
                "target_input_classification": artifact["target_input_classification"],
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


def audit_layer0_warm_conv_gdn_oracle(
    *,
    projection_artifact_path: Path,
    plan_path: Path,
    position0_artifact_path: Path,
    model_path: Path | None = None,
    max_sequence_length: int | None = None,
    input_atol: float = 2.5e-4,
    conv_atol: float = 2.5e-4,
    recurrent_atol: float = 1.0e-3,
    bf16_atol: float = 2.5e-4,
    attn_out_atol: float = 2.5e-4,
    iteration: int = 326,
    boundary_capture_fn: BoundaryCaptureFn | None = None,
    replay_input_builder: ReplayInputBuilder | None = None,
    conv_gdn_weight_loader: ConvGDNWeightLoader | None = None,
) -> dict[str, Any]:
    projection = json.loads(projection_artifact_path.read_text())
    plan = json.loads(plan_path.read_text())
    position0 = json.loads(position0_artifact_path.read_text())
    validate_inputs(projection, plan, position0)
    resolved_model = Path(model_path or projection["model"])
    layer_id = int(projection["layer_id"])
    target_position = int(projection["position"])
    prompt_tokens = tuple(int(token) for token in projection["prompt_tokens"])
    token_id = int(prompt_tokens[target_position])
    capture_fn = boundary_capture_fn or capture_hipengine_linear_boundary
    target_capture = capture_fn(
        resolved_model,
        prompt_tokens,
        target_position,
        layer_id,
        max_sequence_length,
    )
    if target_capture.get("status") != "captured":
        return unavailable_artifact(
            projection_artifact_path=projection_artifact_path,
            plan_path=plan_path,
            position0_artifact_path=position0_artifact_path,
            model_path=resolved_model,
            layer_id=layer_id,
            target_position=target_position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=target_capture,
            iteration=iteration,
        )
    input_builder = replay_input_builder or build_layer0_projection_inputs_for_prompt
    replay_inputs, input_metadata = input_builder(
        resolved_model,
        prompt_tokens[: target_position + 1],
        layer_id,
        target_position,
    )
    weight_loader = conv_gdn_weight_loader or load_conv_gdn_weights
    conv_gdn_weights = weight_loader(resolved_model, layer_id)
    tolerances = {
        "conv_out_f32": float(conv_atol),
        "recurrent_out_f32": float(recurrent_atol),
        "recurrent_bf16_f32": float(bf16_atol),
        "attn_out_f32": float(attn_out_atol),
    }
    target_input_results = compare_target_inputs(
        replay_inputs[target_position],
        target_capture,
        near_atol=float(input_atol),
    )
    oracle_results, replay_summary = build_warm_oracle_results(
        target_capture=target_capture,
        replay_inputs=replay_inputs,
        weights=conv_gdn_weights,
        dimensions=plan["model_metadata"]["dimensions"],
        eps=float(plan["model_metadata"].get("rms_norm_eps", 1.0e-6)),
        target_position=target_position,
        tolerances=tolerances,
    )
    target_input_classification = classify_target_inputs(target_input_results)
    classification = classify_warm_oracle(target_input_classification, oracle_results)
    return {
        "schema": 1,
        "kind": "layer0_warm_bf16_contracted_conv_gdn_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "projection_artifact_path": str(projection_artifact_path),
        "plan_path": str(plan_path),
        "position0_artifact_path": str(position0_artifact_path),
        "model": str(resolved_model),
        "layer_id": layer_id,
        "target_position": target_position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "replay_contract": {
            "source": (
                "GGUF token_embedding rows -> BF16 attn_norm/projections -> "
                "CPU conv/GDN replay"
            ),
            "replayed_positions": list(range(target_position + 1)),
            "target_capture_mode": "hipEngine resident capture after stepping prior tokens",
            "starts_from_zero_state": True,
            "conv_state_floats": plan["model_metadata"]["dimensions"]["conv_state_floats"],
            "recurrent_state_floats": plan["model_metadata"]["dimensions"][
                "recurrent_state_floats"
            ],
        },
        "model_dimensions": plan["model_metadata"]["dimensions"],
        "input_metadata": input_metadata,
        "weights": summarize_weights(conv_gdn_weights),
        "target_input_results": target_input_results,
        "target_input_classification": target_input_classification,
        "hipengine_capture": summarize_capture(target_capture),
        "replay_summary": replay_summary,
        "oracle_results": oracle_results,
        "tolerances": {"input": float(input_atol), **tolerances},
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_inputs(
    projection: Mapping[str, Any],
    plan: Mapping[str, Any],
    position0: Mapping[str, Any],
) -> None:
    if projection.get("status") != "ready":
        raise ValueError("projection artifact must be ready")
    if not str(projection.get("classification", "")).startswith("layer0_projections_match"):
        raise ValueError("projection artifact must have matched layer-0 projections")
    if plan.get("status") != "ready":
        raise ValueError("conv/GDN plan must be ready")
    if position0.get("status") != "ready":
        raise ValueError("position-0 conv/GDN artifact must be ready")
    if not str(position0.get("classification", "")).startswith(
        "layer0_position0_conv_gdn_matches"
    ):
        raise ValueError("position-0 conv/GDN oracle must have matched")


def build_layer0_projection_inputs_for_prompt(
    model_path: Path,
    prompt_tokens: tuple[int, ...],
    layer_id: int,
    _target_position: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    reader = GGUFReader(model_path)
    embedding_rows, embedding_meta = load_token_embedding_rows(reader, prompt_tokens)
    norm_weight, eps, norm_meta = load_attn_norm_weight(model_path, layer_id)
    projection_weights = load_projection_weights(model_path, layer_id)
    alpha_beta_weights = load_alpha_beta_weights(model_path, layer_id)
    inputs: list[dict[str, np.ndarray]] = []
    for row in embedding_rows:
        hidden_bf16 = bf16_roundtrip_array(np.asarray(row, dtype=np.float32))
        attn_norm = bf16_roundtrip_array(rmsnorm_f32(hidden_bf16, norm_weight, eps))
        linear_qkv = bf16_roundtrip_array(
            project_f32(attn_norm, projection_weights["attn_qkv"][0])
        )
        linear_z = bf16_roundtrip_array(
            project_f32(attn_norm, projection_weights["attn_gate"][0])
        )
        alpha = bf16_roundtrip_array(project_f32(attn_norm, alpha_beta_weights["alpha"][0]))
        beta = bf16_roundtrip_array(project_f32(attn_norm, alpha_beta_weights["beta"][0]))
        inputs.append(
            {
                "attn_norm_f32": attn_norm,
                "linear_qkv_f32": linear_qkv,
                "linear_z_f32": linear_z,
                "ssm_alpha_f32": alpha,
                "ssm_beta_f32": beta,
            }
        )
    metadata = {
        "token_embedding": embedding_meta,
        "attn_norm": norm_meta,
        "projection_weights": {
            name: meta for name, (_values, meta) in projection_weights.items()
        },
        "alpha_beta_weights": {
            name: meta for name, (_values, meta) in alpha_beta_weights.items()
        },
        "position_count": int(len(inputs)),
        "first_position_summary": summarize_input_record(0, prompt_tokens[0], inputs[0]),
        "target_position_summary": summarize_input_record(
            _target_position,
            prompt_tokens[_target_position],
            inputs[_target_position],
        ),
        "stack_sha256": {
            field: sha256_float32(
                np.asarray([record[field] for record in inputs], dtype=np.float32).reshape(-1)
            )
            for field in INPUT_FIELDS
        },
    }
    return inputs, metadata


def summarize_input_record(
    position: int,
    token_id: int,
    record: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "position": int(position),
        "token_id": int(token_id),
        "field_summaries": {
            field: summarize_array(np.asarray(record[field], dtype=np.float32))
            for field in INPUT_FIELDS
        },
    }


def load_token_embedding_rows(
    reader: GGUFReader,
    token_ids: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    tensor_name = "token_embd.weight"
    tensor = reader.tensor_info(tensor_name)
    raw = reader.tensor_data(tensor_name)
    token_array = np.asarray([int(token) for token in token_ids], dtype=np.int64)
    if np.any(token_array < 0) or np.any(token_array >= int(tensor.shape[0])):
        raise ValueError("token id outside token_embd.weight rows")
    row_storage = np.asarray(raw[token_array])
    rows = np.asarray(dequantize_gguf_data(row_storage, tensor.ggml_type), dtype=np.float32)
    rows = rows.reshape((len(token_array), int(tensor.shape[1])))
    return rows, {
        "tensor_name": tensor_name,
        "ggml_type": tensor.ggml_type_name,
        "shape": list(tensor.shape),
        "selected_token_count": int(len(token_array)),
        "selected_token_ids": [int(token) for token in token_array.tolist()],
        "selected_rows_summary": summarize_array(rows.reshape(-1)),
        "selected_rows_sha256": sha256_float32(rows.reshape(-1)),
    }


def load_alpha_beta_weights(
    model_path: Path,
    layer_id: int,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    reader = GGUFReader(model_path)
    weights: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for slot, suffix in (("alpha", "ssm_alpha.weight"), ("beta", "ssm_beta.weight")):
        tensor_name = f"blk.{layer_id}.{suffix}"
        tensor = reader.tensor_info(tensor_name)
        values = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
        weights[slot] = (
            values,
            {
                "tensor_name": tensor_name,
                "ggml_type": tensor.ggml_type_name,
                "shape": list(values.shape),
                "summary": summarize_array(values.reshape(-1)),
                "sha256": sha256_float32(values.reshape(-1)),
            },
        )
    return weights


def build_warm_oracle_results(
    *,
    target_capture: Mapping[str, Any],
    replay_inputs: Sequence[Mapping[str, np.ndarray]],
    weights: Mapping[str, tuple[np.ndarray, dict[str, Any]]],
    dimensions: Mapping[str, Any],
    eps: float,
    target_position: int,
    tolerances: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay_records, replay_summary = replay_conv_gdn_sequence(
        replay_inputs=replay_inputs,
        weights=weights,
        dimensions=dimensions,
        eps=eps,
    )
    target_record = replay_records[int(target_position)]
    fields = target_capture["fields"]
    results: dict[str, Any] = {}
    for field in FIELD_ORDER:
        oracle = np.asarray(target_record[field], dtype=np.float32)
        hip = np.asarray(fields[field], dtype=np.float32)
        delta = delta_summary(oracle, hip)
        results[field] = {
            "field": field,
            "oracle": target_record["oracle_descriptions"][field],
            "oracle_summary": summarize_array(oracle),
            "hipengine_summary": summarize_array(hip),
            "delta_oracle_vs_hip": delta,
            "near_atol": float(tolerances[field]),
            "classification": classify_field_delta(delta, near_atol=float(tolerances[field])),
        }
    replay_summary = dict(replay_summary)
    replay_summary["target_position_summary"] = summarize_replay_record(target_record)
    return results, replay_summary


def replay_conv_gdn_sequence(
    *,
    replay_inputs: Sequence[Mapping[str, np.ndarray]],
    weights: Mapping[str, tuple[np.ndarray, dict[str, Any]]],
    dimensions: Mapping[str, Any],
    eps: float,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    conv_weight = weights["ssm_conv1d"][0]
    ssm_out_weight = weights["ssm_out"][0]
    conv_state = np.zeros(
        (int(dimensions["linear_qkv_width"]), int(dimensions["ssm_conv_kernel"])),
        dtype=np.float32,
    )
    recurrent_state = np.zeros(
        (
            int(dimensions["ssm_time_step_rank"]),
            int(dimensions["ssm_state_size"]),
            int(dimensions["ssm_value_dim"]),
        ),
        dtype=np.float32,
    )
    records: list[dict[str, np.ndarray]] = []
    for position, record in enumerate(replay_inputs):
        conv_out = conv_decode_step(record["linear_qkv_f32"], conv_weight, conv_state)
        recurrent_out = gdn_recurrent_step(
            conv_out=conv_out,
            gate=record["linear_z_f32"],
            alpha=record["ssm_alpha_f32"],
            beta=record["ssm_beta_f32"],
            dt_bias=weights["ssm_dt_bias"][0],
            a_log=weights["ssm_a_log"][0],
            norm_weight=weights["ssm_norm"][0],
            recurrent_state=recurrent_state,
            eps=float(eps),
            num_k_heads=int(dimensions["ssm_group_count"]),
            num_v_heads=int(dimensions["ssm_time_step_rank"]),
            head_k_dim=int(dimensions["ssm_state_size"]),
            head_v_dim=int(dimensions["ssm_value_dim"]),
        )
        recurrent_bf16 = bf16_roundtrip_array(recurrent_out)
        attn_out = bf16_roundtrip_array(project_f32(recurrent_bf16, ssm_out_weight))
        records.append(
            {
                "position": np.asarray([position], dtype=np.int64),
                "conv_out_f32": conv_out,
                "recurrent_out_f32": recurrent_out,
                "recurrent_bf16_f32": recurrent_bf16,
                "attn_out_f32": attn_out,
                "oracle_descriptions": {
                    "conv_out_f32": "stateful conv decode replay from BF16 linear_qkv",
                    "recurrent_out_f32": (
                        "stateful GDN recurrent RMSNorm+gate replay before BF16 cast"
                    ),
                    "recurrent_bf16_f32": "BF16(recurrent_out_f32) copied back to F32",
                    "attn_out_f32": "BF16(ssm_out(Q8_0) @ recurrent_bf16)",
                },
            }
        )
    summary = {
        "replayed_positions": int(len(replay_inputs)),
        "final_conv_state_summary": summarize_array(conv_state.reshape(-1)),
        "final_recurrent_state_summary": summarize_array(recurrent_state.reshape(-1)),
        "final_conv_state_sha256": sha256_float32(conv_state.reshape(-1)),
        "final_recurrent_state_sha256": sha256_float32(recurrent_state.reshape(-1)),
    }
    return records, summary


def conv_decode_step(
    linear_qkv: np.ndarray,
    conv_weight: np.ndarray,
    conv_state: np.ndarray,
) -> np.ndarray:
    x = np.asarray(linear_qkv, dtype=np.float32).reshape(-1)
    w = np.asarray(conv_weight, dtype=np.float32)
    state = np.asarray(conv_state, dtype=np.float32)
    if w.ndim != 2 or state.shape != w.shape or w.shape[0] != x.shape[0]:
        raise ValueError(
            f"conv shape mismatch: input={x.shape}, weight={w.shape}, state={state.shape}"
        )
    acc = np.zeros((x.shape[0],), dtype=np.float32)
    for idx in range(w.shape[1] - 1):
        value = state[:, idx + 1].copy()
        acc = np.asarray(acc + np.asarray(value * w[:, idx], dtype=np.float32), dtype=np.float32)
        state[:, idx] = value
    newest = x
    acc = np.asarray(
        acc + np.asarray(newest * w[:, w.shape[1] - 1], dtype=np.float32),
        dtype=np.float32,
    )
    state[:, w.shape[1] - 1] = newest
    return silu_array_f32(acc)


def gdn_recurrent_step(
    *,
    conv_out: np.ndarray,
    gate: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    norm_weight: np.ndarray,
    recurrent_state: np.ndarray,
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
    state = np.asarray(recurrent_state, dtype=np.float32)
    key_dim = int(num_k_heads) * int(head_k_dim)
    value_dim = int(num_v_heads) * int(head_v_dim)
    expected_state = (int(num_v_heads), int(head_k_dim), int(head_v_dim))
    if conv.shape[0] != 2 * key_dim + value_dim:
        raise ValueError("conv_out width does not match Q/K/V dimensions")
    if gate32.shape[0] != value_dim:
        raise ValueError("gate width does not match value dimensions")
    if state.shape != expected_state:
        raise ValueError(f"recurrent_state shape mismatch: {state.shape} != {expected_state}")
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
        decay = decay_f32(alpha32[v_head], dt_bias32[v_head], a_log32[v_head])
        head_raw = np.empty((int(head_v_dim),), dtype=np.float32)
        tail_start = int(head_k_dim) & ~7
        for value_idx in range(int(head_v_dim)):
            previous = state[v_head, :, value_idx].copy()
            kv_mem = np.float32(0.0)
            for idx in range(0, tail_start, 8):
                for offset in range(8):
                    k_norm = np.float32(k[idx + offset] * k_scale)
                    state_value = np.float32(previous[idx + offset] * decay)
                    kv_mem = np.float32(kv_mem + np.float32(k_norm * state_value))
            for idx in range(tail_start, int(head_k_dim)):
                k_norm = np.float32(k[idx] * k_scale)
                state_value = np.float32(previous[idx] * decay)
                kv_mem = np.float32(kv_mem + np.float32(k_norm * state_value))
            delta = np.float32(np.float32(value[value_idx] - kv_mem) * beta_value)
            out_acc = np.float32(0.0)
            for idx in range(0, tail_start, 8):
                group_acc = np.float32(0.0)
                for offset in range(8):
                    flat_idx = idx + offset
                    q_norm = np.float32(q[flat_idx] * q_scale)
                    k_norm = np.float32(k[flat_idx] * k_scale)
                    new_state = np.float32(previous[flat_idx] * decay + k_norm * delta)
                    state[v_head, flat_idx, value_idx] = new_state
                    group_acc = np.float32(group_acc + np.float32(q_norm * new_state))
                out_acc = np.float32(out_acc + group_acc)
            for idx in range(tail_start, int(head_k_dim)):
                q_norm = np.float32(q[idx] * q_scale)
                k_norm = np.float32(k[idx] * k_scale)
                new_state = np.float32(previous[idx] * decay + k_norm * delta)
                state[v_head, idx, value_idx] = new_state
                out_acc = np.float32(out_acc + np.float32(q_norm * new_state))
            head_raw[value_idx] = out_acc
        norm_sum = tree_reduce_sum(square_f32(head_raw), threads=128, group=1)
        inv_rms = np.float32(
            1.0 / math.sqrt(float(norm_sum / np.float32(head_v_dim) + np.float32(eps)))
        )
        head_base = v_head * int(head_v_dim)
        for value_idx in range(int(head_v_dim)):
            out[head_base + value_idx] = np.float32(
                head_raw[value_idx]
                * inv_rms
                * norm32[value_idx]
                * silu_f32(gate32[head_base + value_idx])
            )
    return out


def compare_target_inputs(
    replay_target_input: Mapping[str, np.ndarray],
    target_capture: Mapping[str, Any],
    *,
    near_atol: float,
) -> dict[str, Any]:
    fields = target_capture["fields"]
    results: dict[str, Any] = {}
    for field in INPUT_FIELDS:
        oracle = np.asarray(replay_target_input[field], dtype=np.float32)
        hip = np.asarray(fields[field], dtype=np.float32)
        delta = delta_summary(oracle, hip)
        results[field] = {
            "field": field,
            "oracle_summary": summarize_array(oracle),
            "hipengine_summary": summarize_array(hip),
            "delta_oracle_vs_hip": delta,
            "near_atol": float(near_atol),
            "classification": classify_field_delta(delta, near_atol=float(near_atol)),
        }
    return results


def classify_field_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "warm_field_oracle_unavailable"
    if delta.get("exact_match"):
        return "warm_field_matches_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "warm_field_matches_oracle_within_tolerance"
    return "warm_field_mismatch_after_replay_oracle"


def classify_target_inputs(target_input_results: Mapping[str, Any]) -> str:
    classes = [target_input_results[name]["classification"] for name in INPUT_FIELDS]
    if all(item == "warm_field_matches_oracle_exactly" for item in classes):
        return "target_inputs_match_replay_exactly"
    if all(item.startswith("warm_field_matches_oracle") for item in classes):
        return "target_inputs_match_replay_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "target_inputs_mismatch_before_conv_gdn_replay"
    return "target_inputs_unavailable"


def classify_warm_oracle(
    target_input_classification: str,
    oracle_results: Mapping[str, Any],
) -> str:
    if target_input_classification in {
        "target_inputs_mismatch_before_conv_gdn_replay",
        "target_inputs_unavailable",
    }:
        return "layer0_warm_conv_gdn_blocked_target_input_mismatch"
    classes = [oracle_results[name]["classification"] for name in FIELD_ORDER]
    if all(item == "warm_field_matches_oracle_exactly" for item in classes):
        return "layer0_warm_conv_gdn_matches_oracle_exactly"
    if all(item.startswith("warm_field_matches_oracle") for item in classes):
        return "layer0_warm_conv_gdn_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer0_warm_conv_gdn_mismatch_after_replay_oracle"
    return "layer0_warm_conv_gdn_oracle_unavailable"


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
        "layer0_warm_conv_gdn_matches_oracle_exactly",
        "layer0_warm_conv_gdn_matches_oracle_within_tolerance",
    }:
        return "continue_layer0_bisection_after_attn_out_or_residual"
    if classification == "layer0_warm_conv_gdn_blocked_target_input_mismatch":
        return "audit_warm_target_projection_inputs_from_token_embedding"
    if classification == "layer0_warm_conv_gdn_mismatch_after_replay_oracle":
        return "inspect_first_warm_conv_gdn_replay_mismatch"
    return "rerun_warm_conv_gdn_oracle_on_rocm_host"


def summarize_replay_record(record: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        field: summarize_array(np.asarray(record[field], dtype=np.float32))
        for field in FIELD_ORDER
    }


def unavailable_artifact(
    *,
    projection_artifact_path: Path,
    plan_path: Path,
    position0_artifact_path: Path,
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
        "kind": "layer0_warm_bf16_contracted_conv_gdn_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer0_warm_conv_gdn_oracle_unavailable",
        "projection_artifact_path": str(projection_artifact_path),
        "plan_path": str(plan_path),
        "position0_artifact_path": str(position0_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "target_position": int(target_position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "target_input_classification": "target_inputs_unavailable",
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_warm_conv_gdn_oracle_on_rocm_host",
    }


if __name__ == "__main__":
    main()
