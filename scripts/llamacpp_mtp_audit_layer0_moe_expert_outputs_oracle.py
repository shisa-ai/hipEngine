#!/usr/bin/env python3
"""Audit layer-0 MoE selected/shared expert outputs from verified router state."""

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

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.quant.gguf import dequantize_gguf_data  # noqa: E402
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_moe_router_oracle import (  # noqa: E402
    bf16_roundtrip_like,
    sigmoid_f32,
)
from scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle import (  # noqa: E402
    capture_hipengine_linear_layer,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)

DEFAULT_ROUTER_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter328-layer0-moe-router-oracle.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter329-layer0-moe-expert-outputs-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
ExpertWeightLoader = Callable[
    [Path, int, np.ndarray],
    tuple[dict[str, np.ndarray], dict[str, Any]],
]

EXPERT_COMPARE_FIELDS = (
    "ffn_or_moe_down_f32",
    "moe_shared_out_f32",
    "layer_out_f32",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-artifact", type=Path, default=DEFAULT_ROUTER_ARTIFACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--selected-atol", type=float, default=0.0)
    parser.add_argument("--shared-atol", type=float, default=0.0)
    parser.add_argument("--layer-out-atol", type=float, default=2.5e-4)
    parser.add_argument("--iteration", type=int, default=329)
    args = parser.parse_args()

    artifact = audit_layer0_moe_expert_outputs_oracle(
        router_artifact_path=args.router_artifact,
        model_path=args.model,
        max_sequence_length=args.max_sequence_length,
        selected_atol=args.selected_atol,
        shared_atol=args.shared_atol,
        layer_out_atol=args.layer_out_atol,
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
                "field_classifications": {
                    name: artifact["oracle_results"][name]["classification"]
                    for name in EXPERT_COMPARE_FIELDS
                    if name in artifact.get("oracle_results", {})
                },
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer0_moe_expert_outputs_oracle(
    *,
    router_artifact_path: Path,
    model_path: Path | None = None,
    max_sequence_length: int | None = None,
    selected_atol: float = 0.0,
    shared_atol: float = 0.0,
    layer_out_atol: float = 2.5e-4,
    iteration: int = 329,
    layer_capture_fn: LayerCaptureFn | None = None,
    expert_weight_loader: ExpertWeightLoader | None = None,
) -> dict[str, Any]:
    router_artifact = json.loads(router_artifact_path.read_text())
    validate_router_artifact(router_artifact)
    resolved_model = Path(model_path or router_artifact["model"])
    layer_id = int(router_artifact["layer_id"])
    target_position = int(router_artifact["target_position"])
    prompt_tokens = tuple(int(token) for token in router_artifact["prompt_tokens"])
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
            router_artifact_path=router_artifact_path,
            model_path=resolved_model,
            layer_id=layer_id,
            target_position=target_position,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            capture=capture,
            iteration=iteration,
        )
    if not capture["summary"].get("is_moe", False):
        raise ValueError("layer capture is not MoE-enabled")
    selected = np.asarray(capture["fields"]["moe_selected_experts_i64"], dtype=np.int64)
    router_selected = np.asarray(
        router_artifact["oracle_results"]["selected_experts_i64"]["oracle_values"],
        dtype=np.int64,
    )
    if selected.shape != router_selected.shape or not np.array_equal(selected, router_selected):
        raise ValueError("current capture selected experts differ from router artifact")
    loader = expert_weight_loader or load_expert_output_weights
    weights, weight_metadata = loader(resolved_model, layer_id, selected)
    oracle = compute_moe_expert_outputs_oracle(
        post_norm=np.asarray(capture["fields"]["post_norm_f32"], dtype=np.float32),
        residual=np.asarray(capture["fields"]["residual_f32"], dtype=np.float32),
        selected_experts=selected,
        routing_weights=np.asarray(
            capture["fields"]["moe_routing_weights_f32"],
            dtype=np.float32,
        ),
        shared_gate_logit=np.asarray(
            capture["fields"]["moe_shared_gate_f32"],
            dtype=np.float32,
        ),
        weights=weights,
    )
    oracle_results = compare_expert_output_oracle(
        oracle=oracle,
        capture=capture,
        selected_atol=float(selected_atol),
        shared_atol=float(shared_atol),
        layer_out_atol=float(layer_out_atol),
    )
    classification = classify_expert_output_oracle(oracle_results)
    return {
        "schema": 1,
        "kind": "layer0_moe_expert_outputs_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "router_artifact_path": str(router_artifact_path),
        "model": str(resolved_model),
        "layer_id": layer_id,
        "target_position": target_position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "verified post_norm_f32/router -> raw GGUF expert branches",
            "router_source": router_artifact.get("classification"),
            "selected_experts": [int(value) for value in selected.tolist()],
            "selected_branch": (
                "Q4_K gate/up BF16 GEMV -> BF16 SiLU*up -> Q5_K down BF16 GEMV"
            ),
            "shared_branch": "Q8_0 gate/up/down BF16 GEMV chain",
            "gemv_threads": 128,
            "combine": (
                "weighted selected branch is BF16-rounded before residual + "
                "sigmoid(shared_gate_logit) * shared_out"
            ),
        },
        "weights": weight_metadata,
        "hipengine_capture": summarize_capture(capture),
        "expert_oracle": summarize_expert_oracle(oracle),
        "oracle_results": oracle_results,
        "tolerances": {
            "ffn_or_moe_down_f32": float(selected_atol),
            "moe_shared_out_f32": float(shared_atol),
            "layer_out_f32": float(layer_out_atol),
        },
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_router_artifact(router: Mapping[str, Any]) -> None:
    if router.get("status") != "ready":
        raise ValueError("router artifact must be ready")
    if not str(router.get("classification", "")).startswith("layer0_moe_router_matches"):
        raise ValueError("router artifact must have matched")
    if router.get("next_action") != "audit_layer0_moe_selected_and_shared_expert_outputs":
        raise ValueError("router artifact must point to expert-output audit")


def load_expert_output_weights(
    model_path: Path,
    layer_id: int,
    selected_experts: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reader = GGUFReader(model_path)
    selected = np.asarray(selected_experts, dtype=np.int64)
    specs = {
        "selected_gate": (f"blk.{layer_id}.ffn_gate_exps.weight", selected),
        "selected_up": (f"blk.{layer_id}.ffn_up_exps.weight", selected),
        "selected_down": (f"blk.{layer_id}.ffn_down_exps.weight", selected),
        "shared_gate": (f"blk.{layer_id}.ffn_gate_shexp.weight", None),
        "shared_up": (f"blk.{layer_id}.ffn_up_shexp.weight", None),
        "shared_down": (f"blk.{layer_id}.ffn_down_shexp.weight", None),
    }
    weights: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for slot, (tensor_name, expert_index) in specs.items():
        tensor = reader.tensor_info(tensor_name)
        raw = reader.tensor_data(tensor_name)
        if expert_index is None:
            raw_slice = raw
            selection = "all_rows"
        else:
            raw_slice = raw[expert_index]
            selection = [int(value) for value in selected.tolist()]
        dequant = np.ascontiguousarray(
            dequantize_gguf_data(raw_slice, tensor.ggml_type),
            dtype=np.float32,
        )
        weights[slot] = dequant
        metadata[slot] = {
            "tensor_name": tensor_name,
            "source_ggml_type": tensor.ggml_type_name,
            "source_shape": list(tensor.shape),
            "loaded_shape": list(dequant.shape),
            "selection": selection,
            "loaded_summary": summarize_array(dequant.reshape(-1)),
            "loaded_sha256": sha256_float32(dequant.reshape(-1)),
            "contract": "dequantized selected/all raw GGUF rows for CPU oracle only",
        }
    return weights, metadata


def compute_moe_expert_outputs_oracle(
    *,
    post_norm: np.ndarray,
    residual: np.ndarray,
    selected_experts: np.ndarray,
    routing_weights: np.ndarray,
    shared_gate_logit: np.ndarray,
    weights: Mapping[str, np.ndarray],
    threads: int = 128,
) -> dict[str, np.ndarray]:
    hidden = np.asarray(post_norm, dtype=np.float32).reshape(-1)
    residual_f32 = np.asarray(residual, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected_experts, dtype=np.int64).reshape(-1)
    routing = np.asarray(routing_weights, dtype=np.float32).reshape(-1)
    if selected.shape != routing.shape:
        raise ValueError("selected experts and routing weights must have matching shape")

    selected_gate_w = np.asarray(weights["selected_gate"], dtype=np.float32)
    selected_up_w = np.asarray(weights["selected_up"], dtype=np.float32)
    selected_down_w = np.asarray(weights["selected_down"], dtype=np.float32)
    shared_gate_w = np.asarray(weights["shared_gate"], dtype=np.float32)
    shared_up_w = np.asarray(weights["shared_up"], dtype=np.float32)
    shared_down_w = np.asarray(weights["shared_down"], dtype=np.float32)
    top_k = int(selected.shape[0])
    if selected_gate_w.shape[:1] != (top_k,) or selected_up_w.shape[:1] != (top_k,):
        raise ValueError("selected gate/up weights must be loaded in top-k order")
    if selected_down_w.shape[:1] != (top_k,):
        raise ValueError("selected down weights must be loaded in top-k order")

    ffn_len = int(selected_gate_w.shape[1])
    gate_flat = kernel_order_dot_matrix(
        selected_gate_w.reshape(top_k * ffn_len, hidden.shape[0]),
        hidden,
        threads=threads,
    ).reshape(top_k, ffn_len)
    up_flat = kernel_order_dot_matrix(
        selected_up_w.reshape(top_k * ffn_len, hidden.shape[0]),
        hidden,
        threads=threads,
    ).reshape(top_k, ffn_len)
    selected_gate_bf16 = bf16_roundtrip_like(gate_flat)
    selected_up_bf16 = bf16_roundtrip_like(up_flat)
    selected_intermediate = silu_mul_bf16(selected_gate_bf16, selected_up_bf16)
    selected_down = np.empty((top_k, residual_f32.shape[0]), dtype=np.float32)
    for row in range(top_k):
        selected_down[row] = bf16_roundtrip_like(
            kernel_order_dot_matrix(
                selected_down_w[row],
                selected_intermediate[row],
                threads=threads,
            )
        )

    shared_gate = bf16_roundtrip_like(
        kernel_order_dot_matrix(shared_gate_w, hidden, threads=threads)
    )
    shared_up = bf16_roundtrip_like(
        kernel_order_dot_matrix(shared_up_w, hidden, threads=threads)
    )
    shared_intermediate = silu_mul_bf16(shared_gate, shared_up)
    shared_out = bf16_roundtrip_like(
        kernel_order_dot_matrix(shared_down_w, shared_intermediate, threads=threads)
    )

    selected_weighted = weighted_sum_bf16(selected_down, routing)
    gate = sigmoid_f32(np.asarray(shared_gate_logit, dtype=np.float32).reshape(1))[0]
    layer_value = np.float32(
        np.float32(residual_f32 + selected_weighted)
        + np.float32(np.float32(gate) * shared_out)
    )
    layer_out = bf16_roundtrip_like(layer_value)
    return {
        "selected_experts_i64": selected.astype(np.int64),
        "routing_weights_f32": routing.astype(np.float32),
        "selected_gate_f32": selected_gate_bf16.astype(np.float32),
        "selected_up_f32": selected_up_bf16.astype(np.float32),
        "selected_intermediate_f32": selected_intermediate.astype(np.float32),
        "selected_down_f32": selected_down.astype(np.float32),
        "selected_weighted_f32": selected_weighted.astype(np.float32),
        "shared_gate_f32": shared_gate.astype(np.float32),
        "shared_up_f32": shared_up.astype(np.float32),
        "shared_intermediate_f32": shared_intermediate.astype(np.float32),
        "shared_out_f32": shared_out.astype(np.float32),
        "shared_gate_sigmoid_f32": np.asarray([gate], dtype=np.float32),
        "layer_out_f32": layer_out.astype(np.float32),
    }


def kernel_order_dot_matrix(
    weight_rows: np.ndarray,
    x: np.ndarray,
    *,
    threads: int = 128,
) -> np.ndarray:
    weights = np.asarray(weight_rows, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError("weight_rows must be [rows, in_features]")
    act = np.asarray(x, dtype=np.float32).reshape(-1)
    if weights.shape[1] != act.shape[0]:
        raise ValueError("weight_rows width must match activation length")
    if threads <= 0 or threads % 32 != 0:
        raise ValueError("threads must be a positive multiple of 32")
    partial = np.zeros((int(threads), weights.shape[0]), dtype=np.float32)
    for tid in range(int(threads)):
        acc = np.zeros((weights.shape[0],), dtype=np.float32)
        for k in range(tid, act.shape[0], int(threads)):
            acc = np.float32(acc + np.float32(weights[:, k] * act[k]))
        partial[tid] = acc
    wave_sums = []
    for base in range(0, int(threads), 32):
        lanes = partial[base : base + 32].copy()
        for offset in (16, 8, 4, 2, 1):
            lanes[: 32 - offset] = np.float32(
                lanes[: 32 - offset] + lanes[offset:32]
            )
        wave_sums.append(lanes[0])
    total = np.zeros((weights.shape[0],), dtype=np.float32)
    for wave in wave_sums:
        total = np.float32(total + wave)
    return total.astype(np.float32)


def silu_mul_bf16(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    g = np.asarray(gate, dtype=np.float32)
    u = np.asarray(up, dtype=np.float32)
    sig = np.asarray(1.0 / (1.0 + np.exp(-g, dtype=np.float32)), dtype=np.float32)
    value = np.float32(np.float32(g * sig) * u)
    return bf16_roundtrip_like(value)


def weighted_sum_bf16(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if vals.ndim != 2 or vals.shape[0] != w.shape[0]:
        raise ValueError("values must be [rows, features] matching weights")
    acc = np.zeros((vals.shape[1],), dtype=np.float32)
    for row in range(vals.shape[0]):
        acc = np.float32(acc + np.float32(vals[row] * w[row]))
    return bf16_roundtrip_like(acc)


def compare_expert_output_oracle(
    *,
    oracle: Mapping[str, np.ndarray],
    capture: Mapping[str, Any],
    selected_atol: float,
    shared_atol: float,
    layer_out_atol: float,
) -> dict[str, Any]:
    fields = capture["fields"]
    hidden = int(capture["summary"]["hidden_size"])
    top_k = int(capture["summary"]["top_k"])
    selected_delta = delta_summary(
        np.asarray(oracle["selected_down_f32"], dtype=np.float32).reshape(-1),
        np.asarray(fields["ffn_or_moe_down_f32"], dtype=np.float32)
        .reshape(top_k, hidden)
        .reshape(-1),
    )
    shared_delta = delta_summary(
        np.asarray(oracle["shared_out_f32"], dtype=np.float32),
        np.asarray(fields["moe_shared_out_f32"], dtype=np.float32),
    )
    layer_delta = delta_summary(
        np.asarray(oracle["layer_out_f32"], dtype=np.float32),
        np.asarray(fields["layer_out_f32"], dtype=np.float32),
    )
    return {
        "ffn_or_moe_down_f32": {
            "field": "ffn_or_moe_down_f32",
            "oracle": "per-selected expert down output after BF16 gate/up SiLU chain",
            "oracle_summary": summarize_array(oracle["selected_down_f32"].reshape(-1)),
            "hipengine_summary": summarize_array(fields["ffn_or_moe_down_f32"].reshape(-1)),
            "delta_oracle_vs_hip": selected_delta,
            "near_atol": float(selected_atol),
            "classification": classify_delta(selected_delta, near_atol=float(selected_atol)),
        },
        "moe_shared_out_f32": {
            "field": "moe_shared_out_f32",
            "oracle": "shared expert down output after Q8_0 gate/up/down BF16 chain",
            "oracle_summary": summarize_array(oracle["shared_out_f32"]),
            "hipengine_summary": summarize_array(fields["moe_shared_out_f32"]),
            "delta_oracle_vs_hip": shared_delta,
            "near_atol": float(shared_atol),
            "classification": classify_delta(shared_delta, near_atol=float(shared_atol)),
        },
        "layer_out_f32": {
            "field": "layer_out_f32",
            "oracle": "residual + weighted selected BF16 + sigmoid(shared_gate) * shared BF16",
            "oracle_summary": summarize_array(oracle["layer_out_f32"]),
            "hipengine_summary": summarize_array(fields["layer_out_f32"]),
            "delta_oracle_vs_hip": layer_delta,
            "near_atol": float(layer_out_atol),
            "classification": classify_delta(layer_delta, near_atol=float(layer_out_atol)),
        },
    }


def classify_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "expert_output_oracle_unavailable"
    if delta.get("exact_match"):
        return "expert_output_matches_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "expert_output_matches_oracle_within_tolerance"
    return "expert_output_mismatch_after_oracle"


def classify_expert_output_oracle(oracle_results: Mapping[str, Any]) -> str:
    classes = [oracle_results[name]["classification"] for name in EXPERT_COMPARE_FIELDS]
    if all(item == "expert_output_matches_oracle_exactly" for item in classes):
        return "layer0_moe_expert_outputs_match_oracle_exactly"
    if all(item.startswith("expert_output_matches_oracle") for item in classes):
        return "layer0_moe_expert_outputs_match_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer0_moe_expert_outputs_mismatch_after_oracle"
    return "layer0_moe_expert_outputs_oracle_unavailable"


def status_from_classification(classification: str) -> str:
    if classification.endswith("unavailable"):
        return "unavailable"
    if "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer0_moe_expert_outputs_match_oracle_exactly",
        "layer0_moe_expert_outputs_match_oracle_within_tolerance",
    }:
        return "continue_layer0_bisection_after_moe_ffn_or_compare_layer_out"
    if classification == "layer0_moe_expert_outputs_mismatch_after_oracle":
        return "inspect_layer0_moe_expert_output_dtype_or_dequant_semantics"
    return "rerun_layer0_moe_expert_outputs_oracle_on_rocm_host"


def summarize_expert_oracle(oracle: Mapping[str, np.ndarray]) -> dict[str, Any]:
    selected_down = np.asarray(oracle["selected_down_f32"], dtype=np.float32)
    return {
        "selected_experts": [int(value) for value in oracle["selected_experts_i64"].tolist()],
        "routing_weights_summary": summarize_array(oracle["routing_weights_f32"]),
        "selected_gate_summary": summarize_array(oracle["selected_gate_f32"].reshape(-1)),
        "selected_up_summary": summarize_array(oracle["selected_up_f32"].reshape(-1)),
        "selected_intermediate_summary": summarize_array(
            oracle["selected_intermediate_f32"].reshape(-1)
        ),
        "selected_down_summary": summarize_array(selected_down.reshape(-1)),
        "selected_down_sha256": sha256_float32(selected_down.reshape(-1)),
        "selected_weighted_summary": summarize_array(oracle["selected_weighted_f32"]),
        "selected_weighted_sha256": sha256_float32(oracle["selected_weighted_f32"]),
        "shared_gate_summary": summarize_array(oracle["shared_gate_f32"]),
        "shared_up_summary": summarize_array(oracle["shared_up_f32"]),
        "shared_intermediate_summary": summarize_array(oracle["shared_intermediate_f32"]),
        "shared_out_summary": summarize_array(oracle["shared_out_f32"]),
        "shared_out_sha256": sha256_float32(oracle["shared_out_f32"]),
        "shared_gate_sigmoid_summary": summarize_array(oracle["shared_gate_sigmoid_f32"]),
        "layer_out_summary": summarize_array(oracle["layer_out_f32"]),
        "layer_out_sha256": sha256_float32(oracle["layer_out_f32"]),
    }


def unavailable_artifact(
    *,
    router_artifact_path: Path,
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
        "kind": "layer0_moe_expert_outputs_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer0_moe_expert_outputs_oracle_unavailable",
        "router_artifact_path": str(router_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "target_position": int(target_position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_layer0_moe_expert_outputs_oracle_on_rocm_host",
    }


if __name__ == "__main__":
    main()
