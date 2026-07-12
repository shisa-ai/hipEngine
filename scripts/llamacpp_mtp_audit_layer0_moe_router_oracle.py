#!/usr/bin/env python3
"""Audit layer-0 MoE router/top-k/shared-gate from verified post_norm."""

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
from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (  # noqa: E402
    bf16_roundtrip_array,
    delta_summary,
    sha256_float32,
    summarize_array,
)
from scripts.llamacpp_mtp_audit_layer0_post_attn_residual_oracle import (  # noqa: E402
    capture_hipengine_linear_layer,
)
from scripts.llamacpp_mtp_audit_layer0_projection_oracle import (  # noqa: E402
    summarize_capture,
)

DEFAULT_POST_ATTN_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter327-layer0-post-attn-residual-oracle.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter328-layer0-moe-router-oracle.json"
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

LayerCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
RouterWeightLoader = Callable[[Path, int], tuple[dict[str, np.ndarray], dict[str, Any]]]

ROUTER_COMPARE_FIELDS = (
    "selected_experts_i64",
    "routing_weights_f32",
    "shared_gate_logit_f32",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-attn-artifact", type=Path, default=DEFAULT_POST_ATTN_ARTIFACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--routing-atol", type=float, default=1.0e-6)
    parser.add_argument("--shared-gate-atol", type=float, default=1.0e-6)
    parser.add_argument("--iteration", type=int, default=328)
    args = parser.parse_args()

    artifact = audit_layer0_moe_router_oracle(
        post_attn_artifact_path=args.post_attn_artifact,
        model_path=args.model,
        max_sequence_length=args.max_sequence_length,
        routing_atol=args.routing_atol,
        shared_gate_atol=args.shared_gate_atol,
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
                    for name in ROUTER_COMPARE_FIELDS
                    if name in artifact.get("oracle_results", {})
                },
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer0_moe_router_oracle(
    *,
    post_attn_artifact_path: Path,
    model_path: Path | None = None,
    max_sequence_length: int | None = None,
    routing_atol: float = 1.0e-6,
    shared_gate_atol: float = 1.0e-6,
    iteration: int = 328,
    layer_capture_fn: LayerCaptureFn | None = None,
    router_weight_loader: RouterWeightLoader | None = None,
) -> dict[str, Any]:
    post_attn = json.loads(post_attn_artifact_path.read_text())
    validate_post_attn_artifact(post_attn)
    resolved_model = Path(model_path or post_attn["model"])
    layer_id = int(post_attn["layer_id"])
    target_position = int(post_attn["target_position"])
    prompt_tokens = tuple(int(token) for token in post_attn["prompt_tokens"])
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
            post_attn_artifact_path=post_attn_artifact_path,
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
    selected_weight_loader = router_weight_loader or load_router_weights
    weights, weight_metadata = selected_weight_loader(resolved_model, layer_id)
    top_k = int(capture["summary"]["top_k"])
    router = compute_router_oracle(
        np.asarray(capture["fields"]["post_norm_f32"], dtype=np.float32),
        weights["router_weight_bf16_f32"],
        weights["shared_gate_weight_bf16_f32"],
        top_k=top_k,
    )
    oracle_results = compare_router_oracle(
        router=router,
        capture=capture,
        routing_atol=float(routing_atol),
        shared_gate_atol=float(shared_gate_atol),
    )
    classification = classify_moe_router_oracle(oracle_results)
    return {
        "schema": 1,
        "kind": "layer0_moe_router_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_classification(classification),
        "classification": classification,
        "post_attn_artifact_path": str(post_attn_artifact_path),
        "model": str(resolved_model),
        "layer_id": layer_id,
        "target_position": target_position,
        "token_id": token_id,
        "prompt_tokens": list(prompt_tokens),
        "oracle_contract": {
            "source": "verified post_norm_f32 -> BF16-contracted MoE router/shared gate",
            "post_attn_source": post_attn.get("classification"),
            "router_weight_contract": "GGUF F32 rounded to resident BF16 before dot product",
            "top_k": top_k,
            "selection": "descending logits, softmax over top-k logits",
            "shared_gate_capture": "raw shared-gate logit; combine kernel applies sigmoid later",
        },
        "weights": weight_metadata,
        "hipengine_capture": summarize_capture(capture),
        "router_oracle": summarize_router(router),
        "oracle_results": oracle_results,
        "tolerances": {
            "routing_weights_f32": float(routing_atol),
            "shared_gate_logit_f32": float(shared_gate_atol),
        },
        "external_checkout_modified": False,
        "next_action": next_action(classification),
    }


def validate_post_attn_artifact(post_attn: Mapping[str, Any]) -> None:
    if post_attn.get("status") != "ready":
        raise ValueError("post-attn residual artifact must be ready")
    if post_attn.get("classification") != "layer0_post_attn_residual_matches_oracle_exactly":
        raise ValueError("post-attn residual artifact must have exact classification")
    if post_attn.get("next_action") != "audit_layer0_moe_router_from_post_norm":
        raise ValueError("post-attn artifact must point to MoE router audit")


def load_router_weights(
    model_path: Path,
    layer_id: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reader = GGUFReader(model_path)
    specs = {
        "router_weight_bf16_f32": "ffn_gate_inp.weight",
        "shared_gate_weight_bf16_f32": "ffn_gate_inp_shexp.weight",
    }
    weights: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for slot, suffix in specs.items():
        tensor_name = f"blk.{layer_id}.{suffix}"
        tensor = reader.tensor_info(tensor_name)
        source = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
        resident = bf16_roundtrip_like(source)
        weights[slot] = resident
        metadata[slot] = {
            "tensor_name": tensor_name,
            "source_ggml_type": tensor.ggml_type_name,
            "shape": list(source.shape),
            "source_summary": summarize_array(source.reshape(-1)),
            "resident_bf16_summary": summarize_array(resident.reshape(-1)),
            "source_sha256": sha256_float32(source.reshape(-1)),
            "resident_bf16_sha256": sha256_float32(resident.reshape(-1)),
            "contract": "source F32 rounded to BF16 for current hipEngine router kernels",
        }
    return weights, metadata


def bf16_roundtrip_like(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return bf16_roundtrip_array(arr.reshape(-1)).reshape(arr.shape)


def compute_router_oracle(
    post_norm: np.ndarray,
    router_weight: np.ndarray,
    shared_gate_weight: np.ndarray,
    *,
    top_k: int,
    threads: int = 256,
) -> dict[str, np.ndarray]:
    hidden = np.asarray(post_norm, dtype=np.float32).reshape(-1)
    router_w = np.asarray(router_weight, dtype=np.float32)
    shared_w = np.asarray(shared_gate_weight, dtype=np.float32).reshape(-1)
    if router_w.ndim != 2 or router_w.shape[1] != hidden.shape[0]:
        raise ValueError("router weight shape must be [experts, hidden]")
    if shared_w.shape[0] != hidden.shape[0]:
        raise ValueError("shared-gate weight shape must match hidden width")
    if top_k <= 0 or top_k > router_w.shape[0]:
        raise ValueError("top_k must be within router expert count")
    expert_logits = np.asarray(
        [router_dot_kernel_order(hidden, row, threads=threads) for row in router_w],
        dtype=np.float32,
    )
    shared_gate_logit = np.asarray(
        [router_dot_kernel_order(hidden, shared_w, threads=threads)],
        dtype=np.float32,
    )
    selected = select_topk_descending(expert_logits, top_k)
    routing = softmax_f32(expert_logits[selected])
    return {
        "expert_logits_f32": expert_logits,
        "shared_gate_logit_f32": shared_gate_logit,
        "selected_experts_i64": selected.astype(np.int64),
        "routing_weights_f32": routing.astype(np.float32),
        "shared_gate_sigmoid_f32": sigmoid_f32(shared_gate_logit).astype(np.float32),
    }


def router_dot_kernel_order(
    hidden: np.ndarray,
    weight_row: np.ndarray,
    *,
    threads: int = 256,
) -> np.float32:
    hidden32 = np.asarray(hidden, dtype=np.float32).reshape(-1)
    weight32 = np.asarray(weight_row, dtype=np.float32).reshape(-1)
    if hidden32.shape != weight32.shape:
        raise ValueError("hidden and weight row shapes must match")
    partial = np.zeros((int(threads),), dtype=np.float32)
    hidden_size = int(hidden32.shape[0])
    vec_stride = int(threads) * 8
    tail = hidden_size & ~7
    for tid in range(int(threads)):
        acc = np.float32(0.0)
        k = tid * 8
        while k + 7 < hidden_size:
            for offset in range(8):
                acc = np.float32(
                    acc + np.float32(hidden32[k + offset] * weight32[k + offset])
                )
            k += vec_stride
        k = tail + tid
        while k < hidden_size:
            acc = np.float32(acc + np.float32(hidden32[k] * weight32[k]))
            k += int(threads)
        partial[tid] = acc
    stride = int(threads) // 2
    while stride > 0:
        for tid in range(stride):
            partial[tid] = np.float32(partial[tid] + partial[tid + stride])
        stride //= 2
    return np.float32(partial[0])


def select_topk_descending(logits: np.ndarray, top_k: int) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32).copy()
    selected = np.empty((int(top_k),), dtype=np.int64)
    for pos in range(int(top_k)):
        idx = int(np.argmax(values))
        selected[pos] = idx
        values[idx] = np.float32(-np.inf)
    return selected


def softmax_f32(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    max_value = np.float32(vals[0])
    denom = np.float32(0.0)
    exps = np.empty_like(vals)
    for idx, value in enumerate(vals):
        exps[idx] = np.float32(np.exp(np.float32(value - max_value)))
        denom = np.float32(denom + exps[idx])
    denom = np.float32(max(float(denom), 1.0e-20))
    return np.asarray(exps / denom, dtype=np.float32)


def sigmoid_f32(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    return np.asarray(1.0 / (1.0 + np.exp(-vals, dtype=np.float32)), dtype=np.float32)


def compare_router_oracle(
    *,
    router: Mapping[str, np.ndarray],
    capture: Mapping[str, Any],
    routing_atol: float,
    shared_gate_atol: float,
) -> dict[str, Any]:
    fields = capture["fields"]
    selected_oracle = np.asarray(router["selected_experts_i64"], dtype=np.int64)
    selected_hip = np.asarray(fields["moe_selected_experts_i64"], dtype=np.int64)
    routing_delta = delta_summary(
        np.asarray(router["routing_weights_f32"], dtype=np.float32),
        np.asarray(fields["moe_routing_weights_f32"], dtype=np.float32),
    )
    shared_delta = delta_summary(
        np.asarray(router["shared_gate_logit_f32"], dtype=np.float32),
        np.asarray(fields["moe_shared_gate_f32"], dtype=np.float32),
    )
    selected_match = bool(
        selected_oracle.shape == selected_hip.shape
        and np.array_equal(selected_oracle, selected_hip)
    )
    return {
        "selected_experts_i64": {
            "field": "moe_selected_experts_i64",
            "oracle": "top-k descending expert logits from BF16-contracted router",
            "oracle_values": [int(value) for value in selected_oracle.tolist()],
            "hipengine_values": [int(value) for value in selected_hip.tolist()],
            "shape_match": selected_oracle.shape == selected_hip.shape,
            "exact_match": selected_match,
            "classification": (
                "router_field_matches_oracle_exactly"
                if selected_match
                else "router_field_mismatch_after_oracle"
            ),
        },
        "routing_weights_f32": {
            "field": "moe_routing_weights_f32",
            "oracle": "softmax over selected top-k expert logits",
            "oracle_summary": summarize_array(router["routing_weights_f32"]),
            "hipengine_summary": summarize_array(fields["moe_routing_weights_f32"]),
            "delta_oracle_vs_hip": routing_delta,
            "near_atol": float(routing_atol),
            "classification": classify_delta(routing_delta, near_atol=float(routing_atol)),
        },
        "shared_gate_logit_f32": {
            "field": "moe_shared_gate_f32",
            "oracle": "raw shared-gate logit from BF16-contracted ffn_gate_inp_shexp",
            "oracle_summary": summarize_array(router["shared_gate_logit_f32"]),
            "hipengine_summary": summarize_array(fields["moe_shared_gate_f32"]),
            "delta_oracle_vs_hip": shared_delta,
            "near_atol": float(shared_gate_atol),
            "classification": classify_delta(shared_delta, near_atol=float(shared_gate_atol)),
        },
    }


def classify_delta(delta: Mapping[str, Any], *, near_atol: float) -> str:
    if not delta.get("available") or not delta.get("shape_match"):
        return "router_field_oracle_unavailable"
    if delta.get("exact_match"):
        return "router_field_matches_oracle_exactly"
    if float(delta.get("max_abs_diff", float("inf"))) <= float(near_atol):
        return "router_field_matches_oracle_within_tolerance"
    return "router_field_mismatch_after_oracle"


def classify_moe_router_oracle(oracle_results: Mapping[str, Any]) -> str:
    classes = [oracle_results[name]["classification"] for name in ROUTER_COMPARE_FIELDS]
    if all(item == "router_field_matches_oracle_exactly" for item in classes):
        return "layer0_moe_router_matches_oracle_exactly"
    if all(item.startswith("router_field_matches_oracle") for item in classes):
        return "layer0_moe_router_matches_oracle_within_tolerance"
    if any("mismatch" in item for item in classes):
        return "layer0_moe_router_mismatch_after_oracle"
    return "layer0_moe_router_oracle_unavailable"


def status_from_classification(classification: str) -> str:
    if classification.endswith("unavailable"):
        return "unavailable"
    if "mismatch" in classification:
        return "mismatched"
    return "ready"


def next_action(classification: str) -> str:
    if classification in {
        "layer0_moe_router_matches_oracle_exactly",
        "layer0_moe_router_matches_oracle_within_tolerance",
    }:
        return "audit_layer0_moe_selected_and_shared_expert_outputs"
    if classification == "layer0_moe_router_mismatch_after_oracle":
        return "inspect_layer0_moe_router_dtype_or_topk_semantics"
    return "rerun_layer0_moe_router_oracle_on_rocm_host"


def summarize_router(router: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "expert_logits_summary": summarize_array(router["expert_logits_f32"]),
        "expert_logits_sha256": sha256_float32(router["expert_logits_f32"]),
        "selected_experts": [int(value) for value in router["selected_experts_i64"].tolist()],
        "selected_logits": [
            float(router["expert_logits_f32"][int(index)])
            for index in router["selected_experts_i64"]
        ],
        "routing_weights_summary": summarize_array(router["routing_weights_f32"]),
        "shared_gate_logit_summary": summarize_array(router["shared_gate_logit_f32"]),
        "shared_gate_sigmoid_summary": summarize_array(router["shared_gate_sigmoid_f32"]),
    }


def unavailable_artifact(
    *,
    post_attn_artifact_path: Path,
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
        "kind": "layer0_moe_router_oracle",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(capture.get("status", "unavailable")),
        "classification": "layer0_moe_router_oracle_unavailable",
        "post_attn_artifact_path": str(post_attn_artifact_path),
        "model": str(model_path),
        "layer_id": int(layer_id),
        "target_position": int(target_position),
        "token_id": int(token_id),
        "prompt_tokens": list(prompt_tokens),
        "hipengine_capture": summarize_capture(capture),
        "external_checkout_modified": False,
        "next_action": "rerun_layer0_moe_router_oracle_on_rocm_host",
    }


if __name__ == "__main__":
    main()
