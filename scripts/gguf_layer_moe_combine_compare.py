#!/usr/bin/env python3
"""Compare GGUF MoE/shared-expert layer combine against CPU formula."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.materialize import float_array_to_bf16_bits  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32  # noqa: E402

DEFAULT_CAPTURE = Path("benchmarks/results/mtp-gguf-iter276-linear-layer-routing-full-arrays.json")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter276-layer-moe-combine-compare.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=276)
    args = parser.parse_args()

    artifact = build_moe_combine_artifact(args.capture, iteration=args.iteration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "layer_out_max_abs": artifact["layer_out_vs_cpu"]["max_abs_diff"],
                "layer_out_rms": artifact["layer_out_vs_cpu"]["rms_abs_diff"],
                "within_tolerance": artifact["within_tolerance"],
            },
            indent=2,
        )
    )


def build_moe_combine_artifact(capture_path: Path, *, iteration: int = 276) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    arrays = capture.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("capture artifact must include full arrays; rerun with --include-arrays")
    top_k = int(capture["capture_summary"]["top_k"])
    hidden_size = int(capture["capture_summary"]["hidden_size"])
    comparison = compare_moe_combine(
        residual=_read_array(arrays, "residual_f32"),
        selected_down=_read_array(arrays, "ffn_or_moe_down_f32").reshape(top_k, hidden_size),
        routing_weights=_read_array(arrays, "moe_routing_weights_f32"),
        shared_out=_read_array(arrays, "moe_shared_out_f32"),
        shared_gate=_read_array(arrays, "moe_shared_gate_f32"),
        layer_out=_read_array(arrays, "layer_out_f32"),
    )
    within = comparison["layer_out_vs_cpu"]["max_abs_diff"] == 0.0
    return {
        "schema": 1,
        "kind": "mtp_gguf_layer_moe_combine_cpu_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "source_capture": str(capture_path),
        "source_iteration": capture.get("iteration"),
        "model": capture.get("model"),
        "layer_id": capture.get("layer_id"),
        "position": capture.get("position"),
        "token_id": capture.get("token_id"),
        "top_k": top_k,
        "selected_experts": arrays.get("moe_selected_experts_i64"),
        "routing_weights": arrays.get("moe_routing_weights_f32"),
        "shared_gate_logit": arrays.get("moe_shared_gate_f32"),
        "shared_gate_sigmoid": comparison["shared_gate_sigmoid"],
        "weighted_selected_vs_bf16": comparison["weighted_selected_vs_bf16"],
        "layer_out_vs_cpu": comparison["layer_out_vs_cpu"],
        "samples": comparison["samples"],
        "within_tolerance": bool(within),
        "conclusion": _conclusion(comparison, within),
    }


def compare_moe_combine(
    *,
    residual: np.ndarray,
    selected_down: np.ndarray,
    routing_weights: np.ndarray,
    shared_out: np.ndarray,
    shared_gate: np.ndarray,
    layer_out: np.ndarray,
) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float32).reshape(-1)
    selected_down = np.asarray(selected_down, dtype=np.float32)
    routing_weights = np.asarray(routing_weights, dtype=np.float32).reshape(-1)
    shared_out = np.asarray(shared_out, dtype=np.float32).reshape(-1)
    shared_gate = np.asarray(shared_gate, dtype=np.float32).reshape(-1)
    layer_out = np.asarray(layer_out, dtype=np.float32).reshape(-1)
    if selected_down.ndim != 2:
        raise ValueError("selected_down must have shape [top_k, hidden_size]")
    if selected_down.shape[0] != routing_weights.shape[0]:
        raise ValueError("routing_weights length must match selected_down top_k")
    if not (selected_down.shape[1] == residual.size == shared_out.size == layer_out.size):
        raise ValueError("hidden-size arrays must match selected_down width")
    if shared_gate.size != 1:
        raise ValueError("shared_gate must contain one scalar")

    weighted_f32 = np.sum(selected_down * routing_weights[:, None], axis=0, dtype=np.float32)
    weighted_bf16 = _round_to_bf16(weighted_f32)
    gate = _sigmoid(float(shared_gate[0]))
    cpu_layer_out = _round_to_bf16(residual + weighted_bf16 + np.float32(gate) * shared_out)
    return {
        "shared_gate_sigmoid": float(gate),
        "weighted_selected_vs_bf16": _diff_metrics(weighted_f32, weighted_bf16),
        "layer_out_vs_cpu": _diff_metrics(cpu_layer_out, layer_out),
        "samples": {
            "cpu_layer_out": [float(x) for x in cpu_layer_out[:8]],
            "device_layer_out": [float(x) for x in layer_out[:8]],
            "weighted_selected_bf16": [float(x) for x in weighted_bf16[:8]],
        },
    }


def _round_to_bf16(array: np.ndarray) -> np.ndarray:
    return bf16_to_float32(float_array_to_bf16_bits(np.asarray(array, dtype=np.float32))).astype(
        np.float32
    )


def _sigmoid(value: float) -> float:
    return float(np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(-value))))


def _read_array(arrays: dict[str, object], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"capture artifact missing arrays.{key}")
    return np.asarray(arrays[key], dtype=np.float32)


def _diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    diff = candidate - reference
    return {
        "count": int(reference.size),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rms_abs_diff": float(np.sqrt(np.mean(diff * diff, dtype=np.float32)))
        if diff.size
        else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff), dtype=np.float32)) if diff.size else 0.0,
        "reference_rms": float(np.sqrt(np.mean(reference * reference, dtype=np.float32)))
        if reference.size
        else 0.0,
        "candidate_rms": float(np.sqrt(np.mean(candidate * candidate, dtype=np.float32)))
        if candidate.size
        else 0.0,
    }


def _conclusion(comparison: dict[str, Any], within: bool) -> str:
    layer = comparison["layer_out_vs_cpu"]
    if within:
        return (
            "Layer-0 MoE/shared-expert combine is bit-exact against the CPU kernel formula: "
            "BF16(weighted selected experts) + sigmoid(shared_gate)*shared + residual. "
            "Layer-0 final output is now explained; continue to later layers or final head."
        )
    return (
        "Layer-0 MoE/shared-expert combine diverges from CPU formula; inspect routing weights, "
        f"shared gate, or combine kernel. max_abs={layer['max_abs_diff']:.6g}."
    )


if __name__ == "__main__":
    main()
