#!/usr/bin/env python3
"""Compare GGUF post-attention residual add and RMSNorm against CPU oracles."""

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

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.loading.materialize import float_array_to_bf16_bits  # noqa: E402
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_CAPTURE = Path("benchmarks/results/mtp-gguf-iter274-linear-layer-full-arrays.json")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter275-layer-residual-norm-compare.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=275)
    args = parser.parse_args()

    artifact = build_residual_norm_artifact(
        model=args.model,
        capture_path=args.capture,
        layer_id=args.layer,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "residual_max_abs": artifact["residual_vs_cpu_bf16"]["max_abs_diff"],
                "post_norm_max_abs": artifact["post_norm_vs_cpu_bf16"]["max_abs_diff"],
                "within_tolerance": artifact["within_tolerance"],
            },
            indent=2,
        )
    )


def build_residual_norm_artifact(
    *,
    model: Path,
    capture_path: Path,
    layer_id: int = 0,
    iteration: int = 275,
) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    arrays = capture.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("capture artifact must include full arrays; rerun with --include-arrays")
    hidden_in = _read_array(arrays, "hidden_in_f32")
    attn_out = _read_array(arrays, "attn_out_f32")
    residual = _read_array(arrays, "residual_f32")
    post_norm = _read_array(arrays, "post_norm_f32")

    reader = GGUFReader(model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    layer = model_map.layers[int(layer_id)]
    norm_name = layer.tensor("post_attention_norm").name
    norm_weight = reader.dequantize_tensor(norm_name).astype(np.float32)
    comparison = compare_residual_norm(
        hidden_in=hidden_in,
        attn_out=attn_out,
        residual=residual,
        post_norm=post_norm,
        norm_weight=norm_weight,
        eps=float(model_map.config.rms_norm_eps),
    )
    within = (
        comparison["residual_vs_cpu_bf16"]["max_abs_diff"] == 0.0
        and comparison["post_norm_vs_cpu_bf16"]["max_abs_diff"] <= 5.0e-2
    )
    return {
        "schema": 1,
        "kind": "mtp_gguf_layer_residual_norm_cpu_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "model": str(model),
        "source_capture": str(capture_path),
        "source_iteration": capture.get("iteration"),
        "layer_id": int(layer_id),
        "position": capture.get("position"),
        "token_id": capture.get("token_id"),
        "norm_weight_name": norm_name,
        "residual_vs_cpu_bf16": comparison["residual_vs_cpu_bf16"],
        "post_norm_vs_cpu_bf16": comparison["post_norm_vs_cpu_bf16"],
        "post_norm_cpu_f32_vs_bf16": comparison["post_norm_cpu_f32_vs_bf16"],
        "samples": comparison["samples"],
        "within_tolerance": bool(within),
        "conclusion": _conclusion(comparison, within),
    }


def compare_residual_norm(
    *,
    hidden_in: np.ndarray,
    attn_out: np.ndarray,
    residual: np.ndarray,
    post_norm: np.ndarray,
    norm_weight: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    hidden_in = np.asarray(hidden_in, dtype=np.float32).reshape(-1)
    attn_out = np.asarray(attn_out, dtype=np.float32).reshape(-1)
    residual = np.asarray(residual, dtype=np.float32).reshape(-1)
    post_norm = np.asarray(post_norm, dtype=np.float32).reshape(-1)
    norm_weight = np.asarray(norm_weight, dtype=np.float32).reshape(-1)
    if not (hidden_in.shape == attn_out.shape == residual.shape == post_norm.shape):
        raise ValueError("hidden/attention/residual/post_norm arrays must have the same shape")
    if norm_weight.shape != residual.shape:
        raise ValueError("norm_weight must match hidden size")

    cpu_residual = _round_to_bf16(hidden_in + attn_out)
    cpu_norm_f32 = _rmsnorm(cpu_residual, norm_weight, eps=float(eps))
    cpu_norm_bf16 = _round_to_bf16(cpu_norm_f32)
    return {
        "residual_vs_cpu_bf16": _diff_metrics(cpu_residual, residual),
        "post_norm_vs_cpu_bf16": _diff_metrics(cpu_norm_bf16, post_norm),
        "post_norm_cpu_f32_vs_bf16": _diff_metrics(cpu_norm_f32, cpu_norm_bf16),
        "samples": {
            "cpu_residual": [float(x) for x in cpu_residual[:8]],
            "device_residual": [float(x) for x in residual[:8]],
            "cpu_post_norm_bf16": [float(x) for x in cpu_norm_bf16[:8]],
            "device_post_norm": [float(x) for x in post_norm[:8]],
        },
    }


def _rmsnorm(x: np.ndarray, weight: np.ndarray, *, eps: float) -> np.ndarray:
    inv = np.float32(1.0) / np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(eps))
    return np.ascontiguousarray((x * inv * weight).astype(np.float32))


def _round_to_bf16(array: np.ndarray) -> np.ndarray:
    return bf16_to_float32(float_array_to_bf16_bits(np.asarray(array, dtype=np.float32))).astype(
        np.float32
    )


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
    residual = comparison["residual_vs_cpu_bf16"]
    post_norm = comparison["post_norm_vs_cpu_bf16"]
    if within:
        return (
            "Layer-0 residual add is bit-exact versus BF16(hidden_in + attn_out), and "
            f"post_attention_norm matches the CPU direct-weight RMSNorm within BF16-scale "
            f"tolerance (max_abs={post_norm['max_abs_diff']:.6g}). The next unchecked "
            "boundary is MoE/shared-expert combine or later layers."
        )
    return (
        "Layer-0 residual/RMSNorm diverges from CPU oracle; inspect add+rmsnorm kernel. "
        f"residual max_abs={residual['max_abs_diff']:.6g}, "
        f"post_norm max_abs={post_norm['max_abs_diff']:.6g}."
    )


if __name__ == "__main__":
    main()
