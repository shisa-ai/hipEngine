#!/usr/bin/env python3
"""Probe early GGUF F32-vs-BF16 precision boundaries without launching HIP.

The probe is intentionally narrow: it compares llama.cpp-style F32 GGML graph
math against hipEngine's current BF16 resident activation/aux-weight contract for
one real token embedding row and one early Qwen35MoE linear-attention layer.
"""

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
from hipengine.quant.gguf import bf16_to_float32, dequantize_gguf_data  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter253-early-boundary-probe.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token-id", type=int, default=271)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_probe_artifact(args.model, layer_id=args.layer, token_id=args.token_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "layer": args.layer,
                "token_id": args.token_id,
                "norm_max_abs": artifact["attn_norm_boundary"]["max_abs_diff"],
                "ssm_alpha_max_abs": artifact["projections"]["ssm_alpha"]["max_abs_diff"],
                "ssm_beta_max_abs": artifact["projections"]["ssm_beta"]["max_abs_diff"],
            },
            indent=2,
        )
    )


def build_probe_artifact(model: Path, *, layer_id: int, token_id: int) -> dict[str, Any]:
    reader = GGUFReader(model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    cfg = model_map.config
    if layer_id < 0 or layer_id >= len(model_map.layers):
        max_layer = len(model_map.layers) - 1
        raise ValueError(f"layer {layer_id} outside executable layer range 0..{max_layer}")
    layer = model_map.layers[layer_id]
    if layer.layer_type != "linear_attention":
        raise ValueError(f"layer {layer_id} is {layer.layer_type!r}; expected linear_attention")

    token_embedding = _dequant_row(reader, model_map.root("token_embedding").name, token_id)
    attn_norm_weight = reader.dequantize_tensor(layer.tensor("attn_norm").name).astype(np.float32)
    ssm_alpha = reader.dequantize_tensor(layer.tensor("ssm_alpha").name).astype(np.float32)
    ssm_beta = reader.dequantize_tensor(layer.tensor("ssm_beta").name).astype(np.float32)

    llama_embedding = token_embedding.astype(np.float32)
    llama_norm = _rmsnorm(llama_embedding, attn_norm_weight, eps=float(cfg.rms_norm_eps))

    hip_embedding = _round_to_bf16_float(llama_embedding)
    hip_norm_pre_store = _rmsnorm(hip_embedding, attn_norm_weight, eps=float(cfg.rms_norm_eps))
    hip_norm = _round_to_bf16_float(hip_norm_pre_store)

    projections = {
        "ssm_alpha": _projection_boundary_metrics(llama_norm, hip_norm, ssm_alpha),
        "ssm_beta": _projection_boundary_metrics(llama_norm, hip_norm, ssm_beta),
    }

    return {
        "schema": 1,
        "kind": "mtp_gguf_early_precision_boundary_probe",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": 253,
        "model": str(model),
        "layer_id": layer_id,
        "layer_type": layer.layer_type,
        "token_id": token_id,
        "token_context": "last token of the captured reasoning-off greeting prompt is token 271",
        "contract": {
            "llama_cpp": (
                "token embedding, RMSNorm output, and F32 aux projections stay in F32 "
                "GGML graph tensors"
            ),
            "hipengine": (
                "token embedding/RMSNorm residents are BF16 and ssm_alpha/ssm_beta F32 "
                "weights are materialized as BF16"
            ),
        },
        "attn_norm_boundary": _diff_metrics(llama_norm, hip_norm),
        "projections": projections,
        "conclusion": _conclusion(projections),
    }


def _dequant_row(reader: GGUFReader, name: str, row: int) -> np.ndarray:
    info = reader.tensor_info(name)
    if len(info.shape) != 2:
        raise ValueError(f"expected a matrix tensor for row dequantization: {name}")
    if row < 0 or row >= info.shape[0]:
        raise ValueError(f"row {row} outside {name} row range 0..{info.shape[0] - 1}")
    raw = reader.tensor_data(name)
    return dequantize_gguf_data(raw[row : row + 1], info.ggml_type)[0].astype(np.float32)


def _round_to_bf16_float(array: np.ndarray) -> np.ndarray:
    return bf16_to_float32(float_array_to_bf16_bits(array)).astype(np.float32)


def _rmsnorm(x: np.ndarray, weight: np.ndarray, *, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    scale = np.float32(1.0) / np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(eps))
    return np.ascontiguousarray((x * scale * weight).astype(np.float32))


def _projection_boundary_metrics(
    llama_input: np.ndarray,
    hip_input: np.ndarray,
    weight_f32: np.ndarray,
) -> dict[str, Any]:
    weight_bf16 = _round_to_bf16_float(weight_f32)
    llama = np.matmul(weight_f32, llama_input).astype(np.float32)
    hip_pre_store = np.matmul(weight_bf16, hip_input).astype(np.float32)
    hip = _round_to_bf16_float(hip_pre_store)
    metrics = _diff_metrics(llama, hip)
    metrics.update(
        {
            "rows": int(weight_f32.shape[0]),
            "cols": int(weight_f32.shape[1]),
            "llama_sample": _sample(llama),
            "hipengine_sample": _sample(hip),
        }
    )
    return metrics


def _diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    diff = candidate - reference
    denom = np.maximum(np.abs(reference), np.float32(1.0e-6))
    dot = float(np.dot(reference.reshape(-1), candidate.reshape(-1)))
    ref_norm = float(np.linalg.norm(reference.reshape(-1)))
    cand_norm = float(np.linalg.norm(candidate.reshape(-1)))
    cosine = dot / (ref_norm * cand_norm) if ref_norm > 0.0 and cand_norm > 0.0 else 0.0
    return {
        "shape": list(reference.shape),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rms_abs_diff": float(np.sqrt(np.mean(diff * diff, dtype=np.float32))),
        "max_rel_diff": float(np.max(np.abs(diff) / denom)),
        "mean_abs_reference": float(np.mean(np.abs(reference), dtype=np.float32)),
        "mean_abs_candidate": float(np.mean(np.abs(candidate), dtype=np.float32)),
        "cosine_similarity": float(cosine),
    }


def _sample(array: np.ndarray, *, count: int = 8) -> list[float]:
    return [float(x) for x in np.asarray(array, dtype=np.float32).reshape(-1)[:count]]


def _conclusion(projections: dict[str, dict[str, Any]]) -> str:
    alpha = projections["ssm_alpha"]
    beta = projections["ssm_beta"]
    return (
        "Layer-0 embedding->attn_norm->GDN projection already has measurable "
        "F32-vs-BF16 drift on the real greeting boundary: "
        f"alpha max_abs={alpha['max_abs_diff']:.6g}, beta max_abs={beta['max_abs_diff']:.6g}. "
        "This is not a full target-AR oracle, but it pins the earliest numeric boundary "
        "to compare before MTP acceptance tuning."
    )


if __name__ == "__main__":
    main()
