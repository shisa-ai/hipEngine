#!/usr/bin/env python3
"""Compare CPU-recomputed GGUF ssm_out against a boundary capture."""

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
DEFAULT_CAPTURE = Path(
    "benchmarks/results/mtp-gguf-iter265-extended-linear-boundary-full-arrays.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter267-ssm-out-cpu-compare.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=267)
    args = parser.parse_args()

    artifact = build_ssm_out_compare_artifact(
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
                "max_abs_diff": artifact["device_vs_cpu_bf16"]["max_abs_diff"],
                "rms_abs_diff": artifact["device_vs_cpu_bf16"]["rms_abs_diff"],
                "within_bf16_tolerance": artifact["within_bf16_tolerance"],
            },
            indent=2,
        )
    )


def build_ssm_out_compare_artifact(
    *,
    model: Path,
    capture_path: Path,
    layer_id: int = 0,
    iteration: int = 267,
) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    arrays = capture.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("capture artifact must include full arrays; rerun with --include-arrays")
    recurrent_bf16 = _read_array(arrays, "recurrent_bf16_f32")
    device_attn_out = _read_array(arrays, "attn_out_f32")

    reader = GGUFReader(model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    layer = model_map.layers[int(layer_id)]
    ssm_out_name = layer.tensor("ssm_out").name
    ssm_out_weight = reader.dequantize_tensor(ssm_out_name).astype(np.float32)
    comparison = compare_ssm_out(ssm_out_weight, recurrent_bf16, device_attn_out)

    return {
        "schema": 1,
        "kind": "mtp_gguf_ssm_out_cpu_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "model": str(model),
        "source_capture": str(capture_path),
        "source_iteration": capture.get("iteration"),
        "layer_id": int(layer_id),
        "weight_name": ssm_out_name,
        "weight_shape": list(ssm_out_weight.shape),
        "input_shape": list(recurrent_bf16.shape),
        "output_shape": list(device_attn_out.shape),
        "device_vs_cpu_bf16": comparison["device_vs_cpu_bf16"],
        "cpu_f32_vs_cpu_bf16": comparison["cpu_f32_vs_cpu_bf16"],
        "within_bf16_tolerance": comparison["device_vs_cpu_bf16"]["max_abs_diff"]
        <= 1.0e-5,
        "samples": comparison["samples"],
        "conclusion": _conclusion(comparison),
    }


def compare_ssm_out(
    weight_f32: np.ndarray,
    recurrent_bf16_f32: np.ndarray,
    device_attn_out_f32: np.ndarray,
) -> dict[str, Any]:
    weight_f32 = np.asarray(weight_f32, dtype=np.float32)
    recurrent_bf16_f32 = np.asarray(recurrent_bf16_f32, dtype=np.float32).reshape(-1)
    device_attn_out_f32 = np.asarray(device_attn_out_f32, dtype=np.float32).reshape(-1)
    if weight_f32.ndim != 2:
        raise ValueError("weight_f32 must be a 2D matrix")
    if weight_f32.shape[1] != recurrent_bf16_f32.shape[0]:
        raise ValueError("weight columns must match recurrent_bf16 length")
    if weight_f32.shape[0] != device_attn_out_f32.shape[0]:
        raise ValueError("weight rows must match device attn_out length")

    cpu_f32 = np.matmul(weight_f32, recurrent_bf16_f32).astype(np.float32)
    cpu_bf16 = bf16_to_float32(float_array_to_bf16_bits(cpu_f32)).astype(np.float32)
    return {
        "device_vs_cpu_bf16": _diff_metrics(cpu_bf16, device_attn_out_f32),
        "cpu_f32_vs_cpu_bf16": _diff_metrics(cpu_f32, cpu_bf16),
        "samples": {
            "cpu_bf16": [float(x) for x in cpu_bf16[:8]],
            "device_attn_out": [float(x) for x in device_attn_out_f32[:8]],
        },
    }


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
        "max_abs_reference": float(np.max(np.abs(reference))) if reference.size else 0.0,
        "max_abs_candidate": float(np.max(np.abs(candidate))) if candidate.size else 0.0,
    }


def _read_array(arrays: dict[str, object], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"capture artifact missing arrays.{key}")
    return np.asarray(arrays[key], dtype=np.float32)


def _conclusion(comparison: dict[str, Any]) -> str:
    device = comparison["device_vs_cpu_bf16"]
    if device["max_abs_diff"] <= 1.0e-5:
        return (
            "Device attn_out matches CPU ssm_out(recurrent_bf16) after BF16 rounding; "
            f"max_abs={device['max_abs_diff']:.6g}, rms_abs={device['rms_abs_diff']:.6g}. "
            "Layer-0 triage should move earlier into GDN/conv/recurrent inputs or later layers."
        )
    return (
        "Device attn_out diverges from CPU ssm_out(recurrent_bf16); investigate GGUF GEMV "
        f"or ssm_out quant/dequant path. max_abs={device['max_abs_diff']:.6g}."
    )


if __name__ == "__main__":
    main()
