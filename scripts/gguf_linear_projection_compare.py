#!/usr/bin/env python3
"""Compare GGUF linear-attention projections against a boundary capture."""

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
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter268-linear-projection-cpu-compare.json")
PROJECTIONS = {
    "attn_qkv": "linear_qkv_f32",
    "attn_gate": "linear_z_f32",
    "ssm_alpha": "ssm_alpha_f32",
    "ssm_beta": "ssm_beta_f32",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=268)
    args = parser.parse_args()

    artifact = build_projection_compare_artifact(
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
                "within_bf16_tolerance": artifact["within_bf16_tolerance"],
                "max_abs_by_projection": {
                    name: result["device_vs_cpu_bf16"]["max_abs_diff"]
                    for name, result in artifact["projections"].items()
                },
            },
            indent=2,
        )
    )


def build_projection_compare_artifact(
    *,
    model: Path,
    capture_path: Path,
    layer_id: int = 0,
    iteration: int = 268,
) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    arrays = capture.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("capture artifact must include full arrays; rerun with --include-arrays")
    attn_norm = _read_array(arrays, "attn_norm_f32")

    reader = GGUFReader(model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    layer = model_map.layers[int(layer_id)]
    projection_results: dict[str, Any] = {}
    for slot, array_key in PROJECTIONS.items():
        weight_name = layer.tensor(slot).name
        weight = reader.dequantize_tensor(weight_name).astype(np.float32)
        device = _read_array(arrays, array_key)
        projection_results[slot] = compare_projection(weight, attn_norm, device)
        projection_results[slot]["weight_name"] = weight_name
        projection_results[slot]["array_key"] = array_key
        projection_results[slot]["weight_shape"] = list(weight.shape)

    within = all(
        result["device_vs_cpu_bf16"]["max_abs_diff"] <= 1.0e-3
        for result in projection_results.values()
    )
    return {
        "schema": 1,
        "kind": "mtp_gguf_linear_projection_cpu_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "model": str(model),
        "source_capture": str(capture_path),
        "source_iteration": capture.get("iteration"),
        "layer_id": int(layer_id),
        "input_shape": list(attn_norm.shape),
        "projections": projection_results,
        "within_bf16_tolerance": bool(within),
        "conclusion": _conclusion(projection_results, within),
    }


def compare_projection(
    weight_f32: np.ndarray,
    input_f32: np.ndarray,
    device_output_f32: np.ndarray,
) -> dict[str, Any]:
    weight_f32 = np.asarray(weight_f32, dtype=np.float32)
    input_f32 = np.asarray(input_f32, dtype=np.float32).reshape(-1)
    device_output_f32 = np.asarray(device_output_f32, dtype=np.float32).reshape(-1)
    if weight_f32.ndim != 2:
        raise ValueError("weight_f32 must be a 2D matrix")
    if weight_f32.shape[1] != input_f32.shape[0]:
        raise ValueError("weight columns must match input length")
    if weight_f32.shape[0] != device_output_f32.shape[0]:
        raise ValueError("weight rows must match device output length")

    cpu_f32 = np.matmul(weight_f32, input_f32).astype(np.float32)
    cpu_bf16 = bf16_to_float32(float_array_to_bf16_bits(cpu_f32)).astype(np.float32)
    return {
        "device_vs_cpu_bf16": _diff_metrics(cpu_bf16, device_output_f32),
        "cpu_f32_vs_cpu_bf16": _diff_metrics(cpu_f32, cpu_bf16),
        "samples": {
            "cpu_bf16": [float(x) for x in cpu_bf16[:8]],
            "device_output": [float(x) for x in device_output_f32[:8]],
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
    }


def _read_array(arrays: dict[str, object], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"capture artifact missing arrays.{key}")
    return np.asarray(arrays[key], dtype=np.float32)


def _conclusion(projections: dict[str, Any], within: bool) -> str:
    max_abs = max(
        result["device_vs_cpu_bf16"]["max_abs_diff"] for result in projections.values()
    )
    if within:
        return (
            "Layer-0 linear projections from attn_norm match CPU GGUF GEMV BF16 outputs; "
            f"worst max_abs={max_abs:.6g}. Remaining search moves into conv/GDN recurrent "
            "math or the state entering layer 0."
        )
    return (
        "At least one layer-0 projection diverges from CPU GGUF GEMV; inspect projection "
        f"weight/layout before GDN. worst max_abs={max_abs:.6g}."
    )


if __name__ == "__main__":
    main()
