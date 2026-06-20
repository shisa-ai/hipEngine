#!/usr/bin/env python3
"""Compare GGUF ssm_conv1d/conv_out against a multi-position boundary capture."""

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
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_CAPTURE = Path(
    "benchmarks/results/mtp-gguf-iter269-final-token-conv-window-capture.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter270-conv-out-cpu-compare.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=270)
    args = parser.parse_args()

    artifact = build_conv_out_compare_artifact(
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
                "max_abs_diff": artifact["device_vs_cpu"]["max_abs_diff"],
                "rms_abs_diff": artifact["device_vs_cpu"]["rms_abs_diff"],
                "within_float_tolerance": artifact["within_float_tolerance"],
            },
            indent=2,
        )
    )


def build_conv_out_compare_artifact(
    *,
    model: Path,
    capture_path: Path,
    layer_id: int = 0,
    iteration: int = 270,
) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    captures = capture.get("captures")
    if not isinstance(captures, list):
        raise ValueError("capture artifact must be a multi-position batch with captures")
    reader = GGUFReader(model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    cfg = model_map.config
    layer = model_map.layers[int(layer_id)]
    conv_name = layer.tensor("ssm_conv1d").name
    conv_weight = reader.dequantize_tensor(conv_name).astype(np.float32)
    comparison = compare_conv_out(
        conv_weight,
        _window_linear_qkv(captures, int(cfg.ssm_conv_kernel)),
        _read_array(captures[-1], "conv_out_f32"),
    )

    return {
        "schema": 1,
        "kind": "mtp_gguf_conv_out_cpu_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "model": str(model),
        "source_capture": str(capture_path),
        "source_iteration": capture.get("iteration"),
        "layer_id": int(layer_id),
        "weight_name": conv_name,
        "weight_shape": list(conv_weight.shape),
        "positions": capture.get("positions"),
        "token_ids": capture.get("token_ids"),
        "target_position": captures[-1].get("position"),
        "target_token_id": captures[-1].get("token_id"),
        "device_vs_cpu": comparison["device_vs_cpu"],
        "within_float_tolerance": comparison["device_vs_cpu"]["max_abs_diff"] <= 1.0e-6,
        "samples": comparison["samples"],
        "conclusion": _conclusion(comparison),
    }


def compare_conv_out(
    conv_weight_f32: np.ndarray,
    linear_qkv_window_f32: np.ndarray,
    device_conv_out_f32: np.ndarray,
) -> dict[str, Any]:
    conv_weight_f32 = np.asarray(conv_weight_f32, dtype=np.float32)
    linear_qkv_window_f32 = np.asarray(linear_qkv_window_f32, dtype=np.float32)
    device_conv_out_f32 = np.asarray(device_conv_out_f32, dtype=np.float32).reshape(-1)
    if conv_weight_f32.ndim != 2:
        raise ValueError("conv_weight_f32 must be a 2D [channels, kernel] matrix")
    channels, kernel_size = conv_weight_f32.shape
    if linear_qkv_window_f32.shape != (kernel_size, channels):
        raise ValueError("linear_qkv_window_f32 must have shape [kernel, channels]")
    if device_conv_out_f32.shape != (channels,):
        raise ValueError("device_conv_out_f32 length must match channels")

    acc = np.sum(linear_qkv_window_f32 * conv_weight_f32.T, axis=0, dtype=np.float32)
    cpu_conv = _silu(acc).astype(np.float32)
    return {
        "device_vs_cpu": _diff_metrics(cpu_conv, device_conv_out_f32),
        "samples": {
            "cpu_conv_out": [float(x) for x in cpu_conv[:8]],
            "device_conv_out": [float(x) for x in device_conv_out_f32[:8]],
        },
    }


def _window_linear_qkv(captures: list[object], kernel_size: int) -> np.ndarray:
    if len(captures) < kernel_size:
        raise ValueError(f"need at least {kernel_size} captures for conv window")
    window = [_read_array(capture, "linear_qkv_f32") for capture in captures[-kernel_size:]]
    return np.stack(window, axis=0).astype(np.float32)


def _read_array(capture: object, key: str) -> np.ndarray:
    if not isinstance(capture, dict) or not isinstance(capture.get("arrays"), dict):
        raise ValueError("capture records must include arrays; rerun with --include-arrays")
    arrays = capture["arrays"]
    if key not in arrays:
        raise ValueError(f"capture artifact missing arrays.{key}")
    return np.asarray(arrays[key], dtype=np.float32)


def _silu(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.float32(1.0) + np.exp(-values, dtype=np.float32))


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


def _conclusion(comparison: dict[str, Any]) -> str:
    device = comparison["device_vs_cpu"]
    if device["max_abs_diff"] <= 1.0e-6:
        return (
            "Device conv_out matches CPU ssm_conv1d over the captured linear_qkv window; "
            f"max_abs={device['max_abs_diff']:.6g}, rms_abs={device['rms_abs_diff']:.6g}. "
            "Layer-0 triage moves downstream into GDN recurrence or outside layer 0."
        )
    return (
        "Device conv_out diverges from CPU ssm_conv1d over the captured linear_qkv window; "
        f"inspect conv state/order/weight layout. max_abs={device['max_abs_diff']:.6g}."
    )


if __name__ == "__main__":
    main()
