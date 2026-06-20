#!/usr/bin/env python3
"""Quantify recurrent_out F32 -> BF16 cast error in GGUF boundary captures."""

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

DEFAULT_INPUT = Path(
    "benchmarks/results/mtp-gguf-iter265-extended-linear-boundary-full-arrays.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter266-recurrent-bf16-cast-error.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=266)
    args = parser.parse_args()

    artifact = build_cast_error_artifact(args.input, iteration=args.iteration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "device_matches_expected_bf16": artifact["device_matches_expected_bf16"],
                "max_abs_cast_error": artifact["cast_error"]["max_abs_diff"],
                "rms_abs_cast_error": artifact["cast_error"]["rms_abs_diff"],
            },
            indent=2,
        )
    )


def build_cast_error_artifact(capture_path: Path, *, iteration: int = 266) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    arrays = capture.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("capture artifact must include full arrays; rerun with --include-arrays")
    recurrent_out = _read_array(arrays, "recurrent_out_f32")
    recurrent_bf16 = _read_array(arrays, "recurrent_bf16_f32")
    comparison = compare_recurrent_bf16_cast(recurrent_out, recurrent_bf16)

    return {
        "schema": 1,
        "kind": "mtp_gguf_recurrent_bf16_cast_error",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "source_capture": str(capture_path),
        "source_iteration": capture.get("iteration"),
        "model": capture.get("model"),
        "layer_id": capture.get("layer_id"),
        "position": capture.get("position"),
        "token_id": capture.get("token_id"),
        "shape": list(recurrent_out.shape),
        "device_matches_expected_bf16": comparison["device_matches_expected_bf16"],
        "device_expected_bf16_diff": comparison["device_expected_bf16_diff"],
        "cast_error": comparison["cast_error"],
        "conclusion": _conclusion(comparison),
    }


def compare_recurrent_bf16_cast(
    recurrent_out_f32: np.ndarray,
    recurrent_bf16_f32: np.ndarray,
) -> dict[str, Any]:
    recurrent_out_f32 = np.asarray(recurrent_out_f32, dtype=np.float32).reshape(-1)
    recurrent_bf16_f32 = np.asarray(recurrent_bf16_f32, dtype=np.float32).reshape(-1)
    if recurrent_out_f32.shape != recurrent_bf16_f32.shape:
        raise ValueError("recurrent_out_f32 and recurrent_bf16_f32 must have the same shape")
    expected_bf16 = bf16_to_float32(float_array_to_bf16_bits(recurrent_out_f32)).astype(np.float32)
    device_expected_diff = _diff_metrics(expected_bf16, recurrent_bf16_f32)
    return {
        "device_matches_expected_bf16": bool(np.array_equal(expected_bf16, recurrent_bf16_f32)),
        "device_expected_bf16_diff": device_expected_diff,
        "cast_error": _diff_metrics(recurrent_out_f32, recurrent_bf16_f32),
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
    cast = comparison["cast_error"]
    if comparison["device_matches_expected_bf16"]:
        return (
            "Device recurrent_bf16 exactly matches host BF16 rounding of recurrent_out; "
            f"the cast itself contributes max_abs={cast['max_abs_diff']:.6g} and "
            f"rms_abs={cast['rms_abs_diff']:.6g}, so downstream attn_out triage should "
            "focus on ssm_out/input precision or earlier GDN math, not a cast mismatch."
        )
    expected = comparison["device_expected_bf16_diff"]
    return (
        "Device recurrent_bf16 does not match host BF16 rounding; investigate f32_to_bf16 "
        "or buffer ordering before ssm_out. "
        f"device_expected max_abs={expected['max_abs_diff']:.6g}."
    )


if __name__ == "__main__":
    main()
