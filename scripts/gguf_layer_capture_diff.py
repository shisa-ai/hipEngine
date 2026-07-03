#!/usr/bin/env python3
"""Compare arrays from two GGUF attention-layer capture artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_REFERENCE = Path(
    "benchmarks/results/mtp-gguf-iter278-layer3-full-attn-routing-full-arrays.json"
)
DEFAULT_CANDIDATE = Path(
    "benchmarks/results/mtp-gguf-iter280-layer3-full-attn-actual-routing-full-arrays.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter280-layer3-direct-vs-actual-diff.json")
DEFAULT_KEYS = ("hidden_in_f32", "attn_out_f32", "layer_out_f32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=280)
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    args = parser.parse_args()

    keys = tuple(key.strip() for key in args.keys.split(",") if key.strip())
    artifact = build_capture_diff_artifact(
        reference_path=args.reference,
        candidate_path=args.candidate,
        keys=keys,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "keys": list(artifact["comparisons"]),
                "max_abs_by_key": {
                    key: value["max_abs_diff"]
                    for key, value in artifact["comparisons"].items()
                },
            },
            indent=2,
        )
    )


def build_capture_diff_artifact(
    *,
    reference_path: Path,
    candidate_path: Path,
    keys: tuple[str, ...] = DEFAULT_KEYS,
    iteration: int = 280,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    comparisons = compare_capture_arrays(reference, candidate, keys=keys)
    return {
        "schema": 1,
        "kind": "mtp_gguf_layer_capture_array_diff",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "compared",
        "reference": _capture_metadata(reference_path, reference),
        "candidate": _capture_metadata(candidate_path, candidate),
        "comparisons": comparisons,
        "conclusion": _conclusion(comparisons, reference, candidate),
    }


def compare_capture_arrays(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    keys: tuple[str, ...] = DEFAULT_KEYS,
) -> dict[str, Any]:
    if not keys:
        raise ValueError("at least one array key is required")
    ref_arrays = _arrays(reference, label="reference")
    cand_arrays = _arrays(candidate, label="candidate")
    return {
        key: _diff_metrics(
            _array(ref_arrays, key, "reference"),
            _array(cand_arrays, key, "candidate"),
        )
        for key in keys
    }


def _capture_metadata(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    summary = artifact.get("capture_summary") or {}
    return {
        "path": str(path),
        "iteration": artifact.get("iteration"),
        "status": artifact.get("status"),
        "layer_id": artifact.get("layer_id"),
        "layer_type": artifact.get("layer_type") or summary.get("layer_type"),
        "position": artifact.get("position"),
        "token_id": artifact.get("token_id"),
        "run_preceding_layers": artifact.get("run_preceding_layers"),
        "preceding_layer_count": summary.get("preceding_layer_count"),
    }


def _arrays(artifact: dict[str, Any], *, label: str) -> dict[str, Any]:
    arrays = artifact.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError(f"{label} artifact must include arrays")
    return arrays


def _array(arrays: dict[str, Any], key: str, label: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"{label} artifact missing arrays.{key}")
    return np.asarray(arrays[key], dtype=np.float32).reshape(-1)


def _diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate arrays must have the same shape")
    diff = candidate - reference
    abs_diff = np.abs(diff)
    return {
        "count": int(reference.size),
        "max_abs_diff": float(np.max(abs_diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(abs_diff, dtype=np.float32)) if diff.size else 0.0,
        "rms_abs_diff": float(np.sqrt(np.mean(diff * diff, dtype=np.float32)))
        if diff.size
        else 0.0,
        "reference_rms": float(np.sqrt(np.mean(reference * reference, dtype=np.float32)))
        if diff.size
        else 0.0,
        "candidate_rms": float(np.sqrt(np.mean(candidate * candidate, dtype=np.float32)))
        if diff.size
        else 0.0,
        "reference_sample": [float(x) for x in reference[:8]],
        "candidate_sample": [float(x) for x in candidate[:8]],
        "diff_sample": [float(x) for x in diff[:8]],
    }


def _conclusion(
    comparisons: dict[str, Any], reference: dict[str, Any], candidate: dict[str, Any]
) -> str:
    candidate_preceding = bool(candidate.get("run_preceding_layers"))
    hidden = comparisons.get("hidden_in_f32", {})
    attn = comparisons.get("attn_out_f32", {})
    if candidate_preceding:
        return (
            "Actual in-stack capture differs from direct-layer capture because the selected token "
            "is first propagated through preceding layers. "
            f"hidden_in max_abs={hidden.get('max_abs_diff', 0.0):.6g}; "
            f"attn_out max_abs={attn.get('max_abs_diff', 0.0):.6g}."
        )
    return (
        "Candidate capture did not declare run_preceding_layers=true; compare metadata before "
        "using this artifact for cross-layer propagation conclusions."
    )


if __name__ == "__main__":
    main()
