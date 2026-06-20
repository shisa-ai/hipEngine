#!/usr/bin/env python3
"""Create a compact summary of a full-array GGUF layer capture artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CAPTURE = Path(
    "benchmarks/results/mtp-gguf-iter280-layer3-full-attn-actual-routing-full-arrays.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter281-layer3-actual-checkpoint-summary.json"
)
DEFAULT_KEYS = (
    "hidden_in_f32",
    "attn_out_f32",
    "post_norm_f32",
    "residual_f32",
    "ffn_or_moe_down_f32",
    "moe_shared_out_f32",
    "moe_routing_weights_f32",
    "moe_shared_gate_f32",
    "moe_selected_experts_i64",
    "layer_out_f32",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=281)
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    keys = tuple(key.strip() for key in args.keys.split(",") if key.strip())
    artifact = build_capture_summary_artifact(
        args.capture,
        keys=keys,
        top_n=args.top_n,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "array_count": len(artifact["arrays"]),
                "layer_id": artifact["capture"]["layer_id"],
                "layer_type": artifact["capture"]["layer_type"],
                "run_preceding_layers": artifact["capture"][
                    "run_preceding_layers"
                ],
            },
            indent=2,
        )
    )


def build_capture_summary_artifact(
    capture_path: Path,
    *,
    keys: tuple[str, ...] = DEFAULT_KEYS,
    top_n: int = 8,
    iteration: int = 281,
) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text())
    arrays = _arrays(capture)
    if not keys:
        raise ValueError("at least one array key is required")
    summaries = {key: summarize_array(_array(arrays, key), top_n=top_n) for key in keys}
    return {
        "schema": 1,
        "kind": "mtp_gguf_layer_capture_compact_summary",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "summarized",
        "source_capture": str(capture_path),
        "capture": _capture_metadata(capture),
        "top_n": int(top_n),
        "arrays": summaries,
        "conclusion": _conclusion(capture, summaries),
    }


def summarize_array(array: np.ndarray, *, top_n: int = 8) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    values = np.asarray(array)
    flat = values.reshape(-1)
    finite = bool(np.all(np.isfinite(flat.astype(np.float32)))) if flat.size else True
    payload = _hash_payload(values)
    summary: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "count": int(flat.size),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "finite": finite,
        "sample_first": _sample(flat[:top_n]),
        "sample_last": _sample(flat[-top_n:]) if flat.size else [],
    }
    if np.issubdtype(values.dtype, np.integer):
        summary.update(
            {
                "min": int(np.min(flat)) if flat.size else 0,
                "max": int(np.max(flat)) if flat.size else 0,
                "top_abs": _top_abs_payload(flat.astype(np.float32), top_n=top_n),
            }
        )
    else:
        float_flat = flat.astype(np.float32)
        summary.update(
            {
                "min": float(np.min(float_flat)) if flat.size else 0.0,
                "max": float(np.max(float_flat)) if flat.size else 0.0,
                "mean": float(np.mean(float_flat, dtype=np.float32)) if flat.size else 0.0,
                "rms": float(np.sqrt(np.mean(float_flat * float_flat, dtype=np.float32)))
                if flat.size
                else 0.0,
                "top_abs": _top_abs_payload(float_flat, top_n=top_n),
            }
        )
    return summary


def _capture_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    summary = capture.get("capture_summary") or {}
    return {
        "iteration": capture.get("iteration"),
        "status": capture.get("status"),
        "model": capture.get("model"),
        "position": capture.get("position"),
        "token_id": capture.get("token_id"),
        "layer_id": capture.get("layer_id"),
        "layer_type": capture.get("layer_type") or summary.get("layer_type"),
        "run_preceding_layers": capture.get("run_preceding_layers"),
        "preceding_layer_count": summary.get("preceding_layer_count"),
        "hidden_size": summary.get("hidden_size"),
        "top_k": summary.get("top_k"),
        "finite": summary.get("finite"),
    }


def _arrays(capture: dict[str, Any]) -> dict[str, Any]:
    arrays = capture.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("capture artifact must include arrays")
    return arrays


def _array(arrays: dict[str, Any], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"capture artifact missing arrays.{key}")
    values = arrays[key]
    if key.endswith("_i64"):
        return np.asarray(values, dtype=np.int64)
    return np.asarray(values, dtype=np.float32)


def _hash_payload(values: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(values)
    header = f"{contiguous.dtype}|{tuple(contiguous.shape)}|".encode("utf-8")
    return header + contiguous.tobytes()


def _sample(values: np.ndarray) -> list[int] | list[float]:
    if np.issubdtype(values.dtype, np.integer):
        return [int(x) for x in values]
    return [float(x) for x in values.astype(np.float32)]


def _top_abs_payload(values: np.ndarray, *, top_n: int) -> list[dict[str, float | int]]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return []
    k = int(min(top_n, flat.size))
    indices = np.argpartition(-np.abs(flat), k - 1)[:k]
    order = np.lexsort((indices, -np.abs(flat[indices])))
    return [
        {"index": int(idx), "value": float(flat[idx]), "abs": float(abs(flat[idx]))}
        for idx in indices[order]
    ]


def _conclusion(capture: dict[str, Any], summaries: dict[str, Any]) -> str:
    metadata = _capture_metadata(capture)
    layer_id = metadata["layer_id"]
    layer_type = metadata["layer_type"]
    preceding = metadata["preceding_layer_count"]
    hidden_hash = summaries.get("hidden_in_f32", {}).get("sha256", "missing")
    attn_hash = summaries.get("attn_out_f32", {}).get("sha256", "missing")
    return (
        f"Compact checkpoint summary for layer {layer_id} ({layer_type}) with "
        f"preceding_layer_count={preceding}; hidden_in sha256={hidden_hash}, "
        f"attn_out sha256={attn_hash}. Use these hashes/samples to align external "
        "llama.cpp trace checkpoints without reloading the full-array artifact."
    )


if __name__ == "__main__":
    main()
