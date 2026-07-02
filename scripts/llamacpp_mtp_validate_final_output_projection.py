#!/usr/bin/env python3
"""Validate a llama.cpp final_output tap by reprojecting it through hipEngine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-02-mtp-target-layer0-final-output-reprojection-diagnostic.json"
)

ProjectFn = Callable[[np.ndarray], np.ndarray]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hipengine-raw", type=Path, required=True)
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True)
    parser.add_argument("--llamacpp-cycle", type=int, required=True)
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_projection_validation_artifact(
        model_path=args.model,
        hipengine_raw_path=args.hipengine_raw,
        llamacpp_jsonl_path=args.llamacpp_jsonl,
        llamacpp_cycle=args.llamacpp_cycle,
        row=args.row,
        layer=args.layer,
        command=" ".join(sys.argv),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "assessment": artifact["assessment"]["status"],
                "hip_reproject_mae": artifact["reprojection_deltas"][
                    "hip_recurrent_reproject_vs_hip_capture_attn_out"
                ]["mean_abs_diff"],
                "llama_final_reproject_mae": artifact["reprojection_deltas"][
                    "llama_final_reproject_vs_llama_linear"
                ]["mean_abs_diff"],
                "conclusion": artifact["conclusion"],
            },
            indent=2,
        )
    )


def build_projection_validation_artifact(
    *,
    model_path: Path | str,
    hipengine_raw_path: Path,
    llamacpp_jsonl_path: Path,
    llamacpp_cycle: int,
    row: int,
    layer: int,
    command: str | None = None,
    project_fn: ProjectFn | None = None,
) -> dict[str, Any]:
    hip_artifact = json.loads(hipengine_raw_path.read_text())
    hip_capture = _hip_capture(hip_artifact, layer=layer, row=row)
    hip_values = _hip_values(hip_capture)
    llama_cycle = _llamacpp_cycle(llamacpp_jsonl_path, cycle=llamacpp_cycle)
    llama_values = _llamacpp_row_values(llama_cycle, row=row)

    hip_recurrent = _array(hip_values, "recurrent_out", "hipEngine")
    hip_attn_out = _array(hip_values, "attn_out", "hipEngine")
    llama_final = _array(llama_values, f"final_output_{int(layer)}", "llama.cpp")
    llama_linear = _array(llama_values, f"linear_attn_out_{int(layer)}", "llama.cpp")

    projector = project_fn or _build_hip_ssm_out_projector(model_path, layer=layer)
    hip_reprojected = np.asarray(projector(hip_recurrent), dtype=np.float32).reshape(-1)
    llama_reprojected = np.asarray(projector(llama_final), dtype=np.float32).reshape(-1)

    reprojection_deltas = {
        "hip_capture_attn_out_vs_llama_linear": _numeric_delta(
            llama_linear, hip_attn_out
        ),
        "hip_recurrent_reproject_vs_hip_capture_attn_out": _numeric_delta(
            hip_attn_out, hip_reprojected
        ),
        "hip_recurrent_reproject_vs_llama_linear": _numeric_delta(
            llama_linear, hip_reprojected
        ),
        "llama_final_reproject_vs_llama_linear": _numeric_delta(
            llama_linear, llama_reprojected
        ),
        "llama_final_reproject_vs_hip_capture_attn_out": _numeric_delta(
            hip_attn_out, llama_reprojected
        ),
    }
    assessment = _assessment(reprojection_deltas)
    artifact = {
        "schema": 1,
        "kind": "mtp_target_layer0_final_output_reprojection",
        "status": "complete",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "inputs": {
            "model": str(model_path),
            "hipengine_raw": str(hipengine_raw_path),
            "llamacpp_jsonl": str(llamacpp_jsonl_path),
            "llamacpp_cycle": int(llamacpp_cycle),
            "row": int(row),
            "layer": int(layer),
        },
        "hipengine": {
            "cycle": hip_artifact.get("probe", {}).get("cycle"),
            "sampled_tokens": hip_artifact.get("result", {}).get("sampled_tokens"),
            "accepted_draft_tokens": hip_artifact.get("result", {}).get(
                "accepted_draft_tokens"
            ),
            "layer": hip_capture.get("layer"),
            "row": hip_capture.get("row"),
            "position": hip_capture.get("position"),
            "input_token": hip_capture.get("input_token"),
            "trace_target_token": hip_capture.get("trace_target_token"),
        },
        "llamacpp": {
            "cycle": llama_cycle.get("cycle"),
            "accepted_draft_tokens": llama_cycle.get("accepted_draft_tokens"),
            "accepted_token_ids": llama_cycle.get("accepted_token_ids"),
            "bonus_token_id": llama_cycle.get("bonus_token_id"),
        },
        "reprojection_deltas": reprojection_deltas,
        "samples": {
            "hip_reprojected": _sample(hip_reprojected),
            "llama_final_reprojected": _sample(llama_reprojected),
            "hip_capture_attn_out": _sample(hip_attn_out),
            "llama_linear_attn_out": _sample(llama_linear),
        },
        "assessment": assessment,
    }
    artifact["conclusion"] = _conclusion(artifact)
    return artifact


def _build_hip_ssm_out_projector(model_path: Path | str, *, layer: int) -> ProjectFn:
    os.environ.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")

    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.runtime.qwen35_gguf_runner import (
        GGUF_ACTIVATION_F32,
        Qwen35GGUFOneLayerProbe,
        bf16_to_float32,
        launch_gguf_linear,
    )

    runtime = get_hip_runtime()
    probe = Qwen35GGUFOneLayerProbe(model_path, layer_id=layer, runtime=runtime)
    in_features = int(probe.weights.config.ssm_inner_size)
    out_features = int(probe.weights.config.hidden_size)
    weight = probe.weights.layer(layer).weight("ssm_out")

    def project(values: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(1, -1))
        if x.shape != (1, in_features):
            raise ValueError(f"expected input shape (1, {in_features}), got {x.shape}")
        bits = np.empty((1, out_features), dtype=np.uint16)
        buffers = []
        try:
            in_buf = malloc(x.nbytes, runtime=runtime)
            out_buf = malloc(bits.nbytes, runtime=runtime)
            buffers.extend((in_buf, out_buf))
            copy_host_to_device(in_buf, host_array_ptr(x), x.nbytes, runtime=runtime)
            launch_gguf_linear(
                weight,
                in_buf.ptr,
                out_buf.ptr,
                rows=1,
                in_features=in_features,
                out_features=out_features,
                activation_dtype=GGUF_ACTIVATION_F32,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(bits), out_buf, bits.nbytes, runtime=runtime)
            return bf16_to_float32(bits.reshape(-1))
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)

    return project


def _hip_capture(artifact: dict[str, Any], *, layer: int, row: int) -> dict[str, Any]:
    captures = artifact.get("result", {}).get("layer_boundary_captures", [])
    for capture in captures:
        if int(capture.get("layer", -1)) == layer and int(capture.get("row", -1)) == row:
            return capture
    raise ValueError(f"hipEngine artifact has no layer_boundary_capture for layer={layer} row={row}")


def _hip_values(capture: dict[str, Any]) -> dict[str, np.ndarray]:
    values = capture.get("values")
    if not isinstance(values, dict):
        raise ValueError("hipEngine capture must include raw values")
    return {key: np.asarray(value, dtype=np.float32).reshape(-1) for key, value in values.items()}


def _llamacpp_cycle(path: Path, *, cycle: int) -> dict[str, Any]:
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("cycle", -1)) == cycle:
                return record
    raise ValueError(f"llama.cpp JSONL has no cycle={cycle}")


def _llamacpp_row_values(cycle_record: dict[str, Any], *, row: int) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for trace in cycle_record.get("draft_hidden_state_trace", []):
        if int(trace.get("row_index", -1)) != row or "values" not in trace:
            continue
        values.setdefault(
            str(trace["label"]), np.asarray(trace["values"], dtype=np.float32).reshape(-1)
        )
    return values


def _array(values: dict[str, np.ndarray], key: str, owner: str) -> np.ndarray:
    if key not in values:
        raise ValueError(f"{owner} trace missing raw values for {key}")
    return values[key]


def _numeric_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, candidate={candidate.shape}")
    diff = candidate - reference
    abs_diff = np.abs(diff)
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    return {
        "count": int(reference.size),
        "mean_abs_diff": float(np.mean(abs_diff, dtype=np.float32)) if reference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff, dtype=np.float32))) if reference.size else 0.0,
        "max_abs_diff": float(np.max(abs_diff)) if reference.size else 0.0,
        "cosine": float(np.dot(reference, candidate) / (reference_norm * candidate_norm))
        if reference_norm and candidate_norm
        else None,
        "reference_rms": float(np.sqrt(np.mean(reference * reference, dtype=np.float32)))
        if reference.size
        else 0.0,
        "candidate_rms": float(np.sqrt(np.mean(candidate * candidate, dtype=np.float32)))
        if candidate.size
        else 0.0,
    }


def _assessment(deltas: dict[str, Any]) -> dict[str, Any]:
    hip_reproject = deltas["hip_recurrent_reproject_vs_hip_capture_attn_out"][
        "mean_abs_diff"
    ]
    llama_reproject = deltas["llama_final_reproject_vs_llama_linear"]["mean_abs_diff"]
    if hip_reproject <= 1.0e-7 and llama_reproject >= 1.0e-2:
        return {
            "status": "final_output_trace_not_projectable",
            "reason": (
                "hipEngine recurrent_out exactly reconstructs hipEngine attn_out "
                "through ssm_out, but llama.cpp final_output does not reconstruct "
                "llama.cpp linear_attn_out through the same weight"
            ),
            "hip_reproject_mae": float(hip_reproject),
            "llama_final_reproject_mae": float(llama_reproject),
        }
    return {
        "status": "projection_consistent",
        "reason": "both pre-ssm vectors reproject consistently enough for semantic comparison",
        "hip_reproject_mae": float(hip_reproject),
        "llama_final_reproject_mae": float(llama_reproject),
    }


def _sample(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float32).reshape(-1)[:8]]


def _conclusion(artifact: dict[str, Any]) -> str:
    assessment = artifact["assessment"]
    deltas = artifact["reprojection_deltas"]
    return (
        f"{assessment['status']}: hip recurrent_out -> ssm_out MAE "
        f"{deltas['hip_recurrent_reproject_vs_hip_capture_attn_out']['mean_abs_diff']:.6g}; "
        f"llama final_output -> ssm_out MAE "
        f"{deltas['llama_final_reproject_vs_llama_linear']['mean_abs_diff']:.6g}."
    )


if __name__ == "__main__":
    main()
