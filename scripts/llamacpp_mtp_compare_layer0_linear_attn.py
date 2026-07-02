#!/usr/bin/env python3
"""Compare hipEngine layer-0 linear attention taps against llama.cpp traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-02-mtp-target-layer0-linear-attn-cross-engine-diagnostic.json"
)

TENSOR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("linear_attn_out", "attn_out", "linear_attn_out_{layer}"),
    ("attn_residual", "attn_residual", "attn_residual_{layer}"),
    ("attn_post_norm", "attn_post_norm", "attn_post_norm_{layer}"),
    ("ffn_out", "ffn_out_combined_from_components", "ffn_out_{layer}"),
    ("post_moe", "post_moe_rounded_from_components", "post_moe_{layer}"),
    ("layer_out", "layer_out", "post_moe_{layer}"),
)

PRE_SSM_OUT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("recurrent_out_vs_final_output", "recurrent_out", "final_output_{layer}"),
    ("recurrent_bf16_vs_final_output", "recurrent_bf16", "final_output_{layer}"),
)

POST_SSM_OUT_CLOSE_MAE = 1.0e-3
PRE_SSM_OUT_MISMATCH_MAE = 1.0e-2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipengine-raw", type=Path, required=True)
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True)
    parser.add_argument("--llamacpp-cycle", type=int, required=True)
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_linear_attn_compare_artifact(
        hipengine_raw_path=args.hipengine_raw,
        llamacpp_jsonl_path=args.llamacpp_jsonl,
        llamacpp_cycle=args.llamacpp_cycle,
        row=args.row,
        layer=args.layer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "linear_attn_out_mae": artifact["tensor_deltas"]["linear_attn_out"][
                    "mean_abs_diff"
                ],
                "attn_residual_mae": artifact["tensor_deltas"]["attn_residual"][
                    "mean_abs_diff"
                ],
                "post_moe_mae": artifact["tensor_deltas"]["post_moe"]["mean_abs_diff"],
                "pre_ssm_out_label_assessment": artifact[
                    "pre_ssm_out_label_assessment"
                ]["status"],
                "conclusion": artifact["conclusion"],
            },
            indent=2,
        )
    )


def build_linear_attn_compare_artifact(
    *,
    hipengine_raw_path: Path,
    llamacpp_jsonl_path: Path,
    llamacpp_cycle: int,
    row: int,
    layer: int,
) -> dict[str, Any]:
    hip_artifact = json.loads(hipengine_raw_path.read_text())
    hip_capture = _hip_capture(hip_artifact, layer=layer, row=row)
    hip_values = _hip_values(hip_capture)
    llama_cycle = _llamacpp_cycle(llamacpp_jsonl_path, cycle=llamacpp_cycle)
    llama_values, duplicates = _llamacpp_row_values(llama_cycle, row=row)

    tensor_deltas = {
        name: _numeric_delta(
            _array_label(llama_values, llama_label.format(layer=layer), "llama.cpp"),
            _array(hip_values, hip_key, "hipEngine"),
        )
        for name, hip_key, llama_label in TENSOR_PAIRS
    }
    pre_ssm_out_deltas = _optional_deltas(
        pairs=PRE_SSM_OUT_PAIRS,
        hip_values=hip_values,
        llama_values=llama_values,
        layer=layer,
    )

    artifact = {
        "schema": 1,
        "kind": "mtp_target_layer0_linear_attn_cross_engine_compare",
        "status": "complete",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "hipengine_raw": str(hipengine_raw_path),
            "llamacpp_jsonl": str(llamacpp_jsonl_path),
            "llamacpp_cycle": int(llamacpp_cycle),
            "row": int(row),
            "layer": int(layer),
        },
        "hipengine": _hip_metadata(hip_artifact, hip_capture),
        "llamacpp": _llamacpp_metadata(llama_cycle, duplicates),
        "tensor_deltas": tensor_deltas,
        "pre_ssm_out_deltas": pre_ssm_out_deltas,
        "pre_ssm_out_label_assessment": _pre_ssm_out_label_assessment(
            tensor_deltas=tensor_deltas,
            pre_ssm_out_deltas=pre_ssm_out_deltas,
        ),
        "final_output_summary": _summary_for_optional_label(
            llama_values, f"final_output_{int(layer)}"
        ),
    }
    artifact["conclusion"] = _conclusion(artifact)
    return artifact


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


def _llamacpp_row_values(
    cycle_record: dict[str, Any], *, row: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    by_label: dict[str, list[np.ndarray]] = {}
    for trace in cycle_record.get("draft_hidden_state_trace", []):
        if int(trace.get("row_index", -1)) != row:
            continue
        if "values" not in trace:
            continue
        label = str(trace["label"])
        by_label.setdefault(label, []).append(np.asarray(trace["values"], dtype=np.float32).reshape(-1))
    values = {label: arrays[0] for label, arrays in by_label.items()}
    duplicates = {
        label: {
            "count": len(arrays),
            "max_abs_vs_first": float(
                max(
                    (
                        np.max(np.abs(candidate - arrays[0]))
                        if candidate.shape == arrays[0].shape
                        else np.inf
                    )
                    for candidate in arrays[1:]
                )
            )
            if len(arrays) > 1
            else 0.0,
        }
        for label, arrays in by_label.items()
        if len(arrays) > 1
    }
    return values, duplicates


def _array(values: dict[str, np.ndarray], key: str, owner: str) -> np.ndarray:
    if key not in values:
        raise ValueError(f"{owner} trace missing raw values for {key}")
    return values[key]


def _array_label(values: dict[str, np.ndarray], label: str, owner: str) -> np.ndarray:
    if label not in values:
        raise ValueError(f"{owner} trace missing raw values for {label}")
    return values[label]


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
        "llamacpp_rms": float(np.sqrt(np.mean(reference * reference, dtype=np.float32)))
        if reference.size
        else 0.0,
        "hipengine_rms": float(np.sqrt(np.mean(candidate * candidate, dtype=np.float32)))
        if candidate.size
        else 0.0,
        "llamacpp_sample": [float(x) for x in reference[:8]],
        "hipengine_sample": [float(x) for x in candidate[:8]],
        "diff_sample": [float(x) for x in diff[:8]],
    }


def _optional_deltas(
    *,
    pairs: tuple[tuple[str, str, str], ...],
    hip_values: dict[str, np.ndarray],
    llama_values: dict[str, np.ndarray],
    layer: int,
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for name, hip_key, llama_label in pairs:
        formatted_label = llama_label.format(layer=layer)
        if hip_key not in hip_values or formatted_label not in llama_values:
            deltas[name] = {
                "status": "missing",
                "hipengine_key": hip_key,
                "llamacpp_label": formatted_label,
                "missing": [
                    owner
                    for owner, present in (
                        ("hipEngine", hip_key in hip_values),
                        ("llama.cpp", formatted_label in llama_values),
                    )
                    if not present
                ],
            }
            continue
        deltas[name] = {
            "status": "complete",
            "hipengine_key": hip_key,
            "llamacpp_label": formatted_label,
            **_numeric_delta(llama_values[formatted_label], hip_values[hip_key]),
        }
    return deltas


def _pre_ssm_out_label_assessment(
    *,
    tensor_deltas: dict[str, Any],
    pre_ssm_out_deltas: dict[str, Any],
) -> dict[str, Any]:
    linear_delta = tensor_deltas["linear_attn_out"]["mean_abs_diff"]
    recurrent = pre_ssm_out_deltas.get("recurrent_out_vs_final_output", {})
    if recurrent.get("status") != "complete":
        return {
            "status": "unavailable",
            "reason": "hipEngine recurrent_out or llama.cpp final_output values are missing",
            "linear_attn_out_mae": float(linear_delta),
        }

    recurrent_mae = float(recurrent["mean_abs_diff"])
    if linear_delta <= POST_SSM_OUT_CLOSE_MAE and recurrent_mae >= PRE_SSM_OUT_MISMATCH_MAE:
        return {
            "status": "unresolved_label_or_layout",
            "reason": (
                "llama.cpp final_output differs much more than downstream "
                "linear_attn_out; do not treat the direct final_output vs "
                "recurrent_out comparison as semantic drift until the trace "
                "label/layout is revalidated"
            ),
            "linear_attn_out_mae": float(linear_delta),
            "recurrent_out_vs_final_output_mae": recurrent_mae,
        }

    return {
        "status": "usable",
        "reason": "pre-ssm and post-ssm deltas are directionally consistent",
        "linear_attn_out_mae": float(linear_delta),
        "recurrent_out_vs_final_output_mae": recurrent_mae,
    }


def _summary_for_optional_label(values: dict[str, np.ndarray], label: str) -> dict[str, Any] | None:
    row = values.get(label)
    if row is None:
        return None
    return {
        "label": label,
        "count": int(row.size),
        "rms": float(np.sqrt(np.mean(row * row, dtype=np.float32))) if row.size else 0.0,
        "sample": [float(x) for x in row[:8]],
    }


def _hip_metadata(artifact: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": artifact.get("model"),
        "source_trace": artifact.get("source_trace"),
        "command": artifact.get("command"),
        "cycle": artifact.get("probe", {}).get("cycle"),
        "sampled_tokens": artifact.get("result", {}).get("sampled_tokens"),
        "accepted_draft_tokens": artifact.get("result", {}).get("accepted_draft_tokens"),
        "layer": capture.get("layer"),
        "row": capture.get("row"),
        "position": capture.get("position"),
        "input_token": capture.get("input_token"),
        "trace_target_token": capture.get("trace_target_token"),
    }


def _llamacpp_metadata(cycle_record: dict[str, Any], duplicates: dict[str, Any]) -> dict[str, Any]:
    return {
        "cycle": cycle_record.get("cycle"),
        "accepted_draft_tokens": cycle_record.get("accepted_draft_tokens"),
        "accepted_token_ids": cycle_record.get("accepted_token_ids"),
        "bonus_token_id": cycle_record.get("bonus_token_id"),
        "cycle_wall_ms": cycle_record.get("cycle_wall_ms"),
        "duplicate_value_labels": duplicates,
    }


def _conclusion(artifact: dict[str, Any]) -> str:
    deltas = artifact["tensor_deltas"]
    linear = deltas["linear_attn_out"]["mean_abs_diff"]
    residual = deltas["attn_residual"]["mean_abs_diff"]
    ffn = deltas["ffn_out"]["mean_abs_diff"]
    post = deltas["post_moe"]["mean_abs_diff"]
    pre_assessment = artifact.get("pre_ssm_out_label_assessment", {})
    pre_note = ""
    if pre_assessment.get("status") == "unresolved_label_or_layout":
        pre_note = (
            " Direct final_output vs hipEngine recurrent_out is label/layout "
            "unresolved because downstream ssm_out still matches closely."
        )
    return (
        "Layer-0 drift is already present at the linear-attention output: "
        f"linear_attn_out MAE {linear:.6g}, attention residual MAE {residual:.6g}, "
        f"ffn_out MAE {ffn:.6g}, post_moe MAE {post:.6g}. "
        "The next semantic target is the layer-0 linear-attention/GDN output contract."
        f"{pre_note}"
    )


if __name__ == "__main__":
    main()
