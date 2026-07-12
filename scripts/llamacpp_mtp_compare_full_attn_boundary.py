#!/usr/bin/env python3
"""Compare hipEngine full-attention layer taps against llama.cpp traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    load_layer0_attn_norm_weight,
)
from scripts.llamacpp_mtp_compare_layer0_linear_attn import (
    _attn_norm_formula_assessment,
    _hip_capture,
    _hip_metadata,
    _hip_values,
    _input_boundary_deltas,
    _llamacpp_cycle,
    _llamacpp_metadata,
    _llamacpp_row_values,
    _maybe_mae,
    _numeric_delta,
    _process_h_input_context,
)

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-03-mtp-target-full-attn-cross-engine-diagnostic.json"
)

TENSOR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("attn_out", "attn_out", "attn_output_{layer}"),
    ("attn_residual", "attn_residual", "attn_residual_{layer}"),
    ("attn_post_norm", "attn_post_norm", "attn_post_norm_{layer}"),
    ("ffn_out", "ffn_out_combined_from_components", "ffn_out_{layer}"),
    ("post_moe", "post_moe_rounded_from_components", "post_moe_{layer}"),
)

OPTIONAL_TENSOR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("attn_post_norm_bf16", "attn_post_norm_bf16", "attn_post_norm_{layer}"),
    ("attn_post_norm_router_input", "attn_post_norm_router_input", "attn_post_norm_{layer}"),
    ("layer_out_vs_post_moe", "layer_out", "post_moe_{layer}"),
    ("layer_out_vs_verify_layer_output", "layer_out", "verify_layer_output_{layer}"),
)

ATTN_OUT_CLOSE_MAE = 1.0e-3
BOUNDARY_CLOSE_MAE = 1.25e-3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipengine-raw", type=Path, required=True)
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True)
    parser.add_argument("--llamacpp-cycle", type=int, required=True)
    parser.add_argument(
        "--llamacpp-task-id",
        type=int,
        default=None,
        help="Optional task_id filter for unfiltered multi-task llama.cpp JSONL traces.",
    )
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument(
        "--hipengine-capture-source",
        choices=("auto", "isolated", "scored"),
        default="auto",
        help="Which hipEngine layer-boundary capture block to compare.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_full_attn_compare_artifact(
        hipengine_raw_path=args.hipengine_raw,
        llamacpp_jsonl_path=args.llamacpp_jsonl,
        llamacpp_cycle=args.llamacpp_cycle,
        llamacpp_task_id=args.llamacpp_task_id,
        row=args.row,
        layer=args.layer,
        hipengine_capture_source=args.hipengine_capture_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "hidden_in_mae": artifact["input_boundary_deltas"][
                    "hidden_in_vs_prev_layer_output"
                ].get("mean_abs_diff"),
                "attn_norm_mae": artifact["input_boundary_deltas"][
                    "attn_norm_input"
                ].get("mean_abs_diff"),
                "attn_out_mae": artifact["tensor_deltas"]["attn_out"][
                    "mean_abs_diff"
                ],
                "post_moe_mae": artifact["tensor_deltas"]["post_moe"][
                    "mean_abs_diff"
                ],
                "boundary_assessment": artifact["boundary_assessment"]["status"],
                "conclusion": artifact["conclusion"],
            },
            indent=2,
        )
    )


def build_full_attn_compare_artifact(
    *,
    hipengine_raw_path: Path,
    llamacpp_jsonl_path: Path,
    llamacpp_cycle: int,
    row: int,
    layer: int,
    llamacpp_task_id: int | None = None,
    hipengine_capture_source: str = "auto",
    attn_norm_weight_loader: Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    hip_artifact = json.loads(hipengine_raw_path.read_text())
    hip_capture = _hip_capture(
        hip_artifact,
        layer=layer,
        row=row,
        capture_source=hipengine_capture_source,
    )
    hip_values = _hip_values(hip_capture)
    llama_cycle = _llamacpp_cycle(
        llamacpp_jsonl_path,
        cycle=llamacpp_cycle,
        task_id=llamacpp_task_id,
    )
    llama_values, duplicates = _llamacpp_row_values(llama_cycle, row=row)

    input_boundary_deltas = _input_boundary_deltas(
        hip_values=hip_values,
        llama_values=llama_values,
        layer=layer,
    )
    attn_norm_formula = _attn_norm_formula_assessment(
        hip_values=hip_values,
        llama_values=llama_values,
        layer=layer,
        model_path=Path(str(hip_artifact.get("model", ""))),
        weight_loader=attn_norm_weight_loader or load_layer0_attn_norm_weight,
    )
    tensor_deltas = {
        name: _numeric_delta(
            _required_label(llama_values, llama_label.format(layer=layer), "llama.cpp"),
            _required_key(hip_values, hip_key, "hipEngine"),
        )
        for name, hip_key, llama_label in TENSOR_PAIRS
    }
    optional_tensor_deltas = _optional_deltas(
        pairs=OPTIONAL_TENSOR_PAIRS,
        hip_values=hip_values,
        llama_values=llama_values,
        layer=layer,
    )

    artifact = {
        "schema": 1,
        "kind": "mtp_target_full_attn_cross_engine_compare",
        "status": "complete",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "hipengine_raw": str(hipengine_raw_path),
            "llamacpp_jsonl": str(llamacpp_jsonl_path),
            "llamacpp_cycle": int(llamacpp_cycle),
            "llamacpp_task_id": int(llamacpp_task_id) if llamacpp_task_id is not None else None,
            "row": int(row),
            "layer": int(layer),
            "hipengine_capture_source": hip_capture.get("capture_source", "isolated_layer_replay"),
        },
        "hipengine": _hip_metadata(hip_artifact, hip_capture),
        "llamacpp": _llamacpp_metadata(llama_cycle, duplicates),
        "input_boundary_deltas": input_boundary_deltas,
        "attn_norm_formula_assessment": attn_norm_formula,
        "tensor_deltas": tensor_deltas,
        "optional_tensor_deltas": optional_tensor_deltas,
        "process_h_input_context": _process_h_input_context(llama_values),
    }
    artifact["boundary_assessment"] = _boundary_assessment(artifact)
    artifact["conclusion"] = _conclusion(artifact)
    return artifact


def _required_key(values: dict[str, np.ndarray], key: str, owner: str) -> np.ndarray:
    if key not in values:
        raise ValueError(f"{owner} trace missing raw values for {key}")
    return values[key]


def _required_label(values: dict[str, np.ndarray], label: str, owner: str) -> np.ndarray:
    if label not in values:
        raise ValueError(f"{owner} trace missing raw values for {label}")
    return values[label]


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


def _boundary_assessment(artifact: dict[str, Any]) -> dict[str, Any]:
    layer = int(artifact["inputs"]["layer"])
    inputs = artifact["input_boundary_deltas"]
    tensor = artifact["tensor_deltas"]
    formula = artifact["attn_norm_formula_assessment"]
    hidden_prev_mae = _maybe_mae(inputs.get("hidden_in_vs_prev_layer_output", {}))
    attn_norm_mae = _maybe_mae(inputs.get("attn_norm_input", {}))
    attn_out_mae = float(tensor["attn_out"]["mean_abs_diff"])
    residual_mae = float(tensor["attn_residual"]["mean_abs_diff"])
    post_norm_mae = float(tensor["attn_post_norm"]["mean_abs_diff"])
    ffn_mae = float(tensor["ffn_out"]["mean_abs_diff"])
    post_moe_mae = float(tensor["post_moe"]["mean_abs_diff"])

    if attn_out_mae > max(ATTN_OUT_CLOSE_MAE, (hidden_prev_mae or 0.0) * 2.0):
        status = "full_attention_output_cliff"
        reason = (
            "full-attention output drift is larger than the incoming hidden drift; "
            "inspect the full-attention graph/kernel before moving farther upstream"
        )
        next_split = f"layer-{layer} full-attention Q/K/V, RoPE, mask, and softmax split"
    elif formula.get("classification") in {
        "attn_norm_delta_explained_by_input_delta",
        "attn_norm_delta_mostly_explained_by_input_delta",
    } and hidden_prev_mae is not None:
        status = "incoming_hidden_drift"
        reason = (
            "attention RMSNorm arithmetic reproduces both engines and the layer's "
            "attention/MoE outputs remain close, so the remaining delta is already "
            "present at this layer input"
        )
        next_split = f"layer-{max(layer - 1, 0)} scored boundary split"
    elif post_moe_mae <= BOUNDARY_CLOSE_MAE:
        status = "no_full_attention_cliff"
        reason = (
            "full-attention and MoE outputs stay near the current boundary threshold; "
            "continue upstream unless a later router cutoff needs a narrower probe"
        )
        next_split = f"layer-{max(layer - 1, 0)} scored boundary split"
    else:
        status = "full_attention_or_moe_drift_present"
        reason = (
            "this full-attention layer increases the post-MoE delta enough to justify "
            "a narrower attention/MoE split before moving upstream"
        )
        next_split = f"layer-{layer} attention-vs-MoE sub-split"

    return {
        "status": status,
        "reason": reason,
        "hidden_in_vs_prev_layer_output_mae": hidden_prev_mae,
        "attn_norm_input_mae": attn_norm_mae,
        "attn_out_mae": attn_out_mae,
        "attn_residual_mae": residual_mae,
        "attn_post_norm_mae": post_norm_mae,
        "ffn_out_mae": ffn_mae,
        "post_moe_mae": post_moe_mae,
        "next_split_needed": next_split,
    }


def _conclusion(artifact: dict[str, Any]) -> str:
    layer = int(artifact["inputs"]["layer"])
    assessment = artifact["boundary_assessment"]
    tensor = artifact["tensor_deltas"]
    hidden_prev = assessment.get("hidden_in_vs_prev_layer_output_mae")
    hidden_prefix = (
        f"input MAE {hidden_prev:.6g}, " if isinstance(hidden_prev, float) else ""
    )
    formula = artifact["attn_norm_formula_assessment"]
    formula_note = ""
    if formula.get("classification") in {
        "attn_norm_delta_explained_by_input_delta",
        "attn_norm_delta_mostly_explained_by_input_delta",
    }:
        formula_note = " CPU RMSNorm reproduces both engines' attn_norm rows."
    return (
        f"Layer-{layer} full-attention split status={assessment['status']}: "
        f"{hidden_prefix}attn_out MAE {tensor['attn_out']['mean_abs_diff']:.6g}, "
        f"attn_residual MAE {tensor['attn_residual']['mean_abs_diff']:.6g}, "
        f"ffn_out MAE {tensor['ffn_out']['mean_abs_diff']:.6g}, "
        f"post_moe MAE {tensor['post_moe']['mean_abs_diff']:.6g}."
        f"{formula_note} Next: {assessment['next_split_needed']}."
    )


if __name__ == "__main__":
    main()
