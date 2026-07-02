#!/usr/bin/env python3
"""Compare hipEngine all-layer router traces against llama.cpp tensor traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-02-mtp-target-router-trace-cross-engine-diagnostic.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipengine-router-trace", type=Path, required=True)
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True)
    parser.add_argument("--llamacpp-cycle", type=int, required=True)
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_router_trace_compare_artifact(
        hipengine_router_trace_path=args.hipengine_router_trace,
        llamacpp_jsonl_path=args.llamacpp_jsonl,
        llamacpp_cycle=args.llamacpp_cycle,
        row=args.row,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "layer_count": len(artifact["layers"]),
                "first_topk_mismatch_layer": artifact["first_topk_mismatch_layer"],
                "matched_topk_layers": artifact["matched_topk_layers"],
                "conclusion": artifact["conclusion"],
            },
            indent=2,
        )
    )


def build_router_trace_compare_artifact(
    *,
    hipengine_router_trace_path: Path,
    llamacpp_jsonl_path: Path,
    llamacpp_cycle: int,
    row: int,
) -> dict[str, Any]:
    hip_artifact = json.loads(hipengine_router_trace_path.read_text())
    hip_trace = _hip_router_trace(hip_artifact, row=row)
    llama_cycle = _llamacpp_cycle(llamacpp_jsonl_path, cycle=llamacpp_cycle)
    llama_values, duplicates = _llamacpp_row_values(llama_cycle, row=row)
    layers = []
    for hip_layer in hip_trace["layers"]:
        layer_id = int(hip_layer["layer"])
        top_k = len(hip_layer["moe_selected_experts"])
        hip_router = _array(hip_layer["values"], "moe_router_logits", "hipEngine")
        llama_router = _array_label(llama_values, f"ffn_moe_logits_{layer_id}", "llama.cpp")
        hip_top = [int(value) for value in hip_layer["moe_selected_experts"]]
        llama_top = _topk(llama_router, top_k)
        llama_weights = _array_label(
            llama_values, f"ffn_moe_weights_norm_{layer_id}", "llama.cpp"
        )
        hip_weights = np.asarray(hip_layer["moe_routing_weights"], dtype=np.float32)
        llama_gate = _array_label(
            llama_values, f"shared_expert_gate_{layer_id}", "llama.cpp"
        )
        hip_gate = np.asarray(hip_layer["moe_shared_gate"], dtype=np.float32)
        layers.append(
            {
                "layer": layer_id,
                "layer_type": hip_layer.get("layer_type"),
                "topk_match": hip_top == llama_top,
                "rank_mismatches": [
                    {
                        "rank": int(rank),
                        "hipengine": int(hip_expert),
                        "llamacpp": int(llama_expert),
                    }
                    for rank, (hip_expert, llama_expert) in enumerate(zip(hip_top, llama_top))
                    if int(hip_expert) != int(llama_expert)
                ],
                "hip_only_experts": [
                    int(expert) for expert in hip_top if int(expert) not in set(llama_top)
                ],
                "llamacpp_only_experts": [
                    int(expert) for expert in llama_top if int(expert) not in set(hip_top)
                ],
                "hipengine_topk": _top_values(hip_router, hip_top),
                "llamacpp_topk": _top_values(llama_router, llama_top),
                "router_delta": _numeric_delta(llama_router, hip_router),
                "routing_weights_delta": _numeric_delta(llama_weights, hip_weights),
                "shared_gate": {
                    "hipengine_logit": float(hip_gate[0]),
                    "llamacpp_logit": float(llama_gate[0]),
                    "delta_hip_minus_llama": float(hip_gate[0] - llama_gate[0]),
                },
            }
        )
    first_mismatch = next((layer["layer"] for layer in layers if not layer["topk_match"]), None)
    matched_topk_layers = [int(layer["layer"]) for layer in layers if layer["topk_match"]]
    artifact = {
        "schema": 1,
        "kind": "mtp_target_router_trace_cross_engine_compare",
        "status": "complete",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "hipengine_router_trace": str(hipengine_router_trace_path),
            "llamacpp_jsonl": str(llamacpp_jsonl_path),
            "llamacpp_cycle": int(llamacpp_cycle),
            "row": int(row),
        },
        "hipengine": {
            "model": hip_artifact.get("model"),
            "source_trace": hip_artifact.get("source_trace"),
            "command": hip_artifact.get("command"),
            "cycle": hip_artifact.get("probe", {}).get("cycle"),
            "sampled_tokens": hip_artifact.get("result", {}).get("sampled_tokens"),
            "accepted_draft_tokens": hip_artifact.get("result", {}).get("accepted_draft_tokens"),
            "trace_row": {
                "row": hip_trace.get("row"),
                "position": hip_trace.get("position"),
                "input_token": hip_trace.get("input_token"),
                "trace_target_token": hip_trace.get("trace_target_token"),
            },
        },
        "llamacpp": {
            "cycle": llama_cycle.get("cycle"),
            "accepted_draft_tokens": llama_cycle.get("accepted_draft_tokens"),
            "accepted_token_ids": llama_cycle.get("accepted_token_ids"),
            "bonus_token_id": llama_cycle.get("bonus_token_id"),
            "duplicate_value_labels": duplicates,
        },
        "first_topk_mismatch_layer": first_mismatch,
        "matched_topk_layers": matched_topk_layers,
        "layers": layers,
    }
    artifact["conclusion"] = _conclusion(artifact)
    return artifact


def _hip_router_trace(artifact: dict[str, Any], *, row: int) -> dict[str, Any]:
    traces = artifact.get("result", {}).get("router_trace_captures", [])
    for trace in traces:
        if int(trace.get("row", -1)) == int(row):
            return trace
    raise ValueError(f"hipEngine artifact has no router_trace_captures row={row}")


def _llamacpp_cycle(path: Path, *, cycle: int) -> dict[str, Any]:
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("cycle", -1)) == int(cycle):
                return record
    raise ValueError(f"llama.cpp JSONL has no cycle={cycle}")


def _llamacpp_row_values(
    cycle_record: dict[str, Any], *, row: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    by_label: dict[str, list[np.ndarray]] = {}
    for trace in cycle_record.get("draft_hidden_state_trace", []):
        if int(trace.get("row_index", -1)) != int(row):
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
            ),
        }
        for label, arrays in by_label.items()
        if len(arrays) > 1
    }
    return values, duplicates


def _array(values: dict[str, Any], key: str, owner: str) -> np.ndarray:
    if key not in values:
        raise ValueError(f"{owner} trace missing raw values for {key}")
    return np.asarray(values[key], dtype=np.float32).reshape(-1)


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
    }


def _topk(values: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > values.size:
        raise ValueError("top_k exceeds vector length")
    return [int(index) for index in np.argsort(values)[::-1][:top_k]]


def _top_values(values: np.ndarray, indices: list[int]) -> list[dict[str, Any]]:
    return [{"expert": int(index), "logit": float(values[index])} for index in indices]


def _conclusion(artifact: dict[str, Any]) -> str:
    first = artifact["first_topk_mismatch_layer"]
    if first is None:
        return "All captured MoE router top-k selections match llama.cpp for this row."
    layer = next(layer for layer in artifact["layers"] if int(layer["layer"]) == int(first))
    delta = layer["router_delta"]
    return (
        f"First router top-k divergence is layer {first}: "
        f"rank mismatches={layer['rank_mismatches']}, "
        f"hip-only={layer['hip_only_experts']}, "
        f"llama-only={layer['llamacpp_only_experts']}; "
        f"router logits MAE/RMSE/cosine={delta['mean_abs_diff']:.6g}/"
        f"{delta['rmse']:.6g}/{delta['cosine']:.6g}."
    )


if __name__ == "__main__":
    main()
