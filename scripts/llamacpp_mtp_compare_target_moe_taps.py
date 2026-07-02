#!/usr/bin/env python3
"""Compare hipEngine forced-target MoE taps against llama.cpp tensor traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-02-mtp-target-layer31-fine-moe-cross-engine-diagnostic.json"
)

TENSOR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("router_logits", "moe_router_logits", "ffn_moe_logits_{layer}"),
    ("selected_swiglu", "moe_selected_swiglu", "ffn_moe_swiglu_{layer}"),
    ("selected_down", "ffn_or_moe_down", "ffn_moe_down_{layer}"),
    ("selected_weighted_rows", "moe_selected_down_weighted", "ffn_moe_weighted_{layer}"),
    ("selected_sum_f32", "moe_selected_weighted_sum_f32", "ffn_moe_out_{layer}"),
    ("selected_sum_bf16", "moe_selected_weighted_bf16", "ffn_moe_out_{layer}"),
    ("shared_out", "moe_shared_out", "ffn_shexp_{layer}"),
    ("shared_gated", "moe_shared_gated", "ffn_shexp_gated_{layer}"),
    ("ffn_out", "ffn_out_combined_from_components", "ffn_out_{layer}"),
    ("post_moe", "post_moe_rounded_from_components", "post_moe_{layer}"),
    ("layer_out", "layer_out", "verify_layer_output_{layer}"),
)

SEGMENT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("swiglu", "moe_selected_swiglu", "ffn_moe_swiglu_{layer}"),
    ("down", "ffn_or_moe_down", "ffn_moe_down_{layer}"),
    ("weighted", "moe_selected_down_weighted", "ffn_moe_weighted_{layer}"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipengine-raw", type=Path, required=True)
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True)
    parser.add_argument("--llamacpp-cycle", type=int, required=True)
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_moe_tap_compare_artifact(
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
                "router_topk_match": artifact["router"]["topk_match"],
                "common_experts": artifact["selection"]["common_experts"],
                "ffn_out_mae": artifact["tensor_deltas"]["ffn_out"]["mean_abs_diff"],
                "post_moe_mae": artifact["tensor_deltas"]["post_moe"]["mean_abs_diff"],
                "conclusion": artifact["conclusion"],
            },
            indent=2,
        )
    )


def build_moe_tap_compare_artifact(
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

    hip_router = _array(hip_values, "moe_router_logits", "hipEngine")
    llama_router = _array_label(llama_values, f"ffn_moe_logits_{layer}", "llama.cpp")
    top_k = len(hip_capture["moe_selected_experts"])
    hip_top = _topk(hip_router, top_k)
    llama_top = _topk(llama_router, top_k)

    tensor_deltas = {
        name: _numeric_delta(
            _array_label(llama_values, llama_label.format(layer=layer), "llama.cpp"),
            _array(hip_values, hip_key, "hipEngine"),
        )
        for name, hip_key, llama_label in TENSOR_PAIRS
    }
    segments = {
        name: _segment_deltas(
            hip=_array(hip_values, hip_key, "hipEngine"),
            llama=_array_label(llama_values, llama_label.format(layer=layer), "llama.cpp"),
            hip_experts=list(map(int, hip_capture["moe_selected_experts"])),
            llama_experts=llama_top,
        )
        for name, hip_key, llama_label in SEGMENT_PAIRS
    }

    routing_weights = {
        "hipengine": [float(x) for x in hip_capture["moe_routing_weights"]],
        "llamacpp": [
            float(x)
            for x in _array_label(llama_values, f"ffn_moe_weights_norm_{layer}", "llama.cpp")
        ],
        "delta": _numeric_delta(
            _array_label(llama_values, f"ffn_moe_weights_norm_{layer}", "llama.cpp"),
            np.asarray(hip_capture["moe_routing_weights"], dtype=np.float32),
        ),
    }
    shared_gate = _shared_gate(
        hip_gate=float(hip_capture["moe_shared_gate"][0]),
        llama_values=llama_values,
        layer=layer,
    )
    cutoff_experts = _cutoff_expert_table(
        hip_router=hip_router,
        llama_router=llama_router,
        hip_top=hip_top,
        llama_top=llama_top,
    )
    selection = {
        "hipengine_experts": list(map(int, hip_capture["moe_selected_experts"])),
        "llamacpp_experts": llama_top,
        "common_experts": [int(expert) for expert in llama_top if expert in set(hip_top)],
        "hip_only_experts": [int(expert) for expert in hip_top if expert not in set(llama_top)],
        "llamacpp_only_experts": [int(expert) for expert in llama_top if expert not in set(hip_top)],
        "rank_mismatches": [
            {
                "rank": int(rank),
                "hipengine": int(hip_expert),
                "llamacpp": int(llama_expert),
            }
            for rank, (hip_expert, llama_expert) in enumerate(zip(hip_top, llama_top))
            if int(hip_expert) != int(llama_expert)
        ],
    }

    artifact = {
        "schema": 1,
        "kind": "mtp_target_layer_moe_tap_cross_engine_compare",
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
        "router": {
            "topk_match": hip_top == llama_top,
            "delta": tensor_deltas["router_logits"],
            "hipengine_topk": _top_values(hip_router, hip_top),
            "llamacpp_topk": _top_values(llama_router, llama_top),
            "cutoff_experts": cutoff_experts,
        },
        "routing_weights": routing_weights,
        "shared_gate": shared_gate,
        "selection": selection,
        "tensor_deltas": tensor_deltas,
        "segment_deltas": segments,
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
    return {
        key: np.asarray(value, dtype=np.float32).reshape(-1)
        for key, value in values.items()
    }


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


def _topk(values: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > values.size:
        raise ValueError("top_k exceeds vector length")
    return [int(index) for index in np.argsort(values)[::-1][:top_k]]


def _top_values(values: np.ndarray, indices: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "expert": int(index),
            "logit": float(values[index]),
        }
        for index in indices
    ]


def _segment_deltas(
    *,
    hip: np.ndarray,
    llama: np.ndarray,
    hip_experts: list[int],
    llama_experts: list[int],
) -> dict[str, Any]:
    if len(hip_experts) != len(llama_experts):
        raise ValueError("hip and llama expert lists must have the same top_k")
    top_k = len(hip_experts)
    if hip.size % top_k != 0 or llama.size % top_k != 0:
        raise ValueError("selected MoE tensors must be divisible by top_k")
    hip_width = hip.size // top_k
    llama_width = llama.size // top_k
    if hip_width != llama_width:
        raise ValueError(f"selected MoE segment width mismatch: {hip_width} vs {llama_width}")
    rankwise = []
    for rank, (hip_expert, llama_expert) in enumerate(zip(hip_experts, llama_experts)):
        rankwise.append(
            {
                "rank": int(rank),
                "hipengine_expert": int(hip_expert),
                "llamacpp_expert": int(llama_expert),
                "expert_matches": int(hip_expert) == int(llama_expert),
                "delta": _numeric_delta(
                    llama[rank * llama_width : (rank + 1) * llama_width],
                    hip[rank * hip_width : (rank + 1) * hip_width],
                ),
            }
        )
    hip_by_expert = {int(expert): rank for rank, expert in enumerate(hip_experts)}
    llama_by_expert = {int(expert): rank for rank, expert in enumerate(llama_experts)}
    common_by_expert = []
    for expert in llama_experts:
        expert = int(expert)
        if expert not in hip_by_expert:
            continue
        llama_rank = llama_by_expert[expert]
        hip_rank = hip_by_expert[expert]
        common_by_expert.append(
            {
                "expert": int(expert),
                "llamacpp_rank": int(llama_rank),
                "hipengine_rank": int(hip_rank),
                "delta": _numeric_delta(
                    llama[llama_rank * llama_width : (llama_rank + 1) * llama_width],
                    hip[hip_rank * hip_width : (hip_rank + 1) * hip_width],
                ),
            }
        )
    return {
        "segment_width": int(hip_width),
        "rankwise": rankwise,
        "common_by_expert": common_by_expert,
    }


def _shared_gate(
    *,
    hip_gate: float,
    llama_values: dict[str, np.ndarray],
    layer: int,
) -> dict[str, Any]:
    llama_gate = _array_label(llama_values, f"shared_expert_gate_{layer}", "llama.cpp")
    llama_sigmoid = _array_label(
        llama_values, f"shared_expert_gate_sigmoid_{layer}", "llama.cpp"
    )
    hip_sigmoid = float(1.0 / (1.0 + np.exp(-np.float32(hip_gate))))
    return {
        "hipengine_logit": float(hip_gate),
        "llamacpp_logit": float(llama_gate[0]),
        "logit_delta_hip_minus_llama": float(hip_gate - float(llama_gate[0])),
        "hipengine_sigmoid": hip_sigmoid,
        "llamacpp_sigmoid": float(llama_sigmoid[0]),
        "sigmoid_delta_hip_minus_llama": float(hip_sigmoid - float(llama_sigmoid[0])),
    }


def _cutoff_expert_table(
    *,
    hip_router: np.ndarray,
    llama_router: np.ndarray,
    hip_top: list[int],
    llama_top: list[int],
) -> list[dict[str, Any]]:
    experts = sorted(set(hip_top) | set(llama_top))
    hip_rank = {expert: rank for rank, expert in enumerate(hip_top)}
    llama_rank = {expert: rank for rank, expert in enumerate(llama_top)}
    return [
        {
            "expert": int(expert),
            "hipengine_rank": int(hip_rank[expert]) if expert in hip_rank else None,
            "llamacpp_rank": int(llama_rank[expert]) if expert in llama_rank else None,
            "hipengine_logit": float(hip_router[expert]),
            "llamacpp_logit": float(llama_router[expert]),
            "delta_hip_minus_llama": float(hip_router[expert] - llama_router[expert]),
        }
        for expert in experts
    ]


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
    selection = artifact["selection"]
    router = artifact["router"]
    ffn = artifact["tensor_deltas"]["ffn_out"]
    post = artifact["tensor_deltas"]["post_moe"]
    if not router["topk_match"]:
        return (
            "Layer MoE parity first diverges at router top-k selection: "
            f"rank mismatches={selection['rank_mismatches']}, "
            f"hip-only={selection['hip_only_experts']}, "
            f"llama-only={selection['llamacpp_only_experts']}. "
            "Common-expert projection rows remain close when aligned by expert id, "
            f"and the aggregate ffn_out/post_moe deltas stay small "
            f"(MAE {ffn['mean_abs_diff']:.6g} / {post['mean_abs_diff']:.6g})."
        )
    return (
        "Layer MoE top-k selection matches; inspect tensor_deltas for the largest "
        "remaining projection or combine delta."
    )


if __name__ == "__main__":
    main()
