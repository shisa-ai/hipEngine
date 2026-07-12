#!/usr/bin/env python3
"""Compare early hipEngine target layer boundaries against llama.cpp traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-02-mtp-target-layer0-1-boundary-cross-engine-diagnostic.json"
)

TENSOR_PAIRS: tuple[tuple[str, int, str, str], ...] = (
    ("layer0_attn_norm", 0, "attn_norm", "attn_norm_0"),
    ("layer0_attn_residual", 0, "attn_residual", "attn_residual_0"),
    ("layer0_attn_post_norm", 0, "attn_post_norm", "attn_post_norm_0"),
    ("layer0_router_logits", 0, "moe_router_logits", "ffn_moe_logits_0"),
    ("layer0_ffn_out", 0, "ffn_out_combined_from_components", "ffn_out_0"),
    ("layer0_post_moe_rounded", 0, "post_moe_rounded_from_components", "post_moe_0"),
    ("layer0_layer_out", 0, "layer_out", "post_moe_0"),
    ("layer1_attn_norm", 1, "attn_norm", "attn_norm_1"),
    ("layer1_attn_residual", 1, "attn_residual", "attn_residual_1"),
    ("layer1_attn_post_norm", 1, "attn_post_norm", "attn_post_norm_1"),
    ("layer1_router_logits", 1, "moe_router_logits", "ffn_moe_logits_1"),
    ("layer1_ffn_out", 1, "ffn_out_combined_from_components", "ffn_out_1"),
    ("layer1_post_moe_rounded", 1, "post_moe_rounded_from_components", "post_moe_1"),
    ("layer1_layer_out", 1, "layer_out", "post_moe_1"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipengine-raw", type=Path, required=True)
    parser.add_argument("--llamacpp-jsonl", type=Path, required=True)
    parser.add_argument("--llamacpp-cycle", type=int, required=True)
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_early_boundary_compare_artifact(
        hipengine_raw_path=args.hipengine_raw,
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
                "hip_layer0_to_layer1_mae": artifact["boundary_deltas"][
                    "hip_layer0_layer_out_vs_layer1_hidden_in"
                ]["mean_abs_diff"],
                "cross_engine_boundary_mae": artifact["boundary_deltas"][
                    "hip_layer1_hidden_in_vs_llamacpp_post_moe_0"
                ]["mean_abs_diff"],
                "layer0_topk_match": artifact["router"][0]["topk_match"],
                "layer1_topk_match": artifact["router"][1]["topk_match"],
                "conclusion": artifact["conclusion"],
            },
            indent=2,
        )
    )


def build_early_boundary_compare_artifact(
    *,
    hipengine_raw_path: Path,
    llamacpp_jsonl_path: Path,
    llamacpp_cycle: int,
    row: int,
) -> dict[str, Any]:
    hip_artifact = json.loads(hipengine_raw_path.read_text())
    hip_captures = {
        layer: _hip_values(_hip_capture(hip_artifact, layer=layer, row=row))
        for layer in (0, 1)
    }
    hip_capture_meta = {
        layer: _hip_capture(hip_artifact, layer=layer, row=row)
        for layer in (0, 1)
    }
    llama_cycle = _llamacpp_cycle(llamacpp_jsonl_path, cycle=llamacpp_cycle)
    llama_values, duplicates = _llamacpp_row_values(llama_cycle, row=row)

    tensor_deltas = {
        name: _numeric_delta(
            _array_label(llama_values, llama_label, "llama.cpp"),
            _array(hip_captures[layer], hip_key, "hipEngine"),
        )
        for name, layer, hip_key, llama_label in TENSOR_PAIRS
    }
    boundary_deltas = {
        "hip_layer0_layer_out_vs_layer1_hidden_in": _numeric_delta(
            _array(hip_captures[0], "layer_out", "hipEngine"),
            _array(hip_captures[1], "hidden_in", "hipEngine"),
        ),
        "hip_layer0_post_moe_vs_layer1_hidden_in": _numeric_delta(
            _array(hip_captures[0], "post_moe_rounded_from_components", "hipEngine"),
            _array(hip_captures[1], "hidden_in", "hipEngine"),
        ),
        "hip_layer0_layer_out_vs_llamacpp_post_moe_0": _numeric_delta(
            _array_label(llama_values, "post_moe_0", "llama.cpp"),
            _array(hip_captures[0], "layer_out", "hipEngine"),
        ),
        "hip_layer1_hidden_in_vs_llamacpp_post_moe_0": _numeric_delta(
            _array_label(llama_values, "post_moe_0", "llama.cpp"),
            _array(hip_captures[1], "hidden_in", "hipEngine"),
        ),
    }
    router = [
        _router_compare(
            layer=layer,
            capture=hip_capture_meta[layer],
            hip_values=hip_captures[layer],
            llama_values=llama_values,
        )
        for layer in (0, 1)
    ]
    routing_weights = {
        f"layer{layer}": _numeric_delta(
            _array_label(llama_values, f"ffn_moe_weights_norm_{layer}", "llama.cpp"),
            np.asarray(hip_capture_meta[layer]["moe_routing_weights"], dtype=np.float32),
        )
        for layer in (0, 1)
    }
    shared_gate = {
        f"layer{layer}": _shared_gate(
            hip_gate=hip_capture_meta[layer].get("moe_shared_gate"),
            llama_values=llama_values,
            label=f"shared_expert_gate_{layer}",
        )
        for layer in (0, 1)
    }

    artifact = {
        "schema": 1,
        "kind": "mtp_target_layer0_1_boundary_cross_engine_compare",
        "status": "complete",
        "performance_claim": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "hipengine_raw": str(hipengine_raw_path),
            "llamacpp_jsonl": str(llamacpp_jsonl_path),
            "llamacpp_cycle": int(llamacpp_cycle),
            "row": int(row),
        },
        "hipengine": _hip_metadata(hip_artifact, hip_capture_meta),
        "llamacpp": _llamacpp_metadata(llama_cycle, duplicates),
        "boundary_deltas": boundary_deltas,
        "tensor_deltas": tensor_deltas,
        "router": router,
        "routing_weights": routing_weights,
        "shared_gate": shared_gate,
    }
    artifact["alignment_notes"] = _alignment_notes(artifact)
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
            ),
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


def _optional_array_label(values: dict[str, np.ndarray], label: str) -> np.ndarray | None:
    return values.get(label)


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
        "llamacpp_or_reference_rms": float(np.sqrt(np.mean(reference * reference, dtype=np.float32)))
        if reference.size
        else 0.0,
        "hipengine_or_candidate_rms": float(np.sqrt(np.mean(candidate * candidate, dtype=np.float32)))
        if candidate.size
        else 0.0,
        "reference_sample": [float(x) for x in reference[:8]],
        "candidate_sample": [float(x) for x in candidate[:8]],
        "diff_sample": [float(x) for x in diff[:8]],
    }


def _router_compare(
    *,
    layer: int,
    capture: dict[str, Any],
    hip_values: dict[str, np.ndarray],
    llama_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    hip_router = _array(hip_values, "moe_router_logits", "hipEngine")
    llama_router = _array_label(llama_values, f"ffn_moe_logits_{layer}", "llama.cpp")
    hip_selected = [int(value) for value in capture["moe_selected_experts"]]
    top_k = len(hip_selected)
    hip_from_logits = _topk(hip_router, top_k)
    llama_top = _topk(llama_router, top_k)
    return {
        "layer": int(layer),
        "topk_match": hip_selected == llama_top,
        "hipengine_selected": hip_selected,
        "hipengine_topk_from_logits": hip_from_logits,
        "hipengine_selected_matches_logits": hip_selected == hip_from_logits,
        "llamacpp_topk": llama_top,
        "rank_mismatches": [
            {
                "rank": int(rank),
                "hipengine": int(hip_expert),
                "llamacpp": int(llama_expert),
            }
            for rank, (hip_expert, llama_expert) in enumerate(zip(hip_selected, llama_top))
            if int(hip_expert) != int(llama_expert)
        ],
        "hip_only_experts": [
            int(expert) for expert in hip_selected if int(expert) not in set(llama_top)
        ],
        "llamacpp_only_experts": [
            int(expert) for expert in llama_top if int(expert) not in set(hip_selected)
        ],
        "cutoff_experts": _cutoff_expert_table(
            hip_router=hip_router,
            llama_router=llama_router,
            hip_top=hip_selected,
            llama_top=llama_top,
        ),
        "delta": _numeric_delta(llama_router, hip_router),
    }


def _topk(values: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > values.size:
        raise ValueError("top_k exceeds vector length")
    return [int(index) for index in np.argsort(values)[::-1][:top_k]]


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


def _shared_gate(
    *,
    hip_gate: Any,
    llama_values: dict[str, np.ndarray],
    label: str,
) -> dict[str, Any] | None:
    llama_gate = _optional_array_label(llama_values, label)
    if hip_gate is None or llama_gate is None:
        return None
    hip_value = float(np.asarray(hip_gate, dtype=np.float32).reshape(-1)[0])
    llama_value = float(llama_gate[0])
    return {
        "hipengine_logit": hip_value,
        "llamacpp_logit": llama_value,
        "delta_hip_minus_llama": float(hip_value - llama_value),
    }


def _hip_metadata(
    artifact: dict[str, Any], captures: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "model": artifact.get("model"),
        "source_trace": artifact.get("source_trace"),
        "command": artifact.get("command"),
        "cycle": artifact.get("probe", {}).get("cycle"),
        "sampled_tokens": artifact.get("result", {}).get("sampled_tokens"),
        "accepted_draft_tokens": artifact.get("result", {}).get("accepted_draft_tokens"),
        "captures": {
            str(layer): {
                "layer": capture.get("layer"),
                "row": capture.get("row"),
                "position": capture.get("position"),
                "input_token": capture.get("input_token"),
                "trace_target_token": capture.get("trace_target_token"),
            }
            for layer, capture in captures.items()
        },
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


def _alignment_notes(artifact: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    tensor_deltas = artifact["tensor_deltas"]
    layer0_attn_norm = tensor_deltas["layer0_attn_norm"]
    layer0_residual = tensor_deltas["layer0_attn_residual"]
    if (
        layer0_attn_norm["mean_abs_diff"] > 0.1
        and layer0_residual["mean_abs_diff"] < 0.001
    ):
        notes.append(
            "llama.cpp attn_norm_0 raw values do not align with hipEngine layer-0 "
            "attn_norm for this dirty trace, while downstream layer-0 residual "
            "and post-MoE tensors align closely; do not use attn_norm_0 as the "
            "boundary split without deeper llama.cpp trace validation."
        )
    return notes


def _conclusion(artifact: dict[str, Any]) -> str:
    hip_self = artifact["boundary_deltas"]["hip_layer0_layer_out_vs_layer1_hidden_in"]
    boundary = artifact["boundary_deltas"]["hip_layer1_hidden_in_vs_llamacpp_post_moe_0"]
    layer1_router = artifact["router"][1]
    router_delta = layer1_router["delta"]
    if layer1_router["topk_match"]:
        split = "layer 1 router top-k still matches llama.cpp"
    else:
        split = (
            "layer 1 router top-k diverges "
            f"(rank mismatches={layer1_router['rank_mismatches']})"
        )
    return (
        "hipEngine's local layer0->layer1 handoff is "
        f"{hip_self['mean_abs_diff']:.6g} MAE, so the handoff is internally stable. "
        "The cross-engine layer0 output / layer1 input boundary is already "
        f"{boundary['mean_abs_diff']:.6g} MAE / {boundary['rmse']:.6g} RMSE / "
        f"{boundary['cosine']:.6g} cosine before layer 1 routing; {split}. "
        "Layer 1 router logits differ by "
        f"{router_delta['mean_abs_diff']:.6g} MAE / {router_delta['rmse']:.6g} RMSE / "
        f"{router_delta['cosine']:.6g} cosine."
    )


if __name__ == "__main__":
    main()
