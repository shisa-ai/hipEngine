#!/usr/bin/env python3
"""Inventory llama.cpp MTP trace artifacts for parity-oracle coverage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_TRACE = Path("benchmarks/results/llamacpp-mtp-greeting-b2-draft-trace.json")
DEFAULT_PLAN = Path("benchmarks/results/llamacpp-mtp-greeting-b2-trace-plan.json")
DEFAULT_CHECKPOINT = Path(
    "benchmarks/results/mtp-gguf-iter281-layer3-actual-checkpoint-summary.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter282-llamacpp-trace-inventory.json")
CHECKPOINT_ARRAY_KEYS = (
    "hidden_in_f32",
    "attn_out_f32",
    "post_norm_f32",
    "residual_f32",
    "ffn_or_moe_down_f32",
    "moe_shared_out_f32",
    "layer_out_f32",
)
TRACE_HINT_KEYS = (
    "checkpoint",
    "hidden",
    "attn",
    "attn_out",
    "layer_out",
    "post_norm",
    "residual",
    "logits",
    "tensor",
    "sha256",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--checkpoint-summary", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=282)
    args = parser.parse_args()

    artifact = build_trace_inventory_artifact(
        trace_path=args.trace,
        plan_path=args.plan,
        checkpoint_summary_path=args.checkpoint_summary,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "prompt_tokens_match": artifact["alignment"]["prompt_tokens_match"],
                "has_numeric_layer_checkpoints": artifact["trace_coverage"][
                    "has_numeric_layer_checkpoints"
                ],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_trace_inventory_artifact(
    *,
    trace_path: Path,
    plan_path: Path | None = None,
    checkpoint_summary_path: Path | None = None,
    iteration: int = 282,
) -> dict[str, Any]:
    trace = _read_json(trace_path)
    plan = _read_json(plan_path) if plan_path and plan_path.exists() else None
    checkpoint = (
        _read_json(checkpoint_summary_path)
        if checkpoint_summary_path and checkpoint_summary_path.exists()
        else None
    )
    trace_summary = summarize_trace(trace)
    coverage = infer_trace_checkpoint_coverage(trace)
    alignment = summarize_alignment(trace_summary, checkpoint)
    status = "inventory_complete"
    return {
        "schema": 1,
        "kind": "llamacpp_mtp_trace_checkpoint_inventory",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "trace_path": str(trace_path),
        "plan_path": None if plan_path is None else str(plan_path),
        "checkpoint_summary_path": (
            None if checkpoint_summary_path is None else str(checkpoint_summary_path)
        ),
        "trace_summary": trace_summary,
        "plan_summary": summarize_plan(plan),
        "checkpoint_summary": summarize_checkpoint(checkpoint),
        "trace_coverage": coverage,
        "alignment": alignment,
        "next_action": next_action(coverage, alignment),
        "conclusion": conclusion(coverage, alignment),
    }


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    calls = trace.get("calls") or []
    if not isinstance(calls, list):
        raise ValueError("trace.calls must be a list when present")
    call_summaries = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        candidates = call.get("candidates") or []
        positions = sorted(
            {int(item["pos"]) for item in candidates if isinstance(item, dict) and "pos" in item}
        )
        top_by_pos: dict[str, dict[str, Any]] = {}
        for item in candidates:
            if not isinstance(item, dict) or item.get("rank") != 0:
                continue
            pos = str(int(item.get("pos", 0)))
            top_by_pos[pos] = {
                "token_id": _maybe_int(item.get("token_id")),
                "piece": item.get("piece"),
                "prob": item.get("prob"),
            }
        call_summaries.append(
            {
                "call_count": call.get("call_count"),
                "generated": call.get("generated"),
                "accepted": call.get("accepted"),
                "accept_generated": call.get("accept_generated"),
                "hist_size": call.get("hist_size"),
                "candidate_count": len(candidates),
                "positions": positions,
                "top_by_pos": top_by_pos,
            }
        )
    return {
        "schema": trace.get("schema"),
        "kind": trace.get("kind"),
        "prompt_tokens": trace.get("prompt_tokens"),
        "source_log": trace.get("source_log"),
        "summary": trace.get("summary") or {},
        "llamacpp_timing_summary": trace.get("llamacpp_timing_summary") or {},
        "request": (trace.get("metadata") or {}).get("request") or {},
        "call_count": len(call_summaries),
        "calls": call_summaries,
    }


def summarize_plan(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    command = plan.get("server_command") or []
    payload = plan.get("request_payload") or {}
    return {
        "kind": plan.get("kind"),
        "server_binary": command[0] if command else None,
        "server_log": plan.get("server_log"),
        "trace_json": plan.get("trace_json"),
        "request_endpoint": plan.get("request_endpoint"),
        "request_payload": payload,
        "metadata": plan.get("metadata") or {},
    }


def summarize_checkpoint(checkpoint: dict[str, Any] | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    capture = checkpoint.get("capture") or {}
    arrays = checkpoint.get("arrays") or {}
    return {
        "kind": checkpoint.get("kind"),
        "source_capture": checkpoint.get("source_capture"),
        "capture": capture,
        "array_keys": sorted(arrays),
        "array_hashes": {
            key: value.get("sha256")
            for key, value in arrays.items()
            if isinstance(value, dict)
        },
    }


def infer_trace_checkpoint_coverage(trace: dict[str, Any]) -> dict[str, Any]:
    known_key_paths = []
    hint_key_paths = []
    numeric_array_paths = []
    for path, key, value in _walk_json(trace):
        lower_key = str(key).lower()
        if key in CHECKPOINT_ARRAY_KEYS:
            known_key_paths.append(path)
        if any(hint in lower_key for hint in TRACE_HINT_KEYS):
            hint_key_paths.append(path)
        if _is_numeric_array(value):
            numeric_array_paths.append(
                {
                    "path": path,
                    "count": len(value),
                    "sample": [float(x) for x in value[:8]],
                }
            )
    has_numeric_layer_checkpoints = bool(known_key_paths)
    return {
        "has_numeric_layer_checkpoints": has_numeric_layer_checkpoints,
        "known_checkpoint_array_paths": known_key_paths,
        "hint_key_paths": hint_key_paths,
        "numeric_array_paths_len_ge_16": numeric_array_paths,
        "supports_layer3_checkpoint_alignment": bool(has_numeric_layer_checkpoints),
        "supports_token_draft_alignment": bool((trace.get("calls") or [])),
    }


def summarize_alignment(
    trace_summary: dict[str, Any], checkpoint: dict[str, Any] | None
) -> dict[str, Any]:
    if checkpoint is None:
        return {
            "checkpoint_available": False,
            "prompt_tokens_match": None,
            "layer_checkpoint_target": None,
        }
    capture = checkpoint.get("capture") or {}
    position = capture.get("position")
    expected_prompt_tokens = None if position is None else int(position) + 1
    prompt_tokens = trace_summary.get("prompt_tokens")
    return {
        "checkpoint_available": True,
        "trace_prompt_tokens": prompt_tokens,
        "checkpoint_position": position,
        "expected_prompt_tokens_from_checkpoint": expected_prompt_tokens,
        "prompt_tokens_match": bool(prompt_tokens == expected_prompt_tokens),
        "layer_checkpoint_target": {
            "layer_id": capture.get("layer_id"),
            "layer_type": capture.get("layer_type"),
            "run_preceding_layers": capture.get("run_preceding_layers"),
            "preceding_layer_count": capture.get("preceding_layer_count"),
        },
    }


def next_action(coverage: dict[str, Any], alignment: dict[str, Any]) -> str:
    if not alignment.get("prompt_tokens_match"):
        return "fix_prompt_token_alignment_before_numeric_trace"
    if not coverage["has_numeric_layer_checkpoints"]:
        return "capture_llamacpp_numeric_layer_checkpoint_or_add_llama_trace_tap"
    return "compare_llamacpp_numeric_checkpoint_to_hipengine_summary"


def conclusion(coverage: dict[str, Any], alignment: dict[str, Any]) -> str:
    if not alignment.get("prompt_tokens_match"):
        return (
            "Existing llama.cpp trace prompt length does not match the hipEngine checkpoint "
            "position+1; align tokenization/request setup before numeric comparison."
        )
    if not coverage["has_numeric_layer_checkpoints"]:
        return (
            "Existing llama.cpp greeting trace aligns at prompt-token count but only carries "
            "draft token/probability events; it lacks hidden/attn/layer checkpoint arrays or "
            "hashes required to compare the hipEngine actual layer-3 checkpoint."
        )
    return (
        "Existing llama.cpp trace appears to contain numeric checkpoint arrays; "
        "compare them next."
    )


def _walk_json(value: Any, path: str = "$", key: str = "$") -> Iterable[tuple[str, str, Any]]:
    yield path, key, value
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _walk_json(child_value, f"{path}.{child_key}", str(child_key))
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            yield from _walk_json(child_value, f"{path}[{index}]", str(index))


def _is_numeric_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 16
        and all(isinstance(item, int | float) for item in value)
    )


def _maybe_int(value: Any) -> int | Any:
    return int(value) if isinstance(value, int) else value


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("path must not be None")
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
