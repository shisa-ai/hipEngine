#!/usr/bin/env python3
"""Report Qwen3.5/PARO native-prefill layer coverage.

This is a correctness/blocker planning helper, not a benchmark. It records the
retained single-request native prefill coverage and the first unsupported layer
type, if any, for the selected prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading import load_weight_index, qwen35_paro_config_from_hf
from hipengine.runtime.qwen35_paro_runner import qwen35_paro_native_prefill_plan

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_ACCEPTED_LINEAR_PREFIX_ARTIFACT = (
    "benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-scratch-restore-sweep.json"
)


def _component_blockers(first_unsupported_type: str | None) -> list[dict[str, Any]]:
    if first_unsupported_type != "full_attention":
        return []
    return [
        {
            "component": "full_attention_prefill_orchestrator",
            "path": "hipengine/runtime/qwen35_paro.py",
            "symbol": "Qwen35ParoDecodeState.run_full_attention_moe_c1_layer_fp16",
            "current_guard": "raises ValueError when tokens != 1",
            "required_for_native_prefill": "tokens>1 full-attention prefill orchestration or an explicitly labelled serial fallback",
        },
        {
            "component": "full_attention_qkv_projection_layout",
            "path": "hipengine/runtime/qwen35_paro.py",
            "symbol": "Qwen35ParoDecodeState.project_full_attention_qkv_fp16",
            "current_guard": "raises ValueError when tokens != 1",
            "required_for_native_prefill": "batched q/k/v projections with contiguous q, key, value, and gate views for downstream RoPE/KV writes",
        },
        {
            "component": "full_attention_rope_prepare_positions",
            "path": "hipengine/runtime/qwen35_paro.py",
            "symbol": "Qwen35ParoDecodeState.prepare_full_attention_qkv_fp16",
            "current_guard": "raises ValueError when tokens != 1 and consumes one device position scalar",
            "required_for_native_prefill": "per-token prompt positions for Q/K RMSNorm plus partial rotary instead of a single decode position",
        },
        {
            "component": "full_attention_prefill_kv_append",
            "path": "hipengine/kernels/hip_gfx1100/attention/paged_kv_write.py",
            "symbol": "qwen35_write_paged_kv_mixed_value_fp16_batch_spans",
            "current_status": "batch KV writer wrapper exists, but resident native prefill does not wire token row positions/live counts for prompt rows",
            "required_for_native_prefill": "KVLiveSpans with row/token positions that append all prompt K/V rows at their correct cache positions",
        },
        {
            "component": "full_attention_causal_prefill_attention",
            "path": "hipengine/runtime/qwen35_paro.py",
            "symbol": "Qwen35ParoDecodeState.decode_full_attention_context_gate_fp16",
            "current_status": "decode context attention reads one query against existing cache; no multi-query causal prefill attention path is wired",
            "required_for_native_prefill": "causal full-attention prefill over the prompt, or a labelled serial c=1 fallback until the native kernel exists",
        },
    ]


def _boundary_payload(
    layer_types: Sequence[str],
    *,
    model: str,
    max_layers: int,
    accepted_linear_prefix_artifact: str = DEFAULT_ACCEPTED_LINEAR_PREFIX_ARTIFACT,
    command: str,
) -> dict[str, Any]:
    layer_limit = len(layer_types) if int(max_layers) == 0 else int(max_layers)
    plan = qwen35_paro_native_prefill_plan(tuple(layer_types), layer_limit=layer_limit)
    first_unsupported_layer = plan.first_unsupported_layer
    first_unsupported_type = plan.first_unsupported_type
    blocked = not plan.full_layer_limit_native
    type_counts = Counter(layer_types[:layer_limit])
    payload = {
        "schema": 1,
        "status": "blocked" if blocked else "accepted",
        "blocked_reason": (
            None
            if not blocked
            else "native prefill encountered an unsupported layer type"
        ),
        "model": str(Path(model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_native_prefill_boundary_plan",
        "command": command,
        "performance_claim": False,
        "layer_limit": layer_limit,
        "layer_type_counts": dict(type_counts),
        "native_prefill_plan": plan.to_json_dict(),
        "accepted_linear_prefix_artifact": accepted_linear_prefix_artifact,
        "accepted_linear_prefix_layers": plan.linear_prefix_layers,
        "first_unsupported_layer": first_unsupported_layer,
        "first_unsupported_type": first_unsupported_type,
        "component_blockers": _component_blockers(first_unsupported_type),
        "next_actions": [],
        "notes": [
            "Correctness/blocker planning only; no timings are collected and no throughput claim is made.",
            "Single-request native prefill covers linear_attention and full_attention layers; this artifact narrows any remaining blocker to an unsupported layer type.",
        ],
    }
    if blocked and first_unsupported_type == "full_attention":
        payload["next_actions"] = [
            "Add a layer-boundary diagnostic comparing serial c=1 layer-3 full-attention prefill rows against any new prefill implementation.",
            "Wire batched full-attention q/k/v projection and per-token RoPE preparation for prompt rows.",
            "Wire batched KV append with KVLiveSpans row/token positions, then add causal prefill attention or an explicitly-labelled serial c=1 fallback.",
        ]
    return payload


def _command(args: argparse.Namespace) -> str:
    command = f"python3 scripts/qwen35_native_prefill_boundary.py --model {args.model}"
    if args.max_layers:
        command += f" --max-layers {args.max_layers}"
    if args.accepted_linear_prefix_artifact != DEFAULT_ACCEPTED_LINEAR_PREFIX_ARTIFACT:
        command += f" --accepted-linear-prefix-artifact {args.accepted_linear_prefix_artifact}"
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-layers", type=int, default=0, help="Layer prefix to inspect; 0 means all configured layers")
    parser.add_argument(
        "--accepted-linear-prefix-artifact",
        default=DEFAULT_ACCEPTED_LINEAR_PREFIX_ARTIFACT,
        help="Artifact proving the predecessor linear prefix is accepted.",
    )
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    index = load_weight_index(Path(args.model))
    config = qwen35_paro_config_from_hf(index.config)
    payload = _boundary_payload(
        config.layer_types,
        model=args.model,
        max_layers=args.max_layers,
        accepted_linear_prefix_artifact=str(args.accepted_linear_prefix_artifact),
        command=_command(args),
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if payload["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
