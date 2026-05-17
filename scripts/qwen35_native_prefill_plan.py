#!/usr/bin/env python3
"""Emit the current Qwen3.5/PARO native-prefill coverage plan.

This is a planning/blocker artifact helper, not a benchmark.  It reads the HF
config only, computes the resident native-prefill prefix that hipEngine can run
without falling back to ``step_batch_serial``, and records the first layer that
requires the native compact/full-attention prefill port.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading import load_weight_index, qwen35_paro_config_from_hf
from hipengine.runtime.qwen35_paro_runner import qwen35_paro_native_prefill_plan

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)


def _command(args: argparse.Namespace) -> str:
    command = f"python3 scripts/qwen35_native_prefill_plan.py --model {args.model}"
    if args.layer_limit is not None:
        command += f" --layer-limit {args.layer_limit}"
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def _prefix_preview(layer_types: tuple[str, ...], *, limit: int, width: int = 8) -> list[dict[str, Any]]:
    return [
        {"layer": layer_id, "type": layer_types[layer_id]}
        for layer_id in range(min(limit, len(layer_types), width))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen3.5/PARO model directory")
    parser.add_argument("--layer-limit", type=int, help="Optional layer limit to plan instead of the full model")
    parser.add_argument("--json", type=Path, help="Optional path to write the compact JSON payload")
    args = parser.parse_args(argv)

    index = load_weight_index(Path(args.model))
    config = qwen35_paro_config_from_hf(index.config)
    layer_limit = config.num_hidden_layers if args.layer_limit is None else int(args.layer_limit)
    plan = qwen35_paro_native_prefill_plan(config.layer_types, layer_limit=layer_limit)
    counts = Counter(config.layer_types[:layer_limit])
    payload = {
        "schema": 1,
        "status": "blocked",
        "blocked_reason": "native compact/full-attention prefill is not implemented past the linear-attention prefix",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_native_prefill_plan",
        "command": _command(args),
        "performance_claim": False,
        "config": {
            "architecture": config.architecture,
            "num_hidden_layers": config.num_hidden_layers,
            "layer_limit": layer_limit,
            "layer_type_counts_in_limit": dict(sorted(counts.items())),
            "layer_prefix_preview": _prefix_preview(config.layer_types, limit=layer_limit),
        },
        "native_prefill_plan": plan.to_json_dict(),
        "next_required_port_scope": [
            "grouped/compact MoE prefill at the first unsupported layer",
            "full-attention prefill/KV append for compact token batches",
            "c-aware decode graph replay after native compact prefill is available",
        ],
        "notes": [
            "This helper reads HF config metadata only; it does not materialize weights or run kernels.",
            "The payload is blocker evidence for Task #15 and is not a throughput artifact.",
        ],
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
