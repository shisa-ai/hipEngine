#!/usr/bin/env python3
"""Run a partial StepFun GGUF resident layer-prefix prompt smoke.

This is a correctness bring-up diagnostic, not a full generation path or a
throughput benchmark: it executes a host-composed resident layer prefix and
explicitly reports which decoder layers are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading.gguf import scan_gguf_splits
from hipengine.loading.stepfun_gguf import build_stepfun_gguf_tensor_map
from hipengine.runtime.stepfun_gguf_runner import (
    StepFunResidentSession,
    stepfun_layer_prefix_slot_paths,
    stepfun_slot_tensor,
)

DEFAULT_GGUF_DIR = Path("/data/models/gguf")
DEFAULT_PATTERN = "Step-3.7-flash-Q3_K_L-*.gguf"
FORBIDDEN_TEXT_ONLY_FRAGMENTS = ("vision", "projector", "mmproj", "mtp", "nextn")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_GGUF_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--layer-count", type=int, default=4)
    parser.add_argument("--message", default="hello")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument(
        "--sample-token-id",
        action="append",
        type=int,
        default=None,
        help="Token id to include in sampled final logits; may be repeated.",
    )
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
        help="Scan metadata and print the layer-prefix slot/resource plan without HIP allocation.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def _scope(layer_count: int, block_count: int) -> str:
    if layer_count >= block_count:
        return f"layers_0_{block_count - 1}_prefix_no_skipped_layers"
    return f"layers_0_{layer_count - 1}_prefix_only_layers_{layer_count}_{block_count - 1}_skipped"


def _skipped_layers(layer_count: int, block_count: int) -> list[int]:
    if layer_count >= block_count:
        return []
    return [layer_count, block_count - 1]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = tuple(sorted(args.model_dir.glob(args.pattern)))
    if not paths:
        raise FileNotFoundError(f"no GGUF shards matched {args.model_dir / args.pattern}")

    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    selected_slots = stepfun_layer_prefix_slot_paths(model_map, args.layer_count)
    no_modal_slots = not any(
        fragment in slot for slot in selected_slots for fragment in FORBIDDEN_TEXT_ONLY_FRAGMENTS
    )
    if not no_modal_slots:
        raise RuntimeError("layer-prefix text smoke selected a vision/projector/MTP slot")
    resident_weight_nbytes = sum(stepfun_slot_tensor(model_map, slot).nbytes for slot in selected_slots)
    if args.dry_run_plan:
        result = {
            "status": "planned",
            "scope": _scope(args.layer_count, model_map.config.block_count),
            "model": args.pattern.removesuffix("-*.gguf"),
            "backend": "hip_gfx1151",
            "command": "python3 scripts/stepfun_layer_prefix_smoke.py "
            f"--dry-run-plan --layer-count {args.layer_count} --message {json.dumps(args.message)} --pretty",
            "paths": [str(path) for path in paths],
            "split_count": info.split_count,
            "tensor_count": info.tensor_count,
            "layer_count": args.layer_count,
            "skipped_layers": _skipped_layers(args.layer_count, model_map.config.block_count),
            "selected_slot_count": len(selected_slots),
            "selected_slots": list(selected_slots),
            "no_vision_projector_mtp_slots": True,
            "resident_weight_nbytes": int(resident_weight_nbytes),
            "resident_weight_gib": resident_weight_nbytes / 2**30,
            "note": (
                "Metadata-only layer-prefix prompt plan: no HIP runtime was initialized and no weights, "
                "KV buffers, prompt embeddings, or logits were allocated/computed."
            ),
        }
        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
        return 0

    runtime = get_hip_runtime()
    reset_memory_stats()
    free_before, total = runtime.mem_get_info()
    started = time.perf_counter()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=selected_slots,
        runtime=runtime,
    )
    free_after_load, _ = runtime.mem_get_info()
    result: dict[str, object] = {}
    try:
        probe = session.layer_prefix_prompt_logits_probe_bf16(
            [{"role": "user", "content": args.message}],
            layer_count=args.layer_count,
            reasoning_effort=args.reasoning_effort,
            runtime=runtime,
        )
        sample_ids = args.sample_token_id or [0, 1, 128007, model_map.config.vocab_size - 1]
        sample_ids = [token_id if token_id >= 0 else model_map.config.vocab_size + token_id for token_id in sample_ids]
        if any(token_id < 0 or token_id >= model_map.config.vocab_size for token_id in sample_ids):
            raise ValueError(
                f"sample token ids out of range for vocab_size={model_map.config.vocab_size}: {sample_ids}"
            )
        result = {
            "status": "partial_prompt_smoke",
            "scope": _scope(args.layer_count, model_map.config.block_count),
            "model": args.pattern.removesuffix("-*.gguf"),
            "backend": "hip_gfx1151",
            "command": "python3 scripts/stepfun_layer_prefix_smoke.py "
            f"--layer-count {args.layer_count} --message {json.dumps(args.message)} --pretty",
            "paths": [str(path) for path in paths],
            "split_count": info.split_count,
            "tensor_count": info.tensor_count,
            "layer_count": args.layer_count,
            "skipped_layers": _skipped_layers(args.layer_count, model_map.config.block_count),
            "selected_slot_count": len(selected_slots),
            "selected_slots": list(selected_slots),
            "no_vision_projector_mtp_slots": True,
            "resident_weight_nbytes": session.weights.allocated_nbytes,
            "prompt": probe.prompt.rendered_prompt,
            "input_ids": [int(token_id) for token_id in probe.prompt.input_ids],
            "prompt_length": probe.prompt.prompt_length,
            "layer_hidden_shape": list(probe.layer_hidden.shape),
            "logits_shape": list(probe.logits.shape),
            "sampled_logits": {str(token_id): float(probe.logits[0, token_id]) for token_id in sample_ids},
            "next_token_id": probe.next_token_id,
            "next_token_logit": probe.next_token_logit,
            "hip_total_gib": total / 2**30,
            "hip_free_before_gib": free_before / 2**30,
            "hip_free_after_load_gib": free_after_load / 2**30,
            "elapsed_s": time.perf_counter() - started,
            "memory_stats_before_free": memory_stats(),
            "hip_mem_get_info_note": (
                "hipMemGetInfo total/free are known to be contradictory under the configured Strix Halo GTT "
                "setup; use hipEngine allocation stats and relative before/load/free readings as diagnostic only."
            ),
            "note": (
                "Partial correctness smoke only: host-composed layer-prefix bridge uses resident root/layer "
                "weights and final sampled logits; skipped layers and KV-backed decode remain open, so this "
                "is not full next-token parity or throughput evidence."
            ),
        }
    finally:
        session.free(runtime=runtime)

    free_after_free, _ = runtime.mem_get_info()
    result["hip_free_after_free_gib"] = free_after_free / 2**30
    result["memory_stats_after_free"] = memory_stats()
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
