#!/usr/bin/env python3
"""Plan compact c>N native prompt slabs and emit current blocker artifact.

This is a metadata/correctness-planning helper, not a throughput benchmark. It
validates that the scheduler can form compact prompt slabs bucketed by uniform
block-table length, then records the remaining native-kernel blockers that keep
`prefill_native_packed(...)` from replacing the serial scheduler bridge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.generation import ResidentBatchScheduler

DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"
DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)

BLOCKERS = (
    "c-aware decode graph replay is not wired, so compact c>N throughput is not claim-eligible",
)


def _load_prompt_slices(path: Path, *, prompt_length: int, batch_size: int) -> list[list[int]]:
    fixture = json.loads(path.read_text())
    prompt_ids = [int(item) for item in fixture["prompt_ids"]]
    need = int(prompt_length) * int(batch_size)
    if len(prompt_ids) < need:
        raise ValueError(f"fixture has {len(prompt_ids)} prompt ids; need at least {need}")
    return [prompt_ids[i * prompt_length : (i + 1) * prompt_length] for i in range(batch_size)]


def _slab_payload(slab) -> dict[str, Any]:
    return {
        "request_ids": list(slab.request_ids),
        "slot_ids": list(slab.physical_slot_ids),
        "rows": slab.rows,
        "request_count": slab.request_count,
        "block_count": slab.block_count,
        "block_size": slab.block_size,
        "cu_seqlens_q": list(slab.cu_seqlens_q),
        "cu_seqlens_k": list(slab.cu_seqlens_k),
        "row_to_request": list(slab.row_to_request[:32]),
        "row_to_request_truncated": len(slab.row_to_request) > 32,
        "positions_head": list(slab.positions[:16]),
        "positions_tail": list(slab.positions[-16:]),
        "context_counts_head": list(slab.context_counts[:16]),
        "context_counts_tail": list(slab.context_counts[-16:]),
        "token_rows_lengths": [len(row) for row in slab.token_rows],
    }


def _command(args: argparse.Namespace) -> str:
    command = (
        "python3 scripts/qwen35_native_compact_prefill_plan.py "
        f"--model {args.model} --fixture {args.fixture} --batch-size {args.batch_size} "
        f"--prompt-length {args.prompt_length} --chunk-size {args.chunk_size} --block-size {args.block_size}"
    )
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _load_prompt_slices(args.fixture, prompt_length=args.prompt_length, batch_size=args.batch_size)
    scheduler = ResidentBatchScheduler(capacity=args.batch_size)
    request_ids = tuple(scheduler.submit(prompt, max_new_tokens=1) for prompt in prompts)
    admitted = scheduler.admit_pending()
    buckets = scheduler.bucketize_by_block_count(chunk_size=args.chunk_size, block_size=args.block_size)
    slabs = scheduler.next_compact_prefill_slabs(chunk_size=args.chunk_size, block_size=args.block_size)
    rows = sum(slab.rows for slab in slabs)
    generated_equal_gate_ready = rows == min(args.prompt_length, args.chunk_size) * args.batch_size
    return {
        "schema": 1,
        "status": "planned",
        "blocked_reason": "metadata-only planning artifact; use qwen35_batch_packed_prefill_correctness.py for native packed execution",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "qwen35_paro_native_prefill_compact_cN_plan",
        "command": _command(args),
        "performance_claim": False,
        "fixture": args.fixture.as_posix(),
        "batch_size": int(args.batch_size),
        "prompt_length": int(args.prompt_length),
        "chunk_size": int(args.chunk_size),
        "block_size": int(args.block_size),
        "request_ids": list(request_ids),
        "admitted_request_ids": list(admitted),
        "bucketize_by_block_count": [
            {"block_count": bucket.block_count, "request_ids": list(bucket.request_ids)} for bucket in buckets
        ],
        "compact_slabs": [_slab_payload(slab) for slab in slabs],
        "total_rows_in_first_wave": rows,
        "scheduler_cursors_after_first_wave": {
            str(request_id): scheduler.active_batch.requests[request_id].next_prompt_index for request_id in request_ids
        },
        "generated_equality_gate_ready": generated_equal_gate_ready,
        "native_prefill_packed_ready": True,
        "blockers": list(BLOCKERS),
        "next_actions": [
            "Use qwen35_batch_packed_prefill_correctness.py for c=2/4/8 native compact prefill equality gates.",
            "Add c-aware decode graph replay before claiming end-to-end c>N throughput.",
            "Run a retained c=8/T=512 compact-prefill benchmark only after correctness and decode-path labeling are clean.",
        ],
        "notes": [
            "Metadata-only planning artifact; no native packed kernels launch and no throughput is claimed.",
            "All requests in each slab share block_count so the current row-shaped block-table writer ABI can be used once packed kernels exist.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", type=Path, default=Path(DEFAULT_FIXTURE))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    payload = run(args)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
