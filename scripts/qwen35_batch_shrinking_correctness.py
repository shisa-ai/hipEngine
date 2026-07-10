#!/usr/bin/env python3
"""Qwen3.5/PARO c=N to c=1 shrinking-batch correctness gate.

The gate starts with a packed prompt batch, decodes a fixed number of tokens at
each live width, and cancels rows in a hole-producing order. Generated token IDs
for every row are compared with independent single-request prefill/decode runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType
from hipengine.generation import GeneratedToken, ResidentBatchScheduler
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_retained_bench import (
    DEFAULT_FIXTURE,
    DEFAULT_MODEL,
    _compiler_version,
    _hardware_context,
    _isolated_c1_route_env,
    _load_prompt_slices,
    _software_context,
)


def _payload_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _command(argv: Sequence[str] | None) -> str:
    parts = ["python3", "scripts/qwen35_batch_shrinking_correctness.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _cancellation_order(batch_size: int) -> tuple[int, ...]:
    """Leave slot zero alive while creating holes early in the transition."""

    pending = list(range(1, int(batch_size)))
    order: list[int] = []
    while pending:
        index = len(pending) // 2
        order.append(pending.pop(index))
    return tuple(order)


def _run_c1_reference(
    runner: Qwen35ParoNextTokenRunner,
    prompts: Sequence[Sequence[int]],
    decode_counts: Sequence[int],
    *,
    max_layers: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> list[list[int]]:
    rows: list[list[int]] = []
    with (
        _isolated_c1_route_env(),
        Qwen35ParoResidentSession(
            runner,
            max_sequence_length=max_sequence_length,
            max_layers=max_layers,
            max_batch_size=1,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
        ) as session,
    ):
        for prompt, decode_tokens in zip(prompts, decode_counts, strict=True):
            seed = session.prefill_native(prompt, sample=True)
            if seed is None:
                raise RuntimeError("c1 prefill did not produce a seed token")
            next_token = int(seed.token_id)
            sequence = [next_token]
            for step in range(int(decode_tokens)):
                result = session.step(next_token, position=len(prompt) + step, sample=True)
                if result is None:
                    raise RuntimeError("c1 decode did not produce a token")
                next_token = int(result.token_id)
                sequence.append(next_token)
            rows.append(sequence)
            session.reset()
    return rows


def run(args: argparse.Namespace, argv: Sequence[str] | None = None) -> dict[str, Any]:
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if args.prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if args.steps_per_width <= 0:
        raise ValueError("steps_per_width must be positive")
    if args.max_layers <= 0:
        raise ValueError("max_layers must be positive")
    total_decode_steps = int(args.batch_size) * int(args.steps_per_width)
    if args.max_sequence_length < args.prompt_length + total_decode_steps + 1:
        raise ValueError("max_sequence_length must cover every shrinking decode step")

    os.environ.setdefault("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    compiler_version = _compiler_version(args.compiler_version_file)
    prompts = _load_prompt_slices(
        Path(args.fixture),
        prompt_length=args.prompt_length,
        batch_size=args.batch_size,
    )
    runner = Qwen35ParoNextTokenRunner(args.model)
    cancellation_order = _cancellation_order(args.batch_size)
    generated_by_request: dict[int, list[int]] = {}
    decoded_count_by_request: dict[int, int] = {}
    transition_trace: list[dict[str, Any]] = []

    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=args.max_sequence_length,
        max_layers=args.max_layers,
        max_batch_size=args.batch_size,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        scheduler = ResidentBatchScheduler(capacity=args.batch_size)
        request_ids = [
            scheduler.submit(prompt, max_new_tokens=total_decode_steps)
            for prompt in prompts
        ]
        admitted = scheduler.admit_pending()
        if tuple(request_ids) != tuple(admitted):
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
        slabs = scheduler.next_compact_prefill_slabs(
            chunk_size=args.prompt_length,
            block_size=session.block_size,
        )
        if len(slabs) != 1:
            raise RuntimeError(f"expected one packed prefill slab, got {len(slabs)}")
        seed_results = session.prefill_native_packed(slabs[0], sample=True)
        next_token_by_request: dict[int, int] = {}
        for request_id, result in zip(request_ids, seed_results, strict=True):
            if result is None:
                raise RuntimeError("packed prefill did not produce a seed token")
            token_id = int(result.token_id)
            generated_by_request[request_id] = [token_id]
            decoded_count_by_request[request_id] = 0
            next_token_by_request[request_id] = token_id

        for stage_index, live_width in enumerate(range(args.batch_size, 0, -1)):
            stage_slots: list[list[int]] = []
            stage_request_ids: list[list[int]] = []
            for _step in range(args.steps_per_width):
                work = scheduler.next_decode_work()
                if work is None:
                    raise RuntimeError("scheduler did not emit shrinking decode work")
                active_ids = tuple(
                    request_id
                    for request_id in work.request_ids
                    if request_id in next_token_by_request
                )
                if len(active_ids) != live_width:
                    raise RuntimeError(
                        f"expected live width {live_width}, scheduler emitted {len(active_ids)}"
                    )
                slots = [scheduler.active_batch.slot_for(request_id) for request_id in active_ids]
                positions = [
                    scheduler.active_batch.requests[request_id].context_len
                    for request_id in active_ids
                ]
                stage_slots.append(slots)
                stage_request_ids.append(list(active_ids))
                step = (
                    session.step_batch_native
                    if args.decode_execution == "native"
                    else session.step_batch_serial
                )
                results = step(
                    [next_token_by_request[request_id] for request_id in active_ids],
                    positions=positions,
                    slots=slots,
                    sample=True,
                )
                generated: list[GeneratedToken] = []
                for request_id, result in zip(active_ids, results, strict=True):
                    if result is None:
                        raise RuntimeError("shrinking decode did not produce a token")
                    token_id = int(result.token_id)
                    generated_by_request[request_id].append(token_id)
                    decoded_count_by_request[request_id] += 1
                    next_token_by_request[request_id] = token_id
                    generated.append(GeneratedToken(request_id, token_id))
                scheduler.record_generated(generated)

            cancelled_request: int | None = None
            cancelled_slot: int | None = None
            if live_width > 1:
                cancelled_slot = cancellation_order[stage_index]
                cancelled_request = request_ids[cancelled_slot]
                if scheduler.cancel(cancelled_request) is None:
                    raise RuntimeError(f"failed to cancel request {cancelled_request}")
                next_token_by_request.pop(cancelled_request, None)
            transition_trace.append(
                {
                    "live_width": live_width,
                    "request_ids": stage_request_ids,
                    "physical_slots": stage_slots,
                    "cancelled_request_id": cancelled_request,
                    "cancelled_slot": cancelled_slot,
                }
            )

    decode_counts = [decoded_count_by_request[request_id] for request_id in request_ids]
    c1_sequences = _run_c1_reference(
        runner,
        prompts,
        decode_counts,
        max_layers=args.max_layers,
        max_sequence_length=args.max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    batch_sequences = [generated_by_request[request_id] for request_id in request_ids]
    row_equality = [batch == c1 for batch, c1 in zip(batch_sequences, c1_sequences, strict=True)]
    passed = all(row_equality)
    payload: dict[str, Any] = {
        "schema": 1,
        "status": "eq_ok" if passed else "mismatch_found",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": f"qwen35_paro_shrinking_{args.decode_execution}_decode_gate",
        "command": _command(argv),
        "performance_claim": False,
        "hardware": _hardware_context(),
        "software": _software_context(),
        "workload": {
            "model": str(args.model),
            "fixture": str(args.fixture),
            "prompt_length": int(args.prompt_length),
            "initial_batch_size": int(args.batch_size),
            "steps_per_width": int(args.steps_per_width),
            "decode_tokens_by_request": decode_counts,
            "max_layers": int(args.max_layers),
            "max_sequence_length": int(args.max_sequence_length),
            "decode_execution": str(args.decode_execution),
            "native_compact_prefill": True,
            "cancellation_order": list(cancellation_order),
            "transitions": transition_trace,
        },
        "correctness": {
            "oracle": "independent_single_request_prefill_decode",
            "passed": passed,
            "row_equality": row_equality,
            "batch_sequences": batch_sequences,
            "c1_sequences": c1_sequences,
        },
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_payload_json(payload) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-width", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument(
        "--decode-execution",
        choices=("serial", "native"),
        default="serial",
        help="Use the exact serial bridge or the explicit native-width diagnostic.",
    )
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args, argv)
    print(_payload_json(payload))
    return 0 if payload["status"] == "eq_ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
