#!/usr/bin/env python3
"""Qwen3.5/PARO sparse-slot native decode correctness smoke.

The diagnostic pre-fills three requests, cancels the middle request, then runs
native c>N decode over the remaining sparse physical slots (0 and 2).  It
compares generated-token IDs with independent c=1 resident sessions and emits a
correctness-only JSON artifact.
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
from scripts.qwen35_batch_retained_bench import DEFAULT_FIXTURE, DEFAULT_MODEL, _compiler_version, _load_prompt_slices


def _payload_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _command(argv: Sequence[str] | None) -> str:
    parts = ["python3", "scripts/qwen35_batch_sparse_slot_correctness.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _run_c1_reference(
    runner: Qwen35ParoNextTokenRunner,
    prompts: Sequence[Sequence[int]],
    *,
    decode_tokens: int,
    max_layers: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> list[list[int]]:
    rows: list[list[int]] = []
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        for prompt in prompts:
            seed = session.prefill_native(prompt, sample=True)
            if seed is None:
                raise RuntimeError("c=1 prefill did not produce a seed token")
            next_token = int(seed.token_id)
            seq = [next_token]
            for step in range(decode_tokens):
                result = session.step(next_token, position=len(prompt) + step, sample=True)
                if result is None:
                    raise RuntimeError("c=1 decode did not produce a token")
                next_token = int(result.token_id)
                seq.append(next_token)
            rows.append(seq)
            session.reset()
    return rows


def run(args: argparse.Namespace, argv: Sequence[str] | None = None) -> dict[str, Any]:
    if args.prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if args.decode_tokens <= 0:
        raise ValueError("decode_tokens must be positive")
    if args.max_layers <= 0:
        raise ValueError("max_layers must be positive")
    if args.max_sequence_length < args.prompt_length + args.decode_tokens + 1:
        raise ValueError("max_sequence_length must cover prompt_length + decode_tokens + 1")
    os.environ.setdefault("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    compiler_version = _compiler_version(args.compiler_version_file)
    prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=3)
    runner = Qwen35ParoNextTokenRunner(args.model)

    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=args.max_sequence_length,
        max_layers=args.max_layers,
        max_batch_size=3,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        scheduler = ResidentBatchScheduler(capacity=3)
        request_ids = [scheduler.submit(prompt, max_new_tokens=args.decode_tokens) for prompt in prompts]
        admitted = scheduler.admit_pending()
        if tuple(request_ids) != tuple(admitted):
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
        slabs = scheduler.next_compact_prefill_slabs(chunk_size=args.prompt_length, block_size=session.block_size)
        if len(slabs) != 1:
            raise RuntimeError(f"expected one prefill slab, got {len(slabs)}")
        seed_results = session.prefill_native_packed(slabs[0], sample=True)
        seed_by_request: dict[int, int] = {}
        for request_id, result in zip(request_ids, seed_results, strict=True):
            if result is None:
                raise RuntimeError("native packed prefill did not produce a seed token")
            seed_by_request[request_id] = int(result.token_id)

        cancelled = scheduler.cancel(request_ids[1])
        if cancelled is None:
            raise RuntimeError("middle request was not cancelled")
        active_request_ids = tuple(request_id for request_id in request_ids if request_id != request_ids[1])
        next_token_by_request = {request_id: seed_by_request[request_id] for request_id in active_request_ids}
        generated_by_request = {request_id: [] for request_id in active_request_ids}
        slot_history: list[list[int]] = []
        for _ in range(args.decode_tokens):
            work = scheduler.next_decode_work()
            if work is None:
                raise RuntimeError("scheduler did not emit sparse-slot decode work")
            active_ids = tuple(request_id for request_id in work.request_ids if request_id in next_token_by_request)
            slots = [scheduler.active_batch.slot_for(request_id) for request_id in active_ids]
            positions = [scheduler.active_batch.requests[request_id].context_len for request_id in active_ids]
            slot_history.append(slots)
            results = session.step_batch_native(
                [next_token_by_request[request_id] for request_id in active_ids],
                positions=positions,
                slots=slots,
                sample=True,
            )
            generated_tokens: list[GeneratedToken] = []
            for request_id, result in zip(active_ids, results, strict=True):
                if result is None:
                    raise RuntimeError("native sparse-slot decode did not produce a token")
                token_id = int(result.token_id)
                generated_by_request[request_id].append(token_id)
                next_token_by_request[request_id] = token_id
                generated_tokens.append(GeneratedToken(request_id, token_id))
            scheduler.record_generated(generated_tokens)

    batch_sequences = [[seed_by_request[request_id], *generated_by_request[request_id]] for request_id in active_request_ids]
    c1_sequences = _run_c1_reference(
        runner,
        [prompts[0], prompts[2]],
        decode_tokens=args.decode_tokens,
        max_layers=args.max_layers,
        max_sequence_length=args.max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    passed = batch_sequences == c1_sequences
    payload: dict[str, Any] = {
        "schema": 1,
        "status": "eq_ok" if passed else "mismatch_found",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "qwen35_paro_sparse_slot_native_decode_smoke",
        "command": _command(argv),
        "performance_claim": False,
        "workload": {
            "model": str(args.model),
            "fixture": str(args.fixture),
            "prompt_length": int(args.prompt_length),
            "decode_tokens": int(args.decode_tokens),
            "max_layers": int(args.max_layers),
            "max_sequence_length": int(args.max_sequence_length),
            "initial_slots": [0, 1, 2],
            "active_slots_history": slot_history,
            "cancelled_slot": 1,
            "native_compact_prefill": True,
            "native_caware_decode": True,
        },
        "correctness": {
            "oracle": "generated-token equality vs independent c=1",
            "passed": passed,
            "batch_sequences": batch_sequences,
            "c1_sequences": c1_sequences,
        },
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_payload_json(payload) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=64)
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
