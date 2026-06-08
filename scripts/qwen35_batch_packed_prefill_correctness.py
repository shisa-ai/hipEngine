#!/usr/bin/env python3
"""Correctness smoke for native compact c>N prefill plus serial decode.

Compares a shared resident session using ``prefill_native_packed`` for prompt
slabs against independent c=1 resident sessions using ``prefill_native``. Decode
still uses ``step_batch_serial``; this is a correctness gate, not a throughput
claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.generation import ResidentBatchScheduler
from hipengine.kvcache import ResolvedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_artifact_schema import _load_payload
from scripts.qwen35_kv_policy_args import add_kv_policy_args, append_kv_policy_flags, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"


def _payload_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _load_prompt_slices(path: Path, *, prompt_length: int, batch_size: int) -> list[list[int]]:
    fixture = _load_payload(path)
    tokens = [int(token) for token in fixture["prompt_ids"]]
    needed = prompt_length * batch_size
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if len(tokens) < needed:
        raise ValueError(f"fixture contains {len(tokens)} tokens, need at least {needed}")
    return [tokens[row * prompt_length : (row + 1) * prompt_length] for row in range(batch_size)]


def _compiler_version(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text()


def _format_pair(seed, decode) -> dict[str, Any]:
    return {
        "seed": seed.token_id,
        "decode": decode.token_id,
        "seed_logit": seed.logit,
        "decode_logit": decode.logit,
    }


def _all_logits_finite(rows: list[dict[str, Any]]) -> bool:
    return all(math.isfinite(float(row["seed_logit"])) and math.isfinite(float(row["decode_logit"])) for row in rows)


def _run_c1(
    runner: Qwen35ParoNextTokenRunner,
    prompt: list[int],
    *,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    kv_policy: ResolvedKVPolicy,
) -> dict[str, Any]:
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=len(prompt) + 4,
        max_layers=max_layers,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        seed = session.prefill_native(prompt, sample=True)
        if seed is None:
            raise RuntimeError("native c=1 prefill did not produce a seed token")
        decode = session.step(seed.token_id, position=len(prompt), sample=True)
        if decode is None:
            raise RuntimeError("native c=1 decode did not produce a token")
    return _format_pair(seed, decode)


def _run_packed(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    kv_policy: ResolvedKVPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prompt_lengths = {len(prompt) for prompt in prompts}
    if len(prompt_lengths) != 1:
        raise ValueError("current smoke expects equal prompt lengths")
    prompt_length = prompt_lengths.pop()
    scheduler = ResidentBatchScheduler(capacity=len(prompts))
    request_ids = [scheduler.submit(prompt, max_new_tokens=1) for prompt in prompts]
    admitted = scheduler.admit_pending()
    if admitted != tuple(request_ids):
        raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
    metadata: dict[str, Any] = {
        "request_ids": list(request_ids),
        "admitted": list(admitted),
        "prefill_slabs": [],
    }
    seed_by_request: dict[int, Any] = {}
    decode_by_request: dict[int, Any] = {}
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=prompt_length + 4,
        max_layers=max_layers,
        max_batch_size=len(prompts),
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        batch_execution = session.batch_execution_metadata(scheduler_owned=True).to_json_dict()
        slabs = scheduler.next_compact_prefill_slabs(chunk_size=prompt_length, block_size=session.block_size)
        for slab in slabs:
            metadata["prefill_slabs"].append(
                {
                    "request_ids": list(slab.request_ids),
                    "slot_ids": list(slab.physical_slot_ids),
                    "rows": slab.rows,
                    "block_count": slab.block_count,
                    "cu_seqlens_q": list(slab.cu_seqlens_q),
                    "cu_seqlens_k": list(slab.cu_seqlens_k),
                }
            )
            results = session.prefill_native_packed(slab, sample=True)
            if len(results) != slab.request_count:
                raise RuntimeError("packed prefill result count mismatch")
            for request_id, result in zip(slab.request_ids, results, strict=True):
                if result is None:
                    raise RuntimeError("packed prefill did not produce seed token")
                seed_by_request[request_id] = result
        metadata["last_prefill_execution"] = session.last_prefill_execution
        decode_work = scheduler.next_decode_work()
        if decode_work is None:
            raise RuntimeError("scheduler did not emit decode work")
        decode_results = session.step_batch_serial(
            [seed_by_request[request_id].token_id for request_id in decode_work.request_ids],
            positions=[scheduler.active_batch.requests[request_id].context_len for request_id in decode_work.request_ids],
            slots=[scheduler.active_batch.slot_for(request_id) for request_id in decode_work.request_ids],
            sample=True,
        )
        for request_id, result in zip(decode_work.request_ids, decode_results, strict=True):
            if result is None:
                raise RuntimeError("decode did not produce a token")
            decode_by_request[request_id] = result
    actual = [_format_pair(seed_by_request[request_id], decode_by_request[request_id]) for request_id in request_ids]
    return actual, metadata, batch_execution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--compiler-version-file")
    parser.add_argument("--require-cached", action="store_true")
    add_kv_policy_args(parser, help_prefix="Resident KV storage for packed-prefill correctness")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size)
    compiler_version = _compiler_version(args.compiler_version_file)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    expected = [
        _run_c1(
            runner,
            prompt,
            max_layers=args.max_layers,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached,
            kv_policy=kv_policy,
        )
        for prompt in prompts
    ]
    actual, scheduler_metadata, batch_execution = _run_packed(
        runner,
        prompts,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached,
        kv_policy=kv_policy,
    )
    command = (
        "python3 scripts/qwen35_batch_packed_prefill_correctness.py "
        f"--model {args.model} --fixture {args.fixture} --prompt-length {args.prompt_length} "
        f"--max-layers {args.max_layers} --batch-size {args.batch_size}"
    )
    if args.compiler_version_file is not None:
        command += f" --compiler-version-file {args.compiler_version_file}"
    if args.require_cached:
        command += " --require-cached"
    command = append_kv_policy_flags(command, args)
    if args.json is not None:
        command += f" --json {args.json}"
    generated_match = actual == expected
    finite_logits = _all_logits_finite(expected) and _all_logits_finite(actual)
    payload = {
        "schema": 1,
        "status": "accepted_correctness_smoke" if generated_match and finite_logits else "failed_correctness_smoke",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": f"resident_c{args.batch_size}_native_compact_prefill_serial_decode_correctness",
        "command": command,
        "batch_execution": batch_execution,
        "benchmark_eligible": False,
        "kv_storage_dtype": kv_policy.storage_dtype.value,
        "kv_policy": kv_policy_json(kv_policy),
        "batch_size": args.batch_size,
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "max_layers": args.max_layers,
        "expected_c1_native": expected,
        "packed_prefill_serial_decode": actual,
        "scheduler_metadata": scheduler_metadata,
        "generated_match": generated_match,
        "finite_logits": finite_logits,
        "passed": generated_match and finite_logits,
        "notes": [
            "Native compact c>N prefill with packed prompt slabs; decode remains step_batch_serial.",
            "Correctness gate only; not a throughput claim until c-aware decode graph replay lands.",
        ],
    }
    text = _payload_json(payload)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
