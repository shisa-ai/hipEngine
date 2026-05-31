#!/usr/bin/env python3
"""Correctness smoke for the Qwen3.5/PARO serial c>N resident slot bridge.

This is not a benchmark.  It compares a shared resident session using
``step_batch_serial`` against independent c=1 resident sessions for
deterministic prompt slices.
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
        seed = None
        for pos, token in enumerate(prompt):
            seed = session.step(token, position=pos, sample=(pos == len(prompt) - 1))
        if seed is None:
            raise RuntimeError("prefill did not produce a seed token")
        decode = session.step(seed.token_id, position=len(prompt), sample=True)
        if decode is None:
            raise RuntimeError("decode did not produce a token")
    return {
        "seed": seed.token_id,
        "decode": decode.token_id,
        "seed_logit": seed.logit,
        "decode_logit": decode.logit,
    }


def _format_pair(seed, decode) -> dict[str, Any]:
    return {
        "seed": seed.token_id,
        "decode": decode.token_id,
        "seed_logit": seed.logit,
        "decode_logit": decode.logit,
    }


def _all_logits_finite(rows: list[dict[str, Any]]) -> bool:
    return all(math.isfinite(float(row["seed_logit"])) and math.isfinite(float(row["decode_logit"])) for row in rows)


def _shape_key_payload(key) -> dict[str, Any]:
    return {
        "mode": key.mode.value,
        "active_c": key.active_c,
        "context_bucket": key.context_bucket,
        "active_mask": list(key.active_mask),
        "kv_storage_dtype": key.kv_storage_dtype,
        "layer_plan": key.layer_plan,
        "top_k": key.top_k,
        "experts_per_token": key.experts_per_token,
        "replay_steps": key.replay_steps,
        "draft_depth": key.draft_depth,
        "tree_shape": list(key.tree_shape),
    }


def _run_batch_serial(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    kv_policy: ResolvedKVPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_lengths = {len(prompt) for prompt in prompts}
    if len(prompt_lengths) != 1:
        raise ValueError("current smoke expects equal prompt lengths")
    prompt_length = prompt_lengths.pop()
    slots = list(range(len(prompts)))
    batch_execution: dict[str, Any] = {}
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
        batch_execution = session.batch_execution_metadata(scheduler_owned=False).to_json_dict()
        seed_results = None
        for pos in range(prompt_length):
            seed_results = session.step_batch_serial(
                [prompt[pos] for prompt in prompts],
                positions=[pos] * len(prompts),
                slots=slots,
                sample=(pos == prompt_length - 1),
            )
        if seed_results is None or any(result is None for result in seed_results):
            raise RuntimeError("batch prefill did not produce seed tokens")
        decode_results = session.step_batch_serial(
            [result.token_id for result in seed_results if result is not None],
            positions=[prompt_length] * len(prompts),
            slots=slots,
            sample=True,
        )
        if any(result is None for result in decode_results):
            raise RuntimeError("batch decode did not produce tokens")
    actual = [_format_pair(seed, decode) for seed, decode in zip(seed_results, decode_results, strict=True) if seed is not None and decode is not None]
    return actual, batch_execution


def _run_batch_serial_scheduler(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    kv_policy: ResolvedKVPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run the serial slot bridge through ResidentBatchScheduler ownership."""

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
        "slot_to_request_after_admit": list(scheduler.active_batch.slot_to_request),
        "active_count_after_admit": scheduler.active_count,
        "prefill_work_items": 0,
        "prefill_request_order": [],
    }
    seed_by_request: dict[int, Any] = {}
    decode_by_request: dict[int, Any] = {}
    batch_execution: dict[str, Any] = {}
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
        while True:
            work = scheduler.next_prefill_work(chunk_size=1)
            if work is None:
                break
            metadata["prefill_work_items"] += 1
            request_id = work.request_ids[0]
            metadata["prefill_request_order"].append(request_id)
            token = work.token_rows[0][0]
            request = scheduler.active_batch.requests[request_id]
            position = request.next_prompt_index - 1
            slot = scheduler.active_batch.slot_for(request_id)
            sample = request.remaining_prefill == 0
            result = session.step_batch_serial([token], positions=[position], slots=[slot], sample=sample)[0]
            if sample:
                if result is None:
                    raise RuntimeError("prefill did not produce a seed token")
                seed_by_request[request_id] = result
        decode_work = scheduler.next_decode_work()
        if decode_work is None:
            raise RuntimeError("scheduler did not emit decode work")
        shape_key = scheduler.shape_key(
            mode="decode",
            top_k=8,
            experts_per_token=8,
            replay_steps=1,
            kv_storage_dtype=kv_policy.storage_dtype.value,
            layer_plan=f"max_layers={int(max_layers)}",
        )
        scheduler.graph_buckets.get_or_create(shape_key, _shape_key_payload)
        scheduler.graph_buckets.get(shape_key)
        stats = scheduler.graph_buckets.stats
        metadata["decode_shape_key"] = _shape_key_payload(shape_key)
        metadata["graph_bucket_stats"] = stats.to_json_dict()
        metadata["slot_to_request_at_decode"] = list(scheduler.active_batch.slot_to_request)
        metadata["active_count_at_decode"] = scheduler.active_count
        decode_request_ids = decode_work.request_ids
        metadata["decode_request_ids"] = list(decode_request_ids)
        decode_results = session.step_batch_serial(
            [seed_by_request[request_id].token_id for request_id in decode_request_ids],
            positions=[scheduler.active_batch.requests[request_id].context_len for request_id in decode_request_ids],
            slots=[scheduler.active_batch.slot_for(request_id) for request_id in decode_request_ids],
            sample=True,
        )
        completed = scheduler.record_generated(
            (request_id, result.token_id) for request_id, result in zip(decode_request_ids, decode_results, strict=True) if result is not None
        )
        for request_id, result in zip(decode_request_ids, decode_results, strict=True):
            if result is None:
                raise RuntimeError("decode did not produce a token")
            decode_by_request[request_id] = result
    actual = [_format_pair(seed_by_request[request_id], decode_by_request[request_id]) for request_id in request_ids]
    completed_payload = [
        {
            "request_id": done.request_id,
            "generated_tokens": list(done.generated_tokens),
            "finished": done.finished,
        }
        for done in completed
    ]
    metadata["active_count_after_completion"] = scheduler.active_count
    metadata["slot_to_request_after_completion"] = list(scheduler.active_batch.slot_to_request)
    return actual, completed_payload, metadata, batch_execution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--compiler-version-file")
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--scheduler", action="store_true", help="Drive step_batch_serial from ResidentBatchScheduler work items")
    add_kv_policy_args(parser, help_prefix="Resident KV storage for serial c>N correctness")
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
    scheduler_completed: list[dict[str, Any]] = []
    scheduler_metadata: dict[str, Any] = {}
    if args.scheduler:
        actual, scheduler_completed, scheduler_metadata, batch_execution = _run_batch_serial_scheduler(
            runner,
            prompts,
            max_layers=args.max_layers,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached,
            kv_policy=kv_policy,
        )
    else:
        actual, batch_execution = _run_batch_serial(
            runner,
            prompts,
            max_layers=args.max_layers,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached,
            kv_policy=kv_policy,
        )
    command = (
        "python3 scripts/qwen35_batch_serial_correctness.py "
        f"--prompt-length {args.prompt_length} --max-layers {args.max_layers} --batch-size {args.batch_size}"
    )
    if args.scheduler:
        command += " --scheduler"
    command = append_kv_policy_flags(command, args)
    if args.json is not None:
        command += f" --json {args.json}"
    generated_match = actual == expected
    finite_logits = _all_logits_finite(expected) and _all_logits_finite(actual)
    payload = {
        "schema": 1,
        "status": "accepted_correctness_smoke",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": f"resident_c{args.batch_size}_scheduler_serial_slot_runner_correctness" if args.scheduler else f"resident_c{args.batch_size}_serial_slot_runner_correctness",
        "command": command,
        "scheduler_completed": scheduler_completed,
        "scheduler_metadata": scheduler_metadata,
        "batch_execution": batch_execution,
        "benchmark_eligible": bool(batch_execution.get("throughput_claim_eligible")),
        "kv_storage_dtype": kv_policy.storage_dtype.value,
        "kv_policy": kv_policy_json(kv_policy),
        "batch_size": args.batch_size,
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "max_layers": args.max_layers,
        "expected_c1": expected,
        "batch_serial": actual,
        "generated_match": generated_match,
        "finite_logits": finite_logits,
        "passed": generated_match and finite_logits,
        "notes": [
            "Correctness-first serial c>N bridge over batch-shaped resident slot buffers; not a throughput claim.",
            "batch_execution.throughput_claim_eligible=false until native compact/c-aware kernels replace step_batch_serial.",
            "Compares a shared resident session against independent c=1 resident sessions.",
            "Scheduler mode consumes ResidentBatchScheduler prefill/decode work items and physical slots."
            if args.scheduler
            else "Direct mode drives physical slots without scheduler ownership.",
        ],
    }
    text = _payload_json(payload)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
