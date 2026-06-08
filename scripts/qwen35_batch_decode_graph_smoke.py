#!/usr/bin/env python3
"""C3.0b piece D smoke: capture + replay a device-resident c>1 decode graph.

Validates ``Qwen35ParoResidentSession.capture_batch_decode_graph`` /
``Qwen35ParoBatchDecodeGraph`` end to end on the real PARO model:

1. Prefill ``rows`` compact rows from the fixture.
2. Run ``warmup`` eager device-resident decode steps (allocates every scratch /
   cache so graph capture is allocation-free).
3. Reference: run ``decode`` more eager device-resident steps, recording the
   per-step row tokens.
4. Capture a single decode step at the post-warmup position, rewind the device
   counters, re-seed the warmup's last tokens, and replay ``decode`` steps,
   collecting each step's tokens.
5. Assert the replayed token stream is byte-identical to the eager reference.

Because replay regenerates identical tokens it rewrites identical KV, so running
the eager reference first (which advances KV) does not perturb the replay gate:
any divergence is detected from the first mismatching token onward.

Exit code 0 + ``"passed": true`` only when capture succeeds, replay launches the
graph, and every replayed token matches the eager reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow ``python3 scripts/qwen35_batch_decode_graph_smoke.py`` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipengine.core.dtype import DType
from hipengine.generation import ResidentBatchScheduler
from hipengine.kvcache import ResolvedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_retained_bench import _load_prompt_slices

DEFAULT_MODEL = "/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16"
DEFAULT_FIXTURE = "/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json"


def _prefill_and_warmup(session, scheduler, *, rows, prompt_length, warmup, slot_list):
    """Prefill compact rows + eager device-resident warmup; return (seed_tokens, position)."""

    slabs = scheduler.next_compact_prefill_slabs(chunk_size=prompt_length, block_size=session.block_size)
    seed_by_request: dict[int, object] = {}
    for slab in slabs:
        results = session.prefill_native_packed(slab, sample=True)
        for request_id, result in zip(slab.request_ids, results, strict=True):
            if result is None:
                raise RuntimeError("prefill did not produce a seed token")
            seed_by_request[request_id] = result
    slot_to_request = list(scheduler.active_batch.slot_to_request)
    cur_tokens = [int(seed_by_request[slot_to_request[s]].token_id) for s in range(rows)]
    pos = prompt_length
    for _ in range(warmup):
        results = session.step_batch_native(
            cur_tokens, positions=[pos] * rows, slots=slot_list, sample=True, device_resident=True
        )
        cur_tokens = [int(r.token_id) for r in results]
        pos += 1
    return list(cur_tokens), pos


def _new_session_scheduler(runner, *, rows, prompts, warmup, decode, prompt_length, compiler_version, require_cached_build, kv_policy):
    scheduler = ResidentBatchScheduler(capacity=rows)
    request_ids = [scheduler.submit(prompt, max_new_tokens=warmup + decode) for prompt in prompts]
    admitted = scheduler.admit_pending()
    if admitted != tuple(request_ids):
        raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
    session = Qwen35ParoResidentSession(
        runner,
        max_sequence_length=prompt_length + warmup + decode + 1,
        max_layers=40,
        max_batch_size=rows,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=kv_policy.create_policy(),
    )
    return session, scheduler


def run(
    *,
    model: Path,
    fixture: Path,
    prompt_length: int,
    rows: int,
    warmup: int,
    decode: int,
    compiler_version_file: Path | None,
    require_cached_build: bool,
) -> dict:
    os.environ["HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"] = "1"
    prompts = _load_prompt_slices(fixture, prompt_length=prompt_length, batch_size=rows)
    runner = Qwen35ParoNextTokenRunner(model)
    compiler_version = compiler_version_file.read_text() if compiler_version_file else None
    kv_policy = ResolvedKVPolicy(requested_storage="bf16", storage_dtype=DType.BF16, block_size=256)
    slot_list = list(range(rows))

    # PARO linear-attention layers carry recurrent state that advances every
    # decode step, so the eager reference and the capture/replay must each run
    # from their own fresh session (prefill+warmup is deterministic, so both
    # reach the identical post-warmup state).

    # ---- Pass A: eager device-resident reference ----
    session_a, scheduler_a = _new_session_scheduler(
        runner, rows=rows, prompts=prompts, warmup=warmup, decode=decode, prompt_length=prompt_length,
        compiler_version=compiler_version, require_cached_build=require_cached_build, kv_policy=kv_policy,
    )
    with session_a:
        seed_after_warmup, start_position = _prefill_and_warmup(
            session_a, scheduler_a, rows=rows, prompt_length=prompt_length, warmup=warmup, slot_list=slot_list
        )
        eager_tokens: list[list[int]] = []
        ref_tokens = list(seed_after_warmup)
        ref_pos = start_position
        eager_start = time.perf_counter()
        for _ in range(decode):
            results = session_a.step_batch_native(
                ref_tokens, positions=[ref_pos] * rows, slots=slot_list, sample=True, device_resident=True
            )
            ref_tokens = [int(r.token_id) for r in results]
            eager_tokens.append(list(ref_tokens))
            ref_pos += 1
        eager_seconds = time.perf_counter() - eager_start

    # ---- Pass B: capture + replay from an identical fresh session ----
    session_b, scheduler_b = _new_session_scheduler(
        runner, rows=rows, prompts=prompts, warmup=warmup, decode=decode, prompt_length=prompt_length,
        compiler_version=compiler_version, require_cached_build=require_cached_build, kv_policy=kv_policy,
    )
    with session_b:
        seed_b, start_b = _prefill_and_warmup(
            session_b, scheduler_b, rows=rows, prompt_length=prompt_length, warmup=warmup, slot_list=slot_list
        )
        if seed_b != seed_after_warmup or start_b != start_position:
            raise RuntimeError("warmup diverged across sessions; prefill/warmup is not deterministic")
        graph = session_b.capture_batch_decode_graph(
            rows=rows,
            positions=[start_position] * rows,
            slots=slot_list,
            max_replay_steps=decode,
        )
        try:
            graph.reset_positions()
            graph.seed_tokens(seed_after_warmup)
            replay_start = time.perf_counter()
            replay_tokens = graph.replay_collect(decode)
            replay_seconds = time.perf_counter() - replay_start
            collect_launches = int(graph.launches)
            # Throughput: back-to-back replay (single sync) from the same start,
            # the mode a generator would use; correctness already established.
            graph.reset_positions()
            graph.seed_tokens(seed_after_warmup)
            bb_start = time.perf_counter()
            graph.replay(decode)
            replay_bb_seconds = time.perf_counter() - bb_start
            total_launches = int(graph.launches)
        finally:
            graph.close()

    passed = bool(
        replay_tokens == eager_tokens
        and collect_launches == decode
        and total_launches == 2 * decode
    )
    first_mismatch = None
    for step_index, (eager_row, replay_row) in enumerate(zip(eager_tokens, replay_tokens)):
        if eager_row != replay_row:
            first_mismatch = {"step": step_index, "eager": eager_row, "replay": replay_row}
            break

    return {
        "passed": passed,
        "comparison": "batch_decode_graph_replay_vs_eager_device_resident",
        "rows": rows,
        "prompt_length": prompt_length,
        "warmup_decode_tokens": warmup,
        "decode_tokens": decode,
        "start_position": start_position,
        "replay_collect_launches": collect_launches,
        "replay_total_launches": total_launches,
        "eager_decode_seconds": eager_seconds,
        "replay_decode_seconds": replay_seconds,
        "eager_tok_s": (rows * decode) / eager_seconds if eager_seconds > 0 else None,
        "replay_collect_tok_s": (rows * decode) / replay_seconds if replay_seconds > 0 else None,
        "replay_backtoback_seconds": replay_bb_seconds,
        "replay_backtoback_tok_s": (rows * decode) / replay_bb_seconds if replay_bb_seconds > 0 else None,
        "replay_speedup_vs_eager": (eager_seconds / replay_bb_seconds) if replay_bb_seconds > 0 else None,
        "first_mismatch": first_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--warmup-decode-tokens", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result = run(
        model=Path(args.model),
        fixture=Path(args.fixture),
        prompt_length=args.prompt_length,
        rows=args.rows,
        warmup=args.warmup_decode_tokens,
        decode=args.decode_tokens,
        compiler_version_file=args.compiler_version_file,
        require_cached_build=args.require_cached_build,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.write_text(payload + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
