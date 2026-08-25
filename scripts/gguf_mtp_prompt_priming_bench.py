#!/usr/bin/env python3
"""Measure exact GGUF target prefill plus shifted NextN prompt priming."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Sequence

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNDraftProvider,
    borrow_qwen35_gguf_nextn_fallback_weights,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _parse_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item) for item in str(value).split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prompt lengths must be comma-separated integers") from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("prompt lengths must be positive")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("prompt lengths must be unique")
    return lengths


def _prompt(length: int, seed: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(seed[index % len(seed)]) for index in range(int(length)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lengths", type=_parse_lengths, default=(512,))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--allow-build", action="store_true")
    args = parser.parse_args()
    lengths = tuple(int(item) for item in args.lengths)
    max_positions = max(lengths) + 8
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    seed = (7734, 264, 12654, 709, 421, 4523, 279, 307, 7324, 76938, 1324, 1608, 20781, 1954, 13)
    rows = []

    with Qwen35GGUFResidentSession(
        args.model,
        max_sequence_length=max_positions,
        compiler_version=compiler_version,
        require_cached_build=not args.allow_build,
    ) as target:
        target.select_prefill_quant("gguf_q4_k_m")
        borrowed = borrow_qwen35_gguf_nextn_fallback_weights(target)
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            args.model,
            max_positions=max_positions,
            max_requests=1,
            runtime=target.runtime,
            compiler_version=compiler_version,
            require_cached_build=not args.allow_build,
            borrowed_fallback_weights=borrowed,
        )
        try:
            decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
            decoder.target = target
            decoder.draft_provider = provider
            # Warm all one-row draft kernels and target bulk setup without scoring.
            target.reset()
            provider.reset_request(0)
            decoder._prefill_target_and_draft(
                tuple(seed[:4]), request_id=0, use_bulk=True
            )
            for length in lengths:
                prompt = _prompt(length, seed)
                target.reset()
                pure_started = time.perf_counter()
                pure_result = target.prefill(prompt, use_bulk=True, return_logits=False)
                target_prefill_seconds = time.perf_counter() - pure_started
                target.reset()
                provider.reset_request(0)
                reset_memory_stats()
                before = memory_stats()
                free_before, total = target.runtime.mem_get_info()
                minimum_free = [int(free_before)]
                stop = threading.Event()

                def monitor() -> None:
                    while not stop.wait(0.01):
                        free_now, _ = target.runtime.mem_get_info()
                        minimum_free[0] = min(minimum_free[0], int(free_now))

                watcher = threading.Thread(target=monitor, daemon=True)
                watcher.start()
                started = time.perf_counter()
                try:
                    result = decoder._prefill_target_and_draft(
                        prompt, request_id=0, use_bulk=True
                    )
                finally:
                    elapsed = time.perf_counter() - started
                    stop.set()
                    watcher.join()
                after = memory_stats()
                free_after, _ = target.runtime.mem_get_info()
                slot = provider.executor._request_slots[0]
                draft_scratch = provider.executor.scratch.for_slot(slot, span_role="decode")
                row = {
                    "label": args.label,
                    "prompt_length": length,
                    "token_id": int(result.token_id),
                    "pure_target_token_id": int(pure_result.token_id),
                    "target_prefill_seconds": target_prefill_seconds,
                    "target_prefill_tok_s": length / target_prefill_seconds,
                    "prefill_seconds": elapsed,
                    "target_position": int(target.position),
                    "draft_position": int(draft_scratch.position_host[0]),
                    "draft_context": int(draft_scratch.context_host[0]),
                    "tracked_current_before": int(before["current_allocated_bytes"]),
                    "tracked_peak_after": int(after["peak_allocated_bytes"]),
                    "tracked_transient_peak_delta": int(after["peak_allocated_bytes"] - before["current_allocated_bytes"]),
                    "tracked_total_allocated": int(after["total_allocated_bytes"]),
                    "hip_free_before": int(free_before),
                    "hip_minimum_free": int(minimum_free[0]),
                    "hip_free_after": int(free_after),
                    "hip_transient_peak_delta": int(free_before - minimum_free[0]),
                    "hip_total": int(total),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        finally:
            provider.close()
    payload = {"label": args.label, "model": str(args.model), "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
