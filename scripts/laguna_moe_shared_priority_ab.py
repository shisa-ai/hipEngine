#!/usr/bin/env python3
"""Compare Laguna decode with low- and normal-priority shared MoE streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
    _sha256_json,
)

_MODES = ("low_priority", "decode_normal_priority")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _child_session(
    owner: LagunaGGUFResidentSession,
    *,
    compiler_version_file: Path | None,
    require_cached_build: bool,
) -> LagunaGGUFResidentSession:
    assert owner.weights is not None
    return LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=4096,
        backend="hip_gfx1151",
        runtime=owner.runtime,
        compiler_version=_compiler_version(compiler_version_file),
        require_cached_build=require_cached_build,
        prefill_chunk_size=2048,
        prefill_global_attention_chunk_size=128,
        moe_branch_concurrency=True,
        moe_decode_branch_concurrency=True,
        moe_shared_after_router=True,
        moe_shared_low_priority=True,
        moe_decode_shared_normal_priority=True,
    )


def _run_arm(
    session: LagunaGGUFResidentSession,
    tokens: list[int],
    *,
    decode_normal_priority: bool,
) -> dict[str, object]:
    session.set_moe_decode_shared_normal_priority(decode_normal_priority)
    session.reset_state()
    prefill_started = time.perf_counter()
    result = session.prefill(tokens, use_bulk=True)
    session.runtime.device_synchronize()
    prefill_seconds = time.perf_counter() - prefill_started
    generated = [int(result.next_token_id)]
    decode_started = time.perf_counter()
    while len(generated) < 128:
        result = session.forward_token(result.next_token_id)
        generated.append(int(result.next_token_id))
    session.runtime.device_synchronize()
    decode_seconds = time.perf_counter() - decode_started
    return {
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": 512.0 / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tok_s": 127.0 / decode_seconds,
        "next_token_id": generated[0],
        "final_token_id": generated[-1],
        "final_position": int(session.position),
        "generated_ids_sha256": _sha256_json(generated),
    }


def main() -> int:
    args = _parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    reader = GGUFReader(DEFAULT_MODEL)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(DEFAULT_PROMPTS, tokenizer)
    token_stream, _ = _profile_token_stream(prompts, 512)
    tokens = list(token_stream)
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    owner = LagunaGGUFResidentSession(
        DEFAULT_MODEL,
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        progress=_progress,
        repacked_cache=DEFAULT_CACHE,
        model_sha256=DEFAULT_MODEL_SHA256,
        prefill_chunk_size=128,
        prefill_global_attention_chunk_size=128,
        moe_branch_concurrency=False,
        moe_decode_branch_concurrency=False,
    )
    session: LagunaGGUFResidentSession | None = None
    rows: dict[str, list[dict[str, object]]] = {
        mode: [] for mode in _MODES
    }
    try:
        session = _child_session(
            owner,
            compiler_version_file=args.compiler_version_file,
            require_cached_build=args.require_cached_build,
        )
        for mode in _MODES:
            session.set_moe_decode_shared_normal_priority(
                mode == "decode_normal_priority"
            )
            session.prefill(tokens[:128], use_bulk=True)
            runtime.device_synchronize()
        for repetition in range(args.repetitions):
            order = (
                _MODES
                if repetition % 2 == 0
                else tuple(reversed(_MODES))
            )
            for mode in order:
                row = _run_arm(
                    session,
                    tokens,
                    decode_normal_priority=(
                        mode == "decode_normal_priority"
                    ),
                )
                row["repetition"] = repetition
                rows[mode].append(row)
                print(
                    f"{mode} rep={repetition} "
                    f"prefill={row['prefill_tok_s']:.6f} tok/s "
                    f"decode={row['decode_tok_s']:.6f} tok/s "
                    f"tokens={row['next_token_id']}->{row['final_token_id']}",
                    flush=True,
                )
        result: dict[str, object] = {
            "protocol": {
                "shape": "p512/d128 eager c=1 matrix2048 attention128",
                "repetitions": args.repetitions,
                "timed_order": "alternating low/normal priority",
                "decode_forward_calls_per_arm": 127,
                "gpu_max_hw_queues_required": 2,
            },
            "modes": {},
        }
        for mode in _MODES:
            mode_rows = rows[mode]
            decode_seconds = [
                float(row["decode_seconds"]) for row in mode_rows
            ]
            result["modes"][mode] = {
                "decode_shared_stream_low_priority": mode == "low_priority",
                "prefill_shared_stream_low_priority": True,
                "shared_stream_priority_range": session.moe_shared_priority_range,
                "decode_samples_seconds": decode_seconds,
                "decode_median_seconds": statistics.median(decode_seconds),
                "decode_median_tok_s": (
                    127.0 / statistics.median(decode_seconds)
                ),
                "prefill_samples_seconds": [
                    float(row["prefill_seconds"]) for row in mode_rows
                ],
                "next_token_ids": [
                    int(row["next_token_id"]) for row in mode_rows
                ],
                "final_token_ids": [
                    int(row["final_token_id"]) for row in mode_rows
                ],
                "final_positions": [
                    int(row["final_position"]) for row in mode_rows
                ],
                "generated_ids_sha256": [
                    str(row["generated_ids_sha256"]) for row in mode_rows
                ],
            }
    finally:
        if session is not None:
            session.close()
        owner.close()
    runtime.device_synchronize()
    result["memory"] = {
        "tracked_before": tracked_before,
        "tracked_after": memory_stats(),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
