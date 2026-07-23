#!/usr/bin/env python3
"""Profile Laguna LPF-5 prefill at 512, 1K, and 4K context lengths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from hipengine.benchmark.provenance import collect_artifact_provenance
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
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LENGTHS = (512, 1024, 4096)
DEFAULT_CHUNK_SIZE = 128
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf5-long-context-profile.json"
)


def _parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item.strip())
    if not lengths or any(item <= 0 for item in lengths):
        raise argparse.ArgumentTypeError("context lengths must be positive integers")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("context lengths must be distinct")
    return lengths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=max(DEFAULT_LENGTHS))
    parser.add_argument("--lengths", type=_parse_lengths, default=DEFAULT_LENGTHS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-rows", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _timing_order(lengths: Sequence[int], repetition: int) -> tuple[int, ...]:
    ordered = tuple(int(value) for value in lengths)
    return ordered if int(repetition) % 2 == 0 else tuple(reversed(ordered))


def _summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("long-context summary requires at least one sample")
    lengths = {int(sample["length"]) for sample in samples}
    if len(lengths) != 1:
        raise ValueError("long-context summary samples must have one length")
    seconds = [float(sample["prefill_seconds"]) for sample in samples]
    if any(not math.isfinite(value) or value <= 0.0 for value in seconds):
        raise ValueError("long-context timing samples must be finite and positive")
    length = lengths.pop()
    median = statistics.median(seconds)
    return {
        "length": length,
        "samples_seconds": seconds,
        "median_seconds": median,
        "median_tok_s": length / median,
        "next_token_ids": [int(sample["next_token_id"]) for sample in samples],
        "repeat_deterministic": len({int(sample["next_token_id"]) for sample in samples}) == 1,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    lengths = tuple(int(value) for value in args.lengths)
    if lengths != DEFAULT_LENGTHS:
        raise ValueError(f"retained LPF-5 profiling requires exact lengths {DEFAULT_LENGTHS}")
    if args.chunk_size != DEFAULT_CHUNK_SIZE:
        raise ValueError(
            f"retained LPF-5 profiling requires chunk size {DEFAULT_CHUNK_SIZE}"
        )
    if args.context_length < max(lengths):
        raise ValueError("largest LPF-5 length exceeds admitted context")
    if args.repetitions <= 0:
        raise ValueError("LPF-5 repetitions must be positive")
    if args.warmup_rows <= 0 or args.warmup_rows > args.chunk_size:
        raise ValueError("LPF-5 warmup rows must fit one retained chunk")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna LPF-5 profiling requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_lpf5_long_context",
        timing_protocol="prefill_only_512_1024_4096_chunk128",
        warmups=1,
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(lengths))

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    rows: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            progress=_progress,
            repacked_cache=args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=args.chunk_size,
        )
        load_seconds = time.perf_counter() - load_started
        owner.prefill(token_stream[: args.warmup_rows], use_bulk=True)
        runtime.device_synchronize()
        for repetition in range(args.repetitions):
            for length in _timing_order(lengths, repetition):
                owner.reset_state()
                started = time.perf_counter()
                result = owner.prefill(token_stream[:length], use_bulk=True)
                runtime.device_synchronize()
                elapsed = time.perf_counter() - started
                row = {
                    "length": length,
                    "chunks": math.ceil(length / args.chunk_size),
                    "prefill_seconds": elapsed,
                    "prefill_tok_s": length / elapsed,
                    "next_token_id": int(result.next_token_id),
                    "final_position": int(owner.position),
                    "repetition": repetition,
                }
                rows.append(row)
                print(
                    f"rep={repetition} length={length} chunks={row['chunks']} "
                    f"prefill={row['prefill_tok_s']:.3f} tok/s next={result.next_token_id}",
                    file=sys.stderr,
                    flush=True,
                )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna LPF-5 profiling")

    summaries = {
        str(length): _summarize_samples(
            [row for row in rows if int(row["length"]) == length]
        )
        for length in lengths
    }
    positions_exact = all(int(row["final_position"]) == int(row["length"]) - 1 for row in rows)
    deterministic = all(bool(summary["repeat_deterministic"]) for summary in summaries.values())
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    passed = bool(positions_exact and deterministic and recovered)
    prompt_payload = args.prompts.read_bytes()
    manifest_path = args.repacked_cache / "manifest.json"
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_lpf5_long_context_profile",
        "status": "accepted_attribution_baseline" if passed else "rejected",
        "pass": passed,
        "performance_claim": False,
        "scope": "Laguna S 2.1 c=1 prefill-only long-context attribution baseline",
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
            ),
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "lengths": list(lengths),
            "chunk_size": args.chunk_size,
            "chunks_per_length": {
                str(length): math.ceil(length / args.chunk_size) for length in lengths
            },
            "context_length": args.context_length,
            "repetitions": args.repetitions,
            "warmup_rows": args.warmup_rows,
            "timed_order": "ascending then alternating direction by repetition",
            "timing_scope": "reset complete through synchronized first-token projection; load excluded",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
        },
        "load": {"seconds_excluded": load_seconds, "resident_nbytes": resident_nbytes},
        "rows": rows,
        "timings": summaries,
        "correctness": {
            "pass": passed,
            "final_positions_exact": positions_exact,
            "repeat_next_token_deterministic": deterministic,
            "tracked_returned_to_baseline": recovered,
            "boundary_fixture_evidence": [
                "tests/test_laguna_cpu_reference.py::test_laguna_global_and_swa_masks_match_transformers_at_511_512_513",
                "tests/test_laguna_kv_attention.py::test_laguna_global_and_swa_attention_match_cpu_across_wrap_and_eviction",
            ],
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "This is an attribution baseline, not a speedup or long-context support claim.",
            "The deterministic stream repeats the longest canonical prompt without its leading BOS.",
            "Run once under cached-only rocprofv3 and attach a raw-trace summary separately.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
