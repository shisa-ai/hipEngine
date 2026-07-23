#!/usr/bin/env python3
"""Benchmark Laguna LPF-4 chunk-128 prefill against the retained chunk-64 policy."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import copy
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
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_ORACLE,
    DEFAULT_ORACLE_LOGPROBS,
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    RETAINED_HORIZONS,
    _compiler_version,
    _load_prompts,
    _oracle_gate,
    _progress,
    _repo_state,
    _run_target,
    _sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZES = (64, 128)
MODES = tuple(f"chunk_{value}" for value in CHUNK_SIZES)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf4-chunk128.json"
)


def _parse_chunk_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item.strip())
    if not sizes or any(item <= 0 for item in sizes):
        raise argparse.ArgumentTypeError("chunk sizes must be positive integers")
    return sizes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--oracle-logprobs", type=Path, default=DEFAULT_ORACLE_LOGPROBS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--chunk-sizes", type=_parse_chunk_sizes, default=CHUNK_SIZES)
    parser.add_argument(
        "--output-horizons",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=RETAINED_HORIZONS,
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup-output-tokens", type=int, default=2)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(prompt_index: int, repetition: int) -> tuple[str, str]:
    return MODES if (int(prompt_index) + int(repetition)) % 2 == 0 else tuple(reversed(MODES))


def _args_for_chunk(args: argparse.Namespace, chunk_size: int) -> argparse.Namespace:
    selected = copy(args)
    selected.chunk_size = int(chunk_size)
    return selected


def _paired_correctness(
    rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["prompt_id"]), int(row["repetition"]))][str(row["mode"])] = row
    comparisons = []
    for (prompt_id, repetition), modes in sorted(grouped.items()):
        if set(modes) != set(MODES):
            raise ValueError(f"missing chunk pair for {prompt_id} repetition {repetition}")
        checks = {}
        for horizon in horizons:
            baseline = modes[MODES[0]]["checkpoints"][str(horizon)]["generated_token_ids"]
            candidate = modes[MODES[1]]["checkpoints"][str(horizon)]["generated_token_ids"]
            checks[str(horizon)] = baseline == candidate
        comparisons.append(
            {
                "prompt_id": prompt_id,
                "repetition": repetition,
                "horizons_exact": checks,
                "pass": all(checks.values()),
            }
        )

    deterministic = True
    prompt_ids = {str(row["prompt_id"]) for row in rows}
    for mode in MODES:
        for prompt_id in prompt_ids:
            selected = [
                row for row in rows if row["mode"] == mode and row["prompt_id"] == prompt_id
            ]
            for horizon in horizons:
                hashes = {
                    row["checkpoints"][str(horizon)]["generated_ids_sha256"]
                    for row in selected
                }
                deterministic = deterministic and len(hashes) == 1
    return {
        "pass": bool(all(item["pass"] for item in comparisons) and deterministic),
        "chunk_pairs": comparisons,
        "same_mode_repeat_deterministic": bool(deterministic),
    }


def _aggregate_selected(
    rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]
) -> dict[str, Any]:
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    prefill_seconds = sum(float(row["prefill_seconds"]) for row in rows)
    result: dict[str, Any] = {
        "runs": len(rows),
        "prompt_tokens": prompt_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": prompt_tokens / prefill_seconds,
        "ttft_median_seconds": statistics.median(float(row["ttft_seconds"]) for row in rows),
        "horizons": {},
    }
    for horizon in horizons:
        checkpoints = [row["checkpoints"][str(horizon)] for row in rows]
        decode_calls = sum(int(item["decode_forward_calls"]) for item in checkpoints)
        decode_seconds = sum(float(item["decode_seconds"]) for item in checkpoints)
        output_tokens = sum(int(item["output_tokens"]) for item in checkpoints)
        total_seconds = sum(float(item["total_seconds"]) for item in checkpoints)
        result["horizons"][str(horizon)] = {
            "output_tokens": output_tokens,
            "decode_forward_calls": decode_calls,
            "decode_seconds": decode_seconds,
            "decode_tok_s": decode_calls / decode_seconds,
            "total_seconds": total_seconds,
            "e2e_output_tok_s": output_tokens / total_seconds,
        }
    return result


def _aggregate(
    rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        mode_result = _aggregate_selected(selected, horizons)
        categories = {}
        for category in sorted({str(row["category"]) for row in selected}):
            categories[category] = _aggregate_selected(
                [row for row in selected if row["category"] == category],
                horizons,
            )
        mode_result["categories"] = categories
        result[mode] = mode_result

    baseline = result[MODES[0]]
    candidate = result[MODES[1]]
    comparisons: dict[str, Any] = {
        "prefill_speedup": candidate["prefill_tok_s"] / baseline["prefill_tok_s"],
        "ttft_speedup": baseline["ttft_median_seconds"] / candidate["ttft_median_seconds"],
        "categories": {},
        "horizons": {},
    }
    for category in sorted(baseline["categories"]):
        base_category = baseline["categories"][category]
        candidate_category = candidate["categories"][category]
        comparisons["categories"][category] = {
            "prefill_speedup": candidate_category["prefill_tok_s"]
            / base_category["prefill_tok_s"],
            "horizons": {
                str(horizon): {
                    "e2e_speedup": candidate_category["horizons"][str(horizon)][
                        "e2e_output_tok_s"
                    ]
                    / base_category["horizons"][str(horizon)]["e2e_output_tok_s"]
                }
                for horizon in horizons
            },
        }
    for horizon in horizons:
        base_checkpoint = baseline["horizons"][str(horizon)]
        candidate_checkpoint = candidate["horizons"][str(horizon)]
        comparisons["horizons"][str(horizon)] = {
            "decode_speedup": candidate_checkpoint["decode_tok_s"]
            / base_checkpoint["decode_tok_s"],
            "e2e_speedup": candidate_checkpoint["e2e_output_tok_s"]
            / base_checkpoint["e2e_output_tok_s"],
        }
    result["chunk128_vs_chunk64"] = comparisons
    return result


def _promotion_gate(
    aggregate: Mapping[str, Any],
    correctness: Mapping[str, Any],
    oracle: Mapping[str, Any],
    *,
    horizons: Sequence[int],
    recovered: bool,
) -> dict[str, Any]:
    comparison = aggregate["chunk128_vs_chunk64"]
    failed: list[str] = []
    if not correctness["pass"]:
        failed.append("chunk_outputs_not_exact")
    if not oracle["pass"]:
        failed.append("poolside_oracle_failed")
    if not recovered:
        failed.append("tracked_lifecycle_not_recovered")
    if float(comparison["prefill_speedup"]) <= 1.0:
        failed.append("aggregate_prefill_not_faster")
    for category, values in comparison["categories"].items():
        if float(values["prefill_speedup"]) <= 1.0:
            failed.append(f"{category}_prefill_not_faster")
        for horizon in horizons:
            if float(values["horizons"][str(horizon)]["e2e_speedup"]) < 1.0:
                failed.append(f"{category}_h{horizon}_e2e_regressed")
    for horizon in horizons:
        values = comparison["horizons"][str(horizon)]
        if float(values["e2e_speedup"]) < 1.0:
            failed.append(f"h{horizon}_e2e_regressed")
        decode_speedup = float(values["decode_speedup"])
        if not math.isfinite(decode_speedup) or not 0.98 <= decode_speedup <= 1.02:
            failed.append(f"h{horizon}_decode_outside_2pct")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "policy": (
            "exact outputs and oracle; aggregate plus every-category prefill/E2E non-regressive; "
            "decode within 2%; lifecycle exact"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    chunk_sizes = tuple(int(value) for value in args.chunk_sizes)
    horizons = tuple(int(value) for value in args.output_horizons)
    if chunk_sizes != CHUNK_SIZES:
        raise ValueError(f"retained LPF-4 gate requires exact chunk sizes {CHUNK_SIZES}")
    if horizons != RETAINED_HORIZONS:
        raise ValueError(f"retained LPF-4 gate requires exact horizons {RETAINED_HORIZONS}")
    if args.repetitions < 2:
        raise ValueError("retained LPF-4 gate requires at least two repetitions")
    if args.warmup_output_tokens <= 0:
        raise ValueError("warmup output tokens must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna LPF-4 gate requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_lpf4_chunk_ab",
        timing_protocol="same_session_balanced_chunk64_vs_chunk128_category_h16_h32",
        warmups=len(CHUNK_SIZES),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    if min(int(prompt["prompt_tokens"]) for prompt in prompts) <= CHUNK_SIZES[0]:
        raise RuntimeError("LPF-4 canonical prompts must all cross the chunk-64 boundary")
    if max(int(prompt["prompt_tokens"]) for prompt in prompts) > CHUNK_SIZES[1]:
        raise RuntimeError("LPF-4 canonical prompts must fit in one chunk-128 pass")

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
            prefill_chunk_size=max(CHUNK_SIZES),
        )
        load_seconds = time.perf_counter() - load_started
        for chunk_size in CHUNK_SIZES:
            _run_target(
                owner,
                prompts[0],
                mode="bulk",
                horizons=(int(args.warmup_output_tokens),),
                repetition=-1,
                args=_args_for_chunk(args, chunk_size),
            )
        for repetition in range(args.repetitions):
            for prompt_index, prompt in enumerate(prompts):
                for mode in _mode_order(prompt_index, repetition):
                    chunk_size = int(mode.removeprefix("chunk_"))
                    row = _run_target(
                        owner,
                        prompt,
                        mode="bulk",
                        horizons=horizons,
                        repetition=repetition,
                        args=_args_for_chunk(args, chunk_size),
                    )
                    row["mode"] = mode
                    row["chunk_size"] = chunk_size
                    rows.append(row)
                    print(
                        f"rep={repetition} prompt={prompt['id']} chunk={chunk_size} "
                        f"prefill={row['prefill_tok_s']:.3f} tok/s",
                        file=sys.stderr,
                        flush=True,
                    )
        oracle_args = _args_for_chunk(args, CHUNK_SIZES[1])
        oracle = _oracle_gate(owner, oracle_args)
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna LPF-4 gate")

    correctness = _paired_correctness(rows, horizons)
    aggregate = _aggregate(rows, horizons)
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    promotion = _promotion_gate(
        aggregate,
        correctness,
        oracle,
        horizons=horizons,
        recovered=recovered,
    )
    manifest_path = args.repacked_cache / "manifest.json"
    prompt_payload = args.prompts.read_bytes()
    passed = bool(promotion["pass"])
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_lpf4_chunk_ab",
        "status": "retained" if passed else "rejected",
        "pass": passed,
        "performance_claim": passed,
        "performance_claim_scope": (
            "same-session Laguna chunk-64 versus chunk-128 target AR over all canonical categories; "
            "LPF-1 tiled F16 default active; model load excluded"
        ),
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
            "chunk_sizes": list(CHUNK_SIZES),
            "output_horizons": list(horizons),
            "repetitions": args.repetitions,
            "warmup_output_tokens_per_chunk": args.warmup_output_tokens,
            "timed_order": "alternating chunk64/chunk128 per prompt and reversed next repetition",
            "timing_scope": "prefill plus fixed-horizon decode; resident model load excluded",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "all_prompts_cross_64_and_fit_128": True,
        },
        "load": {"seconds_excluded": load_seconds, "resident_nbytes": resident_nbytes},
        "rows": rows,
        "correctness": {
            **correctness,
            "poolside_oracle": oracle,
            "tracked_returned_to_baseline": recovered,
        },
        "aggregate": aggregate,
        "promotion": promotion,
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "Both chunk policies share resident weights and use isolated bounded sessions per prompt.",
            "Every canonical prompt crosses 64 tokens and fits in one 128-row physical pass.",
            "Chunk-128 changes scheduling only; kernels, arithmetic, KV visibility, and decode are unchanged.",
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
