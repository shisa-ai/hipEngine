#!/usr/bin/env python3
"""Benchmark adaptive exact Laguna grouped-small-M down against direct GEMV."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_prefill_profile import _profile_token_stream, _summarize_timing_samples
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
PROFILE_ROWS = (16, 32, 55, 64, 122, 128)
MODES = ("direct", "adaptive_grouped_smallm")
GROUPED_MIN_ROWS = 32
FALLBACK_MIN_RATIO = 0.995
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-down-ab.json"
)


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not rows or any(item <= 1 for item in rows):
        raise argparse.ArgumentTypeError(
            "grouped-down rows must be distinct integers greater than one"
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--rows", type=_parse_rows, default=PROFILE_ROWS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(shape_index: int, repetition: int) -> tuple[str, str]:
    """Alternate A/B order for every shape and reverse it on the next pass."""

    return MODES if (int(shape_index) + int(repetition)) % 2 == 0 else tuple(reversed(MODES))


def _set_mode(owner: LagunaGGUFResidentSession, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown Laguna grouped-down mode {mode!r}")
    owner.set_selected_down_mode(mode)


def _comparison_summary(
    samples: Mapping[str, Mapping[int, Sequence[float]]],
    next_tokens: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    rows: Sequence[int],
) -> dict[str, Any]:
    parsed_rows = tuple(int(value) for value in rows)
    if not parsed_rows or tuple(sorted(set(parsed_rows))) != parsed_rows:
        raise ValueError("grouped-down comparison rows must be sorted and distinct")
    if set(samples) != set(MODES) or set(next_tokens) != set(MODES):
        raise ValueError(f"grouped-down comparison requires modes {MODES}")

    candidate = "adaptive_grouped_smallm"
    shapes: dict[str, Any] = {}
    exact = True
    direct_seconds = 0.0
    candidate_seconds = 0.0
    total_tokens = 0
    regressed_rows: list[int] = []
    non_improving_grouped_rows: list[int] = []
    for value in parsed_rows:
        mode_summaries: dict[str, Any] = {}
        mode_tokens: dict[str, list[int]] = {}
        sample_counts: set[int] = set()
        for mode in MODES:
            mode_samples = [float(item) for item in samples[mode][value]]
            mode_ids = [int(item) for item in next_tokens[mode][value]]
            if len(mode_samples) != len(mode_ids):
                raise ValueError(f"rows={value} mode={mode} timing/token counts differ")
            sample_counts.add(len(mode_samples))
            mode_summaries[mode] = _summarize_timing_samples(
                mode_samples, rows=value
            )
            mode_tokens[mode] = mode_ids
        if sample_counts == {0} or len(sample_counts) != 1:
            raise ValueError(f"rows={value} modes require equal non-empty sample counts")
        shape_ids = mode_tokens["direct"] + mode_tokens[candidate]
        shape_exact = len(set(shape_ids)) == 1
        exact = exact and shape_exact
        speedup = (
            mode_summaries["direct"]["median_seconds"]
            / mode_summaries[candidate]["median_seconds"]
        )
        if speedup < FALLBACK_MIN_RATIO:
            regressed_rows.append(value)
        if value >= GROUPED_MIN_ROWS and speedup <= 1.0:
            non_improving_grouped_rows.append(value)
        direct_seconds += sum(float(item) for item in samples["direct"][value])
        candidate_seconds += sum(float(item) for item in samples[candidate][value])
        total_tokens += value * len(samples["direct"][value])
        shapes[str(value)] = {
            "rows": value,
            "route": "grouped_smallm" if value >= GROUPED_MIN_ROWS else "direct_fallback",
            "direct": mode_summaries["direct"],
            candidate: mode_summaries[candidate],
            "adaptive_grouped_smallm_vs_direct_speedup": speedup,
            "next_token_ids": mode_tokens,
            "exact_next_token": shape_exact,
        }

    effective_speedup = (
        direct_seconds / candidate_seconds if candidate_seconds > 0.0 else None
    )
    failed: list[str] = []
    if not exact:
        failed.append("output_ids_not_exact")
    if regressed_rows:
        failed.append("candidate_below_no_regression_floor")
    if non_improving_grouped_rows:
        failed.append("grouped_route_not_faster")
    if (
        effective_speedup is None
        or not math.isfinite(effective_speedup)
        or effective_speedup <= 1.0
    ):
        failed.append("effective_profile_not_faster")

    return {
        "shapes": shapes,
        "correctness": {
            "pass": exact,
            "all_modes_and_repetitions_exact_next_token": exact,
        },
        "promotion": {
            "pass": not failed,
            "failed_checks": failed,
            "regressed_rows": regressed_rows,
            "non_improving_grouped_rows": non_improving_grouped_rows,
            "grouped_min_rows": GROUPED_MIN_ROWS,
            "fallback_min_ratio": FALLBACK_MIN_RATIO,
            "policy": (
                "exact IDs; direct fallback >=0.995x; every grouped shape >1x; "
                "aggregate wall >1x"
            ),
            "effective_profile_tokens": total_tokens,
            "direct_profile_seconds": direct_seconds,
            "adaptive_grouped_smallm_profile_seconds": candidate_seconds,
            "effective_speedup": effective_speedup,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tuple(int(value) for value in args.rows)
    if rows != PROFILE_ROWS:
        raise ValueError(
            f"retained grouped-down A/B requires exact rows {PROFILE_ROWS}"
        )
    if args.repetitions < 2:
        raise ValueError(
            "retained grouped-down A/B requires at least two repetitions"
        )
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    if max(rows) > args.context_length:
        raise ValueError("largest grouped-down row shape exceeds admitted context")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError(
            "retained Laguna grouped-down A/B requires a clean tracked worktree"
        )

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_grouped_down_ab",
        timing_protocol=(
            "same_session_balanced_direct_vs_adaptive_grouped_smallm_down_prefill"
        ),
        warmups=args.warmups * len(rows) * len(MODES),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(rows))

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    samples = {mode: {value: [] for value in rows} for mode in MODES}
    next_tokens = {mode: {value: [] for value in rows} for mode in MODES}
    owner: LagunaGGUFResidentSession | None = None
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
            prefill_chunk_size=max(rows),
        )
        load_seconds = time.perf_counter() - load_started
        for _ in range(args.warmups):
            for shape_index, value in enumerate(rows):
                for mode in _mode_order(shape_index, 0):
                    owner.reset_state()
                    _set_mode(owner, mode)
                    owner.prefill(token_stream[:value], use_bulk=True)
        for repetition in range(args.repetitions):
            for shape_index, value in enumerate(rows):
                for mode in _mode_order(shape_index, repetition):
                    owner.reset_state()
                    _set_mode(owner, mode)
                    started = time.perf_counter()
                    result = owner.prefill(token_stream[:value], use_bulk=True)
                    runtime.device_synchronize()
                    elapsed = time.perf_counter() - started
                    samples[mode][value].append(elapsed)
                    next_tokens[mode][value].append(int(result.next_token_id))
                    print(
                        f"rep={repetition} rows={value} mode={mode} "
                        f"prefill={value / elapsed:.3f} tok/s next={result.next_token_id}",
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
        raise RuntimeError("HIP total memory changed during grouped-down A/B")

    comparison = _comparison_summary(samples, next_tokens, rows=rows)
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"]
        == tracked_before["active_allocations"]
    )
    passed = bool(
        comparison["correctness"]["pass"]
        and comparison["promotion"]["pass"]
        and recovered
    )
    prompts_payload = args.prompts.read_bytes()
    manifest_path = args.repacked_cache / "manifest.json"
    manifest_sha256 = (
        _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_grouped_down_ab",
        "status": "retained" if passed else "rejected",
        "pass": passed,
        "performance_claim": passed,
        "performance_claim_scope": (
            "same-session Laguna direct selected-down prefill versus exact adaptive grouped-"
            "small-M Q4/Q6 down; retained tiled F16 default active; one physical chunk; "
            "model load excluded"
        ),
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": manifest_sha256,
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "rows": list(rows),
            "modes": list(MODES),
            "one_physical_chunk": True,
            "prefill_chunk_size": max(rows),
            "context_length": args.context_length,
            "repetitions": args.repetitions,
            "warmups_per_shape_and_mode": args.warmups,
            "timed_order": (
                "alternating direct/adaptive_grouped_smallm per shape and reversed "
                "next repetition"
            ),
            "timing_scope": (
                "reset complete through synchronized first-token projection; load excluded"
            ),
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompts_payload),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
            "candidate_selection": (
                "session-local selected_down_mode=adaptive_grouped_smallm; direct below "
                f"{GROUPED_MIN_ROWS} rows"
            ),
            "control_selection": "session-local selected_down_mode=direct",
            "f16_prefill_selection": "gfx1151 retained LPF-1 backend default",
        },
        "load": {
            "seconds_excluded": load_seconds,
            "resident_nbytes": resident_nbytes,
        },
        **comparison,
        "correctness": {
            **comparison["correctness"],
            "tracked_returned_to_baseline": recovered,
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
            "Direct and adaptive grouped spans share one resident model, runtime, "
            "token stream, and bounded scratch owner.",
            "The candidate groups exact top-10 lanes without padding and restores "
            "original lane order for weighted sum.",
            "The separate canonical category artifact must own free-running h16/h32 "
            "and E2E promotion gates.",
            "Run rocprofv3 separately with cached builds; do not profile model JIT "
            "compilation.",
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
