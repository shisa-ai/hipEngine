#!/usr/bin/env python3
"""Screen Laguna's M16 T16-WMMA sparse-down route at 256/512 rows."""

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

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import host_array_ptr, memory_stats
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
    _normalized_log_probs,
    _progress,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROWS = (256, 512)
MODES = ("adaptive_grouped_smallm", "wmma16_down")
BASELINE_MODE = MODES[0]
CANDIDATE_MODE = MODES[1]
MAX_KL = 0.05
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-wmma16-down-screen.json"
)


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not rows or any(item <= 1 for item in rows):
        raise argparse.ArgumentTypeError("WMMA screen rows must be distinct integers above one")
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
    return MODES if (int(shape_index) + int(repetition)) % 2 == 0 else tuple(reversed(MODES))


def _quality_pair(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    baseline_log_probs = _normalized_log_probs(baseline)
    candidate_log_probs = _normalized_log_probs(candidate)
    probabilities = np.exp(baseline_log_probs)
    kl = float(np.sum(probabilities * (baseline_log_probs - candidate_log_probs)))
    baseline_top1 = int(np.argmax(baseline))
    candidate_top1 = int(np.argmax(candidate))
    finite = bool(
        np.isfinite(baseline).all()
        and np.isfinite(candidate).all()
        and math.isfinite(kl)
    )
    return {
        "kl_divergence": kl,
        "baseline_top1": baseline_top1,
        "candidate_top1": candidate_top1,
        "top1_agreement": baseline_top1 == candidate_top1,
        "finite": finite,
    }


def _comparison_summary(
    samples: Mapping[str, Mapping[int, Sequence[float]]],
    next_tokens: Mapping[str, Mapping[int, Sequence[int]]],
    quality: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    rows: Sequence[int],
) -> dict[str, Any]:
    parsed_rows = tuple(int(value) for value in rows)
    if parsed_rows != tuple(sorted(set(parsed_rows))) or not parsed_rows:
        raise ValueError("WMMA comparison rows must be sorted and distinct")
    if set(samples) != set(MODES) or set(next_tokens) != set(MODES):
        raise ValueError(f"WMMA comparison requires modes {MODES}")

    shapes: dict[str, Any] = {}
    baseline_seconds = 0.0
    candidate_seconds = 0.0
    failed: list[str] = []
    slower_rows: list[int] = []
    all_quality: list[Mapping[str, Any]] = []
    next_ids_agree = True
    for value in parsed_rows:
        mode_summaries: dict[str, Any] = {}
        mode_ids: dict[str, list[int]] = {}
        sample_counts: set[int] = set()
        for mode in MODES:
            timings = [float(item) for item in samples[mode][value]]
            ids = [int(item) for item in next_tokens[mode][value]]
            if len(timings) != len(ids):
                raise ValueError(f"rows={value} mode={mode} timing/token counts differ")
            sample_counts.add(len(timings))
            mode_summaries[mode] = _summarize_timing_samples(timings, rows=value)
            mode_ids[mode] = ids
        if sample_counts == {0} or len(sample_counts) != 1:
            raise ValueError(f"rows={value} modes require equal non-empty samples")
        shape_quality = [dict(item) for item in quality[value]]
        if len(shape_quality) != next(iter(sample_counts)):
            raise ValueError(f"rows={value} requires one quality pair per repetition")
        all_quality.extend(shape_quality)
        shape_ids_agree = mode_ids[BASELINE_MODE] == mode_ids[CANDIDATE_MODE]
        next_ids_agree = next_ids_agree and shape_ids_agree
        speedup = (
            mode_summaries[BASELINE_MODE]["median_seconds"]
            / mode_summaries[CANDIDATE_MODE]["median_seconds"]
        )
        if speedup <= 1.0:
            slower_rows.append(value)
        baseline_seconds += sum(float(item) for item in samples[BASELINE_MODE][value])
        candidate_seconds += sum(float(item) for item in samples[CANDIDATE_MODE][value])
        shapes[str(value)] = {
            "rows": value,
            BASELINE_MODE: mode_summaries[BASELINE_MODE],
            CANDIDATE_MODE: mode_summaries[CANDIDATE_MODE],
            "wmma16_vs_grouped_smallm_speedup": speedup,
            "next_token_ids": mode_ids,
            "next_token_ids_agree": shape_ids_agree,
            "quality": shape_quality,
        }

    finite = bool(all(bool(item["finite"]) for item in all_quality))
    top1_agreement = sum(bool(item["top1_agreement"]) for item in all_quality) / len(
        all_quality
    )
    max_kl = max(float(item["kl_divergence"]) for item in all_quality)
    aggregate_speedup = baseline_seconds / candidate_seconds
    if not finite:
        failed.append("nonfinite_logits")
    if max_kl > MAX_KL:
        failed.append("max_kl_above_0.05")
    if top1_agreement < 1.0 or not next_ids_agree:
        failed.append("top1_or_next_id_mismatch")
    if slower_rows:
        failed.append("wmma16_not_faster_at_every_shape")
    if aggregate_speedup <= 1.0:
        failed.append("wmma16_aggregate_not_faster")

    return {
        "shapes": shapes,
        "correctness": {
            "pass": finite and max_kl <= MAX_KL and top1_agreement == 1.0 and next_ids_agree,
            "finite": finite,
            "max_kl_divergence": max_kl,
            "top1_agreement": top1_agreement,
            "next_token_ids_agree": next_ids_agree,
            "thresholds": {
                "max_kl_divergence": MAX_KL,
                "minimum_top1_agreement": 1.0,
            },
        },
        "screen": {
            "pass": not failed,
            "failed_checks": failed,
            "slower_rows": slower_rows,
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "aggregate_speedup": aggregate_speedup,
            "policy": (
                "final logits finite; KL<=0.05; all final top-1/next IDs agree; "
                "M16 faster at both 256/512 and in aggregate"
            ),
        },
    }


def _copy_logits(
    owner: LagunaGGUFResidentSession,
    result: Any,
    destination: np.ndarray,
) -> None:
    owner.runtime.memcpy(
        host_array_ptr(destination),
        result.logits.ptr,
        destination.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tuple(int(value) for value in args.rows)
    if rows != PROFILE_ROWS:
        raise ValueError(f"retained WMMA screen requires exact rows {PROFILE_ROWS}")
    if args.repetitions < 2:
        raise ValueError("retained WMMA screen requires at least two repetitions")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    if max(rows) > args.context_length:
        raise ValueError("largest WMMA row shape exceeds admitted context")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna WMMA screen requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_wmma16_down_screen",
        timing_protocol="same_session_balanced_grouped_smallm_vs_wmma16_down_prefill",
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
    quality: dict[int, list[dict[str, Any]]] = {value: [] for value in rows}
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
        logits = {
            mode: np.empty(owner.config.vocab_size, dtype=np.float32) for mode in MODES
        }
        for _ in range(args.warmups):
            for shape_index, value in enumerate(rows):
                for mode in _mode_order(shape_index, 0):
                    owner.reset_state()
                    owner.set_selected_down_mode(mode)
                    owner.prefill(token_stream[:value], use_bulk=True)
        for repetition in range(args.repetitions):
            for shape_index, value in enumerate(rows):
                for mode in _mode_order(shape_index, repetition):
                    owner.reset_state()
                    owner.set_selected_down_mode(mode)
                    started = time.perf_counter()
                    result = owner.prefill(token_stream[:value], use_bulk=True)
                    runtime.device_synchronize()
                    elapsed = time.perf_counter() - started
                    samples[mode][value].append(elapsed)
                    next_tokens[mode][value].append(int(result.next_token_id))
                    _copy_logits(owner, result, logits[mode])
                    print(
                        f"rep={repetition} rows={value} mode={mode} "
                        f"prefill={value / elapsed:.3f} tok/s next={result.next_token_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                quality[value].append(
                    _quality_pair(logits[BASELINE_MODE], logits[CANDIDATE_MODE])
                )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during WMMA screen")

    comparison = _comparison_summary(samples, next_tokens, quality, rows=rows)
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    passed = bool(comparison["screen"]["pass"] and recovered)
    if not recovered:
        comparison["screen"]["failed_checks"].append("tracked_ownership_not_recovered")
        comparison["screen"]["pass"] = False
    prompts_payload = args.prompts.read_bytes()
    manifest_path = args.repacked_cache / "manifest.json"
    manifest_sha256 = (
        _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_wmma16_down_screen",
        "status": "screen_passed" if passed else "rejected",
        "pass": passed,
        "performance_claim": False,
        "performance_claim_scope": (
            "candidate-selection screen only; full ten-prompt category quality/E2E gate "
            "is required before promotion"
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
            "prefill_chunk_size": max(rows),
            "context_length": args.context_length,
            "repetitions": args.repetitions,
            "warmups_per_shape_and_mode": args.warmups,
            "timed_order": "counterbalanced per shape and repetition",
            "timing_scope": (
                "reset excluded; prefill through synchronized first-token projection; "
                "full-logit D2H excluded; model load excluded"
            ),
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompts_payload),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
            "candidate_selection": (
                "session-local selected_down_mode=wmma16_down; device grouping/gather, "
                "one scalar padded-row read per sparse layer, M16 T16-WMMA Q4/Q6 down"
            ),
            "control_selection": (
                "session-local selected_down_mode=adaptive_grouped_smallm; retained gfx1151 default"
            ),
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
            "Natural token routing comes from the complete 48-layer model, not a synthetic bucket map.",
            "The candidate reuses resident Q4T16/Q6T16 tiles and caller-owned bounded metadata scratch.",
            "A passing screen admits, but does not replace, the full category quality/E2E gate.",
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
