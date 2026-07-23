#!/usr/bin/env python3
"""Gate Laguna M16 sparse down on expanded full-category 256/512 workloads."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
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
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    RETAINED_HORIZONS,
    _compiler_version,
    _load_prompts,
    _normalized_log_probs,
    _progress,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)
from scripts.laguna_wmma16_down_prefill_bench import (
    BASELINE_MODE,
    CANDIDATE_MODE,
    MAX_KL,
    MODES,
    PROFILE_ROWS,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREEN = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-wmma16-down-screen.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-wmma16-down-category.json"
)
CATEGORY_E2E_FLOOR = 0.98
DECODE_RATIO_FLOOR = 0.98
DECODE_RATIO_CEILING = 1.02


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--rows",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=PROFILE_ROWS,
    )
    parser.add_argument(
        "--output-horizons",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=RETAINED_HORIZONS,
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(shape_index: int, prompt_index: int, repetition: int) -> tuple[str, str]:
    parity = int(shape_index) + int(prompt_index) + int(repetition)
    return MODES if parity % 2 == 0 else tuple(reversed(MODES))


def _expanded_prompt_tokens(prompt: Mapping[str, Any], rows: int) -> tuple[int, ...]:
    target = int(rows)
    source = tuple(int(token) for token in prompt["token_ids"])
    if target <= 0 or not source:
        raise ValueError("expanded category prompts require positive rows and source tokens")
    if len(source) >= target:
        return source[:target]
    extension = source[1:] if len(source) > 1 else source
    result = list(source)
    while len(result) < target:
        result.extend(extension[: target - len(result)])
    return tuple(result)


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


def _quality_pair(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    repetition: int,
    rows: int,
    prompt: Mapping[str, Any],
) -> dict[str, Any]:
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
        "repetition": int(repetition),
        "rows": int(rows),
        "prompt_id": str(prompt["id"]),
        "category": str(prompt["category"]),
        "baseline_top1": baseline_top1,
        "candidate_top1": candidate_top1,
        "top1_agreement": baseline_top1 == candidate_top1,
        "kl_divergence": kl,
        "finite": finite,
    }


def _run_mode(
    owner: LagunaGGUFResidentSession,
    prompt: Mapping[str, Any],
    token_ids: Sequence[int],
    *,
    rows: int,
    mode: str,
    repetition: int,
    horizons: Sequence[int],
) -> tuple[dict[str, Any], np.ndarray]:
    owner.reset_state()
    owner.set_selected_down_mode(mode)
    prefill_started = time.perf_counter()
    result = owner.prefill(token_ids, use_bulk=True)
    owner.runtime.device_synchronize()
    prefill_seconds = time.perf_counter() - prefill_started
    logits = np.empty(owner.config.vocab_size, dtype=np.float32)
    _copy_logits(owner, result, logits)
    generated = [int(result.next_token_id)]
    decode_steps: list[float] = []
    while len(generated) < max(horizons):
        started = time.perf_counter()
        result = owner.forward_token(result.next_token_id)
        owner.runtime.device_synchronize()
        decode_steps.append(time.perf_counter() - started)
        generated.append(int(result.next_token_id))
    checkpoints: dict[str, Any] = {}
    for horizon in horizons:
        decode_seconds = float(sum(decode_steps[: max(0, horizon - 1)]))
        checkpoints[str(horizon)] = {
            "output_tokens": int(horizon),
            "decode_forward_calls": max(0, int(horizon) - 1),
            "decode_seconds": decode_seconds,
            "total_seconds": prefill_seconds + decode_seconds,
            "generated_token_ids": generated[: int(horizon)],
            "generated_ids_sha256": _sha256_json(generated[: int(horizon)]),
        }
    return (
        {
            "repetition": int(repetition),
            "rows": int(rows),
            "prompt_id": str(prompt["id"]),
            "category": str(prompt["category"]),
            "mode": str(mode),
            "prompt_tokens": int(rows),
            "expanded_token_ids_sha256": _sha256_json(token_ids),
            "source_prompt_tokens": int(prompt["prompt_tokens"]),
            "source_token_ids_sha256": str(prompt["token_ids_sha256"]),
            "prefill_seconds": prefill_seconds,
            "prefill_tok_s": int(rows) / prefill_seconds,
            "next_token_id": int(generated[0]),
            "checkpoints": checkpoints,
        },
        logits,
    )


def _aggregate_selected(
    selected: Sequence[Mapping[str, Any]], horizons: Sequence[int]
) -> dict[str, Any]:
    if not selected:
        raise ValueError("category aggregate slice must not be empty")
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in selected)
    prefill_seconds = sum(float(row["prefill_seconds"]) for row in selected)
    result: dict[str, Any] = {
        "runs": len(selected),
        "prompt_tokens": prompt_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": prompt_tokens / prefill_seconds,
        "ttft_median_seconds": statistics.median(
            float(row["prefill_seconds"]) for row in selected
        ),
        "horizons": {},
    }
    for horizon in horizons:
        checkpoints = [row["checkpoints"][str(horizon)] for row in selected]
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


def _mode_aggregate(
    runs: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    rows: Sequence[int],
    categories: Sequence[str],
    horizons: Sequence[int],
) -> dict[str, Any]:
    selected = [row for row in runs if row["mode"] == mode]
    return {
        "overall": _aggregate_selected(selected, horizons),
        "shapes": {
            str(shape): _aggregate_selected(
                [row for row in selected if int(row["rows"]) == int(shape)], horizons
            )
            for shape in rows
        },
        "categories": {
            category: _aggregate_selected(
                [row for row in selected if row["category"] == category], horizons
            )
            for category in categories
        },
        "shape_categories": {
            str(shape): {
                category: _aggregate_selected(
                    [
                        row
                        for row in selected
                        if int(row["rows"]) == int(shape)
                        and row["category"] == category
                    ],
                    horizons,
                )
                for category in categories
            }
            for shape in rows
        },
    }


def _compare_slice(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    horizons: Sequence[int],
) -> dict[str, Any]:
    return {
        "prefill_speedup": candidate["prefill_tok_s"] / baseline["prefill_tok_s"],
        "ttft_speedup": baseline["ttft_median_seconds"]
        / candidate["ttft_median_seconds"],
        "horizons": {
            str(horizon): {
                "decode_speedup": candidate["horizons"][str(horizon)]["decode_tok_s"]
                / baseline["horizons"][str(horizon)]["decode_tok_s"],
                "e2e_speedup": candidate["horizons"][str(horizon)]["e2e_output_tok_s"]
                / baseline["horizons"][str(horizon)]["e2e_output_tok_s"],
            }
            for horizon in horizons
        },
    }


def _aggregate(
    runs: Sequence[Mapping[str, Any]],
    *,
    rows: Sequence[int],
    horizons: Sequence[int],
) -> dict[str, Any]:
    categories = tuple(sorted({str(row["category"]) for row in runs}))
    result = {
        mode: _mode_aggregate(
            runs,
            mode=mode,
            rows=rows,
            categories=categories,
            horizons=horizons,
        )
        for mode in MODES
    }
    baseline = result[BASELINE_MODE]
    candidate = result[CANDIDATE_MODE]
    result["comparison"] = {
        "overall": _compare_slice(baseline["overall"], candidate["overall"], horizons),
        "shapes": {
            str(shape): _compare_slice(
                baseline["shapes"][str(shape)],
                candidate["shapes"][str(shape)],
                horizons,
            )
            for shape in rows
        },
        "categories": {
            category: _compare_slice(
                baseline["categories"][category],
                candidate["categories"][category],
                horizons,
            )
            for category in categories
        },
        "shape_categories": {
            str(shape): {
                category: _compare_slice(
                    baseline["shape_categories"][str(shape)][category],
                    candidate["shape_categories"][str(shape)][category],
                    horizons,
                )
                for category in categories
            }
            for shape in rows
        },
    }
    return result


def _quality_summary(quality: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not quality:
        raise ValueError("category quality pairs must not be empty")
    finite = all(bool(item["finite"]) for item in quality)
    max_kl = max(float(item["kl_divergence"]) for item in quality)
    matches = sum(bool(item["top1_agreement"]) for item in quality)
    agreement = matches / len(quality)
    categories: dict[str, Any] = {}
    failed: list[str] = []
    if not finite:
        failed.append("nonfinite_logits")
    if max_kl > MAX_KL:
        failed.append("max_kl_above_0.05")
    if agreement < 0.9:
        failed.append("suite_top1_below_0.9")
    for category in sorted({str(item["category"]) for item in quality}):
        selected = [item for item in quality if item["category"] == category]
        category_agreement = sum(bool(item["top1_agreement"]) for item in selected) / len(
            selected
        )
        category_max_kl = max(float(item["kl_divergence"]) for item in selected)
        category_finite = all(bool(item["finite"]) for item in selected)
        categories[category] = {
            "pairs": len(selected),
            "top1_agreement": category_agreement,
            "max_kl_divergence": category_max_kl,
            "finite": category_finite,
        }
        if category_agreement < 0.9:
            failed.append(f"{category}_top1_below_0.9")
        if not category_finite:
            failed.append(f"{category}_nonfinite_logits")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "pairs": len(quality),
        "top1_agreement": agreement,
        "max_kl_divergence": max_kl,
        "finite": finite,
        "categories": categories,
        "thresholds": {
            "max_kl_divergence": MAX_KL,
            "minimum_suite_top1_agreement": 0.9,
            "minimum_each_category_top1_agreement": 0.9,
        },
    }


def _free_running_summary(
    runs: Sequence[Mapping[str, Any]], horizons: Sequence[int]
) -> dict[str, Any]:
    grouped: dict[tuple[int, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    deterministic: dict[tuple[str, int, str, int], set[str]] = defaultdict(set)
    for row in runs:
        key = (int(row["repetition"]), int(row["rows"]), str(row["prompt_id"]))
        grouped[key][str(row["mode"])] = row
        for horizon in horizons:
            deterministic[
                (
                    str(row["mode"]),
                    int(row["rows"]),
                    str(row["prompt_id"]),
                    int(horizon),
                )
            ].add(str(row["checkpoints"][str(horizon)]["generated_ids_sha256"]))
    mismatches: list[dict[str, Any]] = []
    for (repetition, rows, prompt_id), pair in sorted(grouped.items()):
        if set(pair) != set(MODES):
            raise ValueError("free-running category pair is missing a mode")
        for horizon in horizons:
            baseline_ids = pair[BASELINE_MODE]["checkpoints"][str(horizon)][
                "generated_token_ids"
            ]
            candidate_ids = pair[CANDIDATE_MODE]["checkpoints"][str(horizon)][
                "generated_token_ids"
            ]
            if baseline_ids != candidate_ids:
                mismatches.append(
                    {
                        "repetition": repetition,
                        "rows": rows,
                        "prompt_id": prompt_id,
                        "horizon": int(horizon),
                    }
                )
    nondeterministic = [
        {
            "mode": key[0],
            "rows": key[1],
            "prompt_id": key[2],
            "horizon": key[3],
        }
        for key, hashes in sorted(deterministic.items())
        if len(hashes) != 1
    ]
    return {
        "pass": not mismatches and not nondeterministic,
        "pairs": len(grouped),
        "mismatches": mismatches,
        "nondeterministic": nondeterministic,
    }


def _promotion_gate(
    aggregate: Mapping[str, Any],
    quality: Mapping[str, Any],
    free_running: Mapping[str, Any],
    *,
    rows: Sequence[int],
    horizons: Sequence[int],
) -> dict[str, Any]:
    comparison = aggregate["comparison"]
    failed: list[str] = []
    if not quality["pass"]:
        failed.append("quality_gate_failed")
    if not free_running["pass"]:
        failed.append("free_running_gate_failed")
    if comparison["overall"]["prefill_speedup"] <= 1.0:
        failed.append("aggregate_prefill_not_faster")
    for shape in rows:
        shape_key = str(shape)
        if comparison["shapes"][shape_key]["prefill_speedup"] <= 1.0:
            failed.append(f"rows_{shape}_prefill_not_faster")
        for category, item in comparison["shape_categories"][shape_key].items():
            if item["prefill_speedup"] <= 1.0:
                failed.append(f"rows_{shape}_{category}_prefill_not_faster")
            for horizon in horizons:
                if item["horizons"][str(horizon)]["e2e_speedup"] < CATEGORY_E2E_FLOOR:
                    failed.append(
                        f"rows_{shape}_{category}_h{horizon}_e2e_below_floor"
                    )
        for horizon in horizons:
            item = comparison["shapes"][shape_key]["horizons"][str(horizon)]
            if item["e2e_speedup"] <= 1.0:
                failed.append(f"rows_{shape}_h{horizon}_e2e_not_faster")
            if not DECODE_RATIO_FLOOR <= item["decode_speedup"] <= DECODE_RATIO_CEILING:
                failed.append(f"rows_{shape}_h{horizon}_decode_outside_2pct")
    for horizon in horizons:
        if comparison["overall"]["horizons"][str(horizon)]["e2e_speedup"] <= 1.0:
            failed.append(f"aggregate_h{horizon}_e2e_not_faster")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "thresholds": {
            "quality_max_kl": MAX_KL,
            "shape_and_category_prefill_speedup": ">1.0",
            "aggregate_and_shape_e2e_speedup": ">1.0",
            "shape_category_e2e_floor": CATEGORY_E2E_FLOOR,
            "decode_speedup_range": [DECODE_RATIO_FLOOR, DECODE_RATIO_CEILING],
            "free_running_mode_pairs_and_repeats": "exact",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tuple(int(value) for value in args.rows)
    horizons = tuple(sorted({int(value) for value in args.output_horizons}))
    if rows != PROFILE_ROWS:
        raise ValueError(f"retained M16 category gate requires rows {PROFILE_ROWS}")
    if horizons != RETAINED_HORIZONS:
        raise ValueError(
            f"retained M16 category gate requires horizons {RETAINED_HORIZONS}"
        )
    if args.repetitions < 3:
        raise ValueError("retained M16 category gate requires at least three repetitions")
    if not args.model.is_file() or not args.prompts.is_file():
        raise FileNotFoundError("Laguna model and prompt suite are required")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    if not screen.get("pass") or screen.get("kind") != "hipengine_laguna_prefill_wmma16_down_screen":
        raise ValueError("M16 category gate requires the accepted 256/512 screen")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained M16 category gate requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_wmma16_down_category",
        timing_protocol="one_load_counterbalanced_expanded_category_h16_h32",
        warmups=len(rows) * len(MODES),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    categories = tuple(sorted({str(prompt["category"]) for prompt in prompts}))
    if categories != ("code", "general_en", "general_ja", "mixed_ja_en"):
        raise ValueError("M16 category gate requires all four canonical categories")
    expanded = {
        (str(prompt["id"]), shape): _expanded_prompt_tokens(prompt, shape)
        for prompt in prompts
        for shape in rows
    }

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    runs: list[dict[str, Any]] = []
    quality_pairs: list[dict[str, Any]] = []
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
        warm_prompt = prompts[0]
        for shape in rows:
            token_ids = expanded[(str(warm_prompt["id"]), shape)]
            for mode in MODES:
                owner.reset_state()
                owner.set_selected_down_mode(mode)
                owner.prefill(token_ids, use_bulk=True)
        for repetition in range(args.repetitions):
            for shape_index, shape in enumerate(rows):
                for prompt_index, prompt in enumerate(prompts):
                    token_ids = expanded[(str(prompt["id"]), shape)]
                    logits: dict[str, np.ndarray] = {}
                    for mode in _mode_order(shape_index, prompt_index, repetition):
                        row, mode_logits = _run_mode(
                            owner,
                            prompt,
                            token_ids,
                            rows=shape,
                            mode=mode,
                            repetition=repetition,
                            horizons=horizons,
                        )
                        runs.append(row)
                        logits[mode] = mode_logits
                        print(
                            f"rep={repetition} rows={shape} prompt={prompt['id']} mode={mode} "
                            f"prefill={row['prefill_tok_s']:.3f} tok/s "
                            f"next={row['next_token_id']}",
                            file=sys.stderr,
                            flush=True,
                        )
                    quality_pairs.append(
                        _quality_pair(
                            logits[BASELINE_MODE],
                            logits[CANDIDATE_MODE],
                            repetition=repetition,
                            rows=shape,
                            prompt=prompt,
                        )
                    )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during M16 category gate")

    aggregate = _aggregate(runs, rows=rows, horizons=horizons)
    quality = _quality_summary(quality_pairs)
    free_running = _free_running_summary(runs, horizons)
    promotion = _promotion_gate(
        aggregate,
        quality,
        free_running,
        rows=rows,
        horizons=horizons,
    )
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    if not recovered:
        promotion["failed_checks"].append("tracked_ownership_not_recovered")
        promotion["pass"] = False
    passed = bool(promotion["pass"] and recovered)
    prompt_bytes = args.prompts.read_bytes()
    manifest_path = args.repacked_cache / "manifest.json"
    manifest_sha256 = (
        _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_wmma16_down_category",
        "status": "retained" if passed else "rejected",
        "pass": passed,
        "performance_claim": passed,
        "performance_claim_scope": (
            "explicit one-physical-chunk 256/512 Poolside Laguna M16 down versus "
            "retained grouped-small-M; default chunk remains 128 pending AR-O3"
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
            "output_horizons": list(horizons),
            "repetitions": args.repetitions,
            "warmups": len(rows) * len(MODES),
            "prefill_chunk_size": max(rows),
            "one_physical_chunk": True,
            "context_length": args.context_length,
            "timed_order": "counterbalanced by repetition, shape, and prompt",
            "timing_scope": (
                "reset and final-logit D2H excluded; prefill and each decode transition "
                "synchronized; model load excluded"
            ),
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_bytes),
            "categories": list(categories),
            "expansion": "repeat each prompt without repeating its leading BOS to exact rows",
            "expanded_streams": {
                f"{prompt['id']}:{shape}": {
                    "rows": shape,
                    "sha256": _sha256_json(expanded[(str(prompt["id"]), shape)]),
                }
                for prompt in prompts
                for shape in rows
            },
            "accepted_screen": str(args.screen.resolve()),
            "accepted_screen_sha256": _sha256_bytes(args.screen.read_bytes()),
        },
        "load": {
            "seconds_excluded": load_seconds,
            "resident_nbytes": resident_nbytes,
        },
        "aggregate": aggregate,
        "quality": quality,
        "free_running": free_running,
        "promotion": promotion,
        "runs": runs,
        "quality_pairs": quality_pairs,
        "correctness": {
            "pass": quality["pass"] and free_running["pass"] and recovered,
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
            "All ten canonical prompts contribute at both 256 and 512 rows.",
            "Repeated content preserves each prompt/category and avoids fixed-ID specialization.",
            "A pass qualifies adaptive M16 from 256 rows; AR-O3 still owns default chunk sizing.",
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
