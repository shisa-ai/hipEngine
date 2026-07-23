#!/usr/bin/env python3
"""Gate exact adaptive Laguna grouped-small-M down on the full AR suite."""

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
    DEFAULT_ORACLE,
    DEFAULT_ORACLE_LOGPROBS,
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    RETAINED_HORIZONS,
    _compiler_version,
    _load_prompts,
    _normalized_log_probs,
    _oracle_gate,
    _progress,
    _repo_state,
    _session,
    _sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
MODES = ("direct", "adaptive_grouped_smallm")
BASELINE_MODE = MODES[0]
CANDIDATE_MODE = MODES[1]
DEFAULT_SCREEN = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-down-ab.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-down-category.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--oracle-logprobs", type=Path, default=DEFAULT_ORACLE_LOGPROBS)
    parser.add_argument("--shape-screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--output-horizons",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=RETAINED_HORIZONS,
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-output-tokens", type=int, default=2)
    parser.add_argument("--teacher-forced-tokens", type=int, default=32)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(prompt_index: int, repetition: int) -> tuple[str, str]:
    return MODES if (int(prompt_index) + int(repetition)) % 2 == 0 else tuple(reversed(MODES))


def _session_for_mode(
    owner: LagunaGGUFResidentSession,
    args: argparse.Namespace,
    mode: str,
) -> LagunaGGUFResidentSession:
    if mode not in MODES:
        raise ValueError(f"unknown Laguna selected-down mode {mode!r}")
    session = _session(owner, args)
    session.set_selected_down_mode(mode)
    return session


def _run_target_mode(
    owner: LagunaGGUFResidentSession,
    prompt: Mapping[str, Any],
    *,
    mode: str,
    horizons: Sequence[int],
    repetition: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    session = _session_for_mode(owner, args, mode)
    try:
        prefill_started = time.perf_counter()
        result = session.prefill(prompt["token_ids"], use_bulk=True)
        prefill_seconds = time.perf_counter() - prefill_started
        generated = [int(result.next_token_id)]
        decode_steps: list[float] = []
        while len(generated) < max(horizons):
            started = time.perf_counter()
            result = session.forward_token(result.next_token_id)
            decode_steps.append(time.perf_counter() - started)
            generated.append(int(result.next_token_id))
        checkpoints = {}
        for horizon in horizons:
            decode_seconds = float(sum(decode_steps[: max(0, horizon - 1)]))
            total_seconds = prefill_seconds + decode_seconds
            checkpoints[str(horizon)] = {
                "output_tokens": int(horizon),
                "generated_token_ids": generated[:horizon],
                "generated_ids_sha256": _sha256_bytes(
                    json.dumps(generated[:horizon], separators=(",", ":")).encode()
                ),
                "decode_forward_calls": max(0, int(horizon) - 1),
                "decode_seconds": decode_seconds,
                "decode_tok_s": (
                    (int(horizon) - 1) / decode_seconds if horizon > 1 else None
                ),
                "total_seconds": total_seconds,
                "e2e_output_tok_s": int(horizon) / total_seconds,
            }
        return {
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "prompt_tokens": prompt["prompt_tokens"],
            "prompt_token_ids_sha256": prompt["token_ids_sha256"],
            "mode": mode,
            "repetition": int(repetition),
            "prefill_seconds": prefill_seconds,
            "ttft_seconds": prefill_seconds,
            "prefill_tok_s": prompt["prompt_tokens"] / prefill_seconds,
            "checkpoints": checkpoints,
        }
    finally:
        session.close()


def _paired_free_running(
    rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["prompt_id"]), int(row["repetition"]))][str(row["mode"])] = row
    comparisons = []
    for (prompt_id, repetition), modes in sorted(grouped.items()):
        if set(modes) != set(MODES):
            raise ValueError(
                f"missing grouped-down pair for {prompt_id} repetition {repetition}"
            )
        checks = {}
        for horizon in horizons:
            baseline = modes[BASELINE_MODE]["checkpoints"][str(horizon)][
                "generated_token_ids"
            ]
            candidate = modes[CANDIDATE_MODE]["checkpoints"][str(horizon)][
                "generated_token_ids"
            ]
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
        "all_pairs_exact": bool(all(item["pass"] for item in comparisons)),
        "same_mode_repeat_deterministic": bool(deterministic),
        "pairs": comparisons,
        "admission_role": "exact mode pairs and deterministic repeats are required",
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
        mode_result["categories"] = {
            category: _aggregate_selected(
                [row for row in selected if row["category"] == category], horizons
            )
            for category in sorted({str(row["category"]) for row in selected})
        }
        result[mode] = mode_result

    baseline = result[BASELINE_MODE]
    candidate = result[CANDIDATE_MODE]
    comparison: dict[str, Any] = {
        "prefill_speedup": candidate["prefill_tok_s"] / baseline["prefill_tok_s"],
        "ttft_speedup": baseline["ttft_median_seconds"] / candidate["ttft_median_seconds"],
        "categories": {},
        "horizons": {},
    }
    for category in sorted(baseline["categories"]):
        base_category = baseline["categories"][category]
        candidate_category = candidate["categories"][category]
        comparison["categories"][category] = {
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
        comparison["horizons"][str(horizon)] = {
            "decode_speedup": candidate_checkpoint["decode_tok_s"]
            / base_checkpoint["decode_tok_s"],
            "e2e_speedup": candidate_checkpoint["e2e_output_tok_s"]
            / base_checkpoint["e2e_output_tok_s"],
        }
    result["adaptive_grouped_smallm_vs_direct"] = comparison
    return result


def _teacher_forced_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("teacher-forced quality requires at least one prompt")
    all_steps = [step for row in rows for step in row["steps"]]
    if not all_steps:
        raise ValueError("teacher-forced quality requires at least one step")
    max_kl = max(float(step["kl_divergence"]) for step in all_steps)
    finite = all(bool(step["finite"]) for step in all_steps)
    matches = sum(bool(step["top1_agreement"]) for step in all_steps)
    top1_agreement = matches / len(all_steps)
    categories: dict[str, Any] = {}
    failed: list[str] = []
    if not finite:
        failed.append("nonfinite_logits")
    if not math.isfinite(max_kl) or max_kl > 0.05:
        failed.append("max_kl_above_0.05")
    if top1_agreement < 0.9:
        failed.append("suite_top1_below_0.9")
    for category in sorted({str(row["category"]) for row in rows}):
        category_steps = [
            step for row in rows if row["category"] == category for step in row["steps"]
        ]
        category_matches = sum(bool(step["top1_agreement"]) for step in category_steps)
        agreement = category_matches / len(category_steps)
        category_max_kl = max(float(step["kl_divergence"]) for step in category_steps)
        category_finite = all(bool(step["finite"]) for step in category_steps)
        categories[category] = {
            "steps": len(category_steps),
            "top1_matches": category_matches,
            "top1_agreement": agreement,
            "max_kl_divergence": category_max_kl,
            "finite": category_finite,
        }
        if agreement < 0.9:
            failed.append(f"{category}_top1_below_0.9")
        if not category_finite:
            failed.append(f"{category}_nonfinite_logits")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "steps": len(all_steps),
        "top1_matches": matches,
        "top1_agreement": top1_agreement,
        "max_kl_divergence": max_kl,
        "finite": finite,
        "categories": categories,
        "thresholds": {
            "max_kl_divergence": 0.05,
            "minimum_suite_top1_agreement": 0.9,
            "minimum_each_category_top1_agreement": 0.9,
        },
    }


def _copy_logits(
    session: LagunaGGUFResidentSession,
    result: Any,
    destination: np.ndarray,
) -> None:
    session.runtime.memcpy(
        host_array_ptr(destination),
        result.logits.ptr,
        destination.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )


def _teacher_forced_prompt(
    owner: LagunaGGUFResidentSession,
    prompt: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sessions = {mode: _session_for_mode(owner, args, mode) for mode in MODES}
    logits = {
        mode: np.empty(sessions[mode].config.vocab_size, dtype=np.float32)
        for mode in MODES
    }
    try:
        results = {
            mode: sessions[mode].prefill(prompt["token_ids"], use_bulk=True)
            for mode in MODES
        }
        steps = []
        for index in range(args.teacher_forced_tokens):
            for mode in MODES:
                _copy_logits(sessions[mode], results[mode], logits[mode])
            baseline_log_probs = _normalized_log_probs(logits[BASELINE_MODE])
            candidate_log_probs = _normalized_log_probs(logits[CANDIDATE_MODE])
            probabilities = np.exp(baseline_log_probs)
            kl = float(
                np.sum(probabilities * (baseline_log_probs - candidate_log_probs))
            )
            baseline_top1 = int(np.argmax(logits[BASELINE_MODE]))
            candidate_top1 = int(np.argmax(logits[CANDIDATE_MODE]))
            finite = bool(
                np.isfinite(logits[BASELINE_MODE]).all()
                and np.isfinite(logits[CANDIDATE_MODE]).all()
                and math.isfinite(kl)
            )
            steps.append(
                {
                    "index": index,
                    "teacher_token_id": baseline_top1,
                    "direct_top1": baseline_top1,
                    "adaptive_grouped_smallm_top1": candidate_top1,
                    "top1_agreement": baseline_top1 == candidate_top1,
                    "kl_divergence": kl,
                    "finite": finite,
                }
            )
            if index + 1 < args.teacher_forced_tokens:
                results = {
                    mode: sessions[mode].forward_token(baseline_top1)
                    for mode in MODES
                }
    finally:
        for session in sessions.values():
            session.close()
    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt_tokens": prompt["prompt_tokens"],
        "prompt_token_ids_sha256": prompt["token_ids_sha256"],
        "steps": steps,
    }


def _promotion_gate(
    aggregate: Mapping[str, Any],
    free_running: Mapping[str, Any],
    teacher_forced: Mapping[str, Any],
    oracle: Mapping[str, Any],
    shape_screen: Mapping[str, Any],
    *,
    horizons: Sequence[int],
    recovered: bool,
) -> dict[str, Any]:
    comparison = aggregate["adaptive_grouped_smallm_vs_direct"]
    failed: list[str] = []
    if not shape_screen["pass"]:
        failed.append("shape_screen_failed")
    if not teacher_forced["pass"]:
        failed.append("teacher_forced_quality_failed")
    if not oracle["pass"]:
        failed.append("poolside_oracle_failed")
    if not free_running["same_mode_repeat_deterministic"]:
        failed.append("free_running_repeat_not_deterministic")
    if not free_running["all_pairs_exact"]:
        failed.append("free_running_pairs_not_exact")
    if not recovered:
        failed.append("tracked_lifecycle_not_recovered")
    if float(comparison["prefill_speedup"]) <= 1.0:
        failed.append("aggregate_prefill_not_faster")
    for category, values in comparison["categories"].items():
        if float(values["prefill_speedup"]) <= 1.0:
            failed.append(f"{category}_prefill_regressed")
        for horizon in horizons:
            if float(values["horizons"][str(horizon)]["e2e_speedup"]) < 0.98:
                failed.append(f"{category}_h{horizon}_e2e_below_0.98")
    for horizon in horizons:
        values = comparison["horizons"][str(horizon)]
        if float(values["e2e_speedup"]) <= 1.0:
            failed.append(f"h{horizon}_aggregate_e2e_not_faster")
        decode_speedup = float(values["decode_speedup"])
        if not math.isfinite(decode_speedup) or not 0.98 <= decode_speedup <= 1.02:
            failed.append(f"h{horizon}_decode_outside_2pct")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "policy": {
            "shape_screen": (
                "direct fallback >=0.995x; rows>=32 grouped shapes and aggregate faster"
            ),
            "quality": "KL <= 0.05 and top-1 >= 90% suite-wide and per category",
            "free_running_ids": "all mode pairs exact and same-mode repeats deterministic",
            "performance": (
                "aggregate and every-category prefill faster; aggregate E2E faster; "
                "each-category E2E >= 0.98x; decode within 2%"
            ),
            "lifecycle": "tracked allocations return exactly to baseline",
        },
    }


def _load_shape_screen(args: argparse.Namespace) -> dict[str, Any]:
    artifact = json.loads(args.shape_screen.read_text(encoding="utf-8"))
    promotion = artifact.get("promotion", {})
    model = artifact.get("model", {})
    passed = bool(
        artifact.get("kind") == "hipengine_laguna_prefill_grouped_down_ab"
        and artifact.get("status") == "retained"
        and artifact.get("pass")
        and promotion.get("pass")
        and not promotion.get("regressed_rows")
        and not promotion.get("non_improving_grouped_rows")
        and model.get("sha256") == args.model_sha256
    )
    return {
        "pass": passed,
        "path": str(args.shape_screen.resolve()),
        "sha256": _sha256_bytes(args.shape_screen.read_bytes()),
        "revision": artifact.get("repo", {}).get("revision"),
        "aggregate_speedup": promotion.get("effective_speedup"),
        "grouped_min_rows": promotion.get("grouped_min_rows"),
        "model_sha256": model.get("sha256"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    horizons = tuple(int(value) for value in args.output_horizons)
    if horizons != RETAINED_HORIZONS:
        raise ValueError(
            f"retained grouped-down gate requires horizons {RETAINED_HORIZONS}"
        )
    if args.repetitions < 3:
        raise ValueError(
            "retained grouped-down gate requires at least three repetitions"
        )
    if args.chunk_size != 128:
        raise ValueError("retained grouped-down gate requires chunk size 128")
    if args.teacher_forced_tokens != max(RETAINED_HORIZONS):
        raise ValueError("retained grouped-down gate requires 32 teacher-forced tokens")
    if args.warmup_output_tokens <= 0:
        raise ValueError("warmup output tokens must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.shape_screen.is_file():
        raise FileNotFoundError(
            f"Laguna grouped-down shape screen not found: {args.shape_screen}"
        )
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError(
            "retained grouped-down category gate requires a clean tracked worktree"
        )
    shape_screen = _load_shape_screen(args)
    if not shape_screen["pass"]:
        raise ValueError("Laguna grouped-down shape screen is not accepted")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_grouped_down_category",
        timing_protocol=(
            "same_owner_balanced_direct_vs_adaptive_grouped_down_category_h16_h32"
        ),
        warmups=len(MODES),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
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
        for mode in MODES:
            _run_target_mode(
                owner,
                prompts[0],
                mode=mode,
                horizons=(int(args.warmup_output_tokens),),
                repetition=-1,
                args=args,
            )
        for repetition in range(args.repetitions):
            for prompt_index, prompt in enumerate(prompts):
                for mode in _mode_order(prompt_index, repetition):
                    row = _run_target_mode(
                        owner,
                        prompt,
                        mode=mode,
                        horizons=horizons,
                        repetition=repetition,
                        args=args,
                    )
                    rows.append(row)
                    print(
                        f"rep={repetition} prompt={prompt['id']} mode={mode} "
                        f"prefill={row['prefill_tok_s']:.3f} tok/s",
                        file=sys.stderr,
                        flush=True,
                    )
        for prompt in prompts:
            row = _teacher_forced_prompt(owner, prompt, args)
            teacher_rows.append(row)
            matches = sum(bool(step["top1_agreement"]) for step in row["steps"])
            max_kl = max(float(step["kl_divergence"]) for step in row["steps"])
            print(
                f"teacher prompt={prompt['id']} top1={matches}/{len(row['steps'])} "
                f"max_kl={max_kl:.6g}",
                file=sys.stderr,
                flush=True,
            )
        oracle = _oracle_gate(owner, args)
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during grouped-down category gate")

    free_running = _paired_free_running(rows, horizons)
    teacher_forced = _teacher_forced_quality(teacher_rows)
    aggregate = _aggregate(rows, horizons)
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    promotion = _promotion_gate(
        aggregate,
        free_running,
        teacher_forced,
        oracle,
        shape_screen,
        horizons=horizons,
        recovered=recovered,
    )
    passed = bool(promotion["pass"])
    manifest_path = args.repacked_cache / "manifest.json"
    prompt_payload = args.prompts.read_bytes()
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_grouped_down_category",
        "status": "retained_category_gate" if passed else "rejected_category_gate",
        "pass": passed,
        "performance_claim": passed,
        "performance_claim_scope": (
            "same-owner Laguna direct versus exact adaptive grouped-small-M down over "
            "all ten canonical category prompts at h16/h32; model load excluded"
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
            "modes": list(MODES),
            "chunk_size": args.chunk_size,
            "output_horizons": list(horizons),
            "repetitions": args.repetitions,
            "warmup_output_tokens_per_mode": args.warmup_output_tokens,
            "teacher_forced_tokens_per_prompt": args.teacher_forced_tokens,
            "timed_order": (
                "alternating direct/adaptive_grouped_smallm per prompt and reversed "
                "next repetition"
            ),
            "timing_scope": "prefill plus fixed-horizon decode; resident model load excluded",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "activation_quantization_included": False,
            "decode_route": "identical exact c=1 path for both modes",
        },
        "shape_screen": shape_screen,
        "load": {"seconds_excluded": load_seconds, "resident_nbytes": resident_nbytes},
        "rows": rows,
        "quality": {
            "poolside_oracle": oracle,
            "teacher_forced": teacher_forced,
            "teacher_forced_prompts": teacher_rows,
            "free_running": free_running,
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
            "Both modes share resident weights and use isolated bounded request sessions.",
            "Adaptive grouped down stays BF16 throughout and falls back to direct "
            "below 32 rows.",
            "Teacher forcing feeds direct-route top-1 IDs to both routes and compares "
            "full logits.",
            "Complete free-running IDs are reported, while KL/top-1 thresholds remain "
            "authoritative.",
            "AR decode uses the identical exact c=1 route in both modes.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
