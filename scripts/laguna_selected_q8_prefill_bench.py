#!/usr/bin/env python3
"""Screen inclusive Laguna selected-Q4 Q8_1/dp4a prefill."""

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
MODES = ("split", "q8_dp4a")
BASELINE_MODE = MODES[0]
CANDIDATE_MODE = MODES[1]
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-ar-o1-q8-dp4a-screen.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(shape_index: int, repetition: int) -> tuple[str, str]:
    """Counterbalance mode order by shape and repetition."""

    return MODES if (int(shape_index) + int(repetition)) % 2 == 0 else tuple(reversed(MODES))


def _comparison_summary(
    samples: Mapping[str, Mapping[int, Sequence[float]]],
    next_tokens: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    rows: Sequence[int] = PROFILE_ROWS,
) -> dict[str, Any]:
    parsed_rows = tuple(int(value) for value in rows)
    if not parsed_rows or tuple(sorted(set(parsed_rows))) != parsed_rows:
        raise ValueError("Laguna selected Q8 rows must be sorted and distinct")
    if set(samples) != set(MODES) or set(next_tokens) != set(MODES):
        raise ValueError(f"Laguna selected Q8 comparison requires modes {MODES}")

    shapes: dict[str, Any] = {}
    all_ids_agree = True
    all_shapes_faster = True
    baseline_seconds = 0.0
    candidate_seconds = 0.0
    timed_tokens = 0
    for value in parsed_rows:
        summaries: dict[str, Any] = {}
        ids: dict[str, list[int]] = {}
        counts: set[int] = set()
        for mode in MODES:
            mode_samples = [float(item) for item in samples[mode][value]]
            mode_ids = [int(item) for item in next_tokens[mode][value]]
            if len(mode_samples) != len(mode_ids):
                raise ValueError(f"rows={value} mode={mode} timing/token counts differ")
            counts.add(len(mode_samples))
            summaries[mode] = _summarize_timing_samples(mode_samples, rows=value)
            ids[mode] = mode_ids
        if counts == {0} or len(counts) != 1:
            raise ValueError(f"rows={value} modes require equal non-empty sample counts")
        ids_agree = len(set(ids[BASELINE_MODE] + ids[CANDIDATE_MODE])) == 1
        speedup = (
            summaries[BASELINE_MODE]["median_seconds"]
            / summaries[CANDIDATE_MODE]["median_seconds"]
        )
        all_ids_agree = all_ids_agree and ids_agree
        all_shapes_faster = all_shapes_faster and speedup > 1.0
        baseline_seconds += sum(float(item) for item in samples[BASELINE_MODE][value])
        candidate_seconds += sum(float(item) for item in samples[CANDIDATE_MODE][value])
        timed_tokens += value * len(samples[BASELINE_MODE][value])
        shapes[str(value)] = {
            "rows": value,
            BASELINE_MODE: summaries[BASELINE_MODE],
            CANDIDATE_MODE: summaries[CANDIDATE_MODE],
            "q8_dp4a_vs_split_speedup": speedup,
            "next_token_ids": ids,
            "next_token_agreement": ids_agree,
        }

    aggregate_speedup = baseline_seconds / candidate_seconds
    failed: list[str] = []
    if not all_shapes_faster:
        failed.append("candidate_not_faster_at_every_shape")
    if not math.isfinite(aggregate_speedup) or aggregate_speedup <= 1.0:
        failed.append("aggregate_candidate_not_faster")
    return {
        "shapes": shapes,
        "next_token_diagnostic": {
            "all_modes_and_repetitions_agree": all_ids_agree,
        },
        "performance_screen": {
            "pass": not failed,
            "failed_checks": failed,
            "all_measured_shapes_strictly_faster": all_shapes_faster,
            "timed_tokens_per_mode": timed_tokens,
            "split_seconds": baseline_seconds,
            "q8_dp4a_seconds": candidate_seconds,
            "aggregate_speedup": aggregate_speedup,
            "quality_gate_required_for_promotion": True,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repetitions < 3:
        raise ValueError("retained selected Q8 screen requires at least three repetitions")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    if max(PROFILE_ROWS) > args.context_length:
        raise ValueError("largest selected Q8 row shape exceeds admitted context")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained selected Q8 screen requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_ar_o1_q8_dp4a_screen",
        timing_protocol="same_session_balanced_split_vs_inclusive_q8_dp4a",
        warmups=args.warmups * len(PROFILE_ROWS) * len(MODES),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(PROFILE_ROWS))

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    samples = {mode: {value: [] for value in PROFILE_ROWS} for mode in MODES}
    next_tokens = {mode: {value: [] for value in PROFILE_ROWS} for mode in MODES}
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
            prefill_chunk_size=max(PROFILE_ROWS),
            selected_gate_up_mode=BASELINE_MODE,
        )
        load_seconds = time.perf_counter() - load_started
        for warmup in range(args.warmups):
            for shape_index, value in enumerate(PROFILE_ROWS):
                for mode in _mode_order(shape_index, warmup):
                    owner.reset_state()
                    owner.set_selected_gate_up_mode(mode)
                    owner.prefill(token_stream[:value], use_bulk=True)
        for repetition in range(args.repetitions):
            for shape_index, value in enumerate(PROFILE_ROWS):
                for mode in _mode_order(shape_index, repetition):
                    owner.reset_state()
                    owner.set_selected_gate_up_mode(mode)
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
        raise RuntimeError("HIP total memory changed during selected Q8 screen")

    comparison = _comparison_summary(samples, next_tokens)
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    passed = bool(comparison["performance_screen"]["pass"] and recovered)
    prompts_payload = args.prompts.read_bytes()
    manifest_path = args.repacked_cache / "manifest.json"
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_ar_o1_q8_dp4a_screen",
        "status": "screen_pass" if passed else "rejected_performance",
        "pass": passed,
        "performance_claim": False,
        "performance_claim_scope": None,
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
            "rows": list(PROFILE_ROWS),
            "modes": list(MODES),
            "one_physical_chunk": True,
            "prefill_chunk_size": max(PROFILE_ROWS),
            "context_length": args.context_length,
            "repetitions": args.repetitions,
            "warmups_per_shape_mode": args.warmups,
            "timed_order": "counterbalanced by shape and reversed each repetition",
            "timing_scope": "reset complete through synchronized first-token projection",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompts_payload),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
            "activation_quantization_included": True,
            "activation_quantization_rows": "producer rows before top-k expansion",
        },
        "load": {"seconds_excluded": load_seconds, "resident_nbytes": resident_nbytes},
        "comparison": comparison,
        "correctness": {
            "quality_gate_status": "not_run_by_screen",
            "next_token_diagnostic": comparison["next_token_diagnostic"],
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
            "Both routes use the same resident weights, scratch, token stream, and session.",
            "Candidate timing includes one GGML Q8_1 activation quantization per producer row per sparse layer.",
            "AR decode remains on the exact split route in both modes.",
            "A screen pass is not promotion; full Poolside/category quality gates remain mandatory.",
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
