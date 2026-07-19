#!/usr/bin/env python3
"""Benchmark GGUF continuation reuse against prefix-cache off.

Source preparation is outside the timed window.  The source may remain live or
be released before continuation admission.  Each continuation contains the
exact source prefix plus a non-empty suffix and runs through the production
resident runner in two scheduler chunks.  ``off`` privately processes both;
``radix`` shares the cached page, restores hybrid state, skips the prefix, and
executes only the suffix.  One warmup per mode is discarded and measured mode
order alternates across repetitions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_CORRECTNESS_ARTIFACT = Path(
    "benchmarks/results/2026-07-19-gfx1151-gguf-active-prefix-reuse-correctness.json"
)
_HARDWARE_LABELS = {
    "hip_gfx1100": "AMD Radeon Pro W7900 (gfx1100)",
    "hip_gfx1151": "AMD Radeon 8060S (gfx1151)",
}


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("distribution requires at least one sample")
    return {
        "samples": samples,
        "count": len(samples),
        "median": float(statistics.median(samples)),
        "p95": float(np.percentile(np.asarray(samples, dtype=np.float64), 95)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "stdev": float(statistics.pstdev(samples)),
    }


def _paired_delta_distribution(
    baseline: Sequence[Mapping[str, Any]],
    radix: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> dict[str, Any]:
    baseline_rows = list(baseline)
    radix_rows = list(radix)
    if not baseline_rows or len(baseline_rows) != len(radix_rows):
        raise ValueError("paired delta requires matched non-empty rows")
    summary = _distribution(
        [
            float(off[key]) - float(hit[key])
            for off, hit in zip(baseline_rows, radix_rows, strict=True)
        ]
    )
    summary["all_positive"] = all(value > 0.0 for value in summary["samples"])
    return summary


def _summarize_comparison(
    baseline: Sequence[Mapping[str, Any]],
    radix: Sequence[Mapping[str, Any]],
    *,
    prefix_tokens: int,
    source_lifecycle: str = "active",
) -> dict[str, Any]:
    baseline_rows = list(baseline)
    radix_rows = list(radix)
    if not baseline_rows or len(baseline_rows) != len(radix_rows):
        raise ValueError("comparison requires paired non-empty mode rows")
    baseline_ttft = _distribution(
        [float(row["continuation_ttft_ms"]) for row in baseline_rows]
    )
    radix_ttft = _distribution(
        [float(row["continuation_ttft_ms"]) for row in radix_rows]
    )
    baseline_median = float(baseline_ttft["median"])
    radix_median = float(radix_ttft["median"])
    speedup = baseline_median / radix_median if radix_median > 0.0 else math.inf
    reduction = (
        (baseline_median - radix_median) / baseline_median * 100.0
        if baseline_median > 0.0
        else -math.inf
    )
    correctness_exact = all(
        int(off["continuation_token_id"]) == int(hit["continuation_token_id"])
        for off, hit in zip(baseline_rows, radix_rows, strict=True)
    )
    radix_hits = sum(int(row["prefix_usable_hits_delta"]) for row in radix_rows)
    hit_rate = radix_hits / len(radix_rows)
    baseline_live_pages = int(
        round(statistics.median(int(row["refcounted_pages"]) for row in baseline_rows))
    )
    radix_live_pages = int(
        round(statistics.median(int(row["refcounted_pages"]) for row in radix_rows))
    )
    saved_live_pages = baseline_live_pages - radix_live_pages
    fallback_free = all(
        int(row["prefix_admission_fallbacks_delta"]) == 0 for row in radix_rows
    )
    reuse_exact = all(
        int(row["prefix_reused_tokens"]) == int(prefix_tokens) for row in radix_rows
    )
    final_drain = all(
        int(row["final_refcounted_pages"]) == 0
        for row in (*baseline_rows, *radix_rows)
    )
    if source_lifecycle == "active":
        snapshot_lifecycle_exact = True
        capacity_contract_exact = saved_live_pages >= 1
    elif source_lifecycle == "completed":
        snapshot_lifecycle_exact = all(
            bool(row.get("prefix_snapshot_hit"))
            and int(row.get("cache_refcount_after_source_release", -1)) == 1
            and int(row.get("cache_refcount_after_continuation_release", -1)) == 1
            and bool(row.get("snapshot_evicted"))
            for row in radix_rows
        )
        capacity_contract_exact = saved_live_pages == 0
    else:
        raise ValueError(f"unsupported source_lifecycle {source_lifecycle!r}")
    passed = bool(
        correctness_exact
        and hit_rate == 1.0
        and fallback_free
        and reuse_exact
        and capacity_contract_exact
        and snapshot_lifecycle_exact
        and final_drain
        and math.isfinite(speedup)
        and speedup > 1.0
    )
    return {
        "passed": passed,
        "correctness_exact": correctness_exact,
        "radix_hits": radix_hits,
        "radix_hit_rate": hit_rate,
        "fallback_free": fallback_free,
        "reuse_exact": reuse_exact,
        "final_drain": final_drain,
        "snapshot_lifecycle_exact": snapshot_lifecycle_exact,
        "capacity_contract_exact": capacity_contract_exact,
        "baseline_live_pages": baseline_live_pages,
        "radix_live_pages": radix_live_pages,
        "saved_live_pages": saved_live_pages,
        "baseline_ttft_ms": baseline_ttft,
        "radix_ttft_ms": radix_ttft,
        "ttft_speedup": speedup,
        "ttft_reduction_percent": reduction,
    }


def _correctness_prerequisite_matches(
    correctness: Mapping[str, Any],
    source_lifecycle: str,
) -> bool:
    if correctness.get("passed") is not True:
        return False
    if source_lifecycle == "active":
        return correctness.get("kind") == "gguf_active_prefix_reuse_correctness_gate"
    if source_lifecycle == "completed":
        return bool(
            correctness.get("kind")
            == "gguf_completed_prefix_reuse_correctness_gate"
            and correctness.get("workload", {}).get("source_lifecycle")
            == "completed"
        )
    raise ValueError(f"unsupported source_lifecycle {source_lifecycle!r}")


def _request(prompt: tuple[int, ...], *, max_tokens: int) -> Any:
    from hipengine.generation.registry import GenerationRequest

    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def _prefill_work(request_id: int, tokens: tuple[int, ...]) -> Any:
    from hipengine.dispatch import WorkItem, WorkKind

    return WorkItem(
        kind=WorkKind.PREFILL,
        request_ids=(int(request_id),),
        row_to_request=(int(request_id),),
        token_rows=(tokens,),
    )


def _counter(snapshot: Mapping[str, Any], name: str) -> int:
    return int(snapshot.get("prefix_cache", {}).get(name, 0))


def _run_case(
    runner: Any,
    base_config: Any,
    *,
    mode: str,
    prefix: tuple[int, ...],
    suffix: tuple[int, ...],
    request_id_base: int,
    source_lifecycle: str = "active",
) -> dict[str, Any]:
    source_id = int(request_id_base)
    continuation_id = source_id + 1
    source_state = SimpleNamespace(request_id=source_id)
    continuation_state = SimpleNamespace(request_id=continuation_id)
    continued_prompt = (*prefix, *suffix)
    runner.configure_engine_loop(replace(base_config, prefix_cache=str(mode)))
    runtime = runner._shared_runner.runtime
    pool = runner.kv_pool
    if runtime is None or pool is None:
        raise RuntimeError("GGUF prefix benchmark requires a live HIP runtime/device KV pool")
    source_released = False
    cache_refcount_after_source_release = 0
    source_session_reset_before_admission = False
    try:
        source_request = _request(prefix, max_tokens=2)
        runner.register_batch((source_id,), source_request, prompt_rows=(prefix,))
        runner.reserve_admission(source_state)
        runner.prefill_batch(_prefill_work(source_id, prefix), commit=True)
        source_row = runner._rows[source_id]
        if (
            source_row.slot is None
            or source_row.lease is None
            or source_row.kv_allocation is None
        ):
            raise RuntimeError("GGUF prefix benchmark source did not become resident")
        source_token_id = int(source_row.slot.prev_token)
        source_session = source_row.lease.session
        source_block_ids = tuple(
            int(block_id) for block_id in source_row.kv_allocation.block_ids
        )
        runtime.device_synchronize()
        if source_lifecycle == "completed":
            if mode == "radix":
                runner._release_row_resources(
                    source_row,
                    retain_prefix_snapshots=True,
                )
                runner._rows.pop(source_id)
                cache_refcount_after_source_release = int(
                    pool.refcount(source_block_ids[0])
                )
            else:
                runner.discard((source_id,))
            source_released = True
            source_session_reset_before_admission = bool(
                int(source_session.position) == 0
                and source_session.device_kv_allocation is None
                and any(lease.session is source_session for lease in runner._available)
            )
            runtime.device_synchronize()

        before_prefix = runner.observability_snapshot()
        memory_before = runner.kv_pool_memory_snapshot()
        continuation_request = _request(continued_prompt, max_tokens=1)
        runner.register_batch(
            (continuation_id,),
            continuation_request,
            prompt_rows=(continued_prompt,),
        )

        runtime.device_synchronize()
        total_start = time.perf_counter()
        admission_start = total_start
        runner.reserve_admission(continuation_state)
        runtime.device_synchronize()
        admission_end = time.perf_counter()
        runner.prefill_batch(_prefill_work(continuation_id, prefix), commit=True)
        runner.prefill_batch(_prefill_work(continuation_id, suffix), commit=True)
        runtime.device_synchronize()
        prefill_end = time.perf_counter()

        continuation_row = runner._rows[continuation_id]
        if continuation_row.slot is None or continuation_row.kv_allocation is None:
            raise RuntimeError("GGUF prefix benchmark continuation did not finish")
        after_prefix = runner.observability_snapshot()
        memory_after = runner.kv_pool_memory_snapshot()
        pool_after = pool.stats.to_json_dict()
        allocation = continuation_row.kv_allocation
        row = {
            "mode": str(mode),
            "source_request_id": source_id,
            "continuation_request_id": continuation_id,
            "source_token_id": source_token_id,
            "continuation_token_id": int(continuation_row.slot.prev_token),
            "source_lifecycle": source_lifecycle,
            "source_session_reset_before_admission": (
                source_session_reset_before_admission
            ),
            "admission_ms": (admission_end - admission_start) * 1000.0,
            "prefill_to_first_token_ms": (prefill_end - admission_end) * 1000.0,
            "continuation_ttft_ms": (prefill_end - total_start) * 1000.0,
            "prefix_usable_hits_delta": (
                _counter(after_prefix, "usable_hits")
                - _counter(before_prefix, "usable_hits")
            ),
            "prefix_admission_fallbacks_delta": (
                _counter(after_prefix, "admission_fallbacks")
                - _counter(before_prefix, "admission_fallbacks")
            ),
            "prefix_reused_tokens": int(continuation_row.prefix_reused_tokens),
            "prefix_state_clone_bytes": int(continuation_row.prefix_state_clone_bytes),
            "prefix_snapshot_hit": bool(continuation_row.prefix_snapshot_hit),
            "prefix_snapshot_hits_delta": (
                _counter(after_prefix, "snapshot_hits")
                - _counter(before_prefix, "snapshot_hits")
            ),
            "snapshot_entries": _counter(after_prefix, "snapshot_entries"),
            "snapshot_bytes": _counter(after_prefix, "snapshot_bytes"),
            "cache_refcount_after_source_release": (
                cache_refcount_after_source_release
            ),
            "continuation_block_ids": [int(block_id) for block_id in allocation.block_ids],
            "continuation_reused_block_ids": [
                int(block_id) for block_id in allocation.reused_block_ids
            ],
            "page_bytes": int(pool.page_bytes),
            "current_pages": int(pool_after["current_pages"]),
            "current_pool_bytes": int(pool_after["current_bytes"]),
            "refcounted_pages": int(pool_after["refcounted_pages"]),
            "free_pages": int(pool_after["free_pages"]),
            "tracked_current_before_bytes": int(
                memory_before["tracked_allocator"]["current_allocated_bytes"]
            ),
            "tracked_current_after_bytes": int(
                memory_after["tracked_allocator"]["current_allocated_bytes"]
            ),
            "hip_used_before_bytes": int(memory_before["hip_used_current_bytes"]),
            "hip_used_after_bytes": int(memory_after["hip_used_current_bytes"]),
            "hip_used_peak_sampled_bytes": int(
                memory_after["hip_used_peak_sampled_bytes"]
            ),
            "route_counts": after_prefix.get("routes", {}).get("counts", {}),
        }
        row["cache_resident_bytes"] = int(row["snapshot_bytes"]) + (
            int(pool.page_bytes)
            if int(row["cache_refcount_after_source_release"]) > 0
            else 0
        )

        if not source_released:
            runner.discard((source_id,))
            source_released = True
            cache_refcount_after_source_release = (
                0
                if not allocation.reused_block_ids
                else int(pool.refcount(allocation.reused_block_ids[0]))
            )
            row["cache_refcount_after_source_release"] = (
                cache_refcount_after_source_release
            )
        runner.discard((continuation_id,))
        row["cache_refcount_after_continuation_release"] = (
            0
            if not allocation.reused_block_ids
            else int(pool.refcount(allocation.reused_block_ids[0]))
        )
        row["snapshot_evicted"] = False
        if source_lifecycle == "completed" and mode == "radix":
            row["snapshot_evicted"] = bool(
                runner._evict_prefix_snapshot(prefix)
            )
        row["final_refcounted_pages"] = int(pool.stats.refcounted_pages)
        return row
    finally:
        runner.discard((source_id, continuation_id))
        runner._clear_prefix_snapshots()


def _repo_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {
        "revision": revision,
        "tracked_clean": not status,
        "tracked_status": status,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine import LLM

    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    correctness_path = args.correctness_artifact.expanduser().resolve()
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    source_lifecycle = str(args.source_lifecycle)
    if not _correctness_prerequisite_matches(correctness, source_lifecycle):
        raise RuntimeError(
            "prefix benchmark correctness prerequisite does not match source lifecycle"
        )
    prefix_tokens = int(args.prefix_tokens)
    suffix_tokens = int(args.suffix_tokens)
    warmups = int(args.warmups)
    repetitions = int(args.repetitions)
    if prefix_tokens <= 0 or prefix_tokens % 256:
        raise ValueError("prefix_tokens must be a positive multiple of 256")
    if suffix_tokens <= 0 or warmups < 0 or repetitions <= 0:
        raise ValueError("suffix_tokens/repetitions must be positive and warmups non-negative")
    prefix = (int(args.prefix_token_id),) * prefix_tokens
    suffix = (int(args.suffix_token_id),) * suffix_tokens
    required_positions = prefix_tokens + suffix_tokens
    if required_positions > int(args.max_sequence_length):
        raise ValueError("max_sequence_length does not cover the continuation")

    repo = _repo_state()
    llm = LLM(
        str(model),
        backend=str(args.backend),
        quant=str(args.quant),
        max_active_requests=3,
        prefix_cache="off",
    )
    warmup_rows: list[dict[str, Any]] = []
    measured: dict[str, list[dict[str, Any]]] = {"off": [], "radix": []}
    try:
        llm.prepare(max_sequence_length=int(args.max_sequence_length))
        wrapper = llm._get_text_generator()
        runner = wrapper._runner
        base_config = wrapper._loop.config
        case_index = 0
        for _ in range(warmups):
            for mode in ("off", "radix"):
                warmup_rows.append(
                    _run_case(
                        runner,
                        base_config,
                        mode=mode,
                        prefix=prefix,
                        suffix=suffix,
                        request_id_base=10_000 + case_index * 10,
                        source_lifecycle=source_lifecycle,
                    )
                )
                case_index += 1
        measured_order: list[str] = []
        for repetition in range(repetitions):
            order = ("off", "radix") if repetition % 2 == 0 else ("radix", "off")
            for mode in order:
                measured_order.append(mode)
                measured[mode].append(
                    _run_case(
                        runner,
                        base_config,
                        mode=mode,
                        prefix=prefix,
                        suffix=suffix,
                        request_id_base=20_000 + case_index * 10,
                        source_lifecycle=source_lifecycle,
                    )
                )
                case_index += 1
    finally:
        llm.close()

    comparison = _summarize_comparison(
        measured["off"],
        measured["radix"],
        prefix_tokens=prefix_tokens,
        source_lifecycle=source_lifecycle,
    )
    page_bytes = int(measured["radix"][0]["page_bytes"])
    saved_live_bytes = int(comparison["saved_live_pages"]) * page_bytes
    timing = {
        mode: {
            "admission_ms": _distribution([row["admission_ms"] for row in rows]),
            "prefill_to_first_token_ms": _distribution(
                [row["prefill_to_first_token_ms"] for row in rows]
            ),
            "continuation_ttft_ms": _distribution(
                [row["continuation_ttft_ms"] for row in rows]
            ),
        }
        for mode, rows in measured.items()
    }
    memory = {
        mode: {
            "refcounted_pages": _distribution(
                [float(row["refcounted_pages"]) for row in rows]
            ),
            "snapshot_entries": _distribution(
                [float(row["snapshot_entries"]) for row in rows]
            ),
            "snapshot_bytes": _distribution(
                [float(row["snapshot_bytes"]) for row in rows]
            ),
            "cache_resident_bytes": _distribution(
                [float(row["cache_resident_bytes"]) for row in rows]
            ),
            "tracked_current_before_bytes": _distribution(
                [float(row["tracked_current_before_bytes"]) for row in rows]
            ),
            "tracked_current_after_bytes": _distribution(
                [float(row["tracked_current_after_bytes"]) for row in rows]
            ),
            "hip_used_after_bytes": _distribution(
                [float(row["hip_used_after_bytes"]) for row in rows]
            ),
            "hip_used_peak_sampled_bytes": _distribution(
                [float(row["hip_used_peak_sampled_bytes"]) for row in rows]
            ),
        }
        for mode, rows in measured.items()
    }
    tracked_current_paired_delta = _paired_delta_distribution(
        measured["off"],
        measured["radix"],
        key="tracked_current_after_bytes",
    )
    hip_current_paired_delta = _paired_delta_distribution(
        measured["off"],
        measured["radix"],
        key="hip_used_after_bytes",
    )
    physical_current_reduction_claim = bool(
        hip_current_paired_delta["all_positive"]
        and float(hip_current_paired_delta["median"]) > 0.0
    )
    passed = bool(comparison["passed"] and repo["tracked_clean"])
    return {
        "schema": 1,
        "kind": (
            "gguf_active_prefix_reuse_economics"
            if source_lifecycle == "active"
            else "gguf_completed_prefix_reuse_economics"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "correctness_claim": True,
        "performance_claim": passed,
        "model": str(model),
        "quant": str(args.quant),
        "backend": str(args.backend),
        "hardware": str(
            args.hardware_label
            or _HARDWARE_LABELS.get(str(args.backend), str(args.backend))
        ),
        "repo": repo,
        "correctness_prerequisite": {
            "path": str(correctness_path),
            "kind": correctness.get("kind"),
            "passed": correctness.get("passed"),
            "teacher_forced_kl_mean": correctness.get("teacher_forced", {}).get("kl_mean"),
            "teacher_forced_top1_agreement": correctness.get("teacher_forced", {}).get("top1_agreement"),
            "initial_state_exact": correctness.get("prefill_oracle", {}).get("initial_state_exact"),
            "final_state_exact": correctness.get("teacher_forced", {}).get("final_state_exact"),
        },
        "workload": {
            "prefix_token_id": int(args.prefix_token_id),
            "prefix_tokens": prefix_tokens,
            "suffix_token_id": int(args.suffix_token_id),
            "suffix_tokens": suffix_tokens,
            "max_sequence_length": int(args.max_sequence_length),
            "max_active_requests": 3,
            "prefill_chunks": [prefix_tokens, suffix_tokens],
            "sampling": "greedy_top1_ignore_eos",
            "kv_dtype": "bf16",
            "source_lifecycle": source_lifecycle,
            "warmups_per_mode": warmups,
            "measured_repetitions_per_mode": repetitions,
            "measured_order": measured_order,
        },
        "timing_protocol": {
            "scope": (
                "already-live source; synchronized continuation admission through first token"
                if source_lifecycle == "active"
                else "completed cache-ready source; synchronized continuation admission through first token"
            ),
            "source_prefill_timed": False,
            "mode_reconfiguration_timed": False,
            "synchronize": "HIP device synchronize before/after admission and after suffix first-token work",
            "alternating_order": True,
        },
        "timing": timing,
        "comparison": {
            **comparison,
            "saved_live_bytes": saved_live_bytes,
        },
        "memory": {
            **memory,
            "page_bytes": page_bytes,
            "saved_live_pages": int(comparison["saved_live_pages"]),
            "saved_live_bytes": saved_live_bytes,
            "tracked_current_paired_delta_bytes": tracked_current_paired_delta,
            "hip_used_current_paired_delta_bytes": hip_current_paired_delta,
            "physical_current_reduction_claim": physical_current_reduction_claim,
            "scope_note": (
                "Default fixed-capacity pool backing is preallocated for both modes. "
                + (
                    "Live-page headroom is the retained memory benefit unless HIP current also falls."
                    if source_lifecycle == "active"
                    else "Completed reuse trades a retained hybrid snapshot plus one idle cache page for TTFT; during continuation its unique live-page count must equal off."
                )
            ),
        },
        "warmup_rows": warmup_rows,
        "measured_rows": measured,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "proc_cmdline": Path("/proc/cmdline").read_text(encoding="utf-8").strip(),
            "HIPENGINE_COMPILER_VERSION_FILE": os.environ.get(
                "HIPENGINE_COMPILER_VERSION_FILE"
            ),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
        "notes": [
            (
                "The source prefill is outside continuation TTFT: reuse economics begin at a cacheable live source."
                if source_lifecycle == "active"
                else "Source prefill, snapshot capture, and normal cache promotion are outside continuation TTFT; the source session is reset before timing."
            ),
            "No prompt-conditioned runtime branch or expected-token rerank is used.",
            "Snapshot bytes, live pages, and process/GPU-visible current bytes are reported separately.",
            (
                "The first production slice is active-current, greedy, BF16-KV, and non-empty-suffix only."
                if source_lifecycle == "active"
                else "This slice is completed-source, greedy, BF16-KV, positive aligned prefix, and non-empty suffix only."
            ),
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        choices=("hip_gfx1100", "hip_gfx1151"),
        default="hip_gfx1151",
    )
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prefix-token-id", type=int, default=9707)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-token-id", type=int, default=9708)
    parser.add_argument("--suffix-tokens", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--source-lifecycle",
        choices=("active", "completed"),
        default="active",
    )
    parser.add_argument(
        "--correctness-artifact",
        type=Path,
        default=DEFAULT_CORRECTNESS_ARTIFACT,
    )
    parser.add_argument("--hardware-label")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    command_args = list(sys.argv[1:] if argv is None else argv)
    payload["command"] = shlex.join(
        [sys.executable, "scripts/gguf_prefix_reuse_bench.py", *command_args]
    )
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
