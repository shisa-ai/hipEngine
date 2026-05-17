#!/usr/bin/env python3
"""Resident Qwen3.5 GGUF c=1 benchmark harness.

The harness measures the public GGUF resident execution surface directly: a
single persistent ``Qwen35GGUFResidentSession`` per run, token-serial prefill,
one optional warmup decode token, and one-step HIP graph replay for measured
decode.  It is intentionally shape-driven so the retained artifacts can compare
512/128 and 4K/128 against PARO resident diagnostics and llama.cpp GGUF rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

DEFAULT_MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--token-id", type=int, default=9707, help="Repeated token id for fixed-length prompt")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument(
        "--graph-replay-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one-step HIP graph replay for measured decode (default).",
    )
    parser.add_argument("--graph-steps-per-replay", type=int, default=1)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=None,
        help="Read precomputed hipcc --version text so profiled/bench runs do not spawn hipcc.",
    )
    parser.add_argument(
        "--require-cached-build",
        action="store_true",
        help="Fail instead of rebuilding resident runtime/lm-head HIP libraries.",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be positive")
    if args.decode_tokens < 0 or args.warmup_decode_tokens < 0:
        raise ValueError("decode token counts must be non-negative")
    if args.warmup_runs < 0 or args.measured_runs <= 0:
        raise ValueError("--warmup-runs must be >=0 and --measured-runs must be positive")
    if args.graph_steps_per_replay <= 0:
        raise ValueError("--graph-steps-per-replay must be positive")
    if args.graph_replay_decode and args.decode_tokens % args.graph_steps_per_replay != 0:
        raise ValueError("--decode-tokens must be divisible by --graph-steps-per-replay")

    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file else None
    prompt_tokens = [int(args.token_id)] * int(args.prompt_length)
    max_sequence_length = len(prompt_tokens) + args.warmup_decode_tokens + args.decode_tokens + 1

    runs: list[dict[str, Any]] = []
    for run_index in range(args.warmup_runs + args.measured_runs):
        measured = run_index >= args.warmup_runs
        run = _run_once(
            model=args.model,
            quant=args.quant,
            prompt_tokens=prompt_tokens,
            decode_tokens=args.decode_tokens,
            warmup_decode_tokens=args.warmup_decode_tokens,
            max_sequence_length=max_sequence_length,
            graph_replay_decode=args.graph_replay_decode,
            graph_steps_per_replay=args.graph_steps_per_replay,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            measured=measured,
            run_index=(run_index - args.warmup_runs + 1 if measured else run_index + 1),
        )
        runs.append(run)
        label = "measured" if measured else "warmup"
        print(
            f"{label}_run={run['run_index']} prefill_tok_s={run['throughput']['prefill_tok_s']:.6f} "
            f"decode_tok_s={run['throughput']['decode_tok_s']:.6f} "
            f"peak_gib={run['memory']['tracked_peak_allocated_gib']:.6f}",
            file=sys.stderr,
            flush=True,
        )

    measured_runs = [run for run in runs if run["measured"]]
    output = {
        "schema": 1,
        "model": str(args.model),
        "quant": args.quant,
        "backend": "hip_gfx1100",
        "mode": "resident_token_serial_prefill_graph_decode",
        "prompt_source": "repeated_token_id",
        "token_id": int(args.token_id),
        "prompt_length": int(args.prompt_length),
        "decode_tokens": int(args.decode_tokens),
        "warmup_decode_tokens": int(args.warmup_decode_tokens),
        "warmup_runs": int(args.warmup_runs),
        "measured_runs": int(args.measured_runs),
        "max_sequence_length": int(max_sequence_length),
        "graph_replay_decode": bool(args.graph_replay_decode),
        "graph_steps_per_replay": int(args.graph_steps_per_replay if args.graph_replay_decode else 0),
        "require_cached_build": bool(args.require_cached_build),
        "compiler_version_file": None if args.compiler_version_file is None else str(args.compiler_version_file),
        "compiler_version_first_line": None if compiler_version is None else compiler_version.splitlines()[0],
        "runs": runs,
        "summary": _summary(measured_runs),
        "notes": [
            "Prefill is currently token-serial through Qwen35GGUFResidentSession.prefill(); this benchmark is not a promoted throughput row.",
            "Measured decode excludes graph capture time when graph_replay_decode=true.",
        ],
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


def _run_once(
    *,
    model: Path,
    quant: str,
    prompt_tokens: list[int],
    decode_tokens: int,
    warmup_decode_tokens: int,
    max_sequence_length: int,
    graph_replay_decode: bool,
    graph_steps_per_replay: int,
    compiler_version: str | None,
    require_cached_build: bool,
    measured: bool,
    run_index: int,
) -> dict[str, Any]:
    runtime = get_hip_runtime()
    reset_memory_stats()
    memory_snapshots: dict[str, Any] = {"before_load": _memory_snapshot("before_load", runtime)}
    load_start = time.perf_counter()
    session = Qwen35GGUFResidentSession(
        model,
        runtime=runtime,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        max_sequence_length=max_sequence_length,
    )
    load_seconds = time.perf_counter() - load_start
    memory_snapshots["after_load"] = _memory_snapshot("after_load", runtime, session)

    generated_token_ids: list[int] = []
    final = None
    graph_capture_seconds = 0.0
    try:
        prefill_start = time.perf_counter()
        first = session.prefill(prompt_tokens, return_logits=False)
        prefill_seconds = time.perf_counter() - prefill_start
        generated_token_ids.append(first.token_id)
        next_token = first.token_id
        memory_snapshots["after_prefill"] = _memory_snapshot("after_prefill", runtime, session)

        warmup_start = time.perf_counter()
        for _ in range(warmup_decode_tokens):
            warmup = session.step(next_token)
            next_token = warmup.token_id
            generated_token_ids.append(warmup.token_id)
        warmup_decode_seconds = time.perf_counter() - warmup_start
        memory_snapshots["after_warmup_decode"] = _memory_snapshot("after_warmup_decode", runtime, session)

        if graph_replay_decode and decode_tokens:
            capture_start = time.perf_counter()
            graph = session.capture_decode_graph(
                position=session.position,
                steps_per_replay=graph_steps_per_replay,
                max_replay_steps=decode_tokens,
                record_steps=decode_tokens,
            )
            graph_capture_seconds = time.perf_counter() - capture_start
            try:
                decode_start = time.perf_counter()
                graph.replay(decode_tokens)
                decode_seconds = time.perf_counter() - decode_start
                generated_token_ids.extend(graph.read_generated_token_ids(decode_tokens))
                final = graph.read_sample()
            finally:
                graph.close()
        else:
            decode_start = time.perf_counter()
            for _ in range(decode_tokens):
                final = session.step(next_token)
                next_token = final.token_id
                generated_token_ids.append(next_token)
            decode_seconds = time.perf_counter() - decode_start
        memory_snapshots["after_decode"] = _memory_snapshot("after_decode", runtime, session)
        final_token_id = None if final is None else final.token_id
        final_logit = None if final is None else final.logit
        finite_logits = None if final is None else bool(np.all(np.isfinite(final.logits)))
    finally:
        memory_snapshots["before_close"] = _memory_snapshot("before_close", runtime, session)
        session.close()
        memory_snapshots["after_close"] = _memory_snapshot("after_close", runtime)

    return {
        "run_index": int(run_index),
        "measured": bool(measured),
        "model": str(model),
        "quant": quant,
        "prompt_length": len(prompt_tokens),
        "decode_tokens": int(decode_tokens),
        "warmup_decode_tokens": int(warmup_decode_tokens),
        "timings": {
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "warmup_decode_seconds": warmup_decode_seconds,
            "graph_capture_seconds": graph_capture_seconds,
            "decode_seconds_excluding_graph_capture": decode_seconds,
            "wall_seconds_excluding_load": prefill_seconds + warmup_decode_seconds + graph_capture_seconds + decode_seconds,
        },
        "throughput": {
            "prefill_tok_s": len(prompt_tokens) / prefill_seconds if prefill_seconds else None,
            "decode_tok_s": decode_tokens / decode_seconds if decode_seconds else None,
            "decode_ms_per_token": (decode_seconds / decode_tokens) * 1000.0 if decode_tokens else None,
        },
        "correctness_sanity": {
            "finite_final_logits": finite_logits,
            "final_token_id": final_token_id,
            "final_logit": final_logit,
            "generated_preview_token_ids": generated_token_ids[:16],
            "generated_tail_token_ids": generated_token_ids[-16:],
            "generated_count_including_prefill_sample_and_warmup": len(generated_token_ids),
        },
        "memory": _memory_summary(memory_snapshots),
        "memory_snapshots": memory_snapshots,
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prefill_tok_s": _stats([run["throughput"]["prefill_tok_s"] for run in runs]),
        "decode_tok_s": _stats([run["throughput"]["decode_tok_s"] for run in runs]),
        "prefill_seconds": _stats([run["timings"]["prefill_seconds"] for run in runs]),
        "decode_seconds": _stats([run["timings"]["decode_seconds_excluding_graph_capture"] for run in runs]),
        "graph_capture_seconds": _stats([run["timings"]["graph_capture_seconds"] for run in runs]),
        "tracked_peak_allocated_gib": _stats([run["memory"]["tracked_peak_allocated_gib"] for run in runs]),
        "tracked_current_allocated_gib_before_close": _stats(
            [run["memory"]["tracked_current_allocated_gib_before_close"] for run in runs]
        ),
        "owned_session_peak_gib": _stats([run["memory"]["owned_session_peak_gib"] for run in runs]),
        "hip_used_peak_sampled_gib": _stats([run["memory"].get("hip_used_peak_sampled_gib") for run in runs]),
        "finite_final_logits_all": all(bool(run["correctness_sanity"]["finite_final_logits"]) for run in runs),
        "final_token_ids": [run["correctness_sanity"]["final_token_id"] for run in runs],
    }


def _stats(values: list[Any]) -> dict[str, Any]:
    samples = [float(value) for value in values if value is not None]
    if not samples:
        return {"samples": [], "median": None, "p95": None, "min": None, "max": None, "stdev": None}
    sorted_samples = sorted(samples)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) >= 2 else 0.0
    return {
        "samples": samples,
        "median": median,
        "p95": sorted_samples[min(len(sorted_samples) - 1, int(0.95 * (len(sorted_samples) - 1)))],
        "min": min(samples),
        "max": max(samples),
        "stdev": stdev,
        "stdev_pct_of_median": None if median == 0 else 100.0 * stdev / median,
    }


def _memory_snapshot(label: str, runtime: HipRuntime, session: Qwen35GGUFResidentSession | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": label,
        "tracked": memory_stats(),
        "hip": _hip_memory_info(runtime),
    }
    if session is not None:
        payload["owned_session_bytes"] = _owned_device_bytes(session)
        payload["owned_session_gib"] = _bytes_to_gib(payload["owned_session_bytes"])
        if session.scratch is not None:
            payload["scratch_max_positions"] = int(session.scratch.max_positions)
            payload["scratch_block_table_len"] = int(session.scratch.block_table_tensor.numel)
    return payload


def _memory_summary(snapshots: dict[str, Any]) -> dict[str, Any]:
    tracked_peak = max(
        int(snapshot.get("tracked", {}).get("peak_allocated_bytes", 0)) for snapshot in snapshots.values()
    ) if snapshots else 0
    tracked_before_close = int(
        snapshots.get("before_close", {}).get("tracked", {}).get("current_allocated_bytes", 0)
    )
    tracked_after_close = int(
        snapshots.get("after_close", {}).get("tracked", {}).get("current_allocated_bytes", 0)
    )
    owned_peak = max(int(snapshot.get("owned_session_bytes", 0)) for snapshot in snapshots.values()) if snapshots else 0
    hip_used_values = [
        int(snapshot.get("hip", {}).get("used_bytes", 0))
        for snapshot in snapshots.values()
        if snapshot.get("hip", {}).get("available")
    ]
    hip_used_peak = max(hip_used_values) if hip_used_values else None
    return {
        "tracked_peak_allocated_bytes": tracked_peak,
        "tracked_peak_allocated_gib": _bytes_to_gib(tracked_peak),
        "tracked_current_allocated_bytes_before_close": tracked_before_close,
        "tracked_current_allocated_gib_before_close": _bytes_to_gib(tracked_before_close),
        "tracked_current_allocated_bytes_after_close": tracked_after_close,
        "tracked_current_allocated_gib_after_close": _bytes_to_gib(tracked_after_close),
        "owned_session_peak_bytes": owned_peak,
        "owned_session_peak_gib": _bytes_to_gib(owned_peak),
        "hip_used_peak_sampled_bytes": hip_used_peak,
        "hip_used_peak_sampled_gib": _bytes_to_gib(hip_used_peak) if hip_used_peak is not None else None,
        "notes": [
            "tracked_* covers hipENGINE allocations through hipengine.core.memory.malloc and keeps a high-water mark.",
            "hip_used_peak_sampled_* is sampled via hipMemGetInfo at phase boundaries, not a continuous device-wide peak.",
            "owned_session_* sums resident weights, scratch, KV/state, and per-session buffers owned by the GGUF session.",
        ],
    }


def _owned_device_bytes(session: Qwen35GGUFResidentSession) -> int:
    total = 0
    if session.runner is not None and session.runner.weights is not None:
        for weight in session.runner.weights.weights:
            for allocation in weight.allocations.values():
                if allocation.owns_buffer:
                    total += int(allocation.buffer.nbytes)
    if session.scratch is not None:
        total += sum(int(buffer.nbytes) for buffer in session.scratch.buffers)
    total += sum(int(buffer.nbytes) for buffer in session._buffers if buffer is not None)
    return total


def _hip_memory_info(runtime: HipRuntime) -> dict[str, Any]:
    try:
        free_bytes, total_bytes = runtime.mem_get_info()
    except Exception as exc:  # pragma: no cover - HIP failure path only
        return {"available": False, "error": str(exc)}
    used_bytes = total_bytes - free_bytes
    return {
        "available": True,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "free_gib": _bytes_to_gib(free_bytes),
        "total_gib": _bytes_to_gib(total_bytes),
        "used_gib": _bytes_to_gib(used_bytes),
    }


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1 << 30)


def _read_compiler_version(path: Path) -> str:
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"compiler version file {path} is empty")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
