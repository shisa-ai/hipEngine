#!/usr/bin/env python3
"""Load one resident Qwen3.6 model once and run README sweep repetitions.

This is the llama-bench-style harness for hipEngine hardware comparison rows: a
single resident session is created for the largest requested workload, then each
prompt/decode shape is run multiple times with ``session.reset()`` between runs.
The measured timing window excludes load/build and graph capture; load is still
reported once so GGUF decode-repack cost remains visible without multiplying it
by every shape/repetition.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_gguf_bench import (
    _memory_snapshot as _gguf_memory_snapshot,
    _memory_summary as _gguf_memory_summary,
    _prefill_chunk_sizes as _gguf_prefill_chunk_sizes,
    _run_existing_session_once as _run_existing_gguf_session_once,
)
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy
from scripts.qwen35_paro_bench import _memory_snapshot as _paro_memory_snapshot
from scripts.qwen35_paro_bench import _memory_summary as _paro_memory_summary
from scripts.qwen35_paro_bench import _prompt_tokens, _read_compiler_version

DEFAULT_WORKLOADS = ("512/128", "1K/128", "4K/128", "32K/128", "64K/128", "128K/128")
DEFAULT_PARO_MODEL = Path(
    "/home/lhl/.cache/huggingface/hub/"
    "models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/"
    "snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
)
DEFAULT_GGUF_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("paro", "gguf"), required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--quant", default="gguf_q4_k_s", help="GGUF quant label for --engine gguf")
    parser.add_argument("--workloads", nargs="+", default=list(DEFAULT_WORKLOADS), help="Workloads like 512/128, 4K/128, 128K/128")
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--warmup-decode-tokens", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument("--shared-expert-format", choices=("auto", "legacy_fp16", "packed_paro_w4"), default="packed_paro_w4")
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--prefill-linear-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-moe-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-query-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-post-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-full-attn-rope-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-chunk-autotune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefill-chunk-memory-budget-gib", type=float, default=0.0)
    parser.add_argument("--graph-replay-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-steps-per-replay", type=int, default=1)
    parser.add_argument("--force-bulk-prefill", action="store_true", help="GGUF: pass use_bulk=True")
    parser.add_argument("--no-bulk-prefill", action="store_true", help="GGUF: pass use_bulk=False")
    parser.add_argument("--bulk-prefill-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument("--prefill-chunk-size", type=int, default=0, help="GGUF all-layer chunk override")
    parser.add_argument("--use-expert-sidecar", action="store_true")
    parser.add_argument("--expert-sidecar-cache-dir", type=Path, default=None)
    parser.add_argument("--require-expert-sidecar", action="store_true")
    parser.add_argument("--preload-expert-sidecars", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True)
    add_kv_policy_args(
        parser,
        legacy_storage_flags=("--kv-storage-dtype",),
        help_prefix="Resident full-attention KV storage for prefill and decode",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    workloads = [_parse_workload(item) for item in args.workloads]
    workloads.sort(key=lambda item: (item[0], item[1]))
    if not workloads:
        raise ValueError("at least one workload is required")
    if args.warmup_runs < 0 or args.measured_runs <= 0:
        raise ValueError("--warmup-runs must be >= 0 and --measured-runs must be positive")
    if args.graph_steps_per_replay <= 0:
        raise ValueError("--graph-steps-per-replay must be positive")
    for prompt_length, decode_tokens in workloads:
        if prompt_length <= 0 or decode_tokens < 0:
            raise ValueError(f"invalid workload {prompt_length}/{decode_tokens}")
        if args.graph_replay_decode and decode_tokens % args.graph_steps_per_replay != 0:
            raise ValueError(f"decode tokens for {prompt_length}/{decode_tokens} must be divisible by --graph-steps-per-replay")
    for name in (
        "prefill_linear_chunk_size",
        "prefill_moe_chunk_size",
        "prefill_full_attn_query_chunk_size",
        "prefill_full_attn_post_chunk_size",
        "prefill_full_attn_rope_chunk_size",
        "prefill_chunk_size",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.force_bulk_prefill and args.no_bulk_prefill:
        raise ValueError("--force-bulk-prefill and --no-bulk-prefill are mutually exclusive")

    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file else None
    model = args.model or (DEFAULT_PARO_MODEL if args.engine == "paro" else DEFAULT_GGUF_MODEL)
    warmup_decode_tokens = int(args.warmup_decode_tokens if args.warmup_decode_tokens is not None else (4 if args.engine == "paro" else 1))
    max_sequence_length = max(prompt + warmup_decode_tokens + decode + 1 for prompt, decode in workloads)
    prefill_config = PrefillConfig(
        linear_chunk_size=args.prefill_linear_chunk_size,
        moe_chunk_size=args.prefill_moe_chunk_size,
        full_attn_query_chunk_size=args.prefill_full_attn_query_chunk_size,
        full_attn_post_chunk_size=args.prefill_full_attn_post_chunk_size,
        full_attn_rope_chunk_size=args.prefill_full_attn_rope_chunk_size,
        auto_tune_chunk_sizes=args.prefill_chunk_autotune,
        chunk_tune_memory_budget_gib=args.prefill_chunk_memory_budget_gib,
        attn_aotriton_min_tokens=args.attn_aotriton_min_tokens,
    )

    if args.engine == "paro":
        output = _run_paro_sweep(args, model, workloads, warmup_decode_tokens, max_sequence_length, compiler_version, prefill_config)
    else:
        output = _run_gguf_sweep(args, model, workloads, warmup_decode_tokens, max_sequence_length, compiler_version, prefill_config)

    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


def _run_paro_sweep(
    args: argparse.Namespace,
    model: Path,
    workloads: list[tuple[int, int]],
    warmup_decode_tokens: int,
    max_sequence_length: int,
    compiler_version: str | None,
    prefill_config: PrefillConfig,
) -> dict[str, Any]:
    runner = Qwen35ParoNextTokenRunner(
        model,
        shared_expert_format=None if args.shared_expert_format == "auto" else args.shared_expert_format,
        backend=args.backend,
    )
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    reset_memory_stats()
    persistent_memory: dict[str, Any] = {"before_load": _paro_memory_snapshot("before_load", runner.runtime)}
    load_start = time.perf_counter()
    session = Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    )
    load_seconds = time.perf_counter() - load_start
    persistent_memory["after_load"] = _paro_memory_snapshot("after_load", session.runtime, session)
    runs_by_workload: dict[str, list[dict[str, Any]]] = {}
    try:
        for prompt_length, decode_tokens in workloads:
            label = _format_workload(prompt_length, decode_tokens)
            prompt_tokens = _prompt_tokens(model, "Hello", args.token_id, prompt_length)
            runs: list[dict[str, Any]] = []
            for raw_index in range(args.warmup_runs + args.measured_runs):
                measured = raw_index >= args.warmup_runs
                run_index = raw_index - args.warmup_runs + 1 if measured else raw_index + 1
                run = _run_existing_paro_session_once(
                    session=session,
                    model=model,
                    runner=runner,
                    prompt_tokens=prompt_tokens,
                    decode_tokens=decode_tokens,
                    warmup_decode_tokens=warmup_decode_tokens,
                    graph_replay_decode=args.graph_replay_decode,
                    graph_steps_per_replay=args.graph_steps_per_replay,
                    measured=measured,
                    run_index=run_index,
                    load_seconds=load_seconds,
                )
                runs.append(run)
                _print_run(label, run)
            runs_by_workload[label] = runs
    finally:
        persistent_memory["before_close"] = _paro_memory_snapshot("before_close", session.runtime, session)
        session.close()
        persistent_memory["after_close"] = _paro_memory_snapshot("after_close", runner.runtime)

    return _sweep_output(
        engine="paro",
        model=model,
        quant="w4_paro",
        workloads=workloads,
        warmup_decode_tokens=warmup_decode_tokens,
        measured_runs=args.measured_runs,
        warmup_runs=args.warmup_runs,
        max_sequence_length=max_sequence_length,
        load_seconds=load_seconds,
        persistent_memory={"summary": _paro_memory_summary(persistent_memory), "snapshots": persistent_memory},
        runs_by_workload=runs_by_workload,
        compiler_version_file=args.compiler_version_file,
        compiler_version=compiler_version,
        extra={
            "backend": runner.backend,
            "requested_backend": args.backend,
            "target_arch": runner.target_arch,
            "shared_expert_format": args.shared_expert_format,
            "max_layers": args.max_layers or runner.config.num_hidden_layers,
            "kv_storage_dtype": kv_policy.storage_dtype.value,
            "kv_policy": kv_policy_json(kv_policy),
            "attn_aotriton_min_tokens": args.attn_aotriton_min_tokens,
        },
    )


def _run_existing_paro_session_once(
    *,
    session: Qwen35ParoResidentSession,
    model: Path,
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: Sequence[int],
    decode_tokens: int,
    warmup_decode_tokens: int,
    graph_replay_decode: bool,
    graph_steps_per_replay: int,
    measured: bool,
    run_index: int,
    load_seconds: float,
) -> dict[str, Any]:
    memory_snapshots: dict[str, Any] = {
        "after_load": _paro_memory_snapshot("after_load", session.runtime, session),
        "before_reset": _paro_memory_snapshot("before_reset", session.runtime, session),
    }
    session.reset()
    # Match historical per-shape auto-tuning while keeping one large resident allocation.
    session._resolve_prefill_config_for_length(len(prompt_tokens))
    memory_snapshots["after_reset"] = _paro_memory_snapshot("after_reset", session.runtime, session)
    generated: list[dict[str, Any]] = []
    graph_capture_seconds = 0.0
    prefill_start = time.perf_counter()
    first = session.prefill_native(prompt_tokens, sample=True)
    prefill_seconds = time.perf_counter() - prefill_start
    generated.append(first.to_json_dict())
    next_token = first.token_id
    memory_snapshots["after_prefill"] = _paro_memory_snapshot("after_prefill", session.runtime, session)

    warmup_start = time.perf_counter()
    for offset in range(warmup_decode_tokens):
        warmup = session.step(next_token, position=len(prompt_tokens) + offset, sample=True)
        if warmup is None:
            raise RuntimeError("PARO warmup decode did not produce a token")
        next_token = warmup.token_id
        generated.append(warmup.to_json_dict())
    warmup_decode_seconds = time.perf_counter() - warmup_start
    memory_snapshots["after_warmup_decode"] = _paro_memory_snapshot("after_warmup_decode", session.runtime, session)

    final = None
    decode_start_pos = len(prompt_tokens) + warmup_decode_tokens
    decode_graph_reused = False
    if graph_replay_decode and decode_tokens:
        capture_start = time.perf_counter()
        graph = session.capture_decode_graph(
            position=decode_start_pos,
            steps_per_replay=graph_steps_per_replay,
            max_replay_steps=decode_tokens,
            record_steps=0,
        )
        graph_capture_seconds = time.perf_counter() - capture_start
        try:
            decode_start = time.perf_counter()
            graph.replay(decode_tokens)
            decode_seconds = time.perf_counter() - decode_start
            final = graph.read_sample()
            generated.append(final.to_json_dict())
        finally:
            graph.close()
    else:
        decode_start = time.perf_counter()
        for offset in range(decode_tokens):
            final = session.step(next_token, position=decode_start_pos + offset, sample=True)
            if final is None:
                raise RuntimeError("PARO measured decode did not produce a token")
            next_token = final.token_id
            generated.append(final.to_json_dict())
        decode_seconds = time.perf_counter() - decode_start
    memory_snapshots["after_decode"] = _paro_memory_snapshot("after_decode", session.runtime, session)
    memory_snapshots["before_close"] = _paro_memory_snapshot("before_close", session.runtime, session)
    return {
        "run_index": int(run_index),
        "measured": bool(measured),
        "persistent_session": True,
        "model": str(model),
        "quant": "w4_paro",
        "backend": runner.backend,
        "prompt_length": len(prompt_tokens),
        "decode_tokens": int(decode_tokens),
        "warmup_decode_tokens": int(warmup_decode_tokens),
        "native_prefill_execution": session.native_prefill_plan().path,
        "prefill_execution_detail": getattr(session, "last_prefill_execution", None),
        "prefill_chunk_sizes": {
            "linear": session.prefill_config.linear_chunk_size,
            "moe": session.prefill_config.moe_chunk_size,
            "full_attn_query": session.prefill_config.full_attn_query_chunk_size,
            "full_attn_post": session.prefill_config.full_attn_post_chunk_size,
            "full_attn_rope": session.prefill_config.full_attn_rope_chunk_size,
        },
        "prefill_chunk_tuning": session.prefill_chunk_tuning,
        "decode_graph_reused": bool(decode_graph_reused),
        "timings": {
            "load_seconds": float(load_seconds),
            "load_seconds_is_shared_session": True,
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
            "finite_final_logit": None if final is None else bool(math.isfinite(final.logit)),
            "final_token_id": None if final is None else final.token_id,
            "final_logit": None if final is None else final.logit,
            "generated_preview_token_ids": [int(item["token_id"]) for item in generated[:16]],
            "generated_tail_token_ids": [int(item["token_id"]) for item in generated[-16:]],
        },
        "memory": _paro_memory_summary(memory_snapshots),
        "memory_snapshots": memory_snapshots,
    }


def _run_gguf_sweep(
    args: argparse.Namespace,
    model: Path,
    workloads: list[tuple[int, int]],
    warmup_decode_tokens: int,
    max_sequence_length: int,
    compiler_version: str | None,
    prefill_config: PrefillConfig,
) -> dict[str, Any]:
    from hipengine.core.hip import get_hip_runtime

    use_bulk_prefill = True if args.force_bulk_prefill else False if args.no_bulk_prefill else None
    runtime = get_hip_runtime()
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    reset_memory_stats()
    persistent_memory: dict[str, Any] = {"before_load": _gguf_memory_snapshot("before_load", runtime)}
    load_start = time.perf_counter()
    session = Qwen35GGUFResidentSession(
        model,
        runtime=runtime,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        max_sequence_length=max_sequence_length,
        use_expert_sidecar=args.use_expert_sidecar,
        expert_sidecar_cache_dir=args.expert_sidecar_cache_dir,
        require_expert_sidecar=args.require_expert_sidecar,
        preload_expert_sidecars=args.preload_expert_sidecars,
        use_wmma_prefill=args.use_wmma_prefill,
        use_gemv_decode=args.use_gemv_decode,
        prefill_chunk_size=args.prefill_chunk_size,
        prefill_config=prefill_config,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    )
    load_seconds = time.perf_counter() - load_start
    host_token_embedding_enabled = bool(getattr(session, "host_token_embedding_enabled", False))
    host_token_embedding_reason = getattr(session, "host_token_embedding_reason", None)
    persistent_memory["after_load"] = _gguf_memory_snapshot("after_load", runtime, session)
    runs_by_workload: dict[str, list[dict[str, Any]]] = {}
    try:
        for prompt_length, decode_tokens in workloads:
            label = _format_workload(prompt_length, decode_tokens)
            prompt_tokens = [int(args.token_id)] * int(prompt_length)
            graph_holder: dict[str, Any] = {}
            runs: list[dict[str, Any]] = []
            try:
                for raw_index in range(args.warmup_runs + args.measured_runs):
                    measured = raw_index >= args.warmup_runs
                    run_index = raw_index - args.warmup_runs + 1 if measured else raw_index + 1
                    run = _run_existing_gguf_session_once(
                        session=session,
                        runtime=runtime,
                        model=model,
                        quant=args.quant,
                        prompt_tokens=prompt_tokens,
                        decode_tokens=decode_tokens,
                        warmup_decode_tokens=warmup_decode_tokens,
                        graph_replay_decode=args.graph_replay_decode,
                        graph_steps_per_replay=args.graph_steps_per_replay,
                        use_bulk_prefill=use_bulk_prefill,
                        bulk_attention_mode=args.bulk_prefill_attention_mode,
                        use_wmma_prefill=args.use_wmma_prefill,
                        use_gemv_decode=args.use_gemv_decode,
                        prefill_chunk_size=args.prefill_chunk_size,
                        measured=measured,
                        run_index=run_index,
                        load_seconds=load_seconds,
                        persistent_session=True,
                        graph_holder=graph_holder,
                    )
                    runs.append(run)
                    _print_run(label, run)
            finally:
                graph = graph_holder.get("graph")
                if graph is not None:
                    graph.close()
            runs_by_workload[label] = runs
    finally:
        persistent_memory["before_close"] = _gguf_memory_snapshot("before_close", runtime, session)
        session.close()
        persistent_memory["after_close"] = _gguf_memory_snapshot("after_close", runtime)

    return _sweep_output(
        engine="gguf",
        model=model,
        quant=args.quant,
        workloads=workloads,
        warmup_decode_tokens=warmup_decode_tokens,
        measured_runs=args.measured_runs,
        warmup_runs=args.warmup_runs,
        max_sequence_length=max_sequence_length,
        load_seconds=load_seconds,
        persistent_memory={"summary": _gguf_memory_summary(persistent_memory), "snapshots": persistent_memory},
        runs_by_workload=runs_by_workload,
        compiler_version_file=args.compiler_version_file,
        compiler_version=compiler_version,
        extra={
            "backend": "hip_gfx1100",
            "use_bulk_prefill": use_bulk_prefill,
            "bulk_prefill_attention_mode": args.bulk_prefill_attention_mode,
            "use_wmma_prefill": args.use_wmma_prefill,
            "use_gemv_decode": args.use_gemv_decode,
            "effective_use_wmma_prefill": getattr(session, "use_wmma_prefill", None),
            "effective_use_gemv_decode": getattr(session, "use_gemv_decode", None),
            "fastpath_safety": None if session.fastpath_safety is None else session.fastpath_safety.as_dict(),
            "kv_storage_dtype": kv_policy.storage_dtype.value,
            "kv_policy": kv_policy_json(kv_policy),
            "prefill_chunk_sizes_session": _gguf_prefill_chunk_sizes(session.prefill_config),
            "host_token_embedding_enabled": host_token_embedding_enabled,
            "host_token_embedding_reason": host_token_embedding_reason,
        },
    )


def _sweep_output(
    *,
    engine: str,
    model: Path,
    quant: str,
    workloads: list[tuple[int, int]],
    warmup_decode_tokens: int,
    measured_runs: int,
    warmup_runs: int,
    max_sequence_length: int,
    load_seconds: float,
    persistent_memory: dict[str, Any],
    runs_by_workload: dict[str, list[dict[str, Any]]],
    compiler_version_file: Path | None,
    compiler_version: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    summaries = {
        label: _summarize_runs([run for run in runs if run.get("measured")])
        for label, runs in runs_by_workload.items()
    }
    return {
        "schema": 1,
        "mode": "qwen35_readme_persistent_resident_sweep",
        "engine": engine,
        "model": str(model),
        "quant": quant,
        "workloads": [_format_workload(prompt, decode) for prompt, decode in workloads],
        "prompt_source": "repeated_token_id",
        "warmup_decode_tokens": int(warmup_decode_tokens),
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
        "max_sequence_length": int(max_sequence_length),
        "persistent_session_load_seconds": float(load_seconds),
        "persistent_session_memory": persistent_memory,
        "compiler_version_file": None if compiler_version_file is None else str(compiler_version_file),
        "compiler_version_first_line": None if compiler_version is None else compiler_version.splitlines()[0],
        "summary_by_workload": summaries,
        "runs_by_workload": runs_by_workload,
        "extra": extra,
        "notes": [
            "The model/session is loaded once for the largest requested shape and reset between repetitions.",
            "Each workload uses warmup_runs discarded repetitions followed by measured_runs measured repetitions.",
            "Measured decode excludes HIP graph capture time when graph replay is enabled; graph capture is reported per first run of each shape.",
        ],
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    prefill = [run["throughput"]["prefill_tok_s"] for run in runs if run.get("throughput", {}).get("prefill_tok_s") is not None]
    decode = [run["throughput"]["decode_tok_s"] for run in runs if run.get("throughput", {}).get("decode_tok_s") is not None]
    peak = [run["memory"].get("tracked_peak_allocated_gib") for run in runs if run.get("memory", {}).get("tracked_peak_allocated_gib") is not None]
    final_ids = [run.get("correctness_sanity", {}).get("final_token_id") for run in runs]
    return {
        "prefill_tok_s": _stats(prefill),
        "decode_tok_s": _stats(decode),
        "tracked_peak_allocated_gib": _stats(peak),
        "final_token_ids": final_ids,
        "final_token_ids_stable": len(set(final_ids)) <= 1,
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "min": None, "max": None, "mean": None, "stdev": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _print_run(label: str, run: dict[str, Any]) -> None:
    kind = "measured" if run.get("measured") else "warmup"
    throughput = run.get("throughput", {})
    memory = run.get("memory", {})
    print(
        f"{label} {kind}_run={run.get('run_index')} "
        f"prefill_tok_s={throughput.get('prefill_tok_s'):.6f} "
        f"decode_tok_s={throughput.get('decode_tok_s'):.6f} "
        f"peak_gib={memory.get('tracked_peak_allocated_gib'):.6f}",
        file=sys.stderr,
        flush=True,
    )


def _parse_workload(text: str) -> tuple[int, int]:
    if "/" not in text:
        raise argparse.ArgumentTypeError(f"workload must be prompt/decode, got {text!r}")
    prompt, decode = text.split("/", 1)
    return _parse_count(prompt), _parse_count(decode)


def _parse_count(text: str) -> int:
    text = text.strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1024)
    return int(text)


def _format_workload(prompt: int, decode: int) -> str:
    return f"{_format_count(prompt)}/{_format_count(decode)}"


def _format_count(value: int) -> str:
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
