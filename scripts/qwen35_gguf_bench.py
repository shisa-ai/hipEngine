#!/usr/bin/env python3
"""Resident Qwen3.5 GGUF c=1 benchmark harness.

The harness measures the public GGUF resident execution surface directly.  By
default it creates a fresh ``Qwen35GGUFResidentSession`` per warmup/measured
run, matching historical artifacts.  ``--persistent-session`` creates one
resident session and resets sequence state between runs, avoiding repeated GGUF
load/decode-repack work while preserving the same prefill/decode timing window.
Each run uses default resident prefill (bulk when supported, token-serial
fallback for short prompts; qwen35moe uses fast fully bulk attention+MoE by
default), one optional warmup decode token, and one-step HIP graph replay for
measured decode.  It is intentionally shape-driven so retained artifacts can
compare 512/128 and 4K/128 against PARO resident diagnostics and llama.cpp GGUF
rows.
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
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy

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
        "--persistent-session",
        action="store_true",
        help=(
            "Create one Qwen35GGUFResidentSession and call session.reset() between "
            "warmup/measured runs. This avoids repeated GGUF load/decode-repack "
            "work while keeping per-run prefill/decode timing separate."
        ),
    )
    prefill_group = parser.add_mutually_exclusive_group()
    prefill_group.add_argument(
        "--force-bulk-prefill",
        action="store_true",
        help="Pass use_bulk=True to Qwen35GGUFResidentSession.prefill().",
    )
    prefill_group.add_argument(
        "--no-bulk-prefill",
        action="store_true",
        help="Pass use_bulk=False to Qwen35GGUFResidentSession.prefill().",
    )
    parser.add_argument(
        "--bulk-prefill-attention-mode",
        choices=("bulk", "native"),
        default="bulk",
        help="When bulk prefill is forced/selected, use fully bulk attention or native row-serial attention with row-bulk FFN/MoE.",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=0, help="Manual GGUF all-layer prefill chunk override (0 uses PrefillConfig policy).")
    parser.add_argument("--prefill-linear-chunk-size", type=int, default=0, help="Chunk linear-attention prefill layers (0 lets auto policy decide).")
    parser.add_argument("--prefill-moe-chunk-size", type=int, default=0, help="Chunk MoE/post-attention rows where supported (0 lets auto policy decide).")
    parser.add_argument("--prefill-full-attn-query-chunk-size", type=int, default=0, help="Chunk full-attention query rows (0 lets auto policy decide).")
    parser.add_argument("--prefill-full-attn-post-chunk-size", type=int, default=0, help="Limit full-attention post/MoE chunk rows when query chunk is unset.")
    parser.add_argument("--prefill-full-attn-rope-chunk-size", type=int, default=0, help="Limit full-attention RoPE chunk rows when query chunk is unset.")
    parser.add_argument(
        "--prefill-chunk-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-select long-context prefill chunk sizes from the memory budget (default).",
    )
    parser.add_argument(
        "--prefill-chunk-memory-budget-gib",
        type=float,
        default=0.0,
        help="Optional resident high-water budget for long-context chunk tuning; 0 derives a budget from device VRAM.",
    )
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
    parser.add_argument(
        "--use-expert-sidecar",
        action="store_true",
        help="Use explicit qwen35moe GGUF expert pack8 sidecar kernels during bulk prefill.",
    )
    parser.add_argument(
        "--expert-sidecar-cache-dir",
        type=Path,
        default=None,
        help="Directory containing/building qwen35moe GGUF expert pack8 sidecars.",
    )
    parser.add_argument(
        "--require-expert-sidecar",
        action="store_true",
        help="Fail instead of building missing qwen35moe expert sidecar cache files.",
    )
    parser.add_argument(
        "--preload-expert-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load all expert sidecar host arrays during session load so measured prefill only copies host->device per layer.",
    )
    parser.add_argument(
        "--use-wmma-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the GGUF WMMA prefill opt-in for the resident session; omit to use HIPENGINE_GGUF_WMMA_PREFILL.",
    )
    parser.add_argument(
        "--use-gemv-decode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the GGUF rows=1 GEMV decode opt-in for the resident session; omit to use HIPENGINE_GGUF_GEMV_DECODE.",
    )
    add_kv_policy_args(
        parser,
        legacy_storage_flags=("--kv-storage-dtype",),
        help_prefix="GGUF resident full-attention KV",
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
    for name in (
        "prefill_chunk_size",
        "prefill_linear_chunk_size",
        "prefill_moe_chunk_size",
        "prefill_full_attn_query_chunk_size",
        "prefill_full_attn_post_chunk_size",
        "prefill_full_attn_rope_chunk_size",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.prefill_chunk_memory_budget_gib < 0.0:
        raise ValueError("--prefill-chunk-memory-budget-gib must be non-negative")

    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file else None
    if args.force_bulk_prefill:
        use_bulk_prefill = True
    elif args.no_bulk_prefill:
        use_bulk_prefill = False
    else:
        use_bulk_prefill = None
    prompt_tokens = [int(args.token_id)] * int(args.prompt_length)
    max_sequence_length = len(prompt_tokens) + args.warmup_decode_tokens + args.decode_tokens + 1
    prefill_config = PrefillConfig(
        linear_chunk_size=args.prefill_linear_chunk_size,
        moe_chunk_size=args.prefill_moe_chunk_size,
        full_attn_query_chunk_size=args.prefill_full_attn_query_chunk_size,
        full_attn_post_chunk_size=args.prefill_full_attn_post_chunk_size,
        full_attn_rope_chunk_size=args.prefill_full_attn_rope_chunk_size,
        auto_tune_chunk_sizes=args.prefill_chunk_autotune,
        chunk_tune_memory_budget_gib=args.prefill_chunk_memory_budget_gib,
    )
    kv_policy = resolve_args_kv_policy(args, block_size=256)

    if args.persistent_session:
        runs, persistent_session_load_seconds, persistent_session_memory = _run_persistent_session(
            model=args.model,
            quant=args.quant,
            prompt_tokens=prompt_tokens,
            decode_tokens=args.decode_tokens,
            warmup_decode_tokens=args.warmup_decode_tokens,
            max_sequence_length=max_sequence_length,
            graph_replay_decode=args.graph_replay_decode,
            graph_steps_per_replay=args.graph_steps_per_replay,
            use_bulk_prefill=use_bulk_prefill,
            bulk_attention_mode=args.bulk_prefill_attention_mode,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            use_expert_sidecar=args.use_expert_sidecar,
            expert_sidecar_cache_dir=args.expert_sidecar_cache_dir,
            require_expert_sidecar=args.require_expert_sidecar,
            preload_expert_sidecars=args.preload_expert_sidecars,
            use_wmma_prefill=args.use_wmma_prefill,
            use_gemv_decode=args.use_gemv_decode,
            prefill_chunk_size=args.prefill_chunk_size,
            prefill_config=prefill_config,
            kv_policy=kv_policy,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        )
        session_mode = "persistent"
    else:
        runs = []
        persistent_session_load_seconds = None
        persistent_session_memory = None
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
                use_bulk_prefill=use_bulk_prefill,
                bulk_attention_mode=args.bulk_prefill_attention_mode,
                compiler_version=compiler_version,
                require_cached_build=args.require_cached_build,
                use_expert_sidecar=args.use_expert_sidecar,
                expert_sidecar_cache_dir=args.expert_sidecar_cache_dir,
                require_expert_sidecar=args.require_expert_sidecar,
                preload_expert_sidecars=args.preload_expert_sidecars,
                use_wmma_prefill=args.use_wmma_prefill,
                use_gemv_decode=args.use_gemv_decode,
                prefill_chunk_size=args.prefill_chunk_size,
                prefill_config=prefill_config,
                kv_policy=kv_policy,
                measured=measured,
                run_index=(run_index - args.warmup_runs + 1 if measured else run_index + 1),
            )
            runs.append(run)
        session_mode = "per_run"

    for run in runs:
        label = "measured" if run["measured"] else "warmup"
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
        "mode": _mode_name(
            graph_replay_decode=args.graph_replay_decode,
            use_bulk_prefill=use_bulk_prefill,
            bulk_attention_mode=args.bulk_prefill_attention_mode,
        ),
        "session_mode": session_mode,
        "persistent_session": bool(args.persistent_session),
        "persistent_session_load_seconds": persistent_session_load_seconds,
        "persistent_session_memory": persistent_session_memory,
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
        "use_bulk_prefill": use_bulk_prefill,
        "bulk_prefill_attention_mode": args.bulk_prefill_attention_mode,
        "requested_prefill_chunk_size": int(args.prefill_chunk_size),
        "requested_prefill_chunk_sizes": {
            "linear": int(args.prefill_linear_chunk_size),
            "moe": int(args.prefill_moe_chunk_size),
            "full_attn_query": int(args.prefill_full_attn_query_chunk_size),
            "full_attn_post": int(args.prefill_full_attn_post_chunk_size),
            "full_attn_rope": int(args.prefill_full_attn_rope_chunk_size),
        },
        "prefill_chunk_autotune": bool(args.prefill_chunk_autotune),
        "prefill_chunk_memory_budget_gib": float(args.prefill_chunk_memory_budget_gib),
        "prefill_chunk_tuning_all": [run.get("prefill_chunk_tuning") for run in runs],
        "prefill_chunk_sizes_all": [run.get("prefill_chunk_sizes") for run in runs],
        "require_cached_build": bool(args.require_cached_build),
        "use_expert_sidecar": bool(args.use_expert_sidecar),
        "expert_sidecar_cache_dir": None if args.expert_sidecar_cache_dir is None else str(args.expert_sidecar_cache_dir),
        "require_expert_sidecar": bool(args.require_expert_sidecar),
        "preload_expert_sidecars": bool(args.preload_expert_sidecars),
        "use_wmma_prefill": args.use_wmma_prefill,
        "use_gemv_decode": args.use_gemv_decode,
        "requested_use_wmma_prefill": args.use_wmma_prefill,
        "requested_use_gemv_decode": args.use_gemv_decode,
        "kv_storage_dtype": kv_policy.storage_dtype.value,
        "kv_policy": kv_policy_json(kv_policy),
        "effective_use_wmma_prefill_all": [run.get("effective_use_wmma_prefill") for run in runs],
        "effective_use_gemv_decode_all": [run.get("effective_use_gemv_decode") for run in runs],
        "fastpath_safety": [run.get("fastpath_safety") for run in runs],
        "compiler_version_file": None if args.compiler_version_file is None else str(args.compiler_version_file),
        "compiler_version_first_line": None if compiler_version is None else compiler_version.splitlines()[0],
        "runs": runs,
        "summary": _summary(measured_runs),
        "notes": [
            "Prefill mode is controlled by --force-bulk-prefill/--no-bulk-prefill; default delegates to Qwen35GGUFResidentSession.prefill().",
            "--bulk-prefill-attention-mode=bulk selects the fast fully bulk scheduler and is the qwen35moe delegated default.",
            "--bulk-prefill-attention-mode=native preserves row-serial attention while using row-bulk FFN/MoE as a qwen35moe diagnostic fallback.",
            "--use-expert-sidecar enables explicit qwen35moe GGUF expert pack8 sidecar kernels for bulk prefill; generated sidecars live in the requested cache dir.",
            "--use-wmma-prefill opts GGUF bulk prefill into P8 WMMA dispatch, including qwen35moe compact grouped selected-MoE when the raw kernels are available.",
            "--use-gemv-decode opts rows=1 GGUF decode into the P9 pack8 GEMV decode path, including graph-capture decode.",
            "GGUF prefill chunking uses the same PrefillConfig auto policy as PARO unless --prefill-chunk-size or explicit per-surface chunk flags override it.",
            "Measured decode excludes graph capture time when graph_replay_decode=true.",
            "--persistent-session creates one resident session and resets sequence state between warmup/measured runs, avoiding repeated GGUF load/decode-repack work. Historical artifacts used the default per-run session mode.",
        ],
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


def _mode_name(*, graph_replay_decode: bool, use_bulk_prefill: bool | None, bulk_attention_mode: str) -> str:
    if use_bulk_prefill is True:
        prefill = f"bulk_prefill_{bulk_attention_mode}_attention"
    elif use_bulk_prefill is False:
        prefill = "token_serial_prefill"
    else:
        prefill = "default_prefill"
    decode = "graph_decode" if graph_replay_decode else "eager_decode"
    return f"resident_{prefill}_{decode}"


def _run_persistent_session(
    *,
    model: Path,
    quant: str,
    prompt_tokens: list[int],
    decode_tokens: int,
    warmup_decode_tokens: int,
    max_sequence_length: int,
    graph_replay_decode: bool,
    graph_steps_per_replay: int,
    use_bulk_prefill: bool | None,
    bulk_attention_mode: str,
    compiler_version: str | None,
    require_cached_build: bool,
    use_expert_sidecar: bool,
    expert_sidecar_cache_dir: Path | None,
    require_expert_sidecar: bool,
    preload_expert_sidecars: bool,
    use_wmma_prefill: bool | None,
    use_gemv_decode: bool | None,
    prefill_chunk_size: int,
    prefill_config: PrefillConfig,
    kv_policy,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Run warmup/measured iterations inside one resident GGUF session.

    Historical qwen35_gguf_bench artifacts intentionally created a fresh session
    per run so load/repack behavior was visible in every raw run.  For repeated
    performance measurements that is unnecessarily expensive: GGUF Q4_K_S on a
    W7900 spends about 60 seconds in load/decode-repack while a 512/128 timed
    iteration only spends ~1.7 seconds in prefill+decode.  This path loads once,
    calls session.reset() before each run, and closes once at the end.
    """

    runtime = get_hip_runtime()
    reset_memory_stats()
    persistent_memory: dict[str, Any] = {"before_load": _memory_snapshot("before_load", runtime)}
    load_start = time.perf_counter()
    session = Qwen35GGUFResidentSession(
        model,
        runtime=runtime,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        max_sequence_length=max_sequence_length,
        use_expert_sidecar=use_expert_sidecar,
        expert_sidecar_cache_dir=expert_sidecar_cache_dir,
        require_expert_sidecar=require_expert_sidecar,
        preload_expert_sidecars=preload_expert_sidecars,
        use_wmma_prefill=use_wmma_prefill,
        use_gemv_decode=use_gemv_decode,
        prefill_chunk_size=prefill_chunk_size,
        prefill_config=prefill_config,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    )
    load_seconds = time.perf_counter() - load_start
    persistent_memory["after_load"] = _memory_snapshot("after_load", runtime, session)

    runs: list[dict[str, Any]] = []
    # HIP graph capture/instantiate can retain large internal allocations until
    # process teardown on ROCm 7.2 when repeated inside one long-lived session.
    # Persistent mode therefore captures one reusable graph (same prompt shape,
    # same start position after reset+prefill+warmup) and replays it across all
    # warmup/measured runs. Per-run mode keeps historical recapture behavior.
    graph_holder: dict[str, Any] = {}
    try:
        for raw_run_index in range(int(warmup_runs) + int(measured_runs)):
            measured = raw_run_index >= int(warmup_runs)
            run_index = raw_run_index - int(warmup_runs) + 1 if measured else raw_run_index + 1
            run = _run_existing_session_once(
                session=session,
                runtime=runtime,
                model=model,
                quant=quant,
                prompt_tokens=prompt_tokens,
                decode_tokens=decode_tokens,
                warmup_decode_tokens=warmup_decode_tokens,
                graph_replay_decode=graph_replay_decode,
                graph_steps_per_replay=graph_steps_per_replay,
                use_bulk_prefill=use_bulk_prefill,
                bulk_attention_mode=bulk_attention_mode,
                use_wmma_prefill=use_wmma_prefill,
                use_gemv_decode=use_gemv_decode,
                prefill_chunk_size=prefill_chunk_size,
                measured=measured,
                run_index=run_index,
                load_seconds=load_seconds,
                persistent_session=True,
                graph_holder=graph_holder,
            )
            runs.append(run)
    finally:
        persistent_memory["before_close"] = _memory_snapshot("before_close", runtime, session)
        graph = graph_holder.get("graph")
        if graph is not None:
            try:
                graph.close()
            finally:
                graph_holder["graph"] = None
        persistent_memory["after_graph_close"] = _memory_snapshot("after_graph_close", runtime, session)
        session.close()
        persistent_memory["after_close"] = _memory_snapshot("after_close", runtime)

    persistent_summary = _memory_summary(persistent_memory)
    return runs, load_seconds, {"summary": persistent_summary, "snapshots": persistent_memory}


def _run_existing_session_once(
    *,
    session: Qwen35GGUFResidentSession,
    runtime: HipRuntime,
    model: Path,
    quant: str,
    prompt_tokens: list[int],
    decode_tokens: int,
    warmup_decode_tokens: int,
    graph_replay_decode: bool,
    graph_steps_per_replay: int,
    use_bulk_prefill: bool | None,
    bulk_attention_mode: str,
    use_wmma_prefill: bool | None,
    use_gemv_decode: bool | None,
    prefill_chunk_size: int,
    measured: bool,
    run_index: int,
    load_seconds: float,
    persistent_session: bool,
    graph_holder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one prefill/decode iteration on an existing resident session."""

    fastpath_safety = session.fastpath_safety.as_dict() if session.fastpath_safety is not None else None
    memory_snapshots: dict[str, Any] = {
        "after_load": _memory_snapshot("after_load", runtime, session),
        "before_reset": _memory_snapshot("before_reset", runtime, session),
    }
    session.reset()
    memory_snapshots["after_reset"] = _memory_snapshot("after_reset", runtime, session)

    generated_token_ids: list[int] = []
    final = None
    graph_capture_seconds = 0.0
    prefill_seconds = 0.0
    warmup_decode_seconds = 0.0
    decode_seconds = 0.0
    host_token_embedding_graph_disabled = bool(
        graph_replay_decode and getattr(session, "host_token_embedding_enabled", False)
    )
    effective_graph_replay_decode = bool(graph_replay_decode and not host_token_embedding_graph_disabled)
    try:
        prefill_start = time.perf_counter()
        first = session.prefill(
            prompt_tokens,
            use_bulk=use_bulk_prefill,
            bulk_attention_mode=bulk_attention_mode,
            return_logits=False,
        )
        prefill_seconds = time.perf_counter() - prefill_start
        generated_token_ids.append(first.token_id)
        next_token = first.token_id
        memory_snapshots["after_prefill"] = _memory_snapshot("after_prefill", runtime, session)

        warmup_start = time.perf_counter()
        for _ in range(warmup_decode_tokens):
            warmup = session.step(next_token, return_logits=False)
            next_token = warmup.token_id
            generated_token_ids.append(warmup.token_id)
        warmup_decode_seconds = time.perf_counter() - warmup_start
        memory_snapshots["after_warmup_decode"] = _memory_snapshot("after_warmup_decode", runtime, session)

        retained_graph = graph_holder is not None
        decode_graph_reused = False
        decode_graph_recorded_tokens = False
        if effective_graph_replay_decode and decode_tokens:
            graph = graph_holder.get("graph") if graph_holder is not None else None
            if graph is None:
                capture_start = time.perf_counter()
                graph = session.capture_decode_graph(
                    position=session.position,
                    steps_per_replay=graph_steps_per_replay,
                    max_replay_steps=decode_tokens,
                    # Reusable graphs avoid generated-token recording because
                    # the generated index buffer is intentionally stateful across
                    # replays. Final-token sanity still comes from read_sample().
                    record_steps=0 if retained_graph else decode_tokens,
                )
                graph_capture_seconds = time.perf_counter() - capture_start
                if graph_holder is not None:
                    graph_holder["graph"] = graph
            else:
                decode_graph_reused = True
            decode_graph_recorded_tokens = getattr(graph, "generated", None) is not None
            if decode_graph_reused:
                # capture_decode_graph() primes the device position scalar before
                # capture, outside the graph body. A retained graph therefore
                # needs the same scalar reset before each replay after
                # session.reset()+prefill()+warmup.
                stream = runtime.stream_create()
                try:
                    session._set_full_attention_position_device(session.position, stream=stream)
                    runtime.stream_synchronize(stream)
                finally:
                    runtime.stream_destroy(stream)
            try:
                decode_start = time.perf_counter()
                graph.replay(decode_tokens)
                decode_seconds = time.perf_counter() - decode_start
                if decode_graph_recorded_tokens:
                    generated_token_ids.extend(graph.read_generated_token_ids(decode_tokens))
                final = graph.read_sample()
                if not decode_graph_recorded_tokens and final is not None:
                    generated_token_ids.append(final.token_id)
            finally:
                if not retained_graph:
                    graph.close()
        else:
            decode_graph_reused = False
            decode_graph_recorded_tokens = False
            decode_start = time.perf_counter()
            for step_index in range(decode_tokens):
                final = session.step(next_token, return_logits=(step_index == decode_tokens - 1))
                next_token = final.token_id
                generated_token_ids.append(next_token)
            decode_seconds = time.perf_counter() - decode_start
        memory_snapshots["after_decode"] = _memory_snapshot("after_decode", runtime, session)
        final_token_id = None if final is None else final.token_id
        final_logit = None if final is None else final.logit
        finite_logits = None if final is None else bool(np.all(np.isfinite(final.logits)))
    finally:
        memory_snapshots["before_close"] = _memory_snapshot("before_close", runtime, session)

    return {
        "run_index": int(run_index),
        "measured": bool(measured),
        "persistent_session": bool(persistent_session),
        "model": str(model),
        "quant": quant,
        "prompt_length": len(prompt_tokens),
        "decode_tokens": int(decode_tokens),
        "warmup_decode_tokens": int(warmup_decode_tokens),
        "use_bulk_prefill": use_bulk_prefill,
        "bulk_prefill_attention_mode": bulk_attention_mode,
        "use_wmma_prefill": use_wmma_prefill,
        "use_gemv_decode": use_gemv_decode,
        "requested_use_wmma_prefill": use_wmma_prefill,
        "requested_use_gemv_decode": use_gemv_decode,
        "requested_prefill_chunk_size": int(prefill_chunk_size),
        "prefill_chunk_sizes": _prefill_chunk_sizes(session.prefill_config),
        "prefill_chunk_tuning": session.prefill_chunk_tuning,
        "effective_use_wmma_prefill": None if fastpath_safety is None else fastpath_safety.get("effective_wmma_prefill"),
        "effective_use_gemv_decode": None if fastpath_safety is None else fastpath_safety.get("effective_gemv_decode"),
        "fastpath_safety": fastpath_safety,
        "decode_graph_reused": bool(decode_graph_reused),
        "decode_graph_recorded_tokens": bool(decode_graph_recorded_tokens),
        "host_token_embedding_enabled": bool(getattr(session, "host_token_embedding_enabled", False)),
        "host_token_embedding_reason": getattr(session, "host_token_embedding_reason", None),
        "decode_graph_disabled_reason": (
            "host_token_embedding" if host_token_embedding_graph_disabled else None
        ),
        "timings": {
            "load_seconds": load_seconds,
            "load_seconds_is_shared_session": bool(persistent_session),
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
    use_bulk_prefill: bool | None,
    bulk_attention_mode: str,
    compiler_version: str | None,
    require_cached_build: bool,
    use_expert_sidecar: bool,
    expert_sidecar_cache_dir: Path | None,
    require_expert_sidecar: bool,
    preload_expert_sidecars: bool,
    use_wmma_prefill: bool | None,
    use_gemv_decode: bool | None,
    prefill_chunk_size: int,
    prefill_config: PrefillConfig,
    kv_policy,
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
        use_expert_sidecar=use_expert_sidecar,
        expert_sidecar_cache_dir=expert_sidecar_cache_dir,
        require_expert_sidecar=require_expert_sidecar,
        preload_expert_sidecars=preload_expert_sidecars,
        use_wmma_prefill=use_wmma_prefill,
        use_gemv_decode=use_gemv_decode,
        prefill_chunk_size=prefill_chunk_size,
        prefill_config=prefill_config,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    )
    load_seconds = time.perf_counter() - load_start
    fastpath_safety = session.fastpath_safety.as_dict() if session.fastpath_safety is not None else None
    memory_snapshots["after_load"] = _memory_snapshot("after_load", runtime, session)

    generated_token_ids: list[int] = []
    final = None
    graph_capture_seconds = 0.0
    host_token_embedding_graph_disabled = bool(
        graph_replay_decode and getattr(session, "host_token_embedding_enabled", False)
    )
    effective_graph_replay_decode = bool(graph_replay_decode and not host_token_embedding_graph_disabled)
    try:
        prefill_start = time.perf_counter()
        first = session.prefill(
            prompt_tokens,
            use_bulk=use_bulk_prefill,
            bulk_attention_mode=bulk_attention_mode,
            return_logits=False,
        )
        prefill_seconds = time.perf_counter() - prefill_start
        generated_token_ids.append(first.token_id)
        next_token = first.token_id
        memory_snapshots["after_prefill"] = _memory_snapshot("after_prefill", runtime, session)

        warmup_start = time.perf_counter()
        for _ in range(warmup_decode_tokens):
            warmup = session.step(next_token, return_logits=False)
            next_token = warmup.token_id
            generated_token_ids.append(warmup.token_id)
        warmup_decode_seconds = time.perf_counter() - warmup_start
        memory_snapshots["after_warmup_decode"] = _memory_snapshot("after_warmup_decode", runtime, session)

        if effective_graph_replay_decode and decode_tokens:
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
            for step_index in range(decode_tokens):
                final = session.step(next_token, return_logits=(step_index == decode_tokens - 1))
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
        "use_bulk_prefill": use_bulk_prefill,
        "bulk_prefill_attention_mode": bulk_attention_mode,
        "use_wmma_prefill": use_wmma_prefill,
        "use_gemv_decode": use_gemv_decode,
        "requested_use_wmma_prefill": use_wmma_prefill,
        "requested_use_gemv_decode": use_gemv_decode,
        "requested_prefill_chunk_size": int(prefill_chunk_size),
        "prefill_chunk_sizes": _prefill_chunk_sizes(session.prefill_config),
        "prefill_chunk_tuning": session.prefill_chunk_tuning,
        "effective_use_wmma_prefill": None if fastpath_safety is None else fastpath_safety.get("effective_wmma_prefill"),
        "effective_use_gemv_decode": None if fastpath_safety is None else fastpath_safety.get("effective_gemv_decode"),
        "fastpath_safety": fastpath_safety,
        "host_token_embedding_enabled": bool(getattr(session, "host_token_embedding_enabled", False)),
        "host_token_embedding_reason": getattr(session, "host_token_embedding_reason", None),
        "decode_graph_disabled_reason": (
            "host_token_embedding" if host_token_embedding_graph_disabled else None
        ),
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


def _prefill_chunk_sizes(config: PrefillConfig | None) -> dict[str, int | bool] | None:
    if config is None:
        return None
    return {
        "linear": int(config.linear_chunk_size),
        "moe": int(config.moe_chunk_size),
        "full_attn_query": int(config.full_attn_query_chunk_size),
        "full_attn_post": int(config.full_attn_post_chunk_size),
        "full_attn_rope": int(config.full_attn_rope_chunk_size),
        "attn_aotriton_min_tokens": int(config.attn_aotriton_min_tokens),
        "auto_tune": bool(config.auto_tune_chunk_sizes),
        "chunk_tune_min_tokens": int(config.chunk_tune_min_tokens),
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
        payload["owned_session_breakdown"] = _owned_device_breakdown(session)
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


def _owned_device_breakdown(session: Qwen35GGUFResidentSession) -> dict[str, Any]:
    """Return a JSON-serialisable owned-device memory census for GGUF sessions."""

    weights = _owned_weight_breakdown(session)
    decode_scratch = _decode_scratch_breakdown(getattr(session, "scratch", None))
    session_buffers = _session_buffer_breakdown(session)
    total_bytes = int(weights["total_bytes"]) + int(decode_scratch["total_bytes"]) + int(session_buffers["total_bytes"])
    return {
        "total_bytes": total_bytes,
        "total_gib": _bytes_to_gib(total_bytes),
        "families": {
            "weights": weights,
            "decode_scratch": decode_scratch,
            "session_buffers": session_buffers,
        },
        "notes": [
            "weights counts unique resident GGUF allocations that own their device buffer; tied aliases are not double-counted.",
            "decode_scratch is the persistent c=1 decode workspace, including full-attention KV cache and linear-attention recurrent state.",
            "session_buffers includes logits/lm-head temporaries, full-sequence prefill token/hidden buffers, and the bulk-prefill scratch workspace.",
        ],
    }


def _owned_weight_breakdown(session: Qwen35GGUFResidentSession) -> dict[str, Any]:
    by_quant: dict[str, int] = {}
    by_layout: dict[str, int] = {}
    by_quant_layout: dict[str, int] = {}
    by_allocation: dict[str, int] = {}
    total = 0
    count = 0
    if session.runner is not None and session.runner.weights is not None:
        for weight in session.runner.weights.weights:
            spec = weight.spec
            quant_key = str(spec.quant_key)
            layout = str(spec.layout)
            for allocation_name, allocation in weight.allocations.items():
                if not allocation.owns_buffer:
                    continue
                nbytes = _buffer_nbytes(allocation.buffer)
                total += nbytes
                count += 1
                _add_bytes(by_quant, quant_key, nbytes)
                _add_bytes(by_layout, layout, nbytes)
                _add_bytes(by_quant_layout, f"{quant_key}:{layout}", nbytes)
                _add_bytes(by_allocation, str(allocation_name), nbytes)
    return {
        "total_bytes": total,
        "total_gib": _bytes_to_gib(total),
        "allocation_count": count,
        "by_quant_key_bytes": by_quant,
        "by_layout_bytes": by_layout,
        "by_quant_layout_bytes": by_quant_layout,
        "by_allocation_name_bytes": by_allocation,
    }


def _decode_scratch_breakdown(scratch: object | None) -> dict[str, Any]:
    if scratch is None:
        return {"total_bytes": 0, "total_gib": 0.0, "by_component_bytes": {}}
    buffers = tuple(getattr(scratch, "buffers", ()))
    total = _sum_buffers(buffers)
    full_attn_kv = _sum_buffers(tuple(getattr(scratch, "full_key_caches", ())) + tuple(getattr(scratch, "full_value_caches", ())))
    full_attn_kv_scales = _sum_buffers(
        tuple(getattr(scratch, "full_k_scale_caches", ())) + tuple(getattr(scratch, "full_v_scale_caches", ()))
    )
    linear_state = _sum_buffers(tuple(getattr(scratch, "layer_conv_states", ())) + tuple(getattr(scratch, "layer_recurrent_states", ())))
    metadata = _sum_named_buffers(
        scratch,
        (
            "block_table",
            "position_buf",
            "context_buf",
            "cos_table_buf",
            "sin_table_buf",
        ),
    )
    named = {
        "full_attention_kv_cache": full_attn_kv,
        "full_attention_kv_scales": full_attn_kv_scales,
        "linear_attention_state": linear_state,
        "metadata_tables": metadata,
    }
    named["decode_workspace_other"] = max(0, total - sum(named.values()))
    return {
        "total_bytes": total,
        "total_gib": _bytes_to_gib(total),
        "max_positions": _maybe_int(getattr(scratch, "max_positions", None)),
        "block_table_len": _maybe_int(getattr(getattr(scratch, "block_table_tensor", None), "numel", None)),
        "kv_storage_dtype": getattr(getattr(scratch, "kv_storage_dtype", None), "value", None),
        "kv_scale_dtype": getattr(getattr(scratch, "kv_scale_dtype", None), "value", None),
        "by_component_bytes": named,
    }


def _session_buffer_breakdown(session: Qwen35GGUFResidentSession) -> dict[str, Any]:
    decode_runtime = _sum_named_buffers(
        session,
        (
            "_token_buf",
            "_hidden_a",
            "_hidden_b",
            "_logits_buf",
            "_lm_block_values",
            "_lm_block_indices",
            "_lm_out_index",
            "_lm_out_value",
        ),
    )
    prefill_token = _sum_named_buffers(session, ("_prefill_token_buf",))
    prefill_hidden = _sum_named_buffers(session, ("_prefill_hidden_a", "_prefill_hidden_b"))
    bulk_scratch_obj = getattr(session, "_bulk_prefill_scratch", None)
    bulk_scratch = _sum_buffers(getattr(bulk_scratch_obj, "buffers", ())) if bulk_scratch_obj is not None else 0
    total = _sum_buffers(getattr(session, "_buffers", ()))
    named = {
        "decode_logits_and_lm_head": decode_runtime,
        "prefill_token_buffer": prefill_token,
        "prefill_full_sequence_hidden": prefill_hidden,
        "bulk_prefill_scratch": bulk_scratch,
    }
    named["session_buffer_other"] = max(0, total - sum(named.values()))
    payload: dict[str, Any] = {
        "total_bytes": total,
        "total_gib": _bytes_to_gib(total),
        "by_component_bytes": named,
        "prefill_hidden_b_rows": _maybe_int(getattr(session, "_prefill_hidden_b_rows", None)),
    }
    if bulk_scratch_obj is not None:
        payload["bulk_prefill_scratch_rows"] = _maybe_int(getattr(bulk_scratch_obj, "rows", None))
        payload["bulk_prefill_scratch_capacity"] = _maybe_int(getattr(bulk_scratch_obj, "max_positions", None))
    return payload


def _sum_named_buffers(owner: object, names: tuple[str, ...]) -> int:
    return _sum_buffers(getattr(owner, name, None) for name in names)


def _sum_buffers(buffers) -> int:
    return sum(_buffer_nbytes(buffer) for buffer in buffers if buffer is not None)


def _buffer_nbytes(buffer: object | None) -> int:
    if buffer is None:
        return 0
    return int(getattr(buffer, "nbytes", 0))


def _add_bytes(target: dict[str, int], key: str, nbytes: int) -> None:
    target[key] = int(target.get(key, 0)) + int(nbytes)


def _maybe_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
