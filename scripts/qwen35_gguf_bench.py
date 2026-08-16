#!/usr/bin/env python3
# ruff: noqa: E402
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
import ctypes
import hashlib
import json
import os
import shlex
import statistics
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading.gguf import GGUFModelInfo, scan_gguf
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
_QUANT_ATTN_AOTRITON_MIN_TOKENS = {"gguf_ud_q3_k_m": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--token-id", type=int, default=9707, help="Repeated token id for fixed-length prompt")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=1)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=0,
        help=(
            "Resident context capacity; 0 uses prompt + warmup + decode + 1. "
            "A larger value enables no-short-mirror KV diagnostics."
        ),
    )
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
        "--prefill-attn-aotriton-min-tokens",
        type=int,
        default=None,
        help=(
            "AOTriton full-attention crossover; unset uses the quant policy "
            "(UD-Q3_K_M keeps exact native GQA)."
        ),
    )
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
        default=False,
        help=(
            "Explicitly benchmark the production state-bound GGUF decode graph. "
            "If the session cannot capture it, record the disabled reason and "
            "fall back to eager decode."
        ),
    )
    parser.add_argument("--graph-steps-per-replay", type=int, default=1)
    parser.add_argument(
        "--rocprof-selected-region",
        choices=("none", "prefill", "measured_decode_graph", "measured_decode"),
        default="none",
        help=(
            "Call roctxProfilerResume/Pause around one timed phase for "
            "rocprofv3 --selected-regions. Profiler-only; benchmark semantics are unchanged."
        ),
    )
    parser.add_argument(
        "--gpu-stage-timings",
        action="store_true",
        help=(
            "Record profiling-only same-stream device wall-clock stage timings. Requires "
            "--persistent-session and eager decode; throughput from this mode is diagnostic."
        ),
    )
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
    minimum_sequence_length = int(args.prompt_length) + int(args.warmup_decode_tokens) + int(args.decode_tokens) + 1
    if int(args.max_sequence_length) < 0:
        raise ValueError("--max-sequence-length must be non-negative")
    if int(args.max_sequence_length) and int(args.max_sequence_length) < minimum_sequence_length:
        raise ValueError(
            f"--max-sequence-length {int(args.max_sequence_length)} is below required {minimum_sequence_length}"
        )
    if args.warmup_runs < 0 or args.measured_runs <= 0:
        raise ValueError("--warmup-runs must be >=0 and --measured-runs must be positive")
    if args.graph_steps_per_replay <= 0:
        raise ValueError("--graph-steps-per-replay must be positive")
    if args.graph_replay_decode and args.decode_tokens % args.graph_steps_per_replay != 0:
        raise ValueError("--decode-tokens must be divisible by --graph-steps-per-replay")
    if args.gpu_stage_timings and not args.persistent_session:
        raise ValueError("--gpu-stage-timings requires --persistent-session")
    if args.gpu_stage_timings and args.graph_replay_decode and args.decode_tokens:
        raise ValueError("--gpu-stage-timings requires eager decode")
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
    if (
        args.prefill_attn_aotriton_min_tokens is not None
        and int(args.prefill_attn_aotriton_min_tokens) < 0
    ):
        raise ValueError("--prefill-attn-aotriton-min-tokens must be non-negative")

    compiler_version = _read_compiler_version(args.compiler_version_file) if args.compiler_version_file else None
    argv_payload = _exact_command_payload(sys.argv)
    gguf_info = scan_gguf(args.model)
    gguf_inventory = _gguf_tensor_inventory_summary(gguf_info)
    if args.force_bulk_prefill:
        use_bulk_prefill = True
    elif args.no_bulk_prefill:
        use_bulk_prefill = False
    else:
        use_bulk_prefill = None
    prompt_tokens = [int(args.token_id)] * int(args.prompt_length)
    max_sequence_length = int(args.max_sequence_length or minimum_sequence_length)
    default_aotriton_threshold = PrefillConfig().attn_aotriton_min_tokens
    aotriton_threshold = (
        _QUANT_ATTN_AOTRITON_MIN_TOKENS.get(args.quant, default_aotriton_threshold)
        if args.prefill_attn_aotriton_min_tokens is None
        else int(args.prefill_attn_aotriton_min_tokens)
    )
    prefill_config = PrefillConfig(
        linear_chunk_size=args.prefill_linear_chunk_size,
        moe_chunk_size=args.prefill_moe_chunk_size,
        full_attn_query_chunk_size=args.prefill_full_attn_query_chunk_size,
        full_attn_post_chunk_size=args.prefill_full_attn_post_chunk_size,
        full_attn_rope_chunk_size=args.prefill_full_attn_rope_chunk_size,
        attn_aotriton_min_tokens=aotriton_threshold,
        auto_tune_chunk_sizes=args.prefill_chunk_autotune,
        chunk_tune_memory_budget_gib=args.prefill_chunk_memory_budget_gib,
    )
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    roctx = _RoctxProfilerControl(enabled=args.rocprof_selected_region != "none")

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
            roctx=roctx,
            rocprof_selected_region=args.rocprof_selected_region,
            gpu_stage_timings=args.gpu_stage_timings,
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
                roctx=roctx,
                rocprof_selected_region=args.rocprof_selected_region,
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
    resolved_backends = {str(run["resolved_backend"]) for run in runs}
    target_arches = {str(run["target_arch"]) for run in runs}
    if len(resolved_backends) != 1 or len(target_arches) != 1:
        raise RuntimeError(
            "GGUF benchmark runs changed backend identity: "
            f"backends={sorted(resolved_backends)}, arches={sorted(target_arches)}"
        )
    resolved_backend = next(iter(resolved_backends))
    target_arch = next(iter(target_arches))
    provenance = _collect_benchmark_provenance(
        compiler_version=compiler_version,
        repo_root=REPO_ROOT,
        configured_backend="auto",
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant=str(args.quant),
        kv_dtype=kv_policy.storage_dtype.value,
        command=argv_payload["argv"],
        environment={
            "HIPENGINE_BACKEND": os.environ.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "HIPENGINE_GGUF_MOE_GRAPH": os.environ.get("HIPENGINE_GGUF_MOE_GRAPH"),
        },
        build_profile="qwen35_gguf_resident_bench",
        timing_protocol=(
            "resident GGUF session; prefill and decode timed separately; "
            "graph capture excluded from decode throughput and reported separately"
        ),
        warmups=int(args.warmup_runs),
        repetitions=int(args.measured_runs),
        profiler={"enabled": False, "reason": "host wall and residency benchmark"},
    )
    output = {
        "schema": 1,
        "model": str(args.model),
        "quant": args.quant,
        "backend": resolved_backend,
        "resolved_backend": resolved_backend,
        "target_arch": target_arch,
        "provenance": provenance,
        "argv": argv_payload["argv"],
        "command": argv_payload["command"],
        "gguf": gguf_inventory,
        "gguf_tensor_inventory_hash": gguf_inventory["tensor_inventory_hash"],
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
        "rocprof_selected_region": args.rocprof_selected_region,
        "gpu_stage_timings": bool(args.gpu_stage_timings),
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
            "--rocprof-selected-region wraps only the requested timed phase with ROCTX profiler resume/pause controls.",
            "--persistent-session creates one resident session and resets sequence state between warmup/measured runs, avoiding repeated GGUF load/decode-repack work. Historical artifacts used the default per-run session mode.",
        ],
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


def _exact_command_payload(argv: Sequence[object]) -> dict[str, Any]:
    argv_strings = [str(item) for item in argv]
    return {"argv": argv_strings, "command": shlex.join(argv_strings)}


def _gguf_tensor_inventory_summary(info: GGUFModelInfo) -> dict[str, Any]:
    return {
        "path": str(info.path),
        "version": int(info.version),
        "alignment": int(info.alignment),
        "architecture": info.architecture,
        "file_type": info.file_type,
        "file_type_name": info.file_type_name,
        "tensor_count": int(info.tensor_count),
        "total_tensor_nbytes": int(info.total_tensor_nbytes),
        "tensor_data_offset": int(info.tensor_data_offset),
        "tensor_inventory_hash_algorithm": "sha256",
        "tensor_inventory_hash": _gguf_tensor_inventory_hash(info),
    }


def _gguf_tensor_inventory_hash(info: GGUFModelInfo) -> str:
    digest = hashlib.sha256()
    _hash_fields(
        digest,
        (
            "hipengine.gguf_tensor_inventory.v1",
            str(int(info.version)),
            str(int(info.alignment)),
            str(int(info.tensor_data_offset)),
            str(int(info.tensor_count)),
            str(int(info.total_tensor_nbytes)),
        ),
    )
    for tensor in info.tensors:
        _hash_fields(
            digest,
            (
                tensor.name,
                ",".join(str(int(dim)) for dim in tensor.shape),
                ",".join(str(int(dim)) for dim in tensor.ggml_shape),
                str(int(tensor.ggml_type)),
                tensor.ggml_type_name,
                str(int(tensor.n_elements)),
                str(int(tensor.nbytes)),
                str(int(tensor.offset)),
                str(int(tensor.data_offset)),
                ",".join(str(int(dim)) for dim in tensor.byte_shape),
            ),
        )
    return digest.hexdigest()


def _hash_fields(digest: Any, fields: Sequence[str]) -> None:
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    digest.update(b";")


class _RoctxProfilerControl:
    """Open one timed phase for ``rocprofv3 --selected-regions``."""

    def __init__(self, *, enabled: bool) -> None:
        self._library = None
        self._resume = None
        self._pause = None
        if not enabled:
            return

        errors: list[str] = []
        for library_name in ("librocprofiler-sdk-roctx.so", "libroctx64.so"):
            try:
                library = ctypes.CDLL(library_name)
            except OSError as exc:
                errors.append(f"{library_name}: {exc}")
                continue
            resume = getattr(library, "roctxProfilerResume", None)
            pause = getattr(library, "roctxProfilerPause", None)
            if resume is None or pause is None:
                errors.append(f"{library_name}: missing roctxProfilerResume/Pause")
                continue
            self._library = library
            self._resume = resume
            self._pause = pause
            break

        if self._resume is None or self._pause is None:
            print(
                "warning: selected-region profiling controls are unavailable "
                f"({'; '.join(errors)}); rocprofv3 --selected-regions will emit no kernel rows",
                file=sys.stderr,
            )
            return

        self._resume.argtypes = [ctypes.c_uint64]
        self._resume.restype = ctypes.c_int
        self._pause.argtypes = [ctypes.c_uint64]
        self._pause.restype = ctypes.c_int

    def region(self, name: str, *, selected: str) -> "_RoctxProfilerRegion":
        return _RoctxProfilerRegion(self, enabled=(selected == name))

    def resume(self) -> None:
        if self._resume is not None:
            self._resume(0)

    def pause(self) -> None:
        if self._pause is not None:
            self._pause(0)


class _RoctxProfilerRegion:
    def __init__(self, control: _RoctxProfilerControl, *, enabled: bool) -> None:
        self.control = control
        self.enabled = bool(enabled)

    def __enter__(self) -> None:
        if self.enabled:
            self.control.resume()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.enabled:
            self.control.pause()


def _reset_existing_session(session: Any, runtime: HipRuntime) -> None:
    """Drain prior benchmark work before reusing and zeroing resident state."""

    runtime.stream_synchronize(0)
    session.reset()


def _rearm_reused_decode_graph(session: Any, graph: Any, runtime: HipRuntime) -> None:
    """Rearm a retained graph after reset+identical prefill restored its start state."""

    graph.rearm_replay_window()
    stream = runtime.stream_create()
    try:
        session._set_full_attention_position_device(session.position, stream=stream)
        runtime.stream_synchronize(stream)
    finally:
        runtime.stream_destroy(stream)


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
    roctx: "_RoctxProfilerControl",
    rocprof_selected_region: str,
    gpu_stage_timings: bool = False,
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
    session.select_prefill_quant(quant)
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
                roctx=roctx,
                rocprof_selected_region=rocprof_selected_region,
                gpu_stage_timings=bool(gpu_stage_timings and measured),
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
    roctx: "_RoctxProfilerControl",
    rocprof_selected_region: str,
    gpu_stage_timings: bool = False,
) -> dict[str, Any]:
    """Run one prefill/decode iteration on an existing resident session."""

    fastpath_safety = session.fastpath_safety.as_dict() if session.fastpath_safety is not None else None
    memory_snapshots: dict[str, Any] = {
        "after_load": _memory_snapshot("after_load", runtime, session),
        "before_reset": _memory_snapshot("before_reset", runtime, session),
    }
    _reset_existing_session(session, runtime)
    memory_snapshots["after_reset"] = _memory_snapshot("after_reset", runtime, session)

    generated_token_ids: list[int] = []
    final = None
    graph_capture_seconds = 0.0
    prefill_seconds = 0.0
    warmup_decode_seconds = 0.0
    decode_seconds = 0.0
    prefill_gpu_stage_timings_ms: dict[str, float] = {}
    decode_gpu_stage_timings_ms: dict[str, float] = {}
    decode_graph_transport_provenance = None
    decode_graph_disabled_reason = _decode_graph_disabled_reason(session, graph_replay_decode)
    effective_graph_replay_decode = bool(graph_replay_decode and decode_graph_disabled_reason is None)
    try:
        prefill_start = time.perf_counter()
        with roctx.region("prefill", selected=rocprof_selected_region):
            first = session.prefill(
                prompt_tokens,
                use_bulk=use_bulk_prefill,
                bulk_attention_mode=bulk_attention_mode,
                return_logits=False,
                record_gpu_stage_timings=gpu_stage_timings,
            )
        prefill_seconds = time.perf_counter() - prefill_start
        if gpu_stage_timings:
            prefill_gpu_stage_timings_ms = dict(session.last_prefill_gpu_stage_timings_ms)
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
                # reset()+identical prefill+warmup reconstructs the captured
                # recurrent/KV/hidden/token state. Rearm host replay accounting
                # and the device position scalar before the retained launch.
                _rearm_reused_decode_graph(session, graph, runtime)
            try:
                decode_start = time.perf_counter()
                with roctx.region("measured_decode_graph", selected=rocprof_selected_region):
                    graph.replay(decode_tokens)
                decode_seconds = time.perf_counter() - decode_start
                if decode_graph_recorded_tokens:
                    generated_token_ids.extend(graph.read_generated_token_ids(decode_tokens))
                final = graph.read_sample()
                if not decode_graph_recorded_tokens and final is not None:
                    generated_token_ids.append(final.token_id)
                decode_graph_transport_provenance = graph.transport_provenance()
            finally:
                if not retained_graph:
                    graph.close()
        else:
            decode_graph_reused = False
            decode_graph_recorded_tokens = False
            decode_start = time.perf_counter()
            with roctx.region("measured_decode", selected=rocprof_selected_region):
                for step_index in range(decode_tokens):
                    final = session.step(
                        next_token,
                        return_logits=(step_index == decode_tokens - 1),
                        record_gpu_stage_timings=gpu_stage_timings,
                    )
                    if gpu_stage_timings:
                        for name, ms in session.last_decode_gpu_stage_timings_ms.items():
                            decode_gpu_stage_timings_ms[name] = (
                                decode_gpu_stage_timings_ms.get(name, 0.0) + float(ms)
                            )
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
        "resolved_backend": str(session.backend),
        "target_arch": str(session.runner.target_arch),
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
        "requested_graph_replay_decode": bool(graph_replay_decode),
        "effective_graph_replay_decode": bool(effective_graph_replay_decode),
        "decode_graph_reused": bool(decode_graph_reused),
        "decode_graph_recorded_tokens": bool(decode_graph_recorded_tokens),
        "decode_graph_transport_provenance": decode_graph_transport_provenance,
        "rocprof_selected_region": rocprof_selected_region,
        "host_token_embedding_enabled": bool(getattr(session, "host_token_embedding_enabled", False)),
        "host_token_embedding_reason": getattr(session, "host_token_embedding_reason", None),
        "host_token_embedding_mapped": _mapped_host_embedding_audit(session),
        "allocation_arena": session.allocation_arena_audit(),
        "decode_graph_disabled_reason": decode_graph_disabled_reason,
        "gpu_stage_timings_ms": {
            "enabled": bool(gpu_stage_timings),
            "method": "device_wall_clock_marker" if gpu_stage_timings else None,
            "prefill": prefill_gpu_stage_timings_ms,
            "decode": decode_gpu_stage_timings_ms,
            "decode_tokens": int(decode_tokens),
        },
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
    roctx: "_RoctxProfilerControl",
    rocprof_selected_region: str,
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
    session.select_prefill_quant(quant)
    fastpath_safety = session.fastpath_safety.as_dict() if session.fastpath_safety is not None else None
    resolved_backend = str(session.backend)
    target_arch = str(session.runner.target_arch)
    memory_snapshots["after_load"] = _memory_snapshot("after_load", runtime, session)

    generated_token_ids: list[int] = []
    final = None
    graph_capture_seconds = 0.0
    decode_graph_transport_provenance = None
    mapped_host_embedding = _mapped_host_embedding_audit(session)
    decode_graph_disabled_reason = _decode_graph_disabled_reason(session, graph_replay_decode)
    effective_graph_replay_decode = bool(graph_replay_decode and decode_graph_disabled_reason is None)
    try:
        prefill_start = time.perf_counter()
        with roctx.region("prefill", selected=rocprof_selected_region):
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
                with roctx.region("measured_decode_graph", selected=rocprof_selected_region):
                    graph.replay(decode_tokens)
                decode_seconds = time.perf_counter() - decode_start
                generated_token_ids.extend(graph.read_generated_token_ids(decode_tokens))
                final = graph.read_sample()
                decode_graph_transport_provenance = graph.transport_provenance()
            finally:
                graph.close()
        else:
            decode_start = time.perf_counter()
            with roctx.region("measured_decode", selected=rocprof_selected_region):
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
        "resolved_backend": resolved_backend,
        "target_arch": target_arch,
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
        "requested_graph_replay_decode": bool(graph_replay_decode),
        "effective_graph_replay_decode": bool(effective_graph_replay_decode),
        "decode_graph_transport_provenance": decode_graph_transport_provenance,
        "host_token_embedding_enabled": bool(getattr(session, "host_token_embedding_enabled", False)),
        "host_token_embedding_reason": getattr(session, "host_token_embedding_reason", None),
        "host_token_embedding_mapped": mapped_host_embedding,
        "allocation_arena": session.allocation_arena_audit(),
        "decode_graph_disabled_reason": decode_graph_disabled_reason,
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


def _mapped_host_embedding_audit(session: Any) -> dict[str, Any]:
    runner = getattr(session, "runner", None)
    mapped = getattr(runner, "host_token_embedding_mapped_weight", None)
    if mapped is None:
        return {"enabled": False, "nbytes": 0, "device_ptr": None}
    allocation = mapped.allocation("raw")
    return {
        "enabled": True,
        "nbytes": int(allocation.buffer.nbytes),
        "device_ptr": int(allocation.tensor.ptr),
        "owns_buffer": bool(allocation.owns_buffer),
        "storage": "hip_registered_gguf_mmap",
    }


def _decode_graph_disabled_reason(session: Any, requested: bool) -> str | None:
    if not requested:
        return None
    if not callable(getattr(session, "capture_decode_graph", None)):
        return "capture_decode_graph_unavailable"
    if getattr(session, "host_token_embedding_enabled", False) and not callable(
        getattr(session, "_device_token_embedding_weight", None)
    ):
        return "host_token_embedding"
    return None


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
    weights = None if session.runner is None else session.runner.weights
    arena = None if weights is None else weights.allocation_arena
    if arena is not None:
        total += int(arena.capacity_bytes)
    if weights is not None:
        for weight in weights.weights:
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
    logical_bytes = int(weights["total_bytes"]) + int(decode_scratch["total_bytes"]) + int(session_buffers["total_bytes"])
    total_bytes = _owned_device_bytes(session)
    return {
        "total_bytes": total_bytes,
        "total_gib": _bytes_to_gib(total_bytes),
        "logical_requested_bytes": logical_bytes,
        "allocation_arena_padding_bytes": max(0, total_bytes - logical_bytes),
        "families": {
            "weights": weights,
            "decode_scratch": decode_scratch,
            "session_buffers": session_buffers,
        },
        "notes": [
            "weights counts unique resident GGUF logical allocations; selective arena views and dedicated large owners are reported separately.",
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
    physical_owner_bytes = 0
    physical_owner_count = 0
    seen_ptrs: set[int] = set()
    weights = None if session.runner is None else session.runner.weights
    arena = None if weights is None else weights.allocation_arena
    if arena is not None:
        physical_owner_bytes += int(arena.capacity_bytes)
        physical_owner_count += 1
    if weights is not None:
        for weight in weights.weights:
            spec = weight.spec
            quant_key = str(spec.quant_key)
            layout = str(spec.layout)
            for allocation_name, allocation in weight.allocations.items():
                ptr = int(allocation.buffer.ptr)
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)
                nbytes = _buffer_nbytes(allocation.buffer)
                total += nbytes
                count += 1
                if allocation.owns_buffer:
                    physical_owner_bytes += nbytes
                    physical_owner_count += 1
                _add_bytes(by_quant, quant_key, nbytes)
                _add_bytes(by_layout, layout, nbytes)
                _add_bytes(by_quant_layout, f"{quant_key}:{layout}", nbytes)
                _add_bytes(by_allocation, str(allocation_name), nbytes)
    return {
        "total_bytes": total,
        "total_gib": _bytes_to_gib(total),
        "allocation_count": count,
        "physical_owner_count": physical_owner_count,
        "physical_owner_bytes": physical_owner_bytes,
        "alignment_padding_bytes": max(0, physical_owner_bytes - total),
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
    full_attn_bf16_mirrors = _sum_buffers(
        tuple(getattr(scratch, "full_bf16_mirror_key_caches", ()))
        + tuple(getattr(scratch, "full_bf16_mirror_value_caches", ()))
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
        "full_attention_bf16_mirrors": full_attn_bf16_mirrors,
        "linear_attention_state": linear_state,
        "metadata_tables": metadata,
    }
    named["decode_workspace_other"] = max(0, total - sum(named.values()))
    return {
        "total_bytes": total,
        "total_gib": _bytes_to_gib(total),
        "allocation_mode": str(getattr(scratch, "allocation_mode", "dedicated")),
        "physical_owner_count": len(buffers),
        "max_positions": _maybe_int(getattr(scratch, "max_positions", None)),
        "block_table_len": _maybe_int(getattr(getattr(scratch, "block_table_tensor", None), "numel", None)),
        "kv_storage_dtype": getattr(getattr(scratch, "kv_storage_dtype", None), "value", None),
        "kv_storage_layout": getattr(scratch, "kv_storage_layout", "uniform"),
        "kv_scale_dtype": getattr(getattr(scratch, "kv_scale_dtype", None), "value", None),
        "kv_scale_granularity": getattr(scratch, "kv_scale_granularity", None),
        "int8_kv_value_bf16": bool(getattr(scratch, "int8_kv_value_bf16", False)),
        "by_component_bytes": named,
    }


def _bulk_prefill_scratch_census(scratch: object | None) -> dict[str, Any]:
    """Describe physical owners and logical fields in the bulk-prefill scratch.

    ``_GGUFFullAttentionPrefillScratch`` may place mutually-exclusive logical
    fields into one liveness arena.  The ordinary tracked allocator therefore
    reports the physically-owned bytes correctly but cannot explain which
    row-scaled fields drove that arena.  This read-only census preserves both
    views without treating aliased field views as additional ownership.
    """

    if scratch is None:
        return {
            "allocation_mode": None,
            "rows": None,
            "max_positions": None,
            "physical_owner_bytes": 0,
            "logical_field_bytes": 0,
            "logical_alias_overhead_bytes": 0,
            "by_field_bytes": {},
            "allocation_offsets": {},
            "allocation_lifetimes": {},
            "allocation_groups": {},
            "allocation_inplace_aliases": {},
            "allocation_subranges": {},
        }

    head_major_names = {"head_major_key_cache", "head_major_value_cache"}
    head_major_bytes = _sum_named_buffers(scratch, tuple(sorted(head_major_names)))
    physical_owner_bytes = max(
        0,
        _sum_buffers(getattr(scratch, "buffers", ())) - head_major_bytes,
    )
    by_field: dict[str, int] = {}
    attributes = vars(scratch) if hasattr(scratch, "__dict__") else {}
    for name, value in attributes.items():
        if name in head_major_names or name == "buffers" or name.endswith("_tensor"):
            continue
        if not hasattr(value, "ptr"):
            continue
        nbytes = _buffer_nbytes(value)
        if nbytes > 0:
            by_field[str(name)] = nbytes
    by_field = dict(sorted(by_field.items(), key=lambda item: (-item[1], item[0])))
    logical_field_bytes = sum(by_field.values())

    raw_offsets = getattr(scratch, "allocation_offsets", {}) or {}
    allocation_offsets = {
        str(name): {"offset_bytes": int(offset), "nbytes": int(nbytes)}
        for name, (offset, nbytes) in sorted(raw_offsets.items())
    }
    raw_lifetimes = getattr(scratch, "allocation_lifetimes", {}) or {}
    allocation_lifetimes = {
        str(name): [
            {
                "route": str(route),
                "start_stage": int(start),
                "end_stage": int(end),
            }
            for route, start, end in lifetimes
        ]
        for name, lifetimes in sorted(raw_lifetimes.items())
    }
    raw_groups = getattr(scratch, "allocation_groups", {}) or {}
    allocation_groups = {
        str(name): str(group)
        for name, group in sorted(raw_groups.items())
    }
    raw_inplace_aliases = getattr(scratch, "allocation_inplace_aliases", {}) or {}
    allocation_inplace_aliases = {
        str(name): str(source)
        for name, source in sorted(raw_inplace_aliases.items())
    }
    raw_subranges = getattr(scratch, "allocation_subranges", {}) or {}
    allocation_subranges = {
        str(name): [
            {
                "relative_offset_bytes": int(relative_offset),
                "nbytes": int(nbytes),
                "lifetimes": [
                    {
                        "route": str(route),
                        "start_stage": int(start),
                        "end_stage": int(end),
                    }
                    for route, start, end in lifetimes
                ],
            }
            for relative_offset, nbytes, lifetimes in subranges
        ]
        for name, subranges in sorted(raw_subranges.items())
    }
    rows = _maybe_int(getattr(scratch, "rows", None))
    return {
        "allocation_mode": getattr(scratch, "allocation_mode", None),
        "rows": rows,
        "max_positions": _maybe_int(getattr(scratch, "max_positions", None)),
        "physical_owner_bytes": physical_owner_bytes,
        "physical_owner_gib": _bytes_to_gib(physical_owner_bytes),
        "logical_field_bytes": logical_field_bytes,
        "logical_field_gib": _bytes_to_gib(logical_field_bytes),
        "logical_alias_overhead_bytes": max(0, logical_field_bytes - physical_owner_bytes),
        "by_field_bytes": by_field,
        "by_field_bytes_per_row": {
            name: (float(nbytes) / rows if rows else None)
            for name, nbytes in by_field.items()
        },
        "allocation_offsets": allocation_offsets,
        "allocation_lifetimes": allocation_lifetimes,
        "allocation_groups": allocation_groups,
        "allocation_inplace_aliases": allocation_inplace_aliases,
        "allocation_subranges": allocation_subranges,
        "notes": [
            "physical_owner_bytes counts allocator-owned buffers and excludes the separately reported head-major K/V pair.",
            "logical_field_bytes intentionally counts aliased views separately; it is a sizing census, not additional residency.",
        ],
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
    bulk_scratch_total = (
        _sum_buffers(getattr(bulk_scratch_obj, "buffers", ()))
        if bulk_scratch_obj is not None
        else 0
    )
    head_major_kv_scratch = (
        _sum_buffers(
            (
                getattr(bulk_scratch_obj, "head_major_key_cache", None),
                getattr(bulk_scratch_obj, "head_major_value_cache", None),
            )
        )
        if bulk_scratch_obj is not None
        else 0
    )
    bulk_scratch = max(0, bulk_scratch_total - head_major_kv_scratch)
    bulk_scratch_census = _bulk_prefill_scratch_census(bulk_scratch_obj)
    total = _sum_buffers(getattr(session, "_buffers", ()))
    named = {
        "decode_logits_and_lm_head": decode_runtime,
        "prefill_token_buffer": prefill_token,
        "prefill_full_sequence_hidden": prefill_hidden,
        "bulk_prefill_scratch": bulk_scratch,
        "aotriton_head_major_kv_scratch": head_major_kv_scratch,
    }
    named["session_buffer_other"] = max(0, total - sum(named.values()))
    payload: dict[str, Any] = {
        "total_bytes": total,
        "total_gib": _bytes_to_gib(total),
        "by_component_bytes": named,
        "prefill_hidden_b_rows": _maybe_int(getattr(session, "_prefill_hidden_b_rows", None)),
        "bulk_prefill_scratch_census": bulk_scratch_census,
    }
    if bulk_scratch_obj is not None:
        payload["bulk_prefill_scratch_rows"] = _maybe_int(getattr(bulk_scratch_obj, "rows", None))
        payload["bulk_prefill_scratch_capacity"] = _maybe_int(getattr(bulk_scratch_obj, "max_positions", None))
        payload["aotriton_head_major_kv_capacity"] = _maybe_int(
            getattr(bulk_scratch_obj, "head_major_kv_capacity", None)
        )
    return payload


def _sum_named_buffers(owner: object, names: tuple[str, ...]) -> int:
    return _sum_buffers(getattr(owner, name, None) for name in names)


def _sum_buffers(buffers) -> int:
    total = 0
    seen_ptrs: set[int] = set()
    for buffer in buffers:
        if buffer is None:
            continue
        ptr = int(getattr(buffer, "ptr", 0))
        if ptr != 0 and ptr in seen_ptrs:
            continue
        if ptr != 0:
            seen_ptrs.add(ptr)
        total += _buffer_nbytes(buffer)
    return total


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


def _collect_benchmark_provenance(
    *,
    compiler_version: str | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reuse precomputed hipcc text instead of probing inside profiled runs."""

    if compiler_version is not None:
        kwargs["hipcc_version"] = compiler_version
    return collect_artifact_provenance(**kwargs)


def _read_compiler_version(path: Path) -> str:
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"compiler version file {path} is empty")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
