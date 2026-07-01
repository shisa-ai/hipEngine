#!/usr/bin/env python3
"""rocprofv3 host/GPU split for the GGUF resident MTP draft chain.

This diagnostic profiles the retained llama-compat draft shape directly:
B2 resident device-chain drafting, optional Q6_K q8_1/dp4a top-1 lm-head,
and the same MTP KV cache lifecycle used by ``gguf_mtp_bench.py``.  It marks
only the draft proposal window with ROCTX, so the target step used to refresh
the next hidden seed does not enter the kernel totals.

Parent mode warm-builds outside rocprof, pins ``HIPENGINE_COMPILER_VERSION_FILE``,
then runs child mode under ``rocprofv3 --kernel-trace --marker-trace`` with
``require_cached_build=True``.  Child mode emits one ROCTX range per measured
draft chain; parent filters the kernel CSV to those ranges and compares summed
kernel time against the child-measured host wall.

Diagnostic only: this emits no retained speed claim.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_mtp_bench import _rope_tables, llama_cpp_mtp_catchup_rows
from scripts.gguf_mtp_verifier_rocprof import (
    _Roctx,
    _default_roctx_sdk,
    _filter_kernels_by_windows,
    _prepare_roctx_override,
    _read_kernels,
    _read_marker_windows,
    _roctx_sdk_dep_paths,
    _single_file,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPT_IDS = "760,4087,369,220,16,17,18,19"
DRAFT_MARKER_PREFIX = "gguf_mtp_draft_chain_"


def _parse_prompt_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _apply_route_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")
    if bool(args.q6_top1_dp4a):
        os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A"] = "1"
    if bool(args.router_row_parallel):
        os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL"] = "1"
    os.environ["HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS"] = str(int(args.q6_top1_stage1_threads))
    os.environ["HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE"] = str(args.q6_top1_stage1_shape)
    selected_down_x8 = str(args.selected_down_x8_repack)
    if selected_down_x8 != "off":
        os.environ["HIPENGINE_GGUF_SELECTED_X8_REPACK"] = selected_down_x8


def _resident_q8_shared_dual_enabled() -> bool:
    raw = os.environ.get("HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL", "1")
    return raw.strip().lower() not in {"0", "false", "off", "no", ""}


def _sum_stage_timings(rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return dict(sorted(totals.items()))


def _copy_pending_hidden_seed(session: Any) -> np.ndarray:
    from hipengine.core.hip import HipMemcpyKind

    hidden_size = int(session.runner.hidden_size)
    hidden_seed = np.empty((1, hidden_size), dtype=np.float32)
    session.runtime.memcpy(
        hidden_seed.ctypes.data,
        session.fp32_hidden_seed_ptr(),
        hidden_size * 4,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return np.ascontiguousarray(hidden_seed, dtype=np.float32)


def _serial_prefill_with_hidden_trace(session: Any, prompt_ids: list[int]) -> tuple[int, np.ndarray, np.ndarray]:
    session.reset()
    hidden_rows: list[np.ndarray] = []
    hidden_ptr: int | None = None
    for token_id in prompt_ids:
        hidden_ptr = session._run_token_to_final_hidden(  # noqa: SLF001 - diagnostic parity hook
            int(token_id),
            position=session.position,
            capture_hidden_seed_fp32=True,
        )
        session._position += 1  # noqa: SLF001 - mirrors Qwen35GGUFResidentSession.prefill serial path
        hidden_rows.append(_copy_pending_hidden_seed(session)[0].copy())
    if hidden_ptr is None:
        raise RuntimeError("prompt produced no hidden row")
    result = session._sample_from_hidden(hidden_ptr, return_logits=False)  # noqa: SLF001
    return (
        int(result.token_id),
        _copy_pending_hidden_seed(session),
        np.ascontiguousarray(np.stack(hidden_rows, axis=0), dtype=np.float32),
    )


def _load_mtp_inputs(model: Path) -> tuple[dict[str, tuple[np.ndarray, object, object]], np.ndarray, np.ndarray, np.ndarray]:
    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data

    reader = GGUFReader(model)
    weights: dict[str, tuple[np.ndarray, object, object]] = {}
    for tensor in reader.info.tensors:
        if "blk.40" in tensor.name or tensor.name in {"output.weight", "token_embd.weight"}:
            weights[tensor.name] = (reader.tensor_data(tensor.name), tensor.ggml_type, tensor.shape)

    token_raw = weights["token_embd.weight"][0]
    token_qtype = GGMLQuantizationType(weights["token_embd.weight"][1])
    token_embd_f32 = dequantize_gguf_data(token_raw, token_qtype).astype(np.float32)

    meta = reader.info.metadata
    rope_dim = int(meta.get("qwen35moe.rope.dimension_count", 64))
    rope_base = float(meta.get("qwen35moe.rope.freq_base", 10000000.0))
    rope_cos, rope_sin = _rope_tables(max_positions=262144, rotary_dim=rope_dim, base=rope_base)
    return weights, np.ascontiguousarray(token_embd_f32), rope_cos, rope_sin


def _run_child(args: argparse.Namespace) -> int:
    _apply_route_env(args)
    if args.compiler_version_file:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.memory import free, malloc
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner

    prompt_ids = _parse_prompt_ids(args.prompt_ids)
    weights, token_embd_f32, rope_cos, rope_sin = _load_mtp_inputs(args.model)
    roctx = _Roctx()
    host_ms: list[float] = []
    stage_timings: list[dict[str, float]] = []
    draft_token_rows: list[list[int]] = []
    kv_lengths: list[int] = []
    mtp_device_key_cache = None
    mtp_device_value_cache = None

    with Qwen35GGUFResidentSession(
        args.model,
        max_sequence_length=args.max_seq,
        require_cached_build=args.require_cached,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        runtime = session.runtime
        draft = Qwen35GGUFResidentMTPDraftRunner(
            weights,
            token_embd_f32,
            runtime=runtime,
            vocab_cap=int(args.vocab_cap or weights["output.weight"][0].shape[0]),
            device_chain_enabled=True,
            prewarm_device_chain=True,
            sync_stage_timings=bool(args.sync_stage_timings),
            require_cached_build=bool(args.require_cached),
        )
        try:
            prev_token, pending_hidden_seed, prompt_hidden_rows = _serial_prefill_with_hidden_trace(session, prompt_ids)
            seq_position = int(session.position)
            context_tokens, context_hidden_rows = llama_cpp_mtp_catchup_rows(prompt_ids, prompt_hidden_rows)
            kv_heads = int(draft.num_kv_heads)
            qk_head_dim = int(draft.qk_head_dim)
            capacity = max(
                1,
                len(context_tokens)
                + (int(args.steps) + int(args.warmup) + 2) * (2 * int(args.draft_n_max) + 2)
                + 4,
            )
            key_nbytes = capacity * kv_heads * qk_head_dim * 4
            value_nbytes = capacity * kv_heads * qk_head_dim * 4
            mtp_device_key_cache = malloc(key_nbytes, runtime=runtime)
            mtp_device_value_cache = malloc(value_nbytes, runtime=runtime)
            mtp_device_kv_len = 0
            if context_tokens:
                positions = np.arange(len(context_tokens), dtype=np.int64)
                mtp_device_kv_len = draft.write_kv_rows(
                    context_hidden_rows,
                    np.asarray(context_tokens, dtype=np.int64),
                    positions=positions,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    dense_key_cache=mtp_device_key_cache,
                    dense_value_cache=mtp_device_value_cache,
                    dense_cache_len=0,
                )

            def advance_target() -> tuple[int, np.ndarray, int]:
                nonlocal seq_position
                result = session.step(
                    int(prev_token),
                    return_logits=False,
                    capture_hidden_seed_fp32=True,
                )
                runtime.device_synchronize()
                seq_position += 1
                return int(result.token_id), _copy_pending_hidden_seed(session), seq_position

            def run_draft_window(index: int | None) -> None:
                nonlocal mtp_device_kv_len
                base_len = int(mtp_device_kv_len)
                runtime.device_synchronize()
                if index is not None:
                    roctx.push(f"{DRAFT_MARKER_PREFIX}{index}")
                t0 = time.perf_counter()
                draft_tokens, _topk_rows, mtp_device_kv_len = draft.propose_chain(
                    pending_hidden_seed,
                    start_token=int(prev_token),
                    start_position=int(seq_position),
                    draft_n_max=int(args.draft_n_max),
                    top_k=int(args.top_k),
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    dense_key_cache=mtp_device_key_cache,
                    dense_value_cache=mtp_device_value_cache,
                    dense_cache_len=mtp_device_kv_len,
                    draft_p_min=0.0,
                    record_top1_probs=False,
                    record_stage_timings=bool(args.record_stage_timings),
                )
                runtime.device_synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if index is not None:
                    roctx.pop()
                    host_ms.append(elapsed_ms)
                    draft_token_rows.append([int(token) for token in draft_tokens])
                    kv_lengths.append(int(mtp_device_kv_len))
                    if args.record_stage_timings:
                        stage_timings.append(dict(draft.last_stage_timings_ms))
                mtp_device_kv_len = min(int(mtp_device_kv_len), base_len + 1)

            for _ in range(int(args.warmup)):
                run_draft_window(None)
                prev_token, pending_hidden_seed, _seq_position = advance_target()
            for index in range(int(args.steps)):
                run_draft_window(index)
                prev_token, pending_hidden_seed, _seq_position = advance_target()
        finally:
            draft.close()
            if mtp_device_value_cache is not None:
                free(mtp_device_value_cache, runtime=runtime)
            if mtp_device_key_cache is not None:
                free(mtp_device_key_cache, runtime=runtime)

    payload = {
        "schema": "hipengine.gguf_mtp_draft_rocprof.child.v1",
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "draft_n_max": int(args.draft_n_max),
        "top_k": int(args.top_k),
        "q6_top1_dp4a": bool(args.q6_top1_dp4a),
        "q6_top1_stage1_threads": int(args.q6_top1_stage1_threads),
        "q6_top1_stage1_shape": str(args.q6_top1_stage1_shape),
        "q8_shared_dual": _resident_q8_shared_dual_enabled(),
        "router_row_parallel": bool(args.router_row_parallel),
        "router_row_parallel_env": os.environ.get("HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL"),
        "selected_down_x8_repack": str(args.selected_down_x8_repack),
        "host_ms": host_ms,
        "avg_host_ms": sum(host_ms) / len(host_ms) if host_ms else 0.0,
        "draft_tokens": draft_token_rows,
        "mtp_device_kv_lengths_after_draft": kv_lengths,
        "stage_timings_ms": stage_timings if stage_timings else None,
        "stage_timing_totals_ms": _sum_stage_timings(stage_timings) if stage_timings else None,
    }
    if args.child_json:
        args.child_json.parent.mkdir(parents=True, exist_ok=True)
        args.child_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[gguf-mtp-draft] steps={args.steps} avg_host_ms={payload['avg_host_ms']:.3f}")
    return 0


def _run_parent(args: argparse.Namespace) -> int:
    rocprofv3 = shutil.which(args.rocprofv3) or args.rocprofv3
    raw_root = Path(args.raw_root)
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    cvf = Path(args.compiler_version_file)
    if not cvf.exists():
        cvf.parent.mkdir(parents=True, exist_ok=True)
        ver = subprocess.run(["hipcc", "--version"], capture_output=True, text=True, check=False)
        cvf.write_text(ver.stdout or "hipcc-unknown\n", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")
    env.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    env["HIPENGINE_COMPILER_VERSION_FILE"] = str(cvf)
    roctx_override = _prepare_roctx_override(args.roctx_sdk)
    dep_paths = _roctx_sdk_dep_paths(args.roctx_sdk)
    ld_prefix = os.pathsep.join([str(roctx_override), *(str(path) for path in dep_paths)])
    env["LD_LIBRARY_PATH"] = f"{ld_prefix}:{env.get('LD_LIBRARY_PATH', '')}"

    child_json = raw_root / "child.json"
    child_base = [
        sys.executable,
        str(Path(__file__)),
        "--child",
        "--model",
        str(args.model),
        "--prompt-ids",
        str(args.prompt_ids),
        "--steps",
        str(int(args.steps)),
        "--warmup",
        str(int(args.warmup)),
        "--max-seq",
        str(int(args.max_seq)),
        "--draft-n-max",
        str(int(args.draft_n_max)),
        "--top-k",
        str(int(args.top_k)),
        "--vocab-cap",
        str(int(args.vocab_cap)),
        "--q6-top1-stage1-threads",
        str(int(args.q6_top1_stage1_threads)),
        "--q6-top1-stage1-shape",
        str(args.q6_top1_stage1_shape),
        "--selected-down-x8-repack",
        str(args.selected_down_x8_repack),
        "--compiler-version-file",
        str(cvf),
        "--child-json",
        str(child_json),
    ]
    if args.q6_top1_dp4a:
        child_base.append("--q6-top1-dp4a")
    if args.router_row_parallel:
        child_base.append("--router-row-parallel")
    if args.record_stage_timings:
        child_base.append("--record-stage-timings")
    if args.sync_stage_timings:
        child_base.append("--sync-stage-timings")

    if not args.skip_warmbuild:
        print("[gguf-mtp-draft-rocprof] warm-build pass (no profiler)...", flush=True)
        subprocess.run(child_base, cwd=REPO_ROOT, env=env, check=True)

    trace_dir = raw_root / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    rocprof_cmd = [
        rocprofv3,
        "--kernel-trace",
        "--marker-trace",
        "--output-format",
        "csv",
        "-d",
        str(trace_dir),
        "--",
        *child_base,
        "--require-cached",
    ]
    print(f"[gguf-mtp-draft-rocprof] {' '.join(rocprof_cmd)}", flush=True)
    subprocess.run(rocprof_cmd, cwd=REPO_ROOT, env=env, check=True)

    kernel_csv = _single_file(trace_dir, "*_kernel_trace.csv")
    marker_csv = _single_file(trace_dir, "*_marker_api_trace.csv")
    child = json.loads(child_json.read_text(encoding="utf-8"))
    windows = _read_marker_windows(marker_csv, DRAFT_MARKER_PREFIX)
    if len(windows) != int(args.steps):
        raise SystemExit(f"expected {args.steps} marker windows, found {len(windows)}")
    kernels = _read_kernels(kernel_csv)
    selected = _filter_kernels_by_windows(kernels, [(start, end) for _idx, start, end in windows])
    summary = _summarize_draft(selected, host_ms=[float(x) for x in child["host_ms"]], top=int(args.top))
    artifact = {
        "schema": "hipengine.gguf_mtp_draft_rocprof.v1",
        "date": date.today().isoformat(),
        "status": "diagnostic_retained",
        "performance_claim": False,
        "purpose": "Current GGUF MTP resident draft-chain host/GPU split.",
        "model": str(args.model),
        "hardware": "AMD Radeon 8060S / Ryzen AI Max+ 395 (gfx1151)",
        "draft_n_max": int(args.draft_n_max),
        "top_k": int(args.top_k),
        "q6_top1_dp4a": bool(args.q6_top1_dp4a),
        "q6_top1_stage1_threads": int(args.q6_top1_stage1_threads),
        "q6_top1_stage1_shape": str(args.q6_top1_stage1_shape),
        "q8_shared_dual": _resident_q8_shared_dual_enabled(),
        "router_row_parallel": bool(args.router_row_parallel),
        "selected_down_x8_repack": str(args.selected_down_x8_repack),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "prompt_ids": _parse_prompt_ids(args.prompt_ids),
        "marker_prefix": DRAFT_MARKER_PREFIX,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "rocprof_command": " ".join(rocprof_cmd),
        "raw_root": str(raw_root),
        "kernel_trace_csv": str(kernel_csv),
        "marker_trace_csv": str(marker_csv),
        "child": child,
        "summary": summary,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"[gguf-mtp-draft-rocprof] wrote {args.out}")
    _print_summary(summary)
    _print_stage_timings(child)
    return 0


def _family(name: str) -> str:
    value = re.sub(r"^void\s+", "", name.strip()).replace("(anonymous namespace)::", "")
    value = re.sub(r"<[^>]*>", "", value).split("(")[0]
    return re.sub(r"_kernel$", "", value).strip()


def _bucket(family: str) -> str:
    f = family.lower()
    if "top1_stage2_gather" in f or ("q6" in f and ("top1" in f or "pack8" in f)):
        return "draft_lm_head_q6_top1"
    if f.startswith(("q8_0", "gguf_q8_0")) or f == "gguf_k_prefill_out":
        return "draft_dense_shared_gemv"
    if f.startswith(("q4_k_selected", "gguf_q4_k_selected")):
        return "draft_moe_gate_up"
    if f.startswith(("gguf_k_selected", "q5_k_selected")):
        return "draft_moe_down"
    if f == "hipengine_mtp_linear_f32":
        return "draft_f32_router_gate_linear"
    if "router" in f:
        return "draft_moe_router"
    if "dense_attn" in f or "sigmoid_gate" in f:
        return "draft_attention_core"
    if "rmsnorm" in f or "rope" in f or "rotary" in f or "cast" in f or "quantize" in f:
        return "draft_norm_cast_quant_rope"
    if any(key in f for key in ("silu", "weighted_sum", "combine")):
        return "draft_moe_combine_silu"
    if f.startswith("mtp_"):
        return "draft_mtp_elementwise"
    if any(key in f for key in ("copybuffer", "fillbuffer", "memcpy", "memset")):
        return "memcpy_fill"
    return "other"


def _summarize_draft(rows: list[dict[str, Any]], *, host_ms: list[float], top: int) -> dict[str, Any]:
    total_kernel_ms = sum(float(row["duration_ns"]) for row in rows) / 1e6
    total_host_ms = sum(host_ms)
    buckets: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    families: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    for row in rows:
        ms = float(row["duration_ns"]) / 1e6
        family = _family(str(row["kernel"]))
        bucket = _bucket(family)
        families[family][0] += 1
        families[family][1] += ms
        buckets[bucket][0] += 1
        buckets[bucket][1] += ms

    def emit(table: dict[str, list[float]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key, (calls, ms) in sorted(table.items(), key=lambda item: -item[1][1]):
            out.append(
                {
                    "name": key,
                    "calls": int(calls),
                    "total_ms": ms,
                    "pct_kernel": (ms / total_kernel_ms * 100.0) if total_kernel_ms else 0.0,
                    "ms_per_step": ms / len(host_ms) if host_ms else 0.0,
                    "us_per_call": (ms * 1000.0 / calls) if calls else 0.0,
                }
            )
        return out

    return {
        "steps": len(host_ms),
        "kernel_calls": len(rows),
        "kernel_calls_per_step": len(rows) / len(host_ms) if host_ms else 0.0,
        "total_host_ms": total_host_ms,
        "avg_host_ms": total_host_ms / len(host_ms) if host_ms else 0.0,
        "total_kernel_ms": total_kernel_ms,
        "avg_kernel_ms": total_kernel_ms / len(host_ms) if host_ms else 0.0,
        "kernel_time_share_of_host_wall": total_kernel_ms / total_host_ms if total_host_ms else 0.0,
        "host_residual_ms_per_step": ((total_host_ms - total_kernel_ms) / len(host_ms)) if host_ms else 0.0,
        "buckets": emit(buckets),
        "top_kernels": emit(families)[:top],
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "[gguf-mtp-draft-rocprof] "
        f"avg_host_ms={summary['avg_host_ms']:.3f} "
        f"avg_kernel_ms={summary['avg_kernel_ms']:.3f} "
        f"kernel_share={summary['kernel_time_share_of_host_wall']:.3f} "
        f"calls/step={summary['kernel_calls_per_step']:.1f}"
    )
    print("\n=== HIGH-LEVEL BUCKETS ===")
    print(f"{'bucket':30s} {'calls':>7s} {'ms/step':>9s} {'%kernel':>8s}")
    for row in summary["buckets"]:
        print(f"{row['name'][:30]:30s} {row['calls']:7d} {row['ms_per_step']:9.3f} {row['pct_kernel']:8.1f}")
    print("\n=== TOP KERNEL FAMILIES ===")
    print(f"{'family':52s} {'calls':>7s} {'ms/step':>9s} {'%kernel':>8s}")
    for row in summary["top_kernels"]:
        print(f"{row['name'][:52]:52s} {row['calls']:7d} {row['ms_per_step']:9.3f} {row['pct_kernel']:8.1f}")


def _print_stage_timings(child: dict[str, Any]) -> None:
    totals = child.get("stage_timing_totals_ms")
    if not isinstance(totals, dict) or not totals:
        return
    steps = max(1, int(child.get("steps") or 1))
    print("\n=== SYNC STAGE TIMINGS ===")
    print(f"{'stage':52s} {'ms/step':>9s}")
    for name, total_ms in sorted(totals.items(), key=lambda item: -float(item[1])):
        print(f"{str(name)[:52]:52s} {float(total_ms) / steps:9.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", action="store_true", help="internal: process run under rocprofv3")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-ids", default=DEFAULT_PROMPT_IDS)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-seq", type=int, default=512)
    parser.add_argument("--draft-n-max", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--vocab-cap", type=int, default=0)
    parser.add_argument("--q6-top1-dp4a", action="store_true")
    parser.add_argument(
        "--router-row-parallel",
        action="store_true",
        help=(
            "Diagnostic llama-compat draft A/B: use the row-parallel F32 router "
            "logits kernel instead of generic MTP F32 linear for router logits."
        ),
    )
    parser.add_argument("--q6-top1-stage1-threads", type=int, default=128)
    parser.add_argument(
        "--q6-top1-stage1-shape",
        choices=("pack8", "pack16", "pack8_llama", "pack8_scalehoist", "row", "x8", "x8_dscale"),
        default="pack8",
    )
    parser.add_argument("--selected-down-x8-repack", choices=("off", "q5", "q6", "both"), default="off")
    parser.add_argument("--record-stage-timings", action="store_true")
    parser.add_argument("--sync-stage-timings", action="store_true")
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--skip-warmbuild", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=Path("/tmp/hipengine-hipcc-version.txt"))
    parser.add_argument("--child-json", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-gguf-mtp-draft-rocprof"))
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=_default_roctx_sdk())
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / f"{date.today().isoformat()}-gguf-mtp-draft-rocprof.json",
    )
    args = parser.parse_args()
    return _run_child(args) if args.child else _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
