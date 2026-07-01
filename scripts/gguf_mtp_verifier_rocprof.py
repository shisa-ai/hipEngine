#!/usr/bin/env python3
"""rocprofv3 host/GPU split for GGUF MTP target verification.

By default this diagnostic profiles the historical serial target-step shape with
``capture_hidden_seed_fp32=True`` and ``return_logits=False``.  With
``--mode block-verify`` it profiles the B2-like verifier block used by the
llama-compat parity lane: previous token plus draft rows, optional dp4a selected
MoE routing, direct-state commit, and the same ROCTX/kernel-window attribution.
This tells us whether the verifier gap is a host/copy/launch floor or GPU
streaming wall on current code.

Parent mode warm-builds outside rocprof, pins ``HIPENGINE_COMPILER_VERSION_FILE``,
then runs child mode under ``rocprofv3 --kernel-trace --marker-trace`` with
``require_cached_build=True``.  Child mode emits one ROCTX range per measured
verify step; parent filters the kernel CSV to those ranges and compares summed
kernel time against the child-measured host wall.

Diagnostic only: this emits no retained speed claim.
"""

from __future__ import annotations

import argparse
import collections
import csv
import ctypes
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPT_IDS = "760,4087,369,220,16,17,18,19"
SERIAL_MARKER_PREFIX = "gguf_mtp_verify_serial_"
BLOCK_MARKER_PREFIX = "gguf_mtp_verify_block_"


def _marker_prefix(mode: str) -> str:
    return BLOCK_MARKER_PREFIX if mode == "block-verify" else SERIAL_MARKER_PREFIX


def _apply_route_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")
    if getattr(args, "verify_dp4a", False):
        for flag in (
            "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A",
            "HIPENGINE_GGUF_T16_SELECTED_DP4A",
            "HIPENGINE_GGUF_RAW_SELECTED_DP4A",
        ):
            os.environ[flag] = "1"
    if getattr(args, "verify_dense_q8_dp4a", False):
        os.environ["HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"] = "1"
        os.environ["HIPENGINE_GGUF_DENSE_Q8_DP4A"] = "1"
    if getattr(args, "verify_dense_q8_dp4a_all", False):
        os.environ["HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"] = "1"
        os.environ["HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL"] = "1"
    selected_down_x8 = str(getattr(args, "selected_down_x8_repack", "off"))
    if selected_down_x8 != "off":
        os.environ["HIPENGINE_GGUF_SELECTED_X8_REPACK"] = selected_down_x8
    if getattr(args, "selected_gate_up_x8", False):
        os.environ["HIPENGINE_GGUF_SELECTED_GATE_UP_X8"] = "1"
    if getattr(args, "q8_t16_pair_rowtile", False):
        os.environ["HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE"] = "1"
    if getattr(args, "q8_t16_rowtile_all", False):
        os.environ["HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL"] = "1"


def _sum_stage_timings(rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return dict(sorted(totals.items()))


def _default_roctx_sdk() -> Path:
    candidates = [
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "_rocm_sdk_core"
        / "lib"
        / "librocprofiler-sdk-roctx.so.1",
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "_rocm_sdk_devel"
        / "lib"
        / "librocprofiler-sdk-roctx.so.1",
        Path("/opt/rocm/lib/librocprofiler-sdk-roctx.so.1"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


class _Roctx:
    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._push = None
        self._pop = None
        try:
            self._lib = ctypes.CDLL("libroctx64.so")
        except OSError as exc:
            print(f"warning: libroctx64.so unavailable, markers disabled: {exc}", file=sys.stderr)
            return
        self._push = getattr(self._lib, "roctxRangePushA", None)
        self._pop = getattr(self._lib, "roctxRangePop", None)
        if self._push is not None:
            self._push.argtypes = [ctypes.c_char_p]
            self._push.restype = ctypes.c_int
        if self._pop is not None:
            self._pop.argtypes = []
            self._pop.restype = ctypes.c_int

    def push(self, name: str) -> None:
        if self._push is not None:
            self._push(name.encode("utf-8"))

    def pop(self) -> None:
        if self._pop is not None:
            self._pop()


def _parse_prompt_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _run_child(args: argparse.Namespace) -> int:
    _apply_route_env(args)
    if args.compiler_version_file:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    prompt_ids = _parse_prompt_ids(args.prompt_ids)
    roctx = _Roctx()
    host_ms: list[float] = []
    token_ids: list[int] = []
    stage_timings: list[dict[str, float]] = []
    with Qwen35GGUFResidentSession(
        args.model,
        max_sequence_length=args.max_seq,
        require_cached_build=args.require_cached,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        first = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
        cur = int(first.token_id)
        if args.mode == "serial-step":
            for _ in range(int(args.warmup)):
                cur = int(
                    session.step(
                        cur,
                        return_logits=bool(args.return_logits),
                        capture_hidden_seed_fp32=True,
                    ).token_id
                )
            session.runtime.device_synchronize()
            for index in range(int(args.steps)):
                roctx.push(f"{SERIAL_MARKER_PREFIX}{index}")
                t0 = time.perf_counter()
                result = session.step(
                    cur,
                    return_logits=bool(args.return_logits),
                    capture_hidden_seed_fp32=True,
                )
                session.runtime.device_synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                roctx.pop()
                cur = int(result.token_id)
                host_ms.append(elapsed_ms)
                token_ids.append(cur)
        else:
            block_rows = int(args.block_rows)
            if block_rows <= 0:
                raise ValueError("--block-rows must be positive")

            def run_block(index: int | None) -> int:
                nonlocal cur
                start_position = int(session.position)
                block_inputs = [int(cur)] * block_rows
                if index is not None:
                    roctx.push(f"{BLOCK_MARKER_PREFIX}{index}")
                t0 = time.perf_counter()
                result = session.verify_target_block(
                    block_inputs,
                    bulk_attention_mode=str(args.block_verify_mode),
                    use_wmma_prefill=bool(args.block_wmma_prefill),
                    capture_linear_state_rows=bool(args.direct_state_commit),
                    record_stage_timings=bool(args.record_stage_timings),
                    sync_stage_timings=bool(args.sync_stage_timings),
                    defer_linear_state_commit=bool(args.direct_state_commit),
                )
                if args.direct_state_commit:
                    if not result.linear_state_rows_captured:
                        raise RuntimeError("direct-state block profile did not capture linear-state rows")
                    session._commit_verify_linear_state_row(block_rows - 1, position=start_position + block_rows)
                session.runtime.device_synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if index is not None:
                    roctx.pop()
                    host_ms.append(elapsed_ms)
                    token_ids.append(int(result.token_ids[-1]))
                    if args.record_stage_timings:
                        stage_timings.append(dict(session.last_verify_stage_timings_ms))
                cur = int(result.token_ids[-1])
                return cur

            for _ in range(int(args.warmup)):
                run_block(None)
            session.runtime.device_synchronize()
            for index in range(int(args.steps)):
                run_block(index)
    payload = {
        "schema": "hipengine.gguf_mtp_verifier_rocprof.child.v1",
        "mode": str(args.mode),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "return_logits": bool(args.return_logits),
        "block_rows": int(args.block_rows) if args.mode == "block-verify" else None,
        "block_verify_mode": str(args.block_verify_mode) if args.mode == "block-verify" else None,
        "block_wmma_prefill": bool(args.block_wmma_prefill) if args.mode == "block-verify" else None,
        "direct_state_commit": bool(args.direct_state_commit) if args.mode == "block-verify" else None,
        "verify_dp4a": bool(args.verify_dp4a),
        "verify_dense_q8_dp4a": bool(args.verify_dense_q8_dp4a),
        "verify_dense_q8_dp4a_all": bool(args.verify_dense_q8_dp4a_all),
        "selected_down_x8_repack": str(args.selected_down_x8_repack),
        "selected_gate_up_x8": bool(args.selected_gate_up_x8),
        "q8_t16_pair_rowtile": bool(args.q8_t16_pair_rowtile),
        "q8_t16_rowtile_all": bool(args.q8_t16_rowtile_all),
        "host_ms": host_ms,
        "avg_host_ms": sum(host_ms) / len(host_ms) if host_ms else 0.0,
        "token_ids": token_ids,
        "stage_timings_ms": stage_timings if stage_timings else None,
        "stage_timing_totals_ms": _sum_stage_timings(stage_timings) if stage_timings else None,
    }
    if args.child_json:
        args.child_json.parent.mkdir(parents=True, exist_ok=True)
        args.child_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[gguf-mtp-verifier] {args.mode} steps={args.steps} avg_host_ms={payload['avg_host_ms']:.3f}")
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
        "--mode",
        str(args.mode),
        "--block-rows",
        str(int(args.block_rows)),
        "--block-verify-mode",
        str(args.block_verify_mode),
        "--compiler-version-file",
        str(cvf),
        "--child-json",
        str(child_json),
    ]
    if args.return_logits:
        child_base.append("--return-logits")
    else:
        child_base.append("--no-return-logits")
    if args.block_wmma_prefill:
        child_base.append("--block-wmma-prefill")
    if args.direct_state_commit:
        child_base.append("--direct-state-commit")
    else:
        child_base.append("--no-direct-state-commit")
    if args.verify_dp4a:
        child_base.append("--verify-dp4a")
    if args.verify_dense_q8_dp4a:
        child_base.append("--verify-dense-q8-dp4a")
    if args.verify_dense_q8_dp4a_all:
        child_base.append("--verify-dense-q8-dp4a-all")
    if str(args.selected_down_x8_repack) != "off":
        child_base.extend(["--selected-down-x8-repack", str(args.selected_down_x8_repack)])
    if args.selected_gate_up_x8:
        child_base.append("--selected-gate-up-x8")
    if args.q8_t16_pair_rowtile:
        child_base.append("--q8-t16-pair-rowtile")
    if args.q8_t16_rowtile_all:
        child_base.append("--q8-t16-rowtile-all")
    if args.record_stage_timings:
        child_base.append("--record-stage-timings")
    if args.sync_stage_timings:
        child_base.append("--sync-stage-timings")

    if not args.skip_warmbuild:
        print("[gguf-mtp-verifier-rocprof] warm-build pass (no profiler)...", flush=True)
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
    print(f"[gguf-mtp-verifier-rocprof] {' '.join(rocprof_cmd)}", flush=True)
    subprocess.run(rocprof_cmd, cwd=REPO_ROOT, env=env, check=True)

    kernel_csv = _single_file(trace_dir, "*_kernel_trace.csv")
    marker_csv = _single_file(trace_dir, "*_marker_api_trace.csv")
    child = json.loads(child_json.read_text(encoding="utf-8"))
    marker_prefix = _marker_prefix(str(args.mode))
    windows = _read_marker_windows(marker_csv, marker_prefix)
    if len(windows) != int(args.steps):
        raise SystemExit(f"expected {args.steps} marker windows, found {len(windows)}")
    kernels = _read_kernels(kernel_csv)
    selected = _filter_kernels_by_windows(kernels, [(start, end) for _idx, start, end in windows])
    summary = _summarize(selected, host_ms=[float(x) for x in child["host_ms"]], top=int(args.top))
    artifact = {
        "schema": "hipengine.gguf_mtp_verifier_rocprof.v1",
        "date": date.today().isoformat(),
        "status": "diagnostic_retained",
        "performance_claim": False,
        "purpose": "Current GGUF MTP target verifier host/GPU split.",
        "model": str(args.model),
        "hardware": "AMD Radeon 8060S / Ryzen AI Max+ 395 (gfx1151)",
        "mode": str(args.mode),
        "block_rows": int(args.block_rows) if args.mode == "block-verify" else None,
        "block_verify_mode": str(args.block_verify_mode) if args.mode == "block-verify" else None,
        "block_wmma_prefill": bool(args.block_wmma_prefill) if args.mode == "block-verify" else None,
        "direct_state_commit": bool(args.direct_state_commit) if args.mode == "block-verify" else None,
        "return_logits": bool(args.return_logits),
        "verify_dp4a": bool(args.verify_dp4a),
        "verify_dense_q8_dp4a": bool(args.verify_dense_q8_dp4a),
        "verify_dense_q8_dp4a_all": bool(args.verify_dense_q8_dp4a_all),
        "selected_down_x8_repack": str(args.selected_down_x8_repack),
        "selected_gate_up_x8": bool(args.selected_gate_up_x8),
        "q8_t16_pair_rowtile": bool(args.q8_t16_pair_rowtile),
        "q8_t16_rowtile_all": bool(args.q8_t16_rowtile_all),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "prompt_ids": _parse_prompt_ids(args.prompt_ids),
        "marker_prefix": marker_prefix,
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
        print(f"[gguf-mtp-verifier-rocprof] wrote {args.out}")
    _print_summary(summary)
    return 0


def _prepare_roctx_override(sdk_path: Path) -> Path:
    if not sdk_path.exists():
        raise FileNotFoundError(f"rocprofiler SDK ROCTX library not found: {sdk_path}")
    override = Path("/tmp/hipengine-roctx-sdk-override-gguf-mtp")
    override.mkdir(parents=True, exist_ok=True)
    symlink = override / "libroctx64.so"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(sdk_path)
    return override


def _roctx_sdk_dep_paths(sdk_path: Path) -> tuple[Path, ...]:
    sdk_core_lib = sdk_path.parent
    paths: list[Path] = []
    if sdk_core_lib.is_dir():
        paths.append(sdk_core_lib)
        sysdeps = sdk_core_lib / "rocm_sysdeps" / "lib"
        if sysdeps.is_dir():
            paths.append(sysdeps)
    return tuple(paths)


def _single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {root}, found {matches}")
    return matches[0]


def _read_marker_windows(path: Path, prefix: str) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (
                row.get("Function")
                or row.get("Marker_Name")
                or row.get("Marker_Text")
                or row.get("Name")
                or ""
            ).strip()
            if not name.startswith(prefix):
                continue
            try:
                idx = int(name.removeprefix(prefix))
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end >= start:
                windows.append((idx, start, end))
    windows.sort(key=lambda item: item[0])
    return windows


def _read_kernels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            rows.append(
                {
                    "kernel": (row.get("Kernel_Name") or row.get("KernelName") or row.get("Name") or "").strip(),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                }
            )
    return rows


def _filter_kernels_by_windows(
    rows: list[dict[str, Any]], windows: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        start = int(row["start_ns"])
        end = int(row["end_ns"])
        if any(start >= ws and end <= we for ws, we in windows):
            kept.append(row)
    return kept


def _family(name: str) -> str:
    value = re.sub(r"^void\s+", "", name.strip()).replace("(anonymous namespace)::", "")
    value = re.sub(r"<[^>]*>", "", value).split("(")[0]
    return re.sub(r"_kernel$", "", value).strip()


def _bucket(family: str) -> str:
    f = family.lower()
    if "router" in f:
        return "moe_router"
    if any(key in f for key in ("linear_attn", "gdn", "ssm", "_conv_")):
        return "gdn_linear_attn"
    if f.startswith(("q4_k_t16_selected", "qk_t16_selected", "gguf_k_selected", "gguf_q4_k_selected")):
        return "moe_selected_gemv"
    if f.startswith(("q8_0_t16", "q8_0_dp4a")):
        return "dense_q8_0_gemv"
    if "q6_k_t16_gemv" in f or "lm_head" in f or "argmax" in f:
        return "lm_head"
    if "rmsnorm" in f or "rotary" in f:
        return "rmsnorm_rope"
    if any(key in f for key in ("silu", "weighted_sum", "combine")):
        return "moe_combine_silu"
    if any(key in f for key in ("attn", "flash", "softmax", "paged_kv", "paged_full")):
        return "attn_core"
    if "embedding" in f:
        return "embedding"
    if any(key in f for key in ("copybuffer", "fillbuffer", "memcpy", "memset")):
        return "memcpy_fill"
    return "other"


def _summarize(rows: list[dict[str, Any]], *, host_ms: list[float], top: int) -> dict[str, Any]:
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
        "[gguf-mtp-verifier-rocprof] "
        f"avg_host_ms={summary['avg_host_ms']:.3f} "
        f"avg_kernel_ms={summary['avg_kernel_ms']:.3f} "
        f"kernel_share={summary['kernel_time_share_of_host_wall']:.3f} "
        f"calls/step={summary['kernel_calls_per_step']:.1f}"
    )
    print("\n=== HIGH-LEVEL BUCKETS ===")
    print(f"{'bucket':30s} {'calls':>7s} {'ms/step':>9s} {'%kernel':>8s}")
    for row in summary["buckets"]:
        print(f"{row['name'][:30]:30s} {row['calls']:7d} {row['ms_per_step']:9.3f} {row['pct_kernel']:8.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", action="store_true", help="internal: process run under rocprofv3")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-ids", default=DEFAULT_PROMPT_IDS)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-seq", type=int, default=512)
    parser.add_argument(
        "--mode",
        choices=("serial-step", "block-verify"),
        default="serial-step",
        help="Profile the historical serial step verifier or the B2-like target block verifier.",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=3,
        help="Rows passed to verify_target_block in --mode block-verify; B2 uses prev+2 draft rows.",
    )
    parser.add_argument("--block-verify-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument("--block-wmma-prefill", action="store_true")
    parser.add_argument(
        "--direct-state-commit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture and commit the final block row's linear state in block-verify mode.",
    )
    parser.add_argument("--verify-dp4a", action="store_true", help="Enable llama-compat selected-MoE dp4a env flags.")
    parser.add_argument(
        "--verify-dense-q8-dp4a",
        action="store_true",
        help="Enable the rejected dense Q8 raw-sidecar dp4a route for diagnostic profiling.",
    )
    parser.add_argument(
        "--verify-dense-q8-dp4a-all",
        action="store_true",
        help="Enable the raw-sidecar dense Q8 dp4a-all route for diagnostic profiling.",
    )
    parser.add_argument("--selected-down-x8-repack", choices=("off", "q5", "q6", "both"), default="off")
    parser.add_argument(
        "--selected-gate-up-x8",
        action="store_true",
        help="Enable the selected gate/up Q4_K X8 replacement-layout diagnostic.",
    )
    parser.add_argument(
        "--q8-t16-pair-rowtile",
        action="store_true",
        help="Enable the existing Q8T16 attn_qkv+attn_gate pair rowtile diagnostic.",
    )
    parser.add_argument(
        "--q8-t16-rowtile-all",
        action="store_true",
        help="Enable Q8T16 rowtile diagnostics for singleton, pair, and triple verifier projections.",
    )
    parser.add_argument("--record-stage-timings", action="store_true")
    parser.add_argument("--sync-stage-timings", action="store_true")
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument(
        "--return-logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match gguf_mtp_bench.py target step call site by default.",
    )
    parser.add_argument("--skip-warmbuild", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=Path("/tmp/hipengine-hipcc-version.txt"))
    parser.add_argument("--child-json", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-gguf-mtp-verifier-rocprof"))
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=_default_roctx_sdk())
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / f"{date.today().isoformat()}-gguf-mtp-verifier-rocprof.json",
    )
    args = parser.parse_args()
    return _run_child(args) if args.child else _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
