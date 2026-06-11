#!/usr/bin/env python3
"""rocprofv3 kernel breakdown of MTP marker windows.

Wraps ``scripts/mtp_chain_e2e_smoke.py`` under ``rocprofv3 --kernel-trace`` and
processes the kernel CSV into a compact JSON artifact.  The default region is
the historical ``mtp_verify_pass_*`` verifier window, but ``--region`` can also
slice the enclosing cycle or the proposer draft/update windows.

``--selected-regions true`` would have been the natural way to scope the trace
to the steady-state verify window, but rocprofv3 1.1.0 on the gfx1151 host
silently emits no CSV when the roctxProfilerResume/Pause path is used (the
flag is honored under newer rocprofv3 versions — see
``scripts/qwen35_rocprof_audit.py`` for the audit-style harness that targets
1.2.3).  We therefore trace the whole smoke process and divide kernel-level
metrics by the verifier-pass count (smoke's ``len(active_budgets)``) to get
per-pass averages.  Warmup-vs-steady drift on this workload is small (<5%
relative per family in spot checks).  This diagnostic sizes the live MTP
kernel/launch families before retaining or rejecting implementation work.

This is a diagnostic artifact only.  No performance claim is retained from
this run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
DEFAULT_ROCTX_SDK = Path(
    "/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/"
    "_rocm_sdk_core/lib/librocprofiler-sdk-roctx.so.1"
)
DEFAULT_PROMPT_TOKENS = "151646"
REGION_MARKER_PREFIXES = {
    "verify_pass": ("mtp_verify_pass_",),
    "cycle": ("mtp_verify_cycle_",),
    "proposer_draft": ("mtp_proposer_draft_",),
    "proposer_update": ("mtp_proposer_update_",),
    "proposer_all": ("mtp_proposer_draft_", "mtp_proposer_update_"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--decode-tokens", type=int, default=24)
    parser.add_argument("--candidate-budget", type=int, default=3)
    parser.add_argument(
        "--rocprof-warmup-cycles",
        type=int,
        default=0,
        help=(
            "Forwarded to the smoke harness for a roctxProfilerResume window.  "
            "Unused on rocprofv3 1.1.0 (which silently drops --selected-regions "
            "output); kept so the same script works on newer rocprofv3 hosts."
        ),
    )
    parser.add_argument(
        "--rocprof-verify-cycles",
        type=int,
        default=0,
        help=(
            "Forwarded to the smoke harness for a roctxProfilerResume window.  "
            "Unused on rocprofv3 1.1.0; full trace is divided by cycle count."
        ),
    )
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched", "decode_batched"), default="c1_loop")
    parser.add_argument("--graph-mode", choices=("off", "auto", "validate"), default="off")
    parser.add_argument(
        "--region",
        choices=tuple(REGION_MARKER_PREFIXES),
        default="verify_pass",
        help=(
            "ROCTX marker region to summarize. Defaults to the historical verifier "
            "pass window; proposer_* regions require a smoke harness that emits the "
            "2026-06-12 proposer markers."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "benchmarks"
        / "results"
        / f"{date.today().isoformat()}-hipengine-mtp-verifier-rocprof-baseline.json",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/tmp/hipengine-rocprof-mtp-verifier-baseline"),
    )
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=DEFAULT_ROCTX_SDK)
    parser.add_argument(
        "--steady-state-skip",
        type=int,
        default=2,
        help=(
            "Number of leading verifier cycles to drop when filtering kernels by "
            "marker windows.  Default 2 drops cold-cache iterations."
        ),
    )
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=Path("/tmp/hipengine-hipcc-version.txt"),
        help=(
            "Path passed to the smoke harness as HIPENGINE_COMPILER_VERSION_FILE so the "
            "JIT cache key stays stable under rocprofv3 (mirrors qwen35_rocprof_audit.py)."
        ),
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    roctx_override = _prepare_roctx_override(args.roctx_sdk)
    sdk_lib_paths = _roctx_sdk_dep_paths(args.roctx_sdk)
    env = os.environ.copy()
    ld_prefix = os.pathsep.join([str(roctx_override), *(str(p) for p in sdk_lib_paths)])
    env["LD_LIBRARY_PATH"] = f"{ld_prefix}:{env.get('LD_LIBRARY_PATH', '')}"
    # Pin the cache key so JIT-compile sub-invocations (hipcc --version) don't shift
    # under rocprofv3.  qwen35_rocprof_audit uses --compiler-version-file; the build
    # module also reads HIPENGINE_COMPILER_VERSION_FILE from env, which threads through
    # the runner + the NativeMtpChainProposer without needing per-call CLI plumbing.
    if args.compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    smoke_json = args.raw_root / f"mtp-{args.region}-rocprof-smoke.json"
    smoke_log = args.raw_root / f"mtp-{args.region}-rocprof-smoke.log"
    smoke_cmd = _smoke_command(args, smoke_json)
    rocprof_cmd = _rocprof_command(args, smoke_cmd)

    if args.dry_run:
        print(" ".join(rocprof_cmd))
        return 0

    print(f"[rocprof] {' '.join(rocprof_cmd)}", flush=True)
    start = time.perf_counter()
    with smoke_log.open("w") as log_file:
        completed = subprocess.run(
            rocprof_cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    wall_seconds = time.perf_counter() - start
    if completed.returncode != 0:
        tail = smoke_log.read_text().splitlines()[-30:]
        raise SystemExit(
            "rocprofv3 mtp-verifier baseline failed with exit "
            f"{completed.returncode}; tail of {smoke_log}:\n" + "\n".join(tail)
        )

    if not smoke_json.exists():
        tail = smoke_log.read_text().splitlines()[-30:]
        raise SystemExit(
            f"smoke harness did not emit JSON at {smoke_json}; tail of {smoke_log}:\n"
            + "\n".join(tail)
        )

    kernel_csv = _single_file(args.raw_root, "*_kernel_trace.csv")
    smoke = json.loads(smoke_json.read_text())
    kernels = _read_kernels(kernel_csv)
    window = smoke.get("mtp", {}).get("rocprof_window", {}) or {}
    window_seconds = window.get("profiled_cycle_seconds")
    mtp = smoke.get("mtp", {}) or {}
    active_budgets = mtp.get("active_budgets") or []
    verifier_passes = len(active_budgets)
    host_window_seconds = window_seconds if window_seconds else mtp.get("verify_seconds")

    # If --marker-trace produced a marker CSV with roctxRangePush ranges for the
    # selected region, use marker start/end ns to filter the kernel rows down to
    # just that window (excludes prefill + AR baseline + unrelated MTP work).
    # Skip the first ``--steady-state-skip`` cycle's windows so we drop cold-cache
    # iterations.
    marker_csv_candidates = sorted(args.raw_root.glob("*_marker_api_trace.csv"))
    cycle_windows: list[tuple[int, int, int]] = []
    cycle_skip = int(args.steady_state_skip)
    if marker_csv_candidates:
        cycle_windows = _read_marker_windows(
            marker_csv_candidates[0],
            REGION_MARKER_PREFIXES[str(args.region)],
        )
    if str(args.region) != "verify_pass" and not cycle_windows:
        raise SystemExit(
            f"no ROCTX marker windows found for region={args.region}; "
            "run against a smoke harness that emits proposer/cycle markers"
        )
    region_kernels = kernels
    used_cycle_count = verifier_passes
    selected_windows: list[tuple[int, int]] = []
    if cycle_windows:
        steady_windows = [(start, end) for idx, start, end in cycle_windows if idx > cycle_skip]
        if str(args.region) != "verify_pass" and not steady_windows:
            raise SystemExit(
                f"all marker windows for region={args.region} were skipped by "
                f"--steady-state-skip={cycle_skip}"
            )
        if steady_windows:
            selected_windows = steady_windows
            region_kernels = _filter_kernels_by_windows(kernels, steady_windows)
            used_cycle_count = len({idx for idx, _start, _end in cycle_windows if idx > cycle_skip})
            host_window_seconds = sum(end - start for start, end in steady_windows) / 1e9
    summary = _summarize_rows(
        region_kernels,
        top=args.top,
        host_window_seconds=host_window_seconds,
        verifier_passes=used_cycle_count,
    )

    artifact = {
        "schema": 1,
        "status": "diagnostic_retained",
        "performance_claim": False,
        "date": date.today().isoformat(),
        "hardware": "AMD Radeon Pro W7900 / gfx1100 lineage; run target backend below",
        "backend": str(args.backend),
        "model": str(args.model),
        "region": str(args.region),
        "marker_prefixes": list(REGION_MARKER_PREFIXES[str(args.region)]),
        "steady_state_skip": int(args.steady_state_skip),
        "purpose": (
            "MTP B=3 marker-window rocprofv3 diagnostic.  The default verifier-pass "
            "region sizes the verify kernel/launch families; proposer regions size "
            "M12.7 graph-capture and route-batching candidates before implementation."
        ),
        "rocprof_command": " ".join(rocprof_cmd),
        "smoke_command": " ".join(smoke_cmd),
        "wall_seconds": wall_seconds,
        "smoke_json": str(smoke_json),
        "rocprof_log": str(smoke_log),
        "kernel_trace_csv": str(kernel_csv),
        "rocprof_window": window,
        "selected_marker_windows": len(selected_windows),
        "smoke_summary": {
            "exact_ar_match": smoke.get("exact_ar_match"),
            "ar_tok_s": (smoke.get("ar") or {}).get("decode_tok_s") or (smoke.get("ar") or {}).get("tok_s"),
            "mtp_tok_s": (smoke.get("mtp") or {}).get("decode_tok_s") or (smoke.get("mtp") or {}).get("tok_s"),
            "accepted_lengths": (smoke.get("mtp") or {}).get("accepted_lengths"),
            "active_budgets": (smoke.get("mtp") or {}).get("active_budgets"),
            "acceptance_rate": (smoke.get("mtp") or {}).get("acceptance_rate"),
            "candidate_budget": smoke.get("candidate_budget"),
            "chain_attn_mode": (smoke.get("mtp") or {}).get("chain_attn_mode"),
            "graph_mode": str(args.graph_mode),
            "proposal_impl": (smoke.get("mtp") or {}).get("proposal_impl"),
        },
        "summary": summary,
    }
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        f"[rocprof] region={args.region} window cycles={window.get('profiled_cycle_range')} "
        f"seconds={window_seconds!s} kernel_calls={summary['kernel_calls']} "
        f"kernel_ms={summary['kernel_time_ms']:.3f}",
        flush=True,
    )
    print(f"wrote {args.out}")
    return 0


def _smoke_command(args: argparse.Namespace, smoke_json: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/mtp_chain_e2e_smoke.py",
        "--model",
        str(args.model),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--decode-tokens",
        str(int(args.decode_tokens)),
        "--candidate-budget",
        str(int(args.candidate_budget)),
        "--proposal-impl",
        "persistent_device",
        "--backend",
        str(args.backend),
        "--chain-attn-mode",
        str(args.chain_attn_mode),
        "--graph-mode",
        str(args.graph_mode),
        "--rocprof-warmup-cycles",
        str(int(args.rocprof_warmup_cycles)),
        "--rocprof-verify-cycles",
        str(int(args.rocprof_verify_cycles)),
        "--json",
        str(smoke_json),
    ]


def _rocprof_command(args: argparse.Namespace, smoke_cmd: list[str]) -> list[str]:
    cmd = [
        args.rocprofv3,
        "--kernel-trace",
        "--marker-trace",  # captures roctxRangePush/Pop boundaries for per-cycle slicing
        "--output-format",
        "csv",
        "-d",
        str(args.raw_root),
        "-o",
        f"mtp-{args.region}-rocprof",
    ]
    if int(args.rocprof_verify_cycles) > 0:
        cmd.extend(["--selected-regions", "true"])
    cmd.append("--")
    cmd.extend(smoke_cmd)
    return cmd


def _prepare_roctx_override(sdk_path: Path) -> Path:
    if not sdk_path.exists():
        raise FileNotFoundError(f"rocprofiler SDK ROCTX library not found: {sdk_path}")
    override = Path("/tmp/hipengine-roctx-sdk-override-mtp")
    override.mkdir(parents=True, exist_ok=True)
    symlink = override / "libroctx64.so"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(sdk_path)
    return override


def _roctx_sdk_dep_paths(sdk_path: Path) -> tuple[Path, ...]:
    """Find the directories the SDK ROCTX library implicitly depends on.

    The therock SDK ships ``librocprofiler-sdk-roctx.so.1`` with NEEDED entries
    like ``librocm_sysdeps_dw.so.1`` that live under
    ``<sdk_core>/lib`` and ``<sdk_core>/lib/rocm_sysdeps/lib``.  Without these
    on ``LD_LIBRARY_PATH`` ``ctypes.CDLL('libroctx64.so')`` fails with
    ``cannot open shared object file: librocm_sysdeps_dw.so.1``.
    """

    sdk_core_lib = sdk_path.parent
    paths: list[Path] = []
    if sdk_core_lib.is_dir():
        paths.append(sdk_core_lib)
        sysdeps = sdk_core_lib / "rocm_sysdeps" / "lib"
        if sysdeps.is_dir():
            paths.append(sysdeps)
    return tuple(paths)


def _single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {root}, found {matches}")
    return matches[0]


def _read_marker_windows(path: Path, prefixes: tuple[str, ...]) -> list[tuple[int, int, int]]:
    """Read rocprofv3 marker CSV and return (cycle_idx, start_ns, end_ns) windows."""

    windows: list[tuple[int, int, int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # rocprofv3 1.1.0 emits one row per ROCTX_MARKER_CORE_API range with
            # Operation indicating roctxRangePush/Pop and a Marker_Name column.
            name = (
                row.get("Function")
                or row.get("Marker_Name")
                or row.get("Marker_Text")
                or row.get("Name")
                or ""
            ).strip()
            prefix = next((item for item in prefixes if name.startswith(item)), None)
            if prefix is None:
                continue
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            try:
                idx = int(name.removeprefix(prefix))
            except ValueError:
                continue
            windows.append((idx, start, end))
    windows.sort(key=lambda x: (x[0], x[1], x[2]))
    return windows


def _filter_kernels_by_windows(
    rows: list[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Return only kernel rows whose start..end ns fully overlaps any cycle window."""

    if not windows:
        return []
    sorted_windows = sorted(windows)
    kept: list[dict[str, Any]] = []
    for row in rows:
        start = row["start_ns"]
        end = row["end_ns"]
        for window_start, window_end in sorted_windows:
            if start >= window_start and end <= window_end:
                kept.append(row)
                break
            if start > window_end:
                continue
            if end < window_start:
                continue
    return kept


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
                    "vgpr": _int_or_none(row.get("VGPR_Count")),
                    "scratch": _int_or_none(row.get("Scratch_Size")),
                    "lds": _int_or_none(row.get("LDS_Block_Size")),
                }
            )
    return rows


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    top: int,
    host_window_seconds: float | None,
    verifier_passes: int = 0,
) -> dict[str, Any]:
    total_ns = sum(row["duration_ns"] for row in rows)
    by_kernel: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        for table, key in ((by_kernel, row["kernel"]), (by_family, _family(row["kernel"]))):
            entry = table.setdefault(
                key,
                {
                    "name": key,
                    "calls": 0,
                    "total_ns": 0,
                    "max_ns": 0,
                    "scratch_max": 0,
                    "vgpr_max": 0,
                    "lds_max": 0,
                },
            )
            entry["calls"] += 1
            entry["total_ns"] += row["duration_ns"]
            entry["max_ns"] = max(entry["max_ns"], row["duration_ns"])
            entry["scratch_max"] = max(entry["scratch_max"], row.get("scratch") or 0)
            entry["vgpr_max"] = max(entry["vgpr_max"], row.get("vgpr") or 0)
            entry["lds_max"] = max(entry["lds_max"], row.get("lds") or 0)
    host_ns = int(host_window_seconds * 1e9) if host_window_seconds else None
    passes = int(verifier_passes) if verifier_passes else 0
    return {
        "kernel_calls": len(rows),
        "kernel_time_ns": total_ns,
        "kernel_time_ms": total_ns / 1e6,
        "host_window_seconds": host_window_seconds,
        "kernel_time_share_of_host_window": None if not host_ns else total_ns / host_ns,
        "verifier_passes": passes,
        "kernel_time_ms_per_pass": (total_ns / 1e6 / passes) if passes else None,
        "kernel_calls_per_pass": (len(rows) / passes) if passes else None,
        "families": _finish_table(by_family.values(), total_ns, passes),
        "top_kernels": _finish_table(by_kernel.values(), total_ns, passes)[:top],
    }


def _finish_table(entries: Any, total_ns: int, verifier_passes: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        total = entry["total_ns"]
        calls = entry["calls"]
        row = {
            **entry,
            "total_ms": total / 1e6,
            "avg_us": (total / calls) / 1e3 if calls else 0.0,
            "max_us": entry["max_ns"] / 1e3,
            "share": total / total_ns if total_ns else 0.0,
        }
        if verifier_passes:
            row["calls_per_pass"] = calls / verifier_passes
            row["ms_per_pass"] = (total / 1e6) / verifier_passes
        out.append(row)
    out.sort(key=lambda item: item["total_ns"], reverse=True)
    return out


def _family(kernel: str) -> str:
    k = kernel.lower()
    if "aotriton" in k or "attn_fwd" in k:
        return "aotriton_prefill_attention"
    if "qwen35_gdn_prefill_recurrent" in k:
        return "linear_attention_gdn_prefill"
    if "qwen35_gdn_recurrent" in k or "qwen35_gdn_chain" in k:
        return "linear_attention_gdn_decode"
    if "linear_attn_conv" in k:
        return "linear_attention_conv"
    if "awq_selected" in k and "wmma" in k:
        return "moe_awq_wmma"
    if "gemv_awq_selected_dual" in k:
        return "moe_gate_up_dual_gemv"
    if "gemv_awq_selected" in k:
        return "moe_down_gemv"
    # Multi-token QKV / shared-expert / dense-MLP fused-W4 prefill kernels.  Hit
    # the verifier whenever ``tokens > 1`` because the call sites in
    # project_full_attention_qkv_fp16 / project_linear_attention_qkv_z_fp16 /
    # shared_expert_paro_w4_fp16 / dense_mlp_paro_w4_fp16 are gated
    # ``if tokens == 1 … else awq_fusedw4_prefill_*``.
    if "awq_fusedw4_prefill_dual" in k:
        return "w4_dual_prefill_smallbatch"
    if "awq_fusedw4_prefill" in k:
        return "w4_single_prefill_smallbatch"
    if "gemv_awq_dual" in k:
        return "w4_dual_gemv"
    if "gemv_awq_pack8" in k or "gemv_awq" in k:
        return "w4_single_gemv"
    if "w8a16_shared_gate" in k or "w8a16_shared_down" in k:
        return "shared_expert_w8a16"
    if "w8a16" in k:
        return "w8a16_linear"
    if "dflash_dense_bf16_to_bf16_expert" in k:
        return "proposer_expert_dense"
    if "dflash_dense_bf16_to_bf16_wmma" in k:
        return "proposer_dense_bf16"
    if "dflash_dense_bf16_to_f32_wmma" in k:
        return "proposer_dense_f32"
    if "dflash_qkv_proj" in k:
        return "proposer_qkv_projection"
    if "dflash_gqa_attention" in k:
        return "proposer_attention"
    if "mtp_accumulate_route" in k:
        return "proposer_route_accumulate"
    if "mtp_accumulate_sigmoid_gate" in k:
        return "proposer_shared_gate"
    if "mtp_finalize" in k:
        return "proposer_finalize"
    if "mtp_fuse_inputs" in k:
        return "proposer_input_fuse"
    if "router_topk_softmax" in k or "topk_rows_i32" in k or "mtp_softmax_topk" in k:
        return "proposer_topk_router"
    if "dense_dual_gemv" in k or "dense_gemv" in k:
        return "dense_gemv"
    if "router" in k:
        return "router"
    if "moe_group" in k or "moe_wmma_tile" in k or "moe_gather" in k:
        return "moe_metadata"
    if "silu_mul_dual_rotate" in k:
        return "moe_silu_rotate_out"
    if "silu" in k:
        return "silu"
    if "paro_rotate" in k:
        return "moe_paro_rotate_in"
    if "rmsnorm" in k:
        return "rmsnorm"
    if "rotate" in k or "rotary" in k:
        return "rotation_rope"
    if "paged_full_attn_prefill" in k or "causal_gqa" in k:
        return "native_prefill_attention"
    if "paged_full_attn_decode" in k or "full_attn_decode" in k:
        return "decode_attention"
    if "write_paged_kv" in k or "paged_kv" in k:
        return "kv_write"
    if "full_attn_gate" in k:
        return "attention_gate"
    if "lm_head" in k or "argmax" in k:
        return "lm_head_argmax"
    if "weighted" in k or "combine" in k:
        return "moe_combine"
    if "copybuffer" in k or "copy" in k:
        return "runtime_copy"
    # NOTE: do NOT match bare "fill" — it false-matches "prefill".  Use
    # the exact substring "memset" or the rocclr Memset signature.
    if "memset" in k or "memsetd" in k:
        return "runtime_memset"
    return "other"



def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
