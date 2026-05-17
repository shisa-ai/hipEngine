#!/usr/bin/env python3
"""Run Qwen3.5/PARO under rocprofv3 selected regions and emit Amdahl summaries.

The raw rocprof CSVs stay in /tmp.  The committed artifact is a compact JSON with per-family and
per-kernel rankings for prefill and measured decode graph replay.
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
DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_ROCTX_SDK = Path(
    "/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/"
    "_rocm_sdk_core/lib/librocprofiler-sdk-roctx.so.1"
)
REGIONS = ("prefill", "measured_decode_graph")
COMMON_FLAGS = (
    "--token-id", "9707",
    "--decode-tokens", "128",
    "--warmup-decode-tokens", "1",
    "--max-layers", "40",
    "--compiler-version-file", "/tmp/hipengine-hipcc-version.txt",
    "--require-cached-build",
    "--attn-aotriton-min-tokens", "512",
    "--graph-replay-decode",
    "--prefill-chunk-autotune",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=("512/128", "4096/128", "32768/128"),
        help="Prompt/decode pairs; default is the OPTIMIZE.md M.3 comparison set.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Compact JSON output path")
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-rocprof-qwen35-audit"))
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=DEFAULT_ROCTX_SDK)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--profile-decode-tokens",
        type=int,
        default=16,
        help=(
            "Number of graph replays to trace for measured_decode_graph. rocprofv3 1.2.3 "
            "asserts in finalization at 64/128 replays on this host, while 16 is stable; "
            "per-token Amdahl is scaled from this sampled replay window."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workloads = [_parse_workload(text) for text in args.workloads]
    output_path = args.out or (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / f"{date.today().isoformat()}-hipengine-qwen35-rocprof-amdahl-diagnostic.json"
    )

    roctx_override = _prepare_roctx_override(args.roctx_sdk)
    base_env = os.environ.copy()
    base_env["LD_LIBRARY_PATH"] = f"{roctx_override}:{base_env.get('LD_LIBRARY_PATH', '')}"

    result: dict[str, Any] = {
        "schema": 1,
        "status": "diagnostic_retained",
        "performance_claim": False,
        "date": date.today().isoformat(),
        "hardware": "AMD Radeon Pro W7900 / gfx1100",
        "model": "Qwen3.5-35B-A3B-PARO",
        "model_path": args.model,
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "purpose": "hipEngine-local rocprofv3 selected-region Amdahl baseline for docs/OPTIMIZE.md M.3/M.4",
        "profile_method": (
            "rocprofv3 --kernel-trace --selected-regions, with qwen35_paro_bench.py "
            "calling roctxProfilerResume/Pause around prefill and measured_decode_graph. "
            "This avoids rocprofiler marker-trace finalization asserts seen with HIP graph replay."
        ),
        "common_command_flags": list(COMMON_FLAGS),
        "roctx_override": str(roctx_override),
        "raw_root": str(args.raw_root),
        "workloads": [],
    }

    if args.dry_run:
        for prompt_len, decode_tokens in workloads:
            for region in REGIONS:
                trace_decode_tokens = decode_tokens if region == "prefill" else min(decode_tokens, args.profile_decode_tokens)
                print(" ".join(_rocprof_command(args, prompt_len, trace_decode_tokens, Path("/tmp/out.json"), Path("/tmp/run"), region)))
        return 0

    args.raw_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for prompt_len, decode_tokens in workloads:
        label = _workload_label(prompt_len, decode_tokens)
        workload_payload: dict[str, Any] = {
            "workload": label,
            "prompt_length": prompt_len,
            "decode_tokens": decode_tokens,
            "region_profiles": {},
        }
        for region in REGIONS:
            run_dir = args.raw_root / label / region
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            trace_decode_tokens = decode_tokens if region == "prefill" else min(decode_tokens, args.profile_decode_tokens)
            bench_json = run_dir / f"{label}-{region}-bench.json"
            cmd = _rocprof_command(args, prompt_len, trace_decode_tokens, bench_json, run_dir, region)
            print(f"[rocprof] {label}/{region}: {' '.join(cmd)}", flush=True)
            log_path = run_dir / f"{label}-{region}-rocprof.log"
            start = time.perf_counter()
            with log_path.open("w") as log_file:
                completed = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=base_env,
                    text=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            wall_seconds = time.perf_counter() - start
            if completed.returncode != 0:
                raise SystemExit(
                    f"rocprof workload {label}/{region} failed with exit {completed.returncode}; see {log_path}"
                )

            kernel_csv = _single_file(run_dir, "*_kernel_trace.csv")
            bench = json.loads(bench_json.read_text())
            bench_metrics = _bench_metrics(bench)
            kernels = _read_kernels(kernel_csv)
            summary = _summarize_rows(
                kernels,
                top=args.top,
                decode_tokens=trace_decode_tokens if region == "measured_decode_graph" else None,
                host_window_seconds=(
                    bench_metrics.get("decode_seconds")
                    if region == "measured_decode_graph"
                    else bench_metrics.get("prefill_seconds")
                ),
            )
            workload_payload["region_profiles"][region] = {
                "command": " ".join(_bench_command(args.model, prompt_len, trace_decode_tokens, bench_json, region)),
                "rocprof_command": " ".join(cmd),
                "profile_decode_tokens": trace_decode_tokens,
                "workload_decode_tokens": decode_tokens,
                "wall_seconds": wall_seconds,
                "bench_json": str(bench_json),
                "rocprof_log": str(log_path),
                "kernel_trace_csv": str(kernel_csv),
                "bench": bench_metrics,
                "summary": summary,
            }
            output_path.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"[rocprof] {label}/{region}: calls={summary['kernel_calls']} "
                f"kernel_ms={summary['kernel_time_ms']:.3f}",
                flush=True,
            )
        # Use the measured-decode profile's benchmark numbers as the row summary.
        decode_bench = workload_payload["region_profiles"]["measured_decode_graph"]["bench"]
        workload_payload["bench"] = decode_bench
        result["workloads"].append(workload_payload)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        prefill_tok_s = _fmt_float(decode_bench.get("prefill_tok_s"))
        decode_tok_s = _fmt_float(decode_bench.get("decode_tok_s"))
        print(
            f"[rocprof] {label}: prefill={prefill_tok_s} tok/s "
            f"decode_profile={decode_tok_s} tok/s "
            f"({workload_payload['region_profiles']['measured_decode_graph']['profile_decode_tokens']} traced replays), "
            f"wrote partial {output_path}",
            flush=True,
        )

    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_path}")
    return 0


def _rocprof_command(
    args: argparse.Namespace,
    prompt_len: int,
    decode_tokens: int,
    bench_json: Path,
    run_dir: Path,
    region: str,
) -> list[str]:
    return [
        args.rocprofv3,
        "--kernel-trace",
        "--selected-regions",
        "true",
        "--output-format",
        "csv",
        "-d",
        str(run_dir),
        "-o",
        f"qwen35-{_workload_label(prompt_len, decode_tokens)}-{region}",
        "--",
        *_bench_command(args.model, prompt_len, decode_tokens, bench_json, region),
    ]


def _bench_command(model: str, prompt_len: int, decode_tokens: int, bench_json: Path, region: str) -> list[str]:
    flags = list(COMMON_FLAGS)
    if decode_tokens != 128:
        idx = flags.index("--decode-tokens") + 1
        flags[idx] = str(decode_tokens)
    return [
        sys.executable,
        "scripts/qwen35_paro_bench.py",
        "--model",
        model,
        *flags,
        "--rocprof-selected-region",
        region,
        "--prompt-length",
        str(prompt_len),
        "--json",
        str(bench_json),
    ]


def _bench_metrics(bench: dict[str, Any]) -> dict[str, Any]:
    throughput = bench.get("throughput") or {}
    timings = bench.get("timings") or {}
    memory = bench.get("memory") or {}
    return {
        "prefill_tok_s": throughput.get("prefill_tok_s") or bench.get("prefill_tok_s"),
        "decode_tok_s": throughput.get("warmed_decode_tok_s") or bench.get("decode_tok_s"),
        "prefill_seconds": timings.get("prefill_seconds") or bench.get("prefill_seconds"),
        "decode_seconds": timings.get("decode_seconds") or bench.get("decode_seconds"),
        "decode_step_median_s": throughput.get("warmed_decode_step_median_s") or bench.get("decode_step_median_s"),
        "tracked_peak_allocated_gib": memory.get("tracked_peak_allocated_gib") or bench.get("tracked_peak_allocated_gib"),
        "generated": bench.get("generated_preview", bench.get("generated", []))[:2],
    }


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _parse_workload(text: str) -> tuple[int, int]:
    left, sep, right = text.partition("/")
    if not sep:
        raise argparse.ArgumentTypeError(f"invalid workload {text!r}; expected prompt/decode")
    return _parse_int(left), _parse_int(right)


def _parse_int(text: str) -> int:
    text = text.strip().lower()
    mult = 1
    if text.endswith("k"):
        mult = 1024
        text = text[:-1]
    return int(text) * mult


def _workload_label(prompt_len: int, decode_tokens: int) -> str:
    p = f"{prompt_len // 1024}k" if prompt_len % 1024 == 0 else str(prompt_len)
    return f"{p}-{decode_tokens}"


def _prepare_roctx_override(sdk_path: Path) -> Path:
    if not sdk_path.exists():
        raise FileNotFoundError(f"rocprofiler SDK ROCTX library not found: {sdk_path}")
    override = Path("/tmp/hipengine-roctx-sdk-override")
    override.mkdir(parents=True, exist_ok=True)
    symlink = override / "libroctx64.so"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(sdk_path)
    return override


def _single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {root}, found {matches}")
    return matches[0]


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
            rows.append({
                "kernel": (row.get("Kernel_Name") or row.get("KernelName") or row.get("Name") or "").strip(),
                "start_ns": start,
                "end_ns": end,
                "duration_ns": end - start,
                "dispatch_id": row.get("Dispatch_Id"),
                "correlation_id": row.get("Correlation_Id"),
                "vgpr": _int_or_none(row.get("VGPR_Count")),
                "scratch": _int_or_none(row.get("Scratch_Size")),
                "lds": _int_or_none(row.get("LDS_Block_Size")),
            })
    return rows


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    top: int,
    decode_tokens: int | None,
    host_window_seconds: float | None,
) -> dict[str, Any]:
    total_ns = sum(row["duration_ns"] for row in rows)
    by_kernel: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        for table, key in ((by_kernel, row["kernel"]), (by_family, _family(row["kernel"]))):
            entry = table.setdefault(key, {
                "name": key,
                "calls": 0,
                "total_ns": 0,
                "max_ns": 0,
                "scratch_max": 0,
                "vgpr_max": 0,
                "lds_max": 0,
            })
            entry["calls"] += 1
            entry["total_ns"] += row["duration_ns"]
            entry["max_ns"] = max(entry["max_ns"], row["duration_ns"])
            entry["scratch_max"] = max(entry["scratch_max"], row.get("scratch") or 0)
            entry["vgpr_max"] = max(entry["vgpr_max"], row.get("vgpr") or 0)
            entry["lds_max"] = max(entry["lds_max"], row.get("lds") or 0)
    host_ns = int(host_window_seconds * 1e9) if host_window_seconds else None
    return {
        "kernel_calls": len(rows),
        "kernel_time_ns": total_ns,
        "kernel_time_ms": total_ns / 1e6,
        "host_window_seconds": host_window_seconds,
        "kernel_time_share_of_host_window": None if not host_ns else total_ns / host_ns,
        "dispatches_per_token": None if not decode_tokens else len(rows) / decode_tokens,
        "kernel_time_ms_per_token": None if not decode_tokens else (total_ns / 1e6) / decode_tokens,
        "families": _finish_table(by_family.values(), total_ns, decode_tokens),
        "top_kernels": _finish_table(by_kernel.values(), total_ns, decode_tokens)[:top],
    }


def _finish_table(entries: Any, total_ns: int, decode_tokens: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        total = entry["total_ns"]
        calls = entry["calls"]
        row = {
            **entry,
            "total_ms": total / 1e6,
            "avg_us": (total / calls) / 1e3 if calls else 0.0,
            "share": total / total_ns if total_ns else 0.0,
        }
        if decode_tokens:
            row["calls_per_token"] = calls / decode_tokens
            row["ms_per_token"] = (total / 1e6) / decode_tokens
        out.append(row)
    out.sort(key=lambda item: item["total_ns"], reverse=True)
    return out


def _family(kernel: str) -> str:
    k = kernel.lower()
    if "aotriton" in k or "attn_fwd" in k:
        return "aotriton_prefill_attention"
    if "qwen35_gdn_prefill_recurrent" in k:
        return "linear_attention_gdn_prefill"
    if "qwen35_gdn_recurrent" in k:
        return "linear_attention_gdn_decode"
    if "linear_attn_conv" in k:
        return "linear_attention_conv"
    if "awq_selected" in k and "wmma" in k:
        return "moe_awq_wmma"
    if "awq_fusedw4_prefill" in k:
        return "w4_prefill_gemm"
    if "gemv_awq_selected" in k:
        return "selected_moe_w4_gemv"
    if "gemv_awq_dual" in k:
        return "w4_dual_gemv"
    if "gemv_awq_pack8" in k or "gemv_awq" in k:
        return "w4_single_gemv"
    if "w8a16_shared_gate" in k or "w8a16_shared_down" in k:
        return "shared_expert_w8a16"
    if "w8a16" in k:
        return "w8a16_linear"
    if "dense_dual_gemv" in k or "dense_gemv" in k:
        return "dense_gemv"
    if "router" in k:
        return "router"
    if "moe_group" in k or "moe_wmma_tile" in k or "moe_gather" in k:
        return "moe_metadata"
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
    if "silu" in k:
        return "silu"
    if "weighted" in k or "combine" in k:
        return "moe_combine"
    if "copybuffer" in k or "copy" in k:
        return "runtime_copy"
    if "memset" in k or "fill" in k:
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
