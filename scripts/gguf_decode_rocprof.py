#!/usr/bin/env python3
"""rocprofv3 kernel-family breakdown of the GGUF resident AR-decode wall.

This is the GGUF-path counterpart to ``scripts/mtp_verifier_rocprof.py`` (which
targets the PARO BF16 MTP model).  It answers "where does the ~18 ms/token decode
wall actually go, per kernel family" for the production
``Qwen3.6-35B-A3B-UD-Q4_K_M`` resident decode path on gfx1151.

Two modes:

* ``--child`` (internal): prefill + warmup + N timed decode ``step()`` calls.
  This is the process that runs *under* rocprofv3.  ``--require-cached`` forbids
  any ``hipcc`` spawn so the JIT cache + a pinned compiler-version file keep the
  profiled process from invoking the compiler (per AGENTS.md rocprof rules).
* default (parent): warm-build the kernels in a non-profiled child, pin the
  compiler-version file, run the child under ``rocprofv3 --kernel-trace``, then
  bucket the kernel CSV into high-level families and emit a compact JSON
  artifact + ranked table.

Diagnostic only; no perf claim is retained from a single run.  The numbers size
which kernel families dominate the decode wall so optimization targets the right
one (e.g. dense Q8_0 GEMV vs selected-MoE GEMV).

Example::

    PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_decode_rocprof.py \
        --steps 24 --warmup 4 \
        --out benchmarks/results/2026-06-29-gguf-decode-rocprof-families.json
"""

from __future__ import annotations

import argparse
import collections
import csv
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
DEFAULT_PROMPT = [760, 4087, 369, 220, 16, 17, 18, 19]


# --------------------------------------------------------------------------- #
# Child mode: the process that runs under rocprofv3.
# --------------------------------------------------------------------------- #
def _run_child(args: argparse.Namespace) -> int:
    os.environ.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")
    if args.compiler_version_file:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    prompt_ids = [int(t) for t in args.prompt.split(",")] if args.prompt else DEFAULT_PROMPT
    with Qwen35GGUFResidentSession(
        args.model, max_sequence_length=args.max_seq, require_cached_build=args.require_cached
    ) as session:
        first = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
        cur = int(first.token_id)
        for _ in range(args.warmup):
            cur = int(session.step(cur, return_logits=False).token_id)
        session.runtime.device_synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            cur = int(session.step(cur, return_logits=False).token_id)
        session.runtime.device_synchronize()
        ms = (time.perf_counter() - t0) / max(args.steps, 1) * 1000.0
    print(f"[gguf-decode] {args.steps} steps  {ms:.3f} ms/step", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Kernel-family bucketing.
# --------------------------------------------------------------------------- #
def _family(name: str) -> str:
    n = re.sub(r"^void\s+", "", name.strip()).replace("(anonymous namespace)::", "")
    n = re.sub(r"<[^>]*>", "", n).split("(")[0]
    return re.sub(r"_kernel$", "", n).strip()


def _bucket(f: str) -> str:
    if "router" in f:
        return "moe_router"
    if any(k in f for k in ("linear_attn", "gdn", "ssm", "_conv_")):
        return "gdn_linear_attn"
    if f.startswith(("q4_k_t16_selected", "qk_t16_selected", "gguf_k_selected", "gguf_q4_k_selected")):
        return "moe_selected_gemv"
    if f.startswith("q8_0_t16"):
        return "dense_q8_0_gemv"
    if "dense_gemv" in f:
        return "dense_gemv_bf16"
    if "q6_k_t16_gemv" in f or "lm_head" in f or "argmax" in f:
        return "lm_head"
    if "rmsnorm" in f or "rotary" in f:
        return "rmsnorm_rope"
    if any(k in f for k in ("silu", "weighted_sum", "combine")):
        return "moe_combine_silu"
    if any(k in f for k in ("attn", "flash", "softmax", "paged_kv", "paged_full")):
        return "attn_core"
    if "embedding" in f:
        return "embedding"
    if any(k in f for k in ("copyBuffer", "fillBuffer", "rocclr", "memcpy")):
        return "memcpy_fill"
    return "other:" + f


def _process_csv(csv_path: Path, top: int) -> dict[str, Any]:
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit(f"empty kernel trace CSV: {csv_path}")
    cols = rows[0].keys()

    def col(sub: str) -> str:
        return next(c for c in cols if sub in c)

    name_c, s_c, e_c = col("Kernel_Name"), col("Start_Timestamp"), col("End_Timestamp")
    fam_agg: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    bucket_agg: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    total = 0.0
    for r in rows:
        dur = (int(r[e_c]) - int(r[s_c])) / 1000.0  # microseconds
        f = _family(r[name_c])
        b = _bucket(f)
        fam_agg[f][0] += 1
        fam_agg[f][1] += dur
        bucket_agg[b][0] += 1
        bucket_agg[b][1] += dur
        total += dur

    def emit(agg: dict[str, list[float]]) -> list[dict[str, Any]]:
        out = []
        for key, (calls, us) in sorted(agg.items(), key=lambda x: -x[1][1]):
            out.append(
                {
                    "name": key,
                    "calls": int(calls),
                    "total_us": round(us, 1),
                    "pct": round(us / total * 100.0, 2),
                    "us_per_call": round(us / calls, 3) if calls else 0.0,
                }
            )
        return out

    return {
        "total_kernels": len(rows),
        "total_gpu_us": round(total, 1),
        "buckets": emit(bucket_agg),
        "top_kernels": emit(fam_agg)[:top],
    }


# --------------------------------------------------------------------------- #
# Parent mode: warm-build, rocprof the child, process CSV, emit artifact.
# --------------------------------------------------------------------------- #
def _run_parent(args: argparse.Namespace) -> int:
    rocprofv3 = shutil.which("rocprofv3") or "rocprofv3"
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    # Pin the compiler-version file so the JIT cache key is stable under rocprofv3.
    cvf = Path(args.compiler_version_file)
    if not cvf.exists():
        cvf.parent.mkdir(parents=True, exist_ok=True)
        ver = subprocess.run(["hipcc", "--version"], capture_output=True, text=True)
        cvf.write_text(ver.stdout or "hipcc-unknown\n", encoding="utf-8")

    child_base = [
        sys.executable,
        str(Path(__file__)),
        "--child",
        "--model", str(args.model),
        "--steps", str(args.steps),
        "--warmup", str(args.warmup),
        "--max-seq", str(args.max_seq),
        "--compiler-version-file", str(cvf),
    ]
    if args.prompt:
        child_base += ["--prompt", args.prompt]

    env = os.environ.copy()
    env.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")
    env["HIPENGINE_COMPILER_VERSION_FILE"] = str(cvf)

    # 1) Warm-build pass (no rocprof): build + cache every .so so the profiled
    #    process never spawns hipcc.
    if not args.skip_warmbuild:
        print("[gguf-decode-rocprof] warm-build pass (no profiler)...", flush=True)
        subprocess.run(child_base, cwd=REPO_ROOT, env=env, check=True)

    # 2) Profiled pass with --require-cached so any missing .so is a hard error
    #    instead of a silent in-trace hipcc spawn.
    out_dir = raw_root / "trace"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rocprof_cmd = [
        rocprofv3,
        "--kernel-trace",
        "--output-format", "csv",
        "-d", str(out_dir),
        "--",
        *child_base,
        "--require-cached",
    ]
    print(f"[gguf-decode-rocprof] {' '.join(rocprof_cmd)}", flush=True)
    subprocess.run(rocprof_cmd, cwd=REPO_ROOT, env=env, check=True)

    csvs = list(out_dir.rglob("*_kernel_trace.csv"))
    if not csvs:
        raise SystemExit(f"no kernel_trace.csv produced under {out_dir}")
    summary = _process_csv(csvs[0], args.top)
    summary.update(
        {
            "schema": "hipengine.gguf_decode_rocprof.v1",
            "date": date.today().isoformat(),
            "model": str(args.model),
            "hardware": "AMD Radeon 8060S / Ryzen AI Max+ 395 (gfx1151)",
            "passes": {"prefill": 1, "warmup_steps": args.warmup, "timed_steps": args.steps},
            "note": (
                "Whole-process trace (prefill + warmup + timed decode). Family "
                "percentages are of total GPU time across all passes; the decode "
                "families dominate. Diagnostic only."
            ),
            "command": " ".join([Path(sys.executable).name] + sys.argv),
        }
    )

    # Ranked table to stdout.
    print(f"\nTOTAL gpu_us={summary['total_gpu_us']:.0f}  kernels={summary['total_kernels']}")
    print("\n=== HIGH-LEVEL BUCKETS ===")
    print(f"{'bucket':32s} {'calls':>7s} {'tot_us':>10s} {'%':>6s}")
    for b in summary["buckets"]:
        print(f"{b['name'][:32]:32s} {b['calls']:7d} {b['total_us']:10.0f} {b['pct']:6.1f}")
    print("\n=== TOP KERNELS ===")
    print(f"{'kernel':52s} {'calls':>6s} {'tot_us':>9s} {'%':>6s} {'us/call':>9s}")
    for k in summary["top_kernels"]:
        print(f"{k['name'][:52]:52s} {k['calls']:6d} {k['total_us']:9.0f} {k['pct']:6.1f} {k['us_per_call']:9.2f}")

    if args.out:
        import json

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\n[gguf-decode-rocprof] wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", action="store_true", help="internal: the process run under rocprofv3")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="", help="comma-separated prompt token ids")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--require-cached", action="store_true")
    ap.add_argument("--skip-warmbuild", action="store_true")
    ap.add_argument("--compiler-version-file", type=Path, default=Path("/tmp/hipengine-hipcc-version.txt"))
    ap.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-gguf-decode-rocprof"))
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    return _run_child(args) if args.child else _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
