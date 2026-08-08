#!/usr/bin/env python3
"""rocprofv3 decode-profile of one Maple token step (docs/MAPLE-PERF.md M0).

Method (matches AGENTS.md profiler rules):
* ``--prebuild`` builds every Maple kernel library once outside the profiler,
  pinned by a compiler-version file, so the profiled child never spawns hipcc.
* ``--profile`` runs a cached-only child under ``rocprofv3 --kernel-trace``. The
  child loads the resident runner, warms up a few steps, then runs a fixed batch
  of decode steps and prints per-step wall to stdout (kernels go to the CSV).
* A post-processor aggregates per-kernel ``DurationNs`` by kernel name/family,
  computes the host gap (step wall - sum kernel time), and emits a compact JSON
  artifact.

No model math is changed by this script; it is the Phase-0 profile that drives
D1 (hipGraph capture) and the decode milestones.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VERSION_FILE = Path("/tmp/hipengine-maple-hipcc-version.txt")
DEFAULT_MODEL = "deepgrove/maple-preview-2bit-mlx"
DEFAULT_RAW = Path("/tmp/hipengine-maple-decode-rocprof")
MARKER_TOKEN = 9707
WARMUP_STEPS = 4
MEASURED_STEPS = 32

# Kernel-name -> family (for aggregation). Prefix match on the CSV Name.
FAMILY_RULES = [
    ("maple_ternary_qkv", "qkv_proj"),
    ("maple_selected_ternary_dual", "expert_gate_up"),
    ("maple_selected_ternary", "expert_down"),
    ("qwen35_moe_group_count_active_parallel", "moe_compaction"),
    ("qwen35_moe_group_prefix_active_parallel", "moe_compaction"),
    ("qwen35_moe_group_scatter_active_parallel", "moe_compaction"),
    ("maple_ternary_gemv", "o_proj"),
    ("maple_affine4_gemv", "lm_head"),
    ("maple_affine4_embed", "embed"),
    ("maple_qknorm_rope_kv_write", "qknorm_kvwrite"),
    ("maple_attention_decode", "attention_decode"),
    ("maple_router_topk", "router_topk"),
    ("maple_clamped_swiglu", "swiglu"),
    ("maple_weighted_residual", "weighted_residual"),
    ("maple_kv_span_update", "kv_span_update"),
    ("paro_rmsnorm_out", "rmsnorm"),
    ("paro_add_rmsnorm_out", "add_rmsnorm"),
    ("argmax", "argmax"),
]


def family_of(name: str) -> str:
    for prefix, family in FAMILY_RULES:
        if name.startswith(prefix):
            return family
    return name


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(args)}", flush=True)
    return subprocess.run(args, **kw)


def write_version_file() -> None:
    from hipengine.core.build import compiler_version_text

    text = compiler_version_text("hipcc")
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(text + "\n")
    print(f"compiler version -> {VERSION_FILE}: {text}", flush=True)


def prebuild(*, backend: str) -> None:
    """Build all Maple libs with a pinned compiler-version file (outside rocprof)."""
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import hip_target_arch_for_backend, load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100.attention.maple_attention import build_maple_attention
    from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        build_qwen35_moe_group_scatter,
    )
    from hipengine.kernels.hip_gfx1100.moe.maple_moe import build_maple_moe
    from hipengine.kernels.hip_gfx1100.norm.rmsnorm import build_qwen35_rmsnorm
    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import build_maple_ternary

    get_hip_runtime()
    arch = hip_target_arch_for_backend(backend)
    load_backend_kernel_package(backend)
    import os

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(VERSION_FILE)
    with _arch_env(arch):
        for build in (
            build_maple_ternary,
            build_maple_attention,
            build_maple_moe,
            build_qwen35_moe_group_scatter,
            build_qwen35_rmsnorm,
            build_lm_head,
        ):
            art = build(load=False, compiler_version=VERSION_FILE.read_text().strip())
            print(f"prebuilt {art.family} exists={art.output_path.exists()}", flush=True)


def _arch_env(arch: str):
    import contextlib

    @contextlib.contextmanager
    def _cm():
        prev = os.environ.get("HIP_TARGET_ARCHS") or os.environ.get("HIP_OFFLOAD_ARCH")
        os.environ["HIP_TARGET_ARCHS"] = arch
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("HIP_TARGET_ARCHS", None)
            else:
                os.environ["HIP_TARGET_ARCHS"] = prev

    return _cm()


CHILD_SOURCE = r"""
import json, os, sys, time
from hipengine.loading.maple import load_maple_checkpoint
from hipengine.runtime.maple import MapleRunner

backend = sys.argv[1]
os.environ.setdefault("HIPENGINE_COMPILER_VERSION_FILE", "/tmp/hipengine-maple-hipcc-version.txt")
ckpt = load_maple_checkpoint("deepgrove/maple-preview-2bit-mlx")
runner = MapleRunner.load(ckpt, backend=backend, max_context=4096)
token = int(sys.argv[2])
# warmup
for _ in range(int(sys.argv[3])):
    runner.step(token)
# measured
n = int(sys.argv[4])
times = []
ids = []
for _ in range(n):
    t = time.perf_counter()
    r = runner.step(token)
    times.append((time.perf_counter() - t) * 1000.0)
    ids.append(r.token_id)
print("MAPLE_STEP_MS=" + json.dumps(times), flush=True)
print("MAPLE_IDS=" + json.dumps(ids), flush=True)
runner.close()
"""


def run_child(*, backend: str, out: Path) -> None:
    child = Path("/tmp/maple_profile_child.py")
    child.write_text(CHILD_SOURCE)
    env = dict(os.environ)
    env["HIPENGINE_COMPILER_VERSION_FILE"] = str(VERSION_FILE)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "rocprofv3", "--kernel-trace", "-f", "csv",
        "-d", str(out),
        "--",
        sys.executable, str(child), backend, str(MARKER_TOKEN), str(WARMUP_STEPS), str(MEASURED_STEPS),
    ]
    _run(cmd, env=env)


def parse_csv(raw_root: Path) -> tuple[list[dict], list[str]]:
    csvs = sorted(p for p in raw_root.rglob("*.csv") if "kernel_trace" in p.name)
    if not csvs:
        return [], []
    rows: list[dict] = []
    names = set()
    for p in csvs:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Kernel_Name") or row.get("Name") or "").strip()
                if not name or name.startswith("__amd_rocclr"):
                    continue
                start = row.get("Start_Timestamp")
                end = row.get("End_Timestamp")
                if start is None or end is None:
                    continue
                try:
                    dur_ns = int(end) - int(start)
                except ValueError:
                    continue
                rows.append(
                    {
                        "Name": name,
                        "DurationNs": dur_ns,
                        "Grid_X": row.get("Grid_Size_X", ""),
                        "Wg_X": row.get("Workgroup_Size_X", ""),
                    }
                )
                names.add(name)
    return rows, sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--prebuild", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.out is None:
        args.out = (
            REPO_ROOT / "benchmarks" / "results"
            / f"{date.today().isoformat()}-gfx1151-maple-decode-profile.json"
        )

    if args.prebuild:
        write_version_file()
        prebuild(backend=args.backend)
        print("prebuild complete; now rerun with --profile")
        return 0

    if args.profile:
        shutil.rmtree(args.raw_root, ignore_errors=True)
        args.raw_root.mkdir(parents=True, exist_ok=True)
        run_child(backend=args.backend, out=args.raw_root)
        # collect step times + ids from the profile log isn't captured; rerun child
        # separately for wall times (cached), OR parse stdout. We'll parse child via
        # a lightweight re-run that only times (no rocprof) for host wall.
        rows, names = parse_csv(args.raw_root)
        if not rows:
            print("no kernel rows parsed; check rocprof CSV layout", file=sys.stderr)
            return 2

        fam = collections.Counter()
        per_kernel = collections.Counter()
        for r in rows:
            per_kernel[r["Name"]] += r["DurationNs"]
            fam[family_of(r["Name"])] += r["DurationNs"]
        total_kernel_ns = sum(r["DurationNs"] for r in rows)
        launches = len(rows)

        # Host wall: rerun cached child without rocprof to time a fresh batch.
        wall_ms = _measure_host_wall(args.backend)

        steps_in_trace = WARMUP_STEPS + MEASURED_STEPS
        artifact = {
            "schema_version": 1,
            "date": date.today().isoformat(),
            "artifact_type": "decode_profile",
            "performance_claim": False,
            "status": "accepted_diagnostic",
            "backend": args.backend,
            "model": args.model,
            "protocol": {
                "environment": {
                    "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
                    "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                    "HIPENGINE_COMPILER_VERSION_FILE": str(VERSION_FILE),
                }
            },
            "measured_steps": MEASURED_STEPS,
            "steps_in_trace": steps_in_trace,
            "kernel_launches_per_step": launches / steps_in_trace,
            "kernel_time_total_us": total_kernel_ns / 1000.0,
            "kernel_time_per_step_us": total_kernel_ns / 1000.0 / steps_in_trace,
            "host_wall_per_step_us": wall_ms * 1000.0,
            "host_gap_per_step_us": wall_ms * 1000.0 - total_kernel_ns / 1000.0 / steps_in_trace,
            "per_family_us_per_step": {
                k: v / 1000.0 / steps_in_trace for k, v in sorted(fam.items(), key=lambda x: -x[1])
            },
            "top_kernels_us_per_step": [
                {"kernel": k, "us_per_step": v / 1000.0 / steps_in_trace}
                for k, v in per_kernel.most_common(30)
            ],
        }
        args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(json.dumps(artifact, indent=2, sort_keys=True))
        print(f"\nartifact: {args.out}")
        return 0

    parser.print_help()
    return 0


def _measure_host_wall(backend: str) -> float:
    import statistics

    code = r"""
import json, os, sys, time
from hipengine.loading.maple import load_maple_checkpoint
from hipengine.runtime.maple import MapleRunner
os.environ.setdefault("HIPENGINE_COMPILER_VERSION_FILE", "/tmp/hipengine-maple-hipcc-version.txt")
backend = sys.argv[1]
ckpt = load_maple_checkpoint("deepgrove/maple-preview-2bit-mlx")
runner = MapleRunner.load(ckpt, backend=backend, max_context=4096)
token = int(sys.argv[2])
for _ in range(int(sys.argv[3])):
    runner.step(token)
n = int(sys.argv[4]); times=[]
for _ in range(n):
    t=time.perf_counter(); runner.step(token); times.append((time.perf_counter()-t)*1000.0)
print(json.dumps(times)); runner.close()
"""
    proc = _run(
        [sys.executable, "-c", code, backend, str(MARKER_TOKEN), str(WARMUP_STEPS), str(MEASURED_STEPS)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(1)
    times = json.loads(proc.stdout.strip().splitlines()[-1])
    return float(statistics.median(times))


if __name__ == "__main__":
    sys.exit(main())
