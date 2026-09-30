#!/usr/bin/env python3
"""Regenerate both halves of the Gemma 4 26B-A4B scoreboard in one command.

The punchlist publishes two tables that must stay in step: an unprofiled
**topline** (prefill and decode tok/s from the campaign bench) and a
**family breakdown** (device-busy milliseconds per kernel family from a
``rocprofv3 --kernel-trace`` census). They are separate instruments with
separate caveats -- profiled decode runs about 25% slower than unprofiled, and
their difference is not a measured host gap -- so they are captured together,
against one commit, on one physical GPU, and written as one artifact.

Re-run it after each landed optimization with a new ``--tag``::

    scripts/gemma4_scoreboard_snapshot.py --gpu 1 --tag baseline
    scripts/gemma4_scoreboard_snapshot.py --gpu 1 --tag p14

then replace the affected rows in
``docs/campaigns/GEMMA4-26B-A4B-PUNCHLIST.md`` from the artifact this writes.
``--gpu`` is the physical GPU index: 1 is the RX 7900 XTX primary lane, 0 is
the W7900 secondary lane. Keep the two lanes in separate artifacts.

What each phase does, in order:

1. **Warm** -- runs the census driver outside the profiler so the JIT cache is
   built before anything measures. A profiled run with a cold cache would
   either deadlock on ``clang++ --version`` or charge compilation to the trace.
2. **Topline** -- ``gemma4_campaign_bench.py`` at each prompt length, which
   writes its own artifact carrying git, GPU and workload provenance.
3. **Census** -- ``rocprofv3 --kernel-trace`` around the same driver, then
   ``rollup`` bucketing every kernel into a family.

Raw traces land in ``/tmp`` and are never committed, per ``docs/BENCHMARK.md``.
Only the compact JSON leaves this script.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = (1024, 4096)
DEFAULT_OUTPUT = 128
DEFAULT_SAMPLES = 3
DEFAULT_WARMUP = 1
DEFAULT_DECODE_STEPS = 64
DEFAULT_PREFILL_REPEATS = 2
DEFAULT_CONTEXT = 8192


class StepFailed(RuntimeError):
    """A subprocess exited non-zero, so the snapshot is incomplete."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str) -> str:
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (done.stdout or "").strip()


def provenance(gpu: int) -> dict[str, Any]:
    """Commit, working-tree and toolchain state for this snapshot."""

    hipcc = _run(["hipcc", "--version"], timeout=60)
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_files": _git("status", "--porcelain").splitlines()[:40],
        "physical_gpu_index": gpu,
        "gpu_name": _gpu_name(gpu),
        "rocm_smi": _run(
            ["rocm-smi", "--showid", "--showproductname"], timeout=60
        ).splitlines()[-12:],
        "hipcc_version_first_line": (hipcc.splitlines() or [""])[0],
        "captured_at": _now(),
    }


def _run(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 7200) -> str:
    completed = subprocess.run(
        command, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout, check=False
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-25:]
        raise StepFailed(
            f"command failed with {completed.returncode}: {shlex.join(command)}\n"
            + "\n".join(tail)
        )
    return (completed.stdout or "") + (completed.stderr or "")


def _gpu_name(gpu: int) -> str:
    """Name of physical GPU ``gpu`` from rocm-smi.

    rocm-smi puts the ``GPU[N]`` marker and the ``Device Name:`` label on the
    *same* line (``GPU[0]\t\t: Device Name: \t\t<name>``), so parse line by
    line rather than tracking a previous marker line.
    """

    try:
        done = subprocess.run(
            ["rocm-smi", "--showid", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in (done.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("GPU[") or "Device Name:" not in stripped:
            continue
        close = stripped.find("]")
        open_b = stripped.find("[")
        if close < 0 or open_b < 0:
            continue
        if stripped[open_b + 1 : close] != str(gpu):
            continue
        return stripped.split("Device Name:", 1)[1].strip()
    return ""


def base_env(gpu: int) -> dict[str, str]:
    """One HIP visibility story for every phase, so device 0 is always ``gpu``.

    ``HIP_VISIBLE_DEVICES`` is removed rather than set: a stale value outranks
    ``ROCR_VISIBLE_DEVICES`` and would silently measure the other card.
    """

    env = dict(os.environ)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["ROCR_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(REPO)
    return env


def cached_build_env(gpu: int, compiler_file: Path) -> dict[str, str]:
    """The profiled environment: refuse to rebuild, or rocprofv3 deadlocks
    waiting on ``clang++ --version`` from inside the traced process."""

    env = base_env(gpu)
    env["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    env["HIPENGINE_COMPILER_VERSION_FILE"] = str(compiler_file)
    return env


def warm_jit(gpu: int) -> None:
    print("[snapshot] warming the JIT cache (unprofiled, may build)", flush=True)
    _run(
        [
            sys.executable,
            "scripts/gemma4_family_census.py",
            "drive",
            "--warm-only",
        ],
        env=base_env(gpu),
        timeout=3600,
    )


def expect_gpu_for(gpu: int) -> str:
    """The device-name expectation the campaign bench validates for a lane.

    ``gemma4_campaign_bench`` defaults to the XTX name, so a ``--gpu 0``
    snapshot died on the check instead of measuring the W7900.
    """
    return "W7900" if gpu == 0 else "RX 7900 XTX"


def run_topline(gpu: int, prompt: int, out_path: Path, tag: str) -> dict[str, Any]:
    print(f"[snapshot] topline: prompt {prompt}", flush=True)
    command = [
        sys.executable,
        "scripts/gemma4_campaign_bench.py",
        "--prompt", str(prompt),
        "--output", str(DEFAULT_OUTPUT),
        "--samples", str(DEFAULT_SAMPLES),
        "--warmup", str(DEFAULT_WARMUP),
        "--context", str(DEFAULT_CONTEXT),
        "--out", str(out_path),
        "--label", f"scoreboard-{tag}",
        "--expect-gpu", expect_gpu_for(gpu),
    ]
    _run(command, env=base_env(gpu), timeout=7200)
    return json.loads(out_path.read_text())


def run_census(
    gpu: int, prompt: int, tag: str, trace_root: Path
) -> tuple[dict[str, Any], Path]:
    print(f"[snapshot] family census: prompt {prompt}", flush=True)
    compiler_file = Path("/tmp") / f"g4_scoreboard_hipcc_{tag}.txt"
    compiler_file.write_text(_run(["hipcc", "--version"], timeout=60))

    trace_dir = trace_root / f"trace_{prompt}"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "rocprofv3",
            "--kernel-trace",
            "--output-format", "csv",
            "-d", str(trace_dir),
            "-o", "run",
            "--",
            sys.executable,
            "scripts/gemma4_family_census.py",
            "drive",
            "--prompt", str(prompt),
            "--decode", str(DEFAULT_DECODE_STEPS),
            "--prefill-repeats", str(DEFAULT_PREFILL_REPEATS),
        ],
        env=cached_build_env(gpu, compiler_file),
        timeout=7200,
    )

    csv_path = trace_dir / "run_kernel_trace.csv"
    if not csv_path.exists():
        raise StepFailed(f"rocprofv3 produced no kernel trace at {csv_path}")

    rollup_path = trace_root / f"family_{prompt}.json"
    _run(
        [
            sys.executable,
            "scripts/gemma4_family_census.py",
            "rollup",
            "hipengine",
            str(csv_path),
            "--decode-steps", str(DEFAULT_DECODE_STEPS),
            "--json", str(rollup_path),
        ],
        timeout=1800,
    )
    rollup = json.loads(rollup_path.read_text())
    rollup["command"] = shlex.join(
        [
            "rocprofv3", "--kernel-trace", "--output-format", "csv",
            "-d", str(trace_dir), "-o", "run", "--",
            sys.executable, "scripts/gemma4_family_census.py", "drive",
            "--prompt", str(prompt), "--decode", str(DEFAULT_DECODE_STEPS),
            "--prefill-repeats", str(DEFAULT_PREFILL_REPEATS),
        ]
    )
    rollup["trace_kernel_rows"] = sum(1 for _ in csv_path.open()) - 1
    rollup_path.write_text(json.dumps(rollup, indent=1) + "\n")
    return rollup, rollup_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", type=int, default=1, help="physical GPU index (1 = XTX, 0 = W7900)")
    parser.add_argument("--tag", default="baseline", help="suffix for artifact names")
    parser.add_argument("--prompts", type=int, nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--gpu-name-hint", default="", help="override the GPU name in artifacts")
    parser.add_argument("--skip-topline", action="store_true")
    parser.add_argument("--skip-census", action="store_true")
    parser.add_argument("--keep-traces", action="store_true", help="leave raw traces in /tmp")
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results_dir = REPO / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_root = Path("/tmp") / f"g4_scoreboard_{args.tag}"
    trace_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    prov = provenance(args.gpu)
    if args.gpu_name_hint:
        prov["gpu_name"] = args.gpu_name_hint
    print(
        f"[snapshot] gpu={args.gpu} ({prov['gpu_name'] or 'unknown name'}) "
        f"commit={prov['git_commit'][:9]} tag={args.tag} prompts={args.prompts}",
        flush=True,
    )

    topline: dict[str, Any] = {}
    census: dict[str, Any] = {}
    artifacts: dict[str, str] = {}

    if not args.skip_topline:
        warm_jit(args.gpu)
        for prompt in args.prompts:
            name = f"{stamp}-gemma4-topline-{prompt}-{args.tag}.json"
            out_path = results_dir / name
            row = run_topline(args.gpu, prompt, out_path, args.tag)
            stats = row.get("stats", {})
            topline[str(prompt)] = {
                "prefill_tps": stats.get("prefill_tps"),
                "prefill_s": stats.get("prefill_s"),
                "decode_tps": stats.get("decode_tps"),
                "decode_tps_min": stats.get("decode_tps_min"),
                "decode_tps_max": stats.get("decode_tps_max"),
                "decode_tps_stdev": stats.get("decode_tps_stdev"),
                "first_token_s": stats.get("first_token_s"),
                "wall_s": stats.get("wall_s"),
                "artifact": f"benchmarks/results/{name}",
                "command": row.get("command"),
            }
            artifacts[f"topline_{prompt}"] = f"benchmarks/results/{name}"
            print(
                f"[snapshot]   prompt {prompt}: prefill "
                f"{topline[str(prompt)]['prefill_tps']:.1f} tok/s, decode "
                f"{topline[str(prompt)]['decode_tps']:.2f} tok/s",
                flush=True,
            )

    if not args.skip_census:
        warm_jit(args.gpu)
        for prompt in args.prompts:
            rollup, rollup_path = run_census(args.gpu, prompt, args.tag, trace_root)
            name = rollup_path.name
            final = results_dir / f"{stamp}-gemma4-family-{prompt}-{args.tag}.json"
            rollup["prefill_prompt_tokens"] = prompt
            final.write_text(json.dumps(rollup, indent=1) + "\n")
            census[str(prompt)] = rollup
            artifacts[f"family_{prompt}"] = f"benchmarks/results/{final.name}"
            busy = rollup.get("prefill", {}).get("busy_ms")
            print(
                f"[snapshot]   prompt {prompt}: prefill device busy "
                f"{busy if busy is None else round(busy, 1)} ms",
                flush=True,
            )

    combined = {
        "schema": 1,
        "label": f"gemma4-scoreboard-{args.tag}",
        "created_at": _now(),
        "elapsed_s": round(time.time() - started, 1),
        "provenance": prov,
        "protocol": {
            "topline": (
                f"scripts/gemma4_campaign_bench.py --prompt P --output {DEFAULT_OUTPUT} "
                f"--samples {DEFAULT_SAMPLES} --warmup {DEFAULT_WARMUP} "
                f"--context {DEFAULT_CONTEXT}; unprofiled, median of "
                f"{DEFAULT_SAMPLES} samples, fixed length (EOS ignored)"
            ),
            "family": (
                "rocprofv3 --kernel-trace around scripts/gemma4_family_census.py drive "
                f"--prompt P --decode {DEFAULT_DECODE_STEPS} "
                f"--prefill-repeats {DEFAULT_PREFILL_REPEATS}, rolled up by "
                "scripts/gemma4_family_census.py rollup; device-busy sums, profiled"
            ),
            "kv_dtype": "bf16",
            "execution_profile": "production route, default selection",
            "comparator": (
                "llama.cpp rows are unchanged and come from the recorded "
                "a97cce8 comparison; this artifact refreshes hipEngine only"
            ),
        },
        "topline": topline,
        "family": census,
        "artifacts": artifacts,
    }
    combined_path = results_dir / f"{stamp}-gemma4-scoreboard-{args.tag}.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n")

    if not args.keep_traces:
        shutil.rmtree(trace_root, ignore_errors=True)

    print(f"[snapshot] artifact=benchmarks/results/{combined_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StepFailed as failure:
        print(f"[snapshot] FAILED: {failure}", file=sys.stderr, flush=True)
        raise SystemExit(1)