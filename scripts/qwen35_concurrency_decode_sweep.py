#!/usr/bin/env python3
"""Concurrency (c>1) decode-throughput sweep for the Qwen3.6 PARO resident model.

Measures aggregate and per-sequence decode tok/s as the number of concurrent
decode sequences (``c``) grows, on a single fixed shape (prompt 512 / decode 128
per sequence by default).  This is the source measurement behind the
"Concurrency" table in the top-level ``README.md`` and the corresponding row in
``benchmarks/README.md``.

Methodology (kept self-consistent so the rows are comparable):

* ``c=1`` is the single-sequence decode path measured with
  ``scripts/qwen35_paro_bench.py --graph-replay-decode`` (the real generate-time
  HIP-graph-replay path), reporting ``throughput.warmed_decode_tok_s``.
* ``c>=2`` is the native batched decode path measured with
  ``scripts/qwen35_batch_retained_bench.py`` (scheduler-owned compact native
  prefill + ``step_batch_native``), reporting ``measurements.decode_tok_s_aggregate``
  (total tok/s across the batch) and ``measurements.decode_tok_s_per_request``
  (per-sequence tok/s).  The retained bench requires ``--batch-size > 1``.

Each ``(c, rep)`` runs in its own process (fresh resident session); the reported
value per ``c`` is the median across ``--reps`` runs.  All runs use cached kernel
builds (``--require-cached-build``) and a precomputed compiler-version file, per
the ``AGENTS.md`` profiling/JIT guidance.

Example
-------
    HIP_VISIBLE_DEVICES=1 python3 scripts/qwen35_concurrency_decode_sweep.py \\
        --model /models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16 \\
        --fixture /tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json \\
        --compiler-version-file /tmp/hipengine-retained/hipcc-version.txt \\
        --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 8 \\
        --concurrencies 1,2,4,8 --reps 3 \\
        --json benchmarks/results/<date>-hipengine-qwen35-concurrency-decode/summary.json

Exit code is always 0 on a completed sweep; the underlying retained-bench rows
stay ``status=blocked`` (c>1 decode is correctness-gated, not a retained
throughput claim yet), which is expected and does not fail the sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16"
DEFAULT_FIXTURE = "/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json"


def _run_json(cmd: list[str], out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    # The retained bench exits 1 for blocked (non-retained) rows but still
    # writes the JSON; paro_bench exits 0.  Tolerate non-zero and read the JSON.
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if not out.exists():
        raise RuntimeError(f"benchmark did not write {out} for command: {' '.join(cmd)}")
    return json.loads(out.read_text())


def _c1_command(args, out: Path) -> list[str]:
    return [
        "python3", str(REPO_ROOT / "scripts" / "qwen35_paro_bench.py"),
        "--model", args.model,
        "--prompt-length", str(args.prompt_length),
        "--decode-tokens", str(args.decode_tokens),
        "--warmup-decode-tokens", str(args.warmup_decode_tokens),
        "--max-layers", str(args.max_layers),
        "--graph-replay-decode",
        *(["--compiler-version-file", args.compiler_version_file] if args.compiler_version_file else []),
        *(["--require-cached-build"] if args.require_cached_build else []),
        "--json", str(out),
    ]


def _cn_command(args, c: int, out: Path) -> list[str]:
    return [
        "python3", str(REPO_ROOT / "scripts" / "qwen35_batch_retained_bench.py"),
        "--model", args.model,
        "--fixture", args.fixture,
        "--prompt-length", str(args.prompt_length),
        "--batch-size", str(c),
        "--decode-tokens", str(args.decode_tokens),
        "--warmup-decode-tokens", str(args.warmup_decode_tokens),
        "--max-layers", str(args.max_layers),
        *(["--compiler-version-file", args.compiler_version_file] if args.compiler_version_file else []),
        *(["--require-cached-build"] if args.require_cached_build else []),
        "--json", str(out),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--concurrencies", default="1,2,4,8", help="comma-separated c values")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--compiler-version-file")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--work-dir", default="/tmp/hipengine-concurrency-sweep")
    parser.add_argument("--json", type=Path, help="write the aggregated sweep summary here")
    args = parser.parse_args()

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    work_dir = Path(args.work_dir)
    rows: dict[str, dict] = {}

    for c in concurrencies:
        agg: list[float] = []
        per: list[float] = []
        for rep in range(args.reps):
            out = work_dir / f"c{c}-r{rep}.json"
            t0 = time.perf_counter()
            if c == 1:
                payload = _run_json(_c1_command(args, out), out)
                value = float(payload["throughput"]["warmed_decode_tok_s"])
                agg.append(value)
                per.append(value)
            else:
                payload = _run_json(_cn_command(args, c, out), out)
                measurements = payload["measurements"]
                agg.append(float(measurements["decode_tok_s_aggregate"]))
                per.append(float(measurements["decode_tok_s_per_request"]))
            print(f"c{c} rep{rep}: agg={agg[-1]:.2f} per={per[-1]:.2f} ({time.perf_counter()-t0:.0f}s)", flush=True)
        rows[str(c)] = {
            "concurrency": c,
            "decode_tok_s_aggregate_median": statistics.median(agg),
            "decode_tok_s_per_request_median": statistics.median(per),
            "decode_tok_s_aggregate_runs": agg,
            "decode_tok_s_per_request_runs": per,
            "path": "paro_bench_graph_replay" if c == 1 else "retained_native_batch",
        }
        print(
            f"== c{c}: agg_median={rows[str(c)]['decode_tok_s_aggregate_median']:.2f} "
            f"per_median={rows[str(c)]['decode_tok_s_per_request_median']:.2f}",
            flush=True,
        )

    summary = {
        "kind": "concurrency_decode_throughput_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "retained_ready": False,
        "host": f"gfx1100/RDNA3 via HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}",
        "model": Path(args.model).name,
        "shape": {
            "prompt_length": args.prompt_length,
            "decode_tokens": args.decode_tokens,
            "warmup_decode_tokens": args.warmup_decode_tokens,
            "max_layers": args.max_layers,
            "reps": args.reps,
            "kv_storage_dtype": "bf16",
        },
        "methodology": {
            "c1": "scripts/qwen35_paro_bench.py --graph-replay-decode -> throughput.warmed_decode_tok_s",
            "cN": "scripts/qwen35_batch_retained_bench.py --batch-size N -> measurements.decode_tok_s_{aggregate,per_request}",
            "aggregate_across_runs": "median",
        },
        "rows": rows,
    }
    print("FINAL " + json.dumps(summary), flush=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
