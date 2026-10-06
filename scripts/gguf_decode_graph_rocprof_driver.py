#!/usr/bin/env python3
"""Decode kernel-attribution driver for rocprofv3 kernel traces.

Single-window modes with a deliberate 0.5 s GPU idle gap before the measured
window, so the window is the trailing kernel burst after the last long idle
interval in the device timeline - no marker trace is needed to slice.

Decode kernel-profile driver, single-window modes.

mode=graph: prefill, warm, capture, warm-replay, [0.5s GPU idle gap],
``--steps``-step measured graph replay, exit.
mode=eager: prefill, warm, [0.5s GPU idle gap], ``--steps`` eager steps, exit.
mode=prefill: one resident session throughout, mirroring the sweep's own
methodology (`_run_existing_session_once` + `reset()`): a warm-up prefill plus
a few eager steps, then ``reset()`` (resident weights and scratch retained),
[0.5s GPU idle gap], one measured fresh ``--prefill-tokens`` prefill, exit --
so the trailing window is exactly one matched-shape prefill with warm weights
and pools, like the paired baseline's measured prefill (campaign
UD-GFX1151-OPTIMIZE2 E1).

The gap makes the measured window the trailing kernel burst after the last
long GPU idle interval, so no markers are needed to slice the trace.

The profiled process must not compile. ``docs/OPTIMIZATION.md`` section 7
requires the JIT cache to be prebuilt outside the profiler and the profiled run
to pin the compiler version and require the cached artifacts, because a
``rocprofv3``-wrapped ``hipcc``/clang child corrupts the trace. So the
default is to require the cached build: run once without the profiler to build,
then under the profiler with the same ``HIPENGINE_COMPILER_VERSION_FILE``. Pass
``--allow-build`` only for that unprofiled warm build.

Usage (prebuild the JIT cache once without the profiler, then)::

    HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipcc-version.txt \\
        python scripts/gguf_decode_graph_rocprof_driver.py MODEL.gguf graph --allow-build
    HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipcc-version.txt \\
        rocprofv3 --kernel-trace --output-format csv -d OUT_DIR -o NAME -- \\
        python scripts/gguf_decode_graph_rocprof_driver.py MODEL.gguf graph

Per-kernel pure durations come from the trailing window of the kernel trace;
summarize with ``scripts/gguf_decode_census_summary.py``. See
``benchmarks/results/2026-09-11-decode-graph-attribution/`` for the derived
four-arm attribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFResidentSession,
)


def _compiler_version(explicit: Path | None) -> str:
    """The pinned compiler-version text, or a refusal naming the fix."""

    if explicit is not None:
        return explicit.read_text().strip()
    from_env = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE", "")
    if from_env:
        return Path(from_env).read_text().strip()
    raise SystemExit(
        "refusing to profile without a pinned compiler version: pass "
        "--compiler-version-file or set HIPENGINE_COMPILER_VERSION_FILE so the "
        "profiled process can require the cache the warm build populated "
        "(docs/OPTIMIZATION.md section 7)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("mode", choices=("graph", "eager", "prefill"))
    parser.add_argument("--steps", type=int, default=32,
                        help="measured decode steps (default 32, the published decode protocol)")
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--gap-s", type=float, default=0.5,
                        help="idle gap before the measured prefill in mode=prefill "
                             "(default 0.5; the window slicer needs a gap >= the "
                             "summary tool's 0.4 s cut, but idle lets device clocks "
                             "downshift before a single measured call)")
    parser.add_argument("--allow-build", action="store_true",
                        help="permit JIT compilation; only for the unprofiled warm build")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the wall-time record here instead of stdout")
    args = parser.parse_args()

    compiler_version = _compiler_version(args.compiler_version_file)
    steps = int(args.steps)

    def _session() -> Qwen35GGUFResidentSession:
        return Qwen35GGUFResidentSession(
            args.model,
            compiler_version=compiler_version,
            require_cached_build=not args.allow_build,
            max_sequence_length=args.max_sequence_length,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        )

    if args.mode == "prefill":
        # One session, sweep-matched: warm-up prefill (and a few eager steps,
        # as the sweep's warmup runs do) warms weights/pools, reset() zeroes
        # resident state WITHOUT freeing weights or scratch, then after the
        # gap the measured prefill runs fresh-shaped on the warm session.
        # Everything before the gap (construction copies, warm-up kernels,
        # reset memsets) is excluded from the trailing-burst window.
        with _session() as session:
            warm_ids = list(np.random.default_rng(7).integers(1000, 50000, args.prefill_tokens))
            cur = session.prefill(warm_ids, use_bulk=True, bulk_attention_mode="bulk")
            for _ in range(4):
                cur = session.step(int(cur.token_id))
            session.runner.runtime.stream_synchronize(0)
            session.reset()
            session.runner.runtime.stream_synchronize(0)
            time.sleep(args.gap_s)
            ids = list(np.random.default_rng(7).integers(1000, 50000, args.prefill_tokens))
            t0 = time.perf_counter()
            session.prefill(ids, use_bulk=True, bulk_attention_mode="bulk")
            session.runner.runtime.stream_synchronize(0)
            wall = time.perf_counter() - t0
            position = int(session.position)
        record = {
            "model": str(args.model),
            "model_name": args.model.name,
            "mode": args.mode,
            "prefill_tokens": int(args.prefill_tokens),
            "max_sequence_length": int(args.max_sequence_length),
            "require_cached_build": not args.allow_build,
            "wall_s": wall,
            "ms_per_token": 1000.0 * wall / args.prefill_tokens,
            "tok_s": args.prefill_tokens / wall,
            "position_after": position,
        }
        text = json.dumps(record, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.write_text(text)
        print(text, end="")
        return 0

    with _session() as session:
        rng = np.random.default_rng(7)
        ids = list(rng.integers(1000, 50000, args.prefill_tokens))
        cur = session.prefill(ids, use_bulk=True, bulk_attention_mode="bulk")
        for _ in range(4):
            cur = session.step(int(cur.token_id))
        graph = None
        if args.mode == "graph":
            graph = session.capture_decode_graph(
                position=session.position,
                steps_per_replay=1,
                max_replay_steps=steps + 8,
                record_steps=0,
            )
            graph.replay(8)
        session.runner.runtime.stream_synchronize(0)
        time.sleep(0.5)  # deliberate GPU idle gap before the measured window
        t0 = time.perf_counter()
        if args.mode == "graph":
            graph.replay(steps)
            session.runner.runtime.stream_synchronize(0)
        else:
            token = int(cur.token_id)
            for _ in range(steps):
                cur = session.step(token, return_logits=False)
                token = int(cur.token_id)
            session.runner.runtime.stream_synchronize(0)
        wall = time.perf_counter() - t0
        if graph is not None:
            graph.close()
        position = int(session.position)

    record = {
        "model": str(args.model),
        "model_name": args.model.name,
        "mode": args.mode,
        "steps": steps,
        "prefill_tokens": int(args.prefill_tokens),
        "max_sequence_length": int(args.max_sequence_length),
        "require_cached_build": not args.allow_build,
        "wall_s": wall,
        "ms_per_token": 1000.0 * wall / steps,
        "tok_s": steps / wall,
        "position_after": position,
    }
    text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
