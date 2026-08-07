#!/usr/bin/env python3
"""M6 batch-decode throughput benchmark: retained c2/c4/c8 rows.

For each batch size c in {2,4,8} it fills a MapleBatchRunner with c independent
requests, runs them through the continuous-batching owner loop for a fixed
number of generated tokens per request, and reports aggregate decode throughput
(tokens/sec across the whole batch). A correctness gate cross-checks batch
decode against a serial autoregressive decode for one seed.

Usage:
    python3 scripts/maple_batch_decode_bench.py [--steps N]
"""

from __future__ import annotations

import argparse
import json
import time

from hipengine.loading.maple import load_maple_checkpoint
from hipengine.runtime.maple import MapleBatchRunner, MapleRunner
from hipengine.runtime.maple_batch import MapleContinuousBatcher

MODEL = "deepgrove/maple-preview-2bit-mlx"
BACKEND = "hip_gfx1151"


def _serial_generate(checkpoint, seed: int, n: int) -> list[int]:
    runner = MapleRunner.load(checkpoint, backend=BACKEND, max_context=64)
    try:
        out = [runner.step(seed).token_id]
        for _ in range(n - 1):
            out.append(runner.step(out[-1]).token_id)
        return out
    finally:
        runner.close()


def bench(checkpoint, c: int, per_req_tokens: int) -> dict:
    runner = MapleBatchRunner.load(
        checkpoint, backend=BACKEND, batch_size=c, per_capacity=64
    )
    batcher = MapleContinuousBatcher(runner)
    try:
        seeds = [9000 + i for i in range(c)]
        for r in range(c):
            batcher.submit(seeds[r], max_new=per_req_tokens)
        started = time.perf_counter()
        while batcher.active():
            batcher.step()
        elapsed = time.perf_counter() - started
    finally:
        runner.close()
    total_tokens = c * per_req_tokens
    return {
        "c": c,
        "steps": per_req_tokens,
        "total_tokens": total_tokens,
        "elapsed_ms": round(elapsed * 1e3, 3),
        "tokens_per_sec": round(total_tokens / elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64, help="tokens per request")
    parser.add_argument("--out", default=None, help="json artifact path")
    args = parser.parse_args()

    checkpoint = load_maple_checkpoint(MODEL)

    # Correctness gate: batch (c=1) autoregressive == serial autoregressive.
    seed = 9707
    n = 4
    serial = _serial_generate(checkpoint, seed, n)
    gate_runner = MapleBatchRunner.load(
        checkpoint, backend=BACKEND, batch_size=1, per_capacity=64
    )
    gate = MapleContinuousBatcher(gate_runner)
    gate.submit(seed, max_new=n)
    while gate.active():
        gate.step()
    gate_runner.close()
    gate_match = gate.completions[0] == serial
    print(f"correctness gate (batch vs serial): {gate_match}")

    results = []
    for c in (2, 4, 8):
        r = bench(checkpoint, c, args.steps)
        results.append(r)
        print(
            f"c={r['c']}: {r['tokens_per_sec']} tok/s "
            f"({r['total_tokens']} tok / {r['elapsed_ms']} ms)"
        )

    artifact = {
        "model": MODEL,
        "backend": BACKEND,
        "hardware": "AMD Radeon Pro W7900 (gfx1151)",
        "per_request_tokens": args.steps,
        "correctness_gate": gate_match,
        "rows": results,
    }
    print(json.dumps(artifact, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(artifact, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
