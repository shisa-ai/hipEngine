#!/usr/bin/env python3
"""C-width server decode baseline for the IKV-C2 campaign (roadmap P5).

Measures the CURRENT C>1 state on the repaired INT8 route before any
row-batched consumer work: N concurrent completions with the same fixed
token-ID fixture through the real server (packed batch decode, per-width
pool lease included - the capacity>1 gate keeps it by design), reporting
aggregate decode tok/s, per-request latency, and the serial-aggregate
reference (N x C1 rate) for the honest comparison the roadmap requires.

Protocol: one server launch per width; the width-N requests are fired
concurrently after readiness; walls are measured from first-request send to
last-response completion. Prefill and decode are separated by a second
isolated C1 run with max_tokens=1 per width (prefill wall), so the decode
rate is (tokens - prompt) / (wall - prefill share) per request.

Output: JSON artifact with per-width rows and the serial-aggregate ratios.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    ap.add_argument("--widths", default="1,2,4")
    ap.add_argument("--prompt-rows", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-context-tokens", type=int, default=16384)
    ap.add_argument("--vocab-span", type=int, default=32000)
    ap.add_argument("--port", type=int, default=18271)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np

    rng = np.random.default_rng(20260909)
    # Distinct-but-deterministic prompts per lane so lanes do not share a
    # prefix accidentally.
    prompts = [
        [int(t) for t in rng.integers(1000, args.vocab_span, size=args.prompt_rows)]
        for _ in range(max(int(w) for w in str(args.widths).split(",")))
    ]
    model_id = Path(args.model).name

    def launch_server(width: int) -> subprocess.Popen:
        env = os.environ.copy()
        env.setdefault("HIP_VISIBLE_DEVICES", "0")
        env.setdefault("GPU_MAX_HW_QUEUES", "1")
        env.setdefault("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG", "1")
        env.setdefault("HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS", "none")
        cmd = [
            sys.executable, "-m", "hipengine.server",
            "--model", str(args.model),
            "--backend", "hip_gfx1100",
            "--quant", "gguf_q4_k_m",
            "--kv-storage", "int8_per_token_head",
            "--kv-scale-dtype", "fp32",
            "--kv-scale-granularity", "per_token_head",
            "--max-context-tokens", str(int(args.max_context_tokens)),
            "--max-active-requests", str(int(width)),
            "--speculative-mtp-serving", "off",
            "--prefix-cache", "off",
            "--metrics", "prometheus",
            "--port", str(int(args.port)),
        ]
        return subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def wait_ready(srv: subprocess.Popen, timeout: float = 600.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if srv.poll() is not None:
                return False
            time.sleep(1.0)
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/health", timeout=2
                ).read()
                return True
            except Exception:
                continue
        return False

    def one_request(prompt: list[int], max_tokens: int) -> tuple[float, dict]:
        body = json.dumps({
            "model": model_id,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=900) as r:
            payload = json.loads(r.read())
        return time.perf_counter() - t0, payload

    out: dict[str, object] = {
        "kind": "gguf_server_cwidth_decode_baseline",
        "protocol_tier": 1,
        "model": str(args.model),
        "kv": "int8_per_token_head + fp32 scales",
        "route": "repaired C1 route (slot-local layer-outer prefill, packed batch decode above 1 slot, per-width pool lease)",
        "prompt_rows": int(args.prompt_rows),
        "max_tokens": int(args.max_tokens),
        "prompt_kind": "deterministic_varied_rng20260909_lanes",
        "widths": {},
    }

    widths = [int(w) for w in str(args.widths).split(",") if w.strip()]
    for width in widths:
        srv = launch_server(width)
        try:
            if not wait_ready(srv):
                out["widths"][str(width)] = {"status": "server_failed"}
                continue
            results: list[tuple[float, dict]] = []
            errors: list[str] = []
            threads: list[threading.Thread] = []

            def worker(i: int) -> None:
                try:
                    results.append(one_request(prompts[i], args.max_tokens))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"lane {i}: {exc}")

            t_start = time.perf_counter()
            for i in range(width):
                t = threading.Thread(target=worker, args=(i,))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            wall = time.perf_counter() - t_start
            if errors or len(results) != width:
                out["widths"][str(width)] = {
                    "status": "request_failed",
                    "errors": errors[:4],
                }
                continue
            # Prefill isolation: one lane alone with max_tokens=1 measures the
            # single-lane prefill wall under the same server.
            prefill_wall, _ = one_request(prompts[0], 1)
            decode_tokens = int(args.max_tokens) - 1
            # Aggregate decode rate: total decode tokens over the concurrent
            # wall minus each lane's prefill share (prefills serialize at
            # C<=4 with the 1024-row chunk cap under protect_decode).
            decode_wall = max(wall - prefill_wall, 1e-6)
            aggregate = width * decode_tokens / decode_wall
            per_request_latency_ms = [
                1000.0 * w for (w, _) in results
            ]
            out["widths"][str(width)] = {
                "status": "pass",
                "concurrent_wall_s": round(wall, 3),
                "single_lane_prefill_wall_s": round(prefill_wall, 3),
                "aggregate_decode_tok_s": round(aggregate, 3),
                "per_request_wall_s": [round(w, 3) for (w, _) in results],
                "per_request_latency_ms": [round(x, 2) for x in per_request_latency_ms],
                "teardown": "pending",
            }
            print(
                f"[C{width}] wall {wall:.2f} s, aggregate decode "
                f"{aggregate:.2f} tok/s, per-request "
                f"{[round(w,2) for w,_ in results]} s",
                flush=True,
            )
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=30)
                if str(width) in out["widths"]:
                    out["widths"][str(width)]["teardown"] = "clean"
            except Exception:
                srv.kill()

    c1 = out["widths"].get("1", {})
    if c1.get("status") == "pass":
        for width in widths:
            row = out["widths"].get(str(width), {})
            if row.get("status") == "pass":
                row["serial_aggregate_reference_tok_s"] = round(
                    width * c1["aggregate_decode_tok_s"], 3
                )
                row["efficiency_vs_serial"] = round(
                    row["aggregate_decode_tok_s"]
                    / (width * c1["aggregate_decode_tok_s"]),
                    4,
                )

    payload = json.dumps(out, indent=2, default=str)
    print(payload)
    if args.json:
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
