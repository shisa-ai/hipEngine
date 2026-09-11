#!/usr/bin/env python3
"""Capture and attribute the newer halo-box HIP prefill path (#22 queue 1).

Runs the halo-box `5f851647f` HIP llama-server under ``rocprofv3`` with the
identical model/server arguments used by the E0 qualification, then drives
ONE prefill-only request per canonical case (``n_predict=1`` so the
measured window is essentially the prompt processing) and attributes the
resulting kernel trace by kernel-name role buckets.

This is the attribution capture the ms-to-parity ledger calls for: which
owners (dense MMQ, expert gate/up, down, packing/quantize, attention,
host-side gaps) carry the newer engine's 709 tok/s p4096 advantage.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVER = Path(
    "/home/lhl/halo-box-strix-llama-src-5f85164/build-hip-release-5f85164/bin/llama-server")
MODEL = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")

ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("moe_mul_mat", r"mul_mat.*(q4_k|q5_1|q6_k|q8_0|iq4)|q4_k.*mul|q5_1.*mul|exp"),
    ("dense_mul_mat", r"ggml_gemm|mul_mat|rope_"),
    ("quantize_pack", r"quantize_row|quantize_|pack|im2col"),
    ("attention_qsa", r"attn|flash|fattn|softmax|ssm|conv|mamba|gdn"),
    ("elementwise", r"add|mul_|silu|gelu|norm|cpy|scale|sqr|sqrt|div|sub|clamp|tanh|sigmoid|exp_|log_"),
    ("argmax_sample", r"argmax|sample|greedy|top_"),
    ("host_gap", r"__hip_api::|api"),
)


def _role_for(name: str) -> str:
    lname = name.lower()
    for role, pattern in ROLE_PATTERNS:
        if re.search(pattern, lname):
            return role
    return "other"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-id", action="append", default=None,
                   help="fixture case ids to run (default: code-p4096 + one per shape)")
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--port", type=int, default=18173)
    p.add_argument("--trace-root", type=Path, default=Path("/tmp/halobox5f-prefill-attribution"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not SERVER.exists():
        p.error(f"missing server binary: {SERVER}")
    fixture = json.loads((ROOT / "benchmarks/fixtures/"
                          "qwen4exp_canonical_ar_p512_p1024_p4096.json").read_text())
    cases = [c for c in fixture["cases"]
             if args.case_id is None or c["id"] in args.case_id]

    trace_root = args.trace_root
    trace_root.mkdir(parents=True, exist_ok=True)
    server_args = [
        str(SERVER), "-m", str(MODEL),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--parallel", "1", "--no-webui", "-ngl", "999", "-fa", "on",
        "-ctk", "bf16", "-ctv", "bf16", "-c", "4352", "-b", "8192",
        "-ub", "2048", "-t", "4",
    ]
    server_log = trace_root / "server.log"
    with server_log.open("wb") as log_handle:
        process = subprocess.Popen(
            ["rocprofv3", "--kernel-trace", "--output-format", "csv",
             "-d", str(trace_root / "trace"), "--", *server_args],
            stdout=log_handle, stderr=subprocess.STDOUT,
        )
    try:
        base = f"http://127.0.0.1:{args.port}"
        for _ in range(240):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1.0)
        else:
            raise RuntimeError("server did not become healthy")
        # one untraced-warmup-free prefill-only request per case
        request_marks = []
        for case in cases:
            for rep in range(args.repetitions):
                payload = json.dumps({
                    "prompt": [int(t) for t in case["prompt_token_ids"]],
                    "n_predict": 1, "temperature": 0.0, "top_k": 1,
                    "top_p": 1.0, "min_p": 0.0, "seed": 12345,
                    "ignore_eos": True, "cache_prompt": False,
                    "stream": False, "return_tokens": True,
                }).encode()
                req = urllib.request.Request(
                    base + "/completion", data=payload,
                    headers={"Content-Type": "application/json"})
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=600) as r:
                    body = json.loads(r.read())
                ms = body.get("timings", {}).get("prompt_ms")
                request_marks.append({
                    "id": case["id"], "rep": rep,
                    "prompt_tokens": case["prompt_tokens"],
                    "prompt_ms": ms, "client_wall_ms": (time.perf_counter() - t0) * 1e3,
                })
                print(f"{case['id']} rep{rep}: prompt_ms={ms}", flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    kernel_csvs = sorted((trace_root / "trace").rglob("*kernel_trace*.csv"))
    if not kernel_csvs:
        raise RuntimeError("no kernel trace csv produced")
    roles: dict[str, float] = defaultdict(float)
    kernels: dict[str, float] = defaultdict(float)
    total_ns = 0
    with kernel_csvs[-1].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Name") or row.get("Kernel_Name") or ""
            dur = row.get("Duration") or row.get("Duration_ns")
            if dur is None and name and row.get("Start_Timestamp") and row.get("End_Timestamp"):
                dur = int(float(row["End_Timestamp"])) - int(float(row["Start_Timestamp"]))
            if not name or not dur:
                continue
            ns = int(float(dur))
            roles[_role_for(name)] += ns / 1e6
            kernels[name] += ns / 1e6
            total_ns += ns
    top = sorted(kernels.items(), key=lambda kv: -kv[1])[:40]
    report = {
        "schema": 1,
        "kind": "halobox5f_hip_prefill_attribution",
        "server": str(SERVER),
        "server_args": server_args,
        "cases": request_marks,
        "role_ms": {k: round(v, 3) for k, v in sorted(roles.items(), key=lambda kv: -kv[1])},
        "device_total_ms": round(total_ns / 1e6, 3),
        "top_kernels_ms": {k: round(v, 3) for k, v in top},
        "note": ("prefill-only attribution: n_predict=1 so decode work is "
                 "excluded; roles bucketed from kernel names; compare "
                 "prompt_ms against the E0 measured prefill rate"),
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["role_ms"], indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
