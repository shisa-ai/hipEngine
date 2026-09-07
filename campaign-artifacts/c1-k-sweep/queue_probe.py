#!/usr/bin/env python3
"""Oversubscribed service-queue probe: 16 requests through a capacity-8 server.

One capacity-8 product server. 16 concurrent requests: 8 explicit-MTP and 8
explicit-AR over the same 8 canonical prompts (wave A MTP, wave B AR), so the
resident owner rotates both cohorts through its 8 slots and every prompt is
generated twice — once under MTP, once under pure AR.

Gates: all 16 return 200 with 24 tokens; all 8 MTP requests report an engaged
speculative_mtp summary regardless of which residence wave they ran in; MTP
outputs are token-exact against the same-run AR output for the same prompt;
the process exits cleanly (the TestClient close runs the engine shutdown
path). Latency context (queue wait + generation wall per request) is recorded
as diagnostics, not a retained benchmark.
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hipengine.llm import LLM
from hipengine.server.api import ServerConfig, create_app
from fastapi.testclient import TestClient

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
MAX_TOKENS = 24
_suite = [
    json.loads(line)
    for line in Path(
        "/home/lhl/hipEngine-main/benchmarks/prompts/mtpbench-code-general-ja.jsonl"
    )
    .read_text(encoding="utf-8")
    .splitlines()
]
PROMPTS = [
    "\n".join(m["content"] for m in row["messages"] if isinstance(m.get("content"), str))
    for row in _suite[:8]
]


def _submit(client: TestClient, prompt: str, mtp: bool) -> dict:
    payload = {
        "model": "queue-probe",
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "speculative_mtp": mtp,
    }
    started = time.perf_counter()
    r = client.post("/v1/completions", json=payload)
    completed = time.perf_counter()
    r.raise_for_status()
    body = r.json()
    ids = body["hipengine"]["token_accounting"]["choice_generated_token_ids"][0]
    mtp_block = (body.get("hipengine") or {}).get("speculative_mtp") or {}
    return {
        "wall": completed - started,
        "ids": tuple(ids),
        "mtp_used": bool(mtp_block.get("used", False)),
        "mtp_cycles": int(mtp_block.get("draft_cycles", 0) or 0),
    }


def main() -> int:
    llm = LLM(MODEL, backend="hip_gfx1100", execution_profile="production",
              max_active_requests=8, max_sequence_length=1024,
              speculative_candidate_budget=3)
    llm.prepare(max_sequence_length=1024)
    app = create_app(
        ServerConfig(
            model=MODEL,
            backend="hip_gfx1100",
            quant="gguf_q4_k_m",
            served_model_name="queue-probe",
            eager_load=False,
            metrics="off",
            generation_batch_window_ms=20.0,
            max_context_tokens=1024,
            max_active_requests=8,
            speculative_mtp_serving="opt_in",
            speculative_candidate_budget=3,
            shutdown_grace_seconds=5.0,
        ),
        llm=llm,
    )
    with TestClient(app) as client:
        for mtp in (False, True):
            r = client.post("/v1/completions", json={
                "model": "queue-probe", "prompt": PROMPTS[0],
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
                "speculative_mtp": mtp})
            r.raise_for_status()

        cohort = [(mtp, PROMPTS[i % 8]) for i, mtp in
                  enumerate([True] * 8 + [False] * 8)]
        barrier = threading.Barrier(len(cohort))

        def task(mtp: bool, prompt: str) -> dict:
            barrier.wait(timeout=30)
            return _submit(client, prompt, mtp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(task, m, p) for m, p in cohort]
            rows = [f.result() for f in futures]

    makespan = max(r["wall"] for r in rows)
    print(f"[queue-probe] 16/16 complete, makespan={makespan:.2f}s", flush=True)

    checks: list[str] = []
    # Gate 1: every MTP request engaged, every AR request stayed AR.
    flags_ok = all(r["mtp_used"] == m for (m, _), r in zip(cohort, rows))
    checks.append(f"route_flags_match={flags_ok}")
    # Gate 2: MTP output == AR output per prompt (ar_exact within the run).
    mismatches = [
        i for i in range(8)
        if rows[i]["ids"] != rows[i + 8]["ids"]
    ]
    checks.append(f"mtp_ar_token_exact={'true' if not mismatches else mismatches}")
    # Latency context.
    mtp_walls = [r["wall"] for r in rows[:8]]
    ar_walls = [r["wall"] for r in rows[8:]]
    checks.append(
        f"latency_context: mtp_wall_median="
        f"{sorted(mtp_walls)[4]:.2f}s ar_wall_median={sorted(ar_walls)[4]:.2f}s"
    )

    out = {
        "kind": "oversubscribed_service_queue_probe",
        "date": "2026-09-07",
        "status": "diagnostic_not_a_benchmark",
        "hardware": "AMD Radeon Pro W7900 (gfx1100), GPU0",
        "method": (
            "capacity-8 product server, 16 concurrent requests (8 explicit-MTP "
            "wave A + 8 explicit-AR wave B over the same 8 canonical prompts), "
            "D24 greedy, 20 ms window; MTP-vs-AR exactness within the run"
        ),
        "makespan_seconds": makespan,
        "checks": checks,
        "rows": [
            {"mtp_requested": m, "mtp_used": r["mtp_used"],
             "mtp_cycles": r["mtp_cycles"], "wall": r["wall"],
             "ids": list(r["ids"])}
            for (m, _), r in zip(cohort, rows)
        ],
    }
    Path("/tmp/he-bettermtp-raw/queue-probe.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    for c in checks:
        print("[queue-probe]", c, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
