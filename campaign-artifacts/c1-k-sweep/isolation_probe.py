#!/usr/bin/env python3
"""Mixed-cohort isolation probe: explicit-MTP requests among AR requests.

One capacity-8 product server, one model load. Scenarios (each submits its
requests concurrently from threads, all on the same canonical suite prompt
set, D24 greedy, 20 ms batch window):

- control8:  8 explicit-AR requests (speculative_mtp=false)
- mixed1x7:  1 explicit-MTP + 7 explicit-AR
- mixed2x6:  2 explicit-MTP + 6 explicit-AR

Gates per scenario: every request returns 200 with 24 tokens; in the mixed
scenarios every MTP-flagged request reports an engaged speculative_mtp
summary and every AR request reports used=false; AR outputs are token-exact
against the control cohort's same-prompt outputs. The AR cohort's total wall
is reported against the control for the slowdown check (retained C2/K3
economics bounds the expected envelope).

Diagnostics only, not a retained benchmark protocol (single server instance,
untracked-source allowed).
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


def _submit(client: TestClient, prompt: str, mtp: bool | None) -> dict:
    payload = {
        "model": "isolation-probe",
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    if mtp is not None:
        payload["speculative_mtp"] = mtp
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


def _run_scenario(client: TestClient, cohort: list[bool | None]) -> dict:
    barrier = threading.Barrier(len(cohort))
    original = _submit.__code__  # noqa: F841 - barrier handled via threads below

    def task(mtp_value: bool | None, prompt: str) -> dict:
        barrier.wait(timeout=30)
        return _submit(client, prompt, mtp_value)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cohort)) as pool:
        futures = [
            pool.submit(task, mtp, PROMPTS[i % len(PROMPTS)])
            for i, mtp in enumerate(cohort)
        ]
        rows = [f.result() for f in futures]
    return {"rows": rows, "total_wall": max(r["wall"] for r in rows)}


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
            served_model_name="isolation-probe",
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
    results: dict[str, dict] = {}
    with TestClient(app) as client:
        # Warmup (mirrors the retained protocol: AR then MTP, discarded).
        for mtp in (False, True):
            r = client.post("/v1/completions", json={
                "model": "isolation-probe", "prompt": PROMPTS[0],
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
                "speculative_mtp": mtp})
            r.raise_for_status()

        control = _run_scenario(client, [False] * 8)
        results["control8"] = control
        print(f"[isolation] control8 total_wall={control['total_wall']:.2f}s",
              flush=True)

        mixed1 = _run_scenario(client, [True] + [False] * 7)
        results["mixed1x7"] = mixed1
        print(f"[isolation] mixed1x7 total_wall={mixed1['total_wall']:.2f}s",
              flush=True)

        # Capacity-2 mixed cohorts (the qualified C2 configuration; the C2
        # evidence row requires resident_capacity=2, so a capacity-8 2-row
        # group correctly refuses to AR).
        control2 = _run_scenario(client, [False, False])
        results["control2"] = control2
        print(f"[isolation] control2 total_wall={control2['total_wall']:.2f}s",
              flush=True)

        mixed21 = _run_scenario(client, [True, False])
        results["mixed1x1_c2"] = mixed21
        print(f"[isolation] mixed1x1_c2 total_wall={mixed21['total_wall']:.2f}s",
              flush=True)

        mixed22 = _run_scenario(client, [True, True])
        results["mixed2x0_c2"] = mixed22
        print(f"[isolation] mixed2x0_c2 total_wall={mixed22['total_wall']:.2f}s",
              flush=True)

    checks: list[str] = []
    control_ids = [tuple(r["ids"]) for r in results["control8"]["rows"]]
    control2_ids = [tuple(r["ids"]) for r in results["control2"]["rows"]]

    # Gate 1: every MTP-flagged request engaged, every AR request stayed AR.
    cohort_specs = {
        "mixed1x7": ([True] + [False] * 7, control_ids),
        "mixed1x1_c2": ([True, False], control2_ids),
        "mixed2x0_c2": ([True, True], control2_ids),
    }
    for name, (cohort, reference_ids) in cohort_specs.items():
        scenario = results[name]
        mtp_flags = [r["mtp_used"] for r in scenario["rows"]]
        expected_mtp = sum(1 for m in cohort if m)
        ok = all(f == (m is True) for f, m in zip(mtp_flags, cohort))
        checks.append(
            f"{name}: route_flags_match={ok} "
            f"mtp_engaged={sum(mtp_flags)}/{expected_mtp}"
        )
    # Gate 2: AR outputs token-exact vs the matching control at the same
    # prompt index.
    for name, (cohort, reference_ids) in cohort_specs.items():
        scenario = results[name]
        mismatches = []
        for i, (m, row) in enumerate(zip(cohort, scenario["rows"])):
            if m is False and tuple(row["ids"]) != reference_ids[i]:
                mismatches.append(i)
        checks.append(
            f"{name}: ar_token_exact={'true' if not mismatches else mismatches}"
        )
    # Gate 3: slowdown envelope (total wall vs the matching control).
    for name, (cohort, reference_ids) in cohort_specs.items():
        ref_name = "control2" if "_c2" in name else "control8"
        ratio = results[name]["total_wall"] / results[ref_name]["total_wall"]
        checks.append(f"{name}: total_wall_ratio={ratio:.3f} (vs {ref_name})")

    out = {
        "kind": "mixed_cohort_isolation_probe",
        "date": "2026-09-07",
        "status": "diagnostic_not_a_benchmark",
        "scenarios": {
            name: {
                "total_wall": data["total_wall"],
                "rows": [
                    {"wall": r["wall"], "mtp_used": r["mtp_used"],
                     "mtp_cycles": r["mtp_cycles"], "ids": list(r["ids"])}
                    for r in data["rows"]
                ],
            }
            for name, data in results.items()
        },
        "checks": checks,
    }
    Path("/tmp/he-bettermtp-raw/isolation-probe.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    for c in checks:
        print("[isolation]", c, flush=True)
    print(json.dumps(out["scenarios"]["mixed1x7"]["rows"][0]["ids"][:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
