#!/usr/bin/env python3
"""Stage-timing comparison: legacy singleton vs packed one-row C1 route.

Loads the model once, serves two identical explicit-MTP K3 requests through
the product HTTP route (one before the packed-evidence injection, one after),
and prints the per-request adapter stage breakdown from the resident runner's
internal rows: proposal / target / provider-update / accept / commit /
readback milliseconds plus batch-call counts.

Legacy leg: the product default evidence (packed_c1_target=false) engages the
legacy singleton target verifier. Packed leg: the injected one-request
`packed_c1_target` evidence row engages the packed one-row frontier (physical
policy admits (1,3)); the legacy verifier is replaced by a guard so any
legacy engagement fails loudly instead of silently mixing routes.

Diagnostics only, not a benchmark protocol: single prompt, single request,
untracked-source allowed (this is a campaign-artifacts diagnostic, not a
retained measurement).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.qwen38_packet5_k4_watchdog_probe import _inject_k4_evidence_row
from hipengine.generation import qwen35_gguf_mtp2 as mtp2
from hipengine.llm import LLM
from hipengine.server.api import ServerConfig, create_app
from fastapi.testclient import TestClient

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
_suite = [
    json.loads(line)
    for line in Path(
        "/home/lhl/hipEngine-main/benchmarks/prompts/mtpbench-code-general-ja.jsonl"
    )
    .read_text(encoding="utf-8")
    .splitlines()
]
PROMPT = "\n".join(
    m["content"] for m in _suite[0]["messages"] if isinstance(m.get("content"), str)
)
MAX_TOKENS = 24
STAGE_FIELDS = (
    "mtp2_cycles",
    "mtp2_proposal_ms",
    "mtp2_target_ms",
    "mtp2_provider_update_ms",
    "mtp2_accept_ms",
    "mtp2_selected_commit_ms",
    "mtp2_candidate_readback_ms",
    "mtp2_target_readback_ms",
    "mtp2_accept_upload_ms",
    "mtp2_accept_tail_ms",
    "mtp2_accept_enqueue_ms",
    "mtp2_target_batch_calls",
    "mtp2_proposal_batch_calls",
    "mtp2_selected_commit_batch_calls",
    "mtp2_candidate_device_handoffs",
    "mtp2_candidate_d2h_after_target",
)


def _stash_rows(llm: LLM, stash: list) -> None:
    """Keep completed per-request rows: the runner reclaims them on completion.

    Wraps Qwen35GGUFResidentModelRunner.reclaim at class level so the row is
    captured after its final state is written but before the dict pop.
    """
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    if getattr(Qwen35GGUFResidentModelRunner.reclaim, "_stash_wrapped", False):
        return
    original = Qwen35GGUFResidentModelRunner.reclaim

    def wrapped(self, completed):
        row = self._rows.get(int(completed.request_id)) if hasattr(self, "_rows") else None
        if row is not None:
            stash.append(row)
        return original(self, completed)

    wrapped._stash_wrapped = True
    Qwen35GGUFResidentModelRunner.reclaim = wrapped


def _last_row_timing(stash: list) -> dict:
    row = stash[-1]
    out = {k: getattr(row, k) for k in STAGE_FIELDS}
    out["mtp2_execution_routes"] = list(row.mtp2_execution_routes)[-4:]
    for diag in ("mtp2_candidate_budget", "mtp2_prompt_fallback_reason",
                 "prefix_fallback_reason", "prefix_admission_fallback",
                 "mtp2_prompt_streaming", "mtp2_prompt_prime_rows"):
        out[f"diag_{diag}"] = getattr(row, diag, None)
    return out


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
            served_model_name="stage-compare",
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
    stash: list = []
    _stash_rows(llm, stash)
    plan_log: list[dict] = []
    original_resolve = llm.resolve_speculative_mtp_serving_plan

    def logged_resolve(**kwargs):
        decision = original_resolve(**kwargs)
        if isinstance(decision, dict):
            get = decision.get
        else:
            get = lambda name, default=None: getattr(decision, name, default)
        plan_log.append({
            "kwargs": {k: str(v)[:60] for k, v in kwargs.items()},
            "admitted": get("admitted"),
            "route": get("selected_route"),
            "reason": get("reason"),
            "evidence_key": get("evidence_key"),
            "candidate_count": get("selected_candidate_count"),
        })
        return decision

    llm.resolve_speculative_mtp_serving_plan = logged_resolve
    with TestClient(app) as client:
        # Warmup arms (mirror the bench: AR then MTP on prompts[0], discarded)
        # before the measured legs; the first MTP request primes the serving
        # path exactly as the retained protocol does.
        for arm, mtp_value in (("ar", False), ("mtp", True)):
            r = client.post(
                "/v1/completions",
                json={
                    "model": "stage-compare",
                    "prompt": PROMPT,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "speculative_mtp": mtp_value,
                },
            )
            r.raise_for_status()
        print(f"[stage-compare] warmup done, stash={len(stash)}", flush=True)
        # Legacy leg: product default evidence row (packed_c1_target=false).
        r = client.post(
            "/v1/completions",
            json={
                "model": "stage-compare",
                "prompt": PROMPT,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "speculative_mtp": True,
            },
        )
        r.raise_for_status()
        payload = r.json()
        print("[stage-compare] legacy response keys:", sorted(payload.keys()), flush=True)
        print("[stage-compare] legacy mtp block:",
              json.dumps((payload.get("hipengine") or {}).get("speculative_mtp"))[:500],
              flush=True)
        print(f"[stage-compare] stash size after legacy: {len(stash)}", flush=True)
        results["legacy"] = _last_row_timing(stash)
        print("[stage-compare] legacy leg done", flush=True)

        # Packed leg: inject the one-request packed_c1_target evidence row and
        # forbid the legacy singleton verifier.
        key = _inject_k4_evidence_row(1, 3, capacity=8)
        print(f"[stage-compare] injected: {key}", flush=True)

        calls: list[tuple[int, ...]] = []
        original_packed = mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch

        def packed(self, plan, *positional, **kwargs):
            calls.append(tuple(plan.speculative_request_ids))
            return original_packed(self, plan, *positional, **kwargs)

        def forbidden(*positional, **kwargs):
            raise AssertionError("legacy singleton verifier invoked on packed leg")

        mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch = packed
        mtp2.Qwen35GGUFTransactionalVerifier = forbidden
        try:
            r = client.post(
                "/v1/completions",
                json={
                    "model": "stage-compare",
                    "prompt": PROMPT,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "speculative_mtp": True,
                },
            )
            r.raise_for_status()
            payload = r.json()
            print("[stage-compare] packed mtp block:",
                  json.dumps((payload.get("hipengine") or {}).get("speculative_mtp"))[:500],
                  flush=True)
            print(f"[stage-compare] stash size after packed: {len(stash)}", flush=True)
        finally:
            mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch = original_packed
        results["packed"] = _last_row_timing(stash)
        results["packed"]["_frontier_calls"] = len(calls)
        print(f"[stage-compare] packed leg done, frontier_calls={len(calls)}",
              flush=True)

    for entry in plan_log:
        print("[stage-compare] plan:", json.dumps(entry)[:400], flush=True)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
