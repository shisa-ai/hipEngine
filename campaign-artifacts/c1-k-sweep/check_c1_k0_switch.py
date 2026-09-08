"""Prove native K0<->MTP switching through the product server route.

For every canonical prompt this runner drives four sequential single requests
through one resident product server (capacity 8, explicit opt-in): alternating
explicit-MTP and automatic-K0 legs, counterbalanced start per prompt index.
From outside the API it verifies that every switch direction keeps outputs
exact (greedy determinism: the MTP legs must match the AR legs, which proves
provider catch-up across each K0<->MTP transaction boundary), engages only
the MTP legs, stays inside the candidate budget, reaches the packed target,
never touches the legacy singleton verifier, and drains cleanly (process
exit 0, no shutdown timeout). No internal request-lifecycle hooks and no
reclaimed-row consultation: the prior switch probe failed on exactly those
instrumentation assumptions, not on runtime behavior.

Diagnostic evidence only: no performance claim, no public admission change.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import gguf_mtp_c1c8_server_bench as bench
from scripts.c1_k0_switch_probe import plan_legs, summarize_switches, validate_sequence
from scripts.qwen38_packet5_k4_watchdog_probe import _inject_k4_evidence_row
from hipengine.generation import qwen35_gguf_mtp2 as mtp2


def _single_request(client, *, llm, model: str, prompt: str, max_tokens: int, mtp: bool) -> dict:
    # Reuse the bench's proven request machinery (executor + barrier). A plain
    # main-thread TestClient.post left the MTP legs unregistered at the adapter
    # (request unregistered or disabled); the bench's submission path is the
    # validated product-route client behavior.
    result = bench._run_arm(
        client,
        llm=llm,
        model=model,
        prompt=prompt,
        width=1,
        max_tokens=max_tokens,
        arm="mtp",
        mtp_request_mode="explicit" if mtp else "automatic",
    )
    row = result["rows"][0]
    return {
        "leg": "mtp" if mtp else "k0",
        "wall_seconds": result["wall_seconds"],
        "generated_ids": row["generated_ids"],
        "route": row["route"],
        "mtp": row["mtp"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0,
                        help="debug: run only the first N prompts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    faulthandler.dump_traceback_later(900, exit=True)

    # Packed-route harness: inject the diagnostic one-request
    # packed_c1_target evidence row (explicit-only, unqualified test candidate
    # path per Packet 0) so the serving layer admits C1 into the repaired
    # packed target, forbid the legacy singleton verifier for the whole
    # process (any legacy use crashes instead of mislabeling), and count the
    # packed frontier calls. These wrap method entry, not request lifecycle
    # events; the rejected 2026-09-07 probe failed by consulting reclaimed
    # rows inside lifecycle hooks.
    injected_key = _inject_k4_evidence_row(1, args.budget, capacity=args.capacity)
    print(f"[packed-harness] injected evidence row: {injected_key}", flush=True)
    calls: list[tuple[int, ...]] = []
    original_packed = mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch

    def packed(self, plan, *positional, **kwargs):
        calls.append(tuple(plan.speculative_request_ids))
        return original_packed(self, plan, *positional, **kwargs)

    def forbidden(*positional, **kwargs):
        raise AssertionError(
            "C1 switch probe invoked the legacy singleton target verifier"
        )

    mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch = packed
    mtp2.Qwen35GGUFTransactionalVerifier = forbidden

    llm = bench.LLM(
        args.model,
        backend="hip_gfx1100",
        execution_profile="production",
        max_active_requests=args.capacity,
        max_sequence_length=1024,
        speculative_candidate_budget=args.budget,
    )
    failures: list[str] = []
    prompts = bench.load_prompt_suite(
        ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
    )
    if args.limit > 0:
        prompts = prompts[: args.limit]
    try:
        llm.prepare(max_sequence_length=1024)
        app = bench.create_app(bench.ServerConfig(
            model=args.model,
            backend="hip_gfx1100",
            quant="gguf_q4_k_m",
            served_model_name="c1-k0-switch",
            eager_load=False,
            generation_batch_window_ms=20,
            max_context_tokens=1024,
            max_active_requests=args.capacity,
            speculative_mtp_serving="opt_in",
            speculative_candidate_budget=args.budget,
            shutdown_grace_seconds=5.0,
        ), llm=llm)
        with bench.TestClient(app) as client:
            rows = []
            for index, prompt in enumerate(prompts):
                legs = plan_legs(index)
                before = len(calls)
                legs_rows = [
                    _single_request(
                        client,
                        llm=llm,
                        model="c1-k0-switch",
                        prompt=prompt["rendered_prompt"],
                        max_tokens=args.max_tokens,
                        mtp=(leg == "mtp"),
                    )
                    for leg in legs
                ]
                reasons = validate_sequence(legs, legs_rows, budget=args.budget)
                switches = summarize_switches(legs)
                row = {
                    "prompt_id": prompt["id"],
                    "legs": list(legs),
                    "switches": switches,
                    "packed_calls": len(calls) - before,
                    "exact": "divergent" not in " ".join(reasons)
                    and not any("missing_generated_ids" in r for r in reasons),
                    "reasons": reasons,
                    "leg_walls_s": [round(r["wall_seconds"], 4) for r in legs_rows],
                    "leg_draft": [
                        {
                            "used": bool(r["mtp"].get("used")),
                            "cycles": int(r["mtp"].get("draft_cycles", 0) or 0),
                            "tokens": int(r["mtp"].get("draft_tokens", 0) or 0),
                            "summary": r["mtp"],
                        }
                        for r in legs_rows
                    ],
                }
                rows.append(row)
                failures.extend(f"{prompt['id']}: {reason}" for reason in reasons)
                print(json.dumps(row), flush=True)
        if len(calls) == 0:
            failures.append("no packed target calls observed")
        total_switches = {
            key: sum(row["switches"][key] for row in rows)
            for key in ("mtp_to_k0", "k0_to_mtp")
        }
    finally:
        llm.close()

    passed = not failures
    payload = {
        "kind": "native_c1_k0_mtp_switch_proof",
        "diagnostic_only": True,
        "performance_claim": False,
        "passed": passed,
        "failure_reasons": failures,
        "source_commit": os.popen("git rev-parse HEAD").read().strip(),
        "model": args.model,
        "budget": args.budget,
        "capacity": args.capacity,
        "max_tokens": args.max_tokens,
        "prompts": len(prompts),
        "legs_per_prompt": len(plan_legs(0)),
        "packed_target_calls": len(calls),
        "total_switches": total_switches,
        "legacy_target_forbidden": True,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "passed": passed,
        "packed_target_calls": len(calls),
        "total_switches": total_switches,
        "failure_reasons": failures[:6],
    }), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
