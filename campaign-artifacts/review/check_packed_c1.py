"""Check packed C1 routing and token equality; never emit a performance claim.

Runs the full category/heldout suite through the service owner. Injects a
bounded diagnostic C1 row, forbids the legacy target verifier, and checks that
each response engaged MTP and reached the packed target. This is not the
production numerical/lifecycle or repeated-performance gate.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import gguf_mtp_c1c8_server_bench as bench
from scripts.qwen38_packet5_k4_watchdog_probe import _inject_k4_evidence_row
from hipengine.generation import qwen35_gguf_mtp2 as mtp2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--budget", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS"] = "1"
    _inject_k4_evidence_row(1, args.budget)
    faulthandler.dump_traceback_later(300, exit=True)
    calls: list[tuple[int, ...]] = []
    original = mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch

    def packed(self, plan, *positional, **kwargs):
        calls.append(tuple(plan.speculative_request_ids))
        return original(self, plan, *positional, **kwargs)

    def forbidden(*positional, **kwargs):
        raise AssertionError("packed C1 invoked the legacy target verifier")

    mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch = packed
    mtp2.Qwen35GGUFTransactionalVerifier = forbidden
    llm = bench.LLM(
        args.model, backend="hip_gfx1100", execution_profile="production",
        max_active_requests=8, max_sequence_length=1024,
        speculative_candidate_budget=args.budget,
    )
    rows = []
    try:
        llm.prepare(max_sequence_length=1024)
        app = bench.create_app(bench.ServerConfig(
            model=args.model, backend="hip_gfx1100", quant="gguf_q4_k_m",
            served_model_name="review", eager_load=False,
            generation_batch_window_ms=20, max_context_tokens=1024,
            max_active_requests=8, speculative_mtp_serving="opt_in",
            speculative_candidate_budget=args.budget, shutdown_grace_seconds=5.0,
        ), llm=llm)
        prompts = bench.load_prompt_suite(ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl")
        with bench.TestClient(app) as client:
            for prompt in prompts:
                before = len(calls)
                arms = {arm: bench._run_arm(
                    client, llm=llm, model="review", prompt=prompt["rendered_prompt"],
                    width=1, max_tokens=24, arm=arm, mtp_request_mode="explicit",
                ) for arm in bench.ARMS}
                exact = ([r["generated_ids"] for r in arms["ar"]["rows"]]
                         == [r["generated_ids"] for r in arms["mtp"]["rows"]])
                engaged = all(bench._mtp_engaged(r["route"], r["mtp"])
                              for r in arms["mtp"]["rows"])
                budget_ok = all(bench._mtp_budget_conformed(r["mtp"], budget=args.budget)
                                for r in arms["mtp"]["rows"])
                row = dict(prompt_id=prompt["id"], exact=exact, engaged=engaged,
                           budget_conformed=budget_ok, packed_calls=len(calls) - before)
                rows.append(row)
                print(json.dumps(row), flush=True)
                assert exact and engaged and budget_ok and len(calls) > before
                assert all(len(ids) == 1 for ids in calls[before:])
    finally:
        llm.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "diagnostic_only": True, "performance_claim": False,
            "budget": args.budget, "resident_capacity": 8, "cells": rows,
            "passed": len(rows) == 10 and all(
                r["exact"] and r["engaged"] and r["budget_conformed"] and r["packed_calls"] > 0
                for r in rows
            ),
        }, indent=2) + "\n")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
