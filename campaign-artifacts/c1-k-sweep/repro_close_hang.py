"""Reproduce the engine-close hang after a serving-admitted, policy-refused C1 request.

Scenario (from the r1-k1 sweep failure): the injected one-request K1 evidence
row admits the request at the serving layer, the backend physical policy
refuses the (C1, K1) cell, every cycle falls back to AR (outputs stay exact),
and llm.close() then times out waiting for the engine service shutdown
command. This script captures the stuck thread stacks at the timeout instead
of dying silently.
"""
from __future__ import annotations

import faulthandler
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import gguf_mtp_c1c8_server_bench as bench
from scripts.qwen38_packet5_k4_watchdog_probe import _inject_k4_evidence_row


def main() -> int:
    faulthandler.dump_traceback_later(600, exit=True)
    injected_key = _inject_k4_evidence_row(1, 1, capacity=8)
    print(f"[repro] injected evidence row: {injected_key}", flush=True)
    MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
    prompts = bench.load_prompt_suite(
        ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
    )
    llm = bench.LLM(
        MODEL, backend="hip_gfx1100", execution_profile="production",
        max_active_requests=8, max_sequence_length=1024,
        speculative_candidate_budget=1,
    )
    try:
        llm.prepare(max_sequence_length=1024)
        app = bench.create_app(bench.ServerConfig(
            model=MODEL, backend="hip_gfx1100", quant="gguf_q4_k_m",
            served_model_name="repro", eager_load=False,
            generation_batch_window_ms=20, max_context_tokens=1024,
            max_active_requests=8, speculative_mtp_serving="opt_in",
            speculative_candidate_budget=1, shutdown_grace_seconds=5.0,
        ), llm=llm)
        with bench.TestClient(app) as client:
            result = bench._run_arm(
                client, llm=llm, model="repro",
                prompt=prompts[0]["rendered_prompt"],
                width=1, max_tokens=24, arm="mtp", mtp_request_mode="explicit",
            )
            row = result["rows"][0]
            print(f"[repro] used={row['mtp'].get('used')} "
                  f"exact={row['correctness']['passed']}", flush=True)
        print("[repro] request context exited; closing engine...", flush=True)
    except BaseException as exc:  # noqa: BLE001 - diagnostic path
        print(f"[repro] exception before close: {exc!r}", flush=True)
    try:
        llm.close()
        print("[repro] close completed cleanly", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[repro] close raised: {exc!r}", flush=True)
        print("[repro] current thread stacks at close failure:", flush=True)
        for tid, frame in sys._current_frames().items():
            name = "main"
            for t in threading.enumerate():
                if t.ident == tid:
                    name = t.name
                    break
            print(f"--- thread {name} (ident={tid})", flush=True)
            traceback.print_stack(frame, file=sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
