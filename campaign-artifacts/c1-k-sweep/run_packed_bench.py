"""Run the canonical server bench against the fail-closed packed-C1 harness.

Wraps scripts/gguf_mtp_c1c8_server_bench.py without modifying it:
1. Injects the diagnostic one-request evidence row (packed_c1_target=True,
   realized_group_rows=1, explicit-only) so the serving layer admits C1
   requests into the repaired packed target instead of the legacy singleton
   verifier. This is the campaign's designated explicitly-unqualified test
   candidate path (Packet 0), not a public evidence registration.
2. Forbids the legacy singleton target verifier for the whole process: if any
   C1 cycle routed through it, the run crashes with AssertionError instead of
   silently measuring the wrong route.
3. Counts packed-target frontier calls for the route evidence.

Everything after the injection is the unmodified canonical bench protocol.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.qwen38_packet5_k4_watchdog_probe import _inject_k4_evidence_row
from hipengine.generation import qwen35_gguf_mtp2 as mtp2


def _argv_value(name: str, default: str) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def main() -> None:
    width = int(_argv_value("--widths", "1"))
    budget = int(_argv_value("--candidate-budget", "3"))
    capacity = int(_argv_value("--resident-capacity", "8"))
    if width != 1:
        raise SystemExit("packed-C1 harness only wraps width-1 runs")
    injected_key = _inject_k4_evidence_row(width, budget, capacity=capacity)
    print(f"[packed-harness] injected evidence row: {injected_key}", flush=True)

    calls: list[tuple[int, ...]] = []
    original_packed = mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch

    def packed(self, plan, *positional, **kwargs):
        calls.append(tuple(plan.speculative_request_ids))
        return original_packed(self, plan, *positional, **kwargs)

    def forbidden(*positional, **kwargs):
        raise AssertionError(
            "packed-C1 harness invoked the legacy singleton target verifier"
        )

    mtp2.Qwen35GGUFMTP2Adapter._execute_target_frontier_batch = packed
    mtp2.Qwen35GGUFTransactionalVerifier = forbidden

    import atexit

    atexit.register(
        lambda: print(
            f"[packed-harness] packed_target_calls={len(calls)}", flush=True
        )
    )
    sys.argv = [str(Path(__file__).name), *sys.argv[1:]]
    runpy.run_path(
        str(ROOT / "scripts/gguf_mtp_c1c8_server_bench.py"), run_name="__main__"
    )


if __name__ == "__main__":
    main()
