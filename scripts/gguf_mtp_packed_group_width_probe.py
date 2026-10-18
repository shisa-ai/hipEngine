"""Measure the packed target verifier's real group width for a bench run.

Per-request cycle and draft counts, `plan_group_rows`, and `verifier_rows` all
describe intent rather than the call that runs.  `plan_group_rows` is the plan,
`verifier_rows` is a declared shape field, and identical per-request cycle
counts across widths are equally consistent with a fused group stepping in
lockstep and with singleton cycles sharing a wider batch.

This wrapper answers the question directly.  It patches
`Qwen35GGUFResidentSession.verify_target_blocks_batch` -- the one call that runs
the packed verifier, invoked with one job per row of the cycle group -- records
`len(jobs)` for every call, and prints the histogram at exit.  Calls recorded at
`jobs=1` in a run whose width is greater than one mean the cycles are not
grouped, whatever the plan says.

    python3 scripts/gguf_mtp_packed_group_width_probe.py \
        --model ~/models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --backend hip_gfx1151 --quant gguf_q4_k_m \
        --kv-storage int8_per_token_head --mtp-request-mode automatic \
        --widths 2,4 --resident-capacity 4 --expected-mtp-widths 2,4 \
        --candidate-budget 2 --max-tokens 24 \
        --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
        --output /tmp/packed_group_width.json

Every other argument is the benchmark's own; this adds the patch and nothing
else.  The benchmark still writes its artifact, so the grouped run and the
sweep it explains can be read side by side.
"""

from __future__ import annotations

import atexit
import collections
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _entry in (str(_REPO), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFResidentSession,
)

calls: "collections.Counter[int]" = collections.Counter()

_original = Qwen35GGUFResidentSession.verify_target_blocks_batch


def _counting_verify(self, jobs, **kwargs):  # type: ignore[no-untyped-def]
    calls[len(list(jobs))] += 1
    return _original(self, jobs, **kwargs)


Qwen35GGUFResidentSession.verify_target_blocks_batch = _counting_verify


@atexit.register
def _report() -> None:
    print("\n=== packed target verifier group width (jobs per call) ===")
    for width in sorted(calls):
        print(f"  jobs={width}: {calls[width]} calls")
    if not calls:
        print("  no calls recorded: the packed verifier was not reached")
    elif min(calls) == 1:
        print(
            "  WARNING: singleton calls recorded -- cycles are not grouped, "
            "however the plan reports them"
        )
    print(f"  total calls: {sum(calls.values())}")
    print("=== end group width ===")


import gguf_mtp_c1c8_server_bench as bench  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(bench.main())
