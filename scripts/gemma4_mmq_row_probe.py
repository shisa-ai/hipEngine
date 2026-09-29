#!/usr/bin/env python3
"""Print mmq_total_rows when the 32-row MMQ check is about to reject it.

The crash is ``ValueError: mmq_total_rows must be a multiple of 32`` raised by
``_check_mmq32_common`` during a bulk prefill, but the message omits the value,
so the width at which it fires cannot be reasoned about from the outside. This
wraps the check to log its arguments and lets the original raise, so a single
capture run reports the exact row count instead of just the exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill as q8  # noqa: E402

original = q8._check_mmq32_common
seen: list[tuple] = []


def traced(compact_rows, in_features, out_features_a, out_features_b,
           num_experts, mmq_total_rows, expert_stride_rows=0):
    ok = mmq_total_rows % 32 == 0
    seen.append((compact_rows, mmq_total_rows, ok))
    if not ok:
        print(
            f"  REJECTED: mmq_total_rows={mmq_total_rows} "
            f"(mod 32 = {mmq_total_rows % 32}) | compact_rows={compact_rows} "
            f"in={in_features} out_a={out_features_a} out_b={out_features_b} "
            f"experts={num_experts} stride={expert_stride_rows}",
            flush=True,
        )
    else:
        print(
            f"  accepted: mmq_total_rows={mmq_total_rows} compact_rows={compact_rows}",
            flush=True,
        )
    return original(compact_rows, in_features, out_features_a, out_features_b,
                    num_experts, mmq_total_rows, expert_stride_rows)


q8._check_mmq32_common = traced

from scripts.gemma4_teacher_forced_gate import capture_chain, _load_chain  # noqa: E402
from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT  # noqa: E402

width = int(sys.argv[1]) if len(sys.argv) > 1 else 501
length = int(sys.argv[2]) if len(sys.argv) > 2 else 600

print(f"--- width={width} length={length} ---", flush=True)
runner, prompt_ids, _loading = _load_chain(DEFAULT_ARTIFACT, length, 8192)
try:
    logits = capture_chain(runner, prompt_ids, width, None)
    print(f"  capture OK, rows={logits.shape[0]}", flush=True)
except ValueError as exc:
    print(f"  RAISED: {exc}", flush=True)
    print(f"  distinct (compact_rows, mmq_total_rows, ok) seen:", flush=True)
    for row in dict.fromkeys(seen):
        print(f"    {row}", flush=True)
    raise SystemExit(1)