#!/usr/bin/env python3
"""Is a Gemma 4 decode step host-launch-bound?

The traffic census puts one decode step at 3219.6 MB of weight traffic and 825.6
kernel launches. At 256 GB/s that traffic is 12.9 ms of device work, and the
step takes 43.9 ms, so something other than device bandwidth is on the critical
path. This probe removes the device work and keeps the host path: every launch
funnel returns immediately, so the decode loop still runs its full Python call
chain and its synchronization but issues nothing.

Whatever wall time is left is the host floor for this launch structure. If it is
a large fraction of 43.9 ms, the decode step is launch-bound and the lever is
fewer launches, not faster kernels.

Diagnostic only: outputs are wrong on purpose. Only the timing is read.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python3 /tmp/gemma4_host_floor.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SHORT = 8
LONG = 40


def main() -> int:
    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    llm, runner, info = _resolve_generator(resolve_artifact(), 4096)
    print(f"load_s={info['load_s']:.1f} resolution={info['resolution']}")

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as ex
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as gl
    from hipengine.runtime import gguf_linear as ggl
    from hipengine.llm import SamplingParams

    stubs: list[tuple[object, str, object]] = []

    def stub(module, name, value):
        stubs.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    # Every dense GGUF projection, and every expert projection.
    stub(ggl, "launch_gguf_linear", lambda *a, **k: None)
    for name in (
        "gemma4_project_experts_selected",
        "gemma4_project_experts_gate_up_mmq",
        "gemma4_project_experts_grouped_prefill",
    ):
        if hasattr(ex, name):
            stub(ex, name, (lambda *a, **k: False) if name != "gemma4_project_experts_selected" else (lambda *a, **k: True))
    # The elementwise and normalization chain, plus attention and the router.
    for module in (gl, ex):
        for name in (
            "gemma4_rmsnorm_f32w_bf16",
            "gemma4_rmsnorm_weightless_bf16",
            "gemma4_add_rmsnorm_scale_bf16",
            "gemma4_partial_rotary_bf16",
            "gemma4_head_rmsnorm_f32w_bf16",
            "gemma4_branch_add_bf16",
            "gemma4_gelu_tanh_mul_split_bf16",
            "gemma4_attention_prefill_bf16",
            "gemma4_router_topk_bf16",
        ):
            if hasattr(module, name):
                stub(module, name, lambda *a, **k: None)

    prompt_ids = list(range(1000, 1000 + 8))

    def params(n):
        return SamplingParams(max_tokens=int(n), temperature=0.0, ignore_eos=True)

    def wall(n: int) -> float:
        started = time.perf_counter()
        llm.generate_detailed(prompt_ids, params(n))
        return time.perf_counter() - started

    try:
        wall(SHORT)
        wall(LONG)
        shorts = [wall(SHORT) for _ in range(3)]
        longs = [wall(LONG) for _ in range(3)]
    finally:
        for module, name, original in reversed(stubs):
            setattr(module, name, original)

    step = LONG - SHORT
    ms = statistics.median([(longs[i] - shorts[i]) / step * 1000.0 for i in range(3)])
    print(f"\nhost floor, no launches issued: {ms:.2f} ms/token ({1000.0 / ms:.1f} tok/s)")
    print(f"short_s={[round(v, 4) for v in shorts]}")
    print(f"long_s ={[round(v, 4) for v in longs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
