#!/usr/bin/env python3
"""Attribute Gemma 4 gfx1151 prefill time by overhead-free route ablation.

The prefill gap to the same-host llama.cpp reference is 3.74x at 512 tokens
(271.0 against 1012.98 tok/s) and it widens with prompt length (4.85x at 2048),
while llama.cpp's prefill is flat. The campaign's existing prefill attribution is
gfx1100 and does not carry over: gfx1151 is a different lane.

This probe measures the same attribution without tracing, by skipping one route's
launches and timing the remaining prefill with the benchmark's own wall clock.
``rocprofv3 --kernel-trace`` is not usable here: it serializes dispatches on this
host and inflates kernel durations about 19x.

The skip is a diagnostic, not a candidate: outputs are wrong on purpose. Only the
timing is read.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python3 scripts/gemma4_prefill_route_ablation.py --prompt 2048
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def _arms() -> list[tuple[str, str, tuple[str, ...]]]:
    """(arm name, choke point, symbols/functions skipped)"""

    return [
        ("baseline", "", ()),
        # The four dense Q8_0 prefill leaves and the lm head.
        ("no_dense_q8", "k_launch", ("hipengine_gguf_q8_0",)),
        ("no_all_gemv", "k_launch", ("*",)),
        ("no_experts", "experts", ()),
        ("no_attention", "attention", ()),
        ("no_router", "router", ()),
        ("no_norm_rope", "norm_rope", ()),
        ("no_moe_elementwise", "moe_elementwise", ()),
    ]


# Elementwise and normalization entry points, spied per arm. They are attributes
# of the layer and expert modules rather than registry lookups, because the layer
# calls them directly.
_NORM_ROPE_NAMES = (
    "gemma4_rmsnorm_f32w_bf16",
    "gemma4_rmsnorm_weightless_bf16",
    "gemma4_add_rmsnorm_scale_bf16",
    "gemma4_partial_rotary_bf16",
    "gemma4_head_rmsnorm_f32w_bf16",
    "gemma4_branch_add_bf16",
    "gemma4_gelu_tanh_mul_split_bf16",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    model = args.model or str(resolve_artifact())
    llm, _runner, info = _resolve_generator(Path(model), 4096)
    print(f"load_s={info['load_s']:.1f} resolution={info['resolution']}")

    from hipengine.llm import SamplingParams
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as ex
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as gl
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as gk
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_gemv as gq

    orig_k_launch = gk._launch
    orig_q4_k_selected = gq._launch_selected
    orig_experts_rows = ex.gemma4_project_experts_rows
    orig_selected = ex.gemma4_project_experts_selected
    orig_attention = gl.gemma4_attention_prefill_bf16
    orig_router = gl.gemma4_router_topk_bf16
    elementwise_names = [
        name
        for name in (
            "qwen35_moe_group_count",
            "qwen35_moe_group_prefix_active",
            "qwen35_moe_group_compact_active",
            "qwen35_moe_gather_packed_hidden_lowp",
            "gemma4_gelu_tanh_mul_bf16",
            "gemma4_moe_lane_to_row_i32",
            "gemma4_moe_weighted_accumulate_bf16",
        )
        if hasattr(ex, name)
    ]
    orig_elementwise = {name: getattr(ex, name) for name in elementwise_names}
    orig_norm_rope = {
        (module, name): getattr(module, name)
        for module in (gl, ex)
        for name in _NORM_ROPE_NAMES
        if hasattr(module, name)
    }

    skip: set[str] = set()
    mode = ""

    def spy_k_launch(quant, symbol, *a, **kw):
        if mode == "k_launch" and ("*" in skip or any(str(symbol).startswith(s) for s in skip)):
            return None
        return orig_k_launch(quant, symbol, *a, **kw)

    def spy_q4_k_selected(symbol, *a, **kw):
        if mode == "k_launch" and ("*" in skip or any(str(symbol).startswith(s) for s in skip)):
            return None
        return orig_q4_k_selected(symbol, *a, **kw)

    def spy_experts_rows(*a, **kw):
        if mode == "experts":
            return "skipped"
        return orig_experts_rows(*a, **kw)

    def spy_selected(weight, x_ptr, selected_ptr, out_ptr, *a, **kw):
        if mode == "experts":
            return True
        return orig_selected(weight, x_ptr, selected_ptr, out_ptr, *a, **kw)

    def spy_attention(*a, **kw):
        if mode == "attention":
            return None
        return orig_attention(*a, **kw)

    def spy_router(*a, **kw):
        if mode == "router":
            return None
        return orig_router(*a, **kw)

    def make_spy(name):
        def spy(*a, **kw):
            if mode == "moe_elementwise":
                return None
            return orig_elementwise[name](*a, **kw)

        return spy

    def make_norm_rope_spy(module, name):
        def spy(*a, **kw):
            if mode == "norm_rope":
                return None
            return orig_norm_rope[(module, name)](*a, **kw)

        return spy

    gk._launch = spy_k_launch
    gq._launch_selected = spy_q4_k_selected
    ex.gemma4_project_experts_rows = spy_experts_rows
    ex.gemma4_project_experts_selected = spy_selected
    gl.gemma4_attention_prefill_bf16 = spy_attention
    gl.gemma4_router_topk_bf16 = spy_router
    for name in elementwise_names:
        setattr(ex, name, make_spy(name))
    for module, name in orig_norm_rope:
        setattr(module, name, make_norm_rope_spy(module, name))

    prompt_ids = list(range(1000, 1000 + args.prompt))
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def wall() -> float:
        started = time.perf_counter()
        llm.generate_detailed(prompt_ids, params)
        return time.perf_counter() - started

    for _ in range(args.warmup):
        wall()

    results = []
    try:
        for name, arm_mode, symbols in _arms():
            mode = arm_mode
            skip = set(symbols)
            times = [wall() for _ in range(args.repeats)]
            seconds = statistics.median(times)
            row = {
                "arm": name,
                "prefill_s": round(seconds, 4),
                "prefill_tps": round(args.prompt / seconds, 1),
                "samples": [round(v, 4) for v in times],
            }
            results.append(row)
            print(f"{name:22s} prefill {seconds:7.3f} s  {row['prefill_tps']:8.1f} tok/s")
    finally:
        mode = ""
        skip = set()

    baseline = next(r for r in results if r["arm"] == "baseline")
    print("\n--- share of the baseline prefill ---")
    for row in results[1:]:
        delta = baseline["prefill_s"] - row["prefill_s"]
        print(
            f"{row['arm']:22s} saves {delta:7.3f} s  "
            f"({100.0 * delta / baseline['prefill_s']:5.1f}% of the step)"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "schema": "hipengine.gemma4_prefill_route_ablation.v1",
                    "prompt": args.prompt,
                    "repeats": args.repeats,
                    "results": results,
                },
                indent=1,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
