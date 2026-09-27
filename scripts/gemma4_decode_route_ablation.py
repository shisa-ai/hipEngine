#!/usr/bin/env python3
"""Attribute Gemma 4 gfx1151 decode time by overhead-free ablation.

``rocprofv3 --kernel-trace`` serializes dispatches on this host, so a traced
decode run inflates kernel durations about 19x and its shares do not survive
(documented in benchmarks/results/2026-09-28-gemma4-gfx1151-decode-profile.json).
This probe measures the same attribution without tracing: for each route it
skips the route's launches and times the remaining decode with the benchmark's
own wall clock.

The skip is a diagnostic, not a candidate: outputs are wrong on purpose. Only
the timing is read. To keep prefill, first-token latency and sampler cost out of
the number, each arm is timed at two output lengths and the decode rate is the
difference, so the per-token figure is a pure steady-state decode measurement.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python3 scripts/gemma4_decode_route_ablation.py --json out.json
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
        ("no_dense_q8_bf16", "k_launch", ("hipengine_gguf_q8_0_pack8_gemv_bf16_bf16_out",)),
        ("no_lm_head_q8_f32", "k_launch", ("hipengine_gguf_q8_0_pack8_gemv_bf16_f32_out",)),
        (
            "no_dense_q8_both",
            "k_launch",
            (
                "hipengine_gguf_q8_0_pack8_gemv_bf16_bf16_out",
                "hipengine_gguf_q8_0_pack8_gemv_bf16_f32_out",
            ),
        ),
        # Every quantized GEMV the decode step launches. Two funnels cover them:
        # ``gguf_k_gemv._launch`` for the Q8_0/Q5_K/Q6_K families and
        # ``gguf_q4_k_gemv._launch_selected`` for the Q4_K selected family, which
        # is a separate module and would be missed by hooking only the first.
        ("no_all_gemv", "k_launch", ("*",)),
        ("no_experts", "experts", ()),
        ("no_attention", "attention", ()),
        ("no_router", "router", ()),
        ("no_moe_elementwise", "moe_elementwise", ()),
        # The elementwise chain around the projections: normalization, the
        # partial rotary, the branch adds and the gelu. Nothing here reads a
        # weight matrix, so whatever it costs is latency and launch overhead.
        ("no_norm_rope", "norm_rope", ()),
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
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--short-output", type=int, default=8)
    ap.add_argument("--long-output", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=2)
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
        if mode == "k_launch" and ("*" in skip or str(symbol) in skip):
            return None
        return orig_k_launch(quant, symbol, *a, **kw)

    def spy_q4_k_selected(symbol, *a, **kw):
        if mode == "k_launch" and ("*" in skip or str(symbol) in skip):
            return None
        return orig_q4_k_selected(symbol, *a, **kw)

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
    ex.gemma4_project_experts_selected = spy_selected
    gl.gemma4_attention_prefill_bf16 = spy_attention
    gl.gemma4_router_topk_bf16 = spy_router
    for name in elementwise_names:
        setattr(ex, name, make_spy(name))
    for module, name in orig_norm_rope:
        setattr(module, name, make_norm_rope_spy(module, name))

    prompt_ids = list(range(1000, 1000 + args.prompt))
    params = lambda n: SamplingParams(max_tokens=int(n), temperature=0.0, ignore_eos=True)

    def wall(n: int) -> float:
        started = time.perf_counter()
        llm.generate_detailed(prompt_ids, params(n))
        return time.perf_counter() - started

    # Warmup so kernel builds and caches are not inside any arm.
    wall(args.short_output)
    wall(args.long_output)

    results = []
    try:
        for name, arm_mode, symbols in _arms():
            mode = arm_mode
            skip = set(symbols)
            shorts = [wall(args.short_output) for _ in range(args.repeats)]
            longs = [wall(args.long_output) for _ in range(args.repeats)]
            step = args.long_output - args.short_output
            ms = statistics.median(
                [
                    (longs[i] - shorts[i]) / step * 1000.0
                    for i in range(args.repeats)
                ]
            )
            row = {
                "arm": name,
                "decode_ms_per_token": round(ms, 2),
                "decode_tps": round(1000.0 / ms, 2) if ms > 0 else None,
                "short_s": [round(v, 4) for v in shorts],
                "long_s": [round(v, 4) for v in longs],
            }
            results.append(row)
            print(
                f"{name:22s} decode {row['decode_ms_per_token']:7.2f} ms/token  "
                f"{row['decode_tps']:6.2f} tok/s"
            )
    finally:
        mode = ""
        skip = set()
        gk._launch = orig_k_launch
        ex.gemma4_project_experts_selected = orig_selected
        gl.gemma4_attention_prefill_bf16 = orig_attention
        gl.gemma4_router_topk_bf16 = orig_router
        for name in elementwise_names:
            setattr(ex, name, orig_elementwise[name])

    base = next((r for r in results if r["arm"] == "baseline"), None)
    if base is not None:
        print("--- share of the baseline decode step ---")
        for r in results:
            if r is base or not r["decode_ms_per_token"]:
                continue
            delta = base["decode_ms_per_token"] - r["decode_ms_per_token"]
            share = delta / base["decode_ms_per_token"] * 100.0
            r["delta_ms"] = round(delta, 2)
            r["share_pct"] = round(share, 1)
            print(
                f"{r['arm']:22s} saves {delta:7.2f} ms  ({share:5.1f}% of the step)"
            )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "schema": "hipengine.gemma4_decode_route_ablation.v1",
                    "prompt": args.prompt,
                    "short_output": args.short_output,
                    "long_output": args.long_output,
                    "repeats": args.repeats,
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
