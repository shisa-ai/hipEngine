#!/usr/bin/env python3
"""Census of one Gemma 4 decode step: launches, weight bytes, x bytes.

The route ablation says how much time each family costs; it does not say how
many bytes that family moved, so it cannot tell a bandwidth-bound kernel from a
latency-bound one. This probe hooks the launch funnels, sums the weight and
activation traffic of every launch in one decode step, and prints bytes and
effective bandwidth per family.

Diagnostic only: it wraps the funnels, so the decode it measures is slower than
a clean one, and only the ratios are read.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python3 /tmp/gemma4_decode_traffic.py --prompt 8 --output 16
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# GGUF block bytes per element, by quant key.
BPE = {
    "gguf_q8_0": 34 / 32,
    "gguf_q4_k": 144 / 256,
    "gguf_q4_k_m": 144 / 256,
    "gguf_q5_1": 24 / 32,
    "gguf_q5_k": 176 / 256,
    "gguf_q6_k": 210 / 256,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--output", type=int, default=16)
    args = ap.parse_args()

    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    llm, runner, info = _resolve_generator(resolve_artifact(), 4096)
    print(f"load_s={info['load_s']:.1f} resolution={info['resolution']}")

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as ex
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as gl
    from hipengine.runtime import gguf_linear as ggl

    counts: Counter = Counter()
    shape_counts: Counter = Counter()
    bytes_in: Counter = Counter()  # weight + activation read
    bytes_out: Counter = Counter()
    enabled = {"on": False}

    orig_dense = ggl.launch_gguf_linear
    orig_selected = ex.gemma4_project_experts_selected
    orig_mmq = ex.gemma4_project_experts_gate_up_mmq
    orig_rows = ex.gemma4_project_experts_rows
    orig_grouped = ex.gemma4_project_experts_grouped_prefill

    def bpe(quant: str) -> float:
        return BPE.get(quant, 0.0)

    def spy_dense(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw):
        if enabled["on"]:
            quant = weight.spec.quant_key
            name = f"dense {quant}"
            counts[name] += 1
            shape_counts[(quant, rows, in_features, out_features)] += 1
            # One block per output row; each block reads the x row (bf16) and
            # one weight row.
            bytes_in[name] += out_features * in_features * bpe(quant)
            bytes_in[name] += rows * in_features * 2
            bytes_out[name] += rows * out_features * (4 if kw.get("output_dtype") == "f32" else 2)
        return orig_dense(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw)

    def spy_selected(weight, x_ptr, selected_ptr, out_ptr, *a, **kw):
        if enabled["on"] and not isinstance(weight, int):
            quant = weight.spec.quant_key
            name = f"expert-selected {quant}"
            counts[name] += 1
            # Signature: (x, selected, qweight, out, x_rows, rows, num_experts,
            # in_features, out_features). A per-row selected block reads one
            # weight row per (out_col, lane), so weight bytes are rows *
            # out_features * in_features, and the x row is re-read per block.
            rows, num_experts, in_features, out_features = a[1], a[2], a[3], a[4]
            bytes_in[name] += rows * out_features * in_features * bpe(quant)
            bytes_out[name] += rows * out_features * 2
        return orig_selected(weight, x_ptr, selected_ptr, out_ptr, *a, **kw)

    def spy_mmq(weight, *a, **kw):
        if enabled["on"] and not isinstance(weight, int):
            counts["expert-mmq gate_up"] += 1
        return orig_mmq(weight, *a, **kw)

    def spy_rows(*a, **kw):
        route = orig_rows(*a, **kw)
        if enabled["on"]:
            counts[f"route:{route}"] += 1
        return route

    def spy_grouped(weight, *a, **kw):
        if enabled["on"] and not isinstance(weight, int):
            counts[f"expert-grouped {weight.spec.quant_key}"] += 1
        return orig_grouped(weight, *a, **kw)

    ggl.launch_gguf_linear = spy_dense
    ex.gemma4_project_experts_selected = spy_selected
    ex.gemma4_project_experts_gate_up_mmq = spy_mmq
    ex.gemma4_project_experts_rows = spy_rows
    ex.gemma4_project_experts_grouped_prefill = spy_grouped

    # Elementwise and normalization entry points.
    ew_names = (
        "gemma4_rmsnorm_f32w_bf16",
        "gemma4_rmsnorm_weightless_bf16",
        "gemma4_add_rmsnorm_scale_bf16",
        "gemma4_partial_rotary_bf16",
        "gemma4_head_rmsnorm_f32w_bf16",
        "gemma4_branch_add_bf16",
        "gemma4_gelu_tanh_mul_split_bf16",
        "gemma4_attention_prefill_bf16",
        "gemma4_router_topk_bf16",
    )
    originals = {}
    for module in (gl, ex):
        for name in ew_names:
            if not hasattr(module, name):
                continue
            originals[(module, name)] = getattr(module, name)

            def make(module=module, name=name):
                fn = originals[(module, name)]

                def spy(*a, **kw):
                    if enabled["on"]:
                        counts[f"layer {name}"] += 1
                    return fn(*a, **kw)

                return spy

            setattr(module, name, make())

    from hipengine.llm import SamplingParams

    prompt_ids = list(range(1000, 1000 + args.prompt))
    params = SamplingParams(max_tokens=args.output, temperature=0.0, ignore_eos=True)
    llm.generate_detailed(prompt_ids, params)

    counts.clear()
    bytes_in.clear()
    bytes_out.clear()
    enabled["on"] = True
    try:
        llm.generate_detailed(prompt_ids, params)
    finally:
        enabled["on"] = False

    steps = args.output - 1
    print(f"\n=== per decode step ({steps} decode forwards) ===")
    print(f"{'family':34s} {'n/step':>8s} {'W+x MB/step':>12s} {'out MB/step':>12s}")
    total = 0.0
    for name, n in counts.most_common():
        gi = bytes_in[name] / steps / 1e6
        go = bytes_out[name] / steps / 1e6
        total += gi + go
        print(f"{name:34s} {n / steps:8.2f} {gi:12.1f} {go:12.1f}")
    print(f"{'TOTAL':34s} {'':8s} {'':12s} {total:12.1f} MB/step")
    print(f"launches per step: {sum(counts.values()) / steps:.1f}")
    print("\n=== dense projection shapes, per step ===")
    print(f"{'quant':14s} {'rows':>5s} {'in':>6s} {'out':>7s} {'n/step':>8s} {'MB/step':>9s}")
    for (quant, rows, i, o), n in shape_counts.most_common(20):
        print(f"{quant:14s} {rows:5d} {i:6d} {o:7d} {n / steps:8.2f} "
              f"{n * o * i * BPE.get(quant, 0.0) / steps / 1e6:9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
