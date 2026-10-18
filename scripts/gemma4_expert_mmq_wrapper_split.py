#!/usr/bin/env python3
"""Split the Gemma 4 expert block into its wrapper parts and its MMQ kernel.

The routed-expert block is 42.3 percent of a 512-token prefill (370.1 ms) and it
moves 14.4 GB of expert weights, which is 39 GB/s. That is far under what this
part reaches elsewhere, so the question is whether the weight read is slow or
whether the time is going somewhere that is not the read.

``gemma4_project_experts_gate_up_mmq`` is not one launch. It packs the block's
activations to DS4 Q8_1, builds a tile map, then **reads the tile row count back
to the host** and passes it as a launch argument. The read-back is a pipeline
drain inside a 30-layer loop, and neither the census nor the route ablation can
see it: the census wraps the whole call, so the drain lands inside the expert
leaf's own time.

This probe times each part in place with a HIP event pair on the compute stream.
The imports are function-local, so patching the source modules is enough to
intercept them.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1151 \
        PYTHONPATH=. python3 scripts/gemma4_expert_mmq_wrapper_split.py --prompt 512
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    model = args.model or str(resolve_artifact())
    llm, _runner, info = _resolve_generator(Path(model), 4096)
    print(f"load_s={info['load_s']:.1f} resolution={info['resolution']}", flush=True)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.llm import SamplingParams
    from hipengine.core import memory as memory_mod
    from hipengine.kernels.hip_gfx1100.moe import group_scatter as gs
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_moe as gmoe
    # The expert forward binds the group-scatter and elementwise kernels into its
    # own namespace at import time (``from ... import name``), so patching the
    # defining module leaves the consumer's already-bound reference untouched.
    # Patch the consumer instead -- the same trap the route ablation hit.
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as gexp
    # The router selects its logits kernel through a dict keyed on the registered
    # mode, so both names must be patched to see whichever one runs.
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_router as grtr
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_q8_1_selected_prefill as q4sel
    from hipengine.kernels.hip_gfx1100.quant import gguf_q5_1_mmq_selected_prefill as q5down
    from hipengine.kernels.hip_gfx1100.quant import gguf_q5_k_q8_1_selected_prefill as q5k

    runtime = get_hip_runtime()
    # A mutable holder, not a plain bool: ``timed`` would need ``global`` to
    # rebind a module-level name, and that rebinding is invisible to the
    # closures below, which capture ``main``'s local instead.
    state = {"recording": False}
    totals: dict[str, list] = {}

    def make_spy(name: str, fn):
        def spy(*a, **kw):
            if not state["recording"]:
                return fn(*a, **kw)
            start = runtime.event_create()
            stop = runtime.event_create()
            runtime.event_record(start, 0)
            result = fn(*a, **kw)
            runtime.event_record(stop, 0)
            totals.setdefault(name, []).append((start, stop))
            return result

        return spy

    patches = [
        (
            q4sel,
            "gguf_q4_k_selected_dual_q8_1_ds4_mmq32_fused_prefill_compact32_bf16_bf16_out",
            "q4k gate_up MMQ kernel",
        ),
        (
            q5down,
            "gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out",
            "q5_1 down MMQ (dp4a)",
        ),
        # Layer 29's experts are Q5_K and that path takes the WMMA iu8 route while
        # the Q4_K path takes the scalar dp4a one. Timing both at the same
        # geometry is the direct comparison between the two instruction sets.
        (
            q5k,
            "gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
            "q5_k gate_up WMMA iu8 (layer 29)",
        ),
        (q5k, "gguf_q5_k_selected_dual_sparse_exact_repair_bf16", "q5_k exact repair"),
        (q4sel, "gguf_q8_1_mmq_ds4_pack_bf16", "activation pack (gate_up)"),
        (q4sel, "gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3", "activation pack (down)"),
        # The expert block's remaining glue. The census puts 30.32 ms in
        # ``gemma4_experts_forward_bf16`` outside its two MMQ leaves, and the
        # readback and the pack above account for only 3.9 ms of it. These are
        # the kernels that run in between; before this they were unpriced.
        (gexp, "qwen35_moe_group_count", "group count"),
        (gexp, "qwen35_moe_group_prefix_active", "group prefix (active)"),
        (gexp, "qwen35_moe_group_compact_active", "group compact (active)"),
        (gexp, "qwen35_moe_gather_packed_hidden_lowp", "gather packed hidden"),
        (gexp, "gemma4_moe_lane_to_row_i32", "lane -> row"),
        (gexp, "gemma4_moe_weighted_accumulate_bf16", "weighted accumulate"),
        (gexp, "gemma4_gelu_tanh_mul_bf16", "gelu tanh mul"),
        (grtr, "qwen35_router_logits_bf16_f32w", "router logits (untiled)"),
        (grtr, "qwen35_router_logits_bf16_f32w_token_tile_8", "router logits (tile8)"),
        (grtr, "qwen35_router_logits_bf16_f32w_token_tile_16", "router logits (tile16)"),
        (memory_mod, "copy_device_to_host", "device->host readback"),
    ]
    originals = []
    for module, name, label in patches:
        fn = getattr(module, name, None)
        if fn is None:
            print(f"  MISSING {label}: {module.__name__}.{name}", flush=True)
            continue
        originals.append((module, name, fn))
        setattr(module, name, make_spy(label, fn))

    prompt_ids = list(range(1000, 1000 + args.prompt))
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def prefill() -> None:
        llm.generate_detailed(prompt_ids, params)

    def timed() -> tuple[float, dict]:
        for _ in range(args.warmup):
            prefill()
        samples = []
        for _ in range(args.repeats):
            totals.clear()
            state["recording"] = True
            runtime.device_synchronize()
            started = time.perf_counter()
            prefill()
            runtime.device_synchronize()
            samples.append(time.perf_counter() - started)
            state["recording"] = False
        return statistics.median(samples), totals

    try:
        median, totals = timed()
    finally:
        for module, name, fn in originals:
            setattr(module, name, fn)

    wall_ms = median * 1000.0
    print(f"prefill median {median:.4f} s  ({args.prompt / median:.1f} tok/s)")
    print(f"{'part':30s} {'n':>4s} {'total ms':>10s} {'ms/call':>9s} {'% of step':>10s}")
    rows = []
    accounted = 0.0
    for name, spans in sorted(totals.items(), key=lambda kv: -len(kv[1])):
        total = 0.0
        for start, stop in spans:
            total += runtime.event_elapsed_time_ms(start, stop)
        accounted += total
        rows.append(
            {
                "part": name,
                "calls": len(spans),
                "total_ms": round(total, 3),
                "per_call_ms": round(total / len(spans), 4),
                "pct_of_step": round(100.0 * total / wall_ms, 2),
            }
        )
        print(
            f"{name:30s} {len(spans):4d} {total:10.3f} {total / len(spans):9.4f} "
            f"{100.0 * total / wall_ms:9.2f}%"
        )
    print(f"{'sum of parts':30s} {'':4s} {accounted:10.3f} {'':9s} {100.0 * accounted / wall_ms:9.2f}%")
    print(f"{'prefill wall':30s} {'':4s} {wall_ms:10.3f}")

    payload = {
        "kind": "gemma4_expert_mmq_wrapper_split",
        "performance_claim": False,
        "prompt": args.prompt,
        "prefill_s": median,
        "prefill_tps": args.prompt / median,
        "wall_ms": round(wall_ms, 3),
        "parts": rows,
        "parts_sum_ms": round(accounted, 3),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
