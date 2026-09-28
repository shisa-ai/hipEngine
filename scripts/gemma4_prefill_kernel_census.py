#!/usr/bin/env python3
"""Close the Gemma 4 prefill accounting kernel by kernel.

The route ablation skips routes and reads the delta, which is the right way to
price a route but cannot see anything no arm intercepts. At the 2048-token
default it accounts for the experts, attention, router, norm/rope and MoE
elementwise, and ``scripts/gemma4_prefill_dense_probe.py`` adds the dense
projections and the lm head. About 30 percent of the step is still unnamed.

This probe names it. Instead of skipping a route and timing the difference, it
times *every* kernel the decoder layer launches, in place, with a HIP event pair
recorded on the compute stream. The leaves are:

- ``launch_gguf_linear`` -- the dense projections and the lm head
- ``gemma4_attention_prefill_bf16`` -- attention
- ``gemma4_experts_forward_bf16`` -- the whole routed-expert block
- the norm, rotary, gelu and residual-add wrappers
- ``launch_gguf_embedding`` -- the token gather

The expert block is timed both as a whole and through its own leaves, so the
difference is the block's gather, compaction and scatter glue rather than an
unmeasured remainder. Events are recorded per call and synchronised once at the
end, so the pipeline is not drained between calls.

Whatever the leaves plus the expert glue do not account for is host time: Python,
staging and launch submission. That residual is printed rather than left implied,
because it is the one number that says whether the next unit is a kernel or the
host path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


# Names patched on the ``gemma4_layer`` module. Every one is a kernel wrapper the
# layer calls directly, so patching the module attribute reaches the call.
# Attention is not here: the layer reaches it through the variant selection,
# which resolves a launcher from the attention modules instead. See the patch
# loop below.
_LAYER_KERNELS = (
    "gemma4_experts_forward_bf16",
    "gemma4_gelu_tanh_mul_split_bf16",
    "gemma4_add_rmsnorm_scale_bf16",
    "gemma4_branch_add_bf16",
    "gemma4_head_rmsnorm_f32w_bf16",
    "gemma4_rmsnorm_f32w_bf16",
    "gemma4_rmsnorm_weightless_bf16",
    "gemma4_partial_rotary_bf16",
    "gemma4_router_topk_bf16",
)

# Expert-block leaves. ``gemma4_project_experts_rows`` is the outer entry point
# and calls one of the others, so it is not patched: timing it would nest inside
# the block and double-count. These are the innermost launches.
_EXPERT_LEAVES = (
    "gemma4_project_experts_gate_up_mmq",
    "gemma4_project_experts_down_mmq",
    "gemma4_project_experts_grouped_prefill",
    "gemma4_project_experts_grouped_row4",
    "gemma4_project_experts_selected",
    "gemma4_project_experts_by_offset",
)


def main() -> int:
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

    from hipengine.core.hip import get_hip_runtime
    from hipengine.llm import SamplingParams
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as ga
    from hipengine.kernels.hip_gfx1100.gemma4 import (
        gemma4_attention_prefill_wmma as gaw,
    )
    from hipengine.kernels.hip_gfx1100.gemma4 import (
        gemma4_attention_prefill_wmma_full as gawf,
    )
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as ex
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as gl
    from hipengine.runtime import gemma4 as g4
    from hipengine.runtime import gguf_embedding as ge
    from hipengine.runtime import gguf_linear as glinear

    runtime = get_hip_runtime()
    recording = False
    # name -> [calls, [(start, stop), ...]]
    census: dict[str, list] = {}

    def make_spy(name: str, fn):
        def spy(*a, **kw):
            if not recording:
                return fn(*a, **kw)
            start = runtime.event_create()
            stop = runtime.event_create()
            runtime.event_record(start, 0)
            result = fn(*a, **kw)
            runtime.event_record(stop, 0)
            row = census.setdefault(name, [0, []])
            row[0] += 1
            row[1].append((start, stop))
            return result

        return spy

    originals: list[tuple[object, str, object]] = []

    def patch(module, name: str) -> None:
        fn = getattr(module, name, None)
        if fn is None:
            return
        originals.append((module, name, fn))
        setattr(module, name, make_spy(name, fn))

    for name in _LAYER_KERNELS:
        patch(gl, name)
    # Attention is reached through the variant selection, not through a
    # module-scope name on the layer, so every launcher a profile can select is
    # patched where the selection resolves it: the strict kernel is a global of
    # the attention module, and the two candidates are imported inside the
    # selection, which looks them up on their own modules at call time. A tree
    # that routes attention somewhere else reports attention as unaccounted
    # rather than as free, which is the failure mode this guards against.
    patch(ga, "gemma4_attention_prefill_bf16")
    patch(gaw, "gemma4_attention_prefill_wmma_bf16")
    patch(gawf, "gemma4_attention_prefill_wmma_full_bf16")
    for name in _EXPERT_LEAVES:
        patch(ex, name)
    for name in ("launch_gguf_embedding",):
        patch(ge, name)
    # Two module-scope references for the linear funnel: the kernel package
    # imports it inside the call, the runtime imports it at module scope.
    patch(glinear, "launch_gguf_linear")
    patch(g4, "launch_gguf_linear")

    prompt_ids = list(range(1000, 1000 + args.prompt))
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def wall() -> float:
        started = time.perf_counter()
        llm.generate_detailed(prompt_ids, params)
        return time.perf_counter() - started

    for _ in range(args.warmup):
        wall()

    seconds = 0.0
    for index in range(args.repeats):
        recording = index == args.repeats - 1
        seconds = wall()
    recording = False

    for _module, name, fn in originals:
        setattr(_module, name, fn)

    totals: dict[str, float] = {}
    calls: dict[str, int] = {}
    for name, (count, events) in census.items():
        for start, stop in events:
            runtime.event_synchronize(stop)
        totals[name] = sum(
            runtime.event_elapsed_time_ms(start, stop) for start, stop in events
        )
        calls[name] = count

    step_ms = seconds * 1000.0
    expert_ms = totals.get("gemma4_experts_forward_bf16", 0.0)
    expert_leaf_ms = sum(
        totals.get(name, 0.0)
        for name in _EXPERT_LEAVES
        if name in totals
    )
    # The expert leaves are inside the expert block, so they are reported as a
    # breakdown of it and not added to the total. Summing them would count the
    # block twice, which an earlier revision of this probe did -- it printed
    # 134 percent of the step.
    accounted = sum(
        ms for name, ms in totals.items() if name not in _EXPERT_LEAVES
    )

    print(f"\nprefill wall {seconds:.3f} s  ({args.prompt / seconds:.1f} tok/s)")
    print(f"\n{'kernel':40s} {'n':>6s} {'ms':>10s} {'ms/call':>9s} {'share':>7s}")
    for name, ms in sorted(totals.items(), key=lambda kv: -kv[1]):
        # Indent the expert leaves: they are inside the block above them.
        label = ("  " + name) if name in _EXPERT_LEAVES else name
        print(
            f"{label:40s} {calls[name]:6d} {ms:10.2f} {ms / calls[name]:9.3f} "
            f"{100.0 * ms / step_ms:6.1f}%"
        )
    print(
        f"{'(expert-block glue: block minus leaves)':40s} {'':6s} "
        f"{expert_ms - expert_leaf_ms:10.2f} {'':9s} "
        f"{100.0 * (expert_ms - expert_leaf_ms) / step_ms:6.1f}%"
    )
    print(
        f"\nkernels measured {accounted:.1f} ms "
        f"({100.0 * accounted / step_ms:.1f}% of the step)"
    )
    print(
        f"host and unmeasured {step_ms - accounted:.1f} ms "
        f"({100.0 * (step_ms - accounted) / step_ms:.1f}% of the step)"
    )
    # A census that intercepts no attention at all is reporting a routing bug,
    # not a free step. Every attention launcher a profile can select is spied
    # on above, so zero calls across all of them means the call site moved.
    attention_names = (
        "gemma4_attention_prefill_bf16",
        "gemma4_attention_prefill_wmma_bf16",
        "gemma4_attention_prefill_wmma_full_bf16",
    )
    if not any(calls.get(name) for name in attention_names):
        print(
            "\nWARNING: no attention launcher was intercepted "
            f"({', '.join(attention_names)}). The step above charges attention "
            "to host and unmeasured time."
        )

    payload = {
        "kind": "gemma4_prefill_kernel_census",
        "recorded": time.strftime("%Y-%m-%d"),
        "prompt": args.prompt,
        "prefill_s": round(seconds, 4),
        "prefill_tps": round(args.prompt / seconds, 2),
        "kernels": [
            {
                "name": name,
                "calls": calls[name],
                "total_ms": round(ms, 3),
                "per_call_ms": round(ms / calls[name], 4),
                "share": round(ms / step_ms, 4),
            }
            for name, ms in sorted(totals.items(), key=lambda kv: -kv[1])
        ],
        "expert_block_ms": round(expert_ms, 3),
        "expert_leaves_ms": round(expert_leaf_ms, 3),
        "accounted_ms": round(accounted, 3),
        "host_and_unmeasured_ms": round(step_ms - accounted, 3),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
