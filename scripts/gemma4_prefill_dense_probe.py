#!/usr/bin/env python3
"""Attribute the Gemma 4 prefill time no ablation arm reaches.

``scripts/gemma4_prefill_route_ablation.py`` accounts for the expert, attention,
router, norm/rope and MoE-elementwise routes by skipping each one. Its two
catch-all arms do not work: ``no_dense_q8`` matches kernels the prefill does not
launch, and ``no_all_gemv`` skips every ``k_launch`` and reports *negative*
savings, which is impossible. Skipping every GEMV cannot save less than skipping
the experts alone.

The reason is structural rather than a bug in the arm. The dense projections do
not go through the launch funnels those arms patch: ``gemma4_layer_forward_bf16``
calls ``gemma4_project``, which calls ``launch_gguf_linear`` in the runtime
layer. So at the 2048-token default the ablation accounts for the experts
(11.1%), attention (36.5%), the router (0.3%), norm/rope (0.1%) and the MoE
elementwise (1.1%) -- and leaves about half the step unattributed, with no arm
able to say where it goes.

This probe closes that hole. It wraps ``launch_gguf_linear`` -- the single funnel
every dense projection and the lm head passes through -- and times each real
launch with a HIP event pair on the compute stream, grouped by
``(quant, rows, in_features, out_features)``. Events are recorded per call and
synchronised once at the end, so the pipeline is not drained between calls and
the per-call times are the ones the GPU actually spends.

Reading the output:

- ``ms`` is GPU time inside that group, not wall time. The groups plus the
  ablation's routes should account for the whole step; what is left is host time
  and any launch neither probe wraps.
- ``TFLOP/s`` is ``2 * rows * in_features * out_features`` over that GPU time.
  ``GB/s`` is the weight bytes read once over the same time. A dense prefill
  GEMM at ``rows=2048`` should be compute-bound and clear 10 TFLOP/s; a group
  that is far under that while its ``GB/s`` is also low is neither bound and is
  the thing to look at next.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    model = args.model or str(resolve_artifact())
    llm, _runner, info = _resolve_generator(Path(model), 4096)
    print(f"load_s={info['load_s']:.1f} resolution={info['resolution']}")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.llm import SamplingParams
    from hipengine.runtime import gemma4 as g4
    from hipengine.runtime import gguf_linear as gl

    runtime = get_hip_runtime()
    orig_linear = gl.launch_gguf_linear

    # (key) -> [calls, bytes, [(start_event, stop_event), ...]]
    groups: dict[tuple, list] = {}
    recording = False

    def spy_linear(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw):
        nonlocal recording
        if not recording:
            return orig_linear(
                weight, x_ptr, out_ptr, rows, in_features, out_features, **kw
            )
        spec = getattr(weight, "spec", None)
        quant = getattr(spec, "quant_key", None) or "unknown"
        try:
            nbytes = int(weight.allocation().nbytes)
        except Exception:
            nbytes = 0
        key = (quant, int(rows), int(in_features), int(out_features))
        start = runtime.event_create()
        stop = runtime.event_create()
        runtime.event_record(start, 0)
        result = orig_linear(
            weight, x_ptr, out_ptr, rows, in_features, out_features, **kw
        )
        runtime.event_record(stop, 0)
        row = groups.setdefault(key, [0, nbytes, []])
        row[0] += 1
        row[2].append((start, stop))
        return result

    # Two module-scope references, not one. ``gemma4_project`` imports the
    # function inside the call so patching the owning module reaches it, but
    # ``hipengine/runtime/gemma4.py`` imports it at module scope for the lm head
    # -- which is the one launch in the whole step whose weight is read at
    # ``rows=1`` and is therefore the one most likely to be memory-bound. An
    # earlier revision of this probe patched only the first and silently missed
    # the lm head entirely.
    gl.launch_gguf_linear = spy_linear
    g4.launch_gguf_linear = spy_linear

    prompt_ids = list(range(1000, 1000 + args.prompt))
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def wall() -> float:
        started = time.perf_counter()
        llm.generate_detailed(prompt_ids, params)
        return time.perf_counter() - started

    for _ in range(args.warmup):
        wall()

    # Only the last repeat is recorded, so the groups describe a single prefill
    # rather than an accumulation over repeats. The earlier repeats still run so
    # the measured step is not the one paying for first-touch.
    seconds = 0.0
    for index in range(args.repeats):
        recording = index == args.repeats - 1
        seconds = wall()
    recording = False

    for row in groups.values():
        for start, stop in row[2]:
            runtime.event_synchronize(stop)
        row[2] = [
            runtime.event_elapsed_time_ms(start, stop) for start, stop in row[2]
        ]

    gl.launch_gguf_linear = orig_linear
    g4.launch_gguf_linear = orig_linear

    measured_ms = 0.0
    rows_out = []
    for (quant, rows, in_features, out_features), row in groups.items():
        calls, nbytes, times = row
        total_ms = sum(times)
        measured_ms += total_ms
        flops = 2.0 * rows * in_features * out_features * calls
        rows_out.append(
            {
                "quant": quant,
                "rows": rows,
                "in_features": in_features,
                "out_features": out_features,
                "calls": calls,
                "total_ms": round(total_ms, 3),
                "per_launch_ms": round(total_ms / calls, 4),
                "tflops": round(flops / (total_ms / 1000.0) / 1e12, 3),
                "weight_mb": round(nbytes / 1e6, 2),
                "gb_per_s": round(nbytes * calls / (total_ms / 1000.0) / 1e9, 2),
            }
        )
    rows_out.sort(key=lambda r: -r["total_ms"])

    print(f"\nprefill wall {seconds:.3f} s  ({args.prompt / seconds:.1f} tok/s)")
    print(
        f"launch_gguf_linear GPU time {measured_ms:.1f} ms  "
        f"({100.0 * measured_ms / (seconds * 1000.0):.1f}% of the step)"
    )
    print(
        f"\n{'quant':14s} {'rows':>6s} {'in':>6s} {'out':>6s} {'n':>5s} "
        f"{'ms':>9s} {'ms/call':>8s} {'TFLOP/s':>8s} {'GB/s':>7s}"
    )
    for r in rows_out[: args.top]:
        print(
            f"{r['quant']:14s} {r['rows']:6d} {r['in_features']:6d} "
            f"{r['out_features']:6d} {r['calls']:5d} {r['total_ms']:9.2f} "
            f"{r['per_launch_ms']:8.3f} {r['tflops']:8.2f} {r['gb_per_s']:7.1f}"
        )

    payload = {
        "kind": "gemma4_prefill_dense_probe",
        "recorded": time.strftime("%Y-%m-%d"),
        "prompt": args.prompt,
        "prefill_s": round(seconds, 4),
        "prefill_tps": round(args.prompt / seconds, 2),
        "launch_gguf_linear_ms": round(measured_ms, 3),
        "launch_gguf_linear_share": round(
            measured_ms / (seconds * 1000.0), 4
        ),
        "groups": rows_out,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
