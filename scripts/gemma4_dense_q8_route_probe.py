#!/usr/bin/env python3
"""Measure the dense Q8_0 prefill route: bf16 WMMA against the int8 MMQ chain.

Gemma 4's dense attention projections (q/k/v/o) are Q8_0 in `UD-Q4_K_XL` and
run `gguf_q8_0_prefill_wmma_kernel`, 159 ms of a 554 ms per-prefill kernel
budget. An int8 MMQ128 owner for the same quant already exists and the Gemma
runner already carries a crossover map naming these six shapes -- but the map is
inert, because `_wmma_prefill_dispatch` rewrites the incoming
`prefill_bf16_bf16_out` to `wmma_prefill_bf16_bf16_out` before
`_q8_mmq_prefill_dispatch` looks at it, and the MMQ gate only recognises the
un-rewritten name.

This probe answers two questions in one run, on the real model:

1. Which route does each dense shape take today, and what does each cost?
2. With the candidate gate (accept the WMMA-rewritten name) applied as a
   monkeypatch, what does the MMQ chain cost at those same shapes?

It patches nothing in the production modules; the candidate is carried here so
the measurement can be taken before any code change.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        .venv/bin/python scripts/gemma4_dense_q8_route_probe.py \
        --prompt 512 --arm both --json-out /tmp/gemma4_dense_q8_route.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/"
        "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
    )
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument(
        "--arm",
        choices=("incumbent", "candidate", "both"),
        default="both",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-block", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_prefill as q8wmma
    from hipengine.runtime import gguf_linear as gl

    # ---- route census + timing -------------------------------------------
    census: collections.Counter = collections.Counter()
    timing: dict[tuple[str, int, int, int], list[float]] = collections.defaultdict(list)

    orig_wmma_launch = q8wmma._launch

    def spy_wmma(symbol, x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, **kw):
        tm, tn = kw.get("tile_m"), kw.get("tile_n")
        if tm is None or tn is None:
            tm_def, tn_def = q8wmma._default_tiles(rows, in_features, out_features)
            tm = tm_def if tm is None else tm
            tn = tn_def if tn is None else tn
        key = ("wmma", rows, in_features, out_features)
        census[key] += 1
        start = time.perf_counter()
        try:
            return orig_wmma_launch(
                symbol, x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, **kw
            )
        finally:
            timing[key].append((time.perf_counter() - start) * 1e3)
            census[("wmma-tile", tm, tn, 0)] += 1

    orig_mmq_launch = gl._launch_raw_mmq_d4x3

    def spy_mmq(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs):
        key = ("mmq", rows, in_features, out_features)
        census[key] += 1
        start = time.perf_counter()
        try:
            return orig_mmq_launch(
                fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs
            )
        finally:
            timing[key].append((time.perf_counter() - start) * 1e3)

    q8wmma._launch = spy_wmma
    gl._LAUNCH_ABI["raw_mmq_d4x3"] = spy_mmq
    # ---- the candidate gate, as a monkeypatch ----------------------------
    orig_mmq_dispatch = gl._q8_mmq_prefill_dispatch
    _WMMA_REWRITTEN = {
        "wmma_prefill_bf16_bf16_out": "mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out",
        "wmma_prefill_bf16_f32_out": "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
        "wmma_prefill_f32_f32_out": "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
    }

    def candidate_mmq_dispatch(dispatch, *, rows, in_features, out_features):
        result = orig_mmq_dispatch(
            dispatch, rows=rows, in_features=in_features, out_features=out_features
        )
        if result is not dispatch:
            return result
        session = gl._q8_mmq_prefill_session.get()
        if session is None or not session.policy(rows, in_features, out_features):
            return dispatch
        target = _WMMA_REWRITTEN.get(dispatch.key.variant)
        if target is None or dispatch.abi != "wmma_raw":
            return dispatch
        return gl.GGUFLinearDispatch(
            gl.KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                target,
            ),
            "raw_mmq_d4x3",
        )

    # ---- drive one real prefill ------------------------------------------
    import hipengine

    def run_arm(arm: str) -> dict:
        census.clear()
        timing.clear()
        gl._q8_mmq_prefill_dispatch = (
            candidate_mmq_dispatch if arm == "candidate" else orig_mmq_dispatch
        )
        llm = hipengine.LLM(model=args.model)
        generator = llm._get_text_generator()
        generator.context_length = 4096
        runner = generator._ensure_runner()
        if args.max_block:
            runner.max_block = int(args.max_block)
        try:
            tokens = list(generator.tokenize("The quick brown fox jumps over " * 200))[
                : args.prompt
            ]
            from hipengine.core.hip import get_hip_runtime

            sync = get_hip_runtime().device_synchronize
            best = None
            for _ in range(args.repeats):
                runner.reset()
                sync()
                start = time.perf_counter()
                runner.forward(tokens)
                sync()
                elapsed = time.perf_counter() - start
                best = elapsed if best is None else min(best, elapsed)
        finally:
            close = getattr(llm, "close", None)
            if close is not None:
                close()
        per_shape = {}
        for key, samples in timing.items():
            route, rows, in_f, out_f = key
            per_shape[f"{route}:{in_f}->{out_f}"] = {
                "route": route,
                "rows": rows,
                "in_features": in_f,
                "out_features": out_f,
                "launches": len(samples),
                # Host enqueue time, not device time: these launches are async.
                # The device cost is attributed from a rocprofv3 kernel trace;
                # this arm's decision metric is ``wall_s``.
                "host_ms_total": round(sum(samples), 3),
            }
        return {
            "arm": arm,
            "prompt": args.prompt,
            "wall_s": best,
            "shapes": dict(sorted(per_shape.items())),
        }

    arms = ("incumbent", "candidate") if args.arm == "both" else (args.arm,)
    out = [run_arm(arm) for arm in arms]
    payload = {
        "prompt": args.prompt,
        "max_block": args.max_block or None,
        "repeats": args.repeats,
        "arms": out,
    }
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=1)
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
