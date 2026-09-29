"""P3 perf re-measure: time layer 29's routed-expert down projection directly
at prompt 1024.

Rather than mutating dispatch to reproduce HEAD (which raises from a call site
HEAD never reaches during decode, so it is not a faithful control), this wraps
``gemma4_project_experts_wmma`` -- the dispatcher P3's fix re-points -- with
HIP events and measures what it costs now. The P3 row records 44.4 ms / 1024
for that down under the old fallback owner
``gguf_k_selected_prefill_out_kernel<...,8>``; the row's Step 2 subject, the
Q5_K gate_up, is untouched and still the 21.9 ms side.

Only prefill blocks satisfy the ``lanes >= 16 * num_experts`` plan gate, so
decode contributes zero calls by construction -- which is exactly the blindness
documented in the punchlist's gate note.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import hipengine
from hipengine.core.hip import get_hip_runtime
from hipengine.llm import SamplingParams
import hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts as ge

ORIGINAL = ge.gemma4_project_experts_wmma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeat", type=int, default=54)
    ap.add_argument("--label", default="now")
    ap.add_argument("--control", action="store_true",
                    help="force _build_wmma_tile_plan to decline, reproducing the "
                         "pre-fix dispatch: line 407 short-circuits on a falsy "
                         "wmma_rows and the down falls through to line 424")
    args = ap.parse_args()

    runtime = get_hip_runtime()
    if args.control:
        # Faithful HEAD reproduction. The real dispatcher does
        #   _ensure_linear_kernel_registered(key); try: resolve(...) except
        #   MissingKernelError: return False
        # so it never propagates an exception -- it returns False and the caller
        # falls through to the selected-GEMV branch. Removing the registration
        # and blocking lazy re-registration for exactly that key reproduces the
        # pre-fix dispatch, scoped to the Q8_0 down (layer 29) only, while every
        # Q5_1 layer keeps its owner in both arms.
        import hipengine.runtime.gguf_linear as gl
        from hipengine.kernels.registry import KernelKey, unregister

        _key = KernelKey(backend="hip_gfx1100", layer="moe_linear", quant="gguf_q8_0",
                         variant="selected_grouped_wmma_prefill_compact_bf16_bf16_out")
        _orig_ens = gl._ensure_linear_kernel_registered

        def _ens(key):
            if key == _key:
                return
            return _orig_ens(key)

        gl._ensure_linear_kernel_registered = _ens
        unregister(_key)
    calls = {"n": 0, "ms": 0.0, "rows": set()}
    mmq = {"n": 0, "ms": 0.0}
    per_layer = {}
    state = {"fwd": -1}

    def _timed(fn, sink):
        def wrap(*a, **k):
            s, e = runtime.event_create(), runtime.event_create()
            runtime.event_record(s, 0)
            out = fn(*a, **k)
            runtime.event_record(e, 0)
            runtime.event_synchronize(e)
            ms = runtime.event_elapsed_time_ms(s, e)
            sink["ms"] += ms
            sink["n"] += 1
            if sink is calls:
                # layers run 0..29 inside each forward, so the down of the last
                # MoE layer is every 30th call -- that is layer 29, P3's subject
                idx = state["fwd"] % 30
                per_layer[idx] = per_layer.get(idx, 0.0) + ms
            runtime.event_destroy(s)
            runtime.event_destroy(e)
            return out
        return wrap

    ge.gemma4_project_experts_wmma = _timed(ORIGINAL, calls)
    ge.gemma4_project_experts_mmq = _timed(ge.gemma4_project_experts_mmq, mmq)

    # observe the block geometry the run actually used
    fwd = ge.gemma4_experts_forward_bf16
    geom = {"rows": set(), "prefill_tokens": 0, "prefill_blocks": 0}
    total = {"n": 0, "ms": 0.0}
    fwd_rows = {}

    def obs(*a, **k):
        sc, rows = k.get("scratch"), k.get("rows")
        r = None
        if sc is not None:
            r = sc.tokens if rows is None else int(rows)
            geom["rows"].add(r)
            if r > 1:  # decode forwards are single-token; prefill blocks are not
                geom["prefill_tokens"] += r
                geom["prefill_blocks"] += 1
        state["fwd"] += 1
        s, e = runtime.event_create(), runtime.event_create()
        runtime.event_record(s, 0)
        out = fwd(*a, **k)
        runtime.event_record(e, 0)
        runtime.event_synchronize(e)
        ms = runtime.event_elapsed_time_ms(s, e)
        total["ms"] += ms
        total["n"] += 1
        if r and r > 1:
            fwd_rows[r] = fwd_rows.get(r, 0.0) + ms
        runtime.event_destroy(s)
        runtime.event_destroy(e)
        return out

    ge.gemma4_experts_forward_bf16 = obs

    llm = hipengine.LLM(model=args.model)
    base = ("The quarterly report covers revenue, operating cost, headcount, and "
            "retention across every region we serve, together with the assumptions "
            "that changed since the previous cycle. ")
    prompt = base * args.repeat
    t0 = time.perf_counter()
    llm.generate(prompt, SamplingParams(max_tokens=4, temperature=0.0))
    wall = (time.perf_counter() - t0) * 1000.0

    print(f"  label              : {args.label}")
    if args.control:
        print("  MODE               : CONTROL (wmma down disabled -> pre-fix dispatch)")
    print(f"  block rows observed: {sorted(geom['rows'])}")
    print(f"  prefill tokens/blocks: {geom['prefill_tokens']} / {geom['prefill_blocks']} blocks (across all 30 layers -> /30 = {geom['prefill_blocks'] // 30 if geom['prefill_blocks'] else 0} per layer)")
    print(f"  down calls (prefill): {calls['n']}  (forward calls {state['fwd'] + 1})")
    print(f"  down total ms      : {calls['ms']:.3f}")
    if calls["n"]:
        print(f"  down per call ms   : {calls['ms'] / calls['n']:.3f}")
    if mmq["n"]:
        print(f"  mmq-fallback ms    : {mmq['ms']:.3f} over {mmq['n']} calls "
              f"({mmq['ms'] / mmq['n']:.3f} ms/call) -- layer 29 part of this")
    if per_layer:
        l29 = per_layer.get(29, 0.0)
        others = [v for i, v in sorted(per_layer.items()) if i != 29]
        avg = sum(others) / len(others) if others else 0.0
        print(f"  layer 29 down ms   : {l29:.3f}   (per 512-row block)")
        print(f"  other layers avg ms: {avg:.3f}   (Q5_1, the already-fast set)")
        print(f"  l29 vs other layers: {l29 / avg:.3f}x" if avg else "")
    print(f"  expert FFN total ms: {total['ms']:.3f} over {total['n']} calls")
    for rr, v in sorted(fwd_rows.items()):
        print(f"     block rows={rr:<5} total {v:8.3f} ms")
    print(f"  generate wall ms   : {wall:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())