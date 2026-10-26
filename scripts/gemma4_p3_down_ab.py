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

The same run is the Q5_1 down owner's in-process A/B. Production runs the
Q5_1T16 tile leaf (``_gemma4_project_experts_down_wmma_t16``), which launches
outside ``gemma4_project_experts_wmma``; a run that timed only the latter
stopped measuring the Q5_1 layers when that leaf landed. Both owners are timed
on every call, and ``--down-t16 off`` pins the raw-block grouped WMMA owner for
this process through the guard the leaf itself reads. The arm is installed
through ``_Patches`` and undone in ``main``'s ``finally``, so a raising run
cannot leave a pinned or wrapped module behind and two runs in one process
cannot time a wrapper of themselves. A run whose selected arm never launched
its owner prints ``arm check: MISMATCH`` and exits nonzero.
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


class _Patches:
    """Module-attribute patches that are always undone.

    ``main`` installs its timing wrappers and the ``--down-t16 off`` arm through
    this so every exit path restores the module it borrowed.
    """

    def __init__(self) -> None:
        self._saved: list[tuple[object, str, object]] = []

    def set(self, owner: object, name: str, value: object) -> None:
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def restore(self) -> None:
        """Undo every patch, newest first. Idempotent."""

        while self._saved:
            owner, name, value = self._saved.pop()
            setattr(owner, name, value)


def _pin_raw_block_owner(patches: _Patches) -> None:
    """Pin the raw-block grouped WMMA down owner for this process.

    The comparison arm for the Q5_1T16 tile leaf. It moves the guard the leaf
    itself reads -- ``_gemma4_down_t16_tiles_ready`` -- so the leaf declines
    every weight and the caller's existing fall-through takes the raw-block
    owner. No second selector and no environment variable; ``patches.restore()``
    puts the production guard back.
    """

    patches.set(ge, "_gemma4_down_t16_tiles_ready", lambda weight: False)


def _arm_check(
    arm: str, tile_launched: int, raw_launched: int, routes: dict[str, int]
) -> bool:
    """Whether the run's own counters show the selected arm ran its owner.

    ``routes`` is the ``gemma4_moe_expert_route_counts()`` delta for the run.
    The tile leaf records ``down_t16`` once per launch; the raw-block owner
    records no route of its own, so its launches come from the event sink. A
    run whose owner never launched is a mismatch, not a zero-cost result.
    """

    if arm == "on":
        return tile_launched > 0 and routes.get("down_t16", 0) == tile_launched
    return (tile_launched == 0
            and routes.get("down_t16", 0) == 0
            and raw_launched > 0)


def _timed(runtime, fn, sink, on_launch=None):
    """Wrap ``fn`` in a HIP event pair, counting launches and declines apart.

    Both down owners return whether they launched, so only a truthy return
    contributes to ``sink["ms"]``: a decline is a fall-through to another
    owner, not this owner's duration. The event pair is destroyed on every
    exit, including an exception from ``fn``.
    """

    def wrap(*a, **k):
        start, end = runtime.event_create(), runtime.event_create()
        try:
            runtime.event_record(start, 0)
            out = fn(*a, **k)
            runtime.event_record(end, 0)
            runtime.event_synchronize(end)
            sink["n"] += 1
            if not out:
                sink["declined"] += 1
                return out
            ms = runtime.event_elapsed_time_ms(start, end)
            sink["ms"] += ms
            if on_launch is not None:
                on_launch(ms)
            return out
        finally:
            runtime.event_destroy(start)
            runtime.event_destroy(end)

    return wrap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeat", type=int, default=54)
    ap.add_argument("--label", default="now")
    ap.add_argument("--down-t16", choices=("on", "off"), default="on",
                    help="on (default) runs the production Q5_1T16 tile leaf; off "
                         "pins the raw-block grouped WMMA owner in this process, "
                         "the tile leaf's comparison arm. Restored in main's "
                         "finally")
    ap.add_argument("--control", action="store_true",
                    help="force _build_wmma_tile_plan to decline, reproducing the "
                         "pre-fix dispatch: the down short-circuits on a falsy "
                         "wmma_rows and falls through to the selected-GEMV branch")
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

    patches = _Patches()
    calls = {"n": 0, "ms": 0.0, "declined": 0}
    tile = {"n": 0, "ms": 0.0, "declined": 0}
    mmq = {"n": 0, "ms": 0.0, "declined": 0}
    per_layer: dict[int, float] = {}
    state = {"fwd": -1}

    def _record_layer(ms: float) -> None:
        # layers run 0..29 inside each forward, so the down of the last MoE
        # layer is every 30th call -- that is layer 29, P3's subject
        idx = state["fwd"] % 30
        per_layer[idx] = per_layer.get(idx, 0.0) + ms

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
        try:
            runtime.event_record(s, 0)
            out = fwd(*a, **k)
            runtime.event_record(e, 0)
            runtime.event_synchronize(e)
            ms = runtime.event_elapsed_time_ms(s, e)
            total["ms"] += ms
            total["n"] += 1
            if r and r > 1:
                fwd_rows[r] = fwd_rows.get(r, 0.0) + ms
            return out
        finally:
            runtime.event_destroy(s)
            runtime.event_destroy(e)

    llm = None
    try:
        if args.down_t16 == "off":
            _pin_raw_block_owner(patches)
        patches.set(ge, "gemma4_project_experts_wmma",
                    _timed(runtime, ge.gemma4_project_experts_wmma, calls,
                           on_launch=_record_layer))
        patches.set(ge, "_gemma4_project_experts_down_wmma_t16",
                    _timed(runtime, ge._gemma4_project_experts_down_wmma_t16, tile))
        patches.set(ge, "gemma4_project_experts_mmq",
                    _timed(runtime, ge.gemma4_project_experts_mmq, mmq))
        patches.set(ge, "gemma4_experts_forward_bf16", obs)

        llm = hipengine.LLM(model=args.model)
        base = ("The quarterly report covers revenue, operating cost, headcount, and "
                "retention across every region we serve, together with the assumptions "
                "that changed since the previous cycle. ")
        prompt = base * args.repeat
        before = ge.gemma4_moe_expert_route_counts()
        t0 = time.perf_counter()
        llm.generate(prompt, SamplingParams(max_tokens=4, temperature=0.0))
        wall = (time.perf_counter() - t0) * 1000.0
        after = ge.gemma4_moe_expert_route_counts()
    finally:
        patches.restore()
        if llm is not None:
            llm.close()

    observed = {
        name: after.get(name, 0) - before.get(name, 0)
        for name in sorted(set(before) | set(after))
        if after.get(name, 0) - before.get(name, 0)
    }
    tile_launched = tile["n"] - tile["declined"]
    raw_launched = calls["n"] - calls["declined"]
    arm_ok = _arm_check(args.down_t16, tile_launched, raw_launched, observed)

    def _owner_line(name: str, sink: dict, launched: int) -> str:
        # A decline is a fall-through, not a duration for this owner, so only
        # launched calls carry a time.
        line = f"  {name:<19}: {sink['n']} calls, {launched} launched"
        if launched:
            line += f", {sink['ms']:.3f} ms total, {sink['ms'] / launched:.3f} ms/launch"
        return line

    print(f"  label              : {args.label}")
    print(f"  arm                : down-t16={args.down_t16}"
          + ("  (raw-block grouped WMMA pinned; the tile leaf declines)"
             if args.down_t16 == "off"
             else "  (production Q5_1T16 tile leaf)"))
    if args.control:
        print("  MODE               : CONTROL (wmma down disabled -> pre-fix dispatch)")
    print(f"  block rows observed: {sorted(geom['rows'])}")
    print(f"  prefill tokens/blocks: {geom['prefill_tokens']} / {geom['prefill_blocks']} blocks (across all 30 layers -> /30 = {geom['prefill_blocks'] // 30 if geom['prefill_blocks'] else 0} per layer)")
    print(f"  forward calls      : {state['fwd'] + 1}")
    print(_owner_line("tile down", tile, tile_launched))
    print(_owner_line("raw-block down", calls, raw_launched))
    print(f"  engine route counts: {observed}")
    print("  arm check          : "
          + ("ok" if arm_ok else "MISMATCH -- this arm did not run its owner"))
    if mmq["n"]:
        print(f"  mmq-fallback ms    : {mmq['ms']:.3f} over {mmq['n']} calls "
              f"({mmq['ms'] / mmq['n']:.3f} ms/call) -- layer 29 part of this")
    if per_layer:
        l29 = per_layer.get(29, 0.0)
        others = [v for i, v in sorted(per_layer.items()) if i != 29]
        avg = sum(others) / len(others) if others else 0.0
        other_note = ("Q5_1, the already-fast set" if args.down_t16 == "off"
                      else "Q8_0 layers; the Q5_1 layers ran the tile leaf")
        print(f"  raw-block layer 29 ms: {l29:.3f}   (per 512-row block)")
        print(f"  raw-block other layers avg ms: {avg:.3f}   ({other_note})")
        if avg:
            print(f"  l29 vs other layers: {l29 / avg:.3f}x")
    print(f"  expert FFN total ms: {total['ms']:.3f} over {total['n']} calls")
    for rr, v in sorted(fwd_rows.items()):
        print(f"     block rows={rr:<5} total {v:8.3f} ms")
    print(f"  generate wall ms   : {wall:.1f}")
    # A wrapper script or a CI step must be able to tell a stale arm from a
    # result, so the mismatch line is also the exit status.
    return 0 if arm_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
