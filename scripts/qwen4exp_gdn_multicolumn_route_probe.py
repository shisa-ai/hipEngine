#!/usr/bin/env python3
"""Does the promoted multi-column GDN prefill route actually fire?

The `gdn_multi_column` profile gate reports `measurement_valid: true` and
`numerical_passed: true`, but it also reports `dispatch_count_mode: "not_counted"`
and `candidate_dispatch_calls: None` - so a green numerical status does not
establish that the candidate route ran. This is the DE-3 trap: a KL gate over a
route that never engaged is vacuous, and DE-3's first run was discarded for
exactly that reason.

This probe answers the only question that matters before trusting the gate: does
`qwen4_exp_gdn_prefill_tiled16_multi_f32` get dispatched at all, at which shapes,
and does the gate that guards it return true? It answers it by wrapping the
registered candidate callable and the selection predicate and counting real
calls during a real prefill.

Usage:
    python3 scripts/qwen4exp_gdn_multicolumn_route_probe.py [--rows 512]
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT_MODEL = pathlib.Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=4)
    args = ap.parse_args()

    from hipengine.runtime import qwen4_exp_runner as runner
    from hipengine.kernels import registry

    predicate_calls = []
    predicate_env = []
    candidate_calls = []
    other_prefill_calls = []

    # 1. Wrap the selection predicate that guards the tile16 route.
    original_predicate = runner.qwen4_exp_gdn_tile16_prefill_selected

    import traceback
    caller_frames = []

    def counting_predicate(**kwargs):
        result = original_predicate(**kwargs)
        if len(predicate_env) < 4:
            predicate_env.append(os.environ.get("HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT",
                                                "<unset>"))
        if len(caller_frames) < 3:
            stack = traceback.extract_stack()[:-1]
            caller_frames.append([f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}"
                                  for f in stack[-4:]])
        predicate_calls.append((result, kwargs.get("rows"), kwargs.get("num_k_heads"),
                                kwargs.get("num_v_heads"), kwargs.get("head_dim")))
        return result

    runner.qwen4_exp_gdn_tile16_prefill_selected = counting_predicate

    # 2. Wrap every registered GDN prefill callable so we see which one actually runs.
    TARGETS = {
        "qwen4exp_gdn_tiled16_multi_prefill",
        "qwen4exp_gdn_tiled16_dpp_prefill",
        "qwen4exp_gdn_tiled16_prefill",
    }
    wrapped = {}
    for key in list(registry.registered_keys()):
        if key.layer != "gdn_recurrence_norm_gate" or key.variant not in TARGETS:
            continue
        original = registry.resolve(
            backend=key.backend, layer=key.layer, quant=key.quant,
            variant=key.variant,
        )
        if original is None:
            continue

        def make(orig, variant):
            def wrapper(*a, **kw):
                record = {"variant": variant, "rows": a[9] if len(a) > 9 else None}
                if variant == "qwen4exp_gdn_tiled16_multi_prefill":
                    candidate_calls.append(record)
                else:
                    other_prefill_calls.append(record)
                return orig(*a, **kw)
            return wrapper

        wrapped[key.variant] = key
        registry.register(key, make(original, key.variant), replace=True)

    print(f"wrapped registered GDN prefill variants: {sorted(wrapped)}")
    print(f"env HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT = "
          f"{os.environ.get('HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT', '<unset>')}")

    # 3. Run a real prefill+decode through the production profile.
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND, QWEN4_EXP_MODEL, register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    # Now that the full registry exists, wrap the tiled variants that were absent
    # on the first pass.
    for key in list(registry.registered_keys()):
        if key.layer != "gdn_recurrence_norm_gate" or key.variant not in TARGETS:
            continue
        if key.variant in wrapped:
            continue
        original = registry.resolve(
            backend=key.backend, layer=key.layer, quant=key.quant,
            variant=key.variant,
        )
        if original is None:
            continue

        def make3(orig, variant):
            def wrapper(*a, **kw):
                record = {"variant": variant, "rows": a[9] if len(a) > 9 else None}
                if variant == "qwen4exp_gdn_tiled16_multi_prefill":
                    candidate_calls.append(record)
                else:
                    other_prefill_calls.append(record)
                return orig(*a, **kw)
            return wrapper

        wrapped[key.variant] = key
        registry.register(key, make3(original, key.variant), replace=True)
    print(f"after full registration, wrapped: {sorted(wrapped)}")
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant="gguf_q4_k_m", profile=ExecutionProfile.PRODUCTION,
    )
    profile.binder(type("S", (), {"manifest": profile.manifest})(), profile)
    print(f"after binder, HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT = "
          f"{os.environ.get('HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT', '<unset>')}")

    # Re-wrap after the binder: profile resolution can re-register kernels.
    for key in list(registry.registered_keys()):
        if key.layer != "gdn_recurrence_norm_gate" or key.variant not in TARGETS:
            continue
        if key.variant in wrapped:
            continue
        original = registry.resolve(
            backend=key.backend, layer=key.layer, quant=key.quant,
            variant=key.variant,
        )
        if original is None:
            continue

        def make2(orig, variant):
            def wrapper(*a, **kw):
                record = {"variant": variant, "rows": a[9] if len(a) > 9 else None}
                if variant == "qwen4exp_gdn_tiled16_multi_prefill":
                    candidate_calls.append(record)
                else:
                    other_prefill_calls.append(record)
                return orig(*a, **kw)
            return wrapper

        registry.register(key, make2(original, key.variant), replace=True)

    # Catch whoever writes the var to "" between construction and dispatch.
    import traceback as _tb
    from hipengine.generation import qwen4_exp_profiles as _prof
    _orig_apply = _prof._apply_profile_env
    writers = []

    def traced_apply(values):
        if "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT" in values:
            writers.append((values["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"],
                            [f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}"
                             for f in _tb.extract_stack()[-5:-1]]))
        return _orig_apply(values)

    _prof._apply_profile_env = traced_apply

    from hipengine import LLM, SamplingParams

    llm = LLM(model=str(args.model_root), backend=QWEN4_EXP_BACKEND,
              execution_profile="production")

    # LLM construction re-registers kernels, which would clobber wrappers installed
    # before it. Install the final wrappers now, and verify they are what resolve
    # actually returns.
    for key in list(registry.registered_keys()):
        if key.layer != "gdn_recurrence_norm_gate" or key.variant not in TARGETS:
            continue
        current = registry.resolve(
            backend=key.backend, layer=key.layer, quant=key.quant,
            variant=key.variant,
        )
        if current is None or getattr(current, "_probe_wrapped", False):
            continue

        def make4(orig, variant):
            def wrapper(*a, **kw):
                record = {"variant": variant, "rows": a[9] if len(a) > 9 else None}
                if variant == "qwen4exp_gdn_tiled16_multi_prefill":
                    candidate_calls.append(record)
                else:
                    other_prefill_calls.append(record)
                return orig(*a, **kw)
            wrapper._probe_wrapped = True
            return wrapper

        registry.register(key, make4(current, key.variant), replace=True)

    # Confirm the wrapper is on the resolve path for the variant the env selects.
    selected = os.environ.get("HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT", "")
    if selected:
        probe = registry.resolve(backend="hip_gfx1151", layer="gdn_recurrence_norm_gate",
                                 quant="f32_state", variant=selected)
        print(f"resolve(selected={selected!r}) -> {getattr(probe, '__name__', probe)} "
              f"wrapped={getattr(probe, '_probe_wrapped', False)}")
    else:
        print(f"selected variant is empty: the runner will use the directly imported "
              f"qwen4_exp_gdn_prefill_tiled16_f32 base route")

    # Instrument the runner's OWN resolve, which is what the dispatch region calls.
    import inspect
    print(f"runner.resolve is {runner.resolve}")
    try:
        print(f"  from module: {inspect.getmodule(runner.resolve).__name__}")
    except Exception as exc:  # pragma: no cover
        print(f"  (module lookup failed: {exc})")
    resolve_calls = []
    _orig_resolve = runner.resolve

    def logging_resolve(*a, **kw):
        out = _orig_resolve(*a, **kw)
        resolve_calls.append((kw.get("variant"), getattr(out, "__name__", str(out)),
                              getattr(out, "_probe_wrapped", False)))
        return out

    runner.resolve = logging_resolve

    print(f"AFTER LLM construction, GDN_TILE16_VARIANT = "
          f"{os.environ.get('HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT', '<unset>')!r}")

    prompt_ids = [9707] * args.rows
    llm.generate([prompt_ids],
                 SamplingParams(max_tokens=args.max_tokens, temperature=0.0))

    print()
    print("=== runner.resolve calls from the dispatch region ===")
    print(f"  total: {len(resolve_calls)}")
    seen = collections.Counter(v for v, _, _ in resolve_calls)
    print("  distinct variants resolved:")
    for v, n in seen.most_common():
        print(f"    {n:5}x  {v}")
    tiled = [c for c in resolve_calls if c[0] and "tiled16" in str(c[0])]
    print(f"  resolve calls naming a tiled16 variant: {len(tiled)}")

    print()
    print("=== predicate call sites ===")
    for fr in caller_frames:
        print("   " + " <- ".join(reversed(fr)))

    print(f"=== env var seen INSIDE the predicate: {predicate_env} ===")
    print(f"=== _apply_profile_env writes of GDN_TILE16_VARIANT: {len(writers)} ===")
    for val, stack in writers[:4]:
        print(f"    -> {val!r}")
        for fr in stack:
            print(f"         {fr}")
    print("=== predicate ===")
    print(f"  calls: {len(predicate_calls)}")
    counts = collections.Counter(c[0] for c in predicate_calls)
    print(f"  returned True: {counts.get(True, 0)}   returned False: {counts.get(False, 0)}")
    shapes = collections.Counter(c[1:] for c in predicate_calls)
    for shape, n in shapes.most_common(6):
        print(f"    rows/kh/vh/d = {shape}: {n}")

    print()
    print("=== actual dispatches ===")
    print(f"  multi-column calls : {len(candidate_calls)}")
    print(f"  other tiled calls  : {len(other_prefill_calls)}")
    for r in other_prefill_calls[:4]:
        print(f"    {r['variant']} rows={r['rows']}")

    print()
    if candidate_calls:
        print(f"VERDICT: ENGAGED - the multi-column route ran {len(candidate_calls)} times.")
        return 0
    print("VERDICT: NOT ENGAGED - the multi-column route never ran. The gate's "
          "numerical pass is vacuous for this change.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
