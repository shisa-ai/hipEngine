"""Phase-filtered dispatch census for the TP2 bulk prefill.

Two things make a naive census lie, and this tool fixes both.

**Warmup contamination.** ``bulk_prefill`` warms the decode route inside itself
(``_forward_token_eager`` and the layer graph capture), so a log taken "around
``bulk_prefill``" contains decode-shaped resolutions - ``t16_gemv_decode``
variants for tensors the prefill never touches with that kernel - next to the
prefill ones. Reading that list as "what the prefill runs" invents fallbacks that
are not there. This tool filters by call stack instead of by time: a resolution
belongs to the prefill only when one of the bulk-prefill layer helpers is on the
stack. ``--show-excluded`` prints what the filter dropped, which is what the
decode warmup contributed.

**Pre-rewrite keys.** This is the subtler one and it invalidated an earlier
version of this tool. ``resolve_gguf_linear_dispatch`` returns a *candidate*; the
runtime then applies rewrites that replace ``dispatch.key.variant`` before
anything launches - ``_q6_t16_f16_rocblas_prefill_dispatch``, the pair overrides,
and others. The kernel that actually runs is chosen by the final
``kernels.registry.resolve(backend=, layer=, quant=, variant=)``. A census that
records the candidate therefore reports variants that never execute: it will
happily show ``t16_gemv_decode`` for a tensor that ran an f16-rocBLAS kernel.

So the authoritative record is the final ``resolve`` call, and the candidate is
kept only as a *secondary* field. The difference between them is itself the
useful signal - it names exactly which rewrite fired - and ``--rewrites`` prints
it.

Usage::

    # what actually launches on the prefill path
    python scripts/tp2_prefill_dispatch_census.py --json /tmp/census.json

    # which rewrites changed the candidate, and to what
    python scripts/tp2_prefill_dispatch_census.py --rewrites

    # A/B: report only the tensors whose launched variant differs between arms
    python scripts/tp2_prefill_dispatch_census.py --ab-capability \\
        GGUF_Q4_DUAL_SILU_PREFILL_OUT_FEATURES=17408,8704
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/lhl/hipEngine-main")

# A resolution belongs to the prefill only if the stack passes through one of
# these. They are the bulk-prefill layer helpers; anything else reached from
# ``bulk_prefill`` (the decode warmup, the graph capture, the final head) is a
# different route and is reported separately.
PREFILL_FRAMES = frozenset(
    {
        "_run_bulk_prefill_layers",
        "_bulk_attention_layer",
        "_bulk_norm_residual_layer",
        "_bulk_sharded_mlp_layer",
    }
)


def _stack_has(frame_names: frozenset[str]) -> bool:
    """Walk the live stack looking for any of ``frame_names``.

    ``sys._getframe`` rather than ``traceback``: this runs once per resolve, and
    the resolution path is hot enough that building traceback objects for every
    projection measurably perturbs the run.
    """
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name in frame_names:
            return True
        frame = frame.f_back
    return False


def _tensor_of(args: tuple[Any, ...]) -> str:
    for arg in args:
        spec = getattr(arg, "spec", None)
        slot = getattr(spec, "slot_path", None)
        if slot:
            return str(slot)
    return "?"


def _short(slot: str) -> str:
    return slot.split(".", 1)[-1] if slot.startswith("layers.") else slot


def collect(
    model: str,
    devices: tuple[int, ...],
    prompt_tokens: int,
    max_sequence_length: int,
    capability: str | None,
    logits_rows: int,
) -> dict[str, Any]:
    import hipengine.runtime.gguf_linear as gguf_linear

    if capability:
        name, _, value = capability.partition("=")
        widths = frozenset(int(part) for part in value.split(",") if part.strip())
        import hipengine.kernels.hip_gfx1100 as backend

        if not hasattr(backend, name):
            raise SystemExit(f"the gfx1100 backend has no capability named {name!r}")
        previous = getattr(backend, name)
        setattr(backend, name, widths)
        print(f"capability {name}: {sorted(previous)} -> {sorted(widths)}")

    # The authoritative record: the final resolve, which is what picks the
    # kernel that runs. Keyed by (tensor, quant, variant).
    launched: collections.Counter[tuple[str, str, str]] = collections.Counter()
    # The candidate, before the runtime's rewrites. Kept only so the rewrite
    # delta is visible; on its own it does not describe execution.
    candidate: dict[str, tuple[str, str]] = {}
    excluded: collections.Counter[tuple[str, str, str]] = collections.Counter()

    original_resolve = gguf_linear.resolve
    original_candidate = gguf_linear.resolve_gguf_linear_dispatch

    def logged_resolve(*args: Any, **kwargs: Any) -> Any:
        backend = kwargs.get("backend")
        layer = kwargs.get("layer")
        quant = kwargs.get("quant")
        variant = kwargs.get("variant", "")
        entry = (str(quant), str(variant))
        if layer == "linear" and _stack_has(PREFILL_FRAMES):
            launched[("", *entry)] += 1
        elif layer == "linear":
            excluded[("", *entry)] += 1
        return original_resolve(*args, **kwargs)

    def logged_candidate(*args: Any, **kwargs: Any) -> Any:
        out = original_candidate(*args, **kwargs)
        if _stack_has(PREFILL_FRAMES):
            tensor = _tensor_of(args)
            key = out.key
            candidate[tensor] = (str(key.quant), str(key.variant))
        return out

    gguf_linear.resolve = logged_resolve
    gguf_linear.resolve_gguf_linear_dispatch = logged_candidate

    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    session = MlpTP2GenerationSession(
        model,
        devices=devices,
        mode="tp2",
        max_sequence_length=int(max_sequence_length),
        bulk_prefill=True,
        bulk_prefill_rows=int(prompt_tokens),
        reduce_mode="device",
    )
    prompt = [9707] * int(prompt_tokens)
    try:
        session.bulk_prefill(prompt, logits_rows=logits_rows)  # warmup
        gguf_linear.clear_gguf_linear_dispatch_cache()
        launched.clear()
        excluded.clear()
        candidate.clear()
        started = time.perf_counter()
        session.bulk_prefill(prompt, logits_rows=logits_rows)
        wall_ms = (time.perf_counter() - started) * 1000.0
    finally:
        session.close()
        gguf_linear.resolve = original_resolve
        gguf_linear.resolve_gguf_linear_dispatch = original_candidate

    return {
        "schema": 2,
        "kind": "tp2_prefill_dispatch_census",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": model,
        "devices": list(devices),
        "prompt_tokens": int(prompt_tokens),
        "logits_rows": int(logits_rows),
        "capability": capability,
        "wall_ms": round(wall_ms, 3),
        "authority": (
            "launched is the final kernels.registry.resolve call, i.e. the kernel the "
            "launcher selected after the runtime's rewrites. candidate is the pre-rewrite "
            "value from resolve_gguf_linear_dispatch and does NOT describe execution."
        ),
        "launched": [
            {"quant": quant, "variant": variant, "n": n}
            for (_, quant, variant), n in sorted(launched.items())
        ],
        "candidate": [
            {"tensor": tensor, "quant": quant, "variant": variant}
            for tensor, (quant, variant) in sorted(candidate.items())
        ],
        "excluded_decode_warmup": [
            {"quant": quant, "variant": variant, "n": n}
            for (_, quant, variant), n in sorted(excluded.items())
        ],
    }


def _print_launched(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}: {len(rows)} distinct (quant, variant)")
    if not rows:
        return
    print(f"  {'quant':<44}{'variant':<50}{'n':>6}")
    for row in rows:
        print(f"  {row['quant']:<44}{row['variant']:<50}{row['n']:>6}")


def _print_candidates(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}: {len(rows)} tensors")
    if not rows:
        return
    print(f"  {'tensor':<26}{'quant':<40}{'variant':<46}")
    for row in rows:
        print(f"  {row['tensor']:<26}{row['quant']:<40}{row['variant']:<46}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--logits-rows", type=int, default=1)
    parser.add_argument(
        "--ab-capability",
        default=None,
        help="NAME=v1,v2 arm to compare against the default arm",
    )
    parser.add_argument("--show-excluded", action="store_true")
    parser.add_argument(
        "--rewrites",
        action="store_true",
        help="print the pre-rewrite candidate alongside what launched",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    devices = tuple(int(part) for part in str(args.devices).split(","))
    report: dict[str, Any] = {"schema": 2, "kind": "tp2_prefill_dispatch_census_pair"}
    base = collect(
        args.model,
        devices,
        args.prompt_tokens,
        args.max_sequence_length,
        None,
        args.logits_rows,
    )
    report["base"] = base
    _print_launched(base["launched"], "launched on the prefill path (base arm)")
    if args.rewrites:
        _print_candidates(
            base["candidate"],
            "pre-rewrite candidate from resolve_gguf_linear_dispatch (NOT what runs)",
        )
    if args.show_excluded:
        _print_launched(
            base["excluded_decode_warmup"],
            "excluded: decode warmup reached from bulk_prefill",
        )

    if args.ab_capability:
        other = collect(
            args.model,
            devices,
            args.prompt_tokens,
            args.max_sequence_length,
            args.ab_capability,
            args.logits_rows,
        )
        report["arm"] = other
        _print_launched(other["launched"], f"launched on the prefill path ({args.ab_capability})")
        base_map = {(r["quant"], r["variant"]): r["n"] for r in base["launched"]}
        arm_map = {(r["quant"], r["variant"]): r["n"] for r in other["launched"]}
        print("\nlaunched-variant differences between the arms:")
        differences = []
        for key in sorted(set(base_map) | set(arm_map)):
            if base_map.get(key) != arm_map.get(key):
                differences.append(
                    {
                        "quant": key[0],
                        "variant": key[1],
                        "base_n": base_map.get(key, 0),
                        "arm_n": arm_map.get(key, 0),
                    }
                )
        for row in differences:
            print(
                f"  {row['quant']:<44}{row['variant']:<50}"
                f"base {row['base_n']:>5} -> arm {row['arm_n']:>5}"
            )
        if not differences:
            print("  none: the arm did not change any launched variant")
        report["launched_differences"] = differences

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
