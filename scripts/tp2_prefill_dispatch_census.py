"""Phase-filtered dispatch census for the TP2 bulk prefill.

A plain "log every resolve around ``bulk_prefill``" census is wrong, and wrong in
a way that looks like a finding. ``bulk_prefill`` warms the decode route inside
itself (``_forward_token_eager`` and the layer graph capture), so the log
contains decode-shaped resolutions - ``t16_gemv_decode`` variants for tensors the
prefill never touches with that kernel - next to the prefill ones. Reading that
list as "what the prefill runs" invents fallbacks that are not there.

This tool filters by call stack instead of by time: a resolution is attributed to
the prefill only when one of the bulk-prefill layer helpers is on the stack. The
distinction is the whole point of the tool, so it is asserted rather than
assumed - ``--show-excluded`` prints what the filter dropped, which is what the
decode warmup contributed.

It answers two questions the attribution cannot:

- which variant each tensor resolves to on the prefill path, so a silent
  fallback to a decode-shaped or non-WMMA kernel is visible;
- whether that set changes between two arms (a capability toggle, an env flag),
  which is how a comparison verifies that the arm actually changed the kernels
  it claims to change.

Usage::

    # one arm
    python scripts/tp2_prefill_dispatch_census.py --json /tmp/census.json

    # A/B: report only the tensors whose variant differs between arms
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

    prefill: collections.Counter[tuple[str, str, str]] = collections.Counter()
    excluded: collections.Counter[tuple[str, str, str]] = collections.Counter()
    original = gguf_linear.resolve_gguf_linear_dispatch

    def logged(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        key = out.key
        entry = (
            _short(_tensor_of(args)),
            str(getattr(key, "quant", "?")),
            str(getattr(key, "variant", "?")),
        )
        # One frame up is this wrapper, so start the walk at the caller.
        (prefill if _stack_has(PREFILL_FRAMES) else excluded)[entry] += 1
        return out

    gguf_linear.resolve_gguf_linear_dispatch = logged

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
        prefill.clear()
        excluded.clear()
        started = time.perf_counter()
        session.bulk_prefill(prompt, logits_rows=logits_rows)
        wall_ms = (time.perf_counter() - started) * 1000.0
    finally:
        session.close()
        gguf_linear.resolve_gguf_linear_dispatch = original

    return {
        "schema": 1,
        "kind": "tp2_prefill_dispatch_census",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": model,
        "devices": list(devices),
        "prompt_tokens": int(prompt_tokens),
        "logits_rows": int(logits_rows),
        "capability": capability,
        "wall_ms": round(wall_ms, 3),
        "prefill": [
            {"tensor": tensor, "quant": quant, "variant": variant, "n": n}
            for (tensor, quant, variant), n in sorted(prefill.items())
        ],
        "excluded_decode_warmup": [
            {"tensor": tensor, "quant": quant, "variant": variant, "n": n}
            for (tensor, quant, variant), n in sorted(excluded.items())
        ],
    }


def _print_rows(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}: {len(rows)} distinct (tensor, quant, variant)")
    if not rows:
        return
    print(f"  {'tensor':<26}{'quant':<26}{'variant':<48}{'n':>5}")
    for row in rows:
        print(
            f"  {row['tensor']:<26}{row['quant']:<26}{row['variant']:<48}{row['n']:>5}"
        )


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
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    devices = tuple(int(part) for part in str(args.devices).split(","))
    report: dict[str, Any] = {"schema": 1, "kind": "tp2_prefill_dispatch_census_pair"}
    base = collect(
        args.model,
        devices,
        args.prompt_tokens,
        args.max_sequence_length,
        None,
        args.logits_rows,
    )
    report["base"] = base
    _print_rows(base["prefill"], "prefill path (base arm)")
    if args.show_excluded:
        _print_rows(
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
        _print_rows(other["prefill"], f"prefill path ({args.ab_capability})")
        base_map = {(r["tensor"], r["quant"]): r["variant"] for r in base["prefill"]}
        arm_map = {(r["tensor"], r["quant"]): r["variant"] for r in other["prefill"]}
        print("\nvariant differences between the arms:")
        differences = []
        for key in sorted(set(base_map) | set(arm_map)):
            if base_map.get(key) != arm_map.get(key):
                differences.append(
                    {
                        "tensor": key[0],
                        "quant": key[1],
                        "base_variant": base_map.get(key),
                        "arm_variant": arm_map.get(key),
                    }
                )
        for row in differences:
            print(
                f"  {row['tensor']:<26}{row['quant']:<26}"
                f"{str(row['base_variant']):<48} -> {row['arm_variant']}"
            )
        if not differences:
            print("  none: the arm did not change any resolved variant")
        report["variant_differences"] = differences

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
