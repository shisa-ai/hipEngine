#!/usr/bin/env python3
"""Observe attention launcher invocations during a public Gemma LLM.generate request.

Selection telemetry counts successful wrapper invocations, not merely resolution
or generated-ID parity. GPU completion is checked separately with synchronize.
This probe observes the product route; it does not override variant selection.

Both phases are observed, and they are observed through different hooks because
they reach the device differently. Prefill goes through
``layer.select_prefill_attention`` and the layer's native launcher functions.
Decode goes through the registered variants (``gemma4_flash`` and the staged
split), which bypass those functions and instead all pass through
``attention._launch_prefill`` -- so a probe that wraps only the native launchers
reports prefill launches and no decode launches at all, which is exactly the
shape of an answer that looks like "this scope is not reached" when the scope
was reached and merely invisible.

Decode is recorded twice, because the two records answer different questions.
``scoped decode resolution`` is the route the layer resolved for that call, which
is what the plan's decode selection decides; it reports ``exact`` below the
resident LDS bound, where the plan does not select a variant.
``launcher kernel selection`` is the kernel the launcher chose for that call,
read from ``attention.decode_selection`` after the wrapped call returns. That
reading is per-call: the wrapped function is the single funnel for decode
symbols and the launcher writes the counter on every branch it can take. The two
disagree by design below the bound -- ``launcher._launch_prefill`` admits flash
whenever the shape is admitted and ``decode_slices`` asks for more than one
slice, without any plan selection, so a short-context decode launch reports
``exact`` for the resolution and the flash kernel for the selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@contextmanager
def observe_attention_launches():
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer

    original = layer.select_prefill_attention
    records = []

    def observed(**geometry):
        selection = original(**geometry)
        launch = selection.launcher

        def invoked(*args, **kwargs):
            result = launch(*args, **kwargs)
            records.append({
                "variant": selection.variant,
                "reason": selection.reason,
                **{name: geometry.get(name) for name in
                   ("tokens", "keys", "head_dim", "num_heads", "num_kv_heads")},
            })
            return result

        return replace(selection, launcher=invoked)

    native_launches = {
        'gemma4_attention_prefill_tiled': 'tiled',
        'gemma4_attention_prefill_aotriton': 'aotriton',
        'gemma4_attention_prefill_bf16': 'exact',
    }
    originals = {name: getattr(layer, name) for name in native_launches}

    def observe_native(launch, variant):
        def invoked(*args, **kwargs):
            result = launch(*args, **kwargs)
            records.append({
                'variant': variant,
                'reason': 'native capability route',
                **{name: kwargs.get(name) for name in
                   ('tokens', 'keys', 'head_dim', 'num_heads', 'num_kv_heads')},
            })
            return result
        return invoked

    layer.select_prefill_attention = observed
    for name, variant in native_launches.items():
        setattr(layer, name, observe_native(originals[name], variant))

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention

    decode_names = {0: "block", 1: attention.DECODE_ATTENTION_PLAIN,
                    2: attention.DECODE_ATTENTION_STAGED,
                    3: attention.DECODE_ATTENTION_FLASH}
    original_launch = attention._launch_prefill

    def observe_decode(*args, **kwargs):
        result = original_launch(*args, **kwargs)
        if kwargs.get("tokens") == 1:
            selection = int(attention.decode_selection(kwargs.get("library")))
            records.append({
                "variant": decode_names.get(selection, f"selection_{selection}"),
                "reason": "launcher kernel selection",
                **{name: kwargs.get(name) for name in
                   ("tokens", "keys", "head_dim", "num_heads", "num_kv_heads")},
            })
        return result

    attention._launch_prefill = observe_decode

    original_resolver = layer._select_exact_decode_attention

    def observe_resolution(*args, **kwargs):
        launcher, route = original_resolver(*args, **kwargs)
        if kwargs.get("tokens") == 1:
            records.append({
                "variant": route,
                "reason": "scoped decode resolution",
                **{name: kwargs.get(name) for name in
                   ("tokens", "keys", "head_dim", "num_heads", "num_kv_heads",
                    "sliding_window")},
            })
        return launcher, route

    layer._select_exact_decode_attention = observe_resolution
    try:
        yield records
    finally:
        layer._select_exact_decode_attention = original_resolver
        attention._launch_prefill = original_launch
        layer.select_prefill_attention = original
        for name, launch in originals.items():
            setattr(layer, name, launch)


def summarize_attention_launches(records):
    """Count by query width, not semantic phase: prefill can end in one row.

    ``reason`` is part of the key because a decode call produces two records with
different meanings -- the route the layer resolved and the kernel the launcher
    selected -- and they legitimately disagree.
    """
    counts = Counter(
        (r["variant"], r.get("reason"), "singleton" if r["tokens"] == 1 else "multi_token")
        for r in records
    )
    return [{"variant": variant, "reason": reason, "query_width": width,
             "invocations": count}
            for (variant, reason, width), count in sorted(
                counts.items(), key=lambda item: (item[0][2], item[0][0], item[0][1] or ""))]


def main(argv=None):
    from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT, exact_prompt_ids

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--prompt", type=int, default=8192)
    parser.add_argument("--output", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prompt < 1 or args.output < 1:
        parser.error("prompt and output must be positive")

    import hipengine
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.hip import get_hip_runtime
    from hipengine.llm import SamplingParams

    llm = hipengine.LLM(model=str(args.artifact), max_sequence_length=args.prompt + args.output)
    generator = llm._get_text_generator()
    ids = exact_prompt_ids(generator.tokenize, args.prompt)
    try:
        with observe_attention_launches() as records:
            outputs = llm.generate(ids, SamplingParams(max_tokens=args.output,
                                                       temperature=0.0, ignore_eos=True))
            get_hip_runtime().device_synchronize()
        counts = summarize_attention_launches(records)
        profile = getattr(llm, "_resolved_execution_profile", None)
        sources = {}
        for path in sorted((_ROOT / "hipengine/kernels/hip_gfx1100/gemma4").glob("gemma4_attention*")):
            if path.suffix in (".hip", ".py"):
                sources[str(path.relative_to(_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = {
            "kind": "gemma4_public_attention_route_observation",
            "performance_claim": False,
            "prompt_tokens": len(ids), "requested_output_tokens": args.output,
            "surface": "hipengine.LLM.generate", "device_synchronize_passed": True,
            "execution_profile_manifest": getattr(profile, "manifest", None),
            "manifest_sha256": getattr(profile, "manifest_sha256", None),
            "observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": sources,
            "counts": counts,
            "launches": records,
            "outputs": outputs,
            "provenance": collect_artifact_provenance(
                repo_root=_ROOT, model_path=args.artifact, quant="UD-Q4_K_XL", kv_dtype="bf16"),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=dict) + "\n")
        print(json.dumps({"counts": report["counts"], "manifest_sha256": report["manifest_sha256"]}))
        return 0 if records and outputs else 1
    finally:
        llm.close()


if __name__ == "__main__":
    raise SystemExit(main())
