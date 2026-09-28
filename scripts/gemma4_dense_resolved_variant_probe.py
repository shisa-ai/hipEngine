"""Record the RESOLVED prefill variant for the dense Q8_0 term.

The earlier probe (scripts/gemma4_dense_dispatch_probe.py) captured the caller's
``variant=None`` argument, which is not what the dispatch picks. This patches the
registry's ``resolve`` so the actual ``(layer, quant, variant)`` is recorded, and
pairs it with ``launch_gguf_linear``'s row count so prefill and decode are
separable.
"""

from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, ".")


def main() -> int:
    from hipengine.kernels import registry
    from hipengine.runtime import gguf_linear as gl
    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    resolved: Counter[tuple[str, str, str]] = Counter()
    launches: Counter[str] = Counter()
    orig_resolve = registry.resolve
    orig_launch = gl.launch_gguf_linear

    def spy_resolve(*a, **kw):
        layer = kw.get("layer")
        quant = kw.get("quant")
        variant = kw.get("variant")
        if a:
            # positional form: (backend, layer, quant, variant, ...)
            if len(a) >= 4:
                layer, quant, variant = a[1], a[2], a[3]
        if quant is not None:
            resolved[(str(layer), str(quant), str(variant))] += 1
        return orig_resolve(*a, **kw)

    def spy_launch(weight, *a, **kw):
        rows = kw.get("rows")
        if rows is None and len(a) > 2:
            rows = a[2]
        quant = getattr(getattr(weight, "spec", None), "quant_key", None)
        if quant is not None:
            launches[f"quant={quant} rows={rows}"] += 1
        return orig_launch(weight, *a, **kw)

    registry.resolve = spy_resolve
    gl.launch_gguf_linear = spy_launch

    llm, runner, info = _resolve_generator(resolve_artifact(), 4096)
    print("resolution:", info["resolution"])
    resolved.clear()

    from hipengine.llm import SamplingParams

    llm.generate_detailed(
        list(range(1000, 1512)),
        SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
    )

    print("\nlaunch_gguf_linear calls by (quant, rows):")
    for key, count in launches.most_common(8):
        print(f"  {key} -> {count}")

    print("\nRESOLVED kernels for the raw q8_0 dense term:")
    hits = [
        (k, c)
        for k, c in resolved.items()
        if k[1] == "gguf_q8_0" and "prefill" in k[2].lower()
    ]
    if not hits:
        print("  no raw q8_0 prefill variant resolved -- printing every q8_0 key:")
        hits = [(k, c) for k, c in resolved.items() if k[1] == "gguf_q8_0"]
    for (layer, quant, variant), count in sorted(hits, key=lambda x: -x[1])[:14]:
        print(f"  layer={layer:10s} quant={quant:14s} variant={variant:44s} -> {count}")

    print("\nall resolved kernels mentioning wmma_prefill (any quant):")
    for (layer, quant, variant), count in sorted(resolved.items(), key=lambda x: -x[1]):
        if "wmma_prefill" in variant:
            print(f"  layer={layer:10s} quant={quant:22s} variant={variant:44s} -> {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
