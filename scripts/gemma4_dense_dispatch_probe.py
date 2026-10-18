"""Record which kernel the dense term's launch_gguf_linear actually resolves.

The Q8_0 T16 prefill module is provably never reached (scripts/gemma4_q8_t16_wave_probe.py
shows zero gate calls and zero launches), yet the dense term is 189.3 ms across 206
launch_gguf_linear calls. So this records the (quant, variant) each call resolves,
which is the input the registry override needs to match.
"""

from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, ".")


def main() -> int:
    from hipengine.runtime import gguf_linear as gl
    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    keys: Counter[str] = Counter()
    orig = gl.launch_gguf_linear

    def spy(weight, *a, **kw):
        quant = getattr(getattr(weight, "spec", None), "quant_key", None)
        variant = kw.get("registered_variant")
        rows = a[2] if len(a) > 2 else kw.get("rows")
        out_f = a[4] if len(a) > 4 else kw.get("out_features")
        keys[f"quant={quant} variant={variant} rows={rows} out={out_f}"] += 1
        return orig(weight, *a, **kw)

    gl.launch_gguf_linear = spy
    llm, runner, info = _resolve_generator(resolve_artifact(), 4096)
    print("resolution:", info["resolution"])

    from hipengine.llm import SamplingParams

    llm.generate_detailed(
        list(range(1000, 1512)),
        SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
    )

    print("\ndistinct (quant, variant) resolved by launch_gguf_linear:")
    grouped: Counter[tuple[str, str]] = Counter()
    for key, count in keys.items():
        quant = key.split(" variant=")[0].split("quant=")[1]
        variant = key.split(" variant=")[1].split(" rows=")[0]
        grouped[(quant, variant)] += count
    for (quant, variant), count in grouped.most_common():
        print(f"  quant={quant:24s} variant={variant} -> {count}")

    print("\ntop call shapes:")
    for key, count in keys.most_common(8):
        print(f"  {key} -> {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
