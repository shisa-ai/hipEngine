"""Count which Q8_0 T16 prefill wave schedule the dense term actually takes.

Walks nothing and infers nothing: it wraps the two gate functions the Q8_0 T16
prefill wrapper calls, and wraps ``_launch`` to record which symbol is selected.
That is the same technique the expert probe uses -- instrument the consumer --
and it answers the question even if the dispatch key never matches the gfx1151
override.
"""

from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, ".")


def main() -> int:
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_t16_prefill as q8
    from scripts.gemma4_campaign_bench import _resolve_generator, resolve_artifact

    calls: Counter[str] = Counter()
    symbols: Counter[str] = Counter()

    orig_four = q8._four_wave_prefill_applies
    orig_two = q8._two_wave_prefill_applies
    orig_launch = q8._launch

    def four(**kw):
        result = orig_four(**kw)
        calls[f"four_wave_applies -> {result}"] += 1
        return result

    def two(**kw):
        result = orig_two(**kw)
        calls[f"two_wave_applies -> {result}"] += 1
        return result

    def launch(symbol, *a, **kw):
        symbols[str(symbol)] += 1
        return orig_launch(symbol, *a, **kw)

    q8._four_wave_prefill_applies = four
    q8._two_wave_prefill_applies = two
    q8._launch = launch

    llm, runner, info = _resolve_generator(resolve_artifact(), 4096)
    print("resolution:", info["resolution"])

    from hipengine.llm import SamplingParams

    llm.generate_detailed(
        list(range(1000, 1512)),
        SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
    )

    print("\nQ8_0 T16 prefill gate calls:")
    if not calls:
        print("  NONE -- the wrapper was never reached, so the dense term is not")
        print("  dispatching through this module at all.")
    for key, count in calls.most_common():
        print(f"  {key}: {count}")

    print("\nQ8_0 T16 prefill symbols launched:")
    if not symbols:
        print("  NONE")
    for key, count in symbols.most_common():
        print(f"  {key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
