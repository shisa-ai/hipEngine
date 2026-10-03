#!/usr/bin/env python3
"""Does Gemma 4 MTP actually speed up decode through the user-facing API?

`scripts/gemma4_mtp_acceptance.py` measures draft acceptance through the low-level
``Gemma4Runner`` + ``Gemma4MtpDrafter``. Acceptance is not speedup: a drafter that
is right 80% of the time still loses if producing the drafts costs more than the
tokens it saves. And a path reachable only from a harness is not shipped.

This drives the same artifact through ``hipengine.LLM`` both ways -- plain
``generate()`` and ``generate_speculative_mtp_detailed()`` -- and reports tokens
per second for each, plus whether the speculative route is supported at all.

Usage:
    env -u ROCR_VISIBLE_DEVICES PYTHONPATH=. python3 scripts/gemma4_mtp_e2e_speedup.py \
        --tokens 64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACT = (
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
DEFAULT_PROMPT = "Explain in a few sentences why the sky is blue."


def _timed(fn, tokens: int) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    first = out[0] if isinstance(out, list) and out else out
    text = getattr(first, "text", str(first))
    return elapsed, tokens, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    import hipengine
    from hipengine.llm import SamplingParams

    llm = hipengine.LLM(model=str(args.artifact))
    params = SamplingParams(max_tokens=args.tokens, temperature=0.0)

    generator = llm._get_text_generator()
    supports = getattr(generator, "supports_speculative_mtp", None)
    print(f"generator            : {type(generator).__name__}")
    print(f"supports_speculative_mtp: {supports}")
    print(f"speculative_mtp_serving : {llm.speculative_mtp_serving}")
    print(f"max_tokens              : {args.tokens}   repeats: {args.repeats}")
    print()

    plain = []
    for _ in range(args.repeats):
        el, n, text = _timed(lambda: llm.generate([args.prompt], params), args.tokens)
        plain.append(n / el)
    print(f"plain generate()                 : {max(plain):7.2f} tok/s   ({'/'.join(f'{x:.2f}' for x in plain)})")

    try:
        spec = []
        for _ in range(args.repeats):
            el, n, text = _timed(
                lambda: llm.generate_speculative_mtp_detailed([args.prompt], params),
                args.tokens,
            )
            spec.append(n / el)
        print(f"generate_speculative_mtp_detailed: {max(spec):7.2f} tok/s   ({'/'.join(f'{x:.2f}' for x in spec)})")
        print()
        print(f"speedup                          : {max(spec) / max(plain):7.2f}x")
    except NotImplementedError as exc:
        print(f"generate_speculative_mtp_detailed: NOT SUPPORTED -- {exc}")

    print()
    print(f"text sample (plain): {text[:200]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
