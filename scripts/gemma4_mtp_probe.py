"""Run Gemma 4 MTP the way a user reaches it: through ``hipengine.LLM``.

The existing ``gemma4_mtp_e2e_speedup.py`` calls ``LLM(model=...)`` with no
``speculative_provider`` and no ``draft_model``, and ``llm.py`` only enters the
provider-resolution block when ``speculative_provider is not None`` -- so that
script can only ever report ``supports_speculative_mtp: False``. This probe
supplies both, then checks the two things the objective asks for:

* does the speculative route run at all, and
* is its output the same tokens plain greedy decoding produces.

Usage::

    HIPENGINE_HIP_ARCH=gfx1151 ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        .venv/bin/python scripts/gemma4_mtp_probe.py --tokens 64 --budget 2
"""

from __future__ import annotations

import argparse
import os
import time

import hipengine
from hipengine.llm import SamplingParams

MODEL_DIR = "/models/gguf/gemma-4-26B-A4B-it-GGUF"
DEFAULT_ARTIFACT = f"{MODEL_DIR}/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
DEFAULT_DRAFT = f"{MODEL_DIR}/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
DEFAULT_PROMPT = "Explain in a few sentences why the sky is blue."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--draft", default=DEFAULT_DRAFT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--provider", default="gemma4_mtp")
    args = ap.parse_args()

    if not os.path.exists(args.draft):
        raise SystemExit(f"assistant sidecar missing: {args.draft}")

    llm = hipengine.LLM(
        model=args.artifact,
        speculative_provider=args.provider,
        draft_model=args.draft,
        speculative_candidate_budget=args.budget,
    )
    generator = llm._get_text_generator()
    supports = getattr(generator, "supports_speculative_mtp", None)
    print(f"provider                : {args.provider}")
    print(f"draft_model             : {args.draft}")
    print(f"candidate_budget        : {args.budget}")
    print(f"generator               : {type(generator).__name__}")
    print(f"supports_speculative_mtp: {supports}")
    print(f"speculative_mtp_serving : {llm.speculative_mtp_serving}")

    params = SamplingParams(max_tokens=args.tokens, temperature=0.0)

    # Both routes are warmed before either is timed. The first generate() on a
    # fresh process pays the kernel JIT and the first-touch of every weight, and
    # that cost lands entirely on whichever route runs first -- measuring a cold
    # plain call against a warm speculative one reports the JIT, not the speedup.
    print("warming both routes...")
    llm.generate([args.prompt], SamplingParams(max_tokens=8, temperature=0.0))
    try:
        llm.generate_speculative_mtp_detailed(
            [args.prompt], SamplingParams(max_tokens=8, temperature=0.0)
        )
    except Exception as exc:  # noqa: BLE001 - report the refusal verbatim
        print(f"generate_speculative_mtp_detailed: NOT SUPPORTED -- {exc}")
        return 1

    started = time.perf_counter()
    plain = llm.generate([args.prompt], params)
    plain_s = time.perf_counter() - started
    plain_text = plain[0].text if hasattr(plain[0], "text") else str(plain[0])
    print(f"plain generate()        : {args.tokens / plain_s:7.2f} tok/s  ({plain_s:.2f}s)")

    try:
        started = time.perf_counter()
        spec = llm.generate_speculative_mtp_detailed([args.prompt], params)
        spec_s = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - report the refusal verbatim
        print(f"generate_speculative_mtp_detailed: NOT SUPPORTED -- {exc}")
        return 1

    spec_text = spec[0].text if hasattr(spec[0], "text") else str(spec[0])
    print(f"generate_speculative_mtp_detailed: {args.tokens / spec_s:7.2f} tok/s  ({spec_s:.2f}s)")
    plain_tokens = getattr(plain[0], "token_ids", None)
    spec_tokens = getattr(spec[0], "token_ids", None)
    if plain_tokens is not None and spec_tokens is not None:
        print(
            f"tokens emitted          : plain {len(plain_tokens)}, "
            f"spec {len(spec_tokens)}  (max_tokens={args.tokens})"
        )
    print(f"speedup                 : {plain_s / spec_s:7.3f}x")
    print(f"outputs identical       : {plain_text == spec_text}")
    if plain_text != spec_text:
        # A commit cycle ends on a whole group, so hitting max_tokens mid-cycle
        # stops the speculative route at a different point than plain AR. That is
        # a token-count artefact, not a wrong token -- so test the property that
        # separates them: the shorter output must be an exact prefix of the
        # longer, with every shared character equal.
        n = min(len(plain_text), len(spec_text))
        longer = "spec" if len(spec_text) > len(plain_text) else "plain"
        print(
            f"  lengths               : plain {len(plain_text)}, spec {len(spec_text)} chars"
            f"  (longer: {longer})"
        )
        print(f"  common prefix equal   : {plain_text[:n] == spec_text[:n]}")
        if plain_text[:n] != spec_text[:n]:
            for i in range(n):
                if plain_text[i] != spec_text[i]:
                    print(f"  first divergence at char {i}: {plain_text[i]!r} vs {spec_text[i]!r}")
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
