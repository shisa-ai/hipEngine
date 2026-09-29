#!/usr/bin/env python3
"""Is the mmq_total_rows crash reachable the way a user actually calls the engine?

The bulk-prefill crash (``ValueError: mmq_total_rows must be a multiple of 32``)
was found through the gate harness, which forwards ``runner.forward(ids[:N])``
directly. A path reachable only from a harness is not shipped behaviour, so this
drives the public surface instead: ``hipengine.LLM.generate()`` on real prompt
text whose token count is chosen to sit in the predicted crash band.

Prediction from the instrumented probe: a single prefill block of N tokens
builds ``compact_rows = N * 8`` and the MMQ leaf rejects it unless that is a
multiple of 32, i.e. unless N is a multiple of 4. N above ``max_block`` (512)
is safe because it splits into a clean 512 block plus a tail too small to reach
the MMQ path. So the predicted-crash region is ``threshold <= N < 512`` with
``N % 4 != 0``, and ``N % 4 == 0`` in the same region is the predicted-safe
control.

The script adjusts the prompt by appending words until its token count lands on
the requested residue, then reports what actually happened.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT  # noqa: E402

BASE = (
    "The quarterly report covers revenue, operating cost, headcount, and "
    "retention across every region we serve, together with the assumptions "
    "that changed since the previous cycle and the follow-up questions the "
    "board asked us to resolve before the next review. "
)

TARGET_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 501


def build(residue: int, target: int) -> tuple[str, int]:
    """Append filler words until token count == target and target % 4 == residue."""
    import hipengine

    llm = hipengine.LLM(model=str(DEFAULT_ARTIFACT))
    generator = llm._get_text_generator()
    text = BASE
    n = len(generator.tokenize(text))
    filler = " margin detail note value line item segment quarter "
    while n < target:
        text += filler
        n = len(generator.tokenize(text))
    # nudge down toward the exact target by trimming words
    while n > target:
        text = text.rstrip()
        if " " in text:
            text = text[:text.rfind(" ")]
        n = len(generator.tokenize(text))
    return text, n


def main() -> int:
    import hipengine

    print(f"target token count: {TARGET_LEN} (mod 4 = {TARGET_LEN % 4})", flush=True)
    llm = hipengine.LLM(model=str(DEFAULT_ARTIFACT))
    generator = llm._get_text_generator()

    filler = " margin detail note value line item segment quarter "
    text = BASE
    n = len(generator.tokenize(text))
    while n < TARGET_LEN:
        text += filler
        n = len(generator.tokenize(text))
    while n > TARGET_LEN:
        text = text.rstrip()
        if " " in text:
            text = text[: text.rfind(" ")]
        n = len(generator.tokenize(text))

    print(f"prompt token count: {n} (mod 4 = {n % 4})", flush=True)
    # The band that raised before `_mmq_dual_route` learned the 32-row guard.
    # Kept as the pre-fix expectation so a passing run below is meaningful:
    # this probe's value is that the same prompt now succeeds.
    predicted = "CRASH (pre-fix)" if (n < 512 and n % 4 != 0) else "OK"
    print(f"pre-fix prediction: {predicted}", flush=True)

    try:
        out = llm.generate(text)
        print(f"RESULT: generate() SUCCEEDED -> {str(out)[:160]!r}", flush=True)
        return 0
    except ValueError as exc:
        print(f"RESULT: generate() RAISED ValueError: {exc}", flush=True)
        traceback.print_exc(limit=6)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())