"""Measure the Gemma 4 MTP draft acceptance against the backbone's own tokens.

The head's forward is verified against llama.cpp's candidate lists elsewhere
(``tests/test_gpu_gemma4_assistant_forward.py``); this measures the number that
matters for a speculative decode: over a real generation, how often is each draft
slot right?

The comparison is greedy on both sides -- the drafts are the head's argmax and the
verdict comes from the backbone's argmax -- which is the same comparison
llama.cpp's ``--temp 0 --top-k 1`` MTP run reports as ``#acc rate/pos``. Its
numbers on this artifact are ``(1.000, 0.750, 0.500, 0.500)`` at
``--spec-draft-n-max 4``.

A slot's rate is over the rounds that *reached* it, so the counts decay with
position exactly as llama.cpp's do: a round stops at its first rejection, and a
draft that never gets compared is not a miss.

    uv run python scripts/gemma4_mtp_acceptance.py --rounds 32 --drafts 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hipengine.loading.gguf import GGUFReader, scan_gguf
from hipengine.loading.gemma4_assistant_device import load_gemma4_assistant_device_weights
from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata
from hipengine.runtime.gemma4 import Gemma4Runner, load_gemma4_device_weights
from hipengine.runtime.gemma4_assistant import Gemma4AssistantHead, Gemma4MtpDrafter

DEFAULT_BACKBONE = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
DEFAULT_HEAD = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
)

# The sentence used for the candidate-list comparison, under Gemma 4's chat
# template. Fixed so two runs of this script are comparable.
PROMPT = [
    2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 818, 4083, 529, 506,
    10995, 23436, 55462, 919, 1082, 496, 13460, 1518, 236764, 6534, 607, 506,
    37813, 529, 506, 3207, 529, 13706, 528, 506, 35186, 7691, 19339, 532, 16548,
    607, 506, 3798, 529, 506, 16425, 38613, 528, 506, 15778, 7691, 7747, 236761,
    799, 1534, 236764, 13706, 13958, 699, 496, 1944, 106, 107, 105, 4368, 107,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--drafts", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=512)
    args = parser.parse_args()

    reader = GGUFReader(str(args.backbone))
    backbone_weights = load_gemma4_device_weights(reader)
    runner = Gemma4Runner(weights=backbone_weights, capacity=args.capacity)
    head_weights = None
    head = None
    try:
        head_weights = load_gemma4_assistant_device_weights(str(args.head))
        head = Gemma4AssistantHead(
            weights=head_weights,
            backbone=gemma4_gguf_config_from_metadata(scan_gguf(args.backbone)),
            backbone_embedding=backbone_weights.embed_tokens,
            backbone_output_norm=backbone_weights.final_norm.buffer,
            capacity=args.capacity,
            eps=1e-6,
        )
        drafter = Gemma4MtpDrafter(head=head, runner=runner, max_drafts=args.drafts)

        started = time.perf_counter()
        token = int(np.argmax(runner.forward(PROMPT, apply_softcap=False)))
        reached = [0] * args.drafts
        matched = [0] * args.drafts
        tokens = [token]
        for _round in range(args.rounds):
            drafts = drafter.draft(token)
            consumed = token
            for slot, draft in enumerate(drafts):
                reached[slot] += 1
                expected = int(np.argmax(runner.forward([consumed], apply_softcap=False)))
                if draft != expected:
                    # The corrected token is what the next round drafts from; the
                    # backbone has not consumed it yet, which is what keeps the
                    # seed row aligned with the token.
                    token = expected
                    tokens.append(expected)
                    break
                matched[slot] += 1
                consumed = draft
                tokens.append(draft)
            else:
                token = int(np.argmax(runner.forward([consumed], apply_softcap=False)))
                tokens.append(token)
        elapsed = time.perf_counter() - started

        rates = [m / r if r else 0.0 for m, r in zip(matched, reached)]
        print(f"prompt tokens   : {len(PROMPT)}")
        print(f"rounds          : {args.rounds}   drafts per round: {args.drafts}")
        print(f"drafts compared : {reached}")
        print(f"drafts accepted : {matched}")
        print(f"accept rate/pos : ({', '.join(f'{r:.3f}' for r in rates)})")
        total_reached = sum(reached)
        total_matched = sum(matched)
        print(
            f"overall         : {total_matched}/{total_reached} = "
            f"{total_matched / total_reached:.3f}"
            if total_reached
            else "overall         : no drafts"
        )
        mean_length = 1.0 + sum(rates)
        print(f"mean acc length : {mean_length:.2f} tokens per round (including the free one)")
        print(f"tokens produced : {len(tokens)}")
        print(f"elapsed         : {elapsed:.1f}s")
        print(f"generated       : {tokens}")
    finally:
        if head is not None:
            head.close()
        if head_weights is not None:
            head_weights.free()
        runner.close()
        backbone_weights.free()


if __name__ == "__main__":
    main()
