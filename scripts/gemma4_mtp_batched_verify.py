"""Measure the Gemma 4 MTP cycle with a batched verify against serial greedy.

``scripts/gemma4_mtp_acceptance.py`` measures how often each draft slot is right,
and it verifies one token at a time. A serial verify produces one token per
backbone forward, so it cannot be faster than plain decoding -- it is a
correctness harness.

This script implements the cycle a speculative decode actually needs: the whole
draft is forwarded in **one** call with ``logits_rows=len(drafts)``, which returns
the backbone's own distribution at every drafted position. One forward then
commits up to ``len(drafts) + 1`` tokens. The win is that rows are nearly free --
``Gemma4Runner.forward``'s own docstring guarantees each returned row is "the same
computation a single-token forward would produce for that position", so the
batched verify is arithmetically identical to verifying one at a time.

The gate this script enforces is the one that matters: **the batched cycle must
produce exactly the token sequence plain greedy decoding produces.** Acceptance
rate without that is not evidence of anything.

    uv run python scripts/gemma4_mtp_batched_verify.py --tokens 96 --drafts 4
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

# The chat-templated prompt from the acceptance harness, kept identical so the two
# scripts measure the same distribution.
PROMPT = [
    2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 818, 4083, 529, 506,
    10995, 23436, 55462, 919, 1082, 496, 13460, 1518, 236764, 6534, 607, 506,
    37813, 529, 506, 3207, 529, 13706, 528, 506, 35186, 7691, 19339, 532, 16548,
    607, 506, 3798, 529, 506, 16425, 38613, 528, 506, 15778, 7691, 7747, 236761,
    799, 1534, 236764, 13706, 13958, 699, 496, 1944, 106, 107, 105, 4368, 107,
]


def greedy(runner: Gemma4Runner, prompt: list[int], tokens: int) -> list[int]:
    """Plain greedy decode, one token per forward. The oracle."""

    logits = runner.forward(prompt, apply_softcap=False)
    generated: list[int] = []
    while len(generated) < tokens:
        token = int(np.argmax(logits))
        generated.append(token)
        logits = runner.forward([token], apply_softcap=False)
    return generated


def speculative(
    runner: Gemma4Runner,
    drafter: Gemma4MtpDrafter,
    prompt: list[int],
    tokens: int,
    drafts_per_round: int,
    stats: dict[str, int],
) -> list[int]:
    """The batched-verify cycle.

    State between rounds is ``(logits, token, seed_row)``:

    * ``logits`` predicts ``token``, the next token to commit.
    * ``token`` is committed but not yet processed by the backbone. That is what
      keeps the drafter's seed row aligned with it: ``Gemma4MtpDrafter.draft``
      wants the sampled token *and the hidden row that produced it*.
    * ``seed_row`` indexes the most recent forward's hidden rows.

    The verify forwards ``[token] + drafts`` and reads ``len(drafts) + 1`` rows,
    because ``drafts[0]`` predicts the token after ``token`` rather than ``token``
    itself, and because accepting every draft still needs one more row to know
    the token that follows the last one. Row ``i`` is the logits after the
    forward has processed row ``i``, so ``drafts[i]`` is tested against
    ``rows[i]`` and the token after ``j`` accepted drafts comes from ``rows[j]``.

    ``Gemma4Runner.rewind`` documents the seed contract from the other side: the
    hidden rows of a rewound forward survive precisely so a caller can draft from
    the last accepted position, and a draft must not be taken across a forward.
    """

    logits = runner.forward(prompt, apply_softcap=False)
    token = int(np.argmax(logits))
    generated: list[int] = [token]
    seed_row = -1  # the prefill's last row produced `token`

    while len(generated) < tokens:
        drafter.max_drafts = min(drafts_per_round, tokens - len(generated))
        drafts = drafter.draft(token, hidden_row=seed_row)
        if not drafts:
            break

        position_before = int(runner.position)
        rows = runner.forward(
            [token, *drafts], apply_softcap=False, logits_rows=len(drafts) + 1
        )
        stats["verify_forwards"] += 1
        stats["verify_rows"] += len(drafts) + 1

        accepted = 0
        for index, draft in enumerate(drafts):
            if int(np.argmax(rows[index])) != draft:
                break
            accepted += 1

        # ``rows[accepted]`` is the logits after the last accepted row, so it
        # predicts the token that follows it. ``accepted`` is also the hidden row
        # index that produced that token, since row ``accepted`` is that row.
        token = int(np.argmax(rows[accepted]))
        generated.extend(drafts[:accepted])
        generated.append(token)

        # The verify consumed ``token`` plus every draft; the target keeps
        # ``token`` and the accepted prefix, so the rejected tail is given back.
        runner.rewind(position_before + 1 + accepted)
        logits = rows[accepted]
        seed_row = accepted

        stats["rounds"] += 1
        stats["accepted"] += accepted
        stats["proposed"] += len(drafts)
        stats["tokens_from_rounds"] += accepted + 1

    return generated[:tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--drafts", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=512)
    args = parser.parse_args()

    reader = GGUFReader(str(args.backbone))
    backbone_weights = load_gemma4_device_weights(reader)
    runner = Gemma4Runner(
        weights=backbone_weights,
        capacity=args.capacity,
        max_logits_rows=args.drafts + 1,
    )
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
        expected = greedy(runner, PROMPT, args.tokens)
        greedy_s = time.perf_counter() - started

        runner.reset()
        stats = {
            "rounds": 0,
            "rounds_short_circuited": 0,
            "accepted": 0,
            "proposed": 0,
            "verify_forwards": 0,
            "verify_rows": 0,
            "tokens_from_rounds": 0,
        }
        started = time.perf_counter()
        actual = speculative(runner, drafter, PROMPT, args.tokens, args.drafts, stats)
        spec_s = time.perf_counter() - started

        print(f"prompt tokens      : {len(PROMPT)}")
        print(f"tokens requested   : {args.tokens}   drafts per round: {args.drafts}")
        print(f"exact match        : {actual == expected}")
        if actual != expected:
            for index, (want, got) in enumerate(zip(expected, actual)):
                if want != got:
                    print(f"  first divergence at {index}: greedy={want} spec={got}")
                    break
            print(f"  greedy={expected[:12]}")
            print(f"  spec  ={actual[:12]}")
        rounds = max(1, stats["rounds"])
        proposed = max(1, stats["proposed"])
        print(f"rounds             : {stats['rounds']}"
              f"  (+{stats['rounds_short_circuited']} short-circuited)")
        print(f"drafts proposed    : {stats['proposed']}")
        print(f"drafts accepted    : {stats['accepted']}"
              f"  = {stats['accepted'] / proposed:.3f}")
        print(f"tokens per round   : {stats['tokens_from_rounds'] / rounds:.2f}")
        print(f"verify forwards    : {stats['verify_forwards']}"
              f"  rows={stats['verify_rows']}")
        print(f"greedy             : {greedy_s:.2f}s  "
              f"({args.tokens / greedy_s:.1f} tok/s)")
        print(f"speculative        : {spec_s:.2f}s  "
              f"({args.tokens / spec_s:.1f} tok/s)")
        print(f"speedup            : {greedy_s / spec_s:.2f}x")
        forwards_greedy = args.tokens
        print(f"forwards           : greedy {forwards_greedy}"
              f"  spec {stats['verify_forwards'] + stats['rounds_short_circuited']}"
              f"  = {(forwards_greedy / max(1, stats['verify_forwards'] + stats['rounds_short_circuited'])):.2f}x fewer")
    finally:
        if head is not None:
            head.close()
        if head_weights is not None:
            head_weights.free()
        runner.close()
        backbone_weights.free()


if __name__ == "__main__":
    main()
