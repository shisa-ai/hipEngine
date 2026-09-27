#!/usr/bin/env python3
"""Gemma 4 MMQ expert gate/up: divergence against prefill length.

The fused int8-dp4a MMQ32 gate/up route replaces two prefill projections per
layer, so a single forward of N ids under each arm is a complete reproduction of
whatever a longer teacher-forced chain shows: the prefill writes the KV cache and
every later row inherits it.

This probe exists because the campaign's recorded evidence for the route was
taken at one prefill length. ``scripts/gemma4_teacher_forced_gate.py`` scores
``--prompt 2048 --prefill 1024``, and at 1024 ids the route is nearly exact
(7.3e-07 KL against the fp32 grouped owner), which is the best case in the sweep
below rather than a representative one. At 16 to 256 ids the same route is off by
up to 2.0 KL with a greedy decision flip, deterministically.

Run it on the tree that carries the route::

    env -u HIP_VISIBLE_DEVICES PYTHONPATH=. .venv/bin/python \\
      scripts/gemma4_mmq_prefill_length_probe.py --artifact <model.gguf>

Exit status is 0 when the route is within the binding ``kl_max`` bar at every
length and 1 otherwise, so the probe can gate a future fix. It is a diagnostic:
it adds no performance row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Binding production limit, docs/EXECUTION-PROFILES.md.
KL_MAX_BAR = 5e-2

# The length that the campaign's recorded evidence was taken at.
RECORDED_LENGTH = 1024

DEFAULT_LENGTHS: tuple[int, ...] = (
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT, DEFAULT_CONTEXT

    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--lengths", type=int, nargs="*", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--repeats", type=int, default=3,
                        help="arms per length, to separate a defect from a race")
    parser.add_argument("--corpus", choices=("frozen", "probe"), default="frozen")
    parser.add_argument("--out", type=Path, help="JSON record to write")
    args = parser.parse_args(argv)

    from scripts.gemma4_campaign_bench import (
        PROBE_CORPUS_SEED, exact_prompt_ids, probe_corpus, _resolve_generator,
    )
    from scripts.gemma4_teacher_forced_gate import row_kl_divergence
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts

    llm, runner, _ = _resolve_generator(args.artifact, args.context)
    tokenize = llm._get_text_generator().tokenize
    longest = max(int(n) for n in args.lengths)
    if args.corpus == "probe":
        ids = exact_prompt_ids(tokenize, longest, corpus=probe_corpus(seed=PROBE_CORPUS_SEED),
                               require_single_pass=True)
    else:
        ids = exact_prompt_ids(tokenize, longest)

    def arm(length: int, mmq: bool, repeats: int) -> list[np.ndarray]:
        os.environ.pop("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", None)
        if mmq:
            os.environ["HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ"] = "1"
        outs: list[np.ndarray] = []
        for _ in range(repeats):
            runner.reset()
            outs.append(np.array(runner.forward(list(ids[:length])),
                                 dtype=np.float32).reshape(-1))
        return outs

    rows: list[dict[str, object]] = []
    print(f"{'ids':>5s} {'kl':>11s} {'flip':>5s} {'deterministic':>14s}  route")
    for length in args.lengths:
        length = int(length)
        os.environ.pop("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", None)
        before = dict(experts.gemma4_moe_expert_route_counts())
        base = arm(length, False, 1)[0]
        repeats = arm(length, True, max(1, int(args.repeats)))
        after = experts.gemma4_moe_expert_route_counts()
        route = {key: value - before.get(key, 0) for key, value in after.items()
                 if "mmq" in key and value - before.get(key, 0)}
        kl = row_kl_divergence(base, repeats[0])
        flip = bool(np.argmax(base) != np.argmax(repeats[0]))
        deterministic = all(np.array_equal(repeats[0], other) for other in repeats[1:])
        # A two-arm comparison can say the arms disagree but not which one is
        # wrong: at 96 ids it reports exact agreement even if both arms are
        # wrong the same way, and everywhere else it names the MMQ route without
        # having established that the fp32 route is the correct one. Record the
        # greedy token each arm picks so a divergence can be attributed against
        # an independent reference (the model's known-good continuation, the
        # teacher-forced baseline, or the CPU streaming reference) instead of
        # being assigned to whichever arm happens to be the second one run.
        base_argmax = int(np.argmax(base))
        mmq_argmax = int(np.argmax(repeats[0]))
        rows.append({
            "base_argmax": base_argmax,
            "mmq_argmax": mmq_argmax,
            "ids": length,
            "kl": kl,
            "over_bar": kl > KL_MAX_BAR,
            "top1_flip": flip,
            "deterministic": deterministic,
            "mmq_route_launches": route,
            "route_used": bool(route),
        })
        print(f"{length:5d} {kl:11.3e} {int(flip):5d} {str(deterministic):>14s}  "
              f"{route}  base->{base_argmax} mmq->{mmq_argmax}")

    used = [row for row in rows if row["route_used"]]
    over = [row for row in used if row["over_bar"]]
    record = {
        "kind": "gemma4_mmq_gate_up_prefill_length_probe",
        "performance_claim": False,
        "corpus": args.corpus,
        "artifact": str(args.artifact),
        "kl_max_bar": KL_MAX_BAR,
        "recorded_evidence_length": RECORDED_LENGTH,
        "lengths_with_route": len(used),
        "lengths_over_bar": len(over),
        "over_bar_lengths": [row["ids"] for row in over],
        "deterministic": all(row["deterministic"] for row in rows),
        "kl_at_recorded_length": next(
            (row["kl"] for row in rows if row["ids"] == RECORDED_LENGTH), None
        ),
        "kl_max_observed": max((row["kl"] for row in used), default=0.0),
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"\nroute used at {len(used)} lengths; over the {KL_MAX_BAR} bar at "
          f"{len(over)}: {[row['ids'] for row in over]}")
    print(f"kl at the recorded evidence length {RECORDED_LENGTH}: "
          f"{record['kl_at_recorded_length']}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
