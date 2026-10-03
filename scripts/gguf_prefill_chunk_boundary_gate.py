#!/usr/bin/env python3
"""GGUF prefill chunk-boundary numerical gate.

Chunk boundaries change reduction grouping, so changing a prefill chunk size is
an arithmetic change and needs a numerical gate rather than an assumption. This
compares the last-token logits of the same prompt prefilled at different
linear/MoE chunk sizes and reports KL, top-1 agreement and max absolute logit
difference for every pair, **plus a same-config repeat** which is the run-noise
floor on this host. A chunk change is admissible only when its cross-config KL
is far below the measured same-config floor; the historical 512-row decision
recorded KL 0.00013 at 2048 tokens against a 0.034 run-noise floor.

Both arms disable the backend geometry policy and pass the size explicitly, so
the comparison isolates the chunk size. Under the manual-chunk contract that
also leaves ``full_attn_query_chunk_size`` at 0 in every arm, which keeps the
full-attention chunking identical across arms and out of the comparison.

Usage:
    python3 scripts/gguf_prefill_chunk_boundary_gate.py \\
        --chunks 512,1024,4096 --length 4096 --json out.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hipengine.runtime.qwen35_gguf_runner as rm  # noqa: E402
from hipengine.runtime.prefill import PrefillConfig  # noqa: E402

DEFAULT_MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TOKEN_ID = 9707
# docs/EXECUTION-PROFILES.md outer smoke floor for a new arithmetic path.
KL_FLOOR = 0.05
TOP1_FLOOR = 0.90


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(values.max())
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())


def _prefill_logits(model: str, tokens: list[int], chunk: int) -> np.ndarray:
    """Prefill ``tokens`` with an explicit linear/MoE chunk size."""

    with rm.Qwen35GGUFResidentSession(
        model,
        max_sequence_length=len(tokens),
        backend="hip_gfx1151",
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=PrefillConfig(linear_chunk_size=chunk, moe_chunk_size=chunk),
    ) as session:
        result = session.prefill(tokens, use_bulk=True, return_logits=True)
        session.runtime.device_synchronize()
        raw = result.logits if hasattr(result, "logits") else result
        return np.asarray(raw, dtype=np.float32).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument(
        "--chunks",
        default="512,1024,4096",
        help="comma-separated linear/MoE chunk sizes to compare",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="extra same-config repetitions used to estimate the run-noise floor",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    ctypes.CDLL("libamdhip64.so")
    chunks = [int(value) for value in args.chunks.split(",")]
    tokens = [TOKEN_ID] * int(args.length)

    original = rm._gguf_prefill_chunk_sizes_for
    rm._gguf_prefill_chunk_sizes_for = lambda *unused_args, **unused_kwargs: None
    try:
        logits: dict[str, np.ndarray] = {}
        for chunk in chunks:
            logits[f"{chunk}#0"] = _prefill_logits(args.model, tokens, chunk)
            for repeat in range(1, int(args.repeats) + 1):
                logits[f"{chunk}#{repeat}"] = _prefill_logits(args.model, tokens, chunk)
    finally:
        rm._gguf_prefill_chunk_sizes_for = original

    probabilities = {name: _softmax(values) for name, values in logits.items()}

    # Same-config repeats are the run-noise floor; they are the baseline every
    # cross-config pair has to beat.
    noise: list[dict[str, object]] = []
    for chunk in chunks:
        first = f"{chunk}#0"
        for repeat in range(1, int(args.repeats) + 1):
            other = f"{chunk}#{repeat}"
            noise.append(
                {
                    "pair": [first, other],
                    "kl": _kl(probabilities[first], probabilities[other]),
                    "max_abs_logit": float(np.max(np.abs(logits[first] - logits[other]))),
                }
            )
    noise_floor = max((float(entry["kl"]) for entry in noise), default=0.0)

    pairs: list[dict[str, object]] = []
    worst_kl = 0.0
    worst_top1 = 1.0
    for index, left in enumerate(chunks):
        for right in chunks[index + 1 :]:
            a, b = f"{left}#0", f"{right}#0"
            kl = _kl(probabilities[a], probabilities[b])
            top1_a = int(np.argmax(logits[a]))
            top1_b = int(np.argmax(logits[b]))
            entry = {
                "pair": [a, b],
                "kl": kl,
                "top1_a": top1_a,
                "top1_b": top1_b,
                "top1_agree": bool(top1_a == top1_b),
                "max_abs_logit": float(np.max(np.abs(logits[a] - logits[b]))),
                "bit_exact": bool(
                    kl == 0.0 and float(np.max(np.abs(logits[a] - logits[b]))) == 0.0
                ),
                "below_run_noise": bool(kl <= noise_floor),
            }
            pairs.append(entry)
            worst_kl = max(worst_kl, kl)
            worst_top1 = min(worst_top1, 1.0 if top1_a == top1_b else 0.0)

    verdict = {
        "kl_floor": KL_FLOOR,
        "top1_floor": TOP1_FLOOR,
        "worst_kl": worst_kl,
        "worst_top1_agreement": worst_top1,
        "run_noise_floor_kl": noise_floor,
        "passes_smoke_floor": bool(worst_kl <= KL_FLOOR and worst_top1 >= TOP1_FLOOR),
        "all_pairs_bit_exact": all(bool(entry["bit_exact"]) for entry in pairs),
        "all_pairs_within_run_noise": all(bool(entry["below_run_noise"]) for entry in pairs),
    }

    print(f"model={args.model}")
    print(f"length={args.length} chunks={chunks} repeats={args.repeats}")
    print()
    if noise:
        print("same-config repeats (run-noise floor):")
        for entry in noise:
            print(
                f"  {entry['pair'][0]} vs {entry['pair'][1]}: "
                f"KL={float(entry['kl']):.8f} max_abs_logit={float(entry['max_abs_logit']):.8f}"
            )
        print(f"  noise floor KL = {noise_floor:.8f}")
        print()
    print("cross-config pairs:")
    for entry in pairs:
        if entry["bit_exact"]:
            flag = "  BIT-EXACT"
        elif entry["below_run_noise"]:
            flag = "  <= noise floor"
        else:
            flag = "  > NOISE FLOOR"
        print(
            f"  {entry['pair'][0]} vs {entry['pair'][1]}: KL={float(entry['kl']):.8f} "
            f"top1 {entry['top1_a']} vs {entry['top1_b']} "
            f"agree={entry['top1_agree']} max_abs_logit={float(entry['max_abs_logit']):.8f}{flag}"
        )
    print()
    print(
        f"worst KL={worst_kl:.8f} (floor {KL_FLOOR}) "
        f"worst top-1 agreement={worst_top1:.2f} (floor {TOP1_FLOOR}) "
        f"smoke={'PASS' if verdict['passes_smoke_floor'] else 'FAIL'} "
        f"all_bit_exact={verdict['all_pairs_bit_exact']} "
        f"all_within_run_noise={verdict['all_pairs_within_run_noise']}"
    )

    if args.json is not None:
        payload = {
            "model": args.model,
            "length": int(args.length),
            "chunks": chunks,
            "repeats": int(args.repeats),
            "route": {"use_wmma_prefill": True, "use_gemv_decode": True},
            "geometry_policy_disabled": True,
            "noise": noise,
            "pairs": pairs,
            "verdict": verdict,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json}")

    return 0 if verdict["passes_smoke_floor"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
