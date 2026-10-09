#!/usr/bin/env python3
"""Strict-side router flip-plausibility for the calibration protocol.

The 2026-10-07 packet showed the WMMA arm's KL failures are row-local spikes,
and the amplification evidence puts the mechanism in the sparse router: a row
whose top-8-of-128 selection changes when its router input moves by one bf16
unit in the last place can route differently under any legitimate
reassociation -- the 2026-10-03 identical-inputs capture showed strict and
WMMA are two valid roundings of the same float64 attention output, up to one
ulp apart in each direction. This diagnostic asks that question directly on
the STRICT trajectory only: for every decode row and every layer, perturb the
captured residual by plus and minus one bf16 ulp and test whether either
perturbation changes the top-8 expert set. Candidate behavior never enters.

It also records the top-8 boundary margins (8th minus 9th router score) and
correlates the strict-side flip-plausible set with the per-row KL the packet
recorded, which is the diagnostic's validation: flip-plausible rows should be
where the spikes are.

The router math follows `gemma4_router.py`: weightless RMSNorm (eps 1e-6)
times the learned `ffn_gate_inp.scale` times hidden_size**-0.5, rounded to
BF16, projected by the F32 router weight. It is a margin oracle, not a
routing kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gemma4_production_quality import DEPTHS, padded_chat


def _bf16_round(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def _ulp_step(bf16_bits: np.ndarray) -> np.ndarray:
    """One bf16 ulp in each direction, skipping zeros and the packed sign bit.

    The step is computed on the raw bits the way an increment/decrement of the
    mantissa would move: `+1` on the uint16 for the upward step and `-1` for
    the downward one, which is exactly one representable bf16 neighbor each
    way for every nonzero magnitude.
    """

    up = (bf16_bits.astype(np.uint16).astype(np.uint32) + 1).astype(np.uint16)
    down = (bf16_bits.astype(np.uint16).astype(np.uint32) - 1).astype(np.uint16)
    return up, down


def router_scores(hidden_bf16: np.ndarray, weight: np.ndarray,
                  scale: np.ndarray, hidden_size: int) -> np.ndarray:
    """The router's F32 scores for one BF16 residual row."""

    h = _bf16_to_f32(hidden_bf16).astype(np.float64)
    rms = np.sqrt(np.mean(h * h) + 1e-6 * 1e-6)
    prescaled = (h / rms) * np.asarray(scale, dtype=np.float64) * (hidden_size ** -0.5)
    # The kernel's prescale output is BF16; reproduce that rounding before the
    # F32 projection.
    prescaled = _bf16_to_f32(_bf16_round(prescaled)).astype(np.float64)
    return weight.astype(np.float64) @ prescaled


def top8(scores: np.ndarray) -> np.ndarray:
    return np.sort(np.argsort(scores)[::-1][:8])


def row_kl(baseline_row, candidate_row) -> float:
    b = np.asarray(baseline_row, dtype=np.float64)
    c = np.asarray(candidate_row, dtype=np.float64)
    log_p = b - b.max()
    log_p -= np.log(np.exp(log_p).sum())
    log_q = c - c.max()
    log_q -= np.log(np.exp(log_q).sum())
    return float(np.sum(np.exp(log_p) * (log_p - log_q)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--packet-directory", type=Path, required=True,
                        help="the WMMA arm's packet directory, for per-row KL")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from scripts.gemma4_campaign_bench import _resolve_generator
    from hipengine.loading.gguf import GGUFReader
    from hipengine.runtime.gemma4 import Gemma4Runner

    llm, production, _ = _resolve_generator(args.artifact, max(DEPTHS) + args.rows)
    generator = llm._get_text_generator()
    reader = GGUFReader(str(args.artifact))

    cases = []
    for split, filename in (("canonical", "mtpbench-code-general-ja.jsonl"),
                            ("heldout", "gdn-prefill-category-heldouts.jsonl")):
        for line in (_REPO_ROOT / "benchmarks/prompts" / filename).read_text().splitlines():
            cases.append({**json.loads(line), "split": split})
    by_id = {case["id"]: (index, case) for index, case in enumerate(cases)}

    strict = Gemma4Runner(weights=production.weights, capacity=max(DEPTHS) + args.rows,
                           prefill_attention_variants=("gemma4_plain",))
    report = {"kind": "gemma4_router_flip_plausibility", "performance_claim": False,
              "perturbation": "plus/minus one bf16 ulp on the captured residual, "
                              "per layer, strict trajectory only",
              "cases": {}}
    try:
        for name in args.case:
            index, case = by_id[name]
            depth = DEPTHS[index % len(DEPTHS)]
            message = "\n".join(m["content"] for m in case["messages"])
            ids = padded_chat(generator, message, depth, 20261003 + index)

            strict.reset()
            logits = np.asarray(strict.forward(ids), dtype=np.float32).reshape(-1)
            chain = []
            captures = []
            for step in range(args.rows):
                token = int(np.argmax(logits))
                chain.append(token)
                if step + 1 < args.rows:
                    per_step: list[np.ndarray] = []
                    logits = np.asarray(
                        strict.forward([token], capture_layers=per_step), dtype=np.float32
                    ).reshape(-1)
                    captures.append(per_step)

            num_layers = 30
            flip_plausible = np.zeros(args.rows - 1, dtype=bool)
            margins = np.full(args.rows - 1, np.inf)
            for t, per_step in enumerate(captures):
                row_flips = []
                row_margin = np.inf
                for layer in range(1, num_layers):
                    residual = per_step[layer - 1][0]  # router input of block `layer`
                    weight = reader.tensor_data(f"blk.{layer}.ffn_gate_inp.weight")
                    scale = reader.tensor_data(f"blk.{layer}.ffn_gate_inp.scale")
                    hidden_size = weight.shape[1]
                    base = router_scores(residual, weight, scale, hidden_size)
                    base_set = top8(base)
                    ordered = np.sort(base)[::-1]
                    row_margin = min(row_margin, float(ordered[7] - ordered[8]))
                    up_bits, down_bits = _ulp_step(residual)
                    for perturbed_bits in (up_bits, down_bits):
                        pert = router_scores(perturbed_bits, weight, scale, hidden_size)
                        if not np.array_equal(top8(pert), base_set):
                            row_flips.append(layer)
                            break
                flip_plausible[t] = bool(row_flips)
                margins[t] = row_margin

            kls = np.array([
                row_kl(np.load(args.packet_directory / f"{name}-strict.npy")[i],
                       np.load(args.packet_directory / f"{name}-candidate.npy")[i])
                for i in range(args.rows)
            ])[1:]
            spiked = kls > 0.01
            tp = int((spiked & flip_plausible).sum())
            fp = int((spiked & ~flip_plausible).sum())
            fn = int((~spiked & flip_plausible).sum())
            entry = {
                "rows": int(args.rows - 1),
                "flip_plausible_rows": int(flip_plausible.sum()),
                "spiked_rows_kl_over_0.01": int(spiked.sum()),
                "spike_with_flip_plausible": tp,
                "spike_without_flip_plausible": fp,
                "clean_flip_plausible": fn,
                "min_margin_percentiles": {
                    "p50": float(np.percentile(margins, 50)),
                    "p90": float(np.percentile(margins, 90)),
                    "min": float(margins.min()),
                },
                "flip_plausible_rows_list": np.flatnonzero(flip_plausible).tolist(),
                "spiked_rows_list": np.flatnonzero(spiked).tolist(),
                "max_kl_of_non_flip_plausible": (float(kls[~flip_plausible].max())
                                                 if (~flip_plausible).any() else None),
            }
            report["cases"][name] = entry
            print(f"{name}: flip_plausible={flip_plausible.sum()}/{args.rows-1} "
                  f"spiked={spiked.sum()} tp={tp} fp={fp} fn={fn} "
                  f"max_kl_nonflip={(float(kls[~flip_plausible].max()) if (~flip_plausible).any() else float('nan')):.4g}",
                  flush=True)
    finally:
        strict.close()
        llm.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"artifact={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
