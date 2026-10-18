#!/usr/bin/env python3
"""A/B the two YuE2 decode attention routes: numerics and step time.

``attention_route="splitk"`` runs the production split-K span kernel and
``"strict"`` runs its parent. The kernel reassociates the softmax, so it needs both a
numerical bound against the parent and a measured step-time win before it can be the
default; this script produces both from one process.

    python3 scripts/yue2_ar_attention_splitk.py --depths 0 900 --steps 60
    python3 scripts/yue2_ar_attention_splitk.py --numerics-only

The numerical arm drives both routes from the same prefill and the same recorded
trajectory, then reports the per-step KL between the combined CFG rows, the top-1
agreement, and the largest absolute difference in either branch's hidden row. The
timing arm alternates the two routes inside one repetition so device state is shared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_CASE = "mandarin-off-s1234"


def _model_dir(explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    value = os.environ.get("YUE2_MODEL_DIR")
    if value:
        return Path(value)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    matches = sorted(cache.glob("models--m-a-p--YuE2-3B/snapshots/*"))
    if not matches:
        raise SystemExit("no YuE2-3B checkpoint found; set YUE2_MODEL_DIR")
    return matches[-1]


def _kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    """KL(reference || candidate) over the semantic window, both as log-softmax."""

    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    finite = np.isfinite(ref) & np.isfinite(cand)
    ref = ref[finite]
    cand = cand[finite]
    ref = ref - ref.max()
    cand = cand - cand.max()
    p = np.exp(ref - np.log(np.exp(ref).sum()))
    q = np.exp(cand - np.log(np.exp(cand).sum()))
    return float(np.sum(p * (np.log(p) - np.log(np.maximum(q, 1e-300)))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--context", type=int, default=16000)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 900])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numerics-steps", type=int, default=24)
    parser.add_argument("--numerics-only", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from hipengine.generation.yue2 import (
        CODEC_OFFSET,
        Sampling,
        SongRequest,
        YuE2Random,
        combine_cfg,
        distribution_windowed,
        negative_prefix,
        phase_window,
        softmax_f32,
    )
    from hipengine.loading.yue2 import load_yue2_weights
    from hipengine.runtime.yue2_ar import Yue2ArRuntime
    from hipengine.runtime.yue2_session import bf16_bits_to_f32
    from hipengine.tokenization.yue2 import YuE2TextTokenizer

    model_dir = _model_dir(args.model_dir)
    weights = load_yue2_weights(model_dir)
    tokenizer = YuE2TextTokenizer(Path(model_dir) / "qwen.tiktoken")
    runtime = Yue2ArRuntime(weights, branches=2, max_context=args.context)
    routes = {"strict": Yue2ArRuntime(weights, branches=2, max_context=args.context,
                                      attention_route="strict")}
    routes["splitk"] = runtime
    print(json.dumps({"resolved_routes": {name: r.attention_route for name, r in routes.items()}}),
          flush=True)
    if runtime.attention_route != "splitk":
        raise SystemExit("the split-K route did not resolve on this backend")

    case = np.load(REPO / "tests/fixtures/yue2/cases" / f"{args.case}.npz")
    prefix = [int(value) for value in case["prefix"]]
    trajectory = [int(value) + CODEC_OFFSET for value in case["semantic"]]
    song = SongRequest(
        style="warm acoustic pop",
        lyrics="[verse]\nMorning light across the floor",
        cot="off",
        seed=1234,
        id="attention-splitk",
    )
    negative = [int(value) for value in negative_prefix(song, tokenizer.encode, ())]
    window = phase_window("semantic")
    low, high = window

    def warm(rt, depth: int) -> None:
        rt.reset()
        rt.prefill_host_rows([rt.embed_row(t) for t in prefix], branch=0, start_pos=0)
        rt.prefill_host_rows([rt.embed_row(t) for t in negative], branch=1, start_pos=0)
        for step in range(depth):
            token = trajectory[step % len(trajectory)]
            row = rt.embed_row(token)
            p0 = rt.context_length(0)
            p1 = rt.context_length(1)
            rt.push_token(row, p0, branch=0)
            rt.push_token(row, p1, branch=1)
            rt.forward_layers_paired(p0, p1)
        rt.runtime.device_synchronize()

    def rows(rt, depth: int, steps: int):
        """Logits rows and hidden rows for ``steps`` recorded tokens from ``depth``."""

        warm(rt, depth)
        collected = []
        for step in range(depth, depth + steps):
            token = trajectory[step % len(trajectory)]
            combined = combine_cfg(
                bf16_bits_to_f32(rt.logits(0, domain=window)[low:high]),
                bf16_bits_to_f32(rt.logits(1, domain=window)[low:high]),
                song.guidance,
            )
            collected.append((combined.copy(), rt.hidden_state(0).copy(), rt.hidden_state(1).copy()))
            row = rt.embed_row(token)
            p0 = rt.context_length(0)
            p1 = rt.context_length(1)
            rt.push_token(row, p0, branch=0)
            rt.push_token(row, p1, branch=1)
            rt.forward_layers_paired(p0, p1)
        rt.runtime.device_synchronize()
        return collected

    payload: dict = {"case": args.case, "resolved_routes": {k: v.attention_route for k, v in routes.items()}}

    if not args.timing_only:
        numerics = []
        for depth in args.depths:
            parent = rows(routes["strict"], depth, args.numerics_steps)
            candidate = rows(routes["splitk"], depth, args.numerics_steps)
            kls, top1, hidden, hidden_rel = [], 0, 0.0, 0.0
            for (ref_row, ref_h0, ref_h1), (cand_row, cand_h0, cand_h1) in zip(parent, candidate):
                kls.append(_kl(ref_row, cand_row))
                top1 += int(np.argmax(ref_row) == np.argmax(cand_row))
                for ref_h, cand_h in ((ref_h0, cand_h0), (ref_h1, cand_h1)):
                    ref64 = ref_h.astype(np.float64)
                    cand64 = cand_h.astype(np.float64)
                    hidden = max(hidden, float(np.max(np.abs(ref64 - cand64))))
                    scale = float(np.max(np.abs(ref64)))
                    if scale > 0:
                        hidden_rel = max(hidden_rel, float(np.max(np.abs(ref64 - cand64))) / scale)
            record = {
                "depth": depth,
                "context": len(prefix) + depth,
                "steps": args.numerics_steps,
                "mean_kl": float(np.mean(kls)),
                "max_kl": float(np.max(kls)),
                "top1_agreement": top1 / args.numerics_steps,
                "max_hidden_abs_delta": hidden,
                "max_hidden_rel_delta": hidden_rel,
            }
            numerics.append(record)
            print(json.dumps(record), flush=True)
        payload["numerics"] = numerics

    if not args.numerics_only:
        timing = []
        sampling = Sampling(
            temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
            penalty_window=50, min_tokens=0, max_tokens=args.steps,
        )
        for depth in args.depths:
            arms = {"strict": [], "splitk": []}
            for repeat in range(args.repeats):
                order = ["strict", "splitk"] if repeat % 2 == 0 else ["splitk", "strict"]
                for name in order:
                    rt = routes[name]
                    warm(rt, depth)
                    generator = YuE2Random(1234)
                    history: list[int] = []
                    rt.runtime.device_synchronize()
                    started = time.perf_counter()
                    for step in range(args.steps):
                        combined = combine_cfg(
                            bf16_bits_to_f32(rt.logits(0, domain=window)[low:high]),
                            bf16_bits_to_f32(rt.logits(1, domain=window)[low:high]),
                            song.guidance,
                        )
                        scores = distribution_windowed(
                            combined, sampling, history, step, "semantic", window
                        )
                        token = scores.offset + generator.sample_categorical(softmax_f32(scores.values))
                        history.append(token)
                        row = rt.embed_row(token)
                        p0 = rt.context_length(0)
                        p1 = rt.context_length(1)
                        rt.push_token(row, p0, branch=0)
                        rt.push_token(row, p1, branch=1)
                        rt.forward_layers_paired(p0, p1)
                    rt.runtime.device_synchronize()
                    arms[name].append((time.perf_counter() - started) / args.steps * 1000.0)
            record = {
                "depth": depth,
                "context": len(prefix) + depth,
                "steps": args.steps,
                "strict_ms": float(np.median(arms["strict"])),
                "splitk_ms": float(np.median(arms["splitk"])),
                "speedup": float(np.median(arms["strict"]) / np.median(arms["splitk"])),
                "repeats": args.repeats,
            }
            timing.append(record)
            print(json.dumps(record), flush=True)
        payload["timing"] = timing

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
