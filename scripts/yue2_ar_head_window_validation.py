"""Validate the AR head's phase window and measure it.

`distribution` masks every row outside a phase's domain and its end token to `-inf`
before any other arithmetic, so those are the only rows the phase can select. The
runtime's `logits(branch, domain=...)` therefore projects just that slice of the 184 704-row
lm head and returns a full-vocabulary row that is `-inf` outside it. This script checks
that claim in four parts:

* A. the windowed row is bit-identical to the full projection's slice, and `-inf`
  everywhere outside, for both phases;
* B. `distribution` over the windowed row and over the full row produce identical score
  rows, identical softmax distributions and the same top-1;
* C. the head call's own time, full against windowed;
* D. the production session loop's tokens and wall clock, windowed against a control that
  forces the full projection.

    python3 scripts/yue2_ar_head_window_validation.py [--json PATH]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    host_array_ptr,
)
from hipengine.generation.yue2 import (  # noqa: E402
    Sampling,
    phase_window,
)
from hipengine.kernels.hip_gfx1100.linear import dense_gemv  # noqa: E402
from hipengine.loading.yue2 import load_yue2_weights  # noqa: E402
from hipengine.runtime import yue2_session as session_module  # noqa: E402
from hipengine.runtime.yue2_ar import (  # noqa: E402
    Yue2ArRuntime,
    bf16_bits_to_f32,
)
from hipengine.runtime.yue2_session import (  # noqa: E402
    combine_cfg,
    distribution,
    generate_tokens,
    softmax_f32,
)

FIXTURES = REPO / "tests/fixtures/yue2/greedy"
CASE = "mandarin-off-s1234"


def _prefill(runtime, prefix, negative):
    runtime.reset()
    runtime.prefill_host_rows([runtime.embed_row(t) for t in prefix], branch=0, start_pos=0)
    if negative is not None:
        runtime.prefill_host_rows([runtime.embed_row(t) for t in negative], branch=1, start_pos=0)


def _timed_head_once(runtime, *, domain) -> float:
    """Milliseconds for one head fill (both branches), single shot."""

    runtime._reset_logits_cache()
    dense_gemv.get_hip_runtime().device_synchronize()
    started = time.perf_counter()
    runtime.logits(0, as_bf16=False, domain=domain)
    dense_gemv.get_hip_runtime().device_synchronize()
    return (time.perf_counter() - started) * 1000.0


def _stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    count = len(ordered)
    return {
        "n": count,
        "median": float(np.median(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p25": float(np.percentile(ordered, 25)),
        "p75": float(np.percentile(ordered, 75)),
        "samples": [float(value) for value in samples],
    }


def _run_loop(runtime, *, full_control: bool, prefix, negative, sampling, seed):
    """One production-loop arm; `full_control` forces the whole vocabulary."""

    original = session_module.phase_window
    if full_control:
        session_module.phase_window = lambda phase: (0, 184704)
    try:
        runtime.reset()
        started = time.perf_counter()
        ids, tok_timing, truncated = generate_tokens(
            runtime, prefix, sampling, seed, "semantic",
            negative=negative, cfg_scale=1.01 if negative is not None else 1.0,
        )
        elapsed = time.perf_counter() - started
    finally:
        session_module.phase_window = original
    return {
        "ids": ids,
        "seconds": elapsed,
        "ms_per_token": elapsed / max(1, len(ids)) * 1000.0,
        "truncated": truncated,
        "rng_state": tok_timing["random"],
        "rng_state_digest": tok_timing.get("random_state_digest"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    parser.add_argument("--fixtures", default=str(FIXTURES))
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--head-repeats", type=int, default=15)
    parser.add_argument("--loop-repeats", type=int, default=5)
    args = parser.parse_args()

    model_dir = os.environ.get("YUE2_MODEL_DIR") or sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/*"))
    )[0]
    weights = load_yue2_weights(model_dir)
    fixture = np.load(Path(args.fixtures) / f"{args.case}.npz")
    prefix = [int(t) for t in fixture["prefix"]]
    negative = [int(t) for t in fixture["negative_prefix"]] if "negative_prefix" in fixture else None

    report = {
        "protocol": "yue2-ar-head-window-validation-v2",
        "date": time.strftime("%Y-%m-%d"),
        "host": Path("/etc/hostname").read_text().strip(),
        "case": args.case,
        "prefix_tokens": len(prefix),
        "negative_tokens": len(negative) if negative is not None else 0,
        "vocab_size": 184704,
        "parts": {},
    }

    runtime = Yue2ArRuntime(weights, branches=2 if negative is not None else 1, max_context=2056)
    try:
        _prefill(runtime, prefix, negative)

        # A. window against full, per phase.
        equivalence = {}
        for phase in ("semantic", "abc"):
            window = phase_window(phase)
            runtime._reset_logits_cache()
            full = runtime.logits(0, as_bf16=False)
            runtime._reset_logits_cache()
            windowed = runtime.logits(0, as_bf16=False, domain=window)
            low, high = window
            inside = np.array_equal(full[low:high], windowed[low:high])
            outside = windowed[:low].size + windowed[high:].size
            outside_inf = bool(
                np.isneginf(windowed[:low]).all() and np.isneginf(windowed[high:]).all()
            )
            equivalence[phase] = {
                "window": [low, high],
                "rows": high - low,
                "fraction_of_vocab": (high - low) / 184704,
                "inside_bit_identical": bool(inside),
                "inside_max_abs": float(np.abs(full[low:high] - windowed[low:high]).max()),
                "outside_rows": int(outside),
                "outside_all_neg_inf": outside_inf,
            }
            print(f"A {phase}: window [{low}, {high}) = {high - low} rows "
                  f"({(high - low) / 184704:.1%}), inside bit-identical {inside}, "
                  f"outside -inf {outside_inf}")
        report["parts"]["coverage"] = {
            "windowed_execution_covered_by": [
                "A: windowed vs full rows, both phases",
                "B: distribution/softmax/top-1 over windowed rows, both phases",
                "D: the production loop with the real phase window",
                "scripts/yue2_session_gate.py: unassisted token agreement",
            ],
            "fallback_covered_by": [
                "scripts/yue2_ar_replay.py: full-vocabulary rows through the unwindowed "
                "route (this validates the fallback and the paired head, not the window)",
            ],
        }
        report["parts"]["window_equivalence"] = equivalence

        # B. The sampler sees the same distribution through either row.
        sampling = Sampling(temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
                            penalty_window=50, min_tokens=0, max_tokens=args.tokens)
        sampler = {}
        for phase in ("semantic", "abc"):
            window = phase_window(phase)
            runtime._reset_logits_cache()
            full0 = runtime.logits(0, as_bf16=False)
            full1 = runtime.logits(1, as_bf16=False) if negative is not None else None
            runtime._reset_logits_cache()
            win0 = runtime.logits(0, as_bf16=False, domain=window)
            win1 = runtime.logits(1, as_bf16=False, domain=window) if negative is not None else None
            scale = 1.01 if negative is not None else 1.0
            full_row = combine_cfg(full0, full1, scale) if full1 is not None else full0
            win_row = combine_cfg(win0, win1, scale) if win1 is not None else win0
            full_scores = distribution(full_row, sampling, [], 0, phase)
            win_scores = distribution(win_row, sampling, [], 0, phase)
            finite = np.isfinite(full_scores) & np.isfinite(win_scores)
            full_probs = softmax_f32(full_scores)
            win_probs = softmax_f32(win_scores)
            sampler[phase] = {
                "scores_exact": bool(np.array_equal(full_scores, win_scores)),
                "scores_max_abs": float(np.abs(full_scores[finite] - win_scores[finite]).max()),
                "mask_identical": bool(
                    np.array_equal(np.isfinite(full_scores), np.isfinite(win_scores))
                ),
                "finite_rows": int(np.isfinite(full_scores).sum()),
                "probs_exact": bool(np.array_equal(full_probs, win_probs)),
                "probs_max_abs": float(np.abs(full_probs - win_probs).max()),
                "top1_agreement": bool(int(full_scores.argmax()) == int(win_scores.argmax())),
                "top1": int(full_scores.argmax()),
            }
            print(f"B {phase}: scores exact {sampler[phase]['scores_exact']}, "
                  f"mask identical {sampler[phase]['mask_identical']}, "
                  f"finite rows {sampler[phase]['finite_rows']}, "
                  f"probs exact {sampler[phase]['probs_exact']}, "
                  f"top1 {sampler[phase]['top1']}")
        report["parts"]["sampler_equivalence"] = sampler

        # C. The head call's own time, interleaved. Both arms are measured inside one
        # repetition and the order flips every repetition, so drift, clock changes and
        # thermal state hit both arms instead of being attributed to whichever ran
        # second. Best-of-N is deliberately not used: it hides spread.
        timing = {}
        for phase in ("semantic", "abc"):
            window = phase_window(phase)
            full_samples: list[float] = []
            window_samples: list[float] = []
            ratios: list[float] = []
            for repeat in range(args.head_repeats):
                windowed_first = repeat % 2 == 1
                if windowed_first:
                    window_ms = _timed_head_once(runtime, domain=window)
                    full_ms = _timed_head_once(runtime, domain=None)
                else:
                    full_ms = _timed_head_once(runtime, domain=None)
                    window_ms = _timed_head_once(runtime, domain=window)
                full_samples.append(full_ms)
                window_samples.append(window_ms)
                ratios.append(full_ms / window_ms)
            timing[phase] = {
                "window": list(window),
                "rows": window[1] - window[0],
                "full": _stats(full_samples),
                "windowed": _stats(window_samples),
                "paired_ratio_median": float(np.median(ratios)),
                "paired_ratio_min": float(min(ratios)),
                "paired_ratio_max": float(max(ratios)),
                "weight_mib_full": 184704 * 2048 * 2 / 2**20,
                "weight_mib_window": (window[1] - window[0]) * 2048 * 2 / 2**20,
                "full_gbs": 184704 * 2048 * 2 / (np.median(full_samples) * 1e-3) / 1e9,
                "windowed_gbs": (window[1] - window[0]) * 2048 * 2
                / (np.median(window_samples) * 1e-3) / 1e9,
            }
            print(f"C {phase}: head {np.median(full_samples):.3f} ms full -> "
                  f"{np.median(window_samples):.3f} ms windowed "
                  f"({timing[phase]['paired_ratio_median']:.2f}x median of "
                  f"{args.head_repeats} interleaved pairs, ratio "
                  f"{timing[phase]['paired_ratio_min']:.2f}-"
                  f"{timing[phase]['paired_ratio_max']:.2f})")
        report["parts"]["head_timing"] = timing

        # D. The production loop, windowed against a full-projection control, interleaved
        # the same way. One run per arm cannot separate a 7% difference from run-to-run
        # variance, so each arm gets a discarded warmup and `--loop-repeats` measured
        # runs, alternating order.
        loop: dict = {"runs": []}
        for label in ("windowed", "full_control"):
            _run_loop(runtime, full_control=label == "full_control", prefix=prefix,
                      negative=negative, sampling=sampling, seed=1234)
        for repeat in range(args.loop_repeats):
            order = ("windowed", "full_control") if repeat % 2 == 0 else (
                "full_control", "windowed")
            for label in order:
                run = _run_loop(runtime, full_control=label == "full_control", prefix=prefix,
                                negative=negative, sampling=sampling, seed=1234)
                run.update({"repeat": repeat, "arm": label, "tokens": len(run["ids"])})
                loop["runs"].append(run)
                print(f"D {label} repeat {repeat}: {run['tokens']} tokens in "
                      f"{run['seconds']:.3f} s ({run['ms_per_token']:.2f} ms/token)")
        by_arm = {
            label: [r for r in loop["runs"] if r["arm"] == label]
            for label in ("windowed", "full_control")
        }
        loop["windowed"] = _stats([r["ms_per_token"] for r in by_arm["windowed"]])
        loop["full_control"] = _stats([r["ms_per_token"] for r in by_arm["full_control"]])
        paired = [
            by_arm["full_control"][index]["ms_per_token"]
            - by_arm["windowed"][index]["ms_per_token"]
            for index in range(args.loop_repeats)
        ]
        loop["paired_ms_per_token_saved"] = _stats(paired)
        loop["speedup_median"] = float(np.median([
            by_arm["full_control"][index]["seconds"] / by_arm["windowed"][index]["seconds"]
            for index in range(args.loop_repeats)
        ]))
        reference = by_arm["windowed"][0]["ids"]
        loop["tokens_identical"] = all(r["ids"] == reference for r in loop["runs"])
        digests = [r["rng_state_digest"] for r in loop["runs"]]
        loop["rng_state_digest_identical"] = (
            None not in digests and len(set(digests)) == 1
        )
        loop["rng_identity_identical"] = all(
            r["rng_state"] == loop["runs"][0]["rng_state"] for r in loop["runs"]
        )
        loop["rng_state_digest"] = digests[0]
        print(f"D windowed median {loop['windowed']['median']:.2f} ms/token "
              f"({loop['windowed']['min']:.2f}-{loop['windowed']['max']:.2f}), "
              f"full control median {loop['full_control']['median']:.2f} ms/token "
              f"({loop['full_control']['min']:.2f}-{loop['full_control']['max']:.2f})")
        print(f"D paired saving median {loop['paired_ms_per_token_saved']['median']:.2f} ms/token "
              f"({loop['paired_ms_per_token_saved']['min']:.2f}-"
              f"{loop['paired_ms_per_token_saved']['max']:.2f}), "
              f"median speedup {loop['speedup_median']:.3f}x")
        print(f"D tokens identical {loop['tokens_identical']}, "
              f"PCG64 state digest identical {loop['rng_state_digest_identical']} "
              f"({loop['rng_state_digest']})")
        report["parts"]["product_loop"] = loop

        ok = (
            all(v["inside_bit_identical"] and v["outside_all_neg_inf"]
                for v in equivalence.values())
            and all(v["scores_exact"] and v["probs_exact"] and v["top1_agreement"]
                    for v in sampler.values())
            and loop["tokens_identical"] and loop["rng_state_digest_identical"]
            and all(v["paired_ratio_median"] > 1.0 for v in timing.values())
            and loop["paired_ms_per_token_saved"]["median"] > 0.0
        )
        report["status"] = "pass" if ok else "fail"
    finally:
        runtime.close()

    print()
    print(f"status: {report['status']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
