"""Validate and measure the window-relative sampler against the full-row one.

Three parts:

* **A** — equivalence on recorded reference rows. The windowed sampler must produce the
  same survivors, the same softmax probabilities and the same token as the full-row
  sampler for both phases, every sampling setting and both arithmetic modes.
* **B** — the host-side sampler cost per step, interleaved, at both row widths. This is the
  part the change is about: mask, penalty, top-k, top-p, softmax and the inverse-CDF draw
  over 32 769 rows instead of 184 704.
* **C** — the production loop, windowed sampler against a control that samples from the
  reconstructed full row, interleaved, with identical tokens and PCG64 state digests.

Run on the W7900 with the YuE2 weights cached:

    python3 scripts/yue2_sampler_window_validation.py --json /tmp/sampler.json
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
sys.path.insert(0, str(REPO))

from hipengine.generation.yue2 import (  # noqa: E402
    VOCAB_SIZE,
    PhaseScores,
    Sampling,
    YuE2Random,
    distribution,
    distribution_windowed,
    phase_window,
    softmax_f32,
)
from hipengine.kernels.cpu_reference.yue2 import bf16_bits_to_f32  # noqa: E402
from hipengine.loading.yue2 import load_yue2_weights  # noqa: E402
from hipengine.runtime import yue2_session as session_module  # noqa: E402
from hipengine.runtime.yue2_ar import Yue2ArRuntime  # noqa: E402
from hipengine.runtime.yue2_session import generate_tokens  # noqa: E402

FIXTURES = REPO / "tests/fixtures/yue2"
CASE = "mandarin-off-s1234"

SETTINGS = {
    "greedy": Sampling(temperature=0.0, top_p=1.0, top_k=VOCAB_SIZE,
                       repetition_penalty=1.0, penalty_window=1, min_tokens=0, max_tokens=48),
    "sampled": Sampling(temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
                        penalty_window=64, min_tokens=2, max_tokens=48),
    "wide_topk": Sampling(temperature=0.8, top_p=0.99, top_k=2048,
                          repetition_penalty=1.05, penalty_window=100, min_tokens=0,
                          max_tokens=48),
}


def _full_row_sampler(values, sampling, history, step, phase, window, legacy_off=False):
    """The pre-change behaviour: sample from the full row, then keep the window slice.

    The head's full row is `-inf` outside the window, which is what this reconstructs, so
    the control runs the same arithmetic the unwindowed session loop used to.
    """

    low, high = window
    row = np.full(VOCAB_SIZE, -np.inf, dtype=np.float32)
    row[low:high] = values
    scores = distribution(row, sampling, history, step, phase, legacy_off=legacy_off)
    return PhaseScores(values=scores[low:high], offset=low)


def _stats(samples):
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "median": float(np.median(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p25": float(np.percentile(ordered, 25)),
        "p75": float(np.percentile(ordered, 75)),
        "samples": [float(value) for value in samples],
    }


def part_a(rows, report):
    """Equivalence over recorded rows: scores, probabilities and drawn tokens."""

    checked = 0
    mismatches = []
    for name, row in rows:
        for phase in ("abc", "semantic"):
            window = phase_window(phase)
            low, high = window
            history = [int(token) for token in np.load(
                FIXTURES / "ar_replay" / name)["tokens"]][:16]
            for label, sampling in SETTINGS.items():
                for legacy_off in (False, True):
                    full = distribution(row, sampling, history, 3, phase,
                                        legacy_off=legacy_off)
                    windowed = distribution_windowed(
                        row[low:high], sampling, history, 3, phase, window,
                        legacy_off=legacy_off,
                    )
                    exact = bool(np.array_equal(full[low:high], windowed.values))
                    same_token = windowed.argmax_token() == int(np.argmax(full))
                    probs_exact = None
                    token_exact = None
                    if sampling.temperature != 0:
                        full_probs = softmax_f32(full)
                        window_probs = softmax_f32(windowed.values)
                        probs_exact = bool(np.array_equal(full_probs[low:high], window_probs))
                        draws = []
                        for _ in range(8):
                            left = YuE2Random(4242)
                            right = YuE2Random(4242)
                            draws.append(
                                left.sample_categorical(full_probs)
                                == right.sample_categorical(window_probs) + low
                            )
                        token_exact = bool(all(draws))
                    if not exact or not same_token or probs_exact is False or token_exact is False:
                        mismatches.append(
                            f"{name}/{phase}/{label}/legacy={legacy_off}: scores={exact} "
                            f"argmax={same_token} probs={probs_exact} draws={token_exact}"
                        )
                    checked += 1
    report["parts"]["equivalence"] = {
        "cases_checked": checked,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    print(f"A equivalence: {checked} cases, {len(mismatches)} mismatches")
    for line in mismatches[:5]:
        print(f"A   {line}")


def part_b(rows, report, repeats):
    """Host sampler cost per step, interleaved, at both row widths."""

    timings = {}
    generator = YuE2Random(99)
    for phase in ("semantic", "abc"):
        window = phase_window(phase)
        low, high = window
        name, row = rows[0]
        history = [int(token) for token in np.load(
            FIXTURES / "ar_replay" / name)["tokens"]][:16]
        sampling = SETTINGS["sampled"]
        full_samples = []
        window_samples = []
        for repeat in range(repeats):
            order = ("full", "window") if repeat % 2 == 0 else ("window", "full")
            for arm in order:
                started = time.perf_counter()
                for _ in range(5):
                    if arm == "full":
                        scores = distribution(row, sampling, history, 3, phase)
                        probabilities = softmax_f32(scores)
                    else:
                        scores = distribution_windowed(
                            row[low:high], sampling, history, 3, phase, window
                        )
                        probabilities = softmax_f32(scores.values)
                    generator.sample_categorical(probabilities)
                elapsed = (time.perf_counter() - started) / 5.0 * 1000.0
                (full_samples if arm == "full" else window_samples).append(elapsed)
        timings[phase] = {
            "window": list(window),
            "rows_full": VOCAB_SIZE,
            "rows_window": high - low,
            "full": _stats(full_samples),
            "windowed": _stats(window_samples),
            "paired_saving_median": float(np.median(
                [full_samples[i] - window_samples[i] for i in range(repeats)]
            )),
            "paired_ratio_median": float(np.median(
                [full_samples[i] / window_samples[i] for i in range(repeats)]
            )),
        }
        print(f"B {phase}: sampler {timings[phase]['full']['median']:.2f} ms full -> "
              f"{timings[phase]['windowed']['median']:.2f} ms windowed "
              f"({timings[phase]['paired_ratio_median']:.2f}x, "
              f"{timings[phase]['paired_saving_median']:.2f} ms saved per step)")
    report["parts"]["sampler_cost"] = timings


def _run_loop(runtime, *, full_sampler, full_head, prefix, negative, sampling, seed):
    original_sampler = session_module.distribution_windowed
    original_window = session_module.phase_window
    if full_sampler:
        session_module.distribution_windowed = _full_row_sampler
    if full_head:
        session_module.phase_window = lambda phase: (0, VOCAB_SIZE)
    try:
        runtime.reset()
        started = time.perf_counter()
        ids, timing, truncated = generate_tokens(
            runtime, prefix, sampling, seed, "semantic",
            negative=negative, cfg_scale=1.01 if negative is not None else 1.0,
        )
        elapsed = time.perf_counter() - started
    finally:
        session_module.distribution_windowed = original_sampler
        session_module.phase_window = original_window
    return {
        "ids": ids,
        "seconds": elapsed,
        "ms_per_token": elapsed / max(1, len(ids)) * 1000.0,
        "truncated": truncated,
        "rng_state": timing["random"],
        "rng_state_digest": timing.get("random_state_digest"),
    }


ARMS = {
    "windowed": (False, False),
    "windowed_head_full_sampler": (True, False),
    "baseline": (True, True),
}


def part_c(runtime, prefix, negative, sampling, report, repeats):
    """The production loop over three interleaved arms.

    ``baseline`` is both changes reverted: the full-vocabulary head and the full-row
    sampler. ``windowed_head_full_sampler`` isolates the sampler change. Every arm runs
    inside one repetition with the order rotated, so the machine state hits all three.
    """

    loop = {"runs": []}
    labels = list(ARMS)
    for label in labels:
        _run_loop(runtime, full_sampler=ARMS[label][0], full_head=ARMS[label][1],
                  prefix=prefix, negative=negative, sampling=sampling, seed=1234)
    for repeat in range(repeats):
        order = [labels[(repeat + offset) % len(labels)] for offset in range(len(labels))]
        for label in order:
            full_sampler, full_head = ARMS[label]
            run = _run_loop(runtime, full_sampler=full_sampler, full_head=full_head,
                            prefix=prefix, negative=negative, sampling=sampling, seed=1234)
            run.update({"repeat": repeat, "arm": label, "tokens": len(run["ids"])})
            loop["runs"].append(run)
            print(f"C {label} repeat {repeat}: {run['tokens']} tokens in "
                  f"{run['seconds']:.3f} s ({run['ms_per_token']:.2f} ms/token)")
    by_arm = {label: [r for r in loop["runs"] if r["arm"] == label] for label in labels}
    for label in labels:
        loop[label] = _stats([r["ms_per_token"] for r in by_arm[label]])
    loop["paired_vs_baseline"] = {
        label: _stats([
            by_arm["baseline"][index]["ms_per_token"] - by_arm[label][index]["ms_per_token"]
            for index in range(repeats)
        ])
        for label in labels if label != "baseline"
    }
    loop["speedup_vs_baseline"] = {
        label: float(np.median([
            by_arm["baseline"][index]["seconds"] / by_arm[label][index]["seconds"]
            for index in range(repeats)
        ]))
        for label in labels if label != "baseline"
    }
    reference = by_arm["windowed"][0]["ids"]
    loop["tokens_identical"] = all(r["ids"] == reference for r in loop["runs"])
    digests = [r["rng_state_digest"] for r in loop["runs"]]
    loop["rng_state_digest_identical"] = None not in digests and len(set(digests)) == 1
    loop["rng_state_digest"] = digests[0]
    for label in labels:
        print(f"C {label}: median {loop[label]['median']:.2f} ms/token "
              f"({loop[label]['min']:.2f}-{loop[label]['max']:.2f})")
    for label, saving in loop["paired_vs_baseline"].items():
        print(f"C {label} vs baseline: {saving['median']:.2f} ms/token saved "
              f"({saving['min']:.2f}-{saving['max']:.2f}), "
              f"{loop['speedup_vs_baseline'][label]:.3f}x")
    print(f"C tokens identical {loop['tokens_identical']}, PCG64 digest identical "
          f"{loop['rng_state_digest_identical']} ({loop['rng_state_digest']})")
    report["parts"]["product_loop"] = loop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--sampler-repeats", type=int, default=15)
    parser.add_argument("--loop-repeats", type=int, default=5)
    parser.add_argument("--skip-loop", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in sorted((FIXTURES / "ar_replay").glob("*.npz")):
        data = np.load(path)
        for key in ("full_logits_0_0", "prefill_logits"):
            if key in data:
                rows.append((path.name, bf16_bits_to_f32(data[key]).reshape(-1)))

    report = {
        "protocol": "yue2-sampler-window-validation-v1",
        "date": time.strftime("%Y-%m-%d"),
        "host": Path("/etc/hostname").read_text().strip(),
        "case": args.case,
        "vocab_size": VOCAB_SIZE,
        "parts": {},
    }

    part_a(rows, report)
    part_b(rows, report, args.sampler_repeats)

    if args.skip_loop:
        report["status"] = "pass" if report["parts"]["equivalence"]["passed"] else "fail"
    else:
        model_dir = os.environ.get("YUE2_MODEL_DIR") or sorted(
            glob.glob(str(Path.home()
                          / ".cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/*"))
        )[0]
        weights = load_yue2_weights(model_dir)
        # The greedy fixture carries the negative prefix, so the loop runs the real
        # two-branch CFG path the production session uses.
        fixture = np.load(FIXTURES / "greedy" / f"{args.case}.npz")
        prefix = [int(token) for token in fixture["prefix"]]
        negative = ([int(token) for token in fixture["negative_prefix"]]
                    if "negative_prefix" in fixture else None)
        sampling = Sampling(temperature=1.0, top_p=0.95, top_k=100,
                            repetition_penalty=1.2, penalty_window=50, min_tokens=0,
                            max_tokens=args.tokens)
        runtime = Yue2ArRuntime(weights, branches=2 if negative is not None else 1,
                                max_context=2056)
        try:
            part_c(runtime, prefix, negative, sampling, report, args.loop_repeats)
        finally:
            runtime.close()
        report["status"] = "pass" if (
            report["parts"]["equivalence"]["passed"]
            and report["parts"]["product_loop"]["tokens_identical"]
            and report["parts"]["product_loop"]["rng_state_digest_identical"]
            and all(
                saving["median"] > 0
                for saving in report["parts"]["product_loop"]["paired_vs_baseline"].values()
            )
        ) else "fail"

    print()
    print(f"status: {report['status']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
