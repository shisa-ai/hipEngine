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


def _timed_head(runtime, *, domain, repeats: int = 10) -> float:
    """Milliseconds for one head fill (both branches), best of `repeats`."""

    best = None
    for _ in range(repeats):
        runtime._reset_logits_cache()
        dense_gemv.get_hip_runtime().device_synchronize()
        started = time.perf_counter()
        runtime.logits(0, as_bf16=False, domain=domain)
        dense_gemv.get_hip_runtime().device_synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        best = elapsed if best is None else min(best, elapsed)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    parser.add_argument("--fixtures", default=str(FIXTURES))
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--tokens", type=int, default=48)
    args = parser.parse_args()

    model_dir = os.environ.get("YUE2_MODEL_DIR") or sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/*"))
    )[0]
    weights = load_yue2_weights(model_dir)
    fixture = np.load(Path(args.fixtures) / f"{args.case}.npz")
    prefix = [int(t) for t in fixture["prefix"]]
    negative = [int(t) for t in fixture["negative_prefix"]] if "negative_prefix" in fixture else None

    report = {
        "protocol": "yue2-ar-head-window-validation-v1",
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

        # C. The head call's own time.
        timing = {}
        for phase in ("semantic", "abc"):
            window = phase_window(phase)
            full_ms = _timed_head(runtime, domain=None)
            window_ms = _timed_head(runtime, domain=window)
            timing[phase] = {
                "window": list(window),
                "rows": window[1] - window[0],
                "full_ms": full_ms,
                "windowed_ms": window_ms,
                "speedup": full_ms / window_ms,
                "weight_mib_full": 184704 * 2048 * 2 / 2**20,
                "weight_mib_window": (window[1] - window[0]) * 2048 * 2 / 2**20,
                "full_gbs": 184704 * 2048 * 2 / (full_ms * 1e-3) / 1e9,
                "windowed_gbs": (window[1] - window[0]) * 2048 * 2 / (window_ms * 1e-3) / 1e9,
            }
            print(f"C {phase}: head {full_ms:.3f} ms full -> {window_ms:.3f} ms windowed "
                  f"({full_ms / window_ms:.2f}x, {timing[phase]['windowed_gbs']:.0f} GB/s)")
        report["parts"]["head_timing"] = timing

        # D. The production loop, windowed against a full-projection control.
        loop = {}
        for label, patched in (("windowed", False), ("full_control", True)):
            original = session_module.phase_window
            if patched:
                session_module.phase_window = lambda phase: (0, 184704)
            try:
                runtime.reset()
                started = time.perf_counter()
                ids, tok_timing, truncated = generate_tokens(
                    runtime, prefix, sampling, 1234, "semantic",
                    negative=negative, cfg_scale=1.01 if negative is not None else 1.0,
                )
                elapsed = time.perf_counter() - started
            finally:
                session_module.phase_window = original
            loop[label] = {
                "ids": ids,
                "seconds": elapsed,
                "ms_per_token": elapsed / max(1, len(ids)) * 1000.0,
                "truncated": truncated,
                "rng_state": tok_timing["random"],
            }
            print(f"D {label}: {len(ids)} tokens in {elapsed:.3f} s "
                  f"({loop[label]['ms_per_token']:.2f} ms/token)")
        loop["tokens_identical"] = loop["windowed"]["ids"] == loop["full_control"]["ids"]
        loop["rng_identical"] = loop["windowed"]["rng_state"] == loop["full_control"]["rng_state"]
        loop["speedup"] = loop["full_control"]["seconds"] / loop["windowed"]["seconds"]
        loop["ms_per_token_saved"] = (
            loop["full_control"]["ms_per_token"] - loop["windowed"]["ms_per_token"]
        )
        print(f"D tokens identical {loop['tokens_identical']}, rng identical {loop['rng_identical']}, "
              f"{loop['speedup']:.3f}x, {loop['ms_per_token_saved']:.2f} ms/token saved")
        report["parts"]["product_loop"] = loop

        ok = (
            all(v["inside_bit_identical"] and v["outside_all_neg_inf"]
                for v in equivalence.values())
            and all(v["scores_exact"] and v["probs_exact"] and v["top1_agreement"]
                    for v in sampler.values())
            and loop["tokens_identical"] and loop["rng_identical"]
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
