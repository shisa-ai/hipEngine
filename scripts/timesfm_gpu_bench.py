"""TimesFM 2.5 200M GPU decode benchmark and correctness guard.

Modes:
  --metric  print the primary metric: best decode wall-clock seconds for the
            fixed workload (batch=8, context=8192, horizon=512).
  --check   run the fixture-parity guard: GPU decode on the committed oracle
            fixture must match within atol 5e-4 / rtol 1e-2; exit 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "cpu_reference" / "timesfm_2p5_200m_decode.npz"
PINNED_MODEL_ID = "google/timesfm-2.5-200m-pytorch"

METRIC_BATCH = 8
METRIC_CONTEXT = 8192
METRIC_HORIZON = 512


def _load_decoder():
    from hipengine.loading.timesfm import load_timesfm_model
    from hipengine.runtime.timesfm_decode import TimesFMGPUDecoder

    loaded = load_timesfm_model(PINNED_MODEL_ID)
    return loaded, TimesFMGPUDecoder(loaded)


def run_metric(repeats: int = 5) -> float:
    rng = np.random.default_rng(20260917)
    t = np.arange(METRIC_CONTEXT, dtype=np.float64)
    inputs = (
        np.sin(2 * np.pi * t / 48.0) * 3.0
        + 0.5 * rng.standard_normal((METRIC_BATCH, METRIC_CONTEXT))
    ).astype(np.float32)
    masks = np.zeros((METRIC_BATCH, METRIC_CONTEXT), dtype=bool)

    loaded, decoder = _load_decoder()
    try:
        decoder.decode(METRIC_HORIZON, inputs, masks)  # warmup + JIT
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            decoder.decode(METRIC_HORIZON, inputs, masks)
            times.append(time.perf_counter() - t0)
    finally:
        decoder.close()
        loaded.free()
    best = min(times)
    print(f"decode_seconds_b{METRIC_BATCH}_ctx{METRIC_CONTEXT}_h{METRIC_HORIZON}={best:.6f}")
    return best


def run_check() -> int:
    from hipengine.runtime.timesfm_decode import TimesFMGPUDecoder
    from hipengine.loading.timesfm import load_timesfm_model

    fixture = np.load(FIXTURE)
    inputs, masks, horizon = fixture["inputs"], fixture["masks"], int(fixture["horizon"])
    loaded = load_timesfm_model(PINNED_MODEL_ID)
    failures: list[str] = []
    try:
        # Strict FP32 fallback: near-exact parity with the FP32 torch oracle.
        decoder = TimesFMGPUDecoder(loaded, precision="fp32")
        try:
            pf, qs, ar = decoder.decode(horizon, inputs, masks)
        finally:
            decoder.close()
        for name, actual, expected in (
            ("renormed_outputs", pf, fixture["renormed_outputs"]),
            ("quantile_spread", qs, fixture["quantile_spread"]),
            ("ar_outputs", ar, fixture["ar_outputs"]),
        ):
            if actual is None or expected is None:
                if actual is not expected:
                    failures.append(f"fp32 {name}: presence mismatch")
                continue
            if not bool(np.isfinite(actual).all()):
                failures.append(f"fp32 {name}: non-finite output")
                continue
            bad = ~np.isclose(actual, expected, atol=5.0e-4, rtol=1.0e-2)
            if bad.any():
                failures.append(f"fp32 {name}: {int(bad.sum())} strict mismatches")

        # Production FP16 path: calibrated forecasting tolerance vs the oracle.
        decoder = TimesFMGPUDecoder(loaded, precision="fp16")
        try:
            pf, qs, ar = decoder.decode(horizon, inputs, masks)
        finally:
            decoder.close()
        for name, actual, expected in (
            ("renormed_outputs", pf, fixture["renormed_outputs"]),
            ("quantile_spread", qs, fixture["quantile_spread"]),
            ("ar_outputs", ar, fixture["ar_outputs"]),
        ):
            if actual is None or expected is None:
                if actual is not expected:
                    failures.append(f"fp16 {name}: presence mismatch")
                continue
            if actual.shape != expected.shape:
                failures.append(f"fp16 {name}: shape {actual.shape} != {expected.shape}")
                continue
            error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            if not bool(np.isfinite(error).all()):
                failures.append(f"fp16 {name}: non-finite output")
                continue
            for b in range(expected.shape[0]):
                scale = float(np.std(inputs[b][~masks[b]]))
                series_error = error[b]
                max_rel = float(series_error.max()) / scale
                mean_rel = float(series_error.mean()) / scale
                if max_rel > 0.02 or mean_rel > 0.005:
                    failures.append(
                        f"fp16 {name} b{b}: max {100 * max_rel:.2f}% / mean {100 * mean_rel:.3f}% "
                        "of signal scale exceeds production gate (2% / 0.5%)"
                    )
    finally:
        loaded.free()

    if failures:
        print("GUARD FAILED:", "; ".join(failures))
        return 1
    print("guard ok: fp32 strict parity + fp16 production gate both pass")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--metric", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.metric:
        run_metric(repeats=args.repeats)
        return
    sys.exit(run_check())


if __name__ == "__main__":
    main()
