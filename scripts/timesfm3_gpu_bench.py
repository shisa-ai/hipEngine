"""TimesFM 3.0 500M GPU decode benchmark and correctness guard.

Modes:
  --metric  print the primary metric: best decode wall-clock seconds for the
            fixed workload (batch=8, 3 variates [2 targets + 1 past-future
            covariate], context=8192, horizon=512, one non-autoregressive
            forward pass).
  --check   run the fixture-parity guard: GPU decode on the committed oracle
            fixtures (base + edge) must match within fp32 atol 1e-4 and the
            fp16 calibrated production gate; exit 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIXTURES = (
    REPO / "tests" / "fixtures" / "cpu_reference" / "timesfm_3p0_decode.npz",
    REPO / "tests" / "fixtures" / "cpu_reference" / "timesfm_3p0_decode_edge.npz",
)
PINNED_MODEL_ID = "google/timesfm-3.0-pytorch"

METRIC_BATCH = 8
METRIC_CONTEXT = 8192
METRIC_HORIZON = 512
METRIC_TARGETS = 2
METRIC_PF_COVARIATES = 1


def _load_decoder(precision: str):
    from hipengine.loading.timesfm3 import load_timesfm3_model
    from hipengine.runtime.timesfm3_decode import TimesFM3GPUDecoder

    loaded = load_timesfm3_model(PINNED_MODEL_ID)
    return loaded, TimesFM3GPUDecoder(loaded, precision=precision)


def _metric_workload():
    rng = np.random.default_rng(20260918)
    t = np.arange(METRIC_CONTEXT, dtype=np.float64)
    target = (
        np.sin(2 * np.pi * t / 48.0) * 3.0
        + 0.5 * rng.standard_normal((METRIC_BATCH, METRIC_TARGETS, METRIC_CONTEXT))
    ).astype(np.float32)
    dow = np.sin(2 * np.pi * np.arange(METRIC_CONTEXT + METRIC_HORIZON) / 7.0)
    past_future = np.broadcast_to(
        dow.astype(np.float32),
        (METRIC_BATCH, METRIC_PF_COVARIATES, METRIC_CONTEXT + METRIC_HORIZON),
    ).copy()
    return target, past_future


def run_metric(repeats: int = 5) -> float:
    target, past_future = _metric_workload()

    loaded, decoder = _load_decoder("fp16")
    try:
        decoder.decode(
            target, METRIC_HORIZON, past_future_covariates=past_future
        )  # warmup + JIT
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            decoder.decode(target, METRIC_HORIZON, past_future_covariates=past_future)
            times.append(time.perf_counter() - t0)
    finally:
        decoder.close()
        loaded.free()
    best = min(times)
    print(
        f"decode_seconds_b{METRIC_BATCH}_v{METRIC_TARGETS + METRIC_PF_COVARIATES}"
        f"_ctx{METRIC_CONTEXT}_h{METRIC_HORIZON}={best:.6f}"
    )
    return best


def _fixture_kwargs(fixture) -> dict:
    kw: dict = {}
    if "past_only_covariates" in fixture.files:
        kw["past_only_covariates"] = fixture["past_only_covariates"]
    if "past_future_covariates" in fixture.files:
        kw["past_future_covariates"] = fixture["past_future_covariates"]
    if "target_mask" in fixture.files:
        kw["target_mask"] = fixture["target_mask"]
    if "global_mask" in fixture.files:
        kw["mask"] = fixture["global_mask"]
    return kw


def _signal_scales(fixture) -> np.ndarray:
    target = fixture["target"]
    mask = fixture["target_mask"] if "target_mask" in fixture.files else None
    scales = []
    for b in range(target.shape[0]):
        for u in range(target.shape[1]):
            m = mask[b, u] if mask is not None else np.zeros(target.shape[2], bool)
            scales.append(float(np.std(target[b, u][~m])))
    return np.array(scales)


def run_check() -> int:
    failures: list[str] = []
    for fixture_path in FIXTURES:
        if not fixture_path.is_file():
            failures.append(f"{fixture_path.name}: fixture missing")
            continue
        fixture = np.load(fixture_path)
        horizon = int(fixture["horizon"])
        kwargs = _fixture_kwargs(fixture)

        loaded, decoder = _load_decoder("fp32")
        try:
            out = decoder.decode(fixture["target"], horizon, **kwargs)
        finally:
            decoder.close()
            loaded.free()
        if not bool(np.isfinite(out).all()):
            failures.append(f"{fixture_path.name} fp32: non-finite output")
        else:
            bad = ~np.isclose(out, fixture["decode_logits"], atol=1.0e-4, rtol=1.0e-2)
            if bad.any():
                failures.append(
                    f"{fixture_path.name} fp32: {int(bad.sum())} strict mismatches "
                    f"(max {np.abs(out - fixture['decode_logits']).max():.2e})"
                )

        loaded, decoder = _load_decoder("fp16")
        try:
            out = decoder.decode(fixture["target"], horizon, **kwargs)
        finally:
            decoder.close()
            loaded.free()
        error = np.abs(out.astype(np.float64) - fixture["decode_logits"].astype(np.float64))
        if not bool(np.isfinite(error).all()):
            failures.append(f"{fixture_path.name} fp16: non-finite output")
            continue
        num_targets = fixture["target"].shape[1]
        scales = _signal_scales(fixture)
        for b in range(fixture["target"].shape[0]):
            series_error = error[b, :num_targets]
            max_rel = float(series_error.max()) / scales[b]
            mean_rel = float(series_error.mean()) / scales[b]
            if max_rel > 0.02 or mean_rel > 0.005:
                failures.append(
                    f"{fixture_path.name} fp16 b{b}: max {100 * max_rel:.2f}% / mean "
                    f"{100 * mean_rel:.3f}% of signal scale exceeds production gate (2% / 0.5%)"
                )

    if failures:
        print("GUARD FAILED:", "; ".join(failures))
        return 1
    print("guard ok: fp32 strict parity + fp16 production gate pass on both fixtures")
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
