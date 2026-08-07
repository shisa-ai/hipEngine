"""C8 phase-2 static-batch encoder throughput benchmark: real req/s + speedup.

Two workload tables, for B=1/2/4/8 on homogeneous fixed-length audio:

Table A — encoder-only, largest certified bucket (480,000 samples, 30 s ->
1,248 encoder frames), synthetic deterministic audio:
  - c=1 baseline: B independent warm ``MoonshineCudaEncoderRuntime`` sessions
    encoded sequentially on the same audio (full ``encode()``: H2D + DAG).
  - batch: one ``MoonshineCudaBatchEncoderRuntime`` encoding B rows in lockstep.

Table B — full route (encoder -> decoder handoff -> decode to EOS) on the
longest retained real fixture (40960 samples -> 105 frames,
audio-konichiwa.ogenkidesuka-fp16) repeated for every row:
  - c=1 baseline: B sequential c=1 encoder + resident-decoder sessions to EOS.
  - batch: batch encoder -> ``handoff_to`` (device-side batch cross-KV) ->
    lockstep batch decode to EOS.

Both tables report real requests/s (B requests per second over the measured
full-route wall time); in a static lockstep batch every request in one route
completes together, so per-request latency equals the route wall time.  Table B
asserts the batch full-route transcripts are bit-exact to the B c=1 sessions at
every B (EOS-gated), so the throughput numbers are for a verified route.

Run under the GPU-gated env:
  HIPENGINE_RUN_CUDA_SM120A=1 HIPENGINE_CUDA_ARCH=sm_120a CUDA_VISIBLE_DEVICES=0
  PYTHONPATH=$PWD uv run --no-project --with torch python scripts/benchmark_c8_batch_encoder_throughput.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.device import Device
from hipengine.loading.moonshine import load_moonshine_model
from hipengine.runtime.moonshine_encoder_cuda import MoonshineCudaEncoderRuntime
from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime
from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime
from hipengine.runtime.moonshine_encoder_cuda_batch import (
    MoonshineCudaBatchEncoderRuntime,
)

_SNAPSHOT = os.environ.get(
    "HIPENGINE_MOONSHINE_SNAPSHOT",
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/"
    "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
)
_FIXTURE_DIR = os.environ.get(
    "HIPENGINE_MOONSHINE_SIX_FIXTURE_DIR",
    "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures-six",
)
_LONG_FIXTURE = "audio-konichiwa.ogenkidesuka-fp16"  # 105 frames / 40960 samples
_LONG_SAMPLES = 40_960
_ENC_SAMPLES = 480_000  # 30 s at 16 kHz -> the certified 1,248-frame encoder bucket
_EOS = 2
_WARMUP = 3
_ROUTES = 15
_BATCH_SIZES = (1, 2, 4, 8)


def _synthesize_1248(batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ``[B, 480000]`` FP16 audio + ``[B, 480000]`` int64 masks."""
    audio = np.stack(
        [
            np.random.default_rng(seed).standard_normal(_ENC_SAMPLES)
            for seed in range(1, batch + 1)
        ],
        axis=0,
    ).astype(np.float16)
    mask = np.ones((batch, _ENC_SAMPLES), dtype=np.int64)
    return audio, mask


def _load_long_fixture(batch: int) -> tuple[np.ndarray, np.ndarray]:
    """The retained 105-frame real fixture repeated for every row (homogeneous B)."""
    with np.load(os.path.join(_FIXTURE_DIR, f"{_LONG_FIXTURE}.npz")) as data:
        audio = np.repeat(data["input.values"], batch, axis=0).astype(np.float32)
        mask = np.repeat(data["input.attention_mask"], batch, axis=0)
    return audio, mask


def _c1_encoder_route(runtime, loaded, audio, mask) -> float:
    """B sequential c=1 encoder ``encode()`` routes; return median wall seconds."""
    batch = audio.shape[0]
    encoders = [
        MoonshineCudaEncoderRuntime(
            audio_samples=audio.shape[1], loaded_model=loaded, owns_weights=False
        )
        for _ in range(batch)
    ]
    for enc in encoders:
        enc.prepare_encoder_kernels()
    try:
        def route() -> float:
            started = time.perf_counter()
            for b in range(batch):
                encoders[b].encode(audio[b : b + 1], mask[b : b + 1])
            return time.perf_counter() - started

        for _ in range(_WARMUP):
            route()  # first-launch/JIT effects are not part of the baseline
        times = [route() for _ in range(_ROUTES)]
        return statistics.median(times)
    finally:
        for enc in encoders:
            enc.close()


def _batch_encoder_route(benc, audio, mask, warmup, routes) -> list[float]:
    """B rows encoded in lockstep via ``encode()``; return per-route wall seconds."""
    times: list[float] = []
    for _ in range(warmup):
        benc.encode(audio, mask)
    for _ in range(routes):
        started = time.perf_counter()
        benc.encode(audio, mask)
        times.append(time.perf_counter() - started)
    return times


def _c1_full_route(runtime, loaded, audio, mask, seeds) -> tuple[float, list[list[int]]]:
    """B sequential c=1 encoder+decoder sessions to EOS; return (wall_s, transcripts).

    The B encoders/decoders are created and prepared once (outside the timed
    wall) so first-launch/JIT effects are excluded, mirroring the batch path
    which prepares its runtimes once before the warmup/timed routes.
    """
    from hipengine.runtime.moonshine_encoder_cuda import moonshine_encoder_frames_from_audio

    spec = loaded.spec
    batch = audio.shape[0]
    frames = moonshine_encoder_frames_from_audio(audio.shape[1])
    encoders = [
        MoonshineCudaEncoderRuntime(
            audio_samples=audio.shape[1], loaded_model=loaded, owns_weights=False
        )
        for _ in range(batch)
    ]
    decoders = [
        MoonshineCudaResidentRuntime(
            encoder_frames=frames, loaded_model=loaded, owns_weights=False
        )
        for _ in range(batch)
    ]
    for enc in encoders:
        enc.prepare_encoder_kernels()
    for dec in decoders:
        dec.prepare_decoder_kernels()
    try:
        def route() -> tuple[float, list[list[int]]]:
            started = time.perf_counter()
            transcripts: list[list[int]] = []
            for b in range(batch):
                decoders[b].reset_generation(clear_cross_cache=False)
                encoders[b].encode(audio[b : b + 1], mask[b : b + 1])
                encoders[b].handoff_to(decoders[b])
                token_id = seeds[b]
                transcript: list[int] = []
                for _ in range(spec.self_cache_capacity):
                    decoders[b].set_decode_state(
                        token_id=token_id, position=decoders[b].self_cache_length
                    )
                    decoders[b].token_step()
                    token_id = int(decoders[b].read_token())
                    if token_id == _EOS:
                        break
                    transcript.append(token_id)
                transcripts.append(transcript)
            return time.perf_counter() - started, transcripts

        for _ in range(_WARMUP):
            route()  # first-launch/JIT effects are not part of the baseline
        times: list[float] = []
        for _ in range(_ROUTES):
            wall, _ = route()
            times.append(wall)
        return statistics.median(times), route()[1]
    finally:
        for enc in encoders:
            enc.close()
        for dec in decoders:
            dec.close()


def _batch_full_route(runtime, loaded, benc, bdec, audio, mask, seeds) -> tuple[float, list[list[int]]]:
    """One batch encoder -> handoff -> lockstep decode-to-EOS route; (wall_s, transcripts)."""
    spec = loaded.spec
    batch = audio.shape[0]
    started = time.perf_counter()
    benc.encode(audio, mask)
    benc.handoff_to(bdec)
    tokens = np.asarray(seeds, dtype=np.int64)
    done = np.zeros(batch, dtype=bool)
    transcripts: list[list[int]] = [[] for _ in range(batch)]
    while not bool(done.all()):
        bdec.set_batch_decode_state(tokens=tokens.tolist(), position=bdec.self_cache_length)
        bdec.batch_token_step()
        tokens = bdec.read_tokens()
        for row in range(batch):
            if not done[row]:
                if int(tokens[row]) == _EOS:
                    done[row] = True
                else:
                    transcripts[row].append(int(tokens[row]))
        if bdec.self_cache_length >= spec.self_cache_capacity:
            break
    wall = time.perf_counter() - started
    return wall, transcripts


def _pct(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    index = min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[index]


def _print_row(batch: int, label: str, times: list[float], c1_total_s: float) -> dict:
    median = statistics.median(times)
    sorted_times = sorted(times)
    p50 = _pct(sorted_times, 50) * 1000.0
    p95 = _pct(sorted_times, 95) * 1000.0
    req_s = batch / median
    c1_req_s = batch / c1_total_s
    speedup = req_s / c1_req_s
    print(f"{batch:>3} {label:<14} {median*1000:>9.3f} {req_s:>8.1f} {p50:>8.3f} "
          f"{p95:>8.3f} {speedup:>9.2f}x")
    return {"c1_seq_total_s": float(c1_total_s), "batch_median_ms": float(median),
            "batch_req_per_s": float(req_s), "vs_c1_seq_req_per_s": float(speedup)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None, help="write results JSON to this path")
    parser.add_argument("--skip-long-bucket", action="store_true",
                        help="skip the 1,248-frame encoder-only scaling table")
    args = parser.parse_args()

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(_SNAPSHOT, device=Device("cuda", 0), runtime=runtime)
    results: dict[str, dict[str, object]] = {"method": {}, "results": {}}
    try:
        # ---- Table A: production-length (105-frame real fixture) ------------
        print(f"[A] {_LONG_FIXTURE} ({_LONG_SAMPLES} samples -> 105 frames, "
              f"every row identical); encoder-only and full route")
        print(f"{'B':>3} {'route':<14} {'ms/route':>9} {'req/s':>8} {'P50(ms)':>8} "
              f"{'P95(ms)':>8} {'vs c1-seq':>9}")
        for batch in _BATCH_SIZES:
            audio, mask = _load_long_fixture(batch)
            seeds = [1] * batch

            # encoder-only
            c1_enc = _c1_encoder_route(runtime, loaded, audio, mask)
            benc = MoonshineCudaBatchEncoderRuntime(
                max_batch=batch, audio_samples=_LONG_SAMPLES,
                loaded_model=loaded, owns_weights=False,
            )
            benc.prepare_encoder_kernels()
            try:
                times = _batch_encoder_route(benc, audio, mask, warmup=_WARMUP, routes=_ROUTES)
                row_enc = _print_row(batch, "enc-batch", times, c1_enc)
            finally:
                benc.close()

            # full route (encode + handoff + decode to EOS)
            c1_full, c1_transcripts = _c1_full_route(runtime, loaded, audio, mask, seeds)
            benc = MoonshineCudaBatchEncoderRuntime(
                max_batch=batch, audio_samples=_LONG_SAMPLES,
                loaded_model=loaded, owns_weights=False,
            )
            benc.prepare_encoder_kernels()
            bdec = MoonshineCudaBatchRuntime(
                max_batch=batch, encoder_frames=105,
                loaded_model=loaded, owns_weights=False,
            )
            bdec.prepare_decoder_kernels()
            try:
                for _ in range(_WARMUP):
                    _batch_full_route(runtime, loaded, benc, bdec, audio, mask, seeds)
                times_full: list[float] = []
                for _ in range(_ROUTES):
                    wall, _ = _batch_full_route(runtime, loaded, benc, bdec, audio, mask, seeds)
                    times_full.append(wall)
                row_full = _print_row(batch, "full-batch", times_full, c1_full)
                _, batch_transcripts = _batch_full_route(runtime, loaded, benc, bdec, audio, mask, seeds)
                for r in range(batch):
                    assert batch_transcripts[r] == c1_transcripts[r], (
                        f"B={batch} row {r} full-route transcript diverged: "
                        f"batch={batch_transcripts[r]} c1={c1_transcripts[r]}"
                    )
                row_full["transcripts_bit_exact_vs_c1"] = True
            finally:
                bdec.close()
                benc.close()
            results["results"].setdefault(str(batch), {})["encoder_105"] = row_enc
            results["results"].setdefault(str(batch), {})["full_route_105"] = row_full
        print()

        # ---- Table B: long-bucket (1,248-frame) encoder-only scaling --------
        if not args.skip_long_bucket:
            print(f"[B] encoder-only, 480000 samples -> 1,248-frame bucket "
                  f"(synthetic, homogeneous); secondary long-bucket scaling")
            print(f"{'B':>3} {'route':<14} {'ms/route':>9} {'req/s':>8} {'P50(ms)':>8} "
                  f"{'P95(ms)':>8} {'vs c1-seq':>9}")
            for batch in _BATCH_SIZES:
                audio, mask = _synthesize_1248(batch)
                c1_enc = _c1_encoder_route(runtime, loaded, audio, mask)
                benc = MoonshineCudaBatchEncoderRuntime(
                    max_batch=batch, audio_samples=_ENC_SAMPLES,
                    loaded_model=loaded, owns_weights=False,
                )
                benc.prepare_encoder_kernels()
                try:
                    times = _batch_encoder_route(benc, audio, mask, warmup=_WARMUP, routes=_ROUTES)
                    row = _print_row(batch, "enc-batch", times, c1_enc)
                finally:
                    benc.close()
                results["results"].setdefault(str(batch), {})["encoder_1248"] = row
    finally:
        loaded.weights.free(runtime=runtime)

    results["method"] = {
        "benchmark": "scripts/benchmark_c8_batch_encoder_throughput.py",
        "table_a": f"encoder-only + full route on {_LONG_FIXTURE} (40960 samples -> 105 frames) repeated for every row",
        "table_b": "encoder-only, 480000 samples -> 1,248-frame certified bucket, synthetic deterministic homogeneous audio (secondary long-bucket scaling)",
        "baseline": "B independent warm eager c=1 sessions, run sequentially on the same audio; c1 full-route runtimes prepared once outside the timed wall",
        "warmup_routes": _WARMUP,
        "timed_routes": _ROUTES,
    }
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
