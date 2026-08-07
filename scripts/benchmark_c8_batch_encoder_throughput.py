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

from c8_report_common import build_report, percentile, route_timing_result


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


def _batch_enc_handoff_graph_timing(
    runtime, benc, bdec, audio, mask, warmup, routes
) -> tuple[list[float], list[float], dict[str, object]]:
    """Time eager encode+handoff vs one captured encoder-chain graph replay.

    Captures the batch encoder DAG + fresh-generation reset + cross-KV handoff
    as one fixed-address graph on the decoder stream, then returns the per-route
    wall times for (a) the eager encode+handoff and (b) the single graph
    replay, plus the capture/instantiate contract.  Both paths do identical
    device work (the graph adds only the generation-reset memsets that the
    eager ``handoff_to`` also performs); the graph removes the per-request
    Python dispatch of ~101 encoder kernels plus the handoff.
    """

    eager: list[float] = []
    graph: list[float] = []
    benc.upload_input(audio, mask)
    # eager warmup (encode on the encoder stream, handoff on the decoder stream)
    for _ in range(warmup):
        benc.run_encode(synchronize=False)
        benc.handoff_to(bdec, synchronize=False)
        runtime.stream_synchronize(benc.stream)
        runtime.stream_synchronize(bdec.stream)
    for _ in range(routes):
        started = time.perf_counter()
        benc.run_encode(synchronize=False)
        benc.handoff_to(bdec, synchronize=False)
        runtime.stream_synchronize(benc.stream)
        runtime.stream_synchronize(bdec.stream)
        eager.append(time.perf_counter() - started)

    # capture once outside the timed region, then warmup + time graph replays.
    graph_obj = benc.capture_encoder_chain(bdec)
    contract = benc.encoder_chain_graph_contract()
    for _ in range(warmup):
        benc.graph_encode_and_handoff(bdec)
        runtime.stream_synchronize(bdec.stream)
    for _ in range(routes):
        started = time.perf_counter()
        benc.graph_encode_and_handoff(bdec)
        runtime.stream_synchronize(bdec.stream)
        graph.append(time.perf_counter() - started)
    return eager, graph, {
        "captured": contract["captured"],
        "graph": graph_obj.graph if contract["captured"] else 0,
        "capture_wall_ms": contract["capture_wall_ms"],
        "instantiate_wall_ms": contract["instantiate_wall_ms"],
        "replay_count": contract["replay_count"],
    }


def _print_graph_row(batch: int, label: str, eager_ms: float, graph_ms: float,
                    *, eager_raw: list[float] | None = None,
                    graph_raw: list[float] | None = None) -> dict:
    speedup = eager_ms / graph_ms if graph_ms > 0 else float("inf")
    print(f"{batch:>3} {label:<16} {eager_ms:>9.3f} {graph_ms:>9.3f} "
          f"{speedup:>9.2f}x")
    row: dict = {"eager_enc_handoff_ms": float(eager_ms), "graph_enc_handoff_ms": float(graph_ms),
                 "graph_speedup": float(speedup)}
    # RR-6: retain raw per-route wall samples for the graph vs eager pair.
    if eager_raw is not None:
        eager_values = [float(v) for v in eager_raw]
        row["eager_enc_handoff_ms_raw"] = eager_values
        row["eager_p50_ms"] = percentile(sorted(eager_values), 50)
        row["eager_p95_ms"] = percentile(sorted(eager_values), 95)
        row["eager_sample_count"] = len(eager_values)
    if graph_raw is not None:
        graph_values = [float(v) for v in graph_raw]
        row["graph_enc_handoff_ms_raw"] = graph_values
        row["graph_p50_ms"] = percentile(sorted(graph_values), 50)
        row["graph_p95_ms"] = percentile(sorted(graph_values), 95)
        row["graph_sample_count"] = len(graph_values)
    return row


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


def _print_row(batch: int, label: str, times: list[float], c1_total_s: float) -> dict:
    row = route_timing_result(
        times,
        batch=batch,
        c1_seq_total_s=c1_total_s,
    )
    print(
        f"{batch:>3} {label:<14} {row['batch_median_ms']:>9.3f} "
        f"{row['batch_req_per_s']:>8.1f} {row['p50_ms']:>8.3f} "
        f"{row['p95_ms']:>8.3f} {row['vs_c1_seq_req_per_s']:>9.2f}x"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None, help="write results JSON to this path")
    parser.add_argument("--skip-long-bucket", action="store_true",
                        help="skip the 1,248-frame encoder-only scaling table")
    parser.add_argument("--encoder-graph", action="store_true",
                        help="capture the batch encoder-chain (encoder+handoff+cross-KV) "
                             "as one fixed-address graph and time eager vs graph replay "
                             "(Table C, long-bucket 1,248-frame cuBLASLt route)")
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
            print("[B] encoder-only, 480000 samples -> 1,248-frame bucket "
                  "(synthetic, homogeneous); secondary long-bucket scaling")
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

        # ---- Table C: encoder-chain graph vs eager encode+handoff ----------
        if args.encoder_graph:
            print("[C] encoder-chain graph capture vs eager encode+handoff, "
                  "long-bucket cuBLASLt route (480000 samples -> 1,248 frames)")
            print(f"{'B':>3} {'route':<16} {'eager ms':>9} {'graph ms':>9} {'speedup':>9}")
            results["graph"] = {}
            for batch in _BATCH_SIZES:
                audio, mask = _synthesize_1248(batch)
                benc = MoonshineCudaBatchEncoderRuntime(
                    max_batch=batch, audio_samples=_ENC_SAMPLES,
                    loaded_model=loaded, owns_weights=False,
                    projection_route="cublaslt",
                )
                benc.prepare_encoder_kernels()
                bdec = MoonshineCudaBatchRuntime(
                    max_batch=batch, encoder_frames=1248,
                    loaded_model=loaded, owns_weights=False,
                )
                bdec.prepare_decoder_kernels()
                try:
                    eager, graph, contract = _batch_enc_handoff_graph_timing(
                        runtime, benc, bdec, audio, mask, warmup=_WARMUP, routes=_ROUTES
                    )
                    row = _print_graph_row(
                        batch, "lt-graph",
                        statistics.median(eager) * 1000.0,
                        statistics.median(graph) * 1000.0,
                        eager_raw=[value * 1000.0 for value in eager],
                        graph_raw=[value * 1000.0 for value in graph],
                    )
                    row.update(contract)
                    results["graph"][str(batch)] = row
                finally:
                    bdec.close()
                    benc.close()
            print()
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
    # RR-6: fold the encoder-chain graph table (when measured) into the
    # retained report so the graph evidence survives into the JSON (raw
    # per-B eager/graph samples from ``_print_graph_row``).  ``build_report``
    # only serializes ``results["results"]``, so the sibling graph dict must
    # be merged here rather than dropped.
    if "graph" in results:
        results["results"]["graph"] = results["graph"]

    # RR-6: complete retained report with raw samples, P50/P95, full source
    # manifest hashes, clean git revision, and dependency-adjusted bytes.  The
    # in-loop transcript assertions passed (they raise on divergence), so the
    # full-route batch is bit-exact vs the c=1 sessions at every B.
    results.setdefault("correctness", {})
    results["correctness"]["batch_transcripts_bit_exact_vs_c1"] = True
    report = build_report(
        artifact="moonshine_cuda_c8_batch_encoder_throughput",
        scope={
            "target": "batch_encoder_full_route_and_encoder_only",
            "gpu_index": 0,
            "exclusive_gpu": True,
            "fixtures": [_LONG_FIXTURE, "synthetic-1248-long-bucket"],
        },
        environment={
            "torch": _torch_version(),
            "platform": __import__("platform").platform(),
        },
        results=results["results"],
        correctness=results["correctness"],
        method=results["method"],
        dependency_route=(
            "c8_batch_encoder_cublaslt_route"
            if args.encoder_graph
            else "custom_cuda_runtime_subset"
        ),
        benchmark_source="scripts/benchmark_c8_batch_encoder_throughput.py",
    )
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json_out}")


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "n/a"


if __name__ == "__main__":
    main()
