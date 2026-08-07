"""C8 phase-1 static-batch throughput benchmark: real requests/s + per-request latency.

Compares, for B=1/2/4/8 on shared-frame fixtures:
- c=1 baseline: B independent c=1 resident sessions decoded sequentially.
- batch eager: one ``MoonshineCudaBatchRuntime`` decoding B rows in lockstep.
- batch graph: the same runtime replaying captured position-bucket CUDA graphs.

Reports real requests/s (B requests completed per second over the measured
full-route wall time) and per-request P50/P95 completion latency over repeated
routes.  In a static lockstep batch every request in one route completes
together, so per-request latency equals the route wall time.

Run under the GPU-gated env:
  HIPENGINE_RUN_CUDA_SM120A=1 HIPENGINE_CUDA_ARCH=sm_120a CUDA_VISIBLE_DEVICES=0
  PYTHONPATH=$PWD uv run --no-project python scripts/benchmark_c8_batch_throughput.py
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.device import Device
from hipengine.loading.moonshine import load_moonshine_model
from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime
from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

from c8_report_common import build_report, route_timing_result

_FIXTURE_DIR = os.environ.get(
    "HIPENGINE_MOONSHINE_SIX_FIXTURE_DIR",
    "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures-six",
)
_SNAPSHOT = os.environ.get(
    "HIPENGINE_MOONSHINE_SNAPSHOT",
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/"
    "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
)
_SIX_FIXTURES = (
    "audio-hai-fp16",
    "audio-konichiwa-fp16",
    "audio-konichiwa.ogenkidesuka-fp16",
    "audio-kumbawa-fp16",
    "audio-sosososo-fp16",
    "audio-sumimasen-fp16",
)
_EOS = 2
# Warmup + repeated routes for P50/P95.
_WARMUP = 3
_ROUTES = 15


def _pad_row_cache(array: np.ndarray, shared_frames: int) -> np.ndarray:
    arr = np.ascontiguousarray(array, dtype=np.float16)
    if arr.shape[2] == shared_frames:
        return arr
    out = np.zeros(
        (arr.shape[0], arr.shape[1], shared_frames, arr.shape[3]), dtype=np.float16
    )
    out[:, :, : arr.shape[2], :] = arr
    return out


def _load(names: list[str], shared_frames: int):
    seeds, keys_b, values_b, masks = [], [], [], []
    for name in names:
        with open(os.path.join(_FIXTURE_DIR, f"{name}.json")) as handle:
            manifest = json.load(handle)
        with np.load(os.path.join(_FIXTURE_DIR, f"{name}.npz")) as fixture:
            frames = int(manifest["input"]["encoder_frames"])
            reference = [int(token) for token in manifest["decoder"]["token_ids"]]
            keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
            values = [fixture[f"cross.layer_{layer}.value"] for layer in range(8)]
        mask = np.zeros((1, shared_frames), dtype=np.int32)
        mask[0, :frames] = 1
        seeds.append(reference[0])
        keys_b.append([_pad_row_cache(k, shared_frames) for k in keys])
        values_b.append([_pad_row_cache(v, shared_frames) for v in values])
        masks.append(mask)
    stacked_keys = [np.concatenate([k[layer] for k in keys_b], axis=0) for layer in range(8)]
    stacked_values = [np.concatenate([v[layer] for v in values_b], axis=0) for layer in range(8)]
    return (
        np.array(seeds, dtype=np.int64),
        stacked_keys,
        stacked_values,
        np.concatenate(masks, axis=0),
    )


def _route_c1(runtime, loaded, encoder_frames, keys, values, mask, seed) -> float:
    """Decode one warm c=1 session to EOS; return wall seconds."""
    spec = loaded.spec
    decoder = MoonshineCudaResidentRuntime(
        encoder_frames=encoder_frames, loaded_model=loaded, owns_weights=False
    )
    decoder.prepare_decoder_kernels()
    try:
        decoder.load_cross_cache(keys, values, mask=mask)

        def route() -> float:
            start = time.perf_counter()
            token_id = seed
            for position in range(spec.self_cache_capacity):
                decoder.set_decode_state(token_id=token_id, position=position)
                decoder.token_step()
                token_id = int(decoder.read_token())
                if token_id == _EOS:
                    break
            return time.perf_counter() - start

        route()  # warmup: first-launch/JIT effects are not part of the baseline
        decoder.reset_generation(clear_cross_cache=False)
        return route()
    finally:
        decoder.close()


def _route_batch(
    decoder: MoonshineCudaBatchRuntime,
    seeds: np.ndarray,
    *,
    graph: bool,
    warmup: int,
    routes: int,
) -> list[float]:
    """Decode B rows in lockstep to all-EOS; return per-route wall seconds.

    Graph mode uses device-owned decode: ``set_batch_decode_seed`` seeds the
    device token/position buffers once per route and the captured graph tail
    advances the position scalars, so each replay is a single graph launch.
    """
    spec = decoder.spec
    batch = decoder.max_batch
    times: list[float] = []

    def run_route() -> None:
        decoder.reset_generation(clear_cross_cache=False)
        if graph:
            decoder.set_batch_device_owned_decode(True)
            decoder.set_batch_decode_seed(tokens=seeds.tolist())
        else:
            tokens = seeds.astype(np.int64)
        done = np.zeros(batch, dtype=bool)
        while not bool(done.all()):
            if graph:
                decoder.graph_batch_token_step()
            else:
                decoder.set_batch_decode_state(
                    tokens=tokens.tolist(), position=decoder.self_cache_length
                )
                decoder.batch_token_step()
            tokens = decoder.read_tokens()
            for row in range(batch):
                if not done[row] and int(tokens[row]) == _EOS:
                    done[row] = True
            if decoder.self_cache_length >= spec.self_cache_capacity:
                break

    for _ in range(warmup):
        run_route()
    for _ in range(routes):
        start = time.perf_counter()
        run_route()
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None, help="write RR-6 report JSON")
    args = parser.parse_args()

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(_SNAPSHOT, device=Device("cuda", 0), runtime=runtime)
    # Resolve the shared bucket across all six fixtures.
    shared_frames = 0
    for name in _SIX_FIXTURES:
        with open(os.path.join(_FIXTURE_DIR, f"{name}.json")) as handle:
            manifest = json.load(handle)
        shared_frames = max(shared_frames, int(manifest["input"]["encoder_frames"]))
    print(f"shared encoder bucket: {shared_frames} frames")
    print(f"{'B':>3} {'mode':<10} {'ms/route':>9} {'req/s':>8} {'P50(ms)':>8} "
          f"{'P95(ms)':>8} {'vs c1-seq':>9}")
    results: dict[str, object] = {}
    try:
        for batch in (1, 2, 4, 8):
            names = [_SIX_FIXTURES[i % len(_SIX_FIXTURES)] for i in range(batch)]
            seeds, keys_b, values_b, masks_b = _load(names, shared_frames)

            # c=1 sequential baseline
            c1_times = []
            for i, name in enumerate(names):
                keys = [keys_b[layer][i : i + 1] for layer in range(8)]
                values = [values_b[layer][i : i + 1] for layer in range(8)]
                mask = masks_b[i : i + 1]
                c1_times.append(_route_c1(runtime, loaded, shared_frames, keys, values, mask, int(seeds[i])))
            c1_seq_total = sum(c1_times)

            decoder = MoonshineCudaBatchRuntime(
                max_batch=batch,
                encoder_frames=shared_frames,
                loaded_model=loaded,
                owns_weights=False,
            )
            decoder.prepare_decoder_kernels()
            decoder.load_cross_cache_batch(keys_b, values_b, masks=masks_b)
            try:
                for graph in (False, True):
                    if graph:
                        # Capture in device-owned mode so the captured DAG tail
                        # advances the position scalars on replay.
                        decoder.reset_generation(clear_cross_cache=False)
                        decoder.set_batch_device_owned_decode(True)
                        decoder.set_batch_decode_seed(tokens=seeds.tolist())
                        decoder.capture_batch_token_graphs()
                    times = _route_batch(decoder, seeds, graph=graph, warmup=_WARMUP, routes=_ROUTES)
                    timing = route_timing_result(
                        times,
                        batch=batch,
                        c1_seq_total_s=c1_seq_total,
                    )
                    mode = "graph" if graph else "eager"
                    print(
                        f"{batch:>3} {mode:<10} {timing['batch_median_ms']:>9.3f} "
                        f"{timing['batch_req_per_s']:>8.1f} {timing['p50_ms']:>8.3f} "
                        f"{timing['p95_ms']:>8.3f} "
                        f"{timing['vs_c1_seq_req_per_s']:>9.2f}x"
                    )
                    # Schema v2: raw seconds and all derived millisecond fields
                    # come from one tested helper, preventing unit drift.
                    results.setdefault(str(batch), {})[mode] = timing
                # graph-to-eager correctness: transcripts must match
                decoder.reset_generation(clear_cross_cache=False)
                decoder.set_batch_device_owned_decode(False)
                tokens = seeds.astype(np.int64)
                done = np.zeros(batch, dtype=bool)
                eager_trans: list[list[int]] = [[] for _ in range(batch)]
                while not bool(done.all()):
                    decoder.set_batch_decode_state(tokens=tokens.tolist(), position=decoder.self_cache_length)
                    decoder.batch_token_step()
                    tokens = decoder.read_tokens()
                    for row in range(batch):
                        if not done[row]:
                            if int(tokens[row]) == _EOS:
                                done[row] = True
                            else:
                                eager_trans[row].append(int(tokens[row]))
                decoder.reset_generation(clear_cross_cache=False)
                decoder.set_batch_device_owned_decode(True)
                decoder.set_batch_decode_seed(tokens=seeds.tolist())
                done = np.zeros(batch, dtype=bool)
                graph_trans: list[list[int]] = [[] for _ in range(batch)]
                while not bool(done.all()):
                    decoder.graph_batch_token_step()
                    tokens = decoder.read_tokens()
                    for row in range(batch):
                        if not done[row]:
                            if int(tokens[row]) == _EOS:
                                done[row] = True
                            else:
                                graph_trans[row].append(int(tokens[row]))
                assert eager_trans == graph_trans, (
                    f"B={batch} graph vs eager transcripts diverged"
                )
                results.setdefault(str(batch), {})["graph_eager_transcripts_match"] = True
                contract = decoder.batch_token_graph_contract()
                assert contract["captured"] is True
            finally:
                decoder.close()
    finally:
        loaded.weights.free(runtime=runtime)

    # RR-6: complete retained report with raw samples, P50/P95, full source
    # manifest hashes, clean git revision, and dependency-adjusted bytes.
    report = build_report(
        artifact="moonshine_cuda_c8_batch_throughput",
        scope={
            "target": "static_batch_decoder_throughput",
            "gpu_index": 0,
            "exclusive_gpu": True,
            "fixtures": list(_SIX_FIXTURES),
            "shared_encoder_bucket_frames": shared_frames,
        },
        environment={
            "torch": _torch_version(),
            "platform": __import__("platform").platform(),
        },
        results=results,
        correctness={"graph_eager_transcripts_match": True},
        method={
            "benchmark": "scripts/benchmark_c8_batch_throughput.py",
            "runtime": "MoonshineCudaBatchRuntime",
            "baseline": "B independent warm eager c=1 resident sessions decoded sequentially",
            "warmup_routes": _WARMUP,
            "timed_routes": _ROUTES,
        },
        dependency_route="custom_cuda_runtime_subset",
        benchmark_source="scripts/benchmark_c8_batch_throughput.py",
    )
    if args.json_out:
        import pathlib

        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
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
