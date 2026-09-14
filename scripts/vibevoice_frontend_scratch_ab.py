#!/usr/bin/env python3
"""VibeVoice-ASR front-end scratch-arena A/B and allocation accounting.

Measures the front-end encode stage with the shipped ``_ScratchPool.reset``
(rewind a large-enough arena) against the legacy behaviour (close and recreate
the arena on every reset). ``reset`` runs once per encoder pass, so every
forward used to pay two ``hipMalloc``/``hipFree`` pairs for the whole arena.

Also reports, for each clip length, the capacity requested per reset, how many
resets a forward performs, and a SHA-256 of the returned embeddings so a
before/after comparison can show the outputs are unchanged.

Timing boundaries: the only timed region is the ``frontend.forward(pcm)`` call
itself. Input construction, the hash, and any output comparison happen outside
it. Lanes alternate inside one process and one trial loop, so host drift cannot
favour a lane.

Usage:
    python3 scripts/vibevoice_frontend_scratch_ab.py [--trials N] [--json PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

PINNED_HF_MODEL_ID = "microsoft/VibeVoice-ASR-HF"

# Seed per clip, so a clip's waveform does not depend on which other durations
# were requested in the same invocation. A single shared RNG makes the 30 s
# waveform differ between `--seconds 30` and `--seconds 11 30`, which would make
# the recorded output hashes incomparable across runs.
SEED_BASE = 20260914


def _legacy_reset(pool, capacity_bytes=None):
    """Previous behaviour: free and reallocate the arena on every reset."""
    from hipengine.core.memory import DeviceMemoryArena

    if capacity_bytes is not None:
        pool._capacity = max(int(capacity_bytes), 1 << 20)
    if pool._arena is not None:
        pool._arena.close()
    pool._arena = DeviceMemoryArena.create(pool._capacity)


def _assert_shipped_reset_reuses() -> None:
    """Fail loudly if the shipped reset is not the reusing one.

    Without this the A/B would silently compare the legacy path against itself
    after a revert, and report a meaningless zero delta.
    """
    from hipengine.runtime.vibevoice_encoder import _ScratchPool

    probe = _ScratchPool(capacity_bytes=1 << 20)
    try:
        probe.reset(capacity_bytes=1 << 20)
        first = probe._arena
        probe.reset(capacity_bytes=1 << 20)
        if probe._arena is not first:
            raise SystemExit(
                "shipped _ScratchPool.reset() reallocates a fitting arena; "
                "this A/B is stale and would compare the legacy path with itself")
    finally:
        probe.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=6,
                        help="timed passes per lane, alternating")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lane", choices=("both", "legacy", "reused"), default="both",
                        help="restrict to one lane, so a separate-process "
                             "before/after run is reproducible too")
    parser.add_argument("--seconds", type=float, nargs="+", default=[11.0, 30.0, 90.0])
    parser.add_argument("--chunk-samples", type=int, default=1_440_000,
                        help="front-end chunk size; a clip longer than this is "
                             "encoded in several chunks, which is the case where "
                             "a stale arena would show up")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_asr import (load_vibevoice_connector,
                                                 load_vibevoice_encoder)
    from hipengine.runtime.vibevoice_encoder import (VibevoiceFrontendRuntime,
                                                     _ScratchPool)

    _assert_shipped_reset_reuses()

    model = str(resolve_model_path(PINNED_HF_MODEL_ID))
    specs = {k: load_vibevoice_encoder(model, k) for k in ("acoustic", "semantic")}
    for name, (spec, _weights) in specs.items():
        print(f"{name:9s} num_filters={spec.num_filters} ratios={spec.ratios} "
              f"hidden={spec.hidden_size}")

    frontend = VibevoiceFrontendRuntime(
        *specs["acoustic"], *specs["semantic"],
        load_vibevoice_connector(model, "acoustic"),
        load_vibevoice_connector(model, "semantic"),
        frontend_variant="wmma",
    )

    shipped_reset = _ScratchPool.reset
    resets: list[int | None] = []
    original_reset = shipped_reset

    def recording_reset(self, capacity_bytes=None):
        resets.append(capacity_bytes)
        return original_reset(self, capacity_bytes)

    payload: dict[str, object] = {
        "clip_seconds": list(args.seconds),
        "chunk_samples": args.chunk_samples,
        "lane": args.lane,
        "clips": {},
    }
    try:
        rng = None
        for seconds in args.seconds:
            rng = np.random.default_rng(SEED_BASE + int(round(seconds)))
            pcm = (rng.standard_normal(int(16000 * seconds)) * 0.05).astype(np.float32)

            # Allocation accounting: what does one forward ask the pool for?
            _ScratchPool.reset = recording_reset
            resets.clear()
            frontend.forward(pcm, chunk_samples=args.chunk_samples)
            _ScratchPool.reset = shipped_reset
            capacities = [c for c in resets if c]

            lane_plan = []
            if args.lane in ("both", "legacy"):
                lane_plan.append((_legacy_reset, "legacy"))
            if args.lane in ("both", "reused"):
                lane_plan.append((shipped_reset, "reused"))

            for _ in range(args.warmup):
                for reset, _name in lane_plan:
                    _ScratchPool.reset = reset
                    frontend.forward(pcm, chunk_samples=args.chunk_samples)
            _ScratchPool.reset = shipped_reset

            timings: dict[str, list[float]] = {name: [] for _r, name in lane_plan}
            digests: list[str] = []
            for _ in range(args.trials):
                for reset, name in lane_plan:
                    _ScratchPool.reset = reset
                    t0 = time.perf_counter()
                    embeds = frontend.forward(pcm, chunk_samples=args.chunk_samples)
                    timings[name].append((time.perf_counter() - t0) * 1000.0)
                digests.append(hashlib.sha256(
                    np.ascontiguousarray(embeds).tobytes()).hexdigest())
            _ScratchPool.reset = shipped_reset

            legacy = timings.get("legacy", [])
            reused = timings.get("reused", [])
            lm_ = statistics.median(legacy) if legacy else None
            rm_ = statistics.median(reused) if reused else None
            print(f"\n{seconds:.0f}s audio ({pcm.size} samples, "
                  f"chunk_samples={args.chunk_samples}, "
                  f"chunks={-(-pcm.size // args.chunk_samples)})")
            print(f"  resets per forward  : {len(resets)}")
            print(f"  capacity per reset  : "
                  f"{[f'{c / 1e6:.0f}MB' for c in capacities]}")
            if lm_ is not None:
                print(f"  legacy close+create : {lm_:7.1f} ms  {[round(v, 1) for v in legacy]}")
            if rm_ is not None:
                print(f"  reused arena        : {rm_:7.1f} ms  {[round(v, 1) for v in reused]}")
            if lm_ is not None and rm_ is not None:
                print(f"  delta               : {lm_ - rm_:+7.1f} ms "
                      f"({(lm_ / rm_ - 1) * 100:+.2f}%)")
            print(f"  embeddings stable   : {len(set(digests)) == 1} "
                  f"sha={digests[0][:32]}")

            entry: dict[str, object] = {
                "samples": int(pcm.size),
                "chunk_samples": args.chunk_samples,
                "chunks": -(-pcm.size // args.chunk_samples),
                "resets_per_forward": len(resets),
                "capacity_bytes_per_reset": [int(c) for c in capacities],
                "legacy_ms": round(lm_, 1) if lm_ is not None else None,
                "legacy_trials": [round(v, 1) for v in legacy],
                "reused_ms": round(rm_, 1) if rm_ is not None else None,
                "reused_trials": [round(v, 1) for v in reused],
                "embeddings_sha256": digests[0],
                "embeddings_stable_across_trials": len(set(digests)) == 1,
                "embed_shape": list(embeds.shape),
            }
            if lm_ is not None and rm_ is not None:
                entry["delta_ms"] = round(lm_ - rm_, 1)
                entry["delta_pct"] = round((lm_ / rm_ - 1) * 100, 2)
            payload["clips"][f"{seconds:g}s"] = entry
    finally:
        _ScratchPool.reset = shipped_reset
        frontend.close()

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
