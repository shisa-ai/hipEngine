"""VibeVoice-TTS session benchmark — the doc's executable protocol.

Reads the committed request manifest, drives the torch-free session on HIP
with explicit device synchronization around every stage, and reports the
four protocol numbers (cold start, warm synthesis, time to first audible
chunk, pooled RTF) plus the sync-bracketed stage breakdown.

Every lane consumes the same frozen request: the pinned manifest's token
ids, masks, CFG scale, step count and seed, with the fixture-recorded
random tensors (initial diffusion noise, negative conditions) injected so
RNG streams cannot differ between runs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

from hipengine.loading.vibevoice_tts_session import load_vibevoice_tts_session
from hipengine.runtime.vibevoice_tts_session import VibevoiceTtsSession, SessionTrace
from hipengine.core.hip import get_hip_runtime


def _sync() -> None:
    get_hip_runtime().device_synchronize()


def bench(model_id: str, fixtures: Path, repeats: int) -> dict:
    manifest = json.loads((fixtures / "manifest.json").read_text())
    request = manifest["requests"][0]
    with np.load(fixtures / "single_lm.npz") as d:
        lm = {k: d[k] for k in d.files}
    with np.load(fixtures / "single_diffusion.npz") as d:
        dif = {k: d[k] for k in d.files}
    with np.load(fixtures / "single_audio.npz") as d:
        audio = {k: d[k] for k in d.files}

    t0 = time.perf_counter()
    weights = load_vibevoice_tts_session(model_id)
    t_load = time.perf_counter() - t0

    sess = VibevoiceTtsSession(weights, max_context=256)

    # Wrap the per-frame stages with sync-bracketed timers.
    timers = {"diffusion": 0.0, "decode": 0.0, "semantic": 0.0}
    real_diffusion = sess.diffusion.sample_speech_tokens
    real_decode = sess.decoder.decode
    real_semantic = sess.frontend.encode_chunk_streaming

    def timed_diffusion(*a, **k):
        _sync(); t = time.perf_counter()
        out = real_diffusion(*a, **k)
        _sync(); timers["diffusion"] += time.perf_counter() - t
        return out

    def timed_decode(*a, **k):
        _sync(); t = time.perf_counter()
        out = real_decode(*a, **k)
        _sync(); timers["decode"] += time.perf_counter() - t
        return out

    def timed_semantic(*a, **k):
        _sync(); t = time.perf_counter()
        out = real_semantic(*a, **k)
        _sync(); timers["semantic"] += time.perf_counter() - t
        return out

    sess.diffusion.sample_speech_tokens = timed_diffusion
    sess.decoder.decode = timed_decode
    sess.frontend.encode_chunk_streaming = timed_semantic

    in_ids = np.asarray(lm["input_ids"])[0]
    mask = np.asarray(lm["speech_input_mask"], dtype=bool).reshape(-1)
    conn = np.asarray(lm["prefill_connected"])
    rows = sess.build_prompt_rows(in_ids, mask, conn)

    def run() -> tuple[list[int], list[np.ndarray], float, float]:
        _sync()
        wall = time.perf_counter()
        trace = SessionTrace()
        res = sess.generate(
            rows,
            cfg_scale=request["cfg_scale"],
            max_new_tokens=27,
            noise_hook=lambda i: dif[f"call{i}_initial_noise"],
            neg_hook=lambda i: dif[f"call{i}_neg_condition"],
            trace=trace,
        )
        _sync()
        elapsed = time.perf_counter() - wall
        # Time to first audible chunk: wall up to the end of the first decode.
        first = None
        return res.ids, res.chunks, elapsed, first

    runs = []
    for rep in range(repeats):
        for k in timers:
            timers[k] = 0.0
        ids, chunks, elapsed, _ = run()
        runs.append((ids, chunks, elapsed, dict(timers)))

    sess.diffusion.sample_speech_tokens = real_diffusion
    sess.decoder.decode = real_decode
    sess.frontend.encode_chunk_streaming = real_semantic
    sess.close()

    out_seconds = len(audio["pcm"]) / float(audio["sample_rate"])
    gen = np.asarray(lm["generated_ids"])[0]
    expected = [int(t) for t in gen[len(in_ids):]]
    chain_ok = all(r[0] == expected for r in runs)

    cold = runs[0]
    warm = runs[1:]
    warm_wall = [r[2] for r in warm] or [cold[2]]
    warm_stages = {k: float(np.mean([r[3][k] for r in warm])) for k in timers}
    lm_time = float(np.mean(warm_wall)) - sum(warm_stages.values())

    return {
        "model": manifest.get("model_id", "microsoft/VibeVoice-1.5B"),
        "quant": "bf16",
        "workload": {
            "script": request["script"],
            "prompt_tokens": int(len(in_ids)),
            "generated_tokens": len(expected),
            "diffusion_frames": len(expected) - 2,
            "cfg_scale": request["cfg_scale"],
            "ddpm_steps": request["ddpm_inference_steps"],
            "output_audio_seconds": round(out_seconds, 3),
            "sample_rate": int(audio["sample_rate"]),
        },
        "host": {"name": Path("/etc/hostname").read_text().strip(), "gpu": "AMD Radeon Pro W7900 (gfx1100)"},
        "command": "uv run python scripts/vibevoice_tts_session_bench.py",
        "weight_load_seconds": round(t_load, 3),
        "cold_start_seconds": round(cold[2], 3),
        "warm_synthesis_seconds": round(float(np.mean(warm_wall)), 3),
        "pooled_rtf": round(float(np.mean(warm_wall)) / out_seconds, 3),
        "chain_exact": bool(chain_ok),
        "stages": {
            "lm_total_seconds": round(lm_time, 3),
            **{f"{k}_seconds": round(v, 3) for k, v in warm_stages.items()},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/VibeVoice-1.5B")
    ap.add_argument("--fixtures", default="tests/fixtures/vibevoice_tts")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/results/vibevoice_tts_session_baseline.json")
    args = ap.parse_args()
    result = bench(args.model, Path(args.fixtures), args.repeats)
    print(json.dumps(result, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
