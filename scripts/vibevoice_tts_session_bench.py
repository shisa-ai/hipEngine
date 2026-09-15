"""VibeVoice-TTS session benchmark — the doc's executable protocol.

Measures the real request path, not a replay: reference PCM -> voice-prompt
acoustic encode and VAE sampling -> prompt rows -> the positive LM and the
session's own negative LM -> per-frame diffusion -> decode -> semantic feedback
-> PCM. Nothing the model computes is injected.

The frozen request comes from the committed manifest plus the frozen fixtures:
the manifest pins the script, the speaker-reference WAV hash, the CFG scale, the
solver step count and the seed; the fixtures pin the reference PCM the oracle's
processor produced, the resolved prompt token ids, and the recorded random
operands. Only the random operands are injected — the voice-prompt VAE draws and
each diffusion frame's initial noise — because a seed does not align torch and
HIP random streams, so a shared seed is not a shared request.

Tracing is disabled for the timed runs; a separate traced run (not timed) gates
correctness. Reported separately: cold start, warm synthesis, time to first
audible chunk, and pooled RTF, which is wall time over OUTPUT audio seconds.

Reference-audio preprocessing (librosa resample from the 16 kHz voice WAV plus
the fork's dB normalizer) is outside the timed path: the request is defined at
the PCM boundary the model consumes, which is what the manifest's reference-PCM
hash pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from hipengine.loading.vibevoice_tts_session import load_vibevoice_tts_session
from hipengine.runtime.vibevoice_tts_session import VibevoiceTtsSession, SessionTrace
from hipengine.core.hip import get_hip_runtime

SAMPLE_RATE = 24000
FRAME_SAMPLES = 3200


def _frame_token_cap(max_audio_seconds: float) -> int:
    """Token budget from a declared audio budget, not from the oracle's answer.

    Two tokens of overhead (the speech-start and speech-end markers) plus one
    diffusion token per 3200 samples of output audio, the same rule the
    generated-audio quality suite uses.
    """
    frames = math.ceil(float(max_audio_seconds) * SAMPLE_RATE / FRAME_SAMPLES)
    return 2 + frames


def _sync() -> None:
    get_hip_runtime().device_synchronize()


def _gpu_name() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Marketing Name:") and "Radeon" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _sha256_file(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _verify_voice_hashes(request: dict) -> dict:
    """The manifest's voice files must hash to what the manifest recorded."""
    result = {}
    for path, expected in zip(request.get("voices", []), request.get("voice_sha256", [])):
        actual = _sha256_file(path)
        result[Path(path).name] = {
            "expected": expected,
            "actual": actual,
            "verified": actual == expected,
        }
    return result


def bench(model_id: str, fixtures: Path, repeats: int, max_audio_seconds_arg: float | None = None) -> dict:
    manifest = json.loads((fixtures / "manifest.json").read_text())
    request = manifest["requests"][0]
    lm = _npz(fixtures / "single_lm.npz")
    dif = _npz(fixtures / "single_diffusion.npz")
    ref = _npz(fixtures / "single_reference.npz")

    voice_hashes = _verify_voice_hashes(request)
    if voice_hashes and not all(v["verified"] for v in voice_hashes.values()):
        raise RuntimeError(f"manifest voice hash mismatch: {voice_hashes}")

    # The frozen request's reference PCM — the exact waveform the oracle's
    # processor handed the model.
    pcm = np.ascontiguousarray(np.asarray(ref["ref_pcm"])[0], dtype=np.float32)
    pcm_sha256 = hashlib.sha256(pcm.tobytes()).hexdigest()
    voice_noise = np.asarray(ref["encode_draw1"]).reshape(-1, 64)
    voice_noise_scale = np.asarray(ref["encode_draw0"]).reshape(1)

    in_ids = np.asarray(lm["input_ids"])[0]
    mask = np.asarray(lm["speech_input_mask"], dtype=bool).reshape(-1)
    expected = [int(t) for t in np.asarray(lm["generated_ids"])[0][len(in_ids):]]
    sample_rate = int(manifest.get("sampling_rate", 24000))
    # Declared generation budget for the frozen request: the oracle's own audio
    # duration plus one second of headroom, so a correct chain reaches EOS with
    # room to spare and a runaway one is still cut off and reported as truncated.
    # It is fixed by the request, not by the number of tokens under test.
    max_audio_seconds = float(
        max_audio_seconds_arg
        if max_audio_seconds_arg is not None
        else round(float(request["pcm_seconds"]) + 1.0, 3)
    )

    t0 = time.perf_counter()
    weights = load_vibevoice_tts_session(model_id)
    t_load = time.perf_counter() - t0

    sess = VibevoiceTtsSession(weights, max_context=256)

    timers = {"prompt": 0.0, "diffusion": 0.0, "decode": 0.0, "semantic": 0.0}
    request_start = [0.0]
    ttfa = [None]
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
        if ttfa[0] is None:
            # Wall from the start of the request to the end of the first decode.
            ttfa[0] = time.perf_counter() - request_start[0]
        return out

    def timed_semantic(*a, **k):
        _sync(); t = time.perf_counter()
        out = real_semantic(*a, **k)
        _sync(); timers["semantic"] += time.perf_counter() - t
        return out

    sess.diffusion.sample_speech_tokens = timed_diffusion
    sess.decoder.decode = timed_decode
    sess.frontend.encode_chunk_streaming = timed_semantic

    def prepare_prompt() -> list[np.ndarray]:
        """Reference PCM -> voice-prompt rows, on the session's real path."""
        _sync(); t = time.perf_counter()
        _, connected = sess.voice_prompt_rows(
            pcm, noise=voice_noise, noise_scale=voice_noise_scale
        )
        rows = sess.build_prompt_rows(in_ids, mask, connected)
        _sync(); timers["prompt"] += time.perf_counter() - t
        return rows

    def run(*, trace: SessionTrace | None = None) -> tuple[list[int], list[np.ndarray], float, str]:
        for k in timers:
            timers[k] = 0.0
        ttfa[0] = None
        _sync(); request_start[0] = time.perf_counter()
        rows = prepare_prompt()
        res = sess.generate(
            rows,
            cfg_scale=request["cfg_scale"],
            # A declared audio budget, not the oracle's token count. Sizing the
            # budget from the answer under test would make truncation
            # unobservable and the output duration self-fulfilling; the cap is
            # generous enough that an exact chain still stops on EOS.
            max_new_tokens=_frame_token_cap(max_audio_seconds),
            # No neg_hook: the session's own negative LM runs.
            noise_hook=lambda i: dif[f"call{i}_initial_noise"],
            trace=trace,
        )
        _sync(); elapsed = time.perf_counter() - request_start[0]
        return res.ids, res.chunks, elapsed, res.finish_reason

    # Cold start: the first synthesis after load, before anything else runs.
    # Previously the traced correctness run went first, so the reported "cold"
    # figure was a warm synthesis and the label was wrong.
    cold_ids, cold_chunks, cold_wall, cold_finish = run()

    # Correctness: one traced, untimed run against the frozen oracle chain.
    trace = SessionTrace()
    ids_ref, chunks_ref, _, ref_finish = run(trace=trace)
    chain_exact = ids_ref == expected
    neg_ok = True
    for call in (0, 1, 2, 12, 24):
        ref_neg = np.asarray(dif[f"call{call}_neg_condition"]).reshape(-1)
        got = trace.neg_conditions[call]
        peak = float(np.abs(ref_neg).max())
        if np.abs(got - ref_neg).max() / peak > 0.15:
            neg_ok = False

    # Timing: untraced runs only.
    runs = []
    for _ in range(repeats):
        ids, chunks, elapsed, finish = run()
        runs.append((ids, chunks, elapsed, dict(timers), ttfa[0], finish))

    sess.diffusion.sample_speech_tokens = real_diffusion
    sess.decoder.decode = real_decode
    sess.frontend.encode_chunk_streaming = real_semantic
    sess.close()

    timed_chain_ok = all(r[0] == expected for r in runs)
    # Output duration comes from the waveform this lane produced, not the oracle.
    out_seconds = float(sum(int(c.size) for c in runs[0][1])) / sample_rate

    warm = runs
    warm_wall = [r[2] for r in warm]
    warm_stages = {k: float(np.mean([r[3][k] for r in warm])) for k in timers}
    lm_time = float(np.mean(warm_wall)) - sum(warm_stages.values())
    warm_ttfa = [r[4] for r in warm if r[4] is not None]

    return {
        "model": manifest.get("model_id", "microsoft/VibeVoice-1.5B"),
        "quant": "bf16",
        "measurement_basis": "real-request-path",
        "request": {
            "script": request["script"],
            "prompt_tokens": int(len(in_ids)),
            "generated_tokens": len(expected),
            "diffusion_frames": len(expected) - 2,
            "cfg_scale": request["cfg_scale"],
            "ddpm_steps": request["ddpm_inference_steps"],
            "seed": request.get("seed"),
            "reference_pcm_sha256": pcm_sha256,
            "reference_pcm_samples": int(pcm.size),
            "voice_files": voice_hashes,
            "injected": "recorded random operands only (voice VAE draws, "
                        "per-frame initial noise)",
            "not_injected": "prompt embeddings, negative conditions",
            "tracing_during_timing": False,
            "max_audio_seconds": round(max_audio_seconds, 3),
            "token_cap": int(_frame_token_cap(max_audio_seconds)),
            "generated_tokens": len(cold_ids),
            "finish_reason": cold_finish,
            "timed_finish_reasons": sorted({r[5] for r in runs}),
            "oracle_generated_tokens": len(expected),
        },
        "workload": {
            "output_audio_seconds": round(out_seconds, 3),
            "sample_rate": sample_rate,
        },
        "host": {"name": Path("/etc/hostname").read_text().strip(), "gpu": _gpu_name()},
        "command": "uv run python scripts/vibevoice_tts_session_bench.py",
        "weight_load_seconds": round(t_load, 3),
        "cold_start_seconds": round(cold_wall, 3),
        "warm_synthesis_seconds": round(float(np.mean(warm_wall)), 3),
        "time_to_first_audio_seconds": round(float(np.mean(warm_ttfa)), 3) if warm_ttfa else None,
        "pooled_rtf": round(float(np.mean(warm_wall)) / out_seconds, 3),
        "chain_exact": bool(chain_exact and timed_chain_ok),
        "negative_conditions_match": bool(neg_ok),
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
    ap.add_argument(
        "--max-audio-seconds",
        type=float,
        default=None,
        help="declared generation budget; defaults to the request's recorded "
             "duration plus one second",
    )
    ap.add_argument(
        "--out",
        default="benchmarks/results/2026-09-15-gfx1151-vibevoice-tts-session-real-request.json",
    )
    args = ap.parse_args()
    result = bench(args.model, Path(args.fixtures), args.repeats, args.max_audio_seconds)
    print(json.dumps(result, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
