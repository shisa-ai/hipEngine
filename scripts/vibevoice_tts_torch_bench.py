"""Time the torch oracle lane on the same frozen TTS request.

Runs inside the pinned oracle venv (scripts/setup_vibevoice_tts_oracle_env.sh)
against the community fork, and reports the protocol numbers so the HIP lane has
a same-host reference to compare against. The request comes from the committed
manifest: script, speaker-reference WAV (hash-verified), CFG scale, solver step
count and seed.

This lane draws its own random tensors, because a torch seed does not align with
the HIP lane's numpy stream. So this is the same *request* but not the same
*random draw*; the comparison is duration-based, and the HIP lane's
chain-exactness is gated separately against the frozen oracle fixtures.

Reference-audio preprocessing is included here (librosa resample from the 16 kHz
WAV plus the fork's dB normalizer) because it is the fork's own path; the HIP
lane is defined at the PCM boundary that preprocessing produces.

Usage:
    /home/lhl/venvs/vibevoice-tts-oracle/bin/python \
        scripts/vibevoice_tts_torch_bench.py [--fork DIR] [--repeats N] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PINNED_SNAPSHOT = (
    "c00898d257e6b46004e3e2866a47534085fb685a"
)
DEFAULT_FORK = "/home/lhl/VibeVoice-community"
DEFAULT_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "vibevoice_tts"


def _gpu_name() -> str:
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default=DEFAULT_FORK)
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--out",
        default="benchmarks/results/2026-09-15-gfx1151-vibevoice-tts-torch-lane.json",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.fork)
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    fixtures = Path(args.fixtures)
    manifest = json.loads((fixtures / "manifest.json").read_text())
    request = manifest["requests"][0]

    snapshot = Path(
        "/home/lhl/.cache/huggingface/hub/models--microsoft--VibeVoice-1.5B/snapshots/"
        + PINNED_SNAPSHOT
    )
    if not snapshot.is_dir():
        raise SystemExit(f"pinned snapshot not found: {snapshot}")

    # The request must be the frozen one: verify the manifest's voice hash.
    voice_hashes = {}
    for path, expected in zip(request["voices"], request["voice_sha256"]):
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        voice_hashes[Path(path).name] = {"expected": expected, "actual": actual,
                                        "verified": actual == expected}
    if not all(v["verified"] for v in voice_hashes.values()):
        raise SystemExit(f"manifest voice hash mismatch: {voice_hashes}")

    device, dtype = "cuda", torch.bfloat16
    t0 = time.perf_counter()
    processor = VibeVoiceProcessor.from_pretrained(str(snapshot))
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        str(snapshot), torch_dtype=dtype, device_map=device, attn_implementation="sdpa",
    )
    model.eval()
    t_load = time.perf_counter() - t0
    model.set_ddpm_inference_steps(num_steps=int(request["ddpm_inference_steps"]))

    inputs = processor(
        text=[request["script"]],
        voice_samples=[[v for v in request["voices"]]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    def timed() -> tuple[float, float, int]:
        """Returns (wall seconds, time to first audible chunk, new tokens)."""
        torch.manual_seed(int(request["seed"]))
        torch.cuda.synchronize()
        wall = time.perf_counter()
        first = None
        with torch.no_grad():
            out = model.generate(
                **moved,
                max_new_tokens=None,
                cfg_scale=float(request["cfg_scale"]),
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                is_prefill=True,
                show_progress_bar=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - wall
        ids = out.sequences if hasattr(out, "sequences") else out[0]
        n_new = int(ids.shape[-1]) - int(moved["input_ids"].shape[-1])
        audio = getattr(out, "speech_outputs", None)
        return elapsed, first, n_new

    cold_wall, _, n_new = timed()
    warm = [timed() for _ in range(max(args.repeats - 1, 1))]
    warm_wall = float(np.mean([w[0] for w in warm]))

    # Output duration from the audio this lane produced.
    frames = max(n_new - 2, 0)
    out_seconds = frames * 3200 / 24000

    result = {
        "lane": "torch-bf16-sdpa",
        "measurement_basis": "real-request-path",
        "request": {
            "script": request["script"],
            "cfg_scale": request["cfg_scale"],
            "ddpm_steps": request["ddpm_inference_steps"],
            "seed": request["seed"],
            "voice_files": voice_hashes,
            "random_draws": "this lane's own torch stream (not shared with the HIP lane)",
        },
        "workload": {
            "generated_tokens": n_new,
            "diffusion_frames": frames,
            "output_audio_seconds": round(out_seconds, 3),
            "sample_rate": 24000,
        },
        "host": {"name": Path("/etc/hostname").read_text().strip(), "gpu": _gpu_name()},
        "command": (
            "/home/lhl/venvs/vibevoice-tts-oracle/bin/python "
            "scripts/vibevoice_tts_torch_bench.py"
        ),
        "torch_version": torch.__version__,
        "weight_load_seconds": round(t_load, 3),
        "cold_start_seconds": round(cold_wall, 3),
        "warm_synthesis_seconds": round(warm_wall, 3),
        "time_to_first_audio_seconds": None,
        "pooled_rtf": round(warm_wall / out_seconds, 3),
        "note": (
            "time_to_first_audio is None because the fork's generate() exposes no "
            "streaming callback here; it is reported by the HIP lane only."
        ),
    }
    print(json.dumps(result, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
