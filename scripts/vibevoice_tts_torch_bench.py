"""Time the torch oracle lane on the same frozen TTS request.

Runs inside the pinned oracle venv (scripts/setup_vibevoice_tts_oracle_env.sh)
against the community fork, and reports the protocol numbers so the HIP lane has
a same-host reference to compare against. The request comes from the committed
manifest: script, speaker-reference WAV (hash-verified), CFG scale, solver step
count and seed.

This lane consumes the same random operands as the HIP lane. The fork draws
randomness in exactly three places on this path -- a ``randn(batch)`` and a
``randn_like(mean)`` for the voice-prompt VAE latent, and one ``randn(2,
vae_dim)`` per diffusion call -- and the committed fixture records all three, so
``install_recorded_operands`` serves them in place of fresh draws. The served set
is checked against the fixture's recorded step-0 ``eps``; a shim that is subtly
wrong still produces plausible audio, but it does not reproduce that tensor.

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


def install_recorded_operands(diff, ref, device, dtype):
    """Serve the fixture's recorded random operands in place of fresh draws.

    The fork draws randomness in exactly three places on this path, and the fixture
    records all three, so the two lanes can be made to consume the same request
    instead of two independent draws from the same distribution:

    * ``modular_vibevoice_tokenizer.py`` gaussian sampling: ``randn(batch)``
      (``encode_draw0``) and ``randn_like(mean)`` (``encode_draw1``), for the
      voice-prompt VAE latent;
    * ``modeling_vibevoice_inference.py:sample_speech_tokens``: one
      ``randn(2, acoustic_vae_dim)`` per diffusion call (``callN_initial_noise``).

    Serving by shape rather than by call order means the shim is a no-op for any
    draw it does not recognise, and the returned ledger says which recorded
    operands were actually consumed. Returns ``(uninstall, ledger)``.
    """
    draw0 = torch.as_tensor(ref["encode_draw0"])
    draw1 = torch.as_tensor(ref["encode_draw1"])
    n_calls = int(diff["num_calls_recorded"])
    frame_noise = [torch.as_tensor(diff[f"call{i}_initial_noise"]) for i in range(n_calls)]

    served = {"encode_draw0": 0, "encode_draw1": 0, "frame_noise": 0, "unmatched_frames": 0}
    state = {"frame": 0}
    real_randn, real_randn_like = torch.randn, torch.randn_like
    frame_shape = tuple(int(s) for s in frame_noise[0].shape)
    draw0_shape = tuple(int(s) for s in draw0.shape)
    draw1_shape = tuple(int(s) for s in draw1.shape)

    def randn(*size, **kw):
        if kw.get("generator") is None:
            shape = tuple(int(s) for s in size)
            if shape == draw0_shape:
                served["encode_draw0"] += 1
                return draw0.to(device=device, dtype=dtype).clone()
            if shape == frame_shape:
                if state["frame"] < len(frame_noise):
                    out = frame_noise[state["frame"]]
                    state["frame"] += 1
                    served["frame_noise"] += 1
                    return out.to(device=device, dtype=dtype).clone()
                served["unmatched_frames"] += 1
        return real_randn(*size, **kw)

    def randn_like(t, **kw):
        if kw.get("generator") is None and tuple(int(s) for s in t.shape) == draw1_shape:
            served["encode_draw1"] += 1
            return draw1.to(device=t.device, dtype=t.dtype).clone()
        return real_randn_like(t, **kw)

    torch.randn, torch.randn_like = randn, randn_like

    def uninstall():
        torch.randn, torch.randn_like = real_randn, real_randn_like

    return uninstall, served


def install_eps_capture(model):
    """Capture the diffusion head's step-0 output so it can be checked against the
    fixture's ``call0_eps``. A served-noise shim that is subtly wrong still produces
    plausible audio; it does not reproduce the recorded eps."""
    captured: dict[str, np.ndarray] = {}
    inner = model.model.prediction_head

    class _Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        # The fork reads `prediction_head.device` directly, so the shim has to
        # answer for it rather than only for forward().
        @property
        def device(self):
            return self.inner.device

        def forward(self, *a, **kw):
            out = self.inner(*a, **kw)
            if "eps0" not in captured:
                captured["eps0"] = out.detach().float().cpu().numpy()
            return out

    model.model.prediction_head = _Capture()
    return captured


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

    # Share the recorded random operands with the HIP lane so the two rows describe
    # the same request, and check the served noise against the fixture's eps.
    diff_npz = np.load(fixtures / f"{request['name']}_diffusion.npz")
    ref_npz = np.load(fixtures / f"{request['name']}_reference.npz")
    uninstall, served = install_recorded_operands(diff_npz, ref_npz, device, dtype)
    captured = install_eps_capture(model)

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
    cold_served = dict(served)
    warm = [timed() for _ in range(max(args.repeats - 1, 1))]
    warm_wall = float(np.mean([w[0] for w in warm]))
    uninstall()

    # A served-noise shim that is subtly wrong still produces plausible audio; it does
    # not reproduce the fixture's recorded step-0 eps. This is the check that the two
    # lanes really are consuming the same request.
    ref_eps0 = diff_npz["call0_eps"][0].astype(np.float32)
    eps0 = captured.get("eps0")
    if eps0 is None or eps0.shape != ref_eps0.shape:
        eps_check = {"compared": False, "captured_shape": None if eps0 is None else list(eps0.shape),
                     "reference_shape": list(ref_eps0.shape)}
    else:
        absdiff = np.abs(eps0 - ref_eps0)
        scale = max(float(np.abs(ref_eps0).max()), 1e-6)
        eps_check = {
            "compared": True,
            "max_abs_diff": round(float(absdiff.max()), 6),
            "mean_abs_diff": round(float(absdiff.mean()), 6),
            "reference_abs_max": round(scale, 6),
            "max_abs_diff_relative": round(float(absdiff.max()) / scale, 6),
            "matches": bool(float(absdiff.max()) / scale <= 0.05),
            "basis": (
                "served noise must reproduce the fixture's call0_eps[0] to within 5% of "
                "its peak magnitude; the two lanes run bf16 with different kernel "
                "decompositions, so exact equality is not expected"
            ),
        }

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
            "random_draws": (
                "the fixture's recorded operands (encode_draw0/encode_draw1 and "
                "callN_initial_noise), served by shape, so both lanes consume the same "
                "request"
            ),
            "recorded_operands_served_cold_run": cold_served,
            "recorded_frames_available": int(diff_npz["num_calls_recorded"]),
            "eps_check": eps_check,
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
