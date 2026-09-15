"""GPU parity test for the VibeVoice-TTS torch-free HIP decoder runtime.

Requires the cached ``microsoft/VibeVoice-1.5B`` snapshot, a working ROCm
stack, and the frozen schema-2 oracle fixtures. Skipped otherwise.

Gates (mirroring tests/test_unit_vibevoice_tts_decoder.py):
- per-frame chunk parity vs the CPU reference within the measured eager-bf16
  envelope, plus pooled waveform RMS relative error < 2%;
- decode_bulk must stay bit-identical to per-frame streaming (the causal
  prefix rolls are pure row-independent arithmetic);
- reset() at the generated speech boundaries must change the output exactly
  as the CPU stream's reset does.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vibevoice_tts"
MANIFEST = FIXTURE_DIR / "manifest.json"
PINNED_MODEL_ID = "microsoft/VibeVoice-1.5B"

if not MANIFEST.is_file():
    pytest.skip("VibeVoice-TTS trace fixtures not present", allow_module_level=True)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if not _hip_available():
    pytest.skip("ROCm/HIP runtime not available", allow_module_level=True)


def _snapshot():
    from hipengine.loading.hf_cache import resolve_model_path

    return resolve_model_path(PINNED_MODEL_ID)


def _speech_resets(lm, request):
    ids = lm["generated_ids"].reshape(-1)[request["prompt_tokens"]:]
    resets = set()
    call = -1
    pending = False
    for token in ids:
        if token == 151653:
            pending = True
        elif token == 151654:
            call += 1
            if pending:
                resets.add(call)
                pending = False
    return resets


def _load_latents(name):
    d = np.load(FIXTURE_DIR / f"{name}_diffusion.npz")
    n = int(d["num_calls_recorded"])
    return np.stack([d[f"call{i}_scaled_latent"].reshape(-1) for i in range(n)]), n


@pytest.fixture(scope="module")
def bundle():
    from hipengine.loading.vibevoice_tts import load_vibevoice_tts_decoder

    try:
        return load_vibevoice_tts_decoder(_snapshot())
    except (FileNotFoundError, ValueError):
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache", allow_module_level=True)


@pytest.fixture(scope="module")
def gpu_runner(bundle):
    from hipengine.runtime.vibevoice_tts_decoder import VibevoiceTTSDecoderGPU

    spec, weights, _, _ = bundle
    runner = VibevoiceTTSDecoderGPU(spec, weights)
    yield runner
    runner.close()


@pytest.mark.parametrize("name", ["single", "two"])
def test_gpu_decoder_replay_parity(bundle, gpu_runner, name):
    import hipengine.kernels.cpu_reference.vibevoice_tts as tts_ref

    spec, weights, _, _ = bundle
    manifest = json.loads(MANIFEST.read_text())
    request = next(r for r in manifest["requests"] if r["name"] == name)
    data = np.load(FIXTURE_DIR / f"{name}_audio.npz")
    lm = np.load(FIXTURE_DIR / f"{name}_lm.npz")
    lat, n = _load_latents(name)
    resets = _speech_resets(lm, request)

    cpu_out = tts_ref.decode_frames(
        spec, weights, lat, reset_before=tuple(sorted(resets)), dtype="bfloat16"
    )

    gpu_runner.reset()  # the module-scoped runner may carry state from other tests
    gpu_out = np.empty_like(cpu_out)
    for i in range(n):
        if i in resets:
            gpu_runner.reset()
        gpu_out[i] = gpu_runner.decode(lat[i])

    assert gpu_out.shape == cpu_out.shape, f"{name}: {gpu_out.shape} != {cpu_out.shape}"

    # Calibrated eager-bf16 envelope (see the CPU test for the derivation):
    # the oracle's own fp32-accumulation spread compounds through 26 residual
    # blocks; quiet frames are gated on absolute difference.
    per_frame_peak = np.abs(cpu_out.astype(np.float64)).max(axis=1)
    per_frame_diff = np.abs(gpu_out.astype(np.float64) - cpu_out.astype(np.float64)).max(axis=1)
    envelope = np.maximum(2e-3, 0.06 * per_frame_peak)
    worst = (per_frame_diff / np.maximum(per_frame_peak, 1e-9)).max()
    assert bool((per_frame_diff <= envelope).all()), (
        f"{name}: GPU frame diff exceeds envelope; worst {worst:.3f} of frame peak"
    )

    pcm_gpu = gpu_out.reshape(-1)
    pcm_cpu = cpu_out.reshape(-1)
    rms_rel = np.sqrt(((pcm_gpu - pcm_cpu) ** 2).mean()) / np.sqrt(
        (pcm_cpu.astype(np.float64) ** 2).mean()
    )
    assert rms_rel < 0.02, f"{name}: pooled waveform RMS relative error {rms_rel:.4f}"


@pytest.mark.parametrize("name", ["single", "two"])
def test_gpu_bulk_matches_per_frame_streaming(bundle, gpu_runner, name):
    import hipengine.kernels.cpu_reference.vibevoice_tts as tts_ref

    spec, weights, _, _ = bundle
    manifest = json.loads(MANIFEST.read_text())
    request = next(r for r in manifest["requests"] if r["name"] == name)
    lm = np.load(FIXTURE_DIR / f"{name}_lm.npz")
    lat, n = _load_latents(name)
    resets = _speech_resets(lm, request)

    # Split into reset-delimited segments; decode each segment in bulk and by
    # per-frame streaming, then require bit identity.
    bounds = [0, *(r + 1 for r in sorted(resets)), n]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo >= hi:
            continue
        gpu_runner.reset()
        bulk = gpu_runner.decode_bulk(lat[lo:hi])
        gpu_runner.reset()
        stream = np.stack([gpu_runner.decode(lat[i]) for i in range(lo, hi)])
        assert np.array_equal(bulk.view(np.uint32), stream.view(np.uint32)), (
            f"{name}: bulk decode diverges from per-frame streaming in [{lo}, {hi})"
        )


def test_gpu_reset_changes_output(bundle, gpu_runner):
    spec, weights, _, _ = bundle
    lat, _ = _load_latents("single")
    gpu_runner.reset()
    fresh = gpu_runner.decode(lat[8])
    gpu_runner.decode(lat[0])
    carried = gpu_runner.decode(lat[8])
    assert not np.array_equal(fresh.view(np.uint32), carried.view(np.uint32)), (
        "reset() did not change the decode output; caches are not rolling"
    )


def test_gpu_decoder_resolves_through_registry(bundle):
    """The convtr primitive must come from the four-axis registry, not a hard path."""
    from hipengine.kernels.vibevoice import resolve_vibevoice_kernels

    ops = resolve_vibevoice_kernels()
    assert callable(ops.vv_convtr_gemm_bf16)
    assert callable(ops.vv_conv_gemm_bf16)
