"""GPU parity test for the VibeVoice-TTS diffusion head HIP runtime.

Requires the cached ``microsoft/VibeVoice-1.5B`` snapshot, a working ROCm
stack, and the frozen schema-2 oracle fixtures. Skipped otherwise.

Gates (mirroring the decoder lane):
- per-step head outputs and solver trajectory vs the numpy CPU reference
  within the measured eager-bf16 envelope (fp32 accumulation-order noise
  compounding through the 20-step solver);
- final latent and its scaled form vs the frozen fixture chain;
- replay determinism (two runs bit-identical);
- the head primitives resolve through the four-axis registry.
"""

from __future__ import annotations

import ctypes
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


@pytest.fixture(scope="module")
def bundle():
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_tts import load_vibevoice_tts_diffusion_head

    try:
        return load_vibevoice_tts_diffusion_head(resolve_model_path(PINNED_MODEL_ID))
    except (FileNotFoundError, ValueError):
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache", allow_module_level=True)


@pytest.fixture(scope="module")
def gpu_head(bundle):
    from hipengine.runtime.vibevoice_tts_diffusion import VibevoiceTTSDiffusionHeadGPU

    spec, weights, _, _ = bundle
    head = VibevoiceTTSDiffusionHeadGPU(spec, weights)
    yield head
    head.close()


def _replay(head, data, collect):
    return head.sample_speech_tokens(
        data["call0_condition"],
        data["call0_neg_condition"],
        float(data["call0_cfg_scale"]),
        data["call0_initial_noise"],
        collect=collect,
    )


@pytest.mark.parametrize("name", ["single", "two"])
def test_gpu_diffusion_replay_parity(bundle, gpu_head, name):
    import hipengine.kernels.cpu_reference.vibevoice_tts_diffusion as diff_ref

    spec, weights, scale, bias = bundle
    data = np.load(FIXTURE_DIR / f"{name}_diffusion.npz")

    steps: list[dict[str, np.ndarray]] = []
    final_g, _ = _replay(gpu_head, data, steps)
    eps_g = np.stack([s["eps"] for s in steps])
    speech_g = np.stack([s["speech"] for s in steps])

    csteps: list[dict[str, np.ndarray]] = []
    final_c, _ = diff_ref.sample_speech_tokens(
        spec,
        weights,
        data["call0_condition"],
        data["call0_neg_condition"],
        float(data["call0_cfg_scale"]),
        data["call0_initial_noise"],
        collect=csteps,
    )
    eps_c = np.stack([s["eps"] for s in csteps])
    speech_c = np.stack([s["speech"] for s in csteps])

    eps_ratio = (
        np.abs(eps_g - eps_c).max(axis=(1, 2)) / np.maximum(np.abs(eps_c).max(axis=(1, 2)), 1e-9)
    )
    speech_ratio = (
        np.abs(speech_g - speech_c).max(axis=(1, 2))
        / np.maximum(np.abs(speech_c).max(axis=(1, 2)), 1e-9)
    )
    assert eps_ratio.max() < 0.10, (
        f"{name}: GPU eps step {int(eps_ratio.argmax())} off CPU by {eps_ratio.max():.3f} of peak"
    )
    assert speech_ratio.max() < 0.05, (
        f"{name}: GPU speech step {int(speech_ratio.argmax())} off CPU by {speech_ratio.max():.3f}"
    )

    latent_peak = max(np.abs(final_c).max(), 1e-9)
    assert np.abs(final_g - final_c).max() / latent_peak < 0.05, (
        f"{name}: GPU final latent off CPU by "
        f"{np.abs(final_g - final_c).max() / latent_peak:.4f} relative"
    )

    # Fixture chain: the frozen oracle latent and its scaled decoder input.
    fx_rel = np.abs(final_g - data["call0_speech_latent"]).max() / max(
        np.abs(data["call0_speech_latent"]).max(), 1e-9
    )
    assert fx_rel < 0.06, f"{name}: GPU final latent off fixture by {fx_rel:.4f} relative"
    scaled_g = diff_ref.scale_speech_latent(final_g, scale, bias)
    fx_scaled = data["call0_scaled_latent"].reshape(-1)
    scaled_rel = np.abs(scaled_g - fx_scaled).max() / max(np.abs(fx_scaled).max(), 1e-9)
    assert scaled_rel < 0.08, f"{name}: GPU scaled latent off fixture by {scaled_rel:.4f} relative"


def test_gpu_diffusion_replay_is_deterministic(bundle, gpu_head):
    data = np.load(FIXTURE_DIR / "single_diffusion.npz")
    a: list[dict[str, np.ndarray]] = []
    final_a, _ = _replay(gpu_head, data, a)
    b: list[dict[str, np.ndarray]] = []
    final_b, _ = _replay(gpu_head, data, b)
    assert np.array_equal(final_a.view(np.uint32), final_b.view(np.uint32))
    for sa, sb in zip(a, b):
        assert np.array_equal(sa["eps"].view(np.uint32), sb["eps"].view(np.uint32))
        assert np.array_equal(sa["speech"].view(np.uint32), sb["speech"].view(np.uint32))


def test_gpu_head_step0_matches_cpu_reference(bundle, gpu_head):
    """Isolated head call: duplicated first branch at the first timestep."""
    import hipengine.kernels.cpu_reference.vibevoice_tts_diffusion as diff_ref
    from hipengine.runtime.vibevoice_tts_diffusion import _bf16_u16

    spec, weights, _, _ = bundle
    data = np.load(FIXTURE_DIR / "single_diffusion.npz")
    cond = np.concatenate([data["call0_condition"], data["call0_neg_condition"]], axis=0)
    noise = data["call0_initial_noise"]
    combined = np.concatenate([noise[:1], noise[:1]], axis=0)
    t0 = int(data["call0_scheduler_timesteps"][0])

    gpu_eps = gpu_head.forward(_bf16_u16(combined), t0, _bf16_u16(cond))
    cpu_eps = diff_ref.diffusion_head_forward(spec, weights, combined, np.full(2, t0), cond)
    peak = max(np.abs(cpu_eps).max(), 1e-9)
    assert np.abs(gpu_eps - cpu_eps).max() / peak < 0.02, (
        f"isolated head call off CPU by {np.abs(gpu_eps - cpu_eps).max() / peak:.4f} of peak"
    )


def test_gpu_diffusion_resolves_through_registry():
    from hipengine.kernels.vibevoice import resolve_vibevoice_kernels

    ops = resolve_vibevoice_kernels()
    for name in (
        "vv_diff_rmsnorm_bf16",
        "vv_diff_silu_bf16",
        "vv_diff_add_bf16",
        "vv_diff_modulate_bf16",
        "vv_diff_gated_residual_bf16",
        "vv_diff_cfg_combine_bf16",
    ):
        assert callable(getattr(ops, name)), f"{name} not registered in the vibevoice family"
