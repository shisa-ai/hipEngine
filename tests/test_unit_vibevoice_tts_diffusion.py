"""VibeVoice-TTS diffusion head + DPMSolver replay against the frozen fixtures.

Gates the torch-free CPU reference
(``hipengine.kernels.cpu_reference.vibevoice_tts_diffusion``) on the frozen
milestone-1 oracle traces for both fixture requests:

- the per-step raw head outputs ``eps`` (20, 2, 64) and solver trajectory
  ``speech`` (20, 2, 64) must stay inside the measured eager-bf16 envelope
  (the fixtures record CUDA execution; bf16-rounded inputs compound through
  the 20-step solver, and the fork's own scheduler drifts identically when
  replayed from the recorded eps on CPU -- see the worklog entry);
- the final latent and its ``latent / scale - bias`` scaled form gate within
  the same envelope;
- the schedule (linspace timesteps, order-2 first/second-order switching,
  zero final sigma) is asserted exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.maple import bf16_round
from hipengine.kernels.cpu_reference.vibevoice_tts_diffusion import (
    DPMSolverMultistepScheduler,
    sample_speech_tokens,
    scale_speech_latent,
    timestep_embedding,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vibevoice_tts"
PINNED_MODEL_ID = "microsoft/VibeVoice-1.5B"

if not (FIXTURE_DIR / "manifest.json").is_file():
    pytest.skip("VibeVoice-TTS trace fixtures not present", allow_module_level=True)


@pytest.fixture(scope="module")
def bundle():
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_tts import load_vibevoice_tts_diffusion_head

    try:
        return load_vibevoice_tts_diffusion_head(resolve_model_path(PINNED_MODEL_ID))
    except (FileNotFoundError, ValueError):
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache", allow_module_level=True)


def _replay(bundle, name):
    spec, weights, scale, bias = bundle
    data = np.load(FIXTURE_DIR / f"{name}_diffusion.npz")
    steps: list[dict[str, np.ndarray]] = []
    final, _ = sample_speech_tokens(
        spec,
        weights,
        data["call0_condition"],
        data["call0_neg_condition"],
        float(data["call0_cfg_scale"]),
        data["call0_initial_noise"],
        collect=steps,
    )
    eps = np.stack([s["eps"] for s in steps])
    speech = np.stack([s["speech"] for s in steps])
    return data, eps, speech, final, scale, bias


@pytest.mark.parametrize("name", ["single", "two"])
def test_diffusion_replay_matches_fixture_chain(bundle, name):
    data, eps, speech, final, scale, bias = _replay(bundle, name)
    fx_eps, fx_speech = data["call0_eps"], data["call0_speech"]

    assert eps.shape == fx_eps.shape, f"{name}: eps shape {eps.shape} != {fx_eps.shape}"
    assert speech.shape == fx_speech.shape

    # Measured envelope: CUDA-recorded bf16 eps replayed on CPU compound
    # through the solver (the fork's scheduler drifts identically); gate
    # per-step peaks and the pooled trajectory.
    eps_ratio = (
        np.abs(eps - fx_eps).max(axis=(1, 2)) / np.maximum(np.abs(fx_eps).max(axis=(1, 2)), 1e-9)
    )
    speech_ratio = (
        np.abs(speech - fx_speech).max(axis=(1, 2))
        / np.maximum(np.abs(fx_speech).max(axis=(1, 2)), 1e-9)
    )
    assert eps_ratio.max() < 0.08, f"{name}: eps step {int(eps_ratio.argmax())} off by {eps_ratio.max():.3f} of peak"
    assert speech_ratio.max() < 0.05, (
        f"{name}: speech step {int(speech_ratio.argmax())} off by {speech_ratio.max():.3f} of peak"
    )
    pooled = np.sqrt(((speech - fx_speech) ** 2).mean()) / np.sqrt((fx_speech**2).mean())
    assert pooled < 0.02, f"{name}: speech pooled RMS rel {pooled:.4f}"

    latent_rel = np.abs(final - data["call0_speech_latent"]).max() / max(
        np.abs(data["call0_speech_latent"]).max(), 1e-9
    )
    assert latent_rel < 0.03, f"{name}: final latent rel {latent_rel:.4f}"

    scaled = scale_speech_latent(final, scale, bias)
    fx_scaled = data["call0_scaled_latent"].reshape(-1)
    scaled_rel = np.abs(scaled - fx_scaled).max() / max(np.abs(fx_scaled).max(), 1e-9)
    assert scaled_rel < 0.03, f"{name}: scaled latent rel {scaled_rel:.4f}"


def test_scheduler_schedule_matches_frozen_timesteps(bundle):
    spec, _, _, _ = bundle
    data = np.load(FIXTURE_DIR / "single_diffusion.npz")
    sched = DPMSolverMultistepScheduler(spec)
    sched.set_timesteps(spec.num_inference_steps)
    np.testing.assert_array_equal(
        sched.timesteps, data["call0_scheduler_timesteps"].astype(np.int64)
    )
    assert float(sched.sigmas[-1]) == 0.0, "final sigma must be zero (final_sigmas_type)"
    assert sched.num_inference_steps == 20


def test_scheduler_step_rejects_out_of_order_timestep(bundle):
    spec, _, _, _ = bundle
    sched = DPMSolverMultistepScheduler(spec)
    sched.set_timesteps(20)
    t0 = int(sched.timesteps[0])
    with pytest.raises(ValueError, match="does not match schedule position"):
        sched.step(np.zeros((2, 64), dtype=np.float32), t0 + 1, np.zeros((2, 64), dtype=np.float32))


def test_solver_order_switches_first_second_first(bundle):
    """Step 0 and the zero-sigma final step are first order; the middle is 2S."""
    spec, _, _, _ = bundle
    sched = DPMSolverMultistepScheduler(spec)
    sched.set_timesteps(20)
    assert spec.final_sigmas_type == "zero" and spec.solver_order == 2

    eps = np.zeros((2, 64), dtype=np.float32)
    sample = np.zeros((2, 64), dtype=np.float32)
    sched.step(eps, int(sched.timesteps[0]), sample)
    assert sched.lower_order_nums == 1, "first step must consume the first-order path"
    sched.step(eps, int(sched.timesteps[1]), sample)
    assert sched.lower_order_nums == 2
    assert sched.model_outputs[0] is not None, "second-order step requires two model outputs"


def _scalar(v) -> float:
    return float(np.asarray(bf16_round(np.float32(v))).reshape(-1)[0])


def test_timestep_embedding_uses_bf16_cast_timestep():
    """999 must round to the bf16 grid (1000) before the sinusoid (eager)."""
    assert _scalar(999.0) == 1000.0
    np.testing.assert_array_equal(
        timestep_embedding(_scalar(999.0), 256), timestep_embedding(1000.0, 256)
    )
    # 949 rounds DOWN to 948 in bf16.
    assert _scalar(949.0) == 948.0
    np.testing.assert_array_equal(
        timestep_embedding(_scalar(949.0), 256), timestep_embedding(948.0, 256)
    )


def test_replay_is_deterministic(bundle):
    _, eps_a, speech_a, final_a, _, _ = _replay(bundle, "single")
    _, eps_b, speech_b, final_b, _, _ = _replay(bundle, "single")
    assert np.array_equal(eps_a.view(np.uint32), eps_b.view(np.uint32))
    assert np.array_equal(speech_a.view(np.uint32), speech_b.view(np.uint32))
    assert np.array_equal(final_a.view(np.uint32), final_b.view(np.uint32))
