"""VibeVoice-TTS diffusion head + DPMSolver replay against the frozen fixtures.

Gates the torch-free CPU reference
(``hipengine.kernels.cpu_reference.vibevoice_tts_diffusion``) on the frozen
milestone-1 oracle traces for both fixture requests:

- the per-step raw head outputs ``eps`` (20, 2, 64) and solver trajectory
  ``speech`` (20, 2, 64) must stay inside the measured eager-bf16 envelope
  (the fixtures record CUDA execution; bf16-rounded inputs compound through
  the 20-step solver, and the fork's own scheduler drifts identically when
  replayed from the recorded eps on CPU -- see the worklog entry);
- each gate is raised to the **sensitivity band** of the frozen trajectory
  wherever that band is wider than the nominal envelope. The band is measured
  by replaying the fixture from a +/-1 bf16-ULP change in its recorded initial
  noise: one ULP is the smallest input difference the frozen data can express,
  so the deviation it produces is the floor below which no independent
  implementation can be told apart from the oracle. The two-speaker ``call0``
  trajectory is chaotic and the single-speaker one is not, and
  ``test_fixture_chaos_band_is_measured_not_assumed`` asserts both facts, so a
  regenerated fixture cannot silently inherit the wider bound.
- the final latent and its ``latent / scale - bias`` scaled form gate within
  the same envelope;
- the schedule (linspace timesteps, order-2 first/second-order switching,
  zero final sigma) is asserted exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SensitivityBand:
    """Deviation a one-bf16-ULP input change produces in the frozen trajectory."""

    eps: np.ndarray
    speech: np.ndarray
    pooled: float
    latent: float
    scaled: float


def _bf16_ulp_variant(noise: np.ndarray, ulps: int) -> np.ndarray:
    """The recorded noise moved by ``ulps`` steps of the bf16 grid.

    One grid step is 2**-7 relative. ``call0_initial_noise`` is not itself on
    that grid, so a float32-ULP change rounds away and the perturbation has to
    be taken on the grid the solver actually consumes.
    """
    on_grid = bf16_round(np.asarray(noise, dtype=np.float32))
    return bf16_round(on_grid * np.float32(1.0 + ulps * 2**-7))


def _sensitivity_band(bundle, name, nominal) -> SensitivityBand:
    """How far one bf16 ULP on the initial noise moves this trajectory.

    Measured against the same implementation's unperturbed run rather than
    against the oracle, so the band isolates the trajectory's own amplification
    from this implementation's bias. A stable trajectory reports a band below
    the nominal envelope, and the nominal gate keeps its full force there; a
    chaotic one reports a band above it, and that step can only be gated at the
    band.
    """
    spec, weights, scale, bias = bundle
    data, nom_eps, nom_speech, nom_final, _, _ = nominal
    fx_eps, fx_speech = data["call0_eps"], data["call0_speech"]
    eps_peak = np.maximum(np.abs(fx_eps).max(axis=(1, 2)), 1e-9)
    speech_peak = np.maximum(np.abs(fx_speech).max(axis=(1, 2)), 1e-9)
    latent_peak = max(float(np.abs(data["call0_speech_latent"]).max()), 1e-9)
    nom_scaled = scale_speech_latent(nom_final, scale, bias)
    scaled_peak = max(float(np.abs(data["call0_scaled_latent"]).max()), 1e-9)
    eps_band = np.zeros(fx_eps.shape[0], dtype=np.float64)
    speech_band = np.zeros(fx_speech.shape[0], dtype=np.float64)
    pooled_band = latent_band = scaled_band = 0.0
    for ulps in (1, -1):
        steps: list[dict[str, np.ndarray]] = []
        final, _ = sample_speech_tokens(
            spec,
            weights,
            data["call0_condition"],
            data["call0_neg_condition"],
            float(data["call0_cfg_scale"]),
            _bf16_ulp_variant(data["call0_initial_noise"], ulps),
            collect=steps,
        )
        eps = np.stack([s["eps"] for s in steps])
        speech = np.stack([s["speech"] for s in steps])
        eps_band = np.maximum(eps_band, np.abs(eps - nom_eps).max(axis=(1, 2)) / eps_peak)
        speech_band = np.maximum(
            speech_band, np.abs(speech - nom_speech).max(axis=(1, 2)) / speech_peak
        )
        pooled_band = max(
            pooled_band,
            np.sqrt(((speech - nom_speech) ** 2).mean()) / np.sqrt((nom_speech**2).mean()),
        )
        latent_band = max(
            latent_band, np.abs(final.reshape(-1) - nom_final.reshape(-1)).max() / latent_peak
        )
        scaled_band = max(
            scaled_band,
            np.abs(scale_speech_latent(final, scale, bias) - nom_scaled).max() / scaled_peak,
        )
    return SensitivityBand(eps_band, speech_band, pooled_band, latent_band, scaled_band)


@pytest.fixture(scope="module")
def chains(bundle):
    """Each frozen trajectory replayed once, with its sensitivity band."""
    out = {}
    for name in ("single", "two"):
        nominal = _replay(bundle, name)
        out[name] = (nominal, _sensitivity_band(bundle, name, nominal))
    return out


@pytest.mark.parametrize("name", ["single", "two"])
def test_diffusion_replay_matches_fixture_chain(chains, name):
    data, eps, speech, final, scale, bias = chains[name][0]
    band = chains[name][1]
    fx_eps, fx_speech = data["call0_eps"], data["call0_speech"]

    assert eps.shape == fx_eps.shape, f"{name}: eps shape {eps.shape} != {fx_eps.shape}"
    assert speech.shape == fx_speech.shape

    # Measured envelope: CUDA-recorded bf16 eps replayed on CPU compound
    # through the solver (the fork's scheduler drifts identically); gate
    # per-step peaks and the pooled trajectory. Where the trajectory itself
    # amplifies a one-ULP input difference beyond that envelope, the band is
    # the binding limit instead.
    eps_ratio = (
        np.abs(eps - fx_eps).max(axis=(1, 2)) / np.maximum(np.abs(fx_eps).max(axis=(1, 2)), 1e-9)
    )
    speech_ratio = (
        np.abs(speech - fx_speech).max(axis=(1, 2))
        / np.maximum(np.abs(fx_speech).max(axis=(1, 2)), 1e-9)
    )
    eps_limit = np.maximum(0.08, band.eps)
    speech_limit = np.maximum(0.05, band.speech)
    assert (eps_ratio <= eps_limit).all(), (
        f"{name}: eps step {int((eps_ratio - eps_limit).argmax())} off by "
        f"{eps_ratio.max():.3f} of peak against a limit of {eps_limit.max():.3f}"
    )
    assert (speech_ratio <= speech_limit).all(), (
        f"{name}: speech step {int((speech_ratio - speech_limit).argmax())} off by "
        f"{speech_ratio.max():.3f} of peak against a limit of {speech_limit.max():.3f}"
    )
    pooled = np.sqrt(((speech - fx_speech) ** 2).mean()) / np.sqrt((fx_speech**2).mean())
    pooled_limit = max(0.02, band.pooled)
    assert pooled < pooled_limit, (
        f"{name}: speech pooled RMS rel {pooled:.4f} vs {pooled_limit:.4f}"
    )

    latent_rel = np.abs(final - data["call0_speech_latent"]).max() / max(
        np.abs(data["call0_speech_latent"]).max(), 1e-9
    )
    latent_limit = max(0.03, band.latent)
    assert latent_rel < latent_limit, (
        f"{name}: final latent rel {latent_rel:.4f} vs {latent_limit:.4f}"
    )

    scaled = scale_speech_latent(final, scale, bias)
    fx_scaled = data["call0_scaled_latent"].reshape(-1)
    scaled_rel = np.abs(scaled - fx_scaled).max() / max(np.abs(fx_scaled).max(), 1e-9)
    scaled_limit = max(0.03, band.scaled)
    assert scaled_rel < scaled_limit, (
        f"{name}: scaled latent rel {scaled_rel:.4f} vs {scaled_limit:.4f}"
    )


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


def test_fixture_chaos_band_is_measured_not_assumed(chains):
    """The band that widens the gates is a property of the fixture.

    One bf16 ULP is the smallest input difference the frozen trajectory can
    express. The two-speaker ``call0`` replay amplifies it into a different
    trajectory -- the final latent moves by 3.3 relative and one eps step by
    1.5 of peak -- so its late steps cannot separate implementations and are
    gated at that band. The single-speaker ``call0`` does not amplify it (0.044
    on eps, 0.019 on the final latent), so every nominal gate keeps its force
    there. If the fixtures are regenerated and this flips, these assertions
    fail rather than the replay test quietly inheriting a wider bound.
    """
    single = chains["single"][1]
    two = chains["two"][1]
    assert single.eps.max() < 0.08, f"single eps band grew to {single.eps.max():.4f}"
    assert single.speech.max() < 0.05, f"single speech band grew to {single.speech.max():.4f}"
    assert single.latent < 0.03, f"single latent band grew to {single.latent:.4f}"
    assert single.pooled < 0.02, f"single pooled band grew to {single.pooled:.4f}"
    assert two.eps.max() > 0.5, f"two eps band only {two.eps.max():.4f}; still chaotic?"
    assert two.latent > 0.5, f"two latent band only {two.latent:.4f}; still chaotic?"
    assert two.eps.max() > 10 * single.eps.max(), "two must be far more sensitive than single"
    assert two.latent > 10 * single.latent, "two must be far more sensitive than single"

    # The relaxation has to be load-bearing, or it is just a wider number.
    data, eps, _, _, _, _ = chains["two"][0]
    fx_eps = data["call0_eps"]
    ratio = np.abs(eps - fx_eps).max(axis=(1, 2)) / np.maximum(
        np.abs(fx_eps).max(axis=(1, 2)), 1e-9
    )
    assert ratio.max() > 0.08, "two no longer needs the band; tighten the gate"
