"""VibeVoice-TTS diffusion head + DPMSolver replay against the frozen fixtures.

Gates the torch-free CPU reference
(``hipengine.kernels.cpu_reference.vibevoice_tts_diffusion``) on the frozen
milestone-1 oracle traces for both fixture requests:

- every request is gated where the implementation is what is being measured.
  ``call0``'s first six steps run before the trajectory's own conditioning
  dominates, and there the per-step ``eps`` and ``speech`` stay inside a frozen
  envelope on both requests;
- the single-speaker request is additionally gated over the whole 20-step
  trajectory, and on the final latent and its ``latent / scale - bias`` scaled
  form, all against frozen thresholds;
- the two-speaker request is **not** gated past step 5. Its ``call0`` trajectory
  amplifies a one-bf16-ULP input change into a different trajectory -- 1.5 of
  peak on one eps step, 3.3 on the final latent -- so agreement there measures
  the fixture's conditioning, not this implementation.
  ``test_trajectory_conditioning_is_diagnostic`` reports that band and asserts
  that the reason for the missing gate still holds, but nothing it measures
  feeds an acceptance threshold. What covers that request is generated-audio
  quality, not a looser number here;
- the schedule (linspace timesteps, order-2 first/second-order switching,
  zero final sigma) is asserted exactly.

Acceptance thresholds below are frozen constants with recorded provenance. They
are deliberately not computed from the implementation under test: a limit
raised to the current run's own sensitivity grows whenever the implementation
gets less stable, which is the opposite of what a gate is for.
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

# Frozen acceptance thresholds, measured 2026-09-15 from the CPU reference
# against the committed fixtures (see worklog entry
# 20260915T101241.017890Z-lhl-vibevoice-tts-chain-bifurcation-48804e.md).
#
# The step-6 boundary is the amplification onset, not a tuning knob: on the
# single-speaker request the pre-step-6 eps spread is at most 0.013 and on the
# two-speaker one at most 0.007, while past it the two-speaker trajectory's own
# one-ULP band passes the nominal envelope. `_FROZEN_EPS_EARLY` is about 2x the
# worst measured pre-onset value so the gate still fails on a real head defect.
_PRE_AMPLIFICATION_STEPS = 6
_FROZEN_EPS_EARLY = 0.025
_FROZEN_SPEECH_EARLY = 0.025
# Single-speaker whole-trajectory envelope, carried from milestone 3: measured
# eps 0.050 and speech 0.020 at the last step, pooled 0.006, latent 0.011.
_FROZEN_EPS_SINGLE = 0.08
_FROZEN_SPEECH_SINGLE = 0.05
_FROZEN_POOLED_SINGLE = 0.02
_FROZEN_LATENT_SINGLE = 0.03

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


def _step_ratios(eps, speech, fx_eps, fx_speech):
    """Per-step peak deviation from the oracle, relative to the oracle's peak."""
    eps_ratio = np.abs(eps - fx_eps).max(axis=(1, 2)) / np.maximum(
        np.abs(fx_eps).max(axis=(1, 2)), 1e-9
    )
    speech_ratio = np.abs(speech - fx_speech).max(axis=(1, 2)) / np.maximum(
        np.abs(fx_speech).max(axis=(1, 2)), 1e-9
    )
    return eps_ratio, speech_ratio


@pytest.mark.parametrize("name", ["single", "two"])
def test_diffusion_replay_matches_fixture_chain(chains, name):
    data, eps, speech, final, scale, bias = chains[name][0]
    fx_eps, fx_speech = data["call0_eps"], data["call0_speech"]

    assert eps.shape == fx_eps.shape, f"{name}: eps shape {eps.shape} != {fx_eps.shape}"
    assert speech.shape == fx_speech.shape
    assert np.isfinite(eps).all() and np.isfinite(speech).all(), f"{name}: non-finite output"

    eps_ratio, speech_ratio = _step_ratios(eps, speech, fx_eps, fx_speech)

    # Both requests are gated here. Before the amplification onset the trajectory
    # has not yet turned this implementation's arithmetic difference from the
    # oracle into a different trajectory, so the comparison measures the head.
    early = slice(0, _PRE_AMPLIFICATION_STEPS)
    assert eps_ratio[early].max() <= _FROZEN_EPS_EARLY, (
        f"{name}: eps step {int(eps_ratio[early].argmax())} off by "
        f"{eps_ratio[early].max():.4f} of peak against {_FROZEN_EPS_EARLY}"
    )
    assert speech_ratio[early].max() <= _FROZEN_SPEECH_EARLY, (
        f"{name}: speech step {int(speech_ratio[early].argmax())} off by "
        f"{speech_ratio[early].max():.4f} of peak against {_FROZEN_SPEECH_EARLY}"
    )

    if name != "single":
        # Past the amplification onset the two-speaker replay is on a different
        # trajectory than the oracle: 0.37 of peak on eps at step 19, 0.27
        # relative on the final latent. No threshold on that comparison
        # separates a defect from the fixture's conditioning, so none is
        # asserted. The magnitude bounds are a smoke check that the head has not
        # collapsed; the request itself is covered by the generated-audio
        # quality suite rather than by a looser number here.
        assert np.abs(eps).max() <= 4 * np.abs(fx_eps).max(), f"{name}: eps magnitude blew up"
        assert np.abs(speech).max() <= 4 * np.abs(fx_speech).max(), (
            f"{name}: speech magnitude blew up"
        )
        return

    assert eps_ratio.max() <= _FROZEN_EPS_SINGLE, (
        f"{name}: eps step {int(eps_ratio.argmax())} off by {eps_ratio.max():.4f} "
        f"of peak against {_FROZEN_EPS_SINGLE}"
    )
    assert speech_ratio.max() <= _FROZEN_SPEECH_SINGLE, (
        f"{name}: speech step {int(speech_ratio.argmax())} off by "
        f"{speech_ratio.max():.4f} of peak against {_FROZEN_SPEECH_SINGLE}"
    )
    pooled = np.sqrt(((speech - fx_speech) ** 2).mean()) / np.sqrt((fx_speech**2).mean())
    assert pooled < _FROZEN_POOLED_SINGLE, (
        f"{name}: speech pooled RMS rel {pooled:.4f} vs {_FROZEN_POOLED_SINGLE}"
    )

    latent_rel = np.abs(final - data["call0_speech_latent"]).max() / max(
        np.abs(data["call0_speech_latent"]).max(), 1e-9
    )
    assert latent_rel < _FROZEN_LATENT_SINGLE, (
        f"{name}: final latent rel {latent_rel:.4f} vs {_FROZEN_LATENT_SINGLE}"
    )

    scaled = scale_speech_latent(final, scale, bias)
    fx_scaled = data["call0_scaled_latent"].reshape(-1)
    scaled_rel = np.abs(scaled - fx_scaled).max() / max(np.abs(fx_scaled).max(), 1e-9)
    assert scaled_rel < _FROZEN_LATENT_SINGLE, (
        f"{name}: scaled latent rel {scaled_rel:.4f} vs {_FROZEN_LATENT_SINGLE}"
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


def test_trajectory_conditioning_is_diagnostic(chains):
    """Reports why the two-speaker request carries no late-step gate.

    Diagnostic only -- nothing measured here feeds an acceptance threshold. One
    bf16 ULP is the smallest input difference the frozen trajectory can express,
    so the deviation it produces is the floor below which no independent
    implementation can be told apart from the oracle. The single-speaker
    ``call0`` stays inside the nominal envelope under that perturbation; the
    two-speaker one does not.

    The assertions pin the *reason* the late gates are absent. If the fixtures
    are regenerated and the two-speaker request becomes well-conditioned, this
    fails and asks for the late-step gate to be restored, instead of leaving a
    silent hole in coverage.
    """
    single = chains["single"][1]
    two = chains["two"][1]
    assert two.eps.max() > 10 * single.eps.max(), "two must be far more sensitive than single"
    assert two.latent > 10 * single.latent, "two must be far more sensitive than single"
    assert two.eps.max() > 0.5, f"two eps band only {two.eps.max():.4f}; still chaotic?"
    assert two.latent > 0.5, f"two latent band only {two.latent:.4f}; still chaotic?"

    # The single-speaker trajectory is well-conditioned, so its whole-trajectory
    # gate is doing work rather than riding a chaotic band.
    assert single.eps.max() < _FROZEN_EPS_SINGLE, f"single eps band grew to {single.eps.max():.4f}"
    assert single.latent < _FROZEN_LATENT_SINGLE, (
        f"single latent band grew to {single.latent:.4f}"
    )

    # ... and the two-speaker late steps still cannot be gated nominally.
    data, eps, _, _, _, _ = chains["two"][0]
    fx_eps = data["call0_eps"]
    late_ratio = (
        np.abs(eps - fx_eps).max(axis=(1, 2))
        / np.maximum(np.abs(fx_eps).max(axis=(1, 2)), 1e-9)
    )[_PRE_AMPLIFICATION_STEPS:]
    assert late_ratio.max() > _FROZEN_EPS_SINGLE, (
        "two late steps now fit the nominal gate; restore it"
    )
