"""VibeVoice-TTS session on HIP/GPU against the frozen torch-oracle chain.

Gates, in execution order: the voice-prompt entry point (called directly, plus
its checkpoint scale/bias step), the session's own negative-LM branch with
recorded noise and no injected condition, the 121-position prefill trace, the
constrained greedy chain (exact), the negative-LM accumulation under injected
conditions, per-frame diffusion latents, and the decoded PCM.
Skips without HIP or the local HF artifact / fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

if not hip_runtime_available():
    pytest.skip("no usable HIP runtime for VibeVoice TTS session tests", allow_module_level=True)

from hipengine.loading.vibevoice_tts_session import load_vibevoice_tts_session
from hipengine.runtime.vibevoice_tts_session import VibevoiceTtsSession, SessionTrace

FIXTURES = Path(__file__).parent / "fixtures" / "vibevoice_tts"
PINNED_HF_MODEL_ID = "microsoft/VibeVoice-1.5B"

if not FIXTURES.is_dir():
    pytest.skip("VibeVoice TTS fixtures not present", allow_module_level=True)


def _hf_snapshot_or_skip():
    from hipengine.loading.hf_cache import resolve_model_path

    try:
        return resolve_model_path(PINNED_HF_MODEL_ID)
    except Exception:
        pytest.skip(f"{PINNED_HF_MODEL_ID} not in local HF cache")


def _npz(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / name) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def weights():
    yield load_vibevoice_tts_session(_hf_snapshot_or_skip())


@pytest.fixture(scope="module")
def session(weights):
    sess = VibevoiceTtsSession(weights, max_context=256)
    yield sess
    sess.close()


@pytest.fixture(scope="module")
def lm() -> dict[str, np.ndarray]:
    return _npz("single_lm.npz")


@pytest.fixture(scope="module")
def dif() -> dict[str, np.ndarray]:
    return _npz("single_diffusion.npz")


def _prompt_rows(sess, lm):
    in_ids = np.asarray(lm["input_ids"])[0]
    mask = np.asarray(lm["speech_input_mask"], dtype=bool).reshape(-1)
    conn = np.asarray(lm["prefill_connected"])
    return sess.build_prompt_rows(in_ids, mask, conn)


def _run_chain(session, lm, dif, trace: SessionTrace):
    """The frozen chain with recorded noise + negative conditions injected."""
    rows = _prompt_rows(session, lm)
    return session.generate(
        rows,
        cfg_scale=1.3,
        max_new_tokens=27,
        noise_hook=lambda i: dif[f"call{i}_initial_noise"],
        neg_hook=lambda i: dif[f"call{i}_neg_condition"],
        trace=trace,
    )


def test_voice_prompt_rows_matches_reference(session) -> None:
    """ref_pcm -> voice_prompt_rows -> sampled latents + connected rows.

    This calls the entry point directly. An earlier version reconstructed the
    encode/sample/scale/connect chain inline, which hid three defects in
    ``voice_prompt_rows``: a stray ``[0]`` on an already ``(frames, hidden)``
    result, a padding expression that appended a whole extra frame when the
    reference was already 3200-aligned, and an RNG that only existed after
    ``generate()`` ran.
    """
    ref = _npz("single_reference.npz")
    pcm = np.asarray(ref["ref_pcm"])[0]
    ref_lat = np.asarray(ref["encode0_latents"]).reshape(-1, 64)
    frames = ref_lat.shape[0]
    sampled, connected = session.voice_prompt_rows(
        pcm,
        noise=np.asarray(ref["encode_draw1"]).reshape(frames, 64),
        noise_scale=np.asarray(ref["encode_draw0"]).reshape(1),
    )
    assert sampled.shape == (frames, 64), f"{sampled.shape} != {(frames, 64)}"
    peak = float(np.abs(ref_lat).max())
    # The whole frame range now meets one gate. Frames 0..68 sit on the fork's
    # GPU-bf16 noise floor; the tail frame was 0.39 before the encoder
    # reproduced the fork's per-stage right zero padding and is now below the
    # interior frames, so it needs no separate tolerance.
    assert np.abs(sampled - ref_lat).max() / peak <= 0.045, "sampled latents drifted"
    assert np.abs(sampled[-1] - ref_lat[-1]).max() / peak <= 0.045, "tail frame drifted"

    ref_conn = np.asarray(ref["connected"])
    assert connected.shape == ref_conn.shape, f"{connected.shape} != {ref_conn.shape}"
    assert np.abs(connected[:-1] - ref_conn[:-1]).max() <= 0.06 * float(
        np.abs(ref_conn[:-1]).max()
    ), "connected rows drifted"


def test_session_negative_path_without_injection(session, lm, dif) -> None:
    """Recorded noise only: the session's own negative-LM branch is gated.

    ``neg_hook`` is deliberately omitted, so ``_negative_condition`` runs and
    the negative LM accumulates the positive pass's feedback embeddings. The
    generated token chain does not catch a broken negative branch, because the
    chain stays exact even when the branch is wrong; the conditions do.
    """
    rows = _prompt_rows(session, lm)
    trace = SessionTrace()
    res = session.generate(
        rows,
        cfg_scale=1.3,
        max_new_tokens=27,
        noise_hook=lambda i: dif[f"call{i}_initial_noise"],
        trace=trace,
    )
    gen = np.asarray(lm["generated_ids"])[0]
    in_ids = np.asarray(lm["input_ids"])[0]
    assert res.ids == [int(t) for t in gen[len(in_ids):]], "chain moved off the oracle"
    assert len(trace.neg_conditions) == 25
    for call in (0, 1, 2, 12, 24):
        ref = np.asarray(dif[f"call{call}_neg_condition"]).reshape(-1)
        got = trace.neg_conditions[call]
        peak = float(np.abs(ref).max())
        # Measured: 0.014 / 0.012 / 0.020 at calls 0/1/2 and 0.111 / 0.060 at
        # calls 12/24. Feeding the speech-start embedding every frame instead
        # of the accumulated feedback drives these to 1.5-2.2.
        limit = 0.06 if call <= 2 else 0.15
        assert np.abs(got - ref).max() / peak <= limit, f"call{call} negative condition drifted"


def test_voice_prompt_scaled_features_match_reference(session, weights) -> None:
    """The checkpoint scale/bias step, checked against ``features_scaled``.

    ``sampled`` comes from ``voice_prompt_rows``; only the documented
    ``(latent + bias) * scale`` arithmetic is applied here, so this does not
    reconstruct the encode path that the direct test already covers.
    """
    ref = _npz("single_reference.npz")
    pcm = np.asarray(ref["ref_pcm"])[0]
    ref_lat = np.asarray(ref["encode0_latents"]).reshape(-1, 64)
    frames = ref_lat.shape[0]
    sampled, _ = session.voice_prompt_rows(
        pcm,
        noise=np.asarray(ref["encode_draw1"]).reshape(frames, 64),
        noise_scale=np.asarray(ref["encode_draw0"]).reshape(1),
    )
    scaled = (sampled + np.float32(weights.speech_bias_factor)) * np.float32(
        weights.speech_scaling_factor
    )
    ref_scaled = np.asarray(ref["features_scaled"]).reshape(frames, 64)
    # Measured envelope: the encoder noise floor propagated through the
    # recorded-draw sampling (0.87 abs times scale is about 0.17 on this input).
    assert np.abs(scaled[:-1] - ref_scaled[:-1]).max() <= 0.2, "scaled features drifted"


def test_prefill_trace_matches_fixture(session, lm) -> None:
    """Every prompt position's post-final-norm hidden stays on the oracle."""
    ref_h = np.asarray(lm["prefill_last_hidden"])[0]
    rows = _prompt_rows(session, lm)
    session.positive.reset()
    worst = 0.0
    for position, row in enumerate(rows):
        session.positive.push_token(row, position)
        session.positive.forward_layers(position)
        h = session.positive.hidden_state()
        worst = max(worst, float(np.abs(h - ref_h[position]).max() / np.abs(ref_h[position]).max()))
    assert worst <= 0.05, f"prefill hidden drift {worst:.4f}"


def test_greedy_chain_matches_torch(session, lm, dif) -> None:
    """The exact 27-token constrained chain, RNG-independent via fixtures."""
    trace = SessionTrace()
    res = _run_chain(session, lm, dif, trace)
    gen = np.asarray(lm["generated_ids"])[0]
    in_ids = np.asarray(lm["input_ids"])[0]
    expected = [int(t) for t in gen[len(in_ids):]]
    assert res.ids == expected, f"{res.ids} != {expected}"
    assert len(res.chunks) == 25


def test_session_conditions_track_oracle(session, lm, dif) -> None:
    """Positive/negative conditions stay on the oracle through the span."""
    trace = SessionTrace()
    _run_chain(session, lm, dif, trace)
    c0 = np.asarray(dif["call0_condition"]).reshape(-1)
    peak_c = float(np.abs(c0).max())
    got0 = trace.conditions[0]
    assert np.abs(got0 - c0).max() / peak_c <= 0.06, "call0 positive condition drifted"
    n0 = np.asarray(dif["call0_neg_condition"]).reshape(-1)
    peak_n = float(np.abs(n0).max())
    gotn0 = trace.neg_conditions[0]
    assert np.abs(gotn0 - n0).max() / peak_n <= 0.06, "call0 negative condition drifted"
    # The negative LM accumulates the session's own feedback rows; deeper
    # calls inherit the positive-path offset, so gate at a wider envelope.
    for call in (1, 12, 24):
        ref = np.asarray(dif[f"call{call}_neg_condition"]).reshape(-1)
        got = trace.neg_conditions[call]
        peak = float(np.abs(ref).max())
        assert np.abs(got - ref).max() / peak <= 0.15, f"call{call} negative condition drifted"
    # Diffusion latents under the offset conditions: the trajectory
    # compounds the voice-sampling offset with call depth (measured 0.018
    # at call0 up to 0.336 at call24) — the same inherent bf16 compounding
    # class the diffusion fixture drift was attributed to in milestone 3.
    for call in (0, 12, 24):
        ref = np.asarray(dif[f"call{call}_speech_latent"]).reshape(-1)
        got = trace.speech_latents[call]
        assert np.abs(got - ref).max() <= 0.4, f"call{call} speech latent drifted"


def test_session_pcm_matches_oracle(session, lm, dif) -> None:
    """Decoded chunks track the oracle's PCM: waveform within the measured
    compounding envelope and pooled energy within 25%."""
    audio = _npz("single_audio.npz")
    ref_chunks = np.asarray(audio["chunks"]).reshape(25, 3200)
    trace = SessionTrace()
    res = _run_chain(session, lm, dif, trace)
    assert len(res.chunks) == ref_chunks.shape[0]
    for call in (0, 1, 12, 24):
        ref = ref_chunks[call]
        got = res.chunks[call]
        peak = max(float(np.abs(ref).max()), 1e-9)
        assert np.abs(got - ref).max() / peak <= 0.25, f"call{call} PCM drifted"
        rms_g = float(np.sqrt((got.astype(np.float64) ** 2).mean()))
        rms_r = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
        ratio = rms_g / max(rms_r, 1e-12)
        assert 0.75 <= ratio <= 1.25, f"call{call} RMS ratio {ratio:.3f}"
