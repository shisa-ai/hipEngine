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
from hipengine.runtime.vibevoice_encoder import reference_frame_count
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


# -- multi-voice prompts ----------------------------------------------------


@pytest.fixture(scope="module")
def two_session(weights):
    """A session wide enough for the two-speaker request (352 + 59 positions)."""
    sess = VibevoiceTtsSession(weights, max_context=1024)
    yield sess
    sess.close()


def _two_voice_pcms(ref):
    """The two speakers' natural-length PCM.

    ``two_reference.npz`` stores both voices as one ``(2, 665600)`` batch, so
    row 0 carries the shorter speaker's trailing zero pad and its length there
    is the batch's, not the speaker's. The same speaker's natural length is
    recorded by the single-speaker fixture, which is where this reads it.
    """
    alice = np.asarray(_npz("single_reference.npz")["ref_pcm"])[0]
    carter = np.asarray(ref["ref_pcm"])[1]
    return [alice, carter]


def _two_prompt_rows(sess, ref, lm):
    """Two-speaker prompt rows from the frozen reference PCM, RNG-independent."""
    _, connected = sess.voice_prompt_rows_multi(
        _two_voice_pcms(ref),
        noise=np.asarray(ref["encode_draw1"]),
        noise_scale=np.asarray(ref["encode_draw0"]),
    )
    return sess.build_prompt_rows(
        np.asarray(lm["input_ids"]),
        np.asarray(lm["speech_input_mask"], dtype=bool),
        connected,
    )


def _row_rel(got, ref):
    """Per-row max error relative to the reference block's peak."""
    return np.abs(got - ref).max(axis=1) / float(np.abs(ref).max())


def test_reference_frame_count_matches_encoder(session) -> None:
    """The frame-count formula agrees with the encoder on the frozen voices.

    ``voice_prompt_rows_multi`` needs each voice's real frame count to slice the
    batch-padded encode, and it computes that from the sample count rather than
    encoding twice. This pins the formula against the encoder's own returned
    count, including a reference whose length is not a hop multiple.
    """
    single = _npz("single_reference.npz")
    two = _npz("two_reference.npz")
    cases = [
        (np.asarray(single["ref_pcm"])[0], 70),
        (np.asarray(two["ref_pcm"])[0], 208),
        (np.asarray(two["ref_pcm"])[1], 208),
    ]
    for pcm, expected in cases:
        assert reference_frame_count(pcm.size) == expected, f"{pcm.size} samples"
        assert session.frontend.encode_reference(pcm).shape[0] == expected


def test_voice_prompt_rows_multi_single_voice_is_unchanged(session) -> None:
    """The one-voice path is bit-identical to the single-reference entry point.

    ``voice_prompt_rows`` now delegates to the multi-voice builder, so the
    regression this guards is a batch-max pad or an extra RNG draw leaking into
    the single-voice case, which the frozen single-speaker chain depends on.
    """
    ref = _npz("single_reference.npz")
    pcm = np.asarray(ref["ref_pcm"])[0]
    noise = np.asarray(ref["encode_draw1"])
    scale = np.asarray(ref["encode_draw0"])
    sampled, connected = session.voice_prompt_rows(pcm, noise=noise, noise_scale=scale)
    multi_sampled, multi_connected = session.voice_prompt_rows_multi(
        [pcm], noise=noise, noise_scale=scale
    )
    assert len(multi_sampled) == 1
    assert np.array_equal(sampled, multi_sampled[0])
    assert np.array_equal(connected, multi_connected)


def test_two_speaker_prompt_rows_match_reference(two_session) -> None:
    """Two voices -> per-voice latents and 278 spliced connected rows.

    The oracle batches both voices into one ``forward_speech_features`` call, so
    both ``speech_tensors`` rows are zero padded to the longer reference and
    ``speech_masks`` selects 70 and 208 frames afterwards. The encoder is not
    translation invariant at its tail, so a voice encoded outside that batch
    disagrees on its final frame; the second assertion is what makes the
    batch-max padding a contract instead of an implementation detail.

    Envelope: the latents sit on the encoder's bf16 noise floor (0.036 for the
    70-frame voice, 0.072 for the 208-frame one, against 0.036 for the frozen
    single-speaker voice). The connector amplifies that per row, so the median
    row is gated tightly and the tail is gated at the measured 0.206.
    """
    ref = _npz("two_reference.npz")
    single = _npz("single_reference.npz")
    oracle_conn = np.asarray(ref["connected"])
    oracle_lat = np.asarray(ref["encode0_latents"])

    sampled, connected = two_session.voice_prompt_rows_multi(
        _two_voice_pcms(ref),
        noise=np.asarray(ref["encode_draw1"]),
        noise_scale=np.asarray(ref["encode_draw0"]),
    )
    assert [s.shape[0] for s in sampled] == [70, 208]
    assert connected.shape == oracle_conn.shape == (278, 1536)

    offsets = (0, 70)
    for index, (frames, offset) in enumerate(zip((70, 208), offsets)):
        lat = np.abs(sampled[index] - oracle_lat[index][:frames]).max() / float(
            np.abs(oracle_lat[index][:frames]).max()
        )
        assert lat <= 0.09, f"voice {index} sampled latents drifted: {lat:.4f}"
        rows = _row_rel(
            connected[offset : offset + frames], oracle_conn[offset : offset + frames]
        )
        assert np.median(rows) <= 0.02, f"voice {index} median row drifted"
        assert rows.max() <= 0.25, f"voice {index} worst row drifted: {rows.max():.4f}"

    # Batch-max padding is load-bearing: voice 0 encoded alone (batch of one, so
    # no pad) puts its final frame 0.23 off, while the batched encode is 0.04.
    _, alone = two_session.voice_prompt_rows_multi(
        [np.asarray(single["ref_pcm"])[0]],
        noise=np.asarray(single["encode_draw1"]),
        noise_scale=np.asarray(single["encode_draw0"]),
    )
    alone_rel = float(np.abs(alone - oracle_conn[:70]).max() / np.abs(oracle_conn[:70]).max())
    batched_rel = float(
        np.abs(connected[:70] - oracle_conn[:70]).max() / np.abs(oracle_conn[:70]).max()
    )
    assert alone_rel > 0.15, f"unpadded voice 0 matched at {alone_rel:.4f}; check the pad"
    assert batched_rel <= 0.05, f"batched voice 0 drifted: {batched_rel:.4f}"


def test_two_speaker_prefill_logits_match_reference(two_session) -> None:
    """The 352-position two-speaker prompt reaches the oracle's logits.

    This is the end of the prompt path: if the connected rows were spliced into
    the wrong mask positions, or the voices were swapped, the final position's
    argmax would move. Measured 0.012 relative on the final position.
    """
    ref = _npz("two_reference.npz")
    lm = _npz("two_lm.npz")
    rows = _two_prompt_rows(two_session, ref, lm)
    assert len(rows) == 352
    pos = two_session.positive
    pos.reset()
    for index, row in enumerate(rows):
        pos.push_token(row, index)
        pos.forward_layers(index)
    logits, _ = pos.logits_argmax()
    oracle = np.asarray(lm["prefill_logits"])[-1]
    rel = float(np.abs(np.asarray(logits) - oracle).max() / np.abs(oracle).max())
    assert rel <= 0.03, f"two-speaker final-position logits drifted: {rel:.4f}"
    assert int(np.asarray(logits).argmax()) == int(oracle.argmax())


@pytest.mark.xfail(
    strict=True,
    reason="two-speaker chain diverges at the first span end; see "
    "worklog/entries/20260915T091156.145004Z-lhl-vibevoice-tts-two-speaker-prompt-f6fe75.md",
)
def test_two_speaker_greedy_chain_matches_torch(two_session) -> None:
    """RED: the 59-token two-speaker chain against the frozen oracle.

    Measured: the first 31 tokens are exact and the divergence is the 32nd, the
    first span's end, where the oracle emits speech_end and this session emits
    another speech_diffusion (top-2 gap 9 logits, so not a near tie). Localized
    to the prompt rows, not the LM or the diffusion head: the head reproduces
    the oracle's own per-call latents to 0.036, and the LM reproduces the
    oracle's prefill hidden to 0.05 given the oracle's own connected rows. The
    208-frame voice's connected rows are the input that is off, at 0.206.
    """
    ref = _npz("two_reference.npz")
    lm = _npz("two_lm.npz")
    dif = _npz("two_diffusion.npz")
    rows = _two_prompt_rows(two_session, ref, lm)
    calls = int(dif["num_calls_recorded"])
    res = two_session.generate(
        rows,
        cfg_scale=1.3,
        max_new_tokens=59,
        noise_hook=lambda i: dif[f"call{min(i, calls - 1)}_initial_noise"],
        neg_hook=lambda i: dif[f"call{min(i, calls - 1)}_neg_condition"],
    )
    gen = np.asarray(lm["generated_ids"])[0]
    expected = [int(t) for t in gen[len(np.asarray(lm["input_ids"])[0]) :]]
    assert len(res.chunks) == calls, (
        f"session ran {len(res.chunks)} diffusion calls, oracle recorded {calls}"
    )
    assert res.ids == expected, "two-speaker chain moved off the oracle"
