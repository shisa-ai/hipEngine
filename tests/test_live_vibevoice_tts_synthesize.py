"""The VibeVoice-TTS ``LLM.synthesize`` contract: status, cancellation, reset.

The model contract specifies a user-facing ``synthesize(script, speaker_references)``
returning PCM with a completion status, cancellable between steps, with a ``reset()``
that returns the session to its initial state. This exercises that surface through the
public API on the pinned single-speaker request.

Needs the TTS weights, so it is guarded on ROCm and skipped elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vibevoice_tts"
MODEL = "microsoft/VibeVoice-1.5B"
SCRIPT = "Speaker 1: The quick brown fox jumps over the lazy dog."
SEED = 20260915


def _reference() -> np.ndarray:
    """The fixture's 24 kHz reference, trimmed of its batch padding."""

    data = np.load(FIXTURES / "single_reference.npz")
    flat = np.asarray(data["ref_pcm"][0], dtype=np.float32).reshape(-1)
    return flat[: int(np.nonzero(flat)[0].max()) + 1]


@pytest.fixture(scope="module")
def engine():
    if not hip_runtime_available():
        pytest.skip("vibevoice TTS synthesize: HIP runtime unavailable")
    from hipengine import LLM

    instance = LLM(MODEL)
    try:
        yield instance
    finally:
        close = getattr(instance, "close", None)
        if callable(close):
            close()


def test_synthesize_returns_pcm_and_the_oracle_prompt_length(engine):
    """The prompt the adapter builds must be the one the oracle recorded.

    ``manifest.json`` records 121 prompt tokens for this request; a tokenizer or
    speaker-formatting mistake would move that number.
    """

    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    expected_prompt = next(
        r["prompt_tokens"] for r in manifest["requests"] if r["name"] == "single"
    )

    out = engine.synthesize(SCRIPT, [_reference()], sample_rate=24000, seed=SEED)

    assert out.finish_reason == "eos"
    assert out.prompt_tokens == expected_prompt
    assert out.sample_rate == 24000
    assert out.pcm.ndim == 1 and out.pcm.dtype == np.float32
    assert out.pcm.size > 0
    # The codec emits whole frames, and the join accounting has to say so.
    assert set(out.chunk_samples) == {3200}
    assert sum(out.chunk_samples) == out.pcm.size
    assert out.output_seconds == pytest.approx(out.pcm.size / 24000)
    assert out.completed and out.error_reason is None
    assert 0.005 <= float(np.sqrt((out.pcm**2).mean())) <= 1.0
    assert float(np.abs(out.pcm).max()) <= 1.0


def test_a_request_is_reproducible_at_a_fixed_seed(engine):
    """Per-request RNG ownership: the same seed on the same engine gives the same audio.

    The session's generator is shared by the voice-prompt draw and every diffusion
    frame, so this fails if a request does not own its own stream.
    """

    first = engine.synthesize(SCRIPT, [_reference()], sample_rate=24000, seed=SEED)
    second = engine.synthesize(SCRIPT, [_reference()], sample_rate=24000, seed=SEED)

    assert first.finish_reason == second.finish_reason == "eos"
    assert first.generated_token_ids == second.generated_token_ids
    assert np.array_equal(first.pcm, second.pcm)


def test_a_different_seed_gives_a_different_request(engine):
    """The seed has to matter, or the reproducibility above is vacuous."""

    base = engine.synthesize(SCRIPT, [_reference()], sample_rate=24000, seed=SEED)
    other = engine.synthesize(SCRIPT, [_reference()], sample_rate=24000, seed=SEED + 1)

    assert not np.array_equal(base.pcm, other.pcm)


def test_reset_restores_the_initial_state(engine):
    """``reset()`` must clear both KV caches, the codec, the semantic cache and the RNG."""

    engine.reset()
    after_reset = engine.synthesize(SCRIPT, [_reference()], sample_rate=24000, seed=SEED)
    engine.reset()
    again = engine.synthesize(SCRIPT, [_reference()], sample_rate=24000, seed=SEED)

    assert np.array_equal(after_reset.pcm, again.pcm)
    assert after_reset.generated_token_ids == again.generated_token_ids


def test_cancellation_returns_the_audio_so_far_and_says_so(engine):
    """A cancelled request is never a completed synthesis, and it keeps its chunks."""

    seen = {"steps": 0}

    def cancel() -> bool:
        seen["steps"] += 1
        return seen["steps"] > 4

    out = engine.synthesize(
        SCRIPT, [_reference()], sample_rate=24000, seed=SEED, cancel=cancel
    )

    assert out.finish_reason == "cancelled"
    assert not out.completed
    assert out.pcm.size > 0
    assert 0 < len(out.chunk_samples) < 24
    assert sum(out.chunk_samples) == out.pcm.size


def test_immediate_cancellation_returns_no_audio_but_still_reports_cancelled(engine):
    out = engine.synthesize(
        SCRIPT, [_reference()], sample_rate=24000, seed=SEED, cancel=lambda: True
    )

    assert out.finish_reason == "cancelled"
    assert out.chunk_samples == ()
    assert out.pcm.size == 0
    assert out.output_seconds == 0.0
    assert not out.completed


def test_argument_validation_is_explicit(engine):
    reference = _reference()
    with pytest.raises(ValueError, match="nonempty text"):
        engine.synthesize("   ", [reference])
    with pytest.raises(ValueError, match="1 to 4 voices"):
        engine.synthesize(SCRIPT, [])
    with pytest.raises(ValueError, match="1 to 4 voices"):
        engine.synthesize(SCRIPT, [reference] * 5)
    with pytest.raises(ValueError, match="sequence of waveforms"):
        engine.synthesize(SCRIPT, reference)
    # The reference-rate contract is the ASR lane's: resample before calling.
    with pytest.raises(ValueError, match="24000"):
        engine.synthesize(SCRIPT, [reference], sample_rate=16000)
