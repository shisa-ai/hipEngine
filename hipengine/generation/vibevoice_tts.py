"""Serialized, torch-free VibeVoice-TTS synthesis through the registered primitives.

This is the user-facing entry point the model contract specifies: one script plus one
24 kHz mono reference per speaker in, PCM out, with the completion status, cancellation
and reset semantics the contract requires. The generation loop itself lives in
``hipengine.runtime.vibevoice_tts_session``; this adapter owns reference preprocessing,
prompt construction and the request's own random stream, and it never imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import RLock
import json

import numpy as np

from hipengine.generation.registry import register_text_generator
from hipengine.generation.vibevoice_protocol import preprocess_audio
from hipengine.runtime.vibevoice_tts_session import (
    FINISH_CANCELLED,
    FINISH_ERROR,
    FINISH_LENGTH,
    FINISH_STOP,
)

#: The guidance scale the frozen oracle request uses. It is a request parameter
#: rather than a checkpoint value: ``config.json`` carries no ``cfg_scale`` key.
DEFAULT_CFG_SCALE = 1.3

#: Reference voices the checkpoint's card supports.
MAX_SPEAKERS = 4

SAMPLE_RATE = 24000

#: The TTS checkpoint ships no tokenizer bundle -- its snapshot holds only the three
#: safetensors shards, ``config.json`` and ``preprocessor_config.json`` -- so the text
#: tokenizer and the speech control ids come from the ASR checkpoint, which is the
#: same Qwen2.5 tokenizer the TTS model was trained against. Pinning it here rather
#: than accepting any tokenizer keeps the control ids and speaker formatting tied to a
#: known vocabulary.
TOKENIZER_MODEL_ID = "microsoft/VibeVoice-ASR-HF"


#: Reasons that mean the model stopped on its own terms. ``length`` counts: a budget
#: cut the request short, but no step misbehaved. ``cancelled`` and ``error`` do not.
COMPLETED_REASONS = ("eos", "length")


@dataclass(frozen=True)
class SynthesisOutput:
    """One synthesized request.

    ``finish_reason`` is ``eos`` when the model emitted the end token, ``length``
    when a token or frame budget cut generation off, ``cancelled`` for a cancelled
    request and ``error`` when a step raised, with ``error_reason`` set. Truncated
    output is returned as-is and never silently padded; a cancelled or failed
    request is not a completed synthesis.

    ``chunk_samples`` is the per-chunk sample count, which is what a caller needs to
    check that the codec's joins neither dropped nor duplicated samples.
    """

    pcm: np.ndarray
    sample_rate: int
    finish_reason: str
    generated_token_ids: tuple[int, ...]
    prompt_tokens: int
    output_seconds: float
    chunk_samples: tuple[int, ...] = ()
    error_reason: str | None = None

    @property
    def completed(self) -> bool:
        """Whether the model reached a stopping point on its own terms.

        ``length`` counts: a budget cut the request short, but every step the model
        took was a normal step. ``cancelled`` and ``error`` do not.
        """
        return self.finish_reason in COMPLETED_REASONS


class VibeVoiceTTSGenerator:
    """One initialized engine, one serialized request at a time.

    The engine owns weights, codec state and scratch. Per-request randomness is owned
    by the request: ``synthesize`` reseeds the session's generator from ``seed``
    before building the prompt, so the voice-prompt VAE draw and every diffusion
    frame come from one stream that belongs to that request and survives a
    cancellation or a speaker transition inside it.
    """

    supports_audio = False
    max_active_requests = 1

    def __init__(self, model_path, *, max_sequence_length=4096, backend="auto",
                 tokenizer_model=TOKENIZER_MODEL_ID):
        from hipengine.loading.hf_cache import resolve_model_path
        from hipengine.loading.vibevoice_tts_prompt import load_tts_tokenizer
        from hipengine.loading.vibevoice_tts_session import load_vibevoice_tts_session
        from hipengine.runtime.vibevoice_tts_session import VibevoiceTtsSession

        self._lock = RLock()
        self._closed = False
        self.max_context = int(max_sequence_length or 4096)
        path = Path(model_path)
        self.tokenizer = load_tts_tokenizer(resolve_model_path(tokenizer_model))
        processor = json.loads((path / "preprocessor_config.json").read_text())
        # The reference-audio contract is the processor's, not a caller's choice:
        # 24 kHz mono, loudness normalized to the checkpoint's target.
        extractor = processor["audio_processor"]
        self.audio_config = {
            key: extractor[key]
            for key in ("normalize_audio", "target_dB_FS", "eps")
            if key in extractor
        }
        self.reference_sample_rate = int(extractor.get("sampling_rate", SAMPLE_RATE))
        self.ddpm_steps = int(
            json.loads((path / "config.json").read_text())["diffusion_head_config"][
                "ddpm_num_inference_steps"
            ]
        )
        weights = load_vibevoice_tts_session(path)
        self.session = VibevoiceTtsSession(weights, max_context=self.max_context)
        # The solver's step count is part of the oracle contract, and the spec carries
        # it from the checkpoint rather than from a caller's preference.
        if int(weights.diffusion_spec.num_inference_steps) != self.ddpm_steps:
            raise RuntimeError(
                f"diffusion spec solves in {weights.diffusion_spec.num_inference_steps} "
                f"steps but the checkpoint config declares {self.ddpm_steps}"
            )

    # -- request path ------------------------------------------------------

    def synthesize(
        self,
        script,
        speaker_references,
        *,
        sample_rate=SAMPLE_RATE,
        max_new_tokens=None,
        seed=None,
        cfg_scale=DEFAULT_CFG_SCALE,
        cancel=None,
    ) -> SynthesisOutput:
        """Speak ``script`` in the voices of ``speaker_references``.

        ``speaker_references`` is one finite mono float waveform per speaker, in
        script order, at most :data:`MAX_SPEAKERS` of them; other rates are the
        caller's to resample, as in the ASR adapter. Each reference is normalized to
        the processor's target, which is part of the oracle contract rather than a
        convenience.
        """
        from hipengine.loading.vibevoice_tts_prompt import build_tts_prompt, vae_token_count

        if not isinstance(script, str) or not script.strip():
            raise ValueError("script must be nonempty text")
        if isinstance(speaker_references, (str, bytes)):
            raise ValueError("speaker_references must be a sequence of waveforms")
        if isinstance(speaker_references, np.ndarray):
            # A bare array is ambiguous: one waveform, or one row per speaker? Require
            # the caller to say so rather than guessing from a shape.
            if speaker_references.ndim != 2:
                raise ValueError(
                    "speaker_references must be a sequence of waveforms; a bare "
                    "array must be 2-D, one row per speaker"
                )
        elif not hasattr(speaker_references, "__len__"):
            raise ValueError("speaker_references must be a sequence of waveforms")
        if not 1 <= len(speaker_references) <= MAX_SPEAKERS:
            raise ValueError(
                f"speaker_references must hold 1 to {MAX_SPEAKERS} voices, "
                f"got {len(speaker_references)}"
            )
        pcms = [
            preprocess_audio(reference, sample_rate=sample_rate, **self.audio_config)
            for reference in speaker_references
        ]
        counts = [vae_token_count(pcm.size) for pcm in pcms]
        prompt = build_tts_prompt(script, counts, self.tokenizer)

        with self._lock:
            if self._closed:
                raise RuntimeError("synthesis engine is closed")
            # One request owns the random stream through its whole chain.
            if seed is not None:
                self.session.reseed(int(seed))
            _, connected = self.session.voice_prompt_rows_multi(pcms)
            speech_positions = sum(prompt.speech_input_mask)
            if connected.shape[0] != speech_positions:
                raise RuntimeError(
                    f"{connected.shape[0]} connected rows for {speech_positions} "
                    "speech positions"
                )
            rows = self.session.build_prompt_rows(
                prompt.input_ids,
                np.asarray(prompt.speech_input_mask, dtype=bool),
                connected,
            )
            result = self.session.generate(
                rows,
                cfg_scale=float(cfg_scale),
                max_new_tokens=max_new_tokens,
                cancel=cancel,
            )

        chunks = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in result.chunks]
        pcm = (
            np.concatenate(chunks).astype(np.float32, copy=False)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )
        return SynthesisOutput(
            pcm=pcm,
            sample_rate=SAMPLE_RATE,
            finish_reason=_STATUS[result.finish_reason],
            generated_token_ids=tuple(int(token) for token in result.ids),
            prompt_tokens=len(prompt.input_ids),
            output_seconds=round(float(pcm.size) / SAMPLE_RATE, 6),
            chunk_samples=tuple(int(chunk.size) for chunk in chunks),
            error_reason=result.error_reason,
        )

    def reset(self) -> None:
        """Return the session to its initial state: both KV caches, codec streaming
        state, semantic cache and the random stream."""

        with self._lock:
            self.session.reset()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


#: The session's internal reasons mapped to the contract's completion status.
_STATUS = {
    FINISH_STOP: "eos",
    FINISH_LENGTH: "length",
    FINISH_CANCELLED: "cancelled",
    FINISH_ERROR: "error",
}


def make_vibevoice_tts_generator(
    *, model_path, weight_index, model_plugin, max_sequence_length=None, backend="auto"
):
    return VibeVoiceTTSGenerator(
        model_path, max_sequence_length=max_sequence_length, backend=backend
    )


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register_text_generator(
        model="vibevoice",
        backend=_backend,
        quant="bf16",
        factory=partial(make_vibevoice_tts_generator, backend=_backend),
    )
