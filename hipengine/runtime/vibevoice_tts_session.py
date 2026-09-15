"""Torch-free VibeVoice-TTS generation session (single-sample, HIP/GPU).

Drives the frozen-fork generation loop end to end on device:

- prompt prefill on the Qwen2 LM with speech rows (voice-prompt connector
  output) spliced over placeholder token embeddings;
- a constrained greedy loop: only the five fork-valid tokens can be
  generated (speech start/end/diffusion, EOS, BOS), enforced host-side on
  the downloaded logits exactly like the fork's ``VibeVoiceTokenConstraintProcessor``;
- per diffusion frame: positive condition = post-final-norm last hidden of
  the positive LM, negative condition = the same from a negative LM runtime
  that is reset at each speech boundary and then accumulates the positive
  pass's input embeddings (the fork's ``refresh_negative=True`` semantics);
- the frame's speech latent through the acoustic decoder (streaming causal
  state, reset at speech boundaries), the chunk through the semantic
  encoder (caller-owned streaming tails), and
  ``acoustic_connector(latent) + semantic_connector(mean)`` as the next
  LM input embedding.

Randomness is owned by one documented numpy ``PCG64`` generator seeded at
construction; the RED test injects recorded fixture operands (initial noise,
negative condition) so parity comparisons are RNG-independent, exactly like
the ASR lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hipengine.loading.vibevoice_tts_session import (
    EOS_TOKEN_ID,
    SPEECH_DIFFUSION_ID,
    SPEECH_END_ID,
    SPEECH_START_ID,
    VibevoiceTtsSessionWeights,
)
from hipengine.kernels.cpu_reference.vibevoice_asr import vibevoice_connector
from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime, reference_frame_count
from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime
from hipengine.runtime.vibevoice_tts_decoder import VibevoiceTTSDecoderGPU
from hipengine.runtime.vibevoice_tts_diffusion import VibevoiceTTSDiffusionHeadGPU

# The fork's generate() restricts generation to exactly these tokens.
VALID_TOKENS = (
    SPEECH_START_ID,
    SPEECH_END_ID,
    SPEECH_DIFFUSION_ID,
    EOS_TOKEN_ID,
    151644,  # bos <|im_start|>
)


@dataclass
class SessionTrace:
    """Optional per-step boundary snapshots for RED gating and diagnosis."""

    hidden_states: list[np.ndarray] = field(default_factory=list)
    logits: list[tuple[int, np.ndarray]] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    conditions: list[np.ndarray] = field(default_factory=list)
    neg_conditions: list[np.ndarray] = field(default_factory=list)
    speech_latents: list[np.ndarray] = field(default_factory=list)
    chunks: list[np.ndarray] = field(default_factory=list)
    semantic_means: list[np.ndarray] = field(default_factory=list)
    feedback_sums: list[np.ndarray] = field(default_factory=list)


@dataclass
class SessionResult:
    ids: list[int]
    chunks: list[np.ndarray]
    trace: SessionTrace | None


class VibevoiceTtsSession:
    """One TTS generation session: two LM runtimes + speech stack on device."""

    def __init__(
        self, weights: VibevoiceTtsSessionWeights, *, max_context: int = 4096,
        seed: int = 20260915,
    ) -> None:
        self.w = weights
        self.positive = VibevoiceQwen2Runtime(weights.lm, max_context=max_context)
        self.negative = VibevoiceQwen2Runtime(weights.lm, max_context=max_context)
        self.frontend = VibevoiceFrontendRuntime(
            weights.acoustic_encoder_spec,
            weights.acoustic_encoder,
            weights.semantic_encoder_spec,
            weights.semantic_encoder,
            weights.acoustic_connector,
            weights.semantic_connector,
        )
        self.decoder = VibevoiceTTSDecoderGPU(weights.decoder_spec, weights.decoder_weights)
        self.diffusion = VibevoiceTTSDiffusionHeadGPU(weights.diffusion_spec, weights.diffusion_weights)
        self.semantic_state: dict[str, np.ndarray] = {}
        # One generator owns every random draw of the session: the voice-prompt
        # VAE sampling and each diffusion frame's initial noise. Seeding here
        # rather than inside ``generate()`` keeps that one stream intact when the
        # voice prompt is prepared first, which is the order the fork uses (it
        # calls ``torch.manual_seed`` once before generation and draws the voice
        # noise inside the prefill pass).
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._noise_draws = 0
        self._neg_position = 0
        self._prev_feedback: np.ndarray | None = None

    def close(self) -> None:
        self.positive.close()
        self.negative.close()
        self.frontend.close()
        self.decoder.close()
        self.diffusion.close()

    # -- prompt construction ------------------------------------------------

    def voice_prompt_rows(self, ref_pcm: np.ndarray, *, noise_scale: np.ndarray | None = None,
                          noise: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reference PCM -> (sampled latents, connected rows) for the prompt.

        Acoustic-encodes the 24 kHz mono reference in one non-streaming pass,
        samples the VAE with the supplied recorded operands (or the session
        generator), applies the checkpoint ``(latent + bias) * scale`` factors,
        and connects to LM space. Matches the fork's ``_process_speech_inputs``
        audio path, which encodes the whole waveform with per-stage right zero
        padding rather than in aligned chunks.
        """
        sampled, connected = self.voice_prompt_rows_multi(
            [ref_pcm], noise_scale=noise_scale, noise=noise,
        )
        return sampled[0], connected

    def voice_prompt_rows_multi(
        self,
        ref_pcms,
        *,
        noise_scale: np.ndarray | None = None,
        noise: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """One or more reference waveforms -> (per-voice latents, spliced rows).

        The fork batches every voice of a request into a single
        ``forward_speech_features`` call, so ``speech_tensors`` are right zero
        padded to the longest reference in the batch before the encoder runs
        and ``speech_masks`` afterwards selects each voice's real frames. The
        encoder is not translation invariant at its tail: a voice shorter than
        the batch max has its final frame computed with the longer voices' zero
        padding in context, and encoding that voice alone disagrees with the
        fork there by up to 0.39 relative on the frozen two-speaker fixture.
        Padding to the batch max reproduces the fork's frame, and for a single
        voice the pad is empty, so the one-voice path is unchanged.

        Returns each voice's sampled latents, shaped ``(frames_i, 64)``, and
        the concatenation of the per-voice connected rows in mask order, which
        is what ``build_prompt_rows`` splices into the prompt.
        """
        pcms = [np.asarray(p, dtype=np.float32).reshape(-1) for p in ref_pcms]
        if not pcms:
            raise ValueError('at least one reference waveform is required')
        if any(not pcm.size for pcm in pcms):
            raise ValueError('reference PCM must be nonempty')
        batch = len(pcms)
        max_samples = max(pcm.size for pcm in pcms)
        frames = [reference_frame_count(pcm.size) for pcm in pcms]
        max_frames = max(frames)
        # One draw per batch, matching the fork's single ``torch.randn`` over
        # ``speech_mode`` rather than one draw per voice.
        if noise is None:
            noise = self._rng_standard_normal((batch, max_frames, 64))
        if noise_scale is None:
            noise_scale = self._rng_standard_normal((batch,)) * np.float32(0.5 / 0.8)
        noise = np.asarray(noise, dtype=np.float32)
        if noise.ndim == 2 and batch == 1:
            noise = noise.reshape(1, *noise.shape)
        if noise.ndim != 3 or noise.shape[0] != batch or noise.shape[2] != 64:
            raise ValueError(
                f'noise must be ({batch}, frames, 64) or (frames, 64) for one voice, got {noise.shape}'
            )
        if noise.shape[1] < max_frames:
            raise ValueError(
                f'noise has {noise.shape[1]} frames, needs at least {max_frames}'
            )
        noise_scale = np.asarray(noise_scale, dtype=np.float32).reshape(-1)
        if noise_scale.size != batch:
            raise ValueError(
                f'noise_scale must hold {batch} value(s), got {noise_scale.size}'
            )
        sampled: list[np.ndarray] = []
        connected: list[np.ndarray] = []
        for index, pcm in enumerate(pcms):
            if pcm.size < max_samples:
                pcm = np.pad(pcm, (0, max_samples - pcm.size))
            latent = np.asarray(
                self.frontend.encode_reference(pcm), dtype=np.float32
            ).reshape(-1, 64)
            if latent.shape[0] != max_frames:
                raise RuntimeError(
                    f'reference {index} encoded {latent.shape[0]} frames, '
                    f'expected {max_frames} from {pcm.size} samples'
                )
            value = latent[:frames[index]] + noise_scale[index] * noise[index][:frames[index]]
            scaled = (value + np.float32(self.w.speech_bias_factor)) * np.float32(
                self.w.speech_scaling_factor
            )
            rows = vibevoice_connector(self.w.acoustic_connector, scaled, dtype='bfloat16')
            sampled.append(value.astype(np.float32))
            connected.append(np.asarray(rows, dtype=np.float32))
        return sampled, np.vstack(connected)

    def build_prompt_rows(
        self,
        input_ids: np.ndarray,
        speech_input_mask: np.ndarray,
        connected: np.ndarray,
    ) -> list[np.ndarray]:
        """Token ids + spliced speech rows -> the list of prompt embedding rows."""
        ids = np.asarray(input_ids).reshape(-1)
        mask = np.asarray(speech_input_mask, dtype=bool).reshape(-1)
        connected = np.asarray(connected, dtype=np.float32)
        rows = np.array([self.positive.embed_row(int(t)) for t in ids])
        rows[mask] = connected
        return [r for r in rows]

    def _rng_standard_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        self._noise_draws += 1
        return self._rng.standard_normal(shape, dtype=np.float32)

    # -- generation ---------------------------------------------------------

    def _masked_argmax(self, logits: np.ndarray) -> int:
        masked = logits[list(VALID_TOKENS)]
        return VALID_TOKENS[int(masked.argmax())]

    def _negative_reset(self) -> None:
        """Clear the negative LM at speech boundaries (the fork's reset block)."""
        self.negative.reset()
        self._neg_position = 0
        self._prev_feedback = None

    def _negative_condition(self, current_embed: np.ndarray | None) -> np.ndarray:
        """Accumulate the current input embedding on the negative LM.

        Fixture-verified semantics: the negative runtime holds
        ``[embed(speech_start), feedback_0, ..., feedback_{n-1}]`` across the
        diffusion frames of one speech span. The fork feeds it the *positive*
        pass's current input embedding (``inputs_embeds`` at the step that
        produced the diffusion token), so the first frame of a span pushes the
        bare ``speech_start`` embedding and each later frame pushes the previous
        frame's feedback embedding.
        """
        if self._neg_position == 0 or current_embed is None:
            row = self.positive.embed_row(SPEECH_START_ID)
        else:
            row = np.asarray(current_embed, dtype=np.float32).reshape(-1)
        self.negative.push_token(row, self._neg_position)
        self.negative.forward_layers(self._neg_position)
        self._neg_position += 1
        return self.negative.hidden_state()

    def generate(
        self,
        prompt_rows: list[np.ndarray],
        *,
        cfg_scale: float,
        max_new_tokens: int | None = None,
        noise_hook=None,
        neg_hook=None,
        trace: SessionTrace | None = None,
    ) -> SessionResult:
        """Run the frozen-fork generation loop.

        ``noise_hook(call_index)`` and ``neg_hook(call_index)`` let the RED
        test inject recorded ``(noise_scale, noise)`` and the recorded
        negative condition, keeping parity RNG-independent; ``None`` runs the
        session's own generator and negative pass. Pass a ``neg_hook`` only to
        isolate arithmetic: with it set the session's own negative branch is
        bypassed and its output is never checked.
        """
        self._noise_draws = 0
        self.semantic_state = {}
        self.decoder.reset()
        self._negative_reset()
        pos = self.positive
        pos.reset()
        if trace is not None:
            trace.hidden_states.clear()

        for position, row in enumerate(prompt_rows):
            pos.push_token(row, position)
            pos.forward_layers(position)
            if trace is not None:
                trace.hidden_states.append(pos.hidden_state())
        # Hidden of the forward that produced the current ``next_token`` —
        # the fork's diffusion condition (``outputs.last_hidden_state[..., -1]``).
        pending_hidden = pos.hidden_state()
        logits, _ = pos.logits_argmax()
        if trace is not None:
            trace.logits.append((len(prompt_rows) - 1, logits.copy()))
        next_token = self._masked_argmax(logits)
        ids: list[int] = []
        chunks: list[np.ndarray] = []
        call_index = 0
        budget = max_new_tokens if max_new_tokens is not None else 40 * len(prompt_rows)
        for _ in range(budget):
            ids.append(next_token)
            if trace is not None:
                trace.tokens.append(next_token)
            if next_token == EOS_TOKEN_ID:
                break
            # Boundary resets fire on the freshly generated token, exactly
            # like the fork's speech_start / speech_end blocks.
            if next_token == SPEECH_START_ID or next_token == SPEECH_END_ID:
                self.semantic_state = {}
                self.decoder.reset()
                self._negative_reset()
                self._noise_draws = 0
            # Diffusion processing fires on the freshly generated token; its
            # feedback embedding is fed at the NEXT forward, like the fork's
            # ``next_inputs_embeds[diffusion_indices] = diffusion_embeds``.
            next_embed: np.ndarray | None = None
            if next_token == SPEECH_DIFFUSION_ID:
                if neg_hook is not None:
                    neg_condition = np.asarray(neg_hook(call_index), dtype=np.float32).reshape(1, -1)
                else:
                    neg_condition = self._negative_condition(self._prev_feedback)
                condition = pending_hidden.reshape(1, -1)
                if trace is not None:
                    trace.conditions.append(condition.reshape(-1).copy())
                    trace.neg_conditions.append(neg_condition.reshape(-1).copy())
                if noise_hook is not None:
                    initial_noise = np.asarray(noise_hook(call_index), dtype=np.float32).reshape(2, 64)
                else:
                    initial_noise = self._rng_standard_normal((2, 64))
                speech, _ = self.diffusion.sample_speech_tokens(
                    condition, neg_condition.reshape(1, -1), cfg_scale, initial_noise
                )
                speech_latent = speech[0].reshape(64)
                if trace is not None:
                    trace.speech_latents.append(speech_latent.copy())
                scaled = speech_latent / np.float32(self.w.speech_scaling_factor) - np.float32(self.w.speech_bias_factor)
                chunk = self.decoder.decode(scaled.reshape(1, 64)[0])
                chunks.append(chunk.copy())
                if trace is not None:
                    trace.chunks.append(chunk.copy())
                sem = self.frontend.encode_chunk_streaming("semantic", chunk, self.semantic_state)
                sem_mean = sem.reshape(1, 128)
                if trace is not None:
                    trace.semantic_means.append(sem_mean.reshape(-1).copy())
                acoustic_embed = vibevoice_connector(self.w.acoustic_connector, speech_latent.reshape(1, 64), dtype="bfloat16")
                semantic_embed = vibevoice_connector(self.w.semantic_connector, sem_mean, dtype="bfloat16")
                feedback = acoustic_embed + semantic_embed
                if trace is not None:
                    trace.feedback_sums.append(feedback.reshape(-1).copy())
                next_embed = feedback.reshape(-1).astype(np.float32)
                # The next frame's negative pass consumes this as the positive
                # pass's current input embedding.
                self._prev_feedback = next_embed
                call_index += 1
            # Forward the next input: the feedback embedding when this step
            # diffused, otherwise the plain token embedding.
            if next_embed is not None:
                embed = next_embed
            else:
                embed = pos.embed_row(next_token)
            position = len(prompt_rows) + len(ids) - 1
            pos.push_token(embed, position)
            pos.forward_layers(position)
            pending_hidden = pos.hidden_state()
            logits, _ = pos.logits_argmax()
            if trace is not None:
                trace.logits.append((position, logits.copy()))
            next_token = self._masked_argmax(logits)
        return SessionResult(ids=ids, chunks=chunks, trace=trace)
