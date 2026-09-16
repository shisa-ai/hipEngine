"""YuE2 NAR (acoustic flow matching) runtime.

The NAR stage turns a semantic codec sequence into acoustic latents. Upstream
splits the song into *original* chunks whose size is set by the AR prefix, draws
one FP32 noise tensor for the whole song, and solves each chunk with a 32-step
midpoint rule against a cached AR conditioning prefix.

This module keeps the CPU-side protocol (chunking, noise slicing, the FP64
solver schedule, the BF16 sigmoid/time shift, the timestep sinusoid) separate
from the device runtime so the protocol is testable without a GPU. The device
runtime lives in :class:`Yue2NarRuntime` below.

Rounding contract, from the pinned upstream ``yue2/nar.py`` and
``yue2/modeling_yue2.py``:

* the solver's ``t`` and ``logit(t)`` are computed in **FP64** and only then
  clamped to ``[-20, 20]``;
* the shifted timestep is ``shift * sigmoid(raw) / (1 + (shift - 1) * sigmoid(raw))``
  with every step in the **model dtype** (BF16 here), so the raw value is rounded
  to BF16 before the sigmoid;
* the ODE state, both velocity evaluations and the two state updates are BF16;
* the returned latents are FP32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np

from hipengine.generation.yue2 import (
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    MUSIC_END,
    YuE2Random,
    chunk_ranges,
)

#: Frames of acoustic context per original chunk is bounded by the AR prefix:
#: ``size = min((context - prefix_tokens - 3) // 2, context)``.
LATENT_DIM = 64
#: Timestep sinusoid width used by ``TimestepEmbedder``.
TIME_FREQUENCY_SIZE = 256


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round FP32 values to BF16 storage bits (round-half-to-even)."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    """Widen BF16 storage bits to FP32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _bf16(values) -> np.ndarray:
    return from_bf16_bits(to_bf16_bits(np.asarray(values, dtype=np.float32)))


def clamp_logit(t_value: float) -> float:
    """``clamp(logit(t), -20, 20)`` in FP64, exactly as the reference computes it."""
    t = float(t_value)
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    if t == 1.0:
        raw = math.inf
    elif t == 0.0:
        raw = -math.inf
    else:
        raw = math.log(t / (1.0 - t))
    return float(min(max(raw, -20.0), 20.0))


def solver_schedule(steps: int) -> list[tuple[float, float]]:
    """Per-step ``(raw_t, raw_mid)`` pairs for the midpoint rule.

    ``h = 1/steps``, ``t = 1 - i*h``; the midpoint evaluation uses ``t - h/2``.
    Both are clamped logs of FP64 timesteps, matching the reference.
    """

    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    count = int(steps)
    h = 1.0 / count
    schedule = []
    for step in range(count):
        t = 1.0 - step * h
        schedule.append((clamp_logit(t), clamp_logit(t - h / 2)))
    return schedule


def shift_t_value(raw_t: float, shift: float = 1.0) -> np.ndarray:
    """BF16 sigmoid/time shift applied to a raw (clamped-logit) timestep.

    Returns one BF16 scalar as a ``uint16`` bits array, so callers can hand it to
    a device kernel without leaving the model's rounding path.
    """

    raw = _bf16([float(raw_t)])[0]
    t_sig = _bf16([1.0 / (1.0 + math.exp(-float(raw)))])[0]
    shift_bf16 = float(_bf16([float(shift)])[0])
    if shift_bf16 == 1.0:
        return to_bf16_bits(np.asarray([t_sig], dtype=np.float32))
    numerator = _bf16([shift_bf16 * float(t_sig)])[0]
    denominator = _bf16([1.0 + _bf16([(shift_bf16 - 1.0) * float(t_sig)])[0]])[0]
    return to_bf16_bits(np.asarray([_bf16([numerator / denominator])[0]], dtype=np.float32))


def timestep_sinusoid(t_value, frequency_embedding_size: int = TIME_FREQUENCY_SIZE) -> np.ndarray:
    """FP32 ``cat([cos(args), sin(args)])`` for one shifted timestep.

    ``args = t.float() * exp(-log(10000) * arange(half) / half)``. The caller
    rounds the result to BF16 before the MLP, as ``TimestepEmbedder`` does.
    """

    half = int(frequency_embedding_size) // 2
    freqs = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half)
    args = np.asarray(t_value, dtype=np.float32) * freqs
    return np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)


def audio_position_rows(positions: Iterable[int], max_frames: int) -> np.ndarray:
    """Local audio-position indices, clamped to the embedding table."""

    indices = np.asarray([int(position) for position in positions], dtype=np.int64)
    if indices.size and indices.min() < 0:
        raise ValueError("audio positions must be nonnegative")
    return np.minimum(indices, int(max_frames) - 1)


@dataclass(frozen=True)
class AcousticChunk:
    """One original upstream chunk: AR conditioning tokens plus its noise view."""

    index: int
    frame_range: tuple[int, int]
    ar_tokens: tuple[int, ...]
    noise: np.ndarray
    nar_cond_end: int = 0

    @property
    def frames(self) -> int:
        return len(self.noise)

    @property
    def ar_length(self) -> int:
        return len(self.ar_tokens)

    @property
    def nar_length(self) -> int:
        """NAR positions: one leading and one trailing zero-state boundary row."""
        return self.frames + 2


def song_chunks(
    prefix: Sequence[int],
    codec: Sequence[int],
    seed: int,
    *,
    context: int = CONTEXT,
    noise: np.ndarray | None = None,
    rng: YuE2Random | None = None,
    nar_cond_end: int = 0,
) -> list[AcousticChunk]:
    """Split a semantic sequence into original chunks with their noise views.

    The reference draws **one** ``[frames, 64]`` FP32 noise tensor for the whole
    song and then takes a view per chunk, so a chunk's noise depends on the
    global frame index rather than only on its own length. Pass ``noise`` to
    reproduce a recorded draw exactly; otherwise a deterministic request-local
    draw is used (see ``YuE2Random`` for the seeded-equality caveat).
    """

    prefix_ids = [int(token) for token in prefix]
    codec_ids = [int(token) for token in codec]
    if not prefix_ids:
        raise ValueError("prefix must be a nonempty sequence of token IDs")
    if min(prefix_ids) < 0:
        raise ValueError("prefix token IDs must be nonnegative")
    if not codec_ids:
        raise ValueError("codec must be a nonempty sequence of token IDs")
    if min(codec_ids) < 0 or max(codec_ids) >= CODEC_SIZE:
        raise ValueError("codec token IDs are outside their allowed vocabulary")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if nar_cond_end < 0:
        raise ValueError("nar_cond_end must be nonnegative")

    frames = len(codec_ids)
    if noise is None:
        source = rng if rng is not None else YuE2Random(int(seed))
        full = np.ascontiguousarray(
            source.standard_normal((frames, LATENT_DIM), dtype=np.float32)
        )
    else:
        full = np.asarray(noise, dtype=np.float32)
        if full.shape != (frames, LATENT_DIM):
            raise ValueError(
                f"noise must have shape ({frames}, {LATENT_DIM}), got {tuple(full.shape)}"
            )
    if not np.isfinite(full).all():
        raise ValueError("acoustic noise contains non-finite values")

    chunks = []
    for index, (start, stop) in enumerate(chunk_ranges(frames, len(prefix_ids), context)):
        ar_tokens = tuple(
            prefix_ids + [value + CODEC_OFFSET for value in codec_ids[start:stop]] + [MUSIC_END]
        )
        chunks.append(
            AcousticChunk(
                index=index,
                frame_range=(start, stop),
                ar_tokens=ar_tokens,
                noise=np.ascontiguousarray(full[start:stop]),
                nar_cond_end=int(nar_cond_end),
            )
        )
    return chunks


def visible_ar_length(chunk: AcousticChunk) -> int:
    """How many cached AR positions the NAR attention may see.

    ``nar_cond_end = 0`` is the release path: the NAR sees the whole AR prefix.
    A positive value restricts visibility to the first ``nar_cond_end``
    positions, which upstream uses for codec dropout ("text-only" mode).
    """

    if chunk.nar_cond_end <= 0:
        return chunk.ar_length
    return min(int(chunk.nar_cond_end), chunk.ar_length)


def validate_chunk_context(chunk: AcousticChunk, max_position_embeddings: int) -> None:
    """The reference rejects an original chunk that does not fit the context."""

    if chunk.ar_length + chunk.nar_length > int(max_position_embeddings):
        raise ValueError("Original acoustic chunk exceeds the model context")

