"""YuE2 NAR (acoustic flow matching) runtime.

The NAR stage turns a semantic codec sequence into acoustic latents. Upstream
splits the song into *original* chunks whose size is set by the AR prefix, draws
one FP32 noise tensor for the whole song, and solves each chunk with a 32-step
midpoint rule against a cached AR conditioning prefix.

This module keeps the CPU-side protocol (chunking, noise slicing, the FP64
solver schedule, the BF16 sigmoid/time shift, the timestep sinusoid) separate
from the device runtime so the protocol is testable without a GPU. The device
runtime lives in :class:`Yue2NarRuntime` below.

The device runtime composes a resident :class:`~hipengine.runtime.yue2_ar.Yue2ArRuntime`
rather than duplicating it: the AR prefix is prefilled once per chunk through the
AR weights, and every velocity evaluation reads that per-layer K/V cache as the
NAR's conditioning keys. Query rows are processed in tiles so a full-context
chunk costs a few hundred megabytes of scratch instead of a few gigabytes; the
NAR key/value projection for a layer still covers every row before any query tile
attends to it, because the NAR attention is bidirectional.

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
import os
from dataclasses import dataclass, field
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.hipblaslt import HipblasLt
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.generation.yue2 import (
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    MUSIC_END,
    YuE2Random,
    chunk_ranges,
)
from hipengine.kernels.hip_gfx1100.yue2 import nar as nar_kernels
from hipengine.loading.yue2 import YuE2Weights
from hipengine.runtime.yue2_ar import (
    Yue2ArRuntime,
    bf16_bits_to_f32,
    bf16_bits_to_fp16_bits,
    f32_to_bf16_bits,
    rope_tables,
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


# ---------------------------------------------------------------------------
# device runtime
# ---------------------------------------------------------------------------

#: Query rows per tile. The NAR key/value projection for a layer always covers
#: every row; only the query-side scratch (Q, attention, O, MLP) is tiled, which
#: is what keeps a full-context chunk inside a few hundred megabytes.
_TILE_ROWS = 2048


class Yue2NarRuntime:
    """Flow-matching (NAR) runner over a resident :class:`Yue2ArRuntime`.

    ``condition`` prefills one chunk's AR prefix through the AR runtime and keeps
    its K/V cache; ``velocity`` then evaluates the ODE right-hand side at the
    current state, and ``solve`` runs the reference 32-step midpoint rule.
    """

    def __init__(
        self,
        weights: YuE2Weights,
        ar: Yue2ArRuntime,
        *,
        library: ctypes.CDLL | None = None,
    ) -> None:
        spec = weights.config
        if int(ar.spec.hidden_size) != int(spec.hidden_size):
            raise ValueError("AR and NAR runtimes must share the same model config")
        self.spec = spec
        self.ar = ar
        self.weights = weights
        self.runtime = get_hip_runtime()
        self.library = library or ar.library
        self.kernels = ar.kernels
        self._buffers: list[DeviceBuffer] = []
        self._rows = 0
        self._tile = 0
        self._chunk: AcousticChunk | None = None
        self._visible = 0
        self._prepared: set[int] = set()
        #: Diagnostic hook: when set, ``_velocity`` copies the hidden stream after
        #: the embedding stack and after each layer into this list (bf16 bits).
        self.debug_taps: list[np.ndarray] | None = None

        hidden = int(spec.hidden_size)
        ffn = int(spec.intermediate_size)
        latent = int(spec.latent_dim)
        heads = int(spec.num_attention_heads)
        kv_heads = int(spec.num_key_value_heads)
        head_dim = int(spec.head_dim)
        q_width = heads * head_dim
        kv_width = kv_heads * head_dim
        self._q_width = q_width
        self._kv_width = kv_width
        self._head_dim = head_dim
        self._heads = heads
        self._kv_heads = kv_heads
        self._latent = latent
        self._hidden_size = hidden
        self._ffn = ffn
        self._scale = 1.0 / float(np.sqrt(head_dim))
        #: The tensor-core attention implements the production head geometry
        #: (16 query heads over 8 key/value heads at head_dim 128). It changes
        #: arithmetic, so the scalar kernel stays as the strict fallback and
        #: ``HIPENGINE_YUE2_NAR_ATTENTION=scalar`` selects it.
        self._wmma_attention = (
            os.environ.get("HIPENGINE_YUE2_NAR_ATTENTION", "wmma").lower() != "scalar"
            and head_dim == 128
            and heads == 16
            and kv_heads == 8
        )

        def keep(buffer: DeviceBuffer) -> DeviceBuffer:
            self._buffers.append(buffer)
            return buffer

        def upload(host: np.ndarray) -> DeviceBuffer:
            return keep(_upload(np.ascontiguousarray(host)))

        def fp16(bits: np.ndarray) -> DeviceBuffer:
            return upload(bf16_bits_to_fp16_bits(bits))

        self.layers: list[_NarLayerWeights] = []
        for layer in weights.layers:
            attention = layer.nar_self_attn
            self.layers.append(
                _NarLayerWeights(
                    input_ln=upload(layer.nar_input_layernorm),
                    q_norm=upload(attention.q_norm),
                    k_norm=upload(attention.k_norm),
                    pre_ln=upload(layer.nar_pre_mlp_layernorm),
                    q_w16=fp16(attention.q),
                    k_w16=fp16(attention.k),
                    v_w16=fp16(attention.v),
                    o_w16=fp16(attention.o),
                    gate_w16=fp16(layer.nar_mlp.gate),
                    up_w16=fp16(layer.nar_mlp.up),
                    down_w16=fp16(layer.nar_mlp.down),
                )
            )
        self.final_ln = upload(weights.norm)
        self.vae2llm_w16 = fp16(weights.vae2llm_weight)
        self.llm2vae_w16 = fp16(weights.llm2vae_weight)
        self.time_first_w16 = fp16(weights.time_embedder.first_weight)
        self.time_second_w16 = fp16(weights.time_embedder.second_weight)
        # The biases stay FP32: a fused Linear adds them to the FP32 accumulator
        # and rounds once, which the BF16-stored checkpoint values cannot express.
        self.vae2llm_b = upload(bf16_bits_to_f32(weights.vae2llm_bias))
        self.llm2vae_b = upload(bf16_bits_to_f32(weights.llm2vae_bias))
        self.time_first_b = upload(bf16_bits_to_f32(weights.time_embedder.first_bias))
        self.time_second_b = upload(bf16_bits_to_f32(weights.time_embedder.second_bias))
        self.pos_table = upload(weights.latent_pos_embed)
        self.pos_rows = int(weights.latent_pos_embed.shape[0])
        cos, sin = rope_tables(int(spec.max_position_embeddings), head_dim, float(spec.rope_theta))
        self.cos = upload(cos)
        self.sin = upload(sin)
        self._ones = upload(np.full(hidden, 0x3F80, dtype=np.uint16))
        self._freqs = np.exp(
            -math.log(10000)
            * np.arange(TIME_FREQUENCY_SIZE // 2, dtype=np.float32)
            / (TIME_FREQUENCY_SIZE // 2)
        )
        self._lt: HipblasLt | None = None
        self._allocate(0)

    # -- allocation ----------------------------------------------------
    def _allocate(self, rows: int) -> None:
        """(Re)allocate the persistent and tile scratch for ``rows`` positions."""

        hidden = self._hidden_size
        ffn = self._ffn
        latent = self._latent
        q_width = self._q_width
        kv_width = self._kv_width
        for buffer in self._buffers:
            if buffer in getattr(self, "_scratch", ()) or buffer in getattr(self, "_persistent", ()):
                free(buffer)
                self._buffers.remove(buffer)
        self._rows = rows
        self._tile = min(rows, _TILE_ROWS) if rows else 0
        tile = self._tile
        persistent = [
            _alloc(rows * hidden * 2),        # hidden (bf16)
            _alloc(rows * hidden * 2),        # normed (bf16)
            _alloc(rows * kv_width * 2),      # nar keys (bf16)
            _alloc(rows * kv_width * 2),      # nar values (bf16)
            _alloc(rows * latent * 2),        # state (bf16, padded)
            _alloc(rows * latent * 2),        # midpoint state
            _alloc(rows * latent * 2),        # first velocity
            _alloc(rows * latent * 2),        # second velocity
            _alloc(rows * latent * 2),        # fp16 latent GEMM input
            _alloc(rows * 8),                 # rope positions
            _alloc(rows * 8),                 # audio positions
            _alloc(rows * hidden * 4),        # (reserved) hidden f32
        ]
        scratch = [
            _alloc(tile * hidden * 2),        # normed16 (fp16)
            _alloc(tile * hidden * 2),        # normed2 (bf16)
            _alloc(tile * hidden * 2),        # normed216 (fp16)
            _alloc(tile * q_width * 4),       # q f32
            _alloc(tile * q_width * 2),       # q bf16
            _alloc(tile * q_width * 2),       # q normed bf16
            _alloc(tile * q_width * 4),       # q normed f32
            _alloc(tile * q_width * 4),       # q rope out (f32)
            _alloc(tile * kv_width * 4),      # k f32
            _alloc(tile * kv_width * 2),      # k bf16
            _alloc(tile * kv_width * 2),      # k normed bf16
            _alloc(tile * kv_width * 4),      # k normed f32
            _alloc(tile * kv_width * 4),      # k rope out (f32)
            _alloc(tile * kv_width * 4),      # v f32
            _alloc(tile * q_width * 4),       # attention f32
            _alloc(tile * q_width * 2),       # attention bf16
            _alloc(tile * q_width * 2),       # attention fp16
            _alloc(tile * hidden * 4),        # o f32
            _alloc(tile * hidden * 2),        # o bf16
            _alloc(tile * ffn * 4),           # gate f32
            _alloc(tile * ffn * 4),           # up f32
            _alloc(tile * ffn * 2),           # gate bf16
            _alloc(tile * ffn * 2),           # up bf16
            _alloc(tile * ffn * 2),           # act bf16
            _alloc(tile * ffn * 2),           # act fp16
            _alloc(tile * hidden * 4),        # down f32
            _alloc(tile * hidden * 2),        # down bf16
            _alloc(TIME_FREQUENCY_SIZE * 2),  # timestep sinusoid (fp16)
            _alloc(hidden * 4),               # timestep f32
            _alloc(hidden * 2),               # timestep bf16
            _alloc(hidden * 2),               # timestep act bf16
            _alloc(hidden * 2),               # timestep act fp16
        ]
        self._persistent = persistent
        self._scratch = scratch
        self._buffers.extend(persistent)
        self._buffers.extend(scratch)
        (
            self._hidden, self._normed, self._nar_k, self._nar_v, self._state,
            self._mid, self._first_velocity, self._out_latent, self._state16,
            self._rope_positions, self._audio_positions, self._hidden_f32,
        ) = persistent
        (
            self._normed16, self._normed2, self._normed216, self._q_f32, self._q_bf16,
            self._q_normed_bf16, self._q_normed_f32, self._q_out, self._k_f32, self._k_bf16,
            self._k_normed_bf16, self._k_normed_f32, self._k_out, self._v_f32, self._attn_f32,
            self._attn_bf16, self._attn16, self._o_f32, self._o_bf16, self._gate_f32,
            self._up_f32, self._gate_bf16, self._up_bf16, self._act_bf16, self._act16,
            self._down_f32, self._down_bf16, self._time_sin16, self._time_f32,
            self._time_bf16, self._time_act_bf16, self._time_act16,
        ) = scratch
        if rows:
            copy_host_array_to_device(
                self._audio_positions,
                np.ascontiguousarray(
                    audio_position_rows(range(rows), self.pos_rows), dtype=np.int64
                ),
            )
            copy_host_array_to_device(
                self._state, np.zeros((rows, latent), dtype=np.uint16)
            )
            copy_host_array_to_device(
                self._mid, np.zeros((rows, latent), dtype=np.uint16)
            )
        self._prepared.clear()

    def _keep(self, buffer: DeviceBuffer) -> DeviceBuffer:
        self._buffers.append(buffer)
        return buffer

    # -- hipBLASLt plumbing --------------------------------------------
    def _prepare_gemm(self, rows: int) -> None:
        """Build the hipBLASLt problems for this chunk's row counts.

        Buffers are sized for the largest chunk seen, so a later, shorter chunk
        keeps the older ``_tile`` and runs as a single ``rows``-row tile. Problems
        must therefore exist for every row count the solver will actually launch
        with, not just for the buffer's tile.
        """

        tile = self._tile
        counts = sorted({min(tile, rows - start) for start in range(0, rows, tile)}) if rows else []
        if all(count in self._prepared for count in counts):
            return
        if self._lt is None:
            if self.ar._lt is None:
                self.ar._lt = HipblasLt()
            self._lt = self.ar._lt
        hidden = self._hidden_size
        ffn = self._ffn
        latent = self._latent
        q_width = self._q_width
        kv_width = self._kv_width
        keys: list[tuple[int, int, int]] = []
        for count in counts:
            keys.extend(
                [
                    (count, latent, hidden),
                    (count, hidden, latent),
                    (count, hidden, kv_width),
                    (count, hidden, q_width),
                    (count, q_width, hidden),
                    (count, hidden, ffn),
                    (count, ffn, hidden),
                ]
            )
        keys.extend([(1, TIME_FREQUENCY_SIZE, hidden), (1, hidden, hidden)])
        for key in keys:
            if key[0] == 0 or key in self.ar._lt_problems:
                continue
            problem = self._lt.problem(key[0], key[1], key[2], 0)
            self.ar._lt_problems[key] = problem
            self.ar._lt_algos[key] = problem.fast_algorithm()
        self._prepared.update(counts)

    def _gemm(self, x16: DeviceBuffer, weight16: DeviceBuffer, out: DeviceBuffer, rows, inputs, outputs) -> None:
        key = (rows, inputs, outputs)
        self.ar._lt_problems[key].launch(
            self.ar._lt_algos[key], x16.ptr, weight16.ptr, out.ptr, stream=0
        )

    # -- conditioning --------------------------------------------------
    def condition(self, chunk: AcousticChunk) -> None:
        """Prefill one chunk's AR conditioning and keep its per-layer K/V."""

        validate_chunk_context(chunk, int(self.spec.max_position_embeddings))
        if chunk.nar_length > self._rows:
            self._allocate(chunk.nar_length)
        self._chunk = chunk
        self._visible = visible_ar_length(chunk)
        rows = chunk.nar_length
        copy_host_array_to_device(
            self._rope_positions,
            np.ascontiguousarray(
                chunk.ar_length + np.arange(rows, dtype=np.int64), dtype=np.int64
            ),
        )
        self.ar.reset(branch=0)
        self.ar.prefill_host_rows(
            [self.ar.embed_row(int(token)) for token in chunk.ar_tokens], branch=0, start_pos=0
        )
        self._prepare_gemm(rows)
        self.load_state(chunk.noise)

    def load_state(self, state: np.ndarray) -> None:
        """Stage FP32 ``[frames, 64]`` values as the ODE state.

        The reference pads the state with one zero row at each end for every
        velocity evaluation; the runtime keeps that padding in place, so only the
        content rows are written here.
        """

        chunk = self._chunk
        if chunk is None:
            raise RuntimeError("condition() must be called before load_state()")
        values = np.asarray(state, dtype=np.float32)
        if values.shape != (chunk.frames, self._latent):
            raise ValueError(f"state must have shape ({chunk.frames}, {self._latent})")
        if not np.isfinite(values).all():
            raise ValueError("state contains non-finite values")
        padded = np.zeros((chunk.frames + 2, self._latent), dtype=np.uint16)
        padded[1:-1] = f32_to_bf16_bits(values)
        copy_host_array_to_device(self._state, np.ascontiguousarray(padded))
        # ``_mid`` keeps the same zero boundary rows as the reference's own padded
        # midpoint state. The solver's state-update kernel only writes the content
        # rows, so a chunk shorter than the previous one would otherwise evaluate
        # its midpoint against stale boundary values left by the longer chunk.
        copy_host_array_to_device(self._mid, np.ascontiguousarray(padded))

    # -- velocity ------------------------------------------------------
    def _timestep_embedding(self, raw_t: float) -> None:
        """``TimestepEmbedder`` output for one raw timestep, in BF16.

        The reference rounds the raw value to the model dtype *before* the
        sigmoid, then feeds the BF16-shifted timestep through the sinusoid and the
        two-layer MLP; the first linear's bias is added to the FP32 accumulator so
        the projection rounds once, as a fused Linear does.
        """

        hidden = self._hidden_size
        kernels = self.kernels
        shifted = float(bf16_bits_to_f32(shift_t_value(raw_t, float(self.spec.timestep_shift)))[0])
        sinusoid = bf16_bits_to_fp16_bits(f32_to_bf16_bits(timestep_sinusoid(shifted)))
        host = _upload(sinusoid)
        try:
            self._gemm(host, self.time_first_w16, self._time_f32, 1, TIME_FREQUENCY_SIZE, hidden)
        finally:
            free(host)
        kernels.vv_add_bias_f32(
            self._time_f32.ptr, self.time_first_b.ptr, self._time_f32.ptr, hidden, hidden,
            library=self.library, runtime=self.runtime,
        )
        kernels.f32_to_bf16(self._time_f32.ptr, self._time_bf16.ptr, hidden, stream=0, runtime=self.runtime)
        kernels.silu_mul_separate_out_bf16(
            self._time_bf16.ptr, self._ones.ptr, self._time_act_bf16.ptr, 1, hidden,
            stream=0, runtime=self.runtime,
        )
        kernels.bf16_to_fp16(
            self._time_act_bf16.ptr, self._time_act16.ptr, hidden, stream=0, runtime=self.runtime
        )
        self._gemm(self._time_act16, self.time_second_w16, self._time_f32, 1, hidden, hidden)
        kernels.vv_add_bias_f32(
            self._time_f32.ptr, self.time_second_b.ptr, self._time_f32.ptr, hidden, hidden,
            library=self.library, runtime=self.runtime,
        )
        kernels.f32_to_bf16(self._time_f32.ptr, self._time_bf16.ptr, hidden, stream=0, runtime=self.runtime)

    def _velocity(self, state: DeviceBuffer, raw_t: float, out: DeviceBuffer) -> None:
        """Evaluate the flow-matching velocity at ``state`` into ``out``.

        Every row is covered by the key/value projection of each layer before any
        query tile attends, because the NAR attention is bidirectional; only the
        query-side scratch is tiled.
        """

        chunk = self._chunk
        if chunk is None:
            raise RuntimeError("condition() must be called before velocity()")
        rows = chunk.nar_length
        tile = self._tile
        hidden = self._hidden_size
        latent = self._latent
        q_width = self._q_width
        kv_width = self._kv_width
        eps = float(self.spec.rms_norm_eps)
        kernels = self.kernels
        self._timestep_embedding(raw_t)
        for start in range(0, rows, tile):
            count = min(tile, rows - start)
            kernels.bf16_to_fp16(
                state.ptr + start * latent * 2, self._state16.ptr, count * latent,
                stream=0, runtime=self.runtime,
            )
            self._gemm(self._state16, self.vae2llm_w16, self._o_f32, count, latent, hidden)
            kernels.vv_add_bias_f32(
                self._o_f32.ptr, self.vae2llm_b.ptr, self._o_f32.ptr, count * hidden, hidden,
                library=self.library, runtime=self.runtime,
            )
            kernels.f32_to_bf16(
                self._o_f32.ptr, self._hidden.ptr + start * hidden * 2, count * hidden,
                stream=0, runtime=self.runtime,
            )
        nar_kernels.nar_add_broadcast_bf16(
            self._hidden.ptr, self._time_bf16.ptr, self._hidden.ptr, rows, hidden,
            runtime=self.runtime,
        )
        nar_kernels.nar_gather_add_bf16(
            self._hidden.ptr, self.pos_table.ptr, self._audio_positions.ptr, self._hidden.ptr,
            rows, hidden, runtime=self.runtime,
        )
        if self.debug_taps is not None:
            self.debug_taps.append(self._read_rows(self._hidden, rows, hidden))
        for index, layer in enumerate(self.layers):
            kernels.vv_rmsnorm_bf16(
                self._hidden.ptr, layer.input_ln.ptr, self._normed.ptr, rows, hidden, eps,
                library=self.library, runtime=self.runtime,
            )
            ar_k = self.ar.layers[index].k_cache[0]
            ar_v = self.ar.layers[index].v_cache[0]
            for start in range(0, rows, tile):
                count = min(tile, rows - start)
                kernels.bf16_to_fp16(
                    self._normed.ptr + start * hidden * 2, self._normed16.ptr, count * hidden,
                    stream=0, runtime=self.runtime,
                )
                self._gemm(self._normed16, layer.k_w16, self._k_f32, count, hidden, kv_width)
                self._gemm(self._normed16, layer.v_w16, self._v_f32, count, hidden, kv_width)
                kernels.f32_to_bf16(
                    self._k_f32.ptr, self._k_bf16.ptr, count * kv_width, stream=0, runtime=self.runtime
                )
                kernels.vv_rmsnorm_bf16(
                    self._k_bf16.ptr, layer.k_norm.ptr, self._k_normed_bf16.ptr,
                    count * self._kv_heads, self._head_dim, eps,
                    library=self.library, runtime=self.runtime,
                )
                kernels.bf16_to_f32(
                    self._k_normed_bf16.ptr, self._k_normed_f32.ptr, count * kv_width,
                    stream=0, runtime=self.runtime,
                )
                nar_kernels.nar_rope_f32(
                    0, self._k_normed_f32.ptr, self.cos.ptr, self.sin.ptr,
                    self._rope_positions.ptr + start * 8, 0, self._k_out.ptr,
                    count, 0, self._kv_heads, self._head_dim, runtime=self.runtime,
                )
                kernels.f32_to_bf16(
                    self._k_out.ptr, self._nar_k.ptr + start * kv_width * 2, count * kv_width,
                    stream=0, runtime=self.runtime,
                )
                kernels.f32_to_bf16(
                    self._v_f32.ptr, self._nar_v.ptr + start * kv_width * 2, count * kv_width,
                    stream=0, runtime=self.runtime,
                )
            for start in range(0, rows, tile):
                count = min(tile, rows - start)
                hidden_off = start * hidden * 2
                kernels.bf16_to_fp16(
                    self._normed.ptr + hidden_off, self._normed16.ptr, count * hidden,
                    stream=0, runtime=self.runtime,
                )
                self._gemm(self._normed16, layer.q_w16, self._q_f32, count, hidden, q_width)
                kernels.f32_to_bf16(self._q_f32.ptr, self._q_bf16.ptr, count * q_width, stream=0, runtime=self.runtime)
                kernels.vv_rmsnorm_bf16(
                    self._q_bf16.ptr, layer.q_norm.ptr, self._q_normed_bf16.ptr,
                    count * self._heads, self._head_dim, eps,
                    library=self.library, runtime=self.runtime,
                )
                kernels.bf16_to_f32(
                    self._q_normed_bf16.ptr, self._q_normed_f32.ptr, count * q_width,
                    stream=0, runtime=self.runtime,
                )
                nar_kernels.nar_rope_f32(
                    self._q_normed_f32.ptr, 0, self.cos.ptr, self.sin.ptr,
                    self._rope_positions.ptr + start * 8, self._q_out.ptr, 0,
                    count, self._heads, 0, self._head_dim, runtime=self.runtime,
                )
                attention = (
                    nar_kernels.nar_attention_wmma
                    if self._wmma_attention
                    else nar_kernels.nar_attention_f32
                )
                attention(
                    self._q_out.ptr, self._nar_k.ptr, self._nar_v.ptr, ar_k.ptr, ar_v.ptr,
                    self._attn_f32.ptr, count, self._visible, self._heads, self._kv_heads,
                    self._head_dim, self._scale, runtime=self.runtime,
                )
                kernels.f32_to_bf16(self._attn_f32.ptr, self._attn_bf16.ptr, count * q_width, stream=0, runtime=self.runtime)
                kernels.bf16_to_fp16(self._attn_bf16.ptr, self._attn16.ptr, count * q_width, stream=0, runtime=self.runtime)
                self._gemm(self._attn16, layer.o_w16, self._o_f32, count, q_width, hidden)
                kernels.f32_to_bf16(self._o_f32.ptr, self._o_bf16.ptr, count * hidden, stream=0, runtime=self.runtime)
                kernels.vv_scale_residual_bf16(
                    self._hidden.ptr + hidden_off, self._o_bf16.ptr, self._ones.ptr,
                    self._hidden.ptr + hidden_off, count * hidden, hidden,
                    library=self.library, runtime=self.runtime,
                )
                kernels.vv_rmsnorm_bf16(
                    self._hidden.ptr + hidden_off, layer.pre_ln.ptr, self._normed2.ptr,
                    count, hidden, eps, library=self.library, runtime=self.runtime,
                )
                kernels.bf16_to_fp16(self._normed2.ptr, self._normed216.ptr, count * hidden, stream=0, runtime=self.runtime)
                self._gemm(self._normed216, layer.gate_w16, self._gate_f32, count, hidden, self._ffn)
                self._gemm(self._normed216, layer.up_w16, self._up_f32, count, hidden, self._ffn)
                kernels.f32_to_bf16(self._gate_f32.ptr, self._gate_bf16.ptr, count * self._ffn, stream=0, runtime=self.runtime)
                kernels.f32_to_bf16(self._up_f32.ptr, self._up_bf16.ptr, count * self._ffn, stream=0, runtime=self.runtime)
                kernels.silu_mul_separate_out_bf16(
                    self._gate_bf16.ptr, self._up_bf16.ptr, self._act_bf16.ptr, count, self._ffn,
                    stream=0, runtime=self.runtime,
                )
                kernels.bf16_to_fp16(self._act_bf16.ptr, self._act16.ptr, count * self._ffn, stream=0, runtime=self.runtime)
                self._gemm(self._act16, layer.down_w16, self._down_f32, count, self._ffn, hidden)
                kernels.f32_to_bf16(self._down_f32.ptr, self._down_bf16.ptr, count * hidden, stream=0, runtime=self.runtime)
                kernels.vv_scale_residual_bf16(
                    self._hidden.ptr + hidden_off, self._down_bf16.ptr, self._ones.ptr,
                    self._hidden.ptr + hidden_off, count * hidden, hidden,
                    library=self.library, runtime=self.runtime,
                )
            if self.debug_taps is not None:
                self.debug_taps.append(self._read_rows(self._hidden, rows, hidden))
        for start in range(0, rows, tile):
            count = min(tile, rows - start)
            kernels.vv_rmsnorm_bf16(
                self._hidden.ptr + start * hidden * 2, self.final_ln.ptr, self._normed2.ptr,
                count, hidden, eps, library=self.library, runtime=self.runtime,
            )
            kernels.bf16_to_fp16(self._normed2.ptr, self._normed216.ptr, count * hidden, stream=0, runtime=self.runtime)
            self._gemm(self._normed216, self.llm2vae_w16, self._o_f32, count, hidden, latent)
            kernels.vv_add_bias_f32(
                self._o_f32.ptr, self.llm2vae_b.ptr, self._o_f32.ptr, count * latent, latent,
                library=self.library, runtime=self.runtime,
            )
            kernels.f32_to_bf16(
                self._o_f32.ptr, out.ptr + start * latent * 2, count * latent,
                stream=0, runtime=self.runtime,
            )

    def _read_rows(self, buffer: DeviceBuffer, rows: int, width: int) -> np.ndarray:
        bits = np.empty((rows, width), dtype=np.uint16)
        self.runtime.memcpy(
            host_array_ptr(bits), buffer.ptr, rows * width * 2, MemcpyKind.DEVICE_TO_HOST
        )
        return bits

    def velocity_bits(self, raw_t: float) -> np.ndarray:
        """BF16 velocity at the current state, content rows only (diagnostics)."""

        chunk = self._chunk
        if chunk is None:
            raise RuntimeError("condition() must be called before velocity()")
        self._velocity(self._state, raw_t, self._out_latent)
        bits = np.empty((chunk.frames, self._latent), dtype=np.uint16)
        self.runtime.memcpy(
            host_array_ptr(bits), self._out_latent.ptr + self._latent * 2,
            chunk.frames * self._latent * 2, MemcpyKind.DEVICE_TO_HOST,
        )
        return bits

    def state_bits(self) -> np.ndarray:
        """BF16 ODE state for the current chunk, content rows only."""

        chunk = self._chunk
        if chunk is None:
            raise RuntimeError("condition() must be called before solve()")
        bits = np.empty((chunk.frames, self._latent), dtype=np.uint16)
        self.runtime.memcpy(
            host_array_ptr(bits), self._state.ptr + self._latent * 2,
            chunk.frames * self._latent * 2, MemcpyKind.DEVICE_TO_HOST,
        )
        return bits

    def solve(self, steps: int = 32) -> np.ndarray:
        """Run the reference midpoint rule and return FP32 latents [frames, 64]."""

        chunk = self._chunk
        if chunk is None:
            raise RuntimeError("condition() must be called before solve()")
        schedule = solver_schedule(steps)
        h = 1.0 / int(steps)
        frames = chunk.frames
        latent = self._latent
        offset = latent * 2
        for raw, raw_mid in schedule:
            # ``nar_state_update_bf16`` computes ``state - bf16(v * scale)``, so
            # both scales are the positive step sizes: the midpoint state uses
            # ``h/2`` and the step itself uses ``h``.
            self._velocity(self._state, raw, self._first_velocity)
            nar_kernels.nar_state_update_bf16(
                self._state.ptr + offset, self._first_velocity.ptr + offset, h / 2.0,
                self._mid.ptr + offset, frames, latent,
                runtime=self.runtime,
            )
            self._velocity(self._mid, raw_mid, self._out_latent)
            nar_kernels.nar_state_update_bf16(
                self._state.ptr + offset, self._out_latent.ptr + offset, h,
                self._state.ptr + offset, frames, latent,
                runtime=self.runtime,
            )
        result = bf16_bits_to_f32(self.state_bits())
        if not np.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def synthesize(
        self,
        prefix: Sequence[int],
        codec: Sequence[int],
        seed: int,
        *,
        steps: int = 32,
        context: int = CONTEXT,
        noise: np.ndarray | None = None,
        nar_cond_end: int = 0,
    ) -> np.ndarray:
        """Solve every original chunk in order and concatenate the latents."""

        chunks = song_chunks(
            prefix, codec, seed, context=context, noise=noise, nar_cond_end=nar_cond_end
        )
        latents = []
        for chunk in chunks:
            self.condition(chunk)
            latents.append(self.solve(steps))
        return latents[0] if len(latents) == 1 else np.concatenate(latents, axis=0)

    def close(self) -> None:
        for buffer in list(self._buffers):
            free(buffer)
        self._buffers.clear()
        self._persistent = []
        self._scratch = []
        self._rows = 0
        self._tile = 0
        self._chunk = None
        self._prepared.clear()


def _alloc(nbytes: int) -> DeviceBuffer:
    return malloc(max(int(nbytes), 8))


def _upload(host: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(host)
    buffer = malloc(array.nbytes)
    copy_host_array_to_device(buffer, array)
    return buffer


@dataclass
class _NarLayerWeights:
    input_ln: DeviceBuffer
    q_norm: DeviceBuffer
    k_norm: DeviceBuffer
    pre_ln: DeviceBuffer
    q_w16: DeviceBuffer
    k_w16: DeviceBuffer
    v_w16: DeviceBuffer
    o_w16: DeviceBuffer
    gate_w16: DeviceBuffer
    up_w16: DeviceBuffer
    down_w16: DeviceBuffer
