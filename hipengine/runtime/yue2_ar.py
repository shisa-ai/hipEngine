"""YuE2 AR (semantic / ABC) runtime on the gfx11 kernel family.

One code path serves prefill and decode: rows stream through the 28 AR layers of
the mixture-of-transformers with a dense uniform KV cache
``(max_context, num_kv_heads, head_dim)`` bf16 per layer, addressed through the
``KVLiveSpans`` ABI (identity slots, dense policy). The NAR attention/MLP paths
in each layer belong to the flow-matching phase and are not touched here.

Precision follows the released reference: bf16 weights and hidden stream, fp32
projection accumulators rounded to bf16 before the per-head Q/K norm (as the
reference's ``nn.Linear`` output is bf16), fp32 attention and normalization
statistics, bf16 residual adds. The reference builds its RoPE table from an fp32
angle; ``rope_table='reference'`` reproduces that table and ``rope_table='f64'``
computes the same angles in double precision, which is the more accurate of the
two at long positions.

Config-forced generation runs two branches (positive and negative prefixes) over
the same weights; each branch owns its KV caches, staged hidden row and context
length, while the launch scratch is shared because branches run in sequence.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.hipblaslt import HipblasLt
from hipengine.core.memory import (
    DeviceBuffer,
    DeviceMemoryArena,
    copy_device_to_host,
    copy_host_array_to_device,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.core.tensor import Tensor
from hipengine.kernels.vibevoice import resolve_vibevoice_kernels
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.yue2 import YuE2Weights

#: Largest context the in-tree span attention kernel can hold in shared memory.
SPAN_ATTENTION_MAX_CONTEXT = 16000


def _upload(host: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(host)
    buffer = malloc(array.nbytes)
    copy_host_array_to_device(buffer, array)
    return buffer


def _alloc(nbytes: int) -> DeviceBuffer:
    return malloc(max(int(nbytes), 8))


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round fp32 to bf16 bits (round-to-nearest-even, as the reference casts)."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    wide = array.view(np.uint32).astype(np.uint64)
    rounded = ((wide + np.uint64(0x7FFF) + ((wide >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16))
    return rounded.astype(np.uint16)


def bf16_bits_to_fp16_bits(bits: np.ndarray) -> np.ndarray:
    """bf16 bit patterns -> fp16 bit patterns (exact for the checkpoint's range)."""
    return np.ascontiguousarray(bf16_bits_to_f32(bits).astype(np.float16).view(np.uint16))


def rope_tables(
    max_positions: int, head_dim: int, theta: float, *, mode: str = "reference"
) -> tuple[np.ndarray, np.ndarray]:
    """Half-duplicated cos/sin tables of width ``head_dim``.

    ``mode='reference'`` mirrors the released implementation (fp32 positions and
    fp32 inverse frequencies, trig in fp32); ``mode='f64'`` keeps the angle in
    double precision until the final cast.
    """
    half = head_dim // 2
    if mode == "reference":
        # Exponents are k/64 for k in [0, 64), i.e. the reference's
        # arange(0, head_dim, 2) / head_dim, evaluated in fp32 like the reference.
        inverse = (1.0 / (np.float32(theta) ** (np.arange(0, half, dtype=np.float32) / np.float32(half)))).astype(
            np.float32
        )
        positions = np.arange(max_positions, dtype=np.float32)
        angles = positions[:, None] * inverse[None, :]
        cos_half = np.cos(angles).astype(np.float32)
        sin_half = np.sin(angles).astype(np.float32)
    elif mode == "f64":
        inverse = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        angles = np.arange(max_positions, dtype=np.float64)[:, None] * inverse[None, :]
        cos_half = np.cos(angles).astype(np.float32)
        sin_half = np.sin(angles).astype(np.float32)
    else:
        raise ValueError("rope table mode must be 'reference' or 'f64'")
    return (
        np.ascontiguousarray(np.concatenate([cos_half, cos_half], axis=1)),
        np.ascontiguousarray(np.concatenate([sin_half, sin_half], axis=1)),
    )


@dataclass
class _LayerBuffers:
    input_ln: DeviceBuffer
    q_w: DeviceBuffer
    k_w: DeviceBuffer
    v_w: DeviceBuffer
    o_w: DeviceBuffer
    q_norm: DeviceBuffer
    k_norm: DeviceBuffer
    post_ln: DeviceBuffer
    gate_w: DeviceBuffer
    up_w: DeviceBuffer
    down_w: DeviceBuffer
    q_w16: DeviceBuffer
    k_w16: DeviceBuffer
    v_w16: DeviceBuffer
    o_w16: DeviceBuffer
    gate_w16: DeviceBuffer
    up_w16: DeviceBuffer
    down_w16: DeviceBuffer
    k_cache: list[DeviceBuffer]
    v_cache: list[DeviceBuffer]


class _PrefillArena:
    """Per-runtime arena for one prefill call's scratch views."""

    _ALIGNMENT = 4096

    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._arena: DeviceMemoryArena | None = None

    def take(self, sizes: Sequence[int]) -> list[DeviceBuffer]:
        alignment = self._ALIGNMENT
        needed = sum(-(-int(size) // alignment) * alignment for size in sizes)
        arena = self._arena
        if arena is None or arena.closed or arena.capacity_bytes < needed:
            if arena is not None:
                arena.close()
            arena = DeviceMemoryArena.create(needed, runtime=self._runtime, alignment=alignment)
            self._arena = arena
        else:
            arena.rewind()
        return [arena.allocate(int(size)) for size in sizes]

    def close(self) -> None:
        if self._arena is not None:
            self._arena.close()
            self._arena = None


class Yue2ArRuntime:
    """AR-path runner for :class:`~hipengine.loading.yue2.YuE2Weights`."""

    quant_name = "bf16"

    def __init__(
        self,
        weights: YuE2Weights,
        *,
        max_context: int = 4096,
        branches: int = 1,
        backend: str = "auto",
        prefill_variant: str = "hipblaslt",
        rope_table: str = "reference",
        library: ctypes.CDLL | None = None,
    ) -> None:
        if isinstance(max_context, bool) or not isinstance(max_context, Integral):
            raise ValueError("max_context must be an integer")
        if not 0 < max_context <= SPAN_ATTENTION_MAX_CONTEXT:
            raise ValueError(
                f"max_context must be in [1, {SPAN_ATTENTION_MAX_CONTEXT}] for the span attention route"
            )
        if branches not in (1, 2):
            raise ValueError("branches must be 1 (no CFG) or 2 (CFG)")
        if prefill_variant not in ("strict", "hipblaslt"):
            raise ValueError("prefill_variant must be strict or hipblaslt")
        spec = weights.config
        self.weights = weights
        self.spec = spec
        self.max_context = int(max_context)
        self.branches = int(branches)
        self.rope_table_mode = rope_table
        self.prefill_variant = prefill_variant
        self.prefill_fallback_reason: str | None = None
        self.kernels = resolve_vibevoice_kernels(backend)
        self.backend = self.kernels.backend
        self.runtime = get_hip_runtime()
        self.library = library or self.kernels.build_vibevoice_encoder()

        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        kv_heads = spec.num_key_value_heads
        head_dim = spec.head_dim
        ffn = spec.intermediate_size
        self._buffers: list[DeviceBuffer] = []

        def keep(buffer: DeviceBuffer) -> DeviceBuffer:
            self._buffers.append(buffer)
            return buffer

        self.embed_host_bf16 = np.ascontiguousarray(weights.embed_tokens)
        self.embed = keep(_upload(self.embed_host_bf16))
        self.final_ln = keep(_upload(weights.norm))
        self.lm_head = keep(_upload(weights.lm_head))
        self._ones_hidden = keep(_upload(np.full(hidden, 0x3F80, dtype=np.uint16)))

        cos, sin = rope_tables(self.max_context, head_dim, spec.rope_theta, mode=rope_table)
        self._cos = keep(_upload(cos))
        self._sin = keep(_upload(sin))

        kv_bytes = self.max_context * kv_heads * head_dim * 2
        # The fp16 copies exist only for the hipBLASLt prefill route; the decode
        # route and the strict incremental prefill read the bf16 weights.
        use_fp16 = prefill_variant == "hipblaslt"

        def fp16_copy(bits: np.ndarray) -> DeviceBuffer:
            if not use_fp16:
                return keep(_alloc(8))
            return keep(_upload(bf16_bits_to_fp16_bits(bits)))

        self.layers: list[_LayerBuffers] = []
        for layer in weights.layers:
            attention = layer.self_attn
            keep_alive = []
            for _ in range(self.branches):
                keep_alive.append(keep(_alloc(kv_bytes)))
            value_buffers = []
            for _ in range(self.branches):
                value_buffers.append(keep(_alloc(kv_bytes)))
            self.layers.append(
                _LayerBuffers(
                    input_ln=keep(_upload(layer.input_layernorm)),
                    q_w=keep(_upload(attention.q)),
                    k_w=keep(_upload(attention.k)),
                    v_w=keep(_upload(attention.v)),
                    o_w=keep(_upload(attention.o)),
                    q_norm=keep(_upload(attention.q_norm)),
                    k_norm=keep(_upload(attention.k_norm)),
                    post_ln=keep(_upload(layer.post_attention_layernorm)),
                    gate_w=keep(_upload(layer.mlp.gate)),
                    up_w=keep(_upload(layer.mlp.up)),
                    down_w=keep(_upload(layer.mlp.down)),
                    q_w16=fp16_copy(attention.q),
                    k_w16=fp16_copy(attention.k),
                    v_w16=fp16_copy(attention.v),
                    o_w16=fp16_copy(attention.o),
                    gate_w16=fp16_copy(layer.mlp.gate),
                    up_w16=fp16_copy(layer.mlp.up),
                    down_w16=fp16_copy(layer.mlp.down),
                    k_cache=keep_alive,
                    v_cache=value_buffers,
                )
            )

        self._slot_map = keep(_upload(np.arange(self.max_context, dtype=np.int32)))
        self._slot_positions = keep(_upload(np.arange(self.max_context, dtype=np.int64)))
        self._evicted = keep(_upload(np.zeros(self.max_context, dtype=np.bool_)))
        self._ctx_len = [_alloc(8) for _ in range(self.branches)]
        self._ctx_len_host = [np.zeros(1, dtype=np.int64) for _ in range(self.branches)]
        self._row_position = [_alloc(8) for _ in range(self.branches)]
        self._hidden = [keep(_alloc(hidden * 2)) for _ in range(self.branches)]

        self._normed = _alloc(hidden * 2)
        self._q_f32 = _alloc(heads * head_dim * 4)
        self._k_f32 = _alloc(kv_heads * head_dim * 4)
        self._v_f32 = _alloc(kv_heads * head_dim * 4)
        self._q_bf16 = _alloc(heads * head_dim * 2)
        self._k_bf16 = _alloc(kv_heads * head_dim * 2)
        self._v_bf16 = _alloc(kv_heads * head_dim * 2)
        self._q_normed_bf16 = _alloc(heads * head_dim * 2)
        self._k_normed_bf16 = _alloc(kv_heads * head_dim * 2)
        self._q_normed_f32 = _alloc(heads * head_dim * 4)
        self._k_normed_f32 = _alloc(kv_heads * head_dim * 4)
        self._q_out = _alloc(heads * head_dim * 4)
        self._k_out = _alloc(kv_heads * head_dim * 4)
        self._attn = _alloc(heads * head_dim * 4)
        self._attn_bf16 = _alloc(heads * head_dim * 2)
        self._o_f32 = _alloc(hidden * 4)
        self._o_bf16 = _alloc(hidden * 2)
        self._gate_up = _alloc(2 * ffn * 2)
        self._silu = _alloc(ffn * 2)
        self._down_bf16 = _alloc(hidden * 2)
        self._logits_bf16 = _alloc(spec.vocab_size * 2)
        self._logits_f32 = _alloc(spec.vocab_size * 4)
        self._scale = 1.0 / float(np.sqrt(head_dim))
        self._lt: HipblasLt | None = None
        self._lt_problems: dict[tuple[int, int, int], object] = {}
        self._lt_algos: dict[tuple[int, int, int], object] = {}
        self._prefill_arena = _PrefillArena(self.runtime)

    # ------------------------------------------------------------------
    # input staging
    # ------------------------------------------------------------------
    def embed_row(self, token_id: int) -> np.ndarray:
        """Host embedding lookup widened to fp32 (bf16 storage)."""
        row = self.embed_host_bf16[int(token_id)]
        return bf16_bits_to_f32(row)

    def push_token(self, token_or_embed: np.ndarray, position: int, branch: int = 0) -> None:
        """Stage one token's input row (fp32 hidden holding bf16 values).

        Positions are append-only per branch: ``position`` must be the branch's
        current context length. Call :meth:`reset` to start a new sequence.
        """
        self._validate_position(position)
        self._validate_branch(branch)
        expected = int(self._ctx_len_host[branch][0])
        if position != expected:
            raise ValueError(f"position must continue the branch context: expected {expected}, got {position}")
        row = np.ascontiguousarray(token_or_embed, dtype=np.float32).reshape(-1)
        if row.shape[0] != self.spec.hidden_size:
            raise ValueError("token row must be hidden-sized fp32")
        copy_host_array_to_device(self._hidden[branch], f32_to_bf16_bits(row))
        self._ctx_len_host[branch][0] = position + 1
        copy_host_to_device(self._ctx_len[branch], host_array_ptr(self._ctx_len_host[branch]))

    def reset(self, branch: int | None = None) -> None:
        for index in range(self.branches):
            if branch is None or branch == index:
                self._ctx_len_host[index][0] = 0

    def context_length(self, branch: int = 0) -> int:
        """Current context length of a branch, i.e. the next append position."""
        self._validate_branch(branch)
        return int(self._ctx_len_host[branch][0])

    def hidden_state(self, branch: int = 0) -> np.ndarray:
        """Post-final-norm hidden row as fp32 (bf16 values)."""
        hidden = self.spec.hidden_size
        self.kernels.vv_rmsnorm_bf16(
            self._hidden[branch].ptr, self.final_ln.ptr, self._normed.ptr, 1, hidden,
            self.spec.rms_norm_eps, library=self.library, runtime=self.runtime,
        )
        bits = np.empty(hidden, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(bits), self._normed, hidden * 2)
        return bf16_bits_to_f32(bits)

    def logits(self, branch: int = 0, *, as_bf16: bool = True) -> np.ndarray:
        """Final norm, lm head, and the vocabulary row as bf16 bits or fp32.

        Scratch buffers are per runtime, not per branch: call this for a branch
        before advancing another one.
        """
        hidden = self.spec.hidden_size
        self.kernels.vv_rmsnorm_bf16(
            self._hidden[branch].ptr, self.final_ln.ptr, self._normed.ptr, 1, hidden,
            self.spec.rms_norm_eps, library=self.library, runtime=self.runtime,
        )
        self.kernels.dense_gemv_bf16_f32_out(
            self._normed.ptr, self.lm_head.ptr, self._logits_f32.ptr, 1, hidden,
            self.spec.vocab_size, stream=0, runtime=self.runtime,
        )
        host = np.empty(self.spec.vocab_size, dtype=np.float32)
        copy_device_to_host(host_array_ptr(host), self._logits_f32, self.spec.vocab_size * 4)
        if as_bf16:
            return f32_to_bf16_bits(host)
        return host

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def forward_layers(self, position: int, branch: int = 0) -> None:
        """Run the staged hidden row through the AR layer stack at ``position``."""
        self._validate_position(position)
        self._validate_branch(branch)
        spec = self.spec
        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        kv_heads = spec.num_key_value_heads
        head_dim = spec.head_dim
        ffn = spec.intermediate_size
        kernels = self.kernels
        ctx = position + 1
        self._ctx_len_host[branch][0] = ctx
        copy_host_array_to_device(self._ctx_len[branch], self._ctx_len_host[branch])
        copy_host_array_to_device(self._row_position[branch], np.array([position], dtype=np.int64))
        spans = self._spans(branch, 1)

        for layer in self.layers:
            kernels.vv_rmsnorm_bf16(
                self._hidden[branch].ptr, layer.input_ln.ptr, self._normed.ptr, 1, hidden,
                spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            kernels.dense_gemv_bf16_f32_out(
                self._normed.ptr, layer.q_w.ptr, self._q_f32.ptr, 1, hidden, heads * head_dim,
                stream=0, runtime=self.runtime,
            )
            kernels.dense_gemv_bf16_f32_out(
                self._normed.ptr, layer.k_w.ptr, self._k_f32.ptr, 1, hidden, kv_heads * head_dim,
                stream=0, runtime=self.runtime,
            )
            kernels.dense_gemv_bf16_f32_out(
                self._normed.ptr, layer.v_w.ptr, self._v_f32.ptr, 1, hidden, kv_heads * head_dim,
                stream=0, runtime=self.runtime,
            )
            # nn.Linear emits bf16, so the head norm sees bf16-rounded inputs.
            kernels.f32_to_bf16(self._q_f32.ptr, self._q_bf16.ptr, heads * head_dim, stream=0, runtime=self.runtime)
            kernels.f32_to_bf16(self._k_f32.ptr, self._k_bf16.ptr, kv_heads * head_dim, stream=0, runtime=self.runtime)
            kernels.vv_rmsnorm_bf16(
                self._q_bf16.ptr, layer.q_norm.ptr, self._q_normed_bf16.ptr, heads, head_dim,
                spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            kernels.vv_rmsnorm_bf16(
                self._k_bf16.ptr, layer.k_norm.ptr, self._k_normed_bf16.ptr, kv_heads, head_dim,
                spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            kernels.bf16_to_f32(self._q_normed_bf16.ptr, self._q_normed_f32.ptr, heads * head_dim, stream=0, runtime=self.runtime)
            kernels.bf16_to_f32(self._k_normed_bf16.ptr, self._k_normed_f32.ptr, kv_heads * head_dim, stream=0, runtime=self.runtime)
            kernels.qwen35_partial_rotary_f32(
                self._q_normed_f32.ptr, self._k_normed_f32.ptr,
                self._cos.ptr + position * head_dim * 4,
                self._sin.ptr + position * head_dim * 4,
                self._q_out.ptr, self._k_out.ptr, heads, kv_heads, head_dim, head_dim,
                stream=0, runtime=self.runtime,
            )
            kernels.f32_to_bf16(self._k_out.ptr, self._k_bf16.ptr, kv_heads * head_dim, stream=0, runtime=self.runtime)
            kernels.f32_to_bf16(self._v_f32.ptr, self._v_bf16.ptr, kv_heads * head_dim, stream=0, runtime=self.runtime)
            kernels.vv_kv_write_spans(
                self._k_bf16.ptr, self._v_bf16.ptr, layer.k_cache[branch].ptr,
                layer.v_cache[branch].ptr, spans, 1, kv_heads, head_dim,
                library=self.library, runtime=self.runtime,
            )
            kernels.vv_attention_spans(
                self._q_out.ptr, layer.k_cache[branch].ptr, layer.v_cache[branch].ptr,
                self._attn.ptr, spans, 1, heads, kv_heads, head_dim, self._scale,
                library=self.library, runtime=self.runtime,
            )
            kernels.f32_to_bf16(self._attn.ptr, self._attn_bf16.ptr, heads * head_dim, stream=0, runtime=self.runtime)
            kernels.dense_gemv_bf16_f32_out(
                self._attn_bf16.ptr, layer.o_w.ptr, self._o_f32.ptr, 1, heads * head_dim, hidden,
                stream=0, runtime=self.runtime,
            )
            kernels.f32_to_bf16(self._o_f32.ptr, self._o_bf16.ptr, hidden, stream=0, runtime=self.runtime)
            kernels.vv_scale_residual_bf16(
                self._hidden[branch].ptr, self._o_bf16.ptr, self._ones_hidden.ptr,
                self._hidden[branch].ptr, hidden, hidden,
                library=self.library, runtime=self.runtime,
            )
            kernels.vv_rmsnorm_bf16(
                self._hidden[branch].ptr, layer.post_ln.ptr, self._normed.ptr, 1, hidden,
                spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            kernels.dense_dual_gemv_out_bf16(
                self._normed.ptr, layer.gate_w.ptr, layer.up_w.ptr, self._gate_up.ptr,
                1, hidden, ffn, ffn, stream=0, runtime=self.runtime,
            )
            kernels.silu_mul_dual_out_bf16(
                self._gate_up.ptr, self._silu.ptr, 1, ffn, stream=0, runtime=self.runtime,
            )
            kernels.dense_gemv_out_bf16(
                self._silu.ptr, layer.down_w.ptr, self._down_bf16.ptr, 1, ffn, hidden,
                stream=0, runtime=self.runtime,
            )
            kernels.vv_scale_residual_bf16(
                self._hidden[branch].ptr, self._down_bf16.ptr, self._ones_hidden.ptr,
                self._hidden[branch].ptr, hidden, hidden,
                library=self.library, runtime=self.runtime,
            )

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------
    def _prepare_lt(self, rows: int) -> None:
        if self._lt is None:
            self._lt = HipblasLt()
        hidden = self.spec.hidden_size
        ffn = self.spec.intermediate_size
        kv = self.spec.num_key_value_heads * self.spec.head_dim
        for inputs, outputs in ((hidden, hidden), (hidden, kv), (ffn, hidden), (hidden, ffn)):
            shape = (rows, inputs, outputs)
            if shape in self._lt_problems:
                continue
            problem = self._lt.problem(rows, inputs, outputs, 0)
            self._lt_problems[shape] = problem
            self._lt_algos[shape] = problem.fast_algorithm()

    def _lt_gemm(self, x16_ptr: int, weight_ptr: int, out_ptr: int, rows: int, inputs: int, outputs: int) -> None:
        shape = (rows, inputs, outputs)
        problem = self._lt_problems[shape]
        problem.launch(self._lt_algos[shape], x16_ptr, weight_ptr, out_ptr, stream=0)

    def prefill_host_rows(
        self, rows: Sequence[np.ndarray], *, branch: int = 0, start_pos: int = 0
    ) -> None:
        """Prefill host-side prompt rows (fp32 rows holding bf16 values)."""
        total = len(rows)
        if total == 0:
            raise ValueError("empty prompt")
        hidden = self.spec.hidden_size
        buffer = _upload(f32_to_bf16_bits(np.asarray(rows, dtype=np.float32).reshape(total, hidden)))
        try:
            self.prefill_rows(buffer, total, start_pos, branch=branch)
            self.runtime.memcpy(
                self._hidden[branch].ptr,
                buffer.ptr + (total - 1) * hidden * 2,
                hidden * 2,
                MemcpyKind.DEVICE_TO_DEVICE,
            )
        finally:
            free(buffer)

    def prefill_rows(
        self, hidden_rows: DeviceBuffer, rows: int, start_pos: int = 0, *, branch: int = 0
    ) -> None:
        """Batched causal prefill of ``rows`` bf16 rows at ``start_pos``."""
        self._validate_branch(branch)
        self._validate_position(start_pos)
        if isinstance(rows, bool) or not isinstance(rows, Integral) or rows <= 0:
            raise ValueError("prefill rows must be a positive integer")
        hidden = self.spec.hidden_size
        if start_pos + rows > self.max_context:
            raise ValueError("prefill exceeds max_context")
        if hidden_rows.nbytes < rows * hidden * 2:
            raise ValueError("prefill input buffer is too small")
        selected = self.prefill_variant
        self.prefill_fallback_reason = None
        if selected == "hipblaslt":
            try:
                self._prepare_lt(rows)
            except (OSError, RuntimeError) as error:
                selected = "strict"
                self.prefill_fallback_reason = str(error)
        if selected == "strict":
            self._prefill_incremental(hidden_rows, rows, start_pos, branch)
        else:
            self._prefill_batched(hidden_rows, rows, start_pos, branch)
        self._ctx_len_host[branch][0] = start_pos + rows
        copy_host_array_to_device(self._ctx_len[branch], self._ctx_len_host[branch])
        self.runtime.memcpy(
            self._hidden[branch].ptr,
            hidden_rows.ptr + (rows - 1) * hidden * 2,
            hidden * 2,
            MemcpyKind.DEVICE_TO_DEVICE,
        )

    def _prefill_incremental(self, hidden_rows: DeviceBuffer, rows: int, start_pos: int, branch: int) -> None:
        width = self.spec.hidden_size * 2
        for row in range(rows):
            position = start_pos + row
            self.runtime.memcpy(
                self._hidden[branch].ptr, hidden_rows.ptr + row * width, width, MemcpyKind.DEVICE_TO_DEVICE
            )
            self.forward_layers(position, branch)
            self.runtime.memcpy(
                hidden_rows.ptr + row * width, self._hidden[branch].ptr, width, MemcpyKind.DEVICE_TO_DEVICE
            )

    def _prefill_batched(self, hidden_rows: DeviceBuffer, rows: int, start_pos: int, branch: int) -> None:
        spec = self.spec
        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        kv_heads = spec.num_key_value_heads
        head_dim = spec.head_dim
        ffn = spec.intermediate_size
        kernels = self.kernels
        q_width = heads * head_dim
        kv_width = kv_heads * head_dim
        positions_host = np.arange(start_pos, start_pos + rows, dtype=np.int64)
        counts_host = positions_host + 1
        scratch = self._prefill_arena.take((
            rows * 8,            # positions
            rows * 8,            # counts
            rows * hidden * 2,   # normed
            rows * hidden * 2,   # normed16
            rows * q_width * 4,  # q_f32
            rows * q_width * 2,  # q_bf16
            rows * kv_width * 4,  # k_f32
            rows * kv_width * 2,  # k_bf16
            rows * kv_width * 4,  # v_f32
            rows * kv_width * 2,  # v_bf16
            rows * q_width * 2,  # q_normed_bf16
            rows * q_width * 4,  # q_normed_f32
            rows * kv_width * 2,  # k_normed_bf16
            rows * kv_width * 4,  # k_normed_f32
            rows * q_width * 4,  # q_out
            rows * kv_width * 4,  # k_out
            rows * kv_width * 2,  # k_out_bf16
            rows * kv_width * 2,  # v_out_bf16
            rows * q_width * 4,  # attn f32
            rows * q_width * 2,  # attn_bf16
            rows * q_width * 2,  # attn16 (fp16 GEMM input)
            rows * hidden * 4,   # o_f32
            rows * hidden * 2,   # o_bf16
            rows * hidden * 2,   # normed2
            rows * hidden * 2,   # normed216
            rows * ffn * 4,      # gate_f32
            rows * ffn * 4,      # up_f32
            rows * ffn * 2,      # gate bf16
            rows * ffn * 2,      # up bf16
            rows * ffn * 2,      # act bf16
            rows * ffn * 2,      # act16
            rows * hidden * 4,   # down_f32
            rows * hidden * 2,   # down_bf16
        ))
        (positions, counts, normed, normed16, q_f32, q_bf16, k_f32, k_bf16, v_f32,
         v_bf16, q_normed_bf16, q_normed_f32, k_normed_bf16, k_normed_f32, q_out,
         k_out, k_out_bf16, v_out_bf16, attn, attn_bf16, attn16, o_f32, o_bf16, normed2,
         normed216, gate_f32, up_f32, gate, up, act, act16, down_f32, down_bf16) = scratch
        copy_host_array_to_device(positions, positions_host)
        copy_host_array_to_device(counts, counts_host)
        spans = self._spans_with(branch, positions, counts, rows)
        for layer in self.layers:
            kernels.vv_rmsnorm_bf16(
                hidden_rows.ptr, layer.input_ln.ptr, normed.ptr, rows, hidden,
                spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            kernels.bf16_to_fp16(normed.ptr, normed16.ptr, rows * hidden, stream=0, runtime=self.runtime)
            self._lt_gemm(normed16.ptr, layer.q_w16.ptr, q_f32.ptr, rows, hidden, q_width)
            self._lt_gemm(normed16.ptr, layer.k_w16.ptr, k_f32.ptr, rows, hidden, kv_width)
            self._lt_gemm(normed16.ptr, layer.v_w16.ptr, v_f32.ptr, rows, hidden, kv_width)
            kernels.f32_to_bf16(q_f32.ptr, q_bf16.ptr, rows * q_width, stream=0, runtime=self.runtime)
            kernels.f32_to_bf16(k_f32.ptr, k_bf16.ptr, rows * kv_width, stream=0, runtime=self.runtime)
            kernels.vv_rmsnorm_bf16(
                q_bf16.ptr, layer.q_norm.ptr, q_normed_bf16.ptr, rows * heads, head_dim,
                spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            kernels.vv_rmsnorm_bf16(
                k_bf16.ptr, layer.k_norm.ptr, k_normed_bf16.ptr, rows * kv_heads, head_dim,
                spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            kernels.bf16_to_f32(q_normed_bf16.ptr, q_normed_f32.ptr, rows * q_width, stream=0, runtime=self.runtime)
            kernels.bf16_to_f32(k_normed_bf16.ptr, k_normed_f32.ptr, rows * kv_width, stream=0, runtime=self.runtime)
            kernels.vv_rope_positions_f32(
                q_normed_f32.ptr, k_normed_f32.ptr, self._cos.ptr, self._sin.ptr, positions.ptr,
                q_out.ptr, k_out.ptr, rows, heads, kv_heads, head_dim,
                stream=0, runtime=self.runtime,
            )
            kernels.f32_to_bf16(k_out.ptr, k_out_bf16.ptr, rows * kv_width, stream=0, runtime=self.runtime)
            kernels.f32_to_bf16(v_f32.ptr, v_out_bf16.ptr, rows * kv_width, stream=0, runtime=self.runtime)
            kernels.vv_kv_write_spans(
                k_out_bf16.ptr, v_out_bf16.ptr, layer.k_cache[branch].ptr, layer.v_cache[branch].ptr,
                spans, rows, kv_heads, head_dim, library=self.library, runtime=self.runtime,
            )
            kernels.vv_attention_spans(
                q_out.ptr, layer.k_cache[branch].ptr, layer.v_cache[branch].ptr, attn.ptr,
                spans, rows, heads, kv_heads, head_dim, self._scale,
                library=self.library, runtime=self.runtime,
            )
            kernels.f32_to_bf16(attn.ptr, attn_bf16.ptr, rows * q_width, stream=0, runtime=self.runtime)
            # The reference O projection consumes the BF16 attention output; the
            # hipBLASLt problem is FP16, so convert rather than reinterpreting
            # BF16 bit patterns as FP16.
            kernels.bf16_to_fp16(attn_bf16.ptr, attn16.ptr, rows * q_width, stream=0, runtime=self.runtime)
            self._lt_gemm(attn16.ptr, layer.o_w16.ptr, o_f32.ptr, rows, q_width, hidden)
            kernels.f32_to_bf16(o_f32.ptr, o_bf16.ptr, rows * hidden, stream=0, runtime=self.runtime)
            kernels.vv_scale_residual_bf16(
                hidden_rows.ptr, o_bf16.ptr, self._ones_hidden.ptr, hidden_rows.ptr,
                rows * hidden, hidden, library=self.library, runtime=self.runtime,
            )
            kernels.vv_rmsnorm_bf16(
                hidden_rows.ptr, layer.post_ln.ptr, normed2.ptr, rows, hidden,
                spec.rms_norm_eps, library=self.library, runtime=self.runtime,
            )
            kernels.bf16_to_fp16(normed2.ptr, normed216.ptr, rows * hidden, stream=0, runtime=self.runtime)
            self._lt_gemm(normed216.ptr, layer.gate_w16.ptr, gate_f32.ptr, rows, hidden, ffn)
            self._lt_gemm(normed216.ptr, layer.up_w16.ptr, up_f32.ptr, rows, hidden, ffn)
            kernels.f32_to_bf16(gate_f32.ptr, gate.ptr, rows * ffn, stream=0, runtime=self.runtime)
            kernels.f32_to_bf16(up_f32.ptr, up.ptr, rows * ffn, stream=0, runtime=self.runtime)
            kernels.silu_mul_separate_out_bf16(
                gate.ptr, up.ptr, act.ptr, rows, ffn, stream=0, runtime=self.runtime,
            )
            kernels.bf16_to_fp16(act.ptr, act16.ptr, rows * ffn, stream=0, runtime=self.runtime)
            self._lt_gemm(act16.ptr, layer.down_w16.ptr, down_f32.ptr, rows, ffn, hidden)
            kernels.f32_to_bf16(down_f32.ptr, down_bf16.ptr, rows * hidden, stream=0, runtime=self.runtime)
            kernels.vv_scale_residual_bf16(
                hidden_rows.ptr, down_bf16.ptr, self._ones_hidden.ptr, hidden_rows.ptr,
                rows * hidden, hidden, library=self.library, runtime=self.runtime,
            )

    # ------------------------------------------------------------------
    # spans
    # ------------------------------------------------------------------
    def _tensor(self, buffer: DeviceBuffer, shape: tuple[int, ...], dtype: str) -> Tensor:
        return Tensor.from_handle(buffer.ptr, shape, dtype, Device("hip"))

    def _spans(self, branch: int, rows: int) -> KVLiveSpans:
        return self._spans_with(
            branch,
            self._row_position[branch],
            self._ctx_len[branch],
            rows,
        )

    def _spans_with(self, branch: int, positions, counts, rows: int) -> KVLiveSpans:
        return KVLiveSpans(
            self._tensor(self._slot_map, (self.max_context,), "int32"),
            self._tensor(counts, (rows,), "int64"),
            self.max_context,
            self._tensor(self._slot_positions, (self.max_context,), "int64"),
            self._tensor(self._evicted, (self.max_context,), "bool"),
            "bf16",
            row_positions=self._tensor(positions, (rows,), "int64"),
            span_role="decode" if rows == 1 else "prefill",
        )

    # ------------------------------------------------------------------
    def _validate_position(self, position: int) -> None:
        if (
            isinstance(position, bool)
            or not isinstance(position, Integral)
            or not 0 <= position < self.max_context
        ):
            raise ValueError(f"position must be an integer in [0, {self.max_context})")

    def _validate_branch(self, branch: int) -> None:
        if branch not in range(self.branches):
            raise ValueError(f"branch must be in [0, {self.branches})")

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._prefill_arena.close()
        if self._lt is not None:
            self._lt.close()
            self._lt = None
        for attribute in (
            "_normed", "_q_f32", "_k_f32", "_v_f32", "_q_bf16", "_k_bf16", "_v_bf16",
            "_q_normed_bf16", "_k_normed_bf16", "_q_normed_f32", "_k_normed_f32", "_q_out",
            "_k_out", "_attn", "_attn_bf16", "_o_f32", "_o_bf16", "_gate_up", "_silu",
            "_down_bf16", "_logits_bf16", "_logits_f32",
        ):
            buffer = getattr(self, attribute, None)
            if isinstance(buffer, DeviceBuffer):
                free(buffer)
                setattr(self, attribute, buffer.__class__(0, 0))
        for index in range(self.branches):
            free(self._ctx_len[index])
            free(self._row_position[index])
        # KV caches and every uploaded weight live in ``_buffers``; freeing them
        # twice raises HIP error 1 (invalid argument).
        for buffer in self._buffers:
            free(buffer)
        self._buffers.clear()
