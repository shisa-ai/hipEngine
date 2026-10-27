"""Resident dense Llama decoder: KV cache plus a one-token decode step.

:class:`~hipengine.runtime.llama_prefill.LlamaPrefillRunner` owns the weights and
the prompt pass. This module adds the two things generation needs on top of it: a
contiguous BF16 K/V cache that the prompt pass populates, and a decode step that
appends one token's K/V and reads the whole context back.

Attention is the tree's dense decode kernel: one block per query head, GQA by
``num_heads // num_kv_heads``, FP32 query and output against a BF16 cache. The
cache row is ``[kv_width]`` per position, which is the ``[num_kv_heads,
head_dim]`` layout that kernel indexes, so one contiguous copy per layer appends
a token.

Both passes tile keys with an online softmax, so neither stages a per-key row in
shared memory and the context is not capped by the 64 KiB LDS budget. What bounds
a request is the checkpoint's declared positions (:func:`max_dense_context`) and,
past that, the memory a KV cache and a causal mask need. The exact block
attention remains the strict fallback and is still LDS-bounded, so
:func:`max_dense_prefill_tokens` reports the tighter prompt bound only for a head
geometry the tiled kernel does not carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.hip_gfx1100.attention.dense_flash_prefill import (
    dense_flash_prefill_admits,
)
from hipengine.kernels.hip_gfx1100.attention.dense_prefill_attention import (
    dense_causal_prefill_bf16,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_bf16_add,
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_separate_out_bf16
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_shared_bytes
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    dense_gemv_bf16_f32_out,
    dense_gemv_out_bf16,
)
from hipengine.kernels.hip_gfx1100.rotary.dense_rope import (
    dense_rope_positions_bf16_inplace,
)
from hipengine.runtime.llama_prefill import LlamaPrefillRunner

if TYPE_CHECKING:
    from hipengine.kvcache.dms_capture import DMSCaptureSink
    from hipengine.loading.llama_safetensors import LlamaLoadedModel
    from hipengine.models.llama import LlamaSpec


_BF16_NBYTES = 2
_F32_NBYTES = 4

#: The exact block attention stages one FP32 logit row per key in LDS, so it is
#: bounded by the 64 KiB workgroup budget. It is the strict fallback now, not the
#: default: the tiled route's shared memory does not mention the context.
_EXACT_LDS_BYTES = 64 * 1024
_EXACT_LDS_FLOATS = _EXACT_LDS_BYTES // _F32_NBYTES


def max_dense_context(spec: "LlamaSpec") -> int:
    """The longest context this path can serve.

    Both attention passes tile keys with an online softmax, so neither stages a
    per-key row in shared memory and neither imposes a context ceiling. What
    bounds a request is the checkpoint's declared positions and, past that, the
    memory a KV cache needs -- a resource failure, not a capability one.

    This is the capability answer. The generator's *default* is separately
    clamped by what actually fits, because a checkpoint may declare more
    positions than the cache can hold beside its own weights.
    """

    return int(spec.max_position_embeddings)


def kv_bytes_per_token(spec: "LlamaSpec") -> int:
    """BF16 K and V for one position, across every layer."""

    return int(spec.num_hidden_layers) * 2 * int(spec.kv_width) * _BF16_NBYTES


#: Host- and device-side working set a prefill needs beyond the KV cache and
#: the resident weights: the attention scratch, logits, and the tokenizer's
#: buffers. Reserved out of the default context budget so a default that fits
#: the cache still fits the pass that fills it.
_PREFILL_SCRATCH_RESERVE = 4 << 30


def default_dense_context_length(spec: "LlamaSpec", weight_index: object) -> int:
    """The context this machine serves by default.

    The checkpoint's positions, capped by the cache that fits beside the weights
    this checkpoint will occupy. The cap is memory, not a round number: it is
    derived from ``hipMemGetInfo`` and the index's own tensor shapes, so the same
    checkpoint gets a larger default on a larger card and a smaller one where it
    would not fit.

    Half of the post-weights budget is available to the cache, so a default
    context never leaves the machine with less free memory than the cache itself
    holds. Asking for a larger ``context_length`` is honoured; if it does not fit
    the allocation fails, which is a resource error, not a refusal.
    """

    checkpoint = max_dense_context(spec)
    per_token = kv_bytes_per_token(spec)
    if per_token <= 0:
        return checkpoint
    try:
        free_bytes, total_bytes = get_hip_runtime().mem_get_info()
    except Exception:
        # No device to ask. The checkpoint's positions are still the right
        # answer; sizing happens when the cache is built.
        return checkpoint
    budget = min(int(free_bytes), int(total_bytes))
    budget -= resident_weight_bytes(weight_index) + _PREFILL_SCRATCH_RESERVE
    if budget <= 0:
        return 1
    return int(max(1, min(checkpoint, (budget // 2) // per_token)))


def resident_weight_bytes(weight_index: object) -> int:
    """Bytes the loader will hold resident, from the index alone.

    The loader materializes BF16 for every tensor it reads, so a float8 source
    costs two bytes per element once resident -- which is why an FP8 checkpoint
    occupies twice its on-disk size. Tensors whose dtype has no known width are
    counted as BF16, which is the loader's own target type.
    """

    tensors = getattr(weight_index, "tensors", {}) or {}
    total = 0
    for info in tensors.values():
        count = 1
        for dim in info.shape:
            count *= int(dim)
        total += count * _BF16_NBYTES
    return total


def max_dense_prefill_tokens(spec: "LlamaSpec", *, strict: bool = False) -> int:
    """The longest prompt the selected prefill kernel serves.

    The tiled kernel serves every key count, so the prompt bound is the
    checkpoint's position count. Only a geometry the tiled kernel does not carry
    -- anything but ``head_dim`` 128 -- falls back to the exact block kernel,
    whose FP32 logit row is ``max(head_dim + keys + threads, keys + 528)`` floats
    and is therefore bounded by the 64 KiB LDS budget. That bound is a property
    of the head geometry, so it is derived from the launcher's own rule rather
    than restated here; the predicate is monotonic in ``keys``, which is what
    makes the bisection sound.
    """

    head_dim = int(spec.head_dim)
    if not strict and dense_flash_prefill_admits(
        tokens=1,
        keys=1,
        num_heads=int(spec.num_attention_heads),
        num_kv_heads=int(spec.num_key_value_heads),
        head_dim=head_dim,
    ) is None:
        return max_dense_context(spec)
    low, high = 1, max_dense_context(spec)
    if not _exact_prefill_fits(head_dim, low):
        raise ValueError(
            f"the causal prefill attention cannot serve head_dim={head_dim} at any "
            "context length"
        )
    while low < high:
        middle = (low + high + 1) // 2
        if _exact_prefill_fits(head_dim, middle):
            low = middle
        else:
            high = middle - 1
    return low


def _exact_prefill_fits(head_dim: int, keys: int) -> bool:
    try:
        gemma4_attention_shared_bytes(head_dim=head_dim, keys=keys)
    except NotImplementedError:
        return False
    return True


class LlamaKVCache:
    """Contiguous BF16 K/V storage, one ``[max_context, kv_width]`` buffer per layer.

    Every Llama layer is full attention and the cache is not paged, so a layer's
    cache is addressed by absolute position and a decode pass reads it linearly.
    """

    def __init__(self, spec: "LlamaSpec", max_context: int, *, runtime: HipRuntime) -> None:
        context = int(max_context)
        if context <= 0:
            raise ValueError("max_context must be positive")
        self.spec = spec
        self.max_context = context
        self.runtime = runtime
        self._row_bytes = spec.kv_width * _BF16_NBYTES
        self._layer_bytes = context * self._row_bytes
        self._key = [
            malloc(self._layer_bytes, runtime=runtime)
            for _ in range(spec.num_hidden_layers)
        ]
        self._value = [
            malloc(self._layer_bytes, runtime=runtime)
            for _ in range(spec.num_hidden_layers)
        ]
        self._context_len = malloc(8, runtime=runtime)
        self._context_len_host = np.zeros(1, dtype=np.int64)
        self.set_context_len(0)

    # -- geometry ----------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return 2 * self.spec.num_hidden_layers * self._layer_bytes

    @property
    def context_len(self) -> int:
        """Number of positions the cache currently holds."""

        return int(self._context_len_host[0])

    @property
    def context_len_ptr(self) -> int:
        return int(self._context_len.ptr)

    def key(self, layer_id: int) -> DeviceBuffer:
        return self._key[self._layer_index(layer_id)]

    def value(self, layer_id: int) -> DeviceBuffer:
        return self._value[self._layer_index(layer_id)]

    def _layer_index(self, layer_id: int) -> int:
        layer = int(layer_id)
        if layer < 0 or layer >= self.spec.num_hidden_layers:
            raise ValueError(
                f"layer {layer} is outside the decoder's {self.spec.num_hidden_layers} layers"
            )
        return layer

    # -- state -------------------------------------------------------------
    def set_context_len(self, tokens: int) -> None:
        """Publish the number of live positions to the decode kernel."""

        count = int(tokens)
        if count < 0 or count > self.max_context:
            raise ValueError(
                f"context length {count} is outside the cache capacity of {self.max_context}"
            )
        self._context_len_host[0] = count
        copy_host_to_device(
            self._context_len,
            host_array_ptr(self._context_len_host),
            8,
            runtime=self.runtime,
        )

    def reset(self) -> None:
        """Drop every position. Cached bytes stay allocated and are overwritten."""

        self.set_context_len(0)

    # -- writes ------------------------------------------------------------
    def write(
        self,
        layer_id: int,
        key_ptr: int,
        value_ptr: int,
        rows: int,
        positions: np.ndarray,
    ) -> None:
        """Copy ``rows`` roped K/V rows from contiguous device buffers into the cache."""

        layer = self._layer_index(layer_id)
        count = int(rows)
        if count <= 0:
            raise ValueError("a cache write needs at least one row")
        row_positions = np.ascontiguousarray(positions, dtype=np.int64).reshape(-1)
        if row_positions.size != count:
            raise ValueError(
                f"positions must describe every written row: {row_positions.size} != {count}"
            )
        if row_positions.min() < 0 or row_positions.max() >= self.max_context:
            raise ValueError(
                f"positions must be in [0, {self.max_context}); got "
                f"[{int(row_positions.min())}, {int(row_positions.max())}]"
            )
        key_base = int(self._key[layer].ptr)
        value_base = int(self._value[layer].ptr)
        if count == 1:
            self._copy_row(key_base, int(key_ptr), int(row_positions[0]))
            self._copy_row(value_base, int(value_ptr), int(row_positions[0]))
            return
        if bool(np.all(np.diff(row_positions) == 1)):
            span = count * self._row_bytes
            offset = int(row_positions[0]) * self._row_bytes
            self.runtime.memcpy(
                key_base + offset, int(key_ptr), span, MemcpyKind.DEVICE_TO_DEVICE
            )
            self.runtime.memcpy(
                value_base + offset, int(value_ptr), span, MemcpyKind.DEVICE_TO_DEVICE
            )
            return
        for row, position in enumerate(row_positions):
            self._copy_row(key_base, int(key_ptr) + row * self._row_bytes, int(position))
            self._copy_row(value_base, int(value_ptr) + row * self._row_bytes, int(position))

    def _copy_row(self, cache_base: int, source_ptr: int, position: int) -> None:
        self.runtime.memcpy(
            cache_base + position * self._row_bytes,
            source_ptr,
            self._row_bytes,
            MemcpyKind.DEVICE_TO_DEVICE,
        )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        for buffer in (*self._key, *self._value, self._context_len):
            free(buffer, runtime=self.runtime)
        self._key = []
        self._value = []

    def __enter__(self) -> "LlamaKVCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass
class _DecodeScratch:
    """Per-step staging the BF16 prefill scratch does not carry.

    A decode step is the tiled kernel at ``tokens == 1`` with a null mask, which
    already means "every key up to this position": ``position_base`` is the last
    cached position, so the causal walk bound covers the whole context. That
    leaves no per-step attention buffer -- only the logit vector the
    language-model head writes.

    A geometry the tiled kernel does not carry falls back to the exact block
    decode, which reads a mask unconditionally. There the row is allocated once
    at cache capacity and filled with ones, because a decode step sees every
    cached key.
    """

    logits: DeviceBuffer
    mask: DeviceBuffer | None = None

    @classmethod
    def allocate(
        cls,
        spec: "LlamaSpec",
        *,
        max_context: int,
        runtime: HipRuntime,
        needs_mask: bool,
    ) -> "_DecodeScratch":
        mask = None
        if needs_mask:
            mask = malloc(int(max_context), runtime=runtime)
            runtime.memset(mask.ptr, 1, int(max_context))
        return cls(
            logits=malloc(spec.vocab_size * _F32_NBYTES, runtime=runtime), mask=mask
        )

    def close(self, *, runtime: HipRuntime) -> None:
        free(self.logits, runtime=runtime)
        if self.mask is not None:
            free(self.mask, runtime=runtime)


class LlamaDenseRunner(LlamaPrefillRunner):
    """Resident dense Llama decoder with a prompt pass and a token-by-token pass."""

    def __init__(
        self,
        loaded: "LlamaLoadedModel",
        *,
        max_context: int,
        runtime: HipRuntime | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__(loaded, runtime=runtime, strict=strict)
        self.cache = LlamaKVCache(self.spec, max_context, runtime=self.runtime)
        self._decode: _DecodeScratch | None = None

    @property
    def max_context(self) -> int:
        return self.cache.max_context

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._decode is not None:
            self._decode.close(runtime=self.runtime)
            self._decode = None
        self.cache.close()
        super().close()

    def reset(self) -> None:
        """Discard the cached context so the next prompt starts at position 0."""

        self.cache.reset()

    def _decode_scratch(self) -> _DecodeScratch:
        if self._decode is None:
            self._decode = _DecodeScratch.allocate(
                self.spec,
                max_context=self.max_context,
                runtime=self.runtime,
                needs_mask=not self._attention_is_tiled,
            )
        return self._decode

    # -- passes ------------------------------------------------------------
    def prefill(
        self,
        token_ids: Sequence[int],
        *,
        capture: "DMSCaptureSink | None" = None,
        logits: str | Sequence[int] = "last",
    ) -> np.ndarray | None:
        """Run one prompt from position 0 and populate the cache."""

        tokens = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        rows = int(tokens.size)
        if rows <= 0:
            raise ValueError("a prefill needs at least one token")
        if rows > self.max_context:
            raise ValueError(
                f"prompt of {rows} tokens exceeds the cache capacity of {self.max_context}"
            )
        result = super().prefill(tokens, capture=capture, kv_cache=self.cache, logits=logits)
        self.cache.set_context_len(rows)
        return result

    def decode_step(self, token_id: int, *, position: int | None = None) -> np.ndarray:
        """Append one token at ``position`` and return its next-token logits in FP32."""

        spec = self.spec
        next_position = self.cache.context_len if position is None else int(position)
        if next_position != self.cache.context_len:
            raise ValueError(
                f"decode position {next_position} does not follow the cached context of "
                f"{self.cache.context_len}"
            )
        if next_position >= self.max_context:
            raise ValueError(
                f"decode position {next_position} is outside the cache capacity of "
                f"{self.max_context}"
            )

        if not self._attention_is_tiled:
            gemma4_attention_shared_bytes(head_dim=spec.head_dim, keys=next_position + 1)

        scratch = self._scratch_for(1)
        decode = self._decode_scratch()
        tokens = np.array([int(token_id)], dtype=np.int64)
        self._gather_embeddings(tokens, scratch)
        self._upload_rope_tables(np.array([next_position], dtype=np.int64), 1, scratch)
        # The context length is the same for every layer, so publish it once
        # instead of once per layer's attention launch.
        self.cache.set_context_len(next_position + 1)

        for layer_id in range(spec.num_hidden_layers):
            weights = self.loaded.layers[layer_id]

            gguf_rmsnorm_bf16_f32_weight(
                scratch.hidden.ptr,
                weights.attn_norm.ptr,
                scratch.norm.ptr,
                1,
                spec.hidden_size,
                spec.rms_norm_eps,
                runtime=self.runtime,
            )
            dense_gemv_out_bf16(
                scratch.norm.ptr,
                weights.q_proj.ptr,
                scratch.query.ptr,
                1,
                spec.hidden_size,
                spec.q_width,
                runtime=self.runtime,
            )
            dense_gemv_out_bf16(
                scratch.norm.ptr,
                weights.k_proj.ptr,
                scratch.key.ptr,
                1,
                spec.hidden_size,
                spec.kv_width,
                runtime=self.runtime,
            )
            dense_gemv_out_bf16(
                scratch.norm.ptr,
                weights.v_proj.ptr,
                scratch.value.ptr,
                1,
                spec.hidden_size,
                spec.kv_width,
                runtime=self.runtime,
            )
            dense_rope_positions_bf16_inplace(
                scratch.query.ptr,
                scratch.key.ptr,
                scratch.cos.ptr,
                scratch.sin.ptr,
                1,
                spec.num_attention_heads,
                spec.num_key_value_heads,
                spec.head_dim,
                spec.head_dim,
                runtime=self.runtime,
            )
            self.cache.write(
                layer_id,
                scratch.key.ptr,
                scratch.value.ptr,
                1,
                np.array([next_position], dtype=np.int64),
            )
            # The tiled prefill kernel is also the decode kernel at tokens == 1:
            # it tiles keys with an online softmax, so its shared memory does not
            # mention the context and there is no LDS ceiling to refuse. One
            # launch replaces the FP32-widened block decode and its BF16
            # narrowing, so Q and the attention output stay BF16 end to end.
            dense_causal_prefill_bf16(
                scratch.query.ptr,
                self.cache.key(layer_id).ptr,
                self.cache.value(layer_id).ptr,
                # 0 means every cached key is visible: at tokens == 1 the causal
                # bound is the last position, i.e. the whole context.
                0 if self._attention_is_tiled else decode.mask.ptr,
                scratch.attention.ptr,
                tokens=1,
                keys=next_position + 1,
                num_heads=spec.num_attention_heads,
                num_kv_heads=spec.num_key_value_heads,
                head_dim=spec.head_dim,
                scale=spec.attention_scale,
                strict=self.strict,
                runtime=self.runtime,
            )
            dense_gemv_out_bf16(
                scratch.attention.ptr,
                weights.o_proj.ptr,
                scratch.projection.ptr,
                1,
                spec.q_width,
                spec.hidden_size,
                runtime=self.runtime,
            )
            gguf_bf16_add(
                scratch.hidden.ptr,
                scratch.projection.ptr,
                scratch.hidden.ptr,
                spec.hidden_size,
                runtime=self.runtime,
            )

            gguf_rmsnorm_bf16_f32_weight(
                scratch.hidden.ptr,
                weights.ffn_norm.ptr,
                scratch.norm.ptr,
                1,
                spec.hidden_size,
                spec.rms_norm_eps,
                runtime=self.runtime,
            )
            dense_gemv_out_bf16(
                scratch.norm.ptr,
                weights.gate_proj.ptr,
                scratch.gate.ptr,
                1,
                spec.hidden_size,
                spec.intermediate_size,
                runtime=self.runtime,
            )
            dense_gemv_out_bf16(
                scratch.norm.ptr,
                weights.up_proj.ptr,
                scratch.up.ptr,
                1,
                spec.hidden_size,
                spec.intermediate_size,
                runtime=self.runtime,
            )
            silu_mul_separate_out_bf16(
                scratch.gate.ptr,
                scratch.up.ptr,
                scratch.activation.ptr,
                1,
                spec.intermediate_size,
                runtime=self.runtime,
            )
            dense_gemv_out_bf16(
                scratch.activation.ptr,
                weights.down_proj.ptr,
                scratch.projection.ptr,
                1,
                spec.intermediate_size,
                spec.hidden_size,
                runtime=self.runtime,
            )
            gguf_bf16_add(
                scratch.hidden.ptr,
                scratch.projection.ptr,
                scratch.hidden.ptr,
                spec.hidden_size,
                runtime=self.runtime,
            )

        gguf_rmsnorm_bf16_f32_weight(
            scratch.hidden.ptr,
            self.loaded.final_norm.ptr,
            scratch.norm.ptr,
            1,
            spec.hidden_size,
            spec.rms_norm_eps,
            runtime=self.runtime,
        )
        dense_gemv_bf16_f32_out(
            scratch.norm.ptr,
            self.loaded.lm_head.ptr,
            decode.logits.ptr,
            1,
            spec.hidden_size,
            spec.vocab_size,
            runtime=self.runtime,
        )
        logits = np.empty(spec.vocab_size, dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits),
            decode.logits,
            spec.vocab_size * _F32_NBYTES,
            runtime=self.runtime,
        )
        return logits


__all__ = [
    "LlamaDenseRunner",
    "LlamaKVCache",
    "default_dense_context_length",
    "kv_bytes_per_token",
    "max_dense_context",
    "max_dense_prefill_tokens",
    "resident_weight_bytes",
]
