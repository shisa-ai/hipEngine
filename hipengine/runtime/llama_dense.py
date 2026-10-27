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

That kernel keeps its score row in workgroup shared memory, so ``max_context`` is
bounded by the LDS budget rather than by the checkpoint's position limit.
:data:`MAX_DENSE_DECODE_CONTEXT` is the largest context it can serve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from hipengine.core.hip import HipRuntime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_full_attn_decode_context_bf16,
)
from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_bf16_add,
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_separate_out_bf16
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

#: The dense decode kernel stages one FP32 score row plus a 256-slot reduction
#: scratch in LDS, so the servable context is the 64 KiB workgroup budget.
_DECODE_LDS_BYTES = 64 * 1024
_DECODE_THREADS = 256
MAX_DENSE_DECODE_CONTEXT = _DECODE_LDS_BYTES // _F32_NBYTES - _DECODE_THREADS


class LlamaKVCache:
    """Contiguous BF16 K/V storage, one ``[max_context, kv_width]`` buffer per layer.

    Every Llama layer is full attention and the cache is not paged, so a layer's
    cache is addressed by absolute position and a decode pass reads it linearly.
    """

    def __init__(self, spec: "LlamaSpec", max_context: int, *, runtime: HipRuntime) -> None:
        context = int(max_context)
        if context <= 0:
            raise ValueError("max_context must be positive")
        if context > MAX_DENSE_DECODE_CONTEXT:
            raise ValueError(
                f"max_context {context} exceeds the dense decode attention budget of "
                f"{MAX_DENSE_DECODE_CONTEXT} tokens (64 KiB LDS score row); a longer "
                "context needs the paged decode path"
            )
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
    """Per-step FP32 staging the BF16 prefill scratch does not carry."""

    query_f32: DeviceBuffer
    attention_f32: DeviceBuffer
    logits: DeviceBuffer

    @classmethod
    def allocate(cls, spec: "LlamaSpec", *, runtime: HipRuntime) -> "_DecodeScratch":
        return cls(
            query_f32=malloc(spec.q_width * _F32_NBYTES, runtime=runtime),
            attention_f32=malloc(spec.q_width * _F32_NBYTES, runtime=runtime),
            logits=malloc(spec.vocab_size * _F32_NBYTES, runtime=runtime),
        )

    def close(self, *, runtime: HipRuntime) -> None:
        for buffer in (self.logits, self.attention_f32, self.query_f32):
            free(buffer, runtime=runtime)


class LlamaDenseRunner(LlamaPrefillRunner):
    """Resident dense Llama decoder with a prompt pass and a token-by-token pass."""

    def __init__(
        self,
        loaded: "LlamaLoadedModel",
        *,
        max_context: int,
        runtime: HipRuntime | None = None,
    ) -> None:
        super().__init__(loaded, runtime=runtime)
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
            self._decode = _DecodeScratch.allocate(self.spec, runtime=self.runtime)
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
            bf16_to_f32(scratch.query.ptr, decode.query_f32.ptr, spec.q_width, runtime=self.runtime)
            qwen35_full_attn_decode_context_bf16(
                decode.query_f32.ptr,
                self.cache.key(layer_id).ptr,
                self.cache.value(layer_id).ptr,
                decode.attention_f32.ptr,
                self.cache.context_len_ptr,
                self.cache.max_context,
                spec.num_attention_heads,
                spec.num_key_value_heads,
                spec.head_dim,
                spec.attention_scale,
                runtime=self.runtime,
            )
            f32_to_bf16(
                decode.attention_f32.ptr,
                scratch.attention.ptr,
                spec.q_width,
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
    "MAX_DENSE_DECODE_CONTEXT",
]
