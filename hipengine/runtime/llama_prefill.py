"""Dense Llama-family prefill execution with DMS capture taps.

One forward pass over a contiguous prompt, torch-free, on the registered
hipEngine primitives:

``embed -> [rmsnorm -> qkv -> rope -> attention -> o_proj -> residual
-> rmsnorm -> gate/up -> silu_mul -> down -> residual] * L
-> final rmsnorm -> lm_head``

The pass exposes the two things external DMS training needs from a teacher: the
predictor input stage (``post_attn_rmsnorm_pre_q_projection``) together with the
post-RoPE Q/K the attention kernel actually consumes, and the next-token logits.

Numerics are BF16 activations with F32 accumulation, and FP8 source weights
dequantized once at load. That is the execution profile recorded with every
capture: Q/K are BF16 values, not pre-rounding F32 values, so the labels describe
the attention the runtime actually computes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

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
from hipengine.kernels.hip_gfx1100.attention.dense_prefill_attention import (
    causal_keep_mask,
    dense_causal_prefill_bf16,
    Gemma4AttentionScratch,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_bf16_add,
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_separate_out_bf16
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    dense_gemv_bf16_f32_out,
    dense_prefill_wmma_out_bf16,
)
from hipengine.kernels.hip_gfx1100.rotary.dense_rope import (
    dense_rope_positions_bf16_inplace,
)

if TYPE_CHECKING:
    from hipengine.kvcache.dms_capture import DMSCaptureSink
    from hipengine.loading.llama_safetensors import LlamaLoadedModel


_BF16_NBYTES = 2


class _KVCacheSink(Protocol):
    """Receives the roped K/V rows a pass produced, one call per layer.

    The runner owns when the rows exist; the cache owns where they land. Keeping
    that split means this module stays free of cache layout, block tables, and
    context-length bookkeeping, and a decode cache and a capture cache can share
    the same pass.
    """

    def write(
        self,
        layer_id: int,
        key_ptr: int,
        value_ptr: int,
        rows: int,
        positions: np.ndarray,
    ) -> None: ...


@dataclass
class _PrefillScratch:
    """Per-geometry device scratch. One instance serves every layer of a pass."""

    rows: int
    hidden: DeviceBuffer
    norm: DeviceBuffer
    query: DeviceBuffer
    key: DeviceBuffer
    value: DeviceBuffer
    attention: DeviceBuffer
    projection: DeviceBuffer
    gate: DeviceBuffer
    up: DeviceBuffer
    activation: DeviceBuffer
    cos: DeviceBuffer
    sin: DeviceBuffer
    mask: DeviceBuffer

    @classmethod
    def allocate(cls, spec, rows: int, *, runtime: HipRuntime) -> "_PrefillScratch":
        def bf16(count: int) -> DeviceBuffer:
            return malloc(count * _BF16_NBYTES, runtime=runtime)

        hidden_bytes = rows * spec.hidden_size * _BF16_NBYTES
        return cls(
            rows=rows,
            hidden=malloc(hidden_bytes, runtime=runtime),
            norm=malloc(hidden_bytes, runtime=runtime),
            query=bf16(rows * spec.q_width),
            key=bf16(rows * spec.kv_width),
            value=bf16(rows * spec.kv_width),
            attention=bf16(rows * spec.q_width),
            projection=malloc(hidden_bytes, runtime=runtime),
            gate=bf16(rows * spec.intermediate_size),
            up=bf16(rows * spec.intermediate_size),
            activation=bf16(rows * spec.intermediate_size),
            cos=malloc(rows * spec.head_dim * 4, runtime=runtime),
            sin=malloc(rows * spec.head_dim * 4, runtime=runtime),
            mask=malloc(rows * rows, runtime=runtime),
        )

    def close(self, *, runtime: HipRuntime) -> None:
        for buffer in (
            self.mask,
            self.sin,
            self.cos,
            self.activation,
            self.up,
            self.gate,
            self.projection,
            self.attention,
            self.value,
            self.key,
            self.query,
            self.norm,
            self.hidden,
        ):
            free(buffer, runtime=runtime)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(
        np.float32
    )


def rope_table(positions: np.ndarray, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return rotate-half cos/sin tables of shape ``[rows, head_dim]``."""

    pos = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1)
    if pos.size <= 0:
        raise ValueError("positions must be non-empty")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be positive and even")
    inv_freq = 1.0 / np.power(float(theta), np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angles = pos[:, None] * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return (
        np.ascontiguousarray(np.concatenate([cos, cos], axis=1)),
        np.ascontiguousarray(np.concatenate([sin, sin], axis=1)),
    )


class LlamaPrefillRunner:
    """Resident dense Llama decoder with a capture-capable prefill pass."""

    def __init__(
        self,
        loaded: "LlamaLoadedModel",
        *,
        runtime: HipRuntime | None = None,
    ) -> None:
        self.loaded = loaded
        self.spec = loaded.spec
        self.runtime = runtime or get_hip_runtime()
        self._scratch: _PrefillScratch | None = None
        self._attention_scratch = Gemma4AttentionScratch()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._scratch is not None:
            self._scratch.close(runtime=self.runtime)
            self._scratch = None
        self._attention_scratch.close()

    def __enter__(self) -> "LlamaPrefillRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------
    def _scratch_for(self, rows: int) -> _PrefillScratch:
        scratch = self._scratch
        if scratch is None or scratch.rows < rows:
            if scratch is not None:
                scratch.close(runtime=self.runtime)
            scratch = _PrefillScratch.allocate(self.spec, rows, runtime=self.runtime)
            self._scratch = scratch
        return scratch

    def _project(
        self,
        x_ptr: int,
        weight_ptr: int,
        out_ptr: int,
        rows: int,
        in_features: int,
        out_features: int,
    ) -> None:
        dense_prefill_wmma_out_bf16(
            x_ptr,
            weight_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            runtime=self.runtime,
        )

    def _gather_embeddings(self, token_ids: np.ndarray, scratch: _PrefillScratch) -> None:
        table = self.loaded.embed_tokens
        table_ptr = int(table.ptr)
        row_bytes = self.spec.hidden_size * _BF16_NBYTES
        hidden_ptr = scratch.hidden.ptr
        vocab = self.spec.vocab_size
        for row, token in enumerate(token_ids):
            index = int(token)
            if index < 0 or index >= vocab:
                raise ValueError(f"token id {index} is outside the vocabulary of {vocab}")
            self.runtime.memcpy(
                hidden_ptr + row * row_bytes,
                table_ptr + index * row_bytes,
                row_bytes,
                MemcpyKind.DEVICE_TO_DEVICE,
            )

    def _upload_rope_tables(
        self,
        positions: np.ndarray,
        rows: int,
        scratch: _PrefillScratch,
    ) -> None:
        cos, sin = rope_table(positions, self.spec.head_dim, self.spec.rope_theta)
        copy_host_to_device(
            scratch.cos,
            host_array_ptr(cos),
            rows * self.spec.head_dim * 4,
            runtime=self.runtime,
        )
        copy_host_to_device(
            scratch.sin,
            host_array_ptr(sin),
            rows * self.spec.head_dim * 4,
            runtime=self.runtime,
        )

    def _upload_mask(self, rows: int, scratch: _PrefillScratch) -> None:
        mask = causal_keep_mask(rows)
        copy_host_to_device(scratch.mask, host_array_ptr(mask), rows * rows, runtime=self.runtime)

    def _copy_bf16_rows(self, buffer: DeviceBuffer, rows: int, width: int) -> np.ndarray:
        host = np.empty((rows, width), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), buffer, rows * width * _BF16_NBYTES, runtime=self.runtime)
        return host

    def _copy_f32_rows(self, buffer: DeviceBuffer, count: int) -> np.ndarray:
        host = np.empty(count, dtype=np.float32)
        copy_device_to_host(host_array_ptr(host), buffer, count * 4, runtime=self.runtime)
        return host

    def _capture_layer(
        self,
        *,
        capture: "DMSCaptureSink",
        physical_layer_id: int,
        positions: np.ndarray,
        rows: int,
        scratch: _PrefillScratch,
    ) -> None:
        hidden_bits = self._copy_bf16_rows(scratch.norm, rows, self.spec.hidden_size)
        query_bits = self._copy_bf16_rows(scratch.query, rows, self.spec.q_width)
        key_bits = self._copy_bf16_rows(scratch.key, rows, self.spec.kv_width)
        capture.capture_chunk(
            physical_layer_id=int(physical_layer_id),
            compact_layer_index=self.spec.compact_layer_index(physical_layer_id),
            positions=np.ascontiguousarray(positions, dtype=np.int32),
            hidden_bf16=hidden_bits,
            query_f32=_bf16_bits_to_float32(query_bits).reshape(
                rows, self.spec.num_attention_heads, self.spec.head_dim
            ),
            key_f32=_bf16_bits_to_float32(key_bits).reshape(
                rows, self.spec.num_key_value_heads, self.spec.head_dim
            ),
        )

    # -- forward -----------------------------------------------------------
    def prefill(
        self,
        token_ids: Sequence[int],
        *,
        capture: "DMSCaptureSink | None" = None,
        kv_cache: "_KVCacheSink | None" = None,
        positions: Sequence[int] | None = None,
        logits: str | Sequence[int] = "last",
    ) -> np.ndarray | None:
        """Run one prompt and return logits for the requested rows.

        ``logits`` is ``"last"`` for the final row (the next-token distribution a
        DMS capture records), ``"all"`` for every row, an explicit sequence of row
        indices, or ``"none"``. Row subsets are produced in FP32; ``"all"`` uses
        the BF16 bulk projection.

        ``kv_cache`` receives each layer's roped K and V for these rows, which is
        what lets a decode pass attend to the prompt. It does not affect the
        returned logits or the captured taps.
        """

        tokens = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        rows = int(tokens.size)
        if rows <= 0:
            raise ValueError("a prefill needs at least one token")
        position_values = (
            np.arange(rows, dtype=np.int64)
            if positions is None
            else np.ascontiguousarray(positions, dtype=np.int64).reshape(-1)
        )
        if position_values.size != rows:
            raise ValueError(
                f"positions must describe every prompt row: {position_values.size} != {rows}"
            )
        if capture is not None:
            if int(capture.hidden_size) != self.spec.hidden_size:
                raise ValueError(
                    f"capture sink hidden_size {capture.hidden_size} does not match "
                    f"the decoder's {self.spec.hidden_size}"
                )
            if int(capture.num_q_heads) != self.spec.num_attention_heads:
                raise ValueError("capture sink query-head count does not match the decoder")
            if int(capture.num_kv_heads) != self.spec.num_key_value_heads:
                raise ValueError("capture sink KV-head count does not match the decoder")
            if int(capture.head_dim) != self.spec.head_dim:
                raise ValueError("capture sink head_dim does not match the decoder")

        spec = self.spec
        scratch = self._scratch_for(rows)
        self._gather_embeddings(tokens, scratch)
        self._upload_rope_tables(position_values, rows, scratch)
        self._upload_mask(rows, scratch)

        for layer_id in range(spec.num_hidden_layers):
            weights = self.loaded.layers[layer_id]

            gguf_rmsnorm_bf16_f32_weight(
                scratch.hidden.ptr,
                weights.attn_norm.ptr,
                scratch.norm.ptr,
                rows,
                spec.hidden_size,
                spec.rms_norm_eps,
                runtime=self.runtime,
            )
            self._project(
                scratch.norm.ptr, weights.q_proj.ptr, scratch.query.ptr,
                rows, spec.hidden_size, spec.q_width,
            )
            self._project(
                scratch.norm.ptr, weights.k_proj.ptr, scratch.key.ptr,
                rows, spec.hidden_size, spec.kv_width,
            )
            self._project(
                scratch.norm.ptr, weights.v_proj.ptr, scratch.value.ptr,
                rows, spec.hidden_size, spec.kv_width,
            )
            dense_rope_positions_bf16_inplace(
                scratch.query.ptr,
                scratch.key.ptr,
                scratch.cos.ptr,
                scratch.sin.ptr,
                rows,
                spec.num_attention_heads,
                spec.num_key_value_heads,
                spec.head_dim,
                spec.head_dim,
                runtime=self.runtime,
            )
            if kv_cache is not None:
                kv_cache.write(
                    layer_id,
                    scratch.key.ptr,
                    scratch.value.ptr,
                    rows,
                    position_values,
                )
            if capture is not None:
                self._capture_layer(
                    capture=capture,
                    physical_layer_id=layer_id,
                    positions=position_values,
                    rows=rows,
                    scratch=scratch,
                )
            dense_causal_prefill_bf16(
                scratch.query.ptr,
                scratch.key.ptr,
                scratch.value.ptr,
                scratch.mask.ptr,
                scratch.attention.ptr,
                tokens=rows,
                num_heads=spec.num_attention_heads,
                num_kv_heads=spec.num_key_value_heads,
                head_dim=spec.head_dim,
                scale=spec.attention_scale,
                runtime=self.runtime,
                scratch=self._attention_scratch,
            )
            self._project(
                scratch.attention.ptr, weights.o_proj.ptr, scratch.projection.ptr,
                rows, spec.q_width, spec.hidden_size,
            )
            gguf_bf16_add(
                scratch.hidden.ptr,
                scratch.projection.ptr,
                scratch.hidden.ptr,
                rows * spec.hidden_size,
                runtime=self.runtime,
            )

            gguf_rmsnorm_bf16_f32_weight(
                scratch.hidden.ptr,
                weights.ffn_norm.ptr,
                scratch.norm.ptr,
                rows,
                spec.hidden_size,
                spec.rms_norm_eps,
                runtime=self.runtime,
            )
            self._project(
                scratch.norm.ptr, weights.gate_proj.ptr, scratch.gate.ptr,
                rows, spec.hidden_size, spec.intermediate_size,
            )
            self._project(
                scratch.norm.ptr, weights.up_proj.ptr, scratch.up.ptr,
                rows, spec.hidden_size, spec.intermediate_size,
            )
            silu_mul_separate_out_bf16(
                scratch.gate.ptr,
                scratch.up.ptr,
                scratch.activation.ptr,
                rows,
                spec.intermediate_size,
                runtime=self.runtime,
            )
            self._project(
                scratch.activation.ptr, weights.down_proj.ptr, scratch.projection.ptr,
                rows, spec.intermediate_size, spec.hidden_size,
            )
            gguf_bf16_add(
                scratch.hidden.ptr,
                scratch.projection.ptr,
                scratch.hidden.ptr,
                rows * spec.hidden_size,
                runtime=self.runtime,
            )

        if logits == "none" or logits is None:
            return None

        gguf_rmsnorm_bf16_f32_weight(
            scratch.hidden.ptr,
            self.loaded.final_norm.ptr,
            scratch.norm.ptr,
            rows,
            spec.hidden_size,
            spec.rms_norm_eps,
            runtime=self.runtime,
        )

        if isinstance(logits, str):
            if logits == "all":
                out = malloc(rows * spec.vocab_size * _BF16_NBYTES, runtime=self.runtime)
                try:
                    self._project(
                        scratch.norm.ptr, self.loaded.lm_head.ptr, out.ptr,
                        rows, spec.hidden_size, spec.vocab_size,
                    )
                    bits = np.empty((rows, spec.vocab_size), dtype=np.uint16)
                    copy_device_to_host(
                        host_array_ptr(bits),
                        out,
                        rows * spec.vocab_size * _BF16_NBYTES,
                        runtime=self.runtime,
                    )
                finally:
                    free(out, runtime=self.runtime)
                values = _bf16_bits_to_float32(bits).reshape(rows, spec.vocab_size)
                if capture is not None:
                    capture.capture_teacher_logits(values[-1].copy())
                return values
            if logits != "last":
                raise ValueError(f"unknown logits selector {logits!r}")
            selected = np.array([rows - 1], dtype=np.int64)
        else:
            selected = np.ascontiguousarray(logits, dtype=np.int64).reshape(-1)
            if selected.size == 0:
                raise ValueError("logits row selection must be non-empty")
            if selected.min() < 0 or selected.max() >= rows:
                raise ValueError("logits row selection is outside the prompt")

        out = malloc(selected.size * spec.vocab_size * 4, runtime=self.runtime)
        norm_host = self._copy_bf16_rows(scratch.norm, rows, spec.hidden_size)
        staging = np.ascontiguousarray(norm_host[selected])
        staging_buffer = malloc(staging.nbytes, runtime=self.runtime)
        try:
            copy_host_to_device(
                staging_buffer, host_array_ptr(staging), staging.nbytes, runtime=self.runtime
            )
            dense_gemv_bf16_f32_out(
                staging_buffer.ptr,
                self.loaded.lm_head.ptr,
                out.ptr,
                int(selected.size),
                spec.hidden_size,
                spec.vocab_size,
                runtime=self.runtime,
            )
            values = self._copy_f32_rows(out, int(selected.size) * spec.vocab_size).reshape(
                int(selected.size), spec.vocab_size
            )
        finally:
            free(staging_buffer, runtime=self.runtime)
            free(out, runtime=self.runtime)
        if capture is not None:
            capture.capture_teacher_logits(values[-1].copy())
        return values


__all__ = [
    "LlamaPrefillRunner",
    "rope_table",
]
