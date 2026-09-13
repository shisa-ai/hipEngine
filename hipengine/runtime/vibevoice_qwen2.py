"""GPU runtime for the VibeVoice-ASR Qwen2 text backbone (v1 incremental).

One code path serves prefill and decode: tokens stream through the layer
stack one at a time with a dense contiguous KV cache
``(max_context, num_kv_heads, head_dim)`` bf16 per layer. This matches the
validated CPU reference semantics exactly and reuses proven in-tree
kernels (dense GEMV family, rotate-half rope, dense GQA decode attention,
silu-mul, casts) plus the vibevoice elementwise family.

Precision: bf16 weights; fp32 for q/k/v projections, bias adds, rope,
attention, and o/down projections' inputs; the hidden stream is bf16.
Greedy argmax runs host-side on the downloaded logits (v1; the lm head is
the dominant GEMV cost anyway).

The batched prefill/decode kernel path (paged KV spans, flash prefill,
fused QKV) is the follow-up optimization once this path is parity-proven.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_full_attn_decode_context_bf16,
)
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_dual_out_bf16
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    dense_dual_gemv_out_bf16,
    dense_gemv_bf16_f32_out,
    dense_gemv_f32_bf16w_f32_out,
    dense_gemv_out_bf16,
)
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import (
    qwen35_partial_rotary_f32,
)
from hipengine.kernels.hip_gfx1100.vibevoice.encoder import (
    build_vibevoice_encoder,
    f32_to_bf16_bits,
    vv_add_bias_f32,
    vv_rmsnorm_bf16,
    vv_scale_residual_bf16,
)


def _upload(host: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(host)
    buffer = malloc(array.nbytes)
    copy_host_array_to_device(buffer, array)
    return buffer


def _alloc(nbytes: int) -> DeviceBuffer:
    return malloc(nbytes)


def _cos_sin_tables(max_positions: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate-half cos/sin tables, per position, duplicated halves (HF layout)."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    angles = np.arange(max_positions, dtype=np.float64)[:, None] * inv[None, :]
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)
    cos = np.concatenate([cos_half, cos_half], axis=1)
    sin = np.concatenate([sin_half, sin_half], axis=1)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


@dataclass
class _LayerBuffers:
    input_ln: DeviceBuffer
    q_w: DeviceBuffer
    q_b: DeviceBuffer
    k_w: DeviceBuffer
    k_b: DeviceBuffer
    v_w: DeviceBuffer
    v_b: DeviceBuffer
    o_w: DeviceBuffer
    post_ln: DeviceBuffer
    gate_w: DeviceBuffer
    up_w: DeviceBuffer
    down_w: DeviceBuffer
    k_cache: DeviceBuffer
    v_cache: DeviceBuffer


class VibevoiceQwen2Runtime:
    """Dense-KV incremental Qwen2 runner over raw device pointers."""

    def __init__(
        self,
        weights,
        *,
        max_context: int = 8192,
        library: ctypes.CDLL | None = None,
    ) -> None:
        spec = weights.spec
        self.spec = spec
        self.max_context = max_context
        self.runtime = get_hip_runtime()
        self.library = library or build_vibevoice_encoder()
        self._buffers: list[DeviceBuffer] = []
        self._scratch: dict[str, DeviceBuffer] = {}

        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        kv_heads = spec.num_key_value_heads
        head_dim = spec.head_dim
        ffn = spec.intermediate_size

        def keep(buf: DeviceBuffer) -> DeviceBuffer:
            self._buffers.append(buf)
            return buf

        self.embed_host_bf16 = f32_to_bf16_bits(np.asarray(weights.embed_tokens, dtype=np.float32))
        self.embed = keep(_upload(self.embed_host_bf16))
        self.final_ln = keep(_upload(f32_to_bf16_bits(weights.final_norm)))
        self.lm_head = keep(_upload(f32_to_bf16_bits(np.asarray(weights.lm_head, dtype=np.float32).reshape(-1))))
        self._ones_hidden = keep(_upload(f32_to_bf16_bits(np.ones(hidden, dtype=np.float32))))

        cos, sin = _cos_sin_tables(max_context, head_dim, spec.rope_theta)
        self._cos = keep(_upload(cos))
        self._sin = keep(_upload(sin))

        self.layers: list[_LayerBuffers] = []
        kv_bytes = max_context * kv_heads * head_dim * 2
        for layer in weights.layers:
            self.layers.append(
                _LayerBuffers(
                    input_ln=keep(_upload(f32_to_bf16_bits(layer.input_layernorm))),
                    q_w=keep(_upload(f32_to_bf16_bits(np.asarray(layer.q_weight, dtype=np.float32).reshape(-1)))),
                    q_b=keep(_upload(np.ascontiguousarray(layer.q_bias, dtype=np.float32))),
                    k_w=keep(_upload(f32_to_bf16_bits(np.asarray(layer.k_weight, dtype=np.float32).reshape(-1)))),
                    k_b=keep(_upload(np.ascontiguousarray(layer.k_bias, dtype=np.float32))),
                    v_w=keep(_upload(f32_to_bf16_bits(np.asarray(layer.v_weight, dtype=np.float32).reshape(-1)))),
                    v_b=keep(_upload(np.ascontiguousarray(layer.v_bias, dtype=np.float32))),
                    o_w=keep(_upload(f32_to_bf16_bits(np.asarray(layer.o_weight, dtype=np.float32).reshape(-1)))),
                    post_ln=keep(_upload(f32_to_bf16_bits(layer.post_attention_layernorm))),
                    gate_w=keep(_upload(f32_to_bf16_bits(np.asarray(layer.gate_proj, dtype=np.float32).reshape(-1)))),
                    up_w=keep(_upload(f32_to_bf16_bits(np.asarray(layer.up_proj, dtype=np.float32).reshape(-1)))),
                    down_w=keep(_upload(f32_to_bf16_bits(np.asarray(layer.down_proj, dtype=np.float32).reshape(-1)))),
                    k_cache=_alloc(kv_bytes),
                    v_cache=_alloc(kv_bytes),
                )
            )

        self._ctx_len = _alloc(8)
        self._ctx_len_host = np.zeros(1, dtype=np.int64)

        # fp32 / bf16 scratch (one token)
        self._hidden = _alloc(hidden * 2)
        self._normed = _alloc(hidden * 2)
        self._q = _alloc(heads * head_dim * 4)
        self._k = _alloc(kv_heads * head_dim * 4)
        self._v = _alloc(kv_heads * head_dim * 4)
        self._q_out = _alloc(heads * head_dim * 4)
        self._k_out = _alloc(kv_heads * head_dim * 4)
        self._k_bf16 = _alloc(kv_heads * head_dim * 2)
        self._v_bf16 = _alloc(kv_heads * head_dim * 2)
        self._attn = _alloc(heads * head_dim * 4)
        self._o_f32 = _alloc(hidden * 4)
        self._o_bf16 = _alloc(hidden * 2)
        self._gate_up = _alloc(2 * ffn * 2)
        self._silu = _alloc(ffn * 2)
        self._down_f32 = _alloc(hidden * 4)
        self._down_bf16 = _alloc(hidden * 2)
        self._logits_bf16 = _alloc(spec.vocab_size * 2)
        self._logits_f32 = _alloc(spec.vocab_size * 4)
        self._scale = 1.0 / float(np.sqrt(head_dim))

    # ------------------------------------------------------------------
    def _bias_add(self, out_ptr: int, x_ptr: int, b_ptr: int, width: int) -> None:
        vv_add_bias_f32(x_ptr, b_ptr, out_ptr, width, width,
                        library=self.library, runtime=self.runtime)

    def push_token(self, token_or_embed: np.ndarray, position: int) -> None:
        """Stage one token's input row (fp32 hidden, shape (hidden,)) at position."""
        row = np.ascontiguousarray(token_or_embed, dtype=np.float32).reshape(-1)
        if row.shape[0] != self.spec.hidden_size:
            raise ValueError("token row must be hidden-sized fp32")
        f32_to_bf16_bits_row = f32_to_bf16_bits(row)
        copy_host_to_device(self._hidden, host_array_ptr(f32_to_bf16_bits_row))
        self._ctx_len_host[0] = position + 1
        copy_host_to_device(self._ctx_len, host_array_ptr(self._ctx_len_host))

    def forward_layers(self, position: int) -> None:
        """Run the staged hidden row through all layers; result in ``_hidden``."""
        spec = self.spec
        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        kv_heads = spec.num_key_value_heads
        head_dim = spec.head_dim
        ffn = spec.intermediate_size
        ctx = position + 1

        for layer in self.layers:
            vv_rmsnorm_bf16(self._hidden.ptr, layer.input_ln.ptr, self._normed.ptr,
                            1, hidden, spec.rms_norm_eps,
                            library=self.library, runtime=self.runtime)
            dense_gemv_bf16_f32_out(self._normed.ptr, layer.q_w.ptr, self._q.ptr,
                                    1, hidden, hidden, stream=0, runtime=self.runtime)
            dense_gemv_bf16_f32_out(self._normed.ptr, layer.k_w.ptr, self._k.ptr,
                                    1, hidden, kv_heads * head_dim, stream=0, runtime=self.runtime)
            dense_gemv_bf16_f32_out(self._normed.ptr, layer.v_w.ptr, self._v.ptr,
                                    1, hidden, kv_heads * head_dim, stream=0, runtime=self.runtime)
            self._bias_add(self._q.ptr, self._q.ptr, layer.q_b.ptr, hidden)
            self._bias_add(self._k.ptr, self._k.ptr, layer.k_b.ptr, kv_heads * head_dim)
            self._bias_add(self._v.ptr, self._v.ptr, layer.v_b.ptr, kv_heads * head_dim)

            cos_ptr = self._cos.ptr + position * head_dim * 4
            sin_ptr = self._sin.ptr + position * head_dim * 4
            qwen35_partial_rotary_f32(
                self._q.ptr, self._k.ptr, cos_ptr, sin_ptr,
                self._q_out.ptr, self._k_out.ptr, heads, kv_heads, head_dim, head_dim,
                stream=0, runtime=self.runtime,
            )

            f32_to_bf16(self._k_out.ptr, self._k_bf16.ptr, kv_heads * head_dim,
                        stream=0, runtime=self.runtime)
            f32_to_bf16(self._v.ptr, self._v_bf16.ptr, kv_heads * head_dim,
                        stream=0, runtime=self.runtime)
            kv_row_bytes = kv_heads * head_dim * 2
            self.runtime.memcpy(
                layer.k_cache.ptr + position * kv_row_bytes, self._k_bf16.ptr,
                kv_row_bytes, MemcpyKind.DEVICE_TO_DEVICE,
            )
            self.runtime.memcpy(
                layer.v_cache.ptr + position * kv_row_bytes, self._v_bf16.ptr,
                kv_row_bytes, MemcpyKind.DEVICE_TO_DEVICE,
            )

            qwen35_full_attn_decode_context_bf16(
                self._q_out.ptr, layer.k_cache.ptr, layer.v_cache.ptr, self._attn.ptr,
                self._ctx_len.ptr, ctx, heads, kv_heads, head_dim, self._scale,
                stream=0, runtime=self.runtime,
            )
            dense_gemv_f32_bf16w_f32_out(self._attn.ptr, layer.o_w.ptr, self._o_f32.ptr,
                                         1, heads * head_dim, hidden, stream=0, runtime=self.runtime)
            f32_to_bf16(self._o_f32.ptr, self._o_bf16.ptr, hidden,
                        stream=0, runtime=self.runtime)
            vv_scale_residual_bf16(self._hidden.ptr, self._o_bf16.ptr, self._ones_hidden.ptr,
                                  self._hidden.ptr, hidden, hidden,
                                  library=self.library, runtime=self.runtime)

            vv_rmsnorm_bf16(self._hidden.ptr, layer.post_ln.ptr, self._normed.ptr,
                            1, hidden, spec.rms_norm_eps,
                            library=self.library, runtime=self.runtime)
            dense_dual_gemv_out_bf16(
                self._normed.ptr, layer.gate_w.ptr, layer.up_w.ptr, self._gate_up.ptr,
                1, hidden, ffn, ffn, stream=0, runtime=self.runtime,
            )
            silu_mul_dual_out_bf16(self._gate_up.ptr, self._silu.ptr, 1, ffn,
                                   stream=0, runtime=self.runtime)
            # bf16 silu output feeds the bf16 GEMV directly; the fp32-input
            # variant would misread bf16 storage as fp32.
            dense_gemv_out_bf16(self._silu.ptr, layer.down_w.ptr, self._down_bf16.ptr,
                                1, ffn, hidden, stream=0, runtime=self.runtime)
            vv_scale_residual_bf16(self._hidden.ptr, self._down_bf16.ptr, self._ones_hidden.ptr,
                                  self._hidden.ptr, hidden, hidden,
                                  library=self.library, runtime=self.runtime)

    def logits_argmax(self) -> tuple[np.ndarray, int]:
        """Final norm + lm head + host argmax over the staged hidden row."""
        spec = self.spec
        hidden = spec.hidden_size
        vv_rmsnorm_bf16(self._hidden.ptr, self.final_ln.ptr, self._normed.ptr,
                        1, hidden, spec.rms_norm_eps,
                        library=self.library, runtime=self.runtime)
        dense_gemv_bf16_f32_out(self._normed.ptr, self.lm_head.ptr, self._logits_f32.ptr,
                                1, hidden, spec.vocab_size, stream=0, runtime=self.runtime)
        host = np.empty(spec.vocab_size, dtype=np.float32)
        copy_device_to_host(host_array_ptr(host), self._logits_f32, spec.vocab_size * 4)
        return host, int(host.argmax())

    def embed_row(self, token_id: int) -> np.ndarray:
        """Host embedding lookup widened to fp32 (bf16 storage)."""
        row16 = self.embed_host_bf16[token_id]
        return (row16.astype(np.uint32) << 16).view(np.float32)

    def reset(self) -> None:
        self._ctx_len_host[0] = 0

    def close(self) -> None:
        for attr in ("_hidden", "_normed", "_q", "_k", "_v", "_q_out", "_k_out",
                     "_k_bf16", "_v_bf16", "_attn", "_o_f32", "_o_bf16", "_gate_up",
                     "_silu", "_down_f32", "_down_bf16", "_logits_bf16", "_logits_f32",
                     "_ctx_len"):
            buf = getattr(self, attr, None)
            if isinstance(buf, DeviceBuffer):
                free(buf)
                setattr(self, attr, buf.__class__(0, 0))
        for layer in self.layers:
            free(layer.k_cache)
            free(layer.v_cache)
        for buf in self._buffers:
            free(buf)
        self._buffers.clear()


def greedy_generate(
    runtime: VibevoiceQwen2Runtime,
    input_rows: Sequence[np.ndarray],
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> list[int]:
    """Stream prompt rows (fp32 hidden-sized), then greedy-decode.

    ``input_rows`` are the per-position input rows (token embeddings, or
    audio embeddings at placeholder positions) already fp32.
    """
    runtime.reset()
    generated: list[int] = []
    total = len(input_rows)
    for pos, row in enumerate(input_rows):
        runtime.push_token(row, pos)
        runtime.forward_layers(pos)
    if total == 0:
        raise ValueError("empty prompt")
    logits, token = runtime.logits_argmax()
    for step in range(max_new_tokens):
        generated.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        pos = total + step
        runtime.push_token(runtime.embed_row(token), pos)
        runtime.forward_layers(pos)
        logits, token = runtime.logits_argmax()
    return generated
