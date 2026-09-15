"""Device-resident VibeVoice Qwen2 backbone over the f16 gfx1100 primitives."""

from __future__ import annotations

import ctypes
import math

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    MemcpyKind,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.vibevoice_qwen2 import (
    Qwen2Geometry,
    bf16_bytes_to_f32,
    expected_weight_shapes,
    load_backbone_weights,
    rope_cos_sin,
)
from hipengine.kernels.hip_gfx1100.vibevoice.qwen2_ops import (
    add_f16,
    append_kv_f16,
    bias_add_f16,
    build_qwen2_ops,
    decode_attn_f16,
    gemm_f16,
    prefill_attn_f16,
    rmsnorm_f16,
    rope_f16,
    silu_mul_f16,
)
from hipengine.core.rocblas import get_rocblas

_F16 = np.float16
_F32 = np.float32


def _upload_f16(runtime, array: np.ndarray):
    array = np.ascontiguousarray(array.astype(_F16))
    buf = malloc(array.nbytes, runtime=runtime)
    copy_host_array_to_device(buf, array, runtime=runtime)
    return buf


def _upload_f32(runtime, array: np.ndarray):
    array = np.ascontiguousarray(array.astype(_F32))
    buf = malloc(array.nbytes, runtime=runtime)
    copy_host_array_to_device(buf, array, runtime=runtime)
    return buf


class VibeVoiceQwen2Device:
    """Qwen2 backbone: resident f16 weights, device KV caches, raw-pointer kernels."""

    def __init__(
        self,
        runtime: HipRuntime | None = None,
        geometry: Qwen2Geometry = Qwen2Geometry(),
        max_context: int = 65536,
        library: ctypes.CDLL | None = None,
    ) -> None:
        self._runtime = runtime or get_hip_runtime()
        self._geometry = geometry
        self._library = library if library is not None else build_qwen2_ops()
        self._rocblas = get_rocblas()
        g = geometry
        self._hidden = g.hidden_size
        self._heads = g.num_attention_heads
        self._kv_heads = g.num_key_value_heads
        self._head_dim = g.head_dim
        self._vocab = g.vocab_size
        self._max_context = max_context

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | None = None,
        *,
        runtime: HipRuntime | None = None,
        geometry: Qwen2Geometry = Qwen2Geometry(),
        max_context: int = 65536,
        read_tensor=None,
    ) -> "VibeVoiceQwen2Device":
        if read_tensor is None:
            if model_path is None:
                raise ValueError("either model_path or read_tensor is required")
            from hipengine.loading.safetensors import (
                load_weight_index,
                read_tensor_storage_bytes,
            )

            index = load_weight_index(model_path)
            read_tensor = lambda name: read_tensor_storage_bytes(  # noqa: E731
                index.require([name])[0]
            )
        self = cls(runtime=runtime, geometry=geometry, max_context=max_context)
        g = geometry
        f32 = load_backbone_weights(read_tensor, g)
        # F16 storage for GEMM operands and the embedding; F32 for norm
        # weights and projection biases (the kernels read those as float*).
        self._w = {}
        for key, value in f32.items():
            if key == "embed_tokens.weight":
                continue
            if key.endswith(".bias") or key.endswith("norm.weight"):
                self._w[key] = _upload_f32(self._runtime, value)
            else:
                self._w[key] = _upload_f16(self._runtime, value)
        self._embed = _upload_f16(self._runtime, f32["embed_tokens.weight"])
        # Whole-context RoPE tables, uploaded once: [max_context, head_dim].
        cos_all, sin_all = rope_cos_sin(
            np.arange(max_context, dtype=np.float64),
            g.head_dim, g.rope_theta,
        )
        self._cos_table = _upload_f32(self._runtime, cos_all)
        self._sin_table = _upload_f32(self._runtime, sin_all)
        self._kv_cache: dict[int, tuple[DeviceBuffer, DeviceBuffer]] = {}
        self._length = 0
        # One bump arena per forward call: every layer's scratch comes from
        # here, so a decode step performs exactly one allocation.
        self._arena_bytes = 4 * max_context * (
            5 * g.hidden_size + 2 * g.intermediate_size
        ) * 2
        self._arena = malloc(self._arena_bytes, runtime=self._runtime)
        self._arena_offset = 0
        return self

    # -- device helpers ---------------------------------------------------

    def _buf_f16(self, rows: int, cols: int):
        buf = malloc(rows * cols * 2, runtime=self._runtime)
        return buf

    def _scratch_f16(self, rows: int, cols: int):
        """Bump-allocate f16 scratch from the per-call arena."""
        nbytes = rows * cols * 2
        self._arena_offset = (self._arena_offset + 255) & ~255
        if self._arena_offset + nbytes > self._arena_bytes:
            raise RuntimeError("qwen2 scratch arena exhausted")
        view = DeviceBuffer(ptr=self._arena.ptr + self._arena_offset,
                            nbytes=nbytes)
        self._arena_offset += nbytes
        return view

    def _kv(self, layer_idx: int) -> tuple[DeviceBuffer, DeviceBuffer]:
        if layer_idx not in self._kv_cache:
            kv_bytes = self._max_context * self._kv_heads * self._head_dim * 2
            self._kv_cache[layer_idx] = (
                malloc(kv_bytes, runtime=self._runtime),
                malloc(kv_bytes, runtime=self._runtime),
            )
        return self._kv_cache[layer_idx]

    def _cos_sin(self, start: int, count: int) -> tuple[DeviceBuffer, DeviceBuffer]:
        """Views into the resident whole-context tables (no allocation)."""
        row = self._head_dim * 4
        return (DeviceBuffer(ptr=self._cos_table.ptr + start * row,
                             nbytes=count * row),
                DeviceBuffer(ptr=self._sin_table.ptr + start * row,
                             nbytes=count * row))

    def reset(self) -> None:
        for k_buf, v_buf in self._kv_cache.values():
            free(k_buf, runtime=self._runtime)
            free(v_buf, runtime=self._runtime)
        self._kv_cache.clear()
        self._length = 0

    def close(self) -> None:
        self.reset()
        free(self._arena, runtime=self._runtime)
        free(self._cos_table, runtime=self._runtime)
        free(self._sin_table, runtime=self._runtime)

    @property
    def length(self) -> int:
        return self._length

    # -- one transformer layer -------------------------------------------

    def _gemm(self, x_buf, w_key, bias_key, out_buf, rows: int, k: int, n: int,
              rt) -> None:
        """Dense f16 GEMM via rocBLAS (f32 accumulate) + optional f32 bias."""
        self._rocblas.gemm_ex_rowmajor_nt_fp16_compute_f32(
            x_buf.ptr, self._w[w_key].ptr, out_buf.ptr,
            rows=rows, in_features=k, out_features=n, stream=0,
        )
        if bias_key is not None:
            bias_add_f16(self._library, out_buf.ptr, self._w[bias_key].ptr,
                         n, rows * n, runtime=rt)

    def _layer(self, layer_idx: int, x: DeviceBuffer, tokens: int,
               cache_len: int, pos_offset: int) -> DeviceBuffer:
        g = self._geometry
        lib, rt, h = self._library, self._runtime, self._hidden
        prefix = f"layers.{layer_idx}."
        p_attn = self._scratch_f16(tokens, h)
        rmsnorm_f16(lib, x.ptr, self._w[prefix + "input_layernorm.weight"].ptr,
                    p_attn.ptr, tokens, h, g.rms_norm_eps, stream=0, runtime=rt)
        # QKV projections: q [T, 1536], k/v [T, 256]; then per-matrix view.
        q = self._scratch_f16(tokens, self._heads * self._head_dim)
        k = self._scratch_f16(tokens, self._kv_heads * self._head_dim)
        v = self._scratch_f16(tokens, self._kv_heads * self._head_dim)
        self._gemm(p_attn, prefix + "self_attn.q_proj.weight",
                   prefix + "self_attn.q_proj.bias", q, tokens, h,
                   self._heads * self._head_dim, rt)
        self._gemm(p_attn, prefix + "self_attn.k_proj.weight",
                   prefix + "self_attn.k_proj.bias", k, tokens, h,
                   self._kv_heads * self._head_dim, rt)
        self._gemm(p_attn, prefix + "self_attn.v_proj.weight",
                   prefix + "self_attn.v_proj.bias", v, tokens, h,
                   self._kv_heads * self._head_dim, rt)
        cos, sin = self._cos_sin(pos_offset, tokens)
        rope_f16(lib, q.ptr, cos.ptr, sin.ptr, tokens, self._heads,
                 self._head_dim, runtime=rt)
        rope_f16(lib, k.ptr, cos.ptr, sin.ptr, tokens, self._kv_heads,
                 self._head_dim, runtime=rt)
        k_cache, v_cache = self._kv(layer_idx)
        ctx = self._scratch_f16(tokens, self._heads * self._head_dim)
        if cache_len == 0 and tokens == 1:
            # Single-token decode: append first, then decode in place.
            append_kv_f16(lib, k.ptr, v.ptr, k_cache.ptr, v_cache.ptr,
                          self._kv_heads, self._head_dim, cache_len, tokens, runtime=rt)
            decode_attn_f16(lib, q.ptr, k_cache.ptr, v_cache.ptr, cos.ptr,
                            sin.ptr, ctx.ptr, self._heads, self._kv_heads,
                            self._head_dim, self._max_context, cache_len, runtime=rt)
        else:
            prefill_attn_f16(lib, q.ptr, k.ptr, v.ptr, k_cache.ptr, v_cache.ptr,
                             cos.ptr, sin.ptr, ctx.ptr, self._heads,
                             self._kv_heads, self._head_dim, self._max_context,
                             cache_len, tokens, runtime=rt)
            append_kv_f16(lib, k.ptr, v.ptr, k_cache.ptr, v_cache.ptr,
                          self._kv_heads, self._head_dim, cache_len, tokens, runtime=rt)
        attn_out = self._scratch_f16(tokens, h)
        self._gemm(ctx, prefix + "self_attn.o_proj.weight", None, attn_out,
                   tokens, self._heads * self._head_dim, h, rt)
        # Residual + post norm + MLP.
        resid = self._scratch_f16(tokens, h)
        add_f16(lib, x.ptr, attn_out.ptr, resid.ptr, tokens * h, runtime=rt)
        p_mlp = self._scratch_f16(tokens, h)
        rmsnorm_f16(lib, resid.ptr, self._w[prefix + "post_attention_layernorm.weight"].ptr,
                    p_mlp.ptr, tokens, h, g.rms_norm_eps, runtime=rt)
        inter = g.intermediate_size
        gate = self._scratch_f16(tokens, inter)
        up = self._scratch_f16(tokens, inter)
        self._gemm(p_mlp, prefix + "mlp.gate_proj.weight", None, gate,
                   tokens, h, inter, rt)
        self._gemm(p_mlp, prefix + "mlp.up_proj.weight", None, up,
                   tokens, h, inter, rt)
        silu_mul_f16(lib, gate.ptr, up.ptr, tokens * inter, runtime=rt)
        mlp_out = self._buf_f16(tokens, h)
        self._gemm(gate, prefix + "mlp.down_proj.weight", None, mlp_out,
                   tokens, inter, h, rt)
        out = self._buf_f16(tokens, h)
        add_f16(lib, resid.ptr, mlp_out.ptr, out.ptr, tokens * h, runtime=rt)
        return out

    # -- device memory helpers --------------------------------------------

    def _download(self, buf, rows: int, cols: int) -> np.ndarray:
        out = np.empty((rows, cols), dtype=_F16)
        copy_device_to_host(host_array_ptr(out), buf, runtime=self._runtime)
        return out

    # -- public API --------------------------------------------------------

    def forward_hidden(self, token_ids: np.ndarray) -> np.ndarray:
        """Prefill tokens; returns the final-norm hidden states [T, H] f32."""
        tokens = int(token_ids.shape[0])
        x = self._gather_embed(token_ids)
        # A fresh prefill starts every layer's cache empty; the RoPE position
        # offset equals the sequence start (0 here).
        self._arena_offset = 0
        for layer_idx in range(self._geometry.num_hidden_layers):
            new_x = self._layer(layer_idx, x, tokens, 0, 0)
            free(x, runtime=self._runtime)
            x = new_x
        self._length = tokens
        normed = self._buf_f16(tokens, self._hidden)
        rmsnorm_f16(self._library, x.ptr, self._w["norm.weight"].ptr,
                    normed.ptr, tokens, self._hidden,
                    self._geometry.rms_norm_eps, runtime=self._runtime)
        hidden = self._download(normed, tokens, self._hidden).astype(_F32)
        free(normed, runtime=self._runtime)
        free(x, runtime=self._runtime)
        return hidden

    def decode_step(self, token_id: int) -> np.ndarray:
        """One autoregressive step from the current cache state. [H] f32."""
        x = self._gather_embed(np.array([token_id], dtype=np.int64))
        cache_len = self._length
        self._arena_offset = 0
        for layer_idx in range(self._geometry.num_hidden_layers):
            new_x = self._layer(layer_idx, x, 1, cache_len, cache_len)
            free(x, runtime=self._runtime)
            x = new_x
        self._length = cache_len + 1
        normed = self._buf_f16(1, self._hidden)
        rmsnorm_f16(self._library, x.ptr, self._w["norm.weight"].ptr,
                    normed.ptr, 1, self._hidden,
                    self._geometry.rms_norm_eps, runtime=self._runtime)
        hidden = self._download(normed, 1, self._hidden).astype(_F32)[0]
        free(normed, runtime=self._runtime)
        free(x, runtime=self._runtime)
        return hidden

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        buf = _upload_f16(self._runtime, hidden)
        out = self._buf_f16(hidden.shape[0], self._vocab)
        self._rocblas.gemm_ex_rowmajor_nt_fp16_compute_f32(
            buf.ptr, self._embed.ptr, out.ptr,
            rows=hidden.shape[0], in_features=self._hidden,
            out_features=self._vocab, stream=0,
        )
        result = self._download(out, hidden.shape[0], self._vocab).astype(_F32)
        free(buf, runtime=self._runtime)
        free(out, runtime=self._runtime)
        return result

    def _gather_embed(self, token_ids: np.ndarray) -> DeviceBuffer:
        ids = np.asarray(token_ids, dtype=np.int64)
        rows = ids.shape[0]
        # The embedding is resident as f16; gather rows via per-row copies.
        out = self._buf_f16(rows, self._hidden)
        row_bytes = self._hidden * 2
        for i, tok in enumerate(ids.tolist()):
            self._runtime.memcpy(
                out.ptr + i * row_bytes,
                self._embed.ptr + int(tok) * row_bytes,
                row_bytes,
                MemcpyKind.DEVICE_TO_DEVICE,
            )
        return out
