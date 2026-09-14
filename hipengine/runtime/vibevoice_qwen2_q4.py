"""Q4_K_M-backed VibeVoice Qwen2 runtime.

Same execution interface as ``VibevoiceQwen2Runtime`` (module-level
``greedy_generate`` works unchanged); the six backbone GEMM weights are
raw Q4_K/Q6_K GGUF blocks and the four linear primitives resolve from
the ``q4_k_m`` registry axis. Norms, biases, the untied bf16 lm_head and
the embedding stay dense. Prefill routes row-by-row through the q4
decode path (correctness-first); decode is the bandwidth win.
"""

from __future__ import annotations

import ctypes
from numbers import Integral

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, malloc
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_bf16_f32_out
from hipengine.kernels.vibevoice import (
    resolve_vibevoice_kernels,
    resolve_vibevoice_prefill_route,
)
from hipengine.runtime.vibevoice_qwen2 import (
    VibevoiceQwen2Runtime,
    _LayerBuffers,
    _alloc,
    _cos_sin_tables,
    _upload,
    f32_to_bf16_bits,
)


class VibevoiceQwen2Q4Runtime(VibevoiceQwen2Runtime):
    """Dense-KV Qwen2 runner over Q4_K_M GGUF block weights."""

    def __init__(
        self,
        q4_weights,
        *,
        max_context: int = 8192,
        library: ctypes.CDLL | None = None,
        backend: str = "auto",
    ) -> None:
        if isinstance(max_context, bool) or not isinstance(max_context, Integral) \
                or not 0 < max_context <= 16000:
            raise ValueError("max_context must be an integer in [1, 16000]")
        self.prefill_variant = "q4"
        self.prefill_fallback_reason = None
        spec = q4_weights.spec
        self.spec = spec
        self.max_context = max_context
        self.kernels = resolve_vibevoice_kernels(backend, quant="q4_k_m")
        self.backend = self.kernels.backend
        self._prefill_routes = {
            "strict": resolve_vibevoice_prefill_route(self.backend, "strict"),
            "q4": resolve_vibevoice_prefill_route(self.backend, "q4", quant="q4_k_m"),
        }
        self.runtime = get_hip_runtime()
        self.library = library or self.kernels.build_vibevoice_encoder()
        self._buffers: list[DeviceBuffer] = []
        self._scratch: dict[str, DeviceBuffer] = {}
        self._q4_owner = q4_weights

        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        kv_heads = spec.num_key_value_heads
        head_dim = spec.head_dim
        ffn = spec.intermediate_size

        def keep(buf: DeviceBuffer) -> DeviceBuffer:
            self._buffers.append(buf)
            return buf

        # embedding rows stay host-side bf16 bits for embed_row()
        embed_host = np.asarray(q4_weights.embed_host_bf16, dtype=np.float32).reshape(-1)
        self.embed_host_bf16 = f32_to_bf16_bits(embed_host).reshape(
            int(embed_host.shape[0]) // hidden, hidden
        )
        self.embed = keep(q4_weights.embed)
        self.final_ln = keep(q4_weights.final_norm)
        self.lm_head = keep(q4_weights.lm_head)
        self._ones_hidden = keep(_upload(f32_to_bf16_bits(np.ones(hidden, dtype=np.float32))))

        cos, sin = _cos_sin_tables(max_context, head_dim, spec.rope_theta)
        self._cos = keep(_upload(cos))
        self._sin = keep(_upload(sin))

        self.layers: list[_LayerBuffers] = []
        self._lt = None
        self._lt_problems: dict[tuple[int, int, int], object] = {}
        self._lt_algos: dict[tuple[int, int, int], object] = {}
        kv_bytes = max_context * kv_heads * head_dim * 2
        empty = DeviceBuffer(0, 0)
        for layer in q4_weights.layers:
            self.layers.append(
                _LayerBuffers(
                    input_ln=keep(layer.input_ln),
                    q_w=keep(layer.q_w),
                    q_b=keep(layer.q_b),
                    k_w=keep(layer.k_w),
                    k_b=keep(layer.k_b),
                    v_w=keep(layer.v_w),
                    v_b=keep(layer.v_b),
                    o_w=keep(layer.o_w),
                    post_ln=keep(layer.post_ln),
                    gate_w=keep(layer.gate_w),
                    up_w=keep(layer.up_w),
                    down_w=keep(layer.down_w),
                    q_w16=empty, k_w16=empty, v_w16=empty, o_w16=empty,
                    gate_w16=empty, up_w16=empty, down_w16=empty,
                    k_cache=_alloc(kv_bytes),
                    v_cache=_alloc(kv_bytes),
                )
            )

        self._ctx_len = _alloc(8)
        self._ctx_len_host = np.zeros(1, dtype=np.int64)
        self._slot_map = keep(_upload(np.arange(max_context, dtype=np.int32)))
        self._slot_positions = keep(_upload(np.arange(max_context, dtype=np.int64)))
        self._evicted = keep(_upload(np.zeros(max_context, dtype=np.bool_)))
        self._row_position = keep(_upload(np.zeros(1, dtype=np.int64)))

        # fp32 / bf16 scratch (one token)
        self._hidden = _alloc(hidden * 2)
        self._normed = _alloc(hidden * 2)
        self._qkv_bf16 = _alloc(hidden * 2)
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

    def logits_argmax(self) -> tuple[np.ndarray, int]:
        """Final norm + dense bf16 lm head + host argmax (lm_head stays dense)."""
        spec = self.spec
        hidden = spec.hidden_size
        from hipengine.core.memory import copy_device_to_host, host_array_ptr

        self.kernels.vv_rmsnorm_bf16(
            self._hidden.ptr, self.final_ln.ptr, self._normed.ptr,
            1, hidden, spec.rms_norm_eps,
            library=self.library, runtime=self.runtime,
        )
        dense_gemv_bf16_f32_out(
            self._normed.ptr, self.lm_head.ptr, self._logits_f32.ptr,
            1, hidden, spec.vocab_size, stream=0, runtime=self.runtime,
        )
        host = np.empty(spec.vocab_size, dtype=np.float32)
        copy_device_to_host(host_array_ptr(host), self._logits_f32, spec.vocab_size * 4)
        return host, int(host.argmax())

    def close(self) -> None:
        # The loader's device buffers were keep()ed into _buffers, so the
        # base close frees them exactly once; nothing else to release.
        super().close()
