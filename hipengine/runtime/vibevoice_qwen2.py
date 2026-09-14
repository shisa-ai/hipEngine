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
from numbers import Integral
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
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
from hipengine.kernels.vibevoice import resolve_vibevoice_kernels,resolve_vibevoice_kernel,DECODER_PRIMITIVES
from hipengine.execution_profiles import build_variant_manifest,VariantSelection,manifest_sha256
from hipengine.loading.vibevoice_layout import f32_to_bf16_bits,conv_rows_out,transpose_conv_weight_t
from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.core.hipblaslt import HipblasLt, HIP_R_16F


def _prefill_gemm_lt(runner, x16_ptr, w16_ptr, out_f32_ptr, rows, in_features, out_features):
    """fp16-in/f32-out hipBLASLt GEMM (fp32 accumulate; same arithmetic class as
    the WMMA chain, which stages fp16 operands with f32 accumulation)."""
    shape = (rows,in_features,out_features)
    problem = runner._lt_problems[shape]
    problem.launch(runner._lt_algos[shape], x16_ptr, w16_ptr, out_f32_ptr, stream=0)


def _bf16_bits_to_fp16_bits(host_bf16: np.ndarray) -> np.ndarray:
    """bf16 bit patterns -> fp16 bit patterns (exact numeric value)."""
    f32 = (np.asarray(host_bf16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.ascontiguousarray(f32.astype(np.float16).view(np.uint16))


def _upload(host: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(host)
    buffer = malloc(array.nbytes)
    copy_host_array_to_device(buffer, array)
    return buffer


def _alloc(nbytes: int) -> DeviceBuffer:
    return malloc(nbytes)


class _PrefillScratchArena:
    """Per-runner arena holding one prefill call's scratch buffers.

    Both prefill routes allocate ~20 scratch buffers per call, and separate
    ``hipMalloc``/``hipFree`` pairs for them measured 4.2 ms of a 220 ms
    prefill (1.9%). The buffers are handed out as aligned views of a single
    arena that is rewound on reuse and replaced only when a larger row count
    needs more room, so repeated prefill calls allocate once.

    The views are owned by the arena: never free them individually, and free
    the arena itself through :meth:`close`.
    """

    _ALIGNMENT = 4096

    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._arena: DeviceMemoryArena | None = None

    def take(self, sizes: Sequence[int]) -> list[DeviceBuffer]:
        """Rewind the arena and return one view per requested size."""
        alignment = self._ALIGNMENT
        needed = sum(-(-int(size) // alignment) * alignment for size in sizes)
        arena = self._arena
        if arena is None or arena.closed or arena.capacity_bytes < needed:
            if arena is not None:
                arena.close()
            arena = DeviceMemoryArena.create(needed, runtime=self._runtime,
                                             alignment=alignment)
            self._arena = arena
        else:
            arena.rewind()
        return [arena.allocate(int(size)) for size in sizes]

    def close(self) -> None:
        if self._arena is not None:
            self._arena.close()
            self._arena = None


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
    q_w16: DeviceBuffer
    k_w16: DeviceBuffer
    v_w16: DeviceBuffer
    o_w16: DeviceBuffer
    gate_w16: DeviceBuffer
    up_w16: DeviceBuffer
    down_w16: DeviceBuffer
    k_cache: DeviceBuffer
    v_cache: DeviceBuffer


class VibevoiceQwen2Runtime:
    """Dense-KV incremental Qwen2 runner over raw device pointers."""

    # Recorded in the execution manifest; subclasses override.
    quant_name = 'bf16'

    def __init__(
        self,
        weights,
        *,
        max_context: int = 8192,
        library: ctypes.CDLL | None = None,
        backend: str = "auto",
        prefill_variant: str = "hipblaslt",
    ) -> None:
        if isinstance(max_context, bool) or not isinstance(max_context, Integral) or not 0 < max_context <= 16000:
            raise ValueError("max_context must be an integer in [1, 16000]")
        if prefill_variant not in {'strict','hipblaslt'}:
            raise ValueError('prefill_variant must be strict or hipblaslt')
        self.prefill_variant = prefill_variant
        self.prefill_fallback_reason = None
        spec = weights.spec
        self.spec = spec
        self.max_context = max_context
        self.kernels = resolve_vibevoice_kernels(backend)
        self.backend = self.kernels.backend
        self._prefill_routes = {variant: resolve_vibevoice_kernel(self.backend,'vibevoice_prefill',variant)
                                for variant in ('strict','hipblaslt')}
        self.runtime = get_hip_runtime()
        self.library = library or self.kernels.build_vibevoice_encoder()
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
        self._lt = None  # hipBLASLt prefill route, created lazily
        self._lt_problems: dict[tuple[int, int, int], object] = {}
        self._lt_algos: dict[tuple[int, int, int], object] = {}
        kv_bytes = max_context * kv_heads * head_dim * 2
        for layer in weights.layers:
            q_w16 = _bf16_bits_to_fp16_bits(f32_to_bf16_bits(np.asarray(layer.q_weight, dtype=np.float32).reshape(-1)))
            k_w16 = _bf16_bits_to_fp16_bits(f32_to_bf16_bits(np.asarray(layer.k_weight, dtype=np.float32).reshape(-1)))
            v_w16 = _bf16_bits_to_fp16_bits(f32_to_bf16_bits(np.asarray(layer.v_weight, dtype=np.float32).reshape(-1)))
            o_w16 = _bf16_bits_to_fp16_bits(f32_to_bf16_bits(np.asarray(layer.o_weight, dtype=np.float32).reshape(-1)))
            gate_w16 = _bf16_bits_to_fp16_bits(f32_to_bf16_bits(np.asarray(layer.gate_proj, dtype=np.float32).reshape(-1)))
            up_w16 = _bf16_bits_to_fp16_bits(f32_to_bf16_bits(np.asarray(layer.up_proj, dtype=np.float32).reshape(-1)))
            down_w16 = _bf16_bits_to_fp16_bits(f32_to_bf16_bits(np.asarray(layer.down_proj, dtype=np.float32).reshape(-1)))
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
                    q_w16=keep(_upload(q_w16)),
                    k_w16=keep(_upload(k_w16)),
                    v_w16=keep(_upload(v_w16)),
                    o_w16=keep(_upload(o_w16)),
                    gate_w16=keep(_upload(gate_w16)),
                    up_w16=keep(_upload(up_w16)),
                    down_w16=keep(_upload(down_w16)),
                    k_cache=_alloc(kv_bytes),
                    v_cache=_alloc(kv_bytes),
                )
            )

        self._ctx_len = _alloc(8)
        self._ctx_len_host = np.zeros(1, dtype=np.int64)
        self._slot_map = keep(_upload(np.arange(max_context,dtype=np.int32)))
        self._slot_positions = keep(_upload(np.arange(max_context,dtype=np.int64)))
        self._evicted = keep(_upload(np.zeros(max_context,dtype=np.bool_)))
        self._row_position = keep(_upload(np.zeros(1,dtype=np.int64)))

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
        # Caller-owned scratch for emulated dual GEMVs (Q4). Left None for
        # fused dense kernels that need no intermediate f32 buffer.
        self._dual_scratch = None
        # Caller-owned bf16 scratch for the Q4 o_proj decode GEMV. Left None
        # for the dense implementation, which takes keyword-only arguments and
        # would reject the extra one. Allocating it per call instead costs a
        # malloc/free pair on every layer of every decoded token.
        #
        # Deliberately NOT in close()'s attribute list: that list and
        # self._buffers are two independent free paths, so a buffer that is
        # keep()ed here must be released only through self._buffers.
        self._o_proj_x_bf16 = None
        self._silu = _alloc(ffn * 2)
        self._down_f32 = _alloc(hidden * 4)
        self._down_bf16 = _alloc(hidden * 2)
        self._logits_bf16 = _alloc(spec.vocab_size * 2)
        self._logits_f32 = _alloc(spec.vocab_size * 4)
        self._scale = 1.0 / float(np.sqrt(head_dim))
        # Per-call prefill scratch, reused across prefill calls on this runner.
        self._prefill_scratch = _PrefillScratchArena(self.runtime)

    # ------------------------------------------------------------------
    def _bias_add(self, out_ptr: int, x_ptr: int, b_ptr: int, width: int) -> None:
        self.kernels.vv_add_bias_f32(x_ptr, b_ptr, out_ptr, width, width,
                        library=self.library, runtime=self.runtime)

    def push_token(self, token_or_embed: np.ndarray, position: int) -> None:
        """Stage one token's input row (fp32 hidden, shape (hidden,)) at position."""
        self._validate_position(position)
        row = np.ascontiguousarray(token_or_embed, dtype=np.float32).reshape(-1)
        if row.shape[0] != self.spec.hidden_size:
            raise ValueError("token row must be hidden-sized fp32")
        f32_to_bf16_bits_row = f32_to_bf16_bits(row)
        copy_host_to_device(self._hidden, host_array_ptr(f32_to_bf16_bits_row))
        self._ctx_len_host[0] = position + 1
        copy_host_to_device(self._ctx_len, host_array_ptr(self._ctx_len_host))

    def forward_layers(self, position: int) -> None:
        """Run the staged hidden row through all layers; result in ``_hidden``."""
        self._validate_position(position)
        spec = self.spec
        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        kv_heads = spec.num_key_value_heads
        head_dim = spec.head_dim
        ffn = spec.intermediate_size
        ctx = position + 1
        self._ctx_len_host[0] = ctx
        copy_host_array_to_device(self._ctx_len,self._ctx_len_host)
        copy_host_array_to_device(self._row_position,np.array([position],dtype=np.int64))
        spans = self._spans(self._row_position,self._ctx_len,1)

        for layer in self.layers:
            self.kernels.vv_rmsnorm_bf16(self._hidden.ptr, layer.input_ln.ptr, self._normed.ptr,
                            1, hidden, spec.rms_norm_eps,
                            library=self.library, runtime=self.runtime)
            self.kernels.dense_gemv_bf16_f32_out(self._normed.ptr, layer.q_w.ptr, self._q.ptr,
                                    1, hidden, hidden, stream=0, runtime=self.runtime)
            self.kernels.dense_gemv_bf16_f32_out(self._normed.ptr, layer.k_w.ptr, self._k.ptr,
                                    1, hidden, kv_heads * head_dim, stream=0, runtime=self.runtime)
            self.kernels.dense_gemv_bf16_f32_out(self._normed.ptr, layer.v_w.ptr, self._v.ptr,
                                    1, hidden, kv_heads * head_dim, stream=0, runtime=self.runtime)
            self._bias_add(self._q.ptr, self._q.ptr, layer.q_b.ptr, hidden)
            self._bias_add(self._k.ptr, self._k.ptr, layer.k_b.ptr, kv_heads * head_dim)
            self._bias_add(self._v.ptr, self._v.ptr, layer.v_b.ptr, kv_heads * head_dim)

            cos_ptr = self._cos.ptr + position * head_dim * 4
            sin_ptr = self._sin.ptr + position * head_dim * 4
            self.kernels.qwen35_partial_rotary_f32(
                self._q.ptr, self._k.ptr, cos_ptr, sin_ptr,
                self._q_out.ptr, self._k_out.ptr, heads, kv_heads, head_dim, head_dim,
                stream=0, runtime=self.runtime,
            )

            self.kernels.f32_to_bf16(self._k_out.ptr, self._k_bf16.ptr, kv_heads * head_dim,
                        stream=0, runtime=self.runtime)
            self.kernels.f32_to_bf16(self._v.ptr, self._v_bf16.ptr, kv_heads * head_dim,
                        stream=0, runtime=self.runtime)
            self.kernels.vv_kv_write_spans(self._k_bf16.ptr,self._v_bf16.ptr,layer.k_cache.ptr,layer.v_cache.ptr,
                              spans,1,kv_heads,head_dim,library=self.library,runtime=self.runtime)
            self.kernels.vv_attention_spans(self._q_out.ptr,layer.k_cache.ptr,layer.v_cache.ptr,self._attn.ptr,
                               spans,1,heads,kv_heads,head_dim,self._scale,library=self.library,runtime=self.runtime)
            self.kernels.dense_gemv_f32_bf16w_f32_out(
                self._attn.ptr, layer.o_w.ptr, self._o_f32.ptr,
                1, heads * head_dim, hidden, stream=0, runtime=self.runtime,
                **({"x_bf16_scratch": self._o_proj_x_bf16}
                   if self._o_proj_x_bf16 else {}),
            )
            self.kernels.f32_to_bf16(self._o_f32.ptr, self._o_bf16.ptr, hidden,
                        stream=0, runtime=self.runtime)
            self.kernels.vv_scale_residual_bf16(self._hidden.ptr, self._o_bf16.ptr, self._ones_hidden.ptr,
                                  self._hidden.ptr, hidden, hidden,
                                  library=self.library, runtime=self.runtime)

            self.kernels.vv_rmsnorm_bf16(self._hidden.ptr, layer.post_ln.ptr, self._normed.ptr,
                            1, hidden, spec.rms_norm_eps,
                            library=self.library, runtime=self.runtime)
            self.kernels.dense_dual_gemv_out_bf16(
                self._normed.ptr, layer.gate_w.ptr, layer.up_w.ptr, self._gate_up.ptr,
                1, hidden, ffn, ffn, stream=0, runtime=self.runtime,
                **({"scratch": self._dual_scratch} if self._dual_scratch else {}),
            )
            self.kernels.silu_mul_dual_out_bf16(self._gate_up.ptr, self._silu.ptr, 1, ffn,
                                   stream=0, runtime=self.runtime)
            self.kernels.dense_gemv_out_bf16(self._silu.ptr, layer.down_w.ptr, self._down_bf16.ptr,
                                1, ffn, hidden, stream=0, runtime=self.runtime)
            self.kernels.vv_scale_residual_bf16(self._hidden.ptr, self._down_bf16.ptr, self._ones_hidden.ptr,
                                  self._hidden.ptr, hidden, hidden,
                                  library=self.library, runtime=self.runtime)

    def logits_argmax(self) -> tuple[np.ndarray, int]:
        """Final norm + lm head + host argmax over the staged hidden row."""
        spec = self.spec
        hidden = spec.hidden_size
        self.kernels.vv_rmsnorm_bf16(self._hidden.ptr, self.final_ln.ptr, self._normed.ptr,
                        1, hidden, spec.rms_norm_eps,
                        library=self.library, runtime=self.runtime)
        self.kernels.dense_gemv_bf16_f32_out(self._normed.ptr, self.lm_head.ptr, self._logits_f32.ptr,
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

    def _spans(self, positions, counts, rows):
        def tensor(buf,shape,dtype):
            return Tensor.from_handle(buf.ptr,shape,dtype,Device('hip'))
        return KVLiveSpans(
            tensor(self._slot_map,(self.max_context,),'int32'),
            tensor(counts,(rows,),'int64'), self.max_context,
            tensor(self._slot_positions,(self.max_context,),'int64'),
            tensor(self._evicted,(self.max_context,),'bool'), 'bf16',
            row_positions=tensor(positions,(rows,),'int64'),
            span_role='decode' if rows == 1 else 'prefill')

    def _validate_position(self, position: int) -> None:
        if (isinstance(position, bool) or not isinstance(position, Integral)
                or not 0 <= position < self.max_context):
            raise ValueError(f"position must be an integer in [0, {self.max_context})")

    # ------------------------------------------------------------------
    def _prepare_lt(self, rows):
        if self._lt is None:
            self._lt = HipblasLt()
        h, f = self.spec.hidden_size, self.spec.intermediate_size
        kv = self.spec.num_key_value_heads*self.spec.head_dim
        for inputs,outputs in ((h,h),(h,kv),(h,f),(f,h)):
            shape=(rows,inputs,outputs)
            if shape in self._lt_problems:
                continue
            problem=self._lt.problem(rows,inputs,outputs,0)
            algorithms=[a for a in problem.algorithms(16) if a.workspace_size == 0]
            if not algorithms:
                raise RuntimeError(f'no zero-workspace hipBLASLt algorithm for {shape}')
            self._lt_problems[shape]=problem
            self._lt_algos[shape]=algorithms[0]

    def prefill_rows(self, hidden_rows: DeviceBuffer, rows: int, start_pos: int) -> None:
        self._validate_position(start_pos)
        if isinstance(rows,bool) or not isinstance(rows,Integral) or rows <= 0:
            raise ValueError('prefill rows must be a positive integer')
        if start_pos+rows > self.max_context or hidden_rows.nbytes < rows*self.spec.hidden_size*2:
            raise ValueError('prefill exceeds cache or input buffer capacity')
        selected = self.prefill_variant
        self.prefill_fallback_reason = None
        if selected == 'hipblaslt':
            try:
                self._prepare_lt(rows)
            except (OSError,RuntimeError) as exc:
                # Capability negotiation is before any cache/hidden mutation.
                # Launch errors never fall back part way through a layer stack.
                selected = 'strict'
                self.prefill_fallback_reason = str(exc)
        call = self._prefill_routes[selected]
        self.variant_manifest = build_variant_manifest(
            profile='strict' if selected == 'strict' else 'production',
            backend=self.backend,model='vibevoice_asr',quant=self.quant_name,
            kv_policy='uniform_block1_spans',graph_policy='eager',
            selections=[VariantSelection(name,'decoder','strict','strict') for name in DECODER_PRIMITIVES]
                + [VariantSelection('vibevoice_prefill','decoder',selected,'strict')])
        self.variant_manifest_sha256 = manifest_sha256(self.variant_manifest)
        call(self,hidden_rows,rows,start_pos)

    def _prefill_batched(self, hidden_rows: DeviceBuffer, rows: int, start_pos: int) -> None:
        """Batched causal prefill of ``rows`` hidden rows at ``start_pos``.

        ``hidden_rows`` is (rows, hidden) bf16 and is overwritten with the
        post-layer-stack result. Audio-embed rows must already be staged by
        the caller (row p uses absolute position ``start_pos + p``).
        """
        self._validate_position(start_pos)
        if isinstance(rows, bool) or not isinstance(rows, Integral) or rows <= 0:
            raise ValueError("prefill rows must be a positive integer")
        if start_pos + rows > self.max_context:
            raise ValueError("prefill exceeds max_context")
        spec = self.spec
        hidden = spec.hidden_size
        heads = spec.num_attention_heads
        kv_heads = spec.num_key_value_heads
        head_dim = spec.head_dim
        ffn = spec.intermediate_size
        max_ctx = start_pos + rows
        if max_ctx > self.max_context:
            raise ValueError("prefill exceeds max_context")

        pos_host = np.arange(start_pos, start_pos + rows, dtype=np.int64)
        counts_host = pos_host + 1
        qkv_w = rows * (heads * head_dim) * 4
        kv_w = rows * kv_heads * head_dim * 4
        # Arena-owned views, rewound per call: freeing them individually would
        # free memory the arena still owns.
        scratch = self._prefill_scratch.take((
            rows * 8,               # positions
            rows * 8,               # counts
            rows * hidden * 2,      # normed
            rows * hidden * 2,      # normed fp16 (hipBLASLt input)
            qkv_w, qkv_w,           # q f32, q_out f32
            kv_w, kv_w, kv_w,       # k, v, k_out f32
            rows * kv_heads * head_dim * 2,  # k bf16
            rows * kv_heads * head_dim * 2,  # v bf16
            qkv_w,                  # attn out f32
            rows * hidden * 2,      # attn fp16
            rows * hidden * 4,      # o f32
            rows * hidden * 2,      # normed2
            rows * hidden * 2,      # normed2 fp16
            rows * ffn * 4, rows * ffn * 4,  # gate f32, up f32
            rows * ffn * 2, rows * ffn * 2,  # gate bf16, up bf16
            rows * ffn * 2,         # silu out bf16
            rows * ffn * 2,         # act fp16
            rows * hidden * 4,      # down f32
            rows * hidden * 2,      # down bf16
        ))
        (positions, counts, normed, normed16, q_f32, q_out, k_f32, v_f32, k_out,
         k_bf16, v_bf16, attn, attn16, o_f32, normed2, normed216, gate_f32,
         up_f32, gate, up, act, act16, down_f32, down_bf16) = scratch
        copy_host_array_to_device(positions, pos_host)
        copy_host_array_to_device(counts, counts_host)
        spans = self._spans(positions, counts, rows)
        kv_row_bytes = kv_heads * head_dim * 2
        for layer in self.layers:
            self.kernels.vv_rmsnorm_bf16(hidden_rows.ptr, layer.input_ln.ptr, normed.ptr,
                            rows, hidden, spec.rms_norm_eps,
                            library=self.library, runtime=self.runtime)
            # q/k/v projections: hipBLASLt fp16 GEMM -> f32, fp32 bias, fp32 rope
            self.kernels.bf16_to_fp16(normed.ptr, normed16.ptr, rows * hidden, stream=0, runtime=self.runtime)
            _prefill_gemm_lt(self, normed16.ptr, layer.q_w16.ptr, q_f32.ptr, rows, hidden, hidden)
            _prefill_gemm_lt(self, normed16.ptr, layer.k_w16.ptr, k_f32.ptr, rows, hidden, kv_heads * head_dim)
            _prefill_gemm_lt(self, normed16.ptr, layer.v_w16.ptr, v_f32.ptr, rows, hidden, kv_heads * head_dim)
            self.kernels.vv_add_bias_f32(q_f32.ptr, layer.q_b.ptr, q_f32.ptr, rows * hidden, hidden,
                            library=self.library, runtime=self.runtime)
            self.kernels.vv_add_bias_f32(k_f32.ptr, layer.k_b.ptr, k_f32.ptr, rows * kv_heads * head_dim, kv_heads * head_dim,
                            library=self.library, runtime=self.runtime)
            self.kernels.vv_add_bias_f32(v_f32.ptr, layer.v_b.ptr, v_f32.ptr, rows * kv_heads * head_dim, kv_heads * head_dim,
                            library=self.library, runtime=self.runtime)
            self.kernels.vv_rope_positions_f32(q_f32.ptr, k_f32.ptr, self._cos.ptr, self._sin.ptr,
                                  positions.ptr, q_out.ptr, k_out.ptr, rows, heads, kv_heads, head_dim,
                                  stream=0, runtime=self.runtime)
            self.kernels.f32_to_bf16(k_out.ptr, k_bf16.ptr, rows * kv_heads * head_dim,
                        stream=0, runtime=self.runtime)
            self.kernels.f32_to_bf16(v_f32.ptr, v_bf16.ptr, rows * kv_heads * head_dim,
                        stream=0, runtime=self.runtime)
            self.kernels.vv_kv_write_spans(k_bf16.ptr,v_bf16.ptr,layer.k_cache.ptr,layer.v_cache.ptr,
                              spans,rows,kv_heads,head_dim,library=self.library,runtime=self.runtime)
            self.kernels.vv_attention_spans(q_out.ptr,layer.k_cache.ptr,layer.v_cache.ptr,attn.ptr,
                               spans,rows,heads,kv_heads,head_dim,self._scale,library=self.library,runtime=self.runtime)
            # o projection: f32 -> fp16, hipBLASLt GEMM -> f32 -> bf16
            self.kernels.f32_to_fp16(attn.ptr, attn16.ptr, rows * heads * head_dim,
                        stream=0, runtime=self.runtime)
            _prefill_gemm_lt(self, attn16.ptr, layer.o_w16.ptr, o_f32.ptr, rows, heads * head_dim, hidden)
            self.kernels.f32_to_bf16(o_f32.ptr, down_bf16.ptr, rows * hidden,
                       stream=0, runtime=self.runtime)
            self.kernels.vv_scale_residual_bf16(hidden_rows.ptr, down_bf16.ptr, self._ones_hidden.ptr,
                                   hidden_rows.ptr, rows * hidden, hidden,
                                   library=self.library, runtime=self.runtime)
            self.kernels.vv_rmsnorm_bf16(hidden_rows.ptr, layer.post_ln.ptr, normed2.ptr,
                            rows, hidden, spec.rms_norm_eps,
                            library=self.library, runtime=self.runtime)
            self.kernels.bf16_to_fp16(normed2.ptr, normed216.ptr, rows * hidden, stream=0, runtime=self.runtime)
            _prefill_gemm_lt(self, normed216.ptr, layer.gate_w16.ptr, gate_f32.ptr, rows, hidden, ffn)
            _prefill_gemm_lt(self, normed216.ptr, layer.up_w16.ptr, up_f32.ptr, rows, hidden, ffn)
            self.kernels.f32_to_bf16(gate_f32.ptr, gate.ptr, rows * ffn, stream=0, runtime=self.runtime)
            self.kernels.f32_to_bf16(up_f32.ptr, up.ptr, rows * ffn, stream=0, runtime=self.runtime)
            self.kernels.silu_mul_separate_out_bf16(gate.ptr, up.ptr, act.ptr, rows, ffn,
                                       stream=0, runtime=self.runtime)
            self.kernels.bf16_to_fp16(act.ptr, act16.ptr, rows * ffn, stream=0, runtime=self.runtime)
            _prefill_gemm_lt(self, act16.ptr, layer.down_w16.ptr, down_f32.ptr, rows, ffn, hidden)
            self.kernels.f32_to_bf16(down_f32.ptr, down_bf16.ptr, rows * hidden, stream=0, runtime=self.runtime)
            self.kernels.vv_scale_residual_bf16(hidden_rows.ptr, down_bf16.ptr, self._ones_hidden.ptr,
                                   hidden_rows.ptr, rows * hidden, hidden,
                                   library=self.library, runtime=self.runtime)
        self._ctx_len_host[0] = start_pos + rows
        copy_host_to_device(self._ctx_len, host_array_ptr(self._ctx_len_host))

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        scratch_arena = getattr(self, "_prefill_scratch", None)
        if scratch_arena is not None:
            scratch_arena.close()
            self._prefill_scratch = None
        if self._lt is not None:
            self._lt.close()
            self._lt = None
        for attr in ("_hidden", "_normed", "_qkv_bf16", "_q", "_k", "_v", "_q_out", "_k_out",
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
    """Batched prefill of the prompt rows, then greedy decode."""
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits as _bits

    generated: list[int] = []
    total = len(input_rows)
    if total == 0:
        raise ValueError("empty prompt")
    if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral)
            or max_new_tokens < 0):
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if total + max(max_new_tokens - 1, 0) > runtime.max_context:
        raise ValueError("prompt and generation exceed max_context")
    if max_new_tokens == 0:
        return generated
    runtime.reset()

    rows_bf16 = _bits(np.asarray(input_rows, dtype=np.float32))  # (total, hidden) uint16
    prompt_buf = _upload(rows_bf16)
    try:
        runtime.prefill_rows(prompt_buf, total, 0)
        hidden = runtime.spec.hidden_size
        runtime.runtime.memcpy(
            runtime._hidden.ptr,
            prompt_buf.ptr + (total - 1) * hidden * 2,
            hidden * 2,
            MemcpyKind.DEVICE_TO_DEVICE,
        )
        _, token = runtime.logits_argmax()
    finally:
        free(prompt_buf)
    for step in range(max_new_tokens):
        generated.append(token)
        if step + 1 == max_new_tokens or (eos_token_id is not None and token == eos_token_id):
            break
        pos = total + step
        runtime.push_token(runtime.embed_row(token), pos)
        runtime.forward_layers(pos)
        _, token = runtime.logits_argmax()
    return generated
