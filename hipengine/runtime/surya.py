"""Torch-free Surya OCR 2 HIP runtime (fp32 strict path, gfx1100/gfx1151).

Runs the Surya text decoder on the HIP device, mirroring the CPU reference
(``hipengine.kernels.cpu_reference.surya``) stage for stage:

- embedding lookup with sequential visual-feature injection at image-pad
  positions (``hipengine_evie_embed_lookup_f32``),
- 18 gated-DeltaNet layers: rocBLAS SGEMM projections, segment-aware fp32
  causal-conv prefill (SiLU inside the kernel, final window written to a
  persistent per-layer state slot), q/K l2-normalization, sigmoid/decay
  gate prep, the normalized cluster8 delta-rule recurrence (persistent
  per-layer state), RMSNormGated, out projection,
- 6 causal full-attention layers: per-head RMSNorm, per-head fused q|gate
  deinterleave, interleaved partial mRoPE (tables built on the host with
  the shared family helper), batched SGEMM scores + causal mask + softmax,
  GQA-mapped AV product, sigmoid gate, o projection, persistent per-layer
  KV caches that decode continues from,
- SiLU MLPs, final RMSNorm, tied LM head over the embedding table.

The single-token decode step reuses the prefill recurrence kernel with
``tokens=1`` (algebraically identical to the CPU reference decode loop:
state decay, keyed memory read, delta write-back, query read-out) plus the
fp32 conv decode kernel. The correctness gate is greedy-token parity and
logit agreement against the CPU reference, not a loose statistical gate.

The vision tower currently runs on the host (CPU reference); its device
port is tracked as follow-up work. This module owns the text stack only.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.core.memory import free as hip_free
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.cpu_reference.evie import text_rope_tables
from hipengine.kernels.cpu_reference.surya import SuryaSpec, SuryaWeights
from hipengine.kernels.hip_gfx1100.evie.evie_ops import build_evie_ops
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    build_qwen35_linear_attn_conv,
    qwen35_linear_attn_conv_decode_f32,
    qwen35_linear_attn_conv_prefill_segments_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_prefill_recurrent_f32,
    qwen35_gdn_prefill_recurrent_normalized_cluster8_f32,
)
from hipengine.kernels.hip_gfx1100.surya.surya_ops import (
    build_surya_ops,
    surya_causal_mask_scale_f32,
    surya_gdn_l2norm_f32,
    surya_scatter_kv_f32,
    surya_split_qgate_f32,
)

_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p

_GEMM_PAD_BYTES = 512

# Vision attention materializes the full score matrix, so its scratch grows as
# heads * patches^2 * 4 bytes. At the checkpoint's own max_pixels ceiling
# (16_777_216 px -> 14400 patches) that is ~9.95 GB for one request. This cap
# admits a 300-DPI A4 page (77x109 grid, ~3.4 GB) and rejects the pathological
# ceiling with a clear error instead of attempting the allocation. Tiled vision
# attention removes the need for a large scratch; until then the cap is the
# admission contract. Pass ``max_vision_scratch_bytes=None`` to disable it.
DEFAULT_MAX_VISION_SCRATCH_BYTES = 4 * 1024**3


class SuryaGpuRuntimeError(RuntimeError):
    pass


def _raw_buffer(ptr: int, nbytes: int) -> DeviceBuffer:
    """A non-owning view over an existing device pointer (for copies)."""

    return DeviceBuffer(ptr=ptr, nbytes=nbytes)


class SuryaGpuRunner:
    """Surya OCR 2 fp32 text decoder on the HIP device."""

    def __init__(
        self,
        weights: SuryaWeights | str,
        spec: SuryaSpec | None = None,
        *,
        max_seq: int = 2048,
        max_vision_scratch_bytes: int | None = DEFAULT_MAX_VISION_SCRATCH_BYTES,
        rocblas: Rocblas | None = None,
        runtime: HipRuntime | None = None,
        library: ctypes.CDLL | None = None,
        conv_library: ctypes.CDLL | None = None,
        gdn_library: ctypes.CDLL | None = None,
    ):
        if isinstance(weights, str):
            weights = SuryaWeights.load(weights)
        self.spec = spec or SuryaSpec()
        self.max_seq = max_seq
        self.max_vision_scratch_bytes = max_vision_scratch_bytes
        self.runtime = runtime or get_hip_runtime()
        self.rocblas = rocblas or Rocblas.load()
        self.rocblas.set_workspace(0, 0)
        self.library = library or build_evie_ops(load=True)
        self.conv_library = conv_library or build_qwen35_linear_attn_conv(load=True)
        self.gdn_library = gdn_library or build_qwen35_linear_attn_gdn(load=True)
        self.surya_library = build_surya_ops(load=True)
        self._w: dict[str, DeviceBuffer] = {}
        # every long-lived allocation, freed by close(); named for what it holds
        self._permanent_bufs: list[DeviceBuffer] = []

        s = self.spec
        self.n_gdn_layers = sum(1 for l in range(s.num_layers) if not s.is_full_attention(l))
        self.n_attn_layers = s.num_layers - self.n_gdn_layers
        # persistent per-layer device state
        self._conv_state: dict[int, DeviceBuffer] = {}
        self._gdn_state: dict[int, DeviceBuffer] = {}
        self._kv_cache: dict[int, tuple[DeviceBuffer, DeviceBuffer]] = {}
        self._upload_weights(weights)
        self._alloc_state()
        # scratch cache (per-key device buffers grown on demand)
        self._scratch: dict[str, DeviceBuffer] = {}
        # cached device pointer arrays (keyed by pointer tuple)
        self._ptr_array_bufs: dict[tuple, DeviceBuffer] = {}
        # persistent tiny staging buffers for per-step scalar/rope uploads
        self._ids_buf = self._permanent(8)
        self._pos_buf = self._permanent(8)
        self._cu_buf = self._permanent(16)
        self._state_idx_buf = self._permanent(8)
        self._rope_cos_buf = self._permanent(s.num_attention_heads * 0 + 256)
        self._rope_sin_buf = self._permanent(256)
        self._visual_buf: DeviceBuffer | None = None
        self._seq_len = 0

    # -- setup ------------------------------------------------------------------

    def _upload_weights(self, weights: SuryaWeights) -> None:
        s = self.spec
        needed: list[str] = ["model.language_model.embed_tokens.weight",
                             "model.language_model.norm.weight"]
        for layer in range(s.num_layers):
            lp = f"model.language_model.layers.{layer}."
            needed += [lp + "input_layernorm.weight", lp + "post_attention_layernorm.weight",
                       lp + "mlp.gate_proj.weight", lp + "mlp.up_proj.weight",
                       lp + "mlp.down_proj.weight"]
            if s.is_full_attention(layer):
                p = lp + "self_attn."
                needed += [p + n for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                           "q_norm.weight", "k_norm.weight", "o_proj.weight")]
            else:
                p = lp + "linear_attn."
                needed += [p + n for n in ("in_proj_qkv.weight", "in_proj_z.weight",
                                           "in_proj_b.weight", "in_proj_a.weight",
                                           "conv1d.weight", "A_log", "dt_bias",
                                           "norm.weight", "out_proj.weight")]
        # vision tower + merger
        needed += ["model.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.bias",
                   "model.visual.pos_embed.weight",
                   "model.visual.merger.norm.weight", "model.visual.merger.norm.bias",
                   "model.visual.merger.linear_fc1.weight", "model.visual.merger.linear_fc1.bias",
                   "model.visual.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.bias"]
        for i in range(s.vision_depth):
            vp = f"model.visual.blocks.{i}."
            needed += [vp + n for n in ("norm1.weight", "norm1.bias",
                                        "attn.qkv.weight", "attn.qkv.bias",
                                        "attn.proj.weight", "attn.proj.bias",
                                        "norm2.weight", "norm2.bias",
                                        "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
                                        "mlp.linear_fc2.weight", "mlp.linear_fc2.bias")]
        for name in needed:
            arr = np.ascontiguousarray(weights[name].astype(np.float32))
            if name.endswith("conv1d.weight"):
                arr = arr.reshape(-1, arr.shape[-1])  # (channels, 1, k) -> (channels, k)
            buf = malloc(arr.nbytes + _GEMM_PAD_BYTES)
            self._upload(buf, arr)
            self._w[name] = buf

    def _alloc_state(self) -> None:
        s = self.spec
        n_conv = 3 * s.gdn_num_value_heads * s.gdn_value_head_dim  # qkv channels
        for layer in range(s.num_layers):
            if not s.is_full_attention(layer):
                self._conv_state[layer] = self._permanent(
                    n_conv * s.gdn_conv_kernel * 4)
                self._gdn_state[layer] = self._permanent(
                    s.gdn_num_value_heads * s.gdn_key_head_dim * s.gdn_value_head_dim * 4)
            else:
                nk, hd = s.num_key_value_heads, s.head_dim
                self._kv_cache[layer] = (
                    self._permanent(nk * self.max_seq * hd * 4),
                    self._permanent(nk * self.max_seq * hd * 4),
                )

    def _permanent(self, nbytes: int) -> DeviceBuffer:
        buf = malloc(nbytes + _GEMM_PAD_BYTES)
        self._permanent_bufs.append(buf)
        return buf

    def _buf(self, key: str, nbytes: int) -> DeviceBuffer:
        buf = self._scratch.get(key)
        if buf is None or buf.nbytes < nbytes + _GEMM_PAD_BYTES:
            if buf is not None:
                hip_free(buf)
            buf = malloc(nbytes + _GEMM_PAD_BYTES)
            self._scratch[key] = buf
        return buf

    # -- low-level helpers --------------------------------------------------------

    def _upload(self, buf: DeviceBuffer, arr: np.ndarray, dtype=np.float32) -> None:
        """H2D copy that pins the host source's lifetime across the transfer.

        On this stack the DMA of an unpinned-source hipMemcpy reads the host
        buffer after the call returns, so a same-statement temporary can be
        freed and recycled before the copy lands (observed as stale-heap
        garbage at the destination). Keep the source alive and synchronize
        before releasing it.
        """
        src = np.ascontiguousarray(arr, dtype=dtype)
        copy_host_to_device(buf, host_array_ptr(src), src.nbytes)
        self.runtime.device_synchronize()

    def _k(self, symbol: str, argtypes: list) -> ctypes._FuncPtr:
        fn = getattr(self.library, symbol, None)
        if fn is None:
            raise SuryaGpuRuntimeError(f"missing symbol {symbol}")
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn

    def _check(self, err: int, what: str) -> None:
        if err != 0:
            raise SuryaGpuRuntimeError(f"{what} failed: hip error {err}")

    def _gemm(self, x_ptr: int, w_ptr: int, out_ptr: int, rows: int, fin: int, fout: int) -> None:
        if rows == 1:
            # Decode is one row: rocBLAS SGEMM runs at ~128 GB/s there (it is
            # tuned for a wide n), while SGEMV reads the same weights at
            # ~316 GB/s. Measured 2.48x on a 3185x1152 fp32 weight on gfx1151.
            # Both accumulate in fp32; the only difference is summation order.
            self.rocblas.sgemv_rowmajor_nt(
                x_ptr, w_ptr, out_ptr, in_features=fin, out_features=fout
            )
            return
        self.rocblas.sgemm_rowmajor_nt(
            x_ptr, w_ptr, out_ptr, rows=rows, in_features=fin, out_features=fout
        )

    def _rmsnorm(self, x_ptr: int, w_ptr: int, out_ptr: int, rows: int, dim: int) -> None:
        # family convention (shared with the Surya CPU reference oracle):
        # out = x * rsqrt(mean(x^2) + eps) * (1 + w)
        err = self._k("hipengine_evie_rmsnorm_f32", [_P, _P, _P, _I, _I, _F, _S])(
            _P(x_ptr), _P(w_ptr), _P(out_ptr), _I(rows), _I(dim), _F(1e-6), _S(0)
        )
        self._check(err, "rmsnorm")

    def _add(self, x_ptr: int, y_ptr: int, out_ptr: int, n: int) -> None:
        err = self._k("hipengine_evie_add_f32", [_P, _P, _P, _I, _S])(
            _P(x_ptr), _P(y_ptr), _P(out_ptr), _I(n), _S(0)
        )
        self._check(err, "add")

    def _dev_ptr_array(self, ptrs: list[int]) -> DeviceBuffer:
        """Upload a host pointer list for rocBLAS batched GEMMs.

        Cached per unique pointer list so the A/B/C arrays of one call
        never alias (the EVIE runner made the same fix).
        """

        key = tuple(ptrs)
        cached = self._ptr_array_bufs.get(key)
        if cached is not None:
            return cached
        arr = np.array(ptrs, dtype=np.uint64)
        buf = malloc(arr.nbytes + _GEMM_PAD_BYTES)
        self._upload(buf, arr, dtype=np.uint64)
        self._ptr_array_bufs[key] = buf
        return buf

    def _h2d_i64(self, values: list[int], buf: DeviceBuffer) -> None:
        self._upload(buf, np.array(values, dtype=np.int64), dtype=np.int64)

    def _h2d_i32(self, values: list[int], buf: DeviceBuffer) -> None:
        self._upload(buf, np.array(values, dtype=np.int32), dtype=np.int32)

    def _rope_tables_device(self, positions: np.ndarray) -> tuple[DeviceBuffer, DeviceBuffer]:
        """Host-built interleaved partial mRoPE tables (shared family math)."""
        cos, sin = text_rope_tables(self.spec, positions)  # (seq, 64) each
        cos_buf = self._buf("rope_cos", cos.nbytes)
        sin_buf = self._buf("rope_sin", sin.nbytes)
        self._upload(cos_buf, cos)
        self._upload(sin_buf, sin)
        return cos_buf, sin_buf

    # -- attention ---------------------------------------------------------------

    def _attention_packed(self, q_ptr, k_ptr, v_ptr, out_ptr, tokens) -> None:
        """Causal prompt attention against the planar KV cache planes.

        q: (tokens, nq*hd); k/v caches: (nk, max_seq, hd) per-head planes.
        """
        s = self.spec
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        repeat = nq // nk
        q_row = nq * hd
        plane = self.max_seq * hd
        head_stride = tokens
        scores = self._buf("scores", nq * tokens * head_stride * 4)
        # scores tile h is the col-major (tokens x tokens) C of batch h with
        # ldc=head_stride, so tile h starts at h*tokens*head_stride (the mask
        # and softmax kernels index the same (heads, tokens, head_stride)
        # row-major layout).
        self.rocblas.sgemm_batched(
            self._dev_ptr_array([k_ptr + (h // repeat) * plane * 4 for h in range(nq)]).ptr,
            self._dev_ptr_array([q_ptr + h * hd * 4 for h in range(nq)]).ptr,
            self._dev_ptr_array([scores.ptr + h * tokens * head_stride * 4 for h in range(nq)]).ptr,
            batch=nq, m=tokens, n=tokens, k=hd,
            lda=hd, ldb=q_row, ldc=head_stride,
            trans_a=True, trans_b=False,
        )
        err = self._fn_surya("hipengine_surya_causal_mask_scale_f32",
                             [_P, _F, _I, _I, _I, _S])(
            _P(scores.ptr), _F(hd ** -0.5), _I(nq), _I(tokens), _I(head_stride), _S(0))
        self._check(err, "causal mask")
        err = self._k("hipengine_evie_softmax_rows_f32", [_P, _I, _I, _I, _I, _S])(
            _P(scores.ptr), _I(nq * tokens), _I(tokens), _I(tokens), _I(tokens * head_stride), _S(0))
        self._check(err, "softmax")
        self.rocblas.sgemm_batched(
            self._dev_ptr_array([v_ptr + (h // repeat) * plane * 4 for h in range(nq)]).ptr,
            self._dev_ptr_array([scores.ptr + h * tokens * head_stride * 4 for h in range(nq)]).ptr,
            self._dev_ptr_array([out_ptr + h * hd * 4 for h in range(nq)]).ptr,
            batch=nq, m=hd, n=tokens, k=tokens,
            lda=hd, ldb=head_stride, ldc=q_row,
            trans_a=False, trans_b=False,
        )

    def _attention_decode(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr, total) -> None:
        """Single-query attention against the persistent KV cache planes."""
        s = self.spec
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        repeat = nq // nk
        head_stride = self.max_seq
        scores = self._buf("scores", nq * head_stride * 4)
        self.rocblas.sgemm_batched(
            self._dev_ptr_array([k_cache_ptr + (h // repeat) * self.max_seq * hd * 4 for h in range(nq)]).ptr,
            self._dev_ptr_array([q_ptr + h * hd * 4 for h in range(nq)]).ptr,
            self._dev_ptr_array([scores.ptr + h * head_stride * 4 for h in range(nq)]).ptr,
            batch=nq, m=total, n=1, k=hd,
            lda=hd, ldb=hd, ldc=head_stride,
            trans_a=True, trans_b=False,
        )
        err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
            _P(scores.ptr), _P(scores.ptr), _F(hd ** -0.5), _I(nq * head_stride), _S(0))
        self._check(err, "scale scores")
        err = self._k("hipengine_evie_softmax_rows_f32", [_P, _I, _I, _I, _I, _S])(
            _P(scores.ptr), _I(nq), _I(total), _I(1), _I(head_stride), _S(0))
        self._check(err, "softmax decode")
        self.rocblas.sgemm_batched(
            self._dev_ptr_array([v_cache_ptr + (h // repeat) * self.max_seq * hd * 4 for h in range(nq)]).ptr,
            self._dev_ptr_array([scores.ptr + h * head_stride * 4 for h in range(nq)]).ptr,
            self._dev_ptr_array([out_ptr + h * hd * 4 for h in range(nq)]).ptr,
            batch=nq, m=hd, n=1, k=total,
            lda=hd, ldb=head_stride, ldc=nq * hd,
            trans_a=False, trans_b=False,
        )

    def _fn_surya(self, symbol: str, argtypes: list) -> ctypes._FuncPtr:
        fn = getattr(self.surya_library, symbol, None)
        if fn is None:
            raise SuryaGpuRuntimeError(f"missing symbol {symbol}")
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn

    # -- layers -------------------------------------------------------------------

    def _gdn_layer_prefill(self, layer: int, norm_ptr: int, out_ptr: int, tokens: int) -> None:
        s = self.spec
        h = s.hidden_size
        nv, hk, hv = s.gdn_num_value_heads, s.gdn_key_head_dim, s.gdn_value_head_dim
        hkv = s.gdn_num_key_heads
        channels = 3 * nv * hv
        p = f"model.language_model.layers.{layer}.linear_attn."
        qkv = self._buf("gdn_qkv", tokens * channels * 4)
        z = self._buf("gdn_z", tokens * nv * hv * 4)
        b_in = self._buf("gdn_b", tokens * nv * 4)
        a_in = self._buf("gdn_a", tokens * nv * 4)
        conv_out = self._buf("gdn_conv", tokens * channels * 4)
        q = self._buf("gdn_q", tokens * nv * hk * 4)
        k = self._buf("gdn_k", tokens * nv * hk * 4)
        v = self._buf("gdn_v", tokens * nv * hv * 4)
        beta = self._buf("gdn_beta", tokens * nv * 4)
        decay = self._buf("gdn_decay", tokens * nv * 4)
        gdn_out = self._buf("gdn_out", tokens * nv * hv * 4)
        gdn_normed = self._buf("gdn_normed", tokens * nv * hv * 4)

        self._gemm(norm_ptr, self._w[p + "in_proj_qkv.weight"].ptr, qkv.ptr, tokens, h, channels)
        self._gemm(norm_ptr, self._w[p + "in_proj_z.weight"].ptr, z.ptr, tokens, h, nv * hv)
        self._gemm(norm_ptr, self._w[p + "in_proj_b.weight"].ptr, b_in.ptr, tokens, h, nv)
        self._gemm(norm_ptr, self._w[p + "in_proj_a.weight"].ptr, a_in.ptr, tokens, h, nv)

        conv_state = self._conv_state[layer]
        # zero the state slot, then the segment-aware prefill writes the
        # final (channels, k) window into slot 0 for the decode step
        err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
            _P(conv_state.ptr), _P(conv_state.ptr), _F(0.0),
            _I(channels * s.gdn_conv_kernel), _S(0))
        self._check(err, "zero conv state")
        # cu_seqlens is int32 in the conv kernel ABI; state_indices is int64
        self._h2d_i32([0, tokens], self._cu_buf)
        self._h2d_i64([0], self._state_idx_buf)
        qwen35_linear_attn_conv_prefill_segments_f32(
            qkv.ptr, conv_state.ptr, self._w[p + "conv1d.weight"].ptr, conv_out.ptr,
            self._cu_buf.ptr, self._state_idx_buf.ptr, tokens, 1, channels,
            s.gdn_conv_kernel, stream=0, library=self.conv_library, runtime=self.runtime,
        )
        # q/K l2-normalization with the 1/sqrt(d_k) query scale applied on
        # the input (plain kernel recurrence consumes q as-is); plain
        # (tokens, heads, dim) output rows
        surya_gdn_l2norm_f32(conv_out.ptr, q.ptr, k.ptr,
                             1.0 / math.sqrt(hk), tokens, hkv, hk, channels,
                             nv * hv, library=self.surya_library,
                             runtime=self.runtime)
        err = self._k("hipengine_evie_gdn_gates_f32", [_P, _P, _P, _P, _P, _P, _I, _I, _S])(
            _P(b_in.ptr), _P(a_in.ptr), _P(self._w[p + "A_log"].ptr),
            _P(self._w[p + "dt_bias"].ptr), _P(beta.ptr), _P(decay.ptr),
            _I(tokens), _I(nv), _S(0))
        self._check(err, "gdn gates")
        # v plane sits at offset 2*key_dim in each conv row (q|k|v)
        err = self._k("hipengine_evie_expand_heads_f32",
                      [_P, _P, _I, _I, _I, _I, _I, _I, _S])(
            _P(conv_out.ptr), _P(v.ptr), _I(tokens), _I(2 * hkv * hk), _I(channels),
            _I(nv), _I(hv), _I(1), _S(0))
        self._check(err, "gdn v expand")
        gdn_state = self._gdn_state[layer]
        err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
            _P(gdn_state.ptr), _P(gdn_state.ptr), _F(0.0),
            _I(nv * hk * hv), _S(0))
        self._check(err, "zero gdn state")
        qwen35_gdn_prefill_recurrent_f32(
            q.ptr, k.ptr, v.ptr, beta.ptr, decay.ptr, gdn_state.ptr, gdn_out.ptr,
            tokens, nv, hk, hv, stream=0, library=self.gdn_library, runtime=self.runtime,
        )
        err = self._k("hipengine_evie_gdn_rmsnorm_gate_f32",
                      [_P, _P, _P, _P, _I, _I, _F, _S])(
            _P(gdn_out.ptr), _P(z.ptr), _P(self._w[p + "norm.weight"].ptr),
            _P(gdn_normed.ptr), _I(tokens * nv), _I(hv), _F(1e-6), _S(0))
        self._check(err, "gdn rmsnorm gate")
        self._gemm(gdn_normed.ptr, self._w[p + "out_proj.weight"].ptr, out_ptr, tokens, nv * hv, h)

    def _gdn_layer_decode(self, layer: int, norm_ptr: int, out_ptr: int) -> None:
        s = self.spec
        h = s.hidden_size
        nv, hk, hv = s.gdn_num_value_heads, s.gdn_key_head_dim, s.gdn_value_head_dim
        hkv = s.gdn_num_key_heads
        channels = 3 * nv * hv
        p = f"model.language_model.layers.{layer}.linear_attn."
        qkv = self._buf("dec_qkv", channels * 4)
        z = self._buf("dec_z", nv * hv * 4)
        b_in = self._buf("dec_b", nv * 4)
        a_in = self._buf("dec_a", nv * 4)
        conv_out = self._buf("dec_conv", channels * 4)
        q = self._buf("dec_q", nv * hk * 4)
        k = self._buf("dec_k", nv * hk * 4)
        v = self._buf("dec_v", nv * hv * 4)
        beta = self._buf("dec_beta", nv * 4)
        decay = self._buf("dec_decay", nv * 4)
        gdn_out = self._buf("dec_gdn_out", nv * hv * 4)
        gdn_normed = self._buf("dec_gdn_normed", nv * hv * 4)

        self._gemm(norm_ptr, self._w[p + "in_proj_qkv.weight"].ptr, qkv.ptr, 1, h, channels)
        self._gemm(norm_ptr, self._w[p + "in_proj_z.weight"].ptr, z.ptr, 1, h, nv * hv)
        self._gemm(norm_ptr, self._w[p + "in_proj_b.weight"].ptr, b_in.ptr, 1, h, nv)
        self._gemm(norm_ptr, self._w[p + "in_proj_a.weight"].ptr, a_in.ptr, 1, h, nv)
        qwen35_linear_attn_conv_decode_f32(
            qkv.ptr, self._conv_state[layer].ptr, self._w[p + "conv1d.weight"].ptr,
            conv_out.ptr, channels, s.gdn_conv_kernel,
            stream=0, library=self.conv_library, runtime=self.runtime,
        )
        err = self._k("hipengine_evie_gdn_l2norm_scale_f32",
                      [_P, _P, _P, _P, _F, _I, _I, _S])(
            _P(conv_out.ptr), _P(conv_out.ptr + nv * hv * 4), _P(q.ptr), _P(k.ptr),
            _F(1.0 / math.sqrt(hk)), _I(nv), _I(hk), _S(0))
        self._check(err, "gdn l2norm decode")
        err = self._k("hipengine_evie_gdn_gates_f32", [_P, _P, _P, _P, _P, _P, _I, _I, _S])(
            _P(b_in.ptr), _P(a_in.ptr), _P(self._w[p + "A_log"].ptr),
            _P(self._w[p + "dt_bias"].ptr), _P(beta.ptr), _P(decay.ptr),
            _I(1), _I(nv), _S(0))
        self._check(err, "gdn gates decode")
        err = self._k("hipengine_evie_expand_heads_f32",
                      [_P, _P, _I, _I, _I, _I, _I, _I, _S])(
            _P(conv_out.ptr), _P(v.ptr), _I(1), _I(2 * hkv * hk), _I(channels),
            _I(nv), _I(hv), _I(1), _S(0))
        self._check(err, "gdn v expand decode")
        qwen35_gdn_prefill_recurrent_f32(
            q.ptr, k.ptr, v.ptr, beta.ptr, decay.ptr, self._gdn_state[layer].ptr,
            gdn_out.ptr, 1, nv, hk, hv, stream=0, library=self.gdn_library,
            runtime=self.runtime,
        )
        err = self._k("hipengine_evie_gdn_rmsnorm_gate_f32",
                      [_P, _P, _P, _P, _I, _I, _F, _S])(
            _P(gdn_out.ptr), _P(z.ptr), _P(self._w[p + "norm.weight"].ptr),
            _P(gdn_normed.ptr), _I(nv), _I(hv), _F(1e-6), _S(0))
        self._check(err, "gdn rmsnorm gate decode")
        self._gemm(gdn_normed.ptr, self._w[p + "out_proj.weight"].ptr, out_ptr, 1, nv * hv, h)

    def _attn_layer_prefill(self, layer: int, norm_ptr: int, out_ptr: int, tokens: int,
                            cos_buf: DeviceBuffer, sin_buf: DeviceBuffer) -> None:
        s = self.spec
        h = s.hidden_size
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        p = f"model.language_model.layers.{layer}.self_attn."
        qp = self._buf("attn_qp", tokens * nq * hd * 2 * 4)
        q = self._buf("attn_q", tokens * nq * hd * 4)
        gate = self._buf("attn_gate", tokens * nq * hd * 4)
        k = self._buf("attn_k", tokens * nk * hd * 4)
        v = self._buf("attn_v", tokens * nk * hd * 4)
        heads_out = self._buf("attn_heads_out", tokens * nq * hd * 4)

        self._gemm(norm_ptr, self._w[p + "q_proj.weight"].ptr, qp.ptr, tokens, h, nq * hd * 2)
        self._gemm(norm_ptr, self._w[p + "k_proj.weight"].ptr, k.ptr, tokens, h, nk * hd)
        self._gemm(norm_ptr, self._w[p + "v_proj.weight"].ptr, v.ptr, tokens, h, nk * hd)
        err = self._fn_surya("hipengine_surya_split_qgate_f32", [_P, _P, _P, _I, _I, _I, _S])(
            _P(qp.ptr), _P(q.ptr), _P(gate.ptr), _I(tokens), _I(nq), _I(hd), _S(0))
        self._check(err, "split qgate")
        self._rmsnorm(q.ptr, self._w[p + "q_norm.weight"].ptr, q.ptr, tokens * nq, hd)
        self._rmsnorm(k.ptr, self._w[p + "k_norm.weight"].ptr, k.ptr, tokens * nk, hd)
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(q.ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(tokens), _I(nq),
            _I(hd), _I(int(hd * s.partial_rotary_factor)), _I(nq * hd), _S(0))
        self._check(err, "rope q")
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(k.ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(tokens), _I(nk),
            _I(hd), _I(int(hd * s.partial_rotary_factor)), _I(nk * hd), _S(0))
        self._check(err, "rope k")
        # write k/v into the persistent cache planes, then attend causally
        k_cache, v_cache = self._kv_cache[layer]
        surya_scatter_kv_f32(k.ptr, k_cache.ptr, tokens, 0, nk, hd, self.max_seq,
                             library=self.surya_library, runtime=self.runtime)
        surya_scatter_kv_f32(v.ptr, v_cache.ptr, tokens, 0, nk, hd, self.max_seq,
                             library=self.surya_library, runtime=self.runtime)
        self._attention_packed(q.ptr, k_cache.ptr, v_cache.ptr, heads_out.ptr, tokens)
        err = self._k("hipengine_evie_sigmoid_mul_f32", [_P, _P, _I, _S])(
            _P(gate.ptr), _P(heads_out.ptr), _I(tokens * nq * hd), _S(0))
        self._check(err, "attn gate")
        self._gemm(heads_out.ptr, self._w[p + "o_proj.weight"].ptr, out_ptr, tokens, nq * hd, h)

    def _attn_layer_decode(self, layer: int, norm_ptr: int, out_ptr: int,
                           cos_buf: DeviceBuffer, sin_buf: DeviceBuffer) -> None:
        s = self.spec
        h = s.hidden_size
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        p = f"model.language_model.layers.{layer}.self_attn."
        qp = self._buf("dec_attn_qp", nq * hd * 2 * 4)
        q = self._buf("dec_attn_q", nq * hd * 4)
        gate = self._buf("dec_attn_gate", nq * hd * 4)
        k = self._buf("dec_attn_k", nk * hd * 4)
        v = self._buf("dec_attn_v", nk * hd * 4)
        heads_out = self._buf("dec_attn_out", nq * hd * 4)

        self._gemm(norm_ptr, self._w[p + "q_proj.weight"].ptr, qp.ptr, 1, h, nq * hd * 2)
        self._gemm(norm_ptr, self._w[p + "k_proj.weight"].ptr, k.ptr, 1, h, nk * hd)
        self._gemm(norm_ptr, self._w[p + "v_proj.weight"].ptr, v.ptr, 1, h, nk * hd)
        err = self._fn_surya("hipengine_surya_split_qgate_f32", [_P, _P, _P, _I, _I, _I, _S])(
            _P(qp.ptr), _P(q.ptr), _P(gate.ptr), _I(1), _I(nq), _I(hd), _S(0))
        self._check(err, "split qgate decode")
        self._rmsnorm(q.ptr, self._w[p + "q_norm.weight"].ptr, q.ptr, nq, hd)
        self._rmsnorm(k.ptr, self._w[p + "k_norm.weight"].ptr, k.ptr, nk, hd)
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(q.ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(1), _I(nq),
            _I(hd), _I(int(hd * s.partial_rotary_factor)), _I(nq * hd), _S(0))
        self._check(err, "rope q decode")
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(k.ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(1), _I(nk),
            _I(hd), _I(int(hd * s.partial_rotary_factor)), _I(nk * hd), _S(0))
        self._check(err, "rope k decode")
        k_cache, v_cache = self._kv_cache[layer]
        pos = self._seq_len
        surya_scatter_kv_f32(k.ptr, k_cache.ptr, 1, pos, nk, hd, self.max_seq,
                             library=self.surya_library, runtime=self.runtime)
        surya_scatter_kv_f32(v.ptr, v_cache.ptr, 1, pos, nk, hd, self.max_seq,
                             library=self.surya_library, runtime=self.runtime)
        self._attention_decode(q.ptr, k_cache.ptr, v_cache.ptr, heads_out.ptr, pos + 1)
        err = self._k("hipengine_evie_sigmoid_mul_f32", [_P, _P, _I, _S])(
            _P(gate.ptr), _P(heads_out.ptr), _I(nq * hd), _S(0))
        self._check(err, "attn gate decode")
        self._gemm(heads_out.ptr, self._w[p + "o_proj.weight"].ptr, out_ptr, 1, nq * hd, h)

    def _mlp(self, prefix: str, norm_ptr: int, out_ptr: int, tokens: int) -> None:
        s = self.spec
        inter = s.intermediate_size
        gate_p = self._buf("mlp_gate", tokens * inter * 4)
        up_p = self._buf("mlp_up", tokens * inter * 4)
        self._gemm(norm_ptr, self._w[prefix + "mlp.gate_proj.weight"].ptr, gate_p.ptr, tokens, s.hidden_size, inter)
        self._gemm(norm_ptr, self._w[prefix + "mlp.up_proj.weight"].ptr, up_p.ptr, tokens, s.hidden_size, inter)
        err = self._k("hipengine_evie_silu_mul_f32", [_P, _P, _I, _S])(
            _P(gate_p.ptr), _P(up_p.ptr), _I(tokens * inter), _S(0))
        self._check(err, "swiglu")
        self._gemm(gate_p.ptr, self._w[prefix + "mlp.down_proj.weight"].ptr, out_ptr, tokens, inter, s.hidden_size)

    # -- public API -----------------------------------------------------------------

    def _embed(self, input_ids: np.ndarray, visual_features: np.ndarray | None) -> DeviceBuffer:
        s = self.spec
        tokens = len(input_ids)
        ids = np.ascontiguousarray(input_ids, dtype=np.int64)
        ids_buf = self._buf("ids", ids.nbytes)
        self._upload(ids_buf, ids, dtype=np.int64)
        x = self._buf("x", tokens * s.hidden_size * 4)
        visual_ptr = 0
        image_token_id = -1
        if visual_features is not None:
            vf = np.ascontiguousarray(visual_features, dtype=np.float32)
            n_img = int((np.asarray(input_ids) == s.image_token_id).sum())
            if vf.ndim != 2 or vf.shape[0] != n_img or vf.shape[1] != s.hidden_size:
                raise SuryaGpuRuntimeError(
                    f"visual_features must be (n_image_tokens, hidden) = "
                    f"({n_img}, {s.hidden_size}); got {vf.shape}")
            if self._visual_buf is None or self._visual_buf.nbytes < vf.nbytes + _GEMM_PAD_BYTES:
                if self._visual_buf is not None:
                    hip_free(self._visual_buf)
                self._visual_buf = malloc(vf.nbytes + _GEMM_PAD_BYTES)
            self._upload(self._visual_buf, vf)
            visual_ptr = self._visual_buf.ptr
            image_token_id = s.image_token_id
        err = self._k("hipengine_evie_embed_lookup_f32", [_P, _P, _P, _P, _I, _I, _I, _S])(
            _P(ids_buf.ptr), _P(self._w["model.language_model.embed_tokens.weight"].ptr),
            _P(visual_ptr), _P(x.ptr), _I(tokens), _I(s.hidden_size), _I(image_token_id), _S(0))
        self._check(err, "embed lookup")
        return x

    def _decode_stack(self, x_ptr: int, tokens: int, cos_buf: DeviceBuffer,
                      sin_buf: DeviceBuffer, *, decode: bool) -> None:
        s = self.spec
        h = s.hidden_size
        norm = self._buf("norm", tokens * h * 4)
        attn_out = self._buf("attn_out", tokens * h * 4)
        for layer in range(s.num_layers):
            lp = f"model.language_model.layers.{layer}."
            self._rmsnorm(x_ptr, self._w[lp + "input_layernorm.weight"].ptr, norm.ptr, tokens, h)
            if s.is_full_attention(layer):
                if decode:
                    self._attn_layer_decode(layer, norm.ptr, attn_out.ptr, cos_buf, sin_buf)
                else:
                    self._attn_layer_prefill(layer, norm.ptr, attn_out.ptr, tokens, cos_buf, sin_buf)
            else:
                if decode:
                    self._gdn_layer_decode(layer, norm.ptr, attn_out.ptr)
                else:
                    self._gdn_layer_prefill(layer, norm.ptr, attn_out.ptr, tokens)
            self._add(x_ptr, attn_out.ptr, x_ptr, tokens * h)
            self._rmsnorm(x_ptr, self._w[lp + "post_attention_layernorm.weight"].ptr, norm.ptr, tokens, h)
            self._mlp(lp, norm.ptr, attn_out.ptr, tokens)
            self._add(x_ptr, attn_out.ptr, x_ptr, tokens * h)
        self._rmsnorm(x_ptr, self._w["model.language_model.norm.weight"].ptr, norm.ptr, tokens, h)
        self._final_norm_ptr = norm.ptr
        # row of the final-norm buffer the lm_head must read (last token)
        self._final_norm_row = norm.ptr + (tokens - 1) * h * 4

    def debug_read(self, ptr: int, n: int) -> np.ndarray:
        # sync first: on this stack a D2H hipMemcpy can complete its call
        # before the producing kernel does, returning stale destination bytes
        self.runtime.device_synchronize()
        out = np.empty(n, dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), _raw_buffer(ptr, n * 4))
        return out

    # -- vision tower -------------------------------------------------------------

    def vision_forward(self, pixel_rows: np.ndarray, grid_thw) -> np.ndarray:
        """Surya vision tower on the HIP device; returns merged features.

        Mirrors ``kernels.cpu_reference.surya.vision_forward``: patch embed
        (conv-as-matmul) + bias, host bilinear position embed, full-dim
        half-split 2-axis rotary, bidirectional packed attention, tanh-GELU
        MLP blocks, then the merger (LayerNorm -> square fc1 -> erf GELU ->
        fc2). Preprocessing and the small pos-embed/rotary tables stay on
        the host; every tensor op runs on the device. Returns the merged
        features (n / merge^2, vision_out_hidden_size) as host fp32.
        """
        from hipengine.kernels.cpu_reference.surya import (
            _merge_block_major_coords,
            _pixel_rows_to_patches,
            vision_pos_embed,
            vision_rotary,
        )

        # Admit before any device work: the attention scratch is quadratic in
        # patch count, so an over-budget page must be rejected here, not after
        # patch embed and a failed multi-GB allocation.
        self.check_vision_capacity(grid_thw)

        s = self.spec
        vh = s.vision_hidden_size
        nh, hd = s.vision_num_heads, s.vision_head_dim()
        merge = s.vision_spatial_merge_size
        inter = s.vision_intermediate_size

        patches = _pixel_rows_to_patches(pixel_rows, s)
        n = patches.shape[0]
        n_merged = n // (merge * merge)
        vis_inter = merge * merge * vh

        x = self._buf("vis_x", n * vh * 4)
        norm = self._buf("vis_norm", n * vh * 4)
        qkv = self._buf("vis_qkv", n * 3 * vh * 4)
        attn = self._buf("vis_attn", n * vh * 4)
        out = self._buf("vis_out", n * vh * 4)
        mlp = self._buf("vis_mlp", n * inter * 4)
        merged_in = self._buf("vis_merged_in", n_merged * vis_inter * 4)
        merged = self._buf("vis_merged", n_merged * s.vision_out_hidden_size * 4)
        scores, head_stride = self._vis_scores(n, nh)

        # patch embed: (n, ch*t*p*p) @ (ch*t*p*p, vh) + bias (separate C —
        # never alias the GEMM input)
        patches_flat = patches.reshape(n, -1)
        patches_buf = self._buf("vis_patches", patches_flat.nbytes)
        self._upload(patches_buf, patches_flat)
        self._gemm(patches_buf.ptr, self._w["model.visual.patch_embed.proj.weight"].ptr,
                   x.ptr, n, patches_flat.shape[1], vh)
        self._add_bias(x.ptr, self._w["model.visual.patch_embed.proj.bias"].ptr,
                       n * vh, vh)

        # position embed: host bilinear resample of the learned table
        coords = _merge_block_major_coords(grid_thw, merge)[:2]
        pos = vision_pos_embed(
            {"model.visual.pos_embed.weight": self._pos_embed_table()},
            s, list(grid_thw), coords,
        )
        self._upload(out, pos)
        self._add(x.ptr, out.ptr, x.ptr, n * vh)

        cos, sin = vision_rotary(s, *coords)
        cos_buf = self._buf("vis_cos", cos.nbytes)
        sin_buf = self._buf("vis_sin", sin.nbytes)
        self._upload(cos_buf, cos)
        self._upload(sin_buf, sin)

        scale = hd ** -0.5
        for i in range(s.vision_depth):
            p = f"model.visual.blocks.{i}."
            self._layernorm(x.ptr, self._w[p + "norm1.weight"].ptr,
                            self._w[p + "norm1.bias"].ptr, norm.ptr, n, vh)
            self._gemm(norm.ptr, self._w[p + "attn.qkv.weight"].ptr, qkv.ptr,
                       n, vh, 3 * vh)
            self._add_bias(qkv.ptr, self._w[p + "attn.qkv.bias"].ptr,
                           n * 3 * vh, 3 * vh)
            # full-dim half-split rotary on the q and k planes of packed qkv
            self._rope(qkv.ptr, cos_buf.ptr, sin_buf.ptr, n, nh, hd, hd, 3 * vh)
            self._rope(qkv.ptr + vh * 4, cos_buf.ptr, sin_buf.ptr, n, nh, hd, hd, 3 * vh)
            self._vision_attention_packed(
                qkv.ptr, qkv.ptr + vh * 4, qkv.ptr + 2 * vh * 4,
                attn.ptr, n, nh, hd, 3 * vh, scores.ptr, head_stride, scale)
            self._gemm(attn.ptr, self._w[p + "attn.proj.weight"].ptr, out.ptr,
                       n, vh, vh)
            self._add_bias(out.ptr, self._w[p + "attn.proj.bias"].ptr, n * vh, vh)
            self._add(x.ptr, out.ptr, x.ptr, n * vh)

            self._layernorm(x.ptr, self._w[p + "norm2.weight"].ptr,
                            self._w[p + "norm2.bias"].ptr, norm.ptr, n, vh)
            self._gemm(norm.ptr, self._w[p + "mlp.linear_fc1.weight"].ptr,
                       mlp.ptr, n, vh, inter)
            self._add_bias(mlp.ptr, self._w[p + "mlp.linear_fc1.bias"].ptr,
                           n * inter, inter)
            self._gelu_tanh(mlp.ptr, mlp.ptr, n * inter)
            self._gemm(mlp.ptr, self._w[p + "mlp.linear_fc2.weight"].ptr,
                       out.ptr, n, inter, vh)
            self._add_bias(out.ptr, self._w[p + "mlp.linear_fc2.bias"].ptr,
                           n * vh, vh)
            self._add(x.ptr, out.ptr, x.ptr, n * vh)

        # merger: LayerNorm -> (n/4, merge^2*vh) -> fc1 -> erf GELU -> fc2
        self._layernorm(x.ptr, self._w["model.visual.merger.norm.weight"].ptr,
                        self._w["model.visual.merger.norm.bias"].ptr,
                        norm.ptr, n, vh)
        self._gemm(norm.ptr, self._w["model.visual.merger.linear_fc1.weight"].ptr,
                   merged_in.ptr, n_merged, vis_inter, vis_inter)
        self._add_bias(merged_in.ptr,
                       self._w["model.visual.merger.linear_fc1.bias"].ptr,
                       n_merged * vis_inter, vis_inter)
        self._gelu_erf(merged_in.ptr, merged_in.ptr, n_merged * vis_inter)
        self._gemm(merged_in.ptr,
                   self._w["model.visual.merger.linear_fc2.weight"].ptr,
                   merged.ptr, n_merged, vis_inter, s.vision_out_hidden_size)
        self._add_bias(merged.ptr,
                       self._w["model.visual.merger.linear_fc2.bias"].ptr,
                       n_merged * s.vision_out_hidden_size,
                       s.vision_out_hidden_size)

        result = np.empty(n_merged * s.vision_out_hidden_size, dtype=np.float32)
        self.runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(result),
                            _raw_buffer(merged.ptr, result.nbytes))
        return result.reshape(n_merged, s.vision_out_hidden_size)

    def _pos_embed_table(self) -> np.ndarray:
        if getattr(self, "_pos_table_host", None) is None:
            pe = self.spec.vision_num_position_embeddings
            vh = self.spec.vision_hidden_size
            self._pos_table_host = self.debug_read(
                self._w["model.visual.pos_embed.weight"].ptr, pe * vh
            ).reshape(pe, vh)
        return self._pos_table_host

    @staticmethod
    def _vis_score_stride(n: int) -> int:
        """Per-head score-tile stride in elements, 16-byte aligned."""

        return (n * n + 3) & ~3

    def _vis_scores(self, n: int, heads: int) -> tuple[DeviceBuffer, int]:
        stride = self._vis_score_stride(n)
        return self._buf("vis_scores", heads * stride * 4), stride

    # -- vision admission --------------------------------------------------

    def vision_scratch_bytes(self, grid_thw) -> int:
        """Device scratch the vision attention scores need for ``grid_thw``.

        Quadratic in patch count. Derived from the same stride helper as
        ``_vis_scores`` so admission can never disagree with the allocation.
        """

        n = int(grid_thw[0][1]) * int(grid_thw[0][2])
        return self.spec.vision_num_heads * self._vis_score_stride(n) * 4

    def check_vision_capacity(self, grid_thw) -> None:
        """Admit a vision grid before any device work or allocation runs.

        Two independent checks, because they fail for different reasons:

        - the configured cap bounds a single request regardless of how much
          memory the host happens to have;
        - free device memory catches the case where the weights plus other
          live runners have already consumed the budget.

        Called before patch embed so an over-budget page costs nothing.
        """

        n = int(grid_thw[0][1]) * int(grid_thw[0][2])
        need = self.vision_scratch_bytes(grid_thw)
        grid = (int(grid_thw[0][0]), int(grid_thw[0][1]), int(grid_thw[0][2]))
        cap = self.max_vision_scratch_bytes
        if cap is not None and need > cap:
            raise SuryaGpuRuntimeError(
                f"vision attention scratch for grid {grid} is {need / 1e9:.2f} GB "
                f"({n} patches), above the {cap / 1e9:.2f} GB cap; reduce the page "
                f"resolution or construct SuryaGpuRunner(max_vision_scratch_bytes=...)"
            )
        # the scratch buffer is cached per key and reused when it is already big
        # enough, so only the growth is charged against free memory
        existing = self._scratch.get("vis_scores")
        growth = max(0, need + _GEMM_PAD_BYTES - (existing.nbytes if existing else 0))
        if growth == 0:
            return
        try:
            free_bytes, _total = self.runtime.mem_get_info()
        except Exception:  # runtime without mem_get_info: cap is the only bound
            return
        if growth > int(free_bytes):
            raise SuryaGpuRuntimeError(
                f"vision attention scratch for grid {grid} needs {growth / 1e9:.2f} GB "
                f"more device memory but only {int(free_bytes) / 1e9:.2f} GB is free"
            )

    def _add_bias(self, x_ptr: int, bias_ptr: int, n: int, row: int) -> None:
        err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
            _P(x_ptr), _P(bias_ptr), _I(n), _I(row), _S(0))
        self._check(err, "add bias")

    def _layernorm(self, x_ptr: int, w_ptr: int, b_ptr: int, out_ptr: int,
                   rows: int, dim: int) -> None:
        err = self._k("hipengine_evie_layernorm_f32",
                      [_P, _P, _P, _P, _I, _I, _F, _S])(
            _P(x_ptr), _P(w_ptr), _P(b_ptr), _P(out_ptr), _I(rows), _I(dim),
            _F(1e-6), _S(0))
        self._check(err, "layernorm")

    def _gelu_tanh(self, x_ptr: int, out_ptr: int, n: int) -> None:
        err = self._k("hipengine_evie_gelu_tanh_f32", [_P, _P, _I, _S])(
            _P(x_ptr), _P(out_ptr), _I(n), _S(0))
        self._check(err, "gelu tanh")

    def _gelu_erf(self, x_ptr: int, out_ptr: int, n: int) -> None:
        err = self._k("hipengine_evie_gelu_erf_f32", [_P, _P, _I, _S])(
            _P(x_ptr), _P(out_ptr), _I(n), _S(0))
        self._check(err, "gelu erf")

    def _rope(self, x_ptr: int, cos_ptr: int, sin_ptr: int, tokens: int,
              heads: int, head_dim: int, rotary_dim: int, row_stride: int) -> None:
        err = self._k("hipengine_evie_rope_f32",
                      [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(x_ptr), _P(cos_ptr), _P(sin_ptr), _I(tokens), _I(heads),
            _I(head_dim), _I(rotary_dim), _I(row_stride), _S(0))
        self._check(err, "rope")

    def _vision_attention_packed(self, q_ptr: int, k_ptr: int, v_ptr: int,
                                 out_ptr: int, tokens: int, heads: int,
                                 head_dim: int, row_stride: int,
                                 scores_ptr: int, head_stride: int,
                                 scale: float) -> None:
        """Bidirectional attention over head planes embedded in packed rows.

        scores tile h is the col-major (tokens x tokens) C of batch h with
        ldc=tokens, so tile h starts at h*head_stride (== h*tokens^2 for the
        aligned stride) — the same layout the softmax kernel indexes.
        """
        self.rocblas.sgemm_strided_batched(
            k_ptr, q_ptr, scores_ptr,
            m=tokens, n=tokens, k=head_dim,
            lda=row_stride, ldb=row_stride, ldc=tokens,
            stride_a=head_dim, stride_b=head_dim, stride_c=head_stride,
            batch=heads, trans_a=True, trans_b=False,
        )
        err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
            _P(scores_ptr), _P(scores_ptr), _F(scale),
            _I(heads * head_stride), _S(0))
        self._check(err, "vision score scale")
        err = self._k("hipengine_evie_softmax_rows_f32", [_P, _I, _I, _I, _I, _S])(
            _P(scores_ptr), _I(heads * tokens), _I(tokens), _I(tokens),
            _I(head_stride), _S(0))
        self._check(err, "vision softmax")
        self.rocblas.sgemm_strided_batched(
            v_ptr, scores_ptr, out_ptr,
            m=head_dim, n=tokens, k=tokens,
            lda=row_stride, ldb=tokens, ldc=heads * head_dim,
            stride_a=head_dim, stride_b=head_stride, stride_c=head_dim,
            batch=heads, trans_a=False, trans_b=False,
        )

    def _logits_last(self) -> np.ndarray:
        s = self.spec
        logits = self._buf("logits", s.vocab_size * 4)
        self._gemm(self._final_norm_row, self._w["model.language_model.embed_tokens.weight"].ptr,
                   logits.ptr, 1, s.hidden_size, s.vocab_size)
        self.runtime.device_synchronize()
        out = np.empty(s.vocab_size, dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), _raw_buffer(logits.ptr, s.vocab_size * 4))
        return out

    def prefill(self, input_ids, positions, visual_features=None) -> np.ndarray:
        """Run the prompt; returns last-token logits and leaves device state
        (conv windows, GDN recurrence states, KV caches) ready for decode."""
        ids = np.asarray(input_ids).reshape(-1)
        pos = np.asarray(positions)
        if pos.shape[0] == 3 and pos.ndim == 3:  # (b, 3, s) fixture layout
            pos = pos[0]
        if pos.ndim != 2 or pos.shape[0] != 3:
            raise ValueError(f"positions must be (3, s); got {pos.shape}")
        self._seq_len = len(ids)
        x = self._embed(ids, visual_features)
        cos_buf, sin_buf = self._rope_tables_device(np.ascontiguousarray(pos, dtype=np.int64))
        self._decode_stack(x.ptr, len(ids), cos_buf, sin_buf, decode=False)
        return self._logits_last()

    def decode_step(self, token_id: int, position: int) -> np.ndarray:
        """Advance one token; position is the absolute rope position."""
        x = self._embed(np.array([token_id]), None)
        cos_buf, sin_buf = self._rope_tables_device(
            np.array([[position], [position], [position]], dtype=np.int64))
        # ``_seq_len`` is the next free KV slot: ``_attn_layer_decode`` scatters
        # into that slot and attends over ``_seq_len + 1`` positions. Advance it
        # only after the step, otherwise the scatter lands one slot too high and
        # attention reads a slot this request never wrote -- stale KV from any
        # earlier, longer request on the same runner.
        self._decode_stack(x.ptr, 1, cos_buf, sin_buf, decode=True)
        self._seq_len += 1
        return self._logits_last()

    def generate(self, input_ids, positions, visual_features=None,
                 max_new_tokens: int = 64, eos_token_id: int = 2) -> list[int]:
        """Greedy generation; mirrors the CPU reference generate loop."""
        logits = self.prefill(input_ids, positions, visual_features)
        generated: list[int] = []
        pos = int(np.asarray(positions).reshape(3, -1)[:, -1].max()) if np.asarray(positions).ndim >= 2 else 0
        for step in range(max_new_tokens):
            nxt = int(np.argmax(logits))
            if nxt == eos_token_id:
                break
            generated.append(nxt)
            logits = self.decode_step(nxt, pos + 1 + step)
        return generated

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for buf in self._permanent_bufs:
            hip_free(buf)
        self._permanent_bufs.clear()
        for buf in self._ptr_array_bufs.values():
            hip_free(buf)
        self._ptr_array_bufs.clear()
        for buf in self._scratch.values():
            hip_free(buf)
        self._scratch.clear()
        if self._visual_buf is not None:
            hip_free(self._visual_buf)
            self._visual_buf = None
        for buf in self._w.values():
            hip_free(buf)
        self._w.clear()
        self._conv_state.clear()
        self._gdn_state.clear()
        self._kv_cache.clear()
