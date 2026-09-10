"""Torch-free EVIE-4.5B retrieval-encoder HIP runtime (fp32 strict path).

Runs the full ColQwen3_5 pipeline on the HIP device with rocBLAS SGEMM
projections and custom fp32 kernels, mirroring the CPU reference
(``hipengine.kernels.cpu_reference.evie``) stage for stage:

- vision tower: Conv3d patch embed as a dense GEMM over patch rows,
  host-precomputed learned position embeddings (bilinear table resample in
  spatial-merge-block order), 24 LayerNorm blocks with bidirectional
  per-image attention (head-plane gathers + sgemm) and tanh-GELU MLPs, and
  the spatial-merge merger.
- text stack: embedding lookup with visual-feature injection, 24
  gated-DeltaNet layers (conv1d prefill kernel + gate prep + recurrent
  delta-rule prefill kernel + RMSNormGated), 8 bidirectional full-attention
  layers with per-head RMSNorm, fused q/gate split, and interleaved partial
  mRoPE (tables built on the host), SiLU MLPs, and the final RMSNorm.
- Prefix-MRL head: projection GEMM, head-dim slice, per-token L2 norm.

This is the correctness-first runtime; the strict gate is exactness against
the torch fp32 oracle fixture, not speed.
"""

from __future__ import annotations

import ctypes
import os
import math
from dataclasses import dataclass, field
from typing import Any

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
from hipengine.kernels.hip_gfx1100.evie.evie_ops import build_evie_ops
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    build_qwen35_linear_attn_conv,
    qwen35_linear_attn_conv_prefill_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_recurrent_normalized_cluster8_f32,
)
from hipengine.loading.evie import EvieLoadedModel
from hipengine.models.evie import EvieModelSpec

_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p


class EvieRuntimeError(RuntimeError):
    pass


# rocBLAS SGEMM tiles can over-read a few rows past small operand buffers.
# Every GEMM-touched allocation is padded AND fully committed (zero-filled)
# at malloc time: on APU/GTT systems pages are demand-mapped, so an
# overread into never-written pad memory raises a GPU "page not present"
# fault that aborts the process.

_GEMM_PAD_BYTES = 1 << 20


def _malloc_committed(nbytes: int) -> DeviceBuffer:
    """Allocate and commit every page so GEMM overreads cannot fault."""

    buf = malloc(nbytes)
    zeros = np.zeros(nbytes, dtype=np.uint8)
    copy_host_to_device(buf, host_array_ptr(zeros), nbytes)
    return buf


@dataclass
class _Scratch:
    """Device scratch buffers for one token capacity."""

    tokens: int
    buffers: dict[str, DeviceBuffer] = field(default_factory=dict)

    def free(self) -> None:
        for buf in self.buffers.values():
            hip_free(buf)
        self.buffers.clear()


def _raw_buffer(ptr: int, nbytes: int) -> DeviceBuffer:
    """A non-owning view over an existing device pointer (for copies)."""

    return DeviceBuffer(ptr=ptr, nbytes=nbytes)


def _fn(library: ctypes.CDLL, symbol: str, argtypes: list) -> "ctypes._FuncPtr":
    fn = getattr(library, symbol)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


class EvieRunner:
    """EVIE-4.5B fp32 encoder on the HIP device."""

    GDN_QKV_DIM = 8192
    GDN_Z_DIM = 4096
    GDN_HEADS = 32
    GDN_KV_HEADS = 16
    GDN_HEAD_DIM = 128
    ATTENTION_HEADS = 16
    ATTENTION_KV_HEADS = 4
    ATTENTION_HEAD_DIM = 256
    ROTARY_DIM = 64
    VISION_HEADS = 16
    VISION_HEAD_DIM = 64

    def __init__(
        self,
        loaded: EvieLoadedModel,
        *,
        rocblas: Rocblas | None = None,
        runtime: HipRuntime | None = None,
        library: ctypes.CDLL | None = None,
        conv_library: ctypes.CDLL | None = None,
        gdn_library: ctypes.CDLL | None = None,
        precision: str = "fp32",
    ):
        if precision not in ("fp32", "fp16"):
            raise ValueError("precision must be 'fp32' or 'fp16'")
        self.precision = precision
        if loaded.precision != precision:
            raise ValueError(
                "runner precision does not match the loaded weights "
                f"({precision!r} runner vs {loaded.precision!r} weights); "
                "reload with load_evie_model(precision=...)"
            )
        self.loaded = loaded
        self.spec: EvieModelSpec = loaded.spec
        self.runtime = runtime or get_hip_runtime()
        self.rocblas = rocblas or Rocblas.load()
        # SGEMM needs no auxiliary workspace; release rocBLAS's lazy ~32-MiB
        # device reserve (mirrors the qwen35 GGUF runner) so everything stays
        # inside hipEngine's tracked allocations.
        self.rocblas.set_workspace(0, 0)
        self.library = library or build_evie_ops(load=True)
        self.conv_library = conv_library or build_qwen35_linear_attn_conv(load=True)
        self.gdn_library = gdn_library or build_qwen35_linear_attn_gdn(load=True)
        self._w = {name: a.buffer.ptr for name, a in loaded.weights.tensors.items()}
        # fp16 GEMM path: activations are cast to a per-scratch fp16 staging
        # buffer; weights were uploaded as fp16 by the loader.
        self._cast16_scratch: DeviceBuffer | None = None
        self._gemm16_out: DeviceBuffer | None = None
        self._scratch: dict[int, _Scratch] = {}
        self._misc_buffers: list[DeviceBuffer] = []
        self._rope_cache: dict[bytes, tuple[DeviceBuffer, DeviceBuffer]] = {}
        self._zero_conv_state: DeviceBuffer | None = None
        self._zero_gdn_state: DeviceBuffer | None = None
        self._gdn_state_zero: DeviceBuffer | None = None
        self._head_bias: DeviceBuffer | None = None
        # GDN prefill recurrence. cluster8 is algebraically exact for this
        # block (the 1/sqrt(d_k) q scale is applied at its output via
        # kOutputScale instead of on the input): single-layer max diff vs
        # k2 is 3e-08, end-to-end strict parity doc 4.4e-05 / query 2.6e-06
        # vs the torch oracle, bit-repeatable, and ~9%/page faster. k2
        # remains the opt-out for rollback/bisection.
        self._gdn_recurrence = (
            qwen35_gdn_prefill_recurrent_k2_f32
            if os.environ.get("HIPENGINE_EVIE_GDN_RECURRENCE") == "k2"
            else qwen35_gdn_prefill_recurrent_normalized_cluster8_f32
        )
        self._pos_embed_table: np.ndarray | None = None

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for scratch in self._scratch.values():
            scratch.free()
        self._scratch.clear()
        for buf in self._misc_buffers:
            hip_free(buf)
        self._misc_buffers.clear()
        self._zero_conv_state = None
        if self._gdn_state_zero is not None:
            hip_free(self._gdn_state_zero)
            self._gdn_state_zero = None
        self._head_bias = None
        if self._cast16_scratch is not None:
            hip_free(self._cast16_scratch)
            self._cast16_scratch = None
        if self._gemm16_out is not None:
            hip_free(self._gemm16_out)
            self._gemm16_out = None
        for bufs in (
            getattr(self, "_scores_bufs", None),
            getattr(self, "_plane16_bufs", None),
            getattr(self, "_ptr_array_bufs", None),
        ):
            if bufs:
                for buf in bufs.values():
                    hip_free(buf)
                bufs.clear()
        self._rope_cache.clear()
        # release the resident weight set so sequential runners in one
        # process are truly single-model-resident
        self.loaded.free(runtime=self.runtime)

    def _release_call_buffers(self) -> None:
        """Free per-call scratch; keep persistent state alive."""

        keep = {self._zero_conv_state.ptr} if self._zero_conv_state is not None else set()
        if self._head_bias is not None:
            keep.add(self._head_bias.ptr)
        for cos_buf, sin_buf in self._rope_cache.values():
            keep.update((cos_buf.ptr, sin_buf.ptr))
        for buf in self._misc_buffers:
            if buf.ptr not in keep:
                hip_free(buf)
        self._misc_buffers = [
            buf for buf in self._misc_buffers if buf.ptr in keep
        ]

    # -- low-level helpers -----------------------------------------------------

    def _k(self, symbol: str, argtypes: list) -> "ctypes._FuncPtr":
        return _fn(self.library, symbol, argtypes)

    def _gemm(self, x_ptr: int, w_ptr: int, out_ptr: int, rows: int, fin: int, fout: int, out_stride: int | None = None) -> None:
        """Row-major NT GEMM; fp16 production path casts x and keeps fp32 out."""

        if self.precision != "fp16":
            self.rocblas.sgemm_rowmajor_nt(
                x_ptr, w_ptr, out_ptr, rows=rows, in_features=fin, out_features=fout
            )
            return
        n = rows * fin
        if self._cast16_scratch is None or self._cast16_scratch.nbytes < n * 2 + _GEMM_PAD_BYTES:
            if self._cast16_scratch is not None:
                hip_free(self._cast16_scratch)
            self._cast16_scratch = _malloc_committed(n * 2 + _GEMM_PAD_BYTES)
        err = self._k("hipengine_evie_cast_f32_to_f16", [_P, _P, _I, _S])(
            _P(x_ptr), _P(self._cast16_scratch.ptr), _I(n), _S(0)
        )
        self._check(err, "cast f32->f16")
        n_out = rows * fout
        if self._gemm16_out is None or self._gemm16_out.nbytes < n_out * 2 + _GEMM_PAD_BYTES:
            if self._gemm16_out is not None:
                hip_free(self._gemm16_out)
            self._gemm16_out = _malloc_committed(n_out * 2 + _GEMM_PAD_BYTES)
        # f16 output runs on the matrix cores (~4-6x faster than the f32-out
        # gemm_ex variant on this APU); cast back to fp32 for downstream
        # kernels and the residual stream.
        self.rocblas.gemm_ex_rowmajor_nt_fp16_compute_f32(
            self._cast16_scratch.ptr,
            w_ptr,
            self._gemm16_out.ptr,
            rows=rows,
            in_features=fin,
            out_features=fout,
        )
        if out_stride is None or out_stride == fout:
            err = self._k("hipengine_evie_cast_f16_to_f32", [_P, _P, _I, _S])(
                _P(self._gemm16_out.ptr), _P(out_ptr), _I(n_out), _S(0)
            )
            self._check(err, "cast f16->f32")
        else:
            err = self._k(
                "hipengine_evie_cast_f16_to_f32_strided", [_P, _P, _I, _I, _I, _S]
            )(
                _P(self._gemm16_out.ptr), _P(out_ptr), _I(rows), _I(fout),
                _I(out_stride), _S(0),
            )
            self._check(err, "cast f16->f32 strided")

    def _to_dev(self, host: np.ndarray) -> DeviceBuffer:
        host = np.ascontiguousarray(host, dtype=np.float32)
        buf = _malloc_committed(host.nbytes + _GEMM_PAD_BYTES)
        copy_host_to_device(buf, host_array_ptr(host), host.nbytes)
        self._misc_buffers.append(buf)
        return buf

    def _to_host(self, ptr: int, n: int) -> np.ndarray:
        host = np.empty(n, dtype=np.float32)
        copy_device_to_host(host_array_ptr(host), _raw_buffer(ptr, n * 4))
        return host

    def _check(self, err: int, what: str) -> None:
        if err != 0:
            raise EvieRuntimeError(f"{what} failed: {err}")

    def _rmsnorm(self, x_ptr: int, w_ptr: int, out_ptr: int, rows: int, hidden: int, eps: float = 1e-6) -> None:
        err = self._k("hipengine_evie_rmsnorm_f32", [_P, _P, _P, _I, _I, _F, _S])(
            _P(x_ptr), _P(w_ptr), _P(out_ptr), _I(rows), _I(hidden), _F(eps), _S(0)
        )
        self._check(err, "rmsnorm")

    def _add(self, x_ptr: int, y_ptr: int, out_ptr: int, n: int) -> None:
        err = self._k("hipengine_evie_add_f32", [_P, _P, _P, _I, _S])(
            _P(x_ptr), _P(y_ptr), _P(out_ptr), _I(n), _S(0)
        )
        self._check(err, "add")

    def _dev_ptr_array(self, ptrs: list[int]) -> int:
        """Upload a host pointer list for rocBLAS batched GEMMs (device arrays).

        Buffers are cached per unique pointer list, so the A/B/C arrays of
        one call never alias.
        """

        if not hasattr(self, "_ptr_array_bufs"):
            self._ptr_array_bufs: dict[tuple, DeviceBuffer] = {}
        key = tuple(ptrs)
        nbytes = len(ptrs) * 8
        buf = self._ptr_array_bufs.get(key)
        if buf is None:
            buf = _malloc_committed(nbytes + _GEMM_PAD_BYTES)
            self._ptr_array_bufs[key] = buf
        host = np.array(ptrs, dtype=np.uint64)
        copy_host_to_device(buf, host_array_ptr(host), nbytes)
        return buf.ptr

    def _evict_caches(self, keep: tuple) -> None:
        """Bound every lazily-grown scratch cache to 4 entries."""

        for cache_name in ("_scores_bufs", "_plane16_bufs"):
            cache = getattr(self, cache_name, None)
            if not isinstance(cache, dict) or len(cache) < 4:
                continue
            keepers = set(keep)
            while len(cache) >= 4:
                oldest = next(iter(cache))
                if oldest in keepers:
                    break
                hip_free(cache.pop(oldest))

    def _scores16_scratch(self, rows: int, heads: int) -> tuple[DeviceBuffer, int]:
        """f16 scores buffer plus the per-head element stride (256B aligned)."""

        stride = (rows * rows + 127) & ~127
        key = ("scores16", rows, heads)
        if not hasattr(self, "_scores_bufs"):
            self._scores_bufs: dict[Any, DeviceBuffer] = {}
        if key not in self._scores_bufs:
            self._evict_caches((key,))
            self._scores_bufs[key] = _malloc_committed(
                heads * stride * 2 + _GEMM_PAD_BYTES
            )
        return self._scores_bufs[key], stride

    def _plane16_scratch(self, key: str, n: int) -> DeviceBuffer:
        """f16 staging plane for attention inputs/outputs."""

        k = ("plane16", key, n)
        if not hasattr(self, "_plane16_bufs"):
            self._plane16_bufs: dict[Any, DeviceBuffer] = {}
        if k not in self._plane16_bufs:
            self._evict_caches((k,))
            self._plane16_bufs[k] = _malloc_committed(n * 2 + _GEMM_PAD_BYTES)
        return self._plane16_bufs[k]

    def _scores_scratch(self, rows: int, heads: int = 1) -> tuple[DeviceBuffer, int]:
        """Scores buffer plus the per-head element stride (16-byte aligned)."""

        stride = (rows * rows + 3) & ~3
        key = ("scores", rows, heads)
        if not hasattr(self, "_scores_bufs"):
            self._scores_bufs: dict[Any, DeviceBuffer] = {}
        if key not in self._scores_bufs:
            self._evict_caches((key,))
            self._scores_bufs[key] = _malloc_committed(
                heads * stride * 4 + _GEMM_PAD_BYTES
            )
        return self._scores_bufs[key], stride

    # -- attention (shared by vision and text) ----------------------------------

    def _attention(
        self,
        q_ptr: int,
        k_ptr: int,
        v_ptr: int,
        out_ptr: int,
        tokens: int,
        heads: int,
        head_dim: int,
        scratch: _Scratch,
        scale: float,
        kv_heads: int | None = None,
    ) -> None:
        """Bidirectional attention over contiguous (tokens, heads, dim) planes.

        Both GEMMs are rocBLAS pointer-array batched SGEMMs; GQA head groups
        map each query head to its key/value head (heads must be a multiple
        of kv_heads), so no repeat expansion is materialized and the AV
        product writes straight into the packed output. In fp16 mode the
        GEMMs run f16 on per-head-contiguous planes (256-byte-aligned batch
        pointers) with an f16 scale+softmax kernel.
        """

        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise EvieRuntimeError("heads must be a multiple of kv_heads")
        repeat = heads // kv_heads
        q_row = heads * head_dim
        kv_row = kv_heads * head_dim
        if self.precision == "fp16":
            plane = (tokens * head_dim + 127) & ~127
            q16 = self._plane16_scratch("q", heads * plane)
            k16 = self._plane16_scratch("k", heads * plane)
            v16 = self._plane16_scratch("v", heads * plane)
            for src, buf, src_repeat, src_row in (
                (q_ptr, q16, 1, q_row),
                (k_ptr, k16, repeat, kv_row),
                (v_ptr, v16, repeat, kv_row),
            ):
                err = self._k(
                    "hipengine_evie_gather_repeat_f16",
                    [_P, _P, _I, _I, _I, _I, _I, _I, _S],
                )(
                    _P(src), _P(buf.ptr), _I(tokens), _I(src_row),
                    _I(heads), _I(src_repeat), _I(head_dim), _I(plane), _S(0),
                )
                self._check(err, "gather repeat f16")
            scores16, head_stride = self._scores16_scratch(tokens, heads)
            self.rocblas.gemm_ex_strided_batched_f16_f32acc(
                k16.ptr,
                q16.ptr,
                scores16.ptr,
                m=tokens,
                n=tokens,
                k=head_dim,
                lda=head_dim,
                ldb=head_dim,
                ldc=tokens,
                stride_a=plane,
                stride_b=plane,
                stride_c=head_stride,
                batch=heads,
                trans_a=True,
                trans_b=False,
            )
            err = self._k(
                "hipengine_evie_scale_softmax_rows_f16",
                [_P, _F, _I, _I, _I, _I, _S],
            )(
                _P(scores16.ptr), _F(scale), _I(heads * tokens), _I(tokens),
                _I(tokens), _I(head_stride), _S(0),
            )
            self._check(err, "softmax f16")
            out16 = self._plane16_scratch("attn_out", heads * plane)
            self.rocblas.gemm_ex_strided_batched_f16_f32acc(
                v16.ptr,
                scores16.ptr,
                out16.ptr,
                m=head_dim,
                n=tokens,
                k=tokens,
                lda=head_dim,
                ldb=tokens,
                ldc=head_dim,
                stride_a=plane,
                stride_b=head_stride,
                stride_c=plane,
                batch=heads,
                trans_a=False,
                trans_b=False,
            )
            err = self._k(
                "hipengine_evie_scatter_heads_f16", [_P, _P, _I, _I, _I, _I, _S]
            )(
                _P(out16.ptr), _P(out_ptr), _I(tokens), _I(heads),
                _I(head_dim), _I(plane), _S(0),
            )
            self._check(err, "scatter out f32")
            return

        scores, head_stride = self._scores_scratch(tokens, heads)
        # scores[h] = q_h @ k_{h//repeat}^T  -> row-major (tokens, tokens) per head
        a_dev = self._dev_ptr_array(
            [k_ptr + (h // repeat) * head_dim * 4 for h in range(heads)]
        )
        b_dev = self._dev_ptr_array(
            [q_ptr + h * head_dim * 4 for h in range(heads)]
        )
        c_dev = self._dev_ptr_array(
            [scores.ptr + h * head_stride * 4 for h in range(heads)]
        )
        self.rocblas.sgemm_batched(
            a_dev,
            b_dev,
            c_dev,
            batch=heads,
            m=tokens,
            n=tokens,
            k=head_dim,
            lda=kv_row,
            ldb=q_row,
            ldc=tokens,
            trans_a=True,
            trans_b=False,
        )
        err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
            _P(scores.ptr), _P(scores.ptr), _F(scale), _I(heads * head_stride), _S(0)
        )
        self._check(err, "scale")
        err = self._k(
            "hipengine_evie_softmax_rows_f32", [_P, _I, _I, _I, _I, _S]
        )(
            _P(scores.ptr), _I(heads * tokens), _I(tokens), _I(tokens),
            _I(head_stride), _S(0),
        )
        self._check(err, "softmax")
        # out[:, h*hd:(h+1)*hd] = scores[h] @ v_{h//repeat}  (direct packed write)
        self.rocblas.sgemm_batched(
            self._dev_ptr_array(
                [v_ptr + (h // repeat) * head_dim * 4 for h in range(heads)]
            ),
            self._dev_ptr_array(
                [scores.ptr + h * head_stride * 4 for h in range(heads)]
            ),
            self._dev_ptr_array(
                [out_ptr + h * head_dim * 4 for h in range(heads)]
            ),
            batch=heads,
            m=head_dim,
            n=tokens,
            k=tokens,
            lda=kv_row,
            ldb=tokens,
            ldc=q_row,
            trans_a=False,
            trans_b=False,
        )

    # -- vision tower -------------------------------------------------------------

    def _vision_pos_embed_host(self, grid_thw: np.ndarray) -> np.ndarray:
        from hipengine.kernels.cpu_reference.evie import _bilinear_interp_indices

        if self._pos_embed_table is None:
            self._pos_embed_table = self._to_host(
                self._w["visual.pos_embed.weight"], 2304 * 1024
            ).reshape(2304, 1024)
        table = self._pos_embed_table
        indices, weights = _bilinear_interp_indices(
            grid_thw, 48, self.spec.vision_spatial_merge_size
        )
        return np.ascontiguousarray((table[indices] * weights[:, :, None]).sum(axis=1))

    def vision_forward(
        self,
        pixel_values: np.ndarray,
        grid_thw: np.ndarray,
        scratch: _Scratch,
    ) -> DeviceBuffer:
        """Run the vision tower; returns the device buffer of merged features."""

        spec = self.spec
        n_patches = len(pixel_values)
        merge = spec.vision_spatial_merge_size
        vh = spec.vision_hidden_size
        heads = self.VISION_HEADS
        head_dim = self.VISION_HEAD_DIM

        pv_dev = self._to_dev(pixel_values)
        x_ptr = scratch.buffers["vx"].ptr
        self._gemm(pv_dev.ptr, self._w["visual.patch_embed.proj.weight"], x_ptr, n_patches, 1536, vh)
        err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
            _P(x_ptr), _P(self._w["visual.patch_embed.proj.bias"]),
            _I(n_patches * vh), _I(vh), _S(0)
        )
        self._check(err, "patch embed bias")

        pos = self._to_dev(self._vision_pos_embed_host(grid_thw))
        self._add(x_ptr, pos.ptr, x_ptr, n_patches * vh)

        from hipengine.kernels.cpu_reference.evie import (
            vision_position_ids_block_major,
        )

        positions = vision_position_ids_block_major(grid_thw, merge)
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, 16, dtype=np.float32) / 16))
        fh = positions[:, 0][:, None].astype(np.float32) * inv_freq[None]
        fw = positions[:, 1][:, None].astype(np.float32) * inv_freq[None]
        freqs = np.concatenate([fh, fw], axis=-1)
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos_dev = self._to_dev(np.cos(emb))
        sin_dev = self._to_dev(np.sin(emb))

        cu = [0]
        for (_t, h, w) in grid_thw:
            cu.append(cu[-1] + h * w)
        if len(cu) > 2:
            raise EvieRuntimeError("multi-image batches not yet supported")

        qkv_ptr = scratch.buffers["vqkv"].ptr
        norm_ptr = scratch.buffers["vnorm"].ptr
        mlp_ptr = scratch.buffers["vmlp"].ptr
        attn_ptr = scratch.buffers["vout"].ptr
        out_ptr = scratch.buffers["vx2"].ptr
        scale = 1.0 / math.sqrt(head_dim)

        for i in range(spec.vision_depth):
            p = f"visual.blocks.{i}."
            err = self._k(
                "hipengine_evie_layernorm_f32", [_P, _P, _P, _P, _I, _I, _F, _S]
            )(
                _P(x_ptr), _P(self._w[p + "norm1.weight"]), _P(self._w[p + "norm1.bias"]),
                _P(norm_ptr), _I(n_patches), _I(vh), _F(1e-6), _S(0),
            )
            self._check(err, "vision layernorm")
            self._gemm(norm_ptr, self._w[p + "attn.qkv.weight"], qkv_ptr, n_patches, vh, 3 * vh)
            err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
                _P(qkv_ptr), _P(self._w[p + "attn.qkv.bias"]),
                _I(n_patches * 3 * vh), _I(3 * vh), _S(0)
            )
            self._check(err, "vision qkv bias")
            # rope q/k inside the packed qkv (row stride 3*vh, planes at 0 and vh)
            err = self._k(
                "hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S]
            )(
                _P(qkv_ptr), _P(cos_dev.ptr), _P(sin_dev.ptr), _I(n_patches),
                _I(heads), _I(head_dim), _I(head_dim), _I(3 * vh), _S(0),
            )
            self._check(err, "vision rope q")
            err = self._k(
                "hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S]
            )(
                _P(qkv_ptr + vh * 4), _P(cos_dev.ptr), _P(sin_dev.ptr), _I(n_patches),
                _I(heads), _I(head_dim), _I(head_dim), _I(3 * vh), _S(0),
            )
            self._check(err, "vision rope k")
            # attention reads q/k/v planes directly from qkv via expansion
            self._attention_from_packed(
                qkv_ptr, qkv_ptr + vh * 4, qkv_ptr + 2 * vh * 4, attn_ptr,
                n_patches, heads, head_dim, 3 * vh, scratch, scale,
            )
            self._gemm(attn_ptr, self._w[p + "attn.proj.weight"], out_ptr, n_patches, vh, vh)
            err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
                _P(out_ptr), _P(self._w[p + "attn.proj.bias"]),
                _I(n_patches * vh), _I(vh), _S(0)
            )
            self._check(err, "vision proj bias")
            self._add(x_ptr, out_ptr, x_ptr, n_patches * vh)
            # mlp
            err = self._k(
                "hipengine_evie_layernorm_f32", [_P, _P, _P, _P, _I, _I, _F, _S]
            )(
                _P(x_ptr), _P(self._w[p + "norm2.weight"]), _P(self._w[p + "norm2.bias"]),
                _P(norm_ptr), _I(n_patches), _I(vh), _F(1e-6), _S(0),
            )
            self._check(err, "vision layernorm 2")
            self._gemm(norm_ptr, self._w[p + "mlp.linear_fc1.weight"], mlp_ptr, n_patches, vh, spec.vision_intermediate_size)
            err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
                _P(mlp_ptr), _P(self._w[p + "mlp.linear_fc1.bias"]),
                _I(n_patches * spec.vision_intermediate_size), _I(spec.vision_intermediate_size), _S(0)
            )
            self._check(err, "vision fc1 bias")
            err = self._k("hipengine_evie_gelu_tanh_f32", [_P, _P, _I, _S])(
                _P(mlp_ptr), _P(mlp_ptr), _I(n_patches * spec.vision_intermediate_size), _S(0)
            )
            self._check(err, "vision gelu")
            self._gemm(mlp_ptr, self._w[p + "mlp.linear_fc2.weight"], out_ptr, n_patches, spec.vision_intermediate_size, vh)
            err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
                _P(out_ptr), _P(self._w[p + "mlp.linear_fc2.bias"]),
                _I(n_patches * vh), _I(vh), _S(0)
            )
            self._check(err, "vision fc2 bias")
            self._add(x_ptr, out_ptr, x_ptr, n_patches * vh)

        # merger
        n_merged = n_patches // (merge * merge)
        err = self._k(
            "hipengine_evie_layernorm_f32", [_P, _P, _P, _P, _I, _I, _F, _S]
        )(
            _P(x_ptr), _P(self._w["visual.merger.norm.weight"]),
            _P(self._w["visual.merger.norm.bias"]), _P(norm_ptr),
            _I(n_patches), _I(vh), _F(1e-6), _S(0),
        )
        self._check(err, "merger layernorm")
        merged_ptr = scratch.buffers["merged_in"].ptr
        self._gemm(
            norm_ptr, self._w["visual.merger.linear_fc1.weight"], merged_ptr,
            n_merged, merge * merge * vh, merge * merge * vh,
        )
        err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
            _P(merged_ptr), _P(self._w["visual.merger.linear_fc1.bias"]),
            _I(n_merged * merge * merge * vh), _I(merge * merge * vh), _S(0)
        )
        self._check(err, "merger fc1 bias")
        err = self._k("hipengine_evie_gelu_erf_f32", [_P, _P, _I, _S])(
            _P(merged_ptr), _P(merged_ptr), _I(n_merged * merge * merge * vh), _S(0)
        )
        self._check(err, "merger gelu")
        out = _malloc_committed(n_merged * spec.vision_out_hidden_size * 4 + _GEMM_PAD_BYTES)
        self._misc_buffers.append(out)
        self._gemm(
            merged_ptr, self._w["visual.merger.linear_fc2.weight"], out.ptr,
            n_merged, merge * merge * vh, spec.vision_out_hidden_size,
        )
        err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
            _P(out.ptr), _P(self._w["visual.merger.linear_fc2.bias"]),
            _I(n_merged * spec.vision_out_hidden_size), _I(spec.vision_out_hidden_size), _S(0)
        )
        self._check(err, "merger fc2 bias")
        return out

    def _attention_from_packed(
        self,
        q_ptr: int,
        k_ptr: int,
        v_ptr: int,
        out_ptr: int,
        tokens: int,
        heads: int,
        head_dim: int,
        row_stride: int,
        scratch: _Scratch,
        scale: float,
    ) -> None:
        """Batched attention over head planes embedded in packed rows.

        Both GEMMs are rocBLAS strided-batched SGEMMs over the packed qkv
        layout (row stride 3*hidden, per-head stride head_dim); the AV
        product writes straight into the packed (tokens, heads*head_dim)
        output, so no per-head gathers or scatters are needed. In fp16 mode
        the attention runs f16 GEMMs over per-head-contiguous planes (the
        batched-ex kernels need 256-byte-aligned batch pointers) with an
        f16 scale+softmax kernel and an f32 scatter at the end.
        """

        if self.precision == "fp16":
            # per-head plane stride, padded to keep every batch pointer
            # 256-byte aligned for gemm_strided_batched_ex
            plane = (tokens * head_dim + 127) & ~127
            qkv16 = self._plane16_scratch("packed", 3 * heads * plane)
            err = self._k(
                "hipengine_evie_gather_qkv_f16", [_P, _P, _I, _I, _I, _I, _I, _S]
            )(
                _P(q_ptr), _P(qkv16.ptr), _I(tokens), _I(row_stride),
                _I(heads), _I(head_dim), _I(plane), _S(0),
            )
            self._check(err, "gather qkv f16")
            scores16, head_stride = self._scores16_scratch(tokens, heads)
            self.rocblas.gemm_ex_strided_batched_f16_f32acc(
                qkv16.ptr + heads * plane * 2,  # k plane
                qkv16.ptr,                       # q plane
                scores16.ptr,
                m=tokens,
                n=tokens,
                k=head_dim,
                lda=head_dim,
                ldb=head_dim,
                ldc=tokens,
                stride_a=plane,
                stride_b=plane,
                stride_c=head_stride,
                batch=heads,
                trans_a=True,
                trans_b=False,
            )
            err = self._k(
                "hipengine_evie_scale_softmax_rows_f16",
                [_P, _F, _I, _I, _I, _I, _S],
            )(
                _P(scores16.ptr), _F(scale), _I(heads * tokens), _I(tokens),
                _I(tokens), _I(head_stride), _S(0),
            )
            self._check(err, "softmax f16")
            out16 = self._plane16_scratch("out", heads * plane)
            self.rocblas.gemm_ex_strided_batched_f16_f32acc(
                qkv16.ptr + 2 * heads * plane * 2,  # v plane
                scores16.ptr,
                out16.ptr,
                m=head_dim,
                n=tokens,
                k=tokens,
                lda=head_dim,
                ldb=tokens,
                ldc=head_dim,
                stride_a=plane,
                stride_b=head_stride,
                stride_c=plane,
                batch=heads,
                trans_a=False,
                trans_b=False,
            )
            err = self._k(
                "hipengine_evie_scatter_heads_f16", [_P, _P, _I, _I, _I, _I, _S]
            )(
                _P(out16.ptr), _P(out_ptr), _I(tokens), _I(heads),
                _I(head_dim), _I(plane), _S(0),
            )
            self._check(err, "scatter out f32")
            return

        scores, head_stride = self._scores_scratch(tokens, heads)
        # scores[h] = q_h @ k_h^T  -> row-major (tokens, tokens) per head
        self.rocblas.sgemm_strided_batched(
            k_ptr,
            q_ptr,
            scores.ptr,
            m=tokens,
            n=tokens,
            k=head_dim,
            lda=row_stride,
            ldb=row_stride,
            ldc=tokens,
            stride_a=head_dim,
            stride_b=head_dim,
            stride_c=head_stride,
            batch=heads,
            trans_a=True,
            trans_b=False,
        )
        err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
            _P(scores.ptr), _P(scores.ptr), _F(scale), _I(heads * head_stride), _S(0)
        )
        self._check(err, "scale")
        err = self._k(
            "hipengine_evie_softmax_rows_f32", [_P, _I, _I, _I, _I, _S]
        )(
            _P(scores.ptr), _I(heads * tokens), _I(tokens), _I(tokens),
            _I(head_stride), _S(0),
        )
        self._check(err, "softmax")
        # out[:, h*hd:(h+1)*hd] = scores[h] @ v_h  (direct packed write)
        self.rocblas.sgemm_strided_batched(
            v_ptr,
            scores.ptr,
            out_ptr,
            m=head_dim,
            n=tokens,
            k=tokens,
            lda=row_stride,
            ldb=tokens,
            ldc=heads * head_dim,
            stride_a=head_dim,
            stride_b=head_stride,
            stride_c=head_dim,
            batch=heads,
            trans_a=False,
            trans_b=False,
        )

    # -- text stack ----------------------------------------------------------------

    def _mrope_tables(self, positions: np.ndarray) -> tuple[DeviceBuffer, DeviceBuffer]:
        key = np.ascontiguousarray(positions, dtype=np.int64).tobytes()
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        inv_freq = 1.0 / (
            self.spec.rope_theta
            ** (np.arange(0, self.ROTARY_DIM, 2, dtype=np.float32) / self.ROTARY_DIM)
        )
        freqs = inv_freq[None, :, None] * positions[:, None, :].astype(np.float32)
        freqs_t = freqs[0].copy()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.spec.mrope_section[axis] * 3
            idx = np.arange(offset, length, 3)
            freqs_t[idx] = freqs[axis, idx]
        emb = np.concatenate([freqs_t, freqs_t], axis=0).T
        cos_buf = self._to_dev(np.cos(emb))
        sin_buf = self._to_dev(np.sin(emb))
        self._rope_cache[key] = (cos_buf, sin_buf)
        return cos_buf, sin_buf

    def text_forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        scratch: _Scratch,
        *,
        visual_ptr: int | None = None,
    ) -> int:
        spec = self.spec
        tokens = len(input_ids)
        h = spec.hidden_size
        ids_host = np.ascontiguousarray(input_ids, dtype=np.int64)
        ids_buf = _malloc_committed(ids_host.nbytes)
        copy_host_to_device(ids_buf, host_array_ptr(ids_host))
        self._misc_buffers.append(ids_buf)
        x_ptr = scratch.buffers["x"].ptr
        table_ptr = self._w["language_model.embed_tokens.weight"]
        err = self._k(
            "hipengine_evie_embed_lookup_f32",
            [_P, _P, _P, _P, _I, _I, _I, _S],
        )(
            _P(ids_buf.ptr), _P(table_ptr),
            _P(visual_ptr if visual_ptr is not None else table_ptr), _P(x_ptr),
            _I(tokens), _I(h),
            _I(spec.image_token_id if visual_ptr is not None else -1), _S(0),
        )
        self._check(err, "embed lookup")

        cos_buf, sin_buf = self._mrope_tables(positions)
        norm_ptr = scratch.buffers["norm"].ptr
        attn_out_ptr = scratch.buffers["attn_out"].ptr

        for layer in range(spec.num_layers):
            lp = f"language_model.layers.{layer}."
            self._rmsnorm(x_ptr, self._w[lp + "input_layernorm.weight"], norm_ptr, tokens, h)
            if spec.is_full_attention(layer):
                self._full_attention_layer(layer, norm_ptr, attn_out_ptr, tokens, cos_buf, sin_buf, scratch)
            else:
                self._gdn_layer(layer, norm_ptr, attn_out_ptr, tokens, scratch)
            self._add(x_ptr, attn_out_ptr, x_ptr, tokens * h)
            self._rmsnorm(x_ptr, self._w[lp + "post_attention_layernorm.weight"], norm_ptr, tokens, h)
            gate_ptr = scratch.buffers["gate_proj"].ptr
            up_ptr = scratch.buffers["up_proj"].ptr
            self._gemm(norm_ptr, self._w[lp + "mlp.gate_proj.weight"], gate_ptr, tokens, h, spec.intermediate_size)
            self._gemm(norm_ptr, self._w[lp + "mlp.up_proj.weight"], up_ptr, tokens, h, spec.intermediate_size)
            err = self._k("hipengine_evie_silu_mul_f32", [_P, _P, _I, _S])(
                _P(gate_ptr), _P(up_ptr), _I(tokens * spec.intermediate_size), _S(0)
            )
            self._check(err, "swiglu")
            self._gemm(gate_ptr, self._w[lp + "mlp.down_proj.weight"], attn_out_ptr, tokens, spec.intermediate_size, h)
            self._add(x_ptr, attn_out_ptr, x_ptr, tokens * h)

        self._rmsnorm(x_ptr, self._w["language_model.norm.weight"], norm_ptr, tokens, h)
        return norm_ptr

    def _full_attention_layer(
        self,
        layer: int,
        norm_ptr: int,
        out_ptr: int,
        tokens: int,
        cos_buf: DeviceBuffer,
        sin_buf: DeviceBuffer,
        scratch: _Scratch,
    ) -> None:
        spec = self.spec
        p = f"language_model.layers.{layer}.self_attn."
        h = spec.hidden_size
        nq = self.ATTENTION_HEADS
        nk = self.ATTENTION_KV_HEADS
        hd = self.ATTENTION_HEAD_DIM
        qkv_ptr = scratch.buffers["qkv_full"].ptr  # (tokens, 8192) q+gate
        q_ptr = scratch.buffers["q_full"].ptr
        gate_ptr = scratch.buffers["gate_full"].ptr
        k_ptr = scratch.buffers["k_full"].ptr
        v_ptr = scratch.buffers["v_full"].ptr
        k_rep = scratch.buffers["k_rep"].ptr
        v_rep = scratch.buffers["v_rep"].ptr
        heads_out = scratch.buffers["heads_out"].ptr

        self._gemm(norm_ptr, self._w[p + "q_proj.weight"], qkv_ptr, tokens, h, nq * hd * 2)
        self._gemm(norm_ptr, self._w[p + "k_proj.weight"], k_ptr, tokens, h, nk * hd)
        self._gemm(norm_ptr, self._w[p + "v_proj.weight"], v_ptr, tokens, h, nk * hd)

        err = self._k("hipengine_evie_split_qgate_f32", [_P, _P, _P, _I, _S])(
            _P(qkv_ptr), _P(q_ptr), _P(gate_ptr), _I(tokens), _S(0)
        )
        self._check(err, "split qgate")
        self._rmsnorm(q_ptr, self._w[p + "q_norm.weight"], q_ptr, tokens * nq, hd)
        self._rmsnorm(k_ptr, self._w[p + "k_norm.weight"], k_ptr, tokens * nk, hd)
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(q_ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(tokens), _I(nq),
            _I(hd), _I(self.ROTARY_DIM), _I(nq * hd), _S(0),
        )
        self._check(err, "rope q")
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])
        err = err(
            _P(k_ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(tokens), _I(nk),
            _I(hd), _I(self.ROTARY_DIM), _I(nk * hd), _S(0),
        )
        self._check(err, "rope k")
        self._attention(
            q_ptr, k_ptr, v_ptr, heads_out, tokens, nq, hd, scratch,
            1.0 / math.sqrt(hd), kv_heads=nk,
        )
        # sigmoid gate
        err = self._k("hipengine_evie_sigmoid_mul_f32", [_P, _P, _I, _S])(
            _P(gate_ptr), _P(heads_out), _I(tokens * nq * hd), _S(0)
        )
        self._check(err, "attn gate")
        self._gemm(heads_out, self._w[p + "o_proj.weight"], out_ptr, tokens, nq * hd, h)

    def _gdn_layer(
        self,
        layer: int,
        norm_ptr: int,
        out_ptr: int,
        tokens: int,
        scratch: _Scratch,
    ) -> None:
        p = f"language_model.layers.{layer}.linear_attn."
        h = self.spec.hidden_size
        qkv_ptr = scratch.buffers["qkv_gdn"].ptr
        z_ptr = scratch.buffers["z_gdn"].ptr
        b_ptr = scratch.buffers["b_gdn"].ptr
        a_ptr = scratch.buffers["a_gdn"].ptr
        conv_ptr = scratch.buffers["conv_out"].ptr
        q_ptr = scratch.buffers["gdn_q"].ptr
        k_ptr = scratch.buffers["gdn_k"].ptr
        beta_ptr = scratch.buffers["gdn_beta"].ptr
        decay_ptr = scratch.buffers["gdn_decay"].ptr
        gdn_out = scratch.buffers["gdn_out"].ptr
        gdn_normed = scratch.buffers["gdn_normed"].ptr

        # qkv split: q [0:2048) and k [2048:4096) tolerate the fast f16-out
        # epilogue; v [4096:8192) feeds the persistent delta-rule state and
        # keeps f32-out. Weight slices are (8192, h) row-major fp16.
        self._gemm(norm_ptr, self._w[p + "in_proj_qkv.weight"], qkv_ptr, tokens, h, self.GDN_QKV_DIM)
        self._gemm(norm_ptr, self._w[p + "in_proj_z.weight"], z_ptr, tokens, h, self.GDN_Z_DIM)
        self._gemm(norm_ptr, self._w[p + "in_proj_b.weight"], b_ptr, tokens, h, self.GDN_HEADS)
        self._gemm(norm_ptr, self._w[p + "in_proj_a.weight"], a_ptr, tokens, h, self.GDN_HEADS)

        if self._zero_conv_state is None:
            # conv-state layout is (channels, kernel_size) with slot 0 unused
            self._zero_conv_state = _malloc_committed(self.GDN_QKV_DIM * 4 * 4)
            self._misc_buffers.append(self._zero_conv_state)
        # the prefill conv kernel chains segments through conv_state, so it
        # must be re-zeroed before every layer invocation (GPU-side scale
        # by 0.0 — an H2D upload here cost a host alloc + copy per layer)
        err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
            _P(self._zero_conv_state.ptr), _P(self._zero_conv_state.ptr),
            _F(0.0), _I(self.GDN_QKV_DIM * 4), _S(0),
        )
        self._check(err, "zero conv state")
        qwen35_linear_attn_conv_prefill_f32(
            qkv_ptr,
            self._zero_conv_state.ptr,
            self._w[p + "conv1d.weight"],
            conv_ptr,
            tokens,
            self.GDN_QKV_DIM,
            4,
            stream=0,
            library=self.conv_library,
            runtime=self.runtime,
        )
        err = self._k(
            "hipengine_evie_gdn_l2norm_scale_repeat_f32",
            [_P, _P, _P, _F, _I, _I, _I, _I, _I, _S],
        )(
            _P(conv_ptr), _P(q_ptr), _P(k_ptr),
            # cluster8 applies the 1/sqrt(d_k) q scale at its output
            # (kOutputScale) instead of on the input, so feed it the
            # unscaled l2-normalized q to avoid double-scaling
            _F(
                1.0
                if self._gdn_recurrence
                is qwen35_gdn_prefill_recurrent_normalized_cluster8_f32
                else 1.0 / math.sqrt(self.GDN_HEAD_DIM)
            ),
            _I(tokens),
            _I(self.GDN_KV_HEADS), _I(self.GDN_HEAD_DIM),
            _I(self.GDN_QKV_DIM), _I(2048), _S(0),
        )
        self._check(err, "gdn l2norm repeat")
        err = self._k(
            "hipengine_evie_gdn_gates_f32",
            [_P, _P, _P, _P, _P, _P, _I, _I, _S],
        )(
            _P(b_ptr), _P(a_ptr), _P(self._w[p + "A_log"]),
            _P(self._w[p + "dt_bias"]), _P(beta_ptr), _P(decay_ptr),
            _I(tokens), _I(self.GDN_HEADS), _S(0),
        )
        self._check(err, "gdn gates")
        # v plane (tokens, 32, 128) lives at offset 4096 in each 8192-stride
        # conv row; expand it into a dense buffer for the recurrent kernel.
        v_dense = scratch.buffers["gdn_v"].ptr
        err = self._k(
            "hipengine_evie_expand_heads_f32",
            [_P, _P, _I, _I, _I, _I, _I, _I, _S],
        )(
            _P(conv_ptr), _P(v_dense), _I(tokens), _I(4096),
            _I(self.GDN_QKV_DIM), _I(self.GDN_HEADS), _I(self.GDN_HEAD_DIM),
            _I(1), _S(0),
        )
        self._check(err, "gdn v expand")
        v_ptr = v_dense
        # zero the recurrent state before the layer: persistent buffer,
        # re-zeroed each time because the recurrence mutates it in place
        n_state = self.GDN_HEADS * self.GDN_HEAD_DIM * self.GDN_HEAD_DIM
        if self._gdn_state_zero is None:
            self._gdn_state_zero = _malloc_committed(n_state * 4 + _GEMM_PAD_BYTES)
        # GPU-side re-zero (x * 0.0) avoids the 2 MB host upload per layer
        err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
            _P(self._gdn_state_zero.ptr), _P(self._gdn_state_zero.ptr),
            _F(0.0), _I(n_state), _S(0),
        )
        self._check(err, "zero gdn state")
        state_zero = self._gdn_state_zero
        self._gdn_recurrence(
            q_ptr,
            k_ptr,
            v_ptr,
            beta_ptr,
            decay_ptr,
            state_zero.ptr,
            gdn_out,
            tokens,
            self.GDN_HEADS,
            self.GDN_HEAD_DIM,
            self.GDN_HEAD_DIM,
            stream=0,
            library=self.gdn_library,
            runtime=self.runtime,
        )
        err = self._k(
            "hipengine_evie_gdn_rmsnorm_gate_f32",
            [_P, _P, _P, _P, _I, _I, _F, _S],
        )(
            _P(gdn_out), _P(z_ptr), _P(self._w[p + "norm.weight"]),
            _P(gdn_normed), _I(tokens * self.GDN_HEADS), _I(self.GDN_HEAD_DIM),
            _F(1e-6), _S(0),
        )
        self._check(err, "gdn rmsnorm gate")
        self._gemm(gdn_normed, self._w[p + "out_proj.weight"], out_ptr, tokens, self.GDN_Z_DIM, h)

    # -- head ----------------------------------------------------------------------

    def project(
        self, hidden_ptr: int, tokens: int, scratch: _Scratch, head_dim: int = 128
    ) -> int:
        # Prefix-MRL: only the first head_dim=128 rows of the (2048, h)
        # projection are needed; they are contiguous at the weight start,
        # so GEMM straight to (tokens, 128) instead of computing 2048
        # channels and slicing (16x less head GEMM work).
        slice_ptr = scratch.buffers["head_slice"].ptr
        self._gemm(
            hidden_ptr, self._w["custom_text_proj.weight"], slice_ptr, tokens,
            self.spec.hidden_size, head_dim,
        )
        if self._head_bias is None:
            # persistent device copy of the bias prefix (a per-encode
            # GPU->CPU->GPU roundtrip was removed)
            bias_host = self._read_bias()[:head_dim].copy()
            buf = _malloc_committed(head_dim * 4 + _GEMM_PAD_BYTES)
            copy_host_to_device(buf, host_array_ptr(bias_host), head_dim * 4)
            self._head_bias = buf
            self._misc_buffers.append(buf)
        err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
            _P(slice_ptr), _P(self._head_bias.ptr), _I(tokens * head_dim), _I(head_dim), _S(0)
        )
        self._check(err, "head bias")
        err = self._k("hipengine_evie_l2norm_rows_f32", [_P, _P, _I, _I, _S])(
            _P(slice_ptr), _P(slice_ptr), _I(tokens), _I(head_dim), _S(0)
        )
        self._check(err, "head l2norm")
        return slice_ptr

    def _read_bias(self) -> np.ndarray:
        return self._to_host(self._w["custom_text_proj.bias"], 2048)

    # -- top-level API ---------------------------------------------------------------

    def _scratch_for(self, tokens: int) -> _Scratch:
        # bucket to 64-token multiples so a stream of distinct input
        # lengths does not grow the cache unboundedly
        tokens = max(1, (tokens + 63) & ~63)
        if tokens in self._scratch:
            return self._scratch[tokens]
        while len(self._scratch) >= 4:
            oldest = next(iter(self._scratch))
            self._scratch.pop(oldest).free()
        h = self.spec.hidden_size
        inter = self.spec.intermediate_size

        def f32(*shape: int) -> DeviceBuffer:
            count = 1
            for dim in shape:
                count *= dim
            return _malloc_committed(count * 4 + _GEMM_PAD_BYTES)

        bufs = {
            "x": f32(tokens, h),
            "norm": f32(tokens, h),
            "attn_out": f32(tokens, h),
            "qkv_full": f32(tokens, 8192),
            "q_full": f32(tokens, 16, 256),
            "gate_full": f32(tokens, 16, 256),
            "k_full": f32(tokens, 4, 256),
            "v_full": f32(tokens, 4, 256),
            "k_rep": f32(tokens, 16, 256),
            "v_rep": f32(tokens, 16, 256),
            "heads_out": f32(tokens, 16, 256),
            "qkv_gdn": f32(tokens, self.GDN_QKV_DIM),
            "z_gdn": f32(tokens, self.GDN_Z_DIM),
            "a_gdn": f32(tokens, self.GDN_HEADS),
            "b_gdn": f32(tokens, self.GDN_HEADS),
            "conv_out": f32(tokens, self.GDN_QKV_DIM),
            "gdn_q": f32(tokens, self.GDN_HEADS, self.GDN_HEAD_DIM),
            "gdn_k": f32(tokens, self.GDN_HEADS, self.GDN_HEAD_DIM),
            "gdn_v": f32(tokens, self.GDN_HEADS, self.GDN_HEAD_DIM),
            "gdn_beta": f32(tokens, self.GDN_HEADS),
            "gdn_decay": f32(tokens, self.GDN_HEADS),
            "gdn_out": f32(tokens, self.GDN_Z_DIM),
            "gdn_normed": f32(tokens, self.GDN_Z_DIM),
            "gate_proj": f32(tokens, inter),
            "up_proj": f32(tokens, inter),
            "head_out": f32(tokens, 2048),
            "head_slice": f32(tokens, 128),
            "vx": f32(tokens, 1024),
            "vx2": f32(tokens, 1024),
            "vqkv": f32(tokens, 3072),
            "vnorm": f32(tokens, 1024),
            "vmlp": f32(tokens, 4096),
            "vout": f32(tokens, 1024),
            "merged_in": f32(tokens, 4096),
        }
        scratch = _Scratch(tokens=tokens, buffers=bufs)
        self._scratch[tokens] = scratch
        return scratch

    def encode_query(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Encode a text query; returns (tokens, head_dim) embeddings."""

        from hipengine.kernels.cpu_reference.evie import lm_rope_positions

        spec = self.spec
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        mask = attention_mask.reshape(-1).astype(bool)
        if not mask.all():
            # positions are valid-token-ordered; drop padding so the
            # execution sequence matches (otherwise rope misaligns and
            # the positions table reads out of bounds)
            ids = ids[mask]
        positions = lm_rope_positions(ids, np.ones(len(ids), dtype=attention_mask.dtype), np.zeros((0, 3), dtype=int), spec)
        scratch = self._scratch_for(len(ids))
        hidden_ptr = self.text_forward(ids, positions, scratch)
        emb_ptr = self.project(hidden_ptr, len(ids), scratch)
        emb = self._to_host(emb_ptr, len(ids) * 128).reshape(len(ids), 128)
        self._release_call_buffers()
        return emb

    def encode_document(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        pixel_values: np.ndarray,
        image_grid_thw: np.ndarray,
    ) -> np.ndarray:
        """Encode one document page; returns (tokens, head_dim) embeddings."""

        from hipengine.kernels.cpu_reference.evie import lm_rope_positions

        spec = self.spec
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        mask = attention_mask.reshape(-1).astype(bool)
        if not mask.all():
            ids = ids[mask]
        grid = np.asarray(image_grid_thw, dtype=int)
        image_tokens = int((ids == spec.image_token_id).sum())
        n_merged = image_tokens  # one merged token per image pad token
        scratch = self._scratch_for(max(len(ids), len(pixel_values)))
        visual = self.vision_forward(
            np.asarray(pixel_values, dtype=np.float32), grid, scratch
        )
        positions = lm_rope_positions(
            ids, np.ones(len(ids), dtype=attention_mask.dtype), grid, spec
        )
        hidden_ptr = self.text_forward(ids, positions, scratch, visual_ptr=visual.ptr)
        emb_ptr = self.project(hidden_ptr, len(ids), scratch)
        emb = self._to_host(emb_ptr, len(ids) * 128).reshape(len(ids), 128)
        self._release_call_buffers()
        return emb


def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
    sims = query @ doc.T
    return float(sims.max(axis=1).sum())
