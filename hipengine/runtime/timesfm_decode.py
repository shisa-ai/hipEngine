"""Torch-free HIP GPU decode path for TimesFM 2.5 200M.

FP16 storage / FP32 math pipeline: rocBLAS ``gemm_ex`` GEMMs take FP16
activations and weights with FP32 accumulation; the fused elementwise/norm/
RoPE/attention kernels in ``hipengine/kernels/hip_gfx1100/timesfm`` load
``__half`` and compute in FP32.  Small norm/scale vectors stay FP32.

Orchestration mirrors the NumPy CPU reference decode step-for-step.
Patch-level running stats, revin, and the AR feedback loop stay on the host
because they are O(batch x patches) scalar work; everything O(rows x hidden)
runs on device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.cpu_reference.timesfm import revin, update_running_stats


def _patch_running_stats(
    patched_inputs: np.ndarray, patched_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-patch running (mu, sigma) mirroring update_running_stats.

    Returns (B, N) arrays of the running mean/std after each patch, computed
    with cumulative sums instead of the per-patch Python loop.
    """

    is_legit = (~patched_masks).astype(np.float64)
    values = patched_inputs.astype(np.float64) * is_legit
    count = is_legit.sum(axis=-1)                      # (B, N)
    total_n = np.cumsum(count, axis=1)                  # (B, N)
    sum_v = np.cumsum(values.sum(axis=-1), axis=1)      # (B, N)
    sum_v2 = np.cumsum((values * values).sum(axis=-1), axis=1)  # (B, N)
    n_safe = np.where(total_n == 0, 1.0, total_n)
    running_mu = sum_v / n_safe
    running_var = sum_v2 / n_safe - running_mu**2
    running_var = np.where(total_n == 0, 0.0, np.clip(running_var, 0.0, None))
    return running_mu.astype(np.float32), np.sqrt(running_var).astype(np.float32)
from hipengine.kernels.hip_gfx1100.timesfm.timesfm import (
    timesfm_add_f32,
    timesfm_qkv_norm_scatter_f16,
    timesfm_rope_norm_scatter_f16,
    timesfm_k_norm_scatter_f16,
    timesfm_flash_attention_f16,
    timesfm_q_norm_transpose_f16,
    timesfm_transpose_heads_f16,
    timesfm_v_scatter_f16,
    timesfm_attention_f32,
    timesfm_bias_f32,
    timesfm_bias_swish_f32,
    timesfm_head_perdim_scale_f32,
    timesfm_head_rmsnorm_f32,
    timesfm_norm_add_f32,
    timesfm_rmsnorm_f32,
    timesfm_rope_f32,
    timesfm_scatter_kv_f32,
    timesfm_swish_f32,
)
from hipengine.loading.timesfm import TimesFMLoadedModel


@dataclass(frozen=True)
class _Buffers:
    """Reusable device scratch for one (batch, patches) decode shape.

    Activation buffers are FP16 (GEMM-adjacent); ``pos``/``num_masked``/
    ``q_offset`` are FP32/INT32 parameter inputs.
    """

    tok_in: DeviceBuffer        # B*N, 64        F16
    hidden: DeviceBuffer         # B*N, 1280      F16
    hidden2: DeviceBuffer        # B*N, 1280      F16
    embeddings: DeviceBuffer     # B*N, 1280      F16
    normed: DeviceBuffer         # B*N, 1280      F16
    qkv: DeviceBuffer            # B*N, 3840      F16
    attn_out: DeviceBuffer       # B*N, 1280      F16
    attn_res: DeviceBuffer       # B*N, 1280      F16
    ff_hidden: DeviceBuffer      # B*N, 1280      F16
    ff_out: DeviceBuffer         # B*N, 1280      F16
    layer_out: DeviceBuffer       # B*N, 1280      F16
    point_out: DeviceBuffer      # B*N, 1280      F16
    quant_scratch: DeviceBuffer  # B*N, 10240     F16
    quantile_out: DeviceBuffer   # B*N, 10240     F16
    qt: DeviceBuffer              # B*H*Q*D       F16 (batched-GEMM attn)
    scores: DeviceBuffer          # B*H*Q*S       F16
    attn_o: DeviceBuffer          # B*H*Q*D       F16
    pos: DeviceBuffer            # B*N            F32
    num_masked: DeviceBuffer     # B              I32
    q_offset: DeviceBuffer       # B              I32
    caches_k: tuple[DeviceBuffer, ...]  # per layer B*S*H*D F16
    caches_v: tuple[DeviceBuffer, ...]

    def free(self) -> None:
        for buffer in (
            self.tok_in, self.hidden, self.hidden2, self.embeddings, self.normed,
            self.qkv, self.attn_out, self.attn_res, self.ff_hidden, self.ff_out,
            self.layer_out, self.point_out, self.quant_scratch, self.quantile_out,
            self.qt, self.scores, self.attn_o,
            self.pos, self.num_masked, self.q_offset, *self.caches_k, *self.caches_v,
        ):
            free(buffer)


_GEMM_WEIGHT_NAMES = (
    "tokenizer.hidden_layer.weight",
    "tokenizer.output_layer.weight",
    "tokenizer.residual_layer.weight",
    "output_projection_point.hidden_layer.weight",
    "output_projection_point.output_layer.weight",
    "output_projection_point.residual_layer.weight",
    "output_projection_quantiles.hidden_layer.weight",
    "output_projection_quantiles.output_layer.weight",
    "output_projection_quantiles.residual_layer.weight",
)


def _fake_buffer(ptr: int, nbytes: int) -> DeviceBuffer:
    """View wrapper for D2H reads of loader-owned weight allocations."""

    return DeviceBuffer(ptr=ptr, nbytes=nbytes)


class TimesFMGPUDecoder:
    """Resident-weights TimesFM 2.5 decoder running on the HIP device.

    ``precision="fp16"`` (default) is the production path: FP16 storage with
    FP32 GEMM accumulation and FP32 kernel math; gated by the calibrated
    TimesFM production tolerance (max <= 2%, mean <= 0.5% of per-series
    signal scale vs the FP32 oracle).  ``precision="fp32"`` is the strict
    exact-parity fallback (rocBLAS SGEMM, FP32 buffers, tolerance atol 5e-4).
    """

    def __init__(
        self,
        loaded: TimesFMLoadedModel,
        *,
        rocblas: Rocblas | None = None,
        precision: str = "fp16",
    ):
        if precision not in ("fp16", "fp32"):
            raise ValueError("precision must be 'fp16' or 'fp32'")
        self.precision = precision
        self.spec = loaded.spec
        self.rocblas = rocblas or Rocblas.load()
        self._loaded = loaded
        self._buffers: dict[tuple[int, int, int], _Buffers] = {}
        half_dim = self.spec.head_dim // 2
        timescale = (
            1.0 * (10_000.0 / 1.0) ** (2.0 * np.arange(half_dim, dtype=np.float32) / self.spec.head_dim)
        ).astype(np.float32)
        self._timescale = malloc(timescale.nbytes)
        copy_host_to_device(self._timescale, host_array_ptr(timescale))
        self._fp16_weights: dict[str, DeviceBuffer] = {}
        raw = {name: alloc.buffer.ptr for name, alloc in loaded.weights.tensors.items()}
        info = {name: alloc for name, alloc in loaded.weights.tensors.items()}
        gemm_names = list(_GEMM_WEIGHT_NAMES)
        for layer in range(self.spec.num_hidden_layers):
            gemm_names.extend(
                (
                    f"stacked_xf.{layer}.attn.qkv_proj.weight",
                    f"stacked_xf.{layer}.attn.out.weight",
                    f"stacked_xf.{layer}.ff0.weight",
                    f"stacked_xf.{layer}.ff1.weight",
                )
            )
        gemm_ptr = raw  # fp32 mode: GEMMs read the loader's FP32 buffers
        if self.precision == "fp16":
            gemm_ptr = {}
            for name in gemm_names:
                source = info[name]
                host = np.empty(int(np.prod(source.source.shape)), dtype=np.float32)
                copy_device_to_host(
                    host_array_ptr(host),
                    _fake_buffer(source.buffer.ptr, source.buffer.nbytes),
                )
                host16 = np.ascontiguousarray(host.astype(np.float16))
                target = malloc(host16.nbytes)
                copy_host_to_device(target, host_array_ptr(host16))
                self._fp16_weights[name] = target
                gemm_ptr[name] = target.ptr
        self._w = {
            "tokenizer_hidden": gemm_ptr["tokenizer.hidden_layer.weight"],
            "tokenizer_hidden_b": raw["tokenizer.hidden_layer.bias"],
            "tokenizer_out": gemm_ptr["tokenizer.output_layer.weight"],
            "tokenizer_out_b": raw["tokenizer.output_layer.bias"],
            "tokenizer_res": gemm_ptr["tokenizer.residual_layer.weight"],
            "tokenizer_res_b": raw["tokenizer.residual_layer.bias"],
            "point_hidden": gemm_ptr["output_projection_point.hidden_layer.weight"],
            "point_out": gemm_ptr["output_projection_point.output_layer.weight"],
            "point_res": gemm_ptr["output_projection_point.residual_layer.weight"],
            "q_hidden": gemm_ptr["output_projection_quantiles.hidden_layer.weight"],
            "q_out": gemm_ptr["output_projection_quantiles.output_layer.weight"],
            "q_res": gemm_ptr["output_projection_quantiles.residual_layer.weight"],
        }
        self._layers = [
            {
                "qkv": gemm_ptr[f"stacked_xf.{i}.attn.qkv_proj.weight"],
                "out": gemm_ptr[f"stacked_xf.{i}.attn.out.weight"],
                "q_ln": raw[f"stacked_xf.{i}.attn.query_ln.scale"],
                "k_ln": raw[f"stacked_xf.{i}.attn.key_ln.scale"],
                "perdim": raw[f"stacked_xf.{i}.attn.per_dim_scale.per_dim_scale"],
                "ff0": gemm_ptr[f"stacked_xf.{i}.ff0.weight"],
                "ff1": gemm_ptr[f"stacked_xf.{i}.ff1.weight"],
                "pre_attn": raw[f"stacked_xf.{i}.pre_attn_ln.scale"],
                "post_attn": raw[f"stacked_xf.{i}.post_attn_ln.scale"],
                "pre_ff": raw[f"stacked_xf.{i}.pre_ff_ln.scale"],
                "post_ff": raw[f"stacked_xf.{i}.post_ff_ln.scale"],
            }
            for i in range(self.spec.num_hidden_layers)
        ]

    def close(self) -> None:
        for buffers in self._buffers.values():
            buffers.free()
        self._buffers.clear()
        free(self._timescale)
        for buffer in self._fp16_weights.values():
            free(buffer)
        self._fp16_weights.clear()

    # -- buffer management ---------------------------------------------------

    def _buffers_for(self, batch: int, patches: int, cache_size: int) -> _Buffers:
        key = (batch, patches, cache_size)
        if key in self._buffers:
            return self._buffers[key]
        h = self.spec.hidden_size
        head_vectors = self.spec.num_attention_heads * self.spec.head_dim
        itemsize = 2 if self.precision == "fp16" else 4
        f16 = lambda n: malloc(n * itemsize)  # noqa: E731
        caches_k = tuple(f16(batch * cache_size * head_vectors) for _ in range(self.spec.num_hidden_layers))
        caches_v = tuple(f16(batch * cache_size * head_vectors) for _ in range(self.spec.num_hidden_layers))
        # The batched attention GEMMs sweep the whole cache including the
        # not-yet-written AR slots; masked weights are zero but 0 * garbage
        # is NaN when the allocator hands back dirty pages, so zero-init.
        runtime = get_hip_runtime()
        for cache in (*caches_k, *caches_v):
            runtime.memset(cache.ptr, 0, cache.nbytes)
        buffers = _Buffers(
            tok_in=f16(batch * patches * self.spec.tokenizer_input_dims),
            hidden=f16(batch * patches * h),
            hidden2=f16(batch * patches * h),
            embeddings=f16(batch * patches * h),
            normed=f16(batch * patches * h),
            qkv=f16(batch * patches * self.spec.qkv_size),
            attn_out=f16(batch * patches * h),
            attn_res=f16(batch * patches * h),
            ff_hidden=f16(batch * patches * h),
            ff_out=f16(batch * patches * h),
            layer_out=f16(batch * patches * h),
            point_out=f16(batch * patches * h),
            quant_scratch=f16(batch * patches * self.spec.quantile_output_dims),
            quantile_out=f16(batch * patches * self.spec.quantile_output_dims),
            qt=f16(batch * self.spec.num_attention_heads * patches * (h // self.spec.num_attention_heads)),
            scores=f16(batch * self.spec.num_attention_heads * patches * cache_size),
            attn_o=f16(batch * self.spec.num_attention_heads * patches * (h // self.spec.num_attention_heads)),
            pos=malloc(batch * patches * 4),
            num_masked=malloc(batch * 4),
            q_offset=malloc(batch * 4),
            caches_k=caches_k,
            caches_v=caches_v,
        )
        self._buffers[key] = buffers
        return buffers

    # -- GEMM helper ---------------------------------------------------------

    def _gemm(self, x_ptr: int, w_ptr: int, out_ptr: int, rows: int, fin: int, fout: int) -> None:
        if self.precision == "fp16":
            self.rocblas.gemm_ex_rowmajor_nt_fp16_compute_f32(
                x_ptr, w_ptr, out_ptr, rows=rows, in_features=fin, out_features=fout
            )
        else:
            self.rocblas.sgemm_rowmajor_nt(
                x_ptr, w_ptr, out_ptr, rows=rows, in_features=fin, out_features=fout
            )

    # -- one forward ---------------------------------------------------------

    def _forward(
        self,
        bufs: _Buffers,
        batch: int,
        n: int,
        *,
        start: int,
        cache_size: int,
        num_masked_host: np.ndarray,
        next_index_host: np.ndarray,
        skip_quantile: bool = False,
    ) -> None:
        """Run the packed tokenizer + 20 layers + output heads on device.

        With ``skip_quantile`` the quantile ResidualBlock head is omitted
        (AR feedback only consumes the point head).
        """

        h = self.spec.hidden_size
        heads = self.spec.num_attention_heads
        hd = self.spec.head_dim
        rows = batch * n
        eps = self.spec.rms_norm_eps
        patch_stride = self.spec.qkv_size
        dt = "f16" if self.precision == "fp16" else "f32"

        copy_host_to_device(bufs.num_masked, host_array_ptr(np.ascontiguousarray(num_masked_host.astype(np.int32))))
        copy_host_to_device(bufs.q_offset, host_array_ptr(np.ascontiguousarray(next_index_host.astype(np.int32))))
        pos_host = (
            np.arange(n, dtype=np.float32)[None, :] + next_index_host[:, None] - num_masked_host[:, None]
        ).astype(np.float32)
        copy_host_to_device(bufs.pos, host_array_ptr(np.ascontiguousarray(pos_host)))

        # Tokenizer ResidualBlock (biased).
        self._gemm(bufs.tok_in.ptr, self._w["tokenizer_hidden"], bufs.hidden.ptr, rows, self.spec.tokenizer_input_dims, h)
        timesfm_bias_swish_f32(bufs.hidden.ptr, self._w["tokenizer_hidden_b"], rows, h, dtype=dt)
        self._gemm(bufs.hidden.ptr, self._w["tokenizer_out"], bufs.hidden2.ptr, rows, h, h)
        timesfm_bias_f32(bufs.hidden2.ptr, self._w["tokenizer_out_b"], rows, h, dtype=dt)
        self._gemm(bufs.tok_in.ptr, self._w["tokenizer_res"], bufs.embeddings.ptr, rows, self.spec.tokenizer_input_dims, h)
        timesfm_bias_f32(bufs.embeddings.ptr, self._w["tokenizer_res_b"], rows, h, dtype=dt)
        timesfm_add_f32(bufs.hidden2.ptr, bufs.embeddings.ptr, bufs.embeddings.ptr, rows * h, dtype=dt)

        # Stacked transformer layers (ping-pong between two activation buffers).
        x = bufs.embeddings
        alt = bufs.layer_out
        for layer, w in enumerate(self._layers):
            timesfm_rmsnorm_f32(x.ptr, w["pre_attn"], bufs.normed.ptr, rows, h, eps, dtype=dt)
            self._gemm(bufs.normed.ptr, w["qkv"], bufs.qkv.ptr, rows, h, self.spec.qkv_size)
            if dt == "f16":
                # Batched-GEMM attention (head-major caches).
                timesfm_qkv_norm_scatter_f16(
                    bufs.qkv.ptr, bufs.pos.ptr, self._timescale.ptr,
                    w["q_ln"], w["k_ln"], w["perdim"],
                    batch, n, cache_size, heads, hd, patch_stride, start,
                    bufs.qt.ptr, bufs.caches_k[layer].ptr, bufs.caches_v[layer].ptr,
                )
                timesfm_flash_attention_f16(
                    bufs.qt.ptr, bufs.caches_k[layer].ptr, bufs.caches_v[layer].ptr,
                    bufs.num_masked.ptr, bufs.q_offset.ptr, bufs.attn_o.ptr,
                    batch, n, cache_size, heads, hd,
                )
                timesfm_transpose_heads_f16(
                    bufs.attn_o.ptr, bufs.attn_out.ptr, batch, n, heads, hd
                )
            else:
                timesfm_rope_f32(bufs.qkv.ptr, bufs.pos.ptr, self._timescale.ptr, batch, n, heads, hd, patch_stride, 0, dtype=dt)
                timesfm_rope_f32(bufs.qkv.ptr, bufs.pos.ptr, self._timescale.ptr, batch, n, heads, hd, patch_stride, h, dtype=dt)
                timesfm_head_rmsnorm_f32(bufs.qkv.ptr, w["q_ln"], batch, n, heads, hd, patch_stride, 0, eps, dtype=dt)
                timesfm_head_rmsnorm_f32(bufs.qkv.ptr, w["k_ln"], batch, n, heads, hd, patch_stride, h, eps, dtype=dt)
                timesfm_head_perdim_scale_f32(bufs.qkv.ptr, w["perdim"], batch, n, heads, hd, patch_stride, 0, dtype=dt)
                timesfm_scatter_kv_f32(
                    bufs.qkv.ptr, bufs.caches_k[layer].ptr, bufs.caches_v[layer].ptr,
                    batch, n, cache_size, heads, hd, patch_stride, start, dtype=dt,
                )
                timesfm_attention_f32(
                    bufs.qkv.ptr, bufs.caches_k[layer].ptr, bufs.caches_v[layer].ptr,
                    bufs.num_masked.ptr, bufs.q_offset.ptr, bufs.attn_out.ptr,
                    batch, n, cache_size, heads, hd, patch_stride, dtype=dt,
                )
            self._gemm(bufs.attn_out.ptr, w["out"], bufs.hidden.ptr, rows, h, h)
            # attn_res = post_attn_ln(attn_out) + x
            timesfm_norm_add_f32(bufs.hidden.ptr, x.ptr, w["post_attn"], bufs.attn_res.ptr, rows, h, eps, dtype=dt)
            # ff path
            timesfm_rmsnorm_f32(bufs.attn_res.ptr, w["pre_ff"], bufs.normed.ptr, rows, h, eps, dtype=dt)
            self._gemm(bufs.normed.ptr, w["ff0"], bufs.ff_hidden.ptr, rows, h, h)
            timesfm_swish_f32(bufs.ff_hidden.ptr, rows * h, dtype=dt)
            self._gemm(bufs.ff_hidden.ptr, w["ff1"], bufs.ff_out.ptr, rows, h, h)
            timesfm_norm_add_f32(bufs.ff_out.ptr, bufs.attn_res.ptr, w["post_ff"], alt.ptr, rows, h, eps, dtype=dt)
            x, alt = alt, x

        # Output heads (unbiased ResidualBlocks).
        self._gemm(x.ptr, self._w["point_hidden"], bufs.hidden.ptr, rows, h, h)
        timesfm_swish_f32(bufs.hidden.ptr, rows * h, dtype=dt)
        self._gemm(bufs.hidden.ptr, self._w["point_out"], bufs.hidden2.ptr, rows, h, h)
        self._gemm(x.ptr, self._w["point_res"], bufs.point_out.ptr, rows, h, h)
        timesfm_add_f32(bufs.hidden2.ptr, bufs.point_out.ptr, bufs.point_out.ptr, rows * h, dtype=dt)

        if not skip_quantile:
            qdim = self.spec.quantile_output_dims
            if dt == "f16":
                # Only the last patch's quantiles are consumed downstream;
                # run the quantile head on the B strided last-patch rows via
                # single-column batched GEMMs instead of all `rows` rows.
                x_last = x.ptr + (n - 1) * h * 2  # byte offset for the pointer
                row_stride = n * h  # rocBLAS strides are in elements
                rb = self.rocblas.gemm_ex_strided_batched_f16_f32acc
                # hidden = swish(W_qh @ x_last)
                rb(
                    self._w["q_hidden"], x_last, bufs.hidden.ptr,
                    m=h, n=1, k=h, lda=h, ldb=h, ldc=h,
                    stride_a=0, stride_b=row_stride, stride_c=h,
                    batch=batch, trans_a=True, trans_b=False,
                )
                timesfm_swish_f32(bufs.hidden.ptr, batch * h, dtype=dt)
                rb(
                    self._w["q_out"], bufs.hidden.ptr, bufs.quant_scratch.ptr,
                    m=qdim, n=1, k=h, lda=h, ldb=h, ldc=qdim,
                    stride_a=0, stride_b=h, stride_c=qdim,
                    batch=batch, trans_a=True, trans_b=False,
                )
                rb(
                    self._w["q_res"], x_last, bufs.quantile_out.ptr,
                    m=qdim, n=1, k=h, lda=h, ldb=h, ldc=qdim,
                    stride_a=0, stride_b=row_stride, stride_c=qdim,
                    batch=batch, trans_a=True, trans_b=False,
                )
                timesfm_add_f32(
                    bufs.quant_scratch.ptr, bufs.quantile_out.ptr,
                    bufs.quantile_out.ptr, batch * qdim, dtype=dt,
                )
            else:
                self._gemm(x.ptr, self._w["q_hidden"], bufs.hidden.ptr, rows, h, h)
                timesfm_swish_f32(bufs.hidden.ptr, rows * h, dtype=dt)
                self._gemm(bufs.hidden.ptr, self._w["q_out"], bufs.quant_scratch.ptr, rows, h, qdim)
                self._gemm(x.ptr, self._w["q_res"], bufs.quantile_out.ptr, rows, h, qdim)
                timesfm_add_f32(bufs.quant_scratch.ptr, bufs.quantile_out.ptr, bufs.quantile_out.ptr, rows * qdim, dtype=dt)

    # -- decode --------------------------------------------------------------

    def decode(
        self,
        horizon: int,
        inputs: np.ndarray,
        masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Mirror the NumPy CPU reference decode; returns host arrays."""

        spec = self.spec
        p, o = spec.patch_length, spec.horizon_length
        m = o // p
        host_dt = np.float16 if self.precision == "fp16" else np.float32
        itemsize = 2 if self.precision == "fp16" else 4
        batch, context = inputs.shape
        num_decode_steps = (horizon - 1) // o
        num_input_patches = context // p
        cache_size = num_input_patches + num_decode_steps * m
        # Scratch is sized for the prefill segment (the largest); AR steps
        # reuse the same buffers with smaller row counts.
        bufs = self._buffers_for(batch, num_input_patches, cache_size)

        patched_inputs = inputs.reshape(batch, -1, p)
        patched_masks = masks.reshape(batch, -1, p)

        context_mu, context_sigma = _patch_running_stats(patched_inputs, patched_masks)
        is_legit = (~patched_masks).astype(np.float64)
        total_legit = is_legit.sum(axis=(1, 2))
        total_sum = (patched_inputs.astype(np.float64) * is_legit).sum(axis=(1, 2))
        total_sum2 = ((patched_inputs.astype(np.float64) * is_legit) ** 2).sum(axis=(1, 2))
        n_safe = np.where(total_legit == 0, 1.0, total_legit)
        last_mu = (total_sum / n_safe).astype(np.float32)
        last_var = np.where(total_legit == 0, 0.0, np.clip(total_sum2 / n_safe - last_mu.astype(np.float64) ** 2, 0.0, None))
        last_sigma = np.sqrt(last_var).astype(np.float32)
        last_n = total_legit.astype(np.float32)

        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = np.where(patched_masks, 0.0, normed_inputs).astype(host_dt)
        tok_in = np.concatenate([normed_inputs, patched_masks.astype(host_dt)], axis=-1)
        tok_in = np.ascontiguousarray(tok_in).reshape(batch * num_input_patches, self.spec.tokenizer_input_dims)
        copy_host_to_device(bufs.tok_in, host_array_ptr(tok_in))

        num_masked_seg = np.sum(patched_masks[:, :, -1], axis=1).astype(np.int32)
        next_index = np.zeros(batch, dtype=np.int32)
        num_masked_cum = num_masked_seg.copy()

        self._forward(
            bufs, batch, num_input_patches,
            start=0, cache_size=cache_size,
            num_masked_host=num_masked_cum, next_index_host=next_index,
        )
        next_index = next_index + num_input_patches

        # D2H heads: full point backcast, but only the last quantile patch.
        point_dev = np.empty((batch * num_input_patches, spec.horizon_length * spec.quantile_heads), dtype=host_dt)
        copy_device_to_host(host_array_ptr(point_dev), bufs.point_out)
        renormed_outputs = revin(
            point_dev.astype(np.float32).reshape(batch, num_input_patches, o, -1),
            context_mu, context_sigma, reverse=True,
        )
        quant_last = np.empty((batch, spec.quantile_output_dims), dtype=host_dt)
        qrow = spec.quantile_output_dims * itemsize
        if self.precision == "fp16":
            copy_device_to_host(host_array_ptr(quant_last), bufs.quantile_out, batch * qrow)
        else:
            for b in range(batch):
                copy_device_to_host(
                    host_array_ptr(quant_last[b : b + 1]),
                    _fake_buffer(
                        bufs.quantile_out.ptr + (b * num_input_patches + num_input_patches - 1) * qrow,
                        qrow,
                    ),
                )
        renormed_quantile_spread = revin(
            quant_last.astype(np.float32).reshape(
                batch, spec.quantile_horizon_length, -1
            ),
            context_mu[:, -1], context_sigma[:, -1], reverse=True,
        )

        ar_outputs = []
        last_renormed_output = renormed_outputs[:, -1, :, spec.decode_index]

        for step in range(num_decode_steps):
            new_patched_input = last_renormed_output.reshape(batch, m, p).astype(np.float32)
            new_mask = np.zeros_like(new_patched_input, dtype=bool)

            n_host, mu_host, sigma_host = last_n, last_mu, last_sigma
            new_mus, new_sigmas = [], []
            for i in range(m):
                n_host, mu_host, sigma_host = update_running_stats(
                    n_host, mu_host, sigma_host, new_patched_input[:, i], new_mask[:, i]
                )
                new_mus.append(mu_host.copy())
                new_sigmas.append(sigma_host.copy())
            last_n, last_mu, last_sigma = n_host, mu_host, sigma_host
            new_mu = np.stack(new_mus, axis=1)
            new_sigma = np.stack(new_sigmas, axis=1)

            new_normed_input = revin(new_patched_input, new_mu, new_sigma, reverse=False)
            tok_in = np.concatenate(
                [new_normed_input.astype(host_dt), new_mask.astype(host_dt)], axis=-1
            )
            tok_in = np.ascontiguousarray(tok_in).reshape(batch * m, self.spec.tokenizer_input_dims)
            copy_host_to_device(bufs.tok_in, host_array_ptr(tok_in), batch * m * self.spec.tokenizer_input_dims * itemsize)

            start = int(next_index[0])
            self._forward(
                bufs, batch, m,
                start=start, cache_size=cache_size,
                num_masked_host=num_masked_cum, next_index_host=next_index,
                skip_quantile=True,
            )
            next_index = next_index + m

            point_step = np.empty((batch * m, o * spec.quantile_heads), dtype=host_dt)
            copy_device_to_host(host_array_ptr(point_step), bufs.point_out, batch * m * o * spec.quantile_heads * itemsize)
            new_renormed_output = revin(
                point_step.astype(np.float32).reshape(batch, m, o, -1),
                new_mu, new_sigma, reverse=True,
            )
            ar_outputs.append(new_renormed_output[:, -1, ...])
            last_renormed_output = new_renormed_output[:, -1, :, spec.decode_index]

        ar_renormed_outputs = np.stack(ar_outputs, axis=1) if num_decode_steps > 0 else None
        return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs


__all__ = ["TimesFMGPUDecoder"]
