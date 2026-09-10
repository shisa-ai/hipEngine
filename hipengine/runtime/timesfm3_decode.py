"""Torch-free HIP GPU decode path for TimesFM 3.0 500M.

FP16 storage / FP32 math pipeline mirroring the TimesFM 2.5 decoder
(``hipengine/runtime/timesfm_decode.py``): rocBLAS ``gemm_ex`` GEMMs take
FP16 activations and weights with FP32 accumulation; the fused kernels in
``hipengine/kernels/hip_gfx1100/timesfm3`` (plus the reused 2.5 family)
load ``__half`` and compute in FP32.  Small norm/scale vectors stay FP32.

TimesFM 3.0 decodes with a single non-autoregressive forward pass, so
unlike 2.5 there is no AR loop and no persistent KV cache — the sequence
attention runs as one prefill-shaped pass with ``B = batch * variates``
independent sequences, and one scratch cache pair is reused across layers.

Host/device split mirrors the NumPy CPU reference
(``hipengine/kernels/cpu_reference/timesfm3.py``): padding, detrending,
running stats, the future-covariate roll, CPM RevIN refinement, stitching,
and the final RevIN reversal are host numpy (O(batch x variates x patches)
scalar work); everything O(rows x hidden) runs on device.
"""

from __future__ import annotations

import math
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
from hipengine.kernels.cpu_reference.timesfm3 import (
    _cpm_iterative_revin_refine,
    _get_output_patch_via_roll,
    _get_running_stats,
    _linear_detrend_context,
    _revin,
    _stitch_patches,
)
from hipengine.kernels.hip_gfx1100.timesfm.timesfm import (
    timesfm_add_f32,
    timesfm_attention_f32,
    timesfm_bias_f32,
    timesfm_flash_attention_f16,
    timesfm_head_perdim_scale_f32,
    timesfm_head_rmsnorm_f32,
    timesfm_norm_add_f32,
    timesfm_rmsnorm_f32,
    timesfm_rope_f32,
    timesfm_scatter_kv_f32,
    timesfm_transpose_heads_f16,
)
from hipengine.kernels.hip_gfx1100.timesfm3.timesfm3 import (
    timesfm3_qkv_norm_scatter_f16,
    timesfm3_relu,
    timesfm3_var_attention,
)
from hipengine.loading.timesfm3 import TimesFM3LoadedModel

_RMS_EPS = float(np.finfo(np.float32).eps)


def _fake_buffer(ptr: int, nbytes: int) -> DeviceBuffer:
    """View wrapper for D2H reads of loader-owned weight allocations."""

    return DeviceBuffer(ptr=ptr, nbytes=nbytes)


@dataclass(frozen=True)
class _Buffers:
    """Reusable device scratch for one (batch, variates, patches) shape."""

    tok_in: DeviceBuffer        # B*V*N, 192     F16/F32
    embeddings: DeviceBuffer    # B*V*N, 1280
    hidden: DeviceBuffer        # B*V*N, 1280
    hidden2: DeviceBuffer       # B*V*N, 1280
    normed: DeviceBuffer        # B*V*N, 1280
    qkv: DeviceBuffer           # B*V*N, 3840
    var_q: DeviceBuffer         # B*V*N, 1280
    var_k: DeviceBuffer         # B*V*N, 1280
    var_v: DeviceBuffer         # B*V*N, 1280
    var_out: DeviceBuffer       # B*V*N, 1280
    attn_out: DeviceBuffer      # B*V*N, 1280
    attn_res: DeviceBuffer      # B*V*N, 1280
    ff_hidden: DeviceBuffer     # B*V*N, 1280
    ff_out: DeviceBuffer        # B*V*N, 1280
    layer_out: DeviceBuffer      # B*V*N, 1280
    logits: DeviceBuffer         # B*V*N, 576
    qt: DeviceBuffer            # B*V, H, N, D
    attn_o: DeviceBuffer        # B*V, H, N, D
    cache_k: DeviceBuffer       # B*V, H, N, D
    cache_v: DeviceBuffer       # B*V, H, N, D
    pos: DeviceBuffer           # B*V, N        F32
    front_masked: DeviceBuffer  # B*V           I32
    q_offset: DeviceBuffer      # B*V           I32 (zeros)

    def free(self) -> None:
        for buffer in (
            self.tok_in, self.embeddings, self.hidden, self.hidden2, self.normed,
            self.qkv, self.var_q, self.var_k, self.var_v, self.var_out,
            self.attn_out, self.attn_res, self.ff_hidden, self.ff_out,
            self.layer_out, self.logits, self.qt, self.attn_o,
            self.cache_k, self.cache_v, self.pos, self.front_masked,
            self.q_offset,
        ):
            free(buffer)


class TimesFM3GPUDecoder:
    """Resident-weights TimesFM 3.0 decoder running on the HIP device.

    ``precision="fp16"`` (default) is the production path: FP16 storage with
    FP32 GEMM accumulation and FP32 kernel math.  ``precision="fp32"`` is
    the strict parity fallback (rocBLAS SGEMM, FP32 buffers).
    """

    def __init__(
        self,
        loaded: TimesFM3LoadedModel,
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

        spec = self.spec
        half_dim = spec.head_dim // 2
        timescale = (
            1.0 * (10_000.0 / 1.0)
            ** (2.0 * np.arange(half_dim, dtype=np.float32) / spec.head_dim)
        ).astype(np.float32)
        self._timescale = malloc(timescale.nbytes)
        copy_host_to_device(self._timescale, host_array_ptr(timescale))

        raw = {name: alloc.buffer.ptr for name, alloc in loaded.weights.tensors.items()}
        info = loaded.weights.tensors
        host_dtype = np.float16 if precision == "fp16" else np.float32

        def _gemm_weight(name: str) -> int:
            source = info[name]
            host = np.empty(int(np.prod(source.source.shape)), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(host),
                _fake_buffer(source.buffer.ptr, source.buffer.nbytes),
            )
            host = host.reshape(source.source.shape)
            if precision == "fp16":
                host = np.ascontiguousarray(host.astype(host_dtype))
                target = malloc(host.nbytes)
                copy_host_to_device(target, host_array_ptr(host))
                self._fp16_weights[name] = target
                return target.ptr
            # fp32: GEMMs read the loader's resident FP32 buffers directly.
            return raw[name]

        def _read_f32(ptr: int, n: int) -> np.ndarray:
            host = np.empty(n, dtype=np.float32)
            copy_device_to_host(host_array_ptr(host), _fake_buffer(ptr, n * 4))
            return host

        def _gemm_weight_qkv(name_prefix: str) -> int:
            """Concatenate separate q/k/v projections into one [3d, d] GEMM."""

            pieces = []
            for suffix in ("query_proj", "key_proj", "value_proj"):
                source = info[f"{name_prefix}.{suffix}.weight"]
                host = np.empty(int(np.prod(source.source.shape)), dtype=np.float32)
                copy_device_to_host(
                    host_array_ptr(host),
                    _fake_buffer(source.buffer.ptr, source.buffer.nbytes),
                )
                pieces.append(host.reshape(source.source.shape))
            concat = np.ascontiguousarray(
                np.concatenate(pieces, axis=0).astype(host_dtype)
            )
            target = malloc(concat.nbytes)
            copy_host_to_device(target, host_array_ptr(concat))
            self._fp16_weights[f"{name_prefix}.__qkv_concat"] = target
            return target.ptr

        self._fp16_weights: dict[str, DeviceBuffer] = {}
        d = spec.model_dims
        self._w = {
            "tok_hidden": _gemm_weight("pre_transformer_resblock.hidden_layer.weight"),
            "tok_out": _gemm_weight("pre_transformer_resblock.output_layer.weight"),
            "tok_res": _gemm_weight("pre_transformer_resblock.residual_layer.weight"),
            "head_w": _gemm_weight("output_head.weight"),
            "head_b": raw["output_head.bias"],
        }
        hd = spec.head_dim
        qscale_factor = 1.442695041 / np.sqrt(float(hd))
        # TimesFM 3.0 attention is SDPA with scale=sqrt(head_dim): the score
        # q.k is multiplied by sqrt(D).  Both the 2.5 attention kernels we
        # reuse and the new var-attention kernel apply no implicit scaling,
        # so fold sqrt(D) into the K-side norm weight (scaling K after its
        # RMSNorm is exactly equivalent to scaling the scores).
        k_scale = np.sqrt(float(hd))
        self._layers = []
        for i in range(spec.num_layers):
            prefix = f"transformer_stack.layers.{i}"
            # Scaled K-side weight for the sequence attention.
            k_ln_h = (
                _read_f32(raw[f"{prefix}.seq_attn.key_ln.weight"], hd) * k_scale
            ).astype(np.float32)
            k_ln_buf = malloc(k_ln_h.nbytes)
            copy_host_to_device(k_ln_buf, host_array_ptr(k_ln_h))
            self._fp16_weights[f"__k_ln_scaled.{i}"] = k_ln_buf
            # Fold q_ln * factor * softplus(per_dim) into one qscale vector
            # (applied after the QK-normalize inside the fused kernel).
            q_ln_h = _read_f32(raw[f"{prefix}.seq_attn.query_ln.weight"], hd)
            per_dim_h = _read_f32(raw[f"{prefix}.seq_attn.per_dim_scale.per_dim_scale"], hd)
            qscale_h = (
                q_ln_h * qscale_factor * np.logaddexp(per_dim_h, np.zeros_like(per_dim_h))
            ).astype(np.float32)
            qscale_buf = malloc(qscale_h.nbytes)
            copy_host_to_device(qscale_buf, host_array_ptr(qscale_h))
            self._fp16_weights[f"__qscale.{i}"] = qscale_buf
            # Folded per-dim scale for the variate-attention queries.
            var_q_ln_h = _read_f32(raw[f"{prefix}.var_attn.query_ln.weight"], hd)
            var_per_dim_h = _read_f32(raw[f"{prefix}.var_attn.per_dim_scale.per_dim_scale"], hd)
            var_qscale_h = (
                var_q_ln_h * qscale_factor * np.logaddexp(var_per_dim_h, np.zeros_like(var_per_dim_h))
            ).astype(np.float32)
            var_qscale_buf = malloc(var_qscale_h.nbytes)
            copy_host_to_device(var_qscale_buf, host_array_ptr(var_qscale_h))
            self._fp16_weights[f"__var_qscale.{i}"] = var_qscale_buf
            self._layers.append({
                # q/k/v projections concatenated into one [3d, d] GEMM weight.
                "qkv": _gemm_weight_qkv(f"{prefix}.seq_attn"),
                "var_q": _gemm_weight(f"{prefix}.var_attn.query_proj.weight"),
                "var_k": _gemm_weight(f"{prefix}.var_attn.key_proj.weight"),
                "var_v": _gemm_weight(f"{prefix}.var_attn.value_proj.weight"),
                "var_out": _gemm_weight(f"{prefix}.var_attn.out_proj.weight"),
                "out": _gemm_weight(f"{prefix}.seq_attn.out_proj.weight"),
                "ff0": _gemm_weight(f"{prefix}.ff0.weight"),
                "ff1": _gemm_weight(f"{prefix}.ff1.weight"),
                "qscale": qscale_buf.ptr,
                "q_ln": raw[f"{prefix}.seq_attn.query_ln.weight"],
                "k_ln": k_ln_buf.ptr,
                "perdim": raw[f"{prefix}.seq_attn.per_dim_scale.per_dim_scale"],
                "pre_seq": raw[f"{prefix}.pre_seq_attn_ln.weight"],
                "post_seq": raw[f"{prefix}.post_seq_attn_ln.weight"],
                "pre_var": raw[f"{prefix}.pre_var_attn_ln.weight"],
                "var_qscale": var_qscale_buf.ptr,
                "var_k_ln": raw[f"{prefix}.var_attn.key_ln.weight"],
                "post_var": raw[f"{prefix}.post_var_attn_ln.weight"],
                "pre_ff": raw[f"{prefix}.pre_ff_ln.weight"],
                "post_ff": raw[f"{prefix}.post_ff_ln.weight"],
            })

    def close(self) -> None:
        for buffers in self._buffers.values():
            buffers.free()
        self._buffers.clear()
        free(self._timescale)
        for buffer in self._fp16_weights.values():
            free(buffer)
        self._fp16_weights.clear()

    # -- buffer management ---------------------------------------------------

    def _buffers_for(self, batch: int, variates: int, n: int) -> _Buffers:
        key = (batch, variates, n)
        if key in self._buffers:
            return self._buffers[key]
        spec = self.spec
        d = spec.model_dims
        rows = batch * variates * n
        itemsize = 2 if self.precision == "fp16" else 4
        f = lambda count: malloc(count * itemsize)  # noqa: E731
        head_vectors = spec.num_heads * spec.head_dim
        buffers = _Buffers(
            tok_in=f(rows * spec.tokenizer_input_dims),
            embeddings=f(rows * d),
            hidden=f(rows * d),
            hidden2=f(rows * d),
            normed=f(rows * d),
            qkv=f(rows * 3 * d),
            var_q=f(rows * d),
            var_k=f(rows * d),
            var_v=f(rows * d),
            var_out=f(rows * d),
            attn_out=f(rows * d),
            attn_res=f(rows * d),
            ff_hidden=f(rows * d),
            ff_out=f(rows * d),
            layer_out=f(rows * d),
            logits=f(rows * spec.output_head_dims),
            qt=f(batch * variates * head_vectors * n),
            attn_o=f(batch * variates * head_vectors * n),
            cache_k=f(batch * variates * head_vectors * n),
            cache_v=f(batch * variates * head_vectors * n),
            pos=malloc(batch * variates * n * 4),
            front_masked=malloc(batch * variates * 4),
            q_offset=malloc(batch * variates * 4),
        )
        runtime = get_hip_runtime()
        runtime.memset(buffers.q_offset.ptr, 0, buffers.q_offset.nbytes)
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

    # -- one forward pass ----------------------------------------------------

    def _forward(
        self,
        bufs: _Buffers,
        batch: int,
        variates: int,
        n: int,
        front_masked_host: np.ndarray,
    ) -> None:
        """Tokenizer + 20 mixing layers + output head on device."""

        spec = self.spec
        d = spec.model_dims
        heads = spec.num_heads
        hd = spec.head_dim
        rows = batch * variates * n
        seq_batch = batch * variates  # independent (b, v) sequences
        dt = "f16" if self.precision == "fp16" else "f32"
        itemsize = 2 if self.precision == "fp16" else 4

        copy_host_to_device(
            bufs.front_masked,
            host_array_ptr(np.ascontiguousarray(front_masked_host.astype(np.int32))),
        )
        pos_host = np.tile(np.arange(n, dtype=np.float32), (seq_batch, 1))
        copy_host_to_device(bufs.pos, host_array_ptr(np.ascontiguousarray(pos_host)))

        # Tokenizer ResidualBlock (ReLU, no biases).
        self._gemm(bufs.tok_in.ptr, self._w["tok_hidden"], bufs.hidden.ptr, rows, spec.tokenizer_input_dims, d)
        timesfm3_relu(bufs.hidden.ptr, rows * d, dtype=dt)
        self._gemm(bufs.hidden.ptr, self._w["tok_out"], bufs.hidden2.ptr, rows, d, d)
        self._gemm(bufs.tok_in.ptr, self._w["tok_res"], bufs.embeddings.ptr, rows, spec.tokenizer_input_dims, d)
        timesfm_add_f32(bufs.hidden2.ptr, bufs.embeddings.ptr, bufs.embeddings.ptr, rows * d, dtype=dt)

        # Stacked mixing transformer layers (ping-pong activations).
        x = bufs.embeddings
        alt = bufs.layer_out
        patch_stride = 3 * d
        for w in self._layers:
            # --- sequence attention (causal, RoPE) ---
            timesfm_rmsnorm_f32(x.ptr, w["pre_seq"], bufs.normed.ptr, rows, d, _RMS_EPS, dtype=dt)
            self._gemm(bufs.normed.ptr, w["qkv"], bufs.qkv.ptr, rows, d, 3 * d)
            if dt == "f16":
                timesfm3_qkv_norm_scatter_f16(
                    bufs.qkv.ptr, bufs.pos.ptr, self._timescale.ptr,
                    w["qscale"], w["k_ln"],
                    seq_batch, n, n, heads, hd, patch_stride, 0, _RMS_EPS,
                    bufs.qt.ptr, bufs.cache_k.ptr, bufs.cache_v.ptr,
                )
                timesfm_flash_attention_f16(
                    bufs.qt.ptr, bufs.cache_k.ptr, bufs.cache_v.ptr,
                    bufs.front_masked.ptr, bufs.q_offset.ptr,
                    bufs.attn_o.ptr,
                    seq_batch, n, n, heads, hd,
                )
                timesfm_transpose_heads_f16(
                    bufs.attn_o.ptr, bufs.attn_out.ptr, seq_batch, n, heads, hd
                )
            else:
                timesfm_rope_f32(bufs.qkv.ptr, bufs.pos.ptr, self._timescale.ptr, seq_batch, n, heads, hd, patch_stride, 0, dtype=dt)
                timesfm_rope_f32(bufs.qkv.ptr, bufs.pos.ptr, self._timescale.ptr, seq_batch, n, heads, hd, patch_stride, d, dtype=dt)
                timesfm_head_rmsnorm_f32(bufs.qkv.ptr, w["q_ln"], seq_batch, n, heads, hd, patch_stride, 0, _RMS_EPS, dtype=dt)
                timesfm_head_rmsnorm_f32(bufs.qkv.ptr, w["k_ln"], seq_batch, n, heads, hd, patch_stride, d, _RMS_EPS, dtype=dt)
                timesfm_head_perdim_scale_f32(bufs.qkv.ptr, w["perdim"], seq_batch, n, heads, hd, patch_stride, 0, dtype=dt)
                timesfm_scatter_kv_f32(
                    bufs.qkv.ptr, bufs.cache_k.ptr, bufs.cache_v.ptr,
                    seq_batch, n, n, heads, hd, patch_stride, 0, dtype=dt,
                )
                timesfm_attention_f32(
                    bufs.qkv.ptr, bufs.cache_k.ptr, bufs.cache_v.ptr,
                    bufs.front_masked.ptr, bufs.q_offset.ptr,
                    bufs.attn_out.ptr,
                    seq_batch, n, n, heads, hd, patch_stride, dtype=dt,
                )
            self._gemm(bufs.attn_out.ptr, w["out"], bufs.hidden.ptr, rows, d, d)
            timesfm_norm_add_f32(bufs.hidden.ptr, x.ptr, w["post_seq"], bufs.attn_res.ptr, rows, d, _RMS_EPS, dtype=dt)

            # --- variate attention (non-causal, no RoPE) ---
            timesfm_rmsnorm_f32(bufs.attn_res.ptr, w["pre_var"], bufs.normed.ptr, rows, d, _RMS_EPS, dtype=dt)
            self._gemm(bufs.normed.ptr, w["var_q"], bufs.var_q.ptr, rows, d, d)
            self._gemm(bufs.normed.ptr, w["var_k"], bufs.var_k.ptr, rows, d, d)
            self._gemm(bufs.normed.ptr, w["var_v"], bufs.var_v.ptr, rows, d, d)
            # Fused var attention: raw GEMM outputs; the kernel applies QK
            # RMSNorm + the folded per-dim query scale and the sqrt(D) score
            # scale internally.
            timesfm3_var_attention(
                bufs.var_q.ptr, bufs.var_k.ptr, bufs.var_v.ptr,
                bufs.front_masked.ptr, w["var_qscale"], w["var_k_ln"], _RMS_EPS,
                bufs.var_out.ptr,
                batch, variates, n, heads, hd, dtype=dt,
            )
            self._gemm(bufs.var_out.ptr, w["var_out"], bufs.hidden.ptr, rows, d, d)
            timesfm_norm_add_f32(bufs.hidden.ptr, bufs.attn_res.ptr, w["post_var"], bufs.hidden2.ptr, rows, d, _RMS_EPS, dtype=dt)

            # --- feed-forward (ReLU) ---
            timesfm_rmsnorm_f32(bufs.hidden2.ptr, w["pre_ff"], bufs.normed.ptr, rows, d, _RMS_EPS, dtype=dt)
            self._gemm(bufs.normed.ptr, w["ff0"], bufs.ff_hidden.ptr, rows, d, d)
            timesfm3_relu(bufs.ff_hidden.ptr, rows * d, dtype=dt)
            self._gemm(bufs.ff_hidden.ptr, w["ff1"], bufs.ff_out.ptr, rows, d, d)
            timesfm_norm_add_f32(bufs.ff_out.ptr, bufs.hidden2.ptr, w["post_ff"], alt.ptr, rows, d, _RMS_EPS, dtype=dt)
            x, alt = alt, x

        # Output head (single biased Linear).
        self._gemm(x.ptr, self._w["head_w"], bufs.logits.ptr, rows, d, spec.output_head_dims)
        timesfm_bias_f32(bufs.logits.ptr, self._w["head_b"], rows, spec.output_head_dims, dtype=dt)

    # -- decode --------------------------------------------------------------

    def decode(
        self,
        target: np.ndarray,
        horizon: int,
        past_only_covariates: np.ndarray | None = None,
        past_future_covariates: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
        past_only_mask: np.ndarray | None = None,
        past_future_mask: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Mirror the NumPy CPU reference decode; returns (b, v, horizon, q)."""

        spec = self.spec
        p = spec.input_patch_len
        host_dt = np.float16 if self.precision == "fp16" else np.float32
        itemsize = 2 if self.precision == "fp16" else 4

        target = np.asarray(target, dtype=np.float32)
        batch_size, num_target, context = target.shape
        if past_future_covariates is not None:
            past_future_covariates = np.asarray(past_future_covariates, dtype=np.float32)
            horizon = past_future_covariates.shape[-1] - context
        if horizon <= 0:
            raise ValueError("Decode requires horizon > 0.")

        # ---- host preprocessing (mirrors timesfm3_decode) -------------------

        def _pad_left(x: np.ndarray, fill) -> np.ndarray:
            pad_width = [(0, 0)] * (x.ndim - 1) + [(ctx_padding, 0)]
            return np.pad(x, pad_width, mode="constant", constant_values=fill)

        ctx_padding = (-context) % p
        if ctx_padding > 0:
            target = _pad_left(target, 0.0)
            if mask is not None:
                mask = _pad_left(mask, True)
            if past_only_covariates is not None:
                past_only_covariates = _pad_left(past_only_covariates, 0.0)
            if past_future_covariates is not None:
                past_future_covariates = _pad_left(past_future_covariates, 0.0)
            if target_mask is not None:
                target_mask = _pad_left(target_mask, True)
            if past_only_mask is not None:
                past_only_mask = _pad_left(past_only_mask, True)
            if past_future_mask is not None:
                past_future_mask = _pad_left(past_future_mask, True)
            context = context + ctx_padding

        if mask is None:
            mask = np.zeros((batch_size, context), dtype=bool)
            if ctx_padding > 0:
                mask[:, :ctx_padding] = True

        if spec.use_stitching:
            extract_len = min(2 * p, spec.output_patch_len)
            overlap = extract_len - p
            num_forecast_patches = max(math.ceil((horizon - overlap) / p), 1)
            num_horizon_patches = num_forecast_patches + spec.rolls - 1
            padded_horizon = num_horizon_patches * p
            hor_padding = padded_horizon - horizon
        else:
            hor_padding = (-horizon) % spec.output_patch_len
            padded_horizon = horizon + hor_padding
            num_horizon_patches = padded_horizon // p
        num_context_patches = context // p

        if target_mask is None:
            target_mask = np.zeros_like(target, dtype=bool)
        target_mask = target_mask | mask[:, None, :]

        all_ctx_vals = [target]
        all_ctx_masks = [target_mask]
        num_past_only = 0
        if past_only_covariates is not None:
            num_past_only = past_only_covariates.shape[1]
            if past_only_mask is None:
                past_only_mask = np.zeros_like(past_only_covariates, dtype=bool)
            all_ctx_vals.append(past_only_covariates)
            all_ctx_masks.append(past_only_mask | mask[:, None, :])
        if past_future_covariates is not None:
            if past_future_mask is None:
                past_future_mask = np.zeros_like(past_future_covariates, dtype=bool)
            all_ctx_vals.append(past_future_covariates[..., :context])
            all_ctx_masks.append(past_future_mask[..., :context] | mask[:, None, :])

        ctx_vals = np.concatenate(all_ctx_vals, axis=1)
        ctx_masks = np.concatenate(all_ctx_masks, axis=1)

        if spec.use_linear_detrending:
            ctx_vals, m_trend, c_trend, apply_detrend = _linear_detrend_context(
                ctx_vals, ctx_masks, context, spec.linear_detrending_threshold
            )
        else:
            num_variates = ctx_vals.shape[1]
            m_trend = np.zeros((batch_size, num_variates, 1), dtype=np.float32)
            c_trend = np.zeros((batch_size, num_variates, 1), dtype=np.float32)
            apply_detrend = np.zeros((batch_size, num_variates, 1), dtype=bool)

        ctx_vals = np.where(ctx_masks, np.float32(0.0), ctx_vals)

        all_hor_vals: list[np.ndarray] = [
            np.zeros((batch_size, num_target, padded_horizon), dtype=np.float32),
            np.zeros((batch_size, num_past_only, padded_horizon), dtype=np.float32),
        ]
        all_hor_masks: list[np.ndarray] = [
            np.ones((batch_size, num_target, padded_horizon), dtype=bool),
            np.ones((batch_size, num_past_only, padded_horizon), dtype=bool),
        ]
        if past_future_covariates is not None:
            pf_future_vals = past_future_covariates[..., context : context + horizon]
            pf_future_masks = past_future_mask[..., context : context + horizon]
            if spec.use_linear_detrending:
                m_pf = m_trend[:, num_target + num_past_only :, :]
                c_pf = c_trend[:, num_target + num_past_only :, :]
                apply_detrend_pf = apply_detrend[:, num_target + num_past_only :, :]
                t_hor_pf = np.arange(1, horizon + 1, dtype=np.float32)[None, None, :]
                t_hor_pf = t_hor_pf / np.float32(context)
                pf_trend_hor = m_pf * t_hor_pf + c_pf
                pf_future_vals = np.where(
                    apply_detrend_pf, pf_future_vals - pf_trend_hor, pf_future_vals
                )
            pf_future_vals = np.where(pf_future_masks, np.float32(0.0), pf_future_vals)
            if hor_padding > 0:
                pf_future_vals = np.concatenate(
                    [pf_future_vals, np.zeros((batch_size, pf_future_vals.shape[1], hor_padding), dtype=np.float32)],
                    axis=-1,
                )
                pf_future_masks = np.concatenate(
                    [pf_future_masks, np.ones((batch_size, pf_future_masks.shape[1], hor_padding), dtype=bool)],
                    axis=-1,
                )
            all_hor_vals.append(pf_future_vals)
            all_hor_masks.append(pf_future_masks)

        hor_vals = np.concatenate(all_hor_vals, axis=1)
        hor_masks = np.concatenate(all_hor_masks, axis=1)

        all_vals = np.concatenate([ctx_vals, hor_vals], axis=-1)
        all_masks = np.concatenate([ctx_masks, hor_masks], axis=-1)

        num_variates = all_vals.shape[1]
        patch_is_target = np.zeros(
            (batch_size, num_variates, num_context_patches + num_horizon_patches), dtype=bool
        )
        patch_is_target[:, : num_target + num_past_only, :] = True

        values_bvnp = all_vals.reshape(batch_size, num_variates, -1, p)
        masks_bvnp = all_masks.reshape(batch_size, num_variates, -1, p)

        num_total_patches = num_context_patches + num_horizon_patches
        horizon_cpm_mask = np.zeros((batch_size, num_total_patches), dtype=bool)
        horizon_cpm_mask[:, num_context_patches:] = True

        # ---- _preprocess on host --------------------------------------------

        values_bvnp = np.nan_to_num(values_bvnp, nan=0.0)
        values_bvnp = np.clip(values_bvnp, -spec.value_clip, spec.value_clip)

        running_n, running_mean, running_std = _get_running_stats(values_bvnp, masks_bvnp)

        cpm_bvnp = horizon_cpm_mask[:, None, :, None]
        cpm_target_only = cpm_bvnp & patch_is_target[..., None]
        masks_cpm = masks_bvnp | cpm_target_only

        values_bvnp = _revin(values_bvnp, running_mean, running_std, reverse=False)
        values_bvnp = np.where(masks_cpm, np.float32(0.0), values_bvnp)

        # The roll runs over the RAW (clamped) values, revin'd with the same
        # running stats — mirror the reference exactly.
        values_raw = np.clip(
            np.nan_to_num(
                all_vals.reshape(batch_size, num_variates, -1, p), nan=0.0
            ),
            -spec.value_clip, spec.value_clip,
        )
        values_fcov, wrap_mask = _get_output_patch_via_roll(values_raw, spec.rolls)
        values_fcov = _revin(values_fcov, running_mean, running_std, reverse=False)

        masks_fcov_raw, _ = _get_output_patch_via_roll(masks_cpm, spec.rolls)
        masks_fcov = masks_fcov_raw | patch_is_target[..., None] | wrap_mask
        values_fcov = np.where(masks_fcov, np.float32(0.0), values_fcov)

        values_cat = np.concatenate([values_bvnp, values_fcov], axis=-1)
        masks_cat = np.concatenate([masks_cpm, masks_fcov], axis=-1)
        resblock_input = np.concatenate([values_cat, masks_cat.astype(np.float32)], axis=-1)

        patch_mask_bvn = masks_cat.all(axis=3)
        effective_patch_mask = np.cumprod(patch_mask_bvn.astype(np.int32), axis=2).astype(bool)
        front_masked = np.sum(effective_patch_mask.astype(np.int32), axis=2).astype(np.int32).reshape(-1)

        # ---- device forward --------------------------------------------------

        n = num_total_patches
        bufs = self._buffers_for(batch_size, num_variates, n)
        rows = batch_size * num_variates * n
        tok_in = np.ascontiguousarray(resblock_input.astype(host_dt)).reshape(
            rows, spec.tokenizer_input_dims
        )
        copy_host_to_device(bufs.tok_in, host_array_ptr(tok_in))
        self._forward(bufs, batch_size, num_variates, n, front_masked)

        logits_host = np.empty((rows, spec.output_head_dims), dtype=host_dt)
        copy_device_to_host(host_array_ptr(logits_host), bufs.logits)
        raw_logits = logits_host.astype(np.float32).reshape(
            batch_size, num_variates, n, spec.output_head_dims
        )

        # ---- host post-processing (mirrors timesfm3_forward tail + decode) --

        if spec.use_iterative_cpm_revin:
            refined_mu, refined_sigma = _cpm_iterative_revin_refine(
                raw_logits,
                revin_n=running_n,
                revin_mu=running_mean,
                revin_sigma=running_std,
                patch_cpm_mask=horizon_cpm_mask,
                median_q_idx=spec.median_index,
                rolls=spec.rolls,
                patch_len=spec.input_patch_len,
                num_quantiles=len(spec.quantiles),
                value_clip=spec.value_clip,
            )
            cpm_bvn = horizon_cpm_mask[:, None, :]
            revin_mean = np.where(cpm_bvn, refined_mu, running_mean)
            revin_std = np.where(cpm_bvn, refined_sigma, running_std)
        else:
            revin_mean = running_mean
            revin_std = running_std

        revin_logits = _revin(raw_logits, revin_mean, revin_std, reverse=True)
        clipped_logits = np.clip(revin_logits, -spec.value_clip, spec.value_clip)
        o, q = spec.output_patch_len, len(spec.quantiles)
        logits = clipped_logits.reshape(batch_size, num_variates, n, o, q)

        if spec.use_stitching:
            extract_len = min(2 * p, spec.output_patch_len)
            forecast_indices = np.arange(num_forecast_patches) + (num_context_patches - 1)
            patch_preds = logits[:, :, forecast_indices, :extract_len, :]
            horizon_logits = _stitch_patches(patch_preds, p)[:, :, :horizon, :]
        else:
            num_forecast_chunks = padded_horizon // spec.output_patch_len
            forecast_indices = (
                np.arange(num_forecast_chunks) * spec.rolls + (num_context_patches - 1)
            )
            forecast_logits = logits[:, :, forecast_indices, :, :]
            horizon_logits = forecast_logits.reshape(
                batch_size, num_variates, -1, q
            )[:, :, :horizon, :]

        if spec.use_linear_detrending:
            t_forecast = np.arange(1, horizon + 1, dtype=np.float32) / np.float32(context)
            trend_forecast = (
                m_trend[:, :, 0, None] * t_forecast[None, None, :] + c_trend[:, :, 0, None]
            )
            trend_forecast = np.where(apply_detrend[:, :, 0, None], trend_forecast, np.float32(0.0))
            horizon_logits = horizon_logits + trend_forecast[:, :, :, None]

        return horizon_logits.astype(np.float32)


__all__ = ["TimesFM3GPUDecoder"]
