"""Torch-free HIP runtime for the VibeVoice-TTS acoustic waveform decoder.

Mirrors ``hipengine.kernels.cpu_reference.vibevoice_tts`` op-for-op in bf16
device storage with fp32 accumulation, driven by the registered ``vibevoice``
kernel family: pointwise convs via ``vv_conv_gemm_bf16``, the causal
streaming transposed convs via ``vv_convtr_gemm_bf16``, blocks via
``vv_rmsnorm_bf16`` / ``vv_depthwise_conv_bf16`` / ``dense_gemv_out_bf16`` /
``vv_add_bias_bf16`` / ``vv_gelu_bf16`` / ``vv_scale_residual_bf16``.

Every causal conv keeps a fixed-length left-zero-padded prefix buffer of
``kernel - 1`` rows that rolls forward across calls, so per-frame carries
equal a single-pass forward exactly (zeros are no-ops in the accumulation).
``decode_bulk`` processes many frames per launch with identical per-row
arithmetic and must stay bit-identical to per-frame streaming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import hipengine.kernels.cpu_reference.vibevoice_tts as tts_ref
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.hip_gfx1100.linear import dense_gemv
from hipengine.kernels.hip_gfx1100.vibevoice import decoder as tts_decoder
from hipengine.kernels.hip_gfx1100.vibevoice import encoder as vv_enc
from hipengine.kernels.vibevoice import resolve_vibevoice_kernels
from hipengine.loading.vibevoice_layout import f32_to_bf16_bits


def _upload_u16(host_u16: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(host_u16)
    buffer = malloc(array.nbytes)
    copy_host_array_to_device(buffer, array)
    return buffer


def _zeros_u16(count: int) -> DeviceBuffer:
    return _upload_u16(np.zeros(count, dtype=np.uint16))


@dataclass
class _PrefixCache:
    """Fixed-length left-zero-padded row cache (``prefix_rows`` x ``width``)."""

    prefix_rows: int
    width: int
    buf: DeviceBuffer
    scratch: DeviceBuffer

    @classmethod
    def create(cls, prefix_rows: int, width: int) -> "_PrefixCache":
        count = prefix_rows * width
        return cls(prefix_rows, width, _zeros_u16(count), _zeros_u16(count))

    def roll(self, x_ptr: int, rows: int, runtime) -> None:
        """After a pass over ``rows`` new rows, keep the last prefix_rows."""
        p, w = self.prefix_rows, self.width
        if p == 0 or rows <= 0:
            return
        if rows >= p:
            runtime.memcpy(
                self.buf.ptr, x_ptr + (rows - p) * w * 2, p * w * 2, MemcpyKind.DEVICE_TO_DEVICE
            )
            return
        keep = p - rows
        runtime.memcpy(
            self.scratch.ptr, self.buf.ptr + rows * w * 2, keep * w * 2, MemcpyKind.DEVICE_TO_DEVICE
        )
        runtime.memcpy(self.buf.ptr, self.scratch.ptr, keep * w * 2, MemcpyKind.DEVICE_TO_DEVICE)
        runtime.memcpy(
            self.buf.ptr + keep * w * 2, x_ptr, rows * w * 2, MemcpyKind.DEVICE_TO_DEVICE
        )

    def zero(self, runtime) -> None:
        runtime.memset(self.buf.ptr, 0, self.prefix_rows * self.width * 2)


class VibevoiceTTSDecoderGPU:
    """Streaming acoustic-tokenizer decoder on HIP (gfx11xx peer backends)."""

    def __init__(self, spec, weights, *, runtime=None) -> None:
        from hipengine.core.hip import get_hip_runtime

        self.spec = spec
        self.weights = weights
        self.runtime = runtime or get_hip_runtime()
        self.kernels = resolve_vibevoice_kernels()
        self._libraries = {
            "decoder": tts_decoder.build_vibevoice_decoder(),
            "encoder": vv_enc.build_vibevoice_encoder(),
        }
        self._weights: dict[str, DeviceBuffer] = {}
        self._scratch_map: dict[str, DeviceBuffer] = {}
        self._upload_weights()
        self._alloc_state()

    # -- setup -------------------------------------------------------------
    def _scratch_buf(self, key: str, nbytes: int) -> DeviceBuffer:
        """Grow-on-demand activation buffer (never shrinks within a run)."""
        buffer = self._scratch_map.get(key)
        if buffer is None or buffer.nbytes < nbytes:
            if buffer is not None:
                free(buffer)
            buffer = malloc(nbytes)
            self._scratch_map[key] = buffer
        return buffer

    def _upload_weights(self) -> None:
        spec, w = self.spec, self.weights
        # nn.Conv1d layout [C_out, C_in, K] -> kernel tap-major [K, C_in, C_out]
        self._weights["stem_w"] = _upload_u16(
            f32_to_bf16_bits(w.stem_conv_weight).transpose(2, 1, 0)
        )
        self._weights["stem_b"] = _upload_u16(f32_to_bf16_bits(w.stem_conv_bias.reshape(-1)))
        for i, (cw, cb) in enumerate(zip(w.convtr_weights, w.convtr_biases)):
            self._weights[f"convtr{i}_w"] = _upload_u16(
                f32_to_bf16_bits(cw).transpose(2, 0, 1)
            )
            self._weights[f"convtr{i}_b"] = _upload_u16(f32_to_bf16_bits(cb.reshape(-1)))
        self._weights["head_w"] = _upload_u16(
            f32_to_bf16_bits(w.head_conv_weight).transpose(2, 1, 0)
        )
        self._weights["head_b"] = _upload_u16(f32_to_bf16_bits(w.head_conv_bias.reshape(-1)))
        for bi, block in enumerate(w.blocks):
            width = block["norm_weight"].shape[0]
            self._weights[f"b{bi}_norm"] = _upload_u16(
                f32_to_bf16_bits(block["norm_weight"].reshape(-1))
            )
            self._weights[f"b{bi}_convw"] = _upload_u16(
                f32_to_bf16_bits(block["conv_weight"].reshape(width, -1))
            )
            self._weights[f"b{bi}_convb"] = _upload_u16(
                f32_to_bf16_bits(block["conv_bias"].reshape(-1))
            )
            self._weights[f"b{bi}_gamma"] = _upload_u16(f32_to_bf16_bits(block["gamma"].reshape(-1)))
            self._weights[f"b{bi}_ffnnorm"] = _upload_u16(
                f32_to_bf16_bits(block["ffn_norm_weight"].reshape(-1))
            )
            self._weights[f"b{bi}_l1w"] = _upload_u16(
                f32_to_bf16_bits(block["ffn_linear1_weight"])
            )  # [4w, w] torch layout
            self._weights[f"b{bi}_l1b"] = _upload_u16(
                f32_to_bf16_bits(block["ffn_linear1_bias"].reshape(-1))
            )
            self._weights[f"b{bi}_l2w"] = _upload_u16(
                f32_to_bf16_bits(block["ffn_linear2_weight"])
            )  # [w, 4w]
            self._weights[f"b{bi}_l2b"] = _upload_u16(
                f32_to_bf16_bits(block["ffn_linear2_bias"].reshape(-1))
            )
            self._weights[f"b{bi}_ffngamma"] = _upload_u16(
                f32_to_bf16_bits(block["ffn_gamma"].reshape(-1))
            )

    def _alloc_state(self) -> None:
        layout = self.spec.block_layout()
        k = self.spec.kernel_size
        self._stem_cache = _PrefixCache.create(k - 1, self.spec.dimension)
        self._head_cache = _PrefixCache.create(self.spec.last_kernel_size - 1, layout[-1][0])
        self._convtr_caches = [
            _PrefixCache.create(2 * ratio - 1, layout[i][0])
            for i, ratio in enumerate(self.spec.ratios)
        ]
        self._block_caches: dict[str, _PrefixCache] = {}

    def reset(self) -> None:
        for cache in [self._stem_cache, self._head_cache, *self._convtr_caches]:
            cache.zero(self.runtime)
        for cache in self._block_caches.values():
            cache.zero(self.runtime)

    # -- decode ------------------------------------------------------------
    def _block_forward(self, x_ptr, block_idx: int, width: int, rows: int, key: str) -> int:
        """One ConvNeXt block over row-major (rows, width); returns out ptr."""
        eps = self.spec.rms_norm_eps
        k = self.spec.kernel_size
        lib_enc = self._libraries["encoder"]
        normed = self._scratch_buf(f"{key}_norm", rows * width * 2)
        mixed = self._scratch_buf(f"{key}_mixer", rows * width * 2)
        self.kernels.vv_rmsnorm_bf16(
            x_ptr, self._weights[f"b{block_idx}_norm"].ptr, normed.ptr, rows, width, eps,
            library=lib_enc, runtime=self.runtime,
        )
        mixer_cache = self._block_caches.get(key)
        if mixer_cache is None:
            mixer_cache = _PrefixCache.create(k - 1, width)
            self._block_caches[key] = mixer_cache
        self.kernels.vv_depthwise_conv_bf16(
            mixer_cache.buf.ptr, normed.ptr, x_ptr,
            self._weights[f"b{block_idx}_convw"].ptr, self._weights[f"b{block_idx}_convb"].ptr,
            self._weights[f"b{block_idx}_gamma"].ptr, mixed.ptr,
            k - 1, rows, width, k,
            library=lib_enc, runtime=self.runtime,
        )
        mixer_cache.roll(normed.ptr, rows, self.runtime)
        # ffn: RMSNorm -> linear(4w) -> bias -> erf-GELU -> linear(w) -> bias -> gamma + residual
        ffn_normed = self._scratch_buf(f"{key}_ffnnorm", rows * width * 2)
        self.kernels.vv_rmsnorm_bf16(
            mixed.ptr, self._weights[f"b{block_idx}_ffnnorm"].ptr, ffn_normed.ptr, rows, width, eps,
            library=lib_enc, runtime=self.runtime,
        )
        h = self._scratch_buf(f"{key}_ffn", rows * 4 * width * 2)
        dense_gemv.dense_gemv_out_bf16(
            ffn_normed.ptr, self._weights[f"b{block_idx}_l1w"].ptr, h.ptr,
            rows, width, 4 * width, runtime=self.runtime,
        )
        self.kernels.vv_add_bias_bf16(
            h.ptr, self._weights[f"b{block_idx}_l1b"].ptr, h.ptr, rows * 4 * width, 4 * width,
            library=lib_enc, runtime=self.runtime,
        )
        self.kernels.vv_gelu_bf16(h.ptr, h.ptr, rows * 4 * width, library=lib_enc, runtime=self.runtime)
        h2 = self._scratch_buf(f"{key}_ffn2", rows * width * 2)
        dense_gemv.dense_gemv_out_bf16(
            h.ptr, self._weights[f"b{block_idx}_l2w"].ptr, h2.ptr,
            rows, 4 * width, width, runtime=self.runtime,
        )
        self.kernels.vv_add_bias_bf16(
            h2.ptr, self._weights[f"b{block_idx}_l2b"].ptr, h2.ptr, rows * width, width,
            library=lib_enc, runtime=self.runtime,
        )
        out = self._scratch_buf(f"{key}_out", rows * width * 2)
        # FFN residual adds to the POST-MIXER activation (mixed = conv*gamma + x),
        # matching the fork's ConvNeXt block and the CPU reference.
        self.kernels.vv_scale_residual_bf16(
            mixed.ptr, h2.ptr, self._weights[f"b{block_idx}_ffngamma"].ptr, out.ptr, rows * width, width,
            library=lib_enc, runtime=self.runtime,
        )
        return out.ptr

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """One frame: ``(dimension,)`` fp32 -> ``(hop_length,)`` bf16-rounded fp32."""
        lat = np.asarray(latent, dtype=np.float32).reshape(-1)
        if lat.shape[0] != self.spec.dimension:
            raise ValueError(f"latent must have {self.spec.dimension} dims")
        return self.decode_bulk(lat[None, :])[0]

    def decode_bulk(self, latents: np.ndarray) -> np.ndarray:
        """``(n, dimension)`` fp32 -> ``(n, hop_length)`` bf16-rounded fp32.

        Per-row arithmetic is identical to per-frame streaming; each causal
        conv's prefix rolls across the whole batch exactly as repeated
        single-frame calls would. The caller drives ``reset()`` at speech
        boundaries, matching the fork's generation loop.
        """
        lat = np.asarray(latents, dtype=np.float32)
        if lat.ndim != 2 or lat.shape[1] != self.spec.dimension:
            raise ValueError(f"latents must be (n, {self.spec.dimension})")
        n = lat.shape[0]
        hop = self.spec.hop_length
        lib_dec = self._libraries["decoder"]
        lib_enc = self._libraries["encoder"]
        spec = self.spec
        layout = spec.block_layout()

        lat_bits = f32_to_bf16_bits(np.ascontiguousarray(lat))
        x_in = self._scratch_buf("lat_in", lat_bits.nbytes)
        copy_host_array_to_device(x_in, lat_bits)

        # stem pointwise conv (stride 1, groups 1): (n, dimension) -> (n, width0)
        x = self._scratch_buf("stem_out", n * layout[0][0] * 2)
        self.kernels.vv_conv_gemm_bf16(
            self._stem_cache.buf.ptr, x_in.ptr,
            self._weights["stem_w"].ptr, self._weights["stem_b"].ptr, x.ptr,
            spec.kernel_size - 1, n, spec.dimension, layout[0][0], spec.kernel_size, 1,
            library=lib_enc, runtime=self.runtime,
        )
        self._stem_cache.roll(x_in.ptr, n, self.runtime)

        ptr = x.ptr
        for j in range(layout[0][1]):
            ptr = self._block_forward(ptr, j, layout[0][0], n, f"stage0.{j}")

        block_cursor = layout[0][1]
        rows = n
        for i, ratio in enumerate(spec.ratios):
            c_in, c_out = layout[i][0], layout[i + 1][0]
            k_len = 2 * ratio
            rows_out = rows * ratio
            cache = self._convtr_caches[i]
            y = self._scratch_buf(f"convtr{i}_out", rows_out * c_out * 2)
            tts_decoder.vv_convtr_gemm_bf16(
                cache.buf.ptr, ptr,
                self._weights[f"convtr{i}_w"].ptr, self._weights[f"convtr{i}_b"].ptr, y.ptr,
                k_len - 1, rows, rows_out, c_in, c_out, k_len, ratio,
                library=lib_dec, runtime=self.runtime,
            )
            cache.roll(ptr, rows, self.runtime)
            ptr = y.ptr
            for j in range(layout[i + 1][1]):
                ptr = self._block_forward(ptr, block_cursor + j, c_out, rows_out, f"stage{i + 1}.{j}")
            block_cursor += layout[i + 1][1]
            rows = rows_out

        # head pointwise conv (stride 1): (rows, width_last) -> (rows, 1)
        width_last = layout[-1][0]
        head_out = self._scratch_buf("head_out", rows * 2)
        self.kernels.vv_conv_gemm_bf16(
            self._head_cache.buf.ptr, ptr,
            self._weights["head_w"].ptr, self._weights["head_b"].ptr, head_out.ptr,
            spec.last_kernel_size - 1, rows, width_last, 1, spec.last_kernel_size, 1,
            library=lib_enc, runtime=self.runtime,
        )
        self._head_cache.roll(ptr, rows, self.runtime)

        raw = np.empty(rows, np.uint16)
        copy_device_to_host(raw.ctypes.data, head_out, raw.nbytes, runtime=self.runtime)
        pcm = tts_ref._round("pcm", (raw.astype(np.uint32) << 16).view(np.float32), None)
        if rows != n * hop:
            raise AssertionError(f"decoder produced {rows} samples, expected {n * hop}")
        return pcm.reshape(n, hop)

    def close(self) -> None:
        for buffer in self._scratch_map.values():
            free(buffer)
        self._scratch_map.clear()
        for buffer in self._weights.values():
            free(buffer)
        self._weights.clear()
