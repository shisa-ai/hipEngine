"""GPU runtime for the VibeVoice-ASR audio front-end (bf16, fp32 accumulate).

Runs both causal tokenizer encoders, the recorded-noise acoustic sampling,
and the two connector paths on HIP. Kernels come from the gfx1100 kernel
tree (built per-arch: gfx1100 and gfx1151); FFN and connector matmuls reuse
the dense prefill GEMM.

Data layout is row-major ``(rows = positions, cols = channels)`` bf16.
Causal conv chunk carry uses one prefix buffer per conv layer holding the
previous chunk's tail rows; a full pass uses zero prefixes.

Batching: one recording per forward (the multi-recording padded batch is a
later serving feature). Acoustic sampling noise must be supplied by the
caller (recorded fixtures for parity; production RNG is tracked for the
serving milestone).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc, copy_host_array_to_device, copy_device_to_host, host_array_ptr
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.cpu_reference.vibevoice_asr import (
    VibevoiceConnectorWeights,
    VibevoiceTokenizerEncoderSpec,
    VibevoiceTokenizerEncoderWeights,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_prefill_gemm_out_bf16
from hipengine.kernels.hip_gfx1100.vibevoice.encoder import (
    build_vibevoice_encoder,
    conv_rows_out,
    f32_to_bf16_bits,
    transpose_conv_weight_t,
    vv_add_bias_bf16,
    vv_add_scaled_noise_bf16,
    vv_conv_gemm_bf16,
    vv_depthwise_conv_bf16,
    vv_gelu_bf16,
    vv_rmsnorm_bf16,
    vv_scale_residual_bf16,
)


def _upload_u16(host_u16: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(host_u16)
    buffer = malloc(array.nbytes)
    copy_host_array_to_device(buffer, array)
    return buffer


def _alloc_u16(count: int) -> DeviceBuffer:
    return malloc(count * 2)


def _zeros_u16(count: int) -> DeviceBuffer:
    """Allocate a zero-filled bf16 buffer (causal-conv left-pad buffers)."""
    return _upload_u16(np.zeros(count, dtype=np.uint16))


@dataclass
class _DeviceBlock:
    norm_w: DeviceBuffer
    conv_w: DeviceBuffer
    conv_b: DeviceBuffer
    gamma: DeviceBuffer
    ffn_norm_w: DeviceBuffer
    ffn_gamma: DeviceBuffer
    ffn_w1: DeviceBuffer
    ffn_b1: DeviceBuffer
    ffn_w2: DeviceBuffer
    ffn_b2: DeviceBuffer
    width: int


@dataclass
class _DeviceStage:
    conv_w_t: DeviceBuffer
    conv_b: DeviceBuffer
    stride: int
    k_len: int
    c_in: int
    c_out: int
    prefix: DeviceBuffer
    # prefix holds k_len - stride rows of the stage input (zeroed on full pass)


class VibevoiceFrontendRuntime:
    """Device-resident weights for both encoders + connectors."""

    def __init__(
        self,
        acoustic_spec: VibevoiceTokenizerEncoderSpec,
        acoustic_weights: VibevoiceTokenizerEncoderWeights,
        semantic_spec: VibevoiceTokenizerEncoderSpec,
        semantic_weights: VibevoiceTokenizerEncoderWeights,
        acoustic_connector: VibevoiceConnectorWeights,
        semantic_connector: VibevoiceConnectorWeights,
        *,
        library: ctypes.CDLL | None = None,
    ) -> None:
        self.runtime = get_hip_runtime()
        self.library = library or build_vibevoice_encoder()
        self.text_hidden = int(acoustic_connector.fc1_weight.shape[0])
        self._buffers: list[DeviceBuffer] = []
        self._ones_gamma: DeviceBuffer | None = None

        self.encoders: dict[str, list[_DeviceStage]] = {}
        self.blocks: dict[str, list[list[_DeviceBlock]]] = {}
        self.heads: dict[str, tuple[DeviceBuffer, DeviceBuffer]] = {}
        self.stems: dict[str, tuple[DeviceBuffer, DeviceBuffer]] = {}
        self.specs: dict[str, VibevoiceTokenizerEncoderSpec] = {
            "acoustic": acoustic_spec,
            "semantic": semantic_spec,
        }

        for name, spec, weights in (
            ("acoustic", acoustic_spec, acoustic_weights),
            ("semantic", semantic_spec, semantic_weights),
        ):
            self._build_encoder(name, spec, weights)

        self.connectors: dict[str, tuple[DeviceBuffer, ...]] = {}
        for tok, conn in (("acoustic", acoustic_connector), ("semantic", semantic_connector)):
            bufs = tuple(
                self._keep(_upload_u16(f32_to_bf16_bits(np.asarray(w).reshape(-1))))
                for w in (
                    conn.fc1_weight, conn.fc1_bias, conn.norm_weight,
                    conn.fc2_weight, conn.fc2_bias,
                )
            )
            self.connectors[tok] = bufs

    # ------------------------------------------------------------------
    def _keep(self, buf: DeviceBuffer) -> DeviceBuffer:
        self._buffers.append(buf)
        return buf

    def _build_encoder(self, name: str, spec: VibevoiceTokenizerEncoderSpec, weights: VibevoiceTokenizerEncoderWeights) -> None:
        stem_w = self._keep(_upload_u16(transpose_conv_weight_t(weights.stem_conv_weight)))
        stem_b = self._keep(_upload_u16(f32_to_bf16_bits(weights.stem_conv_bias)))
        self.stems[name] = (stem_w, stem_b)

        stages: list[_DeviceStage] = []
        block_idx = 0
        blocks_per_stage: list[list[_DeviceBlock]] = []
        c_in = spec.num_filters
        # stem blocks at width num_filters
        stem_blocks = []
        for _ in range(spec.depths[0]):
            stem_blocks.append(self._upload_block(weights.blocks[block_idx], spec.num_filters))
            block_idx += 1
        blocks_per_stage.append(stem_blocks)

        for s, ratio in enumerate(spec.ratios):
            c_out = c_in * 2
            k_len = 2 * ratio
            prefix_rows = k_len - ratio
            stages.append(
                _DeviceStage(
                    conv_w_t=self._keep(_upload_u16(transpose_conv_weight_t(weights.stage_conv_weights[s]))),
                    conv_b=self._keep(_upload_u16(f32_to_bf16_bits(weights.stage_conv_biases[s]))),
                    stride=ratio,
                    k_len=k_len,
                    c_in=c_in,
                    c_out=c_out,
                    prefix=self._keep(_upload_u16(np.zeros(prefix_rows * c_in, dtype=np.uint16))),
                )
            )
            stage_blocks = []
            for _ in range(spec.depths[s + 1]):
                stage_blocks.append(self._upload_block(weights.blocks[block_idx], c_out))
                block_idx += 1
            blocks_per_stage.append(stage_blocks)
            c_in = c_out

        head_w = self._keep(_upload_u16(transpose_conv_weight_t(weights.head_conv_weight)))
        head_b = self._keep(_upload_u16(f32_to_bf16_bits(weights.head_conv_bias)))
        self.heads[name] = (head_w, head_b)
        self.encoders[name] = stages
        self.blocks[name] = blocks_per_stage

    def _upload_block(self, block, width: int) -> _DeviceBlock:
        return _DeviceBlock(
            norm_w=self._keep(_upload_u16(f32_to_bf16_bits(block["norm_weight"]))),
            conv_w=self._keep(_upload_u16(f32_to_bf16_bits(block["conv_weight"].reshape(width, -1)))),
            conv_b=self._keep(_upload_u16(f32_to_bf16_bits(block["conv_bias"]))),
            gamma=self._keep(_upload_u16(f32_to_bf16_bits(block["gamma"]))),
            ffn_norm_w=self._keep(_upload_u16(f32_to_bf16_bits(block["ffn_norm_weight"]))),
            ffn_gamma=self._keep(_upload_u16(f32_to_bf16_bits(block["ffn_gamma"]))),
            ffn_w1=self._keep(_upload_u16(f32_to_bf16_bits(block["ffn_linear1_weight"].reshape(-1)))),
            ffn_b1=self._keep(_upload_u16(f32_to_bf16_bits(block["ffn_linear1_bias"]))),
            ffn_w2=self._keep(_upload_u16(f32_to_bf16_bits(block["ffn_linear2_weight"].reshape(-1)))),
            ffn_b2=self._keep(_upload_u16(f32_to_bf16_bits(block["ffn_linear2_bias"]))),
            width=width,
        )

    def _ones(self) -> DeviceBuffer:
        if self._ones_gamma is None:
            self._ones_gamma = self._keep(_upload_u16(np.full(self.text_hidden, 0x3F80, dtype=np.uint16)))
        return self._ones_gamma

    # ------------------------------------------------------------------
    def _encoder_forward(
        self,
        tok: str,
        pcm_u16: DeviceBuffer,
        num_samples: int,
        *,
        chunk_tail: dict[str, np.ndarray] | None = None,
    ) -> tuple[DeviceBuffer, int]:
        """One full pass; returns latents buffer (frames, hidden) and frames."""
        spec = self.specs[tok]
        stages = self.encoders[tok]
        blocks = self.blocks[tok]

        rows = num_samples
        width = 1
        x = pcm_u16
        zero_pad = _zeros_u16((spec.kernel_size - 1) * 1)

        stem_w, stem_b = self.stems[tok]
        rows_out = rows
        x_stem = _alloc_u16(rows_out * spec.num_filters)
        vv_conv_gemm_bf16(
            zero_pad.ptr, x.ptr, stem_w.ptr, stem_b.ptr, x_stem.ptr,
            spec.kernel_size - 1, rows_out, 1, spec.num_filters, spec.kernel_size, 1,
            library=self.library, runtime=self.runtime,
        )
        free(zero_pad)
        x = x_stem
        width = spec.num_filters
        for block in blocks[0]:
            x = self._run_block(x, rows_out, block, prefix_rows=0, prefix_ptr=0)
        for s, stage in enumerate(stages):
            prefix_rows = stage.k_len - stage.stride
            # full pass: zero left-pad (the persistent chunk-carry buffer is
            # only used by the chunked streaming path, a later milestone)
            prefix = _zeros_u16(prefix_rows * stage.c_in)
            rows_out = conv_rows_out(prefix_rows, rows_out, stage.k_len, stage.stride)
            x_next = _alloc_u16(rows_out * stage.c_out)
            vv_conv_gemm_bf16(
                prefix.ptr, x.ptr, stage.conv_w_t.ptr, stage.conv_b.ptr, x_next.ptr,
                prefix_rows, rows_out, stage.c_in, stage.c_out, stage.k_len, stage.stride,
                library=self.library, runtime=self.runtime,
            )
            free(prefix)
            free(x)
            x = x_next
            width = stage.c_out
            for block in blocks[s + 1]:
                x = self._run_block(x, rows_out, block, prefix_rows=0, prefix_ptr=0)
        head_w, head_b = self.heads[tok]
        prefix_rows = spec.kernel_size - 1
        prefix = _zeros_u16(prefix_rows * width)
        frames = conv_rows_out(prefix_rows, rows_out, spec.kernel_size, 1)
        latents = _alloc_u16(frames * spec.hidden_size)
        vv_conv_gemm_bf16(
            prefix.ptr, x.ptr, head_w.ptr, head_b.ptr, latents.ptr,
            prefix_rows, frames, width, spec.hidden_size, spec.kernel_size, 1,
            library=self.library, runtime=self.runtime,
        )
        free(prefix)
        free(x)
        return latents, frames

    def _run_block(
        self,
        x: DeviceBuffer,
        rows: int,
        block: _DeviceBlock,
        *,
        prefix_rows: int,
        prefix_ptr: int,
    ) -> DeviceBuffer:
        width = block.width
        total = rows * width
        normed = _alloc_u16(total)
        vv_rmsnorm_bf16(x.ptr, block.norm_w.ptr, normed.ptr, rows, width, 1e-5,
                        library=self.library, runtime=self.runtime)
        mixed = _alloc_u16(total)
        if prefix_ptr:
            vv_depthwise_conv_bf16(
                prefix_ptr, normed.ptr, x.ptr, block.conv_w.ptr, block.conv_b.ptr,
                block.gamma.ptr, mixed.ptr, prefix_rows, rows, width, 7,
                library=self.library, runtime=self.runtime,
            )
        else:
            zero = _zeros_u16(6 * width)
            vv_depthwise_conv_bf16(
                zero.ptr, normed.ptr, x.ptr, block.conv_w.ptr, block.conv_b.ptr,
                block.gamma.ptr, mixed.ptr, 6, rows, width, 7,
                library=self.library, runtime=self.runtime,
            )
            free(zero)
        free(normed)
        free(x)
        # ffn
        ffn_normed = _alloc_u16(total)
        vv_rmsnorm_bf16(mixed.ptr, block.ffn_norm_w.ptr, ffn_normed.ptr, rows, width, 1e-5,
                         library=self.library, runtime=self.runtime)
        h = _alloc_u16(rows * 4 * width)
        dense_prefill_gemm_out_bf16(
            ffn_normed.ptr, block.ffn_w1.ptr, h.ptr, rows, width, 4 * width,
            stream=0, library=None, runtime=self.runtime,
        )
        vv_add_bias_bf16(h.ptr, block.ffn_b1.ptr, h.ptr, rows * 4 * width, 4 * width,
                         library=self.library, runtime=self.runtime)
        free(ffn_normed)
        vv_gelu_bf16(h.ptr, h.ptr, rows * 4 * width, library=self.library, runtime=self.runtime)
        y = _alloc_u16(total)
        dense_prefill_gemm_out_bf16(
            h.ptr, block.ffn_w2.ptr, y.ptr, rows, 4 * width, width,
            stream=0, library=None, runtime=self.runtime,
        )
        vv_add_bias_bf16(y.ptr, block.ffn_b2.ptr, y.ptr, total, width,
                         library=self.library, runtime=self.runtime)
        free(h)
        vv_scale_residual_bf16(
            mixed.ptr, y.ptr, block.ffn_gamma.ptr, mixed.ptr, total, width,
            library=self.library, runtime=self.runtime,
        )
        free(y)
        return mixed

    # ------------------------------------------------------------------
    def forward(
        self,
        pcm: np.ndarray,
        *,
        noise: np.ndarray | None = None,
        noise_scale: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full front-end: mono PCM fp32 (samples,) -> (latents, audio embeds).

        ``latents`` are the raw acoustic latents fp32 (frames, hidden);
        ``audio embeds`` are the summed connector outputs fp32
        (frames, text_hidden). Supply the recorded acoustic sampling
        ``noise`` (frames, hidden_ac) fp32 and ``noise_scale`` (1,) fp32 to
        reproduce oracle sampling exactly; without them the acoustic path
        uses the un-sampled latents.
        """
        pcm_u16 = _upload_u16(f32_to_bf16_bits(pcm.astype(np.float32).reshape(-1, 1)))
        lat_ac, frames = self._encoder_forward("acoustic", pcm_u16, len(pcm))
        lat_se, frames_se = self._encoder_forward("semantic", pcm_u16, len(pcm))
        free(pcm_u16)
        if frames != frames_se:
            raise RuntimeError(f"encoder frame mismatch: {frames} != {frames_se}")

        if noise is not None and noise_scale is not None:
            gn = _upload_u16(f32_to_bf16_bits(noise.reshape(-1)))
            gs = malloc(4)
            copy_host_array_to_device(gs, np.asarray(noise_scale, dtype=np.float32).reshape(-1))
            sampled = _alloc_u16(frames * self.specs["acoustic"].hidden_size)
            vv_add_scaled_noise_bf16(
                lat_ac.ptr, gs.ptr, gn.ptr, sampled.ptr, frames * self.specs["acoustic"].hidden_size,
                frames * self.specs["acoustic"].hidden_size,
                library=self.library, runtime=self.runtime,
            )
            free(lat_ac)
            free(gn)
            free(gs)
            lat_ac = sampled

        emb = self._connector_sum(lat_ac, lat_se, frames)
        free(lat_ac)
        free(lat_se)
        return emb

    def _connector_sum(self, lat_ac: DeviceBuffer, lat_se: DeviceBuffer, frames: int) -> np.ndarray:
        fc1w, fc1b, normw, fc2w, fc2b = self.connectors["acoustic"]
        h_ac = _alloc_u16(frames * self.text_hidden)
        dense_prefill_gemm_out_bf16(
            lat_ac.ptr, fc1w.ptr, h_ac.ptr, frames, self.specs["acoustic"].hidden_size, self.text_hidden,
            stream=0, library=None, runtime=self.runtime,
        )
        vv_add_bias_bf16(h_ac.ptr, fc1b.ptr, h_ac.ptr, frames * self.text_hidden, self.text_hidden,
                         library=self.library, runtime=self.runtime)
        vv_rmsnorm_bf16(h_ac.ptr, normw.ptr, h_ac.ptr, frames, self.text_hidden, 1e-6,
                        library=self.library, runtime=self.runtime)
        emb_ac = _alloc_u16(frames * self.text_hidden)
        dense_prefill_gemm_out_bf16(
            h_ac.ptr, fc2w.ptr, emb_ac.ptr, frames, self.text_hidden, self.text_hidden,
            stream=0, library=None, runtime=self.runtime,
        )
        vv_add_bias_bf16(emb_ac.ptr, fc2b.ptr, emb_ac.ptr, frames * self.text_hidden, self.text_hidden,
                         library=self.library, runtime=self.runtime)
        free(h_ac)

        fc1w, fc1b, normw, fc2w, fc2b = self.connectors["semantic"]
        h_se = _alloc_u16(frames * self.text_hidden)
        dense_prefill_gemm_out_bf16(
            lat_se.ptr, fc1w.ptr, h_se.ptr, frames, self.specs["semantic"].hidden_size, self.text_hidden,
            stream=0, library=None, runtime=self.runtime,
        )
        vv_add_bias_bf16(h_se.ptr, fc1b.ptr, h_se.ptr, frames * self.text_hidden, self.text_hidden,
                         library=self.library, runtime=self.runtime)
        vv_rmsnorm_bf16(h_se.ptr, normw.ptr, h_se.ptr, frames, self.text_hidden, 1e-6,
                        library=self.library, runtime=self.runtime)
        emb_se = _alloc_u16(frames * self.text_hidden)
        dense_prefill_gemm_out_bf16(
            h_se.ptr, fc2w.ptr, emb_se.ptr, frames, self.text_hidden, self.text_hidden,
            stream=0, library=None, runtime=self.runtime,
        )
        vv_add_bias_bf16(emb_se.ptr, fc2b.ptr, emb_se.ptr, frames * self.text_hidden, self.text_hidden,
                         library=self.library, runtime=self.runtime)
        free(h_se)

        ones = self._ones()
        vv_scale_residual_bf16(
            emb_ac.ptr, emb_se.ptr, ones.ptr, emb_ac.ptr, frames * self.text_hidden, self.text_hidden,
            library=self.library, runtime=self.runtime,
        )
        free(emb_se)
        host = np.empty(frames * self.text_hidden, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), emb_ac)
        free(emb_ac)
        return (host.astype(np.uint32) << 16).view(np.float32).reshape(frames, self.text_hidden)

    def close(self) -> None:
        for buf in self._buffers:
            free(buf)
        self._buffers.clear()
