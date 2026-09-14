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
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    DeviceMemoryArena,
    copy_host_array_to_device,
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.cpu_reference.vibevoice_asr import (
    VibevoiceConnectorWeights,
    VibevoiceTokenizerEncoderSpec,
    VibevoiceTokenizerEncoderWeights,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_prefill_gemm_out_bf16, dense_prefill_wmma_out_bf16
from hipengine.kernels.hip_gfx1100.vibevoice.encoder import (
    vv_im2col_bf16,
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


def _prefill_gemm(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, *, runtime):
    """WMMA bulk GEMM when tile-aligned, else the naive dense prefill GEMM."""
    if out_features % 128 == 0 and in_features % 32 == 0 and rows >= 1:
        dense_prefill_wmma_out_bf16(x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
                                    stream=0, runtime=runtime)
    else:
        dense_prefill_gemm_out_bf16(x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
                                    stream=0, runtime=runtime)


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


class _ScratchPool:
    """Persistent bump-allocated scratch for one forward call.

    Per-call ``malloc``/``free`` round trips dominated front-end wall time
    (hundreds of allocations per forward); this pool hands out aligned
    views into one device allocation and resets between calls, growing
    when a larger workload arrives.
    """

    def __init__(self, capacity_bytes: int = 64 << 20) -> None:
        self._arena: DeviceMemoryArena | None = None
        self._capacity = max(int(capacity_bytes), 1 << 20)

    def reset(self, capacity_bytes: int | None = None) -> None:
        if capacity_bytes is not None:
            self._capacity = max(int(capacity_bytes), 1 << 20)
        if self._arena is not None:
            self._arena.close()
        self._arena = DeviceMemoryArena.create(self._capacity)

    def take(self, nbytes: int) -> DeviceBuffer:
        if self._arena is None:
            self.reset()
        assert self._arena is not None
        # Growing here would invalidate every live view already issued.
        return self._arena.allocate(int(nbytes))

    def take_u16(self, count: int) -> DeviceBuffer:
        return self.take(count * 2)

    def zeros_u16(self, count: int) -> DeviceBuffer:
        view = self.take_u16(count)
        zero = np.zeros(count, dtype=np.uint16)
        copy_host_array_to_device(view, zero)
        return view

    def close(self) -> None:
        if self._arena is not None:
            self._arena.close()
            self._arena = None


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
    conv_w_flat: DeviceBuffer  # [C_out, C_in*K] raw layout for im2col+GEMM
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
        self._scratch = _ScratchPool()
        self._head_zero: dict[str, DeviceBuffer] = {}

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
        # shared causal left-pad zero buffers (6 rows at stem width 1 and at
        # the widest mixer width)
        self._stem_zero = self._keep(_upload_u16(np.zeros(acoustic_spec.kernel_size - 1, dtype=np.uint16)))
        widest = int(acoustic_spec.num_filters * 2 ** len(acoustic_spec.ratios))
        self._mixer_zero = self._keep(_upload_u16(np.zeros((acoustic_spec.kernel_size - 1) * widest, dtype=np.uint16)))

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
                    conv_w_flat=self._keep(_upload_u16(f32_to_bf16_bits(
                        np.ascontiguousarray(weights.stage_conv_weights[s]).reshape(int(weights.stage_conv_weights[s].shape[0]), -1)))),
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
        head_in = int(spec.num_filters * 2 ** len(spec.ratios))
        self._head_zero[name] = self._keep(_upload_u16(np.zeros((spec.kernel_size - 1) * head_in, dtype=np.uint16)))

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
    def _prefix(self, state, key, x, rows, width, count, zero):
        """Snapshot the incoming convolution tail before its scratch is reused."""
        if state is None:
            return zero
        previous = state.get(key, np.zeros((count, width), dtype=np.uint16))
        if previous.shape != (count, width) or previous.dtype != np.uint16:
            raise ValueError(f'invalid convolution state for {key}')
        prefix = self._scratch.take_u16(count * width)
        copy_host_array_to_device(prefix, previous)
        tail_rows = min(rows, count)
        tail = np.empty((tail_rows, width),dtype=np.uint16)
        view = DeviceBuffer(x.ptr + (rows-tail_rows)*width*2, tail.nbytes)
        copy_device_to_host(host_array_ptr(tail), view)
        state[key] = np.concatenate((previous[tail_rows:],tail),axis=0)
        return prefix

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

        pool = self._scratch

        # ping-pong row buffers sized for the widest stage product
        # (rows x width is largest at the stem: samples x num_filters).
        unit = num_samples * spec.num_filters
        buf_a = pool.take_u16(unit)
        buf_b = pool.take_u16(unit)
        scratch = {
            "normed": pool.take_u16(unit),
            "mixed": pool.take_u16(unit),
            "ffn_normed": pool.take_u16(unit),
            "h": pool.take_u16(4 * unit),
            "y": pool.take_u16(unit),
        }

        rows_out = num_samples
        stem_w, stem_b = self.stems[tok]
        prefix = self._prefix(chunk_tail, 'stem', pcm_u16, rows_out, 1,
                              spec.kernel_size-1, self._stem_zero)
        vv_conv_gemm_bf16(
            prefix.ptr, pcm_u16.ptr, stem_w.ptr, stem_b.ptr, buf_a.ptr,
            spec.kernel_size - 1, rows_out, 1, spec.num_filters, spec.kernel_size, 1,
            library=self.library, runtime=self.runtime,
        )
        cur, other = buf_a, buf_b
        for b, block in enumerate(blocks[0]):
            cur, other = other, cur
            self._run_block(other, cur, rows_out, block, scratch, chunk_tail, f'0.{b}')
        width = spec.num_filters
        for s, stage in enumerate(stages):
            prefix_rows = stage.k_len - stage.stride
            input_rows = rows_out
            prefix = self._prefix(chunk_tail, f'down.{s}',cur,input_rows,stage.c_in,
                                  prefix_rows,stage.prefix)
            rows_out = conv_rows_out(prefix_rows, rows_out, stage.k_len, stage.stride)
            flat_features = stage.c_in * stage.k_len
            if flat_features % 32 == 0 and stage.c_out % 128 == 0 and rows_out >= 1:
                # im2col + bulk GEMM: weights stream once instead of once per
                # 4-row tile (dominant cost of the deep stages)
                im2col_buf = self._scratch.take_u16(rows_out * flat_features)
                conv_input = cur
                im2col_pad = prefix_rows
                if chunk_tail is not None:
                    conv_input = self._scratch.take_u16((prefix_rows+input_rows)*stage.c_in)
                    self.runtime.memcpy(conv_input.ptr,prefix.ptr,prefix_rows*stage.c_in*2,MemcpyKind.DEVICE_TO_DEVICE)
                    self.runtime.memcpy(conv_input.ptr+prefix_rows*stage.c_in*2,cur.ptr,
                                        input_rows*stage.c_in*2,MemcpyKind.DEVICE_TO_DEVICE)
                    im2col_pad = 0
                vv_im2col_bf16(
                    conv_input.ptr, im2col_buf.ptr, rows_out, stage.c_in, stage.k_len, stage.stride,
                    im2col_pad, library=self.library, runtime=self.runtime,
                )
                _prefill_gemm(
                    im2col_buf.ptr, stage.conv_w_flat.ptr, other.ptr, rows_out,
                    flat_features, stage.c_out, runtime=self.runtime,
                )
                vv_add_bias_bf16(
                    other.ptr, stage.conv_b.ptr, other.ptr, rows_out * stage.c_out, stage.c_out,
                    library=self.library, runtime=self.runtime,
                )
            else:
                vv_conv_gemm_bf16(
                    prefix.ptr, cur.ptr, stage.conv_w_t.ptr, stage.conv_b.ptr, other.ptr,
                    prefix_rows, rows_out, stage.c_in, stage.c_out, stage.k_len, stage.stride,
                    library=self.library, runtime=self.runtime,
                )
            cur, other = other, cur
            width = stage.c_out
            for b, block in enumerate(blocks[s + 1]):
                cur, other = other, cur
                self._run_block(other, cur, rows_out, block, scratch,chunk_tail,f'{s+1}.{b}')
        head_w, head_b = self.heads[tok]
        head_prefix_rows = spec.kernel_size - 1
        frames = conv_rows_out(head_prefix_rows, rows_out, spec.kernel_size, 1)
        latents = pool.take_u16(frames * spec.hidden_size)
        prefix = self._prefix(chunk_tail,'head',cur,rows_out,width,head_prefix_rows,self._head_zero[tok])
        vv_conv_gemm_bf16(
            prefix.ptr, cur.ptr, head_w.ptr, head_b.ptr, latents.ptr,
            head_prefix_rows, frames, width, spec.hidden_size, spec.kernel_size, 1,
            library=self.library, runtime=self.runtime,
        )
        return latents, frames

    def _run_block(
        self,
        x: DeviceBuffer,
        out: DeviceBuffer,
        rows: int,
        block: _DeviceBlock,
        scratch: dict[str, DeviceBuffer],
        state=None,
        key='',
    ) -> None:
        """One ConvNeXt block: reads ``x`` (rows, width), writes ``out``."""
        width = block.width
        total = rows * width
        normed = scratch["normed"]
        mixed = scratch["mixed"]
        vv_rmsnorm_bf16(x.ptr, block.norm_w.ptr, normed.ptr, rows, width, 1e-5,
                        library=self.library, runtime=self.runtime)
        prefix = self._prefix(state,key,normed,rows,width,6,self._mixer_zero)
        vv_depthwise_conv_bf16(
            prefix.ptr, normed.ptr, x.ptr, block.conv_w.ptr, block.conv_b.ptr,
            block.gamma.ptr, mixed.ptr, 6, rows, width, 7,
            library=self.library, runtime=self.runtime,
        )
        # ffn
        ffn_normed = scratch["ffn_normed"]
        vv_rmsnorm_bf16(mixed.ptr, block.ffn_norm_w.ptr, ffn_normed.ptr, rows, width, 1e-5,
                         library=self.library, runtime=self.runtime)
        h = scratch["h"]
        _prefill_gemm(
            ffn_normed.ptr, block.ffn_w1.ptr, h.ptr, rows, width, 4 * width,
            runtime=self.runtime,
        )
        vv_add_bias_bf16(h.ptr, block.ffn_b1.ptr, h.ptr, rows * 4 * width, 4 * width,
                         library=self.library, runtime=self.runtime)
        vv_gelu_bf16(h.ptr, h.ptr, rows * 4 * width, library=self.library, runtime=self.runtime)
        y = scratch["y"]
        _prefill_gemm(
            h.ptr, block.ffn_w2.ptr, y.ptr, rows, 4 * width, width,
            runtime=self.runtime,
        )
        vv_add_bias_bf16(y.ptr, block.ffn_b2.ptr, y.ptr, total, width,
                         library=self.library, runtime=self.runtime)
        # out = mixed + y * ffn_gamma (in-place on mixed is safe elementwise)
        vv_scale_residual_bf16(
            mixed.ptr, y.ptr, block.ffn_gamma.ptr, out.ptr, total, width,
            library=self.library, runtime=self.runtime,
        )

    # ------------------------------------------------------------------
    def encode(self, pcm: np.ndarray, *, chunk_samples: int = 1_440_000) -> dict[str, np.ndarray]:
        """Encode one recording, carrying every convolution tail across chunks.

        Pad only the final partial frame. State is private to this call and
        released between recordings; sampling is deliberately not done here.
        """
        from numbers import Integral
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1 or not pcm.size or not np.isfinite(pcm).all():
            raise ValueError('PCM must be a nonempty finite mono waveform')
        if (isinstance(chunk_samples, bool) or not isinstance(chunk_samples, Integral)
                or chunk_samples <= 0 or chunk_samples % 3200):
            raise ValueError('chunk_samples must be a positive multiple of 3200')
        if pcm.size % 3200:
            pcm = np.pad(pcm, (0, 3200 - pcm.size % 3200))
        state = {'acoustic': {}, 'semantic': {}}
        results = {name: [] for name in state}
        for start in range(0, pcm.size, chunk_samples):
            chunk = pcm[start:start+chunk_samples]
            pcm_u16 = _upload_u16(f32_to_bf16_bits(chunk))
            try:
                for name in state:
                    spec = self.specs[name]
                    # One encoder pass at a time; returned latents are owned
                    # host arrays, so resetting the arena cannot invalidate them.
                    self._scratch.reset(capacity_bytes=24*chunk.size*spec.num_filters*2+(16 << 20))
                    latent, frames = self._encoder_forward(name, pcm_u16, chunk.size,
                        chunk_tail=state[name] if pcm.size > chunk_samples else None)
                    host = np.empty((frames,spec.hidden_size),dtype=np.uint16)
                    copy_device_to_host(host_array_ptr(host),latent)
                    results[name].append((host.astype(np.uint32) << 16).view(np.float32))
            finally:
                free(pcm_u16)
        return {name: np.concatenate(parts,axis=0) for name,parts in results.items()}

    def forward(
        self,
        pcm: np.ndarray,
        *,
        noise: np.ndarray | None = None,
        noise_scale: np.ndarray | None = None,
        chunk_samples: int = 1_440_000,
    ) -> np.ndarray:
        """Processed mono PCM -> summed connector embeddings.

        Supply both recorded noise operands for sampled inference. Omitting
        both requests the explicit mean-latent diagnostic path.
        """
        if (noise is None) != (noise_scale is None):
            raise ValueError('noise and noise_scale must be supplied together')
        frames = (len(pcm) + 3199) // 3200
        if noise is not None:
            noise = np.asarray(noise,dtype=np.float32)
            noise_scale = np.asarray(noise_scale,dtype=np.float32)
            if noise.shape != (frames,self.specs['acoustic'].hidden_size):
                raise ValueError('acoustic noise shape must match the joined recording')
            if noise_scale.size != 1 or not np.isfinite(noise).all() or not np.isfinite(noise_scale).all():
                raise ValueError('noise operands must be finite with one recording scale')
        latents = self.encode(pcm,chunk_samples=chunk_samples)
        with ExitStack() as owned:
            def keep(buffer):
                owned.callback(free, buffer)
                return buffer
            lat_ac = keep(_upload_u16(f32_to_bf16_bits(latents['acoustic'])))
            lat_se = keep(_upload_u16(f32_to_bf16_bits(latents['semantic'])))
            if noise is not None:
                gn = keep(_upload_u16(f32_to_bf16_bits(noise.reshape(-1))))
                gs = keep(malloc(4))
                copy_host_array_to_device(gs, noise_scale.reshape(-1))
                sampled = keep(_alloc_u16(frames * self.specs['acoustic'].hidden_size))
                vv_add_scaled_noise_bf16(
                    lat_ac.ptr, gs.ptr, gn.ptr, sampled.ptr, frames * self.specs['acoustic'].hidden_size,
                    frames * self.specs['acoustic'].hidden_size,
                    library=self.library, runtime=self.runtime,
                )
                lat_ac = sampled
            return self._connector_sum(lat_ac, lat_se, frames)

    def _connector_sum(self, lat_ac: DeviceBuffer, lat_se: DeviceBuffer, frames: int) -> np.ndarray:
        fc1w, fc1b, normw, fc2w, fc2b = self.connectors["acoustic"]
        h_ac = _alloc_u16(frames * self.text_hidden)
        _prefill_gemm(
            lat_ac.ptr, fc1w.ptr, h_ac.ptr, frames, self.specs["acoustic"].hidden_size, self.text_hidden,
            runtime=self.runtime,
        )
        vv_add_bias_bf16(h_ac.ptr, fc1b.ptr, h_ac.ptr, frames * self.text_hidden, self.text_hidden,
                         library=self.library, runtime=self.runtime)
        vv_rmsnorm_bf16(h_ac.ptr, normw.ptr, h_ac.ptr, frames, self.text_hidden, 1e-6,
                        library=self.library, runtime=self.runtime)
        emb_ac = _alloc_u16(frames * self.text_hidden)
        _prefill_gemm(
            h_ac.ptr, fc2w.ptr, emb_ac.ptr, frames, self.text_hidden, self.text_hidden,
            runtime=self.runtime,
        )
        vv_add_bias_bf16(emb_ac.ptr, fc2b.ptr, emb_ac.ptr, frames * self.text_hidden, self.text_hidden,
                         library=self.library, runtime=self.runtime)
        free(h_ac)

        fc1w, fc1b, normw, fc2w, fc2b = self.connectors["semantic"]
        h_se = _alloc_u16(frames * self.text_hidden)
        _prefill_gemm(
            lat_se.ptr, fc1w.ptr, h_se.ptr, frames, self.specs["semantic"].hidden_size, self.text_hidden,
            runtime=self.runtime,
        )
        vv_add_bias_bf16(h_se.ptr, fc1b.ptr, h_se.ptr, frames * self.text_hidden, self.text_hidden,
                         library=self.library, runtime=self.runtime)
        vv_rmsnorm_bf16(h_se.ptr, normw.ptr, h_se.ptr, frames, self.text_hidden, 1e-6,
                        library=self.library, runtime=self.runtime)
        emb_se = _alloc_u16(frames * self.text_hidden)
        _prefill_gemm(
            h_se.ptr, fc2w.ptr, emb_se.ptr, frames, self.text_hidden, self.text_hidden,
            runtime=self.runtime,
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
        self._scratch.close()
