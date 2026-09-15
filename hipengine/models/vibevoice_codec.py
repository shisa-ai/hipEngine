"""Device-resident VibeVoice acoustic-tokenizer decoder (gfx1100 kernels).

Composes the validated ``vv_codec_v1`` HIP primitives
(``hipengine.kernels.hip_gfx1100.vibevoice.codec_ops``) into the full decoder
graph, mirroring ``hipengine.kernels.cpu_reference.vibevoice_codec`` exactly:

- causal stride-1 ``SConv1d`` stem, mixers and head with per-layer context
  caches (context = kernel - 1, zero-initialized on first use);
- causal ``SConvTranspose1d`` upsampling stages whose caches start empty and
  whose streaming output is the last ``T * stride`` samples of the windowed
  convtr;
- per-block RMSNorm -> depthwise mixer -> gamma layer-scale residual, then
  RMSNorm -> FFN -> gamma layer-scale residual.

Weights upload once as FP32 and stay resident; conv-window caches live in
device ping/pong buffers, so a streaming decode copies only the latent chunk
in (``[T, 64]``) and the PCM samples out (``[T * 3200]``). The non-streaming
``decode_full`` path leaves every cache untouched and pads with zeros,
matching ``decode_full_sequence`` in the reference. Torch is not imported
anywhere on this path.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    malloc,
)
from hipengine.kernels.cpu_reference.vibevoice_codec import (
    DecoderGeometry,
    load_decoder_weights,
)
from hipengine.kernels.hip_gfx1100.vibevoice.codec_ops import build_codec_ops

_ALIGN = 256  # bump-alignment for arena allocations, bytes


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


class _WindowDst:
    """Minimal ``.ptr`` stand-in so ``_gather`` accepts an arena window."""

    __slots__ = ("ptr",)

    def __init__(self, ptr: int) -> None:
        self.ptr = ptr


class _Arena:
    """Bump allocator over one device buffer.

    Capacity is fixed at construction (sized for ``max_chunk_frames``);
    ``allocate`` raises if a call sequence exceeds it, which would be a
    sizing bug rather than a runtime condition. Pointers stay valid for the
    whole call; ``reset`` between calls.
    """

    __slots__ = ("_buffer", "_capacity", "_used")

    def __init__(self, runtime: HipRuntime, capacity_bytes: int) -> None:
        self._buffer = malloc(capacity_bytes, runtime=runtime)
        self._capacity = capacity_bytes
        self._used = 0

    def reset(self) -> None:
        self._used = 0

    def allocate(self, nbytes: int) -> int:
        aligned = _aligned(nbytes)
        if self._used + aligned > self._capacity:
            raise RuntimeError(
                f"decoder scratch arena exhausted: {self._used}+{aligned}"
                f" > {self._capacity} bytes"
            )
        pointer = self._buffer.ptr + self._used
        self._used += aligned
        return pointer

    def free_device(self, runtime: HipRuntime) -> None:
        free(self._buffer, runtime=runtime)


class _ConvLayer:
    """Window buffers plus cache state for one conv layer.

    ``buffers`` is a ping/pong pair of ``[channels, context + max_new]``
    device buffers. The front buffer holds the cache from the previous call,
    written densely with row stride equal to the recorded ``length``; the
    back buffer receives the assembled ``[cache | x]`` window and then
    becomes the new front. ``zero_window`` is a ``[channels, context]`` zero
    buffer serving the non-streaming path as a never-updated cache.
    """

    __slots__ = ("buffers", "zero_window", "front", "length", "context")

    def __init__(
        self, runtime: HipRuntime, channels: int, context: int, max_new: int
    ) -> None:
        self.context = context
        window_nbytes = channels * (context + max_new) * 4
        self.buffers = (
            malloc(window_nbytes, runtime=runtime),
            malloc(window_nbytes, runtime=runtime),
        )
        zero_nbytes = channels * context * 4
        self.zero_window = malloc(zero_nbytes, runtime=runtime)
        host = np.zeros(zero_nbytes // 4, dtype=np.float32)
        copy_host_to_device(
            self.zero_window, host.ctypes.data, zero_nbytes, runtime=runtime
        )
        self.front = 0
        self.length = 0

    def reset_zeros(self, runtime: HipRuntime) -> None:
        """Restart as a zero-initialized sconv cache (context columns)."""

        self.front = 0
        self.length = self.context
        host = np.zeros(self.buffers[0].nbytes // 4, dtype=np.float32)
        for buffer in self.buffers:
            copy_host_to_device(
                buffer, host.ctypes.data, buffer.nbytes, runtime=runtime
            )

    def reset_empty(self) -> None:
        """Restart with an empty cache (convtr first-chunk semantics)."""

        self.front = 0
        self.length = 0

    def free_device(self, runtime: HipRuntime) -> None:
        for buffer in self.buffers:
            free(buffer, runtime=runtime)
        free(self.zero_window, runtime=runtime)


class VibeVoiceCodecDecoderDevice:
    """FP32 device decoder over the pinned VibeVoice acoustic tokenizer."""

    def __init__(
        self,
        weights,
        *,
        geometry: DecoderGeometry = DecoderGeometry(),
        max_chunk_frames: int = 16,
        runtime: HipRuntime | None = None,
    ) -> None:
        self.geometry = geometry
        self.max_chunk_frames = max_chunk_frames
        self.hop_length = geometry.hop_length
        self._runtime = runtime or get_hip_runtime()
        self._lib = build_codec_ops(load=True, require_cached=True)
        self._stream = 0
        self._bind_signatures()

        dims = geometry.dims()
        self._dims = dims

        # Device weights: name -> (pointer, buffer, shape).
        self._weights: dict[str, tuple[int, DeviceBuffer, tuple[int, ...]]] = {}
        for name, array in weights.items():
            host = np.ascontiguousarray(array, dtype=np.float32)
            buffer = malloc(host.nbytes, runtime=self._runtime)
            copy_host_to_device(
                buffer, host.ctypes.data, host.nbytes, runtime=self._runtime
            )
            self._weights[name] = (buffer.ptr, buffer, host.shape)

        # Persistent I/O buffers: latent chunk in, PCM samples out.
        self._input_buffer = malloc(
            geometry.latent_dim * max_chunk_frames * 4, runtime=self._runtime
        )
        self._output_buffer = malloc(
            geometry.channels * max_chunk_frames * geometry.hop_length * 4,
            runtime=self._runtime,
        )

        # Conv layers: stem, head, one mixer per block, one convtr per stage.
        # Each layer's window buffers are sized for the largest window it
        # assembles: context + max_chunk_frames * (cumulative hop up to that
        # stage), because the streaming signal length grows by every stage's
        # upsample ratio. Undersizing here overflows into adjacent device
        # allocations and corrupts them.
        self._stem = _ConvLayer(
            self._runtime, geometry.latent_dim, geometry.kernel_size - 1,
            max_chunk_frames,
        )
        self._mixers: dict[tuple[int, int], _ConvLayer] = {}
        self._convtrs: dict[int, _ConvLayer] = {}
        prefix = 1  # cumulative hop: streaming length = frames * prefix
        for stage, dim in enumerate(dims):
            for block in range(geometry.depths[stage]):
                self._mixers[(stage, block)] = _ConvLayer(
                    self._runtime, dim, geometry.kernel_size - 1,
                    max_chunk_frames * prefix,
                )
            if stage < len(geometry.ratios):
                ratio = geometry.ratios[stage]
                self._convtrs[stage] = _ConvLayer(
                    self._runtime, dims[stage], ratio * 2 - 1,
                    max_chunk_frames * prefix,
                )
                prefix *= ratio
        self._head = _ConvLayer(
            self._runtime, dims[-1], geometry.last_kernel_size - 1,
            max_chunk_frames * prefix,
        )
        self.reset()

        self._arena = _Arena(self._runtime, self._arena_bytes(max_chunk_frames))

    # ------------------------------------------------------------ loading

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        *,
        geometry: DecoderGeometry = DecoderGeometry(),
        max_chunk_frames: int = 16,
        runtime: HipRuntime | None = None,
    ) -> "VibeVoiceCodecDecoderDevice":
        """Load the acoustic decoder weights from the pinned HF checkpoint."""

        from hipengine.loading.safetensors import (
            load_weight_index,
            read_tensor_storage_bytes,
        )

        index = load_weight_index(model_path)

        def read_tensor(full_name: str) -> bytes:
            info = index.require([full_name])[0]
            return read_tensor_storage_bytes(info)

        weights = load_decoder_weights(read_tensor, geometry)
        return cls(
            weights,
            geometry=geometry,
            max_chunk_frames=max_chunk_frames,
            runtime=runtime,
        )

    # ---------------------------------------------------------- internals

    def reset(self) -> None:
        """Clear all conv caches (fresh utterance, matches set_to_zero)."""

        self._stem.reset_zeros(self._runtime)
        self._head.reset_zeros(self._runtime)
        for layer in self._mixers.values():
            layer.reset_zeros(self._runtime)
        for layer in self._convtrs.values():
            layer.reset_empty()

    def _arena_bytes(self, frames: int) -> int:
        """Peak scratch bytes for one call with ``frames`` latent frames.

        Sized for the non-streaming path (the superset): it needs the same
        kernel-output buffers as the streaming path plus exact-size conv
        windows, since full-mode windows grow with the upsampled signal
        length and cannot live in the fixed-size cache buffers.
        """

        geometry = self.geometry
        total = 0

        def add(channels: int, length: int) -> None:
            nonlocal total
            total += _aligned(channels * length * 4)

        add(self._dims[0], frames)  # stem output
        add(geometry.latent_dim, geometry.kernel_size - 1 + frames)  # stem window
        x_len = frames
        for stage, dim in enumerate(self._dims):
            for _ in range(geometry.depths[stage]):
                add(dim, x_len)  # norm out
                add(dim, x_len)  # mixer out
                add(dim, x_len)  # gamma scale
                add(dim, x_len)  # residual add
                add(dim, geometry.kernel_size - 1 + x_len)  # mixer window
                add(dim, x_len)  # ffn norm
                add(4 * dim, x_len)  # ffn1 out
                add(4 * dim, x_len)  # gelu out
                add(dim, x_len)  # ffn2 out
                add(dim, x_len)  # ffn gamma scale
                add(dim, x_len)  # ffn residual add
            if stage < len(geometry.ratios):
                ratio = geometry.ratios[stage]
                add(self._dims[stage], x_len)  # convtr gather window
                add(self._dims[stage + 1], x_len * ratio)  # convtr output
                x_len *= ratio
        add(self._dims[-1], geometry.last_kernel_size - 1 + x_len)  # head window
        return total

    def _bind_signatures(self) -> None:
        i64 = ctypes.c_int64
        vp = ctypes.c_void_p
        f32 = ctypes.c_float

        signatures: dict[str, "ctypes._FuncPtr"] = {}

        def bind(symbol: str, spec: list) -> None:
            fn = getattr(self._lib, symbol)
            fn.argtypes = spec
            fn.restype = ctypes.c_int
            signatures[symbol] = fn

        bind("hipengine_vv_concat_gather", [vp, vp, vp, i64, i64, i64, i64, vp])
        bind(
            "hipengine_vv_conv1d_valid_dense",
            [vp, vp, vp, vp, i64, i64, i64, i64, vp],
        )
        bind(
            "hipengine_vv_conv1d_valid_depthwise",
            [vp, vp, vp, vp, i64, i64, i64, vp],
        )
        bind(
            "hipengine_vv_conv_transpose1d_causal",
            [vp, vp, vp, vp, i64, i64, i64, i64, i64, i64, i64, vp],
        )
        bind("hipengine_vv_rmsnorm_channels", [vp, vp, vp, i64, i64, f32, vp])
        bind("hipengine_vv_gelu_erf", [vp, vp, i64, vp])
        bind("hipengine_vv_add", [vp, vp, vp, i64, vp])
        bind("hipengine_vv_channel_scale", [vp, vp, vp, i64, i64, vp])
        bind("hipengine_vv_linear", [vp, vp, vp, vp, i64, i64, i64, vp])
        self._signatures = signatures

    def _launch(self, symbol: str, args: list) -> None:
        err = self._signatures[symbol](*args)
        if err != 0:
            raise RuntimeError(f"{symbol} launch failed with {err}")

    @staticmethod
    def _p(value: int):
        return ctypes.c_void_p(value)

    def _weight_ptr(self, name: str) -> int:
        return self._weights[name][0]

    def _bias_name(self, weight_name: str) -> str:
        return weight_name.replace(".weight", ".bias")

    def _gather(
        self,
        cache_buffer: DeviceBuffer,
        cache_len: int,
        keep: int,
        x_ptr: int,
        dst_ptr: int,
        channels: int,
        new_len: int,
    ) -> None:
        self._launch(
            "hipengine_vv_concat_gather",
            [
                self._p(cache_buffer.ptr), self._p(x_ptr), self._p(dst_ptr),
                ctypes.c_int64(channels), ctypes.c_int64(keep),
                ctypes.c_int64(cache_len), ctypes.c_int64(new_len),
                ctypes.c_void_p(self._stream),
            ],
        )

    def _sconv_step(
        self,
        layer: _ConvLayer,
        x_ptr: int,
        in_channels: int,
        new_len: int,
        weight_name: str,
        out_ptr: int,
        *,
        update_cache: bool,
        window_ptr: int | None = None,
    ) -> None:
        """Causal stride-1 conv over ``[cache | x]``; kernel width from the
        weight shape; the window is ``context + new_len`` columns.

        Streaming calls assemble into the layer's ping/pong cache buffer;
        non-streaming calls must pass ``window_ptr`` (an exact-size scratch
        window, ``[in_channels, context + new_len]``) because full-mode
        windows grow with the upsampled length and cannot fit the fixed
        cache buffers.
        """

        weight_ptr, _, weight_shape = self._weights[weight_name]
        bias_ptr = self._weight_ptr(self._bias_name(weight_name))
        out_channels, weight_in, kernel = weight_shape
        if weight_in == 1 and out_channels > 1:
            # Depthwise (groups == channels): the channel count is dim 0.
            if out_channels != in_channels:
                raise ValueError(
                    f"{weight_name}: depthwise weight has {out_channels}"
                    f" channels, x has {in_channels}"
                )
        elif weight_in != in_channels:
            raise ValueError(
                f"{weight_name}: weight expects {weight_in} input channels,"
                f" x has {in_channels}"
            )
        context = kernel - 1
        if update_cache:
            cache_buffer = layer.buffers[layer.front]
            cache_len = layer.length
            keep = min(cache_len, context)
            dst = layer.buffers[1 - layer.front]
        else:
            if window_ptr is None:
                raise ValueError(
                    "non-streaming sconv requires an exact-size window_ptr"
                )
            cache_buffer = layer.zero_window
            cache_len = context
            keep = context
            dst = _WindowDst(window_ptr)
        window_len = keep + new_len
        if window_len < new_len + kernel - 1:
            raise RuntimeError(
                f"window {window_len} shorter than context + input"
                f" {new_len + kernel - 1}"
            )
        self._gather(
            cache_buffer, cache_len, keep, x_ptr, dst.ptr, in_channels, new_len
        )
        if weight_in == 1 and out_channels > 1:
            self._launch(
                "hipengine_vv_conv1d_valid_depthwise",
                [
                    self._p(dst.ptr), self._p(weight_ptr), self._p(bias_ptr),
                    self._p(out_ptr), ctypes.c_int64(in_channels),
                    ctypes.c_int64(kernel), ctypes.c_int64(new_len),
                    ctypes.c_void_p(self._stream),
                ],
            )
        else:
            self._launch(
                "hipengine_vv_conv1d_valid_dense",
                [
                    self._p(dst.ptr), self._p(weight_ptr), self._p(bias_ptr),
                    self._p(out_ptr), ctypes.c_int64(in_channels),
                    ctypes.c_int64(out_channels), ctypes.c_int64(kernel),
                    ctypes.c_int64(new_len), ctypes.c_void_p(self._stream),
                ],
            )
        if update_cache:
            layer.front = 1 - layer.front
            layer.length = window_len

    def _convtr_step(
        self,
        layer: _ConvLayer,
        x_ptr: int,
        in_channels: int,
        new_len: int,
        weight_name: str,
        out_ptr: int,
        *,
        update_cache: bool,
        window_ptr: int | None = None,
    ) -> int:
        """Causal convtr over ``[cache | x]``; the kept tail (last
        ``new_len * stride`` samples) is written compacted dense to
        ``out_ptr``; returns 0 (the output starts at the buffer)."""

        weight_ptr, _, weight_shape = self._weights[weight_name]
        bias_ptr = self._weight_ptr(self._bias_name(weight_name))
        # PyTorch ConvTranspose1d layout: weight is [in, out, kernel].
        weight_in, out_channels, kernel = weight_shape
        if weight_in != in_channels:
            raise ValueError(
                f"{weight_name}: weight expects {weight_in} input channels,"
                f" x has {in_channels}"
            )
        stride = kernel // 2
        if kernel != 2 * stride:
            raise ValueError(f"{weight_name}: kernel {kernel} is not 2*stride")
        if update_cache:
            cache_buffer = layer.buffers[layer.front]
            cache_len = layer.length
            keep = min(cache_len, layer.context)
            dst = layer.buffers[1 - layer.front]
        else:
            if window_ptr is None:
                raise ValueError(
                    "non-streaming convtr requires an exact-size window_ptr"
                )
            cache_buffer = layer.buffers[layer.front]  # unread when keep=0
            cache_len = layer.length
            keep = 0
            dst = _WindowDst(window_ptr)
        window_len = keep + new_len
        self._gather(
            cache_buffer, cache_len, keep, x_ptr, dst.ptr, in_channels, new_len
        )
        # The kernel writes only the kept tail positions, compacted densely,
        # so the output is a dense [out_c, new_len * stride] tensor and
        # downstream layers can consume it without stride-aware views.
        out_len = new_len * stride
        self._launch(
            "hipengine_vv_conv_transpose1d_causal",
            [
                self._p(dst.ptr), self._p(weight_ptr), self._p(bias_ptr),
                self._p(out_ptr), ctypes.c_int64(in_channels),
                ctypes.c_int64(out_channels), ctypes.c_int64(kernel),
                ctypes.c_int64(stride), ctypes.c_int64(window_len * stride),
                ctypes.c_int64(keep * stride), ctypes.c_int64(out_len),
                ctypes.c_void_p(self._stream),
            ],
        )
        if update_cache:
            layer.front = 1 - layer.front
            layer.length = window_len
        return 0

    def _rmsnorm(self, x_ptr, weight_name, out_ptr, channels, length) -> None:
        self._launch(
            "hipengine_vv_rmsnorm_channels",
            [
                self._p(x_ptr), self._p(self._weight_ptr(weight_name)),
                self._p(out_ptr), ctypes.c_int64(channels),
                ctypes.c_int64(length), ctypes.c_float(self.geometry.rms_eps),
                ctypes.c_void_p(self._stream),
            ],
        )

    def _channel_scale(self, x_ptr, weight_name, out_ptr, channels, length) -> None:
        self._launch(
            "hipengine_vv_channel_scale",
            [
                self._p(x_ptr), self._p(self._weight_ptr(weight_name)),
                self._p(out_ptr), ctypes.c_int64(channels),
                ctypes.c_int64(length), ctypes.c_void_p(self._stream),
            ],
        )

    def _add(self, x_ptr, y_ptr, out_ptr, n) -> None:
        self._launch(
            "hipengine_vv_add",
            [
                self._p(x_ptr), self._p(y_ptr), self._p(out_ptr),
                ctypes.c_int64(n), ctypes.c_void_p(self._stream),
            ],
        )

    def _gelu(self, x_ptr, out_ptr, n) -> None:
        self._launch(
            "hipengine_vv_gelu_erf",
            [
                self._p(x_ptr), self._p(out_ptr), ctypes.c_int64(n),
                ctypes.c_void_p(self._stream),
            ],
        )

    def _linear(self, x_ptr, weight_name, out_ptr, in_f, out_f, length) -> None:
        self._launch(
            "hipengine_vv_linear",
            [
                self._p(x_ptr), self._p(self._weight_ptr(weight_name)),
                self._p(self._weight_ptr(self._bias_name(weight_name))),
                self._p(out_ptr), ctypes.c_int64(in_f),
                ctypes.c_int64(out_f), ctypes.c_int64(length),
                ctypes.c_void_p(self._stream),
            ],
        )

    def _run_stage(self, stage: int, x_ptr: int, channels: int, length: int,
                   alloc, *, streaming: bool) -> int:
        geometry = self.geometry
        for block in range(geometry.depths[stage]):
            base = f"stages.{stage}.{block}."

            # Mixer path: norm -> depthwise conv -> gamma -> residual add.
            residual = x_ptr
            norm_out = alloc(channels, length)
            self._rmsnorm(x_ptr, base + "norm.weight", norm_out, channels, length)
            mixer_out = alloc(channels, length)
            mixer_window = (
                None
                if streaming
                else alloc(channels, geometry.kernel_size - 1 + length)
            )
            self._sconv_step(
                self._mixers[(stage, block)], norm_out, channels, length,
                base + "mixer.conv.conv.conv.weight", mixer_out,
                update_cache=streaming, window_ptr=mixer_window,
            )
            scaled = alloc(channels, length)
            self._channel_scale(mixer_out, base + "gamma", scaled, channels, length)
            added = alloc(channels, length)
            self._add(residual, scaled, added, channels * length)
            x_ptr = added

            # FFN path: norm -> linear -> GELU -> linear -> gamma -> residual.
            ffn_residual = x_ptr
            norm_out = alloc(channels, length)
            self._rmsnorm(
                x_ptr, base + "ffn_norm.weight", norm_out, channels, length
            )
            ffn1_out = alloc(4 * channels, length)
            self._linear(
                norm_out, base + "ffn.linear1.weight", ffn1_out,
                channels, 4 * channels, length,
            )
            gelu_out = alloc(4 * channels, length)
            self._gelu(ffn1_out, gelu_out, 4 * channels * length)
            ffn2_out = alloc(channels, length)
            self._linear(
                gelu_out, base + "ffn.linear2.weight", ffn2_out,
                4 * channels, channels, length,
            )
            scaled = alloc(channels, length)
            self._channel_scale(
                ffn2_out, base + "ffn_gamma", scaled, channels, length
            )
            added = alloc(channels, length)
            self._add(ffn_residual, scaled, added, channels * length)
            x_ptr = added
        return x_ptr

    # ------------------------------------------------------------- decode

    def decode_chunk(self, frames: np.ndarray) -> np.ndarray:
        """Decode ``[T, latent_dim]`` latents to ``[T * hop_length]`` samples.

        Streaming semantics: conv caches carry across calls; call
        ``reset()`` between utterances.
        """

        frames = self._validate_frames(frames)
        return self._run(frames, streaming=True)

    def decode_full(self, frames: np.ndarray) -> np.ndarray:
        """Non-streaming decode (left zero padding, caches untouched)."""

        frames = self._validate_frames(frames)
        return self._run(frames, streaming=False)

    def _validate_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != self.geometry.latent_dim:
            raise ValueError(
                f"latents must be [T, {self.geometry.latent_dim}],"
                f" got {frames.shape}"
            )
        total = frames.shape[0]
        if total < 1 or total > self.max_chunk_frames:
            raise ValueError(
                f"chunk frames {total} outside 1..{self.max_chunk_frames}"
            )
        return frames

    def _run(self, frames: np.ndarray, *, streaming: bool) -> np.ndarray:
        geometry = self.geometry
        total = frames.shape[0]
        host_input = np.ascontiguousarray(frames.T, dtype=np.float32)
        copy_host_to_device(
            self._input_buffer, host_input.ctypes.data, host_input.nbytes,
            runtime=self._runtime,
        )
        arena = self._arena
        arena.reset()

        def alloc(channels: int, length: int) -> int:
            return arena.allocate(channels * length * 4)

        # Stem: SConv1d(latent_dim -> dims[0]).
        stem_out = alloc(self._dims[0], total)
        self._sconv_step(
            self._stem, self._input_buffer.ptr, geometry.latent_dim, total,
            "upsample_layers.0.0.conv.conv.weight", stem_out,
            update_cache=streaming,
            window_ptr=None if streaming
            else alloc(geometry.latent_dim, geometry.kernel_size - 1 + total),
        )
        x_ptr = stem_out
        x_channels = self._dims[0]
        x_len = total

        x_ptr = self._run_stage(0, x_ptr, x_channels, x_len, alloc,
                                streaming=streaming)

        for stage, ratio in enumerate(geometry.ratios):
            out_len = x_len * ratio
            window = alloc(self._dims[stage + 1], out_len)
            self._convtr_step(
                self._convtrs[stage], x_ptr, x_channels, x_len,
                f"upsample_layers.{stage + 1}.0.convtr.convtr.weight", window,
                update_cache=streaming,
                window_ptr=None if streaming else alloc(x_channels, x_len),
            )
            x_ptr = window
            x_channels = self._dims[stage + 1]
            x_len = out_len
            x_ptr = self._run_stage(stage + 1, x_ptr, x_channels, x_len, alloc,
                                    streaming=streaming)

        # Head: SConv1d(dims[-1] -> channels), written straight into the
        # persistent output buffer for the D2H copy.
        out_len = total * geometry.hop_length
        self._sconv_step(
            self._head, x_ptr, x_channels, x_len,
            "head.conv.conv.weight", self._output_buffer.ptr,
            update_cache=streaming,
            window_ptr=None if streaming
            else alloc(x_channels, geometry.last_kernel_size - 1 + x_len),
        )
        self._runtime.device_synchronize()
        samples = np.empty(out_len, dtype=np.float32)
        copy_device_to_host(
            samples.ctypes.data, self._output_buffer, samples.nbytes,
            runtime=self._runtime,
        )
        return samples

    def close(self) -> None:
        """Release every device allocation owned by this decoder."""

        for _, buffer, _ in self._weights.values():
            free(buffer, runtime=self._runtime)
        self._weights.clear()
        for layer in [self._stem, self._head, *self._mixers.values(),
                      *self._convtrs.values()]:
            layer.free_device(self._runtime)
        free(self._input_buffer, runtime=self._runtime)
        free(self._output_buffer, runtime=self._runtime)
        self._arena.free_device(self._runtime)
