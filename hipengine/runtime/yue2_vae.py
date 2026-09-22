"""YuE2 Oobleck VAE decoder runtime (FP32, torch-free).

The released decoder is FP32 end to end: weight-normalized Conv1d /
ConvTranspose1d, dilated residual units, and SnakeBeta with log-scale alpha/beta
and a ``1e-9`` denominator epsilon. Weight normalization is folded once at load
time by :mod:`hipengine.loading.yue2`; everything here is device FP32 with the
reference's own arithmetic.

Decoding is bounded and tiled. ``decode_tiled`` reproduces the reference's
schedule exactly: each tile is decoded with finite left/right context, and only
the exact interior core is copied out. There is no crossfade, no boundary
smoothing, and no zero padding of the final audio, so a tiled waveform is the
full decoder's waveform up to FP32 rounding in the convolution kernels. Completed
cores are copied to host as they finish, which bounds device residency.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.yue2 import vae as vae_kernels
from hipengine.loading.yue2 import (
    VAE_DOWNSAMPLING_RATIO,
    VAE_SAMPLE_RATE,
    YuE2ConvWeights,
    YuE2SnakeWeights,
    YuE2VaeDecoderWeights,
)

DEFAULT_CORE_FRAMES = 1024
DEFAULT_HALO_FRAMES = 16


class Yue2VaeRuntime:
    """Device FP32 decoder for one loaded, folded VAE decoder weight set."""

    def __init__(
        self,
        weights: YuE2VaeDecoderWeights,
        *,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        self.weights = weights
        self.runtime = runtime or get_hip_runtime()
        self.library = library or vae_kernels.build_yue2_vae()
        if self.library is None:
            raise RuntimeError("yue2_vae build returned no library")
        self.sample_rate = int(weights.sample_rate or VAE_SAMPLE_RATE)
        self.downsampling_ratio = int(weights.downsampling_ratio or VAE_DOWNSAMPLING_RATIO)
        self._owned: list[DeviceBuffer] = []
        self._scratch: list[DeviceBuffer] = []
        self._convs: dict[int, tuple[DeviceBuffer, DeviceBuffer | None]] = {}
        self._snakes: dict[int, tuple[DeviceBuffer, DeviceBuffer]] = {}
        for conv in weights.all_convs():
            self._convs[id(conv)] = self._upload_conv(conv)
        for snake in weights.all_snakes():
            self._snakes[id(snake)] = self._upload_snake(snake)

    # -- device buffers -------------------------------------------------
    def _alloc(self, elements: int) -> DeviceBuffer:
        buffer = malloc(max(int(elements) * 4, 8))
        self._scratch.append(buffer)
        return buffer

    def _persistent(self, array: np.ndarray) -> DeviceBuffer:
        host = np.ascontiguousarray(array, dtype=np.float32)
        buffer = malloc(max(host.nbytes, 8))
        copy_host_array_to_device(buffer, host)
        self._owned.append(buffer)
        return buffer

    def _upload_conv(self, conv: YuE2ConvWeights):
        weight = self._persistent(conv.weight)
        bias = None if conv.bias is None else self._persistent(conv.bias)
        return weight, bias

    def _upload_snake(self, snake: YuE2SnakeWeights):
        return self._persistent(snake.alpha), self._persistent(snake.beta)

    def close(self) -> None:
        for buffer in self._scratch:
            free(buffer)
        self._scratch.clear()
        for buffer in self._owned:
            free(buffer)
        self._owned.clear()
        self._convs.clear()
        self._snakes.clear()

    def __enter__(self) -> Yue2VaeRuntime:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- arithmetic -----------------------------------------------------
    def natural_output_length(self, frames: int) -> int:
        return self.weights.natural_output_length(int(frames))

    def required_halo(self, core_frames: int) -> int:
        """Frames of context this decoder needs on each side of a core.

        Computed from the decoder's own dependency interval rather than assumed:
        the transposed convolutions shrink the required context at every stage,
        so the binding term is the last dilated residual unit.
        """

        return int(self.weights.required_halo(int(core_frames)))

    def _conv(self, conv: YuE2ConvWeights, x: DeviceBuffer, length: int, out: DeviceBuffer) -> int:
        weight, bias = self._convs[id(conv)]
        if conv.transposed:
            # ConvTranspose1d weights are [C_in, C_out, K].
            channels_in, channels_out = conv.weight.shape[0], conv.weight.shape[1]
        else:
            # Conv1d weights are [C_out, C_in, K].
            channels_out, channels_in = conv.weight.shape[0], conv.weight.shape[1]
        out_length = conv.output_length(length)
        kernel = conv.weight.shape[-1]
        if conv.transposed:
            vae_kernels.vae_conv_transpose1d_f32(
                x.ptr, weight.ptr, 0 if bias is None else bias.ptr, out.ptr,
                channels_in, channels_out, length, out_length, kernel, conv.stride,
                conv.padding, library=self.library, runtime=self.runtime,
            )
        else:
            vae_kernels.vae_conv1d_f32(
                x.ptr, weight.ptr, 0 if bias is None else bias.ptr, out.ptr,
                channels_in, channels_out, length, out_length, kernel, conv.stride,
                conv.dilation, conv.padding, library=self.library, runtime=self.runtime,
            )
        return out_length

    def _snake(self, snake: YuE2SnakeWeights, x: DeviceBuffer, channels: int, length: int) -> None:
        alpha, beta = self._snakes[id(snake)]
        vae_kernels.vae_snake_beta_f32(
            x.ptr, alpha.ptr, beta.ptr, x.ptr, channels, length,
            library=self.library, runtime=self.runtime,
        )

    def _residual_unit(self, unit, x: DeviceBuffer, channels: int, length: int) -> None:
        # The reference keeps the unit's input as its residual, so the first
        # SnakeBeta must write to a separate buffer: activating ``x`` in place
        # would add the activation back instead of the original input.
        activated = self._alloc(channels * length)
        vae_kernels.vae_snake_beta_f32(
            x.ptr, *[b.ptr for b in self._snakes[id(unit.activation_in)]], activated.ptr,
            channels, length, library=self.library, runtime=self.runtime,
        )
        conved = self._alloc(channels * length)
        self._conv(unit.conv, activated, length, conved)
        self._snake(unit.activation_out, conved, channels, length)
        pointwise = self._alloc(channels * length)
        self._conv(unit.pointwise, conved, length, pointwise)
        vae_kernels.vae_add_f32(
            x.ptr, pointwise.ptr, x.ptr, channels * length,
            library=self.library, runtime=self.runtime,
        )

    def _decode_device(self, latent: np.ndarray) -> tuple[DeviceBuffer, int]:
        """Decode one ``[1, 64, T]`` latent on the device; returns ``[2, S]``."""

        values = np.ascontiguousarray(latent, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] != self.weights.latent_dim:
            raise ValueError("latent must be [1, latent_dim, frames]")
        if values.shape[-1] < 1:
            raise ValueError("latent must have at least one frame")
        if not np.isfinite(values).all():
            raise ValueError("latent contains non-finite values")
        frames = int(values.shape[-1])
        weights = self.weights
        length = frames
        channels = weights.input_conv.weight.shape[1]
        x = self._persistent(values[0])
        for conv, next_channels in ((weights.input_conv, weights.input_conv.weight.shape[0]),):
            out = self._alloc(next_channels * conv.output_length(length))
            length = self._conv(conv, x, length, out)
            channels = next_channels
            x = out
        for block in weights.blocks:
            self._snake(block.activation, x, channels, length)
            upsampled = self._alloc(block.upsample.weight.shape[1] * block.upsample.output_length(length))
            length = self._conv(block.upsample, x, length, upsampled)
            channels = block.upsample.weight.shape[1]
            x = upsampled
            for unit in block.residual_units:
                self._residual_unit(unit, x, channels, length)
        self._snake(weights.output_activation, x, channels, length)
        out = self._alloc(weights.output_conv.weight.shape[0] * weights.output_conv.output_length(length))
        length = self._conv(weights.output_conv, x, length, out)
        return out, length

    def _read(self, buffer: DeviceBuffer, channels: int, length: int) -> np.ndarray:
        out = np.empty((channels, length), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), buffer)
        return out

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Full waveform ``[1, 2, natural_length]`` in FP32, without clipping."""

        buffer, length = self._decode_device(latent)
        channels = self.weights.output_conv.weight.shape[0]
        result = self._read(buffer, channels, length)[None, ...]
        self._release_scratch()
        return result

    def _release_scratch(self) -> None:
        for buffer in self._scratch:
            free(buffer)
        self._scratch.clear()

    def decode_tiled(
        self,
        latent: np.ndarray,
        *,
        core_frames: int = DEFAULT_CORE_FRAMES,
        halo_frames: int = DEFAULT_HALO_FRAMES,
        on_progress=None,
    ) -> np.ndarray:
        """Bounded decode: exact interior cores, natural end length, host output.

        Mirrors the reference's schedule. Each tile decodes
        ``[left, right)`` frames with enough context for every dependency and
        copies out only ``[start * ratio, min(end * ratio, total))``; the last
        tile's crop is shorter than its own output because the natural length is
        ``1920 * frames - 64``.
        """

        values = np.ascontiguousarray(latent, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] != self.weights.latent_dim:
            raise ValueError("latent must be [1, latent_dim, frames]")
        core = int(core_frames)
        halo = int(halo_frames)
        if core < 1:
            raise ValueError("core_frames must be positive")
        required = self.required_halo(core)
        if halo < required:
            raise ValueError(f"halo_frames must be at least {required} for this decoder")
        frames = int(values.shape[-1])
        ratio = self.downsampling_ratio
        total = self.natural_output_length(frames)
        channels = self.weights.output_conv.weight.shape[0]
        audio = np.empty((1, channels, total), dtype=np.float32)
        tiles = (frames + core - 1) // core
        for tile_index, start in enumerate(range(0, frames, core)):
            end = min(frames, start + core)
            left = max(0, start - halo)
            right = min(frames, end + halo)
            buffer, tile_length = self._decode_device(values[..., left:right])
            tile = self._read(buffer, channels, tile_length)
            self._release_scratch()
            out_start, out_end = start * ratio, min(end * ratio, total)
            crop_start = (start - left) * ratio
            crop = tile[..., crop_start:crop_start + out_end - out_start]
            if crop.shape[-1] != out_end - out_start:
                raise RuntimeError("VAE tile did not cover its requested output core")
            audio[..., out_start:out_end] = crop
            if on_progress is not None:
                on_progress(tile_index + 1, tiles)
        return audio

    def decode_audio(self, latent: np.ndarray, *, chunked: bool = True, **kwargs) -> np.ndarray:
        return self.decode_tiled(latent, **kwargs) if chunked else self.decode(latent)
