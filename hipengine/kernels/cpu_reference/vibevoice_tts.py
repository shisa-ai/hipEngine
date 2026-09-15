"""CPU FP32 reference for the VibeVoice-TTS acoustic waveform decoder.

Mirrors the community fork's ``TokenizerDecoder`` (the ``microsoft/VibeVoice-1.5B``
checkpoint's network, 343,695,969 parameters, decoder depths ``[8,3,3,3,3,3,3]``)
for the torch-free hipEngine port:

- ``VibevoiceDecoderSpec`` / ``VibevoiceDecoderWeights``: static geometry and the
  flat weight bundle in execution order.
- ``VibevoiceDecoderStream``: the per-call streaming decode exactly as the fork's
  generation loop drives it -- one call per speech frame, causal streaming caches
  per conv layer, ``reset`` for the speech-boundary ``set_to_zero``.

Topology per call (T input frames, hop = prod(ratios) = 3200 samples/frame):

- stem ``SConv1d`` 64 -> 2048 (kernel 7, stride 1), then ``depths[0]`` blocks
  at width 2048;
- for each ratio ``r``: causal ``SConvTranspose1d`` (kernel ``2r``, stride ``r``)
  halving width, then ``depths[i+1]`` blocks at the new width;
- head ``SConv1d`` 32 -> 1 (kernel 7); no final norm (``disable_last_norm``).

Blocks are the same ConvNeXt-1D blocks the ASR encoder uses: RMSNorm(1e-5) ->
depthwise causal conv (kernel 7) -> gamma layer scale -> residual; RMSNorm ->
linear(4x) -> erf-GELU -> linear -> ffn_gamma scale -> residual. Both conv
families are ``pad_mode='constant'`` (zero) and causal.

Streaming semantics (verified against the fork's ``SConv1d`` /
``SConvTranspose1d`` ``_forward_streaming``):

- strided/stem/head convs keep the last ``kernel - 1`` input rows as cache and
  compute a valid convolution over ``[cache, x]``;
- transposed convs keep the last ``kernel - 1`` input rows, compute the
  transposed convolution over ``[cache, x]``, unpad ``kernel - stride`` rows
  from the right (causal, ``trim_right_ratio=1``), and return the last
  ``T * stride`` rows (all of it on the first call, when the cache is empty).

Accumulation is FP32 with eager-bf16 rounding at every op boundary (matching
the bf16 torch oracle), controlled by ``dtype``. Weight layouts follow the
checkpoint: conv weights ``[out_channels, in_channels, kernel]``, transposed
conv weights ``[in_channels, out_channels, kernel]`` (``nn.ConvTranspose1d``
convention), linear weights ``[out_features, in_features]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from hipengine.kernels.cpu_reference.vibevoice_asr import (
    _finite,
    _gelu_erf,
    _round,
    vibevoice_linear,
)

ArrayLike = Any

# Qwen2.5 special tokens the fork reuses as speech control tokens.
SPEECH_START_ID = 151652  # <|vision_start|>
SPEECH_END_ID = 151653    # <|vision_end|>
SPEECH_DIFFUSION_ID = 151654  # <|vision_pad|>


def _valid_conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    stride: int,
    groups: int,
    *,
    dtype: str | None,
    accum_dtype: str = "float32",
) -> np.ndarray:
    """Valid-window causal conv over channel-first ``(batch, in_channels, length)``.

    No padding: streaming callers prepend their own cached context rows. Weight
    is ``[out_channels, in_channels // groups, kernel]``.
    """
    xb = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    b = None if bias is None else np.asarray(bias, dtype=np.float32)
    batch, in_channels, length = xb.shape
    out_channels, in_per_group, kernel = w.shape
    if in_channels != in_per_group * groups:
        raise ValueError("channel/group mismatch")
    if length < kernel:
        raise ValueError(f"valid conv needs length >= kernel, got {length} < {kernel}")
    out_length = (length - kernel) // stride + 1
    out = np.zeros((batch, out_channels, out_length), dtype=np.float32)
    out_per_group = out_channels // groups
    for g in range(groups):
        cols = np.empty((batch, in_per_group * kernel, out_length), dtype=np.float32)
        ch_lo, ch_hi = g * in_per_group, (g + 1) * in_per_group
        row_index = np.arange(in_per_group) * kernel
        for k in range(kernel):
            seg = xb[:, ch_lo:ch_hi, k : k + stride * out_length]
            if stride > 1:
                seg = seg[:, :, ::stride]
            if seg.shape[2] != out_length:
                raise AssertionError("window extraction mismatch")
            cols[:, row_index + k, :] = seg
        w_g = w[g * out_per_group : (g + 1) * out_per_group].reshape(out_per_group, -1)
        out[:, g * out_per_group : (g + 1) * out_per_group] = np.einsum("ok,bkl->bol", w_g, cols)
    if b is not None:
        out = out + b.reshape(1, out_channels, 1)
    result = _round("conv_out", out, dtype)
    _finite("conv_out", result)
    return result


def _conv_transpose1d_valid(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    stride: int,
    *,
    dtype: str | None,
) -> np.ndarray:
    """Transposed conv (no padding) over channel-first ``(batch, in, length)``.

    Weight follows ``nn.ConvTranspose1d``: ``[in_channels, out_channels, kernel]``.
    Output length is ``(length - 1) * stride + kernel``.
    """
    xb = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    b = None if bias is None else np.asarray(bias, dtype=np.float32)
    batch, in_channels, length = xb.shape
    if w.shape[0] != in_channels:
        raise ValueError(f"convtr weight in-channels {w.shape[0]} != input {in_channels}")
    out_channels, kernel = w.shape[1], w.shape[2]
    out_length = (length - 1) * stride + kernel
    out = np.zeros((batch, out_channels, out_length), dtype=np.float32)
    # out[o, j*stride + k] = sum_i w[i, o, k] * x[i, j]
    w_k = w.transpose(1, 0, 2)  # [out, in, kernel]
    for k in range(kernel):
        contribution = np.einsum("oi,bij->boj", w_k[:, :, k], xb)
        out[:, :, k : k + (length - 1) * stride + 1 : stride] += contribution
    if b is not None:
        out = out + b.reshape(1, out_channels, 1)
    result = _round("convtr_out", out, dtype)
    _finite("convtr_out", result)
    return result


def _rmsnorm_affine(
    x_rows: np.ndarray,
    weight: np.ndarray,
    eps: float,
    *,
    dtype: str | None,
) -> np.ndarray:
    """Eager-faithful RMSNorm over the last axis of row-major ``x_rows``.

    The fork's ``ConvRMSNorm`` upcasts to FP32, normalizes, rounds back to the
    storage dtype, and only then applies the affine weight (a second rounding).
    """
    xb = np.asarray(x_rows, dtype=np.float32)
    variance = np.mean(xb * xb, axis=-1, keepdims=True)
    normed = _round("rmsnorm_normed", xb / np.sqrt(variance + eps), dtype)
    w = np.asarray(weight, dtype=np.float32)
    return _round("rmsnorm_out", normed * w.reshape(*([1] * (xb.ndim - 1)), -1), dtype)


@dataclass(frozen=True)
class VibevoiceDecoderSpec:
    """Static geometry of the acoustic waveform decoder."""

    dimension: int
    channels: int
    num_filters: int
    depths: tuple[int, ...]
    ratios: tuple[int, ...]
    kernel_size: int
    last_kernel_size: int
    rms_norm_eps: float
    ffn_expansion: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VibevoiceDecoderSpec":
        """Build from the checkpoint's ``acoustic_tokenizer_config``.

        The fork derives decoder depths as ``reversed(encoder_depths)`` when
        ``decoder_depths`` is null; the checkpoint's string is
        ``"3-3-3-3-3-3-8"``, so the decoder builds ``[8,3,3,3,3,3,3]``.
        """
        depths_raw = config.get("decoder_depths")
        if depths_raw is None:
            depths_raw = config["encoder_depths"]
        if isinstance(depths_raw, str):
            depths = tuple(int(d) for d in reversed(depths_raw.split("-")))
        else:
            depths = tuple(int(d) for d in reversed(depths_raw))
        return cls(
            dimension=int(config["vae_dim"]),
            channels=int(config["channels"]),
            num_filters=int(config["decoder_n_filters"]),
            depths=depths,
            ratios=tuple(int(r) for r in config["decoder_ratios"]),
            kernel_size=int(config.get("kernel_size", 7)),
            last_kernel_size=int(config.get("last_kernel_size", 7)),
            rms_norm_eps=float(config.get("layernorm_eps", 1e-5)),
            ffn_expansion=int(config.get("ffn_expansion", 4)),
        )

    @property
    def hop_length(self) -> int:
        return int(math.prod(self.ratios))

    def stage_widths(self) -> tuple[int, ...]:
        """Width at each stage: ``num_filters << (len(depths) - 1 - i)``."""
        return tuple(
            self.num_filters * (2 ** (len(self.depths) - 1 - i)) for i in range(len(self.depths))
        )

    def block_layout(self) -> tuple[tuple[int, int], ...]:
        """``(width, num_blocks)`` per stage in execution order."""
        return tuple(
            (width, depth) for width, depth in zip(self.stage_widths(), self.depths)
        )


@dataclass(frozen=True)
class VibevoiceDecoderWeights:
    """Flat weight bundle for the acoustic waveform decoder.

    ``convtr_weights`` keeps the checkpoint layout ``[in, out, kernel]`` and is
    ordered by execution ratio. ``blocks`` holds the ten tensors per block under
    the same key names as the ASR encoder blocks.
    """

    stem_conv_weight: np.ndarray
    stem_conv_bias: np.ndarray
    convtr_weights: tuple[np.ndarray, ...]
    convtr_biases: tuple[np.ndarray, ...]
    head_conv_weight: np.ndarray
    head_conv_bias: np.ndarray
    blocks: tuple[dict[str, np.ndarray], ...]


class VibevoiceDecoderStream:
    """Per-sample streaming decoder state, driven one call at a time.

    Mirrors the fork's generation-loop contract: ``decode(latent)`` per speech
    frame, ``reset()`` where the loop zeros the streaming caches at a speech
    boundary (``VibeVoiceTokenizerStreamingCache.set_to_zero`` -- zeroing an
    entry and re-initializing it on next use are value-identical here, since
    every cache entry is zero-initialized context).
    """

    def __init__(
        self,
        spec: VibevoiceDecoderSpec,
        weights: VibevoiceDecoderWeights,
        *,
        dtype: str | None = "bfloat16",
    ) -> None:
        widths = spec.stage_widths()
        if weights.stem_conv_weight.shape != (widths[0], spec.dimension, spec.kernel_size):
            raise ValueError(
                f"stem weight {weights.stem_conv_weight.shape} does not match spec"
            )
        expected_convtr = []
        for i, ratio in enumerate(spec.ratios):
            c_in, c_out = widths[i], widths[i + 1]
            expected_convtr.append((c_in, c_out, 2 * ratio))
        actual = [tuple(w.shape) for w in weights.convtr_weights]
        if actual != expected_convtr:
            raise ValueError(f"convtr shapes {actual} do not match spec {expected_convtr}")
        if weights.head_conv_weight.shape != (spec.channels, widths[-1], spec.last_kernel_size):
            raise ValueError(
                f"head weight {weights.head_conv_weight.shape} does not match spec"
            )
        layout = spec.block_layout()
        total_blocks = sum(depth for _, depth in layout)
        if len(weights.blocks) != total_blocks:
            raise ValueError(f"expected {total_blocks} blocks, got {len(weights.blocks)}")
        self.spec = spec
        self.weights = weights
        self.dtype = dtype
        self._cache: dict[str, np.ndarray] = {}
        self._block_cursor = 0
        # Pre-compute the (stage, block) iteration order once.
        self._block_order: list[tuple[int, int]] = [
            (stage, j) for stage, (_, depth) in enumerate(layout) for j in range(depth)
        ]

    # -- cache helpers ----------------------------------------------------
    def _cached(self, key: str, channels: int, ctx: int) -> np.ndarray:
        arr = self._cache.get(key)
        if arr is None or arr.shape != (1, channels, ctx):
            return np.zeros((1, channels, ctx), dtype=np.float32)
        return arr

    def reset(self) -> None:
        """Zero every streaming cache entry (speech-boundary ``set_to_zero``)."""
        self._cache.clear()

    # -- streaming ops ----------------------------------------------------
    def _stream_conv1d(
        self,
        key: str,
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        kernel: int,
        *,
        groups: int,
    ) -> np.ndarray:
        ctx = kernel - 1
        cached = self._cached(key, x.shape[1], ctx)
        full = np.concatenate([cached, x], axis=2) if ctx > 0 else x
        out = _valid_conv1d(full, weight, bias, 1, groups, dtype=self.dtype)
        keep = full[:, :, -ctx:] if ctx > 0 else full
        self._cache[key] = keep
        return out

    def _stream_convtr(
        self,
        key: str,
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        stride: int,
    ) -> np.ndarray:
        kernel = weight.shape[2]
        ctx = kernel - 1
        pad_total = kernel - stride
        cached = self._cached(key, x.shape[1], ctx)
        full = np.concatenate([cached, x], axis=2) if cached.shape[2] > 0 else x
        y = _conv_transpose1d_valid(full, weight, bias, stride, dtype=self.dtype)
        if pad_total > 0:
            y = y[:, :, : y.shape[2] - pad_total]
        t_frames = x.shape[2]
        # Always the last T*stride rows. On the fork's first call the cache is
        # zero-length and this equals "all" after unpad; with zero-padded
        # history the extra leading rows are exact zeros, so the kept values
        # are identical either way.
        out = y[:, :, -t_frames * stride :]
        # The fork's cache grows call by call (first call stores just x); a
        # left-zero-padded fixed-length cache is value-identical, but the
        # length must be stable or the history would be dropped on reload.
        tail = full[:, :, -ctx:] if ctx > 0 else full
        if tail.shape[2] < ctx:
            tail = np.concatenate(
                [np.zeros((1, tail.shape[1], ctx - tail.shape[2]), dtype=np.float32), tail],
                axis=2,
            )
        self._cache[key] = tail
        return out

    def _stream_block(
        self,
        x: np.ndarray,
        block: Mapping[str, np.ndarray],
        width: int,
        key: str,
    ) -> np.ndarray:
        eps = self.spec.rms_norm_eps
        # mixer: RMSNorm over channels, depthwise causal conv, gamma, residual
        normed = _rmsnorm_affine(
            x.transpose(0, 2, 1), block["norm_weight"], eps, dtype=self.dtype
        ).transpose(0, 2, 1)
        mixed = self._stream_conv1d(
            f"{key}.mixer", normed, block["conv_weight"], block["conv_bias"],
            self.spec.kernel_size, groups=width,
        )
        mixed = _round("gamma_mul", mixed * block["gamma"].reshape(1, width, 1), self.dtype)
        x = _round("residual_add", x + mixed, self.dtype)
        # ffn: RMSNorm -> linear(4x) -> erf-GELU -> linear -> ffn_gamma -> residual
        normed = _rmsnorm_affine(
            x.transpose(0, 2, 1), block["ffn_norm_weight"], eps, dtype=self.dtype
        )
        h = vibevoice_linear(
            normed, block["ffn_linear1_weight"], block["ffn_linear1_bias"], dtype=self.dtype
        )
        h = _round("gelu_out", _gelu_erf(h.astype(np.float32)), self.dtype)
        h = vibevoice_linear(
            h, block["ffn_linear2_weight"], block["ffn_linear2_bias"], dtype=self.dtype
        )
        h = _round("ffn_gamma_mul", h * block["ffn_gamma"].reshape(1, 1, -1), self.dtype)
        x = _round("ffn_residual_add", x + h.transpose(0, 2, 1), self.dtype)
        return x

    # -- public API -------------------------------------------------------
    def decode(self, latent: ArrayLike) -> np.ndarray:
        """One decode call: ``(dimension,)`` or ``(batch, dimension)`` latent rows.

        Returns ``(hop_length,)`` or ``(batch, hop_length)`` float samples.
        Each call consumes exactly one cache state transition per conv layer,
        exactly as the fork's ``decode(..., cache=..., use_cache=True)`` does.
        """
        xb = np.asarray(latent, dtype=np.float32)
        single = xb.ndim == 1
        x2 = xb[None, :] if single else xb
        if x2.ndim != 2 or x2.shape[1] != self.spec.dimension:
            raise ValueError(f"latent must be (batch, {self.spec.dimension}), got {x2.shape}")
        outputs = np.stack([self._decode_one(row) for row in x2], axis=0)
        return outputs[0] if single else outputs

    def _decode_one(self, lat: np.ndarray) -> np.ndarray:
        spec = self.spec
        x = lat.reshape(1, spec.dimension, 1)
        # stage 0: stem conv, then depths[0] blocks at the widest width
        x = self._stream_conv1d(
            "stem", x, self.weights.stem_conv_weight, self.weights.stem_conv_bias,
            spec.kernel_size, groups=1,
        )
        layout = spec.block_layout()
        width0, depth0 = layout[0]
        for j in range(depth0):
            x = self._stream_block(x, self.weights.blocks[j], width0, f"stage0.{j}")
        block_cursor = depth0
        for i, ratio in enumerate(spec.ratios):
            width_in, width_out = layout[i][0], layout[i + 1][0]
            x = self._stream_convtr(
                f"convtr{i}", x, self.weights.convtr_weights[i],
                self.weights.convtr_biases[i], ratio,
            )
            for j in range(layout[i + 1][1]):
                x = self._stream_block(
                    x, self.weights.blocks[block_cursor + j], width_out, f"stage{i + 1}.{j}"
                )
            block_cursor += layout[i + 1][1]
        x = self._stream_conv1d(
            "head", x, self.weights.head_conv_weight, self.weights.head_conv_bias,
            spec.last_kernel_size, groups=1,
        )
        result = _round("pcm", x.reshape(-1), self.dtype)
        _finite("pcm", result)
        return result


def decode_frames(
    spec: VibevoiceDecoderSpec,
    weights: VibevoiceDecoderWeights,
    scaled_latents: ArrayLike,
    *,
    reset_before: tuple[int, ...] = (),
    dtype: str | None = "bfloat16",
) -> np.ndarray:
    """Replay a full decode sequence: ``(n, dimension)`` -> ``(n, hop_length)``.

    ``reset_before`` lists frame indices whose call starts with zeroed caches
    (the loop's speech-boundary resets, derived from the generated token
    stream). Returns the per-frame chunks; concatenating along time reproduces
    the waveform the fork's loop assembles.
    """
    stream = VibevoiceDecoderStream(spec, weights, dtype=dtype)
    lat = np.asarray(scaled_latents, dtype=np.float32)
    if lat.ndim != 2:
        raise ValueError("scaled_latents must be (n, dimension)")
    chunks = []
    for i in range(lat.shape[0]):
        if i in reset_before:
            stream.reset()
        chunks.append(stream.decode(lat[i]))
    return np.stack(chunks, axis=0)
