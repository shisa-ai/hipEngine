"""CPU FP32 reference for the VibeVoice ASR audio front-end.

Mirrors the Hugging Face ``VibeVoiceAsr*`` implementation (transformers 5.15.0)
exactly for the torch-free hipEngine port:

- ``vibevoice_causal_conv1d``: causal 1D convolution with explicit left pad
  ``left_pad = (kernel - 1) * dilation - (stride - 1)`` and a valid-window
  convolution (FP32 accumulation, optional FP16/BF16 output rounding).
- ``VibevoiceTokenizerEncoder``: stem conv K7 -> ConvNeXt blocks -> 6 strided
  downsample stages -> head conv; total stride 3200 at 24 kHz.
- ConvNeXt block: RMSNorm over channels -> depthwise causal conv -> gamma
  layer scale -> residual; then RMSNorm -> linear(4x) -> erf-GELU -> linear
  -> ffn_gamma scale -> residual.
- ``vibevoice_sample_acoustic``: per-example scale ``vae_std * randn`` times
  per-element noise, added after all chunks are concatenated.
- ``vibevoice_connector``: linear -> RMSNorm(1e-6) -> linear per latent, the
  acoustic and semantic paths summed.

Accumulation is FP32; the ``dtype`` argument only rounds the *stored* output
tensors, matching the compiled-PyTorch oracle convention of the Moonshine
references. Weight layout follows ``nn.Conv1d``: conv weights are
``[out_channels, in_channels // groups, kernel]``; linear weights are
``[out_features, in_features]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

ArrayLike = Any


def _round(name: str, value: ArrayLike, dtype: str | None) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating-point array")
    if dtype is None:
        return array.astype(np.float32)
    if dtype == "float16":
        return array.astype(np.float16)
    if dtype == "bfloat16":
        # round-to-nearest-even BF16 via the float32-bit trick
        bits = array.astype(np.float32).view(np.uint32)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        return rounded.view(np.float32).astype(np.float32)
    raise ValueError(f"unsupported dtype {dtype!r}")


def _finite(name: str, value: np.ndarray) -> None:
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def vibevoice_causal_conv1d(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    *,
    dtype: str | None = None,
) -> np.ndarray:
    """Causal conv over channel-first ``(batch, in_channels, length)``.

    Left pad is ``(kernel - 1) * dilation - (stride - 1)`` zeros; the
    convolution is a valid window pass with FP32 accumulation. Output length
    is ``(length + left_pad - dilation * (kernel - 1) - 1) // stride + 1``.
    """
    xb = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    b = None if bias is None else np.asarray(bias, dtype=np.float32)
    if xb.ndim != 3:
        raise ValueError("x must be (batch, in_channels, length)")
    batch, in_channels, length = xb.shape
    out_channels, in_per_group, kernel = w.shape
    if in_channels != in_per_group * groups:
        raise ValueError("channel/group mismatch")
    left_pad = (kernel - 1) * dilation - (stride - 1)
    if left_pad < 0:
        raise ValueError("invalid causal padding")
    padded = np.zeros((batch, in_channels, length + left_pad), dtype=np.float32)
    padded[:, :, left_pad:] = xb
    out_length = (length + left_pad - dilation * (kernel - 1) - 1) // stride + 1
    out = np.zeros((batch, out_channels, out_length), dtype=np.float32)
    out_per_group = out_channels // groups
    for g in range(groups):
        # im2col: per-group taps at rows ci*K + k to match nn.Conv1d weight
        # layout [out, in, kernel] reshaped to (out_per_group, in*K)
        cols = np.empty((batch, in_per_group * kernel, out_length), dtype=np.float32)
        ch_lo, ch_hi = g * in_per_group, (g + 1) * in_per_group
        row_index = np.arange(in_per_group) * kernel
        for k in range(kernel):
            seg = padded[:, ch_lo:ch_hi, k * dilation : k * dilation + stride * out_length]
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


def vibevoice_rmsnorm(x: ArrayLike, weight: ArrayLike, eps: float, *, dtype: str | None = None) -> np.ndarray:
    """RMSNorm over the last axis: FP32 statistics, weight applied to the result."""
    xb = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    variance = np.mean(xb * xb, axis=-1, keepdims=True)
    normed = xb / np.sqrt(variance + eps)
    result = _round("rmsnorm_out", normed * w.reshape(*([1] * (xb.ndim - 1)), -1), dtype)
    _finite("rmsnorm_out", result)
    return result


def _gelu_erf(x: np.ndarray) -> np.ndarray:
    """Exact erf GELU (``ACT2FN['gelu']``), FP32.

    Uses ``scipy.special.erf`` when importable (fast, float64 internally) and
    falls back to ``math.erf`` via vectorize, matching the evie reference.
    """
    x32 = x.astype(np.float32)
    half = np.float32(0.5)
    one = np.float32(1.0)
    try:
        from scipy.special import erf as _erf
    except ImportError:
        _erf = np.vectorize(math.erf, otypes=[np.float32])
        with np.errstate(over="ignore"):
            return half * x32 * (one + _erf(x32 / np.float32(math.sqrt(2.0))))
    erf = np.asarray(_erf(x32 / np.float32(math.sqrt(2.0)))).astype(np.float32)
    return half * x32 * (one + erf)


def vibevoice_linear(x: ArrayLike, weight: ArrayLike, bias: ArrayLike | None, *, dtype: str | None = None) -> np.ndarray:
    """Row-major linear ``x @ weight.T + bias`` with FP32 accumulation."""
    xb = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    b = None if bias is None else np.asarray(bias, dtype=np.float32)
    out = np.matmul(xb, w.T)
    if b is not None:
        out = out + b.reshape(*([1] * (xb.ndim - 1)), -1)
    result = _round("linear_out", out, dtype)
    _finite("linear_out", result)
    return result


@dataclass(frozen=True)
class VibevoiceTokenizerEncoderSpec:
    """Static geometry of one tokenizer encoder (acoustic or semantic)."""

    hidden_size: int
    depths: tuple[int, ...]
    ratios: tuple[int, ...]
    num_filters: int
    kernel_size: int
    ffn_expansion: int
    rms_norm_eps: float
    vae_std: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VibevoiceTokenizerEncoderSpec":
        return cls(
            hidden_size=int(config["hidden_size"]),
            depths=tuple(int(d) for d in config["depths"]),
            ratios=tuple(int(r) for r in config["downsampling_ratios"]),
            num_filters=int(config["num_filters"]),
            kernel_size=int(config["kernel_size"]),
            ffn_expansion=int(config["ffn_expansion"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            vae_std=float(config.get("vae_std", 0.625)),
        )

    @property
    def hop_length(self) -> int:
        return int(math.prod(self.ratios))

    def frame_count(self, num_samples: int) -> int:
        """Frames produced by a full pass: nested floor over stage strides."""
        frames = num_samples
        for r in self.ratios:
            frames //= r
        return frames

    def block_layout(self) -> tuple[tuple[int, int, int], ...]:
        """(width, stride_of_entry_conv, num_blocks) per stage, stem first.

        The stem is entry stride 1 with ``depths[0]`` blocks at width
        ``num_filters``; stage ``s`` (0-based) enters with a strided conv of
        kernel ``2*ratio``, stride ``ratio``, doubling width, and holds
        ``depths[s+1]`` ConvNeXt blocks.
        """
        layout = [(self.num_filters, 1, self.depths[0])]
        for s, ratio in enumerate(self.ratios):
            layout.append((self.num_filters * (2 ** (s + 1)), ratio, self.depths[s + 1]))
        return tuple(layout)


@dataclass(frozen=True)
class VibevoiceTokenizerEncoderWeights:
    """Flat weight bundle for one tokenizer encoder.

    Key naming keeps the checkpoint order: ``stem_conv_{weight,bias}``,
    ``stage{s}_conv_{weight,bias}`` for the six strided entry convs,
    ``head_conv_{weight,bias}``, and per-block dictionaries in
    ``blocks`` with ``norm``, ``gamma``, ``conv``, ``ffn_norm``,
    ``ffn_gamma``, ``ffn_linear1``, ``ffn_linear2`` entries (weights and
    biases suffixed ``_weight``/``_bias``).
    """

    stem_conv_weight: np.ndarray
    stem_conv_bias: np.ndarray
    stage_conv_weights: tuple[np.ndarray, ...]
    stage_conv_biases: tuple[np.ndarray, ...]
    head_conv_weight: np.ndarray
    head_conv_bias: np.ndarray
    blocks: tuple[dict[str, np.ndarray], ...]


def _convnext_block(
    x: np.ndarray,
    block: Mapping[str, np.ndarray],
    spec: VibevoiceTokenizerEncoderSpec,
    width: int,
    *,
    dtype: str | None,
) -> np.ndarray:
    """One ConvNeXt-1D block over channel-first ``x`` (batch, width, length)."""
    # mixer: norm over channels at each position
    normed = vibevoice_rmsnorm(
        x.transpose(0, 2, 1), block["norm_weight"], spec.rms_norm_eps, dtype=dtype
    ).transpose(0, 2, 1)
    mixed = vibevoice_causal_conv1d(
        normed, block["conv_weight"], block["conv_bias"], stride=1, dilation=1, groups=width, dtype=dtype
    )
    x = x + mixed * block["gamma"].reshape(1, width, 1)
    # ffn: norm -> linear -> gelu -> linear, positions as rows
    normed = vibevoice_rmsnorm(x.transpose(0, 2, 1), block["ffn_norm_weight"], spec.rms_norm_eps, dtype=dtype)
    h = vibevoice_linear(normed, block["ffn_linear1_weight"], block["ffn_linear1_bias"], dtype=dtype)
    h = _gelu_erf(h.astype(np.float32))
    h = vibevoice_linear(h, block["ffn_linear2_weight"], block["ffn_linear2_bias"], dtype=dtype)
    x = x + h.transpose(0, 2, 1) * block["ffn_gamma"].reshape(1, width, 1)
    return x


def vibevoice_tokenizer_encoder_forward(
    spec: VibevoiceTokenizerEncoderSpec,
    weights: VibevoiceTokenizerEncoderWeights,
    pcm: ArrayLike,
    *,
    dtype: str | None = None,
) -> np.ndarray:
    """Full encoder pass: ``(batch, samples)`` mono -> ``(batch, frames, hidden)``.

    No chunk cache; this is the single-pass reference. Chunk-carry equivalence
    is a kernel/runtime concern and is validated against the chunk fixture.
    """
    xb = np.asarray(pcm, dtype=np.float32)
    if xb.ndim == 1:
        xb = xb[None, :]
    batch, length = xb.shape
    x = xb[:, None, :]  # (batch, 1 channel, samples)
    x = vibevoice_causal_conv1d(x, weights.stem_conv_weight, weights.stem_conv_bias, dtype=dtype)
    width = spec.num_filters
    block_iter = iter(weights.blocks)
    for _ in range(spec.depths[0]):
        x = _convnext_block(x, next(block_iter), spec, width, dtype=dtype)
    for s, ratio in enumerate(spec.ratios):
        x = vibevoice_causal_conv1d(
            x,
            weights.stage_conv_weights[s],
            weights.stage_conv_biases[s],
            stride=ratio,
            dilation=1,
            groups=1,
            dtype=dtype,
        )
        width *= 2
        for _ in range(spec.depths[s + 1]):
            x = _convnext_block(x, next(block_iter), spec, width, dtype=dtype)
    x = vibevoice_causal_conv1d(x, weights.head_conv_weight, weights.head_conv_bias, dtype=dtype)
    latents = x.transpose(0, 2, 1)
    result = _round("latents", latents, dtype)
    _finite("latents", result)
    return result


def vibevoice_sample_acoustic(
    latents: np.ndarray,
    noise_scale: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """Add per-example scaled noise: ``latents + scale[:, None, None] * noise``.

    ``noise_scale`` is the recorded ``vae_std * randn(batch)`` tensor and
    ``noise`` the recorded per-element draw; both come from the oracle fixture
    so torch/HIP RNG differences never enter parity comparisons.
    """
    if latents.shape != noise.shape:
        raise ValueError("latent/noise shape mismatch")
    scale = np.asarray(noise_scale, dtype=np.float32)
    if scale.shape != (latents.shape[0],):
        raise ValueError("noise_scale must be (batch,)")
    return latents + scale[:, None, None] * np.asarray(noise, dtype=np.float32)


@dataclass(frozen=True)
class VibevoiceConnectorWeights:
    """``{acoustic,semantic}_connector`` weight bundle (linear-norm-linear)."""

    fc1_weight: np.ndarray
    fc1_bias: np.ndarray
    norm_weight: np.ndarray
    fc2_weight: np.ndarray
    fc2_bias: np.ndarray


def vibevoice_connector(
    weights: VibevoiceConnectorWeights,
    latents: ArrayLike,
    *,
    dtype: str | None = None,
) -> np.ndarray:
    """One connector path: linear -> RMSNorm(1e-6) -> linear."""
    h = vibevoice_linear(latents, weights.fc1_weight, weights.fc1_bias, dtype=dtype)
    h = vibevoice_rmsnorm(h, weights.norm_weight, 1e-6, dtype=dtype)
    return vibevoice_linear(h, weights.fc2_weight, weights.fc2_bias, dtype=dtype)
