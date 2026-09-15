"""CPU FP32 VibeVoice acoustic-tokenizer decoder reference.

Mirrors the community fork's ``TokenizerDecoder`` forward exactly (fork
``vibevoice/modular/modular_vibevoice_tokenizer.py`` @ ``952326dd``): causal
streaming convolutions with per-layer context caches, RMSNorm over the channel
axis, depthwise-conv mixers with layer-scale residuals, GELU FFNs with their own
layer-scale residuals, and causal transposed-conv upsampling. All accumulation
is FP32; checkpoint BF16 tensors are widened on load.

The reference implements the streaming path as the primary interface (one
latent frame per call, the shape the TTS session runs at 7.5 frames/s) and a
non-streaming full-sequence path for validation: for latents whose length is a
whole number of frames both paths consume identical convolution windows, so
they are bit-comparable at FP32.

Weight names follow the checkpoint exactly, e.g.
``model.acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.weight`` is
the stem ``SConv1d`` (a ``NormConv1d`` wrapping ``nn.Conv1d``) and
``...stages.0.0.mixer.conv.conv.conv.weight`` is a depthwise mixer conv.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

ArrayLike = Any

PREFIX = "model.acoustic_tokenizer.decoder."


def bf16_bytes_to_f32(payload: bytes) -> np.ndarray:
    """Widen little-endian BF16 storage to FP32 (bit shift, no rounding)."""
    raw = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
    return (raw << 16).view("<f4").astype(np.float32)


@dataclass(frozen=True)
class DecoderGeometry:
    """Decoder shape contract, checked against the checkpoint on load."""

    latent_dim: int = 64
    channels: int = 1
    n_filters: int = 32
    ratios: tuple[int, ...] = (8, 5, 5, 4, 2, 2)
    depths: tuple[int, ...] = (8, 3, 3, 3, 3, 3, 3)
    kernel_size: int = 7
    last_kernel_size: int = 7
    rms_eps: float = 1e-5

    @property
    def n_stages(self) -> int:
        return len(self.depths)

    @property
    def hop_length(self) -> int:
        out = 1
        for r in self.ratios:
            out *= r
        return out

    def dims(self) -> tuple[int, ...]:
        return tuple(
            self.n_filters * (2 ** (self.n_stages - 1 - i)) for i in range(self.n_stages)
        )


def expected_decoder_weight_shapes(
    geometry: DecoderGeometry,
) -> dict[str, tuple[int, ...]]:
    """Every decoder tensor name and shape the checkpoint must provide."""

    shapes: dict[str, tuple[int, ...]] = {}
    dims = geometry.dims()
    top = dims[0]

    # Stem: SConv1d(latent_dim -> top, kernel_size, stride 1).
    shapes["upsample_layers.0.0.conv.conv.weight"] = (
        top,
        geometry.latent_dim,
        geometry.kernel_size,
    )
    shapes["upsample_layers.0.0.conv.conv.bias"] = (top,)
    # Upsample stages: SConvTranspose1d(dims[i] -> dims[i+1], kernel 2*r, stride r).
    for i, ratio in enumerate(geometry.ratios):
        shapes[f"upsample_layers.{i + 1}.0.convtr.convtr.weight"] = (
            dims[i],
            dims[i + 1],
            ratio * 2,
        )
        shapes[f"upsample_layers.{i + 1}.0.convtr.convtr.bias"] = (dims[i + 1],)
    # Blocks: depthwise mixer conv, RMSNorm weights, layer scales, GELU FFN.
    for i, dim in enumerate(dims):
        for j in range(geometry.depths[i]):
            base = f"stages.{i}.{j}."
            shapes[base + "norm.weight"] = (dim,)
            shapes[base + "gamma"] = (dim,)
            shapes[base + "mixer.conv.conv.conv.weight"] = (dim, 1, geometry.kernel_size)
            shapes[base + "mixer.conv.conv.conv.bias"] = (dim,)
            shapes[base + "ffn_norm.weight"] = (dim,)
            shapes[base + "ffn_gamma"] = (dim,)
            shapes[base + "ffn.linear1.weight"] = (4 * dim, dim)
            shapes[base + "ffn.linear1.bias"] = (4 * dim,)
            shapes[base + "ffn.linear2.weight"] = (dim, 4 * dim)
            shapes[base + "ffn.linear2.bias"] = (dim,)
    # Head: SConv1d(dims[-1] -> channels, last_kernel_size, stride 1).
    shapes["head.conv.conv.weight"] = (
        geometry.channels,
        dims[-1],
        geometry.last_kernel_size,
    )
    shapes["head.conv.conv.bias"] = (geometry.channels,)
    return shapes


def load_decoder_weights(
    read_tensor,  # Callable[[str], bytes] over full checkpoint names
    geometry: DecoderGeometry = DecoderGeometry(),
) -> dict[str, np.ndarray]:
    """Decode every decoder tensor to FP32 and validate the shape contract."""

    expected = expected_decoder_weight_shapes(geometry)
    weights: dict[str, np.ndarray] = {}
    for short_name, shape in expected.items():
        full = PREFIX + short_name
        # Storage bytes are flat; the expected shape is the contract.
        array = bf16_bytes_to_f32(read_tensor(full)).reshape(shape)
        if array.shape != shape:
            raise ValueError(
                f"{full}: expected shape {shape}, checkpoint has {array.shape}"
            )
        weights[short_name] = array
    return weights


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def conv1d_valid(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    name: str,
) -> np.ndarray:
    """Valid dense causal conv1d over ``[in_channels, length]``.

    ``weight`` is ``[out_channels, in_channels, kernel]``. Output
    ``[out_channels, length - kernel + 1]``. FP32 accumulation, taps summed in
    kernel order.
    """

    in_channels, length = x.shape
    out_channels, weight_in, kernel = weight.shape
    if weight_in != in_channels:
        raise ValueError(f"{name}: weight expects {weight_in} input channels, x has {in_channels}")
    out_length = length - kernel + 1
    if out_length <= 0:
        raise ValueError(f"{name}: kernel {kernel} exceeds input length {length}")
    output = np.zeros((out_channels, out_length), dtype=np.float32)
    for tap in range(kernel):
        # [out, in] @ [in, out_length]
        output += weight[:, :, tap] @ x[:, tap : tap + out_length]
    if bias is not None:
        output += bias[:, None]
    return _finite(name, output)


def depthwise_conv1d_valid(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    name: str,
) -> np.ndarray:
    """Valid depthwise conv1d over ``[channels, length]``.

    ``weight`` is ``[channels, 1, kernel]``; each channel convolves against its
    own taps with no cross-channel mixing.
    """

    channels, length = x.shape
    weight_channels, _, kernel = weight.shape
    if weight_channels != channels:
        raise ValueError(f"{name}: weight expects {weight_channels} channels, x has {channels}")
    out_length = length - kernel + 1
    if out_length <= 0:
        raise ValueError(f"{name}: kernel {kernel} exceeds input length {length}")
    output = np.zeros((channels, out_length), dtype=np.float32)
    for tap in range(kernel):
        output += weight[:, 0, tap][:, None] * x[:, tap : tap + out_length]
    output += bias[:, None]
    return _finite(name, output)


def conv_transpose1d_causal(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int,
    name: str,
) -> np.ndarray:
    """Causal transposed conv1d over ``[in_channels, length]``.

    ``weight`` is ``[in_channels, out_channels, kernel]`` (PyTorch's
    ``ConvTranspose1d`` layout). The fork's causal trim removes
    ``kernel - stride`` samples from the right, so the returned length is
    exactly ``length * stride``: taps that would only feed trimmed positions
    are skipped.
    """

    in_channels, length = x.shape
    weight_in, out_channels, kernel = weight.shape
    if weight_in != in_channels:
        raise ValueError(f"{name}: weight expects {weight_in} input channels, x has {in_channels}")
    trimmed = length * stride
    output = np.zeros((out_channels, trimmed), dtype=np.float32)
    for tap in range(kernel):
        positions = np.arange(length) * stride + tap
        keep = positions < trimmed
        if not keep.any():
            continue
        # [kept, in] @ [in, out] -> [kept, out], scattered into strict rows.
        contribution = x[:, keep].T @ weight[:, :, tap]
        output[:, positions[keep]] += contribution.T
    output += bias[:, None]
    return _finite(name, output)


def rms_norm_channels(x: np.ndarray, weight: np.ndarray, eps: float, name: str) -> np.ndarray:
    """RMSNorm over the channel axis of ``[channels, length]``.

    The fork's ``ConvRMSNorm`` transposes to channels-last, normalizes each
    time step over the channels in FP32, and transposes back.
    """

    channels, length = x.shape
    if weight.shape != (channels,):
        raise ValueError(f"{name}: norm weight expects {channels} channels")
    mean_square = np.mean(x.astype(np.float32) ** 2, axis=0, keepdims=True)
    normalized = x / np.sqrt(mean_square + np.float32(eps))
    return _finite(name, normalized * weight[:, None])


def gelu_exact(x: np.ndarray, name: str) -> np.ndarray:
    """Exact (erf-based) GELU, matching ``ACT2FN['gelu']`` (house pattern from
    ``cpu_reference/evie.py``: ``math.erf`` vectorized, no scipy dep)."""

    import math

    erf = np.vectorize(math.erf, otypes=[np.float32])
    return _finite(name, 0.5 * x * (1.0 + erf(x / math.sqrt(2.0))))


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, name: str) -> np.ndarray:
    """``weight`` is ``[out, in]`` (PyTorch Linear); x is ``[in, length]``."""

    out_features, in_features = weight.shape
    if x.shape[0] != in_features:
        raise ValueError(f"{name}: linear expects {in_features} rows, x has {x.shape[0]}")
    output = weight @ x
    if bias is not None:
        output += bias[:, None]
    return _finite(name, output)


class DecoderStreams:
    """Per-layer context caches for one streaming decode of one sample.

    Mirrors ``VibeVoiceTokenizerStreamingCache`` for the single-sample case:
    every ``SConv1d`` keeps the last ``kernel_size - 1`` input samples and
    every ``SConvTranspose1d`` keeps the last ``kernel_size - 1`` inputs.
    Reset (a fresh instance) at speech boundaries, matching the session's
    ``set_to_zero``.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, np.ndarray] = {}

    def get(self, key: str) -> np.ndarray:
        cached = self._buffers.get(key)
        if cached is None:
            return None
        return cached

    def set(self, key: str, prepended: np.ndarray, context: int) -> None:
        self._buffers[key] = (
            prepended[:, -context:].copy()
            if prepended.shape[1] > context
            else prepended.copy()
        )


def _conv_valid_dispatch(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, name: str
) -> np.ndarray:
    """Valid conv over pre-padded input, dense or depthwise by weight shape.

    An ``SConv1d`` carries either a dense conv (``weight`` ``[out, in, k]``
    with ``in > 1``) or a depthwise conv (groups = channels, ``weight``
    ``[C, 1, k]``); dispatch on the middle dimension.
    """

    if weight.shape[1] == 1 and weight.shape[0] > 1:
        if bias is None:
            raise ValueError(f"{name}: depthwise conv requires a bias in this codec")
        return depthwise_conv1d_valid(x, weight, bias, name)
    return conv1d_valid(x, weight, bias, name)


def _sconv1d_stream(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    key: str,
    streams: DecoderStreams,
    name: str,
) -> np.ndarray:
    """Causal stride-1 ``SConv1d`` with constant padding, streaming.

    ``context_size = kernel_size - 1`` (dilation 1, stride 1); the cache holds
    the last input samples and the conv runs over ``cat(cache, x)`` with no
    additional padding.
    """

    kernel = weight.shape[2]
    context = kernel - 1
    cached = streams.get(key)
    if cached is None:
        # First chunk: the fork initializes the context with zeros (this is
        # the final-flush semantics); a convtr keeps an empty cache instead.
        cached = np.zeros((x.shape[0], context), dtype=np.float32)
    prepended = np.concatenate([cached, x], axis=1)
    output = _conv_valid_dispatch(prepended, weight, bias, name)
    streams.set(key, prepended, context)
    return output


def _sconv1d_full(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, name: str) -> np.ndarray:
    """Non-streaming causal stride-1 ``SConv1d``: left-pad ``kernel - 1`` zeros."""

    kernel = weight.shape[2]
    padded = np.concatenate(
        [np.zeros((x.shape[0], kernel - 1), dtype=np.float32), x], axis=1
    )
    return _conv_valid_dispatch(padded, weight, bias, name)


def _convtr_stream(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int,
    key: str,
    streams: DecoderStreams,
    name: str,
) -> np.ndarray:
    """Causal ``SConvTranspose1d`` streaming: convtr over cache + input, keep
    the last ``T * stride`` samples of the trimmed output."""

    kernel = weight.shape[2]
    context = kernel - 1
    cached = streams.get(key)
    if cached is None:
        cached = np.zeros((x.shape[0], 0), dtype=np.float32)
    full_input = np.concatenate([cached, x], axis=1)
    full_output = conv_transpose1d_causal(full_input, weight, bias, stride, name)
    expected_new = x.shape[1] * stride
    output = (
        full_output[:, -expected_new:]
        if full_output.shape[1] > expected_new
        else full_output
    )
    streams.set(key, full_input, context)
    return output


def _apply_stage(
    x: np.ndarray,
    stage: int,
    weights: Mapping[str, np.ndarray],
    streams: DecoderStreams | None,
    geometry: DecoderGeometry,
) -> np.ndarray:
    """Run ``geometry.depths[stage]`` Block1D modules over ``[C, L]``.

    ``streams=None`` selects the non-streaming path (left zero padding on each
    mixer conv); otherwise each block's mixer keeps a per-block context cache
    keyed ``s<stage>b<block>``.
    """

    for block in range(geometry.depths[stage]):
        base = f"stages.{stage}.{block}."
        residual = x
        y = rms_norm_channels(
            x, weights[base + "norm.weight"], geometry.rms_eps, base + "norm"
        )
        mixer_weight = weights[base + "mixer.conv.conv.conv.weight"]
        mixer_bias = weights[base + "mixer.conv.conv.conv.bias"]
        if streams is None:
            y = _sconv1d_full(y, mixer_weight, mixer_bias, base + "mixer")
        else:
            y = _sconv1d_stream(
                y, mixer_weight, mixer_bias, f"s{stage}b{block}", streams, base + "mixer"
            )
        y = y * weights[base + "gamma"][:, None]
        x = residual + y

        residual = x
        y = rms_norm_channels(
            x, weights[base + "ffn_norm.weight"], geometry.rms_eps, base + "ffn_norm"
        )
        y = linear(
            y,
            weights[base + "ffn.linear1.weight"],
            weights[base + "ffn.linear1.bias"],
            base + "ffn1",
        )
        y = gelu_exact(y, base + "gelu")
        y = linear(
            y,
            weights[base + "ffn.linear2.weight"],
            weights[base + "ffn.linear2.bias"],
            base + "ffn2",
        )
        y = y * weights[base + "ffn_gamma"][:, None]
        x = residual + y
    return x


def decode_latent_frames(
    frames: np.ndarray,
    weights: Mapping[str, np.ndarray],
    streams: DecoderStreams,
    geometry: DecoderGeometry = DecoderGeometry(),
) -> np.ndarray:
    """Decode a chunk of ``[frames, latent_dim]`` latents to PCM samples.

    ``frames`` is ``[T, 64]`` (time-major, the model-space latent BEFORE the
    scale/bias transform); the output is ``[T * hop_length]`` float32 samples.
    Streams are updated in place: call with one frame at a time against the
    same ``DecoderStreams`` to reproduce the session's streaming decode, or
    with all frames at once for a chunked decode.
    """

    T = frames.shape[0]
    if frames.shape[1] != geometry.latent_dim:
        raise ValueError(
            f"latent expects {geometry.latent_dim} dims, got {frames.shape[1]}"
        )
    x = np.ascontiguousarray(frames.T.astype(np.float32))  # [64, T]

    # Stage 0: stem SConv1d (64 -> 2048), then depth-8 blocks at 2048.
    x = _sconv1d_stream(
        x,
        weights["upsample_layers.0.0.conv.conv.weight"],
        weights["upsample_layers.0.0.conv.conv.bias"],
        "stem",
        streams,
        "decoder.stem",
    )
    x = _apply_stage(x, 0, weights, streams, geometry)

    # Stages 1..6: transposed-conv upsample, then depth-3 blocks.
    for i, ratio in enumerate(geometry.ratios):
        x = _convtr_stream(
            x,
            weights[f"upsample_layers.{i + 1}.0.convtr.convtr.weight"],
            weights[f"upsample_layers.{i + 1}.0.convtr.convtr.bias"],
            ratio,
            f"tr{i}",
            streams,
            f"decoder.convtr{i}",
        )
        x = _apply_stage(x, i + 1, weights, streams, geometry)

    # Head: SConv1d(32 -> 1). The last norm is disabled in this checkpoint.
    x = _sconv1d_stream(
        x,
        weights["head.conv.conv.weight"],
        weights["head.conv.conv.bias"],
        "head",
        streams,
        "decoder.head",
    )
    if x.shape != (geometry.channels, T * geometry.hop_length):
        raise ValueError(
            f"decoder produced {x.shape}, expected 1 x {T * geometry.hop_length}"
        )
    return x[0]


def decode_full_sequence(
    frames: np.ndarray,
    weights: Mapping[str, np.ndarray],
    geometry: DecoderGeometry = DecoderGeometry(),
) -> np.ndarray:
    """Non-streaming full-sequence decode; the validation oracle.

    For whole-frame latents the padding windows match the streaming path, so
    outputs are comparable bit-for-bit at FP32.
    """

    T = frames.shape[0]
    x = np.ascontiguousarray(frames.T.astype(np.float32))

    x = _sconv1d_full(
        x,
        weights["upsample_layers.0.0.conv.conv.weight"],
        weights["upsample_layers.0.0.conv.conv.bias"],
        "decoder.stem",
    )
    x = _apply_stage(x, 0, weights, None, geometry)
    for i, ratio in enumerate(geometry.ratios):
        x = conv_transpose1d_causal(
            x,
            weights[f"upsample_layers.{i + 1}.0.convtr.convtr.weight"],
            weights[f"upsample_layers.{i + 1}.0.convtr.convtr.bias"],
            ratio,
            f"decoder.convtr{i}",
        )
        x = _apply_stage(x, i + 1, weights, None, geometry)
    x = _sconv1d_full(
        x,
        weights["head.conv.conv.weight"],
        weights["head.conv.conv.bias"],
        "decoder.head",
    )
    if x.shape != (geometry.channels, T * geometry.hop_length):
        raise ValueError(
            f"decoder produced {x.shape}, expected 1 x {T * geometry.hop_length}"
        )
    return x[0]
