"""CPU FP32-accumulated Moonshine encoder reference for CUDA sm_120a validation.

Mirrors the HF ``MoonshineEncoder`` forward exactly: conv1+tanh, GroupNorm(1,
416), conv2+gelu, conv3+gelu, permute to (batch, frames, hidden), then eight
MoonshineEncoderLayer blocks (input LayerNorm -> non-causal self-attention with
partial RoPE -> residual -> post-attention LayerNorm -> fc1+GELU+fc2 -> residual)
and a final LayerNorm.  All intermediate accumulation is FP32; only the stored
FP16 tensors are rounded to FP16, matching the compiled PyTorch CUDA FP16
encoder used as the C4 bring-up oracle.
"""

from __future__ import annotations

import math

from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.kernels.cpu_reference.moonshine import (
    moonshine_apply_partial_rope,
    moonshine_attention,
    moonshine_layernorm,
)

ArrayLike = Any


def _fp16(name: str, value: ArrayLike) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating-point array")
    return array.astype(np.float16)


def _finite(name: str, value: np.ndarray) -> None:
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def moonshine_conv1d(
    x: ArrayLike,
    weight: ArrayLike,
    stride: int,
    bias: ArrayLike | None = None,
) -> np.ndarray:
    """Valid 1D convolution over channel-first ``(batch, in_channels, length)``.

    FP32 accumulation, FP16 output.  ``weight`` is ``[out, in, kernel]`` and
    ``bias`` (optional) is ``[out]``.  ``L_out = (L_in - kernel)//stride + 1``.
    """

    x_arr = _fp16("x", x)
    weight_arr = _fp16("weight", weight)
    if x_arr.ndim != 3:
        raise ValueError("x must have shape [batch, in_channels, length]")
    if weight_arr.ndim != 3:
        raise ValueError("weight must have shape [out_channels, in_channels, kernel]")
    batch, in_channels, length = x_arr.shape
    out_channels, weight_in, kernel = weight_arr.shape
    if weight_in != in_channels:
        raise ValueError("weight in_channels must match x in_channels")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if kernel > length:
        raise ValueError("kernel cannot exceed input length")
    out_length = (length - kernel) // stride + 1
    if out_length <= 0:
        raise ValueError("convolution produces no output positions")
    bias_arr = None
    if bias is not None:
        bias_arr = _fp16("bias", bias)
        if bias_arr.shape != (out_channels,):
            raise ValueError("bias must have shape [out_channels]")
    output = np.empty((batch, out_channels, out_length), dtype=np.float32)
    for b in range(batch):
        for out_channel in range(out_channels):
            for position in range(out_length):
                window = x_arr[
                    b, :, position * stride : position * stride + kernel
                ].astype(np.float32)
                weights = weight_arr[out_channel].astype(np.float32)
                accumulator = np.sum(window * weights, dtype=np.float32)
                if bias_arr is not None:
                    accumulator = accumulator + bias_arr[out_channel].astype(np.float32)
                output[b, out_channel, position] = accumulator
    result = output.astype(np.float16)
    _finite("conv1d output", result)
    return result


def moonshine_groupnorm(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike,
    eps: float = 1.0e-5,
) -> np.ndarray:
    """GroupNorm with one group over the channel axis.

    For ``num_groups=1`` on ``[batch, channels, length]`` PyTorch normalizes each
    batch sample over every channel and every spatial position together (the
    single group spans the full channel axis and the spatial dims), then applies
    per-channel affine.  This is NOT per-position InstanceNorm/LayerNorm.
    """

    x_arr = _fp16("x", x)
    weight_arr = _fp16("weight", weight)
    bias_arr = _fp16("bias", bias)
    if x_arr.ndim != 3:
        raise ValueError("x must have shape [batch, channels, length]")
    channels = x_arr.shape[1]
    if weight_arr.shape != (channels,) or bias_arr.shape != (channels,):
        raise ValueError("weight and bias must have shape [channels]")
    eps_value = np.float32(eps)
    if not np.isfinite(eps_value) or eps_value <= 0:
        raise ValueError("eps must be positive and finite")
    x32 = x_arr.astype(np.float32)
    mean = np.mean(x32, axis=(1, 2), keepdims=True, dtype=np.float32)
    centered = (x32 - mean).astype(np.float32)
    variance = np.mean(centered * centered, axis=(1, 2), keepdims=True, dtype=np.float32)
    output = (
        centered
        * np.reciprocal(np.sqrt((variance + eps_value).astype(np.float32)))
        * weight_arr.astype(np.float32)[None, :, None]
        + bias_arr.astype(np.float32)[None, :, None]
    ).astype(np.float16)
    _finite("groupnorm output", output)
    return output


def moonshine_gelu(x: ArrayLike) -> np.ndarray:
    """Exact (erf) GELU with FP32 accumulation, matching ``nn.functional.gelu``."""

    x_arr = _fp16("x", x)
    x32 = x_arr.astype(np.float32)
    output = (
        0.5 * x32 * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(x32 / np.sqrt(2.0)))
    ).astype(np.float16)
    _finite("gelu output", output)
    return output


def _projection(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
) -> np.ndarray:
    """FP32-accumulated linear map over the last axis, FP16 output."""

    x_arr = _fp16("x", x)
    weight_arr = _fp16("weight", weight)
    if x_arr.shape[-1] != weight_arr.shape[1]:
        raise ValueError("x features must match weight in_features")
    output = np.matmul(x_arr.astype(np.float32), weight_arr.astype(np.float32).T)
    if bias is not None:
        bias_arr = _fp16("bias", bias)
        if bias_arr.shape != (weight_arr.shape[0],):
            raise ValueError("bias must match weight out_features")
        output = output + bias_arr.astype(np.float32)
    result = output.astype(np.float16)
    _finite("projection output", result)
    return result


def moonshine_encoder_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Non-causal full-sequence encoder self-attention returning ``[batch, seq, hidden]``."""

    query_arr = _fp16("query", query)
    key_arr = _fp16("key", key)
    value_arr = _fp16("value", value)
    if query_arr.ndim != 4 or key_arr.ndim != 4 or value_arr.ndim != 4:
        raise ValueError("query, key, value must have shape [batch, heads, sequence, dim]")
    if query_arr.shape[:2] != key_arr.shape[:2] or key_arr.shape != value_arr.shape:
        raise ValueError("query/key/value batch, head, and sequence dimensions must match")
    head_output = moonshine_attention(
        query_arr,
        key_arr,
        value_arr,
        mask=mask,
        scale=scale,
        causal=False,
    )
    batch, heads, sequence, head_dim = head_output.shape
    output = head_output.transpose(0, 2, 1, 3).reshape(batch, sequence, heads * head_dim)
    result = output.astype(np.float16)
    _finite("encoder attention output", result)
    return result


def moonshine_encoder_mlp(
    x: ArrayLike,
    fc1_weight: ArrayLike,
    fc1_bias: ArrayLike,
    fc2_weight: ArrayLike,
    fc2_bias: ArrayLike,
) -> np.ndarray:
    """Encoder MLP: fc1 -> exact GELU -> fc2 (FP32 accumulation, FP16 output)."""

    intermediate = _projection(x, fc1_weight, fc1_bias)
    activated = moonshine_gelu(intermediate)
    return _projection(activated, fc2_weight, fc2_bias)


def moonshine_encoder_layer(
    hidden: ArrayLike,
    weights: Mapping[str, ArrayLike],
    prefix: str,
    cos: ArrayLike,
    sin: ArrayLike,
    *,
    attention_mask: ArrayLike | None = None,
    heads: int = 8,
    head_dim: int = 52,
    rotary_dim: int = 32,
    scale: float | None = None,
) -> np.ndarray:
    """One MoonshineEncoderLayer block (input LN -> self-attn -> residual -> MLP)."""

    hidden_arr = _fp16("hidden", hidden)
    batch, sequence, hidden_size = hidden_arr.shape
    residual = hidden_arr

    normalized = moonshine_layernorm(
        hidden_arr, weights[f"{prefix}.input_layernorm.weight"], eps=1.0e-5
    )
    query = _projection(normalized, weights[f"{prefix}.self_attn.q_proj.weight"])
    key = _projection(normalized, weights[f"{prefix}.self_attn.k_proj.weight"])
    value = _projection(normalized, weights[f"{prefix}.self_attn.v_proj.weight"])
    query_h = query.reshape(batch, sequence, heads, head_dim).transpose(0, 2, 1, 3)
    key_h = key.reshape(batch, sequence, heads, head_dim).transpose(0, 2, 1, 3)
    value_h = value.reshape(batch, sequence, heads, head_dim).transpose(0, 2, 1, 3)
    positions = np.arange(sequence, dtype=np.int64)[None, :]
    query_h, key_h = moonshine_apply_partial_rope(
        query_h, key_h, cos, sin, position_ids=positions, rotary_dim=rotary_dim
    )
    attention_output = moonshine_encoder_attention(
        query_h,
        key_h,
        value_h,
        mask=attention_mask,
        scale=scale,
    )
    projection = _projection(
        attention_output, weights[f"{prefix}.self_attn.o_proj.weight"]
    )
    hidden_arr = (residual.astype(np.float32) + projection.astype(np.float32)).astype(np.float16)
    _finite("post-attention hidden", hidden_arr)

    residual = hidden_arr
    normalized = moonshine_layernorm(
        hidden_arr, weights[f"{prefix}.post_attention_layernorm.weight"], eps=1.0e-5
    )
    mlp_output = moonshine_encoder_mlp(
        normalized,
        weights[f"{prefix}.mlp.fc1.weight"],
        weights[f"{prefix}.mlp.fc1.bias"],
        weights[f"{prefix}.mlp.fc2.weight"],
        weights[f"{prefix}.mlp.fc2.bias"],
    )
    hidden_arr = (residual.astype(np.float32) + mlp_output.astype(np.float32)).astype(np.float16)
    _finite("post-mlp hidden", hidden_arr)
    return hidden_arr


def moonshine_encoder_forward(
    input_values: ArrayLike,
    weights: dict[str, ArrayLike],
    *,
    conv_eps: float = 1.0e-5,
    layer_norm_eps: float = 1.0e-5,
    heads: int = 8,
    head_dim: int = 52,
    rotary_dim: int = 32,
    input_attention_mask: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Full Moonshine encoder forward returning ``(hidden [1, L, 416], mask)``.

    ``weights`` uses the HF checkpoint key names (``model.encoder.*``).  The
    returned mask is the downsampled int32 encoder attention mask (or None when
    ``input_attention_mask`` is None).
    """

    values = _fp16("input_values", input_values)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("input_values must have shape [1, audio_length]")
    audio_length = values.shape[1]
    hidden_size = weights["model.encoder.conv1.weight"].shape[0]

    x = values[:, None, :]
    x = moonshine_conv1d(x, weights["model.encoder.conv1.weight"], stride=64)
    x = np.tanh(x.astype(np.float32)).astype(np.float16)
    x = moonshine_groupnorm(
        x,
        weights["model.encoder.groupnorm.weight"],
        weights["model.encoder.groupnorm.bias"],
        eps=conv_eps,
    )
    x = moonshine_conv1d(
        x, weights["model.encoder.conv2.weight"], stride=3, bias=weights["model.encoder.conv2.bias"]
    )
    x = moonshine_gelu(x)
    x = moonshine_conv1d(
        x, weights["model.encoder.conv3.weight"], stride=2, bias=weights["model.encoder.conv3.bias"]
    )
    x = moonshine_gelu(x)
    hidden = x.transpose(0, 2, 1).astype(np.float16)  # (1, L, 416)
    _finite("encoder conv hidden", hidden)

    output_mask = None
    if input_attention_mask is not None:
        mask_values = np.asarray(input_attention_mask)
        if mask_values.shape != (1, audio_length):
            raise ValueError("input_attention_mask must have shape [1, audio_length]")
        output_length = hidden.shape[1]
        downsample_stride = 64 * 3 * 2
        output_mask = (
            mask_values[..., ::downsample_stride][..., :output_length].astype(np.int32)
        )

    # Encoder RoPE tables for positions 0..L-1 (rotary_dim / 2 pairs).
    pairs = rotary_dim // 2
    dimensions = np.arange(0, rotary_dim, 2, dtype=np.float32)
    inverse_frequency = np.reciprocal(np.power(np.float32(10_000.0), dimensions / rotary_dim))
    positions = np.arange(hidden.shape[1], dtype=np.float32)
    angles = np.multiply.outer(positions, inverse_frequency).astype(np.float32)
    cos = np.cos(angles).astype(np.float16)
    sin = np.sin(angles).astype(np.float16)

    for layer in range(8):
        prefix = f"model.encoder.layers.{layer}"
        hidden = moonshine_encoder_layer(
            hidden,
            weights,
            prefix,
            cos,
            sin,
            attention_mask=output_mask,
            heads=heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            scale=head_dim**-0.5,
        )

    hidden = moonshine_layernorm(
        hidden, weights["model.encoder.layer_norm.weight"], eps=layer_norm_eps
    )
    return hidden, output_mask
