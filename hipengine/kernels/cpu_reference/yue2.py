"""NumPy reference operators for the YuE2 AR/NAR/VAE paths.

These are the strict-shape oracles the HIP kernels are gated against. They are
small, dependency-light (NumPy only) and deliberately reproduce the released
implementation's *rounding order* where that order is observable:

* RMSNorm casts the FP32 inverse RMS to BF16 **before** multiplying the BF16
  activations and BF16 weight,
* RoPE casts cos/sin to BF16 and rounds after each BF16 product/subtract,
* the ``off`` CFG combination is BF16 throughout,
* the flow-matching solver integrates in the model dtype and returns FP32.

Matrix products accumulate in FP32 and round once, which is the arithmetic class
of every GEMM the runtime uses; the exact reduction order of a particular kernel
is not reproduced here, so comparisons use the documented numerical gates.
"""

from __future__ import annotations

import numpy as np

BF16_EPS = np.float32(1e-6)


def bf16(values) -> np.ndarray:
    """Round to nearest-even BF16, keeping FP32 storage."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding = ((bits >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)
    return np.where(np.isnan(array), array, rounded).astype(np.float32)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    return bf16(values).view(np.uint32).astype(np.uint16)


# ---------------------------------------------------------------------------
# normalization and elementwise
# ---------------------------------------------------------------------------


def rmsnorm(x, weight, eps: float = 1e-6) -> np.ndarray:
    """``x * bf16(rsqrt(mean_f32(x^2) + eps)) * weight`` in BF16."""
    values = bf16(x)
    weight = bf16(weight)
    mean_square = np.mean(np.square(values.astype(np.float32)), axis=-1, keepdims=True)
    inverse = bf16(1.0 / np.sqrt(mean_square + np.float32(eps)))
    return bf16(bf16(values * inverse) * weight)


def head_rmsnorm(x, weight, eps: float = 1e-6) -> np.ndarray:
    """Per-head Q/K RMSNorm: normalize the last (head_dim) axis."""
    return rmsnorm(x, weight, eps)


def silu_mul(gate, up) -> np.ndarray:
    gate = bf16(gate)
    up = bf16(up)
    activated = bf16(gate * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate))))
    return bf16(activated * up)


def linear(x, weight) -> np.ndarray:
    """BF16 in, FP32 accumulate, BF16 out. ``weight`` is ``[out, in]``."""
    values = bf16(x)
    weight = bf16(weight)
    return bf16(values.astype(np.float32) @ weight.astype(np.float32).T)


def linear_f32(x, weight) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T


# ---------------------------------------------------------------------------
# rotary position embedding
# ---------------------------------------------------------------------------


def rope_tables(positions, head_dim: int, theta: float = 1000000.0) -> tuple[np.ndarray, np.ndarray]:
    """FP32 cos/sin tables, half-width (the reference rotates half the head).

    The reference builds the inverse frequencies and the position angles in FP32
    (``1.0 / (base ** (arange(0, head_dim, 2, float32) / head_dim))``), so the
    tables are computed in FP32 here rather than in FP64.
    """
    half = head_dim // 2
    exponent = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    inverse = np.float32(1.0) / np.power(np.float32(theta), exponent)
    angles = np.asarray(positions, dtype=np.float32)[:, None] * inverse[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rotate_half(x, cos, sin) -> np.ndarray:
    """BF16 rotate-half RoPE; cos/sin are cast to BF16 like the reference."""
    values = bf16(x)
    half = values.shape[-1] // 2
    first, second = values[..., :half], values[..., half:]
    cos = bf16(cos)
    sin = bf16(sin)
    return np.concatenate(
        [bf16(bf16(first * cos) - bf16(second * sin)), bf16(bf16(second * cos) + bf16(first * sin))],
        axis=-1,
    )


def apply_rope(x, positions, head_dim: int, theta: float = 1000000.0) -> np.ndarray:
    """``x`` is ``[rows, heads, head_dim]``; positions is ``[rows]``."""
    cos, sin = rope_tables(positions, head_dim, theta)
    return rotate_half(x, cos[:, None, :], sin[:, None, :])


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def gqa_attention(q, k, v, *, scale: float | None = None, causal: bool = False, mask=None) -> np.ndarray:
    """Grouped-query attention in FP32 from BF16 inputs, BF16 output.

    ``q`` is ``[q_rows, q_heads, head_dim]`` and ``k``/``v`` are
    ``[kv_rows, kv_heads, head_dim]``. ``mask`` is an optional boolean
    ``[q_rows, kv_rows]`` (True = attend).
    """
    query = bf16(q).astype(np.float32)
    key = bf16(k).astype(np.float32)
    value = bf16(v).astype(np.float32)
    q_rows, q_heads, head_dim = query.shape
    kv_rows, kv_heads, _ = key.shape
    if q_heads % kv_heads:
        raise ValueError("q_heads must be a multiple of kv_heads")
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    group = q_heads // kv_heads
    out = np.empty((q_rows, q_heads, head_dim), dtype=np.float32)
    for head in range(q_heads):
        kv_head = head // group
        scores = (query[:, head, :] @ key[:, kv_head, :].T).astype(np.float32) * np.float32(scale)
        if causal:
            rows = np.arange(q_rows)[:, None]
            columns = np.arange(kv_rows)[None, :]
            offset = kv_rows - q_rows
            scores = np.where(columns <= rows + offset, scores, -np.inf)
        if mask is not None:
            scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
        maximum = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - maximum)
        total = weights.sum(axis=-1, keepdims=True)
        weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
        out[:, head, :] = weights @ value[:, kv_head, :]
    return bf16(out)


# ---------------------------------------------------------------------------
# VAE operators
# ---------------------------------------------------------------------------


def fold_weight_norm(weight_g, weight_v) -> np.ndarray:
    """PyTorch ``weight_norm`` folding: ``g / ||v||_2 * v`` over dims (1, 2)."""
    weight_g = np.asarray(weight_g, dtype=np.float32)
    weight_v = np.asarray(weight_v, dtype=np.float32)
    norm = np.sqrt(np.sum(np.square(weight_v), axis=(1, 2), keepdims=True))
    return (weight_g / np.maximum(norm, 1e-12)) * weight_v


def snake_beta(x, alpha, beta) -> np.ndarray:
    """FP32 SnakeBeta with the released 1e-9 denominator epsilon."""
    values = np.asarray(x, dtype=np.float32)
    alpha = np.exp(np.asarray(alpha, dtype=np.float32))[:, None]
    beta = np.exp(np.asarray(beta, dtype=np.float32))[:, None]
    return values + (np.float32(1.0) / (beta + np.float32(1e-9))) * np.square(
        np.sin(values * alpha)
    )


def conv1d(x, weight, bias=None, *, stride: int = 1, dilation: int = 1, padding: int = 0) -> np.ndarray:
    """FP32 Conv1d (cross-correlation), ``x`` ``[C_in, T]``, ``weight`` ``[C_out, C_in, K]``."""
    values = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    channels_in, length = values.shape
    channels_out, weight_in, kernel = weight.shape
    if channels_in != weight_in:
        raise ValueError("input channels do not match the convolution weight")
    out_length = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    out = np.zeros((channels_out, out_length), dtype=np.float32)
    for index in range(kernel):
        start = index * dilation - padding
        positions = np.arange(out_length) * stride + start
        valid = (positions >= 0) & (positions < length)
        if not valid.any():
            continue
        selected = values[:, positions[valid]]
        out[:, valid] += weight[:, :, index] @ selected
    if bias is not None:
        out += np.asarray(bias, dtype=np.float32)[:, None]
    return out


def conv_transpose1d(
    x, weight, bias=None, *, stride: int = 1, padding: int = 0, output_padding: int = 0
) -> np.ndarray:
    """FP32 ConvTranspose1d, ``weight`` ``[C_in, C_out, K]``."""
    values = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    channels_in, length = values.shape
    weight_in, channels_out, kernel = weight.shape
    if channels_in != weight_in:
        raise ValueError("input channels do not match the transposed convolution weight")
    out_length = (length - 1) * stride - 2 * padding + kernel + output_padding
    out = np.zeros((channels_out, out_length), dtype=np.float32)
    for index in range(kernel):
        positions = np.arange(length) * stride - padding + index
        valid = (positions >= 0) & (positions < out_length)
        if not valid.any():
            continue
        out[:, positions[valid]] += weight[:, :, index].T @ values[:, valid]
    if bias is not None:
        out += np.asarray(bias, dtype=np.float32)[:, None]
    return out


def dependency_interval(*, length: int, stride: int, dilation: int, kernel: int, padding: int) -> tuple[int, int]:
    """Inclusive output support of an input interval (tiling halo calculation)."""
    low = -(-(0 + padding - dilation * (kernel - 1)) // stride)
    high = (length - 1 + padding) // stride
    return low, high


# ---------------------------------------------------------------------------
# conditioning features and the midpoint solver
# ---------------------------------------------------------------------------


def timestep_embedding(t_values, w0, b0, w2, b2, frequency_size: int = 256) -> np.ndarray:
    """Reference ``TimestepEmbedder``: FP32 sinusoids, BF16 MLP, SiLU between."""
    half = frequency_size // 2
    frequencies = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    t_values = np.asarray(t_values, dtype=np.float32).reshape(-1)
    args = t_values[:, None] * frequencies[None, :]
    embedding = bf16(np.concatenate([np.cos(args), np.sin(args)], axis=-1))
    hidden = linear(embedding, w0)
    hidden = bf16(hidden + bf16(b0))
    hidden = bf16(hidden * (np.float32(1.0) / (np.float32(1.0) + np.exp(-hidden))))
    return bf16(linear(hidden, w2) + bf16(b2))


def audio_position_embedding(pe_rows, positions) -> np.ndarray:
    return np.asarray(pe_rows, dtype=np.float32)[np.asarray(positions, dtype=np.int64)]


def midpoint_solve(velocity, noise, schedule) -> np.ndarray:
    """Device-shaped midpoint integration used by the NAR chunk solver.

    ``velocity(state, raw_t)`` returns the predicted velocity for the current
    state; ``schedule`` is ``midpoint_schedule(steps)``.
    """
    state = bf16(noise)
    for _, raw_t, _, raw_t_mid in schedule:
        first = bf16(velocity(state, raw_t))
        mid = bf16(state - bf16(first * np.float32(0.5)) * np.float32(2.0 / len(schedule)))
        state = bf16(state - bf16(velocity(mid, raw_t_mid) * np.float32(1.0 / len(schedule))))
    return state.astype(np.float32)


def kl_divergence(reference_logits, candidate_logits, mask=None) -> float:
    """Mean KL(reference || candidate) over rows, ignoring non-finite entries."""
    reference = np.asarray(reference_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        reference = np.where(keep, reference, -np.inf)
        candidate = np.where(keep, candidate, -np.inf)
    reference_max = np.nanmax(np.where(np.isfinite(reference), reference, -np.inf), axis=-1, keepdims=True)
    candidate_max = np.nanmax(np.where(np.isfinite(candidate), candidate, -np.inf), axis=-1, keepdims=True)
    reference_exp = np.where(np.isfinite(reference), np.exp(reference - reference_max), 0.0)
    candidate_exp = np.where(np.isfinite(candidate), np.exp(candidate - candidate_max), 0.0)
    reference_prob = reference_exp / reference_exp.sum(axis=-1, keepdims=True)
    candidate_prob = candidate_exp / candidate_exp.sum(axis=-1, keepdims=True)
    terms = np.where(
        reference_prob > 0,
        reference_prob * (np.log(np.maximum(reference_prob, 1e-300)) - np.log(np.maximum(candidate_prob, 1e-300))),
        0.0,
    )
    return float(terms.sum(axis=-1).mean())


def top1_agreement(reference_logits, candidate_logits, mask=None) -> float:
    reference = np.asarray(reference_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        reference = np.where(keep, reference, -np.inf)
        candidate = np.where(keep, candidate, -np.inf)
    return float((reference.argmax(axis=-1) == candidate.argmax(axis=-1)).mean())
