"""Torch-free NumPy CPU reference for TimesFM 2.5 200M.

Mirrors ``google-research/timesfm`` ``src/timesfm/timesfm_2p5/timesfm_2p5_torch.py``
(``TimesFM_2p5_200M_torch_module.forward`` / ``.decode``) plus the shared
``src/timesfm/torch/{transformer,dense,normalization,util}.py`` modules
(RotaryPositionalEmbedding, PerDimScale, MultiHeadAttention, Transformer,
ResidualBlock, revin, update_running_stats).

Differences from torch beyond the framework swap: none intentional.  Attention
runs as an explicit einsum-softmax-einsum with the reference's unscaled
dot-product semantics; masked scores use ``-finfo(float32).max / 2`` exactly
like ``_dot_product_attention``.  Everything computes in float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from safetensors import safe_open

from hipengine.loading.safetensors import load_weight_index
from hipengine.models.timesfm import (
    TimesFMModelSpec,
    expected_timesfm_weight_shapes,
    parse_timesfm_model_spec,
    validate_timesfm_weight_index,
)

_MASKED_SCORE = np.finfo(np.float32).max / 2.0
_SIGMOID_THRESHOLD = 500.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def swish(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, np.zeros_like(x))


def rmsnorm(x: np.ndarray, scale: np.ndarray, epsilon: float) -> np.ndarray:
    """TimesFM RMSNorm: multiplicative ``scale`` (no +1), eps inside rsqrt."""

    var = np.mean(np.square(x), axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(var + epsilon)) * scale


@dataclass(frozen=True)
class TimesFMHostWeights:
    """Validated host-side FP32 weights keyed by checkpoint name."""

    spec: TimesFMModelSpec
    tensors: dict[str, np.ndarray]

    @classmethod
    def load(cls, model_path: str) -> "TimesFMHostWeights":
        index = load_weight_index(model_path)
        spec = parse_timesfm_model_spec(index.config)
        validate_timesfm_weight_index(spec, index)
        expected = expected_timesfm_weight_shapes(spec)
        names_by_shard: dict[Any, list[str]] = {}
        for name in expected:
            names_by_shard.setdefault(index.tensors[name].shard_path, []).append(name)
        tensors: dict[str, np.ndarray] = {}
        for shard in sorted(names_by_shard):
            with safe_open(str(shard), framework="numpy") as handle:
                for name in sorted(names_by_shard[shard]):
                    value = np.asarray(handle.get_tensor(name), dtype=np.float32)
                    if not bool(np.isfinite(value).all()):
                        raise ValueError(f"TimesFM weight {name} must be finite")
                    tensors[name] = np.ascontiguousarray(value)
        return cls(spec=spec, tensors=tensors)


@dataclass
class TimesFMDecodeCache:
    """Per-layer decode cache (patch-position indexed, not token indexed)."""

    next_index: np.ndarray  # (B,) int32
    num_masked: np.ndarray  # (B,) int32
    key: np.ndarray  # (B, cache_size, H, D)
    value: np.ndarray  # (B, cache_size, H, D)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    out = x @ weight.T
    if bias is not None:
        out += bias
    return out


def _residual_block(
    x: np.ndarray,
    tensors: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray:
    return (
        _linear(
            swish(_linear(x, tensors[f"{prefix}.hidden_layer.weight"], tensors[f"{prefix}.hidden_layer.bias"]
                          if f"{prefix}.hidden_layer.bias" in tensors else None)),
            tensors[f"{prefix}.output_layer.weight"],
            tensors[f"{prefix}.output_layer.bias"] if f"{prefix}.output_layer.bias" in tensors else None,
        )
        + _linear(
            x,
            tensors[f"{prefix}.residual_layer.weight"],
            tensors[f"{prefix}.residual_layer.bias"] if f"{prefix}.residual_layer.bias" in tensors else None,
        )
    )


def _rope(position: np.ndarray, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Sin/cos tables for non-interleaved-halves RoPE at ``position`` (B, N)."""

    half = head_dim // 2
    fraction = 2.0 * np.arange(half, dtype=np.float32) / head_dim
    timescale = 1.0 * (10_000.0 / 1.0) ** fraction  # min_timescale=1, max_timescale=1e4
    sinusoid_inp = (position.astype(np.float32)[..., None] / timescale).astype(np.float32)
    return np.sin(sinusoid_inp)[:, :, None, :], np.cos(sinusoid_inp)[:, :, None, :]


def _apply_rope(x: np.ndarray, sin: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """x: (B, N, H, D); sin/cos: (B, N, 1, D/2)."""

    first, second = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return np.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)


def _make_attn_mask(
    query_length: int,
    num_all_masked_kv: np.ndarray,
    query_index_offset: np.ndarray | None,
    kv_length: int,
) -> np.ndarray:
    """Boolean mask (B, 1, Q, K) mirroring ``transformer.make_attn_mask``."""

    q_index = np.arange(query_length)[None, None, :, None]
    if query_index_offset is not None:
        q_index = q_index + query_index_offset[:, None, None, None]
    kv_index = np.arange(kv_length)[None, None, None, :]
    return (q_index >= kv_index) & (kv_index >= num_all_masked_kv[:, None, None, None])


def _dot_product_attention(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Unscaled dot-product attention; query: (B, Q, H, D)."""

    scores = np.einsum("bqhd,bkhd->bhqk", query, key, dtype=np.float32)
    scores = np.where(mask, scores, -_MASKED_SCORE)
    weights = _softmax(scores, axis=-1)
    return np.einsum("bhqk,bkhd->bqhd", weights, value, dtype=np.float32)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=axis, keepdims=True)


def _per_dim_scale(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    num_dims = x.shape[-1]
    factor = ((1.442695041 / np.sqrt(num_dims)) * softplus(scale)).astype(np.float32)
    return x * factor


def _attention(
    tensors: dict[str, np.ndarray],
    prefix: str,
    inputs_q: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    epsilon: float,
    decode_cache: TimesFMDecodeCache | None,
    patch_mask: np.ndarray,
) -> tuple[np.ndarray, TimesFMDecodeCache | None]:
    b, n_patches, model_dims = inputs_q.shape
    if patch_mask is None:
        patch_mask = np.zeros((b, n_patches), dtype=bool)

    qkv = _linear(inputs_q, tensors[f"{prefix}.attn.qkv_proj.weight"])
    query, key, value = np.split(qkv, 3, axis=-1)
    query = query.reshape(b, n_patches, num_heads, head_dim)
    key = key.reshape(b, n_patches, num_heads, head_dim)
    value = value.reshape(b, n_patches, num_heads, head_dim)

    if decode_cache is None:
        num_masked = np.sum(patch_mask.astype(np.int32), axis=-1)
        next_index = np.zeros_like(num_masked)
    else:
        num_masked = np.sum(patch_mask.astype(np.int32), axis=-1) + decode_cache.num_masked
        next_index = decode_cache.next_index.copy()

    position = np.arange(n_patches)[None, :] + next_index[:, None] - num_masked[:, None]
    sin, cos = _rope(position, head_dim)
    query = _apply_rope(query, sin, cos)
    key = _apply_rope(key, sin, cos)

    query = rmsnorm(query, tensors[f"{prefix}.attn.query_ln.scale"], epsilon)
    key = rmsnorm(key, tensors[f"{prefix}.attn.key_ln.scale"], epsilon)

    query = _per_dim_scale(query, tensors[f"{prefix}.attn.per_dim_scale.per_dim_scale"])

    if decode_cache is not None:
        cache_size = decode_cache.value.shape[1]
        start = int(decode_cache.next_index[0])
        end = start + n_patches
        decode_cache.key[:, start:end] = key
        decode_cache.value[:, start:end] = value
        key = decode_cache.key
        value = decode_cache.value
        decode_cache.next_index = decode_cache.next_index + n_patches
        decode_cache.num_masked = num_masked
        attn_mask = _make_attn_mask(
            query_length=n_patches,
            num_all_masked_kv=num_masked,
            query_index_offset=next_index,
            kv_length=cache_size,
        )
    else:
        attn_mask = _make_attn_mask(
            query_length=n_patches,
            num_all_masked_kv=num_masked,
            query_index_offset=None,
            kv_length=n_patches,
        )

    x = _dot_product_attention(query, key, value, attn_mask)
    x = x.reshape(b, n_patches, model_dims)
    out = _linear(x, tensors[f"{prefix}.attn.out.weight"])
    return out, decode_cache


def _transformer_layer(
    tensors: dict[str, np.ndarray],
    prefix: str,
    input_embeddings: np.ndarray,
    patch_mask: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    epsilon: float,
    decode_cache: TimesFMDecodeCache | None,
) -> tuple[np.ndarray, TimesFMDecodeCache | None]:
    attn_input = rmsnorm(input_embeddings, tensors[f"{prefix}.pre_attn_ln.scale"], epsilon)
    attn_output, decode_cache = _attention(
        tensors,
        prefix,
        attn_input,
        num_heads=num_heads,
        head_dim=head_dim,
        epsilon=epsilon,
        decode_cache=decode_cache,
        patch_mask=patch_mask,
    )
    attn_output = rmsnorm(attn_output, tensors[f"{prefix}.post_attn_ln.scale"], epsilon) + input_embeddings
    ff_input = rmsnorm(attn_output, tensors[f"{prefix}.pre_ff_ln.scale"], epsilon)
    ff_output = _linear(
        swish(_linear(ff_input, tensors[f"{prefix}.ff0.weight"])),
        tensors[f"{prefix}.ff1.weight"],
    )
    output_embeddings = (
        rmsnorm(ff_output, tensors[f"{prefix}.post_ff_ln.scale"], epsilon) + attn_output
    )
    return output_embeddings.astype(np.float32), decode_cache


def timesfm_forward(
    weights: TimesFMHostWeights,
    inputs: np.ndarray,
    masks: np.ndarray,
    decode_caches: list[TimesFMDecodeCache] | None = None,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    list[TimesFMDecodeCache],
]:
    """Mirror ``TimesFM_2p5_200M_torch_module.forward``.

    ``inputs``: (B, N_patches, patch_len) float32; ``masks``: (B, N_patches, patch_len) bool.
    Returns ``(input_embeddings, output_embeddings, output_ts, output_quantile_spread)``
    plus the updated decode caches.
    """

    spec = weights.spec
    tokenizer_inputs = np.concatenate([inputs, masks.astype(np.float32)], axis=-1)
    input_embeddings = _residual_block(tokenizer_inputs, weights.tensors, "tokenizer")

    if decode_caches is None:
        decode_caches = [None] * spec.num_hidden_layers

    output_embeddings = input_embeddings
    new_decode_caches: list[TimesFMDecodeCache] = []
    for layer in range(spec.num_hidden_layers):
        output_embeddings, cache = _transformer_layer(
            weights.tensors,
            f"stacked_xf.{layer}",
            output_embeddings,
            masks[..., -1],
            num_heads=spec.num_attention_heads,
            head_dim=spec.head_dim,
            epsilon=spec.rms_norm_eps,
            decode_cache=decode_caches[layer],
        )
        new_decode_caches.append(cache)
    output_ts = _residual_block(output_embeddings, weights.tensors, "output_projection_point")
    output_quantile_spread = _residual_block(
        output_embeddings, weights.tensors, "output_projection_quantiles"
    )
    return (
        input_embeddings,
        output_embeddings,
        output_ts,
        output_quantile_spread,
    ), new_decode_caches


def update_running_stats(
    n: np.ndarray, mu: np.ndarray, sigma: np.ndarray, x: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror ``util.update_running_stats`` (patch-level scalar stats, shape (B,))."""

    is_legit = ~mask
    inc_n = np.sum(is_legit.astype(x.dtype), axis=-1)
    inc_mu_numerator = np.sum(x * is_legit, axis=-1)
    inc_n_safe = np.where(inc_n == 0, 1.0, inc_n)
    inc_mu = np.where(inc_n == 0, 0.0, inc_mu_numerator / inc_n_safe)
    inc_var = np.sum(((x - inc_mu[..., None]) ** 2) * is_legit, axis=-1) / inc_n_safe
    inc_var = np.where(inc_n == 0, 0.0, inc_var)
    inc_sigma = np.sqrt(inc_var)

    new_n = n + inc_n
    new_n_safe = np.where(new_n == 0, 1.0, new_n)
    new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
    new_mu = np.where(new_n == 0, 0.0, new_mu)
    new_var = (
        n * sigma**2
        + inc_n * inc_sigma**2
        + n * (mu - new_mu) ** 2
        + inc_n * (inc_mu - new_mu) ** 2
    ) / new_n_safe
    new_var = np.where(new_n == 0, 0.0, np.clip(new_var, 0.0, None))
    return new_n, new_mu, np.sqrt(new_var)


def revin(
    x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, *, reverse: bool = False
) -> np.ndarray:
    """Mirror ``util.revin``."""

    tolerance = 1e-6
    if len(mu.shape) == len(x.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(x.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]
    if reverse:
        return x * sigma + mu
    return (x - mu) / np.where(sigma < tolerance, 1.0, sigma)


def _empty_decode_cache(
    spec: TimesFMModelSpec, batch_size: int, decode_cache_size: int
) -> list[TimesFMDecodeCache]:
    return [
        TimesFMDecodeCache(
            next_index=np.zeros(batch_size, dtype=np.int32),
            num_masked=np.zeros(batch_size, dtype=np.int32),
            key=np.zeros((batch_size, decode_cache_size, spec.num_attention_heads, spec.head_dim), dtype=np.float32),
            value=np.zeros((batch_size, decode_cache_size, spec.num_attention_heads, spec.head_dim), dtype=np.float32),
        )
        for _ in range(spec.num_hidden_layers)
    ]


def timesfm_decode(
    weights: TimesFMHostWeights,
    horizon: int,
    inputs: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Mirror ``TimesFM_2p5_200M_torch_module.decode``.

    ``inputs``: (B, context) float32 with context a multiple of patch_len;
    ``masks``: (B, context) bool (True = padded front region).
    Returns ``(renormed_outputs, renormed_quantile_spread, ar_renormed_outputs)``.
    """

    spec = weights.spec
    p, o = spec.patch_length, spec.horizon_length
    m = o // p
    batch_size, context = inputs.shape
    num_decode_steps = (horizon - 1) // o
    num_input_patches = context // p
    decode_cache_size = num_input_patches + num_decode_steps * m

    patched_inputs = inputs.reshape(batch_size, -1, p)
    patched_masks = masks.reshape(batch_size, -1, p)

    n = np.zeros(batch_size, dtype=np.float32)
    mu = np.zeros(batch_size, dtype=np.float32)
    sigma = np.zeros(batch_size, dtype=np.float32)
    patch_mu: list[np.ndarray] = []
    patch_sigma: list[np.ndarray] = []
    for i in range(num_input_patches):
        n, mu, sigma = update_running_stats(n, mu, sigma, patched_inputs[:, i], patched_masks[:, i])
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    last_n, last_mu, last_sigma = n, mu, sigma
    context_mu = np.stack(patch_mu, axis=1)
    context_sigma = np.stack(patch_sigma, axis=1)

    decode_caches = _empty_decode_cache(spec, batch_size, decode_cache_size)

    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = np.where(patched_masks, 0.0, normed_inputs)
    (_, _, normed_outputs, normed_quantile_spread), decode_caches = timesfm_forward(
        weights, normed_inputs, patched_masks, decode_caches
    )
    renormed_outputs = revin(
        normed_outputs, context_mu, context_sigma, reverse=True
    ).reshape(batch_size, -1, o, spec.quantile_heads)
    renormed_quantile_spread = revin(
        normed_quantile_spread, context_mu, context_sigma, reverse=True
    ).reshape(batch_size, -1, spec.quantile_horizon_length, spec.quantile_heads)[:, -1, ...]

    ar_outputs: list[np.ndarray] = []
    last_renormed_output = renormed_outputs[:, -1, :, spec.decode_index]

    for _ in range(num_decode_steps):
        new_patched_input = last_renormed_output.reshape(batch_size, m, p)
        new_mask = np.zeros_like(new_patched_input, dtype=bool)

        n, mu, sigma = last_n, last_mu, last_sigma
        new_mus: list[np.ndarray] = []
        new_sigmas: list[np.ndarray] = []
        for i in range(m):
            n, mu, sigma = update_running_stats(n, mu, sigma, new_patched_input[:, i], new_mask[:, i])
            new_mus.append(mu)
            new_sigmas.append(sigma)
        last_n, last_mu, last_sigma = n, mu, sigma
        new_mu = np.stack(new_mus, axis=1)
        new_sigma = np.stack(new_sigmas, axis=1)

        new_normed_input = revin(new_patched_input, new_mu, new_sigma, reverse=False)
        (_, _, new_normed_output, _), decode_caches = timesfm_forward(
            weights, new_normed_input, new_mask, decode_caches
        )

        new_renormed_output = revin(
            new_normed_output, new_mu, new_sigma, reverse=True
        ).reshape(batch_size, m, o, spec.quantile_heads)
        ar_outputs.append(new_renormed_output[:, -1, ...])
        last_renormed_output = new_renormed_output[:, -1, :, spec.decode_index]

    ar_renormed_outputs = np.stack(ar_outputs, axis=1) if num_decode_steps > 0 else None
    return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs


def timesfm_forecast_naive(
    weights: TimesFMHostWeights,
    horizon: int,
    inputs: list[np.ndarray],
) -> list[np.ndarray]:
    """Mirror ``forecast_naive``: point+quantile forecast, no forecasting flags.

    Each input is padded at the front to a multiple of the patch length.
    Returns per-series (horizon, quantile_heads) float32 arrays.
    """

    p = weights.spec.patch_length
    outputs: list[np.ndarray] = []
    for each_input in inputs:
        input_t = np.asarray(each_input, dtype=np.float32)
        mask = np.zeros_like(input_t, dtype=bool)
        len_front_mask = p - (len(input_t) % p)
        if len_front_mask < p:
            input_t = np.concatenate(
                [np.zeros(len_front_mask, dtype=np.float32), input_t], axis=0
            )
            mask = np.concatenate(
                [np.ones(len_front_mask, dtype=bool), mask], axis=0
            )
        input_t = input_t[None, ...]
        mask = mask[None, ...]
        t_pf, _, t_ar = timesfm_decode(weights, horizon, input_t, mask)
        to_concat = [t_pf[:, -1, ...]]
        if t_ar is not None:
            to_concat.append(t_ar.reshape(1, -1, weights.spec.quantile_heads))
        forecast = np.concatenate(to_concat, axis=1)[:, :horizon, :]
        outputs.append(forecast.squeeze(0))
    return outputs


__all__ = [
    "TimesFMDecodeCache",
    "TimesFMHostWeights",
    "revin",
    "timesfm_decode",
    "timesfm_forecast_naive",
    "timesfm_forward",
    "update_running_stats",
]
