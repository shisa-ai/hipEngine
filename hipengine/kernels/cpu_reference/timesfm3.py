"""Torch-free NumPy CPU reference for TimesFM 3.0 500M.

Mirrors ``google-research/timesfm`` ``src/timesfm3/torch/model.py``
(``TimesFM3Torch.forward`` / ``.decode``) plus its ``transformer.py``,
``dense.py``, ``normalization.py``, ``util.py`` and ``cpm_revin_refine.py``
modules (ResidualBlock, MixingTransformer, MultiHeadAttention,
RotaryPositionalEmbedding, PerDimScale, revin, update_running_stats,
get_output_patch_via_roll, stitch_patches, cpm_iterative_revin_refine).

Differences from torch beyond the framework swap: none intentional.
Everything computes in float32.  Attention is the reference's SDPA
configuration: scores are ``q·k`` multiplied by ``sqrt(head_dim)``, bool
masks exclude keys, and a query row whose keys are all masked produces
zeros (matching the CPU SDPA backend the oracle runs on; such rows only
exist at leading-pad positions that decode() slices away).  RMSNorm uses
torch ``nn.RMSNorm`` semantics — ``x * rsqrt(mean(x^2) + eps) * weight``
with ``eps = finfo(float32).eps`` (NOT 2.5's custom 1e-6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from safetensors import safe_open

from hipengine.loading.safetensors import load_weight_index
from hipengine.models.timesfm3 import (
    TimesFM3ModelSpec,
    expected_timesfm3_weight_shapes,
    parse_timesfm3_model_spec,
    validate_timesfm3_weight_index,
)

_TORCH_RMS_EPS = float(np.finfo(np.float32).eps)
_MASKED_SCORE = 1.0e9  # additive mask magnitude; -1e9 matches the reference


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, np.zeros_like(x))


def rmsnorm(x: np.ndarray, weight: np.ndarray, epsilon: float = _TORCH_RMS_EPS) -> np.ndarray:
    """Torch ``nn.RMSNorm``: multiplicative weight, eps inside rsqrt."""

    var = np.mean(np.square(x), axis=-1, keepdims=True, dtype=np.float32)
    return (x * np.float32(1.0) / np.sqrt(var + np.float32(epsilon))) * weight


@dataclass(frozen=True)
class TimesFM3HostWeights:
    """Validated host-side FP32 weights keyed by checkpoint name."""

    spec: TimesFM3ModelSpec
    tensors: dict[str, np.ndarray]

    @classmethod
    def load(cls, model_path: str) -> "TimesFM3HostWeights":
        index = load_weight_index(model_path)
        spec = parse_timesfm3_model_spec(index.config)
        validate_timesfm3_weight_index(spec, index)
        expected = expected_timesfm3_weight_shapes(spec)
        names_by_shard: dict[Any, list[str]] = {}
        for name in expected:
            names_by_shard.setdefault(index.tensors[name].shard_path, []).append(name)
        tensors: dict[str, np.ndarray] = {}
        for shard in sorted(names_by_shard):
            with safe_open(str(shard), framework="numpy") as handle:
                for name in sorted(names_by_shard[shard]):
                    value = np.asarray(handle.get_tensor(name), dtype=np.float32)
                    if not bool(np.isfinite(value).all()):
                        raise ValueError(f"TimesFM 3.0 weight {name} must be finite")
                    tensors[name] = np.ascontiguousarray(value)
        return cls(spec=spec, tensors=tensors)


# ---------------------------------------------------------------------------
# Shared numerical helpers (util.py).
# ---------------------------------------------------------------------------


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    out = x @ weight.T
    if bias is not None:
        out += bias
    return out


def _make_safe_for_division(values: np.ndarray) -> np.ndarray:
    return np.where(values < 1e-6, np.float32(1.0), values)


def _revin(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """Reversible per-instance normalization (``util.revin``)."""

    if mu.ndim == x.ndim - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif mu.ndim == x.ndim - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]
    else:
        raise ValueError(f"Unsupported shapes for x and mu: {x.shape}, {mu.shape}.")
    if reverse:
        return x * sigma + mu
    return (x - mu) / _make_safe_for_division(sigma)


def _update_running_stats(
    n: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    x: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``util.update_running_stats``; x: (b, v, p), mask: (b, v, p) bool."""

    is_legit = ~mask
    is_legit_f = is_legit.astype(np.float32)
    inc_n = is_legit_f.sum(axis=-1)

    x_masked = np.where(is_legit, x, np.float32(0.0))
    inc_sum = x_masked.sum(axis=-1)
    inc_mu = np.where(inc_n == 0, np.float32(0.0), inc_sum / np.where(inc_n == 0, 1.0, inc_n))

    x_diff_sq = np.where(is_legit, (x - inc_mu[..., None]) ** 2, np.float32(0.0))
    inc_var = np.where(inc_n == 0, np.float32(0.0), x_diff_sq.sum(axis=-1) / np.where(inc_n == 0, 1.0, inc_n))
    inc_sigma = np.sqrt(inc_var)

    new_n = n + inc_n
    safe_new_n = np.where(new_n == 0, np.float32(1.0), new_n)
    new_mu = np.where(new_n == 0, np.float32(0.0), (n * mu + inc_mu * inc_n) / safe_new_n)
    new_var = (
        n * sigma * sigma
        + inc_n * inc_sigma * inc_sigma
        + n * (mu - new_mu) * (mu - new_mu)
        + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)
    ) / safe_new_n
    new_sigma = np.sqrt(np.where(new_n == 0, np.float32(0.0), new_var))
    return new_n.astype(np.float32), new_mu.astype(np.float32), new_sigma.astype(np.float32)


def _get_running_stats(
    values: np.ndarray,
    masks: np.ndarray,
    *,
    freeze_after: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative running statistics per patch (``util.get_running_stats``).

    values/masks: (b, v, n, p). Returns (n, mu, sigma) each (b, v, n).
    """

    b, v, n, _ = values.shape
    cur_n = np.zeros((b, v), dtype=np.float32)
    cur_mu = np.zeros((b, v), dtype=np.float32)
    cur_sigma = np.zeros((b, v), dtype=np.float32)

    all_n: list[np.ndarray] = []
    all_mu: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []
    for i in range(n):
        cur_n, cur_mu, cur_sigma = _update_running_stats(
            cur_n, cur_mu, cur_sigma, values[:, :, i, :], masks[:, :, i, :]
        )
        all_n.append(cur_n)
        all_mu.append(cur_mu)
        all_sigma.append(cur_sigma)

    running_n = np.stack(all_n, axis=2)
    running_mu = np.stack(all_mu, axis=2)
    running_sigma = np.stack(all_sigma, axis=2)
    # The oracle freezes mean/std (post-hoc, on the stacked outputs) at the
    # freeze_after patch; counts keep accumulating normally.
    if freeze_after is not None and 0 <= freeze_after < n - 1:
        running_mu[:, :, freeze_after + 1 :] = running_mu[:, :, freeze_after : freeze_after + 1]
        running_sigma[:, :, freeze_after + 1 :] = running_sigma[
            :, :, freeze_after : freeze_after + 1
        ]
    return (
        running_n,
        running_mu,
        running_sigma,
    )


def _get_output_patch_via_roll(
    x: np.ndarray, rolls: int
) -> tuple[np.ndarray, np.ndarray]:
    """``util.get_output_patch_via_roll``; x: (b, v, n, p)."""

    b, v, n, p = x.shape
    rolling = np.empty((b, v, n, rolls + 1, p), dtype=x.dtype)
    rolling[:, :, :, 0, :] = x
    for i in range(rolls):
        rolling[:, :, :, i + 1, :] = np.roll(rolling[:, :, :, i, :], -1, axis=2)
    result = rolling[:, :, :, 1:, :].reshape(b, v, n, rolls * p)

    patch_idx = np.arange(n)
    point_idx = np.arange(rolls * p)
    source_patch = patch_idx[:, None] + 1 + point_idx[None, :] // p
    wrap_mask = source_patch >= n  # (n, rolls*p) -> broadcast (1, 1, n, rolls*p)
    return result, wrap_mask[None, None]


def _stitch_patches(patch_preds: np.ndarray, patch_len: int) -> np.ndarray:
    """``util.stitch_patches``; patch_preds: (b, v, n, patch_len+overlap, q)."""

    b, v, num_patches, total_len, q = patch_preds.shape
    overlap = total_len - patch_len

    if num_patches == 1:
        return patch_preds[:, :, 0, :, :]

    stitch_weights = np.linspace(1.0, 0.0, overlap, dtype=patch_preds.dtype)
    stitch_weights = stitch_weights[None, None, None, :, None]

    first_chunk = patch_preds[:, :, 0, :patch_len, :]

    prev_patches = patch_preds[:, :, :-1, :, :]
    next_patches = patch_preds[:, :, 1:, :, :]

    prev_overlaps = prev_patches[:, :, :, patch_len:, :]
    next_overlaps = next_patches[:, :, :, :overlap, :]

    stitched_overlaps = (
        stitch_weights * prev_overlaps + (1.0 - stitch_weights) * next_overlaps
    )
    middles = next_patches[:, :, :, overlap:patch_len, :]

    output_chunks = np.concatenate([stitched_overlaps, middles], axis=3)
    mid = output_chunks.reshape(b, v, (num_patches - 1) * patch_len, q)
    tail = patch_preds[:, :, -1, patch_len:, :]
    return np.concatenate([first_chunk, mid, tail], axis=2)


# ---------------------------------------------------------------------------
# Attention (transformer.py, SDPA configuration of the reference).
# ---------------------------------------------------------------------------


def _rope_tables(position: np.ndarray, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Sin/cos tables at ``position`` (B, N) -> (B, N, 1, D/2)."""

    half = head_dim // 2
    fraction = 2.0 * np.arange(half, dtype=np.float32) / head_dim
    timescale = 1.0 * (10_000.0 / 1.0) ** fraction
    sinusoid_inp = (position.astype(np.float32)[..., None] / timescale).astype(np.float32)
    return np.sin(sinusoid_inp)[:, :, None, :], np.cos(sinusoid_inp)[:, :, None, :]


def _apply_rope(x: np.ndarray, sin: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """x: (B, N, H, D); sin/cos: (B, N, 1, D/2); non-interleaved halves."""

    first, second = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return np.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    )


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=axis, keepdims=True)


def _per_dim_scale(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    num_dims = x.shape[-1]
    factor = ((1.442695041 / np.sqrt(num_dims)) * softplus(scale)).astype(np.float32)
    return x * factor


def _dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
    *,
    head_dim: int,
) -> np.ndarray:
    """SDPA with ``scale=sqrt(head_dim)``; query/key/value: (B, N, H, D).

    ``mask``: (B, 1, Q, K) bool, True = attend.  Rows whose keys are all
    masked produce zeros (CPU SDPA backend semantics).
    """

    scores = np.einsum("bqhd,bkhd->bhqk", query, key, dtype=np.float32)
    scores = scores * np.float32(math.sqrt(head_dim))
    scores = np.where(mask, scores, np.float32(-_MASKED_SCORE))
    weights = _softmax(scores, axis=-1)
    fully_masked = ~mask.any(axis=-1)  # (B, 1, Q)
    weights = np.where(fully_masked[:, :, :, None], np.float32(0.0), weights)
    return np.einsum("bhqk,bkhd->bqhd", weights, value, dtype=np.float32)


def _multi_head_attention(
    tensors: dict[str, np.ndarray],
    prefix: str,
    inputs_q: np.ndarray,
    patch_mask: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    causal: bool,
    use_rope: bool,
) -> np.ndarray:
    """Full-sequence MultiHeadAttention (``transformer.MultiHeadAttention``).

    ``inputs_q``: (B, N, D); ``patch_mask``: (B, N) bool True = masked.
    """

    b, n_patches, model_dims = inputs_q.shape
    if patch_mask is None:
        patch_mask = np.zeros((b, n_patches), dtype=bool)

    query = _linear(inputs_q, tensors[f"{prefix}.query_proj.weight"]).reshape(
        b, n_patches, num_heads, head_dim
    )
    key = _linear(inputs_q, tensors[f"{prefix}.key_proj.weight"]).reshape(
        b, n_patches, num_heads, head_dim
    )
    value = _linear(inputs_q, tensors[f"{prefix}.value_proj.weight"]).reshape(
        b, n_patches, num_heads, head_dim
    )

    # num_front_masked: leading fully-masked patches (patch_mask is already
    # the cumprod "effective" mask at inference, so cumprod is idempotent).
    num_front_masked = np.sum(
        np.cumprod(patch_mask.astype(np.int32), axis=-1), axis=-1
    )

    if use_rope:
        position = np.arange(n_patches, dtype=np.float32)[None, :]
        sin, cos = _rope_tables(position, head_dim)
        query = _apply_rope(query, sin, cos)
        key = _apply_rope(key, sin, cos)

    query = rmsnorm(query, tensors[f"{prefix}.query_ln.weight"])
    key = rmsnorm(key, tensors[f"{prefix}.key_ln.weight"])
    query = _per_dim_scale(query, tensors[f"{prefix}.per_dim_scale.per_dim_scale"])

    q_index = np.arange(n_patches)[None, None, :, None]
    kv_index = np.arange(n_patches)[None, None, None, :]
    attn_mask = kv_index >= num_front_masked[:, None, None, None]
    if causal:
        attn_mask = (q_index >= kv_index) & attn_mask
    attn_mask = attn_mask & ~patch_mask[:, None, None, :]

    x = _dot_product_attention(query, key, value, attn_mask, head_dim=head_dim)
    x = x.reshape(b, n_patches, model_dims)
    return _linear(x, tensors[f"{prefix}.out_proj.weight"])


def _mixing_transformer_layer(
    tensors: dict[str, np.ndarray],
    prefix: str,
    input_embeddings: np.ndarray,
    patch_mask: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    use_variate_attention: bool,
) -> np.ndarray:
    """One ``MixingTransformer`` layer; input/patch_mask: (b, v, n, d)/(b, v, n)."""

    b, v, n, d = input_embeddings.shape

    # --- Sequence attention ---
    seq_in = rmsnorm(
        input_embeddings, tensors[f"{prefix}.pre_seq_attn_ln.weight"]
    ).reshape(b * v, n, d)
    seq_mask = patch_mask.reshape(b * v, n)
    seq_out = _multi_head_attention(
        tensors,
        f"{prefix}.seq_attn",
        seq_in,
        seq_mask,
        num_heads=num_heads,
        head_dim=head_dim,
        causal=True,
        use_rope=True,
    ).reshape(b, v, n, d)
    h1 = rmsnorm(seq_out, tensors[f"{prefix}.post_seq_attn_ln.weight"]) + input_embeddings

    # --- Variate attention ---
    if use_variate_attention:
        var_in = rmsnorm(h1, tensors[f"{prefix}.pre_var_attn_ln.weight"])
        var_in = var_in.transpose(0, 2, 1, 3).reshape(b * n, v, d)
        var_mask = patch_mask.transpose(0, 2, 1).reshape(b * n, v)
        var_out = _multi_head_attention(
            tensors,
            f"{prefix}.var_attn",
            var_in,
            var_mask,
            num_heads=num_heads,
            head_dim=head_dim,
            causal=False,
            use_rope=False,
        ).reshape(b, n, v, d).transpose(0, 2, 1, 3)
        h2 = rmsnorm(var_out, tensors[f"{prefix}.post_var_attn_ln.weight"]) + h1
    else:
        h2 = h1

    # --- FeedForward (ReLU) ---
    ff_out = _linear(
        np.maximum(
            _linear(rmsnorm(h2, tensors[f"{prefix}.pre_ff_ln.weight"]),
                    tensors[f"{prefix}.ff0.weight"]),
            np.float32(0.0),
        ),
        tensors[f"{prefix}.ff1.weight"],
    )
    return (
        rmsnorm(ff_out, tensors[f"{prefix}.post_ff_ln.weight"]) + h2
    ).astype(np.float32)


def _residual_block_relu(
    x: np.ndarray,
    tensors: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray:
    """ReLU ResidualBlock with no biases and a linear residual connection."""

    hidden = np.maximum(_linear(x, tensors[f"{prefix}.hidden_layer.weight"]), np.float32(0.0))
    return _linear(
        hidden, tensors[f"{prefix}.output_layer.weight"]
    ) + _linear(x, tensors[f"{prefix}.residual_layer.weight"])


# ---------------------------------------------------------------------------
# CPM iterative RevIN refinement (cpm_revin_refine.py).
# ---------------------------------------------------------------------------


def _cpm_iterative_revin_refine(
    raw_logits: np.ndarray,
    revin_n: np.ndarray,
    revin_mu: np.ndarray,
    revin_sigma: np.ndarray,
    patch_cpm_mask: np.ndarray,
    *,
    median_q_idx: int,
    rolls: int,
    patch_len: int,
    num_quantiles: int,
    value_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine RevIN stats at CPM-masked patches (see the torch docstring)."""

    b, v, n_patches, _ = raw_logits.shape
    median_logits = raw_logits.reshape(
        b, v, n_patches, rolls, patch_len, num_quantiles
    )[..., median_q_idx]

    carry_n = np.zeros((b, v), dtype=np.float32)
    carry_mu = np.zeros((b, v), dtype=np.float32)
    carry_sigma = np.zeros((b, v), dtype=np.float32)
    anchor = np.zeros((b, v, rolls, patch_len), dtype=np.float32)
    block_offset = np.zeros((b,), dtype=np.int64)

    refined_mu_list: list[np.ndarray] = []
    refined_sigma_list: list[np.ndarray] = []

    step_masks = np.zeros((b, v, patch_len), dtype=bool)

    for i in range(n_patches):
        actual_n = revin_n[:, :, i]
        actual_mu = revin_mu[:, :, i]
        actual_sigma = revin_sigma[:, :, i]
        current_step_logits = median_logits[:, :, i]
        is_cpm = patch_cpm_mask[:, i : i + 1]  # (b, 1)

        offset_onehot = np.equal(
            np.arange(rolls)[None, :], block_offset[:, None]
        ).astype(np.float32)  # (b, rolls)
        predicted_values_step = np.einsum("br,bvrp->bvp", offset_onehot, anchor)

        new_n, new_mu, new_sigma = _update_running_stats(
            carry_n, carry_mu, carry_sigma, predicted_values_step, step_masks
        )

        out_n = np.where(is_cpm, new_n, actual_n)
        out_mu = np.where(is_cpm, new_mu, actual_mu)
        out_sigma = np.where(is_cpm, new_sigma, actual_sigma)

        new_block_offset = np.where(
            is_cpm.squeeze(-1), (block_offset + 1) % rolls, np.zeros_like(block_offset)
        )
        should_update_anchor = new_block_offset == 0

        step_predicted_values = _revin(current_step_logits, out_mu, out_sigma, reverse=True)
        step_predicted_values = np.clip(
            step_predicted_values, -value_clip, value_clip
        )

        anchor = np.where(
            should_update_anchor.reshape(b, 1, 1, 1), step_predicted_values, anchor
        )
        block_offset = new_block_offset
        carry_n = out_n
        carry_mu = out_mu
        carry_sigma = out_sigma

        refined_mu_list.append(out_mu)
        refined_sigma_list.append(out_sigma)

    return np.stack(refined_mu_list, axis=2), np.stack(refined_sigma_list, axis=2)


# ---------------------------------------------------------------------------
# Model forward (model.py).
# ---------------------------------------------------------------------------


def timesfm3_forward(
    weights: TimesFM3HostWeights,
    values: np.ndarray,
    masks: np.ndarray,
    patch_is_target: np.ndarray,
    patch_cpm_mask: np.ndarray | None,
    freeze_after: int | None = None,
) -> np.ndarray:
    """Mirror ``TimesFM3Torch.forward`` (single-segment, no aux outputs).

    values/masks: (b, v, n, p); patch_is_target: (b, v, n) bool;
    patch_cpm_mask: (b, n) bool.  Returns logits (b, v, n, o, q).
    """

    spec = weights.spec
    tensors = weights.tensors
    b, v, n, p = values.shape
    if p != spec.input_patch_len:
        raise ValueError(
            f"Input patch_len {p} != model input_patch_len {spec.input_patch_len}"
        )
    if v > spec.max_variates:
        raise ValueError(f"variates {v} exceed max_variates {spec.max_variates}")

    values = np.nan_to_num(values, nan=0.0)
    values = np.clip(values, -spec.value_clip, spec.value_clip)
    masks = masks.astype(bool)

    running_n, running_mean, running_std = _get_running_stats(
        values, masks, freeze_after=freeze_after
    )

    # CPM mask: mask target variates at CPM positions.
    if patch_cpm_mask is not None:
        cpm_bvnp = patch_cpm_mask[:, None, :, None]
        cpm_target_only = cpm_bvnp & patch_is_target[..., None]
        masks = masks | cpm_target_only

    values_bvnp = _revin(values, running_mean, running_std, reverse=False)
    values_bvnp = np.where(masks, np.float32(0.0), values_bvnp)

    values_fcov, wrap_mask = _get_output_patch_via_roll(values, spec.rolls)
    values_fcov = _revin(values_fcov, running_mean, running_std, reverse=False)

    masks_fcov_raw, _ = _get_output_patch_via_roll(masks, spec.rolls)
    masks_fcov = masks_fcov_raw | patch_is_target[..., None] | wrap_mask
    values_fcov = np.where(masks_fcov, np.float32(0.0), values_fcov)

    values_cat = np.concatenate([values_bvnp, values_fcov], axis=-1)
    masks_cat = np.concatenate([masks, masks_fcov], axis=-1)

    resblock_input = np.concatenate(
        [values_cat, masks_cat.astype(np.float32)], axis=-1
    )
    transformer_input = _residual_block_relu(
        resblock_input, tensors, "pre_transformer_resblock"
    )

    patch_mask_bvn = masks_cat.all(axis=3)
    effective_patch_mask = np.cumprod(patch_mask_bvn.astype(np.int32), axis=2).astype(bool)

    output = transformer_input
    for layer in range(spec.num_layers):
        output = _mixing_transformer_layer(
            tensors,
            f"transformer_stack.layers.{layer}",
            output,
            effective_patch_mask,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            use_variate_attention=spec.use_variate_attention,
        )

    raw_logits = _linear(
        output, tensors["output_head.weight"], tensors["output_head.bias"]
    )

    if spec.use_iterative_cpm_revin and patch_cpm_mask is not None:
        refined_mu, refined_sigma = _cpm_iterative_revin_refine(
            raw_logits,
            revin_n=running_n,
            revin_mu=running_mean,
            revin_sigma=running_std,
            patch_cpm_mask=patch_cpm_mask,
            median_q_idx=spec.median_index,
            rolls=spec.rolls,
            patch_len=spec.input_patch_len,
            num_quantiles=len(spec.quantiles),
            value_clip=spec.value_clip,
        )
        cpm_bvn = patch_cpm_mask[:, None, :]
        revin_mean = np.where(cpm_bvn, refined_mu, running_mean)
        revin_std = np.where(cpm_bvn, refined_sigma, running_std)
    else:
        revin_mean = running_mean
        revin_std = running_std

    revin_logits = _revin(raw_logits, revin_mean, revin_std, reverse=True)
    clipped_logits = np.clip(revin_logits, -spec.value_clip, spec.value_clip)

    o, q = spec.output_patch_len, len(spec.quantiles)
    return clipped_logits.reshape(b, v, n, o, q)


# ---------------------------------------------------------------------------
# Decode (model.py decode(); non-autoregressive single pass).
# ---------------------------------------------------------------------------


def _linear_detrend_context(
    ctx_vals: np.ndarray, ctx_masks: np.ndarray, context: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the ``use_linear_detrending`` block of ``decode``.

    ctx_vals/ctx_masks: (b, v, context).  Returns
    (detrended_vals, m_trend, c_trend, apply_detrend).
    """

    t_ctx = np.arange(-(context - 1), 1, dtype=np.float32) / np.float32(context)
    t_ctx = t_ctx[None, None, :]

    valid = ~ctx_masks
    n_v = valid.astype(np.float32).sum(axis=-1, keepdims=True)
    sum_t = np.where(valid, t_ctx, np.float32(0.0)).sum(axis=-1, keepdims=True)
    sum_t2 = np.where(valid, t_ctx**2, np.float32(0.0)).sum(axis=-1, keepdims=True)
    sum_y = np.where(valid, ctx_vals, np.float32(0.0)).sum(axis=-1, keepdims=True)
    sum_ty = np.where(valid, t_ctx * ctx_vals, np.float32(0.0)).sum(
        axis=-1, keepdims=True
    )

    det = n_v * sum_t2 - sum_t**2
    safe_det = np.where(det == 0.0, np.float32(1.0), det)
    m_trend = np.where(det == 0.0, np.float32(0.0), (n_v * sum_ty - sum_t * sum_y) / safe_det)
    c_trend = np.where(
        det == 0.0,
        np.where(n_v > 0, sum_y / np.maximum(n_v, 1.0), np.float32(0.0)),
        (sum_y - m_trend * sum_t) / np.maximum(n_v, 1.0),
    )

    ctx_vals_detrended = ctx_vals - (m_trend * t_ctx + c_trend)

    mean_y = sum_y / np.maximum(n_v, 1.0)
    sum_y2 = np.where(valid, ctx_vals**2, np.float32(0.0)).sum(axis=-1, keepdims=True)
    var_orig = np.maximum(sum_y2 / np.maximum(n_v, 1.0) - mean_y**2, np.float32(0.0))
    std_orig = np.sqrt(var_orig)

    sum_yd = np.where(valid, ctx_vals_detrended, np.float32(0.0)).sum(
        axis=-1, keepdims=True
    )
    mean_yd = sum_yd / np.maximum(n_v, 1.0)
    sum_yd2 = np.where(valid, ctx_vals_detrended**2, np.float32(0.0)).sum(
        axis=-1, keepdims=True
    )
    var_det = np.maximum(sum_yd2 / np.maximum(n_v, 1.0) - mean_yd**2, np.float32(0.0))
    std_det = np.sqrt(var_det)

    apply_detrend = std_det < threshold * std_orig
    return (
        np.where(apply_detrend, ctx_vals_detrended, ctx_vals),
        m_trend,
        c_trend,
        apply_detrend,
    )


def timesfm3_decode(
    weights: TimesFM3HostWeights,
    target: np.ndarray,
    horizon: int,
    past_only_covariates: np.ndarray | None = None,
    past_future_covariates: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    past_only_mask: np.ndarray | None = None,
    past_future_mask: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mirror ``TimesFM3Torch.decode`` (non-autoregressive single pass).

    target: (b, u, context_len); past_only_covariates: (b, v_po, context_len);
    past_future_covariates: (b, w, context_len + horizon).  Masks are boolean
    (True = masked).  Returns (b, num_variates, horizon, num_quantiles).
    """

    spec = weights.spec
    p = spec.input_patch_len

    target = np.asarray(target, dtype=np.float32)
    batch_size, num_target, context = target.shape

    if past_future_covariates is not None:
        past_future_covariates = np.asarray(past_future_covariates, dtype=np.float32)
        horizon = past_future_covariates.shape[-1] - context
    if horizon <= 0:
        raise ValueError("Decode requires horizon > 0.")

    # 1. Pad context to a multiple of input_patch_len.  Rank-agnostic: the
    # global ``mask`` is (batch, context) while everything else is
    # (batch, variates, context).
    ctx_padding = (-context) % p
    if ctx_padding > 0:
        def _pad_left(x: np.ndarray, fill) -> np.ndarray:
            pad_width = [(0, 0)] * (x.ndim - 1) + [(ctx_padding, 0)]
            return np.pad(x, pad_width, mode="constant", constant_values=fill)

        target = _pad_left(target, 0.0)
        if mask is not None:
            mask = _pad_left(mask, True)
        if past_only_covariates is not None:
            past_only_covariates = _pad_left(past_only_covariates, 0.0)
        if past_future_covariates is not None:
            past_future_covariates = _pad_left(past_future_covariates, 0.0)
        if target_mask is not None:
            target_mask = _pad_left(target_mask, True)
        if past_only_mask is not None:
            past_only_mask = _pad_left(past_only_mask, True)
        if past_future_mask is not None:
            past_future_mask = _pad_left(past_future_mask, True)
        context = context + ctx_padding

    if mask is None:
        mask = np.zeros((batch_size, context), dtype=bool)
        if ctx_padding > 0:
            mask[:, :ctx_padding] = True

    # 2. Horizon padding for stitching.
    if spec.use_stitching:
        extract_len = min(2 * p, spec.output_patch_len)
        overlap = extract_len - p
        num_forecast_patches = max(
            math.ceil((horizon - overlap) / p), 1
        )
        num_horizon_patches = num_forecast_patches + spec.rolls - 1
        padded_horizon = num_horizon_patches * p
        hor_padding = padded_horizon - horizon
    else:
        hor_padding = (-horizon) % spec.output_patch_len
        padded_horizon = horizon + hor_padding
        num_horizon_patches = padded_horizon // p
    num_context_patches = context // p

    # 3. Context values and masks.
    if target_mask is None:
        target_mask = np.zeros_like(target, dtype=bool)
    target_mask = target_mask | mask[:, None, :]

    all_ctx_vals = [target]
    all_ctx_masks = [target_mask]
    num_past_only = 0
    if past_only_covariates is not None:
        num_past_only = past_only_covariates.shape[1]
        if past_only_mask is None:
            past_only_mask = np.zeros_like(past_only_covariates, dtype=bool)
        all_ctx_vals.append(past_only_covariates)
        all_ctx_masks.append(past_only_mask | mask[:, None, :])
    if past_future_covariates is not None:
        if past_future_mask is None:
            past_future_mask = np.zeros_like(past_future_covariates, dtype=bool)
        all_ctx_vals.append(past_future_covariates[..., :context])
        all_ctx_masks.append(past_future_mask[..., :context] | mask[:, None, :])

    ctx_vals = np.concatenate(all_ctx_vals, axis=1)
    ctx_masks = np.concatenate(all_ctx_masks, axis=1)

    # Linear detrending of the context.
    if spec.use_linear_detrending:
        ctx_vals, m_trend, c_trend, apply_detrend = _linear_detrend_context(
            ctx_vals, ctx_masks, context, spec.linear_detrending_threshold
        )
    else:
        num_variates = ctx_vals.shape[1]
        m_trend = np.zeros((batch_size, num_variates, 1), dtype=np.float32)
        c_trend = np.zeros((batch_size, num_variates, 1), dtype=np.float32)
        apply_detrend = np.zeros((batch_size, num_variates, 1), dtype=bool)

    ctx_vals = np.where(ctx_masks, np.float32(0.0), ctx_vals)

    # Horizon inputs: targets and past-only variates fully masked; future
    # part of past-future covariates visible.
    all_hor_vals: list[np.ndarray] = [
        np.zeros((batch_size, num_target, padded_horizon), dtype=np.float32),
        np.zeros((batch_size, num_past_only, padded_horizon), dtype=np.float32),
    ]
    all_hor_masks: list[np.ndarray] = [
        np.ones((batch_size, num_target, padded_horizon), dtype=bool),
        np.ones((batch_size, num_past_only, padded_horizon), dtype=bool),
    ]
    if past_future_covariates is not None:
        pf_future_vals = past_future_covariates[..., context : context + horizon]
        pf_future_masks = past_future_mask[..., context : context + horizon]
        if spec.use_linear_detrending:
            m_pf = m_trend[:, num_target + num_past_only :, :]
            c_pf = c_trend[:, num_target + num_past_only :, :]
            apply_detrend_pf = apply_detrend[:, num_target + num_past_only :, :]
            t_hor_pf = np.arange(1, horizon + 1, dtype=np.float32)[None, None, :]
            t_hor_pf = t_hor_pf / np.float32(context)
            pf_trend_hor = m_pf * t_hor_pf + c_pf
            pf_future_vals = np.where(
                apply_detrend_pf, pf_future_vals - pf_trend_hor, pf_future_vals
            )
        pf_future_vals = np.where(pf_future_masks, np.float32(0.0), pf_future_vals)
        if hor_padding > 0:
            pf_future_vals = np.concatenate(
                [pf_future_vals, np.zeros((batch_size, pf_future_vals.shape[1], hor_padding), dtype=np.float32)],
                axis=-1,
            )
            pf_future_masks = np.concatenate(
                [pf_future_masks, np.ones((batch_size, pf_future_masks.shape[1], hor_padding), dtype=bool)],
                axis=-1,
            )
        all_hor_vals.append(pf_future_vals)
        all_hor_masks.append(pf_future_masks)

    hor_vals = np.concatenate(all_hor_vals, axis=1)
    hor_masks = np.concatenate(all_hor_masks, axis=1)

    all_vals = np.concatenate([ctx_vals, hor_vals], axis=-1)
    all_masks = np.concatenate([ctx_masks, hor_masks], axis=-1)

    num_variates = all_vals.shape[1]
    patch_is_target = np.zeros(
        (batch_size, num_variates, num_context_patches + num_horizon_patches), dtype=bool
    )
    patch_is_target[:, : num_target + num_past_only, :] = True

    values_bvnp = all_vals.reshape(batch_size, num_variates, -1, p)
    masks_bvnp = all_masks.reshape(batch_size, num_variates, -1, p)

    num_total_patches = num_context_patches + num_horizon_patches
    horizon_cpm_mask = np.zeros((batch_size, num_total_patches), dtype=bool)
    horizon_cpm_mask[:, num_context_patches:] = True

    logits = timesfm3_forward(
        weights,
        values_bvnp,
        masks_bvnp,
        patch_is_target,
        horizon_cpm_mask,
        freeze_after=(num_context_patches - 1) if spec.use_frozen_running_stats else None,
    )

    if spec.use_stitching:
        extract_len = min(2 * p, spec.output_patch_len)
        forecast_indices = np.arange(num_forecast_patches) + (num_context_patches - 1)
        patch_preds = logits[:, :, forecast_indices, :extract_len, :]
        horizon_logits = _stitch_patches(patch_preds, p)[:, :, :horizon, :]
    else:
        num_forecast_chunks = padded_horizon // spec.output_patch_len
        forecast_indices = (
            np.arange(num_forecast_chunks) * spec.rolls + (num_context_patches - 1)
        )
        forecast_logits = logits[:, :, forecast_indices, :, :]
        horizon_logits = forecast_logits.reshape(
            batch_size, num_variates, -1, len(spec.quantiles)
        )[:, :, :horizon, :]

    if spec.use_linear_detrending:
        t_forecast = np.arange(1, horizon + 1, dtype=np.float32) / np.float32(context)
        trend_forecast = (
            m_trend[:, :, 0, None] * t_forecast[None, None, :] + c_trend[:, :, 0, None]
        )
        trend_forecast = np.where(apply_detrend[:, :, 0, None], trend_forecast, np.float32(0.0))
        horizon_logits = horizon_logits + trend_forecast[:, :, :, None]

    return horizon_logits.astype(np.float32)


__all__ = [
    "TimesFM3HostWeights",
    "timesfm3_decode",
    "timesfm3_forward",
]
