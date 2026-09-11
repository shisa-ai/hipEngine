"""Generate the TimesFM 2.5 200M CPU-reference oracle fixture with real torch.

Oracle lineage: google-research/timesfm (Apache-2.0) master
src/timesfm/torch/{transformer,dense,normalization,util}.py and
src/timesfm/timesfm_2p5/timesfm_2p5_torch.py, vendored verbatim below (trimmed
to the classes the 2.5 200M module needs).  This script is a test-fixture
generator only; the hipEngine hot path never imports torch.

Usage:
    python3 scripts/timesfm_oracle_torch.py \
        --output tests/fixtures/cpu_reference/timesfm_2p5_200m_decode.npz

Writes seeded inputs/masks plus the reference decode outputs (point forecast
backcast head, quantile spread, AR outputs) for later RED/GREEN checks of the
NumPy CPU reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

PINNED_MODEL_ID = "google/timesfm-2.5-200m-pytorch"

# ---------------------------------------------------------------------------
# Vendored from google-research/timesfm (Apache-2.0), master.
# ---------------------------------------------------------------------------


class RMSNorm(torch.nn.Module):
  """RMS normalization."""

  def __init__(self, num_features, *, epsilon=1e-6):
    super().__init__()
    self.scale = torch.nn.Parameter(torch.zeros(num_features))
    self.num_features = num_features
    self.epsilon = epsilon

  def forward(self, inputs):
    var = torch.mean(torch.square(inputs), dim=-1, keepdim=True)
    normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
    normed_inputs = normed_inputs * self.scale
    return normed_inputs


class ResidualBlock(torch.nn.Module):
  """Residual block with two linear layers and a linear residual connection."""

  def __init__(self, input_dims, hidden_dims, output_dims, use_bias, activation):
    super().__init__()
    self.hidden_layer = torch.nn.Linear(input_dims, hidden_dims, bias=use_bias)
    self.output_layer = torch.nn.Linear(hidden_dims, output_dims, bias=use_bias)
    self.residual_layer = torch.nn.Linear(input_dims, output_dims, bias=use_bias)
    if activation == "relu":
      self.activation = torch.nn.ReLU()
    elif activation == "swish":
      self.activation = torch.nn.SiLU()
    elif activation == "none":
      self.activation = torch.nn.Identity()
    else:
      raise ValueError(f"Activation: {activation} not supported.")

  def forward(self, x):
    return self.output_layer(
        self.activation(self.hidden_layer(x))
    ) + self.residual_layer(x)


class RotaryPositionalEmbedding(torch.nn.Module):
  """Rotary positional embedding."""

  def __init__(self, embedding_dims, min_timescale=1.0, max_timescale=10000.0):
    super().__init__()
    self.embedding_dims = embedding_dims
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale

  def forward(self, inputs, position=None):
    if self.embedding_dims != inputs.shape[-1]:
      raise ValueError("rotary dims must match hidden dim")
    half_embedding_dim = self.embedding_dims // 2
    fraction = (
        2
        * torch.arange(0, half_embedding_dim, device=inputs.device)
        / self.embedding_dims
    )
    timescale = (
        self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    ).to(inputs.device)
    if position is None:
      seq_length = inputs.shape[1]
      position = torch.arange(seq_length, dtype=torch.float32, device=inputs.device)[
          None, :
      ]
    if len(inputs.shape) == 4:
      position = position[..., None, None]
      timescale = timescale[None, None, None, :]
    elif len(inputs.shape) == 3:
      position = position[..., None]
      timescale = timescale[None, None, :]
    else:
      raise ValueError("Inputs must be of rank 3 or 4.")
    sinusoid_inp = position / timescale
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    first_half, second_half = torch.chunk(inputs, 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return torch.cat([first_part, second_part], dim=-1)


def make_attn_mask(query_length, num_all_masked_kv, query_index_offset=None, kv_length=0):
  if kv_length == 0:
    kv_length = query_length
  q_index = torch.arange(query_length, device=num_all_masked_kv.device)[
      None, None, :, None
  ]
  if query_index_offset is not None:
    q_index = q_index + query_index_offset[:, None, None, None]
  kv_index = torch.arange(kv_length, device=num_all_masked_kv.device)[
      None, None, None, :
  ]
  return torch.logical_and(
      q_index >= kv_index,
      kv_index >= num_all_masked_kv[:, None, None, None],
  )


def _dot_product_attention(query, key, value, mask=None):
  attn_weights = torch.einsum("...qhd,...khd->...hqk", query, key)
  if mask is not None:
    attn_weights = torch.where(
        mask, attn_weights, -torch.finfo(attn_weights.dtype).max / 2
    )
  attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
  return torch.einsum("...hqk,...khd->...qhd", attn_weights, value)


class PerDimScale(torch.nn.Module):
  """Per-dimension scaling."""

  def __init__(self, num_dims):
    super().__init__()
    self.num_dims = num_dims
    self.per_dim_scale = torch.nn.Parameter(torch.zeros(num_dims))

  def forward(self, x):
    scale_factor = (
        1.442695041 / math.sqrt(self.num_dims) * torch.nn.functional.softplus(self.per_dim_scale)
    )
    return x * scale_factor


class MultiHeadAttention(torch.nn.Module):
  """Multi-head attention."""

  def __init__(self, num_heads, in_features, *, fuse_qkv=True):
    super().__init__()
    self.num_heads = num_heads
    self.in_features = in_features
    self.head_dim = in_features // num_heads
    self.fuse_qkv = fuse_qkv
    if self.fuse_qkv:
      self.qkv_proj = torch.nn.Linear(in_features, 3 * in_features, bias=False)
    self.out = torch.nn.Linear(in_features, in_features, bias=False)
    self.query_ln = RMSNorm(self.head_dim)
    self.key_ln = RMSNorm(self.head_dim)
    self.rotary_position_embedding = RotaryPositionalEmbedding(self.head_dim)
    self.per_dim_scale = PerDimScale(num_dims=self.head_dim)

  def forward(self, inputs_q, *, decode_cache=None, patch_mask=None):
    b, n_patches, _ = inputs_q.shape
    if patch_mask is None:
      patch_mask = torch.zeros(b, n_patches, dtype=torch.bool, device=inputs_q.device)

    qkv = self.qkv_proj(inputs_q)
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    query = query.view(b, n_patches, self.num_heads, self.head_dim)
    key = key.view(b, n_patches, self.num_heads, self.head_dim)
    value = value.view(b, n_patches, self.num_heads, self.head_dim)

    if decode_cache is None:
      num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)
      next_index = torch.zeros_like(num_masked, dtype=torch.int32)
    else:
      num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1) + decode_cache.num_masked
      next_index = decode_cache.next_index.clone()

    position = (
        torch.arange(n_patches, device=inputs_q.device)[None, :]
        + next_index[:, None]
        - num_masked[:, None]
    )
    query = self.rotary_position_embedding(query, position)
    key = self.rotary_position_embedding(key, position)

    query = self.query_ln(query)
    key = self.key_ln(key)
    query = self.per_dim_scale(query)

    if decode_cache is not None:
      _, decode_cache_size, _, _ = decode_cache.value.shape
      start = decode_cache.next_index[0]
      end = start + n_patches
      decode_cache.key[:, start:end] = key
      decode_cache.value[:, start:end] = value
      key = decode_cache.key
      value = decode_cache.value
      decode_cache.next_index += n_patches
      decode_cache.num_masked = num_masked
      attn_mask = make_attn_mask(
          query_length=n_patches,
          num_all_masked_kv=num_masked,
          query_index_offset=next_index,
          kv_length=decode_cache_size,
      )
    else:
      attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)

    x = _dot_product_attention(query, key, value, mask=attn_mask)
    x = x.reshape(b, n_patches, self.in_features)
    out = self.out(x)
    return out, decode_cache


class Transformer(torch.nn.Module):
  """Classic Transformer used in TimesFM."""

  def __init__(self, model_dims, hidden_dims, num_heads):
    super().__init__()
    self.pre_attn_ln = RMSNorm(num_features=model_dims)
    self.post_attn_ln = RMSNorm(num_features=model_dims)
    self.attn = MultiHeadAttention(num_heads=num_heads, in_features=model_dims)
    self.pre_ff_ln = RMSNorm(num_features=model_dims)
    self.post_ff_ln = RMSNorm(num_features=model_dims)
    self.ff0 = torch.nn.Linear(model_dims, hidden_dims, bias=False)
    self.ff1 = torch.nn.Linear(hidden_dims, model_dims, bias=False)
    self.activation = torch.nn.SiLU()

  def forward(self, input_embeddings, patch_mask, decode_cache=None):
    attn_output, decode_cache = self.attn(
        inputs_q=self.pre_attn_ln(input_embeddings),
        decode_cache=decode_cache,
        patch_mask=patch_mask,
    )
    attn_output = self.post_attn_ln(attn_output) + input_embeddings
    output_embeddings = (
        self.post_ff_ln(self.ff1(self.activation(self.ff0(self.pre_ff_ln(attn_output)))))
        + attn_output
    )
    return output_embeddings, decode_cache


@dataclasses.dataclass(frozen=False)
class DecodeCache:
  """Cache for decoding."""

  next_index: torch.Tensor
  num_masked: torch.Tensor
  key: torch.Tensor
  value: torch.Tensor


_TOLERANCE = 1e-6


def revin(x, mu, sigma, reverse=False):
  if len(mu.shape) == len(x.shape) - 1:
    mu = mu[..., None]
    sigma = sigma[..., None]
  elif len(mu.shape) == len(x.shape) - 2:
    mu = mu[..., None, None]
    sigma = sigma[..., None, None]
  if reverse:
    return x * sigma + mu
  else:
    return (x - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)


def update_running_stats(n, mu, sigma, x, mask):
  is_legit = torch.logical_not(mask)
  inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)
  inc_mu_numerator = torch.sum(x * is_legit, dim=-1)
  inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
  inc_mu = inc_mu_numerator / inc_n_safe
  inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)
  inc_var_numerator = torch.sum(((x - inc_mu.unsqueeze(-1)) ** 2) * is_legit, dim=-1)
  inc_var = inc_var_numerator / inc_n_safe
  inc_var = torch.where(inc_n == 0, 0.0, inc_var)
  inc_sigma = torch.sqrt(inc_var)
  new_n = n + inc_n
  new_n_safe = torch.where(new_n == 0, 1.0, new_n)
  new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
  new_mu = torch.where(new_n == 0, 0.0, new_mu)
  term1 = n * sigma.pow(2)
  term2 = inc_n * inc_sigma.pow(2)
  term3 = n * (mu - new_mu).pow(2)
  term4 = inc_n * (inc_mu - new_mu).pow(2)
  new_var = (term1 + term2 + term3 + term4) / new_n_safe
  new_var = torch.where(new_n == 0, 0.0, new_var)
  new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))
  return (new_n, new_mu, new_sigma), (new_n, new_mu, new_sigma)


# ---------------------------------------------------------------------------
# TimesFM 2.5 200M module (decode path only).
# ---------------------------------------------------------------------------


class TimesFM25Module(torch.nn.Module):
  def __init__(self, device):
    super().__init__()
    self.p = 32
    self.o = 128
    self.os = 1024
    self.m = self.o // self.p
    self.x = 20
    self.h = 16
    self.md = 1280
    self.hd = self.md // self.h
    self.q = 10
    self.aridx = 5
    self.tokenizer = ResidualBlock(64, 1280, 1280, use_bias=True, activation="swish")
    self.stacked_xf = torch.nn.ModuleList(
        [Transformer(1280, 1280, 16) for _ in range(self.x)]
    )
    self.output_projection_point = ResidualBlock(
        1280, 1280, 1280, use_bias=False, activation="swish"
    )
    self.output_projection_quantiles = ResidualBlock(
        1280, 1280, 10240, use_bias=False, activation="swish"
    )
    self.device = device

  def forward(self, inputs, masks, decode_caches=None):
    tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
    input_embeddings = self.tokenizer(tokenizer_inputs)
    if decode_caches is None:
      decode_caches = [None] * self.x
    output_embeddings = input_embeddings
    new_decode_caches = []
    for i, layer in enumerate(self.stacked_xf):
      output_embeddings, new_cache = layer(
          output_embeddings, masks[..., -1], decode_caches[i]
      )
      new_decode_caches.append(new_cache)
    output_ts = self.output_projection_point(output_embeddings)
    output_quantile_spread = self.output_projection_quantiles(output_embeddings)
    return (
        input_embeddings,
        output_embeddings,
        output_ts,
        output_quantile_spread,
    ), new_decode_caches

  def decode(self, horizon, inputs, masks):
    with torch.no_grad():
      batch_size, context = inputs.shape[0], inputs.shape[1]
      num_decode_steps = (horizon - 1) // self.o
      num_input_patches = context // self.p
      decode_cache_size = num_input_patches + num_decode_steps * self.m

      patched_inputs = torch.reshape(inputs, (batch_size, -1, self.p))
      patched_masks = torch.reshape(masks, (batch_size, -1, self.p))

      n = torch.zeros(batch_size, device=inputs.device)
      mu = torch.zeros(batch_size, device=inputs.device)
      sigma = torch.zeros(batch_size, device=inputs.device)
      patch_mu = []
      patch_sigma = []
      for i in range(num_input_patches):
        (n, mu, sigma), _ = update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)
      last_n, last_mu, last_sigma = n, mu, sigma
      context_mu = torch.stack(patch_mu, dim=1)
      context_sigma = torch.stack(patch_sigma, dim=1)

      decode_caches = [
          DecodeCache(
              next_index=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
              num_masked=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
              key=torch.zeros(
                  batch_size, decode_cache_size, self.h, self.hd, device=inputs.device
              ),
              value=torch.zeros(
                  batch_size, decode_cache_size, self.h, self.hd, device=inputs.device
              ),
          )
          for _ in range(self.x)
      ]

      normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
      normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
      (_, _, normed_outputs, normed_quantile_spread), decode_caches = self(
          normed_inputs, patched_masks, decode_caches
      )
      renormed_outputs = torch.reshape(
          revin(normed_outputs, context_mu, context_sigma, reverse=True),
          (batch_size, -1, self.o, self.q),
      )
      renormed_quantile_spread = torch.reshape(
          revin(normed_quantile_spread, context_mu, context_sigma, reverse=True),
          (batch_size, -1, self.os, self.q),
      )[:, -1, ...]

      ar_outputs = []
      last_renormed_output = renormed_outputs[:, -1, :, self.aridx]

      for _ in range(num_decode_steps):
        new_patched_input = torch.reshape(
            last_renormed_output, (batch_size, self.m, self.p)
        )
        new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)

        n, mu, sigma = last_n, last_mu, last_sigma
        new_mus, new_sigmas = [], []
        for i in range(self.m):
          (n, mu, sigma), _ = update_running_stats(
              n, mu, sigma, new_patched_input[:, i], new_mask[:, i]
          )
          new_mus.append(mu)
          new_sigmas.append(sigma)
        last_n, last_mu, last_sigma = n, mu, sigma
        new_mu = torch.stack(new_mus, dim=1)
        new_sigma = torch.stack(new_sigmas, dim=1)

        new_normed_input = revin(new_patched_input, new_mu, new_sigma, reverse=False)
        (_, _, new_normed_output, _), decode_caches = self(
            new_normed_input, new_mask, decode_caches
        )

        new_renormed_output = torch.reshape(
            revin(new_normed_output, new_mu, new_sigma, reverse=True),
            (batch_size, self.m, self.o, self.q),
        )
        ar_outputs.append(new_renormed_output[:, -1, ...])
        last_renormed_output = new_renormed_output[:, -1, :, self.aridx]

      if num_decode_steps > 0:
        ar_renormed_outputs = torch.stack(ar_outputs, dim=1)
      else:
        ar_renormed_outputs = None

    return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default=PINNED_MODEL_ID)
  parser.add_argument("--output", required=True)
  parser.add_argument("--seed", type=int, default=20260917)
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--context", type=int, default=512)
  parser.add_argument("--horizon", type=int, default=256)
  args = parser.parse_args()

  snapshot = Path(args.model)
  if not snapshot.is_dir():
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(args.model))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = TimesFM25Module(device)
  state = load_file(str(snapshot / "model.safetensors"))
  # The vendored module drops the config indirection but keeps checkpoint names.
  model.load_state_dict(state, strict=True)
  model.to(device)
  model.eval()

  rng = np.random.default_rng(args.seed)
  batch, context, horizon = args.batch, args.context, args.horizon
  # Two regimes: a smooth sinusoid + noise and a random walk.
  t = np.arange(context, dtype=np.float64)
  series0 = np.sin(2 * np.pi * t / 48.0) * 3.0 + 0.5 * rng.standard_normal(context)
  series1 = np.cumsum(rng.standard_normal(context) * 0.25) + 10.0
  inputs = np.stack([series0, series1], axis=0).astype(np.float32)[:, :context]
  if batch > 2:
    inputs = np.concatenate(
        [inputs, np.tile(series1[None, :context], (batch - 2, 1)).astype(np.float32)],
        axis=0,
    )
  masks = np.zeros((batch, context), dtype=bool)
  # Exercise the front-padding mask path on one series.
  pad = 32 * 2
  series1_padded = np.concatenate(
      [np.zeros(pad, dtype=np.float32), inputs[1, : context - pad]], axis=0
  )
  inputs[1] = series1_padded
  masks[1, :pad] = True

  inputs_t = torch.from_numpy(inputs).to(device)
  masks_t = torch.from_numpy(masks).to(device)

  renormed_outputs, quantile_spread, ar_outputs = model.decode(
      horizon, inputs_t, masks_t
  )

  payload = {
      "schema": np.asarray(1, dtype=np.int64),
      "seed": np.asarray(args.seed, dtype=np.int64),
      "inputs": inputs,
      "masks": masks,
      "horizon": np.asarray(horizon, dtype=np.int64),
      "renormed_outputs": renormed_outputs.detach().cpu().numpy(),
      "quantile_spread": quantile_spread.detach().cpu().numpy(),
  }
  if ar_outputs is not None:
    payload["ar_outputs"] = ar_outputs.detach().cpu().numpy()

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **payload)
  print(
      f"wrote {output}: batch={batch} context={context} horizon={horizon}, "
      f"pf={tuple(renormed_outputs.shape)} qs={tuple(quantile_spread.shape)} "
      f"ar={tuple(ar_outputs.shape) if ar_outputs is not None else None}"
  )


if __name__ == "__main__":
  main()
