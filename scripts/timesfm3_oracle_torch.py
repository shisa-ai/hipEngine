"""Generate the TimesFM 3.0 CPU-reference oracle fixture with real torch.

Oracle lineage: google-research/timesfm (Apache-2.0) master
src/timesfm3/torch/{configs,dense,normalization,transformer,util,cpm_revin_refine,model}.py,
vendored verbatim below (the PyTorchModelHubMixin base and save/load helpers
are dropped; only decode-relevant paths are kept).  This script is a
test-fixture generator only; the hipEngine hot path never imports torch.

Usage:
    python3 scripts/timesfm3_oracle_torch.py \
        --output tests/fixtures/cpu_reference/timesfm_3p0_decode.npz

Writes seeded inputs/masks/covariates plus the reference decode outputs
(non-autoregressive single pass) for later RED/GREEN checks of the NumPy
CPU reference.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Vendored from google-research/timesfm (Apache-2.0), master.
# ---------------------------------------------------------------------------

import dataclasses
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# The vendored modules referenced each other as ``configs.X``, ``dense.X``, ...
# All definitions live in this single namespace now.
class _SelfNamespace:
  def __getattr__(self, name):
    return globals()[name]


configs = dense = transformer = util = normalization = cpm_revin_refine_lib = _SelfNamespace()

# --- vendored from src/timesfm3/torch/configs.py ---
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract configs for TimesFM-3 layers."""

from typing import Literal


@dataclasses.dataclass(frozen=True)
class ResidualBlockConfig:
  """Framework-agnostic config for a residual block."""

  hidden_dims: int
  output_dims: int
  use_bias: bool
  activation: Literal["relu", "swish", "none"]
  dropout: float = 0.0
  identity_skip: bool = False
  prenorm: Literal["rms", "none"] = "none"


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Framework-agnostic config for a transformer."""

  model_dims: int
  hidden_dims: int
  num_heads: int
  attention_norm: Literal["rms"]
  feedforward_norm: Literal["rms"]
  qk_norm: Literal["rms", "none"]
  use_bias: bool
  use_rope_seq: bool
  use_rope_var: bool
  ff_activation: Literal["relu", "swish", "none"]
  deterministic: bool
  v_norm: Literal["rms", "none"] = "none"
  causal_attention: bool = True
  debug_no_masking: bool = False
  training: bool = True
  use_memory_efficient_attention: bool = True
  paired_token_skip_second: bool = False
  max_variates: int = 32
  # PyTorch-only: when True uses F.scaled_dot_product_attention.
  use_sdpa: bool = True


@dataclasses.dataclass(frozen=True)
class StackedTransformersConfig:
  """Framework-agnostic config for a stacked transformers."""

  num_layers: int
  transformer: TransformerConfig
  use_remat: bool = True

# --- vendored from src/timesfm3/torch/normalization.py ---
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalization layers for TimesFM3 PyTorch."""


import math


_RECIPROCAL_OF_SOFTPLUS_0 = 1.442695041


class PerDimScale(nn.Module):
  """Per-dimension scaling (Pax-style).

  Replaces the standard 1/sqrt(d) query scaling with a learnable:
    x * RECIPROCAL_OF_SOFTPLUS_0 / sqrt(num_dims) * softplus(per_dim_scale)

  The per_dim_scale parameter is initialized to zeros, so at init time
  softplus(0) ≈ 0.693..., and the net scale is close to 1/sqrt(d).
  """

  def __init__(self, num_dims: int):
    super().__init__()
    self.num_dims = num_dims
    self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Applies per-dim scaling to the last dimension of x."""
    return (
      x
      * _RECIPROCAL_OF_SOFTPLUS_0
      / math.sqrt(self.num_dims)
      * torch.nn.functional.softplus(self.per_dim_scale)
    )

# --- vendored from src/timesfm3/torch/util.py ---
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions and classes for TimesFM3 PyTorch implementation."""


import os
from collections.abc import Callable


_TOLERANCE = 1e-6


def _make_safe_for_division(values: torch.Tensor) -> torch.Tensor:
  """Handles near zero values."""
  return torch.where(values < _TOLERANCE, 1.0, values)


@dataclasses.dataclass
class DecodeCache:
  """Cache for autoregressive decoding.

  Attributes:
    next_index: The next index to decode for each batch element.
      Shape: (batch_leading,).
    num_front_masked: Number of front masked tokens for each batch element.
      Shape: (batch_leading,).
    key: The key cache. Shape: (batch_leading, cache_len, num_heads, head_dim).
    value: The value cache. Same shape as key.
  """

  next_index: torch.Tensor
  num_front_masked: torch.Tensor
  key: torch.Tensor
  value: torch.Tensor

  @classmethod
  def init_decode_cache(
    cls,
    num_layers: int,
    batch_size: int,
    num_variates: int,
    num_total_input_patches: int,
    num_heads: int,
    head_dim: int,
    device: torch.device | None = None,
  ) -> list[DecodeCache]:
    """Initializes a list of decode caches for stacked layers.

    Args:
      num_layers: The number of transformer layers.
      batch_size: The batch size.
      num_variates: The number of variates.
      num_total_input_patches: Total number of patches the cache should hold.
      num_heads: The number of attention heads.
      head_dim: The head dimension.
      device: The device to create tensors on.

    Returns:
      A list of DecodeCache, one per layer.
    """
    leading_size = batch_size * num_variates
    return [
      cls(
        next_index=torch.zeros(leading_size, dtype=torch.int32, device=device),
        num_front_masked=torch.zeros(leading_size, dtype=torch.int32, device=device),
        key=torch.zeros(
          leading_size,
          num_total_input_patches,
          num_heads,
          head_dim,
          device=device,
        ),
        value=torch.zeros(
          leading_size,
          num_total_input_patches,
          num_heads,
          head_dim,
          device=device,
        ),
      )
      for _ in range(num_layers)
    ]


def update_running_stats(
  n: torch.Tensor,
  mu: torch.Tensor,
  sigma: torch.Tensor,
  x: torch.Tensor,
  mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Updates running stats with a new patch of data.

  Args:
    n: Count of seen non-masked elements. Shape: (b, v).
    mu: Running mean. Shape: (b, v).
    sigma: Running std. Shape: (b, v).
    x: New data patch. Shape: (b, v, p).
    mask: Boolean mask where True = masked/invalid. Shape: (b, v, p).

  Returns:
    Tuple of (new_n, new_mu, new_sigma), each of shape (b, v).
  """
  is_legit = ~mask
  is_legit_f = is_legit.float()
  inc_n = is_legit_f.sum(dim=-1)

  # mean of valid elements in patch
  x_masked = torch.where(is_legit, x, torch.zeros_like(x))
  inc_sum = x_masked.sum(dim=-1)
  inc_mu = torch.where(inc_n == 0, torch.zeros_like(inc_sum), inc_sum / inc_n)

  # std of valid elements in patch
  x_diff_sq = torch.where(
    is_legit, (x - inc_mu.unsqueeze(-1)) ** 2, torch.zeros_like(x)
  )
  inc_var = torch.where(
    inc_n == 0,
    torch.zeros_like(inc_sum),
    x_diff_sq.sum(dim=-1) / inc_n,
  )
  inc_sigma = torch.sqrt(inc_var)

  new_n = n + inc_n
  new_mu = torch.where(
    new_n == 0,
    torch.zeros_like(mu),
    (n * mu + inc_mu * inc_n) / new_n,
  )
  new_sigma = torch.sqrt(
    torch.where(
      new_n == 0,
      torch.zeros_like(sigma),
      (
        n * sigma * sigma
        + inc_n * inc_sigma * inc_sigma
        + n * (mu - new_mu) * (mu - new_mu)
        + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)
      )
      / new_n,
    )
  )
  return new_n, new_mu, new_sigma


def get_running_stats(
  values: torch.Tensor,
  masks: torch.Tensor,
  *,
  segment_ids: torch.Tensor | None = None,
  initial_stats: (tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None) = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Computes cumulative running statistics patch-by-patch.

  For each variate, patch `i` gets stats computed from all unmasked values
  in patches 0 through i (inclusive). Statistics reset at segment boundaries.

  Args:
    values: Input values. Shape: (b, v, n, p).
    masks: Boolean mask (True=masked). Shape: (b, v, n, p).
    segment_ids: Segment IDs. Shape: (b, n). If None, no segment resets.
    initial_stats: Optional initial (n, mu, sigma), each shape (b, v).

  Returns:
    Tuple of (running_n, running_mu, running_sigma), each shape (b, v, n).
  """
  b, v, n, _ = values.shape
  device = values.device

  if initial_stats is None:
    init_n = torch.zeros((b, v), dtype=torch.float32, device=device)
    init_mu = torch.zeros((b, v), dtype=torch.float32, device=device)
    init_sigma = torch.zeros((b, v), dtype=torch.float32, device=device)
  else:
    init_n, init_mu, init_sigma = initial_stats

  # Determine segment reset points
  if segment_ids is None:
    is_new_segment = torch.zeros((b, n), dtype=torch.bool, device=device)
  else:
    shifted = F.pad(segment_ids[:, :-1], (1, 0), value=-1)
    is_new_segment = segment_ids != shifted

  all_n = []
  all_mu = []
  all_sigma = []
  cur_n, cur_mu, cur_sigma = init_n, init_mu, init_sigma

  for i in range(n):
    # Reset stats at new segments
    reset = is_new_segment[:, i].unsqueeze(-1)  # (b, 1)  # pyrefly: ignore[bad-index]
    cur_n = torch.where(reset, init_n, cur_n)
    cur_mu = torch.where(reset, init_mu, cur_mu)
    cur_sigma = torch.where(reset, init_sigma, cur_sigma)

    cur_n, cur_mu, cur_sigma = update_running_stats(
      cur_n, cur_mu, cur_sigma, values[:, :, i, :], masks[:, :, i, :]
    )
    all_n.append(cur_n)
    all_mu.append(cur_mu)
    all_sigma.append(cur_sigma)

  return (
    torch.stack(all_n, dim=2),
    torch.stack(all_mu, dim=2),
    torch.stack(all_sigma, dim=2),
  )


def revin(
  x: torch.Tensor,
  mu: torch.Tensor,
  sigma: torch.Tensor,
  reverse: bool = False,
) -> torch.Tensor:
  """Reversible per-instance normalization.

  Automatically expands mu/sigma dims to match x.

  Args:
    x: Input tensor. Shape: (b, ..., d).
    mu: Mean tensor. Shape: (b, ...) with 1 or 2 fewer dims than x.
    sigma: Std tensor. Same shape as mu.
    reverse: If True, applies reverse normalization (denormalize).

  Returns:
    Normalized or denormalized tensor, same shape as x.
  """
  if mu.dim() == x.dim() - 1:
    mu = mu.unsqueeze(-1)
    sigma = sigma.unsqueeze(-1)
  elif mu.dim() == x.dim() - 2:
    mu = mu.unsqueeze(-1).unsqueeze(-1)
    sigma = sigma.unsqueeze(-1).unsqueeze(-1)
  else:
    raise ValueError(f"Unsupported shapes for x and mu: {x.shape}, {mu.shape}.")
  if reverse:
    return x * sigma + mu
  else:
    return (x - mu) / _make_safe_for_division(sigma)


def get_output_patch_via_roll(
  x: torch.Tensor, rolls: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Creates labels of output_patch length by rolling the patched inputs.

  Takes patched input (b, v, n, p) and creates output patches of length
  p*rolls by concatenating shifted views of the patches.

  Args:
    x: Patched inputs. Shape: (b, v, n, p).
    rolls: Number of rolls (= output_patch_len / patch_len).

  Returns:
    Tuple of:
      - Rolled output. Shape: (b, v, n, p * rolls).
      - Wrap-around mask. Shape: (1, 1, n, p * rolls) bool.
  """
  b, v, n, p = x.shape
  device = x.device
  rolling_mat = torch.zeros(b, v, n, rolls + 1, p, device=device, dtype=x.dtype)
  rolling_mat[:, :, :, 0, :] = x

  for i in range(rolls):
    rolling_mat[:, :, :, i + 1, :] = torch.roll(
      rolling_mat[:, :, :, i, :], shifts=-1, dims=2
    )

  # Take [1:] along the roll axis and flatten
  result = rolling_mat[:, :, :, 1:, :].reshape(b, v, n, rolls * p)

  # Build wrap-around mask
  patch_idx = torch.arange(n, device=device)
  point_idx = torch.arange(rolls * p, device=device)
  source_patch = patch_idx[:, None] + 1 + point_idx[None, :] // p
  wrap_mask = (source_patch >= n).unsqueeze(0).unsqueeze(0)

  return result, wrap_mask


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
  "relu": F.relu,
  "swish": F.silu,
  "silu": F.silu,
  "none": lambda x: x,
}


def get_activation_fn(
  activation_name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
  """Returns the activation function for the given name."""
  try:
    return _ACTIVATIONS[activation_name]
  except KeyError:
    raise ValueError(
      f"Activation: {activation_name} not supported. Supported "
      f"activations: {list(_ACTIVATIONS.keys())}"
    ) from None


def load_safetensors(
  pytorch_safetensors_path: str,
  device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
  """Loads a PyTorch state dict from a safetensors file.

  Args:
    pytorch_safetensors_path: Path to the safetensors file.
    device: The device to load the tensors onto.

  Returns:
    A dictionary of PyTorch state dict weights.
  """
  expanded_path = os.path.expanduser(pytorch_safetensors_path)
  return safetensors_torch.load_file(expanded_path, device=str(device))


def stitch_patches(
  patch_preds: torch.Tensor,
  patch_len: int,
) -> torch.Tensor:
  """Stitches overlapping patch predictions.

  Each patch predicts patch_len + overlap timepoints, where
  overlap = patch_preds.shape[3] - patch_len is inferred from the input.
  Consecutive patches share overlap timepoints, which are linearly stitched.

  Args:
    patch_preds: Predictions of shape (batch, variates, num_patches, patch_len
      + overlap, num_quantiles).
    patch_len: The patch length.

  Returns:
    Stitched predictions of shape
    (batch, variates, num_patches * patch_len + overlap, num_quantiles).
  """
  b, v, num_patches, total_len, q = patch_preds.shape
  overlap = total_len - patch_len

  if num_patches == 1:
    return patch_preds[:, :, 0, :, :]

  stitch_weights = torch.linspace(
    1.0, 0.0, overlap, device=patch_preds.device, dtype=patch_preds.dtype
  )
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

  output_chunks = torch.cat([stitched_overlaps, middles], dim=3)

  mid = output_chunks.reshape(b, v, (num_patches - 1) * patch_len, q)

  tail = patch_preds[:, :, -1, patch_len:, :]

  return torch.cat([first_chunk, mid, tail], dim=2)

# --- vendored from src/timesfm3/torch/dense.py ---
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense layers for TimesFM3 PyTorch (inference only)."""





class ResidualBlock(nn.Module):
  """Residual block with two linear layers and a linear residual connection.

  Architecture:
    if prenorm == "rms": x_norm = RMSNorm(x)  else: x_norm = x
    hidden = activation(hidden_layer(x_norm))
    output = output_layer(hidden) + residual_layer(x)  [or + x if identity_skip]
  """

  def __init__(self, config: configs.ResidualBlockConfig):
    super().__init__()
    self.config = config

    # Defining placeholder layers in __init__ ensures PyTorch registers them as
    # submodules. This is required for standard parameter tracking, printing/
    # debugging, and correctly propagating device/dtype moves applied to the
    # parent module (e.g. `model.to(device)`) before any forward handles them.
    # They are safely re-initialized/overwritten with correct dimensions during
    # the first forward pass (using `set_input_dims()`) before the matrix
    # multiplication is evaluated, avoiding shape mismatch errors.
    self.hidden_layer = nn.Linear(
      in_features=config.hidden_dims,  # placeholder, set in first forward
      out_features=config.hidden_dims,
      bias=config.use_bias,
    )
    self.output_layer = nn.Linear(
      in_features=config.hidden_dims,
      out_features=config.output_dims,
      bias=config.use_bias,
    )

    if config.identity_skip:
      self.residual_layer = None
    else:
      self.residual_layer = nn.Linear(
        in_features=config.hidden_dims,  # placeholder
        out_features=config.output_dims,
        bias=config.use_bias,
      )

    self.activation = util.get_activation_fn(config.activation)

    if config.prenorm == "rms":
      self.pre_norm = nn.RMSNorm(config.hidden_dims)
    else:
      self.pre_norm = None

    # Mark layers as lazy so input dim gets set on first use
    self._input_dim_set = False

  def set_input_dims(self, input_dim: int) -> None:
    """Reinitialize linear layers with the correct input dimension."""
    if self._input_dim_set:
      return
    device = self.hidden_layer.weight.device
    dtype = self.hidden_layer.weight.dtype

    self.hidden_layer = nn.Linear(
      input_dim, self.config.hidden_dims, bias=self.config.use_bias
    ).to(device=device, dtype=dtype)
    self.output_layer = nn.Linear(
      self.config.hidden_dims,
      self.config.output_dims,
      bias=self.config.use_bias,
    ).to(device=device, dtype=dtype)
    if self.residual_layer is not None:
      self.residual_layer = nn.Linear(
        input_dim, self.config.output_dims, bias=self.config.use_bias
      ).to(device=device, dtype=dtype)
    if self.pre_norm is not None:
      self.pre_norm = nn.RMSNorm(input_dim).to(device=device, dtype=dtype)
    self._input_dim_set = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass. x shape: (b, ..., input_dim)."""
    if not self._input_dim_set:
      self.set_input_dims(x.shape[-1])

    if self.pre_norm is not None:
      hidden_input = self.pre_norm(x)
    else:
      hidden_input = x

    hidden_output = self.activation(self.hidden_layer(hidden_input))

    if self.residual_layer is not None:
      return self.output_layer(hidden_output) + self.residual_layer(x)
    else:
      return self.output_layer(hidden_output) + x

# --- vendored from src/timesfm3/torch/transformer.py ---
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer layers for TimesFM3 PyTorch (inference only).

Port of the Flax MixingTransformer architecture:
  - RotaryPositionalEmbedding
  - MultiHeadAttention (with KV-cache support)
  - MixingTransformer (sequential seq + variate attention + FFN)
  - StackedMixingTransformer (nn.ModuleList of MixingTransformer)
"""


import math




def make_attn_mask(
  query_length: int,
  num_all_masked_kv: torch.Tensor,
  query_index_offset: torch.Tensor | None = None,
  kv_length: int = 0,
  causal: bool = True,
) -> torch.Tensor:
  """Makes attention mask. True = attend, False = mask.

  Args:
    query_length: Number of query positions.
    num_all_masked_kv: Shape (b,). Number of leading masked KV positions.
    query_index_offset: Shape (b,). Offset for query indices (decode mode).
    kv_length: Length of KV sequence. Defaults to query_length.
    causal: Whether to apply causal masking.

  Returns:
    Boolean mask of shape (b, 1, query_length, kv_length). True = attend.
  """
  if kv_length == 0:
    kv_length = query_length

  device = num_all_masked_kv.device
  q_index = torch.arange(query_length, device=device).view(1, 1, -1, 1)
  if query_index_offset is not None:
    q_index = q_index + query_index_offset.view(-1, 1, 1, 1)
  kv_index = torch.arange(kv_length, device=device).view(1, 1, 1, -1)
  mask = kv_index >= num_all_masked_kv.view(-1, 1, 1, 1)
  if causal:
    return (q_index >= kv_index) & mask
  return mask


def make_segment_mask(
  segment_ids: torch.Tensor,
) -> torch.Tensor:
  """Makes a segment mask from segment ids.

  Args:
    segment_ids: Shape (b, seq_length).

  Returns:
    Boolean mask of shape (b, 1, seq_length, seq_length).
  """
  return (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)


class RotaryPositionalEmbedding(nn.Module):
  """Rotary positional embedding (RoPE).

  Stateless module — no learnable parameters.
  Supports 3D (b, n, d) and 4D (b, n, h, hd) inputs.
  """

  def __init__(
    self,
    embedding_dims: int,
    min_timescale: int = 1,
    max_timescale: int = 10000,
  ):
    super().__init__()
    self.embedding_dims = embedding_dims
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale

    half_dim = embedding_dims // 2
    fraction = 2.0 * torch.arange(half_dim, dtype=torch.float32) / embedding_dims
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    self.register_buffer("timescale", timescale, persistent=False)

  def forward(
    self,
    inputs: torch.Tensor,
    position: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Applies rotary positional embeddings.

    Args:
      inputs: Shape (b, n, d) or (b, n, h, hd).
      position: Shape (b, n). If None, uses arange(n).

    Returns:
      Tensor with same shape as inputs, with RoPE applied.
    """
    if self.embedding_dims != inputs.shape[-1]:
      raise ValueError(
        "The embedding dims of the rotary position embedding "
        "must match the hidden dimension of the inputs."
      )
    timescale = self.timescale.to(inputs.device)

    if position is None:
      seq_length = inputs.shape[1]
      position = torch.arange(
        seq_length, device=inputs.device, dtype=torch.float32
      ).unsqueeze(0)

    if inputs.dim() == 4:
      # (b, n) -> (b, n, 1, 1) for broadcasting with (b, n, h, hd)
      pos = position.unsqueeze(-1).unsqueeze(-1)
      ts = timescale.view(1, 1, 1, -1)
    elif inputs.dim() == 3:
      pos = position.unsqueeze(-1)
      ts = timescale.view(1, 1, -1)
    else:
      raise ValueError("Inputs must be of rank 3 or 4.")

    sinusoid_inp = pos.float() / ts
    sin_val = torch.sin(sinusoid_inp)
    cos_val = torch.cos(sinusoid_inp)
    first_half, second_half = inputs.chunk(2, dim=-1)
    first_part = first_half * cos_val - second_half * sin_val
    second_part = second_half * cos_val + first_half * sin_val
    return torch.cat([first_part, second_part], dim=-1)


class MultiHeadAttention(nn.Module):
  """Multi-head attention with RoPE, QK-norm, PerDimScale, and KV-cache.

  This matches the Flax MultiHeadAttention exactly, including the
  pre-multiplication of query by sqrt(head_dim) which cancels with the
  standard 1/sqrt(d) scaling in dot-product attention.
  """

  def __init__(
    self,
    num_heads: int,
    in_features: int,
    use_per_dim_scale: bool = True,
    use_rotary_position_embeddings: bool = True,
    causal_attention: bool = True,
    use_bias: bool = False,
    qk_norm: str = "rms",
    v_norm: str = "none",
    use_sdpa: bool = False,
    rescale_logits: bool = False,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.in_features = in_features
    self.causal_attention = causal_attention
    self.head_dim = in_features // num_heads
    self.use_sdpa = use_sdpa
    # rescale_logits=False → MEA=True behaviour: Q is pre-multiplied by √d,
    #   no internal division. Matches Flax
    #   memory_efficient_attention(rescale_logits=False).
    # rescale_logits=True  → MEA=False behaviour: Q*√d is passed but divided
    #   by √d internally, so they cancel (net scale = 1.0). Matches Flax
    #   nn.dot_product_attention.
    self.rescale_logits = rescale_logits

    # Q, K, V projections: Linear(in, heads*hd)
    # We'll reshape the output to (b, n, heads, hd)
    self.query_proj = nn.Linear(in_features, in_features, bias=use_bias)
    self.key_proj = nn.Linear(in_features, in_features, bias=use_bias)
    self.value_proj = nn.Linear(in_features, in_features, bias=use_bias)

    # Output projection
    self.out_proj = nn.Linear(in_features, in_features, bias=use_bias)

    # QK normalization
    if qk_norm == "rms":
      self.query_ln = nn.RMSNorm(self.head_dim)
      self.key_ln = nn.RMSNorm(self.head_dim)
    else:
      self.query_ln = None
      self.key_ln = None

    # V normalization
    if v_norm == "rms":
      self.value_ln = nn.RMSNorm(self.head_dim, elementwise_affine=False)
    else:
      self.value_ln = None

    # RoPE
    if use_rotary_position_embeddings:
      self.rotary_position_embedding = RotaryPositionalEmbedding(
        embedding_dims=self.head_dim
      )
    else:
      self.rotary_position_embedding = None

    # PerDimScale
    if use_per_dim_scale:
      self.per_dim_scale = normalization.PerDimScale(num_dims=self.head_dim)
    else:
      self.per_dim_scale = None

  def forward(
    self,
    inputs_q: torch.Tensor,
    *,
    segment_ids: torch.Tensor | None = None,
    segment_pos: torch.Tensor | None = None,
    decode_cache: util.DecodeCache | None = None,
    patch_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, util.DecodeCache | None, torch.Tensor]:
    """Applies multi-head attention.

    Args:
      inputs_q: Shape (b, n, d).
      segment_ids: Shape (b, n). Optional segment IDs for masking.
      segment_pos: Shape (b, n). Optional positions for RoPE.
      decode_cache: Optional KV cache for autoregressive decoding.
      patch_mask: Shape (b, n). True = masked patch.

    Returns:
      Tuple of (output, updated_cache, attn_mask).
      - output: Shape (b, n, d).
      - updated_cache: Updated DecodeCache or None.
      - attn_mask: The attention mask used.
    """
    batch_size, n_patches, _ = inputs_q.shape
    device = inputs_q.device

    if patch_mask is None:
      patch_mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)

    # Project Q, K, V and reshape to (b, n, h, hd)
    query = self.query_proj(inputs_q).view(
      batch_size, n_patches, self.num_heads, self.head_dim
    )
    key = self.key_proj(inputs_q).view(
      batch_size, n_patches, self.num_heads, self.head_dim
    )
    value = self.value_proj(inputs_q).view(
      batch_size, n_patches, self.num_heads, self.head_dim
    )

    if decode_cache is None:
      num_front_masked = torch.sum(torch.cumprod(patch_mask.int(), dim=-1), dim=-1)
      next_index = torch.zeros_like(num_front_masked, dtype=torch.int32)
    else:
      num_front_masked = decode_cache.num_front_masked
      next_index = decode_cache.next_index

    # Apply RoPE
    if self.rotary_position_embedding is not None:
      if segment_pos is None:
        position = torch.arange(n_patches, device=device, dtype=torch.int32).unsqueeze(
          0
        ) + next_index.unsqueeze(-1)
      else:
        position = segment_pos
      query = self.rotary_position_embedding(query, position)
      key = self.rotary_position_embedding(key, position)

    # QK normalization
    if self.query_ln is not None:
      query = self.query_ln(query)
    if self.key_ln is not None:
      key = self.key_ln(key)

    # PerDimScale
    if self.per_dim_scale is not None:
      query = self.per_dim_scale(query)

    # V normalization
    if self.value_ln is not None:
      value = self.value_ln(value)

    if decode_cache is not None:
      # Cached decoding: update cache with new K, V
      cache_size = decode_cache.key.shape[1]
      if torch.all(next_index == next_index[0]):
        idx = next_index[0].item()
        decode_cache.key[:, idx : idx + n_patches, :, :] = key
        decode_cache.value[:, idx : idx + n_patches, :, :] = value
      else:
        for b_idx in range(batch_size):
          idx_b = next_index[b_idx].item()
          decode_cache.key[b_idx, idx_b : idx_b + n_patches, :, :] = key[b_idx]
          decode_cache.value[b_idx, idx_b : idx_b + n_patches, :, :] = value[b_idx]
      key = decode_cache.key
      value = decode_cache.value

      decode_cache = util.DecodeCache(
        next_index=next_index + n_patches,
        num_front_masked=num_front_masked,
        key=key,
        value=value,
      )

      attn_mask = make_attn_mask(
        query_length=n_patches,
        num_all_masked_kv=num_front_masked,
        query_index_offset=next_index,
        kv_length=cache_size,
        causal=self.causal_attention,
      )
    else:
      # Training / full-sequence mode
      attn_mask = make_attn_mask(
        query_length=n_patches,
        num_all_masked_kv=torch.zeros_like(num_front_masked),
        causal=self.causal_attention,
      )
      # Apply patch_mask to K/V positions
      attn_mask = attn_mask & (~patch_mask[:, None, None, :])
      if segment_ids is not None:
        segment_mask = make_segment_mask(segment_ids)
        attn_mask = attn_mask & segment_mask

    # Transpose for attention: (b, n, h, d) -> (b, h, n, d)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if self.use_sdpa:
      # --- F.scaled_dot_product_attention path (PyTorch >= 2.1) ---
      # SDPA computes: softmax(Q @ K^T * scale) @ V.
      if self.rescale_logits:
        # MEA=False equivalent: Flax passes Q*√d to nn.dot_product_attention
        # which divides by √d internally → net scale = 1.0.
        attn_scale = 1.0
      else:
        # MEA=True equivalent: Flax passes Q*√d with rescale_logits=False
        # → no internal division → net scale = √d.
        attn_scale = math.sqrt(self.head_dim)
      x = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.expand(-1, self.num_heads, -1, -1),
        scale=attn_scale,
      )
    else:
      # --- Manual attention path ---
      # Convert mask: True=attend -> 0.0, False=mask -> -1e9 (matching Flax
      # MEA bias).
      float_mask = torch.where(
        attn_mask.expand(-1, self.num_heads, -1, -1),
        torch.tensor(0.0, device=device),
        torch.tensor(-1e9, device=device),
      )
      if self.rescale_logits:
        # MEA=False equivalent: Q*√d then divide by √d → net scale = 1.0.
        # Flax nn.dot_product_attention receives Q*√d and divides by √d
        # internally, so the pre-multiplication and rescaling cancel out.
        query = query * math.sqrt(self.head_dim)
        attn_logits = (
          torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
          + float_mask
        )
      else:
        # MEA=True equivalent: Q*√d, no internal division → net scale = √d.
        # Matches Flax memory_efficient_attention(rescale_logits=False).
        query = query * math.sqrt(self.head_dim)
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) + float_mask
      attn_weights = F.softmax(attn_logits, dim=-1)
      x = torch.matmul(attn_weights, value)

    # Transpose back: (b, h, n, d) -> (b, n, h, d)
    x = x.transpose(1, 2).contiguous()

    # Reshape and project: (b, n, h, d) -> (b, n, h*d)
    x = x.view(batch_size, n_patches, self.in_features)

    out = self.out_proj(x)
    return out, decode_cache, attn_mask


class MixingTransformer(nn.Module):
  """Transformer with sequential sequence and variate attention.

  Attention is applied first across the sequence dimension 'n', then
  across the variate dimension 'v' for inputs of shape 'b v n d'.

  Architecture per layer:
    1. Sequence attention: pre_ln -> reshape(bv,n,d) -> MHA -> post_ln +
    residual
    2. Variate attention: pre_ln -> reshape(bn,v,d) -> MHA -> post_ln + residual
    3. FFN: pre_ln -> ff0 -> activation -> ff1 -> post_ln + residual
  """

  def __init__(
    self,
    config: configs.TransformerConfig,
    use_variate_attention: bool = True,
  ):
    super().__init__()
    self.config = config
    self.use_variate_attention = use_variate_attention

    # Sequence attention norms + module
    self.pre_seq_attn_ln = nn.RMSNorm(config.model_dims)
    self.post_seq_attn_ln = nn.RMSNorm(config.model_dims)
    # rescale_logits mirrors Flax: use_memory_efficient_attention=True →
    #   MEA=True (rescale_logits=False, scale=√d);
    #   use_memory_efficient_attention=False → MEA=False (rescale_logits=True,
    #   net scale=1.0).
    rescale_logits = not getattr(config, "use_memory_efficient_attention", False)
    self.seq_attn = MultiHeadAttention(
      num_heads=config.num_heads,
      in_features=config.model_dims,
      use_per_dim_scale=True,
      use_rotary_position_embeddings=config.use_rope_seq,
      qk_norm=config.qk_norm,
      v_norm=getattr(config, "v_norm", "none"),
      causal_attention=getattr(config, "causal_attention", True),
      use_bias=config.use_bias,
      use_sdpa=config.use_sdpa,
      rescale_logits=rescale_logits,
    )

    # Variate attention norms + module
    if use_variate_attention:
      self.pre_var_attn_ln = nn.RMSNorm(config.model_dims)
      self.post_var_attn_ln = nn.RMSNorm(config.model_dims)
      self.var_attn = MultiHeadAttention(
        num_heads=config.num_heads,
        in_features=config.model_dims,
        use_per_dim_scale=True,
        use_rotary_position_embeddings=config.use_rope_var,
        qk_norm=config.qk_norm,
        v_norm=getattr(config, "v_norm", "none"),
        causal_attention=False,
        use_bias=config.use_bias,
        use_sdpa=config.use_sdpa,
        rescale_logits=rescale_logits,
      )

    # FFN norms + layers
    self.pre_ff_ln = nn.RMSNorm(config.model_dims)
    self.post_ff_ln = nn.RMSNorm(config.model_dims)
    self.ff0 = nn.Linear(config.model_dims, config.hidden_dims, bias=config.use_bias)
    self.ff1 = nn.Linear(config.hidden_dims, config.model_dims, bias=config.use_bias)
    self.activation = util.get_activation_fn(config.ff_activation)

  def forward(
    self,
    input_embeddings: torch.Tensor,
    patch_mask: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    segment_pos: torch.Tensor | None = None,
    decode_cache: util.DecodeCache | None = None,
    var_segment_pos: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, util.DecodeCache | None, torch.Tensor]:
    """Forward pass.

    Args:
      input_embeddings: Shape (b, v, n, d).
      patch_mask: Shape (b, v, n). True = masked.
      segment_ids: Shape (b, n). Optional.
      segment_pos: Shape (b, n). Optional.
      decode_cache: Optional KV cache.
      var_segment_pos: Shape (b*n, v). Optional variate positions.

    Returns:
      (output_embeddings, updated_cache, seq_attn_mask).
    """
    b, v, n, d = input_embeddings.shape

    # --- Sequence Attention ---
    seq_attn_in = self.pre_seq_attn_ln(input_embeddings)
    # (b, v, n, d) -> (b*v, n, d)
    seq_attn_in_flat = seq_attn_in.reshape(b * v, n, d)
    patch_mask_flat = patch_mask.reshape(b * v, n)

    # Broadcast segment_ids/pos across variates
    seq_seg_ids_flat = None
    if segment_ids is not None:
      seg_ids_bvn = segment_ids.unsqueeze(1).expand(b, v, n)
      seq_seg_ids_flat = seg_ids_bvn.reshape(b * v, n)

    seq_seg_pos_flat = None
    if segment_pos is not None:
      seg_pos_bvn = segment_pos.unsqueeze(1).expand(b, v, n)
      seq_seg_pos_flat = seg_pos_bvn.reshape(b * v, n)

    seq_attn_out_flat, decode_cache, seq_attn_mask = self.seq_attn(
      seq_attn_in_flat,
      segment_ids=seq_seg_ids_flat,
      segment_pos=seq_seg_pos_flat,
      decode_cache=decode_cache,
      patch_mask=patch_mask_flat,
    )
    seq_attn_out = seq_attn_out_flat.view(b, v, n, d)
    h1 = self.post_seq_attn_ln(seq_attn_out) + input_embeddings

    # --- Variate Attention ---
    if self.use_variate_attention:
      var_attn_in = self.pre_var_attn_ln(h1)
      # (b, v, n, d) -> (b*n, v, d)
      var_attn_in_flat = var_attn_in.permute(0, 2, 1, 3).reshape(b * n, v, d)
      # Mask: (b, v, n) -> (b, n, v) -> (b*n, v)
      var_patch_mask = patch_mask.permute(0, 2, 1).reshape(b * n, v)

      var_attn_out_flat, _, _ = self.var_attn(
        var_attn_in_flat,
        segment_pos=var_segment_pos,
        decode_cache=None,
        patch_mask=var_patch_mask,
      )
      # (b*n, v, d) -> (b, n, v, d) -> (b, v, n, d)
      var_attn_out = var_attn_out_flat.view(b, n, v, d).permute(0, 2, 1, 3)
      h2 = self.post_var_attn_ln(var_attn_out) + h1
    else:
      h2 = h1

    # --- FeedForward ---
    ff_out = self.ff1(self.activation(self.ff0(self.pre_ff_ln(h2))))
    output_embeddings = self.post_ff_ln(ff_out) + h2

    return output_embeddings, decode_cache, seq_attn_mask


class StackedMixingTransformer(nn.Module):
  """Stacked MixingTransformer layers."""

  def __init__(
    self,
    config: configs.StackedTransformersConfig,
    use_variate_attention: bool = True,
  ):
    super().__init__()
    self.config = config
    self.layers = nn.ModuleList(
      [
        MixingTransformer(
          config=config.transformer,
          use_variate_attention=use_variate_attention,
        )
        for _ in range(config.num_layers)
      ]
    )

  def forward(
    self,
    input_embeddings: torch.Tensor,
    patch_mask: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    segment_pos: torch.Tensor | None = None,
    decode_cache: list[util.DecodeCache] | None = None,
    var_segment_pos: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, list[util.DecodeCache] | None, list[torch.Tensor]]:
    """Forward pass through all layers.

    Args:
      input_embeddings: Shape (b, v, n, d).
      patch_mask: Shape (b, v, n).
      segment_ids: Shape (b, n). Optional.
      segment_pos: Shape (b, n). Optional.
      decode_cache: List of DecodeCache (one per layer), or None.
      var_segment_pos: Shape (b*n, v). Optional.

    Returns:
      (output_embeddings, updated_caches, attn_masks).
    """
    if decode_cache is None:
      decode_cache = [None] * len(self.layers)  # pyrefly: ignore[bad-assignment]

    output = input_embeddings
    new_caches = []
    attn_masks = []

    for i, layer in enumerate(self.layers):
      output, layer_cache, layer_mask = layer(
        output,
        patch_mask,
        segment_ids,
        segment_pos,
        decode_cache[i],  # pyrefly: ignore[unsupported-operation]
        var_segment_pos,
      )
      new_caches.append(layer_cache)
      attn_masks.append(layer_mask)

    return output, new_caches, attn_masks

# --- vendored from src/timesfm3/torch/cpm_revin_refine.py ---
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone iterative RevIN refinement for CPM-masked patches in PyTorch.

Extracted so it can be tested independently of the TimesFM3 model.
"""





def cpm_iterative_revin_refine(
  raw_logits: torch.Tensor,
  revin_n: torch.Tensor,
  revin_mu: torch.Tensor,
  revin_sigma: torch.Tensor,
  patch_cpm_mask: torch.Tensor,
  median_q_idx: int,
  rolls: int,
  patch_len: int,
  num_quantiles: int,
  value_clip: float = 1e9,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Refines RevIN stats at CPM-masked patches via iterative estimation.

  For each CPM-masked position p the currently frozen stats (from the last
  observed patch before the CPM region) are replaced with stats that also
  incorporate model-estimated values for all CPM patches that precede p.

  Args:
    raw_logits: Output of the output head, shape (b, v, n, output_patch_len *
      num_quantiles). In RevIN-normalised space, before reverse RevIN.
    revin_n: Count of valid (unmasked) values accumulated at each patch
      position, shape (b, v, n).
    revin_mu: Running mean per position, (b, v, n).
    revin_sigma: Running std per position, (b, v, n).
    patch_cpm_mask: Boolean mask, True = CPM-masked patch, shape (b, n).
    median_q_idx: Index into quantiles selecting the median quantile used as
      point estimate (typically num_quantiles // 2).
    rolls: Number of output patches per input patch (output_patch_len //
      patch_len).
    patch_len: Length of each input patch.
    num_quantiles: Total number of quantile heads.
    value_clip: Absolute bound for clamping estimated values after reverse
      RevIN.

  Returns:
    Tuple (refined_mu, refined_sigma), each shape (b, v, n).
    Non-CPM positions are identical to revin_mu / revin_sigma.
    CPM positions incorporate estimates of all preceding CPM patches in
    the same block (and all estimates from earlier blocks in the segment).
  """
  b, v, n_patches, _ = raw_logits.shape
  device = raw_logits.device

  # Reshape and slice raw_logits to keep only the median quantile.
  # (b, v, n, oq) -> (b, v, n, rolls, patch_len, num_quantiles)
  # -> (b, v, n, rolls, patch_len)
  median_logits = raw_logits.reshape(b, v, n_patches, rolls, patch_len, num_quantiles)[
    :, :, :, :, :, median_q_idx
  ]

  # Initialise carry with zeros.
  carry_n = torch.zeros((b, v), dtype=torch.float32, device=device)
  carry_mu = torch.zeros((b, v), dtype=torch.float32, device=device)
  carry_sigma = torch.zeros((b, v), dtype=torch.float32, device=device)
  anchor_predicted_values = torch.zeros(
    (b, v, rolls, patch_len), dtype=torch.float32, device=device
  )
  block_offset = torch.zeros((b,), dtype=torch.long, device=device)

  refined_mu_list = []
  refined_sigma_list = []

  step_masks = torch.zeros((b, v, patch_len), dtype=torch.bool, device=device)

  for i in range(n_patches):
    actual_n = revin_n[:, :, i]
    actual_mu = revin_mu[:, :, i]
    actual_sigma = revin_sigma[:, :, i]
    current_step_logits = median_logits[:, :, i]
    is_cpm = patch_cpm_mask[:, i : i + 1]  # (b, 1)

    # Select the block_offset[b]-th patch for each batch element
    offset_onehot = torch.eq(
      torch.arange(rolls, device=device).unsqueeze(0),
      block_offset.unsqueeze(1),
    ).float()
    predicted_values_step = torch.einsum(
      "br,bvrp->bvp", offset_onehot, anchor_predicted_values
    )

    # Update running stats with the estimated patch.
    new_n, new_mu, new_sigma = util.update_running_stats(
      carry_n, carry_mu, carry_sigma, predicted_values_step, step_masks
    )

    out_n = torch.where(is_cpm, new_n, actual_n)
    out_mu = torch.where(is_cpm, new_mu, actual_mu)
    out_sigma = torch.where(is_cpm, new_sigma, actual_sigma)

    # Advance block_offset: +1 (mod rolls) for CPM, reset to 0 for non-CPM.
    new_block_offset = torch.where(
      is_cpm.squeeze(-1),
      (block_offset + 1) % rolls,
      torch.zeros_like(block_offset),
    )

    should_update_anchor = torch.eq(new_block_offset, 0)

    # Pre-calculate predicted values for the new anchor.
    step_predicted_values = util.revin(
      current_step_logits, out_mu, out_sigma, reverse=True
    )
    step_predicted_values = torch.clamp(step_predicted_values, -value_clip, value_clip)

    new_anchor_predicted_values = torch.where(
      should_update_anchor.view(b, 1, 1, 1),
      step_predicted_values,
      anchor_predicted_values,
    )

    carry_n = out_n
    carry_mu = out_mu
    carry_sigma = out_sigma
    anchor_predicted_values = new_anchor_predicted_values
    block_offset = new_block_offset

    refined_mu_list.append(out_mu)
    refined_sigma_list.append(out_sigma)

  refined_mu = torch.stack(refined_mu_list, dim=2)
  refined_sigma = torch.stack(refined_sigma_list, dim=2)
  return refined_mu, refined_sigma

# --- vendored from src/timesfm3/torch/model.py ---
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TimesFM3 PyTorch model (inference only).

Supports:
  - forward(): equivalent to Flax __call__, full-sequence forward pass.
  - decode(): non-autoregressive and cached decoding with frozen stats and
    configurable output_patch_len.
"""


import math
from typing import Any




class TimesFM3Torch(nn.Module):
  """PyTorch inference-only implementation of TimesFM3.

  Attributes:
    input_patch_len: The length of each input patch.
    output_patch_len: The length of the output forecast for each patch.
    quantiles: A list of quantiles to predict.
    residual_block_config: Configuration for the pre-transformer residual block.
    transformer_config: Configuration for the stacked transformers.
    use_variate_attention: Whether to use variate attention.
    value_clip: Absolute value to clip input values to.
    use_stitching: Whether to use stitching for predictions.
    use_linear_detrending: Whether to apply linear detrending on context.
    linear_detrending_threshold: Ratio threshold for applying linear detrending.
    use_iterative_cpm_revin: Whether to use iterative RevIN refinement.
    use_frozen_running_stats: Whether running stats freeze at context boundary.
  """

  def __init__(
    self,
    input_patch_len: int = 32,
    output_patch_len: int = 64,
    quantiles: list[float] | None = None,
    residual_block_config: configs.ResidualBlockConfig | dict[str, Any] | None = None,
    transformer_config: configs.StackedTransformersConfig
    | dict[str, Any]
    | None = None,
    use_variate_attention: bool = True,
    value_clip: float = 1e20,
    use_stitching: bool = True,
    use_linear_detrending: bool = True,
    linear_detrending_threshold: float = 0.5,
    use_iterative_cpm_revin: bool = True,
    use_frozen_running_stats: bool = False,
    input_transform: str = "identity",
  ):
    super().__init__()
    if quantiles is None:
      quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if residual_block_config is None:
      residual_block_config = configs.ResidualBlockConfig(
        hidden_dims=1280,
        output_dims=1280,
        use_bias=False,
        activation="relu",
      )
    elif isinstance(residual_block_config, dict):
      residual_block_config = configs.ResidualBlockConfig(**residual_block_config)

    if transformer_config is None:
      transformer_config = configs.StackedTransformersConfig(
        num_layers=20,
        transformer=configs.TransformerConfig(
          model_dims=1280,
          hidden_dims=1280,
          num_heads=16,
          attention_norm="rms",
          feedforward_norm="rms",
          qk_norm="rms",
          use_rope_seq=True,
          use_rope_var=False,
          use_bias=False,
          ff_activation="relu",
          deterministic=True,
        ),
      )
    elif isinstance(transformer_config, dict):
      t_data = transformer_config.get("transformer", {})
      if isinstance(t_data, dict):
        t_cfg = configs.TransformerConfig(**t_data)
      else:
        t_cfg = t_data
      tc_dict = dict(transformer_config)
      tc_dict["transformer"] = t_cfg
      transformer_config = configs.StackedTransformersConfig(**tc_dict)

    if output_patch_len % input_patch_len != 0:
      raise ValueError(
        f"Output patch len {output_patch_len} must be a multiple of"
        f" input patch len {input_patch_len}."
      )
    if residual_block_config.output_dims != transformer_config.transformer.model_dims:
      raise ValueError("ResidualBlock output_dims must match Transformer model_dims.")

    self.input_patch_len = input_patch_len
    self.output_patch_len = output_patch_len
    self.quantiles = quantiles
    self.num_quantiles = len(quantiles)
    self.rolls = output_patch_len // input_patch_len
    self.residual_block_config = residual_block_config
    self.transformer_config = transformer_config
    self.use_variate_attention = use_variate_attention
    self.value_clip = value_clip
    self.use_stitching = use_stitching
    self.use_linear_detrending = use_linear_detrending
    self.linear_detrending_threshold = linear_detrending_threshold
    self.use_iterative_cpm_revin = use_iterative_cpm_revin
    self.use_frozen_running_stats = use_frozen_running_stats
    self.input_transform = input_transform

    if self.use_stitching:
      if self.output_patch_len <= self.input_patch_len:
        raise ValueError("use_stitching requires output_patch_len > input_patch_len")
      self._stitching_extract_len = min(2 * self.input_patch_len, self.output_patch_len)

    self.pre_transformer_resblock = dense.ResidualBlock(config=residual_block_config)
    self.pre_transformer_resblock.set_input_dims(
      2 * (input_patch_len + output_patch_len)
    )
    self.transformer_stack = transformer.StackedMixingTransformer(
      config=transformer_config,
      use_variate_attention=use_variate_attention,
    )

    self.output_head = nn.Linear(
      transformer_config.transformer.model_dims,
      output_patch_len * self.num_quantiles,
      bias=True,
    )

  def to_dict(self) -> dict[str, Any]:
    """Returns a serializable dictionary of model configuration."""
    return {
      "input_patch_len": self.input_patch_len,
      "output_patch_len": self.output_patch_len,
      "quantiles": list(self.quantiles),
      "residual_block_config": dataclasses.asdict(self.residual_block_config),
      "transformer_config": dataclasses.asdict(self.transformer_config),
      "use_variate_attention": self.use_variate_attention,
      "value_clip": self.value_clip,
      "use_stitching": self.use_stitching,
      "use_linear_detrending": self.use_linear_detrending,
      "linear_detrending_threshold": self.linear_detrending_threshold,
      "use_iterative_cpm_revin": self.use_iterative_cpm_revin,
      "use_frozen_running_stats": self.use_frozen_running_stats,
      "input_transform": self.input_transform,
    }

  def save_pretrained(
    self,
    save_directory: str | Any,
    *,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
  ) -> str | None:
    """Saves model weights and config to a local directory or pushes to Hub."""
    if config is None:
      config = self.to_dict()
    return super().save_pretrained(save_directory, config=config, **kwargs)

  def _preprocess(
    self,
    values: torch.Tensor,
    masks: torch.Tensor,
    patch_is_target: torch.Tensor,
    freeze_after: int | None = None,
    patch_cpm_mask: torch.Tensor | None = None,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
  ]:
    """Applies preprocessing: RevIN, masking, future covariates, ResBlock.

    Args:
      values: (b, v, n, p).
      masks: (b, v, n, p) bool. True=masked.
      patch_is_target: (b, v, n) bool.
      freeze_after: Optional patch index after which running stats freeze.
      patch_cpm_mask: (b, n) bool or None. If given, target variates at True
        positions are additionally masked (used for horizon CPM masking).

    Returns:
      (resblock_input, resblock_output, patch_mask, (running_mean, running_std),
      running_n)
    """
    running_n, running_mean, running_std = util.get_running_stats(values, masks)
    if freeze_after is not None:
      _, _, n, _ = values.shape
      if 0 <= freeze_after < n - 1:
        running_mean[:, :, freeze_after + 1 :] = running_mean[
          :, :, freeze_after : freeze_after + 1
        ]
        running_std[:, :, freeze_after + 1 :] = running_std[
          :, :, freeze_after : freeze_after + 1
        ]

    # Apply CPM mask: mask target variates at CPM positions.
    if patch_cpm_mask is not None:
      cpm_bvnp = patch_cpm_mask[:, None, :, None]  # (b, 1, n, 1)
      cpm_target_only = cpm_bvnp & patch_is_target.unsqueeze(-1)
      masks = masks | cpm_target_only

    values_bvnp = util.revin(values, running_mean, running_std, reverse=False)
    values_bvnp = torch.where(masks, 0.0, values_bvnp)

    # Roll values to get future covariate patches
    values_fcov, wrap_mask = util.get_output_patch_via_roll(values, self.rolls)
    values_fcov = util.revin(values_fcov, running_mean, running_std, reverse=False)

    # Roll the (CPM-modified) masks for future covariate masking.
    masks_fcov_raw, _ = util.get_output_patch_via_roll(masks, self.rolls)
    masks_fcov = masks_fcov_raw | patch_is_target.unsqueeze(-1) | wrap_mask
    values_fcov = torch.where(masks_fcov, 0.0, values_fcov)

    values_cat = torch.cat([values_bvnp, values_fcov], dim=-1)
    masks_cat = torch.cat([masks, masks_fcov], dim=-1)

    resblock_input = torch.cat([values_cat, masks_cat.float()], dim=-1)
    resblock_output = self.pre_transformer_resblock(resblock_input)

    # Patch mask: a patch is fully masked if ALL points are masked
    patch_mask_bvn = masks_cat.all(dim=3)

    return (
      resblock_input,
      resblock_output,
      patch_mask_bvn,
      (running_mean, running_std),
      running_n,
    )

  def forward(
    self,
    inputs: dict[str, Any],
    freeze_after: int | None = None,
    patch_cpm_mask: torch.Tensor | None = None,
    return_aux_outputs: bool = False,
  ) -> dict[str, Any]:
    """Full-sequence forward pass (equivalent to Flax __call__).

    Args:
      inputs: Dictionary with keys: - "values": (b, v, n, p) float - "masks":
        (b, v, n, p) bool - "patch_is_target": (b, v, n) bool
      freeze_after: Optional patch index after which running stats freeze.
      patch_cpm_mask: (b, n) bool or None. Horizon CPM mask.
      return_aux_outputs: Whether to return auxiliary outputs.

    Returns:
      Dictionary with "logits" of shape (b, v, n, output_patch_len,
      num_quantiles).
    """
    values = inputs["values"]
    values = torch.nan_to_num(values, nan=0.0)
    values = torch.clamp(values, -self.value_clip, self.value_clip)
    masks = inputs["masks"].bool()
    patch_is_target = inputs["patch_is_target"]

    _, _, _, p = values.shape
    if p != self.input_patch_len:
      raise ValueError(
        f"Input patch_len {p} != model input_patch_len {self.input_patch_len}"
      )

    # Preprocessing & ResBlock
    (
      resblock_input,
      transformer_input,
      transformer_patch_mask,
      revin_stats,
      running_n,
    ) = self._preprocess(
      values,
      masks,
      patch_is_target,
      freeze_after=freeze_after,
      patch_cpm_mask=patch_cpm_mask,
    )

    # Transformer
    # At inference, only mask *leading* fully-masked patches (left-padding).
    # Flax __call__ does: effective_patch_mask = cumprod(mask, axis=2) when
    # not training.  This keeps horizon patches (which are fully masked but
    # come after valid context) visible to attention.
    effective_patch_mask = torch.cumprod(transformer_patch_mask.int(), dim=2).bool()
    transformer_output, _, seq_attn_mask = self.transformer_stack(
      transformer_input,
      effective_patch_mask,
    )

    # Output head
    raw_logits = self.output_head(transformer_output)
    revin_mean, revin_std = revin_stats

    if self.use_iterative_cpm_revin and patch_cpm_mask is not None:
      refined_mu, refined_sigma = cpm_revin_refine_lib.cpm_iterative_revin_refine(
        raw_logits,
        revin_n=running_n,
        revin_mu=revin_mean,
        revin_sigma=revin_std,
        patch_cpm_mask=patch_cpm_mask,
        median_q_idx=self.num_quantiles // 2,
        rolls=self.rolls,
        patch_len=self.input_patch_len,
        num_quantiles=self.num_quantiles,
        value_clip=self.value_clip,
      )
      cpm_bvn = patch_cpm_mask.unsqueeze(1)  # (b, 1, n)
      revin_mean = torch.where(cpm_bvn, refined_mu, revin_mean)
      revin_std = torch.where(cpm_bvn, refined_sigma, revin_std)

    revin_logits = util.revin(raw_logits, revin_mean, revin_std, reverse=True)
    clipped_logits = torch.clamp(revin_logits, -self.value_clip, self.value_clip)

    # Reshape: (b, v, n, o*q) -> (b, v, n, o, q)
    b, v, n_patches = clipped_logits.shape[:3]
    final_logits = clipped_logits.view(
      b, v, n_patches, self.output_patch_len, self.num_quantiles
    )

    outputs = {"logits": final_logits, "revin_stats": revin_stats}

    if return_aux_outputs:
      outputs["__call__:resblock_input"] = resblock_input
      outputs["__call__:transformer_input"] = transformer_input
      outputs["__call__:seq_attn_mask"] = seq_attn_mask
      outputs["__call__:transformer_output"] = transformer_output

    return outputs

  @torch.no_grad()
  def decode(
    self,
    target: torch.Tensor,
    horizon: int = 0,
    past_only_covariates: torch.Tensor | None = None,
    past_future_covariates: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    past_only_mask: torch.Tensor | None = None,
    past_future_mask: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    return_aux_outputs: bool = False,
  ) -> Any:
    """Non-autoregressive single-pass decoding for TimesFM3.

    Args:
      target: (b, u, context_len).
      horizon: Forecast horizon. Inferred if past_future_covariates given.
      past_only_covariates: (b, v_po, context_len) or None.
      past_future_covariates: (b, w, context_len+horizon) or None.
      target_mask: (b, u, context_len) bool or None.
      past_only_mask: (b, v_po, context_len) bool or None.
      past_future_mask: (b, w, context_len+horizon) bool or None.
      mask: (b, context_len) global bool mask or None.
      return_aux_outputs: If True, return (logits, aux_dict).

    Returns:
      Logits of shape (b, num_variates, horizon, num_quantiles).
    """
    device = target.device
    batch_size, num_target, context = target.shape

    if past_future_covariates is not None:
      horizon = past_future_covariates.shape[-1] - context
    if horizon <= 0:
      raise ValueError("Decode function requires horizon > 0.")

    # 1. Pad context to multiple of input_patch_len
    ctx_padding = (
      self.input_patch_len - (context % self.input_patch_len)
    ) % self.input_patch_len
    if ctx_padding > 0:
      target = torch.nn.functional.pad(target, (ctx_padding, 0))
      if mask is not None:
        mask = torch.nn.functional.pad(mask, (ctx_padding, 0), value=True)
      if past_only_covariates is not None:
        past_only_covariates = torch.nn.functional.pad(
          past_only_covariates, (ctx_padding, 0)
        )
      if past_future_covariates is not None:
        past_future_covariates = torch.nn.functional.pad(
          past_future_covariates, (ctx_padding, 0)
        )
      if target_mask is not None:
        target_mask = torch.nn.functional.pad(target_mask, (ctx_padding, 0), value=True)
      if past_only_mask is not None:
        past_only_mask = torch.nn.functional.pad(
          past_only_mask, (ctx_padding, 0), value=True
        )
      if past_future_mask is not None:
        past_future_mask = torch.nn.functional.pad(
          past_future_mask, (ctx_padding, 0), value=True
        )
      context = context + ctx_padding

    if mask is None:
      mask = torch.zeros(batch_size, context, dtype=torch.bool, device=device)
      if ctx_padding > 0:
        mask[:, :ctx_padding] = True

    # 2. Pad horizon
    if self.use_stitching:
      extract_len = self._stitching_extract_len
      overlap = extract_len - self.input_patch_len
      num_forecast_patches = max(
        math.ceil((horizon - overlap) / self.input_patch_len), 1
      )
      num_horizon_patches = num_forecast_patches + self.rolls - 1
      padded_horizon = num_horizon_patches * self.input_patch_len
      hor_padding = padded_horizon - horizon
    else:
      hor_padding = (-horizon) % self.output_patch_len
      padded_horizon = horizon + hor_padding
      num_horizon_patches = padded_horizon // self.input_patch_len
    num_context_patches = context // self.input_patch_len

    # 3. Build context & horizon inputs
    if target_mask is None:
      target_mask = torch.zeros_like(target, dtype=torch.bool)
    target_mask = target_mask | mask.unsqueeze(1)

    all_ctx_vals = [target]
    all_ctx_masks = [target_mask]
    num_past_only = 0
    if past_only_covariates is not None:
      num_past_only = past_only_covariates.shape[1]
      if past_only_mask is None:
        past_only_mask = torch.zeros_like(past_only_covariates, dtype=torch.bool)
      all_ctx_vals.append(past_only_covariates)
      all_ctx_masks.append(past_only_mask | mask.unsqueeze(1))
    if past_future_covariates is not None:
      if past_future_mask is None:
        past_future_mask = torch.zeros_like(past_future_covariates, dtype=torch.bool)
      all_ctx_vals.append(past_future_covariates[..., :context])
      all_ctx_masks.append(past_future_mask[..., :context] | mask.unsqueeze(1))

    ctx_vals = torch.cat(all_ctx_vals, dim=1)
    ctx_masks = torch.cat(all_ctx_masks, dim=1)

    if self.use_linear_detrending:
      t_ctx = torch.arange(-(context - 1), 1, dtype=torch.float32, device=device)
      t_ctx_bvc = t_ctx[None, None, :]
      t_ctx_bvc_normalized = t_ctx_bvc / context

      valid = ~ctx_masks
      n_v = valid.float().sum(dim=-1, keepdim=True)
      sum_t = torch.where(valid, t_ctx_bvc_normalized, 0.0).sum(dim=-1, keepdim=True)
      sum_t2 = torch.where(valid, t_ctx_bvc_normalized**2, 0.0).sum(
        dim=-1, keepdim=True
      )
      sum_y = torch.where(valid, ctx_vals, 0.0).sum(dim=-1, keepdim=True)
      sum_ty = torch.where(valid, t_ctx_bvc_normalized * ctx_vals, 0.0).sum(
        dim=-1, keepdim=True
      )

      det = n_v * sum_t2 - sum_t**2
      safe_det = torch.where(det == 0.0, 1.0, det)
      m_trend = torch.where(det == 0.0, 0.0, (n_v * sum_ty - sum_t * sum_y) / safe_det)
      c_trend = torch.where(
        det == 0.0,
        torch.where(n_v > 0, sum_y / torch.clamp_min(n_v, 1.0), 0.0),
        (sum_y - m_trend * sum_t) / torch.clamp_min(n_v, 1.0),
      )

      ctx_vals_detrended = ctx_vals - (m_trend * t_ctx_bvc_normalized + c_trend)

      mean_y = sum_y / torch.clamp_min(n_v, 1.0)
      sum_y2 = torch.where(valid, ctx_vals**2, 0.0).sum(dim=-1, keepdim=True)
      var_orig = torch.clamp_min(sum_y2 / torch.clamp_min(n_v, 1.0) - mean_y**2, 0.0)
      std_orig = torch.sqrt(var_orig)

      sum_yd = torch.where(valid, ctx_vals_detrended, 0.0).sum(dim=-1, keepdim=True)
      mean_yd = sum_yd / torch.clamp_min(n_v, 1.0)
      sum_yd2 = torch.where(valid, ctx_vals_detrended**2, 0.0).sum(dim=-1, keepdim=True)
      var_det = torch.clamp_min(sum_yd2 / torch.clamp_min(n_v, 1.0) - mean_yd**2, 0.0)
      std_det = torch.sqrt(var_det)

      apply_detrend = std_det < self.linear_detrending_threshold * std_orig
      ctx_vals = torch.where(apply_detrend, ctx_vals_detrended, ctx_vals)
    else:
      num_variates = ctx_vals.shape[1]
      m_trend = torch.zeros(
        (batch_size, num_variates, 1), dtype=torch.float32, device=device
      )
      c_trend = torch.zeros(
        (batch_size, num_variates, 1), dtype=torch.float32, device=device
      )
      apply_detrend = torch.zeros(
        (batch_size, num_variates, 1), dtype=torch.bool, device=device
      )

    ctx_vals = torch.where(ctx_masks, 0.0, ctx_vals)

    all_hor_vals = [
      torch.zeros(batch_size, num_target, padded_horizon, device=device),
      torch.zeros(batch_size, num_past_only, padded_horizon, device=device),
    ]
    all_hor_masks = [
      torch.ones(
        batch_size,
        num_target,
        padded_horizon,
        dtype=torch.bool,
        device=device,
      ),
      torch.ones(
        batch_size,
        num_past_only,
        padded_horizon,
        dtype=torch.bool,
        device=device,
      ),
    ]

    if past_future_covariates is not None:
      if past_future_mask is None:
        past_future_mask = torch.zeros_like(past_future_covariates, dtype=torch.bool)
      pf_future_vals = past_future_covariates[..., context : context + horizon]
      pf_future_masks = past_future_mask[..., context : context + horizon]
      if self.use_linear_detrending:
        m_pf = m_trend[:, num_target + num_past_only :, :]
        c_pf = c_trend[:, num_target + num_past_only :, :]
        apply_detrend_pf = apply_detrend[:, num_target + num_past_only :, :]
        t_hor_pf = torch.arange(1, horizon + 1, dtype=torch.float32, device=device)[
          None, None, :
        ]
        t_hor_pf_normalized = t_hor_pf / context
        pf_trend_hor = m_pf * t_hor_pf_normalized + c_pf
        pf_future_vals = torch.where(
          apply_detrend_pf, pf_future_vals - pf_trend_hor, pf_future_vals
        )
      pf_future_vals = torch.where(pf_future_masks, 0.0, pf_future_vals)
      if hor_padding > 0:
        pf_future_vals = torch.nn.functional.pad(pf_future_vals, (0, hor_padding))
        pf_future_masks = torch.nn.functional.pad(
          pf_future_masks, (0, hor_padding), value=True
        )
      all_hor_vals.append(pf_future_vals)
      all_hor_masks.append(pf_future_masks)

    hor_vals = torch.cat(all_hor_vals, dim=1)
    hor_masks = torch.cat(all_hor_masks, dim=1)

    all_vals = torch.cat([ctx_vals, hor_vals], dim=-1)
    all_masks = torch.cat([ctx_masks, hor_masks], dim=-1)

    num_variates = all_vals.shape[1]
    patch_is_target = torch.zeros(
      (batch_size, num_variates, num_context_patches + num_horizon_patches),
      dtype=torch.bool,
      device=device,
    )
    patch_is_target[:, : num_target + num_past_only, :] = True

    # Reshape values & masks to patched shape (b, v, n, p)
    values_bvnp = all_vals.reshape(batch_size, num_variates, -1, self.input_patch_len)
    masks_bvnp = all_masks.reshape(batch_size, num_variates, -1, self.input_patch_len)

    inputs = {
      "values": values_bvnp,
      "masks": masks_bvnp,
      "patch_is_target": patch_is_target,
    }

    # Build horizon CPM mask: context=False, horizon=True.
    num_total_patches = num_context_patches + num_horizon_patches
    horizon_cpm_mask = torch.zeros(
      batch_size, num_total_patches, dtype=torch.bool, device=device
    )
    horizon_cpm_mask[:, num_context_patches:] = True

    freeze_after = num_context_patches - 1 if self.use_frozen_running_stats else None
    forward_out = self.forward(
      inputs,
      freeze_after=freeze_after,
      patch_cpm_mask=horizon_cpm_mask,
      return_aux_outputs=return_aux_outputs,
    )
    logits = forward_out["logits"]  # (b, v, n, output_patch_len, num_quantiles)

    if self.use_stitching:
      extract_len = self._stitching_extract_len
      overlap = extract_len - self.input_patch_len
      num_forecast_patches = max(
        math.ceil((horizon - overlap) / self.input_patch_len), 1
      )
      forecast_indices = torch.arange(num_forecast_patches, device=device) + (
        num_context_patches - 1
      )
      patch_preds = logits[:, :, forecast_indices, :extract_len, :]
      horizon_logits = util.stitch_patches(
        patch_preds,
        self.input_patch_len,
      )[:, :, :horizon, :]
    else:
      num_forecast_chunks = padded_horizon // self.output_patch_len
      forecast_indices = torch.arange(
        num_forecast_chunks, device=device
      ) * self.rolls + (num_context_patches - 1)
      forecast_logits = logits[:, :, forecast_indices, :, :]
      horizon_logits = forecast_logits.reshape(
        batch_size, num_variates, -1, self.num_quantiles
      )[:, :, :horizon, :]

    if self.use_linear_detrending:
      t_forecast = torch.arange(1, horizon + 1, dtype=torch.float32, device=device)
      t_forecast_normalized = t_forecast / context
      trend_forecast = (
        m_trend[:, :, 0, None] * t_forecast_normalized[None, None, :]
        + c_trend[:, :, 0, None]
      )
      trend_forecast = torch.where(apply_detrend[:, :, 0, None], trend_forecast, 0.0)
      horizon_logits = horizon_logits + trend_forecast[:, :, :, None]

    if return_aux_outputs:
      return horizon_logits, forward_out
    return horizon_logits



# ---------------------------------------------------------------------------
# Fixture generation.
# ---------------------------------------------------------------------------

PINNED_MODEL_ID = "google/timesfm-3.0-pytorch"


def main() -> None:
  import argparse
  import glob

  import numpy as np
  from safetensors.torch import load_file

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default=PINNED_MODEL_ID)
  parser.add_argument("--output", required=True)
  parser.add_argument("--seed", type=int, default=20260918)
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--context", type=int, default=512)
  parser.add_argument("--horizon", type=int, default=96)
  args = parser.parse_args()

  import json

  snapshot = None
  for pattern in (
      glob.glob("/home/lhl/.cache/huggingface/hub/models--google--timesfm-3.0-pytorch/snapshots/*/"),
      [],
  ):
    if pattern:
      snapshot = Path(pattern[0])
      break
  if snapshot is None or not (snapshot / "model.safetensors").is_file():
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(args.model))

  config = json.load(open(snapshot / "config.json"))
  model = TimesFM3Torch(
      input_patch_len=config["input_patch_len"],
      output_patch_len=config["output_patch_len"],
      quantiles=config["quantiles"],
      residual_block_config=config["residual_block_config"],
      transformer_config=config["transformer_config"],
      use_variate_attention=config["use_variate_attention"],
      value_clip=config["value_clip"],
      use_stitching=config["use_stitching"],
      use_linear_detrending=config["use_linear_detrending"],
      linear_detrending_threshold=config["linear_detrending_threshold"],
      use_iterative_cpm_revin=config["use_iterative_cpm_revin"],
      use_frozen_running_stats=config["use_frozen_running_stats"],
      input_transform=config["input_transform"],
  )
  state = load_file(str(snapshot / "model.safetensors"))
  missing, unexpected = model.load_state_dict(state, strict=False)
  assert not unexpected, f"unexpected keys: {unexpected[:5]}"
  assert not missing, f"missing keys: {missing[:5]}"
  model.eval()

  device = torch.device("cpu")
  rng = np.random.default_rng(args.seed)
  batch, context, horizon = args.batch, args.context, args.horizon

  # Batch 0: two targets + one past-only + one past-future covariate
  # (multivariate mode).  Batch 1: a single target with front padding
  # (univariate mode + the left-pad mask path).
  t = np.arange(context, dtype=np.float64)
  series0 = np.sin(2 * np.pi * t / 48.0) * 3.0 + 0.5 * rng.standard_normal(context)
  series1 = np.cumsum(rng.standard_normal(context) * 0.25) + 10.0
  target = np.zeros((batch, 2, context), dtype=np.float32)
  target[0, 0] = series0
  target[0, 1] = series1
  pad = 32 * 2
  target[1, 0] = np.concatenate(
      [np.zeros(pad, dtype=np.float32), series1[: context - pad]]
  )
  target[1, 1] = target[1, 0]  # duplicate so batch 1 is also 2-target
  target_mask = np.zeros_like(target, dtype=bool)
  target_mask[1, :, :pad] = True

  # Past-only covariate: a lagged copy of series0 (batch 0 only meaningful).
  past_only = np.zeros((batch, 1, context), dtype=np.float32)
  past_only[0, 0] = np.concatenate(
      [np.zeros(4, dtype=np.float32), series0[:-4]]
  )

  # Past-future covariate: day-of-week style periodic signal known over the
  # horizon as well.
  dow = np.sin(2 * np.pi * np.arange(context + horizon) / 7.0)
  past_future = np.zeros((batch, 1, context + horizon), dtype=np.float32)
  past_future[0, 0] = dow.astype(np.float32)
  past_future[1, 0] = dow.astype(np.float32)

  with torch.no_grad():
    logits = model.decode(
        torch.from_numpy(target),
        horizon=horizon,
        past_only_covariates=torch.from_numpy(past_only),
        past_future_covariates=torch.from_numpy(past_future),
        target_mask=torch.from_numpy(target_mask),
    )

  payload = {
      "schema": np.asarray(1, dtype=np.int64),
      "seed": np.asarray(args.seed, dtype=np.int64),
      "target": target,
      "target_mask": target_mask,
      "past_only_covariates": past_only,
      "past_future_covariates": past_future,
      "horizon": np.asarray(horizon, dtype=np.int64),
      "decode_logits": logits.detach().cpu().numpy(),
  }
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **payload)
  print(
      f"wrote {output}: batch={batch} context={context} horizon={horizon} "
      f"logits={tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())}"
  )


if __name__ == "__main__":
  main()
