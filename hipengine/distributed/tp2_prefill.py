"""Rank-local batched/bulk TP2 prefill: geometry planning and reference math.

This module is CPU-only: it performs no device calls and changes no arithmetic.
It is the contract the GPU execution packets consume and the independent oracle
the batched layer math is checked against:

* :func:`plan_batched_prefill` derives the per-layer batched shapes, the MLP
  shard dimensions, the Conv/GDN final-state shapes, the full-attention KV
  geometry, the chunk boundaries, and the exchange payload size from a model
  config. Nothing is hardcoded by model name.
* :func:`batched_sharded_mlp_reference` is the independent numpy reference for
  one batched, column/row-sharded MLP layer: per-rank gate/up (column-parallel),
  per-rank down (row-parallel), SiLU multiply, and the summed full-hidden
  partial. It exists so a GPU implementation can be compared against a
  from-scratch reference rather than against itself.

The production path keeps the MLP sharded and reduced across ranks; this module
never models a full-TP1 forward or a hidden-state copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

#: Quant block alignment the MLP intermediate axis must keep for every rank.
MLP_BLOCK = 256


class PrefillPlanError(ValueError):
    """The requested batched-prefill geometry cannot be partitioned safely."""


@dataclass(frozen=True)
class LayerPrefillShape:
    """One layer's batched-prefill shapes, independent of the row count."""

    layer_id: int
    layer_type: str
    hidden: int
    per_rank_ffn: int
    gate_shape: tuple[int, int]
    up_shape: tuple[int, int]
    down_shape: tuple[int, int]
    linear_qkv_width: int | None = None
    conv_state: tuple[int, ...] | None = None
    recurrent_state: tuple[int, ...] | None = None
    kv_heads: int | None = None
    key_length: int | None = None
    value_length: int | None = None

    @property
    def is_linear_attention(self) -> bool:
        return self.layer_type == LINEAR_ATTENTION


@dataclass(frozen=True)
class BatchedPrefillPlan:
    """The immutable plan one rank-local bulk prefill executes."""

    num_ranks: int
    chunk_rows: int
    hidden: int
    ffn: int
    layers: tuple[LayerPrefillShape, ...]

    @property
    def per_rank_ffn(self) -> int:
        return self.ffn // self.num_ranks

    @property
    def exchange_payload_floats(self) -> int:
        """One layer's down-partial reduction size: ``rows * hidden`` floats."""

        return self.chunk_rows * self.hidden

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(layer.layer_id for layer in self.layers if layer.layer_type == FULL_ATTENTION)

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        return tuple(layer.layer_id for layer in self.layers if layer.is_linear_attention)

    def chunk_ranges(self, total_rows: int) -> tuple[tuple[int, int], ...]:
        return chunk_ranges(total_rows, self.chunk_rows)

    def validate(self) -> None:
        if self.num_ranks < 1:
            raise PrefillPlanError("num_ranks must be positive")
        if self.chunk_rows < 1:
            raise PrefillPlanError("chunk_rows must be positive")
        if self.hidden < 1 or self.ffn < 1:
            raise PrefillPlanError("hidden and ffn must be positive")
        if self.ffn % (self.num_ranks * MLP_BLOCK) != 0:
            raise PrefillPlanError(
                f"ffn {self.ffn} is not divisible into {self.num_ranks} block-aligned ranks"
            )
        for layer in self.layers:
            if layer.hidden != self.hidden:
                raise PrefillPlanError(f"layer {layer.layer_id} hidden mismatch")
            if layer.per_rank_ffn != self.per_rank_ffn:
                raise PrefillPlanError(f"layer {layer.layer_id} per-rank ffn mismatch")
            if layer.gate_shape != (self.per_rank_ffn, self.hidden):
                raise PrefillPlanError(f"layer {layer.layer_id} gate shape mismatch")
            if layer.up_shape != (self.per_rank_ffn, self.hidden):
                raise PrefillPlanError(f"layer {layer.layer_id} up shape mismatch")
            if layer.down_shape != (self.hidden, self.per_rank_ffn):
                raise PrefillPlanError(f"layer {layer.layer_id} down shape mismatch")
            if layer.is_linear_attention:
                if layer.conv_state is None or layer.recurrent_state is None:
                    raise PrefillPlanError(f"linear layer {layer.layer_id} lacks Conv/GDN state")
                if layer.kv_heads is not None:
                    raise PrefillPlanError(f"linear layer {layer.layer_id} declares KV geometry")
            elif layer.layer_type == FULL_ATTENTION:
                if layer.kv_heads is None or layer.key_length is None or layer.value_length is None:
                    raise PrefillPlanError(f"full layer {layer.layer_id} lacks KV geometry")
                if layer.conv_state is not None or layer.recurrent_state is not None:
                    raise PrefillPlanError(f"full layer {layer.layer_id} declares Conv/GDN state")
            else:
                raise PrefillPlanError(f"layer {layer.layer_id} has unknown type {layer.layer_type!r}")


def chunk_ranges(total_rows: int, chunk_rows: int) -> tuple[tuple[int, int], ...]:
    """Half-open ``[start, stop)`` chunk ranges covering ``[0, total_rows)``."""

    total_rows = int(total_rows)
    chunk_rows = int(chunk_rows)
    if total_rows < 0:
        raise PrefillPlanError("total_rows must be non-negative")
    if chunk_rows < 1:
        raise PrefillPlanError("chunk_rows must be positive")
    return tuple(
        (start, min(start + chunk_rows, total_rows))
        for start in range(0, total_rows, chunk_rows)
    )


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish activation, matching the fused gate/up+SiLU contract."""

    return x / (1.0 + np.exp(-x))


def _positive_int(config: Any, name: str) -> int:
    value = int(getattr(config, name))
    if value < 1:
        raise PrefillPlanError(f"config.{name} must be positive")
    return value


def plan_batched_prefill(
    config: Any,
    *,
    num_ranks: int,
    chunk_rows: int,
) -> BatchedPrefillPlan:
    """Derive the rank-local batched-prefill geometry from a model config.

    The MLP intermediate axis is split into ``num_ranks`` block-aligned
    contiguous groups (column-parallel gate/up, row-parallel down); the
    reduction sums the full-hidden partials. Attention, GDN, and the final norm
    stay replicated per rank in this diagnostic, so no head or state axis is
    split here.
    """

    num_ranks = int(num_ranks)
    chunk_rows = int(chunk_rows)
    if num_ranks < 1:
        raise PrefillPlanError("num_ranks must be positive")
    if chunk_rows < 1:
        raise PrefillPlanError("chunk_rows must be positive")

    hidden = _positive_int(config, "hidden_size")
    ffn = _positive_int(config, "feed_forward_length")
    if ffn % (num_ranks * MLP_BLOCK) != 0:
        raise PrefillPlanError(
            f"ffn {ffn} is not divisible into {num_ranks} block-aligned ranks "
            f"(block {MLP_BLOCK})"
        )
    per_rank_ffn = ffn // num_ranks

    ssm_group_count = _positive_int(config, "ssm_group_count")
    ssm_state_size = _positive_int(config, "ssm_state_size")
    ssm_inner_size = _positive_int(config, "ssm_inner_size")
    ssm_conv_kernel = _positive_int(config, "ssm_conv_kernel")
    ssm_time_step_rank = _positive_int(config, "ssm_time_step_rank")
    linear_qkv_width = 2 * ssm_group_count * ssm_state_size + ssm_inner_size
    conv_state = (ssm_conv_kernel, linear_qkv_width)
    recurrent_state = (ssm_time_step_rank, ssm_state_size, ssm_state_size)

    head_count_kv = _positive_int(config, "head_count_kv")
    key_length = _positive_int(config, "key_length")
    value_length = _positive_int(config, "value_length")

    layer_types: Sequence[str] = tuple(str(t) for t in config.layer_types)
    if not layer_types:
        raise PrefillPlanError("config.layer_types is empty")

    layers: list[LayerPrefillShape] = []
    for layer_id, layer_type in enumerate(layer_types):
        is_linear = layer_type == LINEAR_ATTENTION
        layers.append(
            LayerPrefillShape(
                layer_id=layer_id,
                layer_type=layer_type,
                hidden=hidden,
                per_rank_ffn=per_rank_ffn,
                gate_shape=(per_rank_ffn, hidden),
                up_shape=(per_rank_ffn, hidden),
                down_shape=(hidden, per_rank_ffn),
                linear_qkv_width=linear_qkv_width if is_linear else None,
                conv_state=conv_state if is_linear else None,
                recurrent_state=recurrent_state if is_linear else None,
                kv_heads=None if is_linear else head_count_kv,
                key_length=None if is_linear else key_length,
                value_length=None if is_linear else value_length,
            )
        )

    plan = BatchedPrefillPlan(
        num_ranks=num_ranks,
        chunk_rows=chunk_rows,
        hidden=hidden,
        ffn=ffn,
        layers=tuple(layers),
    )
    plan.validate()
    return plan


def batched_sharded_mlp_reference(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    x: np.ndarray,
    *,
    num_ranks: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Independent reference for one batched, column/row-sharded MLP layer.

    ``gate``/``up`` are ``(ffn, hidden)``; ``down`` is ``(hidden, ffn)``; ``x``
    is ``(rows, hidden)``. Rank ``r`` owns the contiguous block
    ``[r * per_rank_ffn, (r + 1) * per_rank_ffn)`` of the intermediate axis
    (gate/up output rows and down input columns). The returned ``out`` is the
    summed full-hidden result and ``partials`` is each rank's contribution, so
    the caller can check both the sum and the reduction identity.
    """

    num_ranks = int(num_ranks)
    if num_ranks < 1:
        raise PrefillPlanError("num_ranks must be positive")
    gate = np.asarray(gate)
    up = np.asarray(up)
    down = np.asarray(down)
    x = np.asarray(x)
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2 or x.ndim != 2:
        raise PrefillPlanError("gate/up/down/x must all be 2-D")
    ffn, hidden = gate.shape
    if up.shape != (ffn, hidden):
        raise PrefillPlanError("up must match gate shape")
    if down.shape != (hidden, ffn):
        raise PrefillPlanError("down must be (hidden, ffn)")
    if x.shape[1] != hidden:
        raise PrefillPlanError("x hidden width must match gate")
    if ffn % num_ranks != 0:
        raise PrefillPlanError("ffn must divide evenly across ranks")
    per_rank_ffn = ffn // num_ranks

    partials: list[np.ndarray] = []
    for rank in range(num_ranks):
        start = rank * per_rank_ffn
        stop = start + per_rank_ffn
        gate_shard = gate[start:stop]
        up_shard = up[start:stop]
        down_shard = down[:, start:stop]
        activated = silu(x @ gate_shard.T) * (x @ up_shard.T)
        partials.append(activated @ down_shard.T)
    out = np.sum(partials, axis=0)
    return out, partials
