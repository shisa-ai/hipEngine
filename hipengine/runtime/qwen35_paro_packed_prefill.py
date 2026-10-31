"""The PARO packed-prefill layer schedule over borrowed scratch and selected ops.

The public state method keeps token validation and scratch fallback. This owner
receives only explicit geometry, borrowed scratch records, and four selected
operations. It never resolves a session, runner, weights, policy, allocation or
synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from hipengine.core.tensor import Tensor

if TYPE_CHECKING:
    from hipengine.runtime.qwen35_paro import (
        Qwen35ParoGroupedMoeScratch,
        Qwen35ParoLinearAttentionScratch,
    )


@dataclass(frozen=True)
class Qwen35ParoPackedPrefillGeometry:
    tokens: int
    segments: int
    group_size: int = 128


@dataclass(frozen=True)
class Qwen35ParoPackedPrefillScratch:
    linear: Qwen35ParoLinearAttentionScratch
    moe: Qwen35ParoGroupedMoeScratch


class InputRMSNorm(Protocol):
    def __call__(
        self,
        hidden: Tensor,
        out: Tensor,
        *,
        tokens: int,
        library: object,
        stream: int,
    ) -> None: ...


class LinearPrefillSegments(Protocol):
    def __call__(
        self,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        cu_seqlens: Tensor,
        state_indices: Tensor,
        segments: int,
        scratch: Qwen35ParoLinearAttentionScratch,
        tokens: int,
        group_size: int,
        library: object,
        stream: int,
    ) -> Tensor: ...


class PostAttentionAddRMSNorm(Protocol):
    def __call__(
        self,
        hidden: Tensor,
        attn_out: Tensor,
        scratch: Qwen35ParoGroupedMoeScratch,
        *,
        tokens: int,
        library: object,
        stream: int,
    ) -> tuple[Tensor, Tensor]: ...


class GroupedMoe(Protocol):
    def __call__(
        self,
        hidden: Tensor,
        residual: Tensor,
        *,
        scratch: Qwen35ParoGroupedMoeScratch,
        tokens: int,
        group_size: int,
        library: object,
        stream: int,
    ) -> Tensor: ...


def run_linear_attention_moe_packed_prefill_layer(
    hidden: Tensor,
    *,
    conv_state: Tensor,
    recurrent_state: Tensor,
    cu_seqlens: Tensor,
    state_indices: Tensor,
    geometry: Qwen35ParoPackedPrefillGeometry,
    scratch: Qwen35ParoPackedPrefillScratch,
    input_rmsnorm: InputRMSNorm,
    linear_prefill_segments: LinearPrefillSegments,
    post_attention_add_rmsnorm: PostAttentionAddRMSNorm,
    grouped_moe: GroupedMoe,
    library: object = None,
    stream: int = 0,
) -> Tensor:
    """Run one packed linear-attention prefill layer.

    ``geometry.tokens`` is validated by the caller at the prior point. Scratch
    is borrowed from the caller; this function performs no allocation,
    synchronization, policy lookup, or error translation.
    """

    input_rmsnorm(
        hidden,
        scratch.linear.attn_input,
        tokens=geometry.tokens,
        library=library,
        stream=stream,
    )
    attn_out = linear_prefill_segments(
        scratch.linear.attn_input,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        segments=geometry.segments,
        scratch=scratch.linear,
        tokens=geometry.tokens,
        group_size=geometry.group_size,
        library=library,
        stream=stream,
    )
    mlp_input, residual = post_attention_add_rmsnorm(
        hidden,
        attn_out,
        scratch.moe,
        tokens=geometry.tokens,
        library=library,
        stream=stream,
    )
    return grouped_moe(
        mlp_input,
        residual,
        scratch=scratch.moe,
        tokens=geometry.tokens,
        group_size=geometry.group_size,
        library=library,
        stream=stream,
    )
