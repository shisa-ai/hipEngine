"""Synchronous layer-stack execution for the packed GGUF target verifier.

This module owns one packed verifier pass's ordered layer walk: it calls each
layer's already-resolved selected operation over the BF16 hidden ping-pong pair,
accumulates the aggregate wall-clock stage marks, and emits the packed
full-attention GPU stage mark. The resident session owns everything else -
layout, initial-state sync, embeddings, output norm, sampling/LM head,
hidden-seed copy, scatter, cursor/commit/rollback, cancellation, recorder
lifetime and graph/prefix ownership.

Every input arrives as a frozen record. Each selected operation has a typed
binding with the concrete layer id, scratch view, state rows, workspace,
indices and stream resolved by the session. The executor owns the call sites
and supplies the hidden ping-pong pointer pair.

This module does not read owner attributes or environment policy. Borrowed
buffers and selected bound operations stay alive only for the synchronous call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION

LINEAR_ATTENTION_STAGE = "packed_verify_linear_attn_layers"
FULL_ATTENTION_STAGE = "packed_verify_full_attn_layers"
FULL_ATTENTION_GPU_STAGE = "packed_verify_gpu_full_attn_layers"

LINEAR_LAYER_STAGE_PREFIX = "packed_verify_gpu_linear_attn"
FULL_ATTENTION_LAYER_STAGE_PREFIX = "target_block_packed_full_attn"


class StageRecorder(Protocol):
    """The GPU stage marker seam (``_HipEventStageRecorder.mark``)."""

    def mark(self, name: str, *aliases: str) -> None: ...


class LinearRowsOperation(Protocol):
    """The ordinary linear-attention row owner this stack calls."""

    def __call__(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch: object,
        *,
        rows: int,
        decode_scratch: object,
        stream: int = 0,
        expert_sidecar: object | None = None,
        linear_state_rows: tuple[object, object] | None = None,
        commit_final_linear_state: bool = True,
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = ...,
        gpu_stage_recorder: StageRecorder | None = None,
    ) -> None: ...


class MoeC2LinearRowsOperation(Protocol):
    """The MoE C2 exact linear-row owner this stack calls."""

    def __call__(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch: object,
        *,
        rows: int,
        decode_scratch: object,
        linear_state_rows: tuple[object, object],
        state_capacity: int,
        state_indices: Sequence[int],
        stream: int = 0,
    ) -> None: ...


class FullAttentionRowsOperation(Protocol):
    """The full-attention decode-batch row owner this stack calls."""

    def __call__(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch: object,
        *,
        stream: int = 0,
        expert_sidecar: object | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = ...,
        split_workspace: object | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class VerifyLayerStackGeometry:
    """Layer order and packed row count this pass was resolved for."""

    layer_types: tuple[str, ...]
    rows: int


@dataclass(frozen=True)
class VerifyLayerStackHidden:
    """The hidden ping-pong pair.

    ``first_ptr``/``second_ptr`` are the only execution inputs; the matching
    buffer fields let the caller receive the final endpoint without the executor
    reading any buffer attribute.
    """

    first: object
    first_ptr: int
    second: object
    second_ptr: int


@dataclass(frozen=True)
class VerifyOrdinaryLinearLayer:
    """One linear-attention layer resolved to the ordinary row owner."""

    layer_id: int
    rows: int
    run_linear_rows: LinearRowsOperation
    scratch: object
    decode_scratch: object
    linear_state_rows: tuple[object, object] | None
    commit_final_linear_state: bool
    stream: int


@dataclass(frozen=True)
class VerifyMoeC2LinearLayer:
    """One linear-attention layer resolved to the MoE C2 exact row owner."""

    layer_id: int
    rows: int
    run_moe_c2_linear_rows: MoeC2LinearRowsOperation
    scratch: object
    decode_scratch: object
    linear_state_rows: tuple[object, object]
    state_capacity: int
    state_indices: Sequence[int]
    stream: int


@dataclass(frozen=True)
class VerifyFullAttentionLayer:
    """One full-attention layer resolved to the decode-batch row owner."""

    layer_id: int
    rows: int
    run_full_attention_rows: FullAttentionRowsOperation
    scratch: object
    split_workspace: object | None
    stream: int


VerifyLayer = (
    VerifyOrdinaryLinearLayer | VerifyMoeC2LinearLayer | VerifyFullAttentionLayer
)


@dataclass(frozen=True)
class VerifyLayerStackPlan:
    """The complete immutable layer-stack input for one packed pass."""

    geometry: VerifyLayerStackGeometry
    hidden: VerifyLayerStackHidden
    bindings: tuple[VerifyLayer, ...]


def _validate_plan(plan: VerifyLayerStackPlan) -> None:
    geometry = plan.geometry
    rows = int(geometry.rows)
    if rows <= 0:
        raise ValueError("packed verifier layer stack requires at least one row")
    layer_types = geometry.layer_types
    bindings = plan.bindings
    if len(bindings) != len(layer_types):
        raise ValueError(
            f"packed verifier layer stack has {len(bindings)} bindings for "
            f"{len(layer_types)} layers"
        )
    hidden = plan.hidden
    if hidden.first_ptr <= 0 or hidden.second_ptr <= 0:
        raise ValueError("packed verifier hidden ping-pong requires valid pointers")
    # Prefill lifetime planning may alias these endpoints for an in-place
    # verifier route. Preserve the supplied views instead of imposing a new
    # allocation contract at this execution boundary.
    for layer_id, binding in enumerate(bindings):
        if int(binding.layer_id) != layer_id:
            raise ValueError(
                f"packed verifier binding {layer_id} names layer {binding.layer_id}"
            )
        if int(binding.rows) != rows:
            raise ValueError(
                f"packed verifier layer {layer_id} was resolved for {binding.rows} rows, "
                f"not {rows}"
            )
        layer_type = layer_types[layer_id]
        if layer_type == LINEAR_ATTENTION:
            if not isinstance(
                binding, (VerifyOrdinaryLinearLayer, VerifyMoeC2LinearLayer)
            ):
                raise ValueError(
                    f"packed verifier layer {layer_id} is {layer_type!r} but was "
                    "resolved as full attention"
                )
        elif layer_type == FULL_ATTENTION:
            if not isinstance(binding, VerifyFullAttentionLayer):
                raise ValueError(
                    f"packed verifier layer {layer_id} is {layer_type!r} but was "
                    "resolved as linear attention"
                )
        else:
            raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
        if isinstance(binding, VerifyMoeC2LinearLayer):
            if binding.linear_state_rows is None:
                raise ValueError(
                    f"packed verifier MoE C2 layer {layer_id} requires captured state rows"
                )
            if int(binding.state_capacity) < 1:
                raise ValueError(
                    f"packed verifier MoE C2 layer {layer_id} requires state capacity"
                )


def _accumulate_stage(
    stage_timings: dict[str, float],
    name: str,
    start: float,
) -> None:
    stage_timings[name] = (
        stage_timings.get(name, 0.0) + (time.perf_counter() - start) * 1000.0
    )


def _run_layer(
    binding: VerifyLayer,
    hidden_in_ptr: int,
    hidden_out_ptr: int,
    gpu_stage_recorder: StageRecorder | None,
) -> None:
    if isinstance(binding, VerifyFullAttentionLayer):
        binding.run_full_attention_rows(
            binding.layer_id,
            hidden_in_ptr,
            hidden_out_ptr,
            binding.scratch,
            stream=binding.stream,
            expert_sidecar=None,
            stage_timings=None,
            sync_stage_timings=False,
            stage_prefix=FULL_ATTENTION_LAYER_STAGE_PREFIX,
            split_workspace=binding.split_workspace,
        )
    elif isinstance(binding, VerifyMoeC2LinearLayer):
        binding.run_moe_c2_linear_rows(
            binding.layer_id,
            hidden_in_ptr,
            hidden_out_ptr,
            binding.scratch,
            rows=binding.rows,
            decode_scratch=binding.decode_scratch,
            linear_state_rows=binding.linear_state_rows,
            state_capacity=binding.state_capacity,
            state_indices=binding.state_indices,
            stream=binding.stream,
        )
    elif isinstance(binding, VerifyOrdinaryLinearLayer):
        binding.run_linear_rows(
            binding.layer_id,
            hidden_in_ptr,
            hidden_out_ptr,
            binding.scratch,
            rows=binding.rows,
            stream=binding.stream,
            decode_scratch=binding.decode_scratch,
            expert_sidecar=None,
            linear_state_rows=binding.linear_state_rows,
            commit_final_linear_state=binding.commit_final_linear_state,
            hidden_f32_ptr=None,
            out_f32_ptr=None,
            stage_timings=None,
            sync_stage_timings=False,
            stage_prefix=LINEAR_LAYER_STAGE_PREFIX,
            gpu_stage_recorder=gpu_stage_recorder,
        )
    else:  # pragma: no cover - the binding union is closed
        raise TypeError(f"unsupported packed verifier layer binding {binding!r}")


def run_verify_layer_stack(
    plan: VerifyLayerStackPlan,
    *,
    stage_timings: dict[str, float],
    gpu_stage_recorder: StageRecorder | None = None,
) -> object:
    """Run every resolved layer in order and return the final hidden buffer.

    ``stage_timings`` accumulates the aggregate per-layer-type wall-clock marks
    exactly as the inline packed loop did. ``gpu_stage_recorder``, when present,
    receives one full-attention GPU mark per full-attention layer; the ordinary
    linear owner also marks its own internal stages through it.
    """

    _validate_plan(plan)
    hidden = plan.hidden
    src = hidden.first
    src_ptr = hidden.first_ptr
    dst = hidden.second
    dst_ptr = hidden.second_ptr
    for binding in plan.bindings:
        layer_start = time.perf_counter()
        _run_layer(binding, src_ptr, dst_ptr, gpu_stage_recorder)
        if isinstance(binding, VerifyFullAttentionLayer):
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(FULL_ATTENTION_GPU_STAGE)
            _accumulate_stage(stage_timings, FULL_ATTENTION_STAGE, layer_start)
        else:
            _accumulate_stage(stage_timings, LINEAR_ATTENTION_STAGE, layer_start)
        src, dst = dst, src
        src_ptr, dst_ptr = dst_ptr, src_ptr
    return src
