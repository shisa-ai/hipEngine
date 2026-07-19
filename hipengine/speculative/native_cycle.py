"""Provider-neutral native speculative-cycle ABI.

This module freezes the host/native boundary used by future GGUF MTP, PARO MTP,
and DFlash launchers.  The ABI is deliberately math-free: provider adapters
resolve registered kernel launchers and populate raw device pointers, while the
common layer validates bounded shapes, lifecycle, and terminal results.

Pointer ownership is borrowed.  The caller must keep every allocation and the
session-owned stream alive until ``launch()`` returns (or an asynchronous
launcher reports a terminal result).  A launcher may mutate only pointees named
as outputs/state destinations; it never frees or retains caller allocations.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field as dataclass_field, fields
from enum import IntEnum, IntFlag
import threading
from typing import Callable, Protocol, runtime_checkable

from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kvcache.spans import KVLiveSpans
from hipengine.speculative.interfaces import TargetVerifyBuffers

NATIVE_SPEC_CYCLE_ABI_VERSION = 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class NativeSpecCycleStage(IntFlag):
    """Ordered pieces of a speculative cycle requested from a launcher."""

    PROPOSE = 1 << 0
    VERIFY = 1 << 1
    ACCEPT = 1 << 2
    COMMIT = 1 << 3
    UPDATE_CURSORS = 1 << 4


_ALL_STAGES = (
    NativeSpecCycleStage.PROPOSE
    | NativeSpecCycleStage.VERIFY
    | NativeSpecCycleStage.ACCEPT
    | NativeSpecCycleStage.COMMIT
    | NativeSpecCycleStage.UPDATE_CURSORS
)


class NativeSpecCycleMode(IntEnum):
    CHAIN = 1
    TREE = 2


class NativeSpecCycleStatus(IntEnum):
    """Lifecycle states shared by synchronous and future yielding launchers."""

    CREATED = 0
    SUBMITTED = 1
    RUNNING = 2
    COMPLETE = 3
    YIELDED = 4
    CANCELLED = 5
    DEADLINE_EXCEEDED = 6
    FAILED = 7


class NativeSpecCycleError(IntEnum):
    NONE = 0
    ABI_MISMATCH = 1
    INVALID_CONTROL = 2
    UNSUPPORTED_SHAPE = 3
    CANCELLED = 4
    DEADLINE_EXCEEDED = 5
    KERNEL_LAUNCH = 6
    INTERNAL = 7


class NativeSpecCycleDType(IntEnum):
    """Element types interpreted directly by the native control ABI."""

    INT32 = 1
    INT64 = 2
    FP16 = 3
    BF16 = 4
    FP32 = 5
    INT8 = 6
    INT8_PER_TOKEN_HEAD = 7


_DTYPE_TO_NATIVE = {
    DType.INT32: NativeSpecCycleDType.INT32,
    DType.INT64: NativeSpecCycleDType.INT64,
    DType.FP16: NativeSpecCycleDType.FP16,
    DType.BF16: NativeSpecCycleDType.BF16,
    DType.FP32: NativeSpecCycleDType.FP32,
    DType.INT8: NativeSpecCycleDType.INT8,
    DType.INT8_PER_TOKEN_HEAD: NativeSpecCycleDType.INT8_PER_TOKEN_HEAD,
}


@dataclass(frozen=True, slots=True)
class NativeSpecCycleShape:
    """Live counts and allocation capacities for one cycle invocation.

    Counts describe the root-prefixed target verifier region.  Capacities are
    the maximum elements that the supplied pointers may address; native code
    must never use a live count as an implicit allocation bound.
    """

    request_count: int
    request_capacity: int
    row_count: int
    active_row_count: int
    row_capacity: int
    candidate_count: int
    active_candidate_count: int
    candidate_capacity: int
    candidate_budget: int
    span_count: int
    span_capacity: int
    max_live_count: int
    context_bucket: int
    hidden_size: int
    hidden_row_capacity: int
    output_stride: int
    metadata_dtype: NativeSpecCycleDType
    hidden_dtype: NativeSpecCycleDType
    kv_dtype: NativeSpecCycleDType

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata_dtype",
            _coerce_enum("metadata_dtype", self.metadata_dtype, NativeSpecCycleDType),
        )
        object.__setattr__(
            self,
            "hidden_dtype",
            _coerce_enum("hidden_dtype", self.hidden_dtype, NativeSpecCycleDType),
        )
        object.__setattr__(
            self,
            "kv_dtype",
            _coerce_enum("kv_dtype", self.kv_dtype, NativeSpecCycleDType),
        )
        for field in fields(self):
            _check_uint(field.name, getattr(self, field.name), 32)
        if self.request_count <= 0:
            raise ValueError("request_count must be positive")
        if self.request_capacity < self.request_count:
            raise ValueError("request_capacity must be at least request_count")
        if self.row_count != self.request_count + self.candidate_count:
            raise ValueError("row_count must equal request_count plus candidate_count")
        if self.active_row_count != self.request_count + self.active_candidate_count:
            raise ValueError("active_row_count must equal request_count plus active_candidate_count")
        if not self.request_count <= self.active_row_count <= self.row_count:
            raise ValueError("active_row_count must be between request_count and row_count")
        if self.active_candidate_count > self.candidate_count:
            raise ValueError("active_candidate_count cannot exceed candidate_count")
        if self.row_capacity < self.row_count:
            raise ValueError("row_capacity must be at least row_count")
        if self.candidate_capacity < self.candidate_count:
            raise ValueError("candidate_capacity must be at least candidate_count")
        if self.request_capacity + self.candidate_capacity > self.row_capacity:
            raise ValueError("row_capacity must bound request_capacity plus candidate_capacity")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if self.candidate_budget <= 0 or self.candidate_budget > self.candidate_capacity:
            raise ValueError("candidate_budget must be positive and no larger than candidate_capacity")
        if self.span_count <= 0:
            raise ValueError("span_count must be positive")
        if self.span_capacity < self.span_count:
            raise ValueError("span_capacity must be at least span_count")
        if self.context_bucket <= 0 or self.context_bucket < self.max_live_count:
            raise ValueError("context_bucket must be positive and at least max_live_count")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.hidden_row_capacity <= 0:
            raise ValueError("hidden_row_capacity must be positive")
        if self.output_stride < self.candidate_budget + 1:
            raise ValueError("output_stride must hold a root/correction plus the candidate budget")
        if self.metadata_dtype not in {NativeSpecCycleDType.INT32, NativeSpecCycleDType.INT64}:
            raise ValueError("metadata_dtype must be INT32 or INT64")
        if self.hidden_dtype not in {
            NativeSpecCycleDType.FP16,
            NativeSpecCycleDType.BF16,
            NativeSpecCycleDType.FP32,
        }:
            raise ValueError("hidden_dtype must be FP16, BF16, or FP32")
        if self.kv_dtype not in {
            NativeSpecCycleDType.FP16,
            NativeSpecCycleDType.BF16,
            NativeSpecCycleDType.FP32,
            NativeSpecCycleDType.INT8,
            NativeSpecCycleDType.INT8_PER_TOKEN_HEAD,
        }:
            raise ValueError("kv_dtype is not a supported KV storage type")


class _PointerGroup:
    def _validate_pointer_fields(self, prefix: str) -> None:
        for field in fields(self):
            _check_pointer(f"{prefix}.{field.name}", getattr(self, field.name))


@dataclass(frozen=True, slots=True)
class NativeSpecCycleMetadataPointers(_PointerGroup):
    request_ids: int = 0
    token_ids: int = 0
    positions: int = 0
    parent_rows: int = 0
    draft_depths: int = 0
    row_to_request: int = 0
    active_mask: int = 0
    candidate_counts: int = 0
    remaining_decode: int = 0

    def __post_init__(self) -> None:
        self._validate_pointer_fields("metadata")


@dataclass(frozen=True, slots=True)
class NativeSpecCycleKVLiveSpanPointers(_PointerGroup):
    """Raw ``KVLiveSpans`` pointers plus the addressed K/V arenas."""

    base_offsets: int = 0
    live_counts: int = 0
    token_positions: int = 0
    evict_mask: int = 0
    request_ids: int = 0
    row_positions: int = 0
    k_scale: int = 0
    v_scale: int = 0
    key_cache: int = 0
    value_cache: int = 0

    def __post_init__(self) -> None:
        self._validate_pointer_fields("kv_live_spans")
        _check_pair("kv_live_spans.k_scale and kv_live_spans.v_scale", self.k_scale, self.v_scale)
        _check_pair("kv_live_spans.key_cache and kv_live_spans.value_cache", self.key_cache, self.value_cache)


@dataclass(frozen=True, slots=True)
class NativeSpecCycleStatePointers(_PointerGroup):
    """Proposal, verifier-hidden, and state/KV commit pointers."""

    hidden_seed_in: int = 0
    proposal_state: int = 0
    candidate_token_ids: int = 0
    candidate_probabilities: int = 0
    draft_key_cache: int = 0
    draft_value_cache: int = 0
    hidden_seed_rows: int = 0
    linear_state_rows: int = 0
    linear_state_dst: int = 0
    key_rows: int = 0
    value_rows: int = 0
    hidden_seed_dst: int = 0

    def __post_init__(self) -> None:
        self._validate_pointer_fields("state")
        _check_pair(
            "state.draft_key_cache and state.draft_value_cache",
            self.draft_key_cache,
            self.draft_value_cache,
        )
        _check_pair(
            "state.linear_state_rows and state.linear_state_dst",
            self.linear_state_rows,
            self.linear_state_dst,
        )
        _check_pair("state.key_rows and state.value_rows", self.key_rows, self.value_rows)
        if self.hidden_seed_dst and not self.hidden_seed_rows:
            raise ValueError("state.hidden_seed_dst requires state.hidden_seed_rows")


@dataclass(frozen=True, slots=True)
class NativeSpecCycleOutputPointers(_PointerGroup):
    """Verifier, accept-summary, scheduler-output, and cancellation pointers."""

    target_logits: int = 0
    target_top1: int = 0
    accepted_counts: int = 0
    commit_rows: int = 0
    commit_tokens: int = 0
    commit_positions: int = 0
    next_tokens: int = 0
    full_accept: int = 0
    committed_output_ids: int = 0
    committed_output_lengths: int = 0
    output_ids: int = 0
    output_lengths: int = 0
    last_positions: int = 0
    context_lengths: int = 0
    cancel_flag: int = 0

    def __post_init__(self) -> None:
        self._validate_pointer_fields("outputs")
        _check_pair(
            "outputs.committed_output_ids and outputs.committed_output_lengths",
            self.committed_output_ids,
            self.committed_output_lengths,
        )
        _check_pair("outputs.output_ids and outputs.output_lengths", self.output_ids, self.output_lengths)
        _check_pair(
            "outputs.last_positions and outputs.context_lengths",
            self.last_positions,
            self.context_lengths,
        )


@dataclass(frozen=True, slots=True)
class NativeSpecCyclePointers:
    metadata: NativeSpecCycleMetadataPointers = dataclass_field(default_factory=NativeSpecCycleMetadataPointers)
    kv_live_spans: NativeSpecCycleKVLiveSpanPointers = dataclass_field(default_factory=NativeSpecCycleKVLiveSpanPointers)
    state: NativeSpecCycleStatePointers = dataclass_field(default_factory=NativeSpecCycleStatePointers)
    outputs: NativeSpecCycleOutputPointers = dataclass_field(default_factory=NativeSpecCycleOutputPointers)

    def __post_init__(self) -> None:
        expected = (
            ("metadata", self.metadata, NativeSpecCycleMetadataPointers),
            ("kv_live_spans", self.kv_live_spans, NativeSpecCycleKVLiveSpanPointers),
            ("state", self.state, NativeSpecCycleStatePointers),
            ("outputs", self.outputs, NativeSpecCycleOutputPointers),
        )
        for name, value, kind in expected:
            if not isinstance(value, kind):
                raise TypeError(f"pointers.{name} must be {kind.__name__}")


class NativeSpecCycleControlC(ctypes.Structure):
    """Fixed-width C mirror of ``NativeSpecCycleControlV1``."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("stage_mask", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("cycle_id", ctypes.c_uint64),
        ("transaction_id", ctypes.c_uint64),
        ("stream", ctypes.c_uint64),
        ("deadline_ns", ctypes.c_uint64),
        *[(name, ctypes.c_uint32) for name in (
            "request_count",
            "request_capacity",
            "row_count",
            "active_row_count",
            "row_capacity",
            "candidate_count",
            "active_candidate_count",
            "candidate_capacity",
            "candidate_budget",
            "span_count",
            "span_capacity",
            "max_live_count",
            "context_bucket",
            "hidden_size",
            "hidden_row_capacity",
            "output_stride",
            "metadata_dtype",
            "hidden_dtype",
            "kv_dtype",
        )],
        *[(name, ctypes.c_uint64) for name in (
            "metadata_request_ids",
            "metadata_token_ids",
            "metadata_positions",
            "metadata_parent_rows",
            "metadata_draft_depths",
            "metadata_row_to_request",
            "metadata_active_mask",
            "metadata_candidate_counts",
            "metadata_remaining_decode",
            "kv_base_offsets",
            "kv_live_counts",
            "kv_token_positions",
            "kv_evict_mask",
            "kv_request_ids",
            "kv_row_positions",
            "kv_k_scale",
            "kv_v_scale",
            "kv_key_cache",
            "kv_value_cache",
            "state_hidden_seed_in",
            "state_proposal_state",
            "state_candidate_token_ids",
            "state_candidate_probabilities",
            "state_draft_key_cache",
            "state_draft_value_cache",
            "state_hidden_seed_rows",
            "state_linear_state_rows",
            "state_linear_state_dst",
            "state_key_rows",
            "state_value_rows",
            "state_hidden_seed_dst",
            "output_target_logits",
            "output_target_top1",
            "output_accepted_counts",
            "output_commit_rows",
            "output_commit_tokens",
            "output_commit_positions",
            "output_next_tokens",
            "output_full_accept",
            "output_committed_output_ids",
            "output_committed_output_lengths",
            "output_output_ids",
            "output_output_lengths",
            "output_last_positions",
            "output_context_lengths",
            "output_cancel_flag",
        )],
    ]


@dataclass(frozen=True, slots=True)
class NativeSpecCycleControl:
    """Validated version-1 control block with borrowed raw pointers."""

    cycle_id: int
    transaction_id: int
    stages: NativeSpecCycleStage
    mode: NativeSpecCycleMode
    shape: NativeSpecCycleShape
    pointers: NativeSpecCyclePointers
    stream: int = 0
    deadline_ns: int = 0
    abi_version: int = NATIVE_SPEC_CYCLE_ABI_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", _coerce_stage_mask(self.stages))
        object.__setattr__(self, "mode", _coerce_enum("mode", self.mode, NativeSpecCycleMode))
        self.validate()

    def validate(self) -> None:
        if self.abi_version != NATIVE_SPEC_CYCLE_ABI_VERSION:
            raise ValueError(
                f"abi_version must be {NATIVE_SPEC_CYCLE_ABI_VERSION}, got {self.abi_version}"
            )
        _check_uint("abi_version", self.abi_version, 32)
        _check_uint("cycle_id", self.cycle_id, 64)
        _check_uint("transaction_id", self.transaction_id, 64)
        _check_pointer("stream", self.stream)
        _check_uint("deadline_ns", self.deadline_ns, 64)
        if not isinstance(self.shape, NativeSpecCycleShape):
            raise TypeError("shape must be NativeSpecCycleShape")
        if not isinstance(self.pointers, NativeSpecCyclePointers):
            raise TypeError("pointers must be NativeSpecCyclePointers")
        _validate_stage_dependencies(self.stages)
        if self.shape.hidden_row_capacity < self.shape.row_count and self.stages & NativeSpecCycleStage.VERIFY:
            raise ValueError("hidden_row_capacity must be at least row_count for VERIFY")

        metadata = self.pointers.metadata
        spans = self.pointers.kv_live_spans
        state = self.pointers.state
        outputs = self.pointers.outputs
        if self.stages & NativeSpecCycleStage.PROPOSE:
            _require_pointer("state.hidden_seed_in", state.hidden_seed_in)
            _require_pointer("state.candidate_token_ids", state.candidate_token_ids)
        if self.stages & NativeSpecCycleStage.VERIFY:
            for name in (
                "token_ids",
                "positions",
                "parent_rows",
                "draft_depths",
                "row_to_request",
                "active_mask",
            ):
                _require_pointer(f"metadata.{name}", getattr(metadata, name))
            _require_pointer("kv_live_spans.base_offsets", spans.base_offsets)
            _require_pointer("kv_live_spans.live_counts", spans.live_counts)
            _require_pointer("state.hidden_seed_rows", state.hidden_seed_rows)
            _require_pointer("outputs.target_top1", outputs.target_top1)
        if self.stages & NativeSpecCycleStage.ACCEPT:
            _require_pointer("metadata.candidate_counts", metadata.candidate_counts)
            for name in (
                "target_top1",
                "accepted_counts",
                "commit_rows",
                "commit_tokens",
                "commit_positions",
                "next_tokens",
                "full_accept",
                "committed_output_ids",
                "committed_output_lengths",
            ):
                _require_pointer(f"outputs.{name}", getattr(outputs, name))
        if self.stages & NativeSpecCycleStage.COMMIT:
            _require_pointer("outputs.accepted_counts", outputs.accepted_counts)
            _require_pointer("outputs.commit_rows", outputs.commit_rows)
            _require_pointer("outputs.commit_positions", outputs.commit_positions)
            has_linear = bool(state.linear_state_rows and state.linear_state_dst)
            has_kv = bool(
                state.key_rows
                and state.value_rows
                and spans.key_cache
                and spans.value_cache
            )
            has_hidden = bool(state.hidden_seed_rows and state.hidden_seed_dst)
            if not (has_linear or has_kv or has_hidden):
                raise ValueError("COMMIT requires at least one bounded linear-state, KV, or hidden-seed destination")
        if self.stages & NativeSpecCycleStage.UPDATE_CURSORS:
            for name in ("output_ids", "output_lengths", "last_positions", "context_lengths"):
                _require_pointer(f"outputs.{name}", getattr(outputs, name))

    @classmethod
    def for_target_verify(
        cls,
        *,
        cycle_id: int,
        buffers: TargetVerifyBuffers,
        kv_live_spans: KVLiveSpans,
        hidden_seed_rows: Tensor,
        context_bucket: int,
        stream: int = 0,
        deadline_ns: int = 0,
        stages: NativeSpecCycleStage = NativeSpecCycleStage.VERIFY,
        active_row_count: int | None = None,
        output_stride: int | None = None,
        candidate_counts_ptr: int = 0,
        remaining_decode_ptr: int = 0,
        target_logits_ptr: int = 0,
        key_cache_ptr: int = 0,
        value_cache_ptr: int = 0,
    ) -> "NativeSpecCycleControl":
        """Adapt existing target buffers and ``KVLiveSpans`` to target-only N1.

        Exact tensor views provide the capacities in this constructor.  Callers
        with a larger fixed owner must bind or describe that owner explicitly
        rather than claiming capacity that is not visible in the tensors.
        """

        if not isinstance(buffers, TargetVerifyBuffers):
            raise TypeError("buffers must be TargetVerifyBuffers")
        if not isinstance(kv_live_spans, KVLiveSpans):
            raise TypeError("kv_live_spans must be KVLiveSpans")
        if not isinstance(hidden_seed_rows, Tensor):
            raise TypeError("hidden_seed_rows must be Tensor")
        if buffers.device != kv_live_spans.device or buffers.device != hidden_seed_rows.device:
            raise ValueError("target verify buffers, KVLiveSpans, and hidden seeds must live on one device")
        if kv_live_spans.span_role != buffers.mode:
            raise ValueError("KVLiveSpans span_role must match target verify mode")
        if hidden_seed_rows.ndim != 2 or hidden_seed_rows.shape[0] < buffers.rows:
            raise ValueError("hidden_seed_rows must have shape (at_least_rows, hidden_size)")
        if hidden_seed_rows.shape[1] <= 0:
            raise ValueError("hidden_seed_rows hidden_size must be positive")
        if hidden_seed_rows.dtype not in {DType.FP16, DType.BF16, DType.FP32}:
            raise ValueError("hidden_seed_rows must be fp16, bf16, or fp32")
        if kv_live_spans.base_offsets.numel <= 0:
            raise ValueError("KVLiveSpans base_offsets must not be empty")
        if kv_live_spans.token_positions is not None and kv_live_spans.token_positions.dtype not in {
            DType.INT32,
            DType.INT64,
        }:
            raise ValueError("KVLiveSpans token_positions must be int32 or int64")

        integer_tensors = [
            buffers.token_ids,
            buffers.positions,
            buffers.parent_rows,
            buffers.draft_depths,
            buffers.row_to_request,
            buffers.target_top1,
            buffers.accepted_counts,
            buffers.commit_rows,
            buffers.commit_tokens,
            buffers.commit_positions,
        ]
        integer_tensors.extend(
            tensor
            for tensor in (
                buffers.next_tokens,
                buffers.committed_output_ids,
                buffers.committed_output_lengths,
            )
            if tensor is not None
        )
        for name, tensor in (
            ("token_ids", buffers.token_ids),
            ("positions", buffers.positions),
            ("parent_rows", buffers.parent_rows),
            ("draft_depths", buffers.draft_depths),
            ("row_to_request", buffers.row_to_request),
            ("active_mask", buffers.active_mask),
            ("target_top1", buffers.target_top1),
            ("accepted_counts", buffers.accepted_counts),
            ("commit_rows", buffers.commit_rows),
            ("commit_tokens", buffers.commit_tokens),
            ("commit_positions", buffers.commit_positions),
            ("hidden_seed_rows", hidden_seed_rows),
            ("kv.base_offsets", kv_live_spans.base_offsets),
            ("kv.live_counts", kv_live_spans.live_counts),
        ):
            _require_contiguous(name, tensor)
        for name, tensor in (
            ("next_tokens", buffers.next_tokens),
            ("full_accept", buffers.full_accept),
            ("committed_output_ids", buffers.committed_output_ids),
            ("committed_output_lengths", buffers.committed_output_lengths),
            ("kv.token_positions", kv_live_spans.token_positions),
            ("kv.evict_mask", kv_live_spans.evict_mask),
            ("kv.request_ids", kv_live_spans.request_ids),
            ("kv.row_positions", kv_live_spans.row_positions),
        ):
            if tensor is not None:
                _require_contiguous(name, tensor)
        if kv_live_spans.scale_metadata is not None:
            _require_contiguous("kv.k_scale", kv_live_spans.scale_metadata.k_scale)
            _require_contiguous("kv.v_scale", kv_live_spans.scale_metadata.v_scale)

        metadata_dtypes = {tensor.dtype for tensor in integer_tensors}
        if len(metadata_dtypes) != 1:
            raise ValueError("native cycle metadata tensors must use one integer dtype")
        metadata_dtype = next(iter(metadata_dtypes))
        if metadata_dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("native cycle metadata dtype must be int32 or int64")

        inferred_output_stride = (
            buffers.committed_output_ids.shape[1]
            if buffers.committed_output_ids is not None
            else buffers.rows
        )
        stride = inferred_output_stride if output_stride is None else int(output_stride)
        if buffers.committed_output_ids is not None and stride > inferred_output_stride:
            raise ValueError("output_stride exceeds committed_output_ids capacity")
        active_rows = buffers.rows if active_row_count is None else int(active_row_count)
        active_candidates = active_rows - buffers.request_count
        draft_depth = int(buffers.draft_depth or 0)
        if draft_depth <= 0:
            raise ValueError("target verify buffers require a positive draft_depth")

        spans = kv_live_spans
        scale = spans.scale_metadata
        pointer_groups = NativeSpecCyclePointers(
            metadata=NativeSpecCycleMetadataPointers(
                token_ids=buffers.token_ids.ptr,
                positions=buffers.positions.ptr,
                parent_rows=buffers.parent_rows.ptr,
                draft_depths=buffers.draft_depths.ptr,
                row_to_request=buffers.row_to_request.ptr,
                active_mask=buffers.active_mask.ptr,
                candidate_counts=candidate_counts_ptr,
                remaining_decode=remaining_decode_ptr,
            ),
            kv_live_spans=NativeSpecCycleKVLiveSpanPointers(
                base_offsets=spans.base_offsets.ptr,
                live_counts=spans.live_counts.ptr,
                token_positions=0 if spans.token_positions is None else spans.token_positions.ptr,
                evict_mask=0 if spans.evict_mask is None else spans.evict_mask.ptr,
                request_ids=0 if spans.request_ids is None else spans.request_ids.ptr,
                row_positions=0 if spans.row_positions is None else spans.row_positions.ptr,
                k_scale=0 if scale is None else scale.k_scale.ptr,
                v_scale=0 if scale is None else scale.v_scale.ptr,
                key_cache=key_cache_ptr,
                value_cache=value_cache_ptr,
            ),
            state=NativeSpecCycleStatePointers(hidden_seed_rows=hidden_seed_rows.ptr),
            outputs=NativeSpecCycleOutputPointers(
                target_logits=target_logits_ptr,
                target_top1=buffers.target_top1.ptr,
                accepted_counts=buffers.accepted_counts.ptr,
                commit_rows=buffers.commit_rows.ptr,
                commit_tokens=buffers.commit_tokens.ptr,
                commit_positions=buffers.commit_positions.ptr,
                next_tokens=0 if buffers.next_tokens is None else buffers.next_tokens.ptr,
                full_accept=0 if buffers.full_accept is None else buffers.full_accept.ptr,
                committed_output_ids=(
                    0 if buffers.committed_output_ids is None else buffers.committed_output_ids.ptr
                ),
                committed_output_lengths=(
                    0
                    if buffers.committed_output_lengths is None
                    else buffers.committed_output_lengths.ptr
                ),
            ),
        )
        shape = NativeSpecCycleShape(
            request_count=buffers.request_count,
            request_capacity=buffers.request_count,
            row_count=buffers.rows,
            active_row_count=active_rows,
            row_capacity=buffers.rows,
            candidate_count=buffers.candidate_rows,
            active_candidate_count=active_candidates,
            candidate_capacity=buffers.candidate_rows,
            candidate_budget=draft_depth,
            span_count=spans.live_counts.numel,
            span_capacity=spans.live_counts.numel,
            max_live_count=int(spans.max_live_count),
            context_bucket=int(context_bucket),
            hidden_size=int(hidden_seed_rows.shape[1]),
            hidden_row_capacity=int(hidden_seed_rows.shape[0]),
            output_stride=stride,
            metadata_dtype=_native_dtype(metadata_dtype),
            hidden_dtype=_native_dtype(hidden_seed_rows.dtype),
            kv_dtype=_native_dtype(spans.storage_dtype),
        )
        mode = (
            NativeSpecCycleMode.CHAIN
            if buffers.mode == "verify_chain"
            else NativeSpecCycleMode.TREE
        )
        return cls(
            cycle_id=int(cycle_id),
            transaction_id=0 if buffers.transaction_id is None else int(buffers.transaction_id),
            stages=stages,
            mode=mode,
            shape=shape,
            pointers=pointer_groups,
            stream=int(stream),
            deadline_ns=int(deadline_ns),
        )

    @classmethod
    def from_ctypes(cls, raw: NativeSpecCycleControlC) -> "NativeSpecCycleControl":
        """Validate and reconstruct a Python control descriptor from C bytes."""

        if not isinstance(raw, NativeSpecCycleControlC):
            raise TypeError("raw must be NativeSpecCycleControlC")
        if int(raw.abi_version) != NATIVE_SPEC_CYCLE_ABI_VERSION:
            raise ValueError("native control ABI version mismatch")
        if int(raw.struct_size) != ctypes.sizeof(NativeSpecCycleControlC):
            raise ValueError("native control struct_size mismatch")

        def pointer_group(kind, prefix: str):
            return kind(
                **{
                    field.name: int(getattr(raw, f"{prefix}_{field.name}"))
                    for field in fields(kind)
                }
            )

        return cls(
            cycle_id=int(raw.cycle_id),
            transaction_id=int(raw.transaction_id),
            stages=NativeSpecCycleStage(int(raw.stage_mask)),
            mode=NativeSpecCycleMode(int(raw.mode)),
            shape=NativeSpecCycleShape(
                **{
                    field.name: int(getattr(raw, field.name))
                    for field in fields(NativeSpecCycleShape)
                }
            ),
            pointers=NativeSpecCyclePointers(
                metadata=pointer_group(NativeSpecCycleMetadataPointers, "metadata"),
                kv_live_spans=pointer_group(NativeSpecCycleKVLiveSpanPointers, "kv"),
                state=pointer_group(NativeSpecCycleStatePointers, "state"),
                outputs=pointer_group(NativeSpecCycleOutputPointers, "output"),
            ),
            stream=int(raw.stream),
            deadline_ns=int(raw.deadline_ns),
            abi_version=int(raw.abi_version),
        )

    def to_ctypes(self) -> NativeSpecCycleControlC:
        self.validate()
        values: dict[str, int] = {
            "abi_version": self.abi_version,
            "struct_size": ctypes.sizeof(NativeSpecCycleControlC),
            "stage_mask": int(self.stages),
            "mode": int(self.mode),
            "cycle_id": self.cycle_id,
            "transaction_id": self.transaction_id,
            "stream": self.stream,
            "deadline_ns": self.deadline_ns,
        }
        values.update({field.name: getattr(self.shape, field.name) for field in fields(self.shape)})
        values.update(_flatten_pointers(self.pointers))
        raw = NativeSpecCycleControlC()
        for name, value in values.items():
            setattr(raw, name, value)
        return raw


class NativeSpecCycleResultC(ctypes.Structure):
    """Fixed-width terminal/yield result returned by a native launcher."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("error_code", ctypes.c_uint32),
        ("completed_stage_mask", ctypes.c_uint32),
        ("failed_stage", ctypes.c_uint32),
        ("request_count", ctypes.c_uint32),
        ("visible_output_count", ctypes.c_uint32),
        ("cycle_id", ctypes.c_uint64),
        ("transaction_id", ctypes.c_uint64),
        ("backend_error_code", ctypes.c_int64),
        ("reserved", ctypes.c_uint64),
    ]


@dataclass(frozen=True, slots=True)
class NativeSpecCycleResult:
    cycle_id: int
    transaction_id: int
    status: NativeSpecCycleStatus
    error: NativeSpecCycleError
    completed_stages: NativeSpecCycleStage
    failed_stage: NativeSpecCycleStage = NativeSpecCycleStage(0)
    request_count: int = 0
    visible_output_count: int = 0
    backend_error_code: int = 0
    abi_version: int = NATIVE_SPEC_CYCLE_ABI_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_enum("status", self.status, NativeSpecCycleStatus))
        object.__setattr__(self, "error", _coerce_enum("error", self.error, NativeSpecCycleError))
        object.__setattr__(self, "completed_stages", _coerce_stage_mask(self.completed_stages, allow_empty=True))
        object.__setattr__(self, "failed_stage", _coerce_stage_mask(self.failed_stage, allow_empty=True))
        if self.abi_version != NATIVE_SPEC_CYCLE_ABI_VERSION:
            raise ValueError(f"abi_version must be {NATIVE_SPEC_CYCLE_ABI_VERSION}")
        _check_uint("cycle_id", self.cycle_id, 64)
        _check_uint("transaction_id", self.transaction_id, 64)
        _check_uint("request_count", self.request_count, 32)
        _check_uint("visible_output_count", self.visible_output_count, 32)
        if isinstance(self.backend_error_code, bool) or not isinstance(self.backend_error_code, int):
            raise TypeError("backend_error_code must be an integer")
        if not -(1 << 63) <= self.backend_error_code < (1 << 63):
            raise ValueError("backend_error_code must fit int64")
        if self.failed_stage and int(self.failed_stage) & (int(self.failed_stage) - 1):
            raise ValueError("failed_stage must contain at most one stage")

    @classmethod
    def complete(
        cls,
        control: NativeSpecCycleControl,
        *,
        visible_output_count: int = 0,
    ) -> "NativeSpecCycleResult":
        return cls(
            cycle_id=control.cycle_id,
            transaction_id=control.transaction_id,
            status=NativeSpecCycleStatus.COMPLETE,
            error=NativeSpecCycleError.NONE,
            completed_stages=control.stages,
            request_count=control.shape.request_count,
            visible_output_count=visible_output_count,
        )

    def validate_for(self, control: NativeSpecCycleControl) -> None:
        control.validate()
        self._validate_for_binding(
            cycle_id=control.cycle_id,
            transaction_id=control.transaction_id,
            request_count=control.shape.request_count,
            stages=control.stages,
            output_stride=control.shape.output_stride,
        )

    def _validate_for_binding(
        self,
        *,
        cycle_id: int,
        transaction_id: int,
        request_count: int,
        stages: NativeSpecCycleStage,
        output_stride: int,
    ) -> None:
        """Validate against an already-validated state-bound graph descriptor."""

        if self.cycle_id != cycle_id or self.transaction_id != transaction_id:
            raise ValueError("result cycle_id/transaction_id must match control")
        if self.request_count != request_count:
            raise ValueError("result request_count must match control")
        if int(self.completed_stages) & ~int(stages):
            raise ValueError("result completed stages must be a subset of requested stages")
        if self.status is NativeSpecCycleStatus.COMPLETE and self.completed_stages != stages:
            raise ValueError("COMPLETE result completed stages must equal requested stages")
        if self.status in {NativeSpecCycleStatus.CREATED, NativeSpecCycleStatus.SUBMITTED, NativeSpecCycleStatus.RUNNING}:
            raise ValueError("launcher result must be terminal or yielded")
        if self.status in {NativeSpecCycleStatus.COMPLETE, NativeSpecCycleStatus.YIELDED}:
            if self.error is not NativeSpecCycleError.NONE:
                raise ValueError("successful result cannot carry an error")
            if self.failed_stage:
                raise ValueError("successful result cannot carry failed_stage")
        elif self.status is NativeSpecCycleStatus.CANCELLED:
            if self.error is not NativeSpecCycleError.CANCELLED:
                raise ValueError("CANCELLED result requires CANCELLED error")
        elif self.status is NativeSpecCycleStatus.DEADLINE_EXCEEDED:
            if self.error is not NativeSpecCycleError.DEADLINE_EXCEEDED:
                raise ValueError("DEADLINE_EXCEEDED result requires matching error")
        elif self.status is NativeSpecCycleStatus.FAILED:
            if self.error is NativeSpecCycleError.NONE:
                raise ValueError("failed result requires a non-NONE error")
        max_outputs = request_count * output_stride
        if self.visible_output_count > max_outputs:
            raise ValueError("visible_output_count exceeds the bounded output capacity")

    def to_ctypes(self) -> NativeSpecCycleResultC:
        raw = NativeSpecCycleResultC()
        raw.abi_version = self.abi_version
        raw.struct_size = ctypes.sizeof(NativeSpecCycleResultC)
        raw.status = int(self.status)
        raw.error_code = int(self.error)
        raw.completed_stage_mask = int(self.completed_stages)
        raw.failed_stage = int(self.failed_stage)
        raw.request_count = self.request_count
        raw.visible_output_count = self.visible_output_count
        raw.cycle_id = self.cycle_id
        raw.transaction_id = self.transaction_id
        raw.backend_error_code = self.backend_error_code
        raw.reserved = 0
        return raw

    @classmethod
    def from_ctypes(cls, raw: NativeSpecCycleResultC) -> "NativeSpecCycleResult":
        if not isinstance(raw, NativeSpecCycleResultC):
            raise TypeError("raw must be NativeSpecCycleResultC")
        if int(raw.abi_version) != NATIVE_SPEC_CYCLE_ABI_VERSION:
            raise ValueError("native result ABI version mismatch")
        if int(raw.struct_size) != ctypes.sizeof(NativeSpecCycleResultC):
            raise ValueError("native result struct_size mismatch")
        if int(raw.reserved) != 0:
            raise ValueError("native result reserved field must be zero")
        return cls(
            cycle_id=int(raw.cycle_id),
            transaction_id=int(raw.transaction_id),
            status=NativeSpecCycleStatus(int(raw.status)),
            error=NativeSpecCycleError(int(raw.error_code)),
            completed_stages=NativeSpecCycleStage(int(raw.completed_stage_mask)),
            failed_stage=NativeSpecCycleStage(int(raw.failed_stage)),
            request_count=int(raw.request_count),
            visible_output_count=int(raw.visible_output_count),
            backend_error_code=int(raw.backend_error_code),
            abi_version=int(raw.abi_version),
        )


@runtime_checkable
class NativeSpecCycleLauncher(Protocol):
    def launch(self, control: NativeSpecCycleControl) -> NativeSpecCycleResult:
        """Submit one validated cycle and return a yielded/terminal result."""
        ...


class FakeNativeSpecCycleLauncher:
    """CPU/fake oracle boundary with the same lifecycle as a native launcher."""

    def __init__(
        self,
        executor: Callable[[NativeSpecCycleControl], NativeSpecCycleResult] | None = None,
    ) -> None:
        self._executor = executor
        self._lock = threading.Lock()
        self._history: list[tuple[NativeSpecCycleControl, NativeSpecCycleResult]] = []

    @property
    def history(self) -> tuple[tuple[NativeSpecCycleControl, NativeSpecCycleResult], ...]:
        return tuple(self._history)

    @property
    def launch_count(self) -> int:
        return len(self._history)

    def launch(self, control: NativeSpecCycleControl) -> NativeSpecCycleResult:
        if not isinstance(control, NativeSpecCycleControl):
            raise TypeError("control must be NativeSpecCycleControl")
        control.validate()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("native speculative cycle launcher already in flight")
        try:
            result = (
                NativeSpecCycleResult.complete(control)
                if self._executor is None
                else self._executor(control)
            )
            if not isinstance(result, NativeSpecCycleResult):
                raise TypeError("native speculative cycle executor must return NativeSpecCycleResult")
            result.validate_for(control)
            self._history.append((control, result))
            return result
        finally:
            self._lock.release()


def _native_dtype(dtype: DType) -> NativeSpecCycleDType:
    try:
        return _DTYPE_TO_NATIVE[dtype]
    except KeyError as exc:
        raise ValueError(f"dtype {dtype.value!r} is not representable in native cycle ABI v1") from exc


def _require_contiguous(name: str, tensor: Tensor) -> None:
    if tensor.strides is None:
        return
    expected: list[int] = []
    stride = 1
    for dim in reversed(tensor.shape):
        expected.append(stride)
        stride *= int(dim)
    if tensor.strides != tuple(reversed(expected)):
        raise ValueError(f"{name} must be a contiguous tensor view")


def _flatten_pointers(pointers: NativeSpecCyclePointers) -> dict[str, int]:
    values: dict[str, int] = {}
    for prefix, group in (
        ("metadata", pointers.metadata),
        ("kv", pointers.kv_live_spans),
        ("state", pointers.state),
        ("output", pointers.outputs),
    ):
        for field in fields(group):
            values[f"{prefix}_{field.name}"] = getattr(group, field.name)
    return values


def _check_uint(name: str, value: int, bits: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    maximum = _UINT32_MAX if bits == 32 else _UINT64_MAX
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must fit uint{bits}")


def _check_pointer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer raw pointer")
    if value < 0 or value > _UINT64_MAX:
        raise ValueError(f"{name} must fit uint64")


def _require_pointer(name: str, value: int) -> None:
    if value == 0:
        raise ValueError(f"{name} must be a non-zero borrowed pointer")


def _check_pair(name: str, first: int, second: int) -> None:
    if bool(first) != bool(second):
        raise ValueError(f"{name} must be both zero or both non-zero")


def _coerce_enum(name: str, value, kind):
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc


def _coerce_stage_mask(value, *, allow_empty: bool = False) -> NativeSpecCycleStage:
    if isinstance(value, bool) or not isinstance(value, (int, NativeSpecCycleStage)):
        raise TypeError("stage mask must be an integer")
    raw = int(value)
    if raw & ~int(_ALL_STAGES):
        raise ValueError("stage mask contains unknown bits")
    if raw == 0 and not allow_empty:
        raise ValueError("stage mask must not be empty")
    return NativeSpecCycleStage(raw)


def _validate_stage_dependencies(stages: NativeSpecCycleStage) -> None:
    if stages & NativeSpecCycleStage.ACCEPT and not stages & NativeSpecCycleStage.VERIFY:
        raise ValueError("ACCEPT requires VERIFY")
    if stages & NativeSpecCycleStage.COMMIT and not stages & NativeSpecCycleStage.ACCEPT:
        raise ValueError("COMMIT requires ACCEPT")
    if stages & NativeSpecCycleStage.UPDATE_CURSORS and not stages & NativeSpecCycleStage.COMMIT:
        raise ValueError("UPDATE_CURSORS requires COMMIT")


__all__ = [
    "NATIVE_SPEC_CYCLE_ABI_VERSION",
    "FakeNativeSpecCycleLauncher",
    "NativeSpecCycleControl",
    "NativeSpecCycleControlC",
    "NativeSpecCycleDType",
    "NativeSpecCycleError",
    "NativeSpecCycleKVLiveSpanPointers",
    "NativeSpecCycleLauncher",
    "NativeSpecCycleMetadataPointers",
    "NativeSpecCycleMode",
    "NativeSpecCycleOutputPointers",
    "NativeSpecCyclePointers",
    "NativeSpecCycleResult",
    "NativeSpecCycleResultC",
    "NativeSpecCycleShape",
    "NativeSpecCycleStage",
    "NativeSpecCycleStatePointers",
    "NativeSpecCycleStatus",
]
