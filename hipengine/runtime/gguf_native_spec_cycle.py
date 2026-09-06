"""GGUF adapter for reusable B1-B3 native speculative target graphs.

The adapter captures two- through four-row target buckets against stable
session/device addresses, binds each capture to a versioned
:class:`NativeSpecCycleControl`, and submits it through one C++ launcher call.
N1 leaves proposal and host acceptance/commit unchanged; N2 can fold strict
acceptance, selected state/hidden commit, and cursor update into the same graph.

Unsupported shapes and capture-unsafe session configurations use the existing
Python verifier when ``fallback=True``.  Launch or correctness failures never
fall back silently because the graph may already have mutated state/KV.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
import os
import time
from typing import Any, Callable, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.convert import f32_to_bf16
from hipengine.kernels.hip_gfx1100.runtime import unpack_verify_chain_dynamic_metadata_i64
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.hip_gfx1100.speculative import (
    ACCEPT_PACKED_PAYLOAD_FIELDS,
    build_dflash_accept,
    build_dflash_commit,
)
from hipengine.kernels.registry import resolve
from hipengine.kvcache import KVLiveSpans
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative.buffers import TargetVerifyBufferOwner, TargetVerifyBufferSpec
from hipengine.speculative.interfaces import (
    DraftBatch,
    TargetStateCommitBuffers,
    TargetVerifyBatch,
    TargetVerifyBuffers,
)
from hipengine.speculative.native_cycle import (
    NativeSpecCycleControl,
    NativeSpecCycleResult,
    NativeSpecCycleStage,
    NativeSpecCycleStatus,
)
# Importing the provider registers the four-axis launcher factory.  Runtime
# dispatch below still resolves by key rather than branching on backend.
from hipengine.speculative.native_cycle_graph import NativeSpecTargetGraphLauncher  # noqa: F401


class NativeSpecTargetGraphUnsupportedError(RuntimeError):
    """The fixed native bucket cannot safely represent this verifier invocation."""


NATIVE_SPEC_TARGET_GRAPH_CONTEXT_BUCKET_MISS = "target_graph_context_bucket_miss"
NATIVE_SPEC_TARGET_GRAPH_OUTPUT_ROOM_MISS = "target_graph_output_room_miss"


def _call_with_f32_verifier_disabled(
    enabled: bool,
    callback: Callable[[], Any],
) -> Any:
    if not enabled:
        return callback()
    names = (
        "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL",
        "HIPENGINE_GGUF_VERIFY_F32_POST_NORM",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "0"
        return callback()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


_COMPLETE_CYCLE_STAGES = (
    NativeSpecCycleStage.PROPOSE
    | NativeSpecCycleStage.VERIFY
    | NativeSpecCycleStage.ACCEPT
    | NativeSpecCycleStage.COMMIT
    | NativeSpecCycleStage.UPDATE_CURSORS
)


@dataclass(frozen=True)
class Qwen35GGUFNativeCompleteCycleResult:
    """Scheduler-facing payload from one GGUF proposal/verify/commit call.

    N3 keeps the N2 target graph as the state-mutating primitive, but moves the
    strict NextN proposal, MTP-KV rollback/accepted-row repair, GGUF reseed, and
    cursor/result accounting behind one public provider-adapter boundary.
    """

    draft_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    accepted_draft_tokens: int
    start_position: int
    end_position: int
    draft_cache_len_before: int
    draft_cache_len_after: int
    target_result: Any
    proposal_wall_ms: float
    target_wall_ms: float
    mtp_kv_commit_wall_ms: float
    call_wall_ms: float
    proposal_native_graph: bool = False
    completed_stages: NativeSpecCycleStage = _COMPLETE_CYCLE_STAGES
    complete_native_cycle: bool = True

    def __post_init__(self) -> None:
        drafts = len(self.draft_token_ids)
        if drafts not in {1, 2, 3, 4, 5, 6, 7}:
            raise ValueError("native complete cycle requires a B1-B7 draft chain")
        if self.accepted_draft_tokens < 0 or self.accepted_draft_tokens > drafts:
            raise ValueError("accepted_draft_tokens is outside the draft chain")
        if len(self.output_token_ids) != self.accepted_draft_tokens + 1:
            raise ValueError("output_token_ids must contain accepted drafts plus one correction")
        if self.end_position != self.start_position + len(self.output_token_ids):
            raise ValueError("complete-cycle cursor must advance by visible output count")
        if self.draft_cache_len_before < 0:
            raise ValueError("draft_cache_len_before must be non-negative")
        expected_cache_len = self.draft_cache_len_before + 1 + self.accepted_draft_tokens
        if self.draft_cache_len_after != expected_cache_len:
            raise ValueError("draft cache must retain the root row plus accepted draft rows")
        if self.completed_stages != _COMPLETE_CYCLE_STAGES:
            raise ValueError("native complete-cycle result must report every N3 stage")
        for name in (
            "proposal_wall_ms",
            "target_wall_ms",
            "mtp_kv_commit_wall_ms",
            "call_wall_ms",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class Qwen35GGUFNativeAcceptCommitResult:
    """Bounded host payload after device-resident N2 accept and state commit."""

    input_token_ids: list[int]
    token_ids: list[int]
    accepted_draft_tokens: int
    commit_row: int
    commit_token: int
    commit_position: int
    next_token: int
    full_accept: bool
    start_position: int
    end_position: int
    hidden_seed_rows_ptr: int
    hidden_seed_row_count: int
    hidden_size: int
    target_top1: list[int] = dataclass_field(default_factory=list)
    proposal_top1_values: tuple[float, ...] = ()
    proposal_device_handoff: bool = False
    verify_buffers: TargetVerifyBuffers | None = None
    state_commit_buffers: TargetStateCommitBuffers | None = None
    linear_state_rows_captured: bool = True
    final_linear_state_committed: bool = True
    device_accept_commit: bool = True
    hidden_seeds: np.ndarray = dataclass_field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    lm_head_logits_f32: np.ndarray | None = None
    compact_result: bool = False

    def __post_init__(self) -> None:
        rows = len(self.input_token_ids)
        if rows not in {2, 3, 4, 5, 6, 7, 8}:
            raise ValueError("native accept/commit result requires B1-B7 input rows")
        if self.accepted_draft_tokens < 0 or self.accepted_draft_tokens >= rows:
            raise ValueError("accepted_draft_tokens is outside the target bucket")
        if len(self.token_ids) != self.accepted_draft_tokens + 1:
            raise ValueError("token_ids must contain accepted drafts plus one correction")
        if self.commit_row != self.accepted_draft_tokens:
            raise ValueError("strict-chain commit_row must equal accepted_draft_tokens")
        expected_commit_token = (
            self.input_token_ids[self.commit_row]
            if not self.compact_result
            else (
                self.input_token_ids[0]
                if self.commit_row == 0
                else self.token_ids[self.commit_row - 1]
            )
        )
        if self.commit_token != expected_commit_token:
            raise ValueError("commit_token must match the selected target input row")
        if self.commit_position != self.start_position + self.commit_row:
            raise ValueError("commit_position must match the selected target row position")
        if self.end_position != self.commit_position + 1:
            raise ValueError("end_position must be the next cursor after the committed row")
        if self.next_token != self.token_ids[-1]:
            raise ValueError("next_token must be the final visible correction")
        if self.hidden_seed_rows_ptr <= 0 or self.hidden_seed_row_count != rows:
            raise ValueError("native accept/commit result requires all device hidden rows")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.target_top1 and (
            len(self.target_top1) != rows or any(token < 0 for token in self.target_top1)
        ):
            raise ValueError("target_top1 must contain one non-negative token per target row")
        if self.proposal_device_handoff:
            if not self.compact_result and len(self.proposal_top1_values) != rows - 1:
                raise ValueError("device handoff must return one proposal value per candidate")
            if self.proposal_top1_values and not np.all(
                np.isfinite(np.asarray(self.proposal_top1_values, dtype=np.float32))
            ):
                raise ValueError("device handoff proposal values must be finite")
        elif self.proposal_top1_values:
            raise ValueError("proposal values require a device-handoff result")
        if self.verify_buffers is not None and self.verify_buffers.rows != rows:
            raise ValueError("native verify buffers must match the target row count")
        if (
            self.state_commit_buffers is not None
            and self.state_commit_buffers.request_ids
            != (() if self.verify_buffers is None else self.verify_buffers.request_ids)
        ):
            raise ValueError("native verify/state buffer request ids must match")
        if self.lm_head_logits_f32 is not None:
            logits = np.asarray(self.lm_head_logits_f32)
            if (
                logits.dtype != np.float32
                or logits.ndim != 2
                or logits.shape[0] != rows
                or not np.isfinite(logits).all()
            ):
                raise ValueError(
                    "diagnostic lm_head_logits_f32 must contain finite FP32 target rows"
                )
        if self.hidden_seeds.shape != (0, 0):
            if (
                self.lm_head_logits_f32 is None
                or self.hidden_seeds.dtype != np.float32
                or self.hidden_seeds.shape != (rows, self.hidden_size)
                or not np.isfinite(self.hidden_seeds).all()
            ):
                raise ValueError(
                    "diagnostic host hidden rows require aligned finite FP32 logits"
                )
        elif self.hidden_seeds.dtype != np.float32:
            raise ValueError("device accept/commit empty hidden rows must be FP32")


def build_native_b2_target_batch(
    input_token_ids: Sequence[int],
    *,
    start_position: int,
    request_id: int = 0,
) -> TargetVerifyBatch:
    """Build provider-neutral root+candidate metadata for a B1-B7 chain."""

    tokens = tuple(int(token) for token in input_token_ids)
    start = int(start_position)
    request = int(request_id)
    if len(tokens) not in {2, 3, 4, 5, 6, 7, 8}:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph requires two to eight rows (one root plus B1-B7)"
        )
    if start < 0:
        raise ValueError("start_position must be non-negative")
    if request < 0:
        raise ValueError("request_id must be non-negative")
    candidate_count = len(tokens) - 1
    draft = DraftBatch(
        request_ids=(request,),
        candidate_tokens=tokens[1:],
        parent_positions=tuple(start + depth - 1 for depth in range(1, candidate_count + 1)),
        draft_depths=tuple(range(1, candidate_count + 1)),
        row_to_request=tuple(request for _ in range(candidate_count)),
        mode="verify_chain",
    )
    return TargetVerifyBatch.from_draft(
        draft,
        root_tokens=tokens[:1],
        root_positions=(start,),
    )


def _tensor_buffer(tensor: Tensor) -> DeviceBuffer:
    return DeviceBuffer(tensor.ptr, tensor.numel * tensor.dtype.itemsize)


def _copy_array_to_tensor(tensor: Tensor, values: np.ndarray, *, runtime) -> None:
    array = np.ascontiguousarray(values)
    expected_nbytes = tensor.numel * tensor.dtype.itemsize
    if array.nbytes != expected_nbytes:
        raise ValueError(
            f"host metadata byte size {array.nbytes} does not match tensor capacity {expected_nbytes}"
        )
    copy_host_to_device(
        _tensor_buffer(tensor),
        host_array_ptr(array),
        array.nbytes,
        runtime=runtime,
    )


def _allocate_native_linear_state_tables(
    workspace: RuntimeWorkspace,
    session: Any,
    *,
    runtime: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Snapshot mutable session pointer tables into graph-owned storage."""

    table_specs = (
        (
            "native_spec_linear_state_src_conv_table",
            session._verify_linear_state_src_conv_host,
        ),
        (
            "native_spec_linear_state_src_recurrent_table",
            session._verify_linear_state_src_recurrent_host,
        ),
        (
            "native_spec_linear_state_dst_conv_table",
            session._verify_linear_state_dst_conv_host,
        ),
        (
            "native_spec_linear_state_dst_recurrent_table",
            session._verify_linear_state_dst_recurrent_host,
        ),
    )
    if any(values is None for _name, values in table_specs):
        raise RuntimeError("GGUF native target graph pointer tables are incomplete")
    graph_tables: list[Tensor] = []
    for name, values in table_specs:
        assert values is not None
        array = np.ascontiguousarray(values, dtype=np.uint64)
        table = workspace.reserve_tensor(name, (array.size,), DType.INT64)
        _copy_array_to_tensor(table, array, runtime=runtime)
        graph_tables.append(table)
    return (
        graph_tables[0],
        graph_tables[1],
        graph_tables[2],
        graph_tables[3],
    )


def _stage_target_batch(
    batch: TargetVerifyBatch,
    buffers: TargetVerifyBuffers,
    *,
    runtime,
) -> None:
    integer = np.int64 if buffers.token_ids.dtype is DType.INT64 else np.int32
    for tensor, values in (
        (buffers.token_ids, batch.tokens),
        (buffers.positions, batch.positions),
        (buffers.parent_rows, batch.parent_rows),
        (buffers.draft_depths, batch.draft_depths),
        (buffers.row_to_request, batch.row_to_request),
    ):
        _copy_array_to_tensor(tensor, np.asarray(values, dtype=integer), runtime=runtime)
    _copy_array_to_tensor(
        buffers.active_mask,
        np.asarray(batch.active_mask, dtype=np.bool_),
        runtime=runtime,
    )
    runtime.memset(buffers.target_top1.ptr, 0, buffers.target_top1.numel * buffers.target_top1.dtype.itemsize)


def _context_bucket(max_live_count: int) -> int:
    value = max(1, int(max_live_count))
    return 1 << (value - 1).bit_length()


_NATIVE_TARGET_SHORT_CONTEXT_LIMIT = 1023
_NATIVE_TARGET_GRAPH_CACHE_MAX_ENTRIES = 8


def _native_target_graph_context_limit(session: Any, *, rows: int) -> int | None:
    """Select the immutable target-graph context bucket for one live cycle."""

    rows = int(rows)
    end = int(getattr(session, "position", 0)) + rows
    scratch = getattr(session, "scratch", None)
    if scratch is None:
        # Fake/legacy owners model the retained short graph only.
        return _NATIVE_TARGET_SHORT_CONTEXT_LIMIT
    capacity = int(getattr(scratch, "max_positions", 0))
    if capacity <= 0:
        return _NATIVE_TARGET_SHORT_CONTEXT_LIMIT
    if rows <= 0 or end > capacity:
        return None
    graph_context_limit = int(
        backend_package_capability(
            str(getattr(session, "backend", "")),
            "GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT",
            capacity,
        )
    )
    if end > graph_context_limit:
        return None
    if end < 1024:
        return min(_NATIVE_TARGET_SHORT_CONTEXT_LIMIT, capacity)
    start = int(getattr(session, "position", 0))
    # A reusable graph must not force one attention schedule across rows that
    # straddle the short/split-K transition or a split-workspace boundary.
    if start < 1024:
        return None
    block_size = int(getattr(scratch, "block_size", 256))
    first_active_context = start + 1
    first_split_count = (first_active_context + block_size - 1) // block_size
    last_split_count = (end + block_size - 1) // block_size
    if first_split_count != last_split_count:
        return None
    context_limit = min(last_split_count * block_size, capacity)
    from hipengine.runtime import qwen35_gguf_runner as runner_module

    runner = getattr(session, "runner", None)
    weights = getattr(runner, "weights", None)
    config = getattr(weights, "config", None)
    if config is not None:
        def split_kernel(active_context: int):
            split_count = (int(active_context) + block_size - 1) // block_size
            return runner_module._gguf_full_attention_split_gate_bf16_fn(
                config,
                backend=str(getattr(session, "backend", "")),
                block_size=block_size,
                num_splits=split_count,
                active_context=int(active_context),
            )

        first_kernel = split_kernel(first_active_context)
        if any(
            split_kernel(active_context) is not first_kernel
            for active_context in range(first_active_context + 1, end + 1)
        ):
            return None
        while context_limit > end and split_kernel(context_limit) is not first_kernel:
            context_limit -= 1
        if context_limit < end:
            return None

    if not runner_module._gguf_prefill_device_metadata_enabled(
        backend=str(getattr(session, "backend", "")),
        prompt_tokens=context_limit,
    ):
        return None
    return context_limit


def _native_target_graph_cache(session: Any) -> dict[tuple[int, bool, int], Any]:
    cache = getattr(session, "_native_spec_target_graphs", None)
    if isinstance(cache, dict):
        return cache
    cache = {}
    setattr(session, "_native_spec_target_graphs", cache)
    return cache


def _cache_native_target_graph(
    session: Any,
    key: tuple[int, bool, int],
    graph: Any,
) -> None:
    cache = _native_target_graph_cache(session)
    cache.pop(key, None)
    cache[key] = graph
    while len(cache) > _NATIVE_TARGET_GRAPH_CACHE_MAX_ENTRIES:
        _evicted_key, evicted = next(iter(cache.items()))
        cache.pop(_evicted_key, None)
        if evicted is graph:
            continue
        close = getattr(evicted, "close", None)
        if callable(close):
            close()


def _dynamic_target_scratch(session: Any, buffers: TargetVerifyBuffers, *, rows: int, context_limit: int):
    """Bind verifier rows to fixed scratch addresses and live device metadata."""

    base = session._bulk_prefill_scratch
    if base is None:
        raise RuntimeError("GGUF resident bulk prefill scratch is closed")
    rows = int(rows)
    context_limit = int(context_limit)
    if rows <= 0 or rows > int(base.rows):
        raise ValueError("dynamic target rows exceed resident bulk scratch capacity")
    if context_limit <= rows or context_limit > int(base.max_positions):
        raise ValueError("dynamic target context limit is outside resident scratch capacity")
    device = base.positions_tensor.device
    block_table = Tensor.from_handle(
        base.block_table.ptr,
        (rows, int(base.blocks)),
        DType.INT32,
        device,
    )
    positions = Tensor.from_handle(base.positions.ptr, (rows,), DType.INT64, device)
    contexts = Tensor.from_handle(base.context_counts.ptr, (rows,), DType.INT64, device)
    append_spans = KVLiveSpans.paged_uniform(
        block_table=block_table,
        live_counts=positions,
        max_live_count=context_limit - 1,
        storage_dtype=DType.BF16,
        row_positions=positions,
        span_role="verify_chain",
    )
    prefill_spans = KVLiveSpans.paged_uniform(
        block_table=block_table,
        live_counts=contexts,
        max_live_count=context_limit,
        storage_dtype=DType.BF16,
        row_positions=positions,
        span_role="verify_chain",
    )
    dynamic_buffers = replace(buffers, positions=positions)
    dynamic_scratch = replace(
        base,
        start=0,
        rows=rows,
        block_table_tensor=block_table,
        positions_tensor=positions,
        context_counts_tensor=contexts,
        append_spans=append_spans,
        prefill_spans=prefill_spans,
        gdn_active_segments=1,
        metadata_prepare_path="native_spec_dynamic",
    )
    return dynamic_buffers, dynamic_scratch


def _dynamic_native_decode_row_scratches(
    session: Any,
    dynamic_scratch: Any,
    *,
    rows: int,
    start_position: int,
    context_limit: int,
) -> tuple[Any, ...]:
    """Build scalar-attention views over graph-updated row metadata.

    Native verification deliberately keeps attention token-serial so its
    arithmetic and recurrent-state transitions match c=1 exactly.  During graph
    replay, however, no host position upload may be captured.  Each view below
    reuses the stable scalar decode temporaries/state/cache while pointing its
    position and ``KVLiveSpans`` ABI at one dynamically unpacked verifier row.
    """

    base = getattr(session, "scratch", None)
    if base is None:
        raise RuntimeError("GGUF resident decode scratch is closed")
    rows = int(rows)
    start = int(start_position)
    context_limit = int(context_limit)
    blocks = int(base.blocks_per_slot)
    if rows <= 0 or rows > int(dynamic_scratch.rows):
        raise ValueError("native graph row views exceed dynamic scratch capacity")
    if blocks <= 0 or blocks * int(base.block_size) < context_limit:
        raise ValueError("native graph row block table does not cover the context bucket")
    device = base.block_table_tensor.device
    block_row_nbytes = blocks * DType.INT32.itemsize
    if int(base.block_table.nbytes) < block_row_nbytes:
        raise ValueError("resident decode block table is smaller than its declared capacity")
    result = []
    for row in range(rows):
        # Every verifier row extends the same resident request, so page
        # indirection comes from the target slot rather than the temporary bulk
        # scratch's synthetic 0..N table.
        block_table = DeviceBuffer(int(base.block_table.ptr), block_row_nbytes)
        position_buf = DeviceBuffer(
            int(dynamic_scratch.positions.ptr) + row * DType.INT64.itemsize,
            DType.INT64.itemsize,
        )
        context_buf = DeviceBuffer(
            int(dynamic_scratch.context_counts.ptr) + row * DType.INT64.itemsize,
            DType.INT64.itemsize,
        )
        block_table_tensor = Tensor.from_handle(
            block_table.ptr,
            (blocks,),
            DType.INT32,
            device,
        )
        position_tensor = Tensor.from_handle(
            position_buf.ptr,
            (1,),
            DType.INT64,
            device,
        )
        context_tensor = Tensor.from_handle(
            context_buf.ptr,
            (1,),
            DType.INT64,
            device,
        )
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=position_tensor,
            max_live_count=context_limit - 1,
            storage_dtype=DType.BF16,
            row_positions=position_tensor,
            span_role="verify_chain",
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=context_tensor,
            max_live_count=context_limit,
            storage_dtype=DType.BF16,
            row_positions=position_tensor,
            span_role="verify_chain",
        )
        captured_position = start + row
        result.append(
            replace(
                base,
                block_table=block_table,
                position_buf=position_buf,
                context_buf=context_buf,
                block_table_tensor=block_table_tensor,
                position_tensor=position_tensor,
                context_tensor=context_tensor,
                append_spans=append_spans,
                decode_spans=decode_spans,
                position_host=np.asarray([captured_position], dtype=np.int64),
                context_host=np.asarray([captured_position + 1], dtype=np.int64),
                slot_count=1,
                blocks_per_slot=blocks,
            )
        )
    return tuple(result)


def _pack_dynamic_metadata(batch: TargetVerifyBatch) -> np.ndarray:
    rows = []
    for token, position in zip(batch.tokens, batch.positions, strict=True):
        token_i64 = int(token)
        position_i64 = int(position)
        if token_i64 < np.iinfo(np.int32).min or token_i64 > np.iinfo(np.int32).max:
            raise ValueError("native target token id must fit int32")
        if position_i64 < 0 or position_i64 > np.iinfo(np.int32).max:
            raise ValueError("native target position must fit non-negative int32")
        rows.append(
            (
                token_i64,
                token_i64,
                position_i64,
                position_i64,
                position_i64 + 1,
            )
        )
    return np.ascontiguousarray(rows, dtype=np.int64)


def _stage_dynamic_metadata(tensor: Tensor, batch: TargetVerifyBatch, *, runtime) -> None:
    _copy_array_to_tensor(tensor, _pack_dynamic_metadata(batch), runtime=runtime)


def _enqueue_device_proposal_handoff(
    *,
    runtime: Any,
    stream: int,
    dynamic_metadata_ptr: int,
    result_payload_ptr: int,
    proposal_result_ptr: int,
    proposal_result_nbytes: int,
    proposal_event: int,
    proposal_budget: int,
    target_rows: int,
    copy_result_payload: bool = True,
) -> None:
    """Wait for proposal IDs and inject both i64/i32 metadata source columns."""

    runtime.stream_wait_event(int(stream), int(proposal_event))
    metadata_row_nbytes = 5 * DType.INT64.itemsize
    result_row_nbytes = 2 * DType.INT32.itemsize
    for depth in range(int(proposal_budget)):
        proposal_token_ptr = int(proposal_result_ptr) + depth * result_row_nbytes
        metadata_token_ptr = (
            int(dynamic_metadata_ptr) + (depth + 1) * metadata_row_nbytes
        )
        # The unpack kernel reads distinct source columns for its i64 embedding
        # IDs and i32 acceptance IDs. Both must receive the exact proposal ID.
        for token_column in (0, DType.INT64.itemsize):
            runtime.memcpy_async(
                metadata_token_ptr + token_column,
                proposal_token_ptr,
                DType.INT32.itemsize,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )
    if copy_result_payload:
        proposal_payload_start = ACCEPT_PACKED_PAYLOAD_FIELDS + 1 + 2 * int(target_rows)
        runtime.memcpy_async(
            int(result_payload_ptr)
            + proposal_payload_start * DType.INT32.itemsize,
            int(proposal_result_ptr),
            int(proposal_result_nbytes),
            HipMemcpyKind.DEVICE_TO_DEVICE,
            int(stream),
        )


def _native_target_configuration_key(
    *,
    bulk_attention_mode: str,
    use_wmma_prefill: bool,
    capture_linear_state_rows: bool,
    capture_pre_output_norm_hidden: bool,
    defer_linear_state_commit: bool,
    device_accept_commit: bool,
    execution_profile_manifest_sha256: str = "legacy",
    recurrent_state_dtype: str = "fp32",
) -> tuple[object, ...]:
    env = tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("HIPENGINE_")
        )
    )
    manifest_hash = str(execution_profile_manifest_sha256 or "legacy")
    state_dtype = str(recurrent_state_dtype).strip().lower()
    if state_dtype not in {"fp16", "fp32"}:
        raise ValueError("native target recurrent_state_dtype must be fp16 or fp32")
    return (
        str(bulk_attention_mode),
        bool(use_wmma_prefill),
        bool(capture_linear_state_rows),
        bool(capture_pre_output_norm_hidden),
        bool(defer_linear_state_commit),
        bool(device_accept_commit),
        manifest_hash,
        state_dtype,
        env,
    )


def _native_target_execution_identity(session: Any) -> tuple[str, str]:
    manifest_hash = str(
        getattr(session, "_specdec2_execution_profile_manifest_sha256", "legacy")
        or "legacy"
    )
    state_dtype = str(
        getattr(session, "_specdec2_recurrent_state_dtype", "") or ""
    ).strip().lower()
    if not state_dtype:
        state_dtype = (
            "fp16"
            if bool(getattr(getattr(session, "runner", None), "fp16_recurrent_state", False))
            else "fp32"
        )
    return manifest_hash, state_dtype


def _native_target_binding_signature(session: Any) -> tuple[int, ...]:
    """Fingerprint every mutable allocation whose address is captured."""

    pointers: list[int] = []

    def add(value: Any) -> None:
        ptr = getattr(value, "ptr", None)
        if ptr is not None and int(ptr) > 0:
            pointers.append(int(ptr))

    for name in (
        "_prefill_hidden_a",
        "_prefill_hidden_b",
        "_verify_lm_out_indices_i32",
        "_lm_out_index",
    ):
        add(getattr(session, name, None))
    scratch = getattr(session, "scratch", None)
    if scratch is not None:
        for name in (
            "position_buf",
            "context_buf",
            "hidden_seed_fp32",
            "cos_table",
            "sin_table",
        ):
            add(getattr(scratch, name, None))
        for name in (
            "layer_conv_states",
            "layer_recurrent_states",
            "full_key_caches",
            "full_value_caches",
        ):
            for value in getattr(scratch, name, ()):
                add(value)
    bulk = getattr(session, "_bulk_prefill_scratch", None)
    if bulk is not None:
        for value in vars(bulk).values():
            add(value)
    for name in (
        "_verify_linear_conv_state_rows",
        "_verify_linear_recurrent_state_rows",
        "_verify_linear_conv_initial_snapshots",
        "_verify_linear_recurrent_initial_snapshots",
    ):
        if "initial_snapshots" in name and int(
            getattr(session, "_verify_linear_initial_snapshot_users", 0)
        ) <= 0:
            continue
        for value in getattr(session, name, ()):
            add(value)
    return tuple(pointers)


def _validate_capture_admission(
    session: Any,
    input_token_ids: Sequence[int],
    *,
    context_limit: int,
    bulk_attention_mode: str,
    use_wmma_prefill: bool,
    capture_lm_head_logits: bool,
    record_stage_timings: bool,
    sync_stage_timings: bool,
) -> None:
    from hipengine.runtime.qwen35_gguf_runner import (
        _gguf_prefill_device_metadata_enabled,
        _gguf_verify_f32_residual_enabled,
        _gguf_verify_f32_token_embedding_enabled,
    )

    rows = len(input_token_ids)
    if rows not in {2, 3, 4, 5, 6, 7, 8}:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph requires two to eight rows (one root plus B1-B7)"
        )
    if bulk_attention_mode not in {"bulk", "native"}:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph requires bulk or exact native attention scheduling"
        )
    if use_wmma_prefill:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 supports only the non-WMMA small-row verifier"
        )
    if sync_stage_timings:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 does not support synchronized stage timings"
        )
    # Full logits are already produced in the resident verifier buffer.  A
    # diagnostic caller may copy them after graph completion without changing
    # capture topology or the default hot path.
    _ = capture_lm_head_logits
    # The outer cycle may time capture+submission wall, but capture-time Python
    # dispatch intervals are not replay stage timings and are intentionally not
    # reported through ``last_verify_stage_timings_ms``.
    _ = record_stage_timings
    if getattr(session, "runner", None) is None or getattr(session, "scratch", None) is None:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph requires an open GGUF resident session"
        )
    if bool(getattr(session, "host_token_embedding_enabled", False)):
        fallback = getattr(session, "_device_token_embedding_weight", None)
        if not callable(fallback):
            raise NativeSpecTargetGraphUnsupportedError(
                "native target graph N1 requires device-resident token embedding"
            )
        fallback(reason="native_mtp_graph")
    if bool(getattr(session, "use_expert_sidecar", False)):
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 requires resident replacement expert layouts"
        )
    if DType.parse(getattr(session, "kv_storage_dtype", DType.BF16)) is not DType.BF16:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 requires BF16 KV storage"
        )
    if resolve(
        backend=str(session.backend),
        layer="speculative_cycle",
        quant="w4_gguf",
        variant="native_v1_b2_target_graph",
        missing="none",
    ) is None:
        raise NativeSpecTargetGraphUnsupportedError(
            f"native target graph N1 is not registered for backend {session.backend!r}"
        )
    end = int(session.position) + rows
    context_limit = int(context_limit)
    if (
        context_limit <= rows
        or end > context_limit
        or context_limit > int(session.scratch.max_positions)
    ):
        raise NativeSpecTargetGraphUnsupportedError(
            NATIVE_SPEC_TARGET_GRAPH_CONTEXT_BUCKET_MISS
        )
    if context_limit >= 1024 and bulk_attention_mode != "native":
        raise NativeSpecTargetGraphUnsupportedError(
            "long-context native target graphs require split-K native attention"
        )
    if not _gguf_prefill_device_metadata_enabled(
        backend=str(session.backend),
        prompt_tokens=context_limit,
    ):
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 requires stream-ordered device metadata preparation"
        )
    if _gguf_verify_f32_residual_enabled() and _gguf_verify_f32_token_embedding_enabled():
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 cannot capture host F32 token-embedding staging"
        )


@dataclass
class Qwen35GGUFNativeB2TargetGraph:
    """Reusable fixed B1-B3 verifier graph with device-driven metadata."""

    session: Any
    graph: int
    graph_exec: int
    stream: int
    launcher: Any
    workspace: RuntimeWorkspace
    batch: TargetVerifyBatch
    buffers: TargetVerifyBuffers
    control: NativeSpecCycleControl
    dynamic_scratch: Any
    hidden_seed_rows: Tensor
    hidden_f32_a: Tensor
    hidden_f32_b: Tensor
    pre_output_norm_hidden_rows: Tensor | None
    pre_output_norm_hidden_bf16_rows: Tensor | None
    selected_pre_output_norm_hidden_bf16: Tensor | None
    dynamic_metadata: Tensor
    token_ids_i32: Tensor
    positions_i32: Tensor
    linear_state_src_conv_table: Tensor | None
    linear_state_src_recurrent_table: Tensor | None
    linear_state_dst_conv_table: Tensor | None
    linear_state_dst_recurrent_table: Tensor | None
    accept_buffers: TargetVerifyBuffers | None
    remaining_decode: Tensor | None
    result_payload: Tensor | None
    visible_output_ids: Tensor | None
    visible_output_lengths: Tensor | None
    target_top1_payload: Tensor | None
    pre_output_commit_buffers: TargetStateCommitBuffers | None
    accept_library: Any | None
    commit_library: Any | None
    device_accept_commit: bool
    start_position: int
    end_position: int
    context_limit: int
    rows: int
    capture_linear_state_rows: bool
    capture_pre_output_norm_hidden: bool
    defer_linear_state_commit: bool
    configuration_key: tuple[object, ...]
    binding_signature: tuple[int, ...]
    capture_wall_ms: float
    capture_reported: bool = False
    closed: bool = False
    native_result: NativeSpecCycleResult | None = None

    def compatible_with(
        self,
        session: Any,
        *,
        context_limit: int,
        bulk_attention_mode: str,
        use_wmma_prefill: bool,
        capture_linear_state_rows: bool,
        capture_pre_output_norm_hidden: bool,
        defer_linear_state_commit: bool,
        device_accept_commit: bool,
    ) -> bool:
        if self.closed or session is not self.session:
            return False
        manifest_hash, state_dtype = _native_target_execution_identity(session)
        expected_key = _native_target_configuration_key(
            bulk_attention_mode=bulk_attention_mode,
            use_wmma_prefill=use_wmma_prefill,
            capture_linear_state_rows=capture_linear_state_rows,
            capture_pre_output_norm_hidden=capture_pre_output_norm_hidden,
            defer_linear_state_commit=defer_linear_state_commit,
            device_accept_commit=device_accept_commit,
            execution_profile_manifest_sha256=manifest_hash,
            recurrent_state_dtype=state_dtype,
        )
        return (
            int(context_limit) == int(self.context_limit)
            and expected_key == self.configuration_key
            and _native_target_binding_signature(session) == self.binding_signature
        )

    def launch_ineligibility_reason(
        self,
        session: Any,
        *,
        position: int,
        rows: int,
        remaining_decode: int | None,
        bulk_attention_mode: str,
        use_wmma_prefill: bool,
        capture_linear_state_rows: bool,
        capture_pre_output_norm_hidden: bool,
        defer_linear_state_commit: bool,
        device_accept_commit: bool,
    ) -> str | None:
        """Return a stable pre-launch rejection reason for this cached graph."""

        if self.closed:
            return "target_graph_closed"
        if session is not self.session:
            return "target_graph_session_miss"
        if int(rows) != int(self.rows):
            return "target_graph_row_shape_miss"
        manifest_hash, state_dtype = _native_target_execution_identity(session)
        expected_key = _native_target_configuration_key(
            bulk_attention_mode=bulk_attention_mode,
            use_wmma_prefill=use_wmma_prefill,
            capture_linear_state_rows=capture_linear_state_rows,
            capture_pre_output_norm_hidden=capture_pre_output_norm_hidden,
            defer_linear_state_commit=defer_linear_state_commit,
            device_accept_commit=device_accept_commit,
            execution_profile_manifest_sha256=manifest_hash,
            recurrent_state_dtype=state_dtype,
        )
        if expected_key != self.configuration_key:
            return "target_graph_configuration_miss"
        if _native_target_binding_signature(session) != self.binding_signature:
            return "target_graph_binding_generation_miss"
        if int(position) + int(rows) > int(self.context_limit):
            return NATIVE_SPEC_TARGET_GRAPH_CONTEXT_BUCKET_MISS
        if bool(device_accept_commit) and (
            remaining_decode is None or int(remaining_decode) < 1
        ):
            return NATIVE_SPEC_TARGET_GRAPH_OUTPUT_ROOM_MISS
        return None

    def can_launch(
        self,
        session: Any,
        *,
        position: int,
        rows: int,
        remaining_decode: int | None,
        bulk_attention_mode: str,
        use_wmma_prefill: bool,
        capture_linear_state_rows: bool,
        capture_pre_output_norm_hidden: bool,
        defer_linear_state_commit: bool,
        device_accept_commit: bool,
    ) -> bool:
        """Return whether the exact live cycle fits this immutable graph owner."""

        return self.launch_ineligibility_reason(
            session,
            position=position,
            rows=rows,
            remaining_decode=remaining_decode,
            bulk_attention_mode=bulk_attention_mode,
            use_wmma_prefill=use_wmma_prefill,
            capture_linear_state_rows=capture_linear_state_rows,
            capture_pre_output_norm_hidden=capture_pre_output_norm_hidden,
            defer_linear_state_commit=defer_linear_state_commit,
            device_accept_commit=device_accept_commit,
        ) is None

    def launch(
        self,
        input_token_ids: Sequence[int] | None = None,
        *,
        cycle_id: int | None = None,
        transaction_id: int | None = None,
        request_id: int = 0,
        remaining_decode: int | None = None,
        device_proposal: Any | None = None,
        compact_result: bool = False,
        capture_lm_head_logits: bool = False,
    ):
        """Stage live metadata, replay once, and return one bounded result."""

        if self.closed:
            raise RuntimeError("native target graph is closed")
        if _native_target_binding_signature(self.session) != self.binding_signature:
            raise RuntimeError("native target graph captured allocation identity changed")
        if device_proposal is not None:
            if input_token_ids is not None:
                raise ValueError("device proposal launch does not accept host candidate IDs")
            if not self.device_accept_commit:
                raise NativeSpecTargetGraphUnsupportedError(
                    "device proposal handoff requires an N2 target graph"
                )
            proposal_budget = int(getattr(device_proposal, "budget", -1))
            proposal_request_id = int(getattr(device_proposal, "request_id", -1))
            proposal_root = int(getattr(device_proposal, "root_token", -1))
            proposal_position = int(getattr(device_proposal, "root_position", -1))
            proposal_result_ptr = int(getattr(device_proposal, "result_ptr", 0))
            proposal_result_nbytes = int(getattr(device_proposal, "result_nbytes", 0))
            proposal_event = int(getattr(device_proposal, "completion_event", 0))
            if proposal_budget + 1 != int(self.rows):
                raise NativeSpecTargetGraphUnsupportedError(
                    "device proposal budget does not match the cached target bucket"
                )
            if proposal_request_id != int(request_id):
                raise ValueError("device proposal request id drifted before target launch")
            if proposal_position != int(self.session.position):
                raise ValueError("device proposal root position drifted before target launch")
            if proposal_root < 0 or proposal_result_ptr <= 0 or proposal_event <= 0:
                raise ValueError("device proposal descriptor is incomplete")
            expected_result_nbytes = proposal_budget * 2 * DType.INT32.itemsize
            if proposal_result_nbytes != expected_result_nbytes:
                raise ValueError("device proposal result span has an invalid size")
            tokens = (proposal_root, *(0 for _ in range(proposal_budget)))
        else:
            tokens = (
                self.batch.tokens
                if input_token_ids is None
                else tuple(int(token) for token in input_token_ids)
            )
        if len(tokens) != int(self.rows):
            raise NativeSpecTargetGraphUnsupportedError(
                f"native target graph B{self.rows - 1} bucket requires {self.rows} rows"
            )
        start = int(self.session.position)
        bucket_end = start + int(self.rows)
        if bucket_end > int(self.context_limit):
            raise NativeSpecTargetGraphUnsupportedError(
                NATIVE_SPEC_TARGET_GRAPH_CONTEXT_BUCKET_MISS
            )
        if self.device_accept_commit and (
            remaining_decode is None or int(remaining_decode) < 1
        ):
            raise NativeSpecTargetGraphUnsupportedError(
                NATIVE_SPEC_TARGET_GRAPH_OUTPUT_ROOM_MISS
            )
        batch = build_native_b2_target_batch(tokens, start_position=start, request_id=request_id)
        runtime = self.session.runtime
        _stage_dynamic_metadata(self.dynamic_metadata, batch, runtime=runtime)
        if self.device_accept_commit:
            if remaining_decode is None or int(remaining_decode) < 1:
                raise ValueError(
                    "N2 native accept/commit requires room for one correction"
                )
            if self.remaining_decode is None:
                raise RuntimeError("N2 native accept/commit remaining-decode buffer is missing")
            _copy_array_to_tensor(
                self.remaining_decode,
                np.asarray([int(remaining_decode)], dtype=np.int32),
                runtime=runtime,
            )
        elif remaining_decode is not None:
            raise ValueError("remaining_decode is only valid for N2 native accept/commit")
        if device_proposal is not None:
            if self.result_payload is None:
                raise RuntimeError("device proposal target payload is missing")
            _enqueue_device_proposal_handoff(
                runtime=runtime,
                stream=self.stream,
                dynamic_metadata_ptr=self.dynamic_metadata.ptr,
                result_payload_ptr=self.result_payload.ptr,
                proposal_result_ptr=proposal_result_ptr,
                proposal_result_nbytes=proposal_result_nbytes,
                proposal_event=proposal_event,
                proposal_budget=proposal_budget,
                target_rows=self.rows,
                copy_result_payload=not bool(compact_result),
            )
        control = replace(
            self.control,
            cycle_id=self.control.cycle_id if cycle_id is None else int(cycle_id),
            transaction_id=(
                self.control.transaction_id if transaction_id is None else int(transaction_id)
            ),
        )

        submit_start = time.perf_counter()
        result = self.launcher.launch(control)
        self.session.last_native_spec_target_submit_ms = (
            time.perf_counter() - submit_start
        ) * 1000.0
        self.batch = batch
        self.control = control
        if result.status is not NativeSpecCycleStatus.COMPLETE:
            self.native_result = result
            raise RuntimeError(
                "native target graph failed: "
                f"status={result.status.name} error={result.error.name} "
                f"backend_error={result.backend_error_code}"
            )

        readback_start = time.perf_counter()
        hidden_size = int(self.session.runner.hidden_size)
        lm_head_logits_host = None
        hidden_seed_rows_host = None
        if capture_lm_head_logits:
            logits_buf = getattr(self.session, "_verify_logits_buf", None)
            if logits_buf is None:
                raise RuntimeError("native target graph diagnostic logits buffer is missing")
            vocab_size = int(self.session.runner.vocab_size)
            lm_head_logits_host = np.empty((self.rows, vocab_size), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(lm_head_logits_host),
                DeviceBuffer(int(logits_buf.ptr), int(lm_head_logits_host.nbytes)),
                lm_head_logits_host.nbytes,
                runtime=runtime,
            )
            hidden_seed_rows_host = np.empty(
                (self.rows, hidden_size), dtype=np.float32
            )
            copy_device_to_host(
                host_array_ptr(hidden_seed_rows_host),
                _tensor_buffer(self.hidden_seed_rows),
                hidden_seed_rows_host.nbytes,
                runtime=runtime,
            )
        if self.device_accept_commit:
            if self.result_payload is None:
                raise RuntimeError("N2 native accept/commit result payload is missing")
            payload_items = (
                ACCEPT_PACKED_PAYLOAD_FIELDS + 1 + self.rows
                if bool(compact_result)
                else self.result_payload.numel
            )
            payload = np.empty((payload_items,), dtype=np.int32)
            copy_device_to_host(
                host_array_ptr(payload),
                DeviceBuffer(
                    self.result_payload.ptr,
                    payload_items * DType.INT32.itemsize,
                ),
                payload.nbytes,
                runtime=runtime,
            )
            accepted = int(payload[0])
            commit_row = int(payload[1])
            commit_token = int(payload[2])
            commit_position = int(payload[3])
            next_token = int(payload[4])
            full_accept = bool(payload[5])
            committed_length = int(payload[6])
            visible_length = int(payload[ACCEPT_PACKED_PAYLOAD_FIELDS])
            output_start = ACCEPT_PACKED_PAYLOAD_FIELDS + 1
            target_top1_start = output_start + self.rows
            if (
                accepted < 0
                or accepted >= self.rows
                or commit_row != accepted
                or committed_length != accepted + 1
                or visible_length != accepted + 1
                or next_token < 0
                or output_start + visible_length > target_top1_start
                or (
                    not bool(compact_result)
                    and target_top1_start + self.rows > payload.size
                )
            ):
                raise RuntimeError(
                    "N2 native accept/commit returned an invalid bounded payload: "
                    f"accepted={accepted} commit_row={commit_row} "
                    f"committed_length={committed_length} "
                    f"visible_length={visible_length} next_token={next_token} "
                    f"remaining_decode={remaining_decode} rows={self.rows}"
                )
            output_tokens = [
                int(token)
                for token in payload[output_start:output_start + visible_length].tolist()
            ]
            target_top1 = (
                []
                if bool(compact_result)
                else [
                    int(token)
                    for token in payload[
                        target_top1_start:target_top1_start + self.rows
                    ].tolist()
                ]
            )
            proposal_top1_values: tuple[float, ...] = ()
            if device_proposal is not None and not bool(compact_result):
                proposal_payload_start = target_top1_start + self.rows
                proposal_payload_end = proposal_payload_start + 2 * proposal_budget
                if proposal_payload_end > payload.size:
                    raise RuntimeError("N2 device proposal payload is truncated")
                proposal_pairs = np.ascontiguousarray(
                    payload[proposal_payload_start:proposal_payload_end]
                ).reshape(proposal_budget, 2)
                proposal_tokens = tuple(int(token) for token in proposal_pairs[:, 0])
                proposal_top1_values = tuple(
                    float(value)
                    for value in np.ascontiguousarray(proposal_pairs[:, 1]).view(np.float32)
                )
                if any(token < 0 for token in proposal_tokens):
                    raise RuntimeError("N2 device proposal returned an invalid candidate ID")
                if not np.all(
                    np.isfinite(np.asarray(proposal_top1_values, dtype=np.float32))
                ):
                    raise RuntimeError("N2 device proposal returned NaN or Inf")
                batch = build_native_b2_target_batch(
                    (proposal_root, *proposal_tokens),
                    start_position=start,
                    request_id=request_id,
                )
            if self.accept_buffers is None or self.pre_output_commit_buffers is None:
                raise RuntimeError("N2 native device buffer descriptors are missing")
            live_verify_buffers = replace(
                self.accept_buffers,
                request_ids=batch.request_ids,
                transaction_id=int(control.transaction_id),
                candidate_counts=batch.candidate_counts,
                draft_depth=batch.draft_depth,
                tree_shape=batch.tree_shape,
            )
            live_state_commit_buffers = replace(
                self.pre_output_commit_buffers,
                request_ids=batch.request_ids,
                transaction_id=int(control.transaction_id),
            )
            if self.selected_pre_output_norm_hidden_bf16 is None:
                raise RuntimeError("N2 native selected trunk-hidden buffer is missing")
            self.session._last_target_hidden_ptr = int(
                self.selected_pre_output_norm_hidden_bf16.ptr
            )
            end = start + visible_length
            result = replace(result, visible_output_count=visible_length)
            result.validate_for(control)
            self.native_result = result
            block_result = Qwen35GGUFNativeAcceptCommitResult(
                input_token_ids=[int(token) for token in batch.tokens],
                token_ids=output_tokens,
                accepted_draft_tokens=accepted,
                commit_row=commit_row,
                commit_token=commit_token,
                commit_position=commit_position,
                next_token=next_token,
                full_accept=full_accept,
                start_position=start,
                end_position=end,
                hidden_seed_rows_ptr=int(self.hidden_seed_rows.ptr),
                hidden_seed_row_count=self.rows,
                hidden_size=hidden_size,
                target_top1=target_top1,
                proposal_top1_values=proposal_top1_values,
                proposal_device_handoff=device_proposal is not None,
                verify_buffers=live_verify_buffers,
                state_commit_buffers=live_state_commit_buffers,
                hidden_seeds=(
                    np.empty((0, 0), dtype=np.float32)
                    if hidden_seed_rows_host is None
                    else np.ascontiguousarray(hidden_seed_rows_host, dtype=np.float32)
                ),
                lm_head_logits_f32=(
                    None
                    if lm_head_logits_host is None
                    else np.ascontiguousarray(lm_head_logits_host, dtype=np.float32)
                ),
                compact_result=bool(compact_result),
            )
        else:
            token_host = np.empty((self.rows,), dtype=np.int64)
            copy_device_to_host(
                host_array_ptr(token_host),
                _tensor_buffer(self.buffers.target_top1),
                token_host.nbytes,
                runtime=runtime,
            )
            hidden_host = np.empty((self.rows, hidden_size), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(hidden_host),
                _tensor_buffer(self.hidden_seed_rows),
                hidden_host.nbytes,
                runtime=runtime,
            )
            pre_output_hidden_host = None
            if self.capture_pre_output_norm_hidden:
                if self.pre_output_norm_hidden_rows is None:
                    raise RuntimeError("native target graph trunk-hidden rows are missing")
                pre_output_hidden_host = np.empty(
                    (self.rows, hidden_size),
                    dtype=np.float32,
                )
                copy_device_to_host(
                    host_array_ptr(pre_output_hidden_host),
                    _tensor_buffer(self.pre_output_norm_hidden_rows),
                    pre_output_hidden_host.nbytes,
                    runtime=runtime,
                )
            session_hidden = self.session._verify_hidden_seed_buf
            if session_hidden is None or int(session_hidden.nbytes) < hidden_host.nbytes:
                raise RuntimeError("GGUF verifier hidden-seed destination is closed or undersized")
            runtime.memcpy(
                session_hidden.ptr,
                self.hidden_seed_rows.ptr,
                hidden_host.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
            end = bucket_end
            self.native_result = result
            from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFBlockVerifyResult

            block_result = Qwen35GGUFBlockVerifyResult(
                input_token_ids=[int(token) for token in batch.tokens],
                token_ids=[int(token) for token in token_host.tolist()],
                hidden_seeds=np.ascontiguousarray(hidden_host, dtype=np.float32),
                start_position=start,
                pre_output_norm_hidden=(
                    None
                    if pre_output_hidden_host is None
                    else np.ascontiguousarray(pre_output_hidden_host, dtype=np.float32)
                ),
                layer_output_hidden=None,
                layer_boundary_hidden=None,
                lm_head_logits_f32=(
                    None
                    if lm_head_logits_host is None
                    else np.ascontiguousarray(lm_head_logits_host, dtype=np.float32)
                ),
                linear_state_rows_captured=bool(self.capture_linear_state_rows),
                final_linear_state_committed=not bool(self.defer_linear_state_commit),
            )

        self.session._verify_hidden_seed_rows_populated = (
            0 if self.device_accept_commit else self.rows
        )
        self.session._hidden_seed_fp32_populated = True
        self.session.last_native_spec_target_submitted = True
        self.session.last_native_spec_target_fallback_reason = None
        self.session.last_native_spec_target_capture_ms = (
            0.0 if self.capture_reported else float(self.capture_wall_ms)
        )
        self.capture_reported = True
        self.session.last_native_spec_target_readback_ms = (
            time.perf_counter() - readback_start
        ) * 1000.0
        self.session._position = end
        self.session.scratch.position_host[0] = end
        self.session.scratch.context_host[0] = end + 1
        self.session.last_verify_stage_timings_ms = {}
        return block_result

    @property
    def launch_count(self) -> int:
        return int(self.launcher.launch_count)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        runtime = self.session.runtime
        try:
            runtime.graph_exec_destroy(self.graph_exec)
        finally:
            try:
                runtime.graph_destroy(self.graph)
            finally:
                try:
                    if self.stream:
                        runtime.stream_destroy(self.stream)
                finally:
                    self.workspace.free()
        graphs = getattr(self.session, "_decode_graphs", None)
        if isinstance(graphs, list) and self in graphs:
            graphs.remove(self)
        unpin = getattr(self.session, "_unpin_device_kv_graph", None)
        if callable(unpin):
            unpin(self)
        cache = getattr(self.session, "_native_spec_target_graphs", None)
        if isinstance(cache, dict):
            for key, value in tuple(cache.items()):
                if value is self:
                    cache.pop(key, None)
        for cache_name in (
            "_native_spec_b1_target_graph",
            "_native_spec_b2_target_graph",
            "_native_spec_b3_target_graph",
            "_native_spec_b4_target_graph",
            "_native_spec_b5_target_graph",
            "_native_spec_b6_target_graph",
            "_native_spec_b7_target_graph",
            "_native_spec_b1_target_graph_n2",
            "_native_spec_b2_target_graph_n2",
            "_native_spec_b3_target_graph_n2",
            "_native_spec_b4_target_graph_n2",
            "_native_spec_b5_target_graph_n2",
            "_native_spec_b6_target_graph_n2",
            "_native_spec_b7_target_graph_n2",
        ):
            if getattr(self.session, cache_name, None) is self:
                setattr(self.session, cache_name, None)

    def __enter__(self) -> "Qwen35GGUFNativeB2TargetGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def capture_qwen35_gguf_native_b2_target_graph(
    session: Any,
    input_token_ids: Sequence[int],
    *,
    context_limit: int | None = None,
    cycle_id: int = 0,
    transaction_id: int = 0,
    request_id: int = 0,
    bulk_attention_mode: str = "bulk",
    use_wmma_prefill: bool = False,
    capture_linear_state_rows: bool = False,
    capture_pre_output_norm_hidden: bool = False,
    capture_lm_head_logits: bool = False,
    record_stage_timings: bool = False,
    sync_stage_timings: bool = False,
    defer_linear_state_commit: bool = False,
    device_accept_commit: bool = False,
) -> Qwen35GGUFNativeB2TargetGraph:
    """Capture one fixed B1-B3 target forward without executing it."""

    capture_start = time.perf_counter()
    tokens = tuple(int(token) for token in input_token_ids)
    rows = len(tokens)
    selected_context_limit = (
        _native_target_graph_context_limit(session, rows=rows)
        if context_limit is None
        else int(context_limit)
    )
    if selected_context_limit is None:
        raise NativeSpecTargetGraphUnsupportedError(
            NATIVE_SPEC_TARGET_GRAPH_CONTEXT_BUCKET_MISS
        )
    _validate_capture_admission(
        session,
        tokens,
        context_limit=int(selected_context_limit),
        bulk_attention_mode=bulk_attention_mode,
        use_wmma_prefill=bool(use_wmma_prefill),
        capture_lm_head_logits=bool(capture_lm_head_logits),
        record_stage_timings=bool(record_stage_timings),
        sync_stage_timings=bool(sync_stage_timings),
    )
    if device_accept_commit and not (
        bool(capture_linear_state_rows) and bool(defer_linear_state_commit)
    ):
        raise NativeSpecTargetGraphUnsupportedError(
            "N2 device accept/commit requires captured deferred linear-state rows"
        )
    start = int(session.position)
    end = start + rows
    if end > int(session.scratch.max_positions):
        raise ValueError("native target graph rows exceed resident cache capacity")
    batch = build_native_b2_target_batch(tokens, start_position=start, request_id=request_id)
    runtime = session.runtime
    session._ensure_verify_block_buffers(rows, runtime=runtime)
    session._ensure_verify_lm_head_buffers(rows, runtime=runtime)
    if capture_linear_state_rows:
        session._ensure_verify_linear_state_row_buffers(rows, runtime=runtime)
    accept_library = None
    commit_library = None
    accept_kernel = None
    hidden_commit_kernel = None
    linear_commit_kernel = None
    if device_accept_commit:
        if not session._ensure_verify_linear_state_commit_tables(runtime=runtime):
            raise NativeSpecTargetGraphUnsupportedError(
                "N2 device accept/commit requires uniform fused linear-state commit tables"
            )
        accept_kernel = resolve(
            backend=str(session.backend),
            layer="speculative_accept_commit",
            quant="w4_gguf",
            variant="native_v1_i32",
            missing="none",
        )
        hidden_commit_kernel = resolve(
            backend=str(session.backend),
            layer="dflash_commit_chain",
            quant="w4_paro",
            variant="i32",
            missing="none",
        )
        linear_commit_variant = (
            "chunked_i32" if session._chunked_linear_state_commit_enabled() else "i32"
        )
        linear_commit_kernel = resolve(
            backend=str(session.backend),
            layer="linear_state_pair_commit",
            quant="w4_paro",
            variant=linear_commit_variant,
            missing="none",
        )
        if accept_kernel is None or hidden_commit_kernel is None or linear_commit_kernel is None:
            raise NativeSpecTargetGraphUnsupportedError(
                "N2 device accept/commit registered primitives are unavailable"
            )
        accept_library = build_dflash_accept(
            load=True,
            compiler_version=getattr(session, "compiler_version", None),
            require_cached=bool(getattr(session, "require_cached_build", False)),
        )
        commit_library = build_dflash_commit(
            load=True,
            compiler_version=getattr(session, "compiler_version", None),
            require_cached=bool(getattr(session, "require_cached_build", False)),
        )
    workspace = RuntimeWorkspace(device=session.scratch.block_table_tensor.device, runtime=runtime)
    graph = 0
    graph_exec = 0
    stream = 0
    try:
        owner = TargetVerifyBufferOwner.allocate(
            TargetVerifyBufferSpec(
                backend=str(session.backend),
                bucket=f"native_v1_b{rows - 1}_target_graph",
                device=workspace.device,
                max_rows=rows,
                max_requests=1,
                mode="verify_chain",
                metadata_dtype=DType.INT64,
            ),
            workspace=workspace,
        )
        buffers = owner.bind(batch, transaction_id=int(transaction_id))
        context_limit = int(selected_context_limit)
        buffers, dynamic_scratch = _dynamic_target_scratch(
            session,
            buffers,
            rows=rows,
            context_limit=context_limit,
        )
        _stage_target_batch(batch, buffers, runtime=runtime)
        hidden_size = int(session.runner.hidden_size)
        hidden_rows = workspace.reserve_tensor(
            "native_spec_hidden_seed_rows",
            (rows, hidden_size),
            DType.FP32,
        )
        hidden_f32_a = workspace.reserve_tensor(
            "native_spec_hidden_f32_a",
            (rows, hidden_size),
            DType.FP32,
        )
        hidden_f32_b = workspace.reserve_tensor(
            "native_spec_hidden_f32_b",
            (rows, hidden_size),
            DType.FP32,
        )
        capture_pre_output_rows = bool(capture_pre_output_norm_hidden) or bool(
            device_accept_commit
        )
        pre_output_norm_hidden_rows = (
            workspace.reserve_tensor(
                "native_spec_pre_output_norm_hidden_rows",
                (rows, hidden_size),
                DType.FP32,
            )
            if capture_pre_output_rows
            else None
        )
        pre_output_norm_hidden_bf16_rows = None
        selected_pre_output_norm_hidden_bf16 = None
        if device_accept_commit:
            pre_output_norm_hidden_bf16_rows = workspace.reserve_tensor(
                "native_spec_pre_output_norm_hidden_bf16_rows",
                (rows, hidden_size),
                DType.BF16,
            )
            selected_hidden = getattr(
                session,
                "_native_spec_selected_hidden_bf16",
                None,
            )
            if selected_hidden is None:
                selected_hidden = malloc(
                    hidden_size * DType.BF16.itemsize,
                    runtime=runtime,
                )
                session._native_spec_selected_hidden_bf16 = selected_hidden
                session._buffers = (*session._buffers, selected_hidden)
            selected_pre_output_norm_hidden_bf16 = Tensor.from_handle(
                selected_hidden.ptr,
                (1, 1, hidden_size),
                DType.BF16,
                workspace.device,
            )
        native_decode_row_scratches = (
            _dynamic_native_decode_row_scratches(
                session,
                dynamic_scratch,
                rows=rows,
                start_position=start,
                context_limit=context_limit,
            )
            if bulk_attention_mode == "native"
            else None
        )
        dynamic_metadata = workspace.reserve_tensor(
            "native_spec_dynamic_metadata",
            (rows, 5),
            DType.INT64,
        )
        token_ids_i32 = workspace.reserve_tensor(
            "native_spec_token_ids_i32",
            (rows,),
            DType.INT32,
        )
        positions_i32 = workspace.reserve_tensor(
            "native_spec_positions_i32",
            (rows,),
            DType.INT32,
        )
        linear_state_src_conv_table = None
        linear_state_src_recurrent_table = None
        linear_state_dst_conv_table = None
        linear_state_dst_recurrent_table = None
        if device_accept_commit:
            (
                linear_state_src_conv_table,
                linear_state_src_recurrent_table,
                linear_state_dst_conv_table,
                linear_state_dst_recurrent_table,
            ) = _allocate_native_linear_state_tables(
                workspace,
                session,
                runtime=runtime,
            )
        accept_buffers = None
        remaining_decode_tensor = None
        result_payload = None
        visible_output_ids = None
        visible_output_lengths = None
        target_top1_payload = None
        commit_buffers = None
        pre_output_commit_buffers = None
        candidate_counts = None
        if device_accept_commit:
            accept_owner = TargetVerifyBufferOwner.allocate(
                TargetVerifyBufferSpec(
                    backend=str(session.backend),
                    bucket=f"native_v1_b{rows - 1}_accept_commit",
                    device=workspace.device,
                    max_rows=rows,
                    max_requests=1,
                    mode="verify_chain",
                    metadata_dtype=DType.INT32,
                ),
                workspace=workspace,
            )
            accept_buffers = accept_owner.bind(batch, transaction_id=int(transaction_id))
            if session._verify_lm_out_indices_i32 is None:
                raise RuntimeError("GGUF verifier int32 top1 rows are closed")
            target_top1_i32 = Tensor.from_handle(
                int(session._verify_lm_out_indices_i32.ptr),
                (rows,),
                DType.INT32,
                workspace.device,
            )
            accept_buffers = replace(
                accept_buffers,
                token_ids=token_ids_i32,
                positions=positions_i32,
                target_top1=target_top1_i32,
            )
            _stage_target_batch(batch, accept_buffers, runtime=runtime)
            candidate_counts = workspace.reserve_tensor(
                "native_spec_candidate_counts",
                (1,),
                DType.INT32,
            )
            remaining_decode_tensor = workspace.reserve_tensor(
                "native_spec_remaining_decode",
                (1,),
                DType.INT32,
            )
            _copy_array_to_tensor(
                candidate_counts,
                np.asarray([rows - 1], dtype=np.int32),
                runtime=runtime,
            )
            _copy_array_to_tensor(
                remaining_decode_tensor,
                np.asarray([rows], dtype=np.int32),
                runtime=runtime,
            )
            result_payload = workspace.reserve_tensor(
                "native_spec_result_payload",
                (ACCEPT_PACKED_PAYLOAD_FIELDS + 1 + 2 * rows + 2 * (rows - 1),),
                DType.INT32,
            )
            visible_output_lengths = Tensor.from_handle(
                result_payload.ptr + ACCEPT_PACKED_PAYLOAD_FIELDS * DType.INT32.itemsize,
                (1,),
                DType.INT32,
                workspace.device,
            )
            visible_output_ids = Tensor.from_handle(
                result_payload.ptr + (ACCEPT_PACKED_PAYLOAD_FIELDS + 1) * DType.INT32.itemsize,
                (rows,),
                DType.INT32,
                workspace.device,
            )
            target_top1_payload = Tensor.from_handle(
                result_payload.ptr
                + (ACCEPT_PACKED_PAYLOAD_FIELDS + 1 + rows) * DType.INT32.itemsize,
                (rows,),
                DType.INT32,
                workspace.device,
            )
            hidden_src = Tensor.from_handle(
                hidden_rows.ptr,
                (1, rows, hidden_size),
                DType.FP32,
                workspace.device,
            )
            hidden_dst = Tensor.from_handle(
                session.scratch.hidden_seed_fp32.ptr,
                (1, 1, hidden_size),
                DType.FP32,
                workspace.device,
            )
            commit_buffers = TargetStateCommitBuffers(
                request_ids=batch.request_ids,
                transaction_id=int(transaction_id),
                accepted_counts=accept_buffers.accepted_counts,
                commit_rows=accept_buffers.commit_rows,
                commit_positions=accept_buffers.commit_positions,
                hidden_taps_src=hidden_src,
                hidden_taps_dst=hidden_dst,
                mode="verify_chain",
            )
            assert pre_output_norm_hidden_bf16_rows is not None
            assert selected_pre_output_norm_hidden_bf16 is not None
            pre_output_commit_buffers = TargetStateCommitBuffers(
                request_ids=batch.request_ids,
                transaction_id=int(transaction_id),
                accepted_counts=accept_buffers.accepted_counts,
                commit_rows=accept_buffers.commit_rows,
                commit_positions=accept_buffers.commit_positions,
                hidden_taps_src=Tensor.from_handle(
                    pre_output_norm_hidden_bf16_rows.ptr,
                    (1, rows, hidden_size),
                    DType.BF16,
                    workspace.device,
                ),
                hidden_taps_dst=selected_pre_output_norm_hidden_bf16,
                mode="verify_chain",
            )
        _stage_dynamic_metadata(dynamic_metadata, batch, runtime=runtime)
        runtime.device_synchronize()
        stream = runtime.stream_create()
        control_buffers = buffers if accept_buffers is None else accept_buffers
        control = NativeSpecCycleControl.for_target_verify(
            cycle_id=int(cycle_id),
            buffers=control_buffers,
            kv_live_spans=dynamic_scratch.prefill_spans,
            hidden_seed_rows=hidden_rows,
            context_bucket=_context_bucket(context_limit),
            stream=stream,
            output_stride=rows,
            candidate_counts_ptr=0 if candidate_counts is None else candidate_counts.ptr,
            remaining_decode_ptr=(
                0 if remaining_decode_tensor is None else remaining_decode_tensor.ptr
            ),
        )
        if device_accept_commit:
            assert accept_buffers is not None
            assert visible_output_ids is not None and visible_output_lengths is not None
            assert linear_state_src_conv_table is not None
            assert linear_state_src_recurrent_table is not None
            assert linear_state_dst_conv_table is not None
            assert linear_state_dst_recurrent_table is not None
            stages = (
                NativeSpecCycleStage.VERIFY
                | NativeSpecCycleStage.ACCEPT
                | NativeSpecCycleStage.COMMIT
                | NativeSpecCycleStage.UPDATE_CURSORS
            )
            control = replace(
                control,
                stages=stages,
                pointers=replace(
                    control.pointers,
                    state=replace(
                        control.pointers.state,
                        linear_state_rows=linear_state_src_conv_table.ptr,
                        linear_state_dst=linear_state_dst_conv_table.ptr,
                        hidden_seed_dst=session.scratch.hidden_seed_fp32.ptr,
                    ),
                    outputs=replace(
                        control.pointers.outputs,
                        output_ids=visible_output_ids.ptr,
                        output_lengths=visible_output_lengths.ptr,
                        last_positions=session.scratch.position_buf.ptr,
                        context_lengths=session.scratch.context_buf.ptr,
                    ),
                ),
            )

        runtime.stream_begin_capture(stream)
        try:
            unpack_verify_chain_dynamic_metadata_i64(
                dynamic_metadata.ptr,
                buffers.token_ids.ptr,
                token_ids_i32.ptr,
                buffers.positions.ptr,
                positions_i32.ptr,
                dynamic_scratch.context_counts_tensor.ptr,
                rows,
                stream=stream,
                library=session._runtime_state_library,
                runtime=runtime,
            )
            session.verify_target_block(
                tokens,
                bulk_attention_mode=bulk_attention_mode,
                use_wmma_prefill=bool(use_wmma_prefill),
                stream=stream,
                capture_linear_state_rows=bool(capture_linear_state_rows),
                capture_pre_output_norm_hidden=capture_pre_output_rows,
                defer_linear_state_commit=bool(defer_linear_state_commit),
                _pre_staged_token_ids_ptr=buffers.token_ids.ptr,
                _target_top1_i64_ptr=(
                    None if device_accept_commit else buffers.target_top1.ptr
                ),
                _target_top1_i32_ptr=(
                    None
                    if accept_buffers is None
                    else accept_buffers.target_top1.ptr
                ),
                _enqueue_only=True,
                _prebuilt_bulk_scratch=dynamic_scratch,
                _dynamic_cursor_advance=True,
                _graph_hidden_seed_buf=hidden_rows,
                _graph_hidden_f32_a=hidden_f32_a,
                _graph_hidden_f32_b=hidden_f32_b,
                _graph_pre_output_norm_hidden_buf=pre_output_norm_hidden_rows,
                _native_decode_row_scratches=native_decode_row_scratches,
                _native_attention_context_limit=(
                    context_limit if native_decode_row_scratches is not None else None
                ),
            )
            if device_accept_commit:
                assert accept_buffers is not None
                assert remaining_decode_tensor is not None
                assert result_payload is not None
                assert visible_output_ids is not None and visible_output_lengths is not None
                assert target_top1_payload is not None
                assert commit_buffers is not None
                assert pre_output_commit_buffers is not None
                assert pre_output_norm_hidden_rows is not None
                assert pre_output_norm_hidden_bf16_rows is not None
                assert accept_kernel is not None
                assert linear_commit_kernel is not None
                assert hidden_commit_kernel is not None
                assert accept_library is not None and commit_library is not None
                f32_to_bf16(
                    pre_output_norm_hidden_rows.ptr,
                    pre_output_norm_hidden_bf16_rows.ptr,
                    rows * hidden_size,
                    stream=stream,
                    library=session.runner._cast_library(),
                    runtime=runtime,
                )
                accept_kernel(
                    token_ids_i32.ptr,
                    positions_i32.ptr,
                    accept_buffers.parent_rows.ptr,
                    accept_buffers.draft_depths.ptr,
                    accept_buffers.active_mask.ptr,
                    accept_buffers.target_top1.ptr,
                    remaining_decode_tensor.ptr,
                    accept_buffers.accepted_counts.ptr,
                    accept_buffers.commit_rows.ptr,
                    accept_buffers.commit_tokens.ptr,
                    accept_buffers.commit_positions.ptr,
                    accept_buffers.next_tokens.ptr,
                    accept_buffers.full_accept.ptr,
                    accept_buffers.committed_output_ids.ptr,
                    accept_buffers.committed_output_lengths.ptr,
                    result_payload.ptr,
                    visible_output_ids.ptr,
                    visible_output_lengths.ptr,
                    session.scratch.position_buf.ptr,
                    session.scratch.context_buf.ptr,
                    1,
                    rows,
                    1,
                    rows,
                    stream=stream,
                    library=accept_library,
                    runtime=runtime,
                )
                linear_commit_kernel(
                    linear_state_src_conv_table.ptr,
                    linear_state_dst_conv_table.ptr,
                    int(session._verify_linear_state_conv_row_nbytes),
                    linear_state_src_recurrent_table.ptr,
                    linear_state_dst_recurrent_table.ptr,
                    int(session._verify_linear_state_recurrent_row_nbytes),
                    accept_buffers.commit_rows.ptr,
                    int(session._verify_linear_state_layer_count),
                    stream=stream,
                    library=commit_library,
                    runtime=runtime,
                )
                hidden_commit_kernel(
                    commit_buffers,
                    target_rows=rows,
                    stream=stream,
                    library=commit_library,
                    runtime=runtime,
                )
                hidden_commit_kernel(
                    pre_output_commit_buffers,
                    target_rows=rows,
                    stream=stream,
                    library=commit_library,
                    runtime=runtime,
                )
                runtime.memcpy_async(
                    target_top1_payload.ptr,
                    accept_buffers.target_top1.ptr,
                    rows * DType.INT32.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            graph = runtime.stream_end_capture(stream)
        except Exception:
            try:
                runtime.stream_end_capture(stream)
            except Exception:
                pass
            raise
        if not graph:
            raise RuntimeError("HIP returned a null native target graph")
        graph_exec = runtime.graph_instantiate(graph)
        if not graph_exec:
            raise RuntimeError("HIP returned a null native target graph executable")

        factory = resolve(
            backend=str(session.backend),
            layer="speculative_cycle",
            quant="w4_gguf",
            variant="native_v1_b2_target_graph",
        )
        launcher = factory(
            graph_exec=graph_exec,
            runtime=runtime,
            bound_control=control,
            compiler_version=getattr(session, "compiler_version", None),
            target_arch=getattr(session.runner, "target_arch", None),
            require_cached=bool(getattr(session, "require_cached_build", False)),
        )
        handle = Qwen35GGUFNativeB2TargetGraph(
            session=session,
            graph=graph,
            graph_exec=graph_exec,
            stream=stream,
            launcher=launcher,
            workspace=workspace,
            batch=batch,
            buffers=buffers,
            control=control,
            dynamic_scratch=dynamic_scratch,
            hidden_seed_rows=hidden_rows,
            hidden_f32_a=hidden_f32_a,
            hidden_f32_b=hidden_f32_b,
            pre_output_norm_hidden_rows=pre_output_norm_hidden_rows,
            pre_output_norm_hidden_bf16_rows=pre_output_norm_hidden_bf16_rows,
            selected_pre_output_norm_hidden_bf16=selected_pre_output_norm_hidden_bf16,
            dynamic_metadata=dynamic_metadata,
            token_ids_i32=token_ids_i32,
            positions_i32=positions_i32,
            linear_state_src_conv_table=linear_state_src_conv_table,
            linear_state_src_recurrent_table=linear_state_src_recurrent_table,
            linear_state_dst_conv_table=linear_state_dst_conv_table,
            linear_state_dst_recurrent_table=linear_state_dst_recurrent_table,
            accept_buffers=accept_buffers,
            remaining_decode=remaining_decode_tensor,
            result_payload=result_payload,
            visible_output_ids=visible_output_ids,
            visible_output_lengths=visible_output_lengths,
            target_top1_payload=target_top1_payload,
            pre_output_commit_buffers=pre_output_commit_buffers,
            accept_library=accept_library,
            commit_library=commit_library,
            device_accept_commit=bool(device_accept_commit),
            start_position=start,
            end_position=end,
            context_limit=context_limit,
            rows=rows,
            capture_linear_state_rows=bool(capture_linear_state_rows),
            capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
            defer_linear_state_commit=bool(defer_linear_state_commit),
            configuration_key=_native_target_configuration_key(
                bulk_attention_mode=bulk_attention_mode,
                use_wmma_prefill=bool(use_wmma_prefill),
                capture_linear_state_rows=bool(capture_linear_state_rows),
                capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
                defer_linear_state_commit=bool(defer_linear_state_commit),
                device_accept_commit=bool(device_accept_commit),
                execution_profile_manifest_sha256=(
                    _native_target_execution_identity(session)[0]
                ),
                recurrent_state_dtype=_native_target_execution_identity(session)[1],
            ),
            binding_signature=_native_target_binding_signature(session),
            capture_wall_ms=(time.perf_counter() - capture_start) * 1000.0,
        )
        session._decode_graphs.append(handle)
        pin = getattr(session, "_pin_device_kv_graph", None)
        if callable(pin):
            pin(handle)
        return handle
    except Exception:
        if graph_exec:
            try:
                runtime.graph_exec_destroy(graph_exec)
            except Exception:
                pass
        if graph:
            try:
                runtime.graph_destroy(graph)
            except Exception:
                pass
        if stream:
            try:
                runtime.stream_destroy(stream)
            except Exception:
                pass
        workspace.free()
        raise


def run_qwen35_gguf_native_mtp_cycle(
    session: Any,
    resident_draft: Any,
    resident_context: Any,
    *,
    root_token: int,
    root_position: int,
    candidate_budget: int,
    remaining_decode: int,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    draft_key_cache: DeviceBuffer,
    draft_value_cache: DeviceBuffer,
    draft_cache_len: int,
    cycle_id: int = 0,
    transaction_id: int = 0,
    request_id: int = 0,
    record_stage_timings: bool = False,
    native_proposal_graph: bool = False,
    target_bulk_attention_mode: str = "bulk",
    k1_disable_f32_verifier: bool = False,
) -> Qwen35GGUFNativeCompleteCycleResult:
    """Own one strict llama-compatible GGUF MTP cycle behind one call.

    The proposal uses the retained device-chained NextN implementation, with an
    optional proposal-only reusable graph around that same chain; target
    mutation remains the byte-exact N2 reusable graph. This N3 adapter is the
    single GGUF boundary that joins those providers, repairs the MTP KV
    transaction after acceptance, advances the GGUF reseed context, and returns
    only bounded scheduler metadata. Unsupported proposal-graph shapes fall
    back before target mutation; unsupported complete-cycle shapes raise so the
    caller can preserve the established exact/eager loop.
    """

    call_start = time.perf_counter()
    budget = int(candidate_budget)
    root = int(root_token)
    start = int(root_position)
    remaining = int(remaining_decode)
    cache_before = int(draft_cache_len)
    if budget not in {1, 2}:
        raise NativeSpecTargetGraphUnsupportedError(
            "native complete cycle supports only B1/B2 strict chains"
        )
    target_mode = str(target_bulk_attention_mode)
    if target_mode not in {"bulk", "native"}:
        raise ValueError("target_bulk_attention_mode must be bulk or native")
    if root < 0:
        raise ValueError("root_token must be non-negative")
    if start < 0 or cache_before < 0:
        raise ValueError("root_position and draft_cache_len must be non-negative")
    if remaining < 1:
        raise NativeSpecTargetGraphUnsupportedError(
            "native complete cycle requires output room for one correction"
        )
    if int(getattr(session, "position", -1)) != start:
        raise ValueError("root_position must match the target session cursor")
    if not bool(getattr(resident_draft, "_device_chain_enabled", False)):
        raise NativeSpecTargetGraphUnsupportedError(
            "native complete cycle requires the device-chained GGUF NextN provider"
        )
    pending_seed = getattr(resident_context, "pending_seed", None)
    pending_seed_ptr = int(getattr(pending_seed, "hidden_ptr", 0))
    if pending_seed_ptr <= 0:
        raise RuntimeError("native complete cycle requires a resident pending MTP seed")
    for name, cache in (
        ("draft_key_cache", draft_key_cache),
        ("draft_value_cache", draft_value_cache),
    ):
        if not isinstance(cache, DeviceBuffer) or int(cache.ptr) <= 0:
            raise TypeError(f"{name} must be a live DeviceBuffer")
    cos = np.asarray(rope_cos)
    sin = np.asarray(rope_sin)
    if cos.dtype != np.float32 or sin.dtype != np.float32 or cos.shape != sin.shape or cos.ndim != 2:
        raise ValueError("rope_cos and rope_sin must be aligned rank-2 float32 tables")
    if start + budget > int(cos.shape[0]):
        raise ValueError("native complete cycle exceeds the supplied RoPE table")

    proposal_start = time.perf_counter()
    proposal_kwargs = {
        "start_token": root,
        "start_position": start,
        "draft_n_max": budget,
        "top_k": 1,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "dense_key_cache": draft_key_cache,
        "dense_value_cache": draft_value_cache,
        "dense_cache_len": cache_before,
        "draft_p_min": 0.0,
        "record_stage_timings": bool(record_stage_timings),
    }
    proposal_graph_used = False
    proposal_method = resident_draft.propose_chain_from_device_seed
    graph_method = (
        getattr(resident_draft, "propose_chain_from_device_seed_graph", None)
        if native_proposal_graph
        else None
    )
    if callable(graph_method):
        try:
            draft_tokens, draft_topk, proposed_cache_len = graph_method(
                pending_seed_ptr,
                **proposal_kwargs,
            )
            proposal_graph_used = True
        except Exception as exc:
            # Import locally to avoid a module cycle: the resident draft provider
            # imports NativeSpecCycle contracts from this package. Only explicit
            # pre-launch admission failures may replay through the exact chain;
            # graph/runtime failures can have mutated K/V and must surface.
            from hipengine.speculative.mtp_resident_draft import (
                NativeSpecProposalGraphUnsupportedError,
            )

            if not isinstance(exc, NativeSpecProposalGraphUnsupportedError):
                raise
            draft_tokens, draft_topk, proposed_cache_len = proposal_method(
                pending_seed_ptr,
                **proposal_kwargs,
            )
    else:
        draft_tokens, draft_topk, proposed_cache_len = proposal_method(
            pending_seed_ptr,
            **proposal_kwargs,
        )
    proposal_wall_ms = (time.perf_counter() - proposal_start) * 1000.0
    drafts = tuple(int(token) for token in draft_tokens)
    if len(drafts) != budget or any(token < 0 for token in drafts):
        raise RuntimeError("device NextN proposal did not fill the requested B1/B2 chain")
    if len(draft_topk) != budget or any(tuple(int(token) for token in row) != (drafts[index],) for index, row in enumerate(draft_topk)):
        raise RuntimeError("native complete cycle requires one top-1 row per draft depth")
    if int(proposed_cache_len) != cache_before + budget:
        raise RuntimeError("device NextN proposal returned an unexpected speculative KV cursor")

    target_start = time.perf_counter()
    f32_override = bool(k1_disable_f32_verifier and budget == 1)
    target_result = _call_with_f32_verifier_disabled(
        f32_override,
        lambda: session.verify_target_block_native_cycle(
            [root, *drafts],
            cycle_id=int(cycle_id),
            transaction_id=int(transaction_id),
            request_id=int(request_id),
            bulk_attention_mode=target_mode,
            use_wmma_prefill=False,
            capture_linear_state_rows=True,
            defer_linear_state_commit=True,
            device_accept_commit=True,
            remaining_decode=remaining,
            fallback=False,
        ),
    )
    if not bool(getattr(target_result, "device_accept_commit", False)):
        raise RuntimeError("native complete cycle target did not execute N2 accept/commit")
    accepted = int(getattr(target_result, "accepted_draft_tokens", -1))
    outputs = tuple(int(token) for token in getattr(target_result, "token_ids", ()))
    end = int(getattr(target_result, "end_position", -1))
    hidden_rows_ptr = int(getattr(target_result, "hidden_seed_rows_ptr", 0))
    hidden_row_count = int(getattr(target_result, "hidden_seed_row_count", 0))
    target_wall_ms = (time.perf_counter() - target_start) * 1000.0
    if accepted < 0 or accepted > budget:
        raise RuntimeError("native target returned an invalid accepted draft count")
    if outputs[:accepted] != drafts[:accepted] or len(outputs) != accepted + 1:
        raise RuntimeError("native target returned an invalid strict-chain visible payload")
    if (
        int(getattr(target_result, "start_position", -1)) != start
        or end != start + len(outputs)
    ):
        raise RuntimeError("native target cursor diverged from the complete-cycle boundary")

    consumed_rows = accepted + 1
    if hidden_rows_ptr <= 0 or hidden_row_count < consumed_rows:
        raise RuntimeError("native target did not expose the consumed verifier hidden rows")
    verify_seeds = tuple(
        session.mtp_verify_seed(
            row,
            token_id=outputs[row],
            position=start + row,
            hidden_seed_base_ptr=hidden_rows_ptr,
            hidden_seed_row_count=hidden_row_count,
        )
        for row in range(consumed_rows)
    )
    resident_context.record_verify_seeds(verify_seeds)
    resident_context.accept(accepted)

    # Proposal writes root+draftee rows.  Keep the root row, replace accepted
    # draftee rows with verifier-derived target hidden rows, and drop rejects.
    committed_cache_len = cache_before + 1
    kv_commit_wall_ms = 0.0
    if accepted:
        commit_tokens = np.ascontiguousarray(outputs[:accepted], dtype=np.int64)
        commit_positions = np.arange(start + 1, start + 1 + accepted, dtype=np.int64)
        kv_start = time.perf_counter()
        committed_cache_len = resident_draft.write_kv_rows_from_device_seed_base(
            hidden_rows_ptr,
            commit_tokens,
            positions=commit_positions,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            dense_key_cache=draft_key_cache,
            dense_value_cache=draft_value_cache,
            dense_cache_len=committed_cache_len,
        )
        kv_commit_wall_ms = (time.perf_counter() - kv_start) * 1000.0
    expected_cache_len = cache_before + 1 + accepted
    if int(committed_cache_len) != expected_cache_len:
        raise RuntimeError("native complete cycle MTP KV commit returned an invalid cursor")

    return Qwen35GGUFNativeCompleteCycleResult(
        draft_token_ids=drafts,
        output_token_ids=outputs,
        accepted_draft_tokens=accepted,
        start_position=start,
        end_position=end,
        draft_cache_len_before=cache_before,
        draft_cache_len_after=int(committed_cache_len),
        target_result=target_result,
        proposal_wall_ms=proposal_wall_ms,
        target_wall_ms=target_wall_ms,
        mtp_kv_commit_wall_ms=kv_commit_wall_ms,
        call_wall_ms=(time.perf_counter() - call_start) * 1000.0,
        proposal_native_graph=proposal_graph_used,
    )


def verify_qwen35_gguf_native_b2_target(
    session: Any,
    input_token_ids: Sequence[int],
    *,
    fallback: bool = True,
    cycle_id: int = 0,
    transaction_id: int = 0,
    request_id: int = 0,
    bulk_attention_mode: str = "bulk",
    use_wmma_prefill: bool = False,
    capture_linear_state_rows: bool = False,
    capture_pre_output_norm_hidden: bool = False,
    capture_lm_head_logits: bool = False,
    record_stage_timings: bool = False,
    sync_stage_timings: bool = False,
    defer_linear_state_commit: bool = False,
    device_accept_commit: bool = False,
    remaining_decode: int | None = None,
):
    """Run reusable N1/N2 when admitted, otherwise preserve the eager verifier."""

    session.last_native_spec_target_submitted = False
    session.last_native_spec_target_fallback_reason = None
    session.last_native_spec_target_capture_ms = 0.0
    session.last_native_spec_target_submit_ms = 0.0
    session.last_native_spec_target_readback_ms = 0.0
    eager_kwargs: dict[str, object] = {
        "bulk_attention_mode": bulk_attention_mode,
        "use_wmma_prefill": bool(use_wmma_prefill),
        "capture_linear_state_rows": bool(capture_linear_state_rows),
        "defer_linear_state_commit": bool(defer_linear_state_commit),
    }
    if capture_pre_output_norm_hidden:
        eager_kwargs["capture_pre_output_norm_hidden"] = True
    if capture_lm_head_logits:
        eager_kwargs["capture_lm_head_logits"] = True
    if record_stage_timings:
        eager_kwargs["record_stage_timings"] = True
    if sync_stage_timings:
        eager_kwargs["sync_stage_timings"] = True
    rows = len(tuple(input_token_ids))
    recurrent_state_dtype = _native_target_execution_identity(session)[1]
    if recurrent_state_dtype == "fp16":
        reason = "FP16 recurrent state keeps target verify on the eager owner"
        session.last_native_spec_target_fallback_reason = reason
        if not fallback:
            raise NativeSpecTargetGraphUnsupportedError(reason)
        return session.verify_target_block(input_token_ids, **eager_kwargs)
    if rows not in {2, 3, 4, 5, 6, 7, 8}:
        reason = "native target graph requires two to eight rows (one root plus B1-B7)"
        session.last_native_spec_target_fallback_reason = reason
        if not fallback:
            raise NativeSpecTargetGraphUnsupportedError(reason)
        return session.verify_target_block(input_token_ids, **eager_kwargs)
    context_limit = _native_target_graph_context_limit(session, rows=rows)
    if context_limit is None:
        reason = NATIVE_SPEC_TARGET_GRAPH_CONTEXT_BUCKET_MISS
        session.last_native_spec_target_fallback_reason = reason
        if not fallback:
            raise NativeSpecTargetGraphUnsupportedError(reason)
        return session.verify_target_block(input_token_ids, **eager_kwargs)
    cache_suffix = "_n2" if device_accept_commit else ""
    cache_name = f"_native_spec_b{rows - 1}_target_graph{cache_suffix}"
    cache_key = (rows - 1, bool(device_accept_commit), int(context_limit))
    cache = _native_target_graph_cache(session)
    graph = cache.get(cache_key)
    if graph is None and int(context_limit) == _NATIVE_TARGET_SHORT_CONTEXT_LIMIT:
        graph = getattr(session, cache_name, None)
    if graph is not None and not graph.compatible_with(
        session,
        context_limit=int(context_limit),
        bulk_attention_mode=bulk_attention_mode,
        use_wmma_prefill=bool(use_wmma_prefill),
        capture_linear_state_rows=bool(capture_linear_state_rows),
        capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
        defer_linear_state_commit=bool(defer_linear_state_commit),
        device_accept_commit=bool(device_accept_commit),
    ):
        graph.close()
        cache.pop(cache_key, None)
        graph = None
    if graph is not None:
        cache.pop(cache_key, None)
        cache[cache_key] = graph
    try:
        if graph is None:
            graph = capture_qwen35_gguf_native_b2_target_graph(
                session,
                input_token_ids,
                context_limit=int(context_limit),
                cycle_id=cycle_id,
                transaction_id=transaction_id,
                request_id=request_id,
                bulk_attention_mode=bulk_attention_mode,
                use_wmma_prefill=use_wmma_prefill,
                capture_linear_state_rows=capture_linear_state_rows,
                capture_pre_output_norm_hidden=capture_pre_output_norm_hidden,
                capture_lm_head_logits=capture_lm_head_logits,
                record_stage_timings=record_stage_timings,
                sync_stage_timings=sync_stage_timings,
                defer_linear_state_commit=defer_linear_state_commit,
                device_accept_commit=device_accept_commit,
            )
            _cache_native_target_graph(session, cache_key, graph)
            if int(context_limit) == _NATIVE_TARGET_SHORT_CONTEXT_LIMIT:
                setattr(session, cache_name, graph)
    except NativeSpecTargetGraphUnsupportedError as exc:
        session.last_native_spec_target_fallback_reason = str(exc)
        if not fallback:
            raise
        return session.verify_target_block(input_token_ids, **eager_kwargs)
    try:
        launch_kwargs = {
            "cycle_id": int(cycle_id),
            "transaction_id": int(transaction_id),
            "request_id": int(request_id),
        }
        if device_accept_commit:
            launch_kwargs["remaining_decode"] = (
                None if remaining_decode is None else int(remaining_decode)
            )
        if capture_lm_head_logits:
            launch_kwargs["capture_lm_head_logits"] = True
        return graph.launch(input_token_ids, **launch_kwargs)
    except NativeSpecTargetGraphUnsupportedError as exc:
        session.last_native_spec_target_fallback_reason = str(exc)
        if not fallback:
            raise
        return session.verify_target_block(input_token_ids, **eager_kwargs)
    except Exception:
        try:
            graph.close()
        except Exception:
            pass
        raise


def verify_qwen35_gguf_native_target_from_device_proposal(
    session: Any,
    device_proposal: Any,
    *,
    cycle_id: int = 0,
    transaction_id: int = 0,
    request_id: int = 0,
    remaining_decode: int,
    bulk_attention_mode: str = "native",
    use_wmma_prefill: bool = False,
    capture_linear_state_rows: bool = True,
    capture_pre_output_norm_hidden: bool = True,
    capture_lm_head_logits: bool = False,
    defer_linear_state_commit: bool = True,
    compact_result: bool = False,
):
    """Retire a cached proposal and cached N2 target behind one synchronization.

    This route is intentionally cached-only. A miss is reported before target
    launch so the caller can use the established host-materialized proposal on
    a later cycle; capture is never attempted with an in-flight proposal.
    """

    session.last_native_spec_target_submitted = False
    session.last_native_spec_target_fallback_reason = None
    session.last_native_spec_target_capture_ms = 0.0
    session.last_native_spec_target_submit_ms = 0.0
    session.last_native_spec_target_readback_ms = 0.0
    if capture_lm_head_logits:
        raise NativeSpecTargetGraphUnsupportedError(
            "device proposal handoff does not support diagnostic logits"
        )
    rows = int(getattr(device_proposal, "budget", -1)) + 1
    if _native_target_execution_identity(session)[1] == "fp16":
        reason = "FP16 recurrent state device proposal requires eager selected commit"
        session.last_native_spec_target_fallback_reason = reason
        raise NativeSpecTargetGraphUnsupportedError(reason)
    if rows not in {2, 3, 4, 5, 6, 7, 8}:
        raise NativeSpecTargetGraphUnsupportedError(
            "device proposal requires one cached B1-B7 target bucket"
        )
    cache_name = f"_native_spec_b{rows - 1}_target_graph_n2"
    graph = getattr(session, cache_name, None)
    if graph is None:
        reason = "device proposal handoff requires a compatible cached N2 target graph"
        session.last_native_spec_target_fallback_reason = reason
        raise NativeSpecTargetGraphUnsupportedError(reason)
    eligibility = getattr(graph, "launch_ineligibility_reason", None)
    if not callable(eligibility):
        reason = "device proposal handoff requires target graph admission metadata"
        session.last_native_spec_target_fallback_reason = reason
        raise NativeSpecTargetGraphUnsupportedError(reason)
    reason = eligibility(
        session,
        position=int(session.position),
        rows=rows,
        remaining_decode=int(remaining_decode),
        bulk_attention_mode=bulk_attention_mode,
        use_wmma_prefill=bool(use_wmma_prefill),
        capture_linear_state_rows=bool(capture_linear_state_rows),
        capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
        defer_linear_state_commit=bool(defer_linear_state_commit),
        device_accept_commit=True,
    )
    if reason is not None:
        session.last_native_spec_target_fallback_reason = reason
        raise NativeSpecTargetGraphUnsupportedError(reason)
    try:
        launch_kwargs = {
            "cycle_id": int(cycle_id),
            "transaction_id": int(transaction_id),
            "request_id": int(request_id),
            "remaining_decode": int(remaining_decode),
            "device_proposal": device_proposal,
        }
        if compact_result:
            launch_kwargs["compact_result"] = True
        return graph.launch(**launch_kwargs)
    except Exception:
        try:
            graph.close()
        except Exception:
            pass
        raise


__all__ = [
    "NATIVE_SPEC_TARGET_GRAPH_CONTEXT_BUCKET_MISS",
    "NATIVE_SPEC_TARGET_GRAPH_OUTPUT_ROOM_MISS",
    "NativeSpecTargetGraphUnsupportedError",
    "Qwen35GGUFNativeAcceptCommitResult",
    "Qwen35GGUFNativeB2TargetGraph",
    "Qwen35GGUFNativeCompleteCycleResult",
    "build_native_b2_target_batch",
    "capture_qwen35_gguf_native_b2_target_graph",
    "run_qwen35_gguf_native_mtp_cycle",
    "verify_qwen35_gguf_native_b2_target",
    "verify_qwen35_gguf_native_target_from_device_proposal",
]
