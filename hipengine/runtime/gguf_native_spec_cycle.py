"""GGUF adapter for reusable B1/B2 native speculative target graphs.

The adapter captures two- and three-row target buckets against stable
session/device addresses, binds each capture to a versioned
:class:`NativeSpecCycleControl`, and submits it through one C++ launcher call.
Proposal and acceptance/commit remain unchanged.

Unsupported shapes and capture-unsafe session configurations use the existing
Python verifier when ``fallback=True``.  Launch or correctness failures never
fall back silently because the graph may already have mutated state/KV.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
import os
import time
from typing import Any, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, host_array_ptr
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.runtime import unpack_verify_chain_dynamic_metadata_i64
from hipengine.kernels.hip_gfx1100.speculative import (
    ACCEPT_PACKED_PAYLOAD_FIELDS,
    build_dflash_accept,
    build_dflash_commit,
    dflash_accept_chain_i32_native_cycle,
    dflash_commit_chain_i32,
    linear_state_pair_commit_chunked_i32,
    linear_state_pair_commit_i32,
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
    linear_state_rows_captured: bool = True
    final_linear_state_committed: bool = True
    device_accept_commit: bool = True
    hidden_seeds: np.ndarray = dataclass_field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    lm_head_logits_f32: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = len(self.input_token_ids)
        if rows not in {2, 3}:
            raise ValueError("native accept/commit result requires B1/B2 input rows")
        if self.accepted_draft_tokens < 0 or self.accepted_draft_tokens >= rows:
            raise ValueError("accepted_draft_tokens is outside the target bucket")
        if len(self.token_ids) != self.accepted_draft_tokens + 1:
            raise ValueError("token_ids must contain accepted drafts plus one correction")
        if self.commit_row != self.accepted_draft_tokens:
            raise ValueError("strict-chain commit_row must equal accepted_draft_tokens")
        if self.commit_token != self.input_token_ids[self.commit_row]:
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
        if self.hidden_seeds.shape != (0, 0) or self.hidden_seeds.dtype != np.float32:
            raise ValueError("device accept/commit must not return host hidden rows")


def build_native_b2_target_batch(
    input_token_ids: Sequence[int],
    *,
    start_position: int,
    request_id: int = 0,
) -> TargetVerifyBatch:
    """Build provider-neutral root+candidate metadata for a B1/B2 chain."""

    tokens = tuple(int(token) for token in input_token_ids)
    start = int(start_position)
    request = int(request_id)
    if len(tokens) not in {2, 3}:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph requires two or three rows (one root plus B1/B2)"
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


def _native_target_configuration_key(
    *,
    bulk_attention_mode: str,
    use_wmma_prefill: bool,
    capture_linear_state_rows: bool,
    defer_linear_state_commit: bool,
    device_accept_commit: bool,
) -> tuple[object, ...]:
    env = tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("HIPENGINE_")
        )
    )
    return (
        str(bulk_attention_mode),
        bool(use_wmma_prefill),
        bool(capture_linear_state_rows),
        bool(defer_linear_state_commit),
        bool(device_accept_commit),
        env,
    )


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
    ):
        for value in getattr(session, name, ()):
            add(value)
    return tuple(pointers)


def _validate_capture_admission(
    session: Any,
    input_token_ids: Sequence[int],
    *,
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
    if rows not in {2, 3}:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph requires two or three rows (one root plus B1/B2)"
        )
    if bulk_attention_mode != "bulk":
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 supports only the retained bulk verifier"
        )
    if use_wmma_prefill:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 supports only the non-WMMA small-row verifier"
        )
    if capture_lm_head_logits or sync_stage_timings:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 does not support logits readback or synchronized stage timings"
        )
    # The outer cycle may time capture+submission wall, but capture-time Python
    # dispatch intervals are not replay stage timings and are intentionally not
    # reported through ``last_verify_stage_timings_ms``.
    _ = record_stage_timings
    if getattr(session, "runner", None) is None or getattr(session, "scratch", None) is None:
        raise RuntimeError("GGUF resident session is closed")
    if bool(getattr(session, "host_token_embedding_enabled", False)):
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 requires device-resident token embedding"
        )
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
    if end >= 1024:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 is limited to the exact decode-batch context below 1024"
        )
    if not _gguf_prefill_device_metadata_enabled(
        backend=str(session.backend),
        prompt_tokens=end,
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
    """Reusable fixed B1/B2 verifier graph with device-driven metadata."""

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
    dynamic_metadata: Tensor
    token_ids_i32: Tensor
    positions_i32: Tensor
    accept_buffers: TargetVerifyBuffers | None
    remaining_decode: Tensor | None
    result_payload: Tensor | None
    visible_output_ids: Tensor | None
    visible_output_lengths: Tensor | None
    accept_library: Any | None
    commit_library: Any | None
    device_accept_commit: bool
    start_position: int
    end_position: int
    context_limit: int
    rows: int
    capture_linear_state_rows: bool
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
        bulk_attention_mode: str,
        use_wmma_prefill: bool,
        capture_linear_state_rows: bool,
        defer_linear_state_commit: bool,
        device_accept_commit: bool,
    ) -> bool:
        if self.closed or session is not self.session:
            return False
        expected_key = _native_target_configuration_key(
            bulk_attention_mode=bulk_attention_mode,
            use_wmma_prefill=use_wmma_prefill,
            capture_linear_state_rows=capture_linear_state_rows,
            defer_linear_state_commit=defer_linear_state_commit,
            device_accept_commit=device_accept_commit,
        )
        return (
            expected_key == self.configuration_key
            and _native_target_binding_signature(session) == self.binding_signature
        )

    def launch(
        self,
        input_token_ids: Sequence[int] | None = None,
        *,
        cycle_id: int | None = None,
        transaction_id: int | None = None,
        request_id: int = 0,
        remaining_decode: int | None = None,
    ):
        """Stage live metadata, replay once, and return one bounded result."""

        if self.closed:
            raise RuntimeError("native target graph is closed")
        if _native_target_binding_signature(self.session) != self.binding_signature:
            raise RuntimeError("native target graph captured allocation identity changed")
        tokens = self.batch.tokens if input_token_ids is None else tuple(int(token) for token in input_token_ids)
        if len(tokens) != int(self.rows):
            raise NativeSpecTargetGraphUnsupportedError(
                f"native target graph B{self.rows - 1} bucket requires {self.rows} rows"
            )
        start = int(self.session.position)
        bucket_end = start + int(self.rows)
        if bucket_end > int(self.context_limit):
            raise NativeSpecTargetGraphUnsupportedError(
                "native target graph dynamic context exceeds the captured below-1024 bucket"
            )
        batch = build_native_b2_target_batch(tokens, start_position=start, request_id=request_id)
        runtime = self.session.runtime
        _stage_dynamic_metadata(self.dynamic_metadata, batch, runtime=runtime)
        if self.device_accept_commit:
            if remaining_decode is None or int(remaining_decode) < int(self.rows):
                raise ValueError(
                    "N2 native accept/commit requires remaining_decode to cover drafts plus correction"
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
        if self.device_accept_commit:
            if self.result_payload is None:
                raise RuntimeError("N2 native accept/commit result payload is missing")
            payload = np.empty((self.result_payload.numel,), dtype=np.int32)
            copy_device_to_host(
                host_array_ptr(payload),
                _tensor_buffer(self.result_payload),
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
            if (
                accepted < 0
                or accepted >= self.rows
                or commit_row != accepted
                or committed_length != accepted + 1
                or visible_length != accepted + 1
                or next_token < 0
                or output_start + visible_length > payload.size
            ):
                raise RuntimeError("N2 native accept/commit returned an invalid bounded payload")
            output_tokens = [
                int(token)
                for token in payload[output_start:output_start + visible_length].tolist()
            ]
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
                pre_output_norm_hidden=None,
                layer_output_hidden=None,
                layer_boundary_hidden=None,
                lm_head_logits_f32=None,
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
        for cache_name in (
            "_native_spec_b1_target_graph",
            "_native_spec_b2_target_graph",
            "_native_spec_b1_target_graph_n2",
            "_native_spec_b2_target_graph_n2",
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
    cycle_id: int = 0,
    transaction_id: int = 0,
    request_id: int = 0,
    bulk_attention_mode: str = "bulk",
    use_wmma_prefill: bool = False,
    capture_linear_state_rows: bool = False,
    capture_lm_head_logits: bool = False,
    record_stage_timings: bool = False,
    sync_stage_timings: bool = False,
    defer_linear_state_commit: bool = False,
    device_accept_commit: bool = False,
) -> Qwen35GGUFNativeB2TargetGraph:
    """Capture one fixed B1/B2 target forward without executing it."""

    capture_start = time.perf_counter()
    tokens = tuple(int(token) for token in input_token_ids)
    rows = len(tokens)
    _validate_capture_admission(
        session,
        tokens,
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
        context_limit = min(1023, int(session.scratch.max_positions))
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
        accept_buffers = None
        remaining_decode_tensor = None
        result_payload = None
        visible_output_ids = None
        visible_output_lengths = None
        commit_buffers = None
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
                (ACCEPT_PACKED_PAYLOAD_FIELDS + 1 + rows,),
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
            assert session._verify_linear_state_src_conv_table_buf is not None
            assert session._verify_linear_state_dst_conv_table_buf is not None
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
                        linear_state_rows=session._verify_linear_state_src_conv_table_buf.ptr,
                        linear_state_dst=session._verify_linear_state_dst_conv_table_buf.ptr,
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
            )
            if device_accept_commit:
                assert accept_buffers is not None
                assert remaining_decode_tensor is not None
                assert result_payload is not None
                assert visible_output_ids is not None and visible_output_lengths is not None
                assert commit_buffers is not None
                assert accept_kernel is not None
                assert linear_commit_kernel is not None
                assert hidden_commit_kernel is not None
                assert accept_library is not None and commit_library is not None
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
                    session._verify_linear_state_src_conv_table_buf.ptr,
                    session._verify_linear_state_dst_conv_table_buf.ptr,
                    int(session._verify_linear_state_conv_row_nbytes),
                    session._verify_linear_state_src_recurrent_table_buf.ptr,
                    session._verify_linear_state_dst_recurrent_table_buf.ptr,
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
            dynamic_metadata=dynamic_metadata,
            token_ids_i32=token_ids_i32,
            positions_i32=positions_i32,
            accept_buffers=accept_buffers,
            remaining_decode=remaining_decode_tensor,
            result_payload=result_payload,
            visible_output_ids=visible_output_ids,
            visible_output_lengths=visible_output_lengths,
            accept_library=accept_library,
            commit_library=commit_library,
            device_accept_commit=bool(device_accept_commit),
            start_position=start,
            end_position=end,
            context_limit=context_limit,
            rows=rows,
            capture_linear_state_rows=bool(capture_linear_state_rows),
            defer_linear_state_commit=bool(defer_linear_state_commit),
            configuration_key=_native_target_configuration_key(
                bulk_attention_mode=bulk_attention_mode,
                use_wmma_prefill=bool(use_wmma_prefill),
                capture_linear_state_rows=bool(capture_linear_state_rows),
                defer_linear_state_commit=bool(defer_linear_state_commit),
                device_accept_commit=bool(device_accept_commit),
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
    if capture_lm_head_logits:
        eager_kwargs["capture_lm_head_logits"] = True
    if record_stage_timings:
        eager_kwargs["record_stage_timings"] = True
    if sync_stage_timings:
        eager_kwargs["sync_stage_timings"] = True
    rows = len(tuple(input_token_ids))
    if rows not in {2, 3}:
        reason = "native target graph requires two or three rows (one root plus B1/B2)"
        session.last_native_spec_target_fallback_reason = reason
        if not fallback:
            raise NativeSpecTargetGraphUnsupportedError(reason)
        return session.verify_target_block(input_token_ids, **eager_kwargs)
    cache_suffix = "_n2" if device_accept_commit else ""
    cache_name = f"_native_spec_b{rows - 1}_target_graph{cache_suffix}"
    graph = getattr(session, cache_name, None)
    if graph is not None and not graph.compatible_with(
        session,
        bulk_attention_mode=bulk_attention_mode,
        use_wmma_prefill=bool(use_wmma_prefill),
        capture_linear_state_rows=bool(capture_linear_state_rows),
        defer_linear_state_commit=bool(defer_linear_state_commit),
        device_accept_commit=bool(device_accept_commit),
    ):
        graph.close()
        graph = None
    try:
        if graph is None:
            graph = capture_qwen35_gguf_native_b2_target_graph(
                session,
                input_token_ids,
                cycle_id=cycle_id,
                transaction_id=transaction_id,
                request_id=request_id,
                bulk_attention_mode=bulk_attention_mode,
                use_wmma_prefill=use_wmma_prefill,
                capture_linear_state_rows=capture_linear_state_rows,
                capture_lm_head_logits=capture_lm_head_logits,
                record_stage_timings=record_stage_timings,
                sync_stage_timings=sync_stage_timings,
                defer_linear_state_commit=defer_linear_state_commit,
                device_accept_commit=device_accept_commit,
            )
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


__all__ = [
    "NativeSpecTargetGraphUnsupportedError",
    "Qwen35GGUFNativeAcceptCommitResult",
    "Qwen35GGUFNativeB2TargetGraph",
    "build_native_b2_target_batch",
    "capture_qwen35_gguf_native_b2_target_graph",
    "verify_qwen35_gguf_native_b2_target",
]
