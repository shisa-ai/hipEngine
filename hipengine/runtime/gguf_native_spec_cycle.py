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

from dataclasses import dataclass, replace
import os
import time
from typing import Any, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, host_array_ptr
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.runtime import unpack_verify_chain_dynamic_metadata_i64
from hipengine.kernels.registry import resolve
from hipengine.kvcache import KVLiveSpans
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative.buffers import TargetVerifyBufferOwner, TargetVerifyBufferSpec
from hipengine.speculative.interfaces import DraftBatch, TargetVerifyBatch, TargetVerifyBuffers
from hipengine.speculative.native_cycle import (
    NativeSpecCycleControl,
    NativeSpecCycleResult,
    NativeSpecCycleStatus,
)
# Importing the provider registers the four-axis launcher factory.  Runtime
# dispatch below still resolves by key rather than branching on backend.
from hipengine.speculative.native_cycle_graph import NativeSpecTargetGraphLauncher  # noqa: F401


class NativeSpecTargetGraphUnsupportedError(RuntimeError):
    """The fixed N1 bucket cannot safely represent this verifier invocation."""


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
    ) -> bool:
        if self.closed or session is not self.session:
            return False
        expected_key = _native_target_configuration_key(
            bulk_attention_mode=bulk_attention_mode,
            use_wmma_prefill=use_wmma_prefill,
            capture_linear_state_rows=capture_linear_state_rows,
            defer_linear_state_commit=defer_linear_state_commit,
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
    ):
        """Stage live chain metadata, replay once, and return the block result."""

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
        end = start + int(self.rows)
        if end > int(self.context_limit):
            raise NativeSpecTargetGraphUnsupportedError(
                "native target graph dynamic context exceeds the captured below-1024 bucket"
            )
        batch = build_native_b2_target_batch(tokens, start_position=start, request_id=request_id)
        runtime = self.session.runtime
        _stage_dynamic_metadata(self.dynamic_metadata, batch, runtime=runtime)
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
        self.native_result = result
        self.batch = batch
        self.control = control
        if result.status is not NativeSpecCycleStatus.COMPLETE:
            raise RuntimeError(
                "native target graph failed: "
                f"status={result.status.name} error={result.error.name} "
                f"backend_error={result.backend_error_code}"
            )

        readback_start = time.perf_counter()
        token_host = np.empty((self.rows,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(token_host),
            _tensor_buffer(self.buffers.target_top1),
            token_host.nbytes,
            runtime=runtime,
        )
        hidden_size = int(self.session.runner.hidden_size)
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

        self.session._verify_hidden_seed_rows_populated = self.rows
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

        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFBlockVerifyResult

        return Qwen35GGUFBlockVerifyResult(
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
        for cache_name in ("_native_spec_b1_target_graph", "_native_spec_b2_target_graph"):
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
    hidden = session._verify_hidden_seed_buf
    if hidden is None:
        raise RuntimeError("GGUF verifier hidden-seed buffer is closed")

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
        _stage_dynamic_metadata(dynamic_metadata, batch, runtime=runtime)
        runtime.device_synchronize()
        stream = runtime.stream_create()
        control = NativeSpecCycleControl.for_target_verify(
            cycle_id=int(cycle_id),
            buffers=buffers,
            kv_live_spans=dynamic_scratch.prefill_spans,
            hidden_seed_rows=hidden_rows,
            context_bucket=_context_bucket(context_limit),
            stream=stream,
            output_stride=rows,
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
                _target_top1_i64_ptr=buffers.target_top1.ptr,
                _enqueue_only=True,
                _prebuilt_bulk_scratch=dynamic_scratch,
                _dynamic_cursor_advance=True,
                _graph_hidden_seed_buf=hidden_rows,
                _graph_hidden_f32_a=hidden_f32_a,
                _graph_hidden_f32_b=hidden_f32_b,
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
):
    """Run N1 when admitted, otherwise preserve the exact Python verifier."""

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
    cache_name = f"_native_spec_b{rows - 1}_target_graph"
    graph = getattr(session, cache_name, None)
    if graph is not None and not graph.compatible_with(
        session,
        bulk_attention_mode=bulk_attention_mode,
        use_wmma_prefill=bool(use_wmma_prefill),
        capture_linear_state_rows=bool(capture_linear_state_rows),
        defer_linear_state_commit=bool(defer_linear_state_commit),
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
            )
            setattr(session, cache_name, graph)
    except NativeSpecTargetGraphUnsupportedError as exc:
        session.last_native_spec_target_fallback_reason = str(exc)
        if not fallback:
            raise
        return session.verify_target_block(input_token_ids, **eager_kwargs)
    try:
        return graph.launch(
            input_token_ids,
            cycle_id=int(cycle_id),
            transaction_id=int(transaction_id),
            request_id=int(request_id),
        )
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
    "Qwen35GGUFNativeB2TargetGraph",
    "build_native_b2_target_batch",
    "capture_qwen35_gguf_native_b2_target_graph",
    "verify_qwen35_gguf_native_b2_target",
]
