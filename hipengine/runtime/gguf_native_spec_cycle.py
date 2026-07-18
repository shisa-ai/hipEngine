"""GGUF adapter for the fixed-B2 native speculative target graph.

N1 is deliberately one-shot and state-generation-bound.  The adapter captures
one three-row target verifier against stable session/device addresses, binds the
capture to a versioned :class:`NativeSpecCycleControl`, and submits it through
one C++ launcher call.  Proposal and acceptance/commit remain unchanged.

Unsupported shapes and capture-unsafe session configurations use the existing
Python verifier when ``fallback=True``.  Launch or correctness failures never
fall back silently because the graph may already have mutated state/KV.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, host_array_ptr
from hipengine.core.tensor import Tensor
from hipengine.kernels.registry import resolve
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
    """Build the provider-neutral root+two-candidate chain metadata for N1."""

    tokens = tuple(int(token) for token in input_token_ids)
    start = int(start_position)
    request = int(request_id)
    if len(tokens) != 3:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 requires exactly three rows (one root plus B2)"
        )
    if start < 0:
        raise ValueError("start_position must be non-negative")
    if request < 0:
        raise ValueError("request_id must be non-negative")
    draft = DraftBatch(
        request_ids=(request,),
        candidate_tokens=tokens[1:],
        parent_positions=(start, start + 1),
        draft_depths=(1, 2),
        row_to_request=(request, request),
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

    if len(input_token_ids) != 3:
        raise NativeSpecTargetGraphUnsupportedError(
            "native target graph N1 requires exactly three rows (one root plus B2)"
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
    end = int(session.position) + 3
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
    """One-shot owner for a state-bound GGUF B2 verifier graph."""

    session: Any
    graph: int
    graph_exec: int
    stream: int
    launcher: Any
    workspace: RuntimeWorkspace
    batch: TargetVerifyBatch
    buffers: TargetVerifyBuffers
    control: NativeSpecCycleControl
    start_position: int
    end_position: int
    capture_linear_state_rows: bool
    defer_linear_state_commit: bool
    capture_wall_ms: float
    launched: bool = False
    closed: bool = False
    native_result: NativeSpecCycleResult | None = None

    def launch(self):
        """Submit the captured verifier once and return the ordinary block result."""

        if self.closed:
            raise RuntimeError("native target graph is closed")
        if self.launched:
            raise RuntimeError("state-bound native target graph may launch only once")
        if int(self.session.position) != self.start_position:
            raise RuntimeError(
                "native target graph state generation mismatch: "
                f"expected position {self.start_position}, observed {self.session.position}"
            )
        self.launched = True
        submit_start = time.perf_counter()
        result = self.launcher.launch(self.control)
        self.session.last_native_spec_target_submit_ms = (
            time.perf_counter() - submit_start
        ) * 1000.0
        self.native_result = result
        if result.status is not NativeSpecCycleStatus.COMPLETE:
            raise RuntimeError(
                "native target graph failed: "
                f"status={result.status.name} error={result.error.name} "
                f"backend_error={result.backend_error_code}"
            )

        readback_start = time.perf_counter()
        runtime = self.session.runtime
        token_host = np.empty((3,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(token_host),
            _tensor_buffer(self.buffers.target_top1),
            token_host.nbytes,
            runtime=runtime,
        )
        hidden_size = int(self.session.runner.hidden_size)
        hidden_host = np.empty((3, hidden_size), dtype=np.float32)
        hidden = self.session._verify_hidden_seed_buf
        if hidden is None:
            raise RuntimeError("GGUF verifier hidden-seed buffer is closed")
        copy_device_to_host(
            host_array_ptr(hidden_host),
            DeviceBuffer(hidden.ptr, hidden_host.nbytes),
            hidden_host.nbytes,
            runtime=runtime,
        )

        self.session._verify_hidden_seed_rows_populated = 3
        self.session._hidden_seed_fp32_populated = True
        self.session.last_native_spec_target_submitted = True
        self.session.last_native_spec_target_fallback_reason = None
        self.session.last_native_spec_target_capture_ms = float(self.capture_wall_ms)
        self.session.last_native_spec_target_readback_ms = (
            time.perf_counter() - readback_start
        ) * 1000.0
        self.session._position = self.end_position
        self.session.scratch.position_host[0] = self.end_position
        self.session.scratch.context_host[0] = self.end_position + 1
        self.session.last_verify_stage_timings_ms = {}

        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFBlockVerifyResult

        return Qwen35GGUFBlockVerifyResult(
            input_token_ids=[int(token) for token in self.batch.tokens],
            token_ids=[int(token) for token in token_host.tolist()],
            hidden_seeds=np.ascontiguousarray(hidden_host, dtype=np.float32),
            start_position=self.start_position,
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
    """Capture one fixed B2 target forward without executing it."""

    capture_start = time.perf_counter()
    tokens = tuple(int(token) for token in input_token_ids)
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
    end = start + 3
    if end > int(session.scratch.max_positions):
        raise ValueError("native target graph rows exceed resident cache capacity")
    batch = build_native_b2_target_batch(tokens, start_position=start, request_id=request_id)
    runtime = session.runtime
    session._ensure_verify_block_buffers(3, runtime=runtime)
    session._ensure_verify_lm_head_buffers(3, runtime=runtime)
    if capture_linear_state_rows:
        session._ensure_verify_linear_state_row_buffers(3, runtime=runtime)
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
                bucket="native_v1_b2_target_graph",
                device=workspace.device,
                max_rows=3,
                max_requests=1,
                mode="verify_chain",
                metadata_dtype=DType.INT64,
            ),
            workspace=workspace,
        )
        buffers = owner.bind(batch, transaction_id=int(transaction_id))
        _stage_target_batch(batch, buffers, runtime=runtime)
        hidden_rows = Tensor.from_handle(
            hidden.ptr,
            (3, int(session.runner.hidden_size)),
            DType.FP32,
            workspace.device,
        )
        spans = replace(
            session.scratch.decode_spans,
            max_live_count=end,
            span_role="verify_chain",
        )
        runtime.device_synchronize()
        stream = runtime.stream_create()
        control = NativeSpecCycleControl.for_target_verify(
            cycle_id=int(cycle_id),
            buffers=buffers,
            kv_live_spans=spans,
            hidden_seed_rows=hidden_rows,
            context_bucket=_context_bucket(end),
            stream=stream,
            output_stride=3,
        )

        runtime.stream_begin_capture(stream)
        try:
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
            start_position=start,
            end_position=end,
            capture_linear_state_rows=bool(capture_linear_state_rows),
            defer_linear_state_commit=bool(defer_linear_state_commit),
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
    try:
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
    except NativeSpecTargetGraphUnsupportedError as exc:
        session.last_native_spec_target_fallback_reason = str(exc)
        if not fallback:
            raise
        return session.verify_target_block(input_token_ids, **eager_kwargs)
    with graph:
        return graph.launch()


__all__ = [
    "NativeSpecTargetGraphUnsupportedError",
    "Qwen35GGUFNativeB2TargetGraph",
    "build_native_b2_target_batch",
    "capture_qwen35_gguf_native_b2_target_graph",
    "verify_qwen35_gguf_native_b2_target",
]
