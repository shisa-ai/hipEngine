"""Native execution and DraftModel provider for a GGUF trailing NextN block."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import resolve_backend
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import argmax_f32_rows_i32
from hipengine.kernels.hip_gfx1100.quant.gguf_q3_k_gemv import register_gguf_q3_k_gemv_kernels
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    build_gguf_q6_k_pack8_gemv,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
)
from hipengine.kernels.hip_gfx1100.runtime import (
    advance_decode_position_i64,
    build_runtime_state,
    copy_i32_to_i64,
    set_decode_position_i64,
)
from hipengine.kernels.registry import KernelKey, MissingKernelError, is_registered, resolve
from hipengine.loading.qwen35_gguf_materialize import Qwen35GGUFDeviceWeight
from hipengine.loading.qwen35_gguf_nextn_materialize import (
    Qwen35GGUFNextNResidentWeights,
    materialize_qwen35_gguf_nextn_weights,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)
from hipengine.speculative.mtp import (
    MTP_CHAIN_CANDIDATE_BUDGETS,
    MtpDraftRequest,
    MtpProposalContext,
    compile_mtp_chain,
)
from hipengine.speculative.interfaces import DraftBatch


@dataclass(frozen=True, slots=True)
class Qwen35GGUFNextNStepResult:
    """One draft-block output before target verification."""

    request_id: int
    input_token: int
    position: int
    token_id: int
    logit: float
    hidden: Tensor
    logits: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Qwen35GGUFNextNStateAdvance:
    """A draft-state append whose discarded prediction was not scored."""

    request_id: int
    input_token: int
    position: int


@dataclass(slots=True)
class Qwen35GGUFNextNRequestCheckpoint:
    """Request-local provisional provider state before one proposal."""

    request_id: int
    slot: int
    state_pairs: tuple[tuple[DeviceBuffer, DeviceBuffer], ...]
    position: int
    context_length: int
    released: bool = False


@dataclass(frozen=True, slots=True)
class Qwen35GGUFNextNDeviceProposal:
    """One in-flight cached proposal graph awaiting target-stream retirement.

    The result buffer stores contiguous ``(int32 token, float32 value)`` rows.
    A target graph may wait on ``completion_event`` and consume those rows on
    device; no host code may read the result buffer before that wait retires.
    """

    request_id: int
    root_token: int
    root_position: int
    budget: int
    result_ptr: int
    result_nbytes: int
    completion_event: int
    stream: int
    final_hidden: Tensor
    hidden_rows: Tensor | None = None

    def __post_init__(self) -> None:
        if self.request_id < 0 or self.root_token < 0 or self.root_position < 0:
            raise ValueError("device proposal identity fields must be non-negative")
        if self.budget not in _NEXTN_EXACT_CHAIN_GRAPH_BUDGETS:
            raise ValueError("device proposal budget is outside the exact graph ladder")
        if self.result_ptr <= 0 or self.completion_event <= 0 or self.stream <= 0:
            raise ValueError("device proposal requires live result, event, and stream handles")
        if self.result_nbytes != self.budget * _NEXTN_TOP1_RESULT_NBYTES:
            raise ValueError("device proposal result span must cover every top-1 row")
        if self.final_hidden.dtype != DType.BF16 or self.final_hidden.shape[0] != 1:
            raise ValueError("device proposal final hidden row must be rank-2 BF16")
        if self.hidden_rows is not None and (
            self.hidden_rows.dtype != DType.BF16
            or self.hidden_rows.shape
            != (self.budget, self.final_hidden.shape[1])
            or self.hidden_rows.device != self.final_hidden.device
        ):
            raise ValueError(
                "device proposal hidden rows must be BF16 [budget, hidden_size]"
            )


@dataclass(frozen=True, slots=True)
class Qwen35GGUFNextNBatchDeviceProposal:
    """Request-major physical proposal rows retained on the device.

    ``token_ids`` contains only active candidate rows, ordered by request and
    then draft depth. ``hidden_rows`` has the same logical ownership and stays
    device-resident for provider checkpoint repair after target verification.
    """

    request_ids: tuple[int, ...]
    root_tokens: tuple[int, ...]
    root_positions: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    token_ids: Tensor
    hidden_rows: tuple[tuple[Tensor, ...], ...]

    def __post_init__(self) -> None:
        rows = len(self.request_ids)
        if rows < 1 or len(set(self.request_ids)) != rows:
            raise ValueError("batch device proposal requires unique request rows")
        if any(
            len(values) != rows
            for values in (
                self.root_tokens,
                self.root_positions,
                self.candidate_counts,
                self.hidden_rows,
            )
        ):
            raise ValueError("batch device proposal fields must align")
        if any(
            value < 0
            for value in (*self.request_ids, *self.root_tokens, *self.root_positions)
        ):
            raise ValueError("batch device proposal identity must be non-negative")
        if any(
            count <= 0 or count not in MTP_CHAIN_CANDIDATE_BUDGETS
            for count in self.candidate_counts
        ):
            raise ValueError("batch device proposal candidate count is unsupported")
        candidate_rows = sum(self.candidate_counts)
        if self.token_ids.dtype != DType.INT32 or self.token_ids.shape != (
            candidate_rows,
        ):
            raise ValueError(
                "batch device proposal token_ids must be INT32 [candidate_rows]"
            )
        for count, hidden in zip(
            self.candidate_counts,
            self.hidden_rows,
            strict=True,
        ):
            if len(hidden) != count:
                raise ValueError("batch device proposal hidden rows must match counts")
            if any(
                row.dtype != DType.BF16 or len(row.shape) != 2 or row.shape[0] != 1
                for row in hidden
            ):
                raise ValueError("batch device proposal hidden rows must be BF16 [1,H]")


@dataclass(frozen=True, slots=True)
class _Qwen35GGUFNextNProposalGraph:
    """One exact greedy proposal-chain graph bound to a resident request slot."""

    slot: int
    budget: int
    stream: int
    graph: int
    graph_exec: int
    completion_event: int


# The fixed graph topology is proven for the production B1-B3 ladder through
# scalar full-attention context 1,023. Larger cache allocations may still use it
# while the live chain fits; requests crossing into split-K retain the exact
# eager chain until that separate topology is gated.
_NEXTN_EXACT_CHAIN_GRAPH_BUDGETS = (1, 2, 3)
_NEXTN_EXACT_CHAIN_GRAPH_MAX_CONTEXT = 1023
_NEXTN_TOP1_RESULT_DTYPE = np.dtype([("token", np.int32), ("value", np.float32)])
_NEXTN_TOP1_RESULT_NBYTES = int(_NEXTN_TOP1_RESULT_DTYPE.itemsize)
_NEXTN_TOP1_RESULT_CAPACITY = max(_NEXTN_EXACT_CHAIN_GRAPH_BUDGETS)


def borrow_qwen35_gguf_nextn_fallback_weights(
    target: Qwen35GGUFResidentSession,
) -> dict[str, Qwen35GGUFDeviceWeight]:
    """Borrow effective target roots without rehydrating a mapped token table."""

    if target.runner is None or target.runner.weights is None:
        raise RuntimeError("target GGUF weights are unavailable")
    return {
        "token_embedding": target.runner.ensure_device_token_embedding(
            runtime=target.runtime,
        ),
        "lm_head": target.runner.weights.root("lm_head"),
    }


def _resolve_nextn_executor_backend(
    requested: str | None,
    borrowed_fallback_weights: Mapping[str, Qwen35GGUFDeviceWeight] | None,
) -> str:
    """Resolve one backend shared by the draft block and borrowed target roots."""

    borrowed_backends = {
        str(weight.backend) for weight in (borrowed_fallback_weights or {}).values()
    }
    if len(borrowed_backends) > 1:
        raise ValueError(
            "borrowed NextN fallback weights use multiple backends: "
            f"{sorted(borrowed_backends)}"
        )
    borrowed_backend = next(iter(borrowed_backends), None)
    normalized = "auto" if requested is None else str(requested).strip()
    if normalized == "auto" and borrowed_backend is not None:
        return borrowed_backend
    resolved = resolve_backend(normalized)
    if borrowed_backend is not None and resolved != borrowed_backend:
        raise ValueError(
            f"NextN backend {resolved!r} does not match borrowed fallback backend "
            f"{borrowed_backend!r}"
        )
    return resolved


class Qwen35GGUFNextNStepExecutor(Protocol):
    hidden_size: int

    def run_step(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        return_logits: bool = False,
    ) -> Qwen35GGUFNextNStepResult: ...

    def run_chain(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
        return_logits: bool = False,
    ) -> tuple[Qwen35GGUFNextNStepResult, ...]: ...

    def advance_state_only(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
    ) -> Qwen35GGUFNextNStateAdvance: ...

    def enqueue_prompt_rows(
        self,
        request_id: int,
        token_ids: tuple[int, ...],
        *,
        position_start: int,
        target_hidden_base_ptr: int,
        hidden_stride_bytes: int,
        stream: int,
    ) -> None: ...

    def finish_prompt_priming(
        self,
        request_id: int,
        *,
        stream: int,
        synchronize: bool,
    ) -> None: ...

    def reset_request(self, request_id: int) -> None: ...

    def close(self) -> None: ...


class Qwen35GGUFNextNExecutor:
    """Resident native executor for the separate GGUF NextN draft model.

    Target-normalized hidden rows enter through ``target_hidden``.  The executor
    applies NextN embedding/hidden normalization and Q8_0 fusion, runs only the
    mapped trailing full-attention block with its architecture-selected FFN,
    then uses the mapped shared-head
    norm and target output fallback.  It never mutates or extends the AR map.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        max_positions: int,
        max_requests: int = 1,
        runtime: HipRuntime | None = None,
        compiler_version: str | None = None,
        require_cached_build: bool = False,
        borrowed_fallback_weights: Mapping[str, Qwen35GGUFDeviceWeight] | None = None,
        backend: str | None = None,
    ) -> None:
        if max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        self.model = Path(model)
        self.backend = _resolve_nextn_executor_backend(
            backend, borrowed_fallback_weights
        )
        self.runtime = runtime or get_hip_runtime()
        self.compiler_version = compiler_version
        self.require_cached_build = bool(require_cached_build)
        self.max_requests = int(max_requests)
        self.closed = False
        register_gguf_q3_k_gemv_kernels()
        self.weights: Qwen35GGUFNextNResidentWeights | None = materialize_qwen35_gguf_nextn_weights(
            self.model,
            borrowed_fallback_weights=borrowed_fallback_weights,
            runtime=self.runtime,
            backend=self.backend,
        )
        adapted = self.weights.as_full_stack_weights()
        self.runner = Qwen35GGUFFullStackRunner(
            self.model,
            runtime=self.runtime,
            compiler_version=self.compiler_version,
            require_cached_build=self.require_cached_build,
            backend=self.backend,
            resident_weights=adapted,
            owns_resident_weights=False,
        )
        self.hidden_size = int(self.runner.hidden_size)
        self.vocab_size = int(self.runner.vocab_size)
        self._batch_session = Qwen35GGUFResidentSession(
            self.model,
            runtime=self.runtime,
            compiler_version=self.compiler_version,
            require_cached_build=self.require_cached_build,
            backend=self.backend,
            shared_runner=self.runner,
            max_sequence_length=int(max_positions),
            max_batch_size=self.max_requests,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        )
        self.scratch = self._batch_session._target_scratch_owner
        if self.scratch is None:
            raise RuntimeError("GGUF NextN batch scratch is unavailable")
        self._batch_sessions = (
            self._batch_session,
            *tuple(
                self._batch_session.resident_slot_view(index)
                for index in range(1, self.max_requests)
            ),
        )
        self.scratch.zero_states(self.runtime)
        root_state_buffers = (
            *self.scratch.layer_conv_states,
            *self.scratch.layer_recurrent_states,
        )
        self._provider_root_state_snapshots = tuple(
            None
            if state is None
            else malloc(int(state.nbytes), runtime=self.runtime)
            for state in root_state_buffers
        )
        self._provider_root_state_metadata: dict[int, tuple[int, int, int]] = {}
        self._request_slots: dict[int, int] = {}
        # Pageable host token arrays backing nonblocking prompt-prime H2D
        # copies stay alive until the target prefill's completion boundary.
        self._prompt_priming_staging: dict[int, list[np.ndarray]] = {}
        self._token_host = np.zeros((self.max_requests,), dtype=np.int64)
        self._logits_host = np.empty((self.max_requests, self.vocab_size), dtype=np.float32)
        hidden_bytes = self.max_requests * self.hidden_size * DType.BF16.itemsize
        self._token_buf = malloc(self._token_host.nbytes, runtime=self.runtime)
        self._embedding_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._enorm_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._hnorm_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._batch_input_hidden = malloc(hidden_bytes, runtime=self.runtime)
        self._batch_candidate_tokens_i32 = malloc(
            self.max_requests
            * _NEXTN_TOP1_RESULT_CAPACITY
            * DType.INT32.itemsize,
            runtime=self.runtime,
        )
        self._fusion_buf = malloc(2 * hidden_bytes, runtime=self.runtime)
        self._fused_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._layer_out_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._final_hidden_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._proposal_history_hidden = malloc(
            self.max_requests
            * _NEXTN_TOP1_RESULT_CAPACITY
            * self.hidden_size
            * DType.BF16.itemsize,
            runtime=self.runtime,
        )
        self._logits_buf = malloc(self._logits_host.nbytes, runtime=self.runtime)
        self._lm_head_top1_kernel = None
        self._lm_head_top1_weight: Qwen35GGUFDeviceWeight | None = None
        self._lm_head_top1_block_values: DeviceBuffer | None = None
        self._lm_head_top1_block_indices: DeviceBuffer | None = None
        self._lm_head_top1_result: DeviceBuffer | None = None
        self._lm_head_top1_libraries: Mapping[str, object] | None = None
        self.last_lm_head_path = "unobserved"
        self._prepare_exact_lm_head_top1()
        self._proposal_graphs: dict[tuple[int, int], _Qwen35GGUFNextNProposalGraph] = {}
        self._proposal_graph_unavailable: set[tuple[int, int]] = set()
        self._proposal_graph_runtime_library: object | None = None
        self._proposal_graph_captures = 0
        self._proposal_graph_replays = 0
        self._proposal_graph_last_status = "unobserved"
        self._proposal_graph_last_error: str | None = None
        self._proposal_target_hidden: DeviceBuffer | None = None
        self._proposal_results: DeviceBuffer | None = None
        self._proposal_results_host: np.ndarray | None = None
        if self._lm_head_top1_kernel is not None:
            self._proposal_target_hidden = malloc(hidden_bytes, runtime=self.runtime)
            self._proposal_results = malloc(
                self.max_requests * _NEXTN_TOP1_RESULT_CAPACITY * _NEXTN_TOP1_RESULT_NBYTES,
                runtime=self.runtime,
            )
            self._proposal_results_host = np.empty(
                (self.max_requests, _NEXTN_TOP1_RESULT_CAPACITY),
                dtype=_NEXTN_TOP1_RESULT_DTYPE,
            )
            self._proposal_graph_last_status = "ready"
        else:
            self._proposal_graph_last_status = "ineligible"
        self._buffers = tuple(
            buffer
            for buffer in (
                self._token_buf,
                self._embedding_buf,
                self._enorm_buf,
                self._hnorm_buf,
                self._batch_input_hidden,
                self._batch_candidate_tokens_i32,
                self._fusion_buf,
                self._fused_buf,
                self._layer_out_buf,
                self._final_hidden_buf,
                self._proposal_history_hidden,
                self._logits_buf,
                self._lm_head_top1_block_values,
                self._lm_head_top1_block_indices,
                self._lm_head_top1_result,
                self._proposal_target_hidden,
                self._proposal_results,
                *self._provider_root_state_snapshots,
            )
            if buffer is not None
        )

    def _prepare_exact_lm_head_top1(self) -> None:
        """Bind compact exact top-1 scoring when the resident head supports it."""

        if self.weights is None or self.hidden_size % 256 != 0 or self.vocab_size % 8 != 0:
            return
        weight = self.weights.fallback("lm_head")
        key = KernelKey(
            self.weights.backend,
            "linear+argmax",
            weight.spec.quant_key,
            "proposal_top1_exact_bf16",
        )
        if not is_registered(key):
            return
        try:
            kernel = resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            )
        except MissingKernelError:
            return
        libraries = {
            "q6_pack8": build_gguf_q6_k_pack8_gemv(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            ),
            "q6_t16": build_gguf_q6_k_t16_gemv(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            ),
        }
        block_nbytes = (self.vocab_size // 8) * DType.FP32.itemsize
        self._lm_head_top1_block_values = malloc(block_nbytes, runtime=self.runtime)
        self._lm_head_top1_block_indices = malloc(block_nbytes, runtime=self.runtime)
        self._lm_head_top1_result = malloc(DType.INT32.itemsize + DType.FP32.itemsize, runtime=self.runtime)
        self._lm_head_top1_kernel = kernel
        self._lm_head_top1_weight = weight
        self._lm_head_top1_libraries = libraries

    def _enqueue_exact_lm_head_top1(
        self,
        hidden_ptr: int,
        token_out_ptr: int,
        value_out_ptr: int,
        *,
        stream: int = 0,
    ) -> bool:
        """Enqueue exact compact scoring into caller-owned device outputs."""

        if (
            self._lm_head_top1_kernel is None
            or self._lm_head_top1_weight is None
            or self._lm_head_top1_block_values is None
            or self._lm_head_top1_block_indices is None
            or self._lm_head_top1_libraries is None
        ):
            return False
        self._lm_head_top1_kernel(
            self._lm_head_top1_weight,
            int(hidden_ptr),
            self._logits_buf.ptr,
            self._lm_head_top1_block_values.ptr,
            self._lm_head_top1_block_indices.ptr,
            int(token_out_ptr),
            int(value_out_ptr),
            1,
            self.hidden_size,
            self.vocab_size,
            stream=int(stream),
            libraries=self._lm_head_top1_libraries,
            runtime=self.runtime,
        )
        return True

    def _run_exact_lm_head_top1(
        self,
        hidden_ptr: int,
        *,
        stream: int = 0,
    ) -> tuple[int, float] | None:
        """Return the exact token/value pair without materializing vocab logits."""

        if self._lm_head_top1_result is None:
            return None
        result_ptr = int(self._lm_head_top1_result.ptr)
        if not self._enqueue_exact_lm_head_top1(
            hidden_ptr,
            result_ptr,
            result_ptr + DType.INT32.itemsize,
            stream=stream,
        ):
            return None
        self.runtime.device_synchronize()
        result_host = np.empty((1,), dtype=_NEXTN_TOP1_RESULT_DTYPE)
        copy_device_to_host(
            host_array_ptr(result_host),
            self._lm_head_top1_result,
            self._lm_head_top1_result.nbytes,
            runtime=self.runtime,
        )
        return int(result_host["token"][0]), float(result_host["value"][0])

    def _sample_lm_head(
        self,
        hidden_ptr: int,
        *,
        return_logits: bool,
        stream: int = 0,
    ) -> tuple[int, float, np.ndarray | None]:
        if not return_logits:
            compact = self._run_exact_lm_head_top1(hidden_ptr, stream=stream)
            if compact is not None:
                self.last_lm_head_path = "exact_q6_top1"
                return compact[0], compact[1], None

        if self.weights is None:
            raise RuntimeError("GGUF NextN executor is closed")
        launch_gguf_linear(
            self.weights.fallback("lm_head"),
            hidden_ptr,
            self._logits_buf.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            stream=int(stream),
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(self._logits_host),
            self._logits_buf,
            self._logits_host.nbytes,
            runtime=self.runtime,
        )
        if not np.all(np.isfinite(self._logits_host)):
            raise FloatingPointError("GGUF NextN logits contain NaN or Inf")
        token = int(np.argmax(self._logits_host[0]))
        self.last_lm_head_path = "full_logits"
        logits = self._logits_host.copy() if return_logits else None
        return token, float(self._logits_host[0, token]), logits

    def _set_batch_session_position(self, slot: int, position: int) -> None:
        sessions = getattr(self, "_batch_sessions", None)
        if sessions is not None:
            sessions[int(slot)]._position = int(position)

    def _publish_batch_consumed_positions(
        self,
        request_ids: Sequence[int],
        positions: Sequence[int],
    ) -> None:
        """Restore c1 cursor semantics after packed hidden-row execution.

        ``step_hidden_batch_native`` advances each session to the next input
        cursor, but its generic packed-state scatter also publishes that cursor
        as the last consumed position and increments context once more. NextN
        checkpoints and after-root snapshots require the c1 convention instead:
        ``position_host=input_position`` and ``context=input_position+1`` while
        ``session.position`` remains the next input cursor.
        """

        ids = tuple(int(value) for value in request_ids)
        consumed = tuple(int(value) for value in positions)
        if len(ids) != len(consumed):
            raise ValueError("batch consumed positions must align with request IDs")
        for request_id, position in zip(ids, consumed, strict=True):
            slot = self._slot(request_id)
            slot_scratch = self.scratch.for_slot(slot, span_role="decode")
            slot_scratch.position_host[0] = position
            slot_scratch.context_host[0] = position + 1
            set_decode_position_i64(
                slot_scratch.position_buf.ptr,
                slot_scratch.context_buf.ptr,
                position,
                stream=0,
                library=self._batch_session._runtime_state_library,
                runtime=self.runtime,
            )

    def _slot(self, request_id: int) -> int:
        request_id = int(request_id)
        slot = self._request_slots.get(request_id)
        if slot is not None:
            return slot
        if len(self._request_slots) >= self.max_requests:
            raise RuntimeError("GGUF NextN executor has no free request slot")
        used = set(self._request_slots.values())
        slot = next(index for index in range(self.max_requests) if index not in used)
        self._request_slots[request_id] = slot
        return slot

    def _run_block(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        stream: int = 0,
        token_ready: bool = False,
        position_ready: bool = False,
        attention_context_cap: int | None = None,
    ) -> tuple[int, int]:
        """Run the state-mutating NextN block before shared norm/head scoring."""

        if self.closed or self.weights is None:
            raise RuntimeError("GGUF NextN executor is closed")
        if target_hidden.dtype != DType.BF16 or target_hidden.shape != (1, self.hidden_size):
            raise ValueError(f"target_hidden must be BF16 with shape (1, {self.hidden_size})")
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError("token_id is outside the GGUF vocabulary")
        slot = self._slot(request_id)
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        if position_ready:
            # Graph capture supplies dynamic device metadata. Keep the host mirror
            # coherent only so the full-attention helper does not enqueue a
            # synchronous legacy-stream upload while another stream is captured.
            slot_scratch.position_host[0] = int(position)
            slot_scratch.context_host[0] = int(position) + 1
        else:
            slot_scratch.set_full_attention_position(int(position), self.runtime)
        hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        token_ptr = self._token_buf.ptr + slot * DType.INT64.itemsize
        embedding_ptr = self._embedding_buf.ptr + slot * hidden_nbytes
        fusion_ptr = self._fusion_buf.ptr + slot * 2 * hidden_nbytes
        fused_ptr = self._fused_buf.ptr + slot * hidden_nbytes
        layer_out_ptr = self._layer_out_buf.ptr + slot * hidden_nbytes
        final_hidden_ptr = self._final_hidden_buf.ptr + slot * hidden_nbytes

        if not token_ready:
            token_host = np.asarray([int(token_id)], dtype=np.int64)
            copy_host_to_device(
                DeviceBuffer(token_ptr, token_host.nbytes),
                host_array_ptr(token_host),
                token_host.nbytes,
                runtime=self.runtime,
            )
        launch_gguf_embedding(
            self.weights.fallback("token_embedding"),
            token_ptr,
            embedding_ptr,
            rows=1,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            stream=int(stream),
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            embedding_ptr,
            self.weights.nextn("enorm").allocation().tensor.ptr,
            fusion_ptr,
            rows=1,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            stream=int(stream),
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            target_hidden.ptr,
            self.weights.nextn("hnorm").allocation().tensor.ptr,
            fusion_ptr + hidden_nbytes,
            rows=1,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            stream=int(stream),
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.weights.nextn("eh_proj"),
            fusion_ptr,
            fused_ptr,
            rows=1,
            in_features=2 * self.hidden_size,
            out_features=self.hidden_size,
            stream=int(stream),
            runtime=self.runtime,
        )
        self.runner._run_full_attention_layer(
            0,
            fused_ptr,
            layer_out_ptr,
            slot_scratch,
            position=int(position),
            stream=int(stream),
            attention_max_context_len=attention_context_cap,
        )
        return layer_out_ptr, final_hidden_ptr

    def run_step(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        return_logits: bool = False,
    ) -> Qwen35GGUFNextNStepResult:
        layer_out_ptr, final_hidden_ptr = self._run_block(
            request_id,
            token_id,
            position,
            target_hidden,
        )
        slot = self._slot(request_id)
        self._set_batch_session_position(slot, int(position) + 1)
        if self.weights is None:
            raise RuntimeError("GGUF NextN executor is closed")
        gguf_rmsnorm_bf16_f32_weight(
            layer_out_ptr,
            self.weights.fallback("output_norm").allocation().tensor.ptr,
            final_hidden_ptr,
            rows=1,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=self.runtime,
        )
        token, logit, logits = self._sample_lm_head(
            final_hidden_ptr,
            return_logits=bool(return_logits),
        )
        hidden = Tensor.from_handle(
            final_hidden_ptr,
            (1, self.hidden_size),
            DType.BF16,
            Device("hip", 0),
        )
        return Qwen35GGUFNextNStepResult(
            request_id=int(request_id),
            input_token=int(token_id),
            position=int(position),
            token_id=token,
            logit=logit,
            hidden=hidden,
            logits=logits,
        )

    def _run_step_batch_impl(
        self,
        request_ids: Sequence[int],
        token_ids: Sequence[int] | None,
        positions: Sequence[int],
        target_hidden: Tensor,
        *,
        score_output: bool,
        token_ids_device: Sequence[Tensor] | None = None,
    ) -> tuple[Qwen35GGUFNextNStepResult | Qwen35GGUFNextNStateAdvance, ...]:
        """Run one physically row-batched NextN state transition."""

        ids = tuple(int(value) for value in request_ids)
        device_tokens = (
            None if token_ids_device is None else tuple(token_ids_device)
        )
        tokens = (
            (0,) * len(ids)
            if device_tokens is not None
            else tuple(int(value) for value in (() if token_ids is None else token_ids))
        )
        pos = tuple(int(value) for value in positions)
        rows = len(ids)
        if rows <= 1 or rows > self.max_requests:
            raise ValueError("NextN batch rows must be in [2, max_requests]")
        if len(set(ids)) != rows or len(tokens) != rows or len(pos) != rows:
            raise ValueError("NextN batch request/token/position rows must align")
        if device_tokens is not None and (
            token_ids is not None
            or len(device_tokens) != rows
            or any(
                tensor.dtype != DType.INT32
                or tensor.shape != (1,)
                or tensor.device.kind != "hip"
                for tensor in device_tokens
            )
        ):
            raise ValueError(
                "device NextN tokens must be one HIP INT32 scalar per request"
            )
        if target_hidden.dtype != DType.BF16 or target_hidden.shape != (
            rows,
            self.hidden_size,
        ):
            raise ValueError("target_hidden must be contiguous BF16 [rows, hidden]")
        if device_tokens is None and any(
            token < 0 or token >= self.vocab_size for token in tokens
        ):
            raise ValueError("NextN batch token is outside the vocabulary")
        slots = tuple(self._slot(request_id) for request_id in ids)
        sessions = tuple(self._batch_sessions[slot] for slot in slots)
        session_positions = tuple(int(session.position) for session in sessions)
        if session_positions != pos:
            scratch_positions = tuple(
                int(
                    self.scratch.for_slot(
                        slot,
                        span_role="decode",
                    ).position_host[0]
                )
                for slot in slots
            )
            raise ValueError(
                "NextN batch position does not match provider cursor: "
                f"sessions={session_positions!r} scratch={scratch_positions!r} "
                f"requested={pos!r} ids={ids!r} slots={slots!r}"
            )
        if device_tokens is None:
            self._token_host[:rows] = np.asarray(tokens, dtype=np.int64)
            copy_host_to_device(
                self._token_buf,
                host_array_ptr(self._token_host),
                rows * DType.INT64.itemsize,
                runtime=self.runtime,
            )
        else:
            for row, token in enumerate(device_tokens):
                copy_i32_to_i64(
                    token.ptr,
                    self._token_buf.ptr + row * DType.INT64.itemsize,
                    1,
                    library=self._batch_session._runtime_state_library,
                    runtime=self.runtime,
                )
        launch_gguf_embedding(
            self.weights.fallback("token_embedding"),
            self._token_buf.ptr,
            self._embedding_buf.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            self._embedding_buf.ptr,
            self.weights.nextn("enorm").allocation().tensor.ptr,
            self._enorm_buf.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            target_hidden.ptr,
            self.weights.nextn("hnorm").allocation().tensor.ptr,
            self._hnorm_buf.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=self.runtime,
        )
        hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        for row in range(rows):
            destination = self._fusion_buf.ptr + row * 2 * hidden_nbytes
            self.runtime.memcpy_async(
                destination,
                self._enorm_buf.ptr + row * hidden_nbytes,
                hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
            self.runtime.memcpy_async(
                destination + hidden_nbytes,
                self._hnorm_buf.ptr + row * hidden_nbytes,
                hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
        launch_gguf_linear(
            self.weights.nextn("eh_proj"),
            self._fusion_buf.ptr,
            self._fused_buf.ptr,
            rows=rows,
            in_features=2 * self.hidden_size,
            out_features=self.hidden_size,
            runtime=self.runtime,
        )
        self._batch_session.step_hidden_batch_native(
            self._fused_buf.ptr,
            sessions=sessions,
            positions=pos,
            output_hidden_ptr=self._final_hidden_buf.ptr,
            logits_ptr=self._logits_buf.ptr,
            score_output=bool(score_output),
        )
        self._publish_batch_consumed_positions(ids, pos)
        if not score_output:
            self.last_lm_head_path = "physical_batch_state_only"
            return tuple(
                Qwen35GGUFNextNStateAdvance(
                    request_id=request_id,
                    input_token=token_id,
                    position=position,
                )
                for request_id, token_id, position in zip(
                    ids,
                    tokens,
                    pos,
                    strict=True,
                )
            )
        logits = np.empty((rows, self.vocab_size), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits),
            self._logits_buf,
            logits.nbytes,
            runtime=self.runtime,
        )
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("GGUF NextN batch logits contain NaN or Inf")
        results: list[Qwen35GGUFNextNStepResult] = []
        for row, (request_id, token_id, position) in enumerate(
            zip(ids, tokens, pos, strict=True)
        ):
            selected = int(np.argmax(logits[row]))
            results.append(
                Qwen35GGUFNextNStepResult(
                    request_id=request_id,
                    input_token=token_id,
                    position=position,
                    token_id=selected,
                    logit=float(logits[row, selected]),
                    hidden=Tensor.from_handle(
                        self._final_hidden_buf.ptr + row * hidden_nbytes,
                        (1, self.hidden_size),
                        DType.BF16,
                        Device("hip", 0),
                    ),
                    logits=None,
                )
            )
        self.last_lm_head_path = "physical_batch_full_logits"
        return tuple(results)

    def run_step_batch(
        self,
        request_ids: Sequence[int],
        token_ids: Sequence[int],
        positions: Sequence[int],
        target_hidden: Tensor,
    ) -> tuple[Qwen35GGUFNextNStepResult, ...]:
        """Run one physically row-batched NextN step for independent requests."""

        results = self._run_step_batch_impl(
            request_ids,
            token_ids,
            positions,
            target_hidden,
            score_output=True,
        )
        if not all(isinstance(result, Qwen35GGUFNextNStepResult) for result in results):
            raise RuntimeError("NextN scored batch returned a state-only result")
        return tuple(
            result
            for result in results
            if isinstance(result, Qwen35GGUFNextNStepResult)
        )

    def advance_state_batch_only(
        self,
        request_ids: Sequence[int],
        token_ids: Sequence[int],
        positions: Sequence[int],
        target_hidden: Tensor,
    ) -> tuple[Qwen35GGUFNextNStateAdvance, ...]:
        """Consume independent accepted inputs in one backbone without scoring."""

        results = self._run_step_batch_impl(
            request_ids,
            token_ids,
            positions,
            target_hidden,
            score_output=False,
        )
        if not all(isinstance(result, Qwen35GGUFNextNStateAdvance) for result in results):
            raise RuntimeError("NextN state-only batch returned a scored result")
        return tuple(
            result
            for result in results
            if isinstance(result, Qwen35GGUFNextNStateAdvance)
        )

    def advance_state_batch_only_device(
        self,
        request_ids: Sequence[int],
        token_ids: Sequence[Tensor],
        positions: Sequence[int],
        target_hidden: Tensor,
    ) -> None:
        """Consume device-resident accepted IDs without host materialization."""

        results = self._run_step_batch_impl(
            request_ids,
            None,
            positions,
            target_hidden,
            score_output=False,
            token_ids_device=token_ids,
        )
        if not all(isinstance(result, Qwen35GGUFNextNStateAdvance) for result in results):
            raise RuntimeError("NextN device state-only batch returned a scored result")

    def _device_top1_rows(self, rows: int) -> Tensor:
        owner = self._batch_session
        owner._ensure_verify_lm_head_buffers(int(rows), runtime=self.runtime)
        if (
            owner._verify_lm_block_values is None
            or owner._verify_lm_block_indices_i32 is None
            or owner._verify_lm_out_indices_i32 is None
            or owner._verify_lm_out_values is None
        ):
            raise RuntimeError("GGUF NextN device top-1 buffers are unavailable")
        argmax_f32_rows_i32(
            self._logits_buf.ptr,
            owner._verify_lm_block_values.ptr,
            owner._verify_lm_block_indices_i32.ptr,
            owner._verify_lm_out_indices_i32.ptr,
            owner._verify_lm_out_values.ptr,
            int(rows),
            self.vocab_size,
            threads=owner._lm_head_threads,
            library=owner._lm_head_library,
            runtime=self.runtime,
        )
        return Tensor.from_handle(
            owner._verify_lm_out_indices_i32.ptr,
            (int(rows),),
            DType.INT32,
            Device("hip", 0),
        )

    def _run_step_batch_device_top1(
        self,
        request_ids: tuple[int, ...],
        positions: tuple[int, ...],
        target_hidden: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Consume pre-staged INT64 tokens and leave row top-1 IDs on device."""

        ids = tuple(int(value) for value in request_ids)
        pos = tuple(int(value) for value in positions)
        rows = len(ids)
        if rows <= 1 or rows > self.max_requests or len(pos) != rows:
            raise ValueError("device NextN batch rows must be in [2, max_requests]")
        if len(set(ids)) != rows:
            raise ValueError("device NextN batch request IDs must be unique")
        if target_hidden.dtype != DType.BF16 or target_hidden.shape != (
            rows,
            self.hidden_size,
        ):
            raise ValueError("device target_hidden must be BF16 [rows,H]")
        slots = tuple(self._slot(request_id) for request_id in ids)
        sessions = tuple(self._batch_sessions[slot] for slot in slots)
        if tuple(int(session.position) for session in sessions) != pos:
            raise ValueError("device NextN batch positions do not match provider cursors")
        launch_gguf_embedding(
            self.weights.fallback("token_embedding"),
            self._token_buf.ptr,
            self._embedding_buf.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            self._embedding_buf.ptr,
            self.weights.nextn("enorm").allocation().tensor.ptr,
            self._enorm_buf.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            target_hidden.ptr,
            self.weights.nextn("hnorm").allocation().tensor.ptr,
            self._hnorm_buf.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=self.runtime,
        )
        hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        for row in range(rows):
            destination = self._fusion_buf.ptr + row * 2 * hidden_nbytes
            self.runtime.memcpy_async(
                destination,
                self._enorm_buf.ptr + row * hidden_nbytes,
                hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
            self.runtime.memcpy_async(
                destination + hidden_nbytes,
                self._hnorm_buf.ptr + row * hidden_nbytes,
                hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
        launch_gguf_linear(
            self.weights.nextn("eh_proj"),
            self._fusion_buf.ptr,
            self._fused_buf.ptr,
            rows=rows,
            in_features=2 * self.hidden_size,
            out_features=self.hidden_size,
            runtime=self.runtime,
        )
        self._batch_session.step_hidden_batch_native(
            self._fused_buf.ptr,
            sessions=sessions,
            positions=pos,
            output_hidden_ptr=self._final_hidden_buf.ptr,
            logits_ptr=self._logits_buf.ptr,
            score_output=True,
        )
        self._publish_batch_consumed_positions(ids, pos)
        token_ids = self._device_top1_rows(rows)
        hidden = tuple(
            Tensor.from_handle(
                self._final_hidden_buf.ptr + row * hidden_nbytes,
                (1, self.hidden_size),
                DType.BF16,
                Device("hip", 0),
            )
            for row in range(rows)
        )
        self.last_lm_head_path = "physical_batch_device_top1"
        return token_ids, hidden

    def _run_step_device_top1(
        self,
        request_id: int,
        position: int,
        target_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Consume one pre-staged slot token without a candidate host readback."""

        rid = int(request_id)
        slot = self._slot(rid)
        if int(self._batch_sessions[slot].position) != int(position):
            raise ValueError("device NextN position does not match provider cursor")
        layer_out_ptr, final_hidden_ptr = self._run_block(
            rid,
            0,
            int(position),
            target_hidden,
            token_ready=True,
        )
        self._set_batch_session_position(slot, int(position) + 1)
        gguf_rmsnorm_bf16_f32_weight(
            layer_out_ptr,
            self.weights.fallback("output_norm").allocation().tensor.ptr,
            final_hidden_ptr,
            rows=1,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.weights.fallback("lm_head"),
            final_hidden_ptr,
            self._logits_buf.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=self.runtime,
        )
        token_ids = self._device_top1_rows(1)
        self.last_lm_head_path = "physical_singleton_device_top1"
        return token_ids, Tensor.from_handle(
            final_hidden_ptr,
            (1, self.hidden_size),
            DType.BF16,
            Device("hip", 0),
        )

    def run_batch_proposal_device(
        self,
        context: MtpProposalContext,
        *,
        candidate_counts: Sequence[int],
    ) -> Qwen35GGUFNextNBatchDeviceProposal:
        """Run shared proposal depths without reading candidate IDs to the host."""

        ids = tuple(int(value) for value in context.request_ids)
        counts = tuple(int(value) for value in candidate_counts)
        rows = len(ids)
        if rows < 1 or len(counts) != rows or any(count <= 0 for count in counts):
            raise ValueError("device batch proposal requires aligned positive request depths")
        if max(counts) not in MTP_CHAIN_CANDIDATE_BUDGETS:
            raise ValueError("device batch proposal budget is unsupported")
        if any(
            int(token) < 0 or int(token) >= self.vocab_size
            for token in context.root_tokens
        ):
            raise ValueError("device batch proposal root token is outside the vocabulary")
        if context.target_hidden is None or context.target_hidden.shape != (
            rows,
            self.hidden_size,
        ):
            raise ValueError("device batch proposal target hidden rows do not align")
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        current_hidden = [
            Tensor.from_handle(
                context.target_hidden.ptr + row * hidden_nbytes,
                (1, self.hidden_size),
                DType.BF16,
                context.target_hidden.device,
            )
            for row in range(rows)
        ]
        hidden_rows: list[list[Tensor]] = [[] for _ in range(rows)]
        for depth in range(max(counts)):
            active = tuple(index for index, count in enumerate(counts) if depth < count)
            for packed_row, source_row in enumerate(active):
                self.runtime.memcpy(
                    self._batch_input_hidden.ptr + packed_row * hidden_nbytes,
                    current_hidden[source_row].ptr,
                    hidden_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
            if depth == 0:
                root_tokens = np.asarray(
                    [int(context.root_tokens[index]) for index in active],
                    dtype=np.int64,
                )
                destination = (
                    self._token_buf.ptr
                    if len(active) > 1
                    else self._token_buf.ptr
                    + self._slot(ids[active[0]]) * DType.INT64.itemsize
                )
                copy_host_to_device(
                    DeviceBuffer(destination, root_tokens.nbytes),
                    host_array_ptr(root_tokens),
                    root_tokens.nbytes,
                    runtime=self.runtime,
                )
            else:
                for packed_row, source_row in enumerate(active):
                    destination_row = (
                        packed_row if len(active) > 1 else self._slot(ids[source_row])
                    )
                    copy_i32_to_i64(
                        self._batch_candidate_tokens_i32.ptr
                        + (offsets[source_row] + depth - 1) * DType.INT32.itemsize,
                        self._token_buf.ptr
                        + destination_row * DType.INT64.itemsize,
                        1,
                        library=self._batch_session._runtime_state_library,
                        runtime=self.runtime,
                    )
            packed_hidden = Tensor.from_handle(
                self._batch_input_hidden.ptr,
                (len(active), self.hidden_size),
                DType.BF16,
                Device("hip", 0),
            )
            if len(active) > 1:
                step_tokens, step_hidden = self._run_step_batch_device_top1(
                    tuple(ids[index] for index in active),
                    tuple(int(context.root_positions[index]) + depth for index in active),
                    packed_hidden,
                )
            else:
                source_row = active[0]
                token, hidden = self._run_step_device_top1(
                    ids[source_row],
                    int(context.root_positions[source_row]) + depth,
                    Tensor.from_handle(
                        packed_hidden.ptr,
                        (1, self.hidden_size),
                        DType.BF16,
                        Device("hip", 0),
                    ),
                )
                step_tokens, step_hidden = token, (hidden,)
            if depth == 0:
                for source_row in active:
                    self.capture_request_root_state(ids[source_row])
            for packed_row, source_row in enumerate(active):
                self.runtime.memcpy(
                    self._batch_candidate_tokens_i32.ptr
                    + (offsets[source_row] + depth) * DType.INT32.itemsize,
                    step_tokens.ptr + packed_row * DType.INT32.itemsize,
                    DType.INT32.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
                retained = self._preserve_proposal_hidden(
                    ids[source_row],
                    depth,
                    step_hidden[packed_row],
                )
                hidden_rows[source_row].append(retained)
                current_hidden[source_row] = retained
        return Qwen35GGUFNextNBatchDeviceProposal(
            request_ids=ids,
            root_tokens=tuple(int(value) for value in context.root_tokens),
            root_positions=tuple(int(value) for value in context.root_positions),
            candidate_counts=counts,
            token_ids=Tensor.from_handle(
                self._batch_candidate_tokens_i32.ptr,
                (offsets[-1],),
                DType.INT32,
                Device("hip", 0),
            ),
            hidden_rows=tuple(tuple(rows) for rows in hidden_rows),
        )

    def materialize_batch_device_proposal(
        self,
        proposal: Qwen35GGUFNextNBatchDeviceProposal,
    ) -> tuple[int, ...]:
        """Read one bounded request-major candidate vector after target execution."""

        values = np.empty(proposal.token_ids.shape, dtype=np.int32)
        copy_device_to_host(
            host_array_ptr(values),
            DeviceBuffer(proposal.token_ids.ptr, values.nbytes),
            values.nbytes,
            runtime=self.runtime,
        )
        return tuple(int(value) for value in values)

    def _capture_exact_chain_graph(
        self,
        request_id: int,
        slot: int,
        budget: int,
    ) -> _Qwen35GGUFNextNProposalGraph:
        """Capture one short-context greedy proposal chain on a private stream."""

        if self.weights is None or self._proposal_target_hidden is None or self._proposal_results is None:
            raise RuntimeError("GGUF NextN proposal graph storage is unavailable")
        if self._proposal_graph_runtime_library is None:
            self._proposal_graph_runtime_library = build_runtime_state(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
        runtime_library = self._proposal_graph_runtime_library
        runtime = self.runtime
        stream = runtime.stream_create()
        graph = 0
        graph_exec = 0
        completion_event = 0
        capturing = False
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        prior_position = int(slot_scratch.position_host[0])
        prior_context = int(slot_scratch.context_host[0])
        hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        target_hidden_ptr = self._proposal_target_hidden.ptr + slot * hidden_nbytes
        token_ptr = self._token_buf.ptr + slot * DType.INT64.itemsize
        result_base = (
            self._proposal_results.ptr
            + slot * _NEXTN_TOP1_RESULT_CAPACITY * _NEXTN_TOP1_RESULT_NBYTES
        )
        try:
            runtime.device_synchronize()
            runtime.stream_begin_capture(stream)
            capturing = True
            hidden_ptr = target_hidden_ptr
            for depth in range(int(budget)):
                layer_out_ptr, final_hidden_ptr = self._run_block(
                    request_id,
                    0,
                    depth,
                    Tensor.from_handle(
                        hidden_ptr,
                        (1, self.hidden_size),
                        DType.BF16,
                        Device("hip", 0),
                    ),
                    stream=stream,
                    token_ready=True,
                    position_ready=True,
                    attention_context_cap=min(
                        int(self.scratch.max_positions),
                        _NEXTN_EXACT_CHAIN_GRAPH_MAX_CONTEXT,
                    ),
                )
                gguf_rmsnorm_bf16_f32_weight(
                    layer_out_ptr,
                    self.weights.fallback("output_norm").allocation().tensor.ptr,
                    final_hidden_ptr,
                    rows=1,
                    hidden_size=self.hidden_size,
                    eps=self.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
                result_ptr = result_base + depth * _NEXTN_TOP1_RESULT_NBYTES
                if not self._enqueue_exact_lm_head_top1(
                    final_hidden_ptr,
                    result_ptr,
                    result_ptr + DType.INT32.itemsize,
                    stream=stream,
                ):
                    raise RuntimeError("GGUF NextN exact top-1 graph route became unavailable")
                hidden_row_ptr = self._proposal_history_hidden.ptr + (
                    slot * _NEXTN_TOP1_RESULT_CAPACITY + depth
                ) * hidden_nbytes
                runtime.memcpy_async(
                    hidden_row_ptr,
                    final_hidden_ptr,
                    hidden_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                hidden_ptr = final_hidden_ptr
                if depth + 1 < int(budget):
                    copy_i32_to_i64(
                        result_ptr,
                        token_ptr,
                        1,
                        stream=stream,
                        library=runtime_library,
                        runtime=runtime,
                    )
                    advance_decode_position_i64(
                        slot_scratch.position_buf.ptr,
                        slot_scratch.context_buf.ptr,
                        stream=stream,
                        library=runtime_library,
                        runtime=runtime,
                    )
            graph = runtime.stream_end_capture(stream)
            capturing = False
            if not graph:
                raise RuntimeError("HIP returned a null GGUF NextN proposal graph")
            graph_exec = runtime.graph_instantiate(graph)
            if not graph_exec:
                raise RuntimeError("HIP returned a null GGUF NextN proposal graph executable")
            completion_event = runtime.event_create(flags=2)
            if not completion_event:
                raise RuntimeError("HIP returned a null GGUF NextN proposal completion event")
            return _Qwen35GGUFNextNProposalGraph(
                slot=int(slot),
                budget=int(budget),
                stream=int(stream),
                graph=int(graph),
                graph_exec=int(graph_exec),
                completion_event=int(completion_event),
            )
        except Exception:
            if capturing:
                try:
                    abandoned = runtime.stream_end_capture(stream)
                    if abandoned:
                        runtime.graph_destroy(abandoned)
                except Exception:
                    pass
            if completion_event:
                runtime.event_destroy(completion_event)
            if graph_exec:
                runtime.graph_exec_destroy(graph_exec)
            if graph:
                runtime.graph_destroy(graph)
            runtime.stream_destroy(stream)
            raise
        finally:
            slot_scratch.position_host[0] = prior_position
            slot_scratch.context_host[0] = prior_context

    def _proposal_graph(
        self,
        request_id: int,
        slot: int,
        budget: int,
    ) -> _Qwen35GGUFNextNProposalGraph | None:
        key = (int(slot), int(budget))
        graph = self._proposal_graphs.get(key)
        if graph is not None:
            return graph
        if key in self._proposal_graph_unavailable:
            self._proposal_graph_last_status = "capture_fallback"
            return None
        try:
            graph = self._capture_exact_chain_graph(request_id, slot, budget)
        except Exception as exc:
            self._proposal_graph_unavailable.add(key)
            self._proposal_graph_last_status = "capture_fallback"
            self._proposal_graph_last_error = f"{type(exc).__name__}: {exc}"
            return None
        self._proposal_graphs[key] = graph
        self._proposal_graph_captures += 1
        self._proposal_graph_last_status = "captured"
        self._proposal_graph_last_error = None
        return graph

    def _launch_graph_chain_device(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
        cached_only: bool,
    ) -> Qwen35GGUFNextNDeviceProposal | None:
        """Launch one graph chain and leave its top-1 rows device-resident."""

        budget = int(candidate_budget)
        if (
            budget not in _NEXTN_EXACT_CHAIN_GRAPH_BUDGETS
            or self._proposal_target_hidden is None
            or self._proposal_results is None
            or self._proposal_results_host is None
        ):
            self._proposal_graph_last_status = "eager_ineligible"
            return None
        graph_context_cap = min(
            int(self.scratch.max_positions),
            _NEXTN_EXACT_CHAIN_GRAPH_MAX_CONTEXT,
        )
        if int(position) + budget > graph_context_cap:
            self._proposal_graph_last_status = "eager_long_context"
            return None
        slot = self._slot(request_id)
        graph = self._proposal_graphs.get((int(slot), budget))
        if graph is None and not cached_only:
            graph = self._proposal_graph(request_id, slot, budget)
        if graph is None:
            self._proposal_graph_last_status = "device_handoff_cache_miss"
            return None
        hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        target_hidden_ptr = self._proposal_target_hidden.ptr + slot * hidden_nbytes
        token_ptr = self._token_buf.ptr + slot * DType.INT64.itemsize
        result_ptr = (
            self._proposal_results.ptr
            + slot * _NEXTN_TOP1_RESULT_CAPACITY * _NEXTN_TOP1_RESULT_NBYTES
        )
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        self._token_host[slot] = int(token_id)
        slot_scratch.position_host[0] = int(position)
        slot_scratch.context_host[0] = int(position) + 1
        token_host = self._token_host[slot : slot + 1]
        stream = graph.stream
        runtime = self.runtime
        if int(target_hidden.ptr) != int(target_hidden_ptr):
            runtime.memcpy_async(
                target_hidden_ptr,
                target_hidden.ptr,
                hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
        runtime.memcpy_async(
            token_ptr,
            host_array_ptr(token_host),
            token_host.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
            stream,
        )
        runtime.memcpy_async(
            slot_scratch.position_buf.ptr,
            host_array_ptr(slot_scratch.position_host),
            slot_scratch.position_host.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
            stream,
        )
        runtime.memcpy_async(
            slot_scratch.context_buf.ptr,
            host_array_ptr(slot_scratch.context_host),
            slot_scratch.context_host.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
            stream,
        )
        runtime.graph_launch(graph.graph_exec, stream)
        runtime.event_record(graph.completion_event, stream)
        final_hidden_ptr = self._final_hidden_buf.ptr + slot * hidden_nbytes
        hidden_rows_ptr = self._proposal_history_hidden.ptr + (
            slot * _NEXTN_TOP1_RESULT_CAPACITY
        ) * hidden_nbytes
        slot_scratch.position_host[0] = int(position) + budget - 1
        slot_scratch.context_host[0] = int(position) + budget
        self._set_batch_session_position(slot, int(position) + budget)
        self.last_lm_head_path = "exact_q6_top1"
        self._proposal_graph_replays += 1
        self._proposal_graph_last_status = "device_handoff"
        self._proposal_graph_last_error = None
        return Qwen35GGUFNextNDeviceProposal(
            request_id=int(request_id),
            root_token=int(token_id),
            root_position=int(position),
            budget=budget,
            result_ptr=int(result_ptr),
            result_nbytes=budget * _NEXTN_TOP1_RESULT_NBYTES,
            completion_event=int(graph.completion_event),
            stream=int(stream),
            final_hidden=Tensor.from_handle(
                final_hidden_ptr,
                (1, self.hidden_size),
                DType.BF16,
                Device("hip", 0),
            ),
            hidden_rows=Tensor.from_handle(
                hidden_rows_ptr,
                (budget, self.hidden_size),
                DType.BF16,
                Device("hip", 0),
            ),
        )

    def prepare_proposal_graph(
        self,
        request_id: int,
        *,
        candidate_budget: int,
    ) -> bool:
        """Capture one exact budget graph without executing proposal state."""

        budget = int(candidate_budget)
        if budget not in _NEXTN_EXACT_CHAIN_GRAPH_BUDGETS:
            return False
        slot = self._slot(int(request_id))
        return self._proposal_graph(int(request_id), slot, budget) is not None

    def launch_cached_graph_chain_device(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
    ) -> Qwen35GGUFNextNDeviceProposal | None:
        """Launch a resident graph without capture or host synchronization."""

        return self._launch_graph_chain_device(
            request_id,
            token_id,
            position,
            target_hidden,
            candidate_budget=int(candidate_budget),
            cached_only=True,
        )

    def materialize_device_proposal(
        self,
        proposal: Qwen35GGUFNextNDeviceProposal,
        *,
        token_ids: tuple[int, ...],
        top1_values: tuple[float, ...],
    ) -> tuple[Qwen35GGUFNextNStepResult, ...]:
        """Build the ordinary draft ABI after the target stream retires."""

        tokens = tuple(int(token) for token in token_ids)
        values = tuple(float(value) for value in top1_values)
        if len(tokens) != proposal.budget or len(values) != proposal.budget:
            raise RuntimeError("device proposal payload omitted a candidate row")
        if any(token < 0 or token >= self.vocab_size for token in tokens):
            raise RuntimeError("GGUF NextN proposal graph produced an invalid token id")
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float32))):
            raise FloatingPointError("GGUF NextN proposal graph produced NaN or Inf")
        rows = []
        current_token = int(proposal.root_token)
        for depth, (next_token, logit) in enumerate(zip(tokens, values, strict=True)):
            rows.append(
                Qwen35GGUFNextNStepResult(
                    request_id=int(proposal.request_id),
                    input_token=current_token,
                    position=int(proposal.root_position) + depth,
                    token_id=next_token,
                    logit=logit,
                    hidden=(
                        proposal.final_hidden
                        if proposal.hidden_rows is None
                        else Tensor.from_handle(
                            proposal.hidden_rows.ptr
                            + depth
                            * proposal.final_hidden.shape[1]
                            * DType.BF16.itemsize,
                            (1, proposal.final_hidden.shape[1]),
                            DType.BF16,
                            proposal.hidden_rows.device,
                        )
                    ),
                    logits=None,
                )
            )
            current_token = next_token
        self._proposal_graph_last_status = "replay"
        return tuple(rows)

    def _run_exact_graph_chain(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
    ) -> tuple[Qwen35GGUFNextNStepResult, ...] | None:
        """Replay one exact device-chained proposal graph, or request eager fallback."""

        proposal = self._launch_graph_chain_device(
            request_id,
            token_id,
            position,
            target_hidden,
            candidate_budget=int(candidate_budget),
            cached_only=False,
        )
        if proposal is None:
            return None
        slot = self._slot(request_id)
        assert self._proposal_results_host is not None
        result_host = self._proposal_results_host[slot, : proposal.budget]
        self.runtime.memcpy_async(
            host_array_ptr(result_host),
            proposal.result_ptr,
            proposal.result_nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
            proposal.stream,
        )
        self.runtime.stream_synchronize(proposal.stream)
        return self.materialize_device_proposal(
            proposal,
            token_ids=tuple(int(value) for value in result_host["token"]),
            top1_values=tuple(float(value) for value in result_host["value"]),
        )

    def run_chain(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
        return_logits: bool = False,
        allow_graph: bool = True,
    ) -> tuple[Qwen35GGUFNextNStepResult, ...]:
        """Run one candidate chain through the exact graph route or eager fallback."""

        budget = int(candidate_budget)
        if budget not in MTP_CHAIN_CANDIDATE_BUDGETS:
            allowed = ", ".join(str(value) for value in MTP_CHAIN_CANDIDATE_BUDGETS)
            raise ValueError(f"candidate_budget must be one of {allowed}")
        if self.closed or self.weights is None:
            raise RuntimeError("GGUF NextN executor is closed")
        if target_hidden.dtype != DType.BF16 or target_hidden.shape != (1, self.hidden_size):
            raise ValueError(f"target_hidden must be BF16 with shape (1, {self.hidden_size})")
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError("token_id is outside the GGUF vocabulary")
        if position < 0 or int(position) + budget > int(self.scratch.max_positions):
            raise ValueError("GGUF NextN proposal positions exceed cache capacity")
        if not return_logits and bool(allow_graph):
            graph_rows = self._run_exact_graph_chain(
                request_id,
                token_id,
                position,
                target_hidden,
                candidate_budget=budget,
            )
            if graph_rows is not None:
                return graph_rows
        rows = []
        current_token = int(token_id)
        current_hidden = target_hidden
        for depth in range(budget):
            result = self.run_step(
                int(request_id),
                current_token,
                int(position) + depth,
                current_hidden,
                return_logits=return_logits,
            )
            preserved_hidden = self._preserve_proposal_hidden(
                request_id,
                depth,
                result.hidden,
            )
            result = Qwen35GGUFNextNStepResult(
                request_id=result.request_id,
                input_token=result.input_token,
                position=result.position,
                token_id=result.token_id,
                logit=result.logit,
                hidden=preserved_hidden,
                logits=result.logits,
            )
            rows.append(result)
            current_token = int(result.token_id)
            current_hidden = result.hidden
        return tuple(rows)

    def _preserve_proposal_hidden(
        self,
        request_id: int,
        depth: int,
        hidden: Tensor,
    ) -> Tensor:
        if not hasattr(self, "_proposal_history_hidden"):
            return hidden
        slot = self._slot(request_id)
        index = slot * _NEXTN_TOP1_RESULT_CAPACITY + int(depth)
        if depth < 0 or depth >= _NEXTN_TOP1_RESULT_CAPACITY:
            raise ValueError("proposal hidden depth exceeds retained capacity")
        nbytes = self.hidden_size * DType.BF16.itemsize
        destination = self._proposal_history_hidden.ptr + index * nbytes
        self.runtime.memcpy(
            destination,
            hidden.ptr,
            nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
        )
        return Tensor.from_handle(
            destination,
            (1, self.hidden_size),
            DType.BF16,
            hidden.device,
        )

    def proposal_graph_contract(self) -> dict[str, object]:
        """Return compact exact-chain graph ownership and replay telemetry."""

        return {
            "eligible": self._proposal_target_hidden is not None,
            "budgets": list(_NEXTN_EXACT_CHAIN_GRAPH_BUDGETS),
            "max_context": _NEXTN_EXACT_CHAIN_GRAPH_MAX_CONTEXT,
            "graphs": len(self._proposal_graphs),
            "captures": int(self._proposal_graph_captures),
            "replays": int(self._proposal_graph_replays),
            "unavailable": [list(key) for key in sorted(self._proposal_graph_unavailable)],
            "last_status": self._proposal_graph_last_status,
            "last_error": self._proposal_graph_last_error,
        }

    def enqueue_prompt_rows(
        self,
        request_id: int,
        token_ids: tuple[int, ...],
        *,
        position_start: int,
        target_hidden_base_ptr: int,
        hidden_stride_bytes: int,
        stream: int = 0,
    ) -> None:
        """Append exact prompt rows without scoring discarded draft logits.

        Tokens and position metadata are enqueued on the target prefill stream.
        The caller keeps the target-hidden source live until these consumers
        retire; :class:`_StreamingNextNPromptSink` does so by inserting every
        append before target prefill reuses its chunk buffers.
        """

        request_id = int(request_id)
        tokens = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        start = int(position_start)
        base_ptr = int(target_hidden_base_ptr)
        stride = int(hidden_stride_bytes)
        stream = int(stream)
        if tokens.size <= 0:
            raise ValueError("NextN prompt priming requires at least one token")
        if start < 0 or start + int(tokens.size) > int(self.scratch.max_positions):
            raise ValueError("NextN prompt priming positions exceed cache capacity")
        if base_ptr <= 0:
            raise ValueError("NextN prompt priming requires a device hidden pointer")
        hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        if stride < hidden_nbytes:
            raise ValueError("NextN prompt priming hidden stride is smaller than one row")
        if self._proposal_graph_runtime_library is None:
            self._proposal_graph_runtime_library = build_runtime_state(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
        runtime_library = self._proposal_graph_runtime_library
        slot = self._slot(request_id)
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        token_ptr = self._token_buf.ptr + slot * DType.INT64.itemsize
        self._prompt_priming_staging.setdefault(request_id, []).append(tokens)
        token_host_ptr = host_array_ptr(tokens)

        for row, token in enumerate(tokens.tolist()):
            position = start + row
            self.runtime.memcpy_async(
                token_ptr,
                token_host_ptr + row * DType.INT64.itemsize,
                DType.INT64.itemsize,
                HipMemcpyKind.HOST_TO_DEVICE,
                stream,
            )
            slot_scratch.position_host[0] = position
            slot_scratch.context_host[0] = position + 1
            set_decode_position_i64(
                slot_scratch.position_buf.ptr,
                slot_scratch.context_buf.ptr,
                position,
                stream=stream,
                library=runtime_library,
                runtime=self.runtime,
            )
            self._run_block(
                request_id,
                int(token),
                position,
                Tensor.from_handle(
                    base_ptr + row * stride,
                    (1, self.hidden_size),
                    DType.BF16,
                    Device("hip", 0),
                ),
                stream=stream,
                token_ready=True,
                position_ready=True,
            )
        # _run_block is the state-mutating primitive; unlike run_step/batch it
        # intentionally omits scoring and cursor publication. Publish the final
        # request cursor once after the full enqueued span so physical proposal
        # validates against the same exact prompt timeline as C1.
        self._set_batch_session_position(slot, start + int(tokens.size))

    def finish_prompt_priming(
        self,
        request_id: int,
        *,
        stream: int = 0,
        synchronize: bool = False,
    ) -> None:
        """Release host staging after the target-owned completion boundary."""

        request_id = int(request_id)
        if synchronize:
            self.runtime.stream_synchronize(int(stream))
        self._prompt_priming_staging.pop(request_id, None)

    def advance_state_only(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
    ) -> Qwen35GGUFNextNStateAdvance:
        """Consume an accepted tail without computing its discarded prediction."""

        self._run_block(request_id, token_id, position, target_hidden)
        slots = getattr(self, "_request_slots", None)
        if slots is not None:
            slot = self._slot(request_id)
            self._set_batch_session_position(slot, int(position) + 1)
        self.runtime.device_synchronize()
        return Qwen35GGUFNextNStateAdvance(
            request_id=int(request_id),
            input_token=int(token_id),
            position=int(position),
        )

    def advance_state_only_device(
        self,
        request_id: int,
        token_id: Tensor,
        position: int,
        target_hidden: Tensor,
    ) -> None:
        """Consume one device-resident accepted ID without reading it to host."""

        if (
            token_id.dtype != DType.INT32
            or token_id.shape != (1,)
            or token_id.device.kind != "hip"
        ):
            raise ValueError("device NextN token must be one HIP INT32 scalar")
        slot = self._slot(request_id)
        copy_i32_to_i64(
            token_id.ptr,
            self._token_buf.ptr + slot * DType.INT64.itemsize,
            1,
            library=self._batch_session._runtime_state_library,
            runtime=self.runtime,
        )
        self._run_block(
            int(request_id),
            0,
            int(position),
            target_hidden,
            token_ready=True,
        )
        self._set_batch_session_position(slot, int(position) + 1)
        self.runtime.device_synchronize()

    def capture_request_root_state(self, request_id: int) -> None:
        """Persist the exact provider state/cursor immediately after its root."""

        rid = int(request_id)
        slot = self._request_slots.get(rid)
        if slot is None:
            raise ValueError("GGUF NextN root snapshot requires an active request")
        owner_states = (
            *self.scratch.layer_conv_states,
            *self.scratch.layer_recurrent_states,
        )
        for state, snapshot in zip(
            owner_states,
            self._provider_root_state_snapshots,
            strict=True,
        ):
            if state is None or snapshot is None:
                continue
            row_nbytes, remainder = divmod(int(state.nbytes), self.max_requests)
            if remainder:
                raise ValueError("GGUF NextN root snapshot state is not slot-major")
            self.runtime.memcpy(
                int(snapshot.ptr) + int(slot) * row_nbytes,
                int(state.ptr) + int(slot) * row_nbytes,
                row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        consumed_position = int(slot_scratch.position_host[0])
        context_length = int(slot_scratch.context_host[0])
        if context_length != consumed_position + 1:
            raise RuntimeError(
                "GGUF NextN after-root snapshot cursor is inconsistent"
            )
        self._provider_root_state_metadata[rid] = (
            int(slot),
            consumed_position,
            context_length,
        )

    def restore_request_root_state(self, request_id: int) -> None:
        """Commit the captured after-root provider state without model replay."""

        rid = int(request_id)
        metadata = self._provider_root_state_metadata.get(rid)
        slot = self._request_slots.get(rid)
        if metadata is None or slot is None or int(metadata[0]) != int(slot):
            raise RuntimeError("GGUF NextN after-root snapshot is unavailable")
        owner_states = (
            *self.scratch.layer_conv_states,
            *self.scratch.layer_recurrent_states,
        )
        for state, snapshot in zip(
            owner_states,
            self._provider_root_state_snapshots,
            strict=True,
        ):
            if state is None or snapshot is None:
                continue
            row_nbytes, remainder = divmod(int(state.nbytes), self.max_requests)
            if remainder:
                raise ValueError("GGUF NextN root snapshot state is not slot-major")
            self.runtime.memcpy(
                int(state.ptr) + int(slot) * row_nbytes,
                int(snapshot.ptr) + int(slot) * row_nbytes,
                row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
        _slot, consumed_position, context_length = metadata
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        slot_scratch.position_host[0] = int(consumed_position)
        slot_scratch.context_host[0] = int(context_length)
        self._set_batch_session_position(slot, int(context_length))
        copy_host_to_device(
            slot_scratch.position_buf,
            host_array_ptr(slot_scratch.position_host),
            slot_scratch.position_host.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            slot_scratch.context_buf,
            host_array_ptr(slot_scratch.context_host),
            slot_scratch.context_host.nbytes,
            runtime=self.runtime,
        )

    def capture_request_checkpoint(
        self,
        request_id: int,
    ) -> Qwen35GGUFNextNRequestCheckpoint:
        """Snapshot mutable Conv/GDN state and logical cursor before proposal."""

        rid = int(request_id)
        slot = self._request_slots.get(rid)
        if slot is None:
            raise ValueError("GGUF NextN checkpoint requires an active request")
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        live_states = tuple(
            state
            for pair in zip(
                slot_scratch.layer_conv_states,
                slot_scratch.layer_recurrent_states,
                strict=True,
            )
            for state in pair
            if state is not None
        )
        allocated: list[DeviceBuffer] = []
        try:
            pairs: list[tuple[DeviceBuffer, DeviceBuffer]] = []
            for state in live_states:
                backup = malloc(int(state.nbytes), runtime=self.runtime)
                allocated.append(backup)
                self.runtime.memcpy(
                    backup.ptr,
                    state.ptr,
                    state.nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
                pairs.append((state, backup))
        except BaseException:
            for backup in reversed(allocated):
                free(backup, runtime=self.runtime)
            raise
        sessions = getattr(self, "_batch_sessions", None)
        logical_position = int(
            slot_scratch.position_host[0]
            if sessions is None
            else sessions[slot].position
        )
        return Qwen35GGUFNextNRequestCheckpoint(
            request_id=rid,
            slot=int(slot),
            state_pairs=tuple(pairs),
            position=logical_position,
            context_length=logical_position + 1,
        )

    def restore_request_checkpoint(
        self,
        checkpoint: Qwen35GGUFNextNRequestCheckpoint,
    ) -> None:
        """Restore provider state/cursor and leave rejected KV suffix invisible."""

        if checkpoint.released:
            raise RuntimeError("GGUF NextN checkpoint is released")
        slot = self._request_slots.get(int(checkpoint.request_id))
        if slot != int(checkpoint.slot):
            raise RuntimeError("GGUF NextN checkpoint slot ownership changed")
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        for state, backup in checkpoint.state_pairs:
            self.runtime.memcpy(
                state.ptr,
                backup.ptr,
                state.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
        slot_scratch.position_host[0] = int(checkpoint.position)
        slot_scratch.context_host[0] = int(checkpoint.context_length)
        self._set_batch_session_position(slot, int(checkpoint.position))
        copy_host_to_device(
            slot_scratch.position_buf,
            host_array_ptr(slot_scratch.position_host),
            slot_scratch.position_host.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            slot_scratch.context_buf,
            host_array_ptr(slot_scratch.context_host),
            slot_scratch.context_host.nbytes,
            runtime=self.runtime,
        )

    def release_request_checkpoint(
        self,
        checkpoint: Qwen35GGUFNextNRequestCheckpoint,
    ) -> None:
        if checkpoint.released:
            return
        for _state, backup in reversed(checkpoint.state_pairs):
            free(backup, runtime=self.runtime)
        checkpoint.released = True

    def clone_request_state(
        self,
        source_request_id: int,
        destination_request_id: int,
    ) -> None:
        """Clone exact provider state into a distinct request-owned slot."""

        source_id = int(source_request_id)
        destination_id = int(destination_request_id)
        if source_id == destination_id:
            raise ValueError("NextN clone source and destination must differ")
        source_slot = self._request_slots.get(source_id)
        if source_slot is None:
            raise ValueError("NextN clone source request is not active")
        destination_slot = self._slot(destination_id)
        if destination_slot == source_slot:
            raise RuntimeError("NextN clone requires distinct request slots")
        source = self.scratch.for_slot(source_slot, span_role="decode")
        destination = self.scratch.for_slot(destination_slot, span_role="decode")
        for source_pair, destination_pair in zip(
            zip(source.layer_conv_states, source.layer_recurrent_states, strict=True),
            zip(
                destination.layer_conv_states,
                destination.layer_recurrent_states,
                strict=True,
            ),
            strict=True,
        ):
            for origin, target in zip(source_pair, destination_pair, strict=True):
                if (origin is None) != (target is None):
                    raise ValueError("NextN clone state layout mismatch")
                if origin is None:
                    continue
                if int(origin.nbytes) != int(target.nbytes):
                    raise ValueError("NextN clone state size mismatch")
                self.runtime.memcpy(
                    target.ptr,
                    origin.ptr,
                    origin.nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
        position = int(source.position_host[0])
        context = int(source.context_host[0])
        if context != position + 1:
            raise RuntimeError("NextN clone source cursor is inconsistent")
        physical_slots = int(getattr(self.scratch, "slot_count", 1))
        max_positions = int(source.max_positions)
        visible_rows = min(context, max_positions)
        for source_pair, destination_pair in zip(
            zip(source.full_key_caches, source.full_value_caches, strict=True),
            zip(
                destination.full_key_caches,
                destination.full_value_caches,
                strict=True,
            ),
            strict=True,
        ):
            for origin, target in zip(source_pair, destination_pair, strict=True):
                if (origin is None) != (target is None):
                    raise ValueError("NextN clone KV layout mismatch")
                if origin is None:
                    continue
                slot_nbytes, remainder = divmod(int(origin.nbytes), physical_slots)
                if remainder or int(target.nbytes) != int(origin.nbytes):
                    raise ValueError("NextN clone KV slot layout mismatch")
                row_nbytes, remainder = divmod(slot_nbytes, max_positions)
                if remainder:
                    raise ValueError("NextN clone KV position layout mismatch")
                self.runtime.memcpy(
                    int(target.ptr) + destination_slot * slot_nbytes,
                    int(origin.ptr) + source_slot * slot_nbytes,
                    visible_rows * row_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
        destination.position_host[0] = position
        destination.context_host[0] = context
        self._set_batch_session_position(destination_slot, position)
        copy_host_to_device(
            destination.position_buf,
            host_array_ptr(destination.position_host),
            destination.position_host.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            destination.context_buf,
            host_array_ptr(destination.context_host),
            destination.context_host.nbytes,
            runtime=self.runtime,
        )
        metadata = self._provider_root_state_metadata.get(source_id)
        if metadata is not None:
            self._provider_root_state_metadata[destination_id] = (
                destination_slot,
                int(metadata[1]),
                int(metadata[2]),
            )
        self.runtime.device_synchronize()

    def request_state_fingerprint(self, request_id: int) -> dict[str, object]:
        """Return exact visible provider state/KV/cursor hashes for gates."""

        rid = int(request_id)
        slot = self._request_slots.get(rid)
        if slot is None:
            raise ValueError("GGUF NextN fingerprint requires an active request")
        slot_scratch = self.scratch.for_slot(slot, span_role="decode")
        self.runtime.device_synchronize()

        def read_digest(buffer, *, nbytes: int | None = None) -> tuple[str, int]:
            count = int(buffer.nbytes if nbytes is None else nbytes)
            host = np.empty((count,), dtype=np.uint8)
            copy_device_to_host(
                host_array_ptr(host),
                DeviceBuffer(int(buffer.ptr), count),
                count,
                runtime=self.runtime,
            )
            return hashlib.sha256(host.tobytes()).hexdigest(), count

        state_hash = hashlib.sha256()
        state_bytes = 0
        for pair in zip(
            slot_scratch.layer_conv_states,
            slot_scratch.layer_recurrent_states,
            strict=True,
        ):
            for state in pair:
                if state is None:
                    continue
                digest, count = read_digest(state)
                state_hash.update(bytes.fromhex(digest))
                state_bytes += count
        context = int(slot_scratch.context_host[0])
        max_positions = int(slot_scratch.max_positions)
        visible_rows = min(max(0, context), max_positions)
        kv_hash = hashlib.sha256()
        kv_bytes = 0
        physical_slots = int(getattr(self.scratch, "slot_count", 1))
        if physical_slots <= 0:
            raise RuntimeError("GGUF NextN KV cache has no physical slots")
        for pair in zip(
            slot_scratch.full_key_caches,
            slot_scratch.full_value_caches,
            strict=True,
        ):
            for cache in pair:
                if cache is None:
                    continue
                slot_nbytes, remainder = divmod(int(cache.nbytes), physical_slots)
                if remainder or slot_nbytes % max_positions:
                    raise RuntimeError("GGUF NextN KV cache is not slot/position divisible")
                count = visible_rows * (slot_nbytes // max_positions)
                if count <= 0:
                    continue
                slot_cache = DeviceBuffer(
                    int(cache.ptr) + int(slot) * slot_nbytes,
                    slot_nbytes,
                )
                digest, count = read_digest(slot_cache, nbytes=count)
                kv_hash.update(bytes.fromhex(digest))
                kv_bytes += count
        return {
            "request_id": rid,
            "slot": int(slot),
            "position": int(slot_scratch.position_host[0]),
            "context_length": context,
            "state_sha256": state_hash.hexdigest(),
            "state_bytes": state_bytes,
            "visible_kv_sha256": kv_hash.hexdigest(),
            "visible_kv_bytes": kv_bytes,
        }

    def reset_request(self, request_id: int) -> None:
        self._provider_root_state_metadata.pop(int(request_id), None)
        slot = self._request_slots.get(int(request_id))
        if slot is None:
            return
        self.scratch.for_slot(slot).zero_states(self.runtime)
        self._set_batch_session_position(slot, 0)

    def release_request(self, request_id: int) -> None:
        request_id = int(request_id)
        self.reset_request(request_id)
        self._prompt_priming_staging.pop(request_id, None)
        self._request_slots.pop(request_id, None)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for proposal_graph in reversed(tuple(self._proposal_graphs.values())):
            self.runtime.event_destroy(proposal_graph.completion_event)
            self.runtime.graph_exec_destroy(proposal_graph.graph_exec)
            self.runtime.graph_destroy(proposal_graph.graph)
            self.runtime.stream_destroy(proposal_graph.stream)
        self._proposal_graphs.clear()
        self._prompt_priming_staging.clear()
        for buffer in reversed(self._buffers):
            free(buffer, runtime=self.runtime)
        self._batch_session.close()
        self.runner.close()
        if self.weights is not None:
            self.weights.free(runtime=self.runtime)
            self.weights = None

    def __enter__(self) -> "Qwen35GGUFNextNExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class Qwen35GGUFNextNDraftProvider:
    """Target-attached NextN DraftModel provider emitting candidate-only rows."""

    def __init__(
        self,
        executor: Qwen35GGUFNextNStepExecutor,
        *,
        pad_token_id: int = 0,
        owns_executor: bool = False,
    ) -> None:
        self.executor = executor
        self.pad_token_id = int(pad_token_id)
        self.owns_executor = bool(owns_executor)
        self.last_results: dict[int, tuple[Qwen35GGUFNextNStepResult, ...]] = {}
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")

    @classmethod
    def from_model(
        cls,
        model: str | Path,
        *,
        pad_token_id: int = 0,
        **executor_kwargs,
    ) -> "Qwen35GGUFNextNDraftProvider":
        return cls(
            Qwen35GGUFNextNExecutor(model, **executor_kwargs),
            pad_token_id=pad_token_id,
            owns_executor=True,
        )

    def propose(
        self,
        context: MtpProposalContext,
        *,
        candidate_budget: int,
        return_logits: bool = False,
        allow_graph: bool = True,
    ) -> DraftBatch:
        budget = int(candidate_budget)
        if budget not in MTP_CHAIN_CANDIDATE_BUDGETS:
            allowed = ", ".join(str(value) for value in MTP_CHAIN_CANDIDATE_BUDGETS)
            raise ValueError(f"candidate_budget must be one of {allowed}")
        if context.target_hidden is None:
            raise ValueError("GGUF NextN proposal requires target_hidden")
        if context.target_hidden.dtype != DType.BF16:
            raise ValueError("GGUF NextN target_hidden must use BF16")
        if context.target_hidden.shape != (len(context.request_ids), self.executor.hidden_size):
            raise ValueError("GGUF NextN target_hidden shape must align with requests")

        row_nbytes = self.executor.hidden_size * DType.BF16.itemsize
        requests: list[MtpDraftRequest] = []
        for index, request_id in enumerate(context.request_ids):
            hidden = Tensor.from_handle(
                context.target_hidden.ptr + index * row_nbytes,
                (1, self.executor.hidden_size),
                DType.BF16,
                context.target_hidden.device,
            )
            current_token = int(context.root_tokens[index])
            position = int(context.root_positions[index])
            run_chain = getattr(self.executor, "run_chain", None)
            if callable(run_chain):
                chain_kwargs = {
                    "candidate_budget": budget,
                    "return_logits": return_logits,
                }
                if not allow_graph:
                    chain_kwargs["allow_graph"] = False
                results = list(
                    run_chain(
                        int(request_id),
                        current_token,
                        position,
                        hidden,
                        **chain_kwargs,
                    )
                )
                if len(results) != budget:
                    raise RuntimeError("GGUF NextN executor chain returned the wrong row count")
            else:
                results = []
                for depth in range(budget):
                    result = self.executor.run_step(
                        int(request_id),
                        current_token,
                        position + depth,
                        hidden,
                        return_logits=return_logits,
                    )
                    results.append(result)
                    current_token = int(result.token_id)
                    hidden = result.hidden
            self.last_results[int(request_id)] = tuple(results)
            requests.append(
                MtpDraftRequest(
                    request_id=int(request_id),
                    root_position=position,
                    candidate_tokens=tuple(result.token_id for result in results),
                    active_count=len(results),
                )
            )
        return compile_mtp_chain(
            requests,
            candidate_budget=budget,
            pad_token_id=self.pad_token_id,
        )

    def propose_batch_device(
        self,
        context: MtpProposalContext,
        *,
        candidate_counts: Sequence[int],
    ) -> Qwen35GGUFNextNBatchDeviceProposal:
        """Launch a physical C>1 proposal without materializing candidate IDs."""

        launch = getattr(self.executor, "run_batch_proposal_device", None)
        if not callable(launch):
            raise NotImplementedError(
                "GGUF NextN executor has no device-resident batch proposal"
            )
        proposal = launch(
            context,
            candidate_counts=tuple(int(value) for value in candidate_counts),
        )
        if not isinstance(proposal, Qwen35GGUFNextNBatchDeviceProposal):
            raise TypeError("device batch proposal returned an invalid descriptor")
        if proposal.request_ids != tuple(int(value) for value in context.request_ids):
            raise ValueError("device batch proposal request IDs changed")
        return proposal

    def materialize_batch_device_proposal(
        self,
        proposal: Qwen35GGUFNextNBatchDeviceProposal,
    ) -> DraftBatch:
        """Read bounded candidate IDs after target lowering and bind repair rows."""

        materialize = getattr(
            self.executor,
            "materialize_batch_device_proposal",
            None,
        )
        if not callable(materialize):
            raise RuntimeError("GGUF NextN executor cannot materialize a batch proposal")
        token_ids = tuple(int(value) for value in materialize(proposal))
        if len(token_ids) != sum(proposal.candidate_counts):
            raise RuntimeError("device batch proposal omitted candidate IDs")
        requests: list[MtpDraftRequest] = []
        cursor = 0
        for request_id, root_token, root_position, count, hidden_rows in zip(
            proposal.request_ids,
            proposal.root_tokens,
            proposal.root_positions,
            proposal.candidate_counts,
            proposal.hidden_rows,
            strict=True,
        ):
            request_tokens = token_ids[cursor : cursor + count]
            cursor += count
            current = int(root_token)
            results: list[Qwen35GGUFNextNStepResult] = []
            for depth, (token, hidden) in enumerate(
                zip(request_tokens, hidden_rows, strict=True)
            ):
                results.append(
                    Qwen35GGUFNextNStepResult(
                        request_id=int(request_id),
                        input_token=current,
                        position=int(root_position) + depth,
                        token_id=int(token),
                        logit=0.0,
                        hidden=hidden,
                        logits=None,
                    )
                )
                current = int(token)
            self.last_results[int(request_id)] = tuple(results)
            requests.append(
                MtpDraftRequest(
                    request_id=int(request_id),
                    root_position=int(root_position),
                    candidate_tokens=request_tokens,
                    active_count=int(count),
                )
            )
        return compile_mtp_chain(
            requests,
            candidate_budget=max(proposal.candidate_counts),
            pad_token_id=self.pad_token_id,
        )

    def propose_batch(
        self,
        context: MtpProposalContext,
        *,
        candidate_counts: Sequence[int],
    ) -> DraftBatch:
        """Physically batch every shared depth and fall back only for ragged tails."""

        counts = tuple(int(value) for value in candidate_counts)
        rows = len(context.request_ids)
        if rows <= 1 or len(counts) != rows or any(count <= 0 for count in counts):
            raise ValueError("physical NextN proposal requires C>1 positive depths")
        budget = max(counts)
        if budget not in MTP_CHAIN_CANDIDATE_BUDGETS:
            raise ValueError("physical NextN proposal budget is unsupported")
        if context.target_hidden is None or context.target_hidden.shape != (
            rows,
            self.executor.hidden_size,
        ):
            raise ValueError("physical NextN target_hidden must align with requests")
        current_tokens = [int(value) for value in context.root_tokens]
        current_hidden = [
            Tensor.from_handle(
                context.target_hidden.ptr
                + row * self.executor.hidden_size * DType.BF16.itemsize,
                (1, self.executor.hidden_size),
                DType.BF16,
                context.target_hidden.device,
            )
            for row in range(rows)
        ]
        per_request: list[list[Qwen35GGUFNextNStepResult]] = [
            [] for _ in range(rows)
        ]
        hidden_nbytes = self.executor.hidden_size * DType.BF16.itemsize
        for depth in range(budget):
            active = tuple(index for index, count in enumerate(counts) if depth < count)
            if len(active) > 1:
                for packed_row, source_row in enumerate(active):
                    self.executor.runtime.memcpy(
                        self.executor._batch_input_hidden.ptr
                        + packed_row * hidden_nbytes,
                        current_hidden[source_row].ptr,
                        hidden_nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                    )
                step_rows = self.executor.run_step_batch(
                    tuple(context.request_ids[index] for index in active),
                    tuple(current_tokens[index] for index in active),
                    tuple(
                        int(context.root_positions[index]) + depth
                        for index in active
                    ),
                    Tensor.from_handle(
                        self.executor._batch_input_hidden.ptr,
                        (len(active), self.executor.hidden_size),
                        DType.BF16,
                        Device("hip", 0),
                    ),
                )
            else:
                index = active[0]
                step_rows = (
                    self.executor.run_step(
                        int(context.request_ids[index]),
                        current_tokens[index],
                        int(context.root_positions[index]) + depth,
                        current_hidden[index],
                        return_logits=False,
                    ),
                )
            if len(step_rows) != len(active):
                raise RuntimeError("physical NextN step returned the wrong row count")
            for source_index, result in zip(active, step_rows, strict=True):
                hidden = self.executor._preserve_proposal_hidden(
                    int(context.request_ids[source_index]),
                    depth,
                    result.hidden,
                )
                retained = Qwen35GGUFNextNStepResult(
                    request_id=result.request_id,
                    input_token=result.input_token,
                    position=result.position,
                    token_id=result.token_id,
                    logit=result.logit,
                    hidden=hidden,
                    logits=result.logits,
                )
                per_request[source_index].append(retained)
                current_tokens[source_index] = int(retained.token_id)
                current_hidden[source_index] = retained.hidden
        requests = []
        for request_id, root_position, results in zip(
            context.request_ids,
            context.root_positions,
            per_request,
            strict=True,
        ):
            self.last_results[int(request_id)] = tuple(results)
            requests.append(
                MtpDraftRequest(
                    request_id=int(request_id),
                    root_position=int(root_position),
                    candidate_tokens=tuple(result.token_id for result in results),
                    active_count=len(results),
                )
            )
        return compile_mtp_chain(
            requests,
            candidate_budget=budget,
            pad_token_id=self.pad_token_id,
        )

    def prepare_device_proposal(
        self,
        request_id: int,
        *,
        candidate_budget: int,
    ) -> bool:
        """Prepare a provider-owned exact graph before the first hot cycle."""

        prepare = getattr(self.executor, "prepare_proposal_graph", None)
        if not callable(prepare):
            return False
        return bool(
            prepare(
                int(request_id),
                candidate_budget=int(candidate_budget),
            )
        )

    def launch_device_proposal(
        self,
        context: MtpProposalContext,
        *,
        candidate_budget: int,
    ) -> Qwen35GGUFNextNDeviceProposal | None:
        """Launch a cached one-request graph while keeping results on device."""

        budget = int(candidate_budget)
        if budget not in MTP_CHAIN_CANDIDATE_BUDGETS:
            raise ValueError("candidate_budget is outside the MTP chain ladder")
        if len(context.request_ids) != 1 or context.target_hidden is None:
            return None
        if context.target_hidden.dtype != DType.BF16:
            raise ValueError("GGUF NextN target_hidden must use BF16")
        if context.target_hidden.shape != (1, self.executor.hidden_size):
            raise ValueError("GGUF NextN target_hidden shape must align with requests")
        launch = getattr(self.executor, "launch_cached_graph_chain_device", None)
        if not callable(launch):
            return None
        return launch(
            int(context.request_ids[0]),
            int(context.root_tokens[0]),
            int(context.root_positions[0]),
            context.target_hidden,
            candidate_budget=budget,
        )

    def placeholder_device_proposal(
        self,
        proposal: Qwen35GGUFNextNDeviceProposal,
    ) -> DraftBatch:
        """Return shape-only scheduler rows while proposal IDs remain on device."""

        return compile_mtp_chain(
            (
                MtpDraftRequest(
                    request_id=int(proposal.request_id),
                    root_position=int(proposal.root_position),
                    candidate_tokens=(self.pad_token_id,) * int(proposal.budget),
                    active_count=int(proposal.budget),
                ),
            ),
            candidate_budget=int(proposal.budget),
            pad_token_id=self.pad_token_id,
        )

    def finish_device_proposal(
        self,
        proposal: Qwen35GGUFNextNDeviceProposal,
        *,
        token_ids: tuple[int, ...],
        top1_values: tuple[float, ...],
    ) -> DraftBatch:
        """Publish a target-retired device proposal through the normal DraftBatch ABI."""

        materialize = getattr(self.executor, "materialize_device_proposal", None)
        if not callable(materialize):
            raise RuntimeError("GGUF NextN executor cannot materialize a device proposal")
        results = tuple(
            materialize(
                proposal,
                token_ids=tuple(int(token) for token in token_ids),
                top1_values=tuple(float(value) for value in top1_values),
            )
        )
        if len(results) != int(proposal.budget):
            raise RuntimeError("GGUF NextN device proposal returned the wrong row count")
        self.last_results[int(proposal.request_id)] = results
        return compile_mtp_chain(
            (
                MtpDraftRequest(
                    request_id=int(proposal.request_id),
                    root_position=int(proposal.root_position),
                    candidate_tokens=tuple(int(result.token_id) for result in results),
                    active_count=len(results),
                ),
            ),
            candidate_budget=int(proposal.budget),
            pad_token_id=self.pad_token_id,
        )

    def advance_full_accept_tail(
        self,
        request_id: int,
        *,
        accepted_count: int,
    ) -> Qwen35GGUFNextNStepResult | Qwen35GGUFNextNStateAdvance | None:
        """Append the last candidate when the whole proposed chain commits.

        A B-token proposal executes inputs ``root, d1, ..., d{B-1}``, so its
        draft KV already covers every partial-accept commit row. A full accept
        additionally commits ``dB``; consume that final candidate once before
        the next proposal so the resident draft KV has the same accepted prefix.
        Rejected suffix cells need no copy or clear because the next root write
        publishes a shorter context and overwrites the first rejected position.
        """

        rid = int(request_id)
        results = self.last_results.get(rid)
        if not results:
            raise ValueError("GGUF NextN accept update requires a prior proposal")
        accepted = int(accepted_count)
        if accepted < 0 or accepted > len(results):
            raise ValueError("accepted_count must be within the prior proposal budget")
        if accepted < len(results):
            return None
        tail = results[-1]
        advance_state_only = getattr(self.executor, "advance_state_only", None)
        if callable(advance_state_only):
            return advance_state_only(
                rid,
                int(tail.token_id),
                int(tail.position) + 1,
                tail.hidden,
            )
        return self.executor.run_step(
            rid,
            int(tail.token_id),
            int(tail.position) + 1,
            tail.hidden,
            return_logits=False,
        )

    def reset_request(self, request_id: int) -> None:
        self.executor.reset_request(int(request_id))
        self.last_results.pop(int(request_id), None)

    def release_request(self, request_id: int) -> None:
        release = getattr(self.executor, "release_request", None)
        if release is None:
            self.executor.reset_request(int(request_id))
        else:
            release(int(request_id))
        self.last_results.pop(int(request_id), None)

    def close(self) -> None:
        if self.owns_executor:
            self.executor.close()


Qwen35GGUFNextNDraftModel = Qwen35GGUFNextNDraftProvider


__all__ = [
    "Qwen35GGUFNextNBatchDeviceProposal",
    "Qwen35GGUFNextNDeviceProposal",
    "Qwen35GGUFNextNRequestCheckpoint",
    "Qwen35GGUFNextNDraftModel",
    "Qwen35GGUFNextNDraftProvider",
    "Qwen35GGUFNextNExecutor",
    "Qwen35GGUFNextNStateAdvance",
    "Qwen35GGUFNextNStepExecutor",
    "Qwen35GGUFNextNStepResult",
    "borrow_qwen35_gguf_nextn_fallback_weights",
]
