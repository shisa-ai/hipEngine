"""Native execution and DraftModel provider for a GGUF trailing NextN block."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_rmsnorm_bf16_f32_weight,
)
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
)
from hipengine.kernels.registry import KernelKey, MissingKernelError, is_registered, resolve
from hipengine.loading.qwen35_gguf_materialize import Qwen35GGUFDeviceWeight
from hipengine.loading.qwen35_gguf_nextn_materialize import (
    Qwen35GGUFNextNResidentWeights,
    materialize_qwen35_gguf_nextn_weights,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner
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


@dataclass(frozen=True, slots=True)
class _Qwen35GGUFNextNProposalGraph:
    """One exact greedy proposal-chain graph bound to a resident request slot."""

    slot: int
    budget: int
    stream: int
    graph: int
    graph_exec: int


# The fixed graph topology is proven for the production B1-B3 ladder through
# scalar full-attention context 1,023. Larger cache allocations may still use it
# while the live chain fits; requests crossing into split-K retain the exact
# eager chain until that separate topology is gated.
_NEXTN_EXACT_CHAIN_GRAPH_BUDGETS = (1, 2, 3)
_NEXTN_EXACT_CHAIN_GRAPH_MAX_CONTEXT = 1023
_NEXTN_TOP1_RESULT_DTYPE = np.dtype([("token", np.int32), ("value", np.float32)])
_NEXTN_TOP1_RESULT_NBYTES = int(_NEXTN_TOP1_RESULT_DTYPE.itemsize)
_NEXTN_TOP1_RESULT_CAPACITY = max(_NEXTN_EXACT_CHAIN_GRAPH_BUDGETS)


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
    ) -> None:
        if max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        self.model = Path(model)
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
        )
        adapted = self.weights.as_full_stack_weights()
        self.runner = Qwen35GGUFFullStackRunner(
            self.model,
            runtime=self.runtime,
            compiler_version=self.compiler_version,
            require_cached_build=self.require_cached_build,
            resident_weights=adapted,
            owns_resident_weights=False,
        )
        self.hidden_size = int(self.runner.hidden_size)
        self.vocab_size = int(self.runner.vocab_size)
        self.scratch = self.runner.allocate_scratch(
            max_sequence_length=int(max_positions),
            max_batch_size=self.max_requests,
        )
        self.scratch.zero_states(self.runtime)
        self._request_slots: dict[int, int] = {}
        self._token_host = np.zeros((self.max_requests,), dtype=np.int64)
        self._logits_host = np.empty((1, self.vocab_size), dtype=np.float32)
        hidden_bytes = self.max_requests * self.hidden_size * DType.BF16.itemsize
        self._token_buf = malloc(self._token_host.nbytes, runtime=self.runtime)
        self._embedding_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._fusion_buf = malloc(2 * hidden_bytes, runtime=self.runtime)
        self._fused_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._layer_out_buf = malloc(hidden_bytes, runtime=self.runtime)
        self._final_hidden_buf = malloc(hidden_bytes, runtime=self.runtime)
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
                self._fusion_buf,
                self._fused_buf,
                self._layer_out_buf,
                self._final_hidden_buf,
                self._logits_buf,
                self._lm_head_top1_block_values,
                self._lm_head_top1_block_indices,
                self._lm_head_top1_result,
                self._proposal_target_hidden,
                self._proposal_results,
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
            return _Qwen35GGUFNextNProposalGraph(
                slot=int(slot),
                budget=int(budget),
                stream=int(stream),
                graph=int(graph),
                graph_exec=int(graph_exec),
            )
        except Exception:
            if capturing:
                try:
                    abandoned = runtime.stream_end_capture(stream)
                    if abandoned:
                        runtime.graph_destroy(abandoned)
                except Exception:
                    pass
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
        graph = self._proposal_graph(request_id, slot, budget)
        if graph is None:
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
        result_host = self._proposal_results_host[slot, :budget]
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
        runtime.memcpy_async(
            host_array_ptr(result_host),
            result_ptr,
            result_host.nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
            stream,
        )
        runtime.stream_synchronize(stream)
        tokens = tuple(int(value) for value in result_host["token"])
        logits = tuple(float(value) for value in result_host["value"])
        if any(value < 0 or value >= self.vocab_size for value in tokens):
            raise RuntimeError("GGUF NextN proposal graph produced an invalid token id")
        if not np.all(np.isfinite(result_host["value"])):
            raise FloatingPointError("GGUF NextN proposal graph produced NaN or Inf")
        final_hidden_ptr = self._final_hidden_buf.ptr + slot * hidden_nbytes
        hidden = Tensor.from_handle(
            final_hidden_ptr,
            (1, self.hidden_size),
            DType.BF16,
            Device("hip", 0),
        )
        rows = []
        current_token = int(token_id)
        for depth, (next_token, logit) in enumerate(zip(tokens, logits, strict=True)):
            rows.append(
                Qwen35GGUFNextNStepResult(
                    request_id=int(request_id),
                    input_token=current_token,
                    position=int(position) + depth,
                    token_id=next_token,
                    logit=logit,
                    hidden=hidden,
                    logits=None,
                )
            )
            current_token = next_token
        slot_scratch.position_host[0] = int(position) + budget - 1
        slot_scratch.context_host[0] = int(position) + budget
        self.last_lm_head_path = "exact_q6_top1"
        self._proposal_graph_replays += 1
        self._proposal_graph_last_status = "replay"
        self._proposal_graph_last_error = None
        return tuple(rows)

    def run_chain(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
        return_logits: bool = False,
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
        if not return_logits:
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
            rows.append(result)
            current_token = int(result.token_id)
            current_hidden = result.hidden
        return tuple(rows)

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

    def advance_state_only(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
    ) -> Qwen35GGUFNextNStateAdvance:
        """Consume an accepted tail without computing its discarded prediction."""

        self._run_block(request_id, token_id, position, target_hidden)
        self.runtime.device_synchronize()
        return Qwen35GGUFNextNStateAdvance(
            request_id=int(request_id),
            input_token=int(token_id),
            position=int(position),
        )

    def reset_request(self, request_id: int) -> None:
        slot = self._request_slots.get(int(request_id))
        if slot is None:
            return
        self.scratch.for_slot(slot).zero_states(self.runtime)

    def release_request(self, request_id: int) -> None:
        request_id = int(request_id)
        self.reset_request(request_id)
        self._request_slots.pop(request_id, None)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for proposal_graph in reversed(tuple(self._proposal_graphs.values())):
            self.runtime.graph_exec_destroy(proposal_graph.graph_exec)
            self.runtime.graph_destroy(proposal_graph.graph)
            self.runtime.stream_destroy(proposal_graph.stream)
        self._proposal_graphs.clear()
        for buffer in reversed(self._buffers):
            free(buffer, runtime=self.runtime)
        for buffer in reversed(self.scratch.buffers):
            free(buffer, runtime=self.runtime)
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
                results = list(
                    run_chain(
                        int(request_id),
                        current_token,
                        position,
                        hidden,
                        candidate_budget=budget,
                        return_logits=return_logits,
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
    "Qwen35GGUFNextNDraftModel",
    "Qwen35GGUFNextNDraftProvider",
    "Qwen35GGUFNextNExecutor",
    "Qwen35GGUFNextNStateAdvance",
    "Qwen35GGUFNextNStepExecutor",
    "Qwen35GGUFNextNStepResult",
]
