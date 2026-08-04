"""Native execution and DraftModel provider for a GGUF trailing NextN block."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
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

    def _run_exact_lm_head_top1(
        self,
        hidden_ptr: int,
        *,
        stream: int = 0,
    ) -> tuple[int, float] | None:
        """Return the exact token/value pair without materializing vocab logits."""

        if (
            self._lm_head_top1_kernel is None
            or self._lm_head_top1_weight is None
            or self._lm_head_top1_block_values is None
            or self._lm_head_top1_block_indices is None
            or self._lm_head_top1_result is None
            or self._lm_head_top1_libraries is None
        ):
            return None
        result_ptr = int(self._lm_head_top1_result.ptr)
        self._lm_head_top1_kernel(
            self._lm_head_top1_weight,
            int(hidden_ptr),
            self._logits_buf.ptr,
            self._lm_head_top1_block_values.ptr,
            self._lm_head_top1_block_indices.ptr,
            result_ptr,
            result_ptr + DType.INT32.itemsize,
            1,
            self.hidden_size,
            self.vocab_size,
            stream=int(stream),
            libraries=self._lm_head_top1_libraries,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        result_host = np.empty(
            (1,),
            dtype=np.dtype([("token", np.int32), ("value", np.float32)]),
        )
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
        slot_scratch.set_full_attention_position(int(position), self.runtime)
        hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        token_ptr = self._token_buf.ptr + slot * DType.INT64.itemsize
        embedding_ptr = self._embedding_buf.ptr + slot * hidden_nbytes
        fusion_ptr = self._fusion_buf.ptr + slot * 2 * hidden_nbytes
        fused_ptr = self._fused_buf.ptr + slot * hidden_nbytes
        layer_out_ptr = self._layer_out_buf.ptr + slot * hidden_nbytes
        final_hidden_ptr = self._final_hidden_buf.ptr + slot * hidden_nbytes

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
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            embedding_ptr,
            self.weights.nextn("enorm").allocation().tensor.ptr,
            fusion_ptr,
            rows=1,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=self.runtime,
        )
        gguf_rmsnorm_bf16_f32_weight(
            target_hidden.ptr,
            self.weights.nextn("hnorm").allocation().tensor.ptr,
            fusion_ptr + hidden_nbytes,
            rows=1,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.weights.nextn("eh_proj"),
            fusion_ptr,
            fused_ptr,
            rows=1,
            in_features=2 * self.hidden_size,
            out_features=self.hidden_size,
            runtime=self.runtime,
        )
        self.runner._run_full_attention_layer(
            0,
            fused_ptr,
            layer_out_ptr,
            slot_scratch,
            position=int(position),
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
            results: list[Qwen35GGUFNextNStepResult] = []
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
