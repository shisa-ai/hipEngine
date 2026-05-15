"""Bring-up runner for real Qwen3.5/PARO one-token decode smokes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import numpy as np
from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16,
)
from hipengine.kernels.hip_gfx1100.convert import fp16_to_bf16
from hipengine.kernels.hip_gfx1100.norm import paro_rmsnorm_out_bf16, paro_rmsnorm_out_fp16
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import w8a16_linear_bf16_f32_out
from hipengine.kernels.hip_gfx1100.runtime import (
    advance_decode_position_i64,
    embedding_lookup_batch_bf16_i64,
    embedding_lookup_batch_fp16_i64,
    embedding_lookup_batch_mapped_bf16_i64,
    embedding_lookup_batch_mapped_fp16_i64,
    embedding_lookup_bf16_i64,
    embedding_lookup_fp16_i64,
    set_decode_position_i64,
    set_decode_positions_i64,
    set_i64_scalar,
    set_i64_vector,
)
from hipengine.dispatch import ActiveBatch, RequestState
from hipengine.kvcache import KVLiveSpans
from hipengine.loading import (
    WeightIndex,
    float_array_to_bf16_bits,
    load_weight_index,
    materialize_qwen35_paro_full_attention_moe_c1_runtime_layer,
    materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer,
    normalize_qwen35_weight_name,
    qwen35_paro_config_from_hf,
)
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    load_host_array_to_device_as_dtype,
    load_tensor_info_to_device,
)
from hipengine.runtime.qwen35_paro import Qwen35ParoDecodeState
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative import DraftBatch, TargetCommitPlan, TargetStateCommitBuffers, TargetVerifyBatch, TargetVerifyBuffers


@dataclass(frozen=True)
class Qwen35ParoLayerRecord:
    """One layer executed by the one-token Qwen3.5/PARO smoke path."""

    layer: int
    type: str

    def to_json_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "type": self.type}


@dataclass(frozen=True)
class Qwen35ParoNextTokenResult:
    """Structured result from the one-token Qwen3.5/PARO bring-up runner."""

    model: str
    prompt: str
    prompt_ids: tuple[int, ...]
    input_token_id: int
    layers_run: tuple[Qwen35ParoLayerRecord, ...]
    next_token_id: int
    next_token_text: str
    next_token_logit: float
    lm_head: str = "cpu_numpy_argmax"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "prompt_ids": list(self.prompt_ids),
            "input_token_id": self.input_token_id,
            "layers_run": [record.to_json_dict() for record in self.layers_run],
            "next_token_id": self.next_token_id,
            "next_token_text": self.next_token_text,
            "next_token_logit": self.next_token_logit,
            "lm_head": self.lm_head,
        }


class Qwen35ParoNextTokenRunner:
    """Torch-free one-token next-token runner for the real Qwen3.5/PARO checkpoint.

    This is a correctness/bring-up path, not a performance path: it materializes one
    layer at a time, runs the c=1 decode layer chain on HIP, applies final RMSNorm on
    HIP, and computes the lm-head argmax on CPU with NumPy chunks.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        index: WeightIndex | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        self.model = Path(model)
        self.index = index or load_weight_index(self.model)
        self.config = qwen35_paro_config_from_hf(self.index.config)
        self.normalized_infos = _normalized_infos(self.index)
        self.runtime = runtime or get_hip_runtime()

    def run_next_token(
        self,
        *,
        prompt: str = "Hello",
        token_id: int | None = None,
        max_layers: int = 0,
        lm_head_chunk: int = 4096,
        progress: Callable[[dict[str, Any]], None] | None = None,
        resident_layers: bool = False,
        lm_head: str = "gpu_fp16_argmax",
    ) -> Qwen35ParoNextTokenResult:
        if lm_head_chunk <= 0:
            raise ValueError("lm_head_chunk must be positive")
        if lm_head not in {"gpu_fp16_argmax", "cpu_numpy_argmax"}:
            raise ValueError("lm_head must be 'gpu_fp16_argmax' or 'cpu_numpy_argmax'")

        def emit(event: str, **fields: Any) -> None:
            if progress is not None:
                progress({"event": event, **fields})

        token_id, prompt_ids = _select_token(self.model, prompt, token_id)
        emit("token_selected", token_id=token_id, prompt_ids=list(prompt_ids))
        runtime = self.runtime
        device = Device("hip", 0)
        buffers: list[DeviceBuffer] = []
        allocations: list[DeviceTensorAllocation] = []

        def dev(array: np.ndarray) -> DeviceBuffer:
            buf = malloc(array.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), runtime=runtime)
            return buf

        hidden_bits = float_array_to_bf16_bits(
            _read_tensor(self.normalized_infos, "language_model.embed_tokens.weight")[
                token_id : token_id + 1
            ]
        )
        if hidden_bits.shape != (1, self.config.hidden_size):
            raise ValueError(
                f"unexpected embedding row shape {hidden_bits.shape}, "
                f"expected (1, {self.config.hidden_size})"
            )
        hidden_a = dev(hidden_bits)
        hidden_b = malloc(hidden_bits.nbytes, runtime=runtime)
        buffers.append(hidden_b)
        hidden = Tensor.from_handle(hidden_a.ptr, hidden_bits.shape, DType.BF16, device)
        next_hidden = Tensor.from_handle(hidden_b.ptr, hidden_bits.shape, DType.BF16, device)

        # One-token decode smoke: all full-attention layers can reuse the same temporary
        # KV page, and all linear layers can reuse zeroed recurrent/conv state inputs.
        block_size = 256
        block_table_arr = np.asarray([0], dtype=np.int32)
        position_arr = np.asarray([0], dtype=np.int64)
        context_arr = np.asarray([1], dtype=np.int64)
        block_table_buf = dev(block_table_arr)
        position_buf = dev(position_arr)
        context_buf = dev(context_arr)
        block_table = Tensor.from_handle(block_table_buf.ptr, block_table_arr.shape, DType.INT32, device)
        position = Tensor.from_handle(position_buf.ptr, position_arr.shape, DType.INT64, device)
        context = Tensor.from_handle(context_buf.ptr, context_arr.shape, DType.INT64, device)
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=position,
            max_live_count=0,
            storage_dtype=DType.BF16,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=context,
            max_live_count=1,
            storage_dtype=DType.BF16,
        )
        cos_arr, sin_arr = _rope_tables(
            max_positions=1,
            rotary_dim=self.config.rotary_dim or self.config.head_dim,
            base=self.config.rope_theta,
        )
        cos_buf = dev(cos_arr)
        sin_buf = dev(sin_arr)
        cos = Tensor.from_handle(cos_buf.ptr, cos_arr.shape, DType.FP32, device)
        sin = Tensor.from_handle(sin_buf.ptr, sin_arr.shape, DType.FP32, device)

        key_cache_arr = np.zeros(
            (1, block_size, self.config.num_key_value_heads, self.config.head_dim),
            dtype=np.uint16,
        )
        value_cache_arr = np.zeros_like(key_cache_arr)
        key_cache_buf = dev(key_cache_arr)
        value_cache_buf = dev(value_cache_arr)
        key_cache = Tensor.from_handle(key_cache_buf.ptr, key_cache_arr.shape, DType.BF16, device)
        value_cache = Tensor.from_handle(value_cache_buf.ptr, value_cache_arr.shape, DType.BF16, device)

        qkv_width = (
            2 * self.config.linear_num_key_heads * self.config.linear_key_head_dim
            + self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        conv_zero = np.zeros((qkv_width, self.config.linear_conv_kernel_dim), dtype=np.float32)
        recurrent_zero = np.zeros(
            (
                self.config.linear_num_value_heads,
                self.config.linear_key_head_dim,
                self.config.linear_value_head_dim,
            ),
            dtype=np.float32,
        )
        conv_buf = dev(conv_zero)
        recurrent_buf = dev(recurrent_zero)
        conv_state = Tensor.from_handle(conv_buf.ptr, conv_zero.shape, DType.FP32, device)
        recurrent_state = Tensor.from_handle(recurrent_buf.ptr, recurrent_zero.shape, DType.FP32, device)

        layer_limit = (
            self.config.num_hidden_layers
            if max_layers <= 0
            else min(max_layers, self.config.num_hidden_layers)
        )
        layer_records: list[Qwen35ParoLayerRecord] = []
        resident_states: list[Qwen35ParoDecodeState] = []
        emit("layers_start", layers=layer_limit, resident=resident_layers)
        try:
            if resident_layers:
                resident_states = self._materialize_resident_states(layer_limit, emit=emit)
            for layer_id in range(layer_limit):
                layer_type = self.config.layer_types[layer_id]
                emit("layer_start", layer=layer_id, type=layer_type)
                state = (
                    resident_states[layer_id]
                    if resident_layers
                    else self._materialize_state(layer_id, layer_type, progress=_progress_forwarder(emit))
                )
                try:
                    out = self._run_layer_state(
                        state,
                        layer_type,
                        hidden,
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                        conv_buf=conv_buf,
                        recurrent_buf=recurrent_buf,
                        conv_zero=conv_zero,
                        recurrent_zero=recurrent_zero,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        key_cache_buf=key_cache_buf,
                        value_cache_buf=value_cache_buf,
                        key_cache_zero=key_cache_arr,
                        value_cache_zero=value_cache_arr,
                        append_spans=append_spans,
                        decode_spans=decode_spans,
                        cos=cos,
                        sin=sin,
                        position=position,
                    )
                    runtime.memcpy(
                        next_hidden.ptr,
                        out.ptr,
                        hidden_bits.nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                    )
                finally:
                    if not resident_layers:
                        state.free()
                hidden, next_hidden = next_hidden, hidden
                layer_records.append(Qwen35ParoLayerRecord(layer=layer_id, type=layer_type))
                emit("layer_done", layer=layer_id, type=layer_type)

            emit("final_norm_start")
            norm_weight_host = np.asarray(_read_tensor(self.normalized_infos, "language_model.norm.weight"), dtype=np.float32)
            norm_bits = float_array_to_bf16_bits(norm_weight_host + np.float32(1.0))
            norm_weight = load_host_array_to_device_as_dtype(
                "model.norm.weight",
                norm_bits,
                DType.BF16,
                runtime=runtime,
            )
            allocations.append(norm_weight)
            norm_out_buf = malloc(hidden_bits.nbytes, runtime=runtime)
            buffers.append(norm_out_buf)
            norm_out = Tensor.from_handle(norm_out_buf.ptr, hidden_bits.shape, DType.BF16, device)
            paro_rmsnorm_out_bf16(
                hidden.ptr,
                norm_weight.tensor.ptr,
                norm_out.ptr,
                1,
                self.config.hidden_size,
                self.config.rms_norm_eps,
                runtime=runtime,
            )
            runtime.device_synchronize()
            emit("final_norm_done")
            emit("lm_head_start", mode=lm_head, chunk_size=lm_head_chunk)
            if lm_head == "gpu_fp16_argmax":
                next_id, next_logit = self._gpu_lm_head_argmax(norm_out, allocations, buffers)
            else:
                final_bits = np.empty(hidden_bits.shape, dtype=np.uint16)
                copy_device_to_host(
                    host_array_ptr(final_bits),
                    DeviceBuffer(norm_out.ptr, final_bits.nbytes),
                    runtime=runtime,
                )
                final_hidden = _bf16_bits_to_float32(final_bits.reshape(-1))
                next_id, next_logit = _lm_head_argmax(
                    self.normalized_infos,
                    final_hidden,
                    chunk_size=lm_head_chunk,
                )
            emit("lm_head_done", next_token_id=next_id, next_token_logit=next_logit)
            return Qwen35ParoNextTokenResult(
                model=str(self.model),
                prompt=prompt,
                prompt_ids=tuple(prompt_ids),
                input_token_id=token_id,
                layers_run=tuple(layer_records),
                next_token_id=next_id,
                next_token_text=_decode_token(self.model, next_id),
                next_token_logit=next_logit,
                lm_head=lm_head,
            )
        finally:
            for state in reversed(resident_states):
                state.free()
            for allocation in reversed(allocations):
                allocation.free(runtime=runtime)
            for buf in reversed(buffers):
                free(buf, runtime=runtime)

    def _materialize_state(
        self,
        layer_id: int,
        layer_type: str,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen35ParoDecodeState:
        if layer_type == "linear_attention":
            return self._materialize_linear_state(layer_id, progress=progress)
        if layer_type == "full_attention":
            return self._materialize_full_state(layer_id, progress=progress)
        raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")

    def _materialize_resident_states(
        self,
        layer_limit: int,
        *,
        emit: Callable[..., None],
    ) -> list[Qwen35ParoDecodeState]:
        states: list[Qwen35ParoDecodeState] = []
        try:
            for layer_id in range(layer_limit):
                layer_type = self.config.layer_types[layer_id]
                emit("materialize_layer_start", layer=layer_id, type=layer_type)
                states.append(self._materialize_state(layer_id, layer_type, progress=_progress_forwarder(emit)))
                emit("materialize_layer_done", layer=layer_id, type=layer_type)
        except Exception:
            for state in reversed(states):
                state.free()
            raise
        return states

    def _run_layer_state(
        self,
        state: Qwen35ParoDecodeState,
        layer_type: str,
        hidden: Tensor,
        *,
        conv_state: Tensor,
        recurrent_state: Tensor,
        conv_buf: DeviceBuffer,
        recurrent_buf: DeviceBuffer,
        conv_zero: np.ndarray,
        recurrent_zero: np.ndarray,
        key_cache: Tensor,
        value_cache: Tensor,
        key_cache_buf: DeviceBuffer,
        value_cache_buf: DeviceBuffer,
        key_cache_zero: np.ndarray,
        value_cache_zero: np.ndarray,
        append_spans: KVLiveSpans,
        decode_spans: KVLiveSpans,
        cos: Tensor,
        sin: Tensor,
        position: Tensor,
    ) -> Tensor:
        if layer_type == "linear_attention":
            _copy_zero(self.runtime, conv_buf, conv_zero)
            _copy_zero(self.runtime, recurrent_buf, recurrent_zero)
            return state.run_linear_attention_moe_c1_layer_bf16(
                hidden,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
            )
        if layer_type == "full_attention":
            _copy_zero(self.runtime, key_cache_buf, key_cache_zero)
            _copy_zero(self.runtime, value_cache_buf, value_cache_zero)
            return state.run_full_attention_moe_c1_layer_bf16(
                hidden,
                key_cache=key_cache,
                value_cache=value_cache,
                append_spans=append_spans,
                decode_spans=decode_spans,
                cos_table=cos,
                sin_table=sin,
                position=position,
                max_positions=1,
            )
        raise ValueError(f"unsupported layer type {layer_type!r}")

    def _gpu_lm_head_argmax(
        self,
        hidden: Tensor,
        allocations: list[DeviceTensorAllocation],
        buffers: list[DeviceBuffer],
    ) -> tuple[int, float]:
        info = self.normalized_infos["lm_head.weight"]
        lm_head_weight = load_tensor_info_to_device(info, runtime=self.runtime)
        allocations.append(lm_head_weight)
        vocab_size, hidden_size = lm_head_weight.tensor.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError(f"lm_head hidden size {hidden_size} does not match {self.config.hidden_size}")
        threads = 256
        stage1_blocks = lm_head_argmax_stage1_blocks(vocab_size, threads=threads)
        logits = malloc(vocab_size * DType.FP32.itemsize, runtime=self.runtime)
        block_values = malloc(stage1_blocks * DType.FP32.itemsize, runtime=self.runtime)
        block_indices = malloc(stage1_blocks * DType.INT64.itemsize, runtime=self.runtime)
        out_index = malloc(DType.INT64.itemsize, runtime=self.runtime)
        out_value = malloc(DType.FP32.itemsize, runtime=self.runtime)
        buffers.extend((logits, block_values, block_indices, out_index, out_value))
        lm_head_fp16_argmax_bf16(
            hidden.ptr,
            lm_head_weight.tensor.ptr,
            logits.ptr,
            block_values.ptr,
            block_indices.ptr,
            out_index.ptr,
            out_value.ptr,
            self.config.hidden_size,
            vocab_size,
            threads=threads,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        index_host = np.empty((1,), dtype=np.int64)
        value_host = np.empty((1,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(index_host), out_index, runtime=self.runtime)
        copy_device_to_host(host_array_ptr(value_host), out_value, runtime=self.runtime)
        return int(index_host[0]), float(value_host[0])

    def _materialize_linear_state(
        self,
        layer_id: int,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen35ParoDecodeState:
        weights = materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(
            self.index,
            layer_id=layer_id,
            runtime=self.runtime,
            progress=progress,
        )
        return Qwen35ParoDecodeState(
            layer_weights=weights,
            workspace=RuntimeWorkspace(runtime=self.runtime),
            runtime=self.runtime,
        )

    def _materialize_full_state(
        self,
        layer_id: int,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Qwen35ParoDecodeState:
        weights = materialize_qwen35_paro_full_attention_moe_c1_runtime_layer(
            self.index,
            layer_id=layer_id,
            runtime=self.runtime,
            progress=progress,
        )
        return Qwen35ParoDecodeState(
            layer_weights=weights,
            workspace=RuntimeWorkspace(runtime=self.runtime),
            runtime=self.runtime,
        )


@dataclass(frozen=True)
class Qwen35ParoAutoregressiveStepResult:
    token_id: int
    token_text: str
    logit: float

    def to_json_dict(self) -> dict[str, Any]:
        return {"token_id": self.token_id, "token_text": self.token_text, "logit": self.logit}


@dataclass(frozen=True)
class Qwen35ParoResidentBatchLayout:
    """Batch-shaped resident buffer layout for Qwen3.5/PARO sessions."""

    max_batch_size: int
    hidden_size: int
    max_sequence_length: int
    block_size: int
    blocks: int
    num_key_value_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.block_size <= 0 or self.blocks <= 0:
            raise ValueError("block_size and blocks must be positive")
        if self.num_key_value_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_key_value_heads and head_dim must be positive")

    @property
    def hidden_shape(self) -> tuple[int, int]:
        return (self.max_batch_size, self.hidden_size)

    @property
    def slot_scalar_shape(self) -> tuple[int, ...]:
        return (self.max_batch_size,)

    @property
    def slot0_hidden_shape(self) -> tuple[int, int]:
        return (1, self.hidden_size)

    @property
    def full_kv_shape(self) -> tuple[int, int, int, int, int]:
        return (self.max_batch_size, self.blocks, self.block_size, self.num_key_value_heads, self.head_dim)

    @property
    def slot0_full_kv_shape(self) -> tuple[int, int, int, int]:
        return (self.blocks, self.block_size, self.num_key_value_heads, self.head_dim)


@dataclass(frozen=True)
class Qwen35ParoNativePrefillPlan:
    """Serializable planning contract for resident native prefill coverage."""

    path: str
    layer_limit: int
    linear_prefix_layers: int
    full_layer_limit_native: bool
    first_unsupported_layer: int | None
    first_unsupported_type: str | None
    blockers: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "layer_limit": self.layer_limit,
            "linear_prefix_layers": self.linear_prefix_layers,
            "full_layer_limit_native": self.full_layer_limit_native,
            "first_unsupported_layer": self.first_unsupported_layer,
            "first_unsupported_type": self.first_unsupported_type,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class Qwen35ParoResidentBatchExecution:
    """Serializable status for the current resident c>N execution path."""

    path: str
    scheduler_owned: bool
    row_execution: str
    native_prefill_plan: Qwen35ParoNativePrefillPlan
    native_compact_prefill: bool
    native_caware_decode: bool
    throughput_claim_eligible: bool
    blockers: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "scheduler_owned": self.scheduler_owned,
            "row_execution": self.row_execution,
            "native_prefill_plan": self.native_prefill_plan.to_json_dict(),
            "native_compact_prefill": self.native_compact_prefill,
            "native_caware_decode": self.native_caware_decode,
            "throughput_claim_eligible": self.throughput_claim_eligible,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class Qwen35ParoResidentSpeculativeExecution:
    """Serializable status for resident speculative target verification."""

    target_verify_batch_metadata: bool
    verify_speculative_batch_metadata: bool
    commit_verified_state_metadata: bool
    native_target_verify_executes_kernels: bool
    commit_verified_state_executes_copies: bool
    native_target_verify_ready: bool
    throughput_claim_eligible: bool
    blockers: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "native_target_verify_batch": self.target_verify_batch_metadata,
            "speculative_verify_batch": self.verify_speculative_batch_metadata,
            "commit_verified_state": self.commit_verified_state_metadata,
            "native_target_verify_executes_kernels": self.native_target_verify_executes_kernels,
            "commit_verified_state_executes_copies": self.commit_verified_state_executes_copies,
            "native_target_verify_ready": self.native_target_verify_ready,
            "throughput_claim_eligible": self.throughput_claim_eligible,
            "blockers": list(self.blockers),
        }


def qwen35_paro_native_prefill_plan(
    layer_types: Sequence[str],
    *,
    layer_limit: int | None = None,
) -> Qwen35ParoNativePrefillPlan:
    """Plan the resident native-prefill coverage for a Qwen3.5/PARO layer prefix."""

    available_layers = len(layer_types)
    limit = available_layers if layer_limit is None else int(layer_limit)
    if limit < 0:
        raise ValueError("layer_limit must be non-negative")
    if limit > available_layers:
        raise ValueError(f"layer_limit {limit} exceeds available layer_types {available_layers}")
    linear_prefix_layers = 0
    first_unsupported_layer: int | None = None
    first_unsupported_type: str | None = None
    for layer_id in range(limit):
        layer_type = str(layer_types[layer_id])
        if layer_type == "linear_attention" and first_unsupported_layer is None:
            linear_prefix_layers += 1
            continue
        first_unsupported_layer = layer_id
        first_unsupported_type = layer_type
        break
    full_layer_limit_native = linear_prefix_layers == limit
    blockers: tuple[str, ...]
    if full_layer_limit_native:
        blockers = ()
        path = "linear_attention_native_full_layer_limit"
    else:
        blockers = (
            "native prefill currently covers only linear-attention layer prefixes",
            "native compact/grouped MoE and full-attention prefill are not wired",
            f"first unsupported layer {first_unsupported_layer} is {first_unsupported_type!r}",
        )
        path = "linear_attention_prefix_only" if linear_prefix_layers else "serial_only"
    return Qwen35ParoNativePrefillPlan(
        path=path,
        layer_limit=limit,
        linear_prefix_layers=linear_prefix_layers,
        full_layer_limit_native=full_layer_limit_native,
        first_unsupported_layer=first_unsupported_layer,
        first_unsupported_type=first_unsupported_type,
        blockers=blockers,
    )


class Qwen35ParoResidentSession:
    """Resident-state autoregressive Qwen3.5/PARO c=1 inference session.

    The session materializes layer weights once, keeps per-layer linear-attention
    recurrent/conv state and per-full-attention KV caches across tokens, and runs
    actual autoregressive prompt+decode token steps. Decode is still c=1. A
    native batched prefill helper is available only for linear-attention-only
    layer prefixes; full-attention and compact/grouped MoE native prefill remain
    separate work.
    """

    def __init__(
        self,
        runner: Qwen35ParoNextTokenRunner,
        *,
        max_sequence_length: int,
        max_layers: int = 0,
        block_size: int = 256,
        chunk_size: int = 256,
        max_batch_size: int = 1,
        compiler_version: str | None = None,
        require_cached_build: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if block_size != 256:
            raise ValueError("current Qwen3.5 paged attention kernels require block_size=256")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.runner = runner
        self.model = runner.model
        self.config = runner.config
        self.runtime = runner.runtime
        self.device = Device("hip", 0)
        self.max_sequence_length = int(max_sequence_length)
        self.block_size = int(block_size)
        self.chunk_size = int(chunk_size)
        self.max_batch_size = int(max_batch_size)
        self.compiler_version = compiler_version
        self.require_cached_build = bool(require_cached_build)
        self.max_splits = (self.max_sequence_length + self.chunk_size - 1) // self.chunk_size
        self.blocks = (self.max_sequence_length + self.block_size - 1) // self.block_size
        self.batch_layout = Qwen35ParoResidentBatchLayout(
            max_batch_size=self.max_batch_size,
            hidden_size=self.config.hidden_size,
            max_sequence_length=self.max_sequence_length,
            block_size=self.block_size,
            blocks=self.blocks,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
        )
        self.layer_limit = (
            self.config.num_hidden_layers
            if max_layers <= 0
            else min(int(max_layers), self.config.num_hidden_layers)
        )
        self.progress = progress
        self.active_batch = ActiveBatch(self.max_batch_size)
        self.active_batch.admit(RequestState.from_tokens(0, (), max_new_tokens=self.max_sequence_length))
        self.buffers: list[DeviceBuffer] = []
        self.allocations: list[DeviceTensorAllocation] = []
        self.states: list[Qwen35ParoDecodeState] = []
        self.linear_states: dict[int, tuple[Tensor, Tensor, DeviceBuffer, DeviceBuffer, np.ndarray, np.ndarray]] = {}
        self.full_caches: dict[int, tuple[Tensor, Tensor, DeviceBuffer, DeviceBuffer]] = {}
        self.linear_scratch = {}
        self.full_scratch = {}
        self.moe_scratch = {}
        self.tokenizer = _load_tokenizer(self.model)
        self.closed = False
        self._build()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for state in reversed(self.states):
            state.free()
        for allocation in reversed(self.allocations):
            allocation.free(runtime=self.runtime)
        for buffer in reversed(self.buffers):
            free(buffer, runtime=self.runtime)

    def __enter__(self) -> "Qwen35ParoResidentSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def step(self, token_id: int, *, position: int, sample: bool = True) -> Qwen35ParoAutoregressiveStepResult | None:
        if self.closed:
            raise RuntimeError("session is closed")
        self._check_position(position)
        self._set_token_embedding(int(token_id))
        self._set_position(position)
        hidden = self._run_layers(position=position, stream=0)
        if not sample:
            return None
        return self._sample_from_hidden(hidden)

    def step_batch_serial(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        positions: list[int] | tuple[int, ...],
        slots: list[int] | tuple[int, ...] | None = None,
        sample: bool = True,
    ) -> tuple[Qwen35ParoAutoregressiveStepResult | None, ...]:
        """Run one decode token per physical batch slot using the resident c=1 layer path.

        This is a correctness-first c>N bridge: it consumes batch-shaped hidden,
        linear-state, and KV-cache rows but executes active rows serially until
        native c-aware layer kernels replace the fallback. Use
        :meth:`batch_execution_metadata` to label artifacts from this path so the
        serial bridge cannot be mistaken for native compact c>N throughput.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        tokens = tuple(int(token) for token in token_ids)
        pos = tuple(int(position) for position in positions)
        if len(tokens) != len(pos):
            raise ValueError("token_ids and positions must have the same length")
        if not tokens:
            raise ValueError("token_ids must be non-empty")
        slot_ids = tuple(range(len(tokens))) if slots is None else tuple(int(slot) for slot in slots)
        if len(slot_ids) != len(tokens):
            raise ValueError("slots must match token_ids length")
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("slots must be unique")

        saved_hidden, saved_next_hidden = self.hidden, self.next_hidden
        results: list[Qwen35ParoAutoregressiveStepResult | None] = []
        try:
            for token_id, position, slot in zip(tokens, pos, slot_ids, strict=True):
                self._check_slot(slot)
                self._check_position(position)
                self._set_slot_token_embedding(token_id, slot=slot)
                self._set_slot_position(position, slot=slot)
                hidden = self._run_layers(position=position, slot=slot, persist_aliases=False, stream=0)
                results.append(self._sample_from_hidden(hidden) if sample else None)
            return tuple(results)
        finally:
            self.hidden, self.next_hidden = saved_hidden, saved_next_hidden

    def native_prefill_plan(self) -> Qwen35ParoNativePrefillPlan:
        """Return the native prefill coverage currently available for this session."""

        return qwen35_paro_native_prefill_plan(self.config.layer_types, layer_limit=self.layer_limit)

    def target_verify_batch(
        self,
        draft: DraftBatch,
        *,
        root_tokens: Sequence[int],
        root_positions: Sequence[int],
    ) -> TargetVerifyBatch:
        """Materialize metadata for a resident target-verification row batch.

        This is a layout/validation helper only.  It does not run a native
        target verifier or commit state/KV rows; those remain separate runtime
        APIs before speculative decoding can become a throughput path.
        """

        if getattr(self, "closed", False):
            raise RuntimeError("session is closed")
        target = TargetVerifyBatch.from_draft(draft, root_tokens=root_tokens, root_positions=root_positions)
        if target.rows > self.max_batch_size:
            raise ValueError("target verify rows exceed resident max_batch_size")
        for position in target.positions:
            self._check_position(position)
        vocab_size = getattr(self, "vocab_size", None)
        if vocab_size is not None:
            for token_id in target.tokens:
                if token_id >= int(vocab_size):
                    raise ValueError(f"target verify token_id {token_id} outside [0, {int(vocab_size)})")
        return target

    def verify_speculative_batch(
        self,
        batch: TargetVerifyBatch,
        *,
        token_ids: Tensor,
        positions: Tensor,
        parent_rows: Tensor,
        draft_depths: Tensor,
        row_to_request: Tensor,
        active_mask: Tensor,
        target_top1: Tensor,
        accepted_counts: Tensor,
        commit_rows: Tensor,
        commit_tokens: Tensor,
        commit_positions: Tensor,
    ) -> TargetVerifyBuffers:
        """Validate resident target-verifier buffers for a speculative batch.

        This is a metadata-only ABI bridge.  It binds a `TargetVerifyBatch` to
        device Tensor handles that a future native target forward and GPU accept
        summary would use, but it does not launch kernels or commit state/KV.
        """

        if getattr(self, "closed", False):
            raise RuntimeError("session is closed")
        if batch.rows > self.max_batch_size:
            raise ValueError("target verify rows exceed resident max_batch_size")
        for position in batch.positions:
            self._check_position(position)
        buffers = TargetVerifyBuffers.for_batch(
            batch,
            token_ids=token_ids,
            positions=positions,
            parent_rows=parent_rows,
            draft_depths=draft_depths,
            row_to_request=row_to_request,
            active_mask=active_mask,
            target_top1=target_top1,
            accepted_counts=accepted_counts,
            commit_rows=commit_rows,
            commit_tokens=commit_tokens,
            commit_positions=commit_positions,
        )
        device = getattr(self, "device", None)
        if device is not None and buffers.device != device:
            raise ValueError("target verify buffers must live on the resident device")
        return buffers

    def commit_verified_state(
        self,
        plan: TargetCommitPlan,
        buffers: TargetStateCommitBuffers,
    ) -> TargetStateCommitBuffers:
        """Validate resident buffers for future verified state/KV commit.

        This metadata-only bridge checks that a transaction-scoped commit plan
        matches the device Tensor handles that a future state/KV copy kernel
        would consume.  It does not copy recurrent state, mutate KV, or mark the
        transaction committed.
        """

        if getattr(self, "closed", False):
            raise RuntimeError("session is closed")
        if plan.request_ids != buffers.request_ids:
            raise ValueError("commit plan request_ids must match state commit buffers")
        if plan.mode != buffers.mode:
            raise ValueError("commit plan mode must match state commit buffers")
        if not buffers.has_linear_state and not buffers.has_kv_rows:
            raise ValueError("state commit buffers must include linear state or KV rows")
        device = getattr(self, "device", None)
        if device is not None and buffers.device != device:
            raise ValueError("state commit buffers must live on the resident device")
        required_src_rows = max(plan.commit_rows) + 1
        accepted_rows = sum(plan.accepted_counts)
        if buffers.linear_state_src is not None and buffers.linear_state_src.shape[0] < required_src_rows:
            raise ValueError("linear state source rows must cover selected commit rows")
        if buffers.kv_rows_src is not None and buffers.kv_rows_src.shape[0] < required_src_rows:
            raise ValueError("KV source rows must cover selected commit rows")
        if buffers.kv_rows_dst is not None and buffers.kv_rows_dst.shape[0] < accepted_rows:
            raise ValueError("KV destination rows must cover accepted token rows")
        return buffers

    def speculative_execution_metadata(self) -> Qwen35ParoResidentSpeculativeExecution:
        """Describe whether resident speculative target verification is executable."""

        target_api = hasattr(type(self), "target_verify_batch")
        verify_api = hasattr(type(self), "verify_speculative_batch")
        commit_api = hasattr(type(self), "commit_verified_state")
        executes_kernels = False
        executes_copies = False
        ready = bool(target_api and verify_api and commit_api and executes_kernels and executes_copies)
        blockers = (
            "target_verify_batch/verify_speculative_batch/commit_verified_state are metadata-only",
            "native root+candidate target forward kernels are not wired",
            "GPU accept-summary kernels are not wired",
            "verified state/KV copy kernels are not wired",
        )
        return Qwen35ParoResidentSpeculativeExecution(
            target_verify_batch_metadata=target_api,
            verify_speculative_batch_metadata=verify_api,
            commit_verified_state_metadata=commit_api,
            native_target_verify_executes_kernels=executes_kernels,
            commit_verified_state_executes_copies=executes_copies,
            native_target_verify_ready=ready,
            throughput_claim_eligible=False,
            blockers=blockers,
        )

    def batch_execution_metadata(self, *, scheduler_owned: bool = False) -> Qwen35ParoResidentBatchExecution:
        """Describe whether the resident c>N path is native or a serial fallback."""

        native_prefill_plan = self.native_prefill_plan()
        blockers = [
            "step_batch_serial executes active physical slots serially through the c=1 layer path",
            "native compact/grouped MoE c>N prefill is not wired",
            "native c-aware full-attention decode graph replay is not wired",
        ]
        blockers.extend(native_prefill_plan.blockers)
        return Qwen35ParoResidentBatchExecution(
            path="scheduler_serial_slot_bridge" if scheduler_owned else "serial_slot_bridge",
            scheduler_owned=bool(scheduler_owned),
            row_execution="serial_c1_layer_path",
            native_prefill_plan=native_prefill_plan,
            native_compact_prefill=False,
            native_caware_decode=False,
            throughput_claim_eligible=False,
            blockers=tuple(dict.fromkeys(blockers)),
        )

    def prefill_linear_tokens_native(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        sample: bool = True,
        allow_rejected_correctness: bool = False,
    ) -> Qwen35ParoAutoregressiveStepResult | None:
        """Run native prefill over currently supported linear-attention prefixes.

        The helper is correctness-accepted for the available all-linear prefix
        coverage (see ``benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-scratch-restore-sweep.json``).
        When the configured layer limit extends past that prefix, remaining
        layers run token-by-token through the existing c=1 resident path as an
        explicitly labelled fallback. It is still not compact/full-attention
        native prefill for the real 40-layer model. ``allow_rejected_correctness``
        is retained only for compatibility with older diagnostic scripts.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("token_ids must be non-empty")
        if len(tokens) > self.max_sequence_length:
            raise ValueError("token_ids exceed session capacity")
        for pos, token in enumerate(tokens):
            self._check_position(pos)
            if token < 0 or token >= self.vocab_size:
                raise ValueError(f"token_id {token} outside [0, {self.vocab_size})")
        native_prefill_plan = self.native_prefill_plan()
        _ = allow_rejected_correctness
        token_arr = np.asarray(tokens, dtype=np.int64)
        token_buf = malloc(token_arr.nbytes, runtime=self.runtime)
        copy_host_to_device(token_buf, host_array_ptr(token_arr), runtime=self.runtime)
        try:
            embedding_lookup_batch_fp16_i64(
                self.embedding.tensor.ptr,
                token_buf.ptr,
                self.prefill_hidden.ptr,
                len(tokens),
                self.config.hidden_size,
                self.vocab_size,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )
            hidden = self._run_linear_prefill_layers(
                tokens=len(tokens),
                layer_limit=native_prefill_plan.linear_prefix_layers,
                stream=0,
            )
            if not native_prefill_plan.full_layer_limit_native:
                hidden = self._run_prefill_suffix_layers_serial(
                    hidden,
                    start_layer=native_prefill_plan.linear_prefix_layers,
                    tokens=len(tokens),
                    stream=0,
                )
            self.runtime.stream_synchronize(0)
            last_ptr = hidden.ptr
            if len(hidden.shape) > 1 and int(hidden.shape[0]) == len(tokens):
                last_ptr += (len(tokens) - 1) * self.hidden_nbytes
            self.runtime.memcpy(self.hidden.ptr, last_ptr, self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)
            self._restore_decode_scratch_after_prefill()
            self._set_position(len(tokens) - 1)
            if not sample:
                return None
            return self._sample_from_hidden(self.hidden)
        finally:
            free(token_buf, runtime=self.runtime)

    def capture_decode_graph(self, *, position: int, steps_per_replay: int = 1) -> "Qwen35ParoDecodeGraph":
        """Capture one generated-token decode step for replay.

        The captured step consumes the current device argmax token (`lm_out_index`),
        writes the next argmax token back to the same device scalar, and advances
        device position/context at the end. Host tokenization/text decode is not
        part of the graph.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        if steps_per_replay <= 0:
            raise ValueError("steps_per_replay must be positive")
        self._check_position(position)
        self._check_position(position + steps_per_replay - 1)
        num_splits = max(1, (position + steps_per_replay + self.chunk_size - 1) // self.chunk_size)
        stream = self.runtime.stream_create()
        self._set_position(position, stream=stream)
        self.runtime.stream_synchronize(stream)
        self.runtime.stream_begin_capture(stream)
        try:
            for offset in range(steps_per_replay):
                self._step_from_device_token(
                    position=position + offset,
                    num_splits=num_splits,
                    advance_position=True,
                    stream=stream,
                )
            graph = self.runtime.stream_end_capture(stream)
        except Exception:
            # If capture fails, try to end capture so the stream is not left in capture mode.
            try:
                self.runtime.stream_end_capture(stream)
            except Exception:
                pass
            self.runtime.stream_destroy(stream)
            raise
        graph_exec = self.runtime.graph_instantiate(graph)
        return Qwen35ParoDecodeGraph(
            session=self,
            graph=graph,
            graph_exec=graph_exec,
            stream=stream,
            position=position,
            num_splits=num_splits,
            steps_per_replay=steps_per_replay,
        )

    def _step_from_device_token(self, *, position: int, num_splits: int, advance_position: bool, stream: int) -> None:
        self._check_position(position)
        self._set_token_embedding_from_ptr(self.lm_out_index.ptr, stream=stream)
        hidden = self._run_layers(position=position, num_splits_override=num_splits, stream=stream)
        self._sample_device_from_hidden(hidden, stream=stream)
        if advance_position:
            advance_decode_position_i64(
                self.position_buf.ptr,
                self.context_buf.ptr,
                stream=stream,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )

    def _slot_hidden_view(self, tensor: Tensor, slot: int) -> Tensor:
        self._check_slot(slot)
        return Tensor.from_handle(
            tensor.ptr + int(slot) * self.hidden_nbytes,
            (1, self.config.hidden_size),
            tensor.dtype,
            tensor.device,
        )

    def _slot_scalar_tensor(self, buffer: DeviceBuffer, slot: int, dtype: DType) -> Tensor:
        self._check_slot(slot)
        return Tensor.from_handle(buffer.ptr + int(slot) * dtype.itemsize, (1,), dtype, self.device)

    def _slot_linear_state(self, layer_id: int, slot: int) -> tuple[Tensor, Tensor]:
        self._check_slot(slot)
        conv_state, recurrent_state, conv_buf, recurrent_buf, _conv_zero, _recurrent_zero = self.linear_states[layer_id]
        conv_nbytes = int(np.prod(conv_state.shape)) * conv_state.dtype.itemsize
        recurrent_nbytes = int(np.prod(recurrent_state.shape)) * recurrent_state.dtype.itemsize
        return (
            Tensor.from_handle(conv_buf.ptr + int(slot) * conv_nbytes, conv_state.shape, conv_state.dtype, conv_state.device),
            Tensor.from_handle(
                recurrent_buf.ptr + int(slot) * recurrent_nbytes,
                recurrent_state.shape,
                recurrent_state.dtype,
                recurrent_state.device,
            ),
        )

    def _slot_full_cache(self, layer_id: int, slot: int) -> tuple[Tensor, Tensor]:
        self._check_slot(slot)
        key_cache, value_cache, key_buf, value_buf = self.full_caches[layer_id]
        cache_nbytes = int(np.prod(key_cache.shape)) * key_cache.dtype.itemsize
        return (
            Tensor.from_handle(key_buf.ptr + int(slot) * cache_nbytes, key_cache.shape, key_cache.dtype, key_cache.device),
            Tensor.from_handle(value_buf.ptr + int(slot) * cache_nbytes, value_cache.shape, value_cache.dtype, value_cache.device),
        )

    def _slot_spans(self, slot: int) -> tuple[Tensor, KVLiveSpans, KVLiveSpans]:
        position_tensor = self._slot_scalar_tensor(self.position_buf, slot, DType.INT64)
        context_tensor = self._slot_scalar_tensor(self.context_buf, slot, DType.INT64)
        append_spans = KVLiveSpans.paged_uniform(
            block_table=self.block_table,
            live_counts=position_tensor,
            max_live_count=self.max_sequence_length - 1,
            storage_dtype=DType.BF16,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=self.block_table,
            live_counts=context_tensor,
            max_live_count=self.max_sequence_length,
            storage_dtype=DType.BF16,
        )
        return position_tensor, append_spans, decode_spans

    def _check_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self.max_batch_size:
            raise ValueError(f"slot {slot} outside batch capacity {self.max_batch_size}")

    def _run_linear_prefill_layers(self, *, tokens: int, layer_limit: int | None = None, stream: int = 0) -> Tensor:
        hidden = Tensor.from_handle(self.prefill_hidden.ptr, (tokens, self.config.hidden_size), DType.FP16, self.device)
        next_hidden = Tensor.from_handle(self.prefill_next_hidden.ptr, (tokens, self.config.hidden_size), DType.FP16, self.device)
        limit = len(self.states) if layer_limit is None else int(layer_limit)
        if limit < 0 or limit > len(self.states):
            raise ValueError("layer_limit outside resident state range")
        for layer_id in range(limit):
            state = self.states[layer_id]
            layer_type = self.config.layer_types[layer_id]
            if layer_type != "linear_attention":
                raise NotImplementedError(f"native linear prefill cannot run layer {layer_id} type {layer_type!r}")
            conv_state, recurrent_state, _conv_buf, _recurrent_buf, _conv_zero, _recurrent_zero = self.linear_states[layer_id]
            linear_scratch = self.linear_scratch[layer_id]
            if linear_scratch.attn_input.shape[0] < tokens:
                linear_scratch = state.reserve_linear_attention_scratch(tokens=tokens, activation_dtype=DType.FP16)
                self.linear_scratch[layer_id] = linear_scratch
            moe_scratch = self.moe_scratch[layer_id]
            if moe_scratch.normed.shape[0] < tokens:
                moe_scratch = state.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
                self.moe_scratch[layer_id] = moe_scratch
            out = state.run_linear_attention_moe_c1_layer_fp16(
                hidden,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                linear_scratch=linear_scratch,
                moe_scratch=moe_scratch,
                tokens=tokens,
                library=self.libraries,
                stream=stream,
            )
            self.runtime.memcpy_async(next_hidden.ptr, out.ptr, tokens * self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
            hidden, next_hidden = next_hidden, hidden
        return hidden

    def _prefill_row_hidden_view(self, tensor: Tensor, row: int) -> Tensor:
        if row < 0 or row >= int(tensor.shape[0]):
            raise ValueError(f"row {row} outside tensor shape {tensor.shape}")
        return Tensor.from_handle(
            tensor.ptr + int(row) * self.hidden_nbytes,
            (1, self.config.hidden_size),
            tensor.dtype,
            tensor.device,
        )

    def _run_prefill_suffix_layers_serial(
        self,
        hidden_rows: Tensor,
        *,
        start_layer: int,
        tokens: int,
        stream: int = 0,
    ) -> Tensor:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if start_layer < 0 or start_layer > len(self.states):
            raise ValueError("start_layer outside resident state range")
        if start_layer == len(self.states):
            return self._prefill_row_hidden_view(hidden_rows, tokens - 1)
        last_hidden: Tensor | None = None
        for position in range(tokens):
            self._set_position(position, stream=stream)
            hidden = self._prefill_row_hidden_view(hidden_rows, position)
            next_hidden = self.next_hidden
            position_tensor, append_spans, decode_spans = self._slot_spans(0)
            for layer_id in range(start_layer, len(self.states)):
                state = self.states[layer_id]
                layer_type = self.config.layer_types[layer_id]
                if layer_type == "linear_attention":
                    conv_state, recurrent_state = self._slot_linear_state(layer_id, 0)
                    out = state.run_linear_attention_moe_c1_layer_fp16(
                        hidden,
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                        linear_scratch=self.linear_scratch[layer_id],
                        moe_scratch=self.moe_scratch[layer_id],
                        library=self.libraries,
                        stream=stream,
                    )
                elif layer_type == "full_attention":
                    key_cache, value_cache = self._slot_full_cache(layer_id, 0)
                    num_splits = max(1, (position + 1 + self.chunk_size - 1) // self.chunk_size)
                    out = state.run_full_attention_moe_c1_layer_fp16(
                        hidden,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        append_spans=append_spans,
                        decode_spans=decode_spans,
                        cos_table=self.cos,
                        sin_table=self.sin,
                        position=position_tensor,
                        max_positions=self.max_sequence_length,
                        attention_scratch=self.full_scratch[layer_id],
                        moe_scratch=self.moe_scratch[layer_id],
                        chunk_size=self.chunk_size,
                        num_splits=num_splits,
                        library=self.libraries,
                        stream=stream,
                    )
                else:
                    raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
                self.runtime.memcpy_async(next_hidden.ptr, out.ptr, self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
                hidden, next_hidden = next_hidden, hidden
            last_hidden = hidden
        if last_hidden is None:
            raise RuntimeError("serial suffix prefill produced no hidden row")
        return last_hidden

    def _restore_decode_scratch_after_prefill(self) -> None:
        for layer_id, state in enumerate(self.states):
            self.moe_scratch[layer_id] = state.reserve_moe_c1_scratch(tokens=1, activation_dtype=DType.FP16)
            if self.config.layer_types[layer_id] == "linear_attention":
                self.linear_scratch[layer_id] = state.reserve_linear_attention_scratch(tokens=1, activation_dtype=DType.FP16)

    def _run_layers(
        self,
        *,
        position: int,
        num_splits_override: int | None = None,
        slot: int = 0,
        persist_aliases: bool = True,
        stream: int = 0,
    ) -> Tensor:
        if slot == 0 and persist_aliases:
            hidden = self.hidden
            next_hidden = self.next_hidden
        else:
            hidden = self._slot_hidden_view(self.batch_hidden, slot)
            next_hidden = self._slot_hidden_view(self.batch_next_hidden, slot)
        position_tensor, append_spans, decode_spans = self._slot_spans(slot)
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            if layer_type == "linear_attention":
                conv_state, recurrent_state = self._slot_linear_state(layer_id, slot)
                out = state.run_linear_attention_moe_c1_layer_fp16(
                    hidden,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    linear_scratch=self.linear_scratch[layer_id],
                    moe_scratch=self.moe_scratch[layer_id],
                    library=self.libraries,
                    stream=stream,
                )
            elif layer_type == "full_attention":
                key_cache, value_cache = self._slot_full_cache(layer_id, slot)
                num_splits = num_splits_override or max(1, (position + 1 + self.chunk_size - 1) // self.chunk_size)
                out = state.run_full_attention_moe_c1_layer_fp16(
                    hidden,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    append_spans=append_spans,
                    decode_spans=decode_spans,
                    cos_table=self.cos,
                    sin_table=self.sin,
                    position=position_tensor,
                    max_positions=self.max_sequence_length,
                    attention_scratch=self.full_scratch[layer_id],
                    moe_scratch=self.moe_scratch[layer_id],
                    chunk_size=self.chunk_size,
                    num_splits=num_splits,
                    library=self.libraries,
                    stream=stream,
                )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
            self.runtime.memcpy_async(next_hidden.ptr, out.ptr, self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
            hidden, next_hidden = next_hidden, hidden
        if persist_aliases:
            self.hidden = hidden
            self.next_hidden = next_hidden
        return hidden

    def _build(self) -> None:
        self._emit("resident_build_start", layers=self.layer_limit, max_sequence_length=self.max_sequence_length)
        self._load_kernel_libraries()
        self._load_embedding()
        self._load_final_norm_and_head()
        self._allocate_common_buffers()
        self._materialize_layers()
        self._emit("resident_build_done", layers=self.layer_limit)

    def _load_kernel_libraries(self) -> None:
        self._emit("load_kernel_libraries_start")
        from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode, build_qwen35_paged_kv_write
        from hipengine.kernels.hip_gfx1100.convert import build_cast
        from hipengine.kernels.hip_gfx1100.fused.paro_combine import build_paro_combine
        from hipengine.kernels.hip_gfx1100.fused.paro_silu import build_paro_silu
        from hipengine.kernels.hip_gfx1100.linear import build_dense_gemv, build_lm_head
        from hipengine.kernels.hip_gfx1100.linear_attn.conv import build_qwen35_linear_attn_conv
        from hipengine.kernels.hip_gfx1100.linear_attn.gdn import build_qwen35_linear_attn_gdn
        from hipengine.kernels.hip_gfx1100.moe.router import build_qwen35_router
        from hipengine.kernels.hip_gfx1100.norm import build_qwen35_rmsnorm
        from hipengine.kernels.hip_gfx1100.runtime import build_runtime_state
        from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv
        from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear
        from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import build_paro_rotate
        from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import build_qwen35_rotary

        build_kwargs = {
            "load": True,
            "compiler_version": self.compiler_version,
            "require_cached": self.require_cached_build,
        }
        self.libraries = {
            "attention": build_qwen35_paged_attn_decode(**build_kwargs),
            "awq": build_paro_awq_gemv(**build_kwargs),
            "cast": build_cast(**build_kwargs),
            "combine": build_paro_combine(**build_kwargs),
            "dense": build_dense_gemv(**build_kwargs),
            "kv": build_qwen35_paged_kv_write(**build_kwargs),
            "linear_conv": build_qwen35_linear_attn_conv(**build_kwargs),
            "linear_gdn": build_qwen35_linear_attn_gdn(**build_kwargs),
            "lm_head": build_lm_head(**build_kwargs),
            "norm": build_qwen35_rmsnorm(**build_kwargs),
            "qwen_rotary": build_qwen35_rotary(**build_kwargs),
            "router": build_qwen35_router(**build_kwargs),
            "rotate": build_paro_rotate(**build_kwargs),
            "runtime_state": build_runtime_state(**build_kwargs),
            "silu": build_paro_silu(**build_kwargs),
            "w8a16": build_w8a16_linear(**build_kwargs),
        }
        self._emit("load_kernel_libraries_done", count=len(self.libraries))

    def _load_embedding(self) -> None:
        self._emit("load_embedding_start")
        embed_fp16 = np.ascontiguousarray(_read_tensor(self.runner.normalized_infos, "language_model.embed_tokens.weight"), dtype=np.float16)
        if embed_fp16.shape[1] != self.config.hidden_size:
            raise ValueError(f"embedding hidden size {embed_fp16.shape[1]} does not match {self.config.hidden_size}")
        self.embedding = load_host_array_to_device_as_dtype(
            "language_model.embed_tokens.weight.fp16",
            embed_fp16,
            DType.FP16,
            runtime=self.runtime,
        )
        self.allocations.append(self.embedding)
        self.vocab_size = int(embed_fp16.shape[0])
        self.hidden_nbytes = int(self.config.hidden_size) * DType.FP16.itemsize
        self.batch_hidden_nbytes = self.max_batch_size * self.hidden_nbytes
        self.prefill_hidden_nbytes = self.max_sequence_length * self.hidden_nbytes
        self._emit("load_embedding_done", vocab_size=self.vocab_size, hidden_size=self.config.hidden_size)

    def _load_final_norm_and_head(self) -> None:
        self._emit("load_final_norm_start")
        norm_weight_host = np.asarray(_read_tensor(self.runner.normalized_infos, "language_model.norm.weight"), dtype=np.float32)
        norm_fp16 = np.ascontiguousarray(norm_weight_host + np.float32(1.0), dtype=np.float16)
        self.norm_weight = load_host_array_to_device_as_dtype(
            "model.norm.weight.fp16",
            norm_fp16,
            DType.FP16,
            runtime=self.runtime,
        )
        self.allocations.append(self.norm_weight)
        self._emit("load_final_norm_done")
        self._emit("load_lm_head_start", mode="w8a16")
        head = _read_tensor(self.runner.normalized_infos, "lm_head.weight")
        head_vocab, head_hidden = head.shape
        if int(head_hidden) != self.config.hidden_size:
            raise ValueError(f"lm_head hidden size {head_hidden} does not match {self.config.hidden_size}")
        if int(head_vocab) != self.vocab_size:
            # Some checkpoints can untie embeddings; this one should match, but the
            # runtime only requires the head's vocabulary for argmax.
            self.vocab_size = int(head_vocab)
        head_q, head_scale = _quantize_w8a16_host(head)
        self.lm_head_weight = load_host_array_to_device_as_dtype(
            "lm_head.weight_w8a16",
            head_q,
            DType.INT8,
            runtime=self.runtime,
        )
        self.lm_head_scale = load_host_array_to_device_as_dtype(
            "lm_head.weight_w8a16_scale",
            head_scale,
            DType.FP32,
            runtime=self.runtime,
        )
        self.allocations.extend((self.lm_head_weight, self.lm_head_scale))
        self._emit("load_lm_head_done", vocab_size=self.vocab_size, mode="w8a16")

    def _allocate_common_buffers(self) -> None:
        hidden_buf = malloc(self.batch_hidden_nbytes, runtime=self.runtime)
        next_hidden_buf = malloc(self.batch_hidden_nbytes, runtime=self.runtime)
        norm_out_buf = malloc(self.batch_hidden_nbytes, runtime=self.runtime)
        norm_out_bf16_buf = malloc(self.batch_hidden_nbytes, runtime=self.runtime)
        prefill_hidden_buf = malloc(self.prefill_hidden_nbytes, runtime=self.runtime)
        prefill_next_hidden_buf = malloc(self.prefill_hidden_nbytes, runtime=self.runtime)
        self.buffers.extend((hidden_buf, next_hidden_buf, norm_out_buf, norm_out_bf16_buf, prefill_hidden_buf, prefill_next_hidden_buf))
        self.batch_hidden = Tensor.from_handle(hidden_buf.ptr, self.batch_layout.hidden_shape, DType.FP16, self.device)
        self.batch_next_hidden = Tensor.from_handle(next_hidden_buf.ptr, self.batch_layout.hidden_shape, DType.FP16, self.device)
        self.batch_norm_out = Tensor.from_handle(norm_out_buf.ptr, self.batch_layout.hidden_shape, DType.FP16, self.device)
        self.batch_norm_out_bf16 = Tensor.from_handle(norm_out_bf16_buf.ptr, self.batch_layout.hidden_shape, DType.BF16, self.device)
        self.hidden = Tensor.from_handle(hidden_buf.ptr, self.batch_layout.slot0_hidden_shape, DType.FP16, self.device)
        self.next_hidden = Tensor.from_handle(next_hidden_buf.ptr, self.batch_layout.slot0_hidden_shape, DType.FP16, self.device)
        self.norm_out = Tensor.from_handle(norm_out_buf.ptr, self.batch_layout.slot0_hidden_shape, DType.FP16, self.device)
        self.norm_out_bf16 = Tensor.from_handle(norm_out_bf16_buf.ptr, self.batch_layout.slot0_hidden_shape, DType.BF16, self.device)
        self.prefill_hidden = Tensor.from_handle(
            prefill_hidden_buf.ptr,
            (self.max_sequence_length, self.config.hidden_size),
            DType.FP16,
            self.device,
        )
        self.prefill_next_hidden = Tensor.from_handle(
            prefill_next_hidden_buf.ptr,
            (self.max_sequence_length, self.config.hidden_size),
            DType.FP16,
            self.device,
        )

        block_table_arr = np.arange(self.blocks, dtype=np.int32)
        self.position_arr = np.zeros(self.batch_layout.slot_scalar_shape, dtype=np.int64)
        self.context_arr = np.ones(self.batch_layout.slot_scalar_shape, dtype=np.int64)
        self.token_id_arr = np.zeros(self.batch_layout.slot_scalar_shape, dtype=np.int64)
        self.active_mask_arr = np.zeros(self.batch_layout.slot_scalar_shape, dtype=np.uint8)
        self.active_mask_arr[0] = 1
        self.block_table_buf = self._dev(block_table_arr)
        self.position_buf = self._dev(self.position_arr)
        self.context_buf = self._dev(self.context_arr)
        self.token_id_buf = self._dev(self.token_id_arr)
        self.active_mask_buf = self._dev(self.active_mask_arr)
        self.block_table = Tensor.from_handle(self.block_table_buf.ptr, block_table_arr.shape, DType.INT32, self.device)
        self.batch_positions = Tensor.from_handle(self.position_buf.ptr, self.batch_layout.slot_scalar_shape, DType.INT64, self.device)
        self.batch_contexts = Tensor.from_handle(self.context_buf.ptr, self.batch_layout.slot_scalar_shape, DType.INT64, self.device)
        self.batch_token_ids = Tensor.from_handle(self.token_id_buf.ptr, self.batch_layout.slot_scalar_shape, DType.INT64, self.device)
        self.active_mask = Tensor.from_handle(self.active_mask_buf.ptr, self.batch_layout.slot_scalar_shape, DType.BOOL, self.device)
        self.position_tensor = Tensor.from_handle(self.position_buf.ptr, (1,), DType.INT64, self.device)
        self.context_tensor = Tensor.from_handle(self.context_buf.ptr, (1,), DType.INT64, self.device)
        self.append_spans = KVLiveSpans.paged_uniform(
            block_table=self.block_table,
            live_counts=self.position_tensor,
            max_live_count=self.max_sequence_length - 1,
            storage_dtype=DType.BF16,
        )
        self.decode_spans = KVLiveSpans.paged_uniform(
            block_table=self.block_table,
            live_counts=self.context_tensor,
            max_live_count=self.max_sequence_length,
            storage_dtype=DType.BF16,
        )

        cos_arr, sin_arr = _rope_tables(
            max_positions=self.max_sequence_length,
            rotary_dim=self.config.rotary_dim or self.config.head_dim,
            base=self.config.rope_theta,
        )
        cos_buf = self._dev(cos_arr)
        sin_buf = self._dev(sin_arr)
        self.cos = Tensor.from_handle(cos_buf.ptr, cos_arr.shape, DType.FP32, self.device)
        self.sin = Tensor.from_handle(sin_buf.ptr, sin_arr.shape, DType.FP32, self.device)

        threads = int(os.environ.get("HIPENGINE_QWEN35_LM_HEAD_THREADS", "128"))
        if threads not in {128, 256, 512}:
            raise ValueError("HIPENGINE_QWEN35_LM_HEAD_THREADS must be one of 128, 256, 512")
        self.lm_head_stage1_blocks = lm_head_argmax_stage1_blocks(self.vocab_size, threads=threads)
        self.lm_head_threads = threads
        self.lm_logits = malloc(self.vocab_size * DType.FP32.itemsize, runtime=self.runtime)
        self.lm_block_values = malloc(self.lm_head_stage1_blocks * DType.FP32.itemsize, runtime=self.runtime)
        self.lm_block_indices = malloc(self.lm_head_stage1_blocks * DType.INT64.itemsize, runtime=self.runtime)
        self.lm_out_index = malloc(DType.INT64.itemsize, runtime=self.runtime)
        self.lm_out_value = malloc(DType.FP32.itemsize, runtime=self.runtime)
        self.buffers.extend((self.lm_logits, self.lm_block_values, self.lm_block_indices, self.lm_out_index, self.lm_out_value))

    def _materialize_layers(self) -> None:
        self.states = self.runner._materialize_resident_states(self.layer_limit, emit=self._emit)
        qkv_width = (
            2 * self.config.linear_num_key_heads * self.config.linear_key_head_dim
            + self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            self.moe_scratch[layer_id] = state.reserve_moe_c1_scratch(tokens=1, activation_dtype=DType.FP16)
            if layer_type == "linear_attention":
                conv_zero = np.zeros(
                    (self.max_batch_size, qkv_width, self.config.linear_conv_kernel_dim),
                    dtype=np.float32,
                )
                recurrent_zero = np.zeros(
                    (
                        self.max_batch_size,
                        self.config.linear_num_value_heads,
                        self.config.linear_key_head_dim,
                        self.config.linear_value_head_dim,
                    ),
                    dtype=np.float32,
                )
                conv_buf = self._dev(conv_zero)
                recurrent_buf = self._dev(recurrent_zero)
                conv_state = Tensor.from_handle(
                    conv_buf.ptr,
                    (qkv_width, self.config.linear_conv_kernel_dim),
                    DType.FP32,
                    self.device,
                )
                recurrent_state = Tensor.from_handle(
                    recurrent_buf.ptr,
                    (
                        self.config.linear_num_value_heads,
                        self.config.linear_key_head_dim,
                        self.config.linear_value_head_dim,
                    ),
                    DType.FP32,
                    self.device,
                )
                self.linear_states[layer_id] = (conv_state, recurrent_state, conv_buf, recurrent_buf, conv_zero, recurrent_zero)
                self.linear_scratch[layer_id] = state.reserve_linear_attention_scratch(tokens=1, activation_dtype=DType.FP16)
            elif layer_type == "full_attention":
                key_zero = np.zeros(self.batch_layout.full_kv_shape, dtype=np.uint16)
                value_zero = np.zeros_like(key_zero)
                key_buf = self._dev(key_zero)
                value_buf = self._dev(value_zero)
                key_cache = Tensor.from_handle(key_buf.ptr, self.batch_layout.slot0_full_kv_shape, DType.BF16, self.device)
                value_cache = Tensor.from_handle(value_buf.ptr, self.batch_layout.slot0_full_kv_shape, DType.BF16, self.device)
                self.full_caches[layer_id] = (key_cache, value_cache, key_buf, value_buf)
                self.full_scratch[layer_id] = state.reserve_full_attention_scratch(
                    tokens=1,
                    num_splits=self.max_splits,
                    activation_dtype=DType.FP16,
                    gated_dtype=DType.FP16,
                )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")

    def _set_token_embedding(self, token_id: int, *, stream: int = 0) -> None:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token_id {token_id} outside [0, {self.vocab_size})")
        set_i64_scalar(
            self.token_id_buf.ptr,
            token_id,
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )
        self._set_token_embedding_from_ptr(self.token_id_buf.ptr, stream=stream)

    def _set_token_embedding_from_ptr(self, token_id_ptr: int, *, stream: int = 0) -> None:
        embedding_lookup_fp16_i64(
            self.embedding.tensor.ptr,
            token_id_ptr,
            self.hidden.ptr,
            self.config.hidden_size,
            self.vocab_size,
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )

    def _set_batch_token_embeddings(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        row_map: list[int] | tuple[int, ...] | None = None,
        stream: int = 0,
    ) -> Tensor:
        """Set batch token ids and gather embeddings into batch-hidden rows."""

        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("token_ids must be non-empty")
        if len(tokens) > self.max_batch_size:
            raise ValueError("token_ids exceed max_batch_size")
        for token in tokens:
            if token < 0 or token >= self.vocab_size:
                raise ValueError(f"token_id {token} outside [0, {self.vocab_size})")
        token_arr = np.asarray(tokens, dtype=np.int64)
        token_buf = malloc(token_arr.nbytes, runtime=self.runtime)
        row_buf = None
        try:
            copy_host_to_device(token_buf, host_array_ptr(token_arr), runtime=self.runtime)
            set_i64_vector(
                self.token_id_buf.ptr,
                token_buf.ptr,
                len(tokens),
                stream=stream,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )
            rows = len(tokens) if row_map is None else len(row_map)
            if row_map is not None:
                row_arr = np.asarray(tuple(int(row) for row in row_map), dtype=np.int32)
                if row_arr.size == 0:
                    raise ValueError("row_map must be non-empty")
                if row_arr.min() < 0 or row_arr.max() >= len(tokens):
                    raise ValueError("row_map entries must reference token_ids")
                row_buf = malloc(row_arr.nbytes, runtime=self.runtime)
                copy_host_to_device(row_buf, host_array_ptr(row_arr), runtime=self.runtime)
            embedding_lookup_batch_mapped_fp16_i64(
                self.embedding.tensor.ptr,
                self.token_id_buf.ptr,
                self.batch_hidden.ptr,
                rows,
                self.config.hidden_size,
                self.vocab_size,
                len(tokens),
                row_map_i32_ptr=None if row_buf is None else row_buf.ptr,
                stream=stream,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )
            return Tensor.from_handle(self.batch_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        finally:
            if row_buf is not None:
                free(row_buf, runtime=self.runtime)
            free(token_buf, runtime=self.runtime)

    def _set_batch_positions(
        self,
        positions: list[int] | tuple[int, ...],
        *,
        active_mask: list[bool] | tuple[bool, ...] | None = None,
        stream: int = 0,
    ) -> None:
        """Set device position/context vectors for active batch slots."""

        pos = tuple(int(position) for position in positions)
        if not pos:
            raise ValueError("positions must be non-empty")
        if len(pos) > self.max_batch_size:
            raise ValueError("positions exceed max_batch_size")
        for position in pos:
            self._check_position(position)
        pos_arr = np.asarray(pos, dtype=np.int64)
        pos_buf = malloc(pos_arr.nbytes, runtime=self.runtime)
        mask_buf = None
        try:
            copy_host_to_device(pos_buf, host_array_ptr(pos_arr), runtime=self.runtime)
            if active_mask is not None:
                mask = tuple(bool(item) for item in active_mask)
                if len(mask) != len(pos):
                    raise ValueError("active_mask must match positions")
                mask_arr = np.asarray(mask, dtype=np.uint8)
                mask_buf = malloc(mask_arr.nbytes, runtime=self.runtime)
                copy_host_to_device(mask_buf, host_array_ptr(mask_arr), runtime=self.runtime)
            set_decode_positions_i64(
                self.position_buf.ptr,
                self.context_buf.ptr,
                pos_buf.ptr,
                len(pos),
                active_mask_u8_ptr=None if mask_buf is None else mask_buf.ptr,
                stream=stream,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )
        finally:
            if mask_buf is not None:
                free(mask_buf, runtime=self.runtime)
            free(pos_buf, runtime=self.runtime)

    def _set_slot_token_embedding(self, token_id: int, *, slot: int, stream: int = 0) -> None:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token_id {token_id} outside [0, {self.vocab_size})")
        token_ptr = self.token_id_buf.ptr + int(slot) * DType.INT64.itemsize
        set_i64_scalar(
            token_ptr,
            token_id,
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )
        embedding_lookup_fp16_i64(
            self.embedding.tensor.ptr,
            token_ptr,
            self._slot_hidden_view(self.batch_hidden, slot).ptr,
            self.config.hidden_size,
            self.vocab_size,
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )

    def _set_position(self, position: int, *, stream: int = 0) -> None:
        set_decode_position_i64(
            self.position_buf.ptr,
            self.context_buf.ptr,
            int(position),
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )

    def _set_slot_position(self, position: int, *, slot: int, stream: int = 0) -> None:
        set_decode_position_i64(
            self.position_buf.ptr + int(slot) * DType.INT64.itemsize,
            self.context_buf.ptr + int(slot) * DType.INT64.itemsize,
            int(position),
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )

    def _check_position(self, position: int) -> None:
        if position < 0 or position >= self.max_sequence_length:
            raise ValueError(f"position {position} outside session capacity {self.max_sequence_length}")

    def _sample_device_from_hidden(self, hidden: Tensor, *, stream: int = 0) -> None:
        paro_rmsnorm_out_fp16(
            hidden.ptr,
            self.norm_weight.tensor.ptr,
            self.norm_out.ptr,
            1,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            stream=stream,
            library=self.libraries["norm"],
            runtime=self.runtime,
        )
        fp16_to_bf16(
            self.norm_out.ptr,
            self.norm_out_bf16.ptr,
            self.config.hidden_size,
            stream=stream,
            library=self.libraries["cast"],
            runtime=self.runtime,
        )
        w8a16_linear_bf16_f32_out(
            self.norm_out_bf16.ptr,
            self.lm_head_weight.tensor.ptr,
            self.lm_head_scale.tensor.ptr,
            self.lm_logits.ptr,
            1,
            self.config.hidden_size,
            self.vocab_size,
            threads=self.lm_head_threads,
            stream=stream,
            library=self.libraries["w8a16"],
            runtime=self.runtime,
        )
        argmax_f32(
            self.lm_logits.ptr,
            self.lm_block_values.ptr,
            self.lm_block_indices.ptr,
            self.lm_out_index.ptr,
            self.lm_out_value.ptr,
            self.vocab_size,
            threads=self.lm_head_threads,
            stream=stream,
            library=self.libraries["lm_head"],
            runtime=self.runtime,
        )

    def _sample_from_hidden(self, hidden: Tensor) -> Qwen35ParoAutoregressiveStepResult:
        self._sample_device_from_hidden(hidden)
        self.runtime.device_synchronize()
        return self._read_sample()

    def _read_sample(self) -> Qwen35ParoAutoregressiveStepResult:
        index_host = np.empty((1,), dtype=np.int64)
        value_host = np.empty((1,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(index_host), self.lm_out_index, runtime=self.runtime)
        copy_device_to_host(host_array_ptr(value_host), self.lm_out_value, runtime=self.runtime)
        token_id = int(index_host[0])
        return Qwen35ParoAutoregressiveStepResult(
            token_id=token_id,
            token_text=_decode_token_cached(self.tokenizer, token_id),
            logit=float(value_host[0]),
        )

    def _dev(self, array: np.ndarray) -> DeviceBuffer:
        buf = malloc(array.nbytes, runtime=self.runtime)
        self.buffers.append(buf)
        copy_host_to_device(buf, host_array_ptr(array), runtime=self.runtime)
        return buf

    def _emit(self, event: str, **fields: Any) -> None:
        if self.progress is not None:
            self.progress({"event": event, **fields})


@dataclass
class Qwen35ParoDecodeGraph:
    session: Qwen35ParoResidentSession
    graph: int
    graph_exec: int
    stream: int
    position: int
    num_splits: int
    steps_per_replay: int = 1
    closed: bool = False

    def replay(self, steps: int) -> None:
        if self.closed:
            raise RuntimeError("decode graph is closed")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if self.steps_per_replay <= 0:
            raise ValueError("steps_per_replay must be positive")
        if steps % self.steps_per_replay != 0:
            raise ValueError("steps must be divisible by steps_per_replay")
        launches = steps // self.steps_per_replay
        for _ in range(launches):
            self.session.runtime.graph_launch(self.graph_exec, self.stream)
        self.session.runtime.stream_synchronize(self.stream)

    def read_sample(self) -> Qwen35ParoAutoregressiveStepResult:
        if self.closed:
            raise RuntimeError("decode graph is closed")
        return self.session._read_sample()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.session.runtime.graph_exec_destroy(self.graph_exec)
        self.session.runtime.graph_destroy(self.graph)
        if self.stream:
            self.session.runtime.stream_destroy(self.stream)

    def __enter__(self) -> "Qwen35ParoDecodeGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _progress_forwarder(emit: Callable[..., None]) -> Callable[[dict[str, Any]], None]:
    def forward(payload: dict[str, Any]) -> None:
        event = str(payload.get("event", "loader"))
        fields = {key: value for key, value in payload.items() if key != "event"}
        emit(event, **fields)

    return forward


def _normalized_infos(index: WeightIndex) -> dict[str, Any]:
    out = {}
    for name, info in index.tensors.items():
        out[normalize_qwen35_weight_name(name)] = info
    return out


def _read_tensor(normalized: dict[str, Any], name: str) -> np.ndarray:
    key = normalize_qwen35_weight_name(name)
    info = normalized[key]
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        return np.ascontiguousarray(handle.get_tensor(info.name))


def _select_token(model: Path, prompt: str, token_id: int | None) -> tuple[int, list[int]]:
    if token_id is not None:
        return int(token_id), [int(token_id)]
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - optional runtime dependency guard
        raise RuntimeError("tokenizers is required unless --token-id is supplied") from exc
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("prompt produced no tokens")
    return int(ids[-1]), [int(x) for x in ids]


def _load_tokenizer(model: Path) -> Any | None:
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(model / "tokenizer.json"))
    except Exception:
        return None


def _decode_token_cached(tokenizer: Any | None, token_id: int) -> str:
    try:
        if tokenizer is None:
            return ""
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _decode_token(model: Path, token_id: int) -> str:
    return _decode_token_cached(_load_tokenizer(model), token_id)


def _copy_zero(runtime: HipRuntime, buffer: DeviceBuffer, zeros: np.ndarray) -> None:
    copy_host_to_device(buffer, host_array_ptr(zeros), runtime=runtime)


def _rope_tables(*, max_positions: int, rotary_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(base), -2.0 * dims / np.float32(rotary_dim))
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32, copy=False)
    sin = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _quantize_w8a16_host(weight: object) -> tuple[np.ndarray, np.ndarray]:
    weight_f32 = np.asarray(weight, dtype=np.float32)
    scale = np.maximum(np.max(np.abs(weight_f32), axis=1), 1.0e-8).astype(np.float32) / np.float32(127.0)
    quantized = np.rint(weight_f32 / scale[:, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scale)


def _lm_head_argmax(
    normalized: dict[str, Any],
    hidden: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[int, float]:
    info = normalized["lm_head.weight"]
    best_id = -1
    best_logit = -float("inf")
    hidden_f32 = hidden.astype(np.float32, copy=False)
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        weight = handle.get_tensor(info.name)
        rows = int(weight.shape[0])
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            logits = weight[start:end].astype(np.float32) @ hidden_f32
            local = int(np.argmax(logits))
            value = float(logits[local])
            if value > best_logit:
                best_logit = value
                best_id = start + local
    return best_id, best_logit
