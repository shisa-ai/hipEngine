"""Bring-up runner for real Qwen3.5/PARO one-token decode smokes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
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
from hipengine.kernels.backends import (
    hip_target_arch_environment,
    hip_target_arch_for_backend,
    resolve_backend,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    argmax_f32_rows_i32,
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16,
)
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_accept, dflash_accept_chain_i32, dflash_commit_chain_i32
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import build_aotriton_wrap
from hipengine.kernels.hip_gfx1100.convert import fp16_to_bf16, fp16_to_bf16_strided_rows
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
    record_i64_scalar_indexed,
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
    materialize_qwen35_paro_full_attention_dense_c1_runtime_layer,
    materialize_qwen35_paro_full_attention_moe_c1_runtime_layer,
    materialize_qwen35_paro_linear_attention_dense_c1_runtime_layer,
    materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer,
    normalize_qwen35_weight_name,
    qwen35_paro_config_from_hf,
)
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    load_host_array_to_device_as_dtype,
    load_tensor_info_to_device,
)
from hipengine.runtime.prefill import PrefillConfig, resolve_prefill_config_for_sequence
from hipengine.runtime.qwen35_paro import (
    Qwen35ParoAttentionScratch,
    Qwen35ParoDecodeState,
    Qwen35ParoDenseMlpScratch,
    Qwen35ParoGroupedMoeScratch,
    Qwen35ParoLinearAttentionScratch,
    Qwen35ParoMoeScratch,
    _use_moe_grouped_compact_prefill,
)
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative import DraftBatch, TargetAcceptSummary, TargetCommitPlan, TargetStateCommitBuffers, TargetVerifyBatch, TargetVerifyBuffers


def _env_int(name: str, default: int, *aliases: str) -> int:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return int(value)
    return default


def _paged_attn_max_splits() -> int:
    return max(
        1,
        _env_int(
            "HIPENGINE_PAGED_ATTN_MAX_SPLITS",
            4096,
            "NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS",
        ),
    )


def _paged_attn_decode_split_config(context_len: int, *, block_size: int, chunk_size: int) -> tuple[int, int]:
    """Return decode split-K chunk size and split count with an env cap."""

    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    max_splits = _paged_attn_max_splits()
    splits = (int(context_len) + int(chunk_size) - 1) // int(chunk_size)
    effective_chunk = int(chunk_size)
    if splits > max_splits:
        effective_chunk = (int(context_len) + max_splits - 1) // max_splits
        effective_chunk = ((effective_chunk + int(block_size) - 1) // int(block_size)) * int(block_size)
        splits = (int(context_len) + effective_chunk - 1) // effective_chunk
    return effective_chunk, max(1, splits)


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
        shared_expert_format: str | None = None,
        backend: str = "auto",
    ) -> None:
        self.model = Path(model)
        self.index = index or load_weight_index(self.model)
        self.config = qwen35_paro_config_from_hf(self.index.config)
        self.normalized_infos = _normalized_infos(self.index)
        self.runtime = runtime or get_hip_runtime()
        self.shared_expert_format = shared_expert_format
        self.backend = resolve_backend(backend)
        try:
            self.target_arch = hip_target_arch_for_backend(self.backend)
        except ValueError as exc:
            raise RuntimeError(
                "Qwen35ParoNextTokenRunner requires a HIP backend. Auto backend selection "
                "fell back to a non-HIP backend; pass backend='hip_gfx1100' or "
                "backend='hip_gfx1151' after validating that target on your GPU."
            ) from exc

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
        head_key = "lm_head.weight" if "lm_head.weight" in self.normalized_infos else "language_model.embed_tokens.weight"
        info = self.normalized_infos[normalize_qwen35_weight_name(head_key)]
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
        if int(getattr(self.config, "num_experts", 1) or 0) <= 0:
            weights = materialize_qwen35_paro_linear_attention_dense_c1_runtime_layer(
                self.index,
                layer_id=layer_id,
                runtime=self.runtime,
                progress=progress,
            )
        else:
            weights = materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(
                self.index,
                layer_id=layer_id,
                runtime=self.runtime,
                progress=progress,
                shared_expert_format=self.shared_expert_format,
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
        if int(getattr(self.config, "num_experts", 1) or 0) <= 0:
            weights = materialize_qwen35_paro_full_attention_dense_c1_runtime_layer(
                self.index,
                layer_id=layer_id,
                runtime=self.runtime,
                progress=progress,
            )
        else:
            weights = materialize_qwen35_paro_full_attention_moe_c1_runtime_layer(
                self.index,
                layer_id=layer_id,
                runtime=self.runtime,
                progress=progress,
                shared_expert_format=self.shared_expert_format,
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
class Qwen35ParoPackedPrefillMetadata:
    token_ids: Tensor
    positions: Tensor
    context_counts: Tensor
    block_tables: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    state_indices: Tensor
    append_spans: KVLiveSpans
    prefill_spans: KVLiveSpans
    temp_buffers: tuple[DeviceBuffer, ...]


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


@dataclass(frozen=True)
class Qwen35ParoBulkVerifyResult:
    """Result from one native root+candidate target-verification forward."""

    target_top1: tuple[int, ...]
    target_top1_values: tuple[float, ...]
    accepted_count: int
    accepted_tokens: tuple[int, ...]
    commit_row: int
    commit_token: int
    commit_position: int
    next_token: int | None
    full_accept: bool
    finite_logits: bool
    gpu_accept_match_cpu: bool
    rows: int
    target_forward_calls: int = 1
    graph: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "target_top1": list(self.target_top1),
            "target_top1_values": list(self.target_top1_values),
            "accepted_count": self.accepted_count,
            "accepted_tokens": list(self.accepted_tokens),
            "commit_row": self.commit_row,
            "commit_token": self.commit_token,
            "commit_position": self.commit_position,
            "next_token": self.next_token,
            "full_accept": self.full_accept,
            "finite_logits": self.finite_logits,
            "gpu_accept_match_cpu": self.gpu_accept_match_cpu,
            "rows": self.rows,
            "target_forward_calls": self.target_forward_calls,
            "graph": self.graph,
        }


@dataclass
class Qwen35ParoVerifierGraphEntry:
    rows: int
    capture_width: int
    base_slot: int
    graph: int
    graph_exec: int
    stream: int
    validation_passed: bool
    replay_count: int = 0


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
    supported_native_types = {"linear_attention", "full_attention"}
    for layer_id in range(limit):
        layer_type = str(layer_types[layer_id])
        if layer_type == "linear_attention" and first_unsupported_layer is None and linear_prefix_layers == layer_id:
            linear_prefix_layers += 1
        if layer_type not in supported_native_types:
            first_unsupported_layer = layer_id
            first_unsupported_type = layer_type
            break
    full_layer_limit_native = first_unsupported_layer is None
    blockers: tuple[str, ...]
    if full_layer_limit_native:
        blockers = ()
        path = "single_request_native_full"
    else:
        blockers = (
            "native prefill supports linear_attention and full_attention layers only",
            f"first unsupported layer {first_unsupported_layer} is {first_unsupported_type!r}",
        )
        path = "unsupported_layer_type"
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
    native single-request prefill helper covers linear-attention and
    full-attention layers with grouped/compact MoE; c>N compact prompt slabs
    remain separate work.
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
        prefill_config: PrefillConfig | None = None,
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
        self.backend = runner.backend
        self.target_arch = runner.target_arch
        self.device = Device("hip", 0)
        self.max_sequence_length = int(max_sequence_length)
        self.block_size = int(block_size)
        self.chunk_size = int(chunk_size)
        self.decode_chunk_size, self.max_splits = _paged_attn_decode_split_config(
            self.max_sequence_length,
            block_size=self.block_size,
            chunk_size=self.chunk_size,
        )
        self.max_batch_size = int(max_batch_size)
        self.compiler_version = compiler_version
        self.require_cached_build = bool(require_cached_build)
        self.requested_prefill_config = prefill_config or PrefillConfig()
        self._resolve_prefill_config_for_length(self.max_sequence_length)
        decode_context_capacity = self.decode_chunk_size * self.max_splits
        self.blocks = (max(self.max_sequence_length, decode_context_capacity) + self.block_size - 1) // self.block_size
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
        self.prefill_workspace = RuntimeWorkspace(runtime=self.runtime)
        self._prefill_scratch_state: Qwen35ParoDecodeState | None = None
        self.prefill_linear_scratch: Qwen35ParoLinearAttentionScratch | None = None
        self.prefill_full_scratch: Qwen35ParoAttentionScratch | None = None
        self.prefill_moe_scratch: Qwen35ParoGroupedMoeScratch | Qwen35ParoMoeScratch | None = None
        self.tokenizer = _load_tokenizer(self.model)
        self.closed = False
        self._build()

    def _prefill_tuning_total_memory_bytes(self, config: PrefillConfig, *, sequence_length: int | None = None) -> int:
        length = self.max_sequence_length if sequence_length is None else int(sequence_length)
        if (
            not config.auto_tune_chunk_sizes
            or config.chunk_tune_memory_budget_gib > 0.0
            or length < config.chunk_tune_min_tokens
        ):
            return 0
        try:
            _free_bytes, total_bytes = self.runtime.mem_get_info()
        except Exception:
            return 0
        return int(total_bytes)

    def _resolve_prefill_config_for_length(self, sequence_length: int) -> None:
        length = int(sequence_length)
        requested = getattr(self, "requested_prefill_config", getattr(self, "prefill_config", PrefillConfig()))
        total_memory_bytes = self._prefill_tuning_total_memory_bytes(
            requested,
            sequence_length=length,
        )
        self.prefill_config, self.prefill_chunk_tuning = resolve_prefill_config_for_sequence(
            requested,
            max_sequence_length=length,
            total_memory_bytes=total_memory_bytes,
        )

    def close(self) -> None:
        if self.closed:
            return
        # Kernel launches use the default stream throughout this resident session.
        # Once prefill no longer spends ~10s in accidental on-demand build calls,
        # callers can close a session while decode/prefill work is still queued;
        # freeing those buffers early can corrupt the next session in the same
        # process.  Synchronize before releasing any device allocations.
        self.runtime.device_synchronize()
        self.closed = True
        for entry in list(getattr(self, "_verify_graph_cache", {}).values()):
            try:
                self.runtime.graph_exec_destroy(entry.graph_exec)
            except Exception:
                pass
            try:
                self.runtime.graph_destroy(entry.graph)
            except Exception:
                pass
            try:
                self.runtime.stream_destroy(entry.stream)
            except Exception:
                pass
        if hasattr(self, "_verify_graph_cache"):
            self._verify_graph_cache.clear()
        self._release_prefill_workspace()
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

    def step_with_hidden_taps(
        self,
        token_id: int,
        *,
        position: int,
        capture_layer_ids: Sequence[int],
        capture_hidden_concat: Tensor,
        capture_row: int,
        sample: bool = True,
    ) -> Qwen35ParoAutoregressiveStepResult | None:
        """Run one token and append DFlash target-hidden taps to a device row.

        The taps are copied as BF16 in the order supplied by
        ``capture_layer_ids``.  This is used by the full-model DFlash benchmark
        driver to build the drafter context without copying hidden states to the
        host.  It is still a c=1 resident step; bulk verifier paths remain
        separate.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        self._check_position(position)
        self._set_token_embedding(int(token_id))
        self._set_position(position)
        hidden = self._run_layers(
            position=position,
            stream=0,
            capture_layer_ids=capture_layer_ids,
            capture_hidden_concat=capture_hidden_concat,
            capture_row=capture_row,
        )
        if not sample:
            return None
        return self._sample_from_hidden(hidden)

    def copy_slot_state(self, src_slot: int, dst_slot: int, *, stream: int = 0) -> None:
        """Copy resident decode state/KV metadata between physical slots.

        This is a correctness-first branch primitive for serial speculative
        verification: slot ``dst_slot`` receives the same recurrent state,
        full-attention KV cache, hidden scratch rows, and position/context
        scalars as ``src_slot``.  It intentionally copies the whole per-slot KV
        capacity; native DFlash verifier kernels avoid this cost with dedicated
        tree scratch and compact commit.
        """

        self._check_slot(src_slot)
        self._check_slot(dst_slot)
        if src_slot == dst_slot:
            return
        for tensor, _state in ((self.batch_hidden, "hidden"), (self.batch_next_hidden, "next_hidden")):
            stride = int(self.config.hidden_size) * tensor.dtype.itemsize
            self.runtime.memcpy_async(
                tensor.ptr + int(dst_slot) * stride,
                tensor.ptr + int(src_slot) * stride,
                stride,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
        for layer_id in self.linear_states:
            conv_state, recurrent_state, conv_buf, recurrent_buf, _conv_zero, _recurrent_zero = self.linear_states[layer_id]
            conv_stride = int(np.prod(conv_state.shape)) * conv_state.dtype.itemsize
            recurrent_stride = int(np.prod(recurrent_state.shape)) * recurrent_state.dtype.itemsize
            self.runtime.memcpy_async(
                conv_buf.ptr + int(dst_slot) * conv_stride,
                conv_buf.ptr + int(src_slot) * conv_stride,
                conv_stride,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
            self.runtime.memcpy_async(
                recurrent_buf.ptr + int(dst_slot) * recurrent_stride,
                recurrent_buf.ptr + int(src_slot) * recurrent_stride,
                recurrent_stride,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
        for layer_id in self.full_caches:
            key_cache, value_cache, key_buf, value_buf = self.full_caches[layer_id]
            cache_stride = int(np.prod(key_cache.shape)) * key_cache.dtype.itemsize
            self.runtime.memcpy_async(
                key_buf.ptr + int(dst_slot) * cache_stride,
                key_buf.ptr + int(src_slot) * cache_stride,
                cache_stride,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
            self.runtime.memcpy_async(
                value_buf.ptr + int(dst_slot) * cache_stride,
                value_buf.ptr + int(src_slot) * cache_stride,
                cache_stride,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
        for buffer in (self.position_buf, self.context_buf, self.token_id_buf):
            self.runtime.memcpy_async(
                buffer.ptr + int(dst_slot) * DType.INT64.itemsize,
                buffer.ptr + int(src_slot) * DType.INT64.itemsize,
                DType.INT64.itemsize,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )

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
        next_tokens: Tensor | None = None,
        transaction_id: int | None = None,
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
            next_tokens=next_tokens,
            transaction_id=transaction_id,
        )
        device = getattr(self, "device", None)
        if device is not None and buffers.device != device:
            raise ValueError("target verify buffers must live on the resident device")
        return buffers

    def commit_verified_state(
        self,
        plan: TargetCommitPlan,
        buffers: TargetStateCommitBuffers,
        *,
        execute_copies: bool = True,
        stream: int = 0,
        library=None,
    ) -> TargetStateCommitBuffers:
        """Commit accepted verifier state/KV rows with device copy kernels.

        The fast path consumes the compact accept-summary buffers already on
        device.  It selects the final linear-attention/hidden-tap row for each
        request, compacts accepted full-attention K/V path rows, updates
        position/context metadata, and copies committed output-ring ids without
        re-forwarding accepted prefixes.  Tests may pass ``execute_copies=False``
        to exercise validation with synthetic pointer handles only.
        """

        if getattr(self, "closed", False):
            raise RuntimeError("session is closed")
        if plan.request_ids != buffers.request_ids:
            raise ValueError("commit plan request_ids must match state commit buffers")
        if plan.transaction_id != buffers.transaction_id:
            raise ValueError("commit plan transaction_id must match state commit buffers")
        if plan.mode != buffers.mode:
            raise ValueError("commit plan mode must match state commit buffers")
        if not (
            buffers.has_linear_state
            or buffers.has_kv_rows
            or buffers.has_hidden_taps
            or buffers.has_output_ring
            or buffers.has_context_metadata
        ):
            raise ValueError("state commit buffers must include state, KV, hidden taps, output ring, or context metadata")
        device = getattr(self, "device", None)
        if device is not None and buffers.device != device:
            raise ValueError("state commit buffers must live on the resident device")
        required_src_rows = max(plan.commit_rows) + 1
        accepted_rows = sum(plan.accepted_counts)
        target_rows = buffers.parent_rows.shape[0] if buffers.parent_rows is not None else required_src_rows
        if target_rows < required_src_rows:
            raise ValueError("parent_rows must cover selected commit rows")
        if buffers.linear_state_src is not None and buffers.linear_state_src.shape[0] < required_src_rows:
            raise ValueError("linear state source rows must cover selected commit rows")
        if buffers.kv_rows_src is not None and buffers.kv_rows_src.shape[0] < required_src_rows:
            raise ValueError("KV source rows must cover selected commit rows")
        if buffers.kv_rows_dst is not None and buffers.kv_rows_dst.shape[0] < accepted_rows:
            raise ValueError("KV destination rows must cover accepted token rows")
        if buffers.hidden_taps_src is not None and buffers.hidden_taps_src.shape[1] < required_src_rows:
            raise ValueError("hidden tap source rows must cover selected commit rows")
        if execute_copies:
            dflash_commit_chain_i32(
                buffers,
                target_rows=target_rows,
                accepted_rows=accepted_rows,
                stream=stream,
                library=library,
                runtime=getattr(self, "runtime", None),
            )
        return buffers

    def speculative_execution_metadata(self) -> Qwen35ParoResidentSpeculativeExecution:
        """Describe whether resident speculative target verification is executable."""

        target_api = hasattr(type(self), "target_verify_batch")
        verify_api = hasattr(type(self), "verify_speculative_batch")
        commit_api = hasattr(type(self), "commit_verified_state")
        executes_kernels = False
        executes_copies = True
        ready = bool(target_api and verify_api and commit_api and executes_kernels and executes_copies)
        blockers = (
            "native root+candidate target forward kernels are not wired",
            "integrated native verifier still must wire GPU target-top1/accept summaries into the runtime loop",
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
            "step_batch_serial executes decode active physical slots serially through the c=1 layer path",
            "native c-aware full-attention decode graph replay is not wired",
        ]
        blockers.extend(native_prefill_plan.blockers)
        return Qwen35ParoResidentBatchExecution(
            path="scheduler_serial_slot_bridge" if scheduler_owned else "serial_slot_bridge",
            scheduler_owned=bool(scheduler_owned),
            row_execution="serial_c1_layer_path",
            native_prefill_plan=native_prefill_plan,
            native_compact_prefill=bool(native_prefill_plan.full_layer_limit_native),
            native_caware_decode=False,
            throughput_claim_eligible=False,
            blockers=tuple(dict.fromkeys(blockers)),
        )

    def prefill_native(
        self,
        token_ids: Sequence[int],
        *,
        sample: bool = True,
        require_full_native: bool | None = None,
    ) -> Qwen35ParoAutoregressiveStepResult | None:
        """Run single-request native prefill, or an explicit oracle path.

        The retained path is native across the selected layer limit: batched
        linear-attention prefill, batched full-attention append-then-attend, and
        grouped/compact MoE. Passing ``require_full_native=False`` remains an
        explicitly-labelled compatibility path for older linear-prefix oracle
        artifacts only.
        """

        tokens = self._validate_prefill_tokens(token_ids, require_min_prompt=True)
        self._resolve_prefill_config_for_length(len(tokens))
        if not self._resolve_require_full_native(require_full_native):
            self.last_prefill_execution = {
                "path": "legacy_native_linear_prefix_serial_suffix_oracle",
                "tokens": len(tokens),
                "full_native": False,
            }
            return self._prefill_linear_tokens_native_legacy(tokens, sample=sample, allow_rejected_correctness=False)
        return self._prefill_tokens_native_full(tokens, sample=sample)

    def prefill_native_packed(
        self,
        slab,
        *,
        sample: bool = True,
    ) -> tuple[Qwen35ParoAutoregressiveStepResult | None, ...]:
        """Run a compact c>N native prompt slab, once packed stages exist.

        The scheduler can already construct validated compact slabs. Executing
        them natively requires row-shaped physical block tables and segment
        metadata. This path launches one packed prompt slab over native linear
        and full-attention layers, then commits/samples one final row per
        physical slot. Decode after those seed tokens still uses the serial
        c=1 bridge until c-aware decode graph replay lands.
        """

        from hipengine.generation.batch_scheduler import CompactPromptSlab

        if not isinstance(slab, CompactPromptSlab):
            raise TypeError("slab must be a CompactPromptSlab")
        if self.closed:
            raise RuntimeError("session is closed")
        if slab.request_count > self.max_batch_size:
            raise ValueError("compact prompt slab request_count exceeds max_batch_size")
        if slab.rows > self.max_sequence_length * self.max_batch_size:
            raise ValueError("compact prompt slab rows exceed session capacity")
        if slab.block_count > self.blocks:
            raise ValueError("compact prompt slab block_count exceeds session block capacity")
        if slab.block_size != self.block_size:
            raise ValueError("compact prompt slab block_size must match session block_size")
        self._resolve_prefill_config_for_length(max(len(row) for row in slab.token_rows))
        native_prefill_plan = self.native_prefill_plan()
        if not native_prefill_plan.full_layer_limit_native:
            raise NotImplementedError(
                "native Qwen3.5/PARO packed prefill cannot cover this layer limit: "
                + "; ".join(native_prefill_plan.blockers)
            )
        metadata = self._materialize_packed_prefill_metadata(slab)
        try:
            embedding_lookup_batch_fp16_i64(
                self.embedding.tensor.ptr,
                metadata.token_ids.ptr,
                self.prefill_hidden.ptr,
                slab.rows,
                self.config.hidden_size,
                self.vocab_size,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )
            hidden = self._run_native_prefill_packed_layers(slab, metadata, stream=0)
            self.runtime.stream_synchronize(0)
            results = self._commit_packed_prefill_final_rows(hidden, slab, sample=sample, stream=0)
            self._restore_decode_scratch_after_prefill()
            self.last_prefill_execution = {
                "path": "native_prefill_compact_cN",
                "full_native": True,
                "request_count": slab.request_count,
                "rows": slab.rows,
                "block_count": slab.block_count,
                "slot_ids": list(slab.physical_slot_ids),
                "linear_prefix_layers": native_prefill_plan.linear_prefix_layers,
                "layer_limit": native_prefill_plan.layer_limit,
            }
            return results
        finally:
            for buffer in reversed(metadata.temp_buffers):
                free(buffer, runtime=self.runtime)

    def prefill_linear_tokens_native(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        sample: bool = True,
        allow_rejected_correctness: bool = False,
    ) -> Qwen35ParoAutoregressiveStepResult | None:
        """Compatibility alias for retained native-prefix artifacts.

        New call sites should use :meth:`prefill_native`. This helper preserves
        the historical linear-prefix plus serial-suffix oracle behavior for
        existing correctness scripts and artifacts.
        """

        return self._prefill_linear_tokens_native_legacy(
            token_ids,
            sample=sample,
            allow_rejected_correctness=allow_rejected_correctness,
        )

    def _prefill_tokens_native_full(
        self,
        token_ids: Sequence[int],
        *,
        sample: bool = True,
    ) -> Qwen35ParoAutoregressiveStepResult | None:
        """Run the retained full-layer single-request native prefill path."""

        tokens = self._validate_prefill_tokens(token_ids, require_min_prompt=True)
        native_prefill_plan = self.native_prefill_plan()
        if not native_prefill_plan.full_layer_limit_native:
            raise NotImplementedError(
                "native Qwen3.5/PARO prefill cannot cover this layer limit: "
                + "; ".join(native_prefill_plan.blockers)
            )
        token_arr = np.asarray(tokens, dtype=np.int64)
        if hasattr(self, "prefill_token_id_buf") and self.prefill_token_id_buf.nbytes >= token_arr.nbytes:
            token_buf = self.prefill_token_id_buf
            owns_token_buf = False
        else:
            token_buf = malloc(token_arr.nbytes, runtime=self.runtime)
            owns_token_buf = True
        copy_host_to_device(token_buf, host_array_ptr(token_arr), token_arr.nbytes, runtime=self.runtime)
        try:
            self._prepare_prefill_context_counts(len(tokens), stream=0)
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
            hidden = self._run_native_prefill_layers(tokens=len(tokens), stream=0)
            self.runtime.stream_synchronize(0)
            last_ptr = hidden.ptr
            if len(hidden.shape) > 1 and int(hidden.shape[0]) == len(tokens):
                last_ptr += (len(tokens) - 1) * self.hidden_nbytes
            self.runtime.memcpy(self.hidden.ptr, last_ptr, self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)
            self._restore_decode_scratch_after_prefill()
            self._set_position(len(tokens) - 1)
            self.last_prefill_execution = {
                "path": native_prefill_plan.path,
                "tokens": len(tokens),
                "full_native": True,
                "linear_prefix_layers": native_prefill_plan.linear_prefix_layers,
                "layer_limit": native_prefill_plan.layer_limit,
                "aotriton_attention": self._prefill_use_aotriton_attention(len(tokens)),
                "attn_aotriton_min_tokens": self.prefill_config.attn_aotriton_min_tokens,
            }
            if not sample:
                return None
            return self._sample_from_hidden(self.hidden)
        finally:
            if owns_token_buf:
                free(token_buf, runtime=self.runtime)

    def _prefill_linear_tokens_native_legacy(
        self,
        token_ids: Sequence[int],
        *,
        sample: bool = True,
        allow_rejected_correctness: bool = False,
    ) -> Qwen35ParoAutoregressiveStepResult | None:
        """Run the legacy linear-prefix native prefill oracle."""

        tokens = self._validate_prefill_tokens(token_ids, require_min_prompt=False)
        native_prefill_plan = self.native_prefill_plan()
        _ = allow_rejected_correctness
        token_arr = np.asarray(tokens, dtype=np.int64)
        if hasattr(self, "prefill_token_id_buf") and self.prefill_token_id_buf.nbytes >= token_arr.nbytes:
            token_buf = self.prefill_token_id_buf
            owns_token_buf = False
        else:
            token_buf = malloc(token_arr.nbytes, runtime=self.runtime)
            owns_token_buf = True
        copy_host_to_device(token_buf, host_array_ptr(token_arr), token_arr.nbytes, runtime=self.runtime)
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
            self.last_prefill_execution = {
                "path": "legacy_native_linear_prefix_serial_suffix_oracle",
                "tokens": len(tokens),
                "full_native": False,
                "linear_prefix_layers": native_prefill_plan.linear_prefix_layers,
                "layer_limit": native_prefill_plan.layer_limit,
            }
            if not sample:
                return None
            return self._sample_from_hidden(self.hidden)
        finally:
            if owns_token_buf:
                free(token_buf, runtime=self.runtime)

    def _validate_prefill_tokens(self, token_ids: Sequence[int], *, require_min_prompt: bool) -> tuple[int, ...]:
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
        if require_min_prompt:
            min_tokens = int(getattr(self.config, "linear_conv_kernel_dim", 1))
            if len(tokens) < min_tokens:
                raise ValueError(
                    "native prefill requires at least linear_conv_kernel_dim "
                    f"tokens ({min_tokens}); got {len(tokens)}"
                )
        return tokens

    def _resolve_require_full_native(self, require_full_native: bool | None) -> bool:
        if require_full_native is not None:
            return bool(require_full_native)
        config = getattr(self, "prefill_config", None)
        if config is None:
            return PrefillConfig().require_full_native
        return bool(config.require_full_native)

    def capture_decode_graph(
        self,
        *,
        position: int,
        steps_per_replay: int = 1,
        max_replay_steps: int | None = None,
        record_steps: int = 0,
    ) -> "Qwen35ParoDecodeGraph":
        """Capture generated-token decode steps for replay.

        The captured step consumes the current device argmax token (`lm_out_index`),
        writes the next argmax token back to the same device scalar, and advances
        device position/context at the end.  ``max_replay_steps`` lets callers
        bake enough split-K attention capacity for the full replay span rather
        than only the captured micro-step.  If ``record_steps`` is positive, each
        replayed token id is appended to a device int64 buffer for correctness
        gates; host tokenization/text decode is not part of the graph.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        if steps_per_replay <= 0:
            raise ValueError("steps_per_replay must be positive")
        if max_replay_steps is not None and max_replay_steps <= 0:
            raise ValueError("max_replay_steps must be positive")
        if record_steps < 0:
            raise ValueError("record_steps must be non-negative")
        self._check_position(position)
        replay_span = int(max_replay_steps) if max_replay_steps is not None else int(steps_per_replay)
        self._check_position(position + replay_span - 1)
        self._check_position(position + steps_per_replay - 1)
        num_splits = max(1, (position + replay_span + self.decode_chunk_size - 1) // self.decode_chunk_size)
        generated_buf: DeviceBuffer | None = None
        generated_index_buf: DeviceBuffer | None = None
        if record_steps:
            generated_buf = malloc(int(record_steps) * DType.INT64.itemsize, runtime=self.runtime)
            generated_index_buf = malloc(DType.INT64.itemsize, runtime=self.runtime)
            self.runtime.memset(generated_buf.ptr, 0xFF, generated_buf.nbytes)
            zero = np.zeros((1,), dtype=np.int64)
            copy_host_to_device(generated_index_buf, host_array_ptr(zero), runtime=self.runtime)
        graph = 0
        stream = self.runtime.stream_create()
        try:
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
                        record_output_ptr=None if generated_buf is None else generated_buf.ptr,
                        record_index_ptr=None if generated_index_buf is None else generated_index_buf.ptr,
                        record_capacity=record_steps,
                    )
                graph = self.runtime.stream_end_capture(stream)
            except Exception:
                # If capture fails, try to end capture so the stream is not left in capture mode.
                try:
                    self.runtime.stream_end_capture(stream)
                except Exception:
                    pass
                raise
            graph_exec = self.runtime.graph_instantiate(graph)
        except Exception:
            if graph:
                try:
                    self.runtime.graph_destroy(graph)
                except Exception:
                    pass
            self.runtime.stream_destroy(stream)
            if generated_index_buf is not None:
                free(generated_index_buf, runtime=self.runtime)
            if generated_buf is not None:
                free(generated_buf, runtime=self.runtime)
            raise
        return Qwen35ParoDecodeGraph(
            session=self,
            graph=graph,
            graph_exec=graph_exec,
            stream=stream,
            position=position,
            num_splits=num_splits,
            steps_per_replay=steps_per_replay,
            max_replay_steps=replay_span,
            generated=generated_buf,
            generated_index=generated_index_buf,
            record_steps=record_steps,
        )

    def _step_from_device_token(
        self,
        *,
        position: int,
        num_splits: int,
        advance_position: bool,
        stream: int,
        record_output_ptr: int | None = None,
        record_index_ptr: int | None = None,
        record_capacity: int = 0,
    ) -> None:
        self._check_position(position)
        self._set_token_embedding_from_ptr(self.lm_out_index.ptr, stream=stream)
        hidden = self._run_layers(position=position, num_splits_override=num_splits, stream=stream)
        self._sample_device_from_hidden(hidden, stream=stream)
        if record_output_ptr is not None:
            if record_index_ptr is None:
                raise ValueError("record_index_ptr is required when recording decode graph outputs")
            record_i64_scalar_indexed(
                self.lm_out_index.ptr,
                record_output_ptr,
                record_index_ptr,
                int(record_capacity),
                stream=stream,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )
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

    def _prefill_rows_tensor(self, tensor: Tensor, rows: int, *, start: int = 0) -> Tensor:
        return Tensor.from_handle(
            tensor.ptr + int(start) * tensor.dtype.itemsize,
            (rows,),
            tensor.dtype,
            tensor.device,
        )

    def _prefill_row_matrix_view(self, tensor: Tensor, start: int, rows: int) -> Tensor:
        if rows <= 0:
            raise ValueError("rows must be positive")
        if len(tensor.shape) != 2:
            raise ValueError(f"expected row-major matrix tensor, got {tensor.shape}")
        width = int(tensor.shape[1])
        if start < 0 or start + rows > int(tensor.shape[0]):
            raise ValueError(f"row view {start}:{start + rows} outside tensor shape {tensor.shape}")
        return Tensor.from_handle(
            tensor.ptr + int(start) * width * tensor.dtype.itemsize,
            (rows, width),
            tensor.dtype,
            tensor.device,
        )

    def _prefill_block_table_rows(self, rows: int, *, start: int = 0) -> Tensor:
        return Tensor.from_handle(
            self.prefill_block_table_buf.ptr + int(start) * self.blocks * DType.INT32.itemsize,
            (rows, self.blocks),
            DType.INT32,
            self.device,
        )

    def _prepare_prefill_context_counts(self, rows: int, *, stream: int = 0) -> None:
        counts = np.full((rows,), int(rows), dtype=np.int64)
        copy_host_to_device(
            self.prefill_context_count_buf,
            host_array_ptr(counts),
            counts.nbytes,
            runtime=self.runtime,
        )

    def _prefill_full_attention_spans(
        self,
        rows: int,
        *,
        start: int = 0,
        total_tokens: int | None = None,
    ) -> tuple[KVLiveSpans, KVLiveSpans]:
        total = rows if total_tokens is None else int(total_tokens)
        block_table = self._prefill_block_table_rows(rows, start=start)
        positions = self._prefill_rows_tensor(self.prefill_positions, rows, start=start)
        context_counts = Tensor.from_handle(
            self.prefill_context_count_buf.ptr + int(start) * DType.INT64.itemsize,
            (rows,),
            DType.INT64,
            self.device,
        )
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=positions,
            max_live_count=total - 1,
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="prefill",
        )
        prefill_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=context_counts,
            max_live_count=total,
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="prefill",
        )
        return append_spans, prefill_spans

    def _full_cache_all_slots(self, layer_id: int) -> tuple[Tensor, Tensor]:
        key_cache, value_cache, key_buf, value_buf = self.full_caches[layer_id]
        shape = (self.max_batch_size * self.blocks, self.block_size, self.config.num_key_value_heads, self.config.head_dim)
        return (
            Tensor.from_handle(key_buf.ptr, shape, key_cache.dtype, key_cache.device),
            Tensor.from_handle(value_buf.ptr, shape, value_cache.dtype, value_cache.device),
        )

    def _prefill_single_cu_seqlens(self, tokens: int) -> Tensor:
        arr = np.asarray([0, int(tokens)], dtype=np.int32)
        copy_host_to_device(self.prefill_single_cu_buf, host_array_ptr(arr), arr.nbytes, runtime=self.runtime)
        return self.prefill_single_cu

    def _prefill_single_cu_seqlens_pair(self, query_tokens: int, key_tokens: int) -> tuple[Tensor, Tensor]:
        q_arr = np.asarray([0, int(query_tokens)], dtype=np.int32)
        k_arr = np.asarray([0, int(key_tokens)], dtype=np.int32)
        copy_host_to_device(self.prefill_single_cu_buf, host_array_ptr(q_arr), q_arr.nbytes, runtime=self.runtime)
        copy_host_to_device(self.prefill_single_cu_k_buf, host_array_ptr(k_arr), k_arr.nbytes, runtime=self.runtime)
        return self.prefill_single_cu, self.prefill_single_cu_k

    @staticmethod
    def _chunk_ranges(total: int, chunk_size: int, *, min_chunk_size: int = 1) -> tuple[tuple[int, int], ...]:
        if total <= 0:
            raise ValueError("total must be positive")
        size = int(chunk_size)
        min_rows = max(1, int(min_chunk_size))
        if size <= 0 or total <= size:
            return ((0, int(total)),)
        ranges = [(start, min(start + size, total)) for start in range(0, total, size)]
        while len(ranges) >= 2 and ranges[-1][1] - ranges[-1][0] < min_rows:
            ranges[-2] = (ranges[-2][0], ranges[-1][1])
            ranges.pop()
        return tuple(ranges)

    @staticmethod
    def _smallest_positive_or_total(total: int, *sizes: int) -> int:
        positives = [int(size) for size in sizes if int(size) > 0]
        return int(total) if not positives else min(int(total), min(positives))

    def _linear_prefill_layer_chunk_size(self, tokens: int) -> int:
        config = self.prefill_config
        size = self._smallest_positive_or_total(tokens, config.linear_chunk_size, config.moe_chunk_size)
        min_rows = int(getattr(self.config, "linear_conv_kernel_dim", 1))
        return min(int(tokens), max(size, min_rows)) if tokens >= min_rows else size

    def _full_attention_prefill_layer_chunk_size(self, tokens: int) -> int:
        config = self.prefill_config
        if int(config.full_attn_query_chunk_size) > 0:
            size = min(int(tokens), int(config.full_attn_query_chunk_size))
        else:
            size = self._smallest_positive_or_total(
                tokens,
                config.full_attn_post_chunk_size,
                config.full_attn_rope_chunk_size,
                config.moe_chunk_size,
            )
        return 2 if tokens > 1 and size == 1 else size

    def _prefill_use_aotriton_attention(self, tokens: int) -> bool:
        threshold = int(self.prefill_config.attn_aotriton_min_tokens)
        return threshold > 0 and int(tokens) >= threshold

    def _materialize_packed_prefill_metadata(self, slab) -> Qwen35ParoPackedPrefillMetadata:
        if slab.rows > self.prefill_capacity_rows:
            raise ValueError("compact prompt slab rows exceed prefill buffer capacity")
        if slab.block_size != self.block_size:
            raise ValueError("compact prompt slab block_size must match session block_size")
        slot_by_request = dict(zip(slab.request_ids, slab.physical_slot_ids, strict=True))
        physical_tables: list[tuple[int, ...]] = []
        for request_id, local_table in zip(slab.row_to_request, slab.block_tables, strict=True):
            slot = int(slot_by_request[int(request_id)])
            self._check_slot(slot)
            row: list[int] = []
            for local_block in local_table:
                block = int(local_block)
                if block < 0 or block >= self.blocks:
                    raise ValueError("compact prompt slab block table references block outside session")
                row.append(slot * self.blocks + block)
            physical_tables.append(tuple(row))
        token_arr = np.asarray(slab.token_ids, dtype=np.int64)
        position_arr = np.asarray(slab.positions, dtype=np.int64)
        context_arr = np.asarray(slab.context_counts, dtype=np.int64)
        block_table_arr = np.asarray(physical_tables, dtype=np.int32)
        copy_host_to_device(self.prefill_token_id_buf, host_array_ptr(token_arr), token_arr.nbytes, runtime=self.runtime)
        copy_host_to_device(self.prefill_position_buf, host_array_ptr(position_arr), position_arr.nbytes, runtime=self.runtime)
        copy_host_to_device(self.prefill_context_count_buf, host_array_ptr(context_arr), context_arr.nbytes, runtime=self.runtime)
        copy_host_to_device(self.prefill_block_table_buf, host_array_ptr(block_table_arr), block_table_arr.nbytes, runtime=self.runtime)
        temp_buffers: list[DeviceBuffer] = []

        def temp_tensor(array: np.ndarray, dtype: DType) -> Tensor:
            contiguous = np.ascontiguousarray(array)
            buffer = malloc(contiguous.nbytes, runtime=self.runtime)
            temp_buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes, runtime=self.runtime)
            return Tensor.from_handle(buffer.ptr, contiguous.shape, dtype, self.device)

        cu_q = temp_tensor(np.asarray(slab.cu_seqlens_q, dtype=np.int32), DType.INT32)
        cu_k = temp_tensor(np.asarray(slab.cu_seqlens_k, dtype=np.int32), DType.INT32)
        state_indices = temp_tensor(np.asarray(slab.physical_slot_ids, dtype=np.int64), DType.INT64)
        token_tensor = Tensor.from_handle(self.prefill_token_id_buf.ptr, (slab.rows,), DType.INT64, self.device)
        position_tensor = Tensor.from_handle(self.prefill_position_buf.ptr, (slab.rows,), DType.INT64, self.device)
        context_tensor = Tensor.from_handle(self.prefill_context_count_buf.ptr, (slab.rows,), DType.INT64, self.device)
        block_table_tensor = Tensor.from_handle(self.prefill_block_table_buf.ptr, block_table_arr.shape, DType.INT32, self.device)
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=position_tensor,
            max_live_count=max(int(value) for value in slab.positions),
            storage_dtype=DType.BF16,
            row_positions=position_tensor,
            span_role="prefill",
        )
        prefill_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=context_tensor,
            max_live_count=max(int(value) for value in slab.context_counts),
            storage_dtype=DType.BF16,
            row_positions=position_tensor,
            span_role="prefill",
        )
        return Qwen35ParoPackedPrefillMetadata(
            token_ids=token_tensor,
            positions=position_tensor,
            context_counts=context_tensor,
            block_tables=block_table_tensor,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            state_indices=state_indices,
            append_spans=append_spans,
            prefill_spans=prefill_spans,
            temp_buffers=tuple(temp_buffers),
        )

    def _packed_prefill_final_rows(self, slab) -> tuple[int, ...]:
        if len(slab.cu_seqlens_q) != slab.request_count + 1:
            raise ValueError("compact slab cu_seqlens_q must align with request_count")
        rows = tuple(int(slab.cu_seqlens_q[index + 1]) - 1 for index in range(slab.request_count))
        if any(row < 0 or row >= slab.rows for row in rows):
            raise ValueError("compact slab final rows are outside slab rows")
        return rows

    def _commit_packed_prefill_final_rows(
        self,
        hidden: Tensor,
        slab,
        *,
        sample: bool = True,
        stream: int = 0,
    ) -> tuple[Qwen35ParoAutoregressiveStepResult | None, ...]:
        """Commit each compact request's final prompt row to its physical slot.

        Linear recurrent state and KV rows are updated by the packed layer
        kernels themselves. This helper commits the remaining per-request decode
        metadata: final hidden row, position, and context count, then samples
        from each final row if requested.
        """

        if len(hidden.shape) != 2 or int(hidden.shape[1]) != self.config.hidden_size:
            raise ValueError("packed prefill hidden must have shape [rows, hidden_size]")
        if int(hidden.shape[0]) < slab.rows:
            raise ValueError("packed prefill hidden rows must cover slab rows")
        final_rows = self._packed_prefill_final_rows(slab)
        slot_ids = tuple(int(slot) for slot in slab.physical_slot_ids)
        if len(slot_ids) != slab.request_count:
            raise ValueError("compact slab slot ids must align with request_count")
        results: list[Qwen35ParoAutoregressiveStepResult | None] = []
        for final_row, slot in zip(final_rows, slot_ids, strict=True):
            self._check_slot(slot)
            position = int(slab.positions[final_row])
            self._check_position(position)
            context = int(slab.context_counts[final_row])
            if context <= 0:
                raise ValueError("compact slab final context count must be positive")
            src_ptr = hidden.ptr + final_row * self.hidden_nbytes
            dst_ptr = self.batch_hidden.ptr + slot * self.hidden_nbytes
            self.runtime.memcpy_async(dst_ptr, src_ptr, self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
            self.position_arr[slot] = position
            self.context_arr[slot] = context
            if hasattr(self, "active_mask_arr"):
                self.active_mask_arr[slot] = 1
        copy_host_to_device(self.position_buf, host_array_ptr(self.position_arr), self.position_arr.nbytes, runtime=self.runtime)
        copy_host_to_device(self.context_buf, host_array_ptr(self.context_arr), self.context_arr.nbytes, runtime=self.runtime)
        if hasattr(self, "active_mask_arr") and hasattr(self, "active_mask_buf"):
            copy_host_to_device(self.active_mask_buf, host_array_ptr(self.active_mask_arr), self.active_mask_arr.nbytes, runtime=self.runtime)
        if not sample:
            self.runtime.stream_synchronize(stream)
            return tuple(None for _ in slot_ids)
        for slot in slot_ids:
            final_hidden = Tensor.from_handle(
                self.batch_hidden.ptr + slot * self.hidden_nbytes,
                (1, self.config.hidden_size),
                DType.FP16,
                self.device,
            )
            results.append(self._sample_from_hidden(final_hidden))
        return tuple(results)

    def _prefill_scratch_owner(self):
        if not self.states:
            raise RuntimeError("prefill scratch requested before layers are materialized")
        if not hasattr(self, "prefill_workspace"):
            return self.states[0]
        if getattr(self, "_prefill_scratch_state", None) is None:
            self._prefill_scratch_state = Qwen35ParoDecodeState(
                layer_weights=self.states[0].layer_weights,
                workspace=self.prefill_workspace,
                runtime=self.runtime,
            )
        return self._prefill_scratch_state

    def _release_prefill_workspace(self) -> None:
        workspace = getattr(self, "prefill_workspace", None)
        if workspace is not None:
            workspace.free()
        self.prefill_linear_scratch = None
        self.prefill_full_scratch = None
        self.prefill_moe_scratch = None

    def _ensure_linear_prefill_scratch(self, *, tokens: int) -> Qwen35ParoLinearAttentionScratch:
        scratch = getattr(self, "prefill_linear_scratch", None)
        if scratch is not None and scratch.attn_input.shape[0] >= tokens:
            return scratch
        scratch = self._prefill_scratch_owner().reserve_linear_attention_scratch(
            tokens=tokens,
            activation_dtype=DType.FP16,
        )
        self.prefill_linear_scratch = scratch
        return scratch

    def _ensure_full_prefill_scratch(self, *, tokens: int) -> Qwen35ParoAttentionScratch:
        scratch = getattr(self, "prefill_full_scratch", None)
        if scratch is not None and scratch.attn_input.shape[0] >= tokens:
            return scratch
        scratch = self._prefill_scratch_owner().reserve_full_attention_scratch(
            tokens=tokens,
            num_splits=1,
            activation_dtype=DType.FP16,
            gated_dtype=DType.FP16,
        )
        self.prefill_full_scratch = scratch
        return scratch

    def _reserve_mlp_scratch(self, state: Qwen35ParoDecodeState, *, tokens: int):
        if int(getattr(self.config, "num_experts", 1) or 0) <= 0:
            return state.reserve_dense_mlp_scratch(tokens=tokens, activation_dtype=DType.FP16)
        if tokens == 1:
            return state.reserve_moe_c1_scratch(tokens=tokens, activation_dtype=DType.FP16)
        return state.reserve_moe_grouped_prefill_scratch(tokens=tokens, activation_dtype=DType.FP16)

    def _ensure_grouped_moe_prefill_scratch(self, layer_id: int | None = None, *, tokens: int):
        _ = layer_id
        scratch = getattr(self, "prefill_moe_scratch", None)
        if int(getattr(self.config, "num_experts", 1) or 0) <= 0:
            if isinstance(scratch, Qwen35ParoDenseMlpScratch) and scratch.normed.shape[0] >= tokens:
                return scratch
            scratch = self._prefill_scratch_owner().reserve_dense_mlp_scratch(
                tokens=tokens,
                activation_dtype=DType.FP16,
            )
            self.prefill_moe_scratch = scratch
            return scratch
        if isinstance(scratch, Qwen35ParoGroupedMoeScratch) and scratch.normed.shape[0] >= tokens:
            return scratch
        scratch = self._prefill_scratch_owner().reserve_moe_grouped_prefill_scratch(
            tokens=tokens,
            activation_dtype=DType.FP16,
        )
        self.prefill_moe_scratch = scratch
        return scratch

    def _ensure_moe_prefill_scratch(
        self,
        layer_id: int | None = None,
        *,
        tokens: int,
    ) -> Qwen35ParoGroupedMoeScratch | Qwen35ParoMoeScratch:
        if _use_moe_grouped_compact_prefill(tokens):
            return self._ensure_grouped_moe_prefill_scratch(layer_id, tokens=tokens)
        _ = layer_id
        scratch = getattr(self, "prefill_moe_scratch", None)
        if isinstance(scratch, Qwen35ParoMoeScratch) and scratch.normed.shape[0] >= tokens:
            return scratch
        scratch = self._prefill_scratch_owner().reserve_moe_c1_scratch(
            tokens=tokens,
            activation_dtype=DType.FP16,
        )
        self.prefill_moe_scratch = scratch
        return scratch

    def _run_native_prefill_layers(self, *, tokens: int, stream: int = 0) -> Tensor:
        hidden = Tensor.from_handle(self.prefill_hidden.ptr, (tokens, self.config.hidden_size), DType.FP16, self.device)
        next_hidden = Tensor.from_handle(self.prefill_next_hidden.ptr, (tokens, self.config.hidden_size), DType.FP16, self.device)
        use_aotriton_attention = self._prefill_use_aotriton_attention(tokens)
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            if layer_type == "linear_attention":
                conv_state, recurrent_state, _conv_buf, _recurrent_buf, _conv_zero, _recurrent_zero = self.linear_states[layer_id]
                chunk_size = self._linear_prefill_layer_chunk_size(tokens)
                for start, end in self._chunk_ranges(
                    tokens,
                    chunk_size,
                    min_chunk_size=int(getattr(self.config, "linear_conv_kernel_dim", 1)),
                ):
                    rows = end - start
                    hidden_chunk = self._prefill_row_matrix_view(hidden, start, rows)
                    linear_scratch = self._ensure_linear_prefill_scratch(tokens=rows)
                    moe_scratch = self._ensure_moe_prefill_scratch(layer_id, tokens=rows)
                    out = state.run_linear_attention_moe_c1_layer_fp16(
                        hidden_chunk,
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                        linear_scratch=linear_scratch,
                        moe_scratch=moe_scratch,
                        tokens=rows,
                        library=self.libraries,
                        stream=stream,
                    )
                    self.runtime.memcpy_async(
                        next_hidden.ptr + start * self.hidden_nbytes,
                        out.ptr,
                        rows * self.hidden_nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
            elif layer_type == "full_attention":
                key_cache, value_cache = self._slot_full_cache(layer_id, 0)
                chunk_size = self._full_attention_prefill_layer_chunk_size(tokens)
                for start, end in self._chunk_ranges(tokens, chunk_size, min_chunk_size=2):
                    rows = end - start
                    hidden_chunk = self._prefill_row_matrix_view(hidden, start, rows)
                    append_spans, prefill_spans = self._prefill_full_attention_spans(rows, start=start, total_tokens=tokens)
                    positions = self._prefill_rows_tensor(self.prefill_positions, rows, start=start)
                    if use_aotriton_attention:
                        cu_seqlens_q, cu_seqlens_k = self._prefill_single_cu_seqlens_pair(rows, end)
                    else:
                        cu_seqlens_q = cu_seqlens_k = None
                    attention_scratch = self._ensure_full_prefill_scratch(tokens=rows)
                    moe_scratch = self._ensure_moe_prefill_scratch(layer_id, tokens=rows)
                    out = state.run_full_attention_moe_prefill_layer_fp16(
                        hidden_chunk,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        append_spans=append_spans,
                        prefill_spans=prefill_spans,
                        cos_table=self.cos,
                        sin_table=self.sin,
                        positions=positions,
                        max_positions=self.max_sequence_length,
                        attention_scratch=attention_scratch,
                        moe_scratch=moe_scratch,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        aotriton_attention=use_aotriton_attention,
                        aotriton_kv_rows=end,
                        tokens=rows,
                        block_size=self.block_size,
                        library=self.libraries,
                        stream=stream,
                    )
                    self.runtime.memcpy_async(
                        next_hidden.ptr + start * self.hidden_nbytes,
                        out.ptr,
                        rows * self.hidden_nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
            hidden, next_hidden = next_hidden, hidden
        return hidden

    def _run_native_prefill_packed_layers(
        self,
        slab,
        metadata: Qwen35ParoPackedPrefillMetadata,
        *,
        stream: int = 0,
    ) -> Tensor:
        rows = int(slab.rows)
        hidden = Tensor.from_handle(self.prefill_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        next_hidden = Tensor.from_handle(self.prefill_next_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            if layer_type == "linear_attention":
                conv_state, recurrent_state, _conv_buf, _recurrent_buf, _conv_zero, _recurrent_zero = self.linear_states[layer_id]
                linear_scratch = self._ensure_linear_prefill_scratch(tokens=rows)
                moe_scratch = self._ensure_grouped_moe_prefill_scratch(layer_id, tokens=rows)
                out = state.run_linear_attention_moe_packed_prefill_layer_fp16(
                    hidden,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    cu_seqlens=metadata.cu_seqlens_q,
                    state_indices=metadata.state_indices,
                    segments=slab.request_count,
                    linear_scratch=linear_scratch,
                    moe_scratch=moe_scratch,
                    tokens=rows,
                    library=self.libraries,
                    stream=stream,
                )
            elif layer_type == "full_attention":
                key_cache, value_cache = self._full_cache_all_slots(layer_id)
                attention_scratch = self._ensure_full_prefill_scratch(tokens=rows)
                moe_scratch = self._ensure_grouped_moe_prefill_scratch(layer_id, tokens=rows)
                out = state.run_full_attention_moe_prefill_varlen_layer_fp16(
                    hidden,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    append_spans=metadata.append_spans,
                    prefill_spans=metadata.prefill_spans,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k=metadata.cu_seqlens_k,
                    segments=slab.request_count,
                    cos_table=self.cos,
                    sin_table=self.sin,
                    positions=metadata.positions,
                    max_positions=self.max_sequence_length,
                    attention_scratch=attention_scratch,
                    moe_scratch=moe_scratch,
                    tokens=rows,
                    block_size=self.block_size,
                    library=self.libraries,
                    stream=stream,
                )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
            self.runtime.memcpy_async(next_hidden.ptr, out.ptr, rows * self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
            hidden, next_hidden = next_hidden, hidden
        return hidden

    def verify_chain_bulk_and_commit(
        self,
        batch: TargetVerifyBatch,
        *,
        base_slot: int,
        capture_layer_ids: Sequence[int],
        capture_hidden_concat: Tensor,
        capture_row_start: int,
        stream: int = 0,
        graph_mode: str = "off",
    ) -> Qwen35ParoBulkVerifyResult:
        """Run one native root+candidate verifier forward and commit the selected row.

        This is the DFlash chain verifier hot path: it executes one B+1-row
        target forward over ``batch`` against ``base_slot`` state, writes target
        hidden taps for every verifier row, computes row-wise target top-1 on the
        GPU, validates GPU accept-summary output against the CPU oracle, and
        commits the selected linear-attention row state plus decode metadata.
        Full-attention K/V rows are appended for every verifier row; unaccepted
        suffix rows are ignored because the committed context length is reset to
        the selected row position.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        if batch.mode != "verify_chain":
            raise ValueError("bulk verifier currently supports verify_chain only")
        if len(batch.request_ids) != 1:
            raise ValueError("bulk verifier E2E path currently supports one request")
        rows = int(batch.rows)
        if rows <= 1:
            raise ValueError("bulk verifier requires root plus at least one candidate row")
        if rows > self.max_batch_size:
            raise ValueError("target verify rows exceed resident max_batch_size")
        self._check_slot(base_slot)
        for position in batch.positions:
            self._check_position(int(position))
        if capture_hidden_concat.dtype != DType.BF16 or capture_hidden_concat.ndim != 2:
            raise ValueError("capture_hidden_concat must be a rank-2 BF16 tensor")
        capture_ids = tuple(int(layer_id) for layer_id in capture_layer_ids)
        if capture_hidden_concat.shape[1] != len(capture_ids) * self.config.hidden_size:
            raise ValueError("capture_hidden_concat width must match captured layers * hidden_size")
        if capture_row_start < 0 or capture_row_start + rows > capture_hidden_concat.shape[0]:
            raise ValueError("capture rows outside capture_hidden_concat")
        if graph_mode not in {"off", "auto", "validate"}:
            raise ValueError("graph_mode must be off, auto, or validate")

        capture_target = capture_hidden_concat
        capture_target_start = capture_row_start
        if graph_mode != "off":
            capture_target = self._verify_capture_staging_tensor(rows=rows, width=int(capture_hidden_concat.shape[1]))
            capture_target_start = 0

        self._write_verify_chain_metadata(batch, base_slot=base_slot, stream=stream)
        try:
            if graph_mode == "off":
                graph_info: dict[str, Any] = {"mode": "off", "status": "disabled", "replayed": False, "validation_passed": None}
                self._launch_verify_chain_forward_accept(
                    batch,
                    base_slot=base_slot,
                    capture_ids=capture_ids,
                    capture_hidden_concat=capture_target,
                    capture_row_start=capture_target_start,
                    rows=rows,
                    stream=stream,
                )
            else:
                graph_info = self._run_verify_graph_or_direct(
                    batch,
                    base_slot=base_slot,
                    capture_ids=capture_ids,
                    capture_hidden_concat=capture_target,
                    capture_row_start=capture_target_start,
                    rows=rows,
                    graph_mode=graph_mode,
                    stream=stream,
                )
            gpu_payload = self._read_verify_accept_payload(len(batch.request_ids), stream=stream)
            target_top1, target_values = self._read_verify_top1(rows)
            cpu_result = batch.accept_from_top1(target_top1, transaction_id=0)
            cpu_summary = TargetAcceptSummary.from_accept_result(batch, cpu_result)
            gpu_accept_match = self._gpu_accept_payload_matches(gpu_payload, cpu_summary)
            selected_row = int(cpu_summary.commit_rows[0])
            if graph_mode != "off":
                self._copy_verify_capture_prefix(
                    capture_target,
                    capture_hidden_concat,
                    capture_row_start=capture_row_start,
                    rows=int(cpu_summary.accepted_counts[0]) + 1,
                    stream=stream,
                )
            self._commit_bulk_linear_states(selected_row, base_slot=base_slot, stream=stream)
            self._set_slot_position(int(cpu_summary.commit_positions[0]), slot=base_slot, stream=stream)
            self.runtime.stream_synchronize(stream)
            next_token = None if cpu_summary.next_tokens is None else cpu_summary.next_tokens[0]
            return Qwen35ParoBulkVerifyResult(
                target_top1=tuple(int(token) for token in target_top1),
                target_top1_values=tuple(float(value) for value in target_values),
                accepted_count=int(cpu_summary.accepted_counts[0]),
                accepted_tokens=tuple(int(token) for token in cpu_summary.accepted_tokens[0]),
                commit_row=selected_row,
                commit_token=int(cpu_summary.commit_tokens[0]),
                commit_position=int(cpu_summary.commit_positions[0]),
                next_token=None if next_token is None else int(next_token),
                full_accept=bool(cpu_summary.full_accept[0]),
                finite_logits=all(math.isfinite(float(value)) for value in target_values),
                gpu_accept_match_cpu=bool(gpu_accept_match),
                rows=rows,
                graph=graph_info,
            )
        finally:
            # Keep verifier-sized scratch live between cycles; c=1 decode kernels
            # only consume the first row/split of the scratch tensors, and
            # avoiding bulk<->decode scratch churn keeps allocations stable for
            # future verifier graph capture experiments.
            pass

    def _verify_capture_staging_tensor(self, *, rows: int, width: int) -> Tensor:
        if rows <= 0 or rows > self.max_batch_size:
            raise ValueError("rows outside verifier staging capacity")
        max_width = len(self.config.layer_types) * self.config.hidden_size
        if width <= 0 or width > max_width:
            raise ValueError("capture width outside verifier staging capacity")
        return Tensor.from_handle(self.verify_capture_hidden_concat.ptr, (rows, width), DType.BF16, self.device)

    def _copy_verify_capture_prefix(
        self,
        src: Tensor,
        dst: Tensor,
        *,
        capture_row_start: int,
        rows: int,
        stream: int = 0,
    ) -> None:
        if rows <= 0:
            return
        if src.dtype != DType.BF16 or dst.dtype != DType.BF16 or src.ndim != 2 or dst.ndim != 2:
            raise ValueError("capture tensors must be rank-2 BF16")
        if src.shape[1] != dst.shape[1]:
            raise ValueError("capture staging width mismatch")
        if capture_row_start < 0 or capture_row_start + rows > dst.shape[0] or rows > src.shape[0]:
            raise ValueError("capture prefix range outside tensor")
        row_nbytes = int(src.shape[1]) * DType.BF16.itemsize
        self.runtime.memcpy_async(
            dst.ptr + int(capture_row_start) * row_nbytes,
            src.ptr,
            int(rows) * row_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            stream,
        )

    def _run_verify_graph_or_direct(
        self,
        batch: TargetVerifyBatch,
        *,
        base_slot: int,
        capture_ids: Sequence[int],
        capture_hidden_concat: Tensor,
        capture_row_start: int,
        rows: int,
        graph_mode: str,
        stream: int = 0,
    ) -> dict[str, Any]:
        key = (int(rows), int(capture_hidden_concat.shape[1]), int(base_slot))
        entry = self._verify_graph_cache.get(key)
        if graph_mode == "auto" and entry is not None:
            self.runtime.graph_launch(entry.graph_exec, entry.stream)
            self.runtime.stream_synchronize(entry.stream)
            entry.replay_count += 1
            return {
                "mode": graph_mode,
                "status": "replayed",
                "replayed": True,
                "validation_passed": entry.validation_passed,
                "bucket_key": {"rows": rows, "capture_width": int(capture_hidden_concat.shape[1]), "base_slot": base_slot},
                "replay_count": entry.replay_count,
            }

        self._launch_verify_chain_forward_accept(
            batch,
            base_slot=base_slot,
            capture_ids=capture_ids,
            capture_hidden_concat=capture_hidden_concat,
            capture_row_start=capture_row_start,
            rows=rows,
            stream=stream,
        )
        self.runtime.stream_synchronize(stream)
        direct_top1, _ = self._read_verify_top1(rows)
        direct_payload = self._read_verify_accept_payload(len(batch.request_ids), stream=stream)
        graph = 0
        graph_stream = 0
        try:
            graph_stream = self.runtime.stream_create()
            self.runtime.stream_begin_capture(graph_stream)
            try:
                self._launch_verify_chain_forward_accept(
                    batch,
                    base_slot=base_slot,
                    capture_ids=capture_ids,
                    capture_hidden_concat=capture_hidden_concat,
                    capture_row_start=capture_row_start,
                    rows=rows,
                    stream=graph_stream,
                )
                graph = self.runtime.stream_end_capture(graph_stream)
            except Exception:
                try:
                    self.runtime.stream_end_capture(graph_stream)
                except Exception:
                    pass
                raise
            graph_exec = self.runtime.graph_instantiate(graph)
            self.runtime.graph_launch(graph_exec, graph_stream)
            self.runtime.stream_synchronize(graph_stream)
            graph_top1, _ = self._read_verify_top1(rows)
            graph_payload = self._read_verify_accept_payload(len(batch.request_ids), stream=graph_stream)
            validation_passed = tuple(graph_top1) == tuple(direct_top1) and graph_payload == direct_payload
            if not validation_passed:
                self.runtime.graph_exec_destroy(graph_exec)
                self.runtime.graph_destroy(graph)
                self.runtime.stream_destroy(graph_stream)
                # Restore direct outputs for the caller.
                self._launch_verify_chain_forward_accept(
                    batch,
                    base_slot=base_slot,
                    capture_ids=capture_ids,
                    capture_hidden_concat=capture_hidden_concat,
                    capture_row_start=capture_row_start,
                    rows=rows,
                    stream=stream,
                )
                return {
                    "mode": graph_mode,
                    "status": "validation_failed_fallback",
                    "replayed": False,
                    "validation_passed": False,
                    "bucket_key": {"rows": rows, "capture_width": int(capture_hidden_concat.shape[1]), "base_slot": base_slot},
                }
            entry = Qwen35ParoVerifierGraphEntry(
                rows=rows,
                capture_width=int(capture_hidden_concat.shape[1]),
                base_slot=base_slot,
                graph=graph,
                graph_exec=graph_exec,
                stream=graph_stream,
                validation_passed=True,
                replay_count=1,
            )
            self._verify_graph_cache[key] = entry
            return {
                "mode": graph_mode,
                "status": "captured_validated" if graph_mode == "validate" else "captured_validated_miss",
                "replayed": graph_mode == "auto",
                "validation_passed": True,
                "bucket_key": {"rows": rows, "capture_width": int(capture_hidden_concat.shape[1]), "base_slot": base_slot},
                "replay_count": entry.replay_count,
            }
        except Exception as exc:
            if graph:
                try:
                    self.runtime.graph_destroy(graph)
                except Exception:
                    pass
            if graph_stream:
                try:
                    self.runtime.stream_destroy(graph_stream)
                except Exception:
                    pass
            # Restore direct outputs for the caller after capture failure.
            self._launch_verify_chain_forward_accept(
                batch,
                base_slot=base_slot,
                capture_ids=capture_ids,
                capture_hidden_concat=capture_hidden_concat,
                capture_row_start=capture_row_start,
                rows=rows,
                stream=stream,
            )
            return {
                "mode": graph_mode,
                "status": "capture_failed_fallback",
                "replayed": False,
                "validation_passed": None,
                "fallback_reason": str(exc),
                "bucket_key": {"rows": rows, "capture_width": int(capture_hidden_concat.shape[1]), "base_slot": base_slot},
            }

    def _launch_verify_chain_forward_accept(
        self,
        batch: TargetVerifyBatch,
        *,
        base_slot: int,
        capture_ids: Sequence[int],
        capture_hidden_concat: Tensor,
        capture_row_start: int,
        rows: int,
        stream: int = 0,
    ) -> None:
        embedding_lookup_batch_fp16_i64(
            self.embedding.tensor.ptr,
            self.verify_token_ids_i64.ptr,
            self.prefill_hidden.ptr,
            rows,
            self.config.hidden_size,
            self.vocab_size,
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )
        hidden = Tensor.from_handle(self.prefill_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        next_hidden = Tensor.from_handle(self.prefill_next_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        parent_rows = Tensor.from_handle(self.verify_parent_rows_i64.ptr, (rows,), DType.INT64, self.device)
        capture_offsets = {layer_id: idx for idx, layer_id in enumerate(capture_ids)}
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            if layer_type == "linear_attention":
                conv_state, recurrent_state = self._slot_linear_state(layer_id, base_slot)
                linear_scratch = state.reserve_linear_attention_scratch(tokens=rows, activation_dtype=DType.FP16)
                self.linear_scratch[layer_id] = linear_scratch
                moe_scratch = self._reserve_mlp_scratch(state, tokens=rows)
                self.moe_scratch[layer_id] = moe_scratch
                out = state.run_linear_attention_moe_tree_tloop_layer_fp16(
                    hidden,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    parent_rows=parent_rows,
                    linear_scratch=linear_scratch,
                    moe_scratch=moe_scratch,
                    tokens=rows,
                    library=self.libraries,
                    stream=stream,
                )
            elif layer_type == "full_attention":
                self._run_full_attention_chain_c1_loop(
                    state,
                    layer_id=layer_id,
                    hidden=hidden,
                    next_hidden=next_hidden,
                    rows=rows,
                    positions=batch.positions,
                    base_slot=base_slot,
                    stream=stream,
                )
                out = next_hidden
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
            if out.ptr != next_hidden.ptr:
                self.runtime.memcpy_async(next_hidden.ptr, out.ptr, rows * self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
            hidden, next_hidden = next_hidden, hidden
            capture_offset = capture_offsets.get(layer_id)
            if capture_offset is not None:
                dst = capture_hidden_concat.ptr + int(capture_row_start) * int(capture_hidden_concat.shape[1]) * DType.BF16.itemsize
                fp16_to_bf16_strided_rows(
                    hidden.ptr,
                    dst,
                    rows,
                    self.config.hidden_size,
                    int(capture_hidden_concat.shape[1]),
                    capture_offset * self.config.hidden_size,
                    stream=stream,
                    library=self.libraries["cast"],
                    runtime=self.runtime,
                )
        self._sample_verify_rows_from_hidden(hidden, rows, stream=stream)
        self._launch_verify_accept_summary(batch, rows=rows, stream=stream)

    def _run_full_attention_chain_c1_loop(
        self,
        state: Qwen35ParoDecodeState,
        *,
        layer_id: int,
        hidden: Tensor,
        next_hidden: Tensor,
        rows: int,
        positions: Sequence[int],
        base_slot: int,
        stream: int = 0,
    ) -> None:
        """Run a full-attention layer over verifier rows with c=1 kernels.

        The prefill-style full-attention kernels are optimized for larger prompt
        chunks and were much slower for B<=4 verifier chains.  This keeps the
        target verifier as one host-side B+1 forward (one top-1/accept sync) but
        uses the resident decode kernels row-by-row inside the layer.
        """

        if len(positions) != rows:
            raise ValueError("positions must match verifier rows")
        key_cache, value_cache = self._slot_full_cache(layer_id, base_slot)
        attention_scratch = self.full_scratch[layer_id]
        moe_scratch = self.moe_scratch[layer_id]
        for row, position in enumerate(positions):
            position_tensor, append_spans, decode_spans = self._verify_chain_row_spans(row)
            row_hidden = Tensor.from_handle(hidden.ptr + row * self.hidden_nbytes, (1, self.config.hidden_size), DType.FP16, self.device)
            row_out = Tensor.from_handle(next_hidden.ptr + row * self.hidden_nbytes, (1, self.config.hidden_size), DType.FP16, self.device)
            num_splits = max(1, (int(position) + 1 + self.decode_chunk_size - 1) // self.decode_chunk_size)
            out = state.run_full_attention_moe_c1_layer_fp16(
                row_hidden,
                key_cache=key_cache,
                value_cache=value_cache,
                append_spans=append_spans,
                decode_spans=decode_spans,
                cos_table=self.cos,
                sin_table=self.sin,
                position=position_tensor,
                max_positions=self.max_sequence_length,
                attention_scratch=attention_scratch,
                moe_scratch=moe_scratch,
                chunk_size=self.decode_chunk_size,
                num_splits=num_splits,
                library=self.libraries,
                stream=stream,
            )
            self.runtime.memcpy_async(row_out.ptr, out.ptr, self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)

    def _verify_chain_row_spans(self, row: int) -> tuple[Tensor, KVLiveSpans, KVLiveSpans]:
        """Return c=1 full-attention spans for one verifier row.

        The row positions/context counts are already materialized by
        ``_write_verify_chain_metadata``.  Using row views avoids launching
        ``set_decode_position_i64`` for every verifier row in every full-attention
        layer; the committed resident slot position is restored once after the
        verifier accept summary chooses the row.
        """

        position_tensor = Tensor.from_handle(self.prefill_position_buf.ptr + int(row) * DType.INT64.itemsize, (1,), DType.INT64, self.device)
        context_tensor = Tensor.from_handle(self.prefill_context_count_buf.ptr + int(row) * DType.INT64.itemsize, (1,), DType.INT64, self.device)
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

    def _write_verify_chain_metadata(self, batch: TargetVerifyBatch, *, base_slot: int, stream: int = 0) -> None:
        rows = int(batch.rows)
        token_i64 = np.asarray(batch.tokens, dtype=np.int64)
        token_i32 = np.asarray(batch.tokens, dtype=np.int32)
        position_i64 = np.asarray(batch.positions, dtype=np.int64)
        position_i32 = np.asarray(batch.positions, dtype=np.int32)
        context_i64 = np.asarray([int(position) + 1 for position in batch.positions], dtype=np.int64)
        parent_i32 = np.asarray(batch.parent_rows, dtype=np.int32)
        parent_i64 = np.asarray(batch.parent_rows, dtype=np.int64)
        depth_i32 = np.asarray(batch.draft_depths, dtype=np.int32)
        row_req_i32 = np.asarray(batch.row_to_request, dtype=np.int32)
        active_u8 = np.asarray(batch.active_mask, dtype=np.uint8)
        physical_blocks = np.arange(base_slot * self.blocks, (base_slot + 1) * self.blocks, dtype=np.int32)
        block_table = np.tile(physical_blocks, (rows, 1))
        copies = (
            (self.verify_token_ids_i64, token_i64),
            (self.verify_token_ids_i32, token_i32),
            (self.prefill_position_buf, position_i64),
            (self.verify_positions_i32, position_i32),
            (self.prefill_context_count_buf, context_i64),
            (self.verify_parent_rows_i32, parent_i32),
            (self.verify_parent_rows_i64, parent_i64),
            (self.verify_draft_depths_i32, depth_i32),
            (self.verify_row_to_request_i32, row_req_i32),
            (self.verify_active_mask_u8, active_u8),
            (self.prefill_block_table_buf, block_table),
        )
        for buffer, array in copies:
            contiguous = np.ascontiguousarray(array)
            copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes, runtime=self.runtime)
        _ = stream

    def _sample_verify_rows_from_hidden(self, hidden: Tensor, rows: int, *, stream: int = 0) -> None:
        norm_out = Tensor.from_handle(self.batch_norm_out.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        norm_out_bf16 = Tensor.from_handle(self.batch_norm_out_bf16.ptr, (rows, self.config.hidden_size), DType.BF16, self.device)
        paro_rmsnorm_out_fp16(
            hidden.ptr,
            self.norm_weight.tensor.ptr,
            norm_out.ptr,
            rows,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            stream=stream,
            library=self.libraries["norm"],
            runtime=self.runtime,
        )
        fp16_to_bf16(
            norm_out.ptr,
            norm_out_bf16.ptr,
            rows * self.config.hidden_size,
            stream=stream,
            library=self.libraries["cast"],
            runtime=self.runtime,
        )
        w8a16_linear_bf16_f32_out(
            norm_out_bf16.ptr,
            self.lm_head_weight.tensor.ptr,
            self.lm_head_scale.tensor.ptr,
            self.verify_lm_logits.ptr,
            rows,
            self.config.hidden_size,
            self.vocab_size,
            threads=self.lm_head_threads,
            stream=stream,
            library=self.libraries["w8a16"],
            runtime=self.runtime,
        )
        argmax_f32_rows_i32(
            self.verify_lm_logits.ptr,
            self.verify_lm_block_values.ptr,
            self.verify_lm_block_indices.ptr,
            self.verify_top1_i32.ptr,
            self.verify_top1_values.ptr,
            rows,
            self.vocab_size,
            threads=self.lm_head_threads,
            stream=stream,
            library=self.libraries["lm_head"],
            runtime=self.runtime,
        )

    def _read_verify_top1(self, rows: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
        ids = np.empty((rows,), dtype=np.int32)
        values = np.empty((rows,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(ids), DeviceBuffer(self.verify_top1_i32.ptr, ids.nbytes), runtime=self.runtime)
        copy_device_to_host(host_array_ptr(values), DeviceBuffer(self.verify_top1_values.ptr, values.nbytes), runtime=self.runtime)
        return tuple(int(item) for item in ids.tolist()), tuple(float(item) for item in values.tolist())

    def _launch_verify_accept_summary(self, batch: TargetVerifyBatch, *, rows: int, stream: int = 0) -> None:
        request_count = len(batch.request_ids)
        dflash_accept_chain_i32(
            self.verify_token_ids_i32.ptr,
            self.verify_positions_i32.ptr,
            self.verify_parent_rows_i32.ptr,
            self.verify_draft_depths_i32.ptr,
            self.verify_active_mask_u8.ptr,
            self.verify_top1_i32.ptr,
            None,
            self.verify_accepted_counts.ptr,
            self.verify_commit_rows.ptr,
            self.verify_commit_tokens.ptr,
            self.verify_commit_positions.ptr,
            self.verify_next_tokens.ptr,
            self.verify_full_accept.ptr,
            self.verify_committed_output_ids.ptr,
            self.verify_committed_output_lengths.ptr,
            rows,
            request_count,
            rows,
            stream=stream,
            library=self.libraries["dflash_accept"],
            runtime=self.runtime,
        )

    def _read_verify_accept_payload(self, request_count: int, *, stream: int = 0) -> dict[str, tuple[int, ...] | tuple[bool, ...]]:
        self.runtime.stream_synchronize(stream)
        accepted = np.empty((request_count,), dtype=np.int32)
        commit_rows = np.empty((request_count,), dtype=np.int32)
        commit_tokens = np.empty((request_count,), dtype=np.int32)
        commit_positions = np.empty((request_count,), dtype=np.int32)
        next_tokens = np.empty((request_count,), dtype=np.int32)
        full_accept = np.empty((request_count,), dtype=np.uint8)
        out_lengths = np.empty((request_count,), dtype=np.int32)
        for host, buffer in (
            (accepted, self.verify_accepted_counts),
            (commit_rows, self.verify_commit_rows),
            (commit_tokens, self.verify_commit_tokens),
            (commit_positions, self.verify_commit_positions),
            (next_tokens, self.verify_next_tokens),
            (full_accept, self.verify_full_accept),
            (out_lengths, self.verify_committed_output_lengths),
        ):
            copy_device_to_host(host_array_ptr(host), DeviceBuffer(buffer.ptr, host.nbytes), runtime=self.runtime)
        return {
            "accepted_counts": tuple(int(x) for x in accepted.tolist()),
            "commit_rows": tuple(int(x) for x in commit_rows.tolist()),
            "commit_tokens": tuple(int(x) for x in commit_tokens.tolist()),
            "commit_positions": tuple(int(x) for x in commit_positions.tolist()),
            "next_tokens": tuple(int(x) for x in next_tokens.tolist()),
            "full_accept": tuple(bool(x) for x in full_accept.tolist()),
            "committed_output_lengths": tuple(int(x) for x in out_lengths.tolist()),
        }

    def _run_verify_accept_summary(self, batch: TargetVerifyBatch, *, rows: int, stream: int = 0) -> dict[str, tuple[int, ...] | tuple[bool, ...]]:
        self._launch_verify_accept_summary(batch, rows=rows, stream=stream)
        return self._read_verify_accept_payload(len(batch.request_ids), stream=stream)

    @staticmethod
    def _gpu_accept_payload_matches(payload: dict[str, tuple[int, ...] | tuple[bool, ...]], summary: TargetAcceptSummary) -> bool:
        expected_next = tuple(-1 if token is None else int(token) for token in (summary.next_tokens or ()))
        return (
            payload["accepted_counts"] == tuple(int(x) for x in summary.accepted_counts)
            and payload["commit_rows"] == tuple(int(x) for x in summary.commit_rows)
            and payload["commit_tokens"] == tuple(int(x) for x in summary.commit_tokens)
            and payload["commit_positions"] == tuple(int(x) for x in summary.commit_positions)
            and payload["next_tokens"] == expected_next
            and payload["full_accept"] == tuple(bool(x) for x in summary.full_accept)
        )

    def _commit_bulk_linear_states(self, selected_row: int, *, base_slot: int, stream: int = 0) -> None:
        for layer_id, scratch in self.linear_scratch.items():
            conv_state, recurrent_state = self._slot_linear_state(layer_id, base_slot)
            conv_row_nbytes = int(np.prod(conv_state.shape)) * conv_state.dtype.itemsize
            recurrent_row_nbytes = int(np.prod(recurrent_state.shape)) * recurrent_state.dtype.itemsize
            self.runtime.memcpy_async(
                conv_state.ptr,
                scratch.tree_conv_state.ptr + int(selected_row) * conv_row_nbytes,
                conv_row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
            self.runtime.memcpy_async(
                recurrent_state.ptr,
                scratch.tree_recurrent_state.ptr + int(selected_row) * recurrent_row_nbytes,
                recurrent_row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )

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
            linear_scratch = self._ensure_linear_prefill_scratch(tokens=tokens)
            if tokens > 1:
                moe_scratch = self._ensure_moe_prefill_scratch(layer_id, tokens=tokens)
            else:
                moe_scratch = self.moe_scratch[layer_id]
                if moe_scratch.normed.shape[0] < tokens:
                    moe_scratch = self._reserve_mlp_scratch(state, tokens=tokens)
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
                    num_splits = max(1, (position + 1 + self.decode_chunk_size - 1) // self.decode_chunk_size)
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
                        chunk_size=self.decode_chunk_size,
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
        self._release_prefill_workspace()
        for layer_id, state in enumerate(self.states):
            self.moe_scratch[layer_id] = self._reserve_mlp_scratch(state, tokens=1)
            if self.config.layer_types[layer_id] == "linear_attention":
                self.linear_scratch[layer_id] = state.reserve_linear_attention_scratch(tokens=1, activation_dtype=DType.FP16)
            elif self.config.layer_types[layer_id] == "full_attention":
                self.full_scratch[layer_id] = state.reserve_full_attention_scratch(
                    tokens=1,
                    num_splits=self.max_splits,
                    activation_dtype=DType.FP16,
                    gated_dtype=DType.FP16,
                )

    def _run_layers(
        self,
        *,
        position: int,
        num_splits_override: int | None = None,
        slot: int = 0,
        persist_aliases: bool = True,
        stream: int = 0,
        capture_layer_ids: Sequence[int] | None = None,
        capture_hidden_concat: Tensor | None = None,
        capture_row: int = 0,
    ) -> Tensor:
        if slot == 0 and persist_aliases:
            hidden = self.hidden
            next_hidden = self.next_hidden
        else:
            hidden = self._slot_hidden_view(self.batch_hidden, slot)
            next_hidden = self._slot_hidden_view(self.batch_next_hidden, slot)
        position_tensor, append_spans, decode_spans = self._slot_spans(slot)
        capture_ids = tuple(int(x) for x in (capture_layer_ids or ()))
        capture_offsets = {layer_id: idx for idx, layer_id in enumerate(capture_ids)}
        if capture_hidden_concat is not None:
            if capture_hidden_concat.dtype != DType.BF16:
                raise ValueError("capture_hidden_concat must use BF16 storage")
            if capture_hidden_concat.ndim != 2:
                raise ValueError("capture_hidden_concat must be rank-2")
            if capture_hidden_concat.shape[1] != len(capture_ids) * self.config.hidden_size:
                raise ValueError("capture_hidden_concat width must equal captured layers * hidden_size")
            if capture_row < 0 or capture_row >= capture_hidden_concat.shape[0]:
                raise ValueError("capture_row outside capture_hidden_concat")
        elif capture_ids:
            raise ValueError("capture_hidden_concat is required when capture_layer_ids is set")
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
                num_splits = num_splits_override or max(1, (position + 1 + self.decode_chunk_size - 1) // self.decode_chunk_size)
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
                    chunk_size=self.decode_chunk_size,
                    num_splits=num_splits,
                    library=self.libraries,
                    stream=stream,
                )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
            self.runtime.memcpy_async(next_hidden.ptr, out.ptr, self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
            hidden, next_hidden = next_hidden, hidden
            capture_offset = capture_offsets.get(layer_id)
            if capture_offset is not None and capture_hidden_concat is not None:
                dst = capture_hidden_concat.ptr + (
                    int(capture_row) * int(capture_hidden_concat.shape[1]) + capture_offset * self.config.hidden_size
                ) * DType.BF16.itemsize
                fp16_to_bf16(
                    hidden.ptr,
                    dst,
                    self.config.hidden_size,
                    stream=stream,
                    library=self.libraries["cast"],
                    runtime=self.runtime,
                )
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
        from hipengine.kernels.hip_gfx1100.moe.group_scatter import build_qwen35_moe_group_scatter
        from hipengine.kernels.hip_gfx1100.moe.router import build_qwen35_router
        from hipengine.kernels.hip_gfx1100.norm import build_qwen35_rmsnorm
        from hipengine.kernels.hip_gfx1100.runtime import build_runtime_state
        from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv
        from hipengine.kernels.hip_gfx1100.quant.paro_marlin_k import build_paro_marlin_k
        from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear
        from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import build_paro_rotate
        from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import build_qwen35_rotary
        from hipengine.kernels.hip_gfx1100.wmma import build_paro_awq_wmma

        with hip_target_arch_environment(self.target_arch):
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
                "dflash_accept": build_dflash_accept(**build_kwargs),
                "group_scatter": build_qwen35_moe_group_scatter(**build_kwargs),
                "kv": build_qwen35_paged_kv_write(**build_kwargs),
                "linear_conv": build_qwen35_linear_attn_conv(**build_kwargs),
                "linear_gdn": build_qwen35_linear_attn_gdn(**build_kwargs),
                "lm_head": build_lm_head(**build_kwargs),
                "marlin_k": build_paro_marlin_k(**build_kwargs),
                "norm": build_qwen35_rmsnorm(**build_kwargs),
                "qwen_rotary": build_qwen35_rotary(**build_kwargs),
                "router": build_qwen35_router(**build_kwargs),
                "rotate": build_paro_rotate(**build_kwargs),
                "runtime_state": build_runtime_state(**build_kwargs),
                "silu": build_paro_silu(**build_kwargs),
                "w8a16": build_w8a16_linear(**build_kwargs),
                "wmma": build_paro_awq_wmma(**build_kwargs),
            }
            if self.prefill_config.attn_aotriton_min_tokens > 0:
                self.libraries["aotriton"] = build_aotriton_wrap(**build_kwargs)
        self._emit(
            "load_kernel_libraries_done",
            count=len(self.libraries),
            backend=self.backend,
            target_arch=self.target_arch,
        )

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
        self.prefill_capacity_rows = self.max_sequence_length * self.max_batch_size
        self.prefill_hidden_nbytes = self.prefill_capacity_rows * self.hidden_nbytes
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
        head_key = "lm_head.weight" if "lm_head.weight" in self.runner.normalized_infos else "language_model.embed_tokens.weight"
        self._emit("load_lm_head_start", mode="w8a16", source=head_key)
        head = _read_tensor(self.runner.normalized_infos, head_key)
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
            (self.prefill_capacity_rows, self.config.hidden_size),
            DType.FP16,
            self.device,
        )
        self.prefill_next_hidden = Tensor.from_handle(
            prefill_next_hidden_buf.ptr,
            (self.prefill_capacity_rows, self.config.hidden_size),
            DType.FP16,
            self.device,
        )

        block_table_arr = np.arange(self.blocks, dtype=np.int32)
        prefill_block_table_arr = np.tile(block_table_arr, (self.prefill_capacity_rows, 1))
        prefill_context_count_arr = np.zeros((self.prefill_capacity_rows,), dtype=np.int64)
        self.position_arr = np.zeros(self.batch_layout.slot_scalar_shape, dtype=np.int64)
        self.context_arr = np.ones(self.batch_layout.slot_scalar_shape, dtype=np.int64)
        self.token_id_arr = np.zeros(self.batch_layout.slot_scalar_shape, dtype=np.int64)
        self.active_mask_arr = np.zeros(self.batch_layout.slot_scalar_shape, dtype=np.uint8)
        self.active_mask_arr[0] = 1
        self.block_table_buf = self._dev(block_table_arr)
        self.prefill_block_table_buf = self._dev(prefill_block_table_arr)
        self.prefill_context_count_buf = self._dev(prefill_context_count_arr)
        self.position_buf = self._dev(self.position_arr)
        self.context_buf = self._dev(self.context_arr)
        self.token_id_buf = self._dev(self.token_id_arr)
        self.active_mask_buf = self._dev(self.active_mask_arr)
        self.block_table = Tensor.from_handle(self.block_table_buf.ptr, block_table_arr.shape, DType.INT32, self.device)
        self.batch_positions = Tensor.from_handle(self.position_buf.ptr, self.batch_layout.slot_scalar_shape, DType.INT64, self.device)
        prefill_token_arr = np.zeros((self.prefill_capacity_rows,), dtype=np.int64)
        prefill_position_arr = np.arange(self.prefill_capacity_rows, dtype=np.int64)
        prefill_single_cu_arr = np.asarray([0, 0], dtype=np.int32)
        self.prefill_token_id_buf = self._dev(prefill_token_arr)
        self.prefill_position_buf = self._dev(prefill_position_arr)
        self.prefill_single_cu_buf = self._dev(prefill_single_cu_arr)
        self.prefill_single_cu_k_buf = self._dev(prefill_single_cu_arr)
        self.prefill_token_ids = Tensor.from_handle(
            self.prefill_token_id_buf.ptr,
            prefill_token_arr.shape,
            DType.INT64,
            self.device,
        )
        self.prefill_positions = Tensor.from_handle(
            self.prefill_position_buf.ptr,
            prefill_position_arr.shape,
            DType.INT64,
            self.device,
        )
        self.prefill_single_cu = Tensor.from_handle(
            self.prefill_single_cu_buf.ptr,
            prefill_single_cu_arr.shape,
            DType.INT32,
            self.device,
        )
        self.prefill_single_cu_k = Tensor.from_handle(
            self.prefill_single_cu_k_buf.ptr,
            prefill_single_cu_arr.shape,
            DType.INT32,
            self.device,
        )
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

        # Fixed-capacity buffers for the native root+candidate verifier path.
        # They are sized by max_batch_size because one DFlash chain bucket is
        # root + candidate rows for a single request.
        verify_rows = self.max_batch_size
        self.verify_token_ids_i64 = malloc(verify_rows * DType.INT64.itemsize, runtime=self.runtime)
        self.verify_token_ids_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_positions_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_parent_rows_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_parent_rows_i64 = malloc(verify_rows * DType.INT64.itemsize, runtime=self.runtime)
        self.verify_draft_depths_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_row_to_request_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_active_mask_u8 = malloc(verify_rows * DType.BOOL.itemsize, runtime=self.runtime)
        self.verify_lm_logits = malloc(verify_rows * self.vocab_size * DType.FP32.itemsize, runtime=self.runtime)
        self.verify_lm_block_values = malloc(verify_rows * self.lm_head_stage1_blocks * DType.FP32.itemsize, runtime=self.runtime)
        self.verify_lm_block_indices = malloc(verify_rows * self.lm_head_stage1_blocks * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_top1_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_top1_values = malloc(verify_rows * DType.FP32.itemsize, runtime=self.runtime)
        self.verify_accepted_counts = malloc(self.max_batch_size * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_commit_rows = malloc(self.max_batch_size * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_commit_tokens = malloc(self.max_batch_size * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_commit_positions = malloc(self.max_batch_size * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_next_tokens = malloc(self.max_batch_size * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_full_accept = malloc(self.max_batch_size * DType.BOOL.itemsize, runtime=self.runtime)
        self.verify_committed_output_ids = malloc(self.max_batch_size * verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_committed_output_lengths = malloc(self.max_batch_size * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_capture_hidden_concat = malloc(
            verify_rows * len(self.config.layer_types) * self.config.hidden_size * DType.BF16.itemsize,
            runtime=self.runtime,
        )
        self._verify_graph_cache: dict[tuple[int, int, int], Qwen35ParoVerifierGraphEntry] = {}
        self.buffers.extend(
            (
                self.verify_token_ids_i64,
                self.verify_token_ids_i32,
                self.verify_positions_i32,
                self.verify_parent_rows_i32,
                self.verify_parent_rows_i64,
                self.verify_draft_depths_i32,
                self.verify_row_to_request_i32,
                self.verify_active_mask_u8,
                self.verify_lm_logits,
                self.verify_lm_block_values,
                self.verify_lm_block_indices,
                self.verify_top1_i32,
                self.verify_top1_values,
                self.verify_accepted_counts,
                self.verify_commit_rows,
                self.verify_commit_tokens,
                self.verify_commit_positions,
                self.verify_next_tokens,
                self.verify_full_accept,
                self.verify_committed_output_ids,
                self.verify_committed_output_lengths,
                self.verify_capture_hidden_concat,
            )
        )

    def _materialize_layers(self) -> None:
        self.states = self.runner._materialize_resident_states(self.layer_limit, emit=self._emit)
        qkv_width = (
            2 * self.config.linear_num_key_heads * self.config.linear_key_head_dim
            + self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            self.moe_scratch[layer_id] = self._reserve_mlp_scratch(state, tokens=1)
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
    max_replay_steps: int = 1
    generated: DeviceBuffer | None = None
    generated_index: DeviceBuffer | None = None
    record_steps: int = 0
    closed: bool = False

    def replay(self, steps: int) -> None:
        if self.closed:
            raise RuntimeError("decode graph is closed")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if self.steps_per_replay <= 0:
            raise ValueError("steps_per_replay must be positive")
        if steps > self.max_replay_steps:
            raise ValueError("steps exceed captured max_replay_steps")
        if self.record_steps and steps > self.record_steps:
            raise ValueError("steps exceed decode graph record capacity")
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

    def read_generated_token_ids(self, count: int | None = None) -> list[int]:
        if self.closed:
            raise RuntimeError("decode graph is closed")
        if self.generated is None:
            raise RuntimeError("decode graph was captured without generated-token recording")
        rows = int(self.record_steps if count is None else count)
        if rows < 0 or rows > self.record_steps:
            raise ValueError("count outside decode graph record capacity")
        host = np.empty((rows,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(self.generated.ptr, rows * DType.INT64.itemsize),
            runtime=self.session.runtime,
        )
        return [int(item) for item in host.tolist()]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.session.runtime.graph_exec_destroy(self.graph_exec)
        self.session.runtime.graph_destroy(self.graph)
        if self.stream:
            self.session.runtime.stream_destroy(self.stream)
        if self.generated_index is not None:
            free(self.generated_index, runtime=self.session.runtime)
            self.generated_index = None
        if self.generated is not None:
            free(self.generated, runtime=self.session.runtime)
            self.generated = None

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
    head_key = "lm_head.weight" if "lm_head.weight" in normalized else "language_model.embed_tokens.weight"
    info = normalized[normalize_qwen35_weight_name(head_key)]
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
