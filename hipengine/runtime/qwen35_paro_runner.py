"""Bring-up runner for real Qwen3.5/PARO one-token decode smokes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
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
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16,
)
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import build_aotriton_wrap
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
    record_i64_scalar_indexed,
    set_decode_position_i64,
    set_decode_positions_i64,
    set_i64_scalar,
    set_i64_vector,
)
from hipengine.dispatch import (
    ActiveBatch,
    BatchSamplerMode,
    ProjectionKernelSelection,
    RequestState,
    plan_batch_sampler_dispatch,
    plan_projection_dispatch,
)
from hipengine.kvcache import FixedPagedKVPolicy, KVLiveSpans, KVScaleMetadata
from hipengine.kvcache.policy import KV_SCALE_GRANULARITY_CHOICES
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
from hipengine.speculative import DraftBatch, TargetCommitPlan, TargetStateCommitBuffers, TargetVerifyBatch, TargetVerifyBuffers


_PREFILL_OVERLAP_MIN_TOKENS = 32768
_LOGGER = logging.getLogger(__name__)


def _env_int(name: str, default: int, *aliases: str) -> int:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return int(value)
    return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() not in {"0", "false", "off", "no"}


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
class Qwen35ParoKVCapacityEstimate:
    """Fast retained-KV capacity estimate from current free HIP memory."""

    requested_context_tokens: int
    requested_context_tokens_rounded: int
    model_max_context_tokens: int
    allocatable_context_tokens: int
    available_bytes: int
    reserve_bytes: int
    usable_bytes: int
    bytes_per_token: int
    requested_kv_bytes: int
    requested_context_overhead_bytes: int
    requested_total_bytes: int
    model_max_kv_bytes: int
    model_max_context_overhead_bytes: int
    model_max_total_bytes: int
    full_attention_layers: int
    kv_storage_dtype: str
    kv_scale_dtype: str | None
    block_size: int
    max_batch_size: int

    @property
    def fits_requested(self) -> bool:
        return self.bytes_per_token == 0 or self.requested_total_bytes <= self.usable_bytes

    @property
    def fits_model_max(self) -> bool:
        return self.model_max_context_tokens <= 0 or self.bytes_per_token == 0 or self.model_max_total_bytes <= self.usable_bytes

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "requested_context_tokens": self.requested_context_tokens,
            "requested_context_tokens_rounded": self.requested_context_tokens_rounded,
            "model_max_context_tokens": self.model_max_context_tokens,
            "allocatable_context_tokens": self.allocatable_context_tokens,
            "available_bytes": self.available_bytes,
            "reserve_bytes": self.reserve_bytes,
            "usable_bytes": self.usable_bytes,
            "bytes_per_token": self.bytes_per_token,
            "requested_kv_bytes": self.requested_kv_bytes,
            "requested_context_overhead_bytes": self.requested_context_overhead_bytes,
            "requested_total_bytes": self.requested_total_bytes,
            "model_max_kv_bytes": self.model_max_kv_bytes,
            "model_max_context_overhead_bytes": self.model_max_context_overhead_bytes,
            "model_max_total_bytes": self.model_max_total_bytes,
            "full_attention_layers": self.full_attention_layers,
            "kv_storage_dtype": self.kv_storage_dtype,
            "kv_scale_dtype": self.kv_scale_dtype,
            "block_size": self.block_size,
            "max_batch_size": self.max_batch_size,
            "fits_requested": self.fits_requested,
            "fits_model_max": self.fits_model_max,
        }


def estimate_qwen35_paro_kv_capacity(
    config: Any,
    *,
    available_bytes: int,
    requested_context_tokens: int,
    storage_dtype: str | DType,
    scale_dtype: str | DType = DType.FP16,
    block_size: int = 256,
    chunk_size: int = 256,
    reserve_bytes: int = 0,
    max_batch_size: int = 1,
) -> Qwen35ParoKVCapacityEstimate:
    """Estimate the largest retained full-attention KV arena that can fit.

    This is deliberately cheap: it uses model metadata and ``hipMemGetInfo``'s
    current free-memory value after resident weights load.  It covers the retained
    full-attention KV payload, INT8 scale metadata, and persistent
    context-dependent metadata such as the prefill block table.  Transient
    prefill workspaces and allocator fragmentation are represented by
    ``reserve_bytes``.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    requested = int(requested_context_tokens)
    if requested <= 0:
        raise ValueError("requested_context_tokens must be positive")
    available = max(0, int(available_bytes))
    reserve = max(0, int(reserve_bytes))
    usable = max(0, available - reserve)
    max_batch = max(1, int(max_batch_size))
    block = int(block_size)
    chunk = int(chunk_size)
    bytes_per_token = qwen35_paro_kv_bytes_per_token(
        config,
        storage_dtype=storage_dtype,
        scale_dtype=scale_dtype,
        max_batch_size=max_batch,
    )
    requested_rounded = _round_up_to_block(requested, block)
    requested_kv_bytes = requested_rounded * bytes_per_token
    requested_overhead = _qwen35_paro_context_overhead_bytes(
        requested,
        block_size=block,
        chunk_size=chunk,
        max_batch_size=max_batch,
    )
    model_max = int(getattr(config, "max_position_embeddings", 0) or 0)
    model_rounded = _round_up_to_block(model_max, block) if model_max > 0 else 0
    model_overhead = (
        _qwen35_paro_context_overhead_bytes(
            model_max,
            block_size=block,
            chunk_size=chunk,
            max_batch_size=max_batch,
        )
        if model_max > 0
        else 0
    )
    if bytes_per_token > 0:
        allocatable = _qwen35_paro_allocatable_context_tokens(
            usable,
            bytes_per_token=bytes_per_token,
            block_size=block,
            chunk_size=chunk,
            max_batch_size=max_batch,
        )
    else:
        allocatable = requested_rounded
    storage = DType.parse(storage_dtype)
    scale = DType.parse(scale_dtype) if storage == DType.INT8_PER_TOKEN_HEAD else None
    return Qwen35ParoKVCapacityEstimate(
        requested_context_tokens=requested,
        requested_context_tokens_rounded=requested_rounded,
        model_max_context_tokens=model_max,
        allocatable_context_tokens=int(allocatable),
        available_bytes=available,
        reserve_bytes=reserve,
        usable_bytes=usable,
        bytes_per_token=bytes_per_token,
        requested_kv_bytes=requested_kv_bytes,
        requested_context_overhead_bytes=requested_overhead,
        requested_total_bytes=requested_kv_bytes + requested_overhead,
        model_max_kv_bytes=model_rounded * bytes_per_token,
        model_max_context_overhead_bytes=model_overhead,
        model_max_total_bytes=model_rounded * bytes_per_token + model_overhead,
        full_attention_layers=_qwen35_paro_full_attention_layers(config),
        kv_storage_dtype=storage.value,
        kv_scale_dtype=None if scale is None else scale.value,
        block_size=block,
        max_batch_size=max_batch,
    )


def _qwen35_paro_allocatable_context_tokens(
    usable_bytes: int,
    *,
    bytes_per_token: int,
    block_size: int,
    chunk_size: int,
    max_batch_size: int,
) -> int:
    if usable_bytes <= 0 or bytes_per_token <= 0:
        return 0
    block = int(block_size)
    high = (int(usable_bytes) // int(bytes_per_token)) // block * block
    low = 0
    while low < high:
        mid_blocks = (low // block + high // block + 1) // 2
        mid = mid_blocks * block
        total = _round_up_to_block(mid, block) * int(bytes_per_token) + _qwen35_paro_context_overhead_bytes(
            mid,
            block_size=block,
            chunk_size=chunk_size,
            max_batch_size=max_batch_size,
        )
        if total <= usable_bytes:
            low = mid
        else:
            high = (mid_blocks - 1) * block
    return int(low)


def _qwen35_paro_context_overhead_bytes(
    context_tokens: int,
    *,
    block_size: int,
    chunk_size: int,
    max_batch_size: int,
) -> int:
    tokens = int(context_tokens)
    if tokens <= 0:
        return 0
    max_batch = max(1, int(max_batch_size))
    decode_chunk_size, max_splits = _paged_attn_decode_split_config(
        tokens,
        block_size=int(block_size),
        chunk_size=int(chunk_size),
    )
    decode_context_capacity = int(decode_chunk_size) * int(max_splits)
    blocks = (max(tokens, decode_context_capacity) + int(block_size) - 1) // int(block_size)
    prefill_rows = tokens * max_batch
    block_table_bytes = blocks * np.dtype(np.int32).itemsize
    prefill_block_table_bytes = prefill_rows * blocks * np.dtype(np.int32).itemsize
    prefill_token_bytes = prefill_rows * np.dtype(np.int64).itemsize
    prefill_position_bytes = prefill_rows * np.dtype(np.int64).itemsize
    prefill_context_count_bytes = prefill_rows * np.dtype(np.int64).itemsize
    return int(
        block_table_bytes
        + prefill_block_table_bytes
        + prefill_token_bytes
        + prefill_position_bytes
        + prefill_context_count_bytes
    )


def qwen35_paro_kv_bytes_per_token(
    config: Any,
    *,
    storage_dtype: str | DType,
    scale_dtype: str | DType = DType.FP16,
    max_batch_size: int = 1,
) -> int:
    storage = DType.parse(storage_dtype)
    if storage not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
        raise ValueError("Qwen3.5/PARO KV storage must be bf16 or int8_per_token_head")
    full_layers = _qwen35_paro_full_attention_layers(config)
    if full_layers <= 0:
        return 0
    kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError("Qwen3.5/PARO KV estimate requires num_key_value_heads and head_dim")
    batch = max(1, int(max_batch_size))
    payload = batch * full_layers * 2 * kv_heads * head_dim * storage.itemsize
    if storage != DType.INT8_PER_TOKEN_HEAD:
        return payload
    scale = DType.parse(scale_dtype)
    if scale not in {DType.FP16, DType.FP32}:
        raise ValueError("INT8 KV scale dtype must be fp16 or fp32")
    return payload + batch * full_layers * 2 * kv_heads * scale.itemsize


def _qwen35_paro_full_attention_layers(config: Any) -> int:
    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    if layer_types:
        return sum(1 for item in layer_types if str(item) == "full_attention")
    return int(getattr(config, "num_hidden_layers", 0) or 0)


def _round_up_to_block(tokens: int, block_size: int) -> int:
    value = int(tokens)
    block = int(block_size)
    if value <= 0:
        return 0
    return ((value + block - 1) // block) * block


def _format_bytes_gib(value: int) -> str:
    return f"{int(value) / 1024**3:.2f} GiB"


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

    @property
    def full_kv_scale_shape(self) -> tuple[int, int, int, int]:
        return (self.max_batch_size, self.blocks, self.block_size, self.num_key_value_heads)

    @property
    def flat_full_kv_scale_shape(self) -> tuple[int, int, int]:
        return (self.max_batch_size * self.blocks, self.block_size, self.num_key_value_heads)

    @property
    def slot0_full_kv_scale_shape(self) -> tuple[int, int, int]:
        return (self.blocks, self.block_size, self.num_key_value_heads)


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
    decode_execution: dict[str, Any] | None = None
    projection_dispatch: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "scheduler_owned": self.scheduler_owned,
            "row_execution": self.row_execution,
            "native_prefill_plan": self.native_prefill_plan.to_json_dict(),
            "native_compact_prefill": self.native_compact_prefill,
            "native_caware_decode": self.native_caware_decode,
            "throughput_claim_eligible": self.throughput_claim_eligible,
            "blockers": list(self.blockers),
        }
        if self.decode_execution is not None:
            payload["decode_execution"] = self.decode_execution
        if self.projection_dispatch is not None:
            payload["projection_dispatch"] = self.projection_dispatch
        return payload


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
        kv_policy: FixedPagedKVPolicy | None = None,
        kv_scale_dtype: str | DType = DType.FP16,
        kv_scale_granularity: str = "per_token_head",
        auto_context_length: bool = False,
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
        self.kv_policy = kv_policy or FixedPagedKVPolicy(block_size=self.block_size, storage_dtype=DType.BF16)
        policy_block_size = int(getattr(self.kv_policy, "block_size", self.block_size))
        if policy_block_size != self.block_size:
            raise ValueError("resident KV policy block_size must match session block_size")
        self.kv_storage_dtype = DType.parse(getattr(self.kv_policy, "storage_dtype", DType.BF16))
        if self.kv_storage_dtype not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
            raise ValueError("resident full-attention KV storage must be bf16 or int8_per_token_head")
        self.kv_scale_dtype = DType.parse(kv_scale_dtype)
        if self.kv_scale_dtype not in {DType.FP16, DType.FP32}:
            raise ValueError("resident INT8 KV scales must use fp16 or fp32")
        if kv_scale_granularity not in KV_SCALE_GRANULARITY_CHOICES:
            raise ValueError("resident INT8 KV scale granularity must be per_token_head")
        self.kv_scale_granularity = kv_scale_granularity
        self.auto_context_length = bool(auto_context_length)
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
        self.full_cache_scales: dict[int, tuple[Tensor, Tensor, DeviceBuffer, DeviceBuffer]] = {}
        self.full_cache_scale_metadata: dict[int, KVScaleMetadata] = {}
        self.linear_scratch = {}
        self.full_scratch = {}
        self.moe_scratch = {}
        self.prefill_workspace = RuntimeWorkspace(runtime=self.runtime)
        self.prefill_hidden_buffer: DeviceBuffer | None = None
        self.prefill_hidden_capacity_rows = 0
        self._prefill_scratch_state: Qwen35ParoDecodeState | None = None
        self.prefill_linear_scratch: Qwen35ParoLinearAttentionScratch | None = None
        self.prefill_full_scratch: Qwen35ParoAttentionScratch | None = None
        self.prefill_moe_scratch: Qwen35ParoGroupedMoeScratch | Qwen35ParoMoeScratch | None = None
        self.tokenizer = _load_tokenizer(self.model)
        self.closed = False
        try:
            self._build()
        except Exception:
            self.close()
            raise

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
        self._release_prefill_workspace()
        self._release_prefill_hidden_buffer()
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

    def reset(self) -> None:
        """Reset sequence state without freeing resident weights or scratch."""

        if self.closed:
            raise RuntimeError("session is closed")
        self.runtime.device_synchronize()
        self.position_arr.fill(0)
        self.context_arr.fill(1)
        self.token_id_arr.fill(0)
        self.active_mask_arr.fill(0)
        self.active_mask_arr[0] = 1
        copy_host_to_device(
            self.position_buf,
            host_array_ptr(self.position_arr),
            self.position_arr.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.context_buf,
            host_array_ptr(self.context_arr),
            self.context_arr.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.token_id_buf,
            host_array_ptr(self.token_id_arr),
            self.token_id_arr.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.active_mask_buf,
            host_array_ptr(self.active_mask_arr),
            self.active_mask_arr.nbytes,
            runtime=self.runtime,
        )
        for state_buffers in self.linear_states.values():
            _conv_state, _recurrent_state, conv_buf, recurrent_buf, _conv_zero, _recurrent_zero = state_buffers
            self.runtime.memset(conv_buf.ptr, 0, conv_buf.nbytes)
            self.runtime.memset(recurrent_buf.ptr, 0, recurrent_buf.nbytes)
        for cache_buffers in self.full_caches.values():
            _key_cache, _value_cache, key_buf, value_buf = cache_buffers
            self.runtime.memset(key_buf.ptr, 0, key_buf.nbytes)
            self.runtime.memset(value_buf.ptr, 0, value_buf.nbytes)
        for scale_buffers in self.full_cache_scales.values():
            _key_scale, _value_scale, key_scale_buf, value_scale_buf = scale_buffers
            self.runtime.memset(key_scale_buf.ptr, 0, key_scale_buf.nbytes)
            self.runtime.memset(value_scale_buf.ptr, 0, value_scale_buf.nbytes)
        self.last_prefill_execution = None

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

    def step_batch_native(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        positions: list[int] | tuple[int, ...],
        slots: list[int] | tuple[int, ...] | None = None,
        sample: bool = True,
    ) -> tuple[Qwen35ParoAutoregressiveStepResult | None, ...]:
        """Run one decode token per active row through native c-aware layer kernels.

        This retained bring-up path runs compact active rows while addressing
        retained KV/linear state through explicit physical slot ids.  Full-
        attention rows at split-K context lengths use the existing per-row
        split-K path until a true row-aware batch reducer lands.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        if not _env_flag("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"):
            raise NotImplementedError(
                "native c>N decode is experimental and currently blocked on generated-token equality; "
                "set HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1 for diagnostics"
            )
        if self.kv_storage_dtype != DType.BF16:
            raise NotImplementedError("native c>N decode currently requires BF16 KV")
        tokens = tuple(int(token) for token in token_ids)
        pos = tuple(int(position) for position in positions)
        if len(tokens) != len(pos):
            raise ValueError("token_ids and positions must have the same length")
        if not tokens:
            raise ValueError("token_ids must be non-empty")
        rows = len(tokens)
        if rows > self.max_batch_size:
            raise ValueError("token_ids exceed max_batch_size")
        slot_ids = tuple(range(rows)) if slots is None else tuple(int(slot) for slot in slots)
        if len(slot_ids) != rows:
            raise ValueError("slots must have the same length as token_ids")
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("native c>N decode slots must be unique")
        if any(slot < 0 or slot >= self.max_batch_size for slot in slot_ids):
            raise ValueError("native c>N decode slots must be within max_batch_size")
        if tuple(sorted(slot_ids)) != slot_ids:
            raise NotImplementedError("native c>N decode currently requires slots in physical-slot order")
        for position in pos:
            self._check_position(position)
        self._set_batch_token_embeddings(tokens, stream=0)
        self._set_batch_positions(pos, stream=0)
        hidden = self._run_layers_batch_decode(rows=rows, positions=pos, slots=slot_ids, stream=0)
        if not sample:
            self.runtime.device_synchronize()
            return tuple(None for _ in tokens)
        return self._sample_batch_from_hidden(hidden, rows=rows)

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
        if plan.transaction_id != buffers.transaction_id:
            raise ValueError("commit plan transaction_id must match state commit buffers")
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

    def batch_execution_metadata(
        self,
        *,
        scheduler_owned: bool = False,
        native_decode: bool = False,
        active_rows: int | None = None,
    ) -> Qwen35ParoResidentBatchExecution:
        """Describe whether the resident c>N path is native or a serial fallback."""

        native_prefill_plan = self.native_prefill_plan()
        blockers = list(native_prefill_plan.blockers)
        decode_execution = getattr(self, "last_batch_decode_execution", None) if native_decode else None
        projection_rows = active_rows
        if projection_rows is None and isinstance(decode_execution, dict):
            decode_rows = decode_execution.get("rows")
            if isinstance(decode_rows, int) and not isinstance(decode_rows, bool):
                projection_rows = decode_rows
        projection_dispatch = None
        if native_decode:
            blockers.extend(
                [
                    "native c>N decode currently supports compact physical-slot-ordered rows; "
                    "full-attention batch context is native only for BF16 KV and context < 1024",
                    "native c>N decode is experimental and blocked until generated-token equality passes",
                ]
            )
            path = "scheduler_native_compact_batch" if scheduler_owned else "native_compact_batch"
            full_attention_path = (
                decode_execution.get("full_attention_decode_path")
                if isinstance(decode_execution, dict)
                else None
            )
            if full_attention_path in {"per_row_splitk_fallback", "per_row_context_fallback"}:
                row_execution = "native_linear_batch_with_per_row_full_attention_fallback"
                native_caware_decode = False
                blockers.append("full-attention decode used a per-row fallback, so this is not native c-aware decode")
            else:
                row_execution = "native_compact_caware_layers"
                native_caware_decode = True
            if projection_rows is not None:
                projection_decision = plan_projection_dispatch(
                    rows=int(projection_rows),
                    row_gemv=ProjectionKernelSelection("linear", "w4_paro", "row_gemv"),
                )
                projection_dispatch = projection_decision.to_json_dict()
                if not projection_decision.throughput_claim_eligible:
                    blockers.extend(f"projection dispatch: {blocker}" for blocker in projection_decision.blockers)
            eligible = False
        else:
            blockers.extend(
                [
                    "step_batch_serial executes decode active physical slots serially through the c=1 layer path",
                    "native c-aware full-attention decode graph replay is not wired",
                ]
            )
            path = "scheduler_serial_slot_bridge" if scheduler_owned else "serial_slot_bridge"
            row_execution = "serial_c1_layer_path"
            native_caware_decode = False
            eligible = False
        return Qwen35ParoResidentBatchExecution(
            path=path,
            scheduler_owned=bool(scheduler_owned),
            row_execution=row_execution,
            native_prefill_plan=native_prefill_plan,
            native_compact_prefill=bool(native_prefill_plan.full_layer_limit_native),
            native_caware_decode=native_caware_decode,
            throughput_claim_eligible=eligible,
            blockers=tuple(dict.fromkeys(blockers)),
            decode_execution=decode_execution if isinstance(decode_execution, dict) else None,
            projection_dispatch=projection_dispatch,
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
        if getattr(self, "kv_storage_dtype", DType.BF16) == DType.INT8_PER_TOKEN_HEAD:
            raise NotImplementedError("compact c>N native prefill is not wired for int8_per_token_head retained KV")
        self._resolve_prefill_config_for_length(max(len(row) for row in slab.token_rows))
        native_prefill_plan = self.native_prefill_plan()
        if not native_prefill_plan.full_layer_limit_native:
            raise NotImplementedError(
                "native Qwen3.5/PARO packed prefill cannot cover this layer limit: "
                + "; ".join(native_prefill_plan.blockers)
            )
        metadata = self._materialize_packed_prefill_metadata(slab)
        minimize_prefill_workspace_overlap = self._should_minimize_prefill_workspace_overlap(slab.rows)
        try:
            if minimize_prefill_workspace_overlap:
                self._release_decode_scratch_for_prefill()
            prefill_hidden = self._prefill_hidden_view_for_rows(slab.rows)
            embedding_lookup_batch_fp16_i64(
                self.embedding.tensor.ptr,
                metadata.token_ids.ptr,
                prefill_hidden.ptr,
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
                "linear_attention_prefill_path": getattr(self, "_last_packed_prefill_linear_path", "packed_segments"),
                "full_attention_prefill_path": getattr(self, "_last_packed_prefill_full_attention_path", "packed_varlen"),
                "blockers": list(getattr(self, "_last_packed_prefill_blockers", [])),
                "decode_scratch_released_for_prefill": minimize_prefill_workspace_overlap,
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
        minimize_prefill_workspace_overlap = self._should_minimize_prefill_workspace_overlap(len(tokens))
        try:
            if minimize_prefill_workspace_overlap:
                self._release_decode_scratch_for_prefill()
            self._prepare_prefill_context_counts(len(tokens), stream=0)
            prefill_hidden = self._prefill_hidden_view_for_rows(len(tokens))
            embedding_lookup_batch_fp16_i64(
                self.embedding.tensor.ptr,
                token_buf.ptr,
                prefill_hidden.ptr,
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
                "aotriton_attention": self._prefill_use_aotriton_attention_resolved(len(tokens)),
                "attn_aotriton_min_tokens": self.prefill_config.attn_aotriton_min_tokens,
                "kv_storage_dtype": self.kv_storage_dtype.value,
                "kv_scale_dtype": self.kv_scale_dtype.value if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD else None,
                "kv_scale_granularity": self.kv_scale_granularity if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD else None,
                "int8_prefill_oracle": self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD,
                "decode_scratch_released_for_prefill": minimize_prefill_workspace_overlap,
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
        minimize_prefill_workspace_overlap = self._should_minimize_prefill_workspace_overlap(len(tokens))
        try:
            if minimize_prefill_workspace_overlap:
                self._release_decode_scratch_for_prefill()
            prefill_hidden = self._prefill_hidden_view_for_rows(len(tokens))
            embedding_lookup_batch_fp16_i64(
                self.embedding.tensor.ptr,
                token_buf.ptr,
                prefill_hidden.ptr,
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
                "decode_scratch_released_for_prefill": minimize_prefill_workspace_overlap,
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

    def _slot_full_scale_metadata(self, layer_id: int, slot: int) -> KVScaleMetadata | None:
        self._check_slot(slot)
        if self.kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
            return None
        k_scale, v_scale, k_buf, v_buf = self.full_cache_scales[layer_id]
        scale_nbytes = int(np.prod(k_scale.shape)) * k_scale.dtype.itemsize
        return KVScaleMetadata(
            k_scale=Tensor.from_handle(
                k_buf.ptr + int(slot) * scale_nbytes,
                k_scale.shape,
                k_scale.dtype,
                k_scale.device,
            ),
            v_scale=Tensor.from_handle(
                v_buf.ptr + int(slot) * scale_nbytes,
                v_scale.shape,
                v_scale.dtype,
                v_scale.device,
            ),
            scale_dtype=k_scale.dtype,
            granularity=self.kv_scale_granularity,
        )

    def _full_cache_scale_metadata_all_slots(self, layer_id: int) -> KVScaleMetadata | None:
        if self.kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
            return None
        k_scale, v_scale, k_buf, v_buf = self.full_cache_scales[layer_id]
        shape = self.batch_layout.flat_full_kv_scale_shape
        return KVScaleMetadata(
            k_scale=Tensor.from_handle(k_buf.ptr, shape, k_scale.dtype, k_scale.device),
            v_scale=Tensor.from_handle(v_buf.ptr, shape, v_scale.dtype, v_scale.device),
            scale_dtype=k_scale.dtype,
            granularity=self.kv_scale_granularity,
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

    def _slot_full_spans(self, layer_id: int, slot: int) -> tuple[Tensor, KVLiveSpans, KVLiveSpans]:
        position_tensor = self._slot_scalar_tensor(self.position_buf, slot, DType.INT64)
        context_tensor = self._slot_scalar_tensor(self.context_buf, slot, DType.INT64)
        scale_metadata = self._slot_full_scale_metadata(layer_id, slot)
        append_max_live_count = self.max_sequence_length - 1
        decode_max_live_count = self.max_sequence_length
        position_arr = getattr(self, "position_arr", None)
        context_arr = getattr(self, "context_arr", None)
        if position_arr is not None and context_arr is not None and int(slot) < len(position_arr):
            append_max_live_count = max(0, int(position_arr[int(slot)]))
            decode_max_live_count = max(1, int(context_arr[int(slot)]))
        append_spans = KVLiveSpans.paged_uniform(
            block_table=self.block_table,
            live_counts=position_tensor,
            max_live_count=append_max_live_count,
            storage_dtype=self.kv_storage_dtype,
            scale_metadata=scale_metadata,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=self.block_table,
            live_counts=context_tensor,
            max_live_count=decode_max_live_count,
            storage_dtype=self.kv_storage_dtype,
            scale_metadata=scale_metadata,
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
        storage_dtype: str | DType | None = None,
        scale_metadata: KVScaleMetadata | None = None,
    ) -> tuple[KVLiveSpans, KVLiveSpans]:
        total = rows if total_tokens is None else int(total_tokens)
        storage = getattr(self, "kv_storage_dtype", DType.BF16) if storage_dtype is None else DType.parse(storage_dtype)
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
            storage_dtype=storage,
            row_positions=positions,
            span_role="prefill",
            scale_metadata=scale_metadata,
        )
        prefill_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=context_counts,
            max_live_count=total,
            storage_dtype=storage,
            row_positions=positions,
            span_role="prefill",
            scale_metadata=scale_metadata,
        )
        return append_spans, prefill_spans

    def _prefill_int8_oracle_cache(self, layer_id: int, *, total_tokens: int) -> tuple[Tensor, Tensor]:
        """Return temporary BF16 K/V cache used only for INT8 native prefill attention."""

        blocks = max(1, (int(total_tokens) + self.block_size - 1) // self.block_size)
        shape = (blocks, self.block_size, self.config.num_key_value_heads, self.config.head_dim)
        # The BF16 oracle cache is needed only while processing the current
        # full-attention layer. Reuse the same workspace slots across layers so
        # long-context INT8 prefill does not retain one full BF16 shadow per
        # layer before _restore_decode_scratch_after_prefill() releases the
        # prefill workspace.
        _ = layer_id
        key = self.prefill_workspace.reserve_tensor("prefill.int8_oracle_key", shape, DType.BF16)
        value = self.prefill_workspace.reserve_tensor("prefill.int8_oracle_value", shape, DType.BF16)
        return key, value

    def _full_cache_all_slots(self, layer_id: int) -> tuple[Tensor, Tensor]:
        key_cache, value_cache, key_buf, value_buf = self.full_caches[layer_id]
        shape = (self.max_batch_size * self.blocks, self.block_size, self.config.num_key_value_heads, self.config.head_dim)
        return (
            Tensor.from_handle(key_buf.ptr, shape, key_cache.dtype, key_cache.device),
            Tensor.from_handle(value_buf.ptr, shape, value_cache.dtype, value_cache.device),
        )

    def owned_buffer_summary(self) -> dict[str, Any]:
        """Return a compact accounting of session-owned resident buffers."""

        full_layers: list[dict[str, Any]] = []
        payload_bytes = 0
        payload_elements = 0
        scale_bytes = 0
        scale_elements = 0
        for layer_id in sorted(self.full_caches):
            key_cache, value_cache, key_buf, value_buf = self.full_caches[layer_id]
            key_elements = int(key_buf.nbytes) // key_cache.dtype.itemsize
            value_elements = int(value_buf.nbytes) // value_cache.dtype.itemsize
            layer_payload_elements = key_elements + value_elements
            layer_payload_bytes = int(key_buf.nbytes) + int(value_buf.nbytes)
            payload_bytes += layer_payload_bytes
            payload_elements += layer_payload_elements
            entry: dict[str, Any] = {
                "layer_id": int(layer_id),
                "storage_dtype": self.kv_storage_dtype.value,
                "payload_dtype": key_cache.dtype.value,
                "key_shape": list(key_cache.shape),
                "value_shape": list(value_cache.shape),
                "key_full_shape": list(getattr(self.batch_layout, "full_kv_shape", key_cache.shape)),
                "value_full_shape": list(getattr(self.batch_layout, "full_kv_shape", value_cache.shape)),
                "key_elements": key_elements,
                "value_elements": value_elements,
                "payload_elements": layer_payload_elements,
                "key_buffer_bytes": int(key_buf.nbytes),
                "value_buffer_bytes": int(value_buf.nbytes),
                "payload_bytes": layer_payload_bytes,
                "payload_bytes_per_element": (layer_payload_bytes / layer_payload_elements) if layer_payload_elements else None,
                "scale_metadata": None,
            }
            scales = self.full_cache_scales.get(layer_id)
            if scales is not None:
                k_scale, v_scale, k_scale_buf, v_scale_buf = scales
                k_scale_elements = int(k_scale_buf.nbytes) // k_scale.dtype.itemsize
                v_scale_elements = int(v_scale_buf.nbytes) // v_scale.dtype.itemsize
                layer_scale_elements = k_scale_elements + v_scale_elements
                layer_scale_bytes = int(k_scale_buf.nbytes) + int(v_scale_buf.nbytes)
                scale_bytes += layer_scale_bytes
                scale_elements += layer_scale_elements
                metadata = self.full_cache_scale_metadata[layer_id]
                entry["scale_metadata"] = {
                    "granularity": metadata.granularity,
                    "scale_dtype": metadata.scale_dtype.value,
                    "k_scale_shape": list(k_scale.shape),
                    "v_scale_shape": list(v_scale.shape),
                    "k_scale_full_shape": list(getattr(self.batch_layout, "flat_full_kv_scale_shape", k_scale.shape)),
                    "v_scale_full_shape": list(getattr(self.batch_layout, "flat_full_kv_scale_shape", v_scale.shape)),
                    "k_scale_elements": k_scale_elements,
                    "v_scale_elements": v_scale_elements,
                    "scale_elements": layer_scale_elements,
                    "k_scale_buffer_bytes": int(k_scale_buf.nbytes),
                    "v_scale_buffer_bytes": int(v_scale_buf.nbytes),
                    "scale_bytes": layer_scale_bytes,
                    "scale_bytes_per_element": (layer_scale_bytes / layer_scale_elements) if layer_scale_elements else None,
                }
            full_layers.append(entry)
        buffer_bytes = sum(int(buffer.nbytes) for buffer in getattr(self, "buffers", ()))
        allocation_bytes = sum(int(allocation.buffer.nbytes) for allocation in getattr(self, "allocations", ()))
        return {
            "kv_storage_dtype": self.kv_storage_dtype.value,
            "kv_scale_dtype": self.kv_scale_dtype.value if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD else None,
            "kv_scale_granularity": self.kv_scale_granularity if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD else None,
            "full_attention_layer_count": len(full_layers),
            "full_attention_layers": full_layers,
            "full_attention_kv_payload_bytes": payload_bytes,
            "full_attention_kv_payload_elements": payload_elements,
            "full_attention_kv_payload_bytes_per_element": (payload_bytes / payload_elements) if payload_elements else None,
            "full_attention_kv_scale_bytes": scale_bytes,
            "full_attention_kv_scale_elements": scale_elements,
            "full_attention_kv_total_bytes": payload_bytes + scale_bytes,
            "buffer_bytes": buffer_bytes,
            "allocation_bytes": allocation_bytes,
            "owned_direct_bytes": buffer_bytes + allocation_bytes,
        }

    def kv_memory_audit(self) -> dict[str, Any]:
        """Audit retained KV storage and flag BF16 shadows for INT8 sessions."""

        summary = self.owned_buffer_summary()
        storage_dtype = DType.parse(summary["kv_storage_dtype"])
        requires_int8 = storage_dtype == DType.INT8_PER_TOKEN_HEAD
        retained_layers = list(summary.get("full_attention_layers", ()))
        persistent_bf16_layers: list[int] = []
        missing_scale_layers: list[int] = []
        payload_dtype_mismatch_layers: list[int] = []
        payload_element_size_mismatch_layers: list[int] = []
        violations: list[str] = []
        for layer in retained_layers:
            layer_id = int(layer.get("layer_id", -1))
            payload_dtype = str(layer.get("payload_dtype"))
            storage_value = str(layer.get("storage_dtype"))
            bytes_per_element = layer.get("payload_bytes_per_element")
            if requires_int8:
                if storage_value != DType.INT8_PER_TOKEN_HEAD.value or payload_dtype == DType.BF16.value:
                    persistent_bf16_layers.append(layer_id)
                if payload_dtype != DType.INT8.value:
                    payload_dtype_mismatch_layers.append(layer_id)
                if bytes_per_element is None or abs(float(bytes_per_element) - 1.0) > 1.0e-6:
                    payload_element_size_mismatch_layers.append(layer_id)
                metadata = layer.get("scale_metadata")
                if not metadata or int(metadata.get("scale_bytes", 0)) <= 0:
                    missing_scale_layers.append(layer_id)
        bf16_shadow_candidates = self._bf16_full_cache_shadow_candidates() if requires_int8 else []
        if persistent_bf16_layers:
            violations.append(f"INT8 retained KV has BF16 payload/storage layers: {persistent_bf16_layers}")
        if payload_dtype_mismatch_layers:
            violations.append(f"INT8 retained KV payload dtype mismatch layers: {payload_dtype_mismatch_layers}")
        if payload_element_size_mismatch_layers:
            violations.append(f"INT8 retained KV payload is not 1 byte/element for layers: {payload_element_size_mismatch_layers}")
        if missing_scale_layers:
            violations.append(f"INT8 retained KV missing scale metadata layers: {missing_scale_layers}")
        if bf16_shadow_candidates:
            names = [f"{item['workspace']}:{item['name']}" for item in bf16_shadow_candidates]
            violations.append(f"persistent BF16 full-cache shadow tensors after prefill: {names}")
        return {
            "required": bool(requires_int8),
            "passed": not violations,
            "kv_storage_dtype": storage_dtype.value,
            "retained_kv_buffers": retained_layers,
            "retained_kv_payload_bytes": int(summary.get("full_attention_kv_payload_bytes", 0)),
            "retained_kv_payload_elements": int(summary.get("full_attention_kv_payload_elements", 0)),
            "retained_kv_payload_bytes_per_element": summary.get("full_attention_kv_payload_bytes_per_element"),
            "retained_kv_scale_bytes": int(summary.get("full_attention_kv_scale_bytes", 0)),
            "retained_kv_scale_elements": int(summary.get("full_attention_kv_scale_elements", 0)),
            "retained_kv_total_bytes": int(summary.get("full_attention_kv_total_bytes", 0)),
            "persistent_bf16_kv_layers": persistent_bf16_layers,
            "missing_int8_scale_layers": missing_scale_layers,
            "payload_dtype_mismatch_layers": payload_dtype_mismatch_layers,
            "payload_element_size_mismatch_layers": payload_element_size_mismatch_layers,
            "bf16_shadow_candidates": bf16_shadow_candidates,
            "persistent_bf16_shadow_exists": bool(persistent_bf16_layers or bf16_shadow_candidates),
            "violations": violations,
        }

    def _bf16_full_cache_shadow_candidates(self) -> list[dict[str, Any]]:
        full_cache_shapes = {
            tuple(getattr(self.batch_layout, "slot0_full_kv_shape", ())),
            tuple(getattr(self.batch_layout, "full_kv_shape", ())),
        }
        full_cache_shapes.discard(())
        candidates: list[dict[str, Any]] = []
        seen_workspaces: set[int] = set()

        def visit_workspace(label: str, workspace: Any) -> None:
            if workspace is None or id(workspace) in seen_workspaces:
                return
            seen_workspaces.add(id(workspace))
            for name in getattr(workspace, "names", ()):  # RuntimeWorkspace.names is a tuple; fakes may expose any iterable.
                try:
                    allocation = workspace.allocation(name)
                except Exception:
                    continue
                tensor = getattr(allocation, "tensor", None)
                buffer = getattr(allocation, "buffer", None)
                if tensor is None or DType.parse(tensor.dtype) != DType.BF16:
                    continue
                reasons: list[str] = []
                if "int8_oracle" in str(name):
                    reasons.append("int8_prefill_oracle")
                if tuple(tensor.shape) in full_cache_shapes:
                    reasons.append("full_cache_shape")
                if reasons:
                    candidates.append(
                        {
                            "workspace": label,
                            "name": str(name),
                            "dtype": tensor.dtype.value,
                            "shape": list(tensor.shape),
                            "bytes": int(getattr(buffer, "nbytes", tensor.numel * tensor.dtype.itemsize)),
                            "reasons": reasons,
                        }
                    )

        visit_workspace("prefill_workspace", getattr(self, "prefill_workspace", None))
        scratch_state = getattr(self, "_prefill_scratch_state", None)
        visit_workspace("prefill_scratch_state.workspace", getattr(scratch_state, "workspace", None))
        for layer_id, state in enumerate(getattr(self, "states", ())):
            visit_workspace(f"state[{layer_id}].workspace", getattr(state, "workspace", None))
        return candidates

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

    def _prefill_use_aotriton_attention_resolved(self, tokens: int) -> bool:
        if not self._prefill_use_aotriton_attention(tokens):
            return False
        # INT8-retained sessions still build a temporary BF16 oracle K/V cache
        # during native prefill, so the BF16 AOTriton attention path is valid
        # and avoids the shared-memory-limited native causal prefill kernel at
        # long contexts. The BF16 oracle workspace is released before decode.
        return self.kv_storage_dtype in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}

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
        prefill_state = getattr(self, "_prefill_scratch_state", None)
        if prefill_state is not None:
            prefill_state._rotate_fuse_ready.clear()
        self.prefill_linear_scratch = None
        self.prefill_full_scratch = None
        self.prefill_moe_scratch = None

    def _release_decode_scratch_for_prefill(self) -> None:
        """Free token-1 decode scratch before allocating bulk prefill workspaces."""

        for state in getattr(self, "states", ()):
            state.workspace.free()
            state._rotate_fuse_ready.clear()
        for name in ("linear_scratch", "full_scratch", "moe_scratch"):
            scratch = getattr(self, name, None)
            if scratch is None:
                setattr(self, name, {})
            else:
                scratch.clear()

    def _should_minimize_prefill_workspace_overlap(self, tokens: int) -> bool:
        """Return true only when chunked prefill needs lower scratch overlap.

        Freeing decode/prefill workspaces during the timed prefill path saves
        memory for chunked long-context runs, but repeated HIP free/alloc churn
        regresses short and mid prompts (4K/8K/16K on W7900) more than the
        ~0.2 GiB tracked-memory saving justifies.  The W7900 sweep crossed over
        around 32K: the release path was tied at 32K and modestly positive by
        48K+, while still saving memory.  Treat the overlap-minimizing path as a
        long-context tactic rather than the default for every prompt that
        happens to use chunked prefill.
        """

        tokens = int(tokens)
        if tokens <= _PREFILL_OVERLAP_MIN_TOKENS:
            return False
        config = getattr(self, "prefill_config", None)
        if config is None:
            return False
        chunk_sizes = (
            int(getattr(config, "linear_chunk_size", 0)),
            int(getattr(config, "moe_chunk_size", 0)),
            int(getattr(config, "full_attn_query_chunk_size", 0)),
            int(getattr(config, "full_attn_post_chunk_size", 0)),
            int(getattr(config, "full_attn_rope_chunk_size", 0)),
        )
        return any(0 < size < tokens for size in chunk_sizes)

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

    def _ensure_full_prefill_scratch(
        self,
        *,
        tokens: int,
        aotriton_attention: bool = False,
    ) -> Qwen35ParoAttentionScratch:
        query_dtype = DType.BF16 if aotriton_attention else DType.FP32
        scratch = getattr(self, "prefill_full_scratch", None)
        if scratch is not None and scratch.attn_input.shape[0] >= tokens and scratch.query.dtype == query_dtype:
            return scratch
        scratch = self._prefill_scratch_owner().reserve_full_attention_scratch(
            tokens=tokens,
            num_splits=1,
            activation_dtype=DType.FP16,
            gated_dtype=DType.FP16,
            query_dtype=query_dtype,
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

    def _trace_linear_input_bits(
        self,
        *,
        trace_attr: str,
        layer_id: int,
        hidden: Tensor,
        rows: int,
        stream: int = 0,
    ) -> None:
        trace = getattr(self, trace_attr, None)
        if not isinstance(trace, list):
            return
        rows = int(rows)
        if rows <= 0:
            return
        if hasattr(self.runtime, "stream_synchronize"):
            self.runtime.stream_synchronize(stream)
        bits = np.empty((rows, self.config.hidden_size), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(bits),
            DeviceBuffer(hidden.ptr, bits.nbytes),
            runtime=self.runtime,
        )
        trace.append({"layer_index": int(layer_id), "bits": bits})

    def _trace_prefill_linear_input(self, *, layer_id: int, hidden: Tensor, rows: int, stream: int = 0) -> None:
        self._trace_linear_input_bits(
            trace_attr="_prefill_linear_input_trace",
            layer_id=layer_id,
            hidden=hidden,
            rows=rows,
            stream=stream,
        )

    def _trace_decode_linear_input(self, *, layer_id: int, hidden: Tensor, rows: int, stream: int = 0) -> None:
        self._trace_linear_input_bits(
            trace_attr="_decode_linear_input_trace",
            layer_id=layer_id,
            hidden=hidden,
            rows=rows,
            stream=stream,
        )

    def _trace_tensor_bits(
        self,
        *,
        trace_attr: str,
        layer_id: int,
        stage: str,
        tensor: Tensor,
        rows: int,
        stream: int = 0,
    ) -> None:
        trace = getattr(self, trace_attr, None)
        if not isinstance(trace, list):
            return
        rows = int(rows)
        if rows <= 0:
            return
        if tensor.dtype.itemsize != 2:
            raise ValueError(f"{stage} trace expects a 16-bit tensor, got {tensor.dtype}")
        if not tensor.shape or int(tensor.shape[0]) < rows:
            raise ValueError(f"{stage} trace tensor must have at least {rows} rows")
        elements_per_row = 1
        for dim in tensor.shape[1:]:
            elements_per_row *= int(dim)
        if elements_per_row <= 0:
            raise ValueError(f"{stage} trace tensor has no row payload")
        if hasattr(self.runtime, "stream_synchronize"):
            self.runtime.stream_synchronize(stream)
        bits = np.empty((rows, elements_per_row), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(bits),
            DeviceBuffer(tensor.ptr, bits.nbytes),
            runtime=self.runtime,
        )
        trace.append(
            {
                "layer_index": int(layer_id),
                "stage": stage,
                "shape": [int(rows), *(int(dim) for dim in tensor.shape[1:])],
                "bits": bits,
            }
        )

    def _trace_tensor_f32(
        self,
        *,
        trace_attr: str,
        layer_id: int,
        stage: str,
        tensor: Tensor,
        rows: int,
        stream: int = 0,
    ) -> None:
        trace = getattr(self, trace_attr, None)
        if not isinstance(trace, list):
            return
        rows = int(rows)
        if rows <= 0:
            return
        if tensor.dtype != DType.FP32:
            raise ValueError(f"{stage} trace expects an fp32 tensor, got {tensor.dtype}")
        if not tensor.shape or int(tensor.shape[0]) < rows:
            raise ValueError(f"{stage} trace tensor must have at least {rows} rows")
        elements_per_row = 1
        for dim in tensor.shape[1:]:
            elements_per_row *= int(dim)
        if elements_per_row <= 0:
            raise ValueError(f"{stage} trace tensor has no row payload")
        if hasattr(self.runtime, "stream_synchronize"):
            self.runtime.stream_synchronize(stream)
        values = np.empty((rows, elements_per_row), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(values),
            DeviceBuffer(tensor.ptr, values.nbytes),
            runtime=self.runtime,
        )
        trace.append(
            {
                "layer_index": int(layer_id),
                "stage": stage,
                "shape": [int(rows), *(int(dim) for dim in tensor.shape[1:])],
                "values": values,
            }
        )

    def _trace_decode_full_attention(
        self,
        *,
        layer_id: int,
        stage: str,
        hidden: Tensor,
        rows: int,
        stream: int = 0,
    ) -> None:
        if stage not in {"input", "attn_input", "gate", "gated_attn", "o_proj", "output"}:
            raise ValueError("decode full-attention trace stage is not recognized")
        self._trace_tensor_bits(
            trace_attr="_decode_full_attention_trace",
            layer_id=layer_id,
            stage=stage,
            tensor=hidden,
            rows=rows,
            stream=stream,
        )

    def _trace_decode_full_attention_query(
        self,
        *,
        layer_id: int,
        query: Tensor | None,
        rows: int,
        stream: int = 0,
    ) -> None:
        if query is None:
            raise ValueError("decode full-attention query trace requires a query tensor")
        self._trace_tensor_f32(
            trace_attr="_decode_full_attention_trace",
            layer_id=layer_id,
            stage="query",
            tensor=query,
            rows=rows,
            stream=stream,
        )

    def _trace_decode_full_attention_context(
        self,
        *,
        layer_id: int,
        context: Tensor | None,
        rows: int,
        stream: int = 0,
    ) -> None:
        if context is None:
            raise ValueError("decode full-attention context trace requires a context tensor")
        if rows == 1 and len(context.shape) == 2:
            context = Tensor.from_handle(
                context.ptr,
                (1, int(context.shape[0]), int(context.shape[1])),
                context.dtype,
                context.device,
            )
        self._trace_tensor_f32(
            trace_attr="_decode_full_attention_trace",
            layer_id=layer_id,
            stage="attn_context",
            tensor=context,
            rows=rows,
            stream=stream,
        )

    def _trace_decode_full_attention_scratch(
        self,
        *,
        layer_id: int,
        attention_scratch: Qwen35ParoAttentionScratch,
        rows: int,
        context: Tensor | None,
        stream: int = 0,
    ) -> None:
        if not isinstance(getattr(self, "_decode_full_attention_trace", None), list):
            return
        self._trace_decode_full_attention(
            layer_id=layer_id,
            stage="attn_input",
            hidden=attention_scratch.attn_input,
            rows=rows,
            stream=stream,
        )
        self._trace_decode_full_attention(
            layer_id=layer_id,
            stage="gate",
            hidden=attention_scratch.gate,
            rows=rows,
            stream=stream,
        )
        self._trace_decode_full_attention_query(
            layer_id=layer_id,
            query=getattr(attention_scratch, "query", None),
            rows=rows,
            stream=stream,
        )
        self._trace_decode_full_attention_context(
            layer_id=layer_id,
            context=context,
            rows=rows,
            stream=stream,
        )
        self._trace_decode_full_attention(
            layer_id=layer_id,
            stage="gated_attn",
            hidden=attention_scratch.gated_attn,
            rows=rows,
            stream=stream,
        )
        self._trace_decode_full_attention(
            layer_id=layer_id,
            stage="o_proj",
            hidden=attention_scratch.o_proj,
            rows=rows,
            stream=stream,
        )

    def _run_native_prefill_layers(self, *, tokens: int, stream: int = 0) -> Tensor:
        hidden = self._prefill_hidden_view_for_rows(tokens)
        use_aotriton_attention = self._prefill_use_aotriton_attention_resolved(tokens)
        release_workspace_between_layer_types = self._should_minimize_prefill_workspace_overlap(tokens)
        previous_layer_type: str | None = None
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            if (
                release_workspace_between_layer_types
                and previous_layer_type is not None
                and layer_type != previous_layer_type
            ):
                self._release_prefill_workspace()
            previous_layer_type = layer_type
            if layer_type == "linear_attention":
                self._trace_prefill_linear_input(layer_id=layer_id, hidden=hidden, rows=tokens, stream=stream)
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
                        hidden_chunk.ptr,
                        out.ptr,
                        rows * self.hidden_nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
            elif layer_type == "full_attention":
                retained_key_cache, retained_value_cache = self._slot_full_cache(layer_id, 0)
                int8_retained = self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                if int8_retained:
                    key_cache, value_cache = self._prefill_int8_oracle_cache(layer_id, total_tokens=tokens)
                else:
                    key_cache, value_cache = retained_key_cache, retained_value_cache
                chunk_size = self._full_attention_prefill_layer_chunk_size(tokens)
                for start, end in self._chunk_ranges(tokens, chunk_size, min_chunk_size=2):
                    rows = end - start
                    hidden_chunk = self._prefill_row_matrix_view(hidden, start, rows)
                    append_spans, prefill_spans = self._prefill_full_attention_spans(
                        rows,
                        start=start,
                        total_tokens=tokens,
                        storage_dtype=DType.BF16,
                    )
                    retained_append_spans = None
                    if int8_retained:
                        retained_append_spans, _ = self._prefill_full_attention_spans(
                            rows,
                            start=start,
                            total_tokens=tokens,
                            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
                            scale_metadata=self._slot_full_scale_metadata(layer_id, 0),
                        )
                    positions = self._prefill_rows_tensor(self.prefill_positions, rows, start=start)
                    if use_aotriton_attention:
                        cu_seqlens_q, cu_seqlens_k = self._prefill_single_cu_seqlens_pair(rows, end)
                    else:
                        cu_seqlens_q = cu_seqlens_k = None
                    attention_scratch = self._ensure_full_prefill_scratch(
                        tokens=rows,
                        aotriton_attention=use_aotriton_attention,
                    )
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
                        retained_key_cache=retained_key_cache if int8_retained else None,
                        retained_value_cache=retained_value_cache if int8_retained else None,
                        retained_append_spans=retained_append_spans,
                        tokens=rows,
                        block_size=self.block_size,
                        library=self.libraries,
                        stream=stream,
                    )
                    self.runtime.memcpy_async(
                        hidden_chunk.ptr,
                        out.ptr,
                        rows * self.hidden_nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
        return hidden

    def _run_native_prefill_packed_layers(
        self,
        slab,
        metadata: Qwen35ParoPackedPrefillMetadata,
        *,
        stream: int = 0,
    ) -> Tensor:
        rows = int(slab.rows)
        hidden = self._prefill_hidden_view_for_rows(rows)
        force_per_segment_linear = _env_flag("HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR")
        force_per_segment_full_attention = _env_flag("HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN")
        blockers: list[str] = []
        if force_per_segment_linear:
            blockers.append("linear-attention packed prefill forced to per-segment diagnostic path")
        if force_per_segment_full_attention:
            blockers.append("full-attention packed prefill forced to per-segment diagnostic path")
        self._last_packed_prefill_linear_path = "per_segment" if force_per_segment_linear else "packed_segments"
        self._last_packed_prefill_full_attention_path = "per_segment" if force_per_segment_full_attention else "packed_varlen"
        self._last_packed_prefill_blockers = blockers
        max_segment_rows = max(
            int(slab.cu_seqlens_q[index + 1]) - int(slab.cu_seqlens_q[index])
            for index in range(int(slab.request_count))
        )
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            copied_layer_output = False
            if layer_type == "linear_attention":
                self._trace_prefill_linear_input(layer_id=layer_id, hidden=hidden, rows=rows, stream=stream)
                if force_per_segment_linear:
                    for segment_index in range(int(slab.request_count)):
                        start = int(slab.cu_seqlens_q[segment_index])
                        end = int(slab.cu_seqlens_q[segment_index + 1])
                        segment_rows = end - start
                        if segment_rows <= 0:
                            continue
                        slot = int(slab.physical_slot_ids[segment_index])
                        hidden_chunk = self._prefill_row_matrix_view(hidden, start, segment_rows)
                        conv_state, recurrent_state = self._slot_linear_state(layer_id, slot)
                        linear_scratch = self._ensure_linear_prefill_scratch(tokens=segment_rows)
                        moe_scratch = self._ensure_moe_prefill_scratch(layer_id, tokens=segment_rows)
                        out = state.run_linear_attention_moe_c1_layer_fp16(
                            hidden_chunk,
                            conv_state=conv_state,
                            recurrent_state=recurrent_state,
                            linear_scratch=linear_scratch,
                            moe_scratch=moe_scratch,
                            tokens=segment_rows,
                            library=self.libraries,
                            stream=stream,
                        )
                        self.runtime.memcpy_async(
                            hidden_chunk.ptr,
                            out.ptr,
                            segment_rows * self.hidden_nbytes,
                            HipMemcpyKind.DEVICE_TO_DEVICE,
                            stream,
                        )
                    copied_layer_output = True
                else:
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
                if force_per_segment_full_attention:
                    block_count = int(slab.block_count)
                    for segment_index in range(int(slab.request_count)):
                        start = int(slab.cu_seqlens_q[segment_index])
                        end = int(slab.cu_seqlens_q[segment_index + 1])
                        segment_rows = end - start
                        if segment_rows <= 0:
                            continue
                        slot = int(slab.physical_slot_ids[segment_index])
                        local_block_table = np.asarray(slab.block_tables[start:end], dtype=np.int32)
                        local_block_table = np.ascontiguousarray(local_block_table)
                        block_table_offset = int(start) * block_count * DType.INT32.itemsize
                        copy_host_to_device(
                            DeviceBuffer(self.prefill_block_table_buf.ptr + block_table_offset, local_block_table.nbytes),
                            host_array_ptr(local_block_table),
                            local_block_table.nbytes,
                            runtime=self.runtime,
                        )
                        hidden_chunk = self._prefill_row_matrix_view(hidden, start, segment_rows)
                        block_table = Tensor.from_handle(
                            self.prefill_block_table_buf.ptr + block_table_offset,
                            (segment_rows, block_count),
                            DType.INT32,
                            self.device,
                        )
                        positions = self._prefill_rows_tensor(self.prefill_positions, segment_rows, start=start)
                        context_counts = Tensor.from_handle(
                            self.prefill_context_count_buf.ptr + int(start) * DType.INT64.itemsize,
                            (segment_rows,),
                            DType.INT64,
                            self.device,
                        )
                        append_spans = KVLiveSpans.paged_uniform(
                            block_table=block_table,
                            live_counts=positions,
                            max_live_count=segment_rows - 1,
                            storage_dtype=DType.BF16,
                            row_positions=positions,
                            span_role="prefill",
                        )
                        prefill_spans = KVLiveSpans.paged_uniform(
                            block_table=block_table,
                            live_counts=context_counts,
                            max_live_count=segment_rows,
                            storage_dtype=DType.BF16,
                            row_positions=positions,
                            span_role="prefill",
                        )
                        key_cache, value_cache = self._slot_full_cache(layer_id, slot)
                        use_aotriton_attention = self._prefill_use_aotriton_attention_resolved(segment_rows)
                        if use_aotriton_attention:
                            cu_seqlens_q, cu_seqlens_k = self._prefill_single_cu_seqlens_pair(segment_rows, segment_rows)
                        else:
                            cu_seqlens_q = cu_seqlens_k = None
                        attention_scratch = self._ensure_full_prefill_scratch(
                            tokens=segment_rows,
                            aotriton_attention=use_aotriton_attention,
                        )
                        moe_scratch = self._ensure_moe_prefill_scratch(layer_id, tokens=segment_rows)
                        if segment_rows == 1:
                            out = state.run_full_attention_moe_c1_layer_fp16(
                                hidden_chunk,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                append_spans=append_spans,
                                decode_spans=prefill_spans,
                                cos_table=self.cos,
                                sin_table=self.sin,
                                position=positions,
                                max_positions=self.max_sequence_length,
                                attention_scratch=attention_scratch,
                                moe_scratch=moe_scratch,
                                tokens=segment_rows,
                                block_size=self.block_size,
                                library=self.libraries,
                                stream=stream,
                            )
                        else:
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
                                aotriton_kv_rows=segment_rows,
                                tokens=segment_rows,
                                block_size=self.block_size,
                                library=self.libraries,
                                stream=stream,
                            )
                        self.runtime.memcpy_async(
                            hidden_chunk.ptr,
                            out.ptr,
                            segment_rows * self.hidden_nbytes,
                            HipMemcpyKind.DEVICE_TO_DEVICE,
                            stream,
                        )
                    copied_layer_output = True
                else:
                    key_cache, value_cache = self._full_cache_all_slots(layer_id)
                    use_aotriton_attention = self._prefill_use_aotriton_attention_resolved(rows)
                    if use_aotriton_attention:
                        self._last_packed_prefill_full_attention_path = "packed_varlen_aotriton"
                    attention_scratch = self._ensure_full_prefill_scratch(
                        tokens=rows,
                        aotriton_attention=use_aotriton_attention,
                    )
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
                        aotriton_attention=use_aotriton_attention,
                        aotriton_max_seqlen_q=max_segment_rows,
                        aotriton_max_seqlen_k=max_segment_rows,
                        library=self.libraries,
                        stream=stream,
                    )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
            if not copied_layer_output:
                self.runtime.memcpy_async(hidden.ptr, out.ptr, rows * self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
        return hidden

    def _run_linear_prefill_layers(self, *, tokens: int, layer_limit: int | None = None, stream: int = 0) -> Tensor:
        hidden = self._prefill_hidden_view_for_rows(tokens)
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
            self.runtime.memcpy_async(hidden.ptr, out.ptr, tokens * self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
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
                    position_tensor, append_spans, decode_spans = self._slot_full_spans(layer_id, 0)
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
        self._release_prefill_hidden_buffer()
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

    def _batch_decode_segment_metadata(
        self,
        *,
        rows: int,
        slots: tuple[int, ...],
    ) -> tuple[Tensor, Tensor, tuple[DeviceBuffer, ...]]:
        cu_arr = np.arange(rows + 1, dtype=np.int32)
        state_arr = np.asarray(slots, dtype=np.int64)
        cu_buf = malloc(cu_arr.nbytes, runtime=self.runtime)
        state_buf = malloc(state_arr.nbytes, runtime=self.runtime)
        copy_host_to_device(cu_buf, host_array_ptr(cu_arr), runtime=self.runtime)
        copy_host_to_device(state_buf, host_array_ptr(state_arr), runtime=self.runtime)
        cu = Tensor.from_handle(cu_buf.ptr, cu_arr.shape, DType.INT32, self.device)
        state_indices = Tensor.from_handle(state_buf.ptr, state_arr.shape, DType.INT64, self.device)
        return cu, state_indices, (cu_buf, state_buf)

    def _batch_full_spans(
        self,
        layer_id: int,
        *,
        rows: int,
        positions: tuple[int, ...],
        slots: tuple[int, ...],
    ) -> tuple[Tensor, KVLiveSpans, KVLiveSpans]:
        _ = layer_id
        # BF16 batch append/decode kernels run compact active rows and add an
        # active-row base internally.  Encode physical slot ids as row-relative
        # block-table offsets so row ``r`` can address slot ``slots[r]`` without
        # moving retained KV cache pages after reclaim/compaction.
        logical_blocks = np.arange(self.blocks, dtype=np.int32)
        block_rows = np.empty((rows, self.blocks), dtype=np.int32)
        for row, slot in enumerate(slots):
            delta_blocks = (int(slot) - row) * self.blocks
            if delta_blocks < 0:
                raise ValueError("batch full-attention slots must be in physical-slot order")
            block_rows[row] = logical_blocks + np.int32(delta_blocks)
        copy_host_to_device(
            self.prefill_block_table_buf,
            host_array_ptr(block_rows),
            block_rows.nbytes,
            runtime=self.runtime,
        )
        block_table = Tensor.from_handle(
            self.prefill_block_table_buf.ptr,
            (rows, self.blocks),
            DType.INT32,
            self.device,
        )
        position_tensor = Tensor.from_handle(self.position_buf.ptr, (rows,), DType.INT64, self.device)
        context_tensor = Tensor.from_handle(self.context_buf.ptr, (rows,), DType.INT64, self.device)
        append_live_counts = [int(position) for position in positions]
        decode_live_counts = [int(position) + 1 for position in positions]
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=position_tensor,
            max_live_count=max(append_live_counts),
            storage_dtype=self.kv_storage_dtype,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=context_tensor,
            max_live_count=max(decode_live_counts),
            storage_dtype=self.kv_storage_dtype,
        )
        self._last_batch_full_spans_metadata = {
            "layer_index": int(layer_id),
            "rows": int(rows),
            "slots": [int(slot) for slot in slots],
            "positions": append_live_counts,
            "append_live_counts": append_live_counts,
            "decode_live_counts": decode_live_counts,
            "append_max_live_count": int(append_spans.max_live_count),
            "decode_max_live_count": int(decode_spans.max_live_count),
            "block_size": int(getattr(self, "block_size", 256)),
            "block_table_len_per_row": int(self.blocks),
            "block_table_rows": block_rows.astype(np.int32, copy=False).tolist(),
            "storage_dtype": DType.parse(self.kv_storage_dtype).value,
        }
        return position_tensor, append_spans, decode_spans

    def _ensure_linear_decode_batch_scratch(self, layer_id: int, rows: int) -> Qwen35ParoLinearAttentionScratch:
        scratch = self.linear_scratch[layer_id]
        if scratch.attn_input.shape[0] < rows:
            scratch = self.states[layer_id].reserve_linear_attention_scratch(tokens=rows, activation_dtype=DType.FP16)
            self.linear_scratch[layer_id] = scratch
        return scratch

    def _ensure_full_decode_batch_scratch(self, layer_id: int, rows: int) -> Qwen35ParoAttentionScratch:
        scratch = self.full_scratch[layer_id]
        if scratch.attn_input.shape[0] < rows:
            scratch = self.states[layer_id].reserve_full_attention_scratch(
                tokens=rows,
                num_splits=self.max_splits,
                activation_dtype=DType.FP16,
                gated_dtype=DType.FP16,
            )
            self.full_scratch[layer_id] = scratch
        return scratch

    def _ensure_moe_decode_batch_scratch(
        self,
        layer_id: int,
        rows: int,
        *,
        force_selected_c1_moe: bool = False,
    ) -> Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch:
        scratch = self.moe_scratch[layer_id]
        if int(getattr(self.config, "num_experts", 1) or 0) <= 0:
            if not isinstance(scratch, Qwen35ParoDenseMlpScratch) or scratch.residual.shape[0] < rows:
                scratch = self.states[layer_id].reserve_dense_mlp_scratch(tokens=rows, activation_dtype=DType.FP16)
                self.moe_scratch[layer_id] = scratch
            return scratch
        if rows > 1 and not force_selected_c1_moe:
            if not isinstance(scratch, Qwen35ParoGroupedMoeScratch) or scratch.residual.shape[0] < rows:
                scratch = self.states[layer_id].reserve_moe_grouped_prefill_scratch(tokens=rows, activation_dtype=DType.FP16)
                self.moe_scratch[layer_id] = scratch
            return scratch
        if not isinstance(scratch, Qwen35ParoMoeScratch) or scratch.residual.shape[0] < rows:
            scratch = self.states[layer_id].reserve_moe_c1_scratch(tokens=rows, activation_dtype=DType.FP16)
            self.moe_scratch[layer_id] = scratch
        return scratch

    def _run_layers_batch_decode(
        self,
        *,
        rows: int,
        positions: tuple[int, ...],
        slots: tuple[int, ...],
        stream: int = 0,
    ) -> Tensor:
        if rows <= 0:
            raise ValueError("rows must be positive")
        if len(positions) != rows or len(slots) != rows:
            raise ValueError("positions and slots must match rows")
        hidden = Tensor.from_handle(self.batch_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        next_hidden = Tensor.from_handle(self.batch_next_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        cu_seqlens, state_indices, temp_buffers = self._batch_decode_segment_metadata(rows=rows, slots=slots)
        linear_segment_metadata = {
            "cu_seqlens": [int(value) for value in range(rows + 1)],
            "state_indices": [int(slot) for slot in slots],
        }
        full_attention_decode_path = "none"
        max_full_attention_context = 0
        native_full_attention_layers = 0
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        force_selected_c1_moe = (not dense_mlp) and rows > 1 and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE")
        force_per_row_linear = _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR")
        use_single_row_c1_linear = rows == 1 and not force_per_row_linear
        use_per_row_linear = force_per_row_linear or use_single_row_c1_linear
        moe_decode_path = "dense_mlp" if dense_mlp else ("selected_c1" if rows == 1 else ("selected_c1_forced" if force_selected_c1_moe else "grouped_compact"))
        moe_grouped_compact_layers = 0
        moe_selected_c1_fallback_layers = 0
        layer_executions: list[dict[str, Any]] = []
        try:
            for layer_id, state in enumerate(self.states):
                layer_type = self.config.layer_types[layer_id]
                copied_layer_output = False
                if layer_type == "linear_attention":
                    self._trace_decode_linear_input(layer_id=layer_id, hidden=hidden, rows=rows, stream=stream)
                    if use_per_row_linear:
                        row_moe_path = "dense_mlp"
                        if not dense_mlp:
                            row_moe_path = "selected_c1" if use_single_row_c1_linear else "selected_c1_per_row_linear_fallback"
                        for row, slot in enumerate(slots):
                            row_hidden = Tensor.from_handle(
                                hidden.ptr + row * self.hidden_nbytes,
                                (1, self.config.hidden_size),
                                hidden.dtype,
                                hidden.device,
                            )
                            conv_state, recurrent_state = self._slot_linear_state(layer_id, slot)
                            linear_scratch = self._ensure_linear_decode_batch_scratch(layer_id, 1)
                            moe_scratch = self._ensure_moe_decode_batch_scratch(
                                layer_id,
                                1,
                                force_selected_c1_moe=not dense_mlp and not use_single_row_c1_linear,
                            )
                            row_out = state.run_linear_attention_moe_c1_layer_fp16(
                                row_hidden,
                                conv_state=conv_state,
                                recurrent_state=recurrent_state,
                                linear_scratch=linear_scratch,
                                moe_scratch=moe_scratch,
                                tokens=1,
                                library=self.libraries,
                                stream=stream,
                            )
                            self.runtime.memcpy_async(
                                next_hidden.ptr + row * self.hidden_nbytes,
                                row_out.ptr,
                                self.hidden_nbytes,
                                HipMemcpyKind.DEVICE_TO_DEVICE,
                                stream,
                            )
                        copied_layer_output = True
                        if not dense_mlp and not use_single_row_c1_linear:
                            moe_selected_c1_fallback_layers += 1
                        layer_executions.append(
                            {
                                "layer_index": int(layer_id),
                                "layer_type": "linear_attention",
                                "rows": int(rows),
                                "slots": [int(slot) for slot in slots],
                                "linear_attention_decode_path": (
                                    "single_row_c1" if use_single_row_c1_linear else "selected_c1_per_row_fallback"
                                ),
                                "linear_attention_segment_metadata": linear_segment_metadata,
                                "linear_attention_row_state_map": [
                                    {"row": int(row), "slot": int(slot), "state_index": int(slot)}
                                    for row, slot in enumerate(slots)
                                ],
                                "full_attention_decode_path": "not_applicable",
                                "native_caware_decode": False,
                                "moe_decode_path": row_moe_path,
                            }
                        )
                    else:
                        conv_state, recurrent_state, _conv_buf, _recurrent_buf, _conv_zero, _recurrent_zero = self.linear_states[layer_id]
                        linear_scratch = self._ensure_linear_decode_batch_scratch(layer_id, rows)
                        if force_selected_c1_moe:
                            moe_scratch = self._ensure_moe_decode_batch_scratch(layer_id, rows, force_selected_c1_moe=True)
                        else:
                            moe_scratch = self._ensure_moe_decode_batch_scratch(layer_id, rows)
                        out = state.run_linear_attention_moe_decode_batch_layer_fp16(
                            hidden,
                            conv_state=conv_state,
                            recurrent_state=recurrent_state,
                            cu_seqlens=cu_seqlens,
                            state_indices=state_indices,
                            segments=rows,
                            linear_scratch=linear_scratch,
                            moe_scratch=moe_scratch,
                            tokens=rows,
                            force_selected_c1_moe=force_selected_c1_moe,
                            library=self.libraries,
                            stream=stream,
                        )
                        layer_moe_path = "dense_mlp" if dense_mlp else ("selected_c1" if rows == 1 else ("selected_c1_forced" if force_selected_c1_moe else "grouped_compact"))
                        if not dense_mlp and rows > 1:
                            if force_selected_c1_moe:
                                moe_selected_c1_fallback_layers += 1
                            else:
                                moe_grouped_compact_layers += 1
                        layer_executions.append(
                            {
                                "layer_index": int(layer_id),
                                "layer_type": "linear_attention",
                                "rows": int(rows),
                                "slots": [int(slot) for slot in slots],
                                "linear_attention_decode_path": "native_batch_segments",
                                "linear_attention_segment_metadata": linear_segment_metadata,
                                "linear_attention_row_state_map": [
                                    {"row": int(row), "slot": int(slot), "state_index": int(slot)}
                                    for row, slot in enumerate(slots)
                                ],
                                "full_attention_decode_path": "not_applicable",
                                "native_caware_decode": True,
                                "moe_decode_path": layer_moe_path,
                            }
                        )
                elif layer_type == "full_attention":
                    self._trace_decode_full_attention(
                        layer_id=layer_id,
                        stage="input",
                        hidden=hidden,
                        rows=rows,
                        stream=stream,
                    )
                    max_context = max(int(position) + 1 for position in positions)
                    max_full_attention_context = max(max_full_attention_context, max_context)
                    native_full = _env_flag("HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE", True) and max_context < 1024
                    if native_full:
                        full_attention_decode_path = "native_batch"
                        native_full_attention_layers += 1
                        key_cache, value_cache = self._full_cache_all_slots(layer_id)
                        position_tensor, append_spans, decode_spans = self._batch_full_spans(
                            layer_id,
                            rows=rows,
                            positions=positions,
                            slots=slots,
                        )
                        attention_scratch = self._ensure_full_decode_batch_scratch(layer_id, rows)
                        if force_selected_c1_moe:
                            moe_scratch = self._ensure_moe_decode_batch_scratch(layer_id, rows, force_selected_c1_moe=True)
                        else:
                            moe_scratch = self._ensure_moe_decode_batch_scratch(layer_id, rows)
                        out = state.run_full_attention_moe_decode_batch_layer_fp16(
                            hidden,
                            key_cache=key_cache,
                            value_cache=value_cache,
                            append_spans=append_spans,
                            decode_spans=decode_spans,
                            cos_table=self.cos,
                            sin_table=self.sin,
                            positions=position_tensor,
                            max_positions=self.max_sequence_length,
                            attention_scratch=attention_scratch,
                            moe_scratch=moe_scratch,
                            tokens=rows,
                            force_selected_c1_moe=force_selected_c1_moe,
                            library=self.libraries,
                            stream=stream,
                        )
                        self._trace_decode_full_attention_scratch(
                            layer_id=layer_id,
                            attention_scratch=attention_scratch,
                            rows=rows,
                            context=getattr(attention_scratch, "query_raw", None),
                            stream=stream,
                        )
                        self._trace_decode_full_attention(
                            layer_id=layer_id,
                            stage="output",
                            hidden=out,
                            rows=rows,
                            stream=stream,
                        )
                        layer_moe_path = "dense_mlp" if dense_mlp else ("selected_c1" if rows == 1 else ("selected_c1_forced" if force_selected_c1_moe else "grouped_compact"))
                        if not dense_mlp and rows > 1:
                            if force_selected_c1_moe:
                                moe_selected_c1_fallback_layers += 1
                            else:
                                moe_grouped_compact_layers += 1
                        layer_execution = {
                            "layer_index": int(layer_id),
                            "layer_type": "full_attention",
                            "rows": int(rows),
                            "slots": [int(slot) for slot in slots],
                            "max_context": int(max_context),
                            "full_attention_decode_path": "native_batch",
                            "native_caware_decode": True,
                            "moe_decode_path": layer_moe_path,
                            "attn_context_trace_source": "attention_scratch.query_raw",
                        }
                        full_spans_metadata = getattr(self, "_last_batch_full_spans_metadata", None)
                        if isinstance(full_spans_metadata, dict):
                            layer_execution["full_attention_segment_metadata"] = full_spans_metadata
                        layer_executions.append(layer_execution)
                        self.runtime.memcpy_async(next_hidden.ptr, out.ptr, rows * self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
                    else:
                        if full_attention_decode_path == "none":
                            full_attention_decode_path = "per_row_splitk_fallback" if max_context >= 1024 else "per_row_context_fallback"
                        if not dense_mlp and rows > 1:
                            moe_selected_c1_fallback_layers += 1
                        row_num_splits: list[int] = []
                        for row, (slot, position) in enumerate(zip(slots, positions, strict=True)):
                            key_cache, value_cache = self._slot_full_cache(layer_id, slot)
                            position_tensor, append_spans, decode_spans = self._slot_full_spans(layer_id, slot)
                            row_hidden = Tensor.from_handle(
                                hidden.ptr + row * self.hidden_nbytes,
                                (1, self.config.hidden_size),
                                hidden.dtype,
                                hidden.device,
                            )
                            num_splits = max(1, (int(position) + 1 + self.decode_chunk_size - 1) // self.decode_chunk_size)
                            row_num_splits.append(int(num_splits))
                            row_out = state.run_full_attention_moe_c1_layer_fp16(
                                row_hidden,
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
                            self._trace_decode_full_attention_scratch(
                                layer_id=layer_id,
                                attention_scratch=self.full_scratch[layer_id],
                                rows=1,
                                context=getattr(self.full_scratch[layer_id], "attn_out", None),
                                stream=stream,
                            )
                            self._trace_decode_full_attention(
                                layer_id=layer_id,
                                stage="output",
                                hidden=row_out,
                                rows=1,
                                stream=stream,
                            )
                            self.runtime.memcpy_async(
                                next_hidden.ptr + row * self.hidden_nbytes,
                                row_out.ptr,
                                self.hidden_nbytes,
                                HipMemcpyKind.DEVICE_TO_DEVICE,
                                stream,
                            )
                        layer_moe_path = "dense_mlp" if dense_mlp else ("selected_c1" if rows == 1 else "selected_c1_per_row_fallback")
                        layer_executions.append(
                            {
                                "layer_index": int(layer_id),
                                "layer_type": "full_attention",
                                "rows": int(rows),
                                "slots": [int(slot) for slot in slots],
                                "max_context": int(max_context),
                                "full_attention_decode_path": full_attention_decode_path,
                                "native_caware_decode": False,
                                "moe_decode_path": layer_moe_path,
                                "num_splits_per_row": row_num_splits,
                            }
                        )
                else:
                    raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
                if layer_type != "full_attention" and not copied_layer_output:
                    self.runtime.memcpy_async(next_hidden.ptr, out.ptr, rows * self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
                hidden, next_hidden = next_hidden, hidden
            decode_blockers: list[str] = []
            if force_selected_c1_moe:
                decode_blockers.append("MoE decode forced to selected-c1 diagnostic path")
            if force_per_row_linear:
                decode_blockers.append("linear-attention decode forced to per-row diagnostic path")
                if not dense_mlp and rows > 1:
                    moe_decode_path = (
                        "selected_c1_forced_with_per_row_linear_attention_fallback"
                        if force_selected_c1_moe
                        else "mixed_grouped_compact_with_per_row_linear_attention_fallback"
                    )
            if full_attention_decode_path in {"per_row_splitk_fallback", "per_row_context_fallback"}:
                decode_blockers.append("full-attention decode used a per-row fallback")
                if not dense_mlp and rows > 1:
                    if force_selected_c1_moe:
                        moe_decode_path = "selected_c1_forced_with_per_row_full_attention_fallback"
                    elif force_per_row_linear:
                        moe_decode_path = "mixed_grouped_compact_with_per_row_linear_and_full_attention_fallback"
                    else:
                        moe_decode_path = "mixed_grouped_compact_with_per_row_full_attention_fallback"
            self.last_batch_decode_execution = {
                "rows": int(rows),
                "slots": [int(slot) for slot in slots],
                "max_full_attention_context": int(max_full_attention_context),
                "native_full_attention_layers": int(native_full_attention_layers),
                "full_attention_decode_path": full_attention_decode_path,
                "native_caware_decode": full_attention_decode_path not in {"per_row_splitk_fallback", "per_row_context_fallback"} and not force_per_row_linear,
                "linear_attention_segment_metadata": linear_segment_metadata,
                "moe_decode_path": moe_decode_path,
                "moe_decode_rows": int(rows),
                "moe_grouped_compact_layers": int(moe_grouped_compact_layers),
                "moe_selected_c1_fallback_layers": int(moe_selected_c1_fallback_layers),
                "layer_executions": layer_executions,
                "blockers": decode_blockers,
            }
            return hidden
        finally:
            for buf in temp_buffers:
                free(buf, runtime=self.runtime)

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
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            if layer_type == "linear_attention":
                self._trace_decode_linear_input(layer_id=layer_id, hidden=hidden, rows=1, stream=stream)
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
                self._trace_decode_full_attention(
                    layer_id=layer_id,
                    stage="input",
                    hidden=hidden,
                    rows=1,
                    stream=stream,
                )
                key_cache, value_cache = self._slot_full_cache(layer_id, slot)
                position_tensor, append_spans, decode_spans = self._slot_full_spans(layer_id, slot)
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
                self._trace_decode_full_attention_scratch(
                    layer_id=layer_id,
                    attention_scratch=self.full_scratch[layer_id],
                    rows=1,
                    context=getattr(self.full_scratch[layer_id], "attn_out", None),
                    stream=stream,
                )
                self._trace_decode_full_attention(
                    layer_id=layer_id,
                    stage="output",
                    hidden=out,
                    rows=1,
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
        self._materialize_layers()
        self._allocate_common_buffers()
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

    def _set_empty_prefill_hidden_views(self) -> None:
        empty = Tensor.from_handle(0, (0, self.config.hidden_size), DType.FP16, self.device)
        self.prefill_hidden = empty
        # Historical diagnostics accessed ``prefill_next_hidden`` directly. The
        # retained prefill path is now single-buffer/in-place, so this is only a
        # compatibility alias unless an older diagnostic script allocates its own
        # tensor on a manually-constructed session.
        self.prefill_next_hidden = empty

    def _ensure_prefill_hidden_capacity(self, rows: int) -> Tensor:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("prefill hidden rows must be positive")
        if rows > self.prefill_capacity_rows:
            raise ValueError(
                f"prefill rows {rows} exceed session capacity {self.prefill_capacity_rows}"
            )
        nbytes = rows * self.hidden_nbytes
        current = getattr(self, "prefill_hidden_buffer", None)
        current_rows = int(getattr(self, "prefill_hidden_capacity_rows", 0) or 0)
        if current is None or current.nbytes < nbytes:
            if current is not None:
                free(current, runtime=self.runtime)
            current = malloc(nbytes, runtime=self.runtime)
            self.prefill_hidden_buffer = current
            self.prefill_hidden_capacity_rows = rows
            current_rows = rows
        self.prefill_hidden = Tensor.from_handle(
            current.ptr,
            (current_rows, self.config.hidden_size),
            DType.FP16,
            self.device,
        )
        self.prefill_next_hidden = self.prefill_hidden
        return Tensor.from_handle(current.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)

    def _prefill_hidden_view_for_rows(self, rows: int) -> Tensor:
        rows = int(rows)
        hidden = getattr(self, "prefill_hidden", None)
        if hidden is None or hidden.ptr == 0 or int(hidden.shape[0]) < rows:
            return self._ensure_prefill_hidden_capacity(rows)
        device = getattr(self, "device", hidden.device)
        return Tensor.from_handle(hidden.ptr, (rows, self.config.hidden_size), DType.FP16, device)

    def _release_prefill_hidden_buffer(self) -> None:
        current = getattr(self, "prefill_hidden_buffer", None)
        if current is None:
            return
        free(current, runtime=self.runtime)
        self.prefill_hidden_buffer = None
        self.prefill_hidden_capacity_rows = 0
        self._set_empty_prefill_hidden_views()

    def _allocate_common_buffers(self) -> None:
        hidden_buf = malloc(self.batch_hidden_nbytes, runtime=self.runtime)
        next_hidden_buf = malloc(self.batch_hidden_nbytes, runtime=self.runtime)
        norm_out_buf = malloc(self.batch_hidden_nbytes, runtime=self.runtime)
        norm_out_bf16_buf = malloc(self.batch_hidden_nbytes, runtime=self.runtime)
        self.buffers.extend((hidden_buf, next_hidden_buf, norm_out_buf, norm_out_bf16_buf))
        self.batch_hidden = Tensor.from_handle(hidden_buf.ptr, self.batch_layout.hidden_shape, DType.FP16, self.device)
        self.batch_next_hidden = Tensor.from_handle(next_hidden_buf.ptr, self.batch_layout.hidden_shape, DType.FP16, self.device)
        self.batch_norm_out = Tensor.from_handle(norm_out_buf.ptr, self.batch_layout.hidden_shape, DType.FP16, self.device)
        self.batch_norm_out_bf16 = Tensor.from_handle(norm_out_bf16_buf.ptr, self.batch_layout.hidden_shape, DType.BF16, self.device)
        self.hidden = Tensor.from_handle(hidden_buf.ptr, self.batch_layout.slot0_hidden_shape, DType.FP16, self.device)
        self.next_hidden = Tensor.from_handle(next_hidden_buf.ptr, self.batch_layout.slot0_hidden_shape, DType.FP16, self.device)
        self.norm_out = Tensor.from_handle(norm_out_buf.ptr, self.batch_layout.slot0_hidden_shape, DType.FP16, self.device)
        self.norm_out_bf16 = Tensor.from_handle(norm_out_bf16_buf.ptr, self.batch_layout.slot0_hidden_shape, DType.BF16, self.device)
        self._set_empty_prefill_hidden_views()

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
        self.batch_lm_logits = malloc(
            self.max_batch_size * self.vocab_size * DType.FP32.itemsize,
            runtime=self.runtime,
        )
        self.batch_lm_block_values = malloc(
            self.max_batch_size * self.lm_head_stage1_blocks * DType.FP32.itemsize,
            runtime=self.runtime,
        )
        self.batch_lm_block_indices = malloc(
            self.max_batch_size * self.lm_head_stage1_blocks * DType.INT64.itemsize,
            runtime=self.runtime,
        )
        self.batch_lm_out_index = malloc(
            self.max_batch_size * DType.INT64.itemsize,
            runtime=self.runtime,
        )
        self.batch_lm_out_value = malloc(
            self.max_batch_size * DType.FP32.itemsize,
            runtime=self.runtime,
        )
        self.buffers.extend(
            (
                self.lm_logits,
                self.lm_block_values,
                self.lm_block_indices,
                self.lm_out_index,
                self.lm_out_value,
                self.batch_lm_logits,
                self.batch_lm_block_values,
                self.batch_lm_block_indices,
                self.batch_lm_out_index,
                self.batch_lm_out_value,
            )
        )

    @staticmethod
    def _zero_array_dtype(dtype: DType):
        if dtype == DType.BF16:
            return np.uint16
        if dtype == DType.INT8:
            return np.int8
        if dtype == DType.FP16:
            return np.float16
        if dtype == DType.FP32:
            return np.float32
        raise ValueError(f"cannot allocate zeroed resident buffer for dtype {dtype.value!r}")

    def _auto_context_length_from_estimate(self, estimate: Qwen35ParoKVCapacityEstimate) -> int:
        if estimate.model_max_context_tokens > 0 and estimate.allocatable_context_tokens > 0:
            return min(estimate.model_max_context_tokens, estimate.allocatable_context_tokens)
        if estimate.model_max_context_tokens > 0:
            return estimate.model_max_context_tokens
        return max(0, estimate.allocatable_context_tokens)

    def _set_sequence_capacity(self, max_sequence_length: int) -> None:
        capacity = int(max_sequence_length)
        if capacity <= 0:
            raise ValueError("max_sequence_length must be positive")
        self.max_sequence_length = capacity
        self.decode_chunk_size, self.max_splits = _paged_attn_decode_split_config(
            self.max_sequence_length,
            block_size=self.block_size,
            chunk_size=self.chunk_size,
        )
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
        if hasattr(self, "hidden_nbytes"):
            self.batch_hidden_nbytes = self.max_batch_size * self.hidden_nbytes
            self.prefill_capacity_rows = self.max_sequence_length * self.max_batch_size
            self.prefill_hidden_nbytes = self.prefill_capacity_rows * self.hidden_nbytes
        self.active_batch = ActiveBatch(self.max_batch_size)
        self.active_batch.admit(RequestState.from_tokens(0, (), max_new_tokens=self.max_sequence_length))

    def _check_retained_kv_capacity_before_allocation(self) -> Qwen35ParoKVCapacityEstimate | None:
        try:
            free_bytes, _total_bytes = self.runtime.mem_get_info()
        except Exception as exc:
            self._emit("kv_capacity_estimate_unavailable", error=str(exc))
            return None
        reserve_mib = max(0, _env_int("HIPENGINE_KV_CAPACITY_RESERVE_MIB", 512))
        reserve_bytes = reserve_mib * 1024**2
        estimate = estimate_qwen35_paro_kv_capacity(
            self.config,
            available_bytes=free_bytes,
            requested_context_tokens=self.max_sequence_length,
            storage_dtype=self.kv_storage_dtype,
            scale_dtype=self.kv_scale_dtype,
            block_size=self.block_size,
            chunk_size=self.chunk_size,
            reserve_bytes=reserve_bytes,
            max_batch_size=self.max_batch_size,
        )
        if self.auto_context_length:
            auto_context = self._auto_context_length_from_estimate(estimate)
            if auto_context <= 0:
                raise MemoryError(
                    "automatic resident KV cache sizing found no allocatable context tokens; "
                    "try freeing GPU memory, setting --kv-storage int8_per_token_head, or "
                    "setting a lower --max-context-tokens manually"
                )
            if auto_context != self.max_sequence_length:
                self._emit(
                    "kv_auto_context_selected",
                    requested_context_tokens=self.max_sequence_length,
                    selected_context_tokens=auto_context,
                    model_max_context_tokens=estimate.model_max_context_tokens,
                    allocatable_context_tokens=estimate.allocatable_context_tokens,
                )
                self._set_sequence_capacity(auto_context)
                estimate = estimate_qwen35_paro_kv_capacity(
                    self.config,
                    available_bytes=free_bytes,
                    requested_context_tokens=self.max_sequence_length,
                    storage_dtype=self.kv_storage_dtype,
                    scale_dtype=self.kv_scale_dtype,
                    block_size=self.block_size,
                    chunk_size=self.chunk_size,
                    reserve_bytes=reserve_bytes,
                    max_batch_size=self.max_batch_size,
                )
        self.kv_capacity_estimate = estimate
        self._emit("kv_capacity_estimate", **estimate.to_json_dict())
        label = "INT8" if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD else self.kv_storage_dtype.value.upper()
        _LOGGER.debug(
            "%s KV capacity estimate: requested resident context %d tokens needs %s KV + %s metadata; "
            "current free HIP memory can fit about %d tokens (%s usable after %s reserve)%s.",
            label,
            estimate.requested_context_tokens,
            _format_bytes_gib(estimate.requested_kv_bytes),
            _format_bytes_gib(estimate.requested_context_overhead_bytes),
            estimate.allocatable_context_tokens,
            _format_bytes_gib(estimate.usable_bytes),
            _format_bytes_gib(estimate.reserve_bytes),
            "" if estimate.model_max_context_tokens <= 0 else f" vs model max {estimate.model_max_context_tokens} tokens",
        )
        if estimate.model_max_context_tokens > 0 and not estimate.fits_model_max:
            _LOGGER.debug(
                "%s KV capacity estimate: current free HIP memory after model load can fit about %d tokens "
                "(%s usable after %s reserve), below model max context %d tokens; requested resident context is %d tokens.",
                label,
                estimate.allocatable_context_tokens,
                _format_bytes_gib(estimate.usable_bytes),
                _format_bytes_gib(estimate.reserve_bytes),
                estimate.model_max_context_tokens,
                estimate.requested_context_tokens,
            )
        if self.kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
            int8_estimate = estimate_qwen35_paro_kv_capacity(
                self.config,
                available_bytes=free_bytes,
                requested_context_tokens=self.max_sequence_length,
                storage_dtype=DType.INT8_PER_TOKEN_HEAD,
                scale_dtype=self.kv_scale_dtype,
                block_size=self.block_size,
                chunk_size=self.chunk_size,
                reserve_bytes=reserve_bytes,
                max_batch_size=self.max_batch_size,
            )
            self.kv_capacity_int8_estimate = int8_estimate
            self._emit("kv_capacity_estimate_int8", **int8_estimate.to_json_dict())
            _LOGGER.debug(
                "INT8 KV capacity estimate: requested resident context %d tokens needs %s KV + %s metadata; "
                "current free HIP memory can fit about %d tokens (%s usable after %s reserve)%s.",
                int8_estimate.requested_context_tokens,
                _format_bytes_gib(int8_estimate.requested_kv_bytes),
                _format_bytes_gib(int8_estimate.requested_context_overhead_bytes),
                int8_estimate.allocatable_context_tokens,
                _format_bytes_gib(int8_estimate.usable_bytes),
                _format_bytes_gib(int8_estimate.reserve_bytes),
                "" if int8_estimate.model_max_context_tokens <= 0 else f" vs model max {int8_estimate.model_max_context_tokens} tokens",
            )
            if int8_estimate.model_max_context_tokens > 0 and not int8_estimate.fits_model_max:
                _LOGGER.debug(
                    "INT8 KV capacity estimate: current free HIP memory after model load can fit about %d tokens "
                    "(%s usable after %s reserve), below model max context %d tokens; requested resident context is %d tokens.",
                    int8_estimate.allocatable_context_tokens,
                    _format_bytes_gib(int8_estimate.usable_bytes),
                    _format_bytes_gib(int8_estimate.reserve_bytes),
                    int8_estimate.model_max_context_tokens,
                    int8_estimate.requested_context_tokens,
                )
        if not estimate.fits_requested:
            raise MemoryError(
                "requested resident KV cache context "
                f"{estimate.requested_context_tokens_rounded} tokens needs "
                f"{_format_bytes_gib(estimate.requested_total_bytes)} "
                f"({_format_bytes_gib(estimate.requested_kv_bytes)} KV + "
                f"{_format_bytes_gib(estimate.requested_context_overhead_bytes)} metadata) but only "
                f"{_format_bytes_gib(estimate.usable_bytes)} is estimated available for retained KV "
                f"after reserve; estimated max context is {estimate.allocatable_context_tokens} tokens "
                f"with {estimate.kv_storage_dtype} KV; try a lower --max-context-tokens "
                "or --kv-storage int8_per_token_head"
            )
        return estimate

    def _allocate_full_attention_cache(self, layer_id: int) -> None:
        payload_dtype = DType.INT8 if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD else DType.BF16
        key_zero = np.zeros(self.batch_layout.full_kv_shape, dtype=self._zero_array_dtype(payload_dtype))
        value_zero = np.zeros_like(key_zero)
        key_buf = self._dev(key_zero)
        value_buf = self._dev(value_zero)
        key_cache = Tensor.from_handle(key_buf.ptr, self.batch_layout.slot0_full_kv_shape, payload_dtype, self.device)
        value_cache = Tensor.from_handle(value_buf.ptr, self.batch_layout.slot0_full_kv_shape, payload_dtype, self.device)
        self.full_caches[layer_id] = (key_cache, value_cache, key_buf, value_buf)

        if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            scale_zero = np.zeros(
                self.batch_layout.flat_full_kv_scale_shape,
                dtype=self._zero_array_dtype(self.kv_scale_dtype),
            )
            k_scale_buf = self._dev(scale_zero)
            v_scale_buf = self._dev(np.zeros_like(scale_zero))
            k_scale = Tensor.from_handle(
                k_scale_buf.ptr,
                self.batch_layout.slot0_full_kv_scale_shape,
                self.kv_scale_dtype,
                self.device,
            )
            v_scale = Tensor.from_handle(
                v_scale_buf.ptr,
                self.batch_layout.slot0_full_kv_scale_shape,
                self.kv_scale_dtype,
                self.device,
            )
            self.full_cache_scales[layer_id] = (k_scale, v_scale, k_scale_buf, v_scale_buf)
            self.full_cache_scale_metadata[layer_id] = KVScaleMetadata(
                k_scale=k_scale,
                v_scale=v_scale,
                scale_dtype=self.kv_scale_dtype,
                granularity=self.kv_scale_granularity,
            )
        else:
            self.full_cache_scales.pop(layer_id, None)
            self.full_cache_scale_metadata.pop(layer_id, None)

    def _materialize_layers(self) -> None:
        self.states = self.runner._materialize_resident_states(self.layer_limit, emit=self._emit)
        self._check_retained_kv_capacity_before_allocation()
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
                self._allocate_full_attention_cache(layer_id)
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
        if hasattr(self, "position_arr") and hasattr(self, "context_arr"):
            self.position_arr[: len(pos)] = pos_arr
            self.context_arr[: len(pos)] = pos_arr + np.int64(1)
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
        if hasattr(self, "position_arr") and hasattr(self, "context_arr"):
            self.position_arr[0] = int(position)
            self.context_arr[0] = int(position) + 1
        set_decode_position_i64(
            self.position_buf.ptr,
            self.context_buf.ptr,
            int(position),
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )

    def _set_slot_position(self, position: int, *, slot: int, stream: int = 0) -> None:
        if hasattr(self, "position_arr") and hasattr(self, "context_arr"):
            self.position_arr[int(slot)] = int(position)
            self.context_arr[int(slot)] = int(position) + 1
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

    def _sample_batch_from_hidden(self, hidden: Tensor, *, rows: int, stream: int = 0) -> tuple[Qwen35ParoAutoregressiveStepResult, ...]:
        if rows <= 0:
            raise ValueError("rows must be positive")
        if rows > self.max_batch_size:
            raise ValueError("rows exceed max_batch_size")
        sample_mode = os.environ.get("HIPENGINE_QWEN35_BATCH_SAMPLE_MODE", "serial_lm_head")
        try:
            sampler_decision = plan_batch_sampler_dispatch(
                rows=rows,
                requested_mode=sample_mode,
                c2_equality_green=_env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_C2_EQ_OK"),
                equality_artifact=os.environ.get("HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ARTIFACT") or None,
                equality_rows=os.environ.get("HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ROWS") or None,
            )
        except ValueError as exc:
            raise ValueError("HIPENGINE_QWEN35_BATCH_SAMPLE_MODE must be serial_lm_head or batched_lm_head") from exc
        self.last_batch_sampler_execution = sampler_decision.to_json_dict()
        decode_execution = getattr(self, "last_batch_decode_execution", None)
        if isinstance(decode_execution, dict):
            decode_execution["sampler_execution"] = sampler_decision.to_json_dict()
        if sampler_decision.mode is BatchSamplerMode.SERIAL_LM_HEAD:
            results: list[Qwen35ParoAutoregressiveStepResult] = []
            for row in range(rows):
                row_hidden = Tensor.from_handle(
                    hidden.ptr + row * self.hidden_nbytes,
                    (1, self.config.hidden_size),
                    hidden.dtype,
                    hidden.device,
                )
                results.append(self._sample_from_hidden(row_hidden))
            return tuple(results)
        paro_rmsnorm_out_fp16(
            hidden.ptr,
            self.norm_weight.tensor.ptr,
            self.batch_norm_out.ptr,
            rows,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            stream=stream,
            library=self.libraries["norm"],
            runtime=self.runtime,
        )
        fp16_to_bf16(
            self.batch_norm_out.ptr,
            self.batch_norm_out_bf16.ptr,
            rows * self.config.hidden_size,
            stream=stream,
            library=self.libraries["cast"],
            runtime=self.runtime,
        )
        w8a16_linear_bf16_f32_out(
            self.batch_norm_out_bf16.ptr,
            self.lm_head_weight.tensor.ptr,
            self.lm_head_scale.tensor.ptr,
            self.batch_lm_logits.ptr,
            rows,
            self.config.hidden_size,
            self.vocab_size,
            threads=self.lm_head_threads,
            stream=stream,
            library=self.libraries["w8a16"],
            runtime=self.runtime,
        )
        for row in range(rows):
            logits_ptr = self.batch_lm_logits.ptr + row * self.vocab_size * DType.FP32.itemsize
            block_values_ptr = self.batch_lm_block_values.ptr + row * self.lm_head_stage1_blocks * DType.FP32.itemsize
            block_indices_ptr = self.batch_lm_block_indices.ptr + row * self.lm_head_stage1_blocks * DType.INT64.itemsize
            out_index_ptr = self.batch_lm_out_index.ptr + row * DType.INT64.itemsize
            out_value_ptr = self.batch_lm_out_value.ptr + row * DType.FP32.itemsize
            argmax_f32(
                logits_ptr,
                block_values_ptr,
                block_indices_ptr,
                out_index_ptr,
                out_value_ptr,
                self.vocab_size,
                threads=self.lm_head_threads,
                stream=stream,
                library=self.libraries["lm_head"],
                runtime=self.runtime,
            )
        self.runtime.device_synchronize()
        index_host = np.empty((rows,), dtype=np.int64)
        value_host = np.empty((rows,), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(index_host),
            DeviceBuffer(self.batch_lm_out_index.ptr, rows * DType.INT64.itemsize),
            runtime=self.runtime,
        )
        copy_device_to_host(
            host_array_ptr(value_host),
            DeviceBuffer(self.batch_lm_out_value.ptr, rows * DType.FP32.itemsize),
            runtime=self.runtime,
        )
        return tuple(
            Qwen35ParoAutoregressiveStepResult(
                token_id=int(index_host[row]),
                token_text=_decode_token_cached(self.tokenizer, int(index_host[row])),
                logit=float(value_host[row]),
            )
            for row in range(rows)
        )

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
