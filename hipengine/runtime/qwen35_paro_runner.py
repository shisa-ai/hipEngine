"""Bring-up runner for real Qwen3.5/PARO one-token decode smokes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import json
import logging
import os
import sys

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
    batch_argmax_f32,
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16,
    w8a16_lm_head_argmax_rows_bf16,
)
from hipengine.kernels.hip_gfx1100.speculative import (
    ACCEPT_PACKED_PAYLOAD_FIELDS,
    build_dflash_accept,
    build_dflash_commit,
    dflash_accept_chain_i32,
    dflash_accept_chain_i32_packed,
    dflash_accept_chain_i32_packed_update_state,
    dflash_commit_chain_i32,
    linear_state_pair_commit_chunked_i32,
    linear_state_pair_commit_i32,
)
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import build_aotriton_wrap
from hipengine.kernels.hip_gfx1100.convert import fp16_to_bf16, fp16_to_bf16_strided_rows
from hipengine.kernels.hip_gfx1100.norm import paro_rmsnorm_out_bf16, paro_rmsnorm_out_fp16
from hipengine.kernels.hip_gfx1100.sampling import (
    apply_processors_f32_rows,
    build_sampler,
    sample_temperature_f32_rows_i32,
    sample_temperature_top_logprobs_f32_rows_i32,
    sample_top_p_temperature_f32_rows_i32,
    sample_topk_temperature_f32_rows_i32,
)
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import (
    w8a16_linear_bf16_f32_multi_row,
    w8a16_linear_bf16_f32_out,
)
from hipengine.kernels.hip_gfx1100.runtime import (
    advance_decode_position_i64,
    advance_decode_positions_i64,
    embedding_lookup_batch_fp16_i64,
    embedding_lookup_batch_mapped_fp16_i64,
    embedding_lookup_fp16_i64,
    record_i64_scalar_indexed,
    set_decode_position_i64,
    set_decode_positions_i64,
    set_i64_scalar,
    set_i64_vector,
    unpack_verify_chain_dynamic_metadata_i64,
)
from hipengine.dispatch import (
    ActiveBatch,
    BatchSamplerMode,
    ProjectionKernelSelection,
    RequestState,
    plan_batch_sampler_dispatch,
    plan_projection_dispatch,
    projection_dispatch_candidates_from_artifact,
    projection_dispatch_evidence_payload_blockers,
)
from hipengine.generation.sampling import RowSamplingState, normalize_logit_bias_pairs, select_token
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
    _reset_shared_rotate_fuse_barrier_state,
    _set_shared_rotate_fuse_barrier_memset_mode,
    _use_moe_grouped_compact_prefill,
    _verify_moe_grouped_min_tokens,
)
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative import DraftBatch, TargetAcceptSummary, TargetCommitPlan, TargetStateCommitBuffers, TargetVerifyBatch, TargetVerifyBuffers


_PREFILL_OVERLAP_MIN_TOKENS = 32768
_INT8_PREFILL_ATTENTION_ENV = "HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION"
_INT8_PREFILL_STREAMING_MIN_TOKENS_ENV = "HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS"
_INT8_PREFILL_STREAMING_MIN_TOKENS_DEFAULT = 224 * 1024
_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB_ENV = "HIPENGINE_QWEN35_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB"
_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB_DEFAULT = 26.0
_INT8_PREFILL_ORACLE_RESERVE_MIB_ENV = "HIPENGINE_QWEN35_INT8_PREFILL_ORACLE_RESERVE_MIB"
_INT8_PREFILL_ORACLE_RESERVE_MIB_DEFAULT = 1024
_LOGGER = logging.getLogger(__name__)
_VERIFY_DYNAMIC_METADATA_FIELDS = 5

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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _native_sampler_needs_processors(params: Any) -> bool:
    return (
        bool(normalize_logit_bias_pairs(getattr(params, "logit_bias", None)))
        or bool(_native_sampler_suppress_token_ids(params))
        or int(getattr(params, "min_tokens", 0)) > 0
        or float(getattr(params, "repetition_penalty", 1.0)) != 1.0
        or float(getattr(params, "presence_penalty", 0.0)) != 0.0
        or float(getattr(params, "frequency_penalty", 0.0)) != 0.0
    )


def _native_sampler_suppress_token_ids(params: Any) -> tuple[int, ...]:
    raw_ids = getattr(params, "suppress_token_ids", None)
    if raw_ids is None:
        raw_ids = getattr(params, "suppress_tokens", ())
    return tuple(int(token) for token in (raw_ids or ()))


def _env_int_set(name: str) -> set[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return set()
    parsed: set[int] = set()
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"{name} has descending range {text!r}")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(text))
    if any(layer < 0 for layer in parsed):
        raise ValueError(f"{name} layer ids must be non-negative")
    return parsed


_PROJECTION_DISPATCH_ARTIFACT_ENV = "HIPENGINE_QWEN35_PROJECTION_DISPATCH_ARTIFACT"


def _path_has_benchmark_results_symlink_parent(path: Path) -> bool:
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    current = path.parent
    while True:
        try:
            current_resolved = current.resolve()
        except OSError:
            return False
        if not current_resolved.is_relative_to(results_root):
            return False
        if current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent


def _load_projection_dispatch_env_artifact(path_text: str, *, label: str) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    path = Path(path_text)
    if path.is_absolute() or len(path.parts) < 3 or path.parts[:2] != ("benchmarks", "results") or ".." in path.parts:
        return None, (f"{label} must be a relative path under benchmarks/results",)
    check_path = Path.cwd() / path
    results_root = (Path.cwd() / "benchmarks" / "results").resolve()
    try:
        if check_path.is_symlink():
            return None, (f"{label} must not be a symlink",)
        if _path_has_benchmark_results_symlink_parent(check_path):
            return None, (f"{label} parent directories must not be symlinks",)
        if check_path.suffix.lower() != ".json":
            return None, (f"{label} must point to a .json artifact",)
        if not check_path.exists():
            return None, (f"{label} must point to an existing JSON artifact",)
        if not check_path.is_file():
            return None, (f"{label} must point to a regular JSON artifact",)
        if not check_path.resolve().is_relative_to(results_root):
            return None, (f"{label} must resolve under benchmarks/results",)
        payload = json.loads(check_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, (f"{label} must point to a readable JSON artifact: {exc}",)
    except json.JSONDecodeError as exc:
        return None, (f"{label} must be valid JSON: {exc}",)
    if not isinstance(payload, Mapping):
        return None, (f"{label} must be a JSON object",)
    return payload, ()


def _artifact_row_count(payload: Mapping[str, Any]) -> Any:
    rows = payload.get("rows")
    if rows is not None:
        return rows
    workload = payload.get("workload")
    if isinstance(workload, Mapping):
        return workload.get("concurrency")
    return None


def _artifact_is_accepted(payload: Mapping[str, Any]) -> bool:
    if payload.get("accepted") is True or payload.get("passed") is True or payload.get("status") == "accepted":
        return True
    decision = payload.get("decision")
    return isinstance(decision, Mapping) and decision.get("accepted") is True


def _projection_candidate_evidence_blockers(candidate: Any) -> tuple[str, ...]:
    evidence = getattr(candidate, "evidence", None)
    if evidence is None:
        return ()
    evidence_payload, errors = _load_projection_dispatch_env_artifact(
        evidence.artifact_path,
        label=f"{candidate.name} evidence artifact_path",
    )
    if errors:
        return errors
    if evidence_payload is None:
        return ()
    blockers: list[str] = []
    if not _artifact_is_accepted(evidence_payload):
        blockers.append(f"{candidate.name} evidence artifact_path artifact must be accepted")
    artifact_rows = _artifact_row_count(evidence_payload)
    if isinstance(artifact_rows, bool) or not isinstance(artifact_rows, int):
        blockers.append(f"{candidate.name} evidence artifact_path rows must be an int")
    elif not candidate.applies_to(artifact_rows):
        blockers.append(f"{candidate.name} evidence artifact_path rows must be within candidate row bounds")
    else:
        blockers.extend(
            f"{candidate.name} evidence {blocker}"
            for blocker in projection_dispatch_evidence_payload_blockers(
                evidence_payload,
                evidence,
                rows=artifact_rows,
                label="artifact_path",
            )
        )
    return tuple(blockers)


def _env_projection_dispatch_candidates() -> tuple[tuple[Any, ...], tuple[str, ...]]:
    raw = os.environ.get(_PROJECTION_DISPATCH_ARTIFACT_ENV)
    if raw is None or not raw.strip():
        return (), ()
    payload, artifact_errors = _load_projection_dispatch_env_artifact(
        raw.strip(),
        label=_PROJECTION_DISPATCH_ARTIFACT_ENV,
    )
    if artifact_errors:
        return (), artifact_errors
    if payload is None:
        return (), ()
    try:
        candidates = projection_dispatch_candidates_from_artifact(payload)
    except ValueError as exc:
        return (), (f"{_PROJECTION_DISPATCH_ARTIFACT_ENV} has invalid projection_dispatch_candidates: {exc}",)
    if not candidates:
        return (), (f"{_PROJECTION_DISPATCH_ARTIFACT_ENV} must include projection_dispatch_candidates",)
    candidate_blockers: list[str] = []
    for candidate in candidates:
        candidate_blockers.extend(_projection_candidate_evidence_blockers(candidate))
    if candidate_blockers:
        return (), tuple(f"{_PROJECTION_DISPATCH_ARTIFACT_ENV} {blocker}" for blocker in candidate_blockers)
    return candidates, ()


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
    logprob: float | None = None
    top_logprobs: tuple[tuple[int, float], ...] = ()
    forced: bool = False
    forced_reason: str | None = None
    forced_tokens_remaining: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "token_text": self.token_text,
            "logit": self.logit,
            "logprob": self.logprob,
            "top_logprobs": [list(item) for item in self.top_logprobs],
            "forced": self.forced,
            "forced_reason": self.forced_reason,
            "forced_tokens_remaining": self.forced_tokens_remaining,
        }


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


def _dflash_verify_fused_lm_head_enabled() -> bool:
    """Return True when the DFlash verifier should use the R3.7 fused W8A16
    LM-head + argmax-rows kernel instead of the unfused
    ``w8a16_linear_bf16_f32_multi_row -> argmax_f32_rows_i32`` pair.

    Controlled by ``HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD`` (``off`` / ``on``).
    Defaults to ``off``; flip to ``on`` to opt into the fused path.  When
    enabled, the per-cycle ``[rows, vocab_size]`` FP32 logits buffer is never
    materialized in HBM, satisfying the R3.7 rocprof gate (no full-vocab
    lm-head kernel in the verifier window).  Bit-exact vs the unfused path
    because the cooperative-per-vocab-row dot product order is preserved.
    """

    value = os.environ.get("HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD", "off").strip().lower()
    if value in {"", "off", "0", "false", "naive", "unfused"}:
        return False
    if value in {"on", "1", "true", "fused"}:
        return True
    raise ValueError(
        "HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD must be one of: off, on (got " + repr(value) + ")"
    )


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
    # Capture-time per-layer verifier scratch (rows=B+1). The captured graph
    # writes these buffers, but `_canonicalize_decode_scratch` swaps the live
    # `self.linear_scratch` map to rows=1 decode handles after every cycle.
    # Direct passes re-reserve rows scratch each cycle; replays do not, so the
    # commit must restore these handles before reading `tree_*_state`.
    linear_scratch: dict[int, Any] | None = None
    moe_scratch: dict[int, Any] | None = None


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
        self._verify_scratch_cache_generation = 0
        self._verify_linear_scratch_cache: dict[tuple[int, int], tuple[int, Qwen35ParoLinearAttentionScratch]] = {}
        self._verify_mlp_scratch_cache: dict[tuple[int, int, str], tuple[int, Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch]] = {}
        self._resident_tensor_view_cache_enabled_value = _env_flag("HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE", True)
        self._slot_linear_state_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        self._slot_full_cache_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        self._full_cache_all_slots_cache: dict[int, tuple[Tensor, Tensor]] = {}
        self.prefill_workspace = RuntimeWorkspace(runtime=self.runtime)
        _reset_shared_rotate_fuse_barrier_state()
        self.prefill_hidden_buffer: DeviceBuffer | None = None
        self.prefill_hidden_capacity_rows = 0
        self._prefill_scratch_state: Qwen35ParoDecodeState | None = None
        self.prefill_linear_scratch: Qwen35ParoLinearAttentionScratch | None = None
        self.prefill_full_scratch: Qwen35ParoAttentionScratch | None = None
        self.prefill_moe_scratch: Qwen35ParoGroupedMoeScratch | Qwen35ParoMoeScratch | None = None
        self._host_sampling_params: Any | None = None
        self._host_sampling_state: RowSamplingState | None = None
        self._host_sampling_states_by_slot: dict[int, RowSamplingState] | None = None
        self._native_sampling_params: Any | None = None
        self._native_sampling_state: RowSamplingState | None = None
        self._native_sampling_states_by_slot: dict[int, RowSamplingState] | None = None
        self._native_sampler_library: Any | None = None
        self._native_sampler_cached_uploads: dict[tuple[Any, ...], DeviceBuffer] = {}
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
        self._invalidate_verify_graph_cache()
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
        self._clear_resident_tensor_view_caches()
        self._decode_full_block_table_key = None
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
                if sample:
                    results.append(self._sample_from_hidden_for_slot(hidden, slot))
                else:
                    results.append(None)
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
        device_resident: bool = False,
    ) -> tuple[Qwen35ParoAutoregressiveStepResult | None, ...]:
        """Run one decode token per active row through native c-aware layer kernels.

        When ``device_resident`` is set the step reads its input tokens from the
        device ``batch_lm_out_index`` buffer, gathers embeddings, runs the layer
        kernels, and writes the next-token argmax back to ``batch_lm_out_index``
        with no host token list or sampler readback on the compute path
        (C3.0b pieces A+B+C) -- the capture-ready decode step.  The eager driver
        seeds the token buffer and reads the result back for parity testing.

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
        if device_resident:
            if not sample:
                raise NotImplementedError("device-resident native c>N decode requires sampling")
            # Seed the device next-token buffer with this step's input tokens,
            # run the fully device-resident step (pieces A+B), and read the
            # produced tokens back.  Eager callers re-set positions each step,
            # so the device position/context counters are not self-advanced.
            token_arr = np.asarray(tokens, dtype=np.int64)
            copy_host_to_device(
                DeviceBuffer(self.batch_lm_out_index.ptr, rows * DType.INT64.itemsize),
                host_array_ptr(token_arr),
                token_arr.nbytes,
                runtime=self.runtime,
            )
            self._set_batch_positions(pos, stream=0)
            self._step_batch_from_device_tokens(
                rows=rows,
                positions=pos,
                slots=slot_ids,
                advance_positions=False,
                stream=0,
            )
            return self._read_batch_next_tokens(rows=rows)
        self._set_batch_token_embeddings(tokens, stream=0)
        self._set_batch_positions(pos, stream=0)
        hidden = self._run_layers_batch_decode(rows=rows, positions=pos, slots=slot_ids, stream=0)
        if not sample:
            self.runtime.device_synchronize()
            return tuple(None for _ in tokens)
        return self._sample_batch_from_hidden(hidden, rows=rows)

    def capture_batch_decode_graph(
        self,
        *,
        rows: int,
        positions: list[int] | tuple[int, ...],
        slots: list[int] | tuple[int, ...] | None = None,
        max_replay_steps: int,
    ) -> "Qwen35ParoBatchDecodeGraph":
        """Capture one device-resident c>1 decode step for HIP-graph replay (C3.0b piece D).

        The captured step reads its input tokens from ``batch_lm_out_index``,
        runs the c-aware layer kernels, writes the next-token argmax back to
        ``batch_lm_out_index``, and advances the device decode position/context
        counters on-stream -- so replaying the single captured graph walks the
        decode forward with no host involvement on the compute path.

        The full-attention span capacity (``max_live_count`` / ``max_context_len``)
        is baked for the *last* replay step (``start + max_replay_steps - 1``)
        while the device ``live_counts`` (``position_buf``/``context_buf``) start
        at ``positions`` and advance each replay; the batch decode kernels read
        the live count from the device pointer and use the baked count only as a
        static upper bound, and full attention uses the session-wide
        ``max_splits`` so no per-step split recompute is needed.

        The caller must have already exercised the eager device-resident path at
        these ``(rows, slots)`` (e.g. a warmup decode) so every scratch buffer
        and the segment/block-table caches are allocated -- HIP graph capture
        forbids ``hipMalloc``/``hipFree`` on the capturing stream.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        if not _env_flag("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"):
            raise NotImplementedError(
                "native c>N decode graph replay is experimental; set "
                "HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1"
            )
        if self.kv_storage_dtype != DType.BF16:
            raise NotImplementedError("native c>N decode graph replay currently requires BF16 KV")
        if rows <= 0:
            raise ValueError("rows must be positive")
        if rows > self.max_batch_size:
            raise ValueError("rows exceed max_batch_size")
        if int(max_replay_steps) <= 0:
            raise ValueError("max_replay_steps must be positive")
        start_positions = tuple(int(position) for position in positions)
        if len(start_positions) != rows:
            raise ValueError("positions must have the same length as rows")
        slot_ids = tuple(range(rows)) if slots is None else tuple(int(slot) for slot in slots)
        if len(slot_ids) != rows:
            raise ValueError("slots must have the same length as rows")
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("native c>N decode slots must be unique")
        if tuple(sorted(slot_ids)) != slot_ids:
            raise NotImplementedError("native c>N decode requires slots in physical-slot order")
        replay_span = int(max_replay_steps)
        capacity_positions = tuple(position + replay_span - 1 for position in start_positions)
        for position in start_positions:
            self._check_position(position)
        for position in capacity_positions:
            self._check_position(position)
        stream = self.runtime.stream_create()
        graph = 0
        try:
            # Seed the device decode counters to the real start positions before
            # capture (this runs, not captured); the captured advance walks them
            # forward each replay.
            self._set_batch_positions(start_positions, stream=stream)
            self.runtime.stream_synchronize(stream)
            self.runtime.stream_begin_capture(stream)
            try:
                self._step_batch_from_device_tokens(
                    rows=rows,
                    positions=capacity_positions,
                    slots=slot_ids,
                    advance_positions=True,
                    stream=stream,
                )
                graph = self.runtime.stream_end_capture(stream)
            except Exception:
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
            raise
        # Capture does not execute, so the device counters are still at start;
        # reset explicitly for clarity so the first replay begins at start.
        self._set_batch_positions(start_positions, stream=stream)
        self.runtime.stream_synchronize(stream)
        return Qwen35ParoBatchDecodeGraph(
            session=self,
            graph=graph,
            graph_exec=graph_exec,
            stream=stream,
            rows=rows,
            slots=slot_ids,
            start_positions=start_positions,
            max_replay_steps=replay_span,
        )

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
            decode_native_caware = (
                bool(decode_execution.get("native_caware_decode", True))
                if isinstance(decode_execution, dict)
                else True
            )
            decode_blockers = (
                [str(blocker) for blocker in decode_execution.get("blockers", [])]
                if isinstance(decode_execution, dict) and isinstance(decode_execution.get("blockers"), list)
                else []
            )
            if full_attention_path in {"per_row_splitk_fallback", "per_row_context_fallback"}:
                row_execution = "native_linear_batch_with_per_row_full_attention_fallback"
                native_caware_decode = False
                blockers.append("full-attention decode used a per-row fallback, so this is not native c-aware decode")
            elif not decode_native_caware:
                row_execution = "native_batch_with_diagnostic_fallback"
                native_caware_decode = False
                blockers.extend(decode_blockers or ["native c>N decode used a diagnostic fallback"])
            else:
                row_execution = "native_compact_caware_layers"
                native_caware_decode = True
            if projection_rows is not None:
                projection_candidates, projection_candidate_blockers = _env_projection_dispatch_candidates()
                projection_decision = plan_projection_dispatch(
                    rows=int(projection_rows),
                    row_gemv=ProjectionKernelSelection("linear", "w4_paro", "row_gemv"),
                    candidates=projection_candidates,
                )
                projection_dispatch = projection_decision.to_json_dict()
                blockers.extend(f"projection dispatch: {blocker}" for blocker in projection_candidate_blockers)
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
            decode_execution=self._batch_decode_execution_with_sampler_audit(decode_execution) if isinstance(decode_execution, dict) else None,
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
                "int8_prefill_oracle": self._prefill_int8_uses_oracle_attention(len(tokens)),
                "int8_prefill_attention": self._prefill_int8_attention_path(len(tokens)),
                "int8_prefill_attention_env": (
                    os.environ.get(_INT8_PREFILL_ATTENTION_ENV, "auto")
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "int8_prefill_streaming_min_tokens": (
                    _env_int(
                        _INT8_PREFILL_STREAMING_MIN_TOKENS_ENV,
                        _INT8_PREFILL_STREAMING_MIN_TOKENS_DEFAULT,
                    )
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "int8_prefill_memory_pressure": (
                    self._prefill_int8_memory_pressure(len(tokens))
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "int8_prefill_low_memory_total_gib": (
                    _env_float(
                        _INT8_PREFILL_LOW_MEMORY_TOTAL_GIB_ENV,
                        _INT8_PREFILL_LOW_MEMORY_TOTAL_GIB_DEFAULT,
                    )
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "int8_prefill_oracle_reserve_mib": (
                    _env_int(
                        _INT8_PREFILL_ORACLE_RESERVE_MIB_ENV,
                        _INT8_PREFILL_ORACLE_RESERVE_MIB_DEFAULT,
                    )
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
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

    def _resident_tensor_view_cache_enabled(self) -> bool:
        return bool(getattr(self, "_resident_tensor_view_cache_enabled_value", False))

    def _clear_resident_tensor_view_caches(self) -> None:
        for name in ("_slot_linear_state_cache", "_slot_full_cache_cache", "_full_cache_all_slots_cache"):
            cache = getattr(self, name, None)
            if cache is not None:
                cache.clear()

    def _slot_linear_state(self, layer_id: int, slot: int) -> tuple[Tensor, Tensor]:
        self._check_slot(slot)
        key = (int(layer_id), int(slot))
        if self._resident_tensor_view_cache_enabled():
            cached = self._slot_linear_state_cache.get(key)
            if cached is not None:
                return cached
        conv_state, recurrent_state, conv_buf, recurrent_buf, _conv_zero, _recurrent_zero = self.linear_states[layer_id]
        conv_nbytes = int(np.prod(conv_state.shape)) * conv_state.dtype.itemsize
        recurrent_nbytes = int(np.prod(recurrent_state.shape)) * recurrent_state.dtype.itemsize
        views = (
            Tensor.from_handle(conv_buf.ptr + int(slot) * conv_nbytes, conv_state.shape, conv_state.dtype, conv_state.device),
            Tensor.from_handle(
                recurrent_buf.ptr + int(slot) * recurrent_nbytes,
                recurrent_state.shape,
                recurrent_state.dtype,
                recurrent_state.device,
            ),
        )
        if self._resident_tensor_view_cache_enabled():
            self._slot_linear_state_cache[key] = views
        return views

    def _slot_full_cache(self, layer_id: int, slot: int) -> tuple[Tensor, Tensor]:
        self._check_slot(slot)
        key = (int(layer_id), int(slot))
        if self._resident_tensor_view_cache_enabled():
            cached = self._slot_full_cache_cache.get(key)
            if cached is not None:
                return cached
        key_cache, value_cache, key_buf, value_buf = self.full_caches[layer_id]
        cache_nbytes = int(np.prod(key_cache.shape)) * key_cache.dtype.itemsize
        views = (
            Tensor.from_handle(key_buf.ptr + int(slot) * cache_nbytes, key_cache.shape, key_cache.dtype, key_cache.device),
            Tensor.from_handle(value_buf.ptr + int(slot) * cache_nbytes, value_cache.shape, value_cache.dtype, value_cache.device),
        )
        if self._resident_tensor_view_cache_enabled():
            self._slot_full_cache_cache[key] = views
        return views

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
        """Return temporary BF16 K/V cache used only for gated INT8 native prefill attention."""

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
        cache_key = int(layer_id)
        if self._resident_tensor_view_cache_enabled():
            cached = self._full_cache_all_slots_cache.get(cache_key)
            if cached is not None:
                return cached
        key_cache, value_cache, key_buf, value_buf = self.full_caches[layer_id]
        shape = (self.max_batch_size * self.blocks, self.block_size, self.config.num_key_value_heads, self.config.head_dim)
        views = (
            Tensor.from_handle(key_buf.ptr, shape, key_cache.dtype, key_cache.device),
            Tensor.from_handle(value_buf.ptr, shape, value_cache.dtype, value_cache.device),
        )
        if self._resident_tensor_view_cache_enabled():
            self._full_cache_all_slots_cache[cache_key] = views
        return views

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

    def _prefill_int8_oracle_bytes(self, tokens: int) -> int:
        blocks = max(1, (int(tokens) + self.block_size - 1) // self.block_size)
        return int(
            2
            * blocks
            * self.block_size
            * int(self.config.num_key_value_heads)
            * int(self.config.head_dim)
            * DType.BF16.itemsize
        )

    def _prefill_int8_memory_pressure(self, tokens: int) -> bool:
        if self.kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
            return False
        try:
            free_bytes, total_bytes = self.runtime.mem_get_info()
        except Exception:
            return False
        total_limit_gib = max(
            0.0,
            _env_float(
                _INT8_PREFILL_LOW_MEMORY_TOTAL_GIB_ENV,
                _INT8_PREFILL_LOW_MEMORY_TOTAL_GIB_DEFAULT,
            ),
        )
        if total_limit_gib > 0.0 and int(total_bytes) <= int(total_limit_gib * 1024**3):
            return True
        reserve_mib = max(
            0,
            _env_int(
                _INT8_PREFILL_ORACLE_RESERVE_MIB_ENV,
                _INT8_PREFILL_ORACLE_RESERVE_MIB_DEFAULT,
            ),
        )
        oracle_plus_reserve = self._prefill_int8_oracle_bytes(tokens) + reserve_mib * 1024**2
        return int(free_bytes) <= int(oracle_plus_reserve)

    def _prefill_int8_attention_path(self, tokens: int) -> str | None:
        if self.kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
            return None
        value = os.environ.get(_INT8_PREFILL_ATTENTION_ENV, "auto").strip().lower()
        if value in {"", "auto"}:
            min_tokens = max(
                0,
                _env_int(
                    _INT8_PREFILL_STREAMING_MIN_TOKENS_ENV,
                    _INT8_PREFILL_STREAMING_MIN_TOKENS_DEFAULT,
                ),
            )
            if int(tokens) < min_tokens:
                return "oracle_bf16"
            return "streaming_direct" if self._prefill_int8_memory_pressure(tokens) else "oracle_bf16"
        if value in {"streaming", "streaming_direct", "direct", "direct_streaming"}:
            return "streaming_direct"
        if value in {"oracle", "bf16_oracle", "aotriton", "oracle_aotriton"}:
            return "oracle_bf16"
        raise ValueError(
            f"{_INT8_PREFILL_ATTENTION_ENV} must be auto, streaming, or oracle "
            f"(got {value!r})"
        )

    def _prefill_int8_uses_direct_attention(self, tokens: int) -> bool:
        return self._prefill_int8_attention_path(tokens) == "streaming_direct"

    def _prefill_int8_uses_oracle_attention(self, tokens: int) -> bool:
        return self._prefill_int8_attention_path(tokens) == "oracle_bf16"

    def _prefill_use_aotriton_attention_resolved(self, tokens: int) -> bool:
        if not self._prefill_use_aotriton_attention(tokens):
            return False
        if self.kv_storage_dtype == DType.BF16:
            return True
        # AOTriton consumes BF16 K/V. For INT8-retained sessions, use it only
        # when the INT8 prefill gate selects the temporary BF16 oracle bridge;
        # the slow direct streaming INT8 path remains available for explicit
        # diagnostics and very-long memory-gate prompts.
        return self._prefill_int8_uses_oracle_attention(tokens)

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
        # Prefill overwrote the shared block-table buffer; force the decode
        # block-table cache to rebuild on the next _batch_full_spans call.
        self._decode_full_block_table_key = None
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
            results.append(self._sample_from_hidden_for_slot(final_hidden, slot))
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
            include_tree_state=False,
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

    def _verify_mlp_scratch_policy(self, rows: int) -> str:
        if int(getattr(self.config, "num_experts", 1) or 0) <= 0:
            return "dense"
        if not _env_flag("HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED", True):
            return "c1" if int(rows) == 1 else "grouped"
        return "grouped" if int(rows) >= _verify_moe_grouped_min_tokens() else "c1"

    def _reserve_verify_mlp_scratch(self, state: Qwen35ParoDecodeState, *, rows: int):
        policy = self._verify_mlp_scratch_policy(rows)
        if policy == "dense":
            return state.reserve_dense_mlp_scratch(tokens=rows, activation_dtype=DType.FP16)
        if policy == "grouped":
            return state.reserve_moe_grouped_prefill_scratch(tokens=rows, activation_dtype=DType.FP16)
        return state.reserve_moe_c1_scratch(tokens=rows, activation_dtype=DType.FP16)

    def _clear_verify_scratch_caches(self) -> None:
        self._verify_scratch_cache_generation = int(getattr(self, "_verify_scratch_cache_generation", 0)) + 1
        self._verify_linear_scratch_cache.clear()
        self._verify_mlp_scratch_cache.clear()

    def _verify_scratch_generation_stamp_enabled(self) -> bool:
        return _env_flag("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", True)

    @staticmethod
    def _workspace_tensor_matches(state: Qwen35ParoDecodeState, name: str, tensor: Tensor) -> bool:
        try:
            current = state.workspace.allocation(name).tensor
        except KeyError:
            return False
        return (
            current.ptr == tensor.ptr
            and current.shape == tensor.shape
            and current.dtype == tensor.dtype
            and current.device == tensor.device
        )

    def _verify_linear_attention_scratch(
        self,
        layer_id: int,
        state: Qwen35ParoDecodeState,
        *,
        rows: int,
    ) -> Qwen35ParoLinearAttentionScratch:
        rows = int(rows)
        if self._verify_scratch_cache_enabled():
            key = (int(layer_id), rows)
            cached_entry = self._verify_linear_scratch_cache.get(key)
            cached_generation, cached = cached_entry if cached_entry is not None else (-1, None)
            generation_matches = cached_generation == int(getattr(self, "_verify_scratch_cache_generation", 0))
            use_generation_stamp = self._verify_scratch_generation_stamp_enabled()
            if (
                isinstance(cached, Qwen35ParoLinearAttentionScratch)
                and cached.attn_input.shape[0] == rows
                and cached.attn_input.dtype == DType.FP16
                and (
                    (use_generation_stamp and generation_matches)
                    or (
                        not use_generation_stamp
                        and self._workspace_tensor_matches(state, "linear_attn.attn_input", cached.attn_input)
                        and self._workspace_tensor_matches(state, "linear_attn.qkv_z", cached.qkv_z)
                        and self._workspace_tensor_matches(state, "linear_attn.tree_recurrent_state", cached.tree_recurrent_state)
                    )
                )
            ):
                return cached
        scratch = state.reserve_linear_attention_scratch(tokens=rows, activation_dtype=DType.FP16)
        if self._verify_scratch_cache_enabled():
            self._verify_linear_scratch_cache[(int(layer_id), rows)] = (
                int(getattr(self, "_verify_scratch_cache_generation", 0)),
                scratch,
            )
        return scratch

    def _verify_mlp_scratch(
        self,
        layer_id: int,
        state: Qwen35ParoDecodeState,
        *,
        rows: int,
    ) -> Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch:
        rows = int(rows)
        policy = self._verify_mlp_scratch_policy(rows)
        if self._verify_scratch_cache_enabled():
            key = (int(layer_id), rows, policy)
            cached_entry = self._verify_mlp_scratch_cache.get(key)
            cached_generation, cached = cached_entry if cached_entry is not None else (-1, None)
            generation_matches = cached_generation == int(getattr(self, "_verify_scratch_cache_generation", 0))
            use_generation_stamp = self._verify_scratch_generation_stamp_enabled()
            if policy == "dense" and isinstance(cached, Qwen35ParoDenseMlpScratch):
                alloc_name = "dense_mlp.normed"
            elif policy == "grouped" and isinstance(cached, Qwen35ParoGroupedMoeScratch):
                alloc_name = "moe.grouped.normed"
            elif policy == "c1" and isinstance(cached, Qwen35ParoMoeScratch):
                alloc_name = "moe.normed"
            else:
                alloc_name = ""
            if (
                alloc_name
                and cached.normed.shape[0] == rows
                and cached.normed.dtype == DType.FP16
                and (
                    (use_generation_stamp and generation_matches)
                    or (
                        not use_generation_stamp
                        and self._workspace_tensor_matches(state, alloc_name, cached.normed)
                    )
                )
            ):
                return cached
        scratch = self._reserve_verify_mlp_scratch(state, rows=rows)
        if self._verify_scratch_cache_enabled():
            self._verify_mlp_scratch_cache[(int(layer_id), rows, policy)] = (
                int(getattr(self, "_verify_scratch_cache_generation", 0)),
                scratch,
            )
        return scratch

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

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        """Allocate serving scratch for an admitted c=1 request shape.

        This is a startup/readiness probe, not a generation path: it reserves the
        prompt-length-dependent prefill buffers and workspaces without launching a
        full prompt prefill or decoding to the output limit.  For long prompts it
        mirrors the real prefill path's workspace lifetime: the prompt hidden
        buffer stays live, while prefill workspaces are released between adjacent
        layer-type phases when `_run_native_prefill_layers()` would do so.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        prompt_rows = max(1, int(max_prompt_tokens))
        new_tokens = max(0, int(max_new_tokens))
        batch_size = max(1, int(max_batch_size))
        if batch_size > self.max_batch_size:
            raise ValueError("scratch probe max_batch_size exceeds resident session max_batch_size")
        if prompt_rows > self.max_sequence_length:
            raise ValueError("scratch probe prompt tokens exceed resident session capacity")
        if prompt_rows + new_tokens + 1 > self.max_sequence_length:
            raise ValueError("scratch probe request shape exceeds resident session capacity")

        self._resolve_prefill_config_for_length(prompt_rows)
        layer_types = tuple(str(item) for item in self.config.layer_types[: len(self.states)])
        phase_order: list[str] = []
        for layer_type in layer_types:
            if layer_type not in {"linear_attention", "full_attention"}:
                continue
            if not phase_order or phase_order[-1] != layer_type:
                phase_order.append(layer_type)
        has_linear = "linear_attention" in phase_order
        has_full = "full_attention" in phase_order
        linear_chunk_rows = 0
        full_chunk_rows = 0
        int8_oracle_bytes = 0
        linear_tree_state_rows = 0
        linear_tree_state_bytes = 0
        linear_tree_state_full_bytes = 0
        live_memory_samples: list[dict[str, int | float | str]] = []
        minimize_workspace_overlap = self._should_minimize_prefill_workspace_overlap(prompt_rows)

        def capture_live_memory(stage: str) -> dict[str, int | float | str] | None:
            self.runtime.device_synchronize()
            try:
                free_bytes, total_bytes = self.runtime.mem_get_info()
            except Exception:
                return None
            free = int(free_bytes)
            total = int(total_bytes)
            sample: dict[str, int | float | str] = {
                "stage": str(stage),
                "free_bytes": free,
                "total_bytes": total,
                "used_bytes": max(0, total - free),
                "free_gib": round(free / 1024**3, 6),
                "used_gib": round(max(0, total - free) / 1024**3, 6),
            }
            live_memory_samples.append(sample)
            return sample

        try:
            if minimize_workspace_overlap:
                self._release_decode_scratch_for_prefill()
            self._prepare_prefill_context_counts(prompt_rows, stream=0)
            self._prefill_hidden_view_for_rows(prompt_rows)
            capture_live_memory("prefill_hidden_live")

            if has_linear:
                linear_chunk = self._linear_prefill_layer_chunk_size(prompt_rows)
                min_rows = int(getattr(self.config, "linear_conv_kernel_dim", 1))
                linear_chunk_rows = max(
                    end - start
                    for start, end in self._chunk_ranges(
                        prompt_rows,
                        linear_chunk,
                        min_chunk_size=min_rows,
                    )
                )
            if has_full:
                full_chunk = self._full_attention_prefill_layer_chunk_size(prompt_rows)
                full_chunk_rows = max(
                    end - start
                    for start, end in self._chunk_ranges(
                        prompt_rows,
                        full_chunk,
                        min_chunk_size=2,
                    )
                )

            previous_phase: str | None = None
            for phase in phase_order:
                if minimize_workspace_overlap and previous_phase is not None and phase != previous_phase:
                    self._release_prefill_workspace()
                    capture_live_memory(f"after_{previous_phase}_workspace_release")
                previous_phase = phase
                if phase == "linear_attention":
                    linear_scratch = self._ensure_linear_prefill_scratch(tokens=linear_chunk_rows)
                    self._ensure_moe_prefill_scratch(None, tokens=linear_chunk_rows)
                    linear_tree_state_rows = int(linear_scratch.tree_recurrent_state.shape[0])
                    linear_tree_state_bytes = int(
                        linear_scratch.tree_conv_state.numel * linear_scratch.tree_conv_state.dtype.itemsize
                        + linear_scratch.tree_recurrent_state.numel * linear_scratch.tree_recurrent_state.dtype.itemsize
                        + linear_scratch.tree_gdn_acc.numel * linear_scratch.tree_gdn_acc.dtype.itemsize
                    )
                    if linear_chunk_rows:
                        full_tree_rows = int(linear_chunk_rows)
                        linear_tree_state_full_bytes = int(
                            full_tree_rows
                            * (
                                int(np.prod(linear_scratch.tree_conv_state.shape[1:]))
                                * linear_scratch.tree_conv_state.dtype.itemsize
                                + int(np.prod(linear_scratch.tree_recurrent_state.shape[1:]))
                                * linear_scratch.tree_recurrent_state.dtype.itemsize
                                + int(np.prod(linear_scratch.tree_gdn_acc.shape[1:]))
                                * linear_scratch.tree_gdn_acc.dtype.itemsize
                            )
                        )
                    capture_live_memory("linear_prefill_scratch_live")
                elif phase == "full_attention":
                    use_aotriton_attention = self._prefill_use_aotriton_attention_resolved(prompt_rows)
                    self._ensure_full_prefill_scratch(
                        tokens=full_chunk_rows,
                        aotriton_attention=use_aotriton_attention,
                    )
                    self._ensure_moe_prefill_scratch(None, tokens=full_chunk_rows)
                    if self._prefill_int8_uses_oracle_attention(prompt_rows):
                        key, value = self._prefill_int8_oracle_cache(0, total_tokens=prompt_rows)
                        int8_oracle_bytes = int(
                            key.numel * key.dtype.itemsize + value.numel * value.dtype.itemsize
                        )
                    capture_live_memory("full_prefill_scratch_live")

            peak_memory = None
            if live_memory_samples:
                peak_memory = max(live_memory_samples, key=lambda item: int(item.get("used_bytes", 0) or 0))
            return {
                "max_prompt_tokens": prompt_rows,
                "max_new_tokens": new_tokens,
                "max_batch_size": batch_size,
                "prefill_hidden_bytes": int(prompt_rows * self.hidden_nbytes),
                "linear_prefill_chunk_rows": int(linear_chunk_rows),
                "full_prefill_chunk_rows": int(full_chunk_rows),
                "linear_prefill_tree_state_rows": int(linear_tree_state_rows),
                "linear_prefill_tree_state_bytes": int(linear_tree_state_bytes),
                "linear_prefill_tree_state_full_bytes": int(linear_tree_state_full_bytes),
                "linear_prefill_tree_state_saved_bytes": int(max(0, linear_tree_state_full_bytes - linear_tree_state_bytes)),
                "int8_oracle_bytes": int(int8_oracle_bytes),
                "int8_prefill_attention": self._prefill_int8_attention_path(prompt_rows),
                "int8_prefill_attention_env": (
                    os.environ.get(_INT8_PREFILL_ATTENTION_ENV, "auto")
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "int8_prefill_streaming_min_tokens": (
                    _env_int(
                        _INT8_PREFILL_STREAMING_MIN_TOKENS_ENV,
                        _INT8_PREFILL_STREAMING_MIN_TOKENS_DEFAULT,
                    )
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "int8_prefill_memory_pressure": (
                    self._prefill_int8_memory_pressure(prompt_rows)
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "int8_prefill_low_memory_total_gib": (
                    _env_float(
                        _INT8_PREFILL_LOW_MEMORY_TOTAL_GIB_ENV,
                        _INT8_PREFILL_LOW_MEMORY_TOTAL_GIB_DEFAULT,
                    )
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "int8_prefill_oracle_reserve_mib": (
                    _env_int(
                        _INT8_PREFILL_ORACLE_RESERVE_MIB_ENV,
                        _INT8_PREFILL_ORACLE_RESERVE_MIB_DEFAULT,
                    )
                    if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
                    else None
                ),
                "decode_scratch_released_for_probe": bool(minimize_workspace_overlap),
                "workspace_overlap_minimized": bool(minimize_workspace_overlap),
                "prefill_phase_order": list(phase_order),
                "prefill_chunk_tuning": dict(getattr(self, "prefill_chunk_tuning", {}) or {}),
                "live_memory": peak_memory,
                "live_memory_samples": list(live_memory_samples),
                "release_after_probe": bool(release_after_probe),
            }
        finally:
            if release_after_probe:
                self._restore_decode_scratch_after_prefill()
            elif minimize_workspace_overlap:
                self._reserve_decode_scratch_after_prefill()

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

    def _trace_decode_linear_output(self, *, layer_id: int, hidden: Tensor, rows: int, stream: int = 0) -> None:
        self._trace_linear_input_bits(
            trace_attr="_decode_linear_output_trace",
            layer_id=layer_id,
            hidden=hidden,
            rows=rows,
            stream=stream,
        )

    def _trace_decode_linear_tensor(
        self,
        *,
        layer_id: int,
        stage: str,
        tensor: Tensor,
        rows: int,
        stream: int = 0,
    ) -> None:
        if stage not in {
            "attn_input",
            "qkv",
            "z",
            "conv_out",
            "recurrent_out",
            "out_proj",
            "residual",
            "mlp_input",
            "output",
        }:
            raise ValueError("decode linear-attention trace stage is not recognized")
        if tensor.dtype == DType.FP32:
            self._trace_tensor_f32(
                trace_attr="_decode_linear_stage_trace",
                layer_id=layer_id,
                stage=stage,
                tensor=tensor,
                rows=rows,
                stream=stream,
            )
            return
        if tensor.dtype.itemsize == 2:
            self._trace_tensor_bits(
                trace_attr="_decode_linear_stage_trace",
                layer_id=layer_id,
                stage=stage,
                tensor=tensor,
                rows=rows,
                stream=stream,
            )
            return
        raise ValueError(f"decode linear-attention tensor trace {stage} does not support dtype {tensor.dtype}")

    def _trace_decode_linear_stages(
        self,
        *,
        layer_id: int,
        linear_scratch: Qwen35ParoLinearAttentionScratch,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch,
        output: Tensor,
        rows: int,
        stream: int = 0,
    ) -> None:
        if not isinstance(getattr(self, "_decode_linear_stage_trace", None), list):
            return
        for stage, tensor in (
            ("attn_input", linear_scratch.attn_input),
            ("qkv", linear_scratch.qkv),
            ("z", linear_scratch.z),
            ("conv_out", linear_scratch.conv_out),
            ("recurrent_out", linear_scratch.recurrent_out),
            ("out_proj", linear_scratch.out_proj),
            ("residual", moe_scratch.residual),
            ("mlp_input", moe_scratch.normed),
            ("output", output),
        ):
            self._trace_decode_linear_tensor(
                layer_id=layer_id,
                stage=stage,
                tensor=tensor,
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
        if stage not in {
            "input",
            "attn_input_pre_qkv",
            "attn_input_after_rotate",
            "attn_input_after_project",
            "attn_input_after_prepare",
            "attn_input",
            "gate",
            "gated_attn",
            "o_proj",
            "residual",
            "mlp_input",
            "output",
        }:
            raise ValueError("decode full-attention trace stage is not recognized")
        self._trace_tensor_bits(
            trace_attr="_decode_full_attention_trace",
            layer_id=layer_id,
            stage=stage,
            tensor=hidden,
            rows=rows,
            stream=stream,
        )

    def _trace_decode_full_attention_tensor(
        self,
        *,
        layer_id: int,
        stage: str,
        tensor: Tensor,
        rows: int,
        stream: int = 0,
    ) -> None:
        if stage not in {
            "q_proj_key_after_project",
            "value_after_project",
            "query_raw_after_split",
            "key_raw_after_cast",
            "gate_after_split",
            "query_after_prepare",
            "key_after_prepare",
        }:
            raise ValueError("decode full-attention tensor trace stage is not recognized")
        if tensor.dtype == DType.FP32:
            self._trace_tensor_f32(
                trace_attr="_decode_full_attention_trace",
                layer_id=layer_id,
                stage=stage,
                tensor=tensor,
                rows=rows,
                stream=stream,
            )
            return
        if tensor.dtype.itemsize == 2:
            self._trace_tensor_bits(
                trace_attr="_decode_full_attention_trace",
                layer_id=layer_id,
                stage=stage,
                tensor=tensor,
                rows=rows,
                stream=stream,
            )
            return
        raise ValueError(f"decode full-attention tensor trace {stage} does not support dtype {tensor.dtype}")

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

    def _trace_decode_full_attention_moe_scratch(
        self,
        *,
        layer_id: int,
        moe_scratch: Qwen35ParoMoeScratch | Qwen35ParoGroupedMoeScratch | Qwen35ParoDenseMlpScratch,
        rows: int,
        stream: int = 0,
    ) -> None:
        if not isinstance(getattr(self, "_decode_full_attention_trace", None), list):
            return
        self._trace_decode_full_attention(
            layer_id=layer_id,
            stage="residual",
            hidden=moe_scratch.residual,
            rows=rows,
            stream=stream,
        )
        self._trace_decode_full_attention(
            layer_id=layer_id,
            stage="mlp_input",
            hidden=moe_scratch.normed,
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
                direct_int8_prefill = self._prefill_int8_uses_direct_attention(tokens)
                if int8_retained and not direct_int8_prefill:
                    key_cache, value_cache = self._prefill_int8_oracle_cache(layer_id, total_tokens=tokens)
                    prefill_storage_dtype = DType.BF16
                    prefill_scale_metadata = None
                else:
                    key_cache, value_cache = retained_key_cache, retained_value_cache
                    prefill_storage_dtype = DType.INT8_PER_TOKEN_HEAD if int8_retained else DType.BF16
                    prefill_scale_metadata = self._slot_full_scale_metadata(layer_id, 0) if int8_retained else None
                chunk_size = self._full_attention_prefill_layer_chunk_size(tokens)
                for start, end in self._chunk_ranges(tokens, chunk_size, min_chunk_size=2):
                    rows = end - start
                    hidden_chunk = self._prefill_row_matrix_view(hidden, start, rows)
                    append_spans, prefill_spans = self._prefill_full_attention_spans(
                        rows,
                        start=start,
                        total_tokens=tokens,
                        storage_dtype=prefill_storage_dtype,
                        scale_metadata=prefill_scale_metadata,
                    )
                    retained_append_spans = None
                    if int8_retained and not direct_int8_prefill:
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
                        retained_key_cache=retained_key_cache if retained_append_spans is not None else None,
                        retained_value_cache=retained_value_cache if retained_append_spans is not None else None,
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
                        self._decode_full_block_table_key = None
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
                        linear_scratch=self._linear_decode_scratch(layer_id, state),
                        moe_scratch=self._mlp_decode_scratch(layer_id, state),
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
                        moe_scratch=self._mlp_decode_scratch(layer_id, state),
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
    def _reserve_decode_scratch_after_prefill(self) -> None:
        self._clear_verify_scratch_caches()
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

    def _restore_decode_scratch_after_prefill(self) -> None:
        self._release_prefill_workspace()
        self._release_prefill_hidden_buffer()
        self._reserve_decode_scratch_after_prefill()

    def _batch_decode_segment_metadata(
        self,
        *,
        rows: int,
        slots: tuple[int, ...],
    ) -> tuple[Tensor, Tensor, tuple[DeviceBuffer, ...]]:
        # cu_seqlens (arange(rows+1)) and state_indices (physical slot ids)
        # depend only on (rows, slots) and are identical across every decode
        # step for a fixed active batch.  Persist them in dedicated device
        # buffers and skip the per-step malloc/free + host->device copies when
        # the (rows, slots) key is unchanged.  This removes the last per-step
        # device allocation from the batch layer pass -- a capture-safety
        # prerequisite for c>1 decode graph replay (C3.0b) and an eager
        # host-overhead trim -- mirroring the cached full-attention block table
        # in _batch_full_spans.  The third return element is retained as an
        # (empty) buffer tuple so the call site's release loop is a no-op.
        seg_key = (int(rows),) + tuple(int(slot) for slot in slots)
        if getattr(self, "_decode_segment_metadata_key", None) != seg_key:
            cu_arr = np.arange(rows + 1, dtype=np.int32)
            state_arr = np.asarray(slots, dtype=np.int64)
            cu_capacity = int(self.max_batch_size + 1) * DType.INT32.itemsize
            cu_buf = getattr(self, "_decode_segment_cu_buf", None)
            if cu_buf is None or cu_buf.nbytes < cu_capacity:
                cu_buf = malloc(cu_capacity, runtime=self.runtime)
                self.buffers.append(cu_buf)
                self._decode_segment_cu_buf = cu_buf
            state_capacity = int(self.max_batch_size) * DType.INT64.itemsize
            state_buf = getattr(self, "_decode_segment_state_buf", None)
            if state_buf is None or state_buf.nbytes < state_capacity:
                state_buf = malloc(state_capacity, runtime=self.runtime)
                self.buffers.append(state_buf)
                self._decode_segment_state_buf = state_buf
            copy_host_to_device(cu_buf, host_array_ptr(cu_arr), cu_arr.nbytes, runtime=self.runtime)
            copy_host_to_device(state_buf, host_array_ptr(state_arr), state_arr.nbytes, runtime=self.runtime)
            self._decode_segment_cu_tensor = Tensor.from_handle(cu_buf.ptr, cu_arr.shape, DType.INT32, self.device)
            self._decode_segment_state_tensor = Tensor.from_handle(
                state_buf.ptr, state_arr.shape, DType.INT64, self.device
            )
            self._decode_segment_metadata_key = seg_key
        return self._decode_segment_cu_tensor, self._decode_segment_state_tensor, ()

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
        #
        # The block table depends only on (rows, slots); it is identical across
        # every decode step for a fixed active batch.  Cache one persistent
        # decode-only device buffer *per distinct (rows, slots) key* so the host
        # build + synchronous host->device copy happens once per key and never
        # repeats on the decode hot path.  A single-key cache thrashed under the
        # row-chunked full-attention path (which calls this with several
        # (chunk_rows, chunk_slots) within one step) -- re-copying every call,
        # which is both wasteful eagerly and *fatal under HIP graph capture*
        # (a host->device copy on a capturing stream is illegal).  Keying per
        # config means that once the eager warmup has touched every chunk key,
        # capture performs no allocation or copy.
        slots_key = (int(rows),) + tuple(int(slot) for slot in slots)
        block_cache = getattr(self, "_decode_full_block_table_cache", None)
        if block_cache is None:
            block_cache = {}
            self._decode_full_block_table_cache = block_cache
        block_entry = block_cache.get(slots_key)
        if block_entry is None:
            logical_blocks = np.arange(self.blocks, dtype=np.int32)
            block_rows = np.empty((rows, self.blocks), dtype=np.int32)
            for row, slot in enumerate(slots):
                delta_blocks = (int(slot) - row) * self.blocks
                if delta_blocks < 0:
                    raise ValueError("batch full-attention slots must be in physical-slot order")
                block_rows[row] = logical_blocks + np.int32(delta_blocks)
            block_buf = malloc(int(block_rows.nbytes), runtime=self.runtime)
            self.buffers.append(block_buf)
            copy_host_to_device(block_buf, host_array_ptr(block_rows), block_rows.nbytes, runtime=self.runtime)
            block_entry = {
                "buf": block_buf,
                "block_table": Tensor.from_handle(block_buf.ptr, (rows, self.blocks), DType.INT32, self.device),
                "block_rows_list": block_rows.astype(np.int32, copy=False).tolist(),
            }
            block_cache[slots_key] = block_entry
        block_table = block_entry["block_table"]
        self._decode_full_block_rows_list = block_entry["block_rows_list"]
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
            "block_table_rows": self._decode_full_block_rows_list,
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
        row_chunked_full_attention_layers: list[int] = []
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        force_selected_c1_moe = (not dense_mlp) and rows > 1 and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE")
        force_per_row_linear_moe = (not dense_mlp) and rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR_MOE"
        )
        force_selected_c1_linear_projections = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_PROJECTIONS"
        )
        force_selected_c1_qkv_z_linear_projections = (
            rows > 1
            and not force_selected_c1_linear_projections
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ")
        )
        force_selected_c1_qkv_z_linear_input = (
            rows > 1
            and not force_selected_c1_linear_projections
            and not force_selected_c1_qkv_z_linear_projections
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ_INPUT")
        )
        force_selected_c1_qkv_linear_projections = (
            rows > 1
            and not force_selected_c1_linear_projections
            and not force_selected_c1_qkv_z_linear_projections
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKV")
        )
        force_selected_c1_z_linear_projections = (
            rows > 1
            and not force_selected_c1_linear_projections
            and not force_selected_c1_qkv_z_linear_projections
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_Z")
        )
        force_selected_c1_ab_linear_projections = (
            rows > 1
            and not force_selected_c1_linear_projections
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_AB")
        )
        force_batch_gemv_linear_projections = (
            rows > 1
            and not force_selected_c1_linear_projections
            and not force_selected_c1_qkv_z_linear_projections
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_LINEAR_PROJECTIONS")
        )
        use_batch_gemv_linear_projections = (
            rows > 1
            and not force_selected_c1_linear_projections
            and not force_selected_c1_qkv_z_linear_projections
        )
        force_selected_c1_linear_state = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_STATE"
        )
        linear_out_env = os.environ.get("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT", "auto")
        linear_out_mode = linear_out_env.strip().lower()
        force_batch_gemv_linear_out = False
        if linear_out_mode in {"", "auto"}:
            force_selected_c1_linear_out: bool | None = None
        elif linear_out_mode in {"1", "true", "on", "yes", "selected_c1"}:
            force_selected_c1_linear_out = rows > 1
        elif linear_out_mode in {"0", "false", "off", "no", "batch"}:
            force_selected_c1_linear_out = False
        elif linear_out_mode in {"batch_gemv", "gemv"}:
            force_selected_c1_linear_out = False
            force_batch_gemv_linear_out = rows > 1
        else:
            raise ValueError(
                "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT must be auto, batch, batch_gemv, or selected_c1"
            )
        linear_attention_projection_path = (
            "selected_c1_forced"
            if force_selected_c1_linear_projections
            else (
                "selected_c1_qkv_z"
                if force_selected_c1_qkv_z_linear_projections
                else (
                    "selected_c1_qkv_z_input"
                    if force_selected_c1_qkv_z_linear_input
                    else "selected_c1_qkv"
                    if force_selected_c1_qkv_linear_projections and not force_selected_c1_z_linear_projections
                    else "selected_c1_z"
                    if force_selected_c1_z_linear_projections and not force_selected_c1_qkv_linear_projections
                    else "selected_c1_qkv_plus_z"
                    if force_selected_c1_qkv_linear_projections and force_selected_c1_z_linear_projections
                    else "batch_gemv_selected_c1_ab"
                    if force_batch_gemv_linear_projections and force_selected_c1_ab_linear_projections
                    else (
                        "batch_gemv"
                        if force_batch_gemv_linear_projections
                        else "selected_c1_ab" if force_selected_c1_ab_linear_projections else "native_batch"
                    )
                )
            )
        )
        linear_attention_state_path = "selected_c1_forced" if force_selected_c1_linear_state else "native_segments"
        linear_attention_output_path = (
            "selected_c1_forced"
            if (force_selected_c1_linear_state if force_selected_c1_linear_out is None else force_selected_c1_linear_out)
            else (
                "batch_gemv_from_f32"
                if force_selected_c1_linear_state and force_batch_gemv_linear_out
                else "batch_from_f32" if force_selected_c1_linear_state else ("batch_gemv" if force_batch_gemv_linear_out else "native_batch")
            )
        )
        force_per_row_linear = _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR")
        force_per_row_full_attention_input = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT"
        )
        force_per_row_full_attention_qkv = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_QKV"
        )
        force_per_row_full_attention_scratch = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SCRATCH"
        )
        force_per_row_full_attention_batch_scratch = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_BATCH_SCRATCH"
        )
        force_per_row_full_attention_attn_batch_moe = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_MOE"
        )
        force_per_row_full_attention_attn_batch_post_moe = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_POST_MOE"
        )
        force_per_row_full_attention_attn_batch_o_post_moe = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_O_POST_MOE"
        )
        force_per_row_full_attention_preqkv_append_batch_context_o_post_moe = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_BATCH_CONTEXT_O_POST_MOE"
        )
        force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_CONTEXT_BATCH_GATE_O_POST_MOE"
        )
        force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_CONTEXT_GATE_BATCH_O_POST_MOE"
        )
        force_per_row_full_attention_persistent_scratch = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PERSISTENT_SCRATCH"
        )
        force_per_row_full_attention_skip_batch_setup = (
            force_per_row_full_attention_persistent_scratch
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SKIP_BATCH_SETUP")
        )
        force_per_row_full_attention_context = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_CONTEXT"
        )
        force_per_row_full_attention_context_only = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_CONTEXT_ONLY"
        )
        force_per_row_full_attention_dense_context_only = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_ONLY"
        )
        force_per_row_full_attention_dense_context_batch_gate = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_BATCH_GATE"
        )
        force_per_row_full_attention_dense_context_batch_gate_layers = (
            _env_int_set("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_BATCH_GATE_LAYERS")
            if rows > 1
            else set()
        )
        force_per_row_full_attention_paged_context_only = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PAGED_CONTEXT_ONLY"
        )
        force_per_row_full_attention_dense_context_layers = (
            _env_int_set("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_LAYERS")
            if rows > 1
            else set()
        )
        force_batch_temp_full_attention_context = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_BATCH_TEMP_FULL_ATTN_CONTEXT"
        )
        force_batch_compact_full_attention_context = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_BATCH_COMPACT_FULL_ATTN_CONTEXT"
        )
        force_per_row_full_attention_gate = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_GATE"
        )
        force_per_row_full_attention_kv_append = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_KV_APPEND"
        )
        force_per_row_full_attention_append_context = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_APPEND_CONTEXT"
        )
        force_per_row_full_attention_suffix = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SUFFIX"
        )
        force_per_row_full_attention_output = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_OUTPUT"
        )
        force_native_full_attention_output = (
            rows > 1
            and not force_per_row_full_attention_output
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_NATIVE_FULL_ATTN_OUTPUT")
        )
        force_batch_gemv_full_attention_output = (
            rows > 1
            and not force_per_row_full_attention_output
            and not force_native_full_attention_output
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_FULL_ATTN_OUTPUT")
        )
        auto_batch_gemv_full_attention_output = (
            rows == 2
            and not force_per_row_full_attention_output
            and not force_native_full_attention_output
            and not force_batch_gemv_full_attention_output
        )
        use_batch_gemv_full_attention_output = (
            force_batch_gemv_full_attention_output or auto_batch_gemv_full_attention_output
        )
        force_native_row_chunk_full_attention_output = (
            rows > 1
            and not force_per_row_full_attention_output
            and not use_batch_gemv_full_attention_output
            and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_NATIVE_ROW_CHUNK_FULL_ATTN_OUTPUT")
        )
        force_per_row_full_attention_layer_copy = rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_LAYER_COPY"
        )
        force_per_row_post_attention = rows > 1 and _env_flag("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN")
        force_per_row_full_attention_moe = (not dense_mlp) and rows > 1 and _env_flag(
            "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_MOE"
        )
        full_attention_row_chunk_env = "HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_SIZE"
        full_attention_row_chunk_size = _env_int(
            full_attention_row_chunk_env,
            0,
        )
        if full_attention_row_chunk_size < 0:
            raise ValueError("HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_SIZE must be non-negative")
        full_attention_row_chunk_env_value = os.environ.get(full_attention_row_chunk_env)
        auto_full_attention_row_chunks = (
            rows in {3, 4, 5, 6, 7, 8}
            and full_attention_row_chunk_size == 0
            and (full_attention_row_chunk_env_value is None or full_attention_row_chunk_env_value.strip() == "")
        )
        if auto_full_attention_row_chunks:
            full_attention_row_chunk_size = 2
        force_full_attention_row_chunks = rows > 1 and 0 < full_attention_row_chunk_size < rows
        full_attention_row_chunk_source = "auto" if auto_full_attention_row_chunks else "env"
        full_attention_row_chunk_layers = (
            _env_int_set("HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_LAYERS") if rows > 1 else set()
        )
        full_attention_input_decode_path = (
            "per_row_rmsnorm_fallback" if force_per_row_full_attention_input else "native_batch"
        )
        full_attention_qkv_decode_path = (
            "per_row_qkv_scratch_fallback" if force_per_row_full_attention_qkv else "native_batch"
        )
        full_attention_scratch_decode_path = (
            "persistent_c1_scratch_fallback"
            if force_per_row_full_attention_persistent_scratch
            else "per_row_preqkv_append_context_gate_batch_o_post_moe_fallback"
            if force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe
            else "per_row_preqkv_append_context_batch_gate_o_post_moe_fallback"
            if force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe
            else "per_row_preqkv_append_batch_context_o_post_moe_fallback"
            if force_per_row_full_attention_preqkv_append_batch_context_o_post_moe
            else "per_row_attention_batch_o_post_moe_fallback"
            if force_per_row_full_attention_attn_batch_o_post_moe
            else "per_row_attention_batch_post_moe_fallback"
            if force_per_row_full_attention_attn_batch_post_moe
            else "per_row_attention_batch_moe_fallback"
            if force_per_row_full_attention_attn_batch_moe
            else "per_row_layer_batch_scratch_fallback"
            if force_per_row_full_attention_batch_scratch
            else "per_row_layer_scratch_fallback" if force_per_row_full_attention_scratch else "native_batch"
        )
        full_attention_context_decode_path = (
            "per_row_context_gate_fallback"
            if force_per_row_full_attention_context
            else "per_row_context_only_fallback"
            if force_per_row_full_attention_context_only
            else "per_row_dense_context_only_fallback"
            if force_per_row_full_attention_dense_context_only
            else "per_row_dense_context_batch_gate_fallback"
            if force_per_row_full_attention_dense_context_batch_gate
            else "per_row_paged_context_only_fallback"
            if force_per_row_full_attention_paged_context_only
            else "per_row_dense_context_layer_override"
            if force_per_row_full_attention_dense_context_layers
            else "per_row_dense_context_batch_gate_layer_override"
            if force_per_row_full_attention_dense_context_batch_gate_layers
            else "batch_temp_output_diagnostic"
            if force_batch_temp_full_attention_context
            else "batch_compact_cache_diagnostic" if force_batch_compact_full_attention_context else "native_batch"
        )
        full_attention_gate_decode_path = (
            "per_row_gate_fallback" if force_per_row_full_attention_gate else "native_batch"
        )
        full_attention_kv_append_decode_path = (
            "per_row_kv_append_fallback" if force_per_row_full_attention_kv_append else "native_batch"
        )
        full_attention_append_context_decode_path = (
            "per_row_append_context_interleaved" if force_per_row_full_attention_append_context else "phased"
        )
        full_attention_suffix_decode_path = (
            "per_row_suffix_interleaved" if force_per_row_full_attention_suffix else "phased"
        )
        full_attention_output_decode_path = (
            "per_row_o_projection_fallback"
            if force_per_row_full_attention_output
            else "batch_gemv"
            if force_batch_gemv_full_attention_output
            else "batch_gemv_auto"
            if auto_batch_gemv_full_attention_output
            else "native_batch_forced"
            if force_native_full_attention_output
            else "native_batch"
        )
        row_chunk_batch_gemv_full_attention_output = False
        row_chunk_native_full_attention_output = False
        full_attention_layer_copy_decode_path = (
            "per_row_layer_copy_fallback" if force_per_row_full_attention_layer_copy else "batch_copy"
        )
        post_attention_decode_path = "per_row_add_rmsnorm_fallback" if force_per_row_post_attention else "native_batch"
        use_single_row_c1_linear = rows == 1 and not force_per_row_linear
        use_per_row_linear = force_per_row_linear or use_single_row_c1_linear
        moe_decode_path = "dense_mlp" if dense_mlp else (
            "selected_c1"
            if rows == 1
            else (
                "selected_c1_per_row_moe_fallback"
                if (
                    force_per_row_linear_moe
                    or force_per_row_full_attention_moe
                    or force_per_row_full_attention_batch_scratch
                    or force_per_row_full_attention_persistent_scratch
                )
                else ("selected_c1_batch" if force_selected_c1_moe else "grouped_compact")
            )
        )
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
                            self._trace_decode_linear_stages(
                                layer_id=layer_id,
                                linear_scratch=linear_scratch,
                                moe_scratch=moe_scratch,
                                output=row_out,
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
                        copied_layer_output = True
                        self._trace_decode_linear_output(layer_id=layer_id, hidden=next_hidden, rows=rows, stream=stream)
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
                        if force_selected_c1_moe or force_per_row_linear_moe:
                            moe_scratch = self._ensure_moe_decode_batch_scratch(layer_id, rows, force_selected_c1_moe=True)
                        else:
                            moe_scratch = self._ensure_moe_decode_batch_scratch(layer_id, rows)
                        selected_c1_linear_state_pairs = (
                            tuple(self._slot_linear_state(layer_id, slot) for slot in slots)
                            if force_selected_c1_linear_state
                            else None
                        )
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
                            force_selected_c1_linear_projections=force_selected_c1_linear_projections,
                            force_selected_c1_qkv_z_linear_projections=force_selected_c1_qkv_z_linear_projections,
                            force_selected_c1_qkv_z_linear_input=force_selected_c1_qkv_z_linear_input,
                            force_selected_c1_qkv_linear_projections=force_selected_c1_qkv_linear_projections,
                            force_selected_c1_z_linear_projections=force_selected_c1_z_linear_projections,
                            force_selected_c1_ab_linear_projections=force_selected_c1_ab_linear_projections,
                            force_batch_gemv_linear_projections=use_batch_gemv_linear_projections,
                            force_selected_c1_linear_state=force_selected_c1_linear_state,
                            selected_c1_linear_state_pairs=selected_c1_linear_state_pairs,
                            force_selected_c1_linear_out=force_selected_c1_linear_out,
                            force_batch_gemv_linear_out=force_batch_gemv_linear_out,
                            force_per_row_moe=force_per_row_linear_moe,
                            library=self.libraries,
                            stream=stream,
                        )
                        self._trace_decode_linear_stages(
                            layer_id=layer_id,
                            linear_scratch=linear_scratch,
                            moe_scratch=moe_scratch,
                            output=out,
                            rows=rows,
                            stream=stream,
                        )
                        self._trace_decode_linear_output(layer_id=layer_id, hidden=out, rows=rows, stream=stream)
                        layer_moe_path = "dense_mlp" if dense_mlp else (
                            "selected_c1"
                            if rows == 1
                            else "selected_c1_per_row_moe_fallback" if force_per_row_linear_moe else ("selected_c1_batch" if force_selected_c1_moe else "grouped_compact")
                        )
                        if not dense_mlp and rows > 1:
                            if force_per_row_linear_moe:
                                moe_selected_c1_fallback_layers += 1
                            elif not force_selected_c1_moe:
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
                                "native_caware_decode": not (
                                    force_per_row_linear_moe
                                    or force_selected_c1_linear_projections
                                    or force_selected_c1_qkv_z_linear_projections
                                    or force_selected_c1_qkv_z_linear_input
                                    or force_selected_c1_qkv_linear_projections
                                    or force_selected_c1_z_linear_projections
                                    or force_selected_c1_ab_linear_projections
                                    or force_batch_gemv_linear_projections
                                    or force_selected_c1_linear_state
                                    or linear_attention_output_path in {"selected_c1_forced", "batch_gemv_from_f32"}
                                ),
                                "linear_attention_projection_path": linear_attention_projection_path,
                                "linear_attention_state_path": linear_attention_state_path,
                                "linear_attention_output_path": linear_attention_output_path,
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
                        layer_force_per_row_dense_context_only = (
                            force_per_row_full_attention_dense_context_only
                            or layer_id in force_per_row_full_attention_dense_context_layers
                        )
                        layer_force_per_row_dense_context_batch_gate = (
                            force_per_row_full_attention_dense_context_batch_gate
                            or layer_id in force_per_row_full_attention_dense_context_batch_gate_layers
                        )
                        layer_force_per_row_paged_context_only = force_per_row_full_attention_paged_context_only
                        if (
                            layer_id in force_per_row_full_attention_dense_context_layers
                            or layer_id in force_per_row_full_attention_dense_context_batch_gate_layers
                        ):
                            layer_force_per_row_paged_context_only = False
                        layer_full_attention_context_decode_path = (
                            "per_row_context_gate_fallback"
                            if force_per_row_full_attention_context
                            else "per_row_context_only_fallback"
                            if force_per_row_full_attention_context_only
                            else "per_row_dense_context_only_fallback"
                            if layer_force_per_row_dense_context_only
                            else "per_row_dense_context_batch_gate_fallback"
                            if layer_force_per_row_dense_context_batch_gate
                            else "per_row_paged_context_only_fallback"
                            if layer_force_per_row_paged_context_only
                            else "batch_temp_output_diagnostic"
                            if force_batch_temp_full_attention_context
                            else "batch_compact_cache_diagnostic"
                            if force_batch_compact_full_attention_context
                            else "native_batch"
                        )
                        layer_force_full_attention_row_chunks = force_full_attention_row_chunks and (
                            not full_attention_row_chunk_layers or layer_id in full_attention_row_chunk_layers
                        )
                        if layer_force_full_attention_row_chunks:
                            full_attention_decode_path = "native_batch_row_chunks"
                            row_chunked_full_attention_layers.append(int(layer_id))
                        elif full_attention_decode_path == "none":
                            full_attention_decode_path = "native_batch"
                        native_full_attention_layers += 1
                        persistent_attention_scratch = self.full_scratch[layer_id] if force_per_row_full_attention_persistent_scratch else None
                        persistent_moe_scratch = self.moe_scratch[layer_id] if force_per_row_full_attention_persistent_scratch else None
                        batch_full_spans_metadata = None
                        if force_per_row_full_attention_skip_batch_setup:
                            key_cache = value_cache = position_tensor = append_spans = decode_spans = None
                        else:
                            key_cache, value_cache = self._full_cache_all_slots(layer_id)
                            if layer_force_full_attention_row_chunks:
                                position_tensor = append_spans = decode_spans = None
                            else:
                                position_tensor, append_spans, decode_spans = self._batch_full_spans(
                                    layer_id,
                                    rows=rows,
                                    positions=positions,
                                    slots=slots,
                                )
                                batch_full_spans_metadata = getattr(self, "_last_batch_full_spans_metadata", None)
                        attention_scratch = (
                            persistent_attention_scratch
                            if force_per_row_full_attention_persistent_scratch
                            else self._ensure_full_decode_batch_scratch(layer_id, rows)
                        )
                        if force_per_row_full_attention_persistent_scratch:
                            moe_scratch = persistent_moe_scratch
                        elif (
                            force_selected_c1_moe
                            or force_per_row_full_attention_moe
                            or force_per_row_full_attention_scratch
                            or force_per_row_full_attention_batch_scratch
                        ):
                            moe_scratch = self._ensure_moe_decode_batch_scratch(layer_id, rows, force_selected_c1_moe=True)
                        else:
                            moe_scratch = self._ensure_moe_decode_batch_scratch(layer_id, rows)
                        post_input_rmsnorm_trace = None
                        input_scratch_trace = None
                        qkv_tensor_trace = None
                        per_row_contexts = None
                        per_row_append_contexts = None
                        if (
                            not force_per_row_full_attention_skip_batch_setup
                            and (
                                force_per_row_full_attention_context
                                or force_per_row_full_attention_context_only
                                or layer_force_per_row_dense_context_only
                                or layer_force_per_row_dense_context_batch_gate
                                or layer_force_per_row_paged_context_only
                                or force_batch_compact_full_attention_context
                                or force_per_row_full_attention_kv_append
                                or force_per_row_full_attention_suffix
                                or force_per_row_full_attention_scratch
                                or force_per_row_full_attention_batch_scratch
                                or force_per_row_full_attention_attn_batch_moe
                                or force_per_row_full_attention_attn_batch_post_moe
                                or force_per_row_full_attention_attn_batch_o_post_moe
                                or force_per_row_full_attention_preqkv_append_batch_context_o_post_moe
                                or force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe
                                or force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe
                                or force_per_row_full_attention_persistent_scratch
                            )
                        ):
                            per_row_contexts = [] if force_per_row_full_attention_context or force_per_row_full_attention_context_only or layer_force_per_row_dense_context_only or layer_force_per_row_dense_context_batch_gate or layer_force_per_row_paged_context_only or force_batch_compact_full_attention_context or force_per_row_full_attention_suffix or force_per_row_full_attention_scratch or force_per_row_full_attention_batch_scratch or force_per_row_full_attention_attn_batch_moe or force_per_row_full_attention_attn_batch_post_moe or force_per_row_full_attention_attn_batch_o_post_moe or force_per_row_full_attention_preqkv_append_batch_context_o_post_moe or force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe or force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe or force_per_row_full_attention_persistent_scratch else None
                            per_row_append_contexts = [] if force_per_row_full_attention_kv_append or force_per_row_full_attention_suffix or force_per_row_full_attention_scratch or force_per_row_full_attention_batch_scratch or force_per_row_full_attention_attn_batch_moe or force_per_row_full_attention_attn_batch_post_moe or force_per_row_full_attention_attn_batch_o_post_moe or force_per_row_full_attention_preqkv_append_batch_context_o_post_moe or force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe or force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe or force_per_row_full_attention_persistent_scratch else None
                            for slot in slots:
                                row_key_cache, row_value_cache = self._slot_full_cache(layer_id, slot)
                                _row_position, row_append_spans, row_decode_spans = self._slot_full_spans(layer_id, slot)
                                if per_row_contexts is not None:
                                    per_row_contexts.append((row_key_cache, row_value_cache, row_decode_spans))
                                if per_row_append_contexts is not None:
                                    per_row_append_contexts.append((row_key_cache, row_value_cache, row_append_spans))
                        if isinstance(getattr(self, "_decode_full_attention_trace", None), list):
                            def post_input_rmsnorm_trace(
                                attention_scratch: Qwen35ParoAttentionScratch,
                                *,
                                _layer_id: int = layer_id,
                                _rows: int = rows,
                                _stream: int = stream,
                            ) -> None:
                                self._trace_decode_full_attention(
                                    layer_id=_layer_id,
                                    stage="attn_input_pre_qkv",
                                    hidden=attention_scratch.attn_input,
                                    rows=_rows,
                                    stream=_stream,
                                )

                            def input_scratch_trace(
                                stage: str,
                                row: int,
                                attention_scratch: Qwen35ParoAttentionScratch,
                                *,
                                _layer_id: int = layer_id,
                                _stream: int = stream,
                            ) -> None:
                                self._trace_decode_full_attention(
                                    layer_id=_layer_id,
                                    stage=stage,
                                    hidden=attention_scratch.attn_input,
                                    rows=1,
                                    stream=_stream,
                                )

                            def qkv_tensor_trace(
                                stage: str,
                                row: int,
                                tensor: Tensor,
                                *,
                                _layer_id: int = layer_id,
                                _stream: int = stream,
                            ) -> None:
                                self._trace_decode_full_attention_tensor(
                                    layer_id=_layer_id,
                                    stage=stage,
                                    tensor=tensor,
                                    rows=1,
                                    stream=_stream,
                                )

                        copied_full_attention_output = False
                        layer_row_chunk_batch_gemv_full_attention_output = False
                        layer_row_chunk_auto_batch_gemv_full_attention_output = False
                        layer_row_chunk_native_full_attention_output = False
                        if layer_force_full_attention_row_chunks:
                            chunk_records: list[dict[str, Any]] = []
                            chunk_size = int(full_attention_row_chunk_size)
                            if key_cache is None or value_cache is None:
                                key_cache, value_cache = self._full_cache_all_slots(layer_id)
                            for chunk_start in range(0, rows, chunk_size):
                                chunk_end = min(rows, chunk_start + chunk_size)
                                chunk_rows = int(chunk_end - chunk_start)
                                chunk_slots = tuple(int(slot) for slot in slots[chunk_start:chunk_end])
                                chunk_positions = tuple(int(position) for position in positions[chunk_start:chunk_end])
                                chunk_hidden = Tensor.from_handle(
                                    hidden.ptr + chunk_start * self.hidden_nbytes,
                                    (chunk_rows, self.config.hidden_size),
                                    hidden.dtype,
                                    hidden.device,
                                )
                                chunk_position_tensor, chunk_append_spans, chunk_decode_spans = self._batch_full_spans(
                                    layer_id,
                                    rows=chunk_rows,
                                    positions=chunk_positions,
                                    slots=chunk_slots,
                                )
                                chunk_metadata = getattr(self, "_last_batch_full_spans_metadata", None)
                                chunk_records.append(
                                    {
                                        "row_start": int(chunk_start),
                                        "rows": int(chunk_rows),
                                        "slots": [int(slot) for slot in chunk_slots],
                                        "positions": [int(position) for position in chunk_positions],
                                        "segment_metadata": chunk_metadata,
                                    }
                                )
                                chunk_attention_scratch = self._ensure_full_decode_batch_scratch(layer_id, chunk_rows)
                                chunk_moe_scratch = self._ensure_moe_decode_batch_scratch(
                                    layer_id,
                                    chunk_rows,
                                    force_selected_c1_moe=(
                                        force_selected_c1_moe
                                        or force_per_row_full_attention_moe
                                        or force_per_row_full_attention_scratch
                                        or force_per_row_full_attention_batch_scratch
                                        or force_per_row_full_attention_suffix
                                    ),
                                )
                                chunk_post_input_rmsnorm_trace = None
                                chunk_input_scratch_trace = None
                                chunk_qkv_tensor_trace = None
                                if isinstance(getattr(self, "_decode_full_attention_trace", None), list):
                                    def chunk_post_input_rmsnorm_trace(
                                        attention_scratch: Qwen35ParoAttentionScratch,
                                        *,
                                        _layer_id: int = layer_id,
                                        _rows: int = chunk_rows,
                                        _stream: int = stream,
                                    ) -> None:
                                        self._trace_decode_full_attention(
                                            layer_id=_layer_id,
                                            stage="attn_input_pre_qkv",
                                            hidden=attention_scratch.attn_input,
                                            rows=_rows,
                                            stream=_stream,
                                        )

                                    def chunk_input_scratch_trace(
                                        stage: str,
                                        row: int,
                                        attention_scratch: Qwen35ParoAttentionScratch,
                                        *,
                                        _layer_id: int = layer_id,
                                        _stream: int = stream,
                                    ) -> None:
                                        self._trace_decode_full_attention(
                                            layer_id=_layer_id,
                                            stage=stage,
                                            hidden=attention_scratch.attn_input,
                                            rows=1,
                                            stream=_stream,
                                        )

                                    def chunk_qkv_tensor_trace(
                                        stage: str,
                                        row: int,
                                        tensor: Tensor,
                                        *,
                                        _layer_id: int = layer_id,
                                        _stream: int = stream,
                                    ) -> None:
                                        self._trace_decode_full_attention_tensor(
                                            layer_id=_layer_id,
                                            stage=stage,
                                            tensor=tensor,
                                            rows=1,
                                            stream=_stream,
                                        )
                                chunk_per_row_contexts = (
                                    per_row_contexts[chunk_start:chunk_end] if per_row_contexts is not None else None
                                )
                                chunk_per_row_append_contexts = (
                                    per_row_append_contexts[chunk_start:chunk_end]
                                    if per_row_append_contexts is not None
                                    else None
                                )
                                chunk_force_batch_gemv_output = use_batch_gemv_full_attention_output
                                if (
                                    force_native_row_chunk_full_attention_output
                                    and not chunk_force_batch_gemv_output
                                    and not force_per_row_full_attention_output
                                    and chunk_rows > 1
                                ):
                                    layer_row_chunk_native_full_attention_output = True
                                    row_chunk_native_full_attention_output = True
                                    chunk_records[-1]["full_attention_output_decode_path"] = "native_batch_row_chunk_forced"
                                elif (
                                    not chunk_force_batch_gemv_output
                                    and not force_per_row_full_attention_output
                                    and chunk_rows > 1
                                ):
                                    # The native fused rows>1 O projection is not
                                    # equality-stable for every prompt/window even
                                    # in the leading row chunk.  Keep the native
                                    # context/MoE work, but use the row-aware GEMV
                                    # O projection that matches c=1 numerics.  Two-
                                    # row chunks inherit the accepted rows=2
                                    # batch_gemv_auto default; larger chunks remain
                                    # an explicit rowchunk O repair until covered by
                                    # equality evidence.  A dedicated diagnostic env
                                    # may temporarily bypass this to re-test native
                                    # row-chunk O with artifact evidence.
                                    chunk_force_batch_gemv_output = True
                                    if chunk_rows == 2:
                                        layer_row_chunk_auto_batch_gemv_full_attention_output = True
                                        chunk_records[-1]["full_attention_output_decode_path"] = "batch_gemv_auto_row_chunk"
                                    else:
                                        layer_row_chunk_batch_gemv_full_attention_output = True
                                        row_chunk_batch_gemv_full_attention_output = True
                                        chunk_records[-1]["full_attention_output_decode_path"] = "batch_gemv_row_chunk"
                                chunk_out = state.run_full_attention_moe_decode_batch_layer_fp16(
                                    chunk_hidden,
                                    key_cache=key_cache,
                                    value_cache=value_cache,
                                    append_spans=chunk_append_spans,
                                    decode_spans=chunk_decode_spans,
                                    cos_table=self.cos,
                                    sin_table=self.sin,
                                    positions=chunk_position_tensor,
                                    max_positions=self.max_sequence_length,
                                    attention_scratch=chunk_attention_scratch,
                                    moe_scratch=chunk_moe_scratch,
                                    tokens=chunk_rows,
                                    force_selected_c1_moe=force_selected_c1_moe,
                                    force_per_row_input_rmsnorm=force_per_row_full_attention_input,
                                    force_per_row_qkv_scratch=force_per_row_full_attention_qkv,
                                    force_per_row_layer_scratch=force_per_row_full_attention_scratch,
                                    force_per_row_layer_batch_scratch=force_per_row_full_attention_batch_scratch,
                                    force_per_row_attention_batch_moe=force_per_row_full_attention_attn_batch_moe,
                                    force_per_row_attention_batch_post_moe=force_per_row_full_attention_attn_batch_post_moe,
                                    force_per_row_attention_batch_o_post_moe=force_per_row_full_attention_attn_batch_o_post_moe,
                                    force_per_row_preqkv_append_batch_context_o_post_moe=force_per_row_full_attention_preqkv_append_batch_context_o_post_moe,
                                    force_per_row_preqkv_append_context_batch_gate_o_post_moe=force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe,
                                    force_per_row_preqkv_append_context_gate_batch_o_post_moe=force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe,
                                    force_per_row_context=force_per_row_full_attention_context,
                                    force_per_row_context_only=force_per_row_full_attention_context_only,
                                    force_per_row_dense_context_only=layer_force_per_row_dense_context_only,
                                    force_per_row_dense_context_batch_gate=layer_force_per_row_dense_context_batch_gate,
                                    force_per_row_paged_context_only=layer_force_per_row_paged_context_only,
                                    force_batch_temp_context=force_batch_temp_full_attention_context,
                                    force_batch_compact_context=force_batch_compact_full_attention_context,
                                    force_per_row_gate=force_per_row_full_attention_gate,
                                    per_row_contexts=chunk_per_row_contexts,
                                    force_per_row_kv_append=force_per_row_full_attention_kv_append,
                                    per_row_append_contexts=chunk_per_row_append_contexts,
                                    force_per_row_append_context=force_per_row_full_attention_append_context,
                                    force_per_row_suffix=force_per_row_full_attention_suffix,
                                    force_per_row_output=force_per_row_full_attention_output,
                                    force_batch_gemv_output=chunk_force_batch_gemv_output,
                                    force_per_row_post_attention=force_per_row_post_attention,
                                    force_per_row_moe=force_per_row_full_attention_moe,
                                    post_input_rmsnorm_trace=chunk_post_input_rmsnorm_trace,
                                    input_scratch_trace=chunk_input_scratch_trace,
                                    qkv_tensor_trace=chunk_qkv_tensor_trace,
                                    library=self.libraries,
                                    stream=stream,
                                )
                                self._trace_decode_full_attention_scratch(
                                    layer_id=layer_id,
                                    attention_scratch=chunk_attention_scratch,
                                    rows=chunk_rows,
                                    context=getattr(chunk_attention_scratch, "query_raw", None),
                                    stream=stream,
                                )
                                self._trace_decode_full_attention_moe_scratch(
                                    layer_id=layer_id,
                                    moe_scratch=chunk_moe_scratch,
                                    rows=chunk_rows,
                                    stream=stream,
                                )
                                if force_per_row_full_attention_layer_copy:
                                    for row in range(chunk_rows):
                                        self.runtime.memcpy_async(
                                            next_hidden.ptr + (chunk_start + row) * self.hidden_nbytes,
                                            chunk_out.ptr + row * self.hidden_nbytes,
                                            self.hidden_nbytes,
                                            HipMemcpyKind.DEVICE_TO_DEVICE,
                                            stream,
                                        )
                                else:
                                    self.runtime.memcpy_async(
                                        next_hidden.ptr + chunk_start * self.hidden_nbytes,
                                        chunk_out.ptr,
                                        chunk_rows * self.hidden_nbytes,
                                        HipMemcpyKind.DEVICE_TO_DEVICE,
                                        stream,
                                    )
                            batch_full_spans_metadata = {
                                "row_chunk_size": int(chunk_size),
                                "chunks": chunk_records,
                            }
                            attention_scratch = self._ensure_full_decode_batch_scratch(layer_id, min(chunk_size, rows))
                            out = next_hidden
                            copied_full_attention_output = True
                        elif force_per_row_full_attention_persistent_scratch:
                            if dense_mlp:
                                raise NotImplementedError(
                                    "persistent c1 full-attention scratch diagnostic is currently wired for MoE layers"
                                )
                            if persistent_attention_scratch is None or persistent_moe_scratch is None or moe_scratch is None:
                                raise ValueError("persistent c1 full-attention scratch diagnostic requires token-1 scratch")
                            row_num_splits: list[int] = []
                            for row, (slot, position) in enumerate(zip(slots, positions, strict=True)):
                                row_key_cache, row_value_cache = self._slot_full_cache(layer_id, slot)
                                row_position, row_append_spans, row_decode_spans = self._slot_full_spans(layer_id, slot)
                                row_hidden = Tensor.from_handle(
                                    hidden.ptr + row * self.hidden_nbytes,
                                    (1, self.config.hidden_size),
                                    hidden.dtype,
                                    hidden.device,
                                )
                                num_splits = max(
                                    1,
                                    (int(position) + 1 + self.decode_chunk_size - 1) // self.decode_chunk_size,
                                )
                                row_num_splits.append(int(num_splits))
                                row_out = state.run_full_attention_moe_c1_layer_fp16(
                                    row_hidden,
                                    key_cache=row_key_cache,
                                    value_cache=row_value_cache,
                                    append_spans=row_append_spans,
                                    decode_spans=row_decode_spans,
                                    cos_table=self.cos,
                                    sin_table=self.sin,
                                    position=row_position,
                                    max_positions=self.max_sequence_length,
                                    attention_scratch=persistent_attention_scratch,
                                    moe_scratch=persistent_moe_scratch,
                                    chunk_size=self.decode_chunk_size,
                                    num_splits=num_splits,
                                    library=self.libraries,
                                    stream=stream,
                                )
                                self._trace_decode_full_attention_scratch(
                                    layer_id=layer_id,
                                    attention_scratch=persistent_attention_scratch,
                                    rows=1,
                                    context=getattr(persistent_attention_scratch, "attn_out", None),
                                    stream=stream,
                                )
                                self._trace_decode_full_attention_moe_scratch(
                                    layer_id=layer_id,
                                    moe_scratch=persistent_moe_scratch,
                                    rows=1,
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
                            out = next_hidden
                            copied_full_attention_output = True
                        else:
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
                                force_per_row_input_rmsnorm=force_per_row_full_attention_input,
                                force_per_row_qkv_scratch=force_per_row_full_attention_qkv,
                                force_per_row_layer_scratch=force_per_row_full_attention_scratch,
                                force_per_row_layer_batch_scratch=force_per_row_full_attention_batch_scratch,
                                force_per_row_attention_batch_moe=force_per_row_full_attention_attn_batch_moe,
                                force_per_row_attention_batch_post_moe=force_per_row_full_attention_attn_batch_post_moe,
                                force_per_row_attention_batch_o_post_moe=force_per_row_full_attention_attn_batch_o_post_moe,
                                force_per_row_preqkv_append_batch_context_o_post_moe=force_per_row_full_attention_preqkv_append_batch_context_o_post_moe,
                                force_per_row_preqkv_append_context_batch_gate_o_post_moe=force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe,
                                force_per_row_preqkv_append_context_gate_batch_o_post_moe=force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe,
                                force_per_row_context=force_per_row_full_attention_context,
                                force_per_row_context_only=force_per_row_full_attention_context_only,
                                force_per_row_dense_context_only=layer_force_per_row_dense_context_only,
                                force_per_row_dense_context_batch_gate=layer_force_per_row_dense_context_batch_gate,
                                force_per_row_paged_context_only=layer_force_per_row_paged_context_only,
                                force_batch_temp_context=force_batch_temp_full_attention_context,
                                force_batch_compact_context=force_batch_compact_full_attention_context,
                                force_per_row_gate=force_per_row_full_attention_gate,
                                per_row_contexts=per_row_contexts,
                                force_per_row_kv_append=force_per_row_full_attention_kv_append,
                                per_row_append_contexts=per_row_append_contexts,
                                force_per_row_append_context=force_per_row_full_attention_append_context,
                                force_per_row_suffix=force_per_row_full_attention_suffix,
                                force_per_row_output=force_per_row_full_attention_output,
                                force_batch_gemv_output=use_batch_gemv_full_attention_output,
                                force_per_row_post_attention=force_per_row_post_attention,
                                force_per_row_moe=force_per_row_full_attention_moe,
                                post_input_rmsnorm_trace=post_input_rmsnorm_trace,
                                input_scratch_trace=input_scratch_trace,
                                qkv_tensor_trace=qkv_tensor_trace,
                                library=self.libraries,
                                stream=stream,
                            )
                        trace_rows = (
                            1
                            if force_per_row_full_attention_persistent_scratch
                            else min(int(full_attention_row_chunk_size), rows) if layer_force_full_attention_row_chunks else rows
                        )
                        if not layer_force_full_attention_row_chunks:
                            self._trace_decode_full_attention_scratch(
                                layer_id=layer_id,
                                attention_scratch=attention_scratch,
                                rows=trace_rows,
                                context=getattr(attention_scratch, "query_raw", None),
                                stream=stream,
                            )
                            self._trace_decode_full_attention_moe_scratch(
                                layer_id=layer_id,
                                moe_scratch=moe_scratch,
                                rows=trace_rows,
                                stream=stream,
                            )
                        self._trace_decode_full_attention(
                            layer_id=layer_id,
                            stage="output",
                            hidden=out,
                            rows=rows,
                            stream=stream,
                        )
                        layer_moe_path = "dense_mlp" if dense_mlp else (
                            "selected_c1"
                            if rows == 1
                            else "selected_c1_per_row_moe_fallback" if force_per_row_full_attention_moe or force_per_row_full_attention_batch_scratch or force_per_row_full_attention_persistent_scratch else ("selected_c1_batch" if force_selected_c1_moe else "grouped_compact")
                        )
                        if not dense_mlp and rows > 1:
                            if (
                                force_per_row_full_attention_moe
                                or force_per_row_full_attention_batch_scratch
                                or force_per_row_full_attention_persistent_scratch
                            ):
                                moe_selected_c1_fallback_layers += 1
                            elif not force_selected_c1_moe:
                                moe_grouped_compact_layers += 1
                        layer_execution = {
                            "layer_index": int(layer_id),
                            "layer_type": "full_attention",
                            "rows": int(rows),
                            "slots": [int(slot) for slot in slots],
                            "max_context": int(max_context),
                            "full_attention_decode_path": "native_batch_row_chunks" if layer_force_full_attention_row_chunks else "native_batch",
                            "native_caware_decode": not (
                                layer_force_full_attention_row_chunks
                                or force_per_row_full_attention_moe
                                or force_per_row_full_attention_input
                                or force_per_row_full_attention_qkv
                                or force_per_row_full_attention_scratch
                                or force_per_row_full_attention_batch_scratch
                                or force_per_row_full_attention_attn_batch_moe
                                or force_per_row_full_attention_attn_batch_post_moe
                                or force_per_row_full_attention_attn_batch_o_post_moe
                                or force_per_row_full_attention_preqkv_append_batch_context_o_post_moe
                                or force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe
                                or force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe
                                or force_per_row_full_attention_persistent_scratch
                                or force_per_row_full_attention_skip_batch_setup
                                or force_per_row_full_attention_output
                                or force_per_row_full_attention_layer_copy
                                or force_per_row_full_attention_context
                                or force_per_row_full_attention_context_only
                                or layer_force_per_row_dense_context_only
                                or layer_force_per_row_dense_context_batch_gate
                                or layer_force_per_row_paged_context_only
                                or force_batch_temp_full_attention_context
                                or force_batch_compact_full_attention_context
                                or force_per_row_full_attention_gate
                                or force_per_row_full_attention_kv_append
                                or force_per_row_full_attention_append_context
                                or force_per_row_full_attention_suffix
                                or force_per_row_post_attention
                            ),
                            "moe_decode_path": layer_moe_path,
                        }
                        if isinstance(getattr(self, "_decode_full_attention_trace", None), list):
                            layer_execution["attn_context_trace_source"] = "attention_scratch.query_raw"
                        if layer_force_full_attention_row_chunks:
                            layer_execution["full_attention_row_chunk_size"] = int(full_attention_row_chunk_size)
                            layer_execution["full_attention_row_chunk_source"] = full_attention_row_chunk_source
                        if force_per_row_full_attention_input:
                            layer_execution["full_attention_input_decode_path"] = full_attention_input_decode_path
                        if force_per_row_full_attention_qkv:
                            layer_execution["full_attention_qkv_decode_path"] = full_attention_qkv_decode_path
                        if (
                            force_per_row_full_attention_scratch
                            or force_per_row_full_attention_batch_scratch
                            or force_per_row_full_attention_attn_batch_moe
                            or force_per_row_full_attention_attn_batch_post_moe
                            or force_per_row_full_attention_attn_batch_o_post_moe
                            or force_per_row_full_attention_preqkv_append_batch_context_o_post_moe
                            or force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe
                            or force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe
                            or force_per_row_full_attention_persistent_scratch
                        ):
                            layer_execution["full_attention_scratch_decode_path"] = full_attention_scratch_decode_path
                        if (
                            force_per_row_full_attention_context
                            or force_per_row_full_attention_context_only
                            or layer_force_per_row_dense_context_only
                            or layer_force_per_row_dense_context_batch_gate
                            or layer_force_per_row_paged_context_only
                            or force_batch_temp_full_attention_context
                            or force_batch_compact_full_attention_context
                        ):
                            layer_execution["full_attention_context_decode_path"] = layer_full_attention_context_decode_path
                        if force_per_row_full_attention_gate:
                            layer_execution["full_attention_gate_decode_path"] = full_attention_gate_decode_path
                        if force_per_row_full_attention_kv_append:
                            layer_execution["full_attention_kv_append_decode_path"] = full_attention_kv_append_decode_path
                        if force_per_row_full_attention_append_context:
                            layer_execution["full_attention_append_context_decode_path"] = full_attention_append_context_decode_path
                        if force_per_row_full_attention_suffix:
                            layer_execution["full_attention_suffix_decode_path"] = full_attention_suffix_decode_path
                        if layer_row_chunk_batch_gemv_full_attention_output:
                            layer_execution["full_attention_output_decode_path"] = "native_batch_with_row_chunk_batch_gemv"
                        elif layer_row_chunk_auto_batch_gemv_full_attention_output:
                            layer_execution["full_attention_output_decode_path"] = "native_batch_row_chunks_with_batch_gemv_auto"
                        elif layer_row_chunk_native_full_attention_output:
                            layer_execution["full_attention_output_decode_path"] = "native_batch_row_chunk_forced"
                        elif (
                            force_per_row_full_attention_output
                            or use_batch_gemv_full_attention_output
                            or force_native_full_attention_output
                        ):
                            layer_execution["full_attention_output_decode_path"] = full_attention_output_decode_path
                        if force_per_row_full_attention_layer_copy:
                            layer_execution["full_attention_layer_copy_decode_path"] = full_attention_layer_copy_decode_path
                        if force_per_row_post_attention:
                            layer_execution["post_attention_decode_path"] = post_attention_decode_path
                        if force_per_row_full_attention_skip_batch_setup:
                            layer_execution["full_attention_batch_setup_decode_path"] = "skipped_for_persistent_c1"
                        full_spans_metadata = batch_full_spans_metadata
                        if isinstance(full_spans_metadata, dict):
                            layer_execution["full_attention_segment_metadata"] = full_spans_metadata
                        layer_executions.append(layer_execution)
                        if not copied_full_attention_output:
                            if force_per_row_full_attention_layer_copy:
                                for row in range(rows):
                                    self.runtime.memcpy_async(
                                        next_hidden.ptr + row * self.hidden_nbytes,
                                        out.ptr + row * self.hidden_nbytes,
                                        self.hidden_nbytes,
                                        HipMemcpyKind.DEVICE_TO_DEVICE,
                                        stream,
                                    )
                            else:
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
                            self._trace_decode_full_attention_moe_scratch(
                                layer_id=layer_id,
                                moe_scratch=self.moe_scratch[layer_id],
                                rows=1,
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
            if force_per_row_linear_moe:
                decode_blockers.append("linear-attention MoE forced to per-row selected-c1 diagnostic path")
            if force_selected_c1_linear_projections:
                decode_blockers.append("linear-attention projections forced to selected-c1 diagnostic path")
            if force_selected_c1_qkv_z_linear_input:
                decode_blockers.append("linear-attention QKV/Z rotary inputs forced to selected-c1 diagnostic path")
            if force_selected_c1_qkv_linear_projections:
                decode_blockers.append("linear-attention QKV projections forced to selected-c1 diagnostic path")
            if force_selected_c1_z_linear_projections:
                decode_blockers.append("linear-attention Z projection forced to selected-c1 diagnostic path")
            if force_selected_c1_ab_linear_projections:
                decode_blockers.append("linear-attention A/B projections forced to selected-c1 diagnostic path")
            if force_batch_gemv_linear_projections:
                decode_blockers.append("linear-attention projections forced to batch GEMV diagnostic path")
            if force_selected_c1_linear_state:
                decode_blockers.append("linear-attention state forced to selected-c1 diagnostic path")
            if linear_attention_output_path == "selected_c1_forced":
                decode_blockers.append("linear-attention output projection forced to selected-c1 diagnostic path")
            if force_per_row_linear:
                decode_blockers.append("linear-attention decode forced to per-row diagnostic path")
                if not dense_mlp and rows > 1:
                    moe_decode_path = (
                        "selected_c1_forced_with_per_row_linear_attention_fallback"
                        if force_selected_c1_moe or force_per_row_linear_moe
                        else "mixed_grouped_compact_with_per_row_linear_attention_fallback"
                    )
            if force_per_row_full_attention_input:
                decode_blockers.append("full-attention input RMSNorm forced to per-row diagnostic path")
            if force_per_row_full_attention_moe:
                decode_blockers.append("full-attention MoE forced to per-row selected-c1 diagnostic path")
            if row_chunked_full_attention_layers:
                if auto_full_attention_row_chunks:
                    decode_blockers.append("full-attention decode auto-selected native row-chunk diagnostic path")
                elif full_attention_row_chunk_layers:
                    decode_blockers.append("full-attention decode forced to native row-chunk diagnostic path on selected layers")
                else:
                    decode_blockers.append("full-attention decode forced to native row-chunk diagnostic path")
            if force_per_row_full_attention_qkv:
                decode_blockers.append("full-attention QKV prep forced to per-row scratch diagnostic path")
            if force_per_row_full_attention_scratch:
                decode_blockers.append("full-attention layer forced to independent per-row scratch diagnostic path")
            if force_per_row_full_attention_batch_scratch:
                decode_blockers.append("full-attention layer forced to batch-view per-row scratch diagnostic path")
            if force_per_row_full_attention_attn_batch_moe:
                decode_blockers.append("full-attention attention/post forced to per-row diagnostic path with grouped batch MoE")
            if force_per_row_full_attention_attn_batch_post_moe:
                decode_blockers.append("full-attention attention forced to per-row diagnostic path with batch post/MoE")
            if force_per_row_full_attention_attn_batch_o_post_moe:
                decode_blockers.append("full-attention pre-O attention forced to per-row diagnostic path with batch O/post/MoE")
            if force_per_row_full_attention_preqkv_append_batch_context_o_post_moe:
                decode_blockers.append("full-attention pre-QKV/append forced to per-row diagnostic path with batch context/O/post/MoE")
            if force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe:
                decode_blockers.append("full-attention pre-QKV/append/context forced to per-row diagnostic path with batch gate/O/post/MoE")
            if force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe:
                decode_blockers.append("full-attention pre-QKV/append/context/gate forced to per-row diagnostic path with batch O/post/MoE")
            if force_per_row_full_attention_persistent_scratch:
                decode_blockers.append("full-attention layer forced to persistent c1 scratch diagnostic path")
            if force_per_row_full_attention_skip_batch_setup:
                decode_blockers.append("full-attention native batch setup skipped for persistent c1 diagnostic path")
            if force_per_row_full_attention_context:
                decode_blockers.append("full-attention context/gate forced to per-row diagnostic path")
            if force_per_row_full_attention_context_only:
                decode_blockers.append("full-attention context forced to per-row diagnostic path with diagnostic gate")
            if force_per_row_full_attention_dense_context_only:
                decode_blockers.append("full-attention context forced to row-local dense diagnostic path with diagnostic gate")
            if force_per_row_full_attention_dense_context_batch_gate:
                decode_blockers.append("full-attention context forced to row-local dense diagnostic path with batch gate")
            if force_per_row_full_attention_dense_context_layers:
                decode_blockers.append("full-attention context forced to row-local dense diagnostic path on selected layers")
            if force_per_row_full_attention_dense_context_batch_gate_layers:
                decode_blockers.append("full-attention context forced to row-local dense diagnostic path with batch gate on selected layers")
            if force_per_row_full_attention_paged_context_only:
                decode_blockers.append("full-attention context forced to row-local paged diagnostic path with diagnostic gate")
            if force_batch_temp_full_attention_context:
                decode_blockers.append("full-attention context forced through temp-output batch diagnostic path")
            if force_batch_compact_full_attention_context:
                decode_blockers.append("full-attention context forced through compact-cache batch diagnostic path")
            if force_per_row_full_attention_gate:
                decode_blockers.append("full-attention gate forced to per-row diagnostic path")
            if force_per_row_full_attention_kv_append:
                decode_blockers.append("full-attention KV append forced to per-row diagnostic path")
            if force_per_row_full_attention_append_context:
                decode_blockers.append("full-attention append+context forced to interleaved per-row diagnostic order")
            if force_per_row_full_attention_suffix:
                decode_blockers.append("full-attention context/output/post/MoE forced to interleaved per-row diagnostic order")
            if force_per_row_full_attention_output:
                decode_blockers.append("full-attention O projection forced to per-row diagnostic path")
            if row_chunk_batch_gemv_full_attention_output:
                decode_blockers.append("full-attention O projection forced to batch GEMV for multi-row chunks")
            if row_chunk_native_full_attention_output:
                decode_blockers.append("full-attention O projection forced to native row-chunk diagnostic path")
            if force_per_row_full_attention_layer_copy:
                decode_blockers.append("full-attention layer output forced to per-row copy diagnostic path")
            if force_per_row_post_attention:
                decode_blockers.append("post-attention add/rmsnorm forced to per-row diagnostic path")
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
                "full_attention_input_decode_path": full_attention_input_decode_path,
                "full_attention_qkv_decode_path": full_attention_qkv_decode_path,
                "full_attention_scratch_decode_path": full_attention_scratch_decode_path,
                "full_attention_context_decode_path": full_attention_context_decode_path,
                "full_attention_kv_append_decode_path": full_attention_kv_append_decode_path,
                "post_attention_decode_path": post_attention_decode_path,
                "native_caware_decode": full_attention_decode_path not in {"per_row_splitk_fallback", "per_row_context_fallback"}
                and not row_chunked_full_attention_layers
                and not force_per_row_linear_moe
                and not force_selected_c1_linear_projections
                and not force_selected_c1_qkv_z_linear_projections
                and not force_selected_c1_qkv_z_linear_input
                and not force_selected_c1_qkv_linear_projections
                and not force_selected_c1_z_linear_projections
                and not force_selected_c1_ab_linear_projections
                and not force_batch_gemv_linear_projections
                and not force_selected_c1_linear_state
                and linear_attention_output_path not in {"selected_c1_forced", "batch_gemv_from_f32"}
                and not force_per_row_linear
                and not force_per_row_full_attention_input
                and not force_per_row_full_attention_qkv
                and not force_per_row_full_attention_scratch
                and not force_per_row_full_attention_batch_scratch
                and not force_per_row_full_attention_attn_batch_moe
                and not force_per_row_full_attention_attn_batch_post_moe
                and not force_per_row_full_attention_attn_batch_o_post_moe
                and not force_per_row_full_attention_preqkv_append_batch_context_o_post_moe
                and not force_per_row_full_attention_preqkv_append_context_batch_gate_o_post_moe
                and not force_per_row_full_attention_preqkv_append_context_gate_batch_o_post_moe
                and not force_per_row_full_attention_persistent_scratch
                and not force_per_row_full_attention_skip_batch_setup
                and not force_per_row_full_attention_context
                and not force_per_row_full_attention_context_only
                and not force_per_row_full_attention_dense_context_only
                and not force_per_row_full_attention_dense_context_batch_gate
                and not force_per_row_full_attention_dense_context_layers
                and not force_per_row_full_attention_dense_context_batch_gate_layers
                and not force_per_row_full_attention_paged_context_only
                and not force_batch_temp_full_attention_context
                and not force_batch_compact_full_attention_context
                and not force_per_row_full_attention_gate
                and not force_per_row_full_attention_kv_append
                and not force_per_row_full_attention_append_context
                and not force_per_row_full_attention_suffix
                and not force_per_row_full_attention_output
                and not force_per_row_full_attention_layer_copy
                and not force_per_row_full_attention_moe
                and not force_per_row_post_attention,
                "linear_attention_segment_metadata": linear_segment_metadata,
                "linear_attention_projection_path": linear_attention_projection_path,
                "linear_attention_state_path": linear_attention_state_path,
                "linear_attention_output_path": linear_attention_output_path,
                "moe_decode_path": moe_decode_path,
                "moe_decode_rows": int(rows),
                "moe_grouped_compact_layers": int(moe_grouped_compact_layers),
                "moe_selected_c1_fallback_layers": int(moe_selected_c1_fallback_layers),
                "layer_executions": layer_executions,
                "blockers": decode_blockers,
            }
            if row_chunked_full_attention_layers:
                self.last_batch_decode_execution["full_attention_row_chunk_size"] = int(full_attention_row_chunk_size)
                self.last_batch_decode_execution["full_attention_row_chunk_source"] = full_attention_row_chunk_source
                self.last_batch_decode_execution["full_attention_row_chunked_layers"] = row_chunked_full_attention_layers
            if full_attention_row_chunk_layers:
                self.last_batch_decode_execution["full_attention_row_chunk_layers"] = sorted(
                    int(layer) for layer in full_attention_row_chunk_layers
                )
            if force_per_row_full_attention_dense_context_layers:
                self.last_batch_decode_execution["full_attention_dense_context_layers"] = sorted(
                    int(layer) for layer in force_per_row_full_attention_dense_context_layers
                )
            if force_per_row_full_attention_dense_context_batch_gate_layers:
                self.last_batch_decode_execution["full_attention_dense_context_batch_gate_layers"] = sorted(
                    int(layer) for layer in force_per_row_full_attention_dense_context_batch_gate_layers
                )
            if force_per_row_full_attention_gate:
                self.last_batch_decode_execution["full_attention_gate_decode_path"] = full_attention_gate_decode_path
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
                self._trace_decode_linear_input(layer_id=layer_id, hidden=hidden, rows=1, stream=stream)
                conv_state, recurrent_state = self._slot_linear_state(layer_id, slot)
                out = state.run_linear_attention_moe_c1_layer_fp16(
                    hidden,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    linear_scratch=self._linear_decode_scratch(layer_id, state),
                    moe_scratch=self._mlp_decode_scratch(layer_id, state),
                    library=self.libraries,
                    stream=stream,
                )
                self._trace_decode_linear_stages(
                    layer_id=layer_id,
                    linear_scratch=self.linear_scratch[layer_id],
                    moe_scratch=self.moe_scratch[layer_id],
                    output=out,
                    rows=1,
                    stream=stream,
                )
                self._trace_decode_linear_output(layer_id=layer_id, hidden=out, rows=1, stream=stream)
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
                post_input_rmsnorm_trace = None
                input_scratch_trace = None
                qkv_tensor_trace = None
                if isinstance(getattr(self, "_decode_full_attention_trace", None), list):
                    def post_input_rmsnorm_trace(
                        attention_scratch: Qwen35ParoAttentionScratch,
                        *,
                        _layer_id: int = layer_id,
                        _stream: int = stream,
                    ) -> None:
                        self._trace_decode_full_attention(
                            layer_id=_layer_id,
                            stage="attn_input_pre_qkv",
                            hidden=attention_scratch.attn_input,
                            rows=1,
                            stream=_stream,
                        )

                    def input_scratch_trace(
                        stage: str,
                        attention_scratch: Qwen35ParoAttentionScratch,
                        *,
                        _layer_id: int = layer_id,
                        _stream: int = stream,
                    ) -> None:
                        self._trace_decode_full_attention(
                            layer_id=_layer_id,
                            stage=stage,
                            hidden=attention_scratch.attn_input,
                            rows=1,
                            stream=_stream,
                        )

                    def qkv_tensor_trace(
                        stage: str,
                        tensor: Tensor,
                        *,
                        _layer_id: int = layer_id,
                        _stream: int = stream,
                    ) -> None:
                        self._trace_decode_full_attention_tensor(
                            layer_id=_layer_id,
                            stage=stage,
                            tensor=tensor,
                            rows=1,
                            stream=_stream,
                        )

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
                    moe_scratch=self._mlp_decode_scratch(layer_id, state),
                    chunk_size=self.decode_chunk_size,
                    num_splits=num_splits,
                    post_input_rmsnorm_trace=post_input_rmsnorm_trace,
                    input_scratch_trace=input_scratch_trace,
                    qkv_tensor_trace=qkv_tensor_trace,
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
                self._trace_decode_full_attention_moe_scratch(
                    layer_id=layer_id,
                    moe_scratch=self.moe_scratch[layer_id],
                    rows=1,
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
                "dflash_accept": build_dflash_accept(**build_kwargs),
                "dflash_commit": build_dflash_commit(**build_kwargs),
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
            # M14.dispatch.1-beta prewarm: if the optional C-side MoE C1
            # dispatcher is enabled, resolve its process-global .so handle and
            # function-pointer table during resident build.  Leaving this lazy
            # charges ~200 ms to verifier cycle 1 and looks like a steady-state
            # regression once the economics harness averages over cycles.
            from hipengine.runtime.moe_c1_dispatch import (
                moe_c1_c_dispatch_enabled,
                prewarm_moe_c1_c_dispatch,
            )

            if moe_c1_c_dispatch_enabled():
                prewarm_moe_c1_c_dispatch()
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

    def _allocate_verify_trunk_buffers(self) -> None:
        """Allocate the DFlash chain verifier's dedicated trunk hidden pair.

        ``_launch_verify_chain_forward_accept`` writes the root+candidate
        verifier rows into ``verify_trunk_hidden`` and ping-pongs it against
        ``verify_trunk_next_hidden`` across the layer stack, so the two must be
        DISTINCT device buffers.  The verifier entrypoints reject
        ``rows > max_batch_size`` and every other verifier buffer is sized to
        ``max_batch_size``, so the trunk pair only needs that verifier-row
        capacity -- not the full prompt prefill capacity.

        This is deliberately separate from main's lazy ``prefill_hidden`` (a
        single, growable, self-aliased buffer sized to the last decode step's
        row count): reusing ``prefill_hidden`` for the verifier forward writes
        out of bounds and faults/hangs the GPU.  Regression covered by
        ``tests/test_qwen35_resident_batch_layout.py``.
        """
        verify_rows = int(self.max_batch_size)
        if verify_rows <= 0:
            raise ValueError("max_batch_size must be positive for verifier trunk allocation")
        verify_trunk_nbytes = verify_rows * self.hidden_nbytes
        verify_trunk_hidden_buf = malloc(verify_trunk_nbytes, runtime=self.runtime)
        verify_trunk_next_hidden_buf = malloc(verify_trunk_nbytes, runtime=self.runtime)
        self.buffers.extend((verify_trunk_hidden_buf, verify_trunk_next_hidden_buf))
        trunk_shape = (verify_rows, self.config.hidden_size)
        self.verify_trunk_hidden = Tensor.from_handle(
            verify_trunk_hidden_buf.ptr, trunk_shape, DType.FP16, self.device
        )
        self.verify_trunk_next_hidden = Tensor.from_handle(
            verify_trunk_next_hidden_buf.ptr, trunk_shape, DType.FP16, self.device
        )

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
        # Merge fix: ours' DFlash chain verifier needs a dedicated, distinct,
        # full-capacity trunk hidden pair -- it must NOT share main's lazy,
        # 1-row, self-aliased `prefill_hidden` (out-of-bounds verifier write ->
        # GPU fault/hang).  See `_allocate_verify_trunk_buffers`.
        self._allocate_verify_trunk_buffers()

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
        # Fixed-capacity buffers for the native root+candidate verifier path.
        # They are sized by max_batch_size because one DFlash chain bucket is
        # root + candidate rows for a single request.
        verify_rows = self.max_batch_size
        self.verify_token_ids_i64 = malloc(verify_rows * DType.INT64.itemsize, runtime=self.runtime)
        self.verify_token_ids_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_dynamic_metadata_i64 = malloc(
            verify_rows * _VERIFY_DYNAMIC_METADATA_FIELDS * DType.INT64.itemsize,
            runtime=self.runtime,
        )
        self.verify_positions_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_parent_rows_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_parent_rows_i64 = malloc(verify_rows * DType.INT64.itemsize, runtime=self.runtime)
        self.verify_draft_depths_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_row_to_request_i32 = malloc(verify_rows * DType.INT32.itemsize, runtime=self.runtime)
        self.verify_active_mask_u8 = malloc(verify_rows * DType.BOOL.itemsize, runtime=self.runtime)
        # Tree ancestor mask: dense [verify_rows, verify_rows] uint8 buffer
        # consumed by qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans.
        # ``ancestor_mask[i * verify_rows + j] == 1`` iff verifier row j is
        # an ancestor of verifier row i (a row is its own ancestor).  Only
        # the leading ``rows`` x ``rows`` submatrix is read each cycle; the
        # full ``verify_rows`` capacity matches every other verify buffer.
        self.verify_ancestor_mask_u8 = malloc(
            verify_rows * verify_rows * DType.BOOL.itemsize,
            runtime=self.runtime,
        )
        # Tree cache-slot buffer: per-row UNIQUE K/V write slot.  Tree mode
        # decouples RoPE phase (depth-based, in ``prefill_position_buf``)
        # from K/V storage slot (per-row, in this buffer) because siblings
        # share a depth but must NOT share a cache cell.  Chain mode leaves
        # this buffer untouched; the chain orchestrator continues to use
        # ``prefill_position_buf`` for both RoPE and write slot (they
        # coincide for chain topology).
        self.verify_cache_slot_buf = malloc(
            verify_rows * DType.INT64.itemsize,
            runtime=self.runtime,
        )
        # Latest tree_committed_count (decoded position where verifier rows
        # start).  Set per-cycle by ``_write_verify_chain_metadata`` when
        # ``batch.mode == 'verify_tree'``; chain mode leaves it at 0.
        self.verify_tree_committed_count: int = 0
        # Device-resident copy: the tree GQA kernel reads it via pointer so a
        # captured graph replays per-cycle counts instead of the frozen value.
        self.verify_tree_committed_buf = malloc(DType.INT64.itemsize, runtime=self.runtime)
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
        self.verify_accept_payload_i32 = malloc(
            self.max_batch_size * ACCEPT_PACKED_PAYLOAD_FIELDS * DType.INT32.itemsize,
            runtime=self.runtime,
        )
        self.verify_capture_hidden_concat = malloc(
            verify_rows * len(self.config.layer_types) * self.config.hidden_size * DType.BF16.itemsize,
            runtime=self.runtime,
        )
        self._verify_graph_cache: dict[tuple[int, int, int, str, str], Qwen35ParoVerifierGraphEntry] = {}
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
                self.verify_token_ids_i64,
                self.verify_token_ids_i32,
                self.verify_dynamic_metadata_i64,
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
                self.verify_accept_payload_i32,
                self.verify_capture_hidden_concat,
                self.verify_ancestor_mask_u8,
                self.verify_cache_slot_buf,
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
        self._clear_resident_tensor_view_caches()
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
        self._clear_resident_tensor_view_caches()

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
                self._clear_resident_tensor_view_caches()
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
        self._build_linear_state_commit_tables(qkv_width=qkv_width)
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
        # Reuse a persistent device scratch instead of malloc/free per decode
        # step (the allocator round-trip + synchronous copy is per-step host
        # overhead on the decode hot path).
        token_scratch_bytes = int(self.max_batch_size) * DType.INT64.itemsize
        token_buf = getattr(self, "_batch_token_id_scratch", None)
        if token_buf is None or token_buf.nbytes < token_scratch_bytes:
            token_buf = malloc(token_scratch_bytes, runtime=self.runtime)
            self.buffers.append(token_buf)
            self._batch_token_id_scratch = token_buf
        row_buf = None
        try:
            copy_host_to_device(token_buf, host_array_ptr(token_arr), token_arr.nbytes, runtime=self.runtime)
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

    def _set_batch_token_embeddings_from_ptr(
        self,
        token_src_ptr: int,
        *,
        rows: int,
        stream: int = 0,
    ) -> Tensor:
        """Gather batch token embeddings straight from a device int64 token buffer.

        Device-resident analog of :meth:`_set_batch_token_embeddings` (C3.0b
        piece A): the next-token ids are read directly from a device buffer
        (e.g. ``batch_lm_out_index``) with no host token list, host->device
        copy, or host-side bounds check, so the gather is safe inside a captured
        c>1 decode graph.  ``out[row] = embedding[token_src[row]]`` for
        ``row in [0, rows)``; the device argmax that fills ``token_src``
        guarantees ids in ``[0, vocab_size)`` by construction (the kernel skips
        any out-of-range id without writing its row).
        """

        if rows <= 0:
            raise ValueError("rows must be positive")
        if rows > self.max_batch_size:
            raise ValueError("rows exceed max_batch_size")
        embedding_lookup_batch_mapped_fp16_i64(
            self.embedding.tensor.ptr,
            token_src_ptr,
            self.batch_hidden.ptr,
            rows,
            self.config.hidden_size,
            self.vocab_size,
            rows,
            row_map_i32_ptr=None,
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )
        return Tensor.from_handle(self.batch_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)

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
        # Reuse a persistent device scratch instead of malloc/free per decode step.
        pos_scratch_bytes = int(self.max_batch_size) * DType.INT64.itemsize
        pos_buf = getattr(self, "_batch_position_scratch", None)
        if pos_buf is None or pos_buf.nbytes < pos_scratch_bytes:
            pos_buf = malloc(pos_scratch_bytes, runtime=self.runtime)
            self.buffers.append(pos_buf)
            self._batch_position_scratch = pos_buf
        mask_buf = None
        try:
            copy_host_to_device(pos_buf, host_array_ptr(pos_arr), pos_arr.nbytes, runtime=self.runtime)
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

    def _record_slot_position_host(self, position: int, *, slot: int) -> None:
        if hasattr(self, "position_arr") and hasattr(self, "context_arr"):
            self.position_arr[int(slot)] = int(position)
            self.context_arr[int(slot)] = int(position) + 1

    def _check_position(self, position: int) -> None:
        if position < 0 or position >= self.max_sequence_length:
            raise ValueError(f"position {position} outside session capacity {self.max_sequence_length}")

    def _batch_decode_execution_with_sampler_audit(self, decode_execution: dict[str, Any]) -> dict[str, Any]:
        argmax_audit = getattr(self, "_batch_argmax_audit", None)
        lm_head_audit = getattr(self, "_batch_lm_head_audit", None)
        lm_head_fence = getattr(self, "_batch_lm_head_fence", None)
        final_norm_audit = getattr(self, "_batch_final_norm_audit", None)
        final_norm_fence = getattr(self, "_batch_final_norm_fence", None)
        final_rmsnorm_fence = getattr(self, "_batch_final_rmsnorm_fence", None)
        final_rmsnorm_temp_fence = getattr(self, "_batch_final_rmsnorm_temp_fence", None)
        final_cast_temp_fence = getattr(self, "_batch_final_cast_temp_fence", None)
        final_cast_tiny_fence = getattr(self, "_batch_final_cast_tiny_fence", None)
        final_cast_elems_fence = getattr(self, "_batch_final_cast_elems_fence", None)
        stabilize_cast_elems_fence = getattr(self, "_batch_stabilize_cast_elems_fence", None)
        sync_fence = getattr(self, "_batch_sync_fence", None)
        suffix_fence = getattr(self, "_batch_suffix_fence", None)
        if (
            not isinstance(argmax_audit, dict)
            and not isinstance(lm_head_audit, dict)
            and not isinstance(lm_head_fence, dict)
            and not isinstance(final_norm_audit, dict)
            and not isinstance(final_norm_fence, dict)
            and not isinstance(final_rmsnorm_fence, dict)
            and not isinstance(final_rmsnorm_temp_fence, dict)
            and not isinstance(final_cast_temp_fence, dict)
            and not isinstance(final_cast_tiny_fence, dict)
            and not isinstance(final_cast_elems_fence, dict)
            and not isinstance(stabilize_cast_elems_fence, dict)
            and not isinstance(sync_fence, dict)
            and not isinstance(suffix_fence, dict)
        ):
            return decode_execution
        execution = dict(decode_execution)
        sampler_execution = dict(execution.get("sampler_execution") or {})
        if isinstance(argmax_audit, dict):
            sampler_execution["argmax_audit"] = dict(argmax_audit)
        if isinstance(lm_head_audit, dict):
            sampler_execution["lm_head_audit"] = dict(lm_head_audit)
        if isinstance(lm_head_fence, dict):
            sampler_execution["lm_head_fence"] = dict(lm_head_fence)
        if isinstance(final_norm_audit, dict):
            sampler_execution["final_norm_audit"] = dict(final_norm_audit)
        if isinstance(final_norm_fence, dict):
            sampler_execution["final_norm_fence"] = dict(final_norm_fence)
        if isinstance(final_rmsnorm_fence, dict):
            sampler_execution["final_rmsnorm_fence"] = dict(final_rmsnorm_fence)
        if isinstance(final_rmsnorm_temp_fence, dict):
            sampler_execution["final_rmsnorm_temp_fence"] = dict(final_rmsnorm_temp_fence)
        if isinstance(final_cast_temp_fence, dict):
            sampler_execution["final_cast_temp_fence"] = dict(final_cast_temp_fence)
        if isinstance(final_cast_tiny_fence, dict):
            sampler_execution["final_cast_tiny_fence"] = dict(final_cast_tiny_fence)
        if isinstance(final_cast_elems_fence, dict):
            sampler_execution["final_cast_elems_fence"] = dict(final_cast_elems_fence)
        if isinstance(stabilize_cast_elems_fence, dict):
            sampler_execution["stabilize_cast_elems_fence"] = dict(stabilize_cast_elems_fence)
        if isinstance(sync_fence, dict):
            sampler_execution["sync_fence"] = dict(sync_fence)
        if isinstance(suffix_fence, dict):
            sampler_execution["suffix_fence"] = dict(suffix_fence)
        execution["sampler_execution"] = sampler_execution
        return execution

    def _publish_batch_sampler_execution(self, sampler_execution: dict[str, Any]) -> None:
        self.last_batch_sampler_execution = sampler_execution
        decode_execution = getattr(self, "last_batch_decode_execution", None)
        if isinstance(decode_execution, dict):
            decode_execution["sampler_execution"] = dict(sampler_execution)

    def _record_batch_argmax_audit(
        self,
        *,
        batch_indices: np.ndarray,
        batch_values: np.ndarray,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        audit = getattr(self, "_batch_argmax_audit", None)
        if not isinstance(audit, dict):
            audit = {
                "enabled": True,
                "checked_steps": 0,
                "checked_rows": 0,
                "mismatch_steps": 0,
                "mismatch_rows": 0,
                "mismatches": [],
            }
            self._batch_argmax_audit = audit
        step_index = int(audit["checked_steps"])
        serial_indices = np.empty((rows,), dtype=np.int64)
        serial_values = np.empty((rows,), dtype=np.float32)
        for row in range(rows):
            argmax_f32(
                self.batch_lm_logits.ptr + row * self.vocab_size * DType.FP32.itemsize,
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
            self.runtime.device_synchronize()
            serial_index = np.empty((1,), dtype=np.int64)
            serial_value = np.empty((1,), dtype=np.float32)
            copy_device_to_host(host_array_ptr(serial_index), self.lm_out_index, runtime=self.runtime)
            copy_device_to_host(host_array_ptr(serial_value), self.lm_out_value, runtime=self.runtime)
            serial_indices[row] = serial_index[0]
            serial_values[row] = serial_value[0]
        mismatches: list[dict[str, Any]] = []
        for row in range(rows):
            if int(batch_indices[row]) != int(serial_indices[row]):
                mismatches.append(
                    {
                        "step_index": step_index,
                        "row": row,
                        "batch_index": int(batch_indices[row]),
                        "serial_index": int(serial_indices[row]),
                        "batch_value": float(batch_values[row]),
                        "serial_value": float(serial_values[row]),
                    }
                )
        audit["checked_steps"] = step_index + 1
        audit["checked_rows"] = int(audit["checked_rows"]) + int(rows)
        if mismatches:
            audit["mismatch_steps"] = int(audit["mismatch_steps"]) + 1
            audit["mismatch_rows"] = int(audit["mismatch_rows"]) + len(mismatches)
            retained = list(audit.get("mismatches") or [])
            retained.extend(mismatches)
            audit["mismatches"] = retained[:16]
        audit["last_step"] = {
            "step_index": step_index,
            "rows": int(rows),
            "mismatch_rows": len(mismatches),
        }
        return dict(audit)

    def _record_batch_lm_head_audit(
        self,
        *,
        batch_indices: np.ndarray,
        batch_values: np.ndarray,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        """Compare batched LM-head+argmax output with serial per-row projection.

        The argmax audit reuses the already-computed batched logits.  This audit
        goes one step earlier: each row is projected through the serial c=1
        LM-head path from the exact same normalized BF16 input row, then argmaxed
        independently.  A clean audit therefore clears the LM-head projection and
        argmax suffix for the observed batch input, leaving earlier hidden-state
        production as the remaining source of generated-token drift.
        """

        audit = getattr(self, "_batch_lm_head_audit", None)
        if not isinstance(audit, dict):
            audit = {
                "enabled": True,
                "checked_steps": 0,
                "checked_rows": 0,
                "mismatch_steps": 0,
                "mismatch_rows": 0,
                "max_abs_value_delta": 0.0,
                "mismatches": [],
            }
            self._batch_lm_head_audit = audit
        step_index = int(audit["checked_steps"])
        serial_indices = np.empty((rows,), dtype=np.int64)
        serial_values = np.empty((rows,), dtype=np.float32)
        for row in range(rows):
            w8a16_linear_bf16_f32_out(
                self.batch_norm_out_bf16.ptr + row * self.hidden_nbytes,
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
            self.runtime.device_synchronize()
            serial_index = np.empty((1,), dtype=np.int64)
            serial_value = np.empty((1,), dtype=np.float32)
            copy_device_to_host(host_array_ptr(serial_index), self.lm_out_index, runtime=self.runtime)
            copy_device_to_host(host_array_ptr(serial_value), self.lm_out_value, runtime=self.runtime)
            serial_indices[row] = serial_index[0]
            serial_values[row] = serial_value[0]
        mismatches: list[dict[str, Any]] = []
        step_max_delta = 0.0
        for row in range(rows):
            value_delta = float(abs(float(batch_values[row]) - float(serial_values[row])))
            if np.isfinite(value_delta):
                step_max_delta = max(step_max_delta, value_delta)
            if int(batch_indices[row]) != int(serial_indices[row]):
                mismatch: dict[str, Any] = {
                    "step_index": step_index,
                    "row": row,
                    "batch_index": int(batch_indices[row]),
                    "serial_index": int(serial_indices[row]),
                    "batch_value": float(batch_values[row]),
                    "serial_value": float(serial_values[row]),
                }
                if np.isfinite(value_delta):
                    mismatch["value_delta"] = value_delta
                mismatches.append(mismatch)
        audit["checked_steps"] = step_index + 1
        audit["checked_rows"] = int(audit["checked_rows"]) + int(rows)
        audit["max_abs_value_delta"] = max(float(audit.get("max_abs_value_delta") or 0.0), step_max_delta)
        if mismatches:
            audit["mismatch_steps"] = int(audit["mismatch_steps"]) + 1
            audit["mismatch_rows"] = int(audit["mismatch_rows"]) + len(mismatches)
            retained = list(audit.get("mismatches") or [])
            retained.extend(mismatches)
            audit["mismatches"] = retained[:16]
        audit["last_step"] = {
            "step_index": step_index,
            "rows": int(rows),
            "mismatch_rows": len(mismatches),
            "max_abs_value_delta": step_max_delta,
        }
        return dict(audit)

    def _record_batch_lm_head_fence(
        self,
        *,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        """Run serial per-row LM-head+argmax kernels as a fence only."""

        fence = getattr(self, "_batch_lm_head_fence", None)
        if not isinstance(fence, dict):
            fence = {
                "enabled": True,
                "kind": "serial_lm_head_argmax_kernel_only",
                "checked_steps": 0,
                "checked_rows": 0,
                "host_reads": 0,
            }
            self._batch_lm_head_fence = fence
        step_index = int(fence["checked_steps"])
        for row in range(rows):
            w8a16_linear_bf16_f32_out(
                self.batch_norm_out_bf16.ptr + row * self.hidden_nbytes,
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
            self.runtime.device_synchronize()
        fence["checked_steps"] = step_index + 1
        fence["checked_rows"] = int(fence["checked_rows"]) + int(rows)
        fence["last_step"] = {"step_index": step_index, "rows": int(rows), "host_reads": 0}
        return dict(fence)

    def _record_batch_final_norm_audit(
        self,
        *,
        hidden: Tensor,
        batch_indices: np.ndarray,
        batch_values: np.ndarray,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        """Compare batched sampler output with a serial final RMSNorm/cast path.

        Each row is re-normalized and re-cast through the serial c=1 sampler
        suffix from the same hidden row that fed the batched sampler.  A clean
        audit clears final RMSNorm, FP16->BF16 cast, LM-head projection, and
        argmax for the observed hidden rows; generated-token drift must then come
        from hidden-state production before the sampler or from timing/probe
        sensitivity outside the audited suffix.
        """

        audit = getattr(self, "_batch_final_norm_audit", None)
        if not isinstance(audit, dict):
            audit = {
                "enabled": True,
                "checked_steps": 0,
                "checked_rows": 0,
                "mismatch_steps": 0,
                "mismatch_rows": 0,
                "max_abs_value_delta": 0.0,
                "mismatches": [],
            }
            self._batch_final_norm_audit = audit
        step_index = int(audit["checked_steps"])
        serial_indices = np.empty((rows,), dtype=np.int64)
        serial_values = np.empty((rows,), dtype=np.float32)
        for row in range(rows):
            row_hidden_ptr = hidden.ptr + row * self.hidden_nbytes
            paro_rmsnorm_out_fp16(
                row_hidden_ptr,
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
            self.runtime.device_synchronize()
            serial_index = np.empty((1,), dtype=np.int64)
            serial_value = np.empty((1,), dtype=np.float32)
            copy_device_to_host(host_array_ptr(serial_index), self.lm_out_index, runtime=self.runtime)
            copy_device_to_host(host_array_ptr(serial_value), self.lm_out_value, runtime=self.runtime)
            serial_indices[row] = serial_index[0]
            serial_values[row] = serial_value[0]
        mismatches: list[dict[str, Any]] = []
        step_max_delta = 0.0
        for row in range(rows):
            value_delta = float(abs(float(batch_values[row]) - float(serial_values[row])))
            if np.isfinite(value_delta):
                step_max_delta = max(step_max_delta, value_delta)
            if int(batch_indices[row]) != int(serial_indices[row]):
                mismatch: dict[str, Any] = {
                    "step_index": step_index,
                    "row": row,
                    "batch_index": int(batch_indices[row]),
                    "serial_index": int(serial_indices[row]),
                    "batch_value": float(batch_values[row]),
                    "serial_value": float(serial_values[row]),
                }
                if np.isfinite(value_delta):
                    mismatch["value_delta"] = value_delta
                mismatches.append(mismatch)
        audit["checked_steps"] = step_index + 1
        audit["checked_rows"] = int(audit["checked_rows"]) + int(rows)
        audit["max_abs_value_delta"] = max(float(audit.get("max_abs_value_delta") or 0.0), step_max_delta)
        if mismatches:
            audit["mismatch_steps"] = int(audit["mismatch_steps"]) + 1
            audit["mismatch_rows"] = int(audit["mismatch_rows"]) + len(mismatches)
            retained = list(audit.get("mismatches") or [])
            retained.extend(mismatches)
            audit["mismatches"] = retained[:16]
        audit["last_step"] = {
            "step_index": step_index,
            "rows": int(rows),
            "mismatch_rows": len(mismatches),
            "max_abs_value_delta": step_max_delta,
        }
        return dict(audit)

    def _record_batch_final_norm_fence(
        self,
        *,
        hidden: Tensor,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        """Run serial per-row final RMSNorm+cast kernels as a fence only."""

        fence = getattr(self, "_batch_final_norm_fence", None)
        if not isinstance(fence, dict):
            fence = {
                "enabled": True,
                "kind": "serial_final_norm_cast_kernel_only",
                "checked_steps": 0,
                "checked_rows": 0,
                "host_reads": 0,
            }
            self._batch_final_norm_fence = fence
        step_index = int(fence["checked_steps"])
        for row in range(rows):
            row_hidden_ptr = hidden.ptr + row * self.hidden_nbytes
            paro_rmsnorm_out_fp16(
                row_hidden_ptr,
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
            self.runtime.device_synchronize()
        fence["checked_steps"] = step_index + 1
        fence["checked_rows"] = int(fence["checked_rows"]) + int(rows)
        fence["last_step"] = {"step_index": step_index, "rows": int(rows), "host_reads": 0}
        return dict(fence)

    def _record_batch_final_rmsnorm_fence(
        self,
        *,
        hidden: Tensor,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        """Run serial per-row final RMSNorm kernels as a fence only."""

        fence = getattr(self, "_batch_final_rmsnorm_fence", None)
        if not isinstance(fence, dict):
            fence = {
                "enabled": True,
                "kind": "serial_final_rmsnorm_kernel_only",
                "checked_steps": 0,
                "checked_rows": 0,
                "host_reads": 0,
            }
            self._batch_final_rmsnorm_fence = fence
        step_index = int(fence["checked_steps"])
        for row in range(rows):
            row_hidden_ptr = hidden.ptr + row * self.hidden_nbytes
            paro_rmsnorm_out_fp16(
                row_hidden_ptr,
                self.norm_weight.tensor.ptr,
                self.norm_out.ptr,
                1,
                self.config.hidden_size,
                self.config.rms_norm_eps,
                stream=stream,
                library=self.libraries["norm"],
                runtime=self.runtime,
            )
            self.runtime.device_synchronize()
        fence["checked_steps"] = step_index + 1
        fence["checked_rows"] = int(fence["checked_rows"]) + int(rows)
        fence["last_step"] = {"step_index": step_index, "rows": int(rows), "host_reads": 0}
        return dict(fence)

    def _record_batch_final_rmsnorm_temp_fence(
        self,
        *,
        hidden: Tensor,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        """Run serial final RMSNorm kernels into a dedicated temp buffer."""

        fence = getattr(self, "_batch_final_rmsnorm_temp_fence", None)
        if not isinstance(fence, dict):
            scratch = getattr(self, "_batch_final_rmsnorm_temp_buffer", None)
            if not isinstance(scratch, DeviceBuffer):
                scratch = malloc(self.hidden_nbytes, runtime=self.runtime)
                self.buffers.append(scratch)
                self._batch_final_rmsnorm_temp_buffer = scratch
            fence = {
                "enabled": True,
                "kind": "serial_final_rmsnorm_temp_kernel_only",
                "checked_steps": 0,
                "checked_rows": 0,
                "host_reads": 0,
                "scratch_ptr": int(scratch.ptr),
            }
            self._batch_final_rmsnorm_temp_fence = fence
        scratch = getattr(self, "_batch_final_rmsnorm_temp_buffer", None)
        if not isinstance(scratch, DeviceBuffer):
            scratch = malloc(self.hidden_nbytes, runtime=self.runtime)
            self.buffers.append(scratch)
            self._batch_final_rmsnorm_temp_buffer = scratch
            fence["scratch_ptr"] = int(scratch.ptr)
        step_index = int(fence["checked_steps"])
        for row in range(rows):
            row_hidden_ptr = hidden.ptr + row * self.hidden_nbytes
            paro_rmsnorm_out_fp16(
                row_hidden_ptr,
                self.norm_weight.tensor.ptr,
                scratch.ptr,
                1,
                self.config.hidden_size,
                self.config.rms_norm_eps,
                stream=stream,
                library=self.libraries["norm"],
                runtime=self.runtime,
            )
            self.runtime.device_synchronize()
        fence["checked_steps"] = step_index + 1
        fence["checked_rows"] = int(fence["checked_rows"]) + int(rows)
        fence["last_step"] = {"step_index": step_index, "rows": int(rows), "host_reads": 0}
        return dict(fence)

    def _record_batch_final_cast_temp_fence(
        self,
        *,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        """Run serial per-row final FP16->BF16 cast kernels into a temp buffer."""

        fence = getattr(self, "_batch_final_cast_temp_fence", None)
        if not isinstance(fence, dict):
            scratch = getattr(self, "_batch_final_cast_temp_buffer", None)
            if not isinstance(scratch, DeviceBuffer):
                scratch = malloc(self.hidden_nbytes, runtime=self.runtime)
                self.buffers.append(scratch)
                self._batch_final_cast_temp_buffer = scratch
            fence = {
                "enabled": True,
                "kind": "serial_final_cast_temp_kernel_only",
                "checked_steps": 0,
                "checked_rows": 0,
                "host_reads": 0,
                "scratch_ptr": int(scratch.ptr),
            }
            self._batch_final_cast_temp_fence = fence
        scratch = getattr(self, "_batch_final_cast_temp_buffer", None)
        if not isinstance(scratch, DeviceBuffer):
            scratch = malloc(self.hidden_nbytes, runtime=self.runtime)
            self.buffers.append(scratch)
            self._batch_final_cast_temp_buffer = scratch
            fence["scratch_ptr"] = int(scratch.ptr)
        step_index = int(fence["checked_steps"])
        for row in range(rows):
            fp16_to_bf16(
                self.batch_norm_out.ptr + row * self.hidden_nbytes,
                scratch.ptr,
                self.config.hidden_size,
                stream=stream,
                library=self.libraries["cast"],
                runtime=self.runtime,
            )
            self.runtime.device_synchronize()
        fence["checked_steps"] = step_index + 1
        fence["checked_rows"] = int(fence["checked_rows"]) + int(rows)
        fence["last_step"] = {"step_index": step_index, "rows": int(rows), "host_reads": 0}
        return dict(fence)

    def _record_batch_final_cast_tiny_fence(
        self,
        *,
        rows: int,
        stream: int,
    ) -> dict[str, Any]:
        """Run one-element serial FP16->BF16 cast kernels into a temp buffer."""

        return self._record_batch_final_cast_elems_fence(rows=rows, stream=stream, elements=1, tiny_alias=True)

    def _record_batch_final_cast_elems_fence(
        self,
        *,
        rows: int,
        stream: int,
        elements: int,
        tiny_alias: bool = False,
        attr_name: str | None = None,
        buffer_name: str | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Run a prefix of each normalized FP16 row through FP16->BF16 cast into temp."""

        elements_per_row = max(1, min(int(elements), int(self.config.hidden_size)))
        if attr_name is None:
            attr_name = "_batch_final_cast_tiny_fence" if tiny_alias else "_batch_final_cast_elems_fence"
        if buffer_name is None:
            buffer_name = "_batch_final_cast_tiny_buffer" if tiny_alias else "_batch_final_cast_elems_buffer"
        if kind is None:
            kind = "serial_final_cast_temp_1elem_kernel_only" if tiny_alias else "serial_final_cast_temp_nelems_kernel_only"
        scratch_nbytes = elements_per_row * DType.BF16.itemsize
        fence = getattr(self, attr_name, None)
        scratch = getattr(self, buffer_name, None)
        if not isinstance(fence, dict) or int(fence.get("elements_per_row", 0)) != elements_per_row:
            if not isinstance(scratch, DeviceBuffer) or int(scratch.nbytes) < scratch_nbytes:
                scratch = malloc(scratch_nbytes, runtime=self.runtime)
                self.buffers.append(scratch)
                setattr(self, buffer_name, scratch)
            fence = {
                "enabled": True,
                "kind": kind,
                "checked_steps": 0,
                "checked_rows": 0,
                "host_reads": 0,
                "elements_per_row": elements_per_row,
                "scratch_nbytes": scratch_nbytes,
                "scratch_ptr": int(scratch.ptr),
            }
            setattr(self, attr_name, fence)
        if not isinstance(scratch, DeviceBuffer) or int(scratch.nbytes) < scratch_nbytes:
            scratch = malloc(scratch_nbytes, runtime=self.runtime)
            self.buffers.append(scratch)
            setattr(self, buffer_name, scratch)
            fence["scratch_ptr"] = int(scratch.ptr)
            fence["scratch_nbytes"] = scratch_nbytes
        step_index = int(fence["checked_steps"])
        for row in range(rows):
            fp16_to_bf16(
                self.batch_norm_out.ptr + row * self.hidden_nbytes,
                scratch.ptr,
                elements_per_row,
                stream=stream,
                library=self.libraries["cast"],
                runtime=self.runtime,
            )
            self.runtime.device_synchronize()
        fence["checked_steps"] = step_index + 1
        fence["checked_rows"] = int(fence["checked_rows"]) + int(rows)
        fence["last_step"] = {
            "step_index": step_index,
            "rows": int(rows),
            "host_reads": 0,
            "elements_per_row": elements_per_row,
        }
        return dict(fence)

    def _record_batch_stabilize_cast_elems_fence(
        self,
        *,
        rows: int,
        stream: int,
        elements: int,
    ) -> dict[str, Any]:
        """Run an opt-in post-sampler cast-work fence without marking diagnostics."""

        return self._record_batch_final_cast_elems_fence(
            rows=rows,
            stream=stream,
            elements=elements,
            attr_name="_batch_stabilize_cast_elems_fence",
            buffer_name="_batch_stabilize_cast_elems_buffer",
            kind="batch_sampler_stabilize_cast_nelems_kernel",
        )

    def _record_batch_sync_fence(self, *, rows: int) -> dict[str, Any]:
        """Run extra device synchronizations as a no-kernel fence."""

        fence = getattr(self, "_batch_sync_fence", None)
        if not isinstance(fence, dict):
            fence = {
                "enabled": True,
                "kind": "device_synchronize_only",
                "checked_steps": 0,
                "checked_rows": 0,
                "host_reads": 0,
                "device_synchronizes": 0,
            }
            self._batch_sync_fence = fence
        step_index = int(fence["checked_steps"])
        for _ in range(rows):
            self.runtime.device_synchronize()
        fence["checked_steps"] = step_index + 1
        fence["checked_rows"] = int(fence["checked_rows"]) + int(rows)
        fence["device_synchronizes"] = int(fence["device_synchronizes"]) + int(rows)
        fence["last_step"] = {
            "step_index": step_index,
            "rows": int(rows),
            "host_reads": 0,
            "device_synchronizes": int(rows),
        }
        return dict(fence)

    def _record_batch_suffix_fence(
        self,
        *,
        hidden: Tensor,
        rows: int,
        stream: int,
        host_read: bool = True,
    ) -> dict[str, Any]:
        """Run a serial sampler suffix as a timing/scratch fence only.

        This intentionally does not compare tokens; the final-norm audit does
        that.  It answers a narrower question for the c2 flake: whether the
        serial suffix work, and optionally the host reads, after each batched
        sample are enough to stabilize the next decode step even without
        retaining parity details.
        """

        kind = (
            "serial_final_norm_cast_lm_head_argmax_host_read"
            if host_read
            else "serial_final_norm_cast_lm_head_argmax_kernel_only"
        )
        fence = getattr(self, "_batch_suffix_fence", None)
        if not isinstance(fence, dict) or fence.get("kind") != kind:
            fence = {
                "enabled": True,
                "kind": kind,
                "checked_steps": 0,
                "checked_rows": 0,
                "host_reads": 0,
            }
            self._batch_suffix_fence = fence
        step_index = int(fence["checked_steps"])
        serial_index = np.empty((1,), dtype=np.int64)
        serial_value = np.empty((1,), dtype=np.float32)
        for row in range(rows):
            row_hidden_ptr = hidden.ptr + row * self.hidden_nbytes
            paro_rmsnorm_out_fp16(
                row_hidden_ptr,
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
            self.runtime.device_synchronize()
            if host_read:
                copy_device_to_host(host_array_ptr(serial_index), self.lm_out_index, runtime=self.runtime)
                copy_device_to_host(host_array_ptr(serial_value), self.lm_out_value, runtime=self.runtime)
        step_host_reads = int(rows) * 2 if host_read else 0
        fence["checked_steps"] = step_index + 1
        fence["checked_rows"] = int(fence["checked_rows"]) + int(rows)
        fence["host_reads"] = int(fence["host_reads"]) + step_host_reads
        fence["last_step"] = {"step_index": step_index, "rows": int(rows), "host_reads": step_host_reads}
        return dict(fence)

    def _step_batch_from_device_tokens(
        self,
        *,
        rows: int,
        positions: tuple[int, ...],
        slots: tuple[int, ...],
        advance_positions: bool,
        stream: int = 0,
    ) -> None:
        """Run one device-resident c>1 decode step (C3.0b pieces A+B+C).

        The next-token ids are read straight from ``batch_lm_out_index``
        (device), gathered into batch embeddings (piece A), passed through the
        c-aware layer kernels, and the resulting hidden is projected + argmaxed
        back into ``batch_lm_out_index`` (piece B) -- all device-resident, with
        no host token list or sampler readback on the compute path.  When
        ``advance_positions`` is set the device decode position/context counters
        are incremented on-stream (piece C) so a captured graph can self-advance
        across replays; eager callers that re-set positions per step pass
        ``advance_positions=False``.
        """

        if rows <= 0:
            raise ValueError("rows must be positive")
        if rows > self.max_batch_size:
            raise ValueError("rows exceed max_batch_size")
        self._set_batch_token_embeddings_from_ptr(self.batch_lm_out_index.ptr, rows=rows, stream=stream)
        hidden = self._run_layers_batch_decode(rows=rows, positions=positions, slots=slots, stream=stream)
        self._write_batch_next_tokens_device(hidden, rows=rows, stream=stream)
        if advance_positions:
            advance_decode_positions_i64(
                self.position_buf.ptr,
                self.context_buf.ptr,
                rows,
                stream=stream,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )

    def _read_batch_next_tokens(self, *, rows: int) -> tuple[Qwen35ParoAutoregressiveStepResult, ...]:
        """Read the device-resident next-token argmax back into host step results.

        Used by the eager ``device_resident`` driver to materialize the tokens a
        captured graph would otherwise leave device-resident; not part of the
        capture region.
        """

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

    def _write_batch_next_tokens_device(self, hidden: Tensor, *, rows: int, stream: int = 0) -> None:
        """Device-resident batched LM-head -> argmax into ``batch_lm_out_index``.

        C3.0b piece B.  Runs the token-determining core of the row-aware
        ``batched_lm_head`` sampler -- batch final RMSNorm -> fp16->bf16 cast ->
        w8a16 LM-head projection -> batch argmax -- writing the next-token id of
        each active row into ``batch_lm_out_index`` (and its value into
        ``batch_lm_out_value``) with **no** host synchronize or device->host
        copy.  This is exactly the kernel sequence the eager
        :meth:`_sample_batch_from_hidden` batched path executes before its
        host readback; the readback plus the stabilize/audit fences are
        eager-only diagnostics that do not affect the argmax result.  Because
        nothing leaves the device, the write is safe to capture inside a c>1
        decode graph and feeds straight back into
        :meth:`_set_batch_token_embeddings_from_ptr` for the next step.
        """

        if rows <= 0:
            raise ValueError("rows must be positive")
        if rows > self.max_batch_size:
            raise ValueError("rows exceed max_batch_size")
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
        batch_argmax_f32(
            self.batch_lm_logits.ptr,
            self.batch_lm_block_values.ptr,
            self.batch_lm_block_indices.ptr,
            self.batch_lm_out_index.ptr,
            self.batch_lm_out_value.ptr,
            rows,
            self.vocab_size,
            threads=self.lm_head_threads,
            stream=stream,
            library=self.libraries["lm_head"],
            runtime=self.runtime,
        )

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
        sampler_norm_path = os.environ.get("HIPENGINE_QWEN35_BATCH_SAMPLE_NORM_PATH", "batch").strip() or "batch"
        if sampler_norm_path not in {"batch", "per_row"}:
            raise ValueError("HIPENGINE_QWEN35_BATCH_SAMPLE_NORM_PATH must be batch or per_row")
        sampler_cast_path = os.environ.get("HIPENGINE_QWEN35_BATCH_SAMPLE_CAST_PATH", "auto").strip() or "auto"
        if sampler_cast_path not in {"auto", "batch", "per_row"}:
            raise ValueError("HIPENGINE_QWEN35_BATCH_SAMPLE_CAST_PATH must be auto, batch, or per_row")
        effective_sampler_cast_path = sampler_norm_path if sampler_cast_path == "auto" else sampler_cast_path
        sampler_argmax_mode = os.environ.get("HIPENGINE_QWEN35_BATCH_SAMPLE_ARGMAX_MODE", "batch").strip() or "batch"
        if sampler_argmax_mode not in {"batch", "serial_per_row"}:
            raise ValueError("HIPENGINE_QWEN35_BATCH_SAMPLE_ARGMAX_MODE must be batch or serial_per_row")
        sampler_argmax_audit = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_ARGMAX_AUDIT")
        sampler_lm_head_audit = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_LM_HEAD_AUDIT")
        sampler_lm_head_kernel_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_LM_HEAD_KERNEL_FENCE")
        sampler_final_norm_audit = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_NORM_AUDIT")
        sampler_final_norm_kernel_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_NORM_KERNEL_FENCE")
        sampler_final_rmsnorm_kernel_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_RMSNORM_KERNEL_FENCE")
        sampler_final_rmsnorm_temp_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_RMSNORM_TEMP_FENCE")
        sampler_final_cast_temp_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_TEMP_FENCE")
        sampler_final_cast_tiny_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_TINY_FENCE")
        sampler_final_cast_elems_fence = max(0, _env_int("HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_ELEMS_FENCE", 0))
        sampler_final_cast_elems_fence = min(sampler_final_cast_elems_fence, int(self.config.hidden_size))
        sampler_stabilize_cast_elems = max(0, _env_int("HIPENGINE_QWEN35_BATCH_SAMPLE_STABILIZE_CAST_ELEMS", 0))
        sampler_stabilize_cast_elems = min(sampler_stabilize_cast_elems, int(self.config.hidden_size))
        sampler_sync_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_SYNC_FENCE")
        sampler_suffix_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_SUFFIX_FENCE")
        sampler_suffix_kernel_fence = _env_flag("HIPENGINE_QWEN35_BATCH_SAMPLE_SUFFIX_KERNEL_FENCE")
        sampler_execution = sampler_decision.to_json_dict()
        sampler_execution["final_norm_path"] = "serial_per_row" if sampler_decision.mode is BatchSamplerMode.SERIAL_LM_HEAD else sampler_norm_path
        sampler_execution["final_cast_path"] = "serial_per_row" if sampler_decision.mode is BatchSamplerMode.SERIAL_LM_HEAD else effective_sampler_cast_path
        sampler_execution["argmax_mode"] = "serial_per_row" if sampler_decision.mode is BatchSamplerMode.SERIAL_LM_HEAD else sampler_argmax_mode
        sampler_execution["argmax_audit_enabled"] = bool(sampler_argmax_audit and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["lm_head_audit_enabled"] = bool(sampler_lm_head_audit and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["lm_head_kernel_fence_enabled"] = bool(sampler_lm_head_kernel_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["final_norm_audit_enabled"] = bool(sampler_final_norm_audit and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["final_norm_kernel_fence_enabled"] = bool(sampler_final_norm_kernel_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["final_rmsnorm_kernel_fence_enabled"] = bool(sampler_final_rmsnorm_kernel_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["final_rmsnorm_temp_fence_enabled"] = bool(sampler_final_rmsnorm_temp_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["final_cast_temp_fence_enabled"] = bool(sampler_final_cast_temp_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["final_cast_tiny_fence_enabled"] = bool(sampler_final_cast_tiny_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["final_cast_elems_fence_elements"] = int(sampler_final_cast_elems_fence)
        sampler_execution["final_cast_elems_fence_enabled"] = bool(sampler_final_cast_elems_fence > 0 and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["stabilize_cast_elems"] = int(sampler_stabilize_cast_elems)
        sampler_execution["stabilize_cast_elems_enabled"] = bool(sampler_stabilize_cast_elems > 0 and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["sync_fence_enabled"] = bool(sampler_sync_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["suffix_fence_enabled"] = bool(sampler_suffix_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        sampler_execution["suffix_kernel_fence_enabled"] = bool(sampler_suffix_kernel_fence and sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "batch")
        if sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_norm_path == "per_row":
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final norm forced to per_row diagnostic")
        if sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and effective_sampler_cast_path == "per_row":
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final cast forced to per_row diagnostic")
        if sampler_decision.mode is BatchSamplerMode.BATCHED_LM_HEAD and sampler_argmax_mode == "serial_per_row":
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head argmax forced to serial_per_row diagnostic")
        if sampler_execution["argmax_audit_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head argmax audit enabled")
        if sampler_execution["lm_head_audit_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head serial projection audit enabled")
        if sampler_execution["lm_head_kernel_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head serial projection kernel fence enabled")
        if sampler_execution["final_norm_audit_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final norm/cast audit enabled")
        if sampler_execution["final_norm_kernel_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final norm/cast kernel fence enabled")
        if sampler_execution["final_rmsnorm_kernel_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final RMSNorm kernel fence enabled")
        if sampler_execution["final_rmsnorm_temp_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final RMSNorm temp-buffer kernel fence enabled")
        if sampler_execution["final_cast_temp_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final cast temp-buffer kernel fence enabled")
        if sampler_execution["final_cast_tiny_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final cast tiny temp-buffer kernel fence enabled")
        if sampler_execution["final_cast_elems_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head final cast prefix temp-buffer kernel fence enabled")
        if sampler_execution["sync_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head sync-only fence enabled")
        if sampler_execution["suffix_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head serial suffix fence enabled")
        if sampler_execution["suffix_kernel_fence_enabled"]:
            sampler_execution["native_row_aware_lm_head"] = False
            sampler_execution["blockers"].append("batched LM-head serial suffix kernel fence enabled")
        self._publish_batch_sampler_execution(sampler_execution)
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
        if sampler_norm_path == "per_row":
            for row in range(rows):
                row_hidden_ptr = hidden.ptr + row * self.hidden_nbytes
                row_norm_ptr = self.batch_norm_out.ptr + row * self.hidden_nbytes
                paro_rmsnorm_out_fp16(
                    row_hidden_ptr,
                    self.norm_weight.tensor.ptr,
                    row_norm_ptr,
                    1,
                    self.config.hidden_size,
                    self.config.rms_norm_eps,
                    stream=stream,
                    library=self.libraries["norm"],
                    runtime=self.runtime,
                )
        else:
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
        if effective_sampler_cast_path == "per_row":
            for row in range(rows):
                row_norm_ptr = self.batch_norm_out.ptr + row * self.hidden_nbytes
                row_norm_bf16_ptr = self.batch_norm_out_bf16.ptr + row * self.hidden_nbytes
                fp16_to_bf16(
                    row_norm_ptr,
                    row_norm_bf16_ptr,
                    self.config.hidden_size,
                    stream=stream,
                    library=self.libraries["cast"],
                    runtime=self.runtime,
                )
        else:
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
        if sampler_argmax_mode == "serial_per_row":
            results: list[Qwen35ParoAutoregressiveStepResult] = []
            for row in range(rows):
                argmax_f32(
                    self.batch_lm_logits.ptr + row * self.vocab_size * DType.FP32.itemsize,
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
                self.runtime.device_synchronize()
                results.append(self._read_sample())
            return tuple(results)
        batch_argmax_f32(
            self.batch_lm_logits.ptr,
            self.batch_lm_block_values.ptr,
            self.batch_lm_block_indices.ptr,
            self.batch_lm_out_index.ptr,
            self.batch_lm_out_value.ptr,
            rows,
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
        if sampler_execution["argmax_audit_enabled"]:
            sampler_execution["argmax_audit"] = self._record_batch_argmax_audit(
                batch_indices=index_host,
                batch_values=value_host,
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["lm_head_audit_enabled"]:
            sampler_execution["lm_head_audit"] = self._record_batch_lm_head_audit(
                batch_indices=index_host,
                batch_values=value_host,
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["lm_head_kernel_fence_enabled"]:
            sampler_execution["lm_head_fence"] = self._record_batch_lm_head_fence(
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["final_norm_audit_enabled"]:
            sampler_execution["final_norm_audit"] = self._record_batch_final_norm_audit(
                hidden=hidden,
                batch_indices=index_host,
                batch_values=value_host,
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["final_norm_kernel_fence_enabled"]:
            sampler_execution["final_norm_fence"] = self._record_batch_final_norm_fence(
                hidden=hidden,
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["final_rmsnorm_kernel_fence_enabled"]:
            sampler_execution["final_rmsnorm_fence"] = self._record_batch_final_rmsnorm_fence(
                hidden=hidden,
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["final_rmsnorm_temp_fence_enabled"]:
            sampler_execution["final_rmsnorm_temp_fence"] = self._record_batch_final_rmsnorm_temp_fence(
                hidden=hidden,
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["final_cast_temp_fence_enabled"]:
            sampler_execution["final_cast_temp_fence"] = self._record_batch_final_cast_temp_fence(
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["final_cast_tiny_fence_enabled"]:
            sampler_execution["final_cast_tiny_fence"] = self._record_batch_final_cast_tiny_fence(
                rows=rows,
                stream=stream,
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["final_cast_elems_fence_enabled"]:
            sampler_execution["final_cast_elems_fence"] = self._record_batch_final_cast_elems_fence(
                rows=rows,
                stream=stream,
                elements=int(sampler_execution["final_cast_elems_fence_elements"]),
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["stabilize_cast_elems_enabled"]:
            sampler_execution["stabilize_cast_elems_fence"] = self._record_batch_stabilize_cast_elems_fence(
                rows=rows,
                stream=stream,
                elements=int(sampler_execution["stabilize_cast_elems"]),
            )
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["sync_fence_enabled"]:
            sampler_execution["sync_fence"] = self._record_batch_sync_fence(rows=rows)
            self._publish_batch_sampler_execution(sampler_execution)
        if sampler_execution["suffix_fence_enabled"] or sampler_execution["suffix_kernel_fence_enabled"]:
            sampler_execution["suffix_fence"] = self._record_batch_suffix_fence(
                hidden=hidden,
                rows=rows,
                stream=stream,
                host_read=bool(sampler_execution["suffix_fence_enabled"]),
            )
            self._publish_batch_sampler_execution(sampler_execution)
        return tuple(
            Qwen35ParoAutoregressiveStepResult(
                token_id=int(index_host[row]),
                token_text=_decode_token_cached(self.tokenizer, int(index_host[row])),
                logit=float(value_host[row]),
            )
            for row in range(rows)
        )

    def configure_host_sampler(self, params: Any | None, state: RowSamplingState | None) -> None:
        """Configure the correctness-first host sampler for subsequent samples."""

        self._host_sampling_params = params
        self._host_sampling_state = state
        self._host_sampling_states_by_slot = None
        self._native_sampling_params = None
        self._native_sampling_state = None
        self._native_sampling_states_by_slot = None

    def configure_native_sampler(self, params: Any | None, state: RowSamplingState | None) -> None:
        """Configure the native GPU sampler for c=1 samples."""

        self._native_sampling_params = params
        self._native_sampling_state = state
        self._native_sampling_states_by_slot = None
        self._host_sampling_params = None
        self._host_sampling_state = None
        self._host_sampling_states_by_slot = None

    def configure_host_sampler_rows(
        self,
        params: Any | None,
        states_by_slot: Mapping[int, RowSamplingState] | None,
    ) -> None:
        """Configure per-slot host sampler state for c>N sampled batches."""

        self._host_sampling_params = params
        self._host_sampling_state = None
        self._native_sampling_params = None
        self._native_sampling_state = None
        self._native_sampling_states_by_slot = None
        if params is None or states_by_slot is None:
            self._host_sampling_states_by_slot = None
        else:
            self._host_sampling_states_by_slot = {int(slot): state for slot, state in states_by_slot.items()}

    def configure_native_sampler_rows(
        self,
        params: Any | None,
        states_by_slot: Mapping[int, RowSamplingState] | None,
    ) -> None:
        """Configure per-slot native GPU sampler state for serial c>N decode."""

        self._native_sampling_params = params
        self._native_sampling_state = None
        self._host_sampling_params = None
        self._host_sampling_state = None
        self._host_sampling_states_by_slot = None
        if params is None or states_by_slot is None:
            self._native_sampling_states_by_slot = None
        else:
            self._native_sampling_states_by_slot = {int(slot): state for slot, state in states_by_slot.items()}

    def _project_logits_device_from_hidden(self, hidden: Tensor, *, stream: int = 0) -> None:
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

    def _select_argmax_device_from_logits(self, *, stream: int = 0) -> None:
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

    def _sample_device_from_hidden(self, hidden: Tensor, *, stream: int = 0) -> None:
        self._project_logits_device_from_hidden(hidden, stream=stream)
        self._select_argmax_device_from_logits(stream=stream)

    def _sample_from_hidden(self, hidden: Tensor) -> Qwen35ParoAutoregressiveStepResult:
        if self._native_sampling_params is not None and self._native_sampling_state is not None:
            return self._sample_from_hidden_native(hidden, self._native_sampling_params, self._native_sampling_state)
        if self._host_sampling_params is not None and self._host_sampling_state is not None:
            return self._sample_from_hidden_host(hidden, self._host_sampling_params, self._host_sampling_state)
        self._sample_device_from_hidden(hidden)
        self.runtime.device_synchronize()
        return self._read_sample()

    def _sample_from_hidden_for_slot(
        self,
        hidden: Tensor,
        slot: int,
    ) -> Qwen35ParoAutoregressiveStepResult:
        native_state = self._native_sampler_state_for_slot(slot)
        native_sampling_params = getattr(self, "_native_sampling_params", None)
        if native_sampling_params is not None and native_state is not None:
            return self._sample_from_hidden_native(hidden, native_sampling_params, native_state)
        host_state = self._host_sampler_state_for_slot(slot)
        host_sampling_params = getattr(self, "_host_sampling_params", None)
        if host_sampling_params is not None and host_state is not None:
            return self._sample_from_hidden_host(hidden, host_sampling_params, host_state)
        return self._sample_from_hidden(hidden)

    def _host_sampler_state_for_slot(self, slot: int) -> RowSamplingState | None:
        states = getattr(self, "_host_sampling_states_by_slot", None)
        if not states:
            return None
        return states.get(int(slot))

    def _native_sampler_state_for_slot(self, slot: int) -> RowSamplingState | None:
        states = getattr(self, "_native_sampling_states_by_slot", None)
        if not states:
            return None
        return states.get(int(slot))

    def _sample_from_hidden_native(
        self,
        hidden: Tensor,
        params: Any,
        state: RowSamplingState,
    ) -> Qwen35ParoAutoregressiveStepResult:
        if state.has_forced_tokens:
            return self._sample_from_hidden_host(hidden, params, state)
        self._project_logits_device_from_hidden(hidden)
        logits_ptr = self._native_sampler_logits_ptr(params, state)
        library = self._native_sampler_library_handle()
        temperature_buf = self._native_sampler_cached_scalar(
            ("temperature",),
            float(getattr(params, "temperature", 0.0)),
            np.float32,
        )
        seed_buf = self._native_sampler_cached_scalar(
            ("seed",),
            int(state.seed) & ((1 << 64) - 1),
            np.uint64,
        )
        out_indices = self._native_sampler_buffer("_native_sampler_out_indices_i32", DType.INT32.itemsize)
        out_logprobs = self._native_sampler_buffer("_native_sampler_out_logprobs_f32", DType.FP32.itemsize)
        top_k = int(getattr(params, "top_k", 0))
        top_p = float(getattr(params, "top_p", 1.0))
        min_p = float(getattr(params, "min_p", 0.0))
        requested_top_logprobs = int(getattr(params, "top_logprobs", 0))
        if requested_top_logprobs > 0 and top_k > 0 and requested_top_logprobs > top_k:
            raise RuntimeError("native bounded top_logprobs require top_logprobs <= top_k")
        out_top_indices = None
        out_top_logprobs = None
        top_logprobs_width = top_k if top_k > 0 else requested_top_logprobs
        if requested_top_logprobs > 0:
            out_top_indices = self._native_sampler_buffer(
                "_native_sampler_top_indices_i32",
                top_logprobs_width * DType.INT32.itemsize,
            )
            out_top_logprobs = self._native_sampler_buffer(
                "_native_sampler_top_logprobs_f32",
                top_logprobs_width * DType.FP32.itemsize,
            )
        uses_probability_filter = top_p < 1.0 or min_p > 0.0
        if top_k > 0:
            top_p_buf = self._native_sampler_cached_scalar(("top_p",), top_p, np.float32) if uses_probability_filter else None
            min_p_buf = self._native_sampler_cached_scalar(("min_p",), min_p, np.float32) if uses_probability_filter else None
            sample_topk_temperature_f32_rows_i32(
                logits_ptr,
                temperature_buf.ptr,
                seed_buf.ptr,
                out_indices.ptr,
                out_logprobs.ptr,
                None if out_top_indices is None else out_top_indices.ptr,
                None if out_top_logprobs is None else out_top_logprobs.ptr,
                1,
                self.vocab_size,
                top_k,
                top_ps_f32_ptr=None if top_p_buf is None else top_p_buf.ptr,
                min_ps_f32_ptr=None if min_p_buf is None else min_p_buf.ptr,
                out_indices_i64_ptr=self.lm_out_index.ptr,
                out_values_f32_ptr=self.lm_out_value.ptr,
                step_index=state.step_index,
                threads=128,
                library=library,
                runtime=self.runtime,
            )
        elif uses_probability_filter:
            top_p_buf = self._native_sampler_cached_scalar(("top_p",), top_p, np.float32)
            min_p_buf = self._native_sampler_cached_scalar(("min_p",), min_p, np.float32)
            retained_counts = self._native_sampler_buffer("_native_sampler_retained_counts_i32", DType.INT32.itemsize)
            sample_top_p_temperature_f32_rows_i32(
                logits_ptr,
                temperature_buf.ptr,
                top_p_buf.ptr,
                min_p_buf.ptr,
                seed_buf.ptr,
                out_indices.ptr,
                out_logprobs.ptr,
                retained_counts.ptr,
                1,
                self.vocab_size,
                out_top_indices_i32_ptr=None if out_top_indices is None else out_top_indices.ptr,
                out_top_logprobs_f32_ptr=None if out_top_logprobs is None else out_top_logprobs.ptr,
                top_logprobs=requested_top_logprobs,
                out_indices_i64_ptr=self.lm_out_index.ptr,
                out_values_f32_ptr=self.lm_out_value.ptr,
                step_index=state.step_index,
                threads=128,
                library=library,
                runtime=self.runtime,
            )
        else:
            sample_temperature_f32_rows_i32(
                logits_ptr,
                temperature_buf.ptr,
                seed_buf.ptr,
                out_indices.ptr,
                out_logprobs.ptr,
                1,
                self.vocab_size,
                out_indices_i64_ptr=self.lm_out_index.ptr,
                out_values_f32_ptr=self.lm_out_value.ptr,
                step_index=state.step_index,
                threads=128,
                library=library,
                runtime=self.runtime,
            )
            if requested_top_logprobs > 0:
                sample_temperature_top_logprobs_f32_rows_i32(
                    logits_ptr,
                    temperature_buf.ptr,
                    out_top_indices.ptr,
                    out_top_logprobs.ptr,
                    1,
                    self.vocab_size,
                    requested_top_logprobs,
                    threads=128,
                    library=library,
                    runtime=self.runtime,
                )
        self.runtime.device_synchronize()
        index_i32 = np.empty((1,), dtype=np.int32)
        logprob_host = np.empty((1,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(index_i32), out_indices, runtime=self.runtime)
        copy_device_to_host(host_array_ptr(logprob_host), out_logprobs, runtime=self.runtime)
        token_id = int(index_i32[0])
        if token_id < 0 or token_id >= self.vocab_size:
            raise RuntimeError(f"native sampler selected invalid token id {token_id}")
        value_host = np.empty((1,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(value_host), self.lm_out_value, runtime=self.runtime)
        top_logprobs: tuple[tuple[int, float], ...] = ()
        if requested_top_logprobs > 0 and out_top_indices is not None and out_top_logprobs is not None:
            top_indices_host = np.empty((top_logprobs_width,), dtype=np.int32)
            top_logprobs_host = np.empty((top_logprobs_width,), dtype=np.float32)
            copy_device_to_host(host_array_ptr(top_indices_host), out_top_indices, runtime=self.runtime)
            copy_device_to_host(host_array_ptr(top_logprobs_host), out_top_logprobs, runtime=self.runtime)
            pairs: list[tuple[int, float]] = []
            for candidate_id, candidate_logprob in zip(top_indices_host, top_logprobs_host, strict=True):
                if len(pairs) >= requested_top_logprobs:
                    break
                token = int(candidate_id)
                logprob = float(candidate_logprob)
                if token < 0 or token >= self.vocab_size or not np.isfinite(logprob):
                    continue
                pairs.append((token, logprob))
            top_logprobs = tuple(pairs)
        state.observe(token_id)
        return Qwen35ParoAutoregressiveStepResult(
            token_id=token_id,
            token_text=_decode_token_cached(self.tokenizer, token_id),
            logit=float(value_host[0]),
            logprob=float(logprob_host[0]),
            top_logprobs=top_logprobs,
        )

    def _native_sampler_logits_ptr(self, params: Any, state: RowSamplingState) -> int:
        if not _native_sampler_needs_processors(params):
            return int(self.lm_logits.ptr)
        processed = self._native_sampler_buffer(
            "_native_sampler_processed_logits",
            self.vocab_size * DType.FP32.itemsize,
        )
        bias_pairs = normalize_logit_bias_pairs(getattr(params, "logit_bias", None))
        for token_id, _bias in bias_pairs:
            if int(token_id) >= self.vocab_size:
                raise ValueError(f"logit_bias token id {token_id} is outside vocab size {self.vocab_size}")
        history_pairs = tuple(
            (int(token), int(count))
            for token, count in sorted(state.history_counts().items())
            if 0 <= int(token) < self.vocab_size
        )
        suppress_ids = _native_sampler_suppress_token_ids(params)
        for token_id in suppress_ids:
            if int(token_id) < 0 or int(token_id) >= self.vocab_size:
                raise ValueError(f"suppress_token_ids token id {token_id} is outside vocab size {self.vocab_size}")
        min_tokens = int(getattr(params, "min_tokens", 0))
        eos_token_id = -1
        if min_tokens > 0:
            raw_eos_token_id = getattr(params, "eos_token_id", None)
            if raw_eos_token_id is None:
                raise ValueError("min_tokens requires eos_token_id")
            eos_token_id = int(raw_eos_token_id)
            if eos_token_id < 0 or eos_token_id >= self.vocab_size:
                raise ValueError(f"eos_token_id {eos_token_id} is outside vocab size {self.vocab_size}")
        bias_offsets = self._native_sampler_cached_upload(
            ("bias_offsets_i32", len(bias_pairs)),
            np.asarray([0, len(bias_pairs)], dtype=np.int32),
        )
        history_offsets = self._native_sampler_upload(
            "_native_sampler_history_offsets_i32",
            np.asarray([0, len(history_pairs)], dtype=np.int32),
        )
        suppress_offsets = self._native_sampler_cached_upload(
            ("suppress_offsets_i32", len(suppress_ids)),
            np.asarray([0, len(suppress_ids)], dtype=np.int32),
        )
        bias_ids = None
        bias_values = None
        if bias_pairs:
            bias_ids = self._native_sampler_cached_upload(
                ("bias_ids_i32", tuple(int(token) for token, _bias in bias_pairs)),
                np.asarray([int(token) for token, _bias in bias_pairs], dtype=np.int32),
            )
            bias_values = self._native_sampler_cached_upload(
                ("bias_values_f32", tuple(float(bias) for _token, bias in bias_pairs)),
                np.asarray([float(bias) for _token, bias in bias_pairs], dtype=np.float32),
            )
        history_ids = None
        history_counts = None
        if history_pairs:
            history_ids = self._native_sampler_upload(
                "_native_sampler_history_ids_i32",
                np.asarray([token for token, _count in history_pairs], dtype=np.int32),
            )
            history_counts = self._native_sampler_upload(
                "_native_sampler_history_counts_i32",
                np.asarray([count for _token, count in history_pairs], dtype=np.int32),
            )
        suppress_ids_buf = None
        if suppress_ids:
            suppress_ids_buf = self._native_sampler_cached_upload(
                ("suppress_ids_i32", suppress_ids),
                np.asarray(suppress_ids, dtype=np.int32),
            )
        min_tokens_buf = None
        eos_token_ids_buf = None
        step_indices_buf = None
        if min_tokens > 0:
            min_tokens_buf = self._native_sampler_cached_scalar(("min_tokens",), min_tokens, np.int32)
            eos_token_ids_buf = self._native_sampler_cached_scalar(("eos_token_id",), eos_token_id, np.int32)
            step_indices_buf = self._native_sampler_upload(
                "_native_sampler_step_indices_u64",
                np.asarray([int(state.step_index)], dtype=np.uint64),
            )
        repetition = self._native_sampler_cached_scalar(
            ("repetition",),
            float(getattr(params, "repetition_penalty", 1.0)),
            np.float32,
        )
        presence = self._native_sampler_cached_scalar(
            ("presence",),
            float(getattr(params, "presence_penalty", 0.0)),
            np.float32,
        )
        frequency = self._native_sampler_cached_scalar(
            ("frequency",),
            float(getattr(params, "frequency_penalty", 0.0)),
            np.float32,
        )
        apply_processors_f32_rows(
            self.lm_logits.ptr,
            processed.ptr,
            bias_offsets.ptr,
            None if bias_ids is None else bias_ids.ptr,
            None if bias_values is None else bias_values.ptr,
            history_offsets.ptr,
            None if history_ids is None else history_ids.ptr,
            None if history_counts is None else history_counts.ptr,
            repetition.ptr,
            presence.ptr,
            frequency.ptr,
            1,
            self.vocab_size,
            suppress_offsets_i32_ptr=suppress_offsets.ptr,
            suppress_token_ids_i32_ptr=None if suppress_ids_buf is None else suppress_ids_buf.ptr,
            min_tokens_i32_ptr=None if min_tokens_buf is None else min_tokens_buf.ptr,
            eos_token_ids_i32_ptr=None if eos_token_ids_buf is None else eos_token_ids_buf.ptr,
            step_indices_u64_ptr=None if step_indices_buf is None else step_indices_buf.ptr,
            threads=128,
            library=self._native_sampler_library_handle(),
            runtime=self.runtime,
        )
        return int(processed.ptr)

    def _native_sampler_library_handle(self):
        library = self._native_sampler_library
        if library is None:
            library = build_sampler(load=True)
            self._native_sampler_library = library
        return library

    def _native_sampler_buffer(self, name: str, nbytes: int) -> DeviceBuffer:
        required = max(int(nbytes), 4)
        current = getattr(self, name, None)
        if not isinstance(current, DeviceBuffer) or int(current.nbytes) < required:
            current = malloc(required, runtime=self.runtime)
            self.buffers.append(current)
            setattr(self, name, current)
        return current

    def _native_sampler_upload(self, name: str, array: np.ndarray) -> DeviceBuffer:
        host = np.ascontiguousarray(array)
        buffer = self._native_sampler_buffer(name, int(host.nbytes))
        copy_host_to_device(buffer, host_array_ptr(host), int(host.nbytes), runtime=self.runtime)
        return buffer

    def _native_sampler_cached_scalar(self, cache_key: tuple[Any, ...], value: Any, dtype: Any) -> DeviceBuffer:
        np_dtype = np.dtype(dtype)
        scalar = np_dtype.type(value).item()
        key = (*cache_key, np_dtype.str, scalar)
        cache = getattr(self, "_native_sampler_cached_uploads", None)
        if not isinstance(cache, dict):
            cache = {}
            self._native_sampler_cached_uploads = cache
        buffer = cache.get(key)
        if not isinstance(buffer, DeviceBuffer):
            host = np.ascontiguousarray(np.asarray([scalar], dtype=np_dtype))
            buffer = malloc(max(int(host.nbytes), 4), runtime=self.runtime)
            self.buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(host), int(host.nbytes), runtime=self.runtime)
            cache[key] = buffer
        return buffer

    def _native_sampler_cached_upload(self, cache_key: tuple[Any, ...], array: np.ndarray) -> DeviceBuffer:
        host = np.ascontiguousarray(array)
        key = (
            *cache_key,
            str(host.dtype),
            tuple(int(dim) for dim in host.shape),
            host.tobytes(),
        )
        cache = getattr(self, "_native_sampler_cached_uploads", None)
        if not isinstance(cache, dict):
            cache = {}
            self._native_sampler_cached_uploads = cache
        buffer = cache.get(key)
        if not isinstance(buffer, DeviceBuffer) or int(buffer.nbytes) < int(host.nbytes):
            buffer = malloc(max(int(host.nbytes), 4), runtime=self.runtime)
            self.buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(host), int(host.nbytes), runtime=self.runtime)
            cache[key] = buffer
        return buffer

    def _sample_from_hidden_host(
        self,
        hidden: Tensor,
        params: Any,
        state: RowSamplingState,
    ) -> Qwen35ParoAutoregressiveStepResult:
        self._project_logits_device_from_hidden(hidden)
        self.runtime.device_synchronize()
        logits_host = np.empty((self.vocab_size,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(logits_host), self.lm_logits, runtime=self.runtime)
        sample = select_token(logits_host, params, state)
        index_host = np.array([sample.token_id], dtype=np.int64)
        value_host = np.array([sample.logit], dtype=np.float32)
        copy_host_to_device(self.lm_out_index, host_array_ptr(index_host), runtime=self.runtime)
        copy_host_to_device(self.lm_out_value, host_array_ptr(value_host), runtime=self.runtime)
        token_id = int(sample.token_id)
        return Qwen35ParoAutoregressiveStepResult(
            token_id=token_id,
            token_text=_decode_token_cached(self.tokenizer, token_id),
            logit=float(sample.logit),
            logprob=sample.logprob,
            top_logprobs=sample.top_logprobs,
            forced=sample.forced,
            forced_reason=sample.forced_reason,
            forced_tokens_remaining=sample.forced_tokens_remaining,
        )

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

    def copy_slot_state(self, src_slot: int, dst_slot: int, *, stream: int = 0, kv_rows: int | None = None) -> None:
        """Copy resident decode state/KV metadata between physical slots.

        This is a correctness-first branch primitive for serial speculative
        verification: slot ``dst_slot`` receives the same recurrent state,
        full-attention KV cache, hidden scratch rows, and position/context
        scalars as ``src_slot``.  By default it copies the whole per-slot KV
        capacity; callers that know the live KV prefix can pass ``kv_rows`` to
        avoid copying unused future cache rows.
        """

        self._check_slot(src_slot)
        self._check_slot(dst_slot)
        if src_slot == dst_slot:
            return
        if kv_rows is not None and int(kv_rows) < 0:
            raise ValueError("kv_rows must be non-negative")
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
            cache_nbytes = cache_stride
            if kv_rows is not None:
                if len(key_cache.shape) < 3:
                    raise ValueError(f"expected paged KV cache shape, got {key_cache.shape}")
                capacity_rows = int(key_cache.shape[0]) * int(key_cache.shape[1])
                row_elems = int(np.prod(key_cache.shape[2:]))
                copy_rows = min(int(kv_rows), capacity_rows)
                cache_nbytes = copy_rows * row_elems * key_cache.dtype.itemsize
            if cache_nbytes > 0:
                self.runtime.memcpy_async(
                    key_buf.ptr + int(dst_slot) * cache_stride,
                    key_buf.ptr + int(src_slot) * cache_stride,
                    cache_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                self.runtime.memcpy_async(
                    value_buf.ptr + int(dst_slot) * cache_stride,
                    value_buf.ptr + int(src_slot) * cache_stride,
                    cache_nbytes,
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

    def _verify_chain_linear_tloop_enabled(self) -> bool:
        value = os.environ.get("HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP")
        if value is None or value.strip() == "":
            return True
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _verify_gpu_accept_enabled(self) -> bool:
        value = os.environ.get("HIPENGINE_VERIFY_GPU_ACCEPT")
        if value is None or value.strip() == "":
            # M12 verifier cycles need the GPU-resident accept path by default;
            # set HIPENGINE_VERIFY_GPU_ACCEPT=0 for the legacy CPU-oracle read
            # path, or =validate to cross-check the GPU payload.
            return True
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _verify_accept_packed_payload_enabled(self) -> bool:
        value = os.environ.get("HIPENGINE_VERIFY_ACCEPT_PACKED_PAYLOAD")
        if value is None or value.strip() == "":
            return True
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _verify_accept_updates_position_enabled(self, batch: TargetVerifyBatch) -> bool:
        if batch.mode != "verify_chain":
            return False
        if len(batch.request_ids) != 1:
            return False
        if not self._verify_gpu_accept_enabled():
            return False
        if not self._verify_accept_packed_payload_enabled():
            return False
        return _env_flag("HIPENGINE_VERIFY_ACCEPT_UPDATES_POSITION", False)

    def _verify_packed_dynamic_metadata_enabled(self, batch: TargetVerifyBatch) -> bool:
        if batch.mode != "verify_chain":
            return False
        return _env_flag("HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA", True)

    def _verify_scratch_cache_enabled(self) -> bool:
        return _env_flag("HIPENGINE_VERIFY_SCRATCH_CACHE", True)

    def _should_use_chain_tloop_linear_verify(self, batch: TargetVerifyBatch, *, rows: int, graph_mode: str) -> bool:
        if not self._verify_chain_linear_tloop_enabled():
            return False
        if batch.mode != "verify_chain" or len(batch.request_ids) != 1:
            return False
        if tuple(batch.root_rows) != (0,) or tuple(batch.candidate_rows) != tuple(range(1, rows)):
            return False
        if tuple(batch.parent_rows) != (-1, *tuple(range(0, rows - 1))):
            return False
        return all(bool(flag) for flag in batch.active_mask)

    def _ensure_full_decode_batch_partials(self, *, rows: int, num_splits: int) -> tuple[Tensor, Tensor, Tensor]:
        owner = self._prefill_scratch_owner()
        partial_out = owner.workspace.reserve_tensor(
            "attn.decode_batch.partial_out",
            (rows, self.config.num_attention_heads, num_splits, self.config.head_dim),
            DType.FP32,
        )
        partial_m = owner.workspace.reserve_tensor(
            "attn.decode_batch.partial_m",
            (rows, self.config.num_attention_heads, num_splits),
            DType.FP32,
        )
        partial_l = owner.workspace.reserve_tensor(
            "attn.decode_batch.partial_l",
            (rows, self.config.num_attention_heads, num_splits),
            DType.FP32,
        )
        return partial_out, partial_m, partial_l

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
        chain_attn_mode: str = "c1_loop",
        canonicalize_after: bool = True,
        synchronize_after_commit: bool = True,
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

        ``chain_attn_mode`` selects how the full-attention layers process the
        B+1 verifier rows. ``c1_loop`` (default) drives the resident decode
        kernels row-by-row, matching the path validated in earlier diagnostics.
        ``batched`` reuses the native prefill-style primitives (RMSNorm + rotate
        + QKV projection over rows, prompt-style batched K/V append, gated
        prefill GQA attention with per-row causal limit, then c=1 MoE with
        ``tokens=rows``) so each full-attention layer runs in one batched pass.
        ``decode_batched`` keeps the row-batched projections/KV/MoE staging but
        swaps the prefill attention kernel for a small-B row-batched decode GQA
        split-K primitive; it is currently graph-off only because the split
        count changes with verifier position buckets.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        if batch.mode != "verify_chain":
            raise ValueError("bulk verifier currently supports verify_chain only")
        if len(batch.request_ids) != 1:
            raise ValueError("bulk verifier E2E path currently supports one request")
        if chain_attn_mode not in {"c1_loop", "batched", "decode_batched"}:
            raise ValueError("chain_attn_mode must be 'c1_loop', 'batched', or 'decode_batched'")
        if chain_attn_mode == "decode_batched" and graph_mode != "off":
            raise ValueError("chain_attn_mode='decode_batched' currently requires graph_mode='off'")
        # NOTE: chain_attn_mode='batched' is now allowed with graph_mode!='off';
        # the verifier graph cache key includes chain_attn_mode/linear_attn_mode
        # so batched and c1_loop captures do not alias.  See `docs/MTP.md` M12.1.
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
        linear_attn_mode = "chain_tloop" if self._should_use_chain_tloop_linear_verify(batch, rows=rows, graph_mode=graph_mode) else "tree_tloop"
        try:
            if graph_mode == "off":
                graph_info: dict[str, Any] = {
                    "mode": "off",
                    "status": "disabled",
                    "replayed": False,
                    "validation_passed": None,
                    "chain_attn_mode": chain_attn_mode,
                    "linear_attn_mode": linear_attn_mode,
                    "linear_attn_fallback": False,
                }
                self._launch_verify_chain_forward_accept(
                    batch,
                    base_slot=base_slot,
                    capture_ids=capture_ids,
                    capture_hidden_concat=capture_target,
                    capture_row_start=capture_target_start,
                    rows=rows,
                    stream=stream,
                    chain_attn_mode=chain_attn_mode,
                    linear_attn_mode=linear_attn_mode,
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
                    chain_attn_mode=chain_attn_mode,
                    linear_attn_mode=linear_attn_mode,
                    stream=stream,
                )
                graph_info["chain_attn_mode"] = chain_attn_mode
                graph_info["linear_attn_mode"] = linear_attn_mode
                graph_info["linear_attn_fallback"] = False
            gpu_payload = self._read_verify_accept_payload(len(batch.request_ids), stream=stream)
            if self._verify_gpu_accept_enabled():
                # Fast path: trust GPU accept-summary kernel, skip CPU top1 read + CPU accept.
                summary = TargetAcceptSummary.from_gpu_payload(batch, gpu_payload)
                selected_row = int(summary.commit_rows[0])
                gpu_accept_match = True
                # Optional validation against CPU oracle.
                if os.environ.get("HIPENGINE_VERIFY_GPU_ACCEPT", "").strip().lower() == "validate":
                    target_top1, target_values = self._read_verify_top1(rows)
                    cpu_result = batch.accept_from_top1(target_top1, transaction_id=0)
                    cpu_summary = TargetAcceptSummary.from_accept_result(batch, cpu_result)
                    gpu_accept_match = self._gpu_accept_payload_matches(gpu_payload, cpu_summary)
                    if not gpu_accept_match:
                        summary = cpu_summary
                        selected_row = int(cpu_summary.commit_rows[0])
                else:
                    target_top1 = ()
                    target_values = ()
            else:
                target_top1, target_values = self._read_verify_top1(rows)
                cpu_result = batch.accept_from_top1(target_top1, transaction_id=0)
                cpu_summary = TargetAcceptSummary.from_accept_result(batch, cpu_result)
                gpu_accept_match = self._gpu_accept_payload_matches(gpu_payload, cpu_summary)
                summary = cpu_summary
                selected_row = int(cpu_summary.commit_rows[0])
            if graph_mode != "off":
                self._copy_verify_capture_prefix(
                    capture_target,
                    capture_hidden_concat,
                    capture_row_start=capture_row_start,
                    rows=int(summary.accepted_counts[0]) + 1,
                    stream=stream,
                )
            self._commit_bulk_linear_states(
                selected_row,
                base_slot=base_slot,
                stream=stream,
                commit_row_ptr=int(self.verify_commit_rows.ptr),
            )
            if self._verify_accept_updates_position_enabled(batch) and gpu_accept_match:
                self._record_slot_position_host(int(summary.commit_positions[0]), slot=base_slot)
            else:
                self._set_slot_position(int(summary.commit_positions[0]), slot=base_slot, stream=stream)
            # Re-pointing the scratch maps to rows=1 decode views re-reserves
            # workspace names with a different shape, which frees the rows=B+1
            # buffers the captured verifier graph holds raw pointers to.
            # Replays then read/write freed memory (#107 graph-auto drift).
            # Keep the verifier-shaped scratch alive while any graph is cached;
            # decode steps lazily re-reserve canonical c=1 views on demand.
            if bool(canonicalize_after) and (graph_mode == "off" or not self._verify_graph_cache):
                self._canonicalize_decode_scratch()
            if bool(synchronize_after_commit):
                self.runtime.stream_synchronize(stream)
            next_token = None if summary.next_tokens is None else summary.next_tokens[0]
            return Qwen35ParoBulkVerifyResult(
                target_top1=tuple(int(token) for token in target_top1) if target_top1 else (),
                target_top1_values=tuple(float(value) for value in target_values) if target_values else (),
                accepted_count=int(summary.accepted_counts[0]),
                accepted_tokens=tuple(int(token) for token in summary.accepted_tokens[0]),
                commit_row=selected_row,
                commit_token=int(summary.commit_tokens[0]),
                commit_position=int(summary.commit_positions[0]),
                next_token=None if next_token is None else int(next_token),
                full_accept=bool(summary.full_accept[0]),
                finite_logits=all(math.isfinite(float(value)) for value in target_values) if target_values else True,
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

    def verify_tree_bulk_and_commit(
        self,
        batch: TargetVerifyBatch,
        *,
        base_slot: int,
        capture_layer_ids: Sequence[int],
        capture_hidden_concat: Tensor,
        capture_row_start: int,
        stream: int = 0,
        graph_mode: str = "off",
        canonicalize_after: bool = True,
    ) -> Qwen35ParoBulkVerifyResult:
        """DDTree variant of ``verify_chain_bulk_and_commit``.

        Runs one batched target-verifier forward over a tree-shaped
        ``TargetVerifyBatch`` (``batch.mode == 'verify_tree'``), with
        sibling/cousin attention isolation enforced by the tree-aware GQA
        gate kernel + ancestor mask.  Linear-attention layers reuse the
        existing parent-indexed tree t-loop kernel.  The accept summary is
        computed by ``dflash_accept_chain_i32`` which already walks
        ``parent_rows`` to find the longest matching path through the tree.

        Multi-cycle-safe: after accept, ``_commit_tree_full_attention_kv``
        compacts the accepted path's K/V cells from their sparse verifier-
        row slots back to the canonical dense slots
        ``[tree_committed_count, tree_committed_count + accepted_count]``
        for every full-attention layer.  Linear-attention recurrent state
        for the accepted leaf is committed by ``_commit_bulk_linear_states``
        as for chain mode.  Subsequent decode cycles can read the correct
        dense context.

        Graph capture is not supported for tree mode (yet) because graph
        capture must observe a fixed kernel/address layout per bucket and
        the tree orchestrator allocates the uniform context/position
        device buffers lazily on first cycle.
        """

        if self.closed:
            raise RuntimeError("session is closed")
        if batch.mode != "verify_tree":
            raise ValueError("verify_tree_bulk_and_commit requires batch.mode == 'verify_tree'")
        if len(batch.request_ids) != 1:
            raise ValueError("verify_tree_bulk_and_commit currently supports one request")
        rows = int(batch.rows)
        if rows <= 1:
            raise ValueError("tree verifier requires root plus at least one candidate row")
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
        if graph_mode != "off" and capture_hidden_concat.shape[1] != 0:
            raise NotImplementedError("verify_tree graph replay currently requires capture width 0 (persistent proposer)")
        capture_target = capture_hidden_concat
        capture_target_start = capture_row_start
        if graph_mode != "off":
            capture_target = self._verify_capture_staging_tensor(rows=rows, width=int(capture_hidden_concat.shape[1]))
            capture_target_start = 0

        self._write_verify_chain_metadata(batch, base_slot=base_slot, stream=stream)
        if graph_mode == "off":
            graph_info: dict[str, Any] = {
                "mode": "off",
                "status": "disabled",
                "replayed": False,
                "validation_passed": None,
                "chain_attn_mode": "tree_batched",
                "verifier_mode": "verify_tree",
            }
            self._launch_verify_chain_forward_accept(
                batch,
                base_slot=base_slot,
                capture_ids=capture_ids,
                capture_hidden_concat=capture_hidden_concat,
                capture_row_start=capture_row_start,
                rows=rows,
                stream=stream,
                # chain_attn_mode is ignored for tree mode; the dispatcher
                # checks batch.mode first and routes to the tree orchestrator.
                chain_attn_mode="batched",
            )
        else:
            # Same one-capture-per-bucket replay as the chain path; tree
            # topology (parent_rows/ancestor mask) is part of the metadata
            # bucket and the cache key includes batch.mode, so chain and tree
            # graphs never alias. Lazy buffers are allocated by the cycle-1
            # direct pass before capture (same trick as chain).
            graph_info = self._run_verify_graph_or_direct(
                batch,
                base_slot=base_slot,
                capture_ids=capture_ids,
                capture_hidden_concat=capture_target,
                capture_row_start=capture_target_start,
                rows=rows,
                graph_mode=graph_mode,
                chain_attn_mode="batched",
                linear_attn_mode="tree_tloop",
                stream=stream,
            )
            graph_info["verifier_mode"] = "verify_tree"
        gpu_payload = self._read_verify_accept_payload(len(batch.request_ids), stream=stream)
        if self._verify_gpu_accept_enabled():
            # Fast path: trust GPU accept-summary kernel, skip CPU top1 read + CPU accept.
            summary = TargetAcceptSummary.from_gpu_payload(batch, gpu_payload)
            selected_row = int(summary.commit_rows[0])
            gpu_accept_match = True
            # Optional validation against CPU oracle.
            if os.environ.get("HIPENGINE_VERIFY_GPU_ACCEPT", "").strip().lower() == "validate":
                target_top1, target_values = self._read_verify_top1(rows)
                cpu_result = batch.accept_from_top1(target_top1, transaction_id=0)
                cpu_summary = TargetAcceptSummary.from_accept_result(batch, cpu_result)
                gpu_accept_match = self._gpu_accept_payload_matches(gpu_payload, cpu_summary)
                if not gpu_accept_match:
                    summary = cpu_summary
                    selected_row = int(cpu_summary.commit_rows[0])
            else:
                target_top1 = ()
                target_values = ()
        else:
            target_top1, target_values = self._read_verify_top1(rows)
            cpu_result = batch.accept_from_top1(target_top1, transaction_id=0)
            cpu_summary = TargetAcceptSummary.from_accept_result(batch, cpu_result)
            gpu_accept_match = self._gpu_accept_payload_matches(gpu_payload, cpu_summary)
            summary = cpu_summary
            selected_row = int(cpu_summary.commit_rows[0])
        # Compact accepted-path K/V cells from their sparse verifier-row
        # slots back to canonical dense slots, so the next decode cycle
        # reads the correct context.  Skipped (no-op) when the accepted
        # path is already in canonical layout (e.g. a degenerate tree
        # that is a chain).
        self._commit_tree_full_attention_kv(
            batch,
            selected_row,
            base_slot=base_slot,
            stream=stream,
        )
        self._commit_tree_capture_hidden_concat(
            batch,
            selected_row,
            capture_hidden_concat,
            capture_row_start=capture_row_start,
            stream=stream,
        )
        # Linear-attention recurrent state commit follows the chain path:
        # the tree t-loop already produced the exact leaf state and
        # ``_commit_bulk_linear_states`` picks it via selected_row.
        self._commit_bulk_linear_states(
            selected_row,
            base_slot=base_slot,
            stream=stream,
            commit_row_ptr=int(self.verify_commit_rows.ptr),
        )
        if self._verify_accept_updates_position_enabled(batch) and gpu_accept_match:
            self._record_slot_position_host(int(summary.commit_positions[0]), slot=base_slot)
        else:
            self._set_slot_position(int(summary.commit_positions[0]), slot=base_slot, stream=stream)
        # Same #107 keepalive as chain: canonicalizing decode scratch frees the
        # rows=B+1 buffers any cached verifier graph holds raw pointers to.
        if bool(canonicalize_after) and (graph_mode == "off" or not self._verify_graph_cache):
            self._canonicalize_decode_scratch()
        self.runtime.stream_synchronize(stream)
        next_token = None if summary.next_tokens is None else summary.next_tokens[0]
        return Qwen35ParoBulkVerifyResult(
            target_top1=tuple(int(token) for token in target_top1) if target_top1 else (),
            target_top1_values=tuple(float(value) for value in target_values) if target_values else (),
            accepted_count=int(summary.accepted_counts[0]),
            accepted_tokens=tuple(int(token) for token in summary.accepted_tokens[0]),
            commit_row=selected_row,
            commit_token=int(summary.commit_tokens[0]),
            commit_position=int(summary.commit_positions[0]),
            next_token=None if next_token is None else int(next_token),
            full_accept=bool(summary.full_accept[0]),
            finite_logits=all(math.isfinite(float(value)) for value in target_values) if target_values else True,
            gpu_accept_match_cpu=bool(gpu_accept_match),
            rows=rows,
            graph=graph_info,
        )

    def _verify_capture_staging_tensor(self, *, rows: int, width: int) -> Tensor:
        if rows <= 0 or rows > self.max_batch_size:
            raise ValueError("rows outside verifier staging capacity")
        max_width = len(self.config.layer_types) * self.config.hidden_size
        if width < 0 or width > max_width:
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
        chain_attn_mode: str = "c1_loop",
        linear_attn_mode: str = "tree_tloop",
        stream: int = 0,
    ) -> dict[str, Any]:
        key = (
            int(rows),
            int(capture_hidden_concat.shape[1]),
            int(base_slot),
            str(chain_attn_mode),
            str(linear_attn_mode),
            str(batch.mode),
        )
        entry = self._verify_graph_cache.get(key)
        if (
            entry is not None
            and os.environ.get("HIPENGINE_VERIFY_GRAPH_RECAPTURE", "").strip() == "1"
        ):
            # Debug-only (#107): drop the cached graph each cycle so replay
            # always executes a graph captured from the current state.
            try:
                self.runtime.graph_exec_destroy(entry.graph_exec)
                self.runtime.graph_destroy(entry.graph)
                self.runtime.stream_destroy(entry.stream)
            except Exception:
                pass
            self._verify_graph_cache.pop(key, None)
            entry = None
        if graph_mode == "auto" and entry is not None:
            # Debug-only (#107): re-run the direct pass before each replay and
            # compare per-row top1 + accept payload to localize replay drift.
            revalidate = os.environ.get("HIPENGINE_VERIFY_GRAPH_REVALIDATE", "").strip() == "1"
            direct_top1: tuple = ()
            direct_payload = None
            if revalidate:
                print(f"[graph-revalidate] replay#{entry.replay_count + 1} direct-begin", file=sys.stderr, flush=True)
                self._launch_verify_chain_forward_accept(
                    batch,
                    base_slot=base_slot,
                    capture_ids=capture_ids,
                    capture_hidden_concat=capture_hidden_concat,
                    capture_row_start=capture_row_start,
                    rows=rows,
                    stream=stream,
                    chain_attn_mode=chain_attn_mode,
                    linear_attn_mode=linear_attn_mode,
                )
                self.runtime.stream_synchronize(stream)
                direct_top1, _ = self._read_verify_top1(rows)
                direct_payload = self._read_verify_accept_payload(len(batch.request_ids), stream=stream)
                print(f"[graph-revalidate] replay#{entry.replay_count + 1} direct-done, replay-begin", file=sys.stderr, flush=True)
            # Restore the capture-time verifier scratch maps: the replayed
            # graph writes the capture-time rows=B+1 tree_*_state buffers, but
            # `_canonicalize_decode_scratch` re-pointed the live maps to rows=1
            # decode scratch after the previous cycle. Without this, the
            # post-replay `_commit_bulk_linear_states` reads the wrong buffers
            # (stale decode rows) and poisons the GDN slot state — the #107
            # graph-auto divergence.
            if entry.linear_scratch is not None:
                self.linear_scratch.update(entry.linear_scratch)
            if entry.moe_scratch is not None:
                self.moe_scratch.update(entry.moe_scratch)
            # Launch on the caller's stream so the subsequent accept-payload
            # read serializes naturally; avoids the extra cross-stream sync we
            # would pay if we launched on the (separate) capture stream.
            self.runtime.graph_launch(entry.graph_exec, stream)
            entry.replay_count += 1
            if revalidate:
                self.runtime.stream_synchronize(stream)
                replay_top1, _ = self._read_verify_top1(rows)
                replay_payload = self._read_verify_accept_payload(len(batch.request_ids), stream=stream)
                top1_match = tuple(replay_top1) == tuple(direct_top1)
                payload_match = replay_payload == direct_payload
                print(
                    f"[graph-revalidate] replay#{entry.replay_count} rows={rows} "
                    f"top1_match={top1_match} payload_match={payload_match} "
                    f"direct_top1={tuple(direct_top1)} replay_top1={tuple(replay_top1)}",
                    file=sys.stderr,
                    flush=True,
                )
            return {
                "mode": graph_mode,
                "status": "replayed",
                "replayed": True,
                "validation_passed": entry.validation_passed,
                "bucket_key": {
                    "rows": rows,
                    "capture_width": int(capture_hidden_concat.shape[1]),
                    "base_slot": base_slot,
                    "chain_attn_mode": str(chain_attn_mode),
                    "linear_attn_mode": str(linear_attn_mode),
                },
                "replay_count": entry.replay_count,
            }

        # Cycle 1 (cache miss): run the direct pass first so all lazily-
        # allocated workspace tensors (linear/full-attention/MoE scratch)
        # exist before we open the capture stream.  HIP does not allow
        # hipMalloc while a stream is in capture mode, so allocating during
        # capture would either skip the recording or fault at replay.  The
        # direct pass also doubles as the cycle-1 forward; capture then
        # records a second (replayable) instance of the same kernel sequence.
        #
        # Keyed staged-rotate barriers must run in capture-safe memset mode for
        # every pass that may be captured: the cumulative host epoch is baked
        # by-value into the graph, so replays of a keyed capture skip the
        # producer→consumer sync (#107 drift) and later keyed direct launches
        # spin on epochs the device never reaches (#107 hang).
        _set_shared_rotate_fuse_barrier_memset_mode(True)
        self._launch_verify_chain_forward_accept(
            batch,
            base_slot=base_slot,
            capture_ids=capture_ids,
            capture_hidden_concat=capture_hidden_concat,
            capture_row_start=capture_row_start,
            rows=rows,
            stream=stream,
            chain_attn_mode=chain_attn_mode,
            linear_attn_mode=linear_attn_mode,
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
                    chain_attn_mode=chain_attn_mode,
                    linear_attn_mode=linear_attn_mode,
                )
                graph = self.runtime.stream_end_capture(graph_stream)
            except Exception:
                try:
                    self.runtime.stream_end_capture(graph_stream)
                except Exception:
                    pass
                raise
            graph_exec = self.runtime.graph_instantiate(graph)
            if graph_mode == "validate":
                # Validation pass: run the captured graph and compare with the
                # direct pass we already executed above.  Launch on a separate
                # stream so the direct outputs (already written into the
                # caller-stream buffers) are not clobbered before we read them.
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
                        chain_attn_mode=chain_attn_mode,
                        linear_attn_mode=linear_attn_mode,
                    )
                    return {
                        "mode": graph_mode,
                        "status": "validation_failed_fallback",
                        "replayed": False,
                        "validation_passed": False,
                        "bucket_key": {
                            "rows": rows,
                            "capture_width": int(capture_hidden_concat.shape[1]),
                            "base_slot": base_slot,
                            "chain_attn_mode": str(chain_attn_mode),
                            "linear_attn_mode": str(linear_attn_mode),
                        },
                    }
            if os.environ.get("HIPENGINE_VERIFY_GRAPH_RECAPTURE", "").strip() == "1":
                # Debug-only (#107): execute the freshly captured graph so the
                # accept payload comes from replay, not the direct pass.
                self.runtime.graph_launch(graph_exec, stream)
            entry = Qwen35ParoVerifierGraphEntry(
                rows=rows,
                capture_width=int(capture_hidden_concat.shape[1]),
                base_slot=base_slot,
                graph=graph,
                graph_exec=graph_exec,
                stream=graph_stream,
                validation_passed=True,
                replay_count=1,
                linear_scratch=dict(self.linear_scratch),
                moe_scratch=dict(self.moe_scratch),
            )
            self._verify_graph_cache[key] = entry
            return {
                "mode": graph_mode,
                "status": "captured_validated" if graph_mode == "validate" else "captured_validated_miss",
                "replayed": graph_mode == "auto",
                "validation_passed": True,
                "bucket_key": {
                    "rows": rows,
                    "capture_width": int(capture_hidden_concat.shape[1]),
                    "base_slot": base_slot,
                    "chain_attn_mode": str(chain_attn_mode),
                    "linear_attn_mode": str(linear_attn_mode),
                },
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
                chain_attn_mode=chain_attn_mode,
                linear_attn_mode=linear_attn_mode,
            )
            return {
                "mode": graph_mode,
                "status": "capture_failed_fallback",
                "replayed": False,
                "validation_passed": None,
                "fallback_reason": str(exc),
                "bucket_key": {
                    "rows": rows,
                    "capture_width": int(capture_hidden_concat.shape[1]),
                    "base_slot": base_slot,
                    "chain_attn_mode": str(chain_attn_mode),
                    "linear_attn_mode": str(linear_attn_mode),
                },
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
        chain_attn_mode: str = "c1_loop",
        linear_attn_mode: str = "tree_tloop",
    ) -> None:
        if linear_attn_mode not in {"tree_tloop", "chain_tloop"}:
            raise ValueError("linear_attn_mode must be tree_tloop or chain_tloop")
        embedding_lookup_batch_fp16_i64(
            self.embedding.tensor.ptr,
            self.verify_token_ids_i64.ptr,
            self.verify_trunk_hidden.ptr,
            rows,
            self.config.hidden_size,
            self.vocab_size,
            stream=stream,
            library=self.libraries["runtime_state"],
            runtime=self.runtime,
        )
        hidden = Tensor.from_handle(self.verify_trunk_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        next_hidden = Tensor.from_handle(self.verify_trunk_next_hidden.ptr, (rows, self.config.hidden_size), DType.FP16, self.device)
        parent_rows = Tensor.from_handle(self.verify_parent_rows_i64.ptr, (rows,), DType.INT64, self.device)
        capture_offsets = {layer_id: idx for idx, layer_id in enumerate(capture_ids)}
        for layer_id, state in enumerate(self.states):
            layer_type = self.config.layer_types[layer_id]
            if layer_type == "linear_attention":
                conv_state, recurrent_state = self._slot_linear_state(layer_id, base_slot)
                linear_scratch = self._verify_linear_attention_scratch(layer_id, state, rows=rows)
                self.linear_scratch[layer_id] = linear_scratch
                moe_scratch = self._verify_mlp_scratch(layer_id, state, rows=rows)
                self.moe_scratch[layer_id] = moe_scratch
                # M13.B.0: pass ``out=next_hidden`` so the linear-attention
                # layer's final MoE combine writes straight into the trunk
                # ``next_hidden`` buffer.  The orchestrator's trailing
                # ``if out.ptr != next_hidden.ptr`` memcpy then becomes a
                # no-op (40 D2D launches/pass eliminated for chain mode).
                if linear_attn_mode == "chain_tloop":
                    out = state.run_linear_attention_moe_chain_tloop_layer_fp16(
                        hidden,
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                        chain_conv_state=linear_scratch.tree_conv_state,
                        chain_recurrent_state=linear_scratch.tree_recurrent_state,
                        linear_scratch=linear_scratch,
                        moe_scratch=moe_scratch,
                        out=next_hidden,
                        tokens=rows,
                        library=self.libraries,
                        stream=stream,
                    )
                else:
                    out = state.run_linear_attention_moe_tree_tloop_layer_fp16(
                        hidden,
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                        parent_rows=parent_rows,
                        linear_scratch=linear_scratch,
                        moe_scratch=moe_scratch,
                        out=next_hidden,
                        tokens=rows,
                        library=self.libraries,
                        stream=stream,
                    )
            elif layer_type == "full_attention":
                if batch.mode == "verify_tree":
                    # Tree topology: sibling rows must not see each other; use
                    # the tree-aware orchestrator regardless of
                    # chain_attn_mode (which only applies to chain mode).
                    self._run_full_attention_tree_batched(
                        state,
                        layer_id=layer_id,
                        hidden=hidden,
                        next_hidden=next_hidden,
                        rows=rows,
                        base_slot=base_slot,
                        stream=stream,
                    )
                elif chain_attn_mode == "batched":
                    self._run_full_attention_chain_batched(
                        state,
                        layer_id=layer_id,
                        hidden=hidden,
                        next_hidden=next_hidden,
                        rows=rows,
                        base_slot=base_slot,
                        stream=stream,
                    )
                elif chain_attn_mode == "decode_batched":
                    self._run_full_attention_chain_decode_batched(
                        state,
                        layer_id=layer_id,
                        hidden=hidden,
                        next_hidden=next_hidden,
                        rows=rows,
                        positions=batch.positions,
                        base_slot=base_slot,
                        stream=stream,
                    )
                else:
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
        self._launch_verify_accept_summary(batch, rows=rows, base_slot=base_slot, stream=stream)

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
            # M13.B.0: ``out=row_out`` lands each row's MoE combine result
            # directly in ``next_hidden``'s row slice; the conditional D2D
            # below becomes a no-op (one D2D/row/layer eliminated for c1_loop
            # mode).
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
                out=row_out,
                chunk_size=self.decode_chunk_size,
                num_splits=num_splits,
                library=self.libraries,
                stream=stream,
            )
            if out.ptr != row_out.ptr:
                self.runtime.memcpy_async(row_out.ptr, out.ptr, self.hidden_nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, stream)

    def _run_full_attention_chain_batched(
        self,
        state: Qwen35ParoDecodeState,
        *,
        layer_id: int,
        hidden: Tensor,
        next_hidden: Tensor,
        rows: int,
        base_slot: int,
        stream: int = 0,
    ) -> None:
        """Run a full-attention layer over verifier rows in one batched pass.

        Reuses the same native primitives the prefill orchestrator uses but
        with verifier-specific spans and forces c=1 MoE (``run_moe_c1_fp16``
        with ``tokens=rows``) instead of the grouped/WMMA prefill MoE.  Grouped
        MoE has higher fixed setup cost than c=1 MoE at small B+1, which is the
        regime DFlash chain verification spends most cycles in.

        Pipeline (one launch per step instead of per-row):
          1. input RMSNorm over ``rows``
          2. paro rotate + QKV projection over ``rows``
          3. multi-token head RMSNorm + partial RoPE (per-row positions)
          4. prompt-style batched K/V append (one launch over all rows)
          5. gated GQA prefill attention with per-row causal limit
          6. paro rotate + O projection over ``rows``
          7. batched post-attention add + RMSNorm
          8. ``run_moe_c1_fp16(tokens=rows)`` (one block per row inside MoE)
        """

        _ = base_slot  # base_slot already encoded in prefill_block_table_buf
        positions_tensor = Tensor.from_handle(
            self.prefill_position_buf.ptr,
            (rows,),
            DType.INT64,
            self.device,
        )
        append_spans, prefill_spans = self._verify_chain_full_attention_spans_batched(rows)
        # ``prefill_block_table_buf`` was populated in ``_write_verify_chain_metadata``
        # with absolute physical-block indices for ``base_slot``, so the global K/V
        # cache view matches the block-table indexing used by both the prompt-style
        # KV write kernel and the prefill GQA gate attention kernel.
        key_cache, value_cache = self._full_cache_all_slots(layer_id)
        attention_scratch = self._ensure_full_prefill_scratch(tokens=rows)
        moe_scratch = self._ensure_moe_c1_prefill_scratch(layer_id=layer_id, tokens=rows)

        state.input_rmsnorm_fp16(
            hidden,
            attention_scratch.attn_input,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        state.rotate_full_attention_inputs_fp16(
            attention_scratch.attn_input,
            attention_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        state.project_full_attention_qkv_fp16(
            attention_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        _query, _key, _value, gate = state.prepare_full_attention_qkv_fp16(
            attention_scratch,
            cos_table=self.cos,
            sin_table=self.sin,
            position=positions_tensor,
            max_positions=self.max_sequence_length,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        state.append_full_attention_kv_fp16_batch(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=append_spans,
            rows=rows,
            block_size=self.block_size,
            library=self.libraries,
            stream=stream,
        )
        gated = state.prefill_full_attention_gqa_gate_fp16(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=prefill_spans,
            rows=rows,
            gate=gate,
            block_size=self.block_size,
            library=self.libraries,
            stream=stream,
        )
        attn_out = state.project_full_attention_o_fp16(
            gated,
            attention_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        mlp_input, residual = state.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        # M13.B.0: ``out=next_hidden`` lets the MoE combine write straight
        # into the trunk ``next_hidden`` buffer, so the conditional D2D
        # below becomes a no-op (10 launches/pass eliminated for the
        # chain-batched full-attention layers).
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        if dense_mlp:
            out = state.run_dense_mlp_residual_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                out=next_hidden,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
        else:
            out = state.run_moe_c1_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                out=next_hidden,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
        if out.ptr != next_hidden.ptr:
            self.runtime.memcpy_async(
                next_hidden.ptr,
                out.ptr,
                rows * self.hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )

    def _run_full_attention_chain_decode_batched(
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
        """Run a full-attention verifier layer with small-B decode attention.

        This keeps the safe row-batched QKV/KV/O/MoE staging from
        ``_run_full_attention_chain_batched`` but replaces the prefill-style GQA
        attention kernel with a row-batched decode split-K primitive.  The K/V
        cache block table is still populated by ``_write_verify_chain_metadata``
        with absolute physical blocks for ``base_slot``.
        """

        if len(positions) != rows:
            raise ValueError("positions must match verifier rows")
        positions_tensor = Tensor.from_handle(
            self.prefill_position_buf.ptr,
            (rows,),
            DType.INT64,
            self.device,
        )
        append_spans, decode_spans = self._verify_chain_full_attention_spans_batched(rows)
        key_cache, value_cache = self._full_cache_all_slots(layer_id)
        attention_scratch = self._ensure_full_prefill_scratch(tokens=rows)
        moe_scratch = self._ensure_moe_c1_prefill_scratch(layer_id=layer_id, tokens=rows)
        max_context = max(int(position) for position in positions) + 1
        num_splits = max(1, (max_context + self.decode_chunk_size - 1) // self.decode_chunk_size)
        force_row_input = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_INPUT", False)
        force_row_qkv = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_QKV", False)
        force_row_qkv_temp = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_QKV_TEMP", False)
        force_exact_suffix = rows > 1 and _env_flag(
            "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX", False
        )
        force_row_layer = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_LAYER", False)
        force_row_layer_batch = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_LAYER_BATCH", False)
        force_row_context = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_CONTEXT", False)
        force_row_context_only = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_CONTEXT_ONLY", False)
        force_row_dense_context_only = rows > 1 and _env_flag(
            "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_DENSE_CONTEXT_ONLY", False
        )
        force_row_paged_context_only = rows > 1 and _env_flag(
            "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_PAGED_CONTEXT_ONLY", False
        )
        force_row_gate = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_GATE", False)
        force_row_kv_append = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_KV_APPEND", False)
        force_row_append_context = rows > 1 and _env_flag(
            "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_APPEND_CONTEXT", False
        )
        force_row_suffix = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_SUFFIX", False)
        force_row_output = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_OUTPUT", False)
        force_batch_gemv_output = rows > 1 and _env_flag(
            "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_BATCH_GEMV_OUTPUT", False
        )
        force_row_post = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_POST", False)
        force_row_moe = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_MOE", False)
        force_decode_helper = rows > 1 and _env_flag("HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_HELPER", False)
        if force_exact_suffix:
            force_row_kv_append = True
            force_row_context = True
            force_row_append_context = True
            force_batch_gemv_output = True
        force_decode_helper = force_decode_helper or any(
            (
                force_row_input,
                force_row_qkv_temp,
                force_exact_suffix,
                force_row_layer,
                force_row_layer_batch,
                force_row_context,
                force_row_context_only,
                force_row_dense_context_only,
                force_row_paged_context_only,
                force_row_gate,
                force_row_kv_append,
                force_row_append_context,
                force_row_suffix,
                force_row_output,
                force_batch_gemv_output,
                force_row_post,
                force_row_moe,
            )
        )

        if force_decode_helper:
            per_row_contexts = None
            per_row_append_contexts = None
            needs_row_contexts = any(
                (
                    force_row_layer,
                    force_row_layer_batch,
                    force_row_context,
                    force_row_context_only,
                    force_row_dense_context_only,
                    force_row_paged_context_only,
                    force_row_append_context,
                    force_row_suffix,
                )
            )
            needs_row_append_contexts = any(
                (
                    force_row_layer,
                    force_row_layer_batch,
                    force_row_kv_append,
                    force_row_append_context,
                    force_row_suffix,
                )
            )
            if needs_row_contexts or needs_row_append_contexts:
                row_key_cache, row_value_cache = self._slot_full_cache(layer_id, base_slot)
                per_row_contexts = [] if needs_row_contexts else None
                per_row_append_contexts = [] if needs_row_append_contexts else None
                for row in range(rows):
                    _row_position, row_append_spans, row_decode_spans = self._verify_chain_row_spans(row)
                    if per_row_contexts is not None:
                        per_row_contexts.append((row_key_cache, row_value_cache, row_decode_spans))
                    if per_row_append_contexts is not None:
                        per_row_append_contexts.append((row_key_cache, row_value_cache, row_append_spans))
            out = state.run_full_attention_moe_decode_batch_layer_fp16(
                hidden,
                key_cache=key_cache,
                value_cache=value_cache,
                append_spans=append_spans,
                decode_spans=decode_spans,
                cos_table=self.cos,
                sin_table=self.sin,
                positions=positions_tensor,
                max_positions=self.max_sequence_length,
                attention_scratch=attention_scratch,
                moe_scratch=moe_scratch,
                tokens=rows,
                block_size=self.block_size,
                force_selected_c1_moe=True,
                force_per_row_input_rmsnorm=force_row_input,
                force_per_row_qkv_scratch=force_row_qkv_temp,
                force_per_row_layer_scratch=force_row_layer,
                force_per_row_layer_batch_scratch=force_row_layer_batch,
                force_per_row_context=force_row_context,
                force_per_row_context_only=force_row_context_only,
                force_per_row_dense_context_only=force_row_dense_context_only,
                force_per_row_paged_context_only=force_row_paged_context_only,
                force_per_row_gate=force_row_gate,
                per_row_contexts=per_row_contexts,
                force_per_row_kv_append=force_row_kv_append,
                per_row_append_contexts=per_row_append_contexts,
                force_per_row_append_context=force_row_append_context,
                force_per_row_suffix=force_row_suffix,
                force_per_row_output=force_row_output,
                force_batch_gemv_output=force_batch_gemv_output,
                force_per_row_post_attention=force_row_post,
                force_per_row_moe=force_row_moe,
                library=self.libraries,
                stream=stream,
            )
            if out.ptr != next_hidden.ptr:
                self.runtime.memcpy_async(
                    next_hidden.ptr,
                    out.ptr,
                    rows * self.hidden_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            return

        partial_out, partial_m, partial_l = self._ensure_full_decode_batch_partials(rows=rows, num_splits=num_splits)

        state.input_rmsnorm_fp16(
            hidden,
            attention_scratch.attn_input,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        if force_row_qkv:
            _query, _key, _value, gate = state.prepare_full_attention_qkv_fp16_decode_rows(
                attention_scratch,
                cos_table=self.cos,
                sin_table=self.sin,
                positions=positions_tensor,
                max_positions=self.max_sequence_length,
                tokens=rows,
                force_per_row_scratch=force_row_qkv_temp,
                library=self.libraries,
                stream=stream,
            )
        else:
            state.rotate_full_attention_inputs_fp16(
                attention_scratch.attn_input,
                attention_scratch,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
            state.project_full_attention_qkv_fp16(
                attention_scratch,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
            _query, _key, _value, gate = state.prepare_full_attention_qkv_fp16(
                attention_scratch,
                cos_table=self.cos,
                sin_table=self.sin,
                position=positions_tensor,
                max_positions=self.max_sequence_length,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
        state.append_full_attention_kv_fp16_batch(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=append_spans,
            rows=rows,
            block_size=self.block_size,
            library=self.libraries,
            stream=stream,
        )
        gated = state.decode_full_attention_gqa_gate_fp16_batch(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=decode_spans,
            rows=rows,
            partial_out=partial_out,
            partial_m=partial_m,
            partial_l=partial_l,
            chunk_size=self.decode_chunk_size,
            num_splits=num_splits,
            gate=gate,
            block_size=self.block_size,
            library=self.libraries,
            stream=stream,
        )
        attn_out = state.project_full_attention_o_fp16(
            gated,
            attention_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        mlp_input, residual = state.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        if dense_mlp:
            out = state.run_dense_mlp_residual_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                out=next_hidden,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
        else:
            out = state.run_moe_c1_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                out=next_hidden,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
        if out.ptr != next_hidden.ptr:
            self.runtime.memcpy_async(
                next_hidden.ptr,
                out.ptr,
                rows * self.hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )

    def _verify_chain_full_attention_spans_batched(self, rows: int) -> tuple[KVLiveSpans, KVLiveSpans]:
        """Build [rows]-sized append and decode spans for the chain batched verifier.

        ``append_spans.live_counts`` are the absolute write positions (consumed
        by ``qwen35_write_paged_kv_mixed_value_fp16_prompt_spans``).
        ``prefill_spans.live_counts`` are the post-append context counts (one
        greater than the position) used by the prefill GQA gate kernel to bound
        the per-row visible context, with ``row_positions`` providing the causal
        limit per row.
        """

        block_table = self._prefill_block_table_rows(rows, start=0)
        positions = Tensor.from_handle(
            self.prefill_position_buf.ptr,
            (rows,),
            DType.INT64,
            self.device,
        )
        context_counts = Tensor.from_handle(
            self.prefill_context_count_buf.ptr,
            (rows,),
            DType.INT64,
            self.device,
        )
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=positions,
            max_live_count=self.max_sequence_length - 1,
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="verify_chain",
        )
        prefill_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=context_counts,
            max_live_count=self.max_sequence_length,
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="verify_chain",
        )
        return append_spans, prefill_spans

    def _run_full_attention_tree_batched(
        self,
        state: Qwen35ParoDecodeState,
        *,
        layer_id: int,
        hidden: Tensor,
        next_hidden: Tensor,
        rows: int,
        base_slot: int,
        stream: int = 0,
    ) -> None:
        """Tree-topology variant of ``_run_full_attention_chain_batched``.

        Pipeline differences vs the chain orchestrator:

          1. Cache slots are taken from ``verify_cache_slot_buf`` (one
             UNIQUE slot per verifier row) so sibling rows do not collide
             on K/V writes.  ``prefill_position_buf`` still carries the
             depth-based RoPE phases ``batch.positions``.
          2. The K/V append writes each row at its cache slot.
          3. The full-attention attention step uses
             ``prefill_full_attention_gqa_gate_tree_fp16`` with the dense
             ``[rows, rows]`` ancestor mask in ``verify_ancestor_mask_u8``
             so siblings/cousins do not see each other.  Committed-context
             positions in ``[0, tree_committed_count)`` are visible to
             every row.
          4. MoE/MLP/post-norm reuse the c=1 multi-token primitives,
             same as chain batched.

        Pre-conditions: ``_write_verify_chain_metadata`` has already run
        with ``batch.mode == 'verify_tree'``, which populated
        ``verify_cache_slot_buf``, ``verify_ancestor_mask_u8``, and
        ``self.verify_tree_committed_count``.
        """

        _ = base_slot  # base_slot already encoded in prefill_block_table_buf
        if self.verify_tree_committed_count < 0:
            raise RuntimeError(
                "verify_tree_committed_count must be set by"
                " _write_verify_chain_metadata before invoking the tree"
                " orchestrator"
            )

        rope_positions = Tensor.from_handle(
            self.prefill_position_buf.ptr,
            (rows,),
            DType.INT64,
            self.device,
        )
        cache_slots = Tensor.from_handle(
            self.verify_cache_slot_buf.ptr,
            (rows,),
            DType.INT64,
            self.device,
        )
        ancestor_mask = Tensor.from_handle(
            self.verify_ancestor_mask_u8.ptr,
            (rows, rows),
            DType.BOOL,
            self.device,
        )
        append_spans, prefill_spans = self._verify_tree_full_attention_spans_batched(
            rows,
            cache_slots=cache_slots,
        )
        key_cache, value_cache = self._full_cache_all_slots(layer_id)
        attention_scratch = self._ensure_full_prefill_scratch(tokens=rows)
        moe_scratch = self._ensure_moe_c1_prefill_scratch(layer_id=layer_id, tokens=rows)

        state.input_rmsnorm_fp16(
            hidden,
            attention_scratch.attn_input,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        state.rotate_full_attention_inputs_fp16(
            attention_scratch.attn_input,
            attention_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        state.project_full_attention_qkv_fp16(
            attention_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        # RoPE phases come from the depth-based ``prefill_position_buf``
        # (== batch.positions); siblings share their RoPE phase.
        _query, _key, _value, gate = state.prepare_full_attention_qkv_fp16(
            attention_scratch,
            cos_table=self.cos,
            sin_table=self.sin,
            position=rope_positions,
            max_positions=self.max_sequence_length,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        # K/V append writes each row at its UNIQUE cache slot so sibling
        # writes do not collide.
        state.append_full_attention_kv_fp16_batch(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=append_spans,
            rows=rows,
            block_size=self.block_size,
            library=self.libraries,
            stream=stream,
        )
        # Tree-aware attention: ancestor mask filters sibling/cousin rows
        # inside the verifier-row K/V block.  Committed-context positions
        # below ``tree_committed_count`` are visible to every row.
        gated = state.prefill_full_attention_gqa_gate_tree_fp16(
            attention_scratch,
            key_cache=key_cache,
            value_cache=value_cache,
            spans=prefill_spans,
            rows=rows,
            ancestor_mask=ancestor_mask,
            tree_committed_count_ptr=int(self.verify_tree_committed_buf.ptr),
            gate=gate,
            block_size=self.block_size,
            library=self.libraries,
            stream=stream,
        )
        attn_out = state.project_full_attention_o_fp16(
            gated,
            attention_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        mlp_input, residual = state.post_attention_add_rmsnorm_fp16(
            hidden,
            attn_out,
            moe_scratch,
            tokens=rows,
            library=self.libraries,
            stream=stream,
        )
        # M13.B.0: ``out=next_hidden`` same rationale as the chain-batched
        # path; the tree orchestrator writes its MoE combine directly into
        # the trunk buffer.
        dense_mlp = int(getattr(self.config, "num_experts", 1) or 0) <= 0
        if dense_mlp:
            out = state.run_dense_mlp_residual_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                out=next_hidden,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
        else:
            out = state.run_moe_c1_fp16(
                mlp_input,
                residual,
                scratch=moe_scratch,
                out=next_hidden,
                tokens=rows,
                library=self.libraries,
                stream=stream,
            )
        if out.ptr != next_hidden.ptr:
            self.runtime.memcpy_async(
                next_hidden.ptr,
                out.ptr,
                rows * self.hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )

    def _verify_tree_full_attention_spans_batched(
        self,
        rows: int,
        *,
        cache_slots: Tensor,
    ) -> tuple[KVLiveSpans, KVLiveSpans]:
        """Build [rows]-sized append and decode spans for the tree verifier.

        ``append_spans.live_counts`` are the UNIQUE per-row write positions
        (``cache_slots``) consumed by
        ``qwen35_write_paged_kv_mixed_value_fp16_prompt_spans``.
        ``prefill_spans.live_counts`` is set to ``tree_committed_count + rows``
        for every row so the tree-aware GQA gate kernel walks every committed
        position plus the entire verifier-row block; the ancestor mask then
        filters out sibling/cousin rows inside that block.  ``row_positions``
        provides the kernel's causal upper bound (also uniform = total length
        minus one) so its existing causal-limit branch is a no-op and the mask
        does all the work.
        """

        if cache_slots.dtype is not DType.INT64 or cache_slots.shape != (rows,):
            raise ValueError("cache_slots must be INT64 with shape (rows,)")
        block_table = self._prefill_block_table_rows(rows, start=0)
        tree_committed = int(self.verify_tree_committed_count)
        total_len = tree_committed + rows
        # Uniform live_counts/row_positions for the prefill spans: total
        # length (committed + verifier-rows).  Each row sees [0, total_len)
        # logically; the ancestor mask filters siblings inside the
        # verifier-row block.
        uniform_counts_host = np.full((rows,), total_len, dtype=np.int64)
        uniform_positions_host = np.full((rows,), total_len - 1, dtype=np.int64)
        # Cache these uniform arrays in pinned device buffers; they are
        # small and reused across layers within a cycle.  Allocate on
        # demand against ``verify_tree_uniform_counts_buf``.
        counts_buf = self._ensure_verify_tree_uniform_buf(
            name="verify_tree_uniform_counts",
            host=uniform_counts_host,
            populate=False,
        )
        positions_buf = self._ensure_verify_tree_uniform_buf(
            name="verify_tree_uniform_positions",
            host=uniform_positions_host,
            populate=False,
        )
        uniform_counts = Tensor.from_handle(
            counts_buf.ptr, (rows,), DType.INT64, self.device,
        )
        uniform_positions = Tensor.from_handle(
            positions_buf.ptr, (rows,), DType.INT64, self.device,
        )
        # Use the session bound, not this cycle's total_len: max_live_count is
        # baked by value into the kernel launch (LDS sizing/loop cap), so a
        # captured graph would otherwise freeze the cycle-1 context length.
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=cache_slots,
            max_live_count=self.max_sequence_length - 1,
            storage_dtype=DType.BF16,
            row_positions=cache_slots,
            span_role="verify_tree",
        )
        prefill_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=uniform_counts,
            max_live_count=self.max_sequence_length,
            storage_dtype=DType.BF16,
            row_positions=uniform_positions,
            span_role="verify_tree",
        )
        return append_spans, prefill_spans

    def _ensure_verify_tree_uniform_buf(self, *, name: str, host: np.ndarray, populate: bool = True):
        """Reserve a small device buffer for the tree verifier uniform vectors.

        Tree verify needs two ``[rows]`` int64 vectors (uniform context counts
        and uniform row positions) per cycle.  Allocate them lazily and reuse
        the buffer across layers within a cycle.  The buffer is keyed by
        ``name`` so multiple tree-only auxiliary arrays can coexist.

        ``populate=False`` returns the existing buffer without an H2D copy —
        required inside the verifier forward, where a synchronous hipMemcpy
        would invalidate HIP graph capture (error 906). Per-cycle population
        happens in ``_write_verify_chain_metadata`` (outside the graph).
        """

        attr = f"_{name}_buf"
        attr_size = f"_{name}_buf_nbytes"
        nbytes = int(host.nbytes)
        existing = getattr(self, attr, None)
        existing_size = getattr(self, attr_size, 0)
        if existing is not None and existing_size >= nbytes:
            if populate:
                copy_host_to_device(
                    existing, host_array_ptr(np.ascontiguousarray(host)), nbytes, runtime=self.runtime,
                )
            return existing
        buf = malloc(nbytes, runtime=self.runtime)
        copy_host_to_device(
            buf, host_array_ptr(np.ascontiguousarray(host)), nbytes, runtime=self.runtime,
        )
        setattr(self, attr, buf)
        setattr(self, attr_size, nbytes)
        self.buffers.append(buf)
        return buf

    def _ensure_moe_c1_prefill_scratch(
        self,
        layer_id: int | None = None,
        *,
        tokens: int,
    ):
        """Reserve c=1 MoE scratch sized for batched verifier rows.

        Unlike ``_ensure_moe_prefill_scratch`` which switches to grouped MoE
        prefill scratch for ``tokens >= _moe_prefill_compact_wmma_min_tokens()``,
        this helper keeps the c=1 layout for any ``tokens`` so the batched
        chain verifier can amortize launches without paying the grouped MoE
        prefill setup cost at small B+1.
        """

        _ = layer_id
        if int(getattr(self.config, "num_experts", 1) or 0) <= 0:
            scratch = getattr(self, "prefill_moe_scratch", None)
            if isinstance(scratch, Qwen35ParoDenseMlpScratch) and scratch.normed.shape[0] >= tokens:
                return scratch
            scratch = self._prefill_scratch_owner().reserve_dense_mlp_scratch(
                tokens=tokens,
                activation_dtype=DType.FP16,
            )
            self.prefill_moe_scratch = scratch
            return scratch
        scratch = getattr(self, "prefill_moe_scratch", None)
        if isinstance(scratch, Qwen35ParoMoeScratch) and scratch.normed.shape[0] >= tokens:
            return scratch
        scratch = self._prefill_scratch_owner().reserve_moe_c1_scratch(
            tokens=tokens,
            activation_dtype=DType.FP16,
        )
        self.prefill_moe_scratch = scratch
        return scratch

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
        # M12.5: cycle-to-cycle invariants (parent_rows, draft_depths,
        # row_to_request, active_mask, block_table) only change when the
        # (rows, base_slot, mode) bucket changes.  Cache by bucket signature
        # so steady-state cycles in a fixed B/B+1 bucket only refresh the
        # per-cycle dynamic buffers (tokens i64/i32, positions i64/i32,
        # context_i64).  The chain path can pack those five logical updates
        # into one H2D copy and unpack them on device.
        bucket_signature = (
            int(rows),
            int(base_slot),
            str(batch.mode),
            tuple(int(x) for x in batch.parent_rows),
            tuple(int(x) for x in batch.draft_depths),
            tuple(int(x) for x in batch.row_to_request),
            tuple(int(x) for x in batch.active_mask),
        )
        if getattr(self, "_verify_metadata_bucket_cache", None) is None:
            self._verify_metadata_bucket_cache = None  # type: ignore[attr-defined]
        packed_dynamic_metadata: np.ndarray | None = None
        if self._verify_packed_dynamic_metadata_enabled(batch):
            packed_dynamic_metadata = np.empty((rows, _VERIFY_DYNAMIC_METADATA_FIELDS), dtype=np.int64)
            packed_dynamic_metadata[:, 0] = token_i64
            packed_dynamic_metadata[:, 1] = token_i64
            packed_dynamic_metadata[:, 2] = position_i64
            packed_dynamic_metadata[:, 3] = position_i64
            packed_dynamic_metadata[:, 4] = context_i64
            dynamic_copies: list[tuple[Any, Any]] = []
        else:
            dynamic_copies = [
                (self.verify_token_ids_i64, token_i64),
                (self.verify_token_ids_i32, token_i32),
                (self.prefill_position_buf, position_i64),
                (self.verify_positions_i32, position_i32),
                (self.prefill_context_count_buf, context_i64),
            ]
        if self._verify_metadata_bucket_cache == bucket_signature:
            copies: list[tuple[Any, Any]] = dynamic_copies
        else:
            copies = dynamic_copies + [
                (self.verify_parent_rows_i32, parent_i32),
                (self.verify_parent_rows_i64, parent_i64),
                (self.verify_draft_depths_i32, depth_i32),
                (self.verify_row_to_request_i32, row_req_i32),
                (self.verify_active_mask_u8, active_u8),
                (self.prefill_block_table_buf, block_table),
            ]
            self._verify_metadata_bucket_cache = bucket_signature
        # Tree topology: build the dense ``[rows, rows]`` ancestor mask, the
        # global ``tree_committed_count``, and the per-row unique cache-slot
        # vector so the tree-aware K/V append + GQA gate kernels can filter
        # sibling/cousin rows without write collisions.  Roots have
        # ``parent == -1`` and are their own ancestors; every other row
        # inherits its parent's ancestor set plus itself.
        if batch.mode == "verify_tree":
            ancestor_mask = np.zeros((rows, rows), dtype=np.uint8)
            parents = list(batch.parent_rows)
            for i in range(rows):
                cursor = i
                while cursor >= 0:
                    ancestor_mask[i, cursor] = 1
                    parent = parents[cursor]
                    cursor = parent if parent >= 0 else -1
            root_positions = [int(batch.positions[r]) for r in batch.root_rows]
            if not root_positions:
                raise ValueError("verify_tree batch must have at least one root row")
            # All roots in a single-request verify share the same global
            # position (the current decode position).  For multi-request
            # batches we conservatively take the min so committed-context
            # positions are visible to every row.
            self.verify_tree_committed_count = min(root_positions)
            # Per-row unique K/V cache slot: ``tree_committed_count + i``.
            # The depth-based ``batch.positions`` already lives in
            # ``prefill_position_buf`` for RoPE; this separate slot vector
            # is what the prompt-style K/V write reads to place each row
            # in its own cache cell.
            cache_slot_i64 = np.arange(
                self.verify_tree_committed_count,
                self.verify_tree_committed_count + rows,
                dtype=np.int64,
            )
            copies.append((self.verify_ancestor_mask_u8, ancestor_mask))
            copies.append((self.verify_cache_slot_buf, cache_slot_i64))
            copies.append((self.verify_tree_committed_buf, np.asarray([self.verify_tree_committed_count], dtype=np.int64)))
            # Tree uniform context counts/positions are consumed by the
            # verifier forward; populate here (outside any graph capture —
            # a sync H2D inside the captured forward is HIP error 906).
            total_len = int(self.verify_tree_committed_count) + rows
            self._ensure_verify_tree_uniform_buf(
                name="verify_tree_uniform_counts",
                host=np.full((rows,), total_len, dtype=np.int64),
            )
            self._ensure_verify_tree_uniform_buf(
                name="verify_tree_uniform_positions",
                host=np.full((rows,), total_len - 1, dtype=np.int64),
            )
        else:
            # Chain mode: leave the ancestor mask and cache-slot buffers
            # alone.  Recorded ``tree_committed_count`` is meaningless for
            # chain; readers must branch on ``batch.mode`` before
            # consulting it.
            self.verify_tree_committed_count = 0
        for buffer, array in copies:
            contiguous = np.ascontiguousarray(array)
            copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes, runtime=self.runtime)
        if packed_dynamic_metadata is not None:
            contiguous = np.ascontiguousarray(packed_dynamic_metadata)
            copy_host_to_device(
                self.verify_dynamic_metadata_i64,
                host_array_ptr(contiguous),
                contiguous.nbytes,
                runtime=self.runtime,
            )
            unpack_verify_chain_dynamic_metadata_i64(
                self.verify_dynamic_metadata_i64.ptr,
                self.verify_token_ids_i64.ptr,
                self.verify_token_ids_i32.ptr,
                self.prefill_position_buf.ptr,
                self.verify_positions_i32.ptr,
                self.prefill_context_count_buf.ptr,
                rows,
                stream=stream,
                library=self.libraries["runtime_state"],
                runtime=self.runtime,
            )

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
        # M12.2: when the verifier processes more than one row, the weight-
        # sharing kernel reads the W8 LM-head weights from HBM once per block
        # and amortizes them across all verifier rows.  For ``rows == 1`` the
        # stock kernel is already optimal and we keep it.  Env override
        # ``HIPENGINE_W8A16_LM_HEAD_MULTI_ROW`` ("0"/"off" to disable) lets us
        # bisect this path.
        #
        # R3.7: when ``HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD=on`` we replace the
        # full-vocab GEMV + argmax-rows pair with a single fused kernel that
        # never materializes the [rows, vocab] FP32 logits buffer in HBM.
        # Bit-exact vs the unfused path (same cooperative-per-vocab-row dot
        # product reduction).
        if _dflash_verify_fused_lm_head_enabled():
            w8a16_lm_head_argmax_rows_bf16(
                norm_out_bf16.ptr,
                self.lm_head_weight.tensor.ptr,
                self.lm_head_scale.tensor.ptr,
                self.verify_lm_block_values.ptr,
                self.verify_lm_block_indices.ptr,
                self.verify_top1_i32.ptr,
                self.verify_top1_values.ptr,
                rows,
                self.config.hidden_size,
                self.vocab_size,
                threads=self.lm_head_threads,
                stream=stream,
                library=self.libraries["lm_head"],
                runtime=self.runtime,
            )
            return
        if rows > 1 and self._w8a16_lm_head_multi_row_enabled():
            w8a16_linear_bf16_f32_multi_row(
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
        else:
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

    def _launch_verify_accept_summary(
        self,
        batch: TargetVerifyBatch,
        *,
        rows: int,
        base_slot: int | None = None,
        stream: int = 0,
    ) -> None:
        request_count = len(batch.request_ids)
        accept_args = (
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
        )
        if self._verify_accept_packed_payload_enabled():
            if self._verify_accept_updates_position_enabled(batch) and base_slot is not None:
                dflash_accept_chain_i32_packed_update_state(
                    *accept_args,
                    self.verify_accept_payload_i32.ptr,
                    self.position_buf.ptr + int(base_slot) * DType.INT64.itemsize,
                    self.context_buf.ptr + int(base_slot) * DType.INT64.itemsize,
                    rows,
                    request_count,
                    rows,
                    stream=stream,
                    library=self.libraries["dflash_accept"],
                    runtime=self.runtime,
                )
                return
            dflash_accept_chain_i32_packed(
                *accept_args,
                self.verify_accept_payload_i32.ptr,
                rows,
                request_count,
                rows,
                stream=stream,
                library=self.libraries["dflash_accept"],
                runtime=self.runtime,
            )
            return
        dflash_accept_chain_i32(
            *accept_args,
            rows,
            request_count,
            rows,
            stream=stream,
            library=self.libraries["dflash_accept"],
            runtime=self.runtime,
        )

    def _read_verify_accept_payload(self, request_count: int, *, stream: int = 0) -> dict[str, tuple[int, ...] | tuple[bool, ...]]:
        self.runtime.stream_synchronize(stream)
        if not self._verify_accept_packed_payload_enabled():
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
        payload = np.empty((request_count, ACCEPT_PACKED_PAYLOAD_FIELDS), dtype=np.int32)
        copy_device_to_host(
            host_array_ptr(payload),
            DeviceBuffer(self.verify_accept_payload_i32.ptr, payload.nbytes),
            runtime=self.runtime,
        )
        accepted = payload[:, 0]
        commit_rows = payload[:, 1]
        commit_tokens = payload[:, 2]
        commit_positions = payload[:, 3]
        next_tokens = payload[:, 4]
        full_accept = payload[:, 5]
        out_lengths = payload[:, 6]
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

    def _tree_path_rows(self, batch: TargetVerifyBatch, selected_row: int) -> tuple[int, ...]:
        """Return accepted tree path rows root→leaf for ``selected_row``."""

        if batch.mode != "verify_tree":
            return (int(selected_row),)
        if selected_row < 0 or selected_row >= batch.rows:
            raise RuntimeError(f"selected tree row {selected_row} outside batch rows")
        path: list[int] = []
        seen: set[int] = set()
        cursor = int(selected_row)
        while cursor >= 0:
            if cursor in seen:
                raise RuntimeError("cycle detected while walking tree parent rows")
            seen.add(cursor)
            path.append(cursor)
            parent = int(batch.parent_rows[cursor])
            if parent < 0:
                break
            cursor = parent
        path.reverse()
        if not path or path[0] not in batch.root_rows:
            raise RuntimeError("accepted tree path did not reach a root row")
        return tuple(path)

    def _commit_tree_capture_hidden_concat(
        self,
        batch: TargetVerifyBatch,
        selected_row: int,
        capture_hidden_concat: Tensor,
        *,
        capture_row_start: int,
        stream: int = 0,
    ) -> None:
        """Compact accepted-path hidden taps into dense context rows.

        ``verify_tree_bulk_and_commit`` writes target-hidden taps in verifier-row
        order: ``capture_row_start + tree_row``.  A real branching tree may
        accept a non-contiguous path such as rows ``[0, 2]`` (root then the
        second root child).  The native DFlash drafter's append-only context
        cache expects committed context rows to be dense at
        ``capture_row_start + depth`` before ``commit_context_rows(start,
        count)`` projects them.  Copy the accepted path root→leaf into that
        dense layout.  Degenerate chains are already dense and become no-ops.
        """

        if batch.mode != "verify_tree":
            return
        path = self._tree_path_rows(batch, int(selected_row))
        if len(path) <= 1:
            return
        row_nbytes = int(capture_hidden_concat.shape[1]) * capture_hidden_concat.dtype.itemsize
        for depth, source_row in enumerate(path):
            src_row = capture_row_start + int(source_row)
            dst_row = capture_row_start + int(depth)
            if src_row == dst_row:
                continue
            self.runtime.memcpy_async(
                capture_hidden_concat.ptr + dst_row * row_nbytes,
                capture_hidden_concat.ptr + src_row * row_nbytes,
                row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )

    def _commit_tree_full_attention_kv(
        self,
        batch: TargetVerifyBatch,
        selected_row: int,
        *,
        base_slot: int,
        stream: int = 0,
    ) -> None:
        """Compact accepted-path K/V cells into canonical dense cache slots.

        After ``verify_tree_bulk_and_commit`` runs, each verifier row's K/V
        lives at cache slot ``tree_committed_count + row_index``, which is
        sparse: unaccepted sibling rows occupy slots between the accepted
        path's slots.  For a multi-cycle decode to read the correct context
        on the next step, the accepted path's K/V must occupy slots
        ``[tree_committed_count, tree_committed_count + accepted_count]``
        densely.

        This method walks ``batch.parent_rows`` from ``selected_row`` back to
        the root, collecting the accepted path (root first, leaf last), and
        for every full-attention layer copies each path row's K/V cell from
        its sparse source slot ``tree_committed_count + path_row`` to its
        canonical dense destination slot ``tree_committed_count + depth``.
        Source slots are strictly greater than (or equal to) destination
        slots because tree row indices grow with depth, so the copy ordering
        root→leaf is safe (no source slot is overwritten before being read).

        Linear-attention recurrent state for the leaf is committed separately
        by ``_commit_bulk_linear_states``; only the leaf state matters because
        linear attention is path-dependent and the tree t-loop already
        produced the leaf's exact state.
        """

        if batch.mode != "verify_tree":
            return
        path = self._tree_path_rows(batch, int(selected_row))
        if not path:
            return
        if int(self.verify_tree_committed_count) < 0:
            raise RuntimeError(
                "verify_tree_committed_count must be set before tree K/V commit"
            )
        committed = int(self.verify_tree_committed_count)
        # Compute the byte stride of a single position's K/V cell from one
        # of the resident full-attention caches.  Layout within a slot view
        # is ``[blocks, block_size, num_kv_heads, head_dim]`` packed
        # contiguously, so position ``P`` lives at byte offset
        # ``P * num_kv_heads * head_dim * itemsize`` from the slot base.
        if not self.full_caches:
            return  # no full-attention layers; nothing to compact
        sample_layer = next(iter(self.full_caches))
        sample_key_cache, _sample_value_cache, _key_buf, _value_buf = self.full_caches[sample_layer]
        cell_nbytes = int(
            self.config.num_key_value_heads
            * self.config.head_dim
            * sample_key_cache.dtype.itemsize
        )
        copies: list[tuple[int, int]] = []  # (dest_position, source_position)
        for depth, source_row in enumerate(path):
            source_position = committed + int(source_row)
            dest_position = committed + depth
            if source_position == dest_position:
                continue
            if dest_position < 0 or dest_position >= self.max_sequence_length:
                raise RuntimeError(
                    f"dest position {dest_position} out of cache range"
                )
            copies.append((dest_position, source_position))
        if not copies:
            return
        for layer_id, (key_cache, _value_cache, _key_buf, _value_buf) in self.full_caches.items():
            layer_type = self.config.layer_types[layer_id]
            if layer_type != "full_attention":
                continue
            # ``_slot_full_cache`` returns slot-view tensors whose ``.ptr``
            # already points at the slot's K/V region inside the global
            # allocation, so position offsets within those tensors are
            # relative to the slot base.
            slot_key, slot_value = self._slot_full_cache(layer_id, base_slot)
            for dest_position, source_position in copies:
                src_off = source_position * cell_nbytes
                dst_off = dest_position * cell_nbytes
                self.runtime.memcpy_async(
                    slot_key.ptr + dst_off,
                    slot_key.ptr + src_off,
                    cell_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                self.runtime.memcpy_async(
                    slot_value.ptr + dst_off,
                    slot_value.ptr + src_off,
                    cell_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            # Force ``key_cache`` to look used so static analyzers don't
            # warn about the dict-iteration variable.
            _ = key_cache

    def _commit_bulk_linear_states(
        self,
        selected_row: int,
        *,
        base_slot: int,
        stream: int = 0,
        commit_row_ptr: int | None = None,
    ) -> None:
        """Commit accepted linear-attention conv+recurrent state for all layers.

        Fast path (M12.4, default): one kernel launch reads ``commit_row`` from
        device-resident ``commit_row_ptr`` and copies the selected row of
        ``scratch.tree_conv_state`` / ``tree_recurrent_state`` into every
        linear-attention layer's canonical slot.  Replaces ``2 * n_layers``
        ``hipMemcpyAsync`` calls (~5 µs each at the Python/ctypes boundary).

        Fallback path: legacy per-layer ``hipMemcpyAsync`` loop, used when the
        device-resident commit row is unavailable (``commit_row_ptr is None``),
        when ``base_slot != 0`` (commit tables are built for slot 0), or when
        ``HIPENGINE_FUSED_LINEAR_STATE_COMMIT`` is explicitly disabled.
        """

        if (
            commit_row_ptr is not None
            and base_slot == 0
            and self._fused_linear_state_commit_enabled()
            and getattr(self, "linear_state_dst_conv_table_buf", None) is not None
            and self.linear_scratch
            and len(self.linear_scratch) == len(self.linear_layer_ids)
        ):
            # Refresh per-layer source pointer tables (per-layer workspaces do
            # not share buffers).  Host-stage current pointers in cached
            # ndarrays and only H2D-copy when at least one pointer changed.
            conv_host = self.linear_state_src_conv_host
            rec_host = self.linear_state_src_recurrent_host
            conv_cached = self.linear_state_src_conv_cached
            rec_cached = self.linear_state_src_recurrent_cached
            for idx, layer_id in enumerate(self.linear_layer_ids):
                scratch = self.linear_scratch[layer_id]
                conv_host[idx] = np.uint64(scratch.tree_conv_state.ptr)
                rec_host[idx] = np.uint64(scratch.tree_recurrent_state.ptr)
            if not np.array_equal(conv_host, conv_cached):
                copy_host_to_device(
                    self.linear_state_src_conv_table_buf,
                    host_array_ptr(conv_host),
                    conv_host.nbytes,
                    runtime=self.runtime,
                )
                np.copyto(conv_cached, conv_host)
            if not np.array_equal(rec_host, rec_cached):
                copy_host_to_device(
                    self.linear_state_src_recurrent_table_buf,
                    host_array_ptr(rec_host),
                    rec_host.nbytes,
                    runtime=self.runtime,
                )
                np.copyto(rec_cached, rec_host)
            linear_commit = (
                linear_state_pair_commit_chunked_i32
                if self._chunked_linear_state_commit_enabled()
                else linear_state_pair_commit_i32
            )
            linear_commit(
                self.linear_state_src_conv_table_buf.ptr,
                self.linear_state_dst_conv_table_buf.ptr,
                self.linear_state_conv_row_nbytes,
                self.linear_state_src_recurrent_table_buf.ptr,
                self.linear_state_dst_recurrent_table_buf.ptr,
                self.linear_state_recurrent_row_nbytes,
                int(commit_row_ptr),
                len(self.linear_layer_ids),
                stream=stream,
                library=self.libraries["dflash_commit"],
                runtime=self.runtime,
            )
            return
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

    def _invalidate_verify_graph_cache(self) -> None:
        """Destroy cached verifier graphs (their baked scratch pointers may dangle)."""

        self._clear_verify_scratch_caches()
        if not getattr(self, "_verify_graph_cache", None):
            return
        # A replay may still be in flight on the default stream; freeing the
        # scratch the graph writes (caller does so right after) corrupts it.
        self.runtime.device_synchronize()
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

    def _mlp_decode_scratch(
        self,
        layer_id: int,
        state: Qwen35ParoDecodeState,
    ) -> Qwen35ParoMoeScratch | Qwen35ParoDenseMlpScratch:
        """Return canonical c=1 MoE/MLP scratch for resident decode."""

        scratch = self.moe_scratch.get(layer_id)
        if int(getattr(self.config, "num_experts", 1) or 0) <= 0:
            if isinstance(scratch, Qwen35ParoDenseMlpScratch) and scratch.normed.shape[0] == 1:
                return scratch
            # Re-reserving workspace names at a new shape frees the rows=B+1
            # buffers any cached verifier graph holds raw pointers to (#107).
            self._invalidate_verify_graph_cache()
            scratch = state.reserve_dense_mlp_scratch(tokens=1, activation_dtype=DType.FP16)
        else:
            if isinstance(scratch, Qwen35ParoMoeScratch) and scratch.normed.shape[0] == 1:
                return scratch
            self._invalidate_verify_graph_cache()
            scratch = state.reserve_moe_c1_scratch(tokens=1, activation_dtype=DType.FP16)
        self.moe_scratch[layer_id] = scratch
        return scratch

    def _canonicalize_decode_scratch(self) -> None:
        """Make resident c=1 scratch views current after bulk/prefill paths."""

        for layer_id, state in enumerate(self.states):
            self.moe_scratch[layer_id] = self._mlp_decode_scratch(layer_id, state)
            if self.config.layer_types[layer_id] == "linear_attention":
                self.linear_scratch[layer_id] = self._linear_decode_scratch(layer_id, state)

    def _linear_decode_scratch(self, layer_id: int, state: Qwen35ParoDecodeState) -> Qwen35ParoLinearAttentionScratch:
        """Return canonical c=1 linear-attention scratch for resident decode.

        Native bulk verifier passes reserve ``tokens=rows`` scratch and store it
        in ``self.linear_scratch`` so the selected ``tree_*_state`` row can be
        committed after accept.  That scratch is **not** safe to reuse for a
        later c=1 decode step: several split views (notably ``qkv/z`` and
        ``a/b``) place the second view at an offset derived from the reserved
        row count, while the c=1 projection kernels write a compact
        one-row-concatenated layout.  Reusing verifier-sized scratch therefore
        makes c=1 GDN read stale z/b rows after a bulk→AR handoff.
        """

        scratch = self.linear_scratch.get(layer_id)
        if not isinstance(scratch, Qwen35ParoLinearAttentionScratch):
            # Re-reserving at rows=1 frees rows=B+1 buffers any cached verifier
            # graph holds raw pointers to (#107).
            self._invalidate_verify_graph_cache()
            scratch = state.reserve_linear_attention_scratch(tokens=1, activation_dtype=DType.FP16)
            self.linear_scratch[layer_id] = scratch
            return scratch
        if (
            scratch.attn_input.shape[0] == 1
            and scratch.qkv_z.shape[0] == 1
            and scratch.ab.shape[0] == 1
            and scratch.tree_recurrent_state.shape[0] == 1
        ):
            return scratch

        cfg = self.config
        qkv_width = (
            2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim
        )
        z_width = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        lowp = scratch.attn_input.dtype

        def row_view(tensor: Tensor, *tail_shape: int) -> Tensor:
            return Tensor.from_handle(tensor.ptr, (1, *tuple(int(x) for x in tail_shape)), tensor.dtype, tensor.device)

        qkv_z = Tensor.from_handle(scratch.qkv_z.ptr, (1, qkv_width + z_width), lowp, scratch.qkv_z.device)
        qkv = Tensor.from_handle(qkv_z.ptr, (1, qkv_width), lowp, qkv_z.device)
        z = Tensor.from_handle(qkv_z.ptr + qkv_width * lowp.itemsize, (1, z_width), lowp, qkv_z.device)
        ab = Tensor.from_handle(scratch.ab.ptr, (1, 2 * cfg.linear_num_value_heads), lowp, scratch.ab.device)
        a = Tensor.from_handle(ab.ptr, (1, cfg.linear_num_value_heads), lowp, ab.device)
        b = Tensor.from_handle(
            ab.ptr + cfg.linear_num_value_heads * lowp.itemsize,
            (1, cfg.linear_num_value_heads),
            lowp,
            ab.device,
        )
        compact = Qwen35ParoLinearAttentionScratch(
            attn_input=row_view(scratch.attn_input, cfg.hidden_size),
            qkv_rot=row_view(scratch.qkv_rot, cfg.hidden_size),
            z_rot=row_view(scratch.z_rot, cfg.hidden_size),
            rotate_fuse_barrier=scratch.rotate_fuse_barrier,
            qkv_z=qkv_z,
            qkv=qkv,
            z=z,
            qkv_f32=row_view(scratch.qkv_f32, qkv_width),
            ab=ab,
            a=a,
            b=b,
            conv_out=row_view(scratch.conv_out, qkv_width),
            prefill_query=row_view(scratch.prefill_query, cfg.linear_num_value_heads, cfg.linear_key_head_dim),
            prefill_key=row_view(scratch.prefill_key, cfg.linear_num_value_heads, cfg.linear_key_head_dim),
            prefill_value=row_view(scratch.prefill_value, cfg.linear_num_value_heads, cfg.linear_value_head_dim),
            prefill_beta=row_view(scratch.prefill_beta, cfg.linear_num_value_heads),
            prefill_decay=row_view(scratch.prefill_decay, cfg.linear_num_value_heads),
            recurrent_out=row_view(scratch.recurrent_out, z_width),
            recurrent_bf16=row_view(scratch.recurrent_bf16, z_width),
            out_rot=row_view(scratch.out_rot, z_width),
            out_proj=row_view(scratch.out_proj, cfg.hidden_size),
            tree_conv_state=row_view(scratch.tree_conv_state, qkv_width, cfg.linear_conv_kernel_dim),
            tree_recurrent_state=row_view(
                scratch.tree_recurrent_state,
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
            ),
            tree_gdn_acc=row_view(scratch.tree_gdn_acc, z_width),
        )
        self.linear_scratch[layer_id] = compact
        return compact

    def _build_linear_state_commit_tables(self, *, qkv_width: int) -> None:
        """Build device-resident destination pointer tables for the M12.4
        single-launch linear-state commit path.

        ``linear_state_pair_commit_i32`` consumes two ``uint64[n_layers]``
        device tables: one of conv destination row pointers, one of recurrent
        destination row pointers, both for ``base_slot=0`` (per-row offset
        within each layer's canonical buffer).  We populate these once after
        ``_materialize_layers`` allocates the canonical layer-state buffers
        and record per-row byte sizes for the launch.
        """

        linear_layer_ids = tuple(
            layer_id
            for layer_id, layer_type in enumerate(self.config.layer_types[: self.layer_limit])
            if layer_type == "linear_attention"
        )
        self.linear_layer_ids: tuple[int, ...] = linear_layer_ids
        if not linear_layer_ids:
            self.linear_state_dst_conv_table_buf = None
            self.linear_state_dst_recurrent_table_buf = None
            self.linear_state_conv_row_nbytes = 0
            self.linear_state_recurrent_row_nbytes = 0
            return
        conv_row_nbytes = (
            int(qkv_width)
            * int(self.config.linear_conv_kernel_dim)
            * DType.FP32.itemsize
        )
        recurrent_row_nbytes = (
            int(self.config.linear_num_value_heads)
            * int(self.config.linear_key_head_dim)
            * int(self.config.linear_value_head_dim)
            * DType.FP32.itemsize
        )
        dst_conv_table = np.empty((len(linear_layer_ids),), dtype=np.uint64)
        dst_recurrent_table = np.empty((len(linear_layer_ids),), dtype=np.uint64)
        for idx, layer_id in enumerate(linear_layer_ids):
            _conv_state, _recurrent_state, conv_buf, recurrent_buf, _, _ = self.linear_states[layer_id]
            # Slot 0 destination: offset 0 within each layer's canonical buffer.
            dst_conv_table[idx] = np.uint64(conv_buf.ptr)
            dst_recurrent_table[idx] = np.uint64(recurrent_buf.ptr)
        # ``_dev`` already registers each new device buffer with ``self.buffers``
        # so the resident session's close path frees them; do not re-append.
        self.linear_state_dst_conv_table_buf = self._dev(dst_conv_table)
        self.linear_state_dst_recurrent_table_buf = self._dev(dst_recurrent_table)
        # Source pointer tables are refreshed each commit because each layer's
        # Qwen35ParoDecodeState owns its own RuntimeWorkspace; per-layer
        # ``tree_conv_state`` / ``tree_recurrent_state`` allocations therefore
        # do NOT share a base address.  We pre-allocate the device-side table
        # buffers once and host-stage the current pointers each cycle.
        zero_table = np.zeros((len(linear_layer_ids),), dtype=np.uint64)
        self.linear_state_src_conv_table_buf = self._dev(zero_table)
        self.linear_state_src_recurrent_table_buf = self._dev(zero_table.copy())
        # Cached src pointer arrays for cycle-to-cycle refresh; lazily filled
        # by ``_commit_bulk_linear_states`` from the live per-layer scratch.
        self.linear_state_src_conv_host = np.zeros((len(linear_layer_ids),), dtype=np.uint64)
        self.linear_state_src_recurrent_host = np.zeros((len(linear_layer_ids),), dtype=np.uint64)
        # Track the previous-cycle src pointers so we can skip the H2D refresh
        # when the workspace did not reallocate (the common case once a stable
        # ``rows`` bucket warms up).
        self.linear_state_src_conv_cached = np.zeros((len(linear_layer_ids),), dtype=np.uint64)
        self.linear_state_src_recurrent_cached = np.zeros((len(linear_layer_ids),), dtype=np.uint64)
        self.linear_state_conv_row_nbytes = conv_row_nbytes
        self.linear_state_recurrent_row_nbytes = recurrent_row_nbytes

    def _fused_linear_state_commit_enabled(self) -> bool:
        value = os.environ.get("HIPENGINE_FUSED_LINEAR_STATE_COMMIT")
        if value is None or value.strip() == "":
            return True
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _chunked_linear_state_commit_enabled(self) -> bool:
        value = os.environ.get("HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED")
        if value is None or value.strip() == "":
            return True
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _w8a16_lm_head_multi_row_enabled(self) -> bool:
        value = os.environ.get("HIPENGINE_W8A16_LM_HEAD_MULTI_ROW")
        if value is None or value.strip() == "":
            return True
        return value.strip().lower() not in {"0", "false", "no", "off"}

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


@dataclass
class Qwen35ParoBatchDecodeGraph:
    """Replayable HIP graph for one device-resident c>1 decode step (C3.0b piece D).

    A single captured step is replayed once per generated token: it reads the
    input tokens from ``batch_lm_out_index``, runs the c-aware layers, writes the
    next-token argmax back to ``batch_lm_out_index``, and advances the device
    decode position/context counters.  ``seed_tokens`` primes the first input,
    ``reset_positions`` rewinds the device counters to the captured start, and
    ``replay``/``replay_collect`` drive the launches.
    """

    session: "Qwen35ParoResidentSession"
    graph: int
    graph_exec: int
    stream: int
    rows: int
    slots: tuple[int, ...]
    start_positions: tuple[int, ...]
    max_replay_steps: int
    launches: int = 0
    closed: bool = False

    def seed_tokens(self, token_ids: Sequence[int]) -> None:
        """Write the first replay step's input tokens into ``batch_lm_out_index``."""

        if self.closed:
            raise RuntimeError("batch decode graph is closed")
        tokens = np.asarray([int(token) for token in token_ids], dtype=np.int64)
        if tokens.size != self.rows:
            raise ValueError("seed token count must match rows")
        copy_host_to_device(
            DeviceBuffer(self.session.batch_lm_out_index.ptr, self.rows * DType.INT64.itemsize),
            host_array_ptr(tokens),
            tokens.nbytes,
            runtime=self.session.runtime,
        )

    def reset_positions(self) -> None:
        """Rewind the device decode counters to the captured start positions."""

        if self.closed:
            raise RuntimeError("batch decode graph is closed")
        self.session._set_batch_positions(self.start_positions, stream=self.stream)
        self.session.runtime.stream_synchronize(self.stream)

    def replay(self, steps: int) -> None:
        """Replay ``steps`` decode tokens back-to-back, syncing once at the end."""

        if self.closed:
            raise RuntimeError("batch decode graph is closed")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps > self.max_replay_steps:
            raise ValueError("steps exceed captured max_replay_steps")
        for _ in range(steps):
            self.session.runtime.graph_launch(self.graph_exec, self.stream)
            self.launches += 1
        self.session.runtime.stream_synchronize(self.stream)

    def step_tokens(self) -> list[int]:
        """Read the current device next-token ids (one per active row)."""

        if self.closed:
            raise RuntimeError("batch decode graph is closed")
        host = np.empty((self.rows,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(self.session.batch_lm_out_index.ptr, self.rows * DType.INT64.itemsize),
            runtime=self.session.runtime,
        )
        return [int(item) for item in host.tolist()]

    def replay_collect(self, steps: int) -> list[list[int]]:
        """Replay ``steps`` tokens one launch at a time, collecting each step's row tokens."""

        if self.closed:
            raise RuntimeError("batch decode graph is closed")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps > self.max_replay_steps:
            raise ValueError("steps exceed captured max_replay_steps")
        collected: list[list[int]] = []
        for _ in range(steps):
            self.session.runtime.graph_launch(self.graph_exec, self.stream)
            self.launches += 1
            self.session.runtime.stream_synchronize(self.stream)
            collected.append(self.step_tokens())
        return collected

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.session.runtime.graph_exec_destroy(self.graph_exec)
        self.session.runtime.graph_destroy(self.graph)
        if self.stream:
            self.session.runtime.stream_destroy(self.stream)

    def __enter__(self) -> "Qwen35ParoBatchDecodeGraph":
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
