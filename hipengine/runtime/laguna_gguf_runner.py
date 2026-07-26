"""Torch-free resident runner for Poolside Laguna S 2.1 GGUF.

The runner keeps exact eager c=1 decode while adding bounded chunked prompt and
B+1 verifier rows over mixed global/SWA ``KVLiveSpans`` attention, deterministic
owned scratch, greedy top-1, and caller-owned DFlash hidden taps.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.kernels.backends import (
    backend_package_capability,
    hip_target_arch_environment,
    hip_target_arch_for_backend,
    load_backend_kernel_package,
    resolve_backend,
)
from hipengine.kernels.hip_gfx1100.convert.cast import (
    bf16_to_fp16,
    bf16_to_fp16_scaled_rows,
    f32_to_bf16,
    f32_scale_rows,
    f32_scale_rows_to_bf16,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import lm_head_argmax_stage1_blocks
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf import (
    DENSE_MLP,
    FULL_ATTENTION,
    PER_HEAD_GATE,
    SLIDING_ATTENTION,
    SPARSE_MOE,
    LagunaGGUFConfig,
    laguna_gguf_config_from_metadata,
)
from hipengine.loading.laguna_gguf_materialize import (
    DEFAULT_LAGUNA_SAFETY_RESERVE_BYTES,
    DEFAULT_LAGUNA_SCRATCH_BYTES,
    LAYOUT_DENSE_F16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    LagunaGGUFRepackedCache,
    LagunaGGUFResidentLayerWeights,
    LagunaGGUFResidentWeights,
    materialize_laguna_gguf_weights,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import (
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
    launch_gguf_linear_pair,
)
from hipengine.runtime.f16_weight_linear import (
    launch_f16_weight_linear,
    launch_f16_weight_linear_triple,
)
from hipengine.runtime.laguna_f16_hipblaslt import (
    LagunaF16HipblasLt,
    laguna_attention_gated_fp16_bound,
    laguna_attention_norm_fp16_bound,
    resolve_laguna_f16_prefill_mode,
)
from hipengine.runtime.laguna_kv import (
    LagunaKVCache,
    allocate_laguna_kv_cache,
    resolve_laguna_global_prefill_variant,
    resolve_laguna_swa_decode_variant,
    resolve_laguna_swa_prefill_variant,
)
from hipengine.runtime.laguna_moe import (
    LagunaMoEKernelPlan,
    LagunaMoEScratch,
    allocate_laguna_moe_scratch,
    laguna_moe_scratch_nbytes,
    resolve_laguna_dense_q4_prefill_mode,
    resolve_laguna_group_compact_mode,
    resolve_laguna_moe_plan,
    resolve_laguna_router_logits_mode,
    resolve_laguna_selected_down_mode,
    resolve_laguna_selected_gate_up_mode,
    run_laguna_moe_c1,
    run_laguna_moe_rows,
    validate_laguna_moe_layer,
)
from hipengine.runtime.laguna_rope import (
    LagunaDeviceRoPETables,
    launch_laguna_head_rmsnorm_rope,
    materialize_laguna_rope_tables,
)

LAGUNA_DFLASH_CAPTURE_DEPTHS = (2, 11, 20, 30, 39, 48)
_INITIAL_MAX_CONTEXT = 4_096
_EXPECTED_HEAD_COUNTS = tuple([48, 72, 72, 72] * 12)
_EXPECTED_LAYER_TYPES = tuple(
    FULL_ATTENTION if layer_id % 4 == 0 else SLIDING_ATTENTION for layer_id in range(48)
)
_BF16_NBYTES = DType.BF16.itemsize
_F32_NBYTES = DType.FP32.itemsize
_I32_NBYTES = DType.INT32.itemsize
_I64_NBYTES = DType.INT64.itemsize
_U8_NBYTES = DType.BOOL.itemsize


def resolve_laguna_moe_branch_concurrency(
    backend: str,
    requested: bool | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve the exact MoE overlap default against process queue capacity."""

    if requested is not None:
        return bool(requested)
    env = os.environ if environ is None else environ
    queue_count_raw = env.get("GPU_MAX_HW_QUEUES")
    try:
        queue_count = 2 if queue_count_raw is None else int(queue_count_raw)
    except ValueError:
        queue_count = 0
    return bool(
        queue_count >= 2
        and backend_package_capability(
            backend,
            "LAGUNA_MOE_BRANCH_CONCURRENCY",
            False,
        )
    )


_PROJECTION_LAYOUT_BY_QUANT = MappingProxyType(
    {
        "fp16": LAYOUT_DENSE_F16,
        "gguf_q5_k": LAYOUT_RAW_GGUF,
        "gguf_q6_k": LAYOUT_RAW_GGUF,
        "gguf_q8_0": LAYOUT_RAW_GGUF,
    }
)
_ROOT_LAYOUTS_BY_SLOT = MappingProxyType(
    {
        "token_embedding": MappingProxyType(
            {"gguf_q4_k": LAYOUT_RAW_GGUF, "gguf_q5_k": LAYOUT_RAW_GGUF}
        ),
        "output_norm": MappingProxyType({"f32": LAYOUT_DENSE_F32}),
        "lm_head": MappingProxyType(
            {"gguf_q4_k": LAYOUT_RAW_GGUF, "gguf_q6_k_t16_v1": LAYOUT_GGUF_Q6_K_T16}
        ),
    }
)
_DENSE_MLP_LAYOUTS_BY_SLOT = MappingProxyType(
    {
        "ffn_gate": MappingProxyType(
            {"gguf_q4_k": LAYOUT_Q4_K_PACK8, "gguf_q5_k": LAYOUT_RAW_GGUF}
        ),
        "ffn_up": MappingProxyType(
            {"gguf_q4_k": LAYOUT_Q4_K_PACK8, "gguf_q5_k": LAYOUT_RAW_GGUF}
        ),
        "ffn_down": MappingProxyType({"gguf_q6_k": LAYOUT_RAW_GGUF}),
    }
)


@dataclass(frozen=True)
class LagunaHiddenCaptureTargets:
    """Caller-owned BF16 destinations for one or more post-layer hidden rows."""

    hidden_size: int
    buffers: Mapping[int, DeviceBuffer]
    rows: int = 1

    def __post_init__(self) -> None:
        hidden_size = int(self.hidden_size)
        rows = int(self.rows)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if rows <= 0:
            raise ValueError("capture rows must be positive")
        expected_nbytes = rows * hidden_size * _BF16_NBYTES
        normalized: dict[int, DeviceBuffer] = {}
        for raw_depth, buffer in self.buffers.items():
            depth = int(raw_depth)
            if depth not in LAGUNA_DFLASH_CAPTURE_DEPTHS:
                raise ValueError(
                    "Laguna hidden captures are limited to the configured DFlash depths "
                    f"{LAGUNA_DFLASH_CAPTURE_DEPTHS}; got {depth}"
                )
            if not isinstance(buffer, DeviceBuffer):
                raise TypeError("Laguna hidden capture destinations must be DeviceBuffer views")
            if buffer.nbytes != expected_nbytes:
                row_label = "one BF16 hidden row" if rows == 1 else f"{rows} BF16 hidden rows"
                raise ValueError(
                    f"each Laguna hidden capture target must hold exactly {row_label}; "
                    f"depth={depth} expected={expected_nbytes} actual={buffer.nbytes}"
                )
            normalized[depth] = buffer
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "buffers", MappingProxyType(normalized))
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True)
class LagunaEagerKernelPlan:
    """Exact registry keys and callables used by the eager session."""

    backend: str
    rmsnorm_key: KernelKey
    rmsnorm_fp16_via_bf16_key: KernelKey
    add_rmsnorm_key: KernelKey
    add_key: KernelKey
    attention_gate_key: KernelKey
    attention_gate_fp16_via_bf16_key: KernelKey
    dense_silu_key: KernelKey
    argmax_key: KernelKey
    f16_triple_key: KernelKey
    f16_f32_key: KernelKey
    f16_bf16_key: KernelKey
    rope_key: KernelKey
    rmsnorm: Callable
    rmsnorm_fp16_via_bf16: Callable
    add_rmsnorm: Callable
    add: Callable
    attention_gate: Callable
    attention_gate_fp16_via_bf16: Callable
    dense_silu: Callable
    argmax: Callable

    @property
    def kernel_keys(self) -> tuple[KernelKey, ...]:
        return (
            self.rmsnorm_key,
            self.rmsnorm_fp16_via_bf16_key,
            self.add_rmsnorm_key,
            self.add_key,
            self.attention_gate_key,
            self.attention_gate_fp16_via_bf16_key,
            self.dense_silu_key,
            self.argmax_key,
            self.f16_triple_key,
            self.f16_f32_key,
            self.f16_bf16_key,
            self.rope_key,
        )


@dataclass
class LagunaEagerScratch:
    """Deterministic c=1 scratch owner sized for Laguna's widest layer."""

    max_query_width: int
    max_query_heads: int
    token_id: DeviceBuffer
    position: DeviceBuffer
    hidden: DeviceBuffer
    norm: DeviceBuffer
    query: DeviceBuffer
    key: DeviceBuffer
    value: DeviceBuffer
    query_rotated: DeviceBuffer
    key_rotated: DeviceBuffer
    gate_logits: DeviceBuffer
    context: DeviceBuffer
    gated_context: DeviceBuffer
    attention_output: DeviceBuffer
    post_attention: DeviceBuffer
    dense_gate: DeviceBuffer
    dense_up: DeviceBuffer
    dense_intermediate: DeviceBuffer
    dense_output: DeviceBuffer
    final_norm: DeviceBuffer
    logits: DeviceBuffer
    argmax_block_values: DeviceBuffer
    argmax_block_indices: DeviceBuffer
    argmax_id: DeviceBuffer
    argmax_value: DeviceBuffer
    _closed: bool = False

    @classmethod
    def allocate(
        cls,
        config: LagunaGGUFConfig,
        *,
        runtime: HipRuntime | None = None,
    ) -> "LagunaEagerScratch":
        max_heads = max(int(value) for value in config.head_counts)
        max_query_width = max_heads * int(config.key_length)
        kv_width = int(config.head_count_kv) * int(config.key_length)
        hidden = int(config.hidden_size)
        dense_ffn = int(config.feed_forward_length)
        vocab = int(config.vocab_size)
        argmax_blocks = lm_head_argmax_stage1_blocks(vocab)
        sizes = (
            _I64_NBYTES,
            _I64_NBYTES,
            hidden * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            max_query_width * _F32_NBYTES,
            kv_width * _F32_NBYTES,
            kv_width * _F32_NBYTES,
            max_query_width * _F32_NBYTES,
            kv_width * _F32_NBYTES,
            max_heads * _F32_NBYTES,
            max_query_width * _F32_NBYTES,
            max_query_width * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            dense_ffn * _BF16_NBYTES,
            dense_ffn * _BF16_NBYTES,
            dense_ffn * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            vocab * _F32_NBYTES,
            argmax_blocks * _F32_NBYTES,
            argmax_blocks * _I64_NBYTES,
            _I64_NBYTES,
            _F32_NBYTES,
        )
        buffers: list[DeviceBuffer] = []
        try:
            buffers.extend(malloc(nbytes, runtime=runtime) for nbytes in sizes)
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            raise
        return cls(max_query_width, max_heads, *buffers)

    @property
    def buffers(self) -> tuple[DeviceBuffer, ...]:
        return (
            self.token_id,
            self.position,
            self.hidden,
            self.norm,
            self.query,
            self.key,
            self.value,
            self.query_rotated,
            self.key_rotated,
            self.gate_logits,
            self.context,
            self.gated_context,
            self.attention_output,
            self.post_attention,
            self.dense_gate,
            self.dense_up,
            self.dense_intermediate,
            self.dense_output,
            self.final_norm,
            self.logits,
            self.argmax_block_values,
            self.argmax_block_indices,
            self.argmax_id,
            self.argmax_value,
        )

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self.buffers)

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for buffer in reversed(self.buffers):
            free(buffer, runtime=runtime)


@dataclass(frozen=True)
class LagunaPrefillChunkPolicy:
    """Bounded matrix/MoE capacity with an independent attention query tile."""

    MAX_MATRIX_ROWS = 2_048

    matrix_rows: int
    attention_rows: int

    @classmethod
    def resolve(
        cls,
        *,
        context_length: int,
        matrix_rows: int,
        attention_rows: int | None = None,
    ) -> "LagunaPrefillChunkPolicy":
        context = int(context_length)
        matrix = int(matrix_rows)
        if matrix <= 0 or matrix > min(context, cls.MAX_MATRIX_ROWS):
            raise ValueError(
                "Laguna prefill matrix rows must be positive and no larger than context/2048"
            )
        attention = min(matrix, 128) if attention_rows is None else int(attention_rows)
        if attention <= 0 or attention > matrix:
            raise ValueError("Laguna prefill attention rows must be within matrix capacity")
        return cls(matrix_rows=matrix, attention_rows=attention)

    def attention_ranges(self, rows: int) -> tuple[tuple[int, int], ...]:
        """Return resident-position-backed ``(offset, rows)`` attention slices."""

        count = int(rows)
        if count <= 0 or count > self.matrix_rows:
            raise ValueError("Laguna prefill rows must fit matrix capacity")
        return tuple(
            (start, min(self.attention_rows, count - start))
            for start in range(0, count, self.attention_rows)
        )


@dataclass
class LagunaRowsScratch:
    """Bounded row-major scratch for chunked prefill and B+1 verification."""

    max_rows: int
    max_query_width: int
    max_query_heads: int
    token_ids: DeviceBuffer
    positions: DeviceBuffer
    hidden: DeviceBuffer
    norm: DeviceBuffer
    query: DeviceBuffer
    key: DeviceBuffer
    value: DeviceBuffer
    query_rotated: DeviceBuffer
    key_rotated: DeviceBuffer
    gate_logits: DeviceBuffer
    context: DeviceBuffer
    gated_context: DeviceBuffer
    attention_output: DeviceBuffer
    post_attention: DeviceBuffer
    dense_gate: DeviceBuffer
    dense_up: DeviceBuffer
    dense_intermediate: DeviceBuffer
    dense_output: DeviceBuffer
    final_norm: DeviceBuffer
    logits: DeviceBuffer
    _closed: bool = False

    @classmethod
    def _planned_sizes(
        cls,
        config: LagunaGGUFConfig,
        *,
        max_rows: int,
    ) -> tuple[int, ...]:
        rows = int(max_rows)
        if rows <= 0:
            raise ValueError("max_rows must be positive")
        max_heads = max(int(value) for value in config.head_counts)
        max_query_width = max_heads * int(config.key_length)
        kv_width = int(config.head_count_kv) * int(config.key_length)
        hidden = int(config.hidden_size)
        dense_ffn = int(config.feed_forward_length)
        vocab = int(config.vocab_size)
        return (
            rows * _I64_NBYTES,
            rows * _I64_NBYTES,
            rows * hidden * _BF16_NBYTES,
            rows * hidden * _BF16_NBYTES,
            rows * max_query_width * _F32_NBYTES,
            rows * kv_width * _F32_NBYTES,
            rows * kv_width * _F32_NBYTES,
            rows * max_query_width * _F32_NBYTES,
            rows * kv_width * _F32_NBYTES,
            rows * max_heads * _F32_NBYTES,
            rows * max_query_width * _F32_NBYTES,
            rows * max_query_width * _BF16_NBYTES,
            rows * hidden * _BF16_NBYTES,
            rows * hidden * _BF16_NBYTES,
            rows * dense_ffn * _BF16_NBYTES,
            rows * dense_ffn * _BF16_NBYTES,
            rows * dense_ffn * _BF16_NBYTES,
            rows * hidden * _BF16_NBYTES,
            rows * hidden * _BF16_NBYTES,
            rows * vocab * _F32_NBYTES,
        )

    @classmethod
    def planned_nbytes(
        cls,
        config: LagunaGGUFConfig,
        *,
        max_rows: int,
    ) -> int:
        """Return exact owned bytes before allocating the bounded row scratch."""

        return sum(cls._planned_sizes(config, max_rows=max_rows))

    @classmethod
    def allocate(
        cls,
        config: LagunaGGUFConfig,
        *,
        max_rows: int,
        runtime: HipRuntime | None = None,
    ) -> "LagunaRowsScratch":
        rows = int(max_rows)
        sizes = cls._planned_sizes(config, max_rows=rows)
        max_heads = max(int(value) for value in config.head_counts)
        max_query_width = max_heads * int(config.key_length)
        buffers: list[DeviceBuffer] = []
        try:
            buffers.extend(malloc(nbytes, runtime=runtime) for nbytes in sizes)
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            raise
        return cls(rows, max_query_width, max_heads, *buffers)

    @property
    def buffers(self) -> tuple[DeviceBuffer, ...]:
        return (
            self.token_ids,
            self.positions,
            self.hidden,
            self.norm,
            self.query,
            self.key,
            self.value,
            self.query_rotated,
            self.key_rotated,
            self.gate_logits,
            self.context,
            self.gated_context,
            self.attention_output,
            self.post_attention,
            self.dense_gate,
            self.dense_up,
            self.dense_intermediate,
            self.dense_output,
            self.final_norm,
            self.logits,
        )

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self.buffers)

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for buffer in reversed(self.buffers):
            free(buffer, runtime=runtime)


@dataclass(frozen=True)
class LagunaPrefillScratchPlan:
    """Exact bounded row-scratch bytes covered by UMA memory admission."""

    matrix_rows: int
    attention_rows: int
    rows_nbytes: int
    moe_nbytes: int

    @classmethod
    def build(
        cls,
        config: LagunaGGUFConfig,
        moe_plan: LagunaMoEKernelPlan,
        *,
        policy: LagunaPrefillChunkPolicy,
    ) -> "LagunaPrefillScratchPlan":
        rows_nbytes = LagunaRowsScratch.planned_nbytes(
            config,
            max_rows=policy.matrix_rows,
        )
        moe_nbytes = laguna_moe_scratch_nbytes(
            moe_plan,
            max_rows=policy.matrix_rows,
        )
        return cls(
            matrix_rows=policy.matrix_rows,
            attention_rows=policy.attention_rows,
            rows_nbytes=rows_nbytes,
            moe_nbytes=moe_nbytes,
        )

    @property
    def total_nbytes(self) -> int:
        return self.rows_nbytes + self.moe_nbytes


@dataclass
class LagunaVerifierScratch:
    """Stable target-verifier staging, argmax, and accept-summary storage.

    K/V rows remain outside the canonical cache until GPU acceptance selects a
    committed prefix.  One fixed allocation therefore makes reject/partial/full
    acceptance safe across SWA wrap without snapshotting overwritten ring rows.
    """

    max_rows: int
    layer_count: int
    kv_width: int
    argmax_blocks: int
    staged_keys: DeviceBuffer
    staged_values: DeviceBuffer
    argmax_block_values: DeviceBuffer
    argmax_block_indices: DeviceBuffer
    target_top1: DeviceBuffer
    target_top1_values: DeviceBuffer
    token_ids: DeviceBuffer
    positions: DeviceBuffer
    parent_rows: DeviceBuffer
    draft_depths: DeviceBuffer
    active_mask: DeviceBuffer
    remaining_decode: DeviceBuffer
    accepted_counts: DeviceBuffer
    commit_rows: DeviceBuffer
    commit_tokens: DeviceBuffer
    commit_positions: DeviceBuffer
    next_tokens: DeviceBuffer
    full_accept: DeviceBuffer
    committed_output_ids: DeviceBuffer
    committed_output_lengths: DeviceBuffer
    packed_payload: DeviceBuffer
    _closed: bool = False

    @classmethod
    def allocate(
        cls,
        config: LagunaGGUFConfig,
        *,
        max_rows: int,
        runtime: HipRuntime | None = None,
    ) -> "LagunaVerifierScratch":
        rows = int(max_rows)
        layers = int(config.block_count)
        kv_width = int(config.head_count_kv) * int(config.key_length)
        blocks = lm_head_argmax_stage1_blocks(int(config.vocab_size))
        if rows <= 0 or layers <= 0 or kv_width <= 0 or blocks <= 0:
            raise ValueError("Laguna verifier scratch dimensions must be positive")
        sizes = (
            layers * rows * kv_width * _F32_NBYTES,
            layers * rows * kv_width * _F32_NBYTES,
            rows * blocks * _F32_NBYTES,
            rows * blocks * _I32_NBYTES,
            rows * _I32_NBYTES,
            rows * _F32_NBYTES,
            rows * _I32_NBYTES,
            rows * _I32_NBYTES,
            rows * _I32_NBYTES,
            rows * _I32_NBYTES,
            rows * _U8_NBYTES,
            _I32_NBYTES,
            _I32_NBYTES,
            _I32_NBYTES,
            _I32_NBYTES,
            _I32_NBYTES,
            _I32_NBYTES,
            _U8_NBYTES,
            rows * _I32_NBYTES,
            _I32_NBYTES,
            7 * _I32_NBYTES,
        )
        buffers: list[DeviceBuffer] = []
        try:
            buffers.extend(malloc(nbytes, runtime=runtime) for nbytes in sizes)
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            raise
        return cls(rows, layers, kv_width, blocks, *buffers)

    @property
    def buffers(self) -> tuple[DeviceBuffer, ...]:
        return (
            self.staged_keys,
            self.staged_values,
            self.argmax_block_values,
            self.argmax_block_indices,
            self.target_top1,
            self.target_top1_values,
            self.token_ids,
            self.positions,
            self.parent_rows,
            self.draft_depths,
            self.active_mask,
            self.remaining_decode,
            self.accepted_counts,
            self.commit_rows,
            self.commit_tokens,
            self.commit_positions,
            self.next_tokens,
            self.full_accept,
            self.committed_output_ids,
            self.committed_output_lengths,
            self.packed_payload,
        )

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self.buffers)

    @property
    def layer_stride_nbytes(self) -> int:
        return self.max_rows * self.kv_width * _F32_NBYTES

    def key_ptr(self, layer_id: int) -> int:
        self._check_layer(layer_id)
        return self.staged_keys.ptr + int(layer_id) * self.layer_stride_nbytes

    def value_ptr(self, layer_id: int) -> int:
        self._check_layer(layer_id)
        return self.staged_values.ptr + int(layer_id) * self.layer_stride_nbytes

    def address_signature(self) -> dict[str, int]:
        names = (
            "staged_keys",
            "staged_values",
            "argmax_block_values",
            "argmax_block_indices",
            "target_top1",
            "target_top1_values",
            "token_ids",
            "positions",
            "parent_rows",
            "draft_depths",
            "active_mask",
            "remaining_decode",
            "accepted_counts",
            "commit_rows",
            "commit_tokens",
            "commit_positions",
            "next_tokens",
            "full_accept",
            "committed_output_ids",
            "committed_output_lengths",
            "packed_payload",
        )
        return {
            f"verifier.{name}": buffer.ptr
            for name, buffer in zip(names, self.buffers, strict=True)
        }

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for buffer in reversed(self.buffers):
            free(buffer, runtime=runtime)

    def _check_layer(self, layer_id: int) -> None:
        layer = int(layer_id)
        if layer < 0 or layer >= self.layer_count:
            raise IndexError(f"verifier layer {layer} outside [0, {self.layer_count})")


@dataclass(frozen=True)
class LagunaEagerLibraries:
    """Loaded JIT libraries, held once for the whole resident session."""

    embedding: object
    gguf_ops: object
    f16_projection: object
    f16_projection_prefill: object
    cast: object
    attention_gate: object
    kv_attention: object
    dense_silu: object
    argmax: object
    q4_linear: object
    q4_decode_linear: object
    q4_prefill_linear: object
    q6_linear: object
    q6_decode_linear: object
    q6_t16_linear: object
    q8_decode_linear: object
    router_logits: object
    router_select: object
    selected_experts: object
    selected_prefill: object
    iq_selected_experts: object
    moe_group: object
    routed_sum: object

    @property
    def embedding_libraries(self) -> Mapping[str, object]:
        return {
            quant: self.embedding
            for quant in ("gguf_q4_k", "gguf_q5_k", "gguf_q6_k", "gguf_q8_0")
        }

    @property
    def f16_linear(self) -> Mapping[str, object]:
        return {
            "fp16_weight": self.f16_projection,
            "fp16_weight:tiled_bf16_f32_out": self.f16_projection_prefill,
            "fp16_weight:tiled_bf16_bf16_out": self.f16_projection_prefill,
            "fp16_weight:wmma_comp_bf16_f32_out": self.f16_projection_prefill,
            "fp16_weight:wmma_comp_bf16_bf16_out": self.f16_projection_prefill,
        }

    @property
    def linear(self) -> Mapping[str, object]:
        return {
            "gguf_q4_k": self.q4_linear,
            "gguf_q4_k:pack8_gemv_decode_bf16_bf16_out": self.q4_decode_linear,
            "gguf_q4_k:pack8_gemv_decode_bf16_f32_out": self.q4_decode_linear,
            "gguf_q4_k:pack8_wmma_prefill_bf16_bf16_out": self.q4_prefill_linear,
            "gguf_q5_k": self.q6_linear,
            "gguf_q6_k": self.q6_linear,
            "gguf_q6_k:wmma_prefill_bf16_bf16_out": self.q4_prefill_linear,
            "gguf_q6_k:pack8_gemv_decode_bf16_bf16_out": self.q6_decode_linear,
            "gguf_q6_k:pack8_gemv_decode_bf16_f32_out": self.q6_decode_linear,
            "gguf_q8_0": self.q6_linear,
            "gguf_q8_0:pack8_gemv_decode_bf16_bf16_out": self.q8_decode_linear,
            "gguf_q8_0:pack8_gemv_decode_bf16_f32_out": self.q8_decode_linear,
            "gguf_q6_k_t16_v1": self.q6_t16_linear,
        }

    @property
    def moe(self) -> Mapping[str, object]:
        return {
            **self.linear,
            "router_logits": self.router_logits,
            "router_select": self.router_select,
            "selected_gate_up": self.selected_experts,
            "selected_gate_up_prefill": self.selected_prefill,
            "selected_gate_up_iq": self.iq_selected_experts,
            "selected_silu": self.dense_silu,
            "selected_down": self.selected_experts,
            "selected_down_iq": self.iq_selected_experts,
            "grouped_metadata": self.moe_group,
            "grouped_gather": self.moe_group,
            "grouped_down": self.selected_experts,
            "grouped_weighted_sum": self.routed_sum,
            "grouped_weighted_sum_shared_add": self.routed_sum,
            "routed_sum": self.routed_sum,
            "routed_sum_rows": self.router_select,
            "shared_silu": self.dense_silu,
            "add": self.gguf_ops,
        }


def _launch_laguna_f16_weight_linear(weight, *args, libraries, **kwargs) -> None:
    launch_f16_weight_linear(weight, *args, libraries=libraries.f16_linear, **kwargs)


def _launch_laguna_raw_weight_linear(weight, *args, libraries, **kwargs) -> None:
    rows = int(args[2])
    kwargs.pop("compensated_wmma_eligible", None)
    launch_gguf_linear(
        weight,
        *args,
        libraries=libraries.linear,
        use_wmma_prefill=False,
        use_gemv_decode=rows == 1,
        **kwargs,
    )


_LAGUNA_WEIGHT_LINEAR_LAUNCHERS = MappingProxyType(
    {
        LAYOUT_DENSE_F16: _launch_laguna_f16_weight_linear,
        LAYOUT_RAW_GGUF: _launch_laguna_raw_weight_linear,
    }
)


def launch_laguna_weight_linear(
    weight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    activation_dtype: str = "bf16",
    output_dtype: str = "bf16",
    backend: str | None = None,
    stream: int = 0,
    libraries: LagunaEagerLibraries,
    runtime: HipRuntime | None = None,
    compensated_wmma_eligible: bool = False,
) -> None:
    """Dispatch one Laguna projection from its validated resident layout."""

    try:
        launch = _LAGUNA_WEIGHT_LINEAR_LAUNCHERS[weight.spec.layout]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Laguna projection resident layout {weight.spec.layout!r}"
        ) from exc
    launch(
        weight,
        x_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        activation_dtype=activation_dtype,
        output_dtype=output_dtype,
        backend=backend,
        stream=stream,
        libraries=libraries,
        runtime=runtime,
        compensated_wmma_eligible=compensated_wmma_eligible,
    )


def _launch_laguna_f16_qkv(
    q_weight,
    k_weight,
    v_weight,
    x_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    rows,
    in_features,
    q_features,
    k_features,
    v_features,
    *,
    backend,
    stream,
    libraries,
    runtime,
    compensated_wmma_eligible,
) -> None:
    launch_f16_weight_linear_triple(
        q_weight,
        k_weight,
        v_weight,
        x_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        rows,
        in_features,
        q_features,
        k_features,
        v_features,
        backend=backend,
        stream=stream,
        libraries=libraries.f16_linear,
        runtime=runtime,
        compensated_wmma_eligible=compensated_wmma_eligible,
    )


def _launch_laguna_raw_qkv(
    q_weight,
    k_weight,
    v_weight,
    x_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    rows,
    in_features,
    q_features,
    k_features,
    v_features,
    *,
    backend,
    stream,
    libraries,
    runtime,
    compensated_wmma_eligible,
) -> None:
    for weight, out_ptr, out_features in (
        (q_weight, q_ptr, q_features),
        (k_weight, k_ptr, k_features),
        (v_weight, v_ptr, v_features),
    ):
        launch_laguna_weight_linear(
            weight,
            x_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            output_dtype="f32",
            backend=backend,
            stream=stream,
            libraries=libraries,
            runtime=runtime,
        )


_LAGUNA_QKV_LAUNCHERS = MappingProxyType(
    {
        (LAYOUT_DENSE_F16,) * 3: _launch_laguna_f16_qkv,
        (LAYOUT_RAW_GGUF,) * 3: _launch_laguna_raw_qkv,
    }
)


def launch_laguna_qkv(
    q_weight,
    k_weight,
    v_weight,
    x_ptr: int,
    q_ptr: int,
    k_ptr: int,
    v_ptr: int,
    rows: int,
    in_features: int,
    q_features: int,
    k_features: int,
    v_features: int,
    *,
    backend: str,
    stream: int,
    libraries: LagunaEagerLibraries,
    runtime: HipRuntime | None,
    compensated_wmma_eligible: bool = False,
) -> None:
    """Preserve the fused F16 QKV path and route raw quants independently."""

    layouts = tuple(weight.spec.layout for weight in (q_weight, k_weight, v_weight))
    try:
        launch = _LAGUNA_QKV_LAUNCHERS[layouts]
    except KeyError as exc:
        raise ValueError(f"unsupported Laguna QKV resident layouts {layouts}") from exc
    launch(
        q_weight,
        k_weight,
        v_weight,
        x_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        rows,
        in_features,
        q_features,
        k_features,
        v_features,
        backend=backend,
        stream=stream,
        libraries=libraries,
        runtime=runtime,
        compensated_wmma_eligible=compensated_wmma_eligible,
    )


def launch_laguna_attention_projections(
    q_weight,
    k_weight,
    v_weight,
    gate_weight,
    x_ptr: int,
    q_ptr: int,
    k_ptr: int,
    v_ptr: int,
    gate_ptr: int,
    rows: int,
    in_features: int,
    q_features: int,
    k_features: int,
    v_features: int,
    gate_features: int,
    *,
    backend: str,
    stream: int,
    libraries: LagunaEagerLibraries,
    runtime: HipRuntime | None,
    compensated_wmma_eligible: bool = False,
) -> bool:
    """Launch exact attention projections and report both raw pairs fused.

    Registered query/gate and K/V pairs are decode-only and fail closed.
    Rows greater than one, registry/shape/quant misses, and unmeasured layouts
    retain the established fused-QKV or singleton fallbacks.
    """

    q_gate_fused = False
    if (
        q_weight.spec.layout == LAYOUT_RAW_GGUF
        and gate_weight.spec.layout == LAYOUT_RAW_GGUF
    ):
        q_gate_fused = launch_gguf_linear_pair(
            q_weight,
            gate_weight,
            x_ptr,
            q_ptr,
            gate_ptr,
            rows,
            in_features,
            q_features,
            out_features_b=gate_features,
            output_dtype=GGUF_OUTPUT_F32,
            backend=backend,
            stream=stream,
            libraries=libraries.linear,
            runtime=runtime,
            use_wmma_prefill=False,
            use_gemv_decode=rows == 1,
            registered_decode_only=True,
        )

    kv_fused = False
    if (
        k_weight.spec.layout == LAYOUT_RAW_GGUF
        and v_weight.spec.layout == LAYOUT_RAW_GGUF
    ):
        kv_fused = launch_gguf_linear_pair(
            k_weight,
            v_weight,
            x_ptr,
            k_ptr,
            v_ptr,
            rows,
            in_features,
            k_features,
            out_features_b=v_features,
            output_dtype=GGUF_OUTPUT_F32,
            backend=backend,
            stream=stream,
            libraries=libraries.linear,
            runtime=runtime,
            use_wmma_prefill=False,
            use_gemv_decode=rows == 1,
            registered_decode_only=True,
        )

    if not q_gate_fused and not kv_fused:
        launch_laguna_qkv(
            q_weight,
            k_weight,
            v_weight,
            x_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            rows,
            in_features,
            q_features,
            k_features,
            v_features,
            backend=backend,
            stream=stream,
            libraries=libraries,
            runtime=runtime,
            compensated_wmma_eligible=compensated_wmma_eligible,
        )
        launch_laguna_weight_linear(
            gate_weight,
            x_ptr,
            gate_ptr,
            rows,
            in_features,
            gate_features,
            output_dtype=GGUF_OUTPUT_F32,
            backend=backend,
            stream=stream,
            libraries=libraries,
            runtime=runtime,
            compensated_wmma_eligible=compensated_wmma_eligible,
        )
        return False

    if not q_gate_fused:
        for weight, out_ptr, out_features in (
            (q_weight, q_ptr, q_features),
            (gate_weight, gate_ptr, gate_features),
        ):
            launch_laguna_weight_linear(
                weight,
                x_ptr,
                out_ptr,
                rows,
                in_features,
                out_features,
                output_dtype=GGUF_OUTPUT_F32,
                backend=backend,
                stream=stream,
                libraries=libraries,
                runtime=runtime,
                compensated_wmma_eligible=compensated_wmma_eligible,
            )
    if not kv_fused:
        for weight, out_ptr, out_features in (
            (k_weight, k_ptr, k_features),
            (v_weight, v_ptr, v_features),
        ):
            launch_laguna_weight_linear(
                weight,
                x_ptr,
                out_ptr,
                rows,
                in_features,
                out_features,
                output_dtype=GGUF_OUTPUT_F32,
                backend=backend,
                stream=stream,
                libraries=libraries,
                runtime=runtime,
                compensated_wmma_eligible=compensated_wmma_eligible,
            )
    return q_gate_fused and kv_fused


@dataclass(frozen=True)
class LagunaEagerTokenResult:
    """One eager token result; device buffers remain owned by the session."""

    position: int
    input_token_id: int
    next_token_id: int
    next_token_logit: float
    logits: DeviceBuffer
    final_hidden: DeviceBuffer
    post_layer_hidden: DeviceBuffer


@dataclass(frozen=True)
class LagunaVerifierRowsResult:
    """Borrowed B+1 target rows with full logits and stable hidden taps."""

    start_position: int
    input_token_ids: tuple[int, ...]
    logits: DeviceBuffer
    final_hidden: DeviceBuffer
    post_layer_hidden: DeviceBuffer
    logits_row_stride: int

    @property
    def rows(self) -> int:
        return len(self.input_token_ids)


@dataclass(frozen=True)
class LagunaPrefillRoutingReplay:
    """Host-owned selected-expert lanes from one diagnostic prefill pass."""

    result: LagunaEagerTokenResult
    rows: int
    expert_count: int
    top_k: int
    selected_experts: Mapping[int, tuple[int, ...]]


@dataclass(frozen=True)
class LagunaDFlashVerifyResult:
    """One GPU-accepted target chain after transactional prefix commit."""

    rows_result: LagunaVerifierRowsResult
    target_top1_ids: tuple[int, ...]
    target_top1_values: tuple[float, ...]
    accepted_draft_count: int
    accepted_token_ids: tuple[int, ...]
    commit_row: int
    commit_token_id: int
    commit_position: int
    next_token_id: int | None
    full_accept: bool
    committed_input_ids: tuple[int, ...]
    visible_output_ids: tuple[int, ...]
    packed_payload: tuple[int, ...]

    @property
    def committed_rows(self) -> int:
        return len(self.committed_input_ids)


def resolve_laguna_eager_kernel_plan(
    config: LagunaGGUFConfig,
    *,
    backend: str,
) -> LagunaEagerKernelPlan:
    """Validate the S 2.1 eager contract and resolve only exact registry keys."""

    if not str(backend).startswith("hip_"):
        raise ValueError("Laguna eager execution requires a concrete HIP backend")
    if config.block_count != 48:
        raise ValueError("Laguna S 2.1 eager execution requires exactly 48 layers")
    if config.hidden_size != 3_072 or config.vocab_size != 100_352:
        raise ValueError("Laguna S 2.1 eager execution requires hidden=3072 and vocab=100352")
    if config.head_count_kv != 8 or config.key_length != 128 or config.value_length != 128:
        raise ValueError("Laguna S 2.1 eager execution requires 8 KV heads of dimension 128")
    if tuple(config.head_counts) != _EXPECTED_HEAD_COUNTS:
        raise ValueError("Laguna S 2.1 eager query-head sequence must repeat 48/72/72/72")
    if tuple(config.layer_types) != _EXPECTED_LAYER_TYPES:
        raise ValueError("Laguna S 2.1 eager layer sequence must repeat global/SWA/SWA/SWA")
    if config.sliding_window != 512:
        raise ValueError("Laguna S 2.1 eager execution requires a 512-token SWA ring")
    if config.leading_dense_block_count != 1:
        raise ValueError("Laguna S 2.1 eager execution requires one leading dense layer")

    load_backend_kernel_package(backend)
    keys = {
        "rmsnorm": KernelKey(backend, "rmsnorm", "gguf_f32_weight", "bf16_out"),
        "rmsnorm_fp16_via_bf16": KernelKey(
            backend,
            "rmsnorm",
            "gguf_f32_weight",
            "fp16_via_bf16_out",
        ),
        "add_rmsnorm": KernelKey(backend, "add_rmsnorm", "gguf_f32_weight", "bf16_out"),
        "add": KernelKey(backend, "elementwise", "bf16", "add"),
        "attention_gate": KernelKey(
            backend, "attention_gate", "f32", "softplus_broadcast_bf16_out"
        ),
        "attention_gate_fp16_via_bf16": KernelKey(
            backend,
            "attention_gate",
            "f32",
            "softplus_broadcast_fp16_via_bf16_out",
        ),
        "dense_silu": KernelKey(backend, "silu_mul_separate", "bf16", "out"),
        "argmax": KernelKey(backend, "argmax", "f32", "top1_i64"),
        "f16_triple": KernelKey(backend, "linear_triple", "fp16_weight", "bf16_f32_out"),
        "f16_f32": KernelKey(backend, "linear", "fp16_weight", "bf16_f32_out"),
        "f16_bf16": KernelKey(backend, "linear", "fp16_weight", "bf16_bf16_out"),
        "rope": KernelKey(
            backend,
            "head_rmsnorm+partial_rotary",
            "laguna_f32_weight",
            "positions_f32",
        ),
    }
    functions = {name: _resolve_exact(key) for name, key in keys.items()}
    return LagunaEagerKernelPlan(
        backend=backend,
        rmsnorm_key=keys["rmsnorm"],
        rmsnorm_fp16_via_bf16_key=keys["rmsnorm_fp16_via_bf16"],
        add_rmsnorm_key=keys["add_rmsnorm"],
        add_key=keys["add"],
        attention_gate_key=keys["attention_gate"],
        attention_gate_fp16_via_bf16_key=keys[
            "attention_gate_fp16_via_bf16"
        ],
        dense_silu_key=keys["dense_silu"],
        argmax_key=keys["argmax"],
        f16_triple_key=keys["f16_triple"],
        f16_f32_key=keys["f16_f32"],
        f16_bf16_key=keys["f16_bf16"],
        rope_key=keys["rope"],
        rmsnorm=functions["rmsnorm"],
        rmsnorm_fp16_via_bf16=functions["rmsnorm_fp16_via_bf16"],
        add_rmsnorm=functions["add_rmsnorm"],
        add=functions["add"],
        attention_gate=functions["attention_gate"],
        attention_gate_fp16_via_bf16=functions[
            "attention_gate_fp16_via_bf16"
        ],
        dense_silu=functions["dense_silu"],
        argmax=functions["argmax"],
    )


def capture_laguna_hidden_tap(
    source_bf16_ptr: int,
    *,
    depth: int,
    targets: LagunaHiddenCaptureTargets | None,
    hidden_size: int,
    runtime: HipRuntime,
    stream: int = 0,
) -> None:
    """Copy one requested tap; the ``None`` path performs no runtime call."""

    if targets is None:
        return
    if targets.rows != 1:
        raise ValueError("single-row Laguna hidden capture requires rows=1 targets")
    if int(targets.hidden_size) != int(hidden_size):
        raise ValueError("Laguna hidden capture hidden_size does not match the session")
    target = targets.buffers.get(int(depth))
    if target is None:
        return
    runtime.memcpy_async(
        target.ptr,
        int(source_bf16_ptr),
        int(hidden_size) * _BF16_NBYTES,
        HipMemcpyKind.DEVICE_TO_DEVICE,
        int(stream),
    )


def capture_laguna_hidden_rows(
    source_bf16_ptr: int,
    *,
    depth: int,
    rows: int,
    targets: LagunaHiddenCaptureTargets | None,
    hidden_size: int,
    runtime: HipRuntime,
    stream: int = 0,
) -> None:
    """Copy one requested row-batched tap into caller-owned storage."""

    if targets is None:
        return
    parsed_rows = int(rows)
    if parsed_rows <= 0 or targets.rows != parsed_rows:
        raise ValueError("Laguna hidden capture rows must match the target row count")
    if int(targets.hidden_size) != int(hidden_size):
        raise ValueError("Laguna hidden capture hidden_size does not match the session")
    target = targets.buffers.get(int(depth))
    if target is None:
        return
    runtime.memcpy_async(
        target.ptr,
        int(source_bf16_ptr),
        parsed_rows * int(hidden_size) * _BF16_NBYTES,
        HipMemcpyKind.DEVICE_TO_DEVICE,
        int(stream),
    )


def capture_laguna_routing_rows(
    selected_experts_ptr: int,
    *,
    layer_id: int,
    leading_dense_layers: int,
    sparse_layers: int,
    rows: int,
    top_k: int,
    capture: DeviceBuffer,
    runtime: HipRuntime,
    stream: int = 0,
) -> None:
    """Copy one sparse layer's selected IDs into a bounded diagnostic plane."""

    parsed_rows = int(rows)
    parsed_top_k = int(top_k)
    dense_layers = int(leading_dense_layers)
    sparse_count = int(sparse_layers)
    layer = int(layer_id)
    if parsed_rows <= 0 or parsed_top_k <= 0 or sparse_count <= 0:
        raise ValueError("Laguna routing replay dimensions must be positive")
    sparse_index = layer - dense_layers
    if sparse_index < 0 or sparse_index >= sparse_count:
        raise ValueError("Laguna routing replay layer is outside the sparse layer range")
    layer_nbytes = parsed_rows * parsed_top_k * _I64_NBYTES
    expected_nbytes = sparse_count * layer_nbytes
    if capture.nbytes != expected_nbytes:
        raise ValueError(
            "Laguna routing replay capture buffer must exactly match all sparse planes: "
            f"expected={expected_nbytes} actual={capture.nbytes}"
        )
    runtime.memcpy_async(
        capture.ptr + sparse_index * layer_nbytes,
        int(selected_experts_ptr),
        layer_nbytes,
        HipMemcpyKind.DEVICE_TO_DEVICE,
        int(stream),
    )


def load_laguna_eager_libraries(
    *,
    backend: str,
    compiler_version: str | None = None,
    require_cached: bool = False,
) -> LagunaEagerLibraries:
    """Build/load every library used by one eager session exactly once."""

    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
    )
    from hipengine.kernels.hip_gfx1100.convert.cast import build_cast
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops
    from hipengine.kernels.hip_gfx1100.fused.laguna_attention import (
        build_laguna_attention,
    )
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import build_paro_combine
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import build_paro_silu
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        build_laguna_f16_projection,
        build_laguna_f16_projection_prefill,
    )
    from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        build_qwen35_moe_group_scatter,
    )
    from hipengine.kernels.hip_gfx1100.moe.laguna_router import build_laguna_router
    from hipengine.kernels.hip_gfx1100.moe.router import build_qwen35_router
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import build_gguf_iq_gemv
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_pack8_gemv import (
        build_gguf_q4_k_pack8_gemv,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
        build_gguf_q4_k_prefill,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        build_gguf_q4_k_q8_1_selected_prefill,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
        build_gguf_q6_k_embedding,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
        build_gguf_q6_k_pack8_gemv,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv import (
        build_gguf_q8_0_pack8_gemv,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
    )
    kwargs = {
        "compiler_version": compiler_version,
        "require_cached": require_cached,
        "load": True,
    }
    target_arch = hip_target_arch_for_backend(backend)
    with hip_target_arch_environment(target_arch):
        return LagunaEagerLibraries(
            embedding=build_gguf_q6_k_embedding(**kwargs),
            gguf_ops=build_gguf_ops(**kwargs),
            f16_projection=build_laguna_f16_projection(**kwargs),
            f16_projection_prefill=build_laguna_f16_projection_prefill(**kwargs),
            cast=build_cast(**kwargs),
            attention_gate=build_laguna_attention(**kwargs),
            kv_attention=build_laguna_kv_attention(**kwargs),
            dense_silu=build_paro_silu(**kwargs),
            argmax=build_lm_head(**kwargs),
            q4_linear=build_gguf_q4_k_gemv(**kwargs),
            q4_decode_linear=build_gguf_q4_k_pack8_gemv(**kwargs),
            q4_prefill_linear=build_gguf_q4_k_prefill(**kwargs),
            q6_linear=build_gguf_k_gemv(**kwargs),
            q6_decode_linear=build_gguf_q6_k_pack8_gemv(**kwargs),
            q6_t16_linear=build_gguf_q6_k_t16_gemv(**kwargs),
            q8_decode_linear=build_gguf_q8_0_pack8_gemv(**kwargs),
            router_logits=build_qwen35_router(**kwargs),
            router_select=build_laguna_router(**kwargs),
            selected_experts=build_gguf_t16_selected_gemv(**kwargs),
            selected_prefill=build_gguf_q4_k_q8_1_selected_prefill(**kwargs),
            iq_selected_experts=build_gguf_iq_gemv(**kwargs),
            moe_group=build_qwen35_moe_group_scatter(**kwargs),
            routed_sum=build_paro_combine(**kwargs),
        )


class LagunaGGUFResidentSession:
    """All-resident eager Laguna S 2.1 c=1 session with BF16 KV state."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        resident_weights: LagunaGGUFResidentWeights | None = None,
        context_length: int = _INITIAL_MAX_CONTEXT,
        backend: str = "hip_gfx1151",
        runtime: HipRuntime | None = None,
        device: Device | None = None,
        compiler_version: str | None = None,
        require_cached_build: bool = False,
        available_bytes: int | None = None,
        safety_reserve_nbytes: int = DEFAULT_LAGUNA_SAFETY_RESERVE_BYTES,
        progress: Callable | None = None,
        repacked_cache: LagunaGGUFRepackedCache | str | Path | None = None,
        model_sha256: str | None = None,
        prefill_chunk_size: int | None = None,
        prefill_attention_chunk_size: int | None = None,
        global_prefill_variant: str | None = None,
        swa_decode_variant: str | None = None,
        swa_prefill_variant: str | None = None,
        prefill_kv_preappend: bool | None = None,
        prefill_cached_meta: bool | None = None,
        prefill_global_qrow6: bool | None = None,
        prefill_dense_initial: bool | None = None,
        q6_qmicro: bool | None = None,
        q6_compact_activation: bool | None = None,
        q6_half_row_activation: bool | None = None,
        q6_skip_padded_activation: bool | None = None,
        moe_branch_concurrency: bool | None = None,
        moe_shared_after_router: bool = False,
    ) -> None:
        self.runtime = runtime or get_hip_runtime()
        self.device = device or Device("hip", 0)
        self.backend = resolve_backend(backend)
        self.context_length = int(context_length)
        matrix_rows = prefill_chunk_size
        if matrix_rows is None:
            matrix_rows = min(
                self.context_length,
                int(
                    backend_package_capability(
                        self.backend,
                        "LAGUNA_PREFILL_MATRIX_ROWS",
                        128,
                    )
                    or 128
                ),
            )
        self.prefill_chunk_policy = LagunaPrefillChunkPolicy.resolve(
            context_length=self.context_length,
            matrix_rows=matrix_rows,
            attention_rows=prefill_attention_chunk_size,
        )
        self.prefill_chunk_size = self.prefill_chunk_policy.matrix_rows
        self.prefill_attention_chunk_size = self.prefill_chunk_policy.attention_rows
        self.global_prefill_variant = resolve_laguna_global_prefill_variant(
            self.backend,
            global_prefill_variant,
        )
        self.swa_decode_variant = resolve_laguna_swa_decode_variant(
            self.backend,
            swa_decode_variant,
        )
        self.swa_prefill_variant = resolve_laguna_swa_prefill_variant(
            self.backend,
            swa_prefill_variant,
        )
        self.prefill_kv_preappend = bool(
            backend_package_capability(
                self.backend,
                "LAGUNA_PREFILL_KV_PREAPPEND",
                False,
            )
            if prefill_kv_preappend is None
            else prefill_kv_preappend
        )
        self.prefill_cached_meta = bool(
            backend_package_capability(
                self.backend,
                "LAGUNA_PREFILL_CACHED_META",
                False,
            )
            if prefill_cached_meta is None
            else prefill_cached_meta
        )
        self.prefill_global_qrow6 = bool(
            backend_package_capability(
                self.backend,
                "LAGUNA_PREFILL_GLOBAL_QROW6",
                False,
            )
            if prefill_global_qrow6 is None
            else prefill_global_qrow6
        )
        self.prefill_dense_initial = bool(
            backend_package_capability(
                self.backend,
                "LAGUNA_PREFILL_DENSE_INITIAL",
                False,
            )
            if prefill_dense_initial is None
            else prefill_dense_initial
        )
        resident_q6_qmicro = (
            getattr(resident_weights, "q6_qmicro", None)
            if resident_weights is not None
            else None
        )
        self.q6_qmicro = bool(
            resident_q6_qmicro
            if q6_qmicro is None and resident_q6_qmicro is not None
            else backend_package_capability(
                self.backend,
                "LAGUNA_Q6_QMICRO",
                False,
            )
            if q6_qmicro is None
            else q6_qmicro
        )
        self.q6_compact_activation = bool(
            backend_package_capability(
                self.backend,
                "LAGUNA_Q6_COMPACT_ACTIVATION",
                False,
            )
            if q6_compact_activation is None
            else q6_compact_activation
        )
        self.q6_half_row_activation = bool(
            self.q6_compact_activation
            and backend_package_capability(
                self.backend,
                "LAGUNA_Q6_HALF_ROW_ACTIVATION",
                False,
            )
            if q6_half_row_activation is None
            else q6_half_row_activation
        )
        if self.q6_half_row_activation and not self.q6_compact_activation:
            raise ValueError(
                "Q6 half-row activation staging requires compact activation"
            )
        self.q6_skip_padded_activation = bool(
            self.q6_half_row_activation
            and backend_package_capability(
                self.backend,
                "LAGUNA_Q6_SKIP_PADDED_ACTIVATION",
                False,
            )
            if q6_skip_padded_activation is None
            else q6_skip_padded_activation
        )
        if (
            self.q6_skip_padded_activation
            and not self.q6_half_row_activation
        ):
            raise ValueError(
                "Q6 skipping padded activation rows requires half-row staging"
            )
        self.selected_down_mode = resolve_laguna_selected_down_mode(self.backend)
        self.selected_gate_up_mode = resolve_laguna_selected_gate_up_mode(self.backend)
        self.fuse_selected_silu_pack = bool(
            backend_package_capability(
                self.backend,
                "LAGUNA_FUSED_SELECTED_SILU_PACK",
                False,
            )
        )
        self.router_logits_mode = resolve_laguna_router_logits_mode(self.backend)
        self.dense_q4_prefill_mode = resolve_laguna_dense_q4_prefill_mode(self.backend)
        self.group_compact_mode = resolve_laguna_group_compact_mode(self.backend)
        self.f16_prefill_mode = resolve_laguna_f16_prefill_mode(self.backend)
        self.fuse_f16_boundaries = bool(
            backend_package_capability(
                self.backend,
                "LAGUNA_F16_BOUNDARY_FUSION",
                False,
            )
        )
        self.moe_branch_concurrency = resolve_laguna_moe_branch_concurrency(
            self.backend,
            moe_branch_concurrency,
        )
        self.moe_shared_after_router = bool(moe_shared_after_router)
        self.position = -1
        self.last_result: LagunaEagerTokenResult | None = None
        self.weights: LagunaGGUFResidentWeights | None = None
        self.kv_cache: LagunaKVCache | None = None
        self.scratch: LagunaEagerScratch | None = None
        self.moe_scratch: LagunaMoEScratch | None = None
        self.rows_scratch: LagunaRowsScratch | None = None
        self.rows_moe_scratch: LagunaMoEScratch | None = None
        self.verifier_scratch: LagunaVerifierScratch | None = None
        self.full_rope: LagunaDeviceRoPETables | None = None
        self.swa_rope: LagunaDeviceRoPETables | None = None
        self.libraries: LagunaEagerLibraries | None = None
        self.kernel_plan: LagunaEagerKernelPlan | None = None
        self.moe_plan: LagunaMoEKernelPlan | None = None
        self.prefill_scratch_plan: LagunaPrefillScratchPlan | None = None
        self.f16_hipblaslt: LagunaF16HipblasLt | None = None
        self.prefill_scratch_admission_nbytes = DEFAULT_LAGUNA_SCRATCH_BYTES
        self._owns_weights = resident_weights is None
        self._closed = False
        self._compiler_version = compiler_version
        self._require_cached_build = bool(require_cached_build)
        self._dflash_accept_library = None
        self._staged_verifier_tokens: tuple[int, ...] | None = None
        self._moe_shared_stream = 0
        self._moe_shared_input_ready_event = 0
        self._moe_shared_output_ready_event = 0

        if self.context_length <= 0 or self.context_length > _INITIAL_MAX_CONTEXT:
            raise ValueError(
                f"initial Laguna eager context_length must be within [1, {_INITIAL_MAX_CONTEXT}]"
            )
        if resident_weights is not None and (
            repacked_cache is not None or model_sha256 is not None
        ):
            raise ValueError(
                "repacked_cache/model_sha256 apply only when the session owns model loading"
            )
        try:
            if resident_weights is None:
                if model_path is None:
                    raise ValueError("model_path is required without resident_weights")
                reader = GGUFReader(model_path)
                config = laguna_gguf_config_from_metadata(reader.info)
            else:
                config = resident_weights.config
                if resident_weights.backend != self.backend:
                    raise ValueError("resident Laguna backend does not match the session backend")

            self.kernel_plan = resolve_laguna_eager_kernel_plan(
                config,
                backend=self.backend,
            )
            self.libraries = load_laguna_eager_libraries(
                backend=self.backend,
                compiler_version=compiler_version,
                require_cached=require_cached_build,
            )
            self.moe_plan = resolve_laguna_moe_plan(
                config,
                backend=self.backend,
                q6_qmicro=self.q6_qmicro,
                q6_compact_activation=self.q6_compact_activation,
                q6_half_row_activation=self.q6_half_row_activation,
                q6_skip_padded_activation=(
                    self.q6_skip_padded_activation
                ),
            )
            self.prefill_scratch_plan = LagunaPrefillScratchPlan.build(
                config,
                self.moe_plan,
                policy=self.prefill_chunk_policy,
            )
            self.prefill_scratch_admission_nbytes = max(
                DEFAULT_LAGUNA_SCRATCH_BYTES,
                self.prefill_scratch_plan.total_nbytes,
            )
            if resident_weights is None:
                self.weights = materialize_laguna_gguf_weights(
                    reader,
                    context_length=self.context_length,
                    available_bytes=available_bytes,
                    scratch_nbytes=self.prefill_scratch_admission_nbytes,
                    safety_reserve_nbytes=safety_reserve_nbytes,
                    device=self.device,
                    runtime=self.runtime,
                    backend=self.backend,
                    progress=progress,
                    repacked_cache=repacked_cache,
                    repacked_cache_source_sha256=model_sha256,
                    q6_qmicro=self.q6_qmicro,
                )
            else:
                self.weights = resident_weights
            self._validate_resident_weights()
            if self.f16_prefill_mode == "hipblaslt_norm_direct":
                self.set_f16_prefill_mode(self.f16_prefill_mode)
            self.full_rope = materialize_laguna_rope_tables(
                self.context_length,
                config.full_rope,
                device=self.device,
                runtime=self.runtime,
            )
            if config.swa_rope is None:
                raise ValueError("Laguna eager session requires SWA RoPE metadata")
            self.swa_rope = materialize_laguna_rope_tables(
                self.context_length,
                config.swa_rope,
                device=self.device,
                runtime=self.runtime,
            )
            self.kv_cache = allocate_laguna_kv_cache(
                config,
                context_length=self.context_length,
                backend=self.backend,
                device=self.device,
                runtime=self.runtime,
                global_prefill_variant=self.global_prefill_variant,
                swa_decode_variant=self.swa_decode_variant,
                swa_prefill_variant=self.swa_prefill_variant,
                prefill_cached_meta=self.prefill_cached_meta,
                prefill_global_qrow6=self.prefill_global_qrow6,
                prefill_dense_initial=self.prefill_dense_initial,
            )
            self.scratch = LagunaEagerScratch.allocate(config, runtime=self.runtime)
            self.moe_scratch = allocate_laguna_moe_scratch(
                self.moe_plan,
                runtime=self.runtime,
            )
            self.rows_scratch = LagunaRowsScratch.allocate(
                config,
                max_rows=self.prefill_chunk_size,
                runtime=self.runtime,
            )
            self.rows_moe_scratch = allocate_laguna_moe_scratch(
                self.moe_plan,
                max_rows=self.prefill_chunk_size,
                runtime=self.runtime,
            )
            if self.moe_branch_concurrency:
                self._moe_shared_stream = self.runtime.stream_create(
                    nonblocking=True
                )
                self._moe_shared_input_ready_event = (
                    self.runtime.event_create(flags=0x2)
                )
                self._moe_shared_output_ready_event = (
                    self.runtime.event_create(flags=0x2)
                )
        except BaseException:
            self._close(suppress_errors=True)
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def config(self) -> LagunaGGUFConfig:
        self._check_open()
        assert self.weights is not None
        return self.weights.config

    def set_selected_down_mode(self, mode: str) -> None:
        """Select the explicit diagnostic sparse-down route for later row runs."""

        self.selected_down_mode = resolve_laguna_selected_down_mode(
            self.backend,
            mode,
        )

    def set_selected_gate_up_mode(self, mode: str) -> None:
        """Select the explicit compact selected gate/up prefill candidate."""

        self.selected_gate_up_mode = resolve_laguna_selected_gate_up_mode(
            self.backend,
            mode,
        )

    def set_fused_selected_silu_pack(self, enabled: bool) -> None:
        """Select exact dual-SiLU packing or its registered primitive fallback."""

        self.fuse_selected_silu_pack = bool(enabled)

    def set_moe_shared_after_router(self, enabled: bool) -> None:
        """Move concurrent shared work behind the exact router prefix."""

        self.moe_shared_after_router = bool(enabled)

    def set_router_logits_mode(self, mode: str) -> None:
        """Select the exact rows>1 router-logit token-reuse schedule."""

        self.router_logits_mode = resolve_laguna_router_logits_mode(
            self.backend,
            mode,
        )

    def set_f16_prefill_mode(self, mode: str) -> None:
        """Select the explicit rows>1 source-F16 projection route."""

        selected = resolve_laguna_f16_prefill_mode(self.backend, mode)
        if selected != "retained":
            assert self.weights is not None
            unsupported = [
                (layer.layer_id, slot, layer.weight(slot).spec.layout)
                for layer in self.weights.layers
                for slot in ("attn_q", "attn_k", "attn_v", "attn_gate", "attn_output")
                if layer.weight(slot).spec.layout != LAYOUT_DENSE_F16
            ]
            if unsupported:
                raise ValueError(
                    "Laguna hipBLASLt prefill requires dense F16 attention weights; "
                    f"first unsupported={unsupported[0]}"
                )
        if selected in {"hipblaslt_norm_direct", "hipblaslt_range_direct"}:
            assert self.weights is not None
            bound = laguna_attention_norm_fp16_bound(
                self.weights.config.hidden_size,
                (
                    getattr(layer.weight("attn_norm"), "source_abs_max", None)
                    for layer in self.weights.layers
                ),
            )
            if bound * 2.0 > 65_504.0:
                raise ValueError(
                    "Laguna direct norm cast requires a 2x-safe finite FP16 bound; "
                    f"bound={bound}"
                )
        if selected == "hipblaslt_range_direct":
            assert self.weights is not None
            bound = laguna_attention_gated_fp16_bound(
                self.weights.config.hidden_size,
                (
                    (
                        getattr(layer.weight("attn_norm"), "source_abs_max", None),
                        getattr(layer.weight("attn_v"), "source_row_l2_max", None),
                        getattr(layer.weight("attn_gate"), "source_row_l2_max", None),
                    )
                    for layer in self.weights.layers
                ),
            )
            if bound * 2.0 > 65_504.0:
                raise ValueError(
                    "Laguna direct attention-output cast requires a 2x-safe "
                    f"finite FP16 bound; bound={bound}"
                )
        self.f16_prefill_mode = selected

    def set_f16_boundary_fusion(self, enabled: bool) -> None:
        """Fuse exact BF16-to-FP16 boundaries in range-direct prefill."""

        self.fuse_f16_boundaries = bool(enabled)

    def set_dense_q4_prefill_mode(self, mode: str) -> None:
        """Select the explicit dense/shared Q4 rows>1 projection route."""

        self.dense_q4_prefill_mode = resolve_laguna_dense_q4_prefill_mode(
            self.backend,
            mode,
        )

    def set_group_compact_mode(self, mode: str) -> None:
        """Select stable serial or parallel MoE metadata compaction."""

        self.group_compact_mode = resolve_laguna_group_compact_mode(
            self.backend,
            mode,
        )

    @property
    def resident_nbytes(self) -> int:
        self._check_open()
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.scratch is not None
        assert self.moe_scratch is not None
        assert self.rows_scratch is not None
        assert self.rows_moe_scratch is not None
        assert self.full_rope is not None
        assert self.swa_rope is not None
        return (
            self.weights.resident_nbytes
            + self.kv_cache.resident_nbytes
            + self.scratch.nbytes
            + self.moe_scratch.nbytes
            + self.rows_scratch.nbytes
            + self.rows_moe_scratch.nbytes
            + (self.verifier_scratch.nbytes if self.verifier_scratch is not None else 0)
            + self.full_rope.cos.buffer.nbytes
            + self.full_rope.sin.buffer.nbytes
            + self.swa_rope.cos.buffer.nbytes
            + self.swa_rope.sin.buffer.nbytes
        )

    def forward_token(
        self,
        token_id: int,
        *,
        captures: LagunaHiddenCaptureTargets | None = None,
        stream: int = 0,
    ) -> LagunaEagerTokenResult:
        """Append and execute one token, then return the borrowed top-1 result."""

        self._check_open()
        self._check_no_staged_verifier()
        config = self.config
        token = int(token_id)
        if token < 0 or token >= config.vocab_size:
            raise ValueError(f"token_id must be within [0, {config.vocab_size})")
        if captures is not None and captures.hidden_size != config.hidden_size:
            raise ValueError("Laguna hidden capture hidden_size does not match the session")
        next_position = self.position + 1
        if next_position >= self.context_length:
            raise ValueError("Laguna eager session exhausted its admitted context")

        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        try:
            _copy_i64(self.scratch.token_id, token, self.runtime)
            _copy_i64(self.scratch.position, next_position, self.runtime)
            self.kv_cache.prepare_position(next_position)
            launch_gguf_embedding(
                self.weights.root("token_embedding"),
                self.scratch.token_id.ptr,
                self.scratch.hidden.ptr,
                1,
                config.hidden_size,
                config.vocab_size,
                backend=self.backend,
                stream=stream,
                libraries=self.libraries.embedding_libraries,
                runtime=self.runtime,
            )
            for layer_id in range(config.block_count):
                self._run_layer(layer_id, stream=stream)
                capture_laguna_hidden_tap(
                    self.scratch.hidden.ptr,
                    depth=layer_id + 1,
                    targets=captures,
                    hidden_size=config.hidden_size,
                    runtime=self.runtime,
                    stream=stream,
                )
            result = self._project_and_sample(
                input_token_id=token,
                position=next_position,
                stream=stream,
            )
            self.position = next_position
            self.last_result = result
            return result
        except BaseException:
            self._close(suppress_errors=True)
            raise

    def prefill(
        self,
        token_ids: Sequence[int],
        *,
        capture_last: LagunaHiddenCaptureTargets | None = None,
        use_bulk: bool = True,
        stream: int = 0,
    ) -> LagunaEagerTokenResult:
        """Chunked exact prefill with an explicit token-serial fallback."""

        self._check_open()
        self._check_no_staged_verifier()
        tokens = tuple(int(value) for value in token_ids)
        if not tokens:
            raise ValueError("Laguna eager prefill requires at least one token")
        if capture_last is not None and capture_last.rows != 1:
            raise ValueError("Laguna prefill capture_last requires one-row targets")
        if not use_bulk or len(tokens) == 1:
            result: LagunaEagerTokenResult | None = None
            for index, token in enumerate(tokens):
                result = self.forward_token(
                    token,
                    captures=capture_last if index == len(tokens) - 1 else None,
                    stream=stream,
                )
            assert result is not None
            return result

        last_chunk: tuple[int, ...] = ()
        for start in range(0, len(tokens), self.prefill_chunk_size):
            chunk = tokens[start : start + self.prefill_chunk_size]
            self._execute_rows(
                chunk,
                capture_last=capture_last if start + len(chunk) == len(tokens) else None,
                stream=stream,
            )
            last_chunk = chunk
        assert last_chunk
        return self._project_rows_last(
            input_token_id=last_chunk[-1],
            position=self.position,
            row_index=len(last_chunk) - 1,
            stream=stream,
        )

    def prefill_routing_replay(
        self,
        token_ids: Sequence[int],
        *,
        stream: int = 0,
    ) -> LagunaPrefillRoutingReplay:
        """Run one physical prefill chunk and return host-owned expert selections.

        This is a diagnostic-only LPF replay surface. Normal generation allocates
        no capture buffer and performs no extra copies.
        """

        self._check_open()
        self._check_no_staged_verifier()
        tokens = tuple(int(value) for value in token_ids)
        rows = len(tokens)
        if rows <= 0 or rows > self.prefill_chunk_size:
            raise ValueError(
                f"Laguna routing replay rows must be within [1, {self.prefill_chunk_size}]"
            )
        config = self.config
        sparse_layers = config.block_count - config.leading_dense_block_count
        layer_lanes = rows * config.expert_used_count
        total_lanes = sparse_layers * layer_lanes
        capture = malloc(total_lanes * _I64_NBYTES, runtime=self.runtime)
        try:
            self._execute_rows(tokens, routing_capture=capture, stream=stream)
            result = self._project_rows_last(
                input_token_id=tokens[-1],
                position=self.position,
                row_index=rows - 1,
                stream=stream,
            )
            host = (ctypes.c_int64 * total_lanes)()
            self.runtime.memcpy(
                ctypes.addressof(host),
                capture.ptr,
                capture.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            selected = {
                layer_id: tuple(
                    int(host[(layer_id - config.leading_dense_block_count) * layer_lanes + lane])
                    for lane in range(layer_lanes)
                )
                for layer_id in range(config.leading_dense_block_count, config.block_count)
            }
            return LagunaPrefillRoutingReplay(
                result=result,
                rows=rows,
                expert_count=config.expert_count,
                top_k=config.expert_used_count,
                selected_experts=MappingProxyType(selected),
            )
        finally:
            free(capture, runtime=self.runtime)

    def verify_rows(
        self,
        root_token_id: int,
        draft_token_ids: Sequence[int],
        *,
        captures: LagunaHiddenCaptureTargets | None = None,
        stream: int = 0,
    ) -> LagunaVerifierRowsResult:
        """Execute one committed B+1 target block and expose every logits/tap row."""

        self._check_open()
        self._check_no_staged_verifier()
        tokens = (int(root_token_id), *(int(value) for value in draft_token_ids))
        if len(tokens) > self.prefill_chunk_size:
            raise ValueError(
                f"Laguna verifier rows exceed prefill capacity {self.prefill_chunk_size}"
            )
        if captures is not None and captures.rows != len(tokens):
            raise ValueError("Laguna verifier captures must provide exactly B+1 rows")
        start_position = self.position + 1
        self._execute_rows(tokens, capture_rows=captures, stream=stream)
        return self._project_rows_all(
            input_token_ids=tokens,
            start_position=start_position,
            stream=stream,
        )

    def verify_dflash_chain(
        self,
        root_token_id: int,
        draft_token_ids: Sequence[int],
        *,
        captures: LagunaHiddenCaptureTargets | None = None,
        remaining_decode: int | None = None,
        stream: int = 0,
    ) -> LagunaDFlashVerifyResult:
        """Verify, GPU-accept, and transactionally commit one linear DFlash chain.

        Root plus draft rows execute once against canonical prior K/V, but their
        per-layer K/V rows are staged separately. Only ``root + accepted`` rows
        are appended after the provider-neutral accept kernel returns its compact
        seven-int payload; rejected suffix rows therefore cannot corrupt SWA
        slots even across positions 511/512/513.
        """

        self._check_open()
        self._check_no_staged_verifier()
        drafts = tuple(int(value) for value in draft_token_ids)
        if not drafts:
            raise ValueError("Laguna DFlash verification requires at least one draft token")
        tokens = (int(root_token_id), *drafts)
        rows = len(tokens)
        if rows > self.prefill_chunk_size:
            raise ValueError(
                f"Laguna DFlash verifier rows exceed capacity {self.prefill_chunk_size}"
            )
        if remaining_decode is not None:
            if isinstance(remaining_decode, bool) or not isinstance(remaining_decode, int):
                raise TypeError("remaining_decode must be an integer or None")
            if remaining_decode < 0:
                raise ValueError("remaining_decode must be non-negative")
        if captures is not None and captures.rows != rows:
            raise ValueError("Laguna DFlash captures must provide exactly B+1 rows")

        self._ensure_verifier_resources()
        start_position = self.position + 1
        positions = tuple(range(start_position, start_position + rows))
        self._execute_rows(
            tokens,
            capture_rows=captures,
            stage_verifier_kv=True,
            stream=stream,
        )
        try:
            rows_result = self._project_rows_all(
                input_token_ids=tokens,
                start_position=start_position,
                stream=stream,
            )
            assert self.verifier_scratch is not None
            assert self.libraries is not None
            verifier = self.verifier_scratch
            from hipengine.kernels.hip_gfx1100.linear.lm_head import argmax_f32_rows_i32
            from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import (
                dflash_accept_chain_i32_packed,
            )

            argmax_f32_rows_i32(
                rows_result.logits.ptr,
                verifier.argmax_block_values.ptr,
                verifier.argmax_block_indices.ptr,
                verifier.target_top1.ptr,
                verifier.target_top1_values.ptr,
                rows,
                self.config.vocab_size,
                stream=stream,
                library=self.libraries.argmax,
                runtime=self.runtime,
            )
            _copy_i32_rows(verifier.token_ids, tokens, self.runtime)
            _copy_i32_rows(verifier.positions, positions, self.runtime)
            _copy_i32_rows(verifier.parent_rows, (-1, *range(rows - 1)), self.runtime)
            _copy_i32_rows(verifier.draft_depths, tuple(range(rows)), self.runtime)
            _copy_u8_rows(verifier.active_mask, (1,) * rows, self.runtime)
            remaining_ptr: int | None = None
            if remaining_decode is not None:
                _copy_i32_rows(verifier.remaining_decode, (remaining_decode,), self.runtime)
                remaining_ptr = verifier.remaining_decode.ptr
            dflash_accept_chain_i32_packed(
                verifier.token_ids.ptr,
                verifier.positions.ptr,
                verifier.parent_rows.ptr,
                verifier.draft_depths.ptr,
                verifier.active_mask.ptr,
                verifier.target_top1.ptr,
                remaining_ptr,
                verifier.accepted_counts.ptr,
                verifier.commit_rows.ptr,
                verifier.commit_tokens.ptr,
                verifier.commit_positions.ptr,
                verifier.next_tokens.ptr,
                verifier.full_accept.ptr,
                verifier.committed_output_ids.ptr,
                verifier.committed_output_lengths.ptr,
                verifier.packed_payload.ptr,
                rows,
                1,
                rows,
                stream=stream,
                library=self._dflash_accept_library,
                runtime=self.runtime,
            )
            if stream:
                self.runtime.stream_synchronize(stream)
            else:
                self.runtime.device_synchronize()
            target_top1 = _read_i32_rows(verifier.target_top1, rows, self.runtime)
            target_values = _read_f32_rows(verifier.target_top1_values, rows, self.runtime)
            payload = _read_i32_rows(verifier.packed_payload, 7, self.runtime)
            accepted, commit_row, commit_token, commit_position, next_raw, full_raw, _ = payload
            _validate_laguna_dflash_accept_payload(
                tokens=tokens,
                positions=positions,
                target_top1=target_top1,
                remaining_decode=remaining_decode,
                payload=payload,
            )
            self._commit_staged_verifier_rows(accepted + 1, stream=stream)
        except BaseException:
            if not self.closed and self._staged_verifier_tokens is not None:
                self._discard_staged_verifier_rows()
            raise

        next_token = None if next_raw < 0 else next_raw
        accepted_tokens = drafts[:accepted]
        committed_inputs = tokens[: accepted + 1]
        visible = (*accepted_tokens, *((next_token,) if next_token is not None else ()))
        hidden_nbytes = self.config.hidden_size * _BF16_NBYTES
        commit_hidden_offset = commit_row * hidden_nbytes
        logits_row_nbytes = self.config.vocab_size * _F32_NBYTES
        self.last_result = LagunaEagerTokenResult(
            position=commit_position,
            input_token_id=commit_token,
            next_token_id=target_top1[commit_row],
            next_token_logit=target_values[commit_row],
            logits=_buffer_view(
                rows_result.logits,
                commit_row * logits_row_nbytes,
                logits_row_nbytes,
            ),
            final_hidden=_buffer_view(
                rows_result.final_hidden,
                commit_hidden_offset,
                hidden_nbytes,
            ),
            post_layer_hidden=_buffer_view(
                rows_result.post_layer_hidden,
                commit_hidden_offset,
                hidden_nbytes,
            ),
        )
        return LagunaDFlashVerifyResult(
            rows_result=rows_result,
            target_top1_ids=target_top1,
            target_top1_values=target_values,
            accepted_draft_count=accepted,
            accepted_token_ids=accepted_tokens,
            commit_row=commit_row,
            commit_token_id=commit_token,
            commit_position=commit_position,
            next_token_id=next_token,
            full_accept=bool(full_raw),
            committed_input_ids=committed_inputs,
            visible_output_ids=visible,
            packed_payload=payload,
        )

    def reset_state(self) -> None:
        """Reset request state while retaining weights and stable scratch."""

        self._check_open()
        self._check_no_staged_verifier()
        assert self.kv_cache is not None
        self.kv_cache.reset()
        self.position = -1
        self.last_result = None

    def verifier_address_signature(self) -> dict[str, int]:
        """Return stable semantic pointers used by verifier graph buckets."""

        self._check_open()
        self._ensure_verifier_resources()
        assert self.rows_scratch is not None
        assert self.verifier_scratch is not None
        signature = self.verifier_scratch.address_signature()
        for index, buffer in enumerate(self.rows_scratch.buffers):
            signature[f"rows_scratch.{index}"] = buffer.ptr
        return signature

    def generate_greedy(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        stop_token_ids: Sequence[int] = (),
        stream: int = 0,
    ) -> tuple[int, ...]:
        """Diagnostic target-only greedy loop over the resident eager state."""

        count = int(max_new_tokens)
        if count <= 0:
            raise ValueError("max_new_tokens must be positive")
        result = self.prefill(prompt_token_ids, stream=stream)
        stops = {int(value) for value in stop_token_ids}
        generated: list[int] = []
        for _ in range(count):
            token = int(result.next_token_id)
            generated.append(token)
            if token in stops:
                break
            if len(generated) == count:
                break
            result = self.forward_token(token, stream=stream)
        return tuple(generated)

    def close(self) -> None:
        self._close(suppress_errors=False)

    def __enter__(self) -> "LagunaGGUFResidentSession":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _execute_rows(
        self,
        token_ids: Sequence[int],
        *,
        capture_last: LagunaHiddenCaptureTargets | None = None,
        capture_rows: LagunaHiddenCaptureTargets | None = None,
        stage_verifier_kv: bool = False,
        routing_capture: DeviceBuffer | None = None,
        stream: int,
    ) -> None:
        self._check_open()
        self._check_no_staged_verifier()
        if capture_last is not None and capture_rows is not None:
            raise ValueError("Laguna row execution accepts capture_last or capture_rows, not both")
        tokens = tuple(int(token) for token in token_ids)
        rows = len(tokens)
        if rows <= 0 or rows > self.prefill_chunk_size:
            raise ValueError(
                f"Laguna row count must be within [1, {self.prefill_chunk_size}]"
            )
        config = self.config
        if any(token < 0 or token >= config.vocab_size for token in tokens):
            raise ValueError(f"token IDs must be within [0, {config.vocab_size})")
        start_position = self.position + 1
        end_position = start_position + rows - 1
        if end_position >= self.context_length:
            raise ValueError("Laguna eager session exhausted its admitted context")
        if capture_last is not None and capture_last.hidden_size != config.hidden_size:
            raise ValueError("Laguna hidden capture hidden_size does not match the session")
        if capture_rows is not None and (
            capture_rows.hidden_size != config.hidden_size or capture_rows.rows != rows
        ):
            raise ValueError("Laguna row captures must match hidden_size and row count")

        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.rows_scratch is not None
        assert self.libraries is not None
        scratch = self.rows_scratch
        positions = tuple(range(start_position, end_position + 1))
        try:
            _copy_i64_rows(scratch.token_ids, tokens, self.runtime)
            _copy_i64_rows(scratch.positions, positions, self.runtime)
            self.kv_cache.prepare_rows(positions)
            launch_gguf_embedding(
                self.weights.root("token_embedding"),
                scratch.token_ids.ptr,
                scratch.hidden.ptr,
                rows,
                config.hidden_size,
                config.vocab_size,
                backend=self.backend,
                stream=stream,
                libraries=self.libraries.embedding_libraries,
                runtime=self.runtime,
            )
            for layer_id in range(config.block_count):
                self._run_layer_rows(
                    layer_id,
                    rows=rows,
                    stage_verifier_kv=stage_verifier_kv,
                    routing_capture=routing_capture,
                    stream=stream,
                )
                depth = layer_id + 1
                capture_laguna_hidden_rows(
                    scratch.hidden.ptr,
                    depth=depth,
                    rows=rows,
                    targets=capture_rows,
                    hidden_size=config.hidden_size,
                    runtime=self.runtime,
                    stream=stream,
                )
                capture_laguna_hidden_tap(
                    scratch.hidden.ptr + (rows - 1) * config.hidden_size * _BF16_NBYTES,
                    depth=depth,
                    targets=capture_last,
                    hidden_size=config.hidden_size,
                    runtime=self.runtime,
                    stream=stream,
                )
            if stage_verifier_kv:
                self._staged_verifier_tokens = tokens
            else:
                self.kv_cache.commit_rows()
                self.position = end_position
        except BaseException:
            self._close(suppress_errors=True)
            raise

    def _commit_staged_verifier_rows(self, rows: int, *, stream: int) -> None:
        """Append only the accepted verifier prefix into canonical target K/V."""

        self._check_open()
        tokens = self._staged_verifier_tokens
        if tokens is None:
            raise RuntimeError("no Laguna verifier transaction is staged")
        commit_rows = int(rows)
        if commit_rows <= 0 or commit_rows > len(tokens):
            raise ValueError("committed verifier rows must be a non-empty staged prefix")
        assert self.kv_cache is not None
        assert self.verifier_scratch is not None
        pending = self.kv_cache.pending_positions
        if len(pending) != len(tokens):
            raise RuntimeError("staged verifier token and KV position counts diverged")
        commit_positions = pending[:commit_rows]
        try:
            self.kv_cache.discard_rows()
            self.kv_cache.prepare_rows(commit_positions)
            for layer_id in range(self.config.block_count):
                self.kv_cache.append_rows(
                    layer_id,
                    self.verifier_scratch.key_ptr(layer_id),
                    self.verifier_scratch.value_ptr(layer_id),
                    commit_rows,
                    stream=stream,
                    library=self.libraries.kv_attention if self.libraries is not None else None,
                )
            self.kv_cache.commit_rows()
            self.position = commit_positions[-1]
            self._staged_verifier_tokens = None
        except BaseException:
            self._close(suppress_errors=True)
            raise

    def _discard_staged_verifier_rows(self) -> None:
        """Cancel a pre-commit verifier transaction without touching canonical K/V."""

        self._check_open()
        if self._staged_verifier_tokens is None:
            raise RuntimeError("no Laguna verifier transaction is staged")
        assert self.kv_cache is not None
        if self.kv_cache.pending_positions:
            self.kv_cache.discard_rows()
        self._staged_verifier_tokens = None

    def _ensure_f16_hipblaslt(self) -> LagunaF16HipblasLt:
        route = self.f16_hipblaslt
        if route is None:
            route = LagunaF16HipblasLt()
            self.f16_hipblaslt = route
        return route

    def _launch_attention_projections_rows(
        self,
        layer: LagunaGGUFResidentLayerWeights,
        scratch: LagunaRowsScratch,
        *,
        rows: int,
        q_width: int,
        kv_width: int,
        heads: int,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.libraries is not None
        config = self.weights.config
        if self.f16_prefill_mode == "retained":
            launch_laguna_attention_projections(
                layer.weight("attn_q"),
                layer.weight("attn_k"),
                layer.weight("attn_v"),
                layer.weight("attn_gate"),
                scratch.norm.ptr,
                scratch.query.ptr,
                scratch.key.ptr,
                scratch.value.ptr,
                scratch.gate_logits.ptr,
                rows,
                config.hidden_size,
                q_width,
                kv_width,
                kv_width,
                heads,
                backend=self.backend,
                stream=stream,
                libraries=self.libraries,
                runtime=self.runtime,
                compensated_wmma_eligible=layer.attention_type == SLIDING_ATTENTION,
            )
            return

        route = self._ensure_f16_hipblaslt()
        direct_norm = self.f16_prefill_mode in {
            "hipblaslt_norm_direct",
            "hipblaslt_range_direct",
        }
        row_scales_ptr = scratch.token_ids.ptr
        if direct_norm:
            if (
                self.f16_prefill_mode != "hipblaslt_range_direct"
                or not self.fuse_f16_boundaries
            ):
                bf16_to_fp16(
                    scratch.norm.ptr,
                    scratch.gated_context.ptr,
                    rows * config.hidden_size,
                    stream=stream,
                    library=self.libraries.cast,
                    runtime=self.runtime,
                )
        else:
            bf16_to_fp16_scaled_rows(
                scratch.norm.ptr,
                scratch.gated_context.ptr,
                row_scales_ptr,
                rows,
                config.hidden_size,
                stream=stream,
                library=self.libraries.cast,
                runtime=self.runtime,
            )
        for slot, out_ptr, out_features in (
            ("attn_q", scratch.query.ptr, q_width),
            ("attn_k", scratch.key.ptr, kv_width),
            ("attn_v", scratch.value.ptr, kv_width),
            ("attn_gate", scratch.gate_logits.ptr, heads),
        ):
            route.launch(
                scratch.gated_context.ptr,
                layer.weight(slot).allocation("raw").tensor.ptr,
                out_ptr,
                rows,
                config.hidden_size,
                out_features,
                stream=stream,
            )
            if not direct_norm:
                f32_scale_rows(
                    out_ptr,
                    row_scales_ptr,
                    rows,
                    out_features,
                    stream=stream,
                    library=self.libraries.cast,
                    runtime=self.runtime,
                )

    def _launch_attention_output_rows(
        self,
        layer: LagunaGGUFResidentLayerWeights,
        scratch: LagunaRowsScratch,
        *,
        rows: int,
        q_width: int,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.libraries is not None
        config = self.weights.config
        if self.f16_prefill_mode == "retained":
            launch_laguna_weight_linear(
                layer.weight("attn_output"),
                scratch.gated_context.ptr,
                scratch.attention_output.ptr,
                rows,
                q_width,
                config.hidden_size,
                backend=self.backend,
                stream=stream,
                libraries=self.libraries,
                runtime=self.runtime,
                compensated_wmma_eligible=layer.attention_type == SLIDING_ATTENTION,
            )
            return

        route = self._ensure_f16_hipblaslt()
        direct_output = self.f16_prefill_mode == "hipblaslt_range_direct"
        row_scales_ptr = scratch.token_ids.ptr
        if direct_output and not self.fuse_f16_boundaries:
            bf16_to_fp16(
                scratch.gated_context.ptr,
                scratch.gated_context.ptr,
                rows * q_width,
                stream=stream,
                library=self.libraries.cast,
                runtime=self.runtime,
            )
        elif not direct_output:
            bf16_to_fp16_scaled_rows(
                scratch.gated_context.ptr,
                scratch.gated_context.ptr,
                row_scales_ptr,
                rows,
                q_width,
                stream=stream,
                library=self.libraries.cast,
                runtime=self.runtime,
            )
        route.launch(
            scratch.gated_context.ptr,
            layer.weight("attn_output").allocation("raw").tensor.ptr,
            scratch.context.ptr,
            rows,
            q_width,
            config.hidden_size,
            stream=stream,
        )
        if direct_output:
            f32_to_bf16(
                scratch.context.ptr,
                scratch.attention_output.ptr,
                rows * config.hidden_size,
                stream=stream,
                library=self.libraries.cast,
                runtime=self.runtime,
            )
        else:
            f32_scale_rows_to_bf16(
                scratch.context.ptr,
                row_scales_ptr,
                scratch.attention_output.ptr,
                rows,
                config.hidden_size,
                stream=stream,
                library=self.libraries.cast,
                runtime=self.runtime,
            )

    def _run_layer_rows(
        self,
        layer_id: int,
        *,
        rows: int,
        stage_verifier_kv: bool = False,
        routing_capture: DeviceBuffer | None = None,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.rows_scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        assert self.full_rope is not None
        assert self.swa_rope is not None
        config = self.weights.config
        layer = self.weights.layer(layer_id)
        scratch = self.rows_scratch
        heads = config.head_count(layer_id)
        q_width = heads * config.key_length
        kv_width = config.head_count_kv * config.key_length

        if (
            self.f16_prefill_mode == "hipblaslt_range_direct"
            and self.fuse_f16_boundaries
        ):
            self.kernel_plan.rmsnorm_fp16_via_bf16(
                scratch.hidden.ptr,
                layer.weight("attn_norm").allocation("raw").tensor.ptr,
                scratch.gated_context.ptr,
                rows,
                config.hidden_size,
                config.rms_norm_eps,
                stream=stream,
                library=self.libraries.gguf_ops,
                runtime=self.runtime,
            )
        else:
            self.kernel_plan.rmsnorm(
                scratch.hidden.ptr,
                layer.weight("attn_norm").allocation("raw").tensor.ptr,
                scratch.norm.ptr,
                rows,
                config.hidden_size,
                config.rms_norm_eps,
                stream=stream,
                library=self.libraries.gguf_ops,
                runtime=self.runtime,
            )
        self._launch_attention_projections_rows(
            layer,
            scratch,
            rows=rows,
            q_width=q_width,
            kv_width=kv_width,
            heads=heads,
            stream=stream,
        )
        rope = self.full_rope if layer.attention_type == FULL_ATTENTION else self.swa_rope
        launch_laguna_head_rmsnorm_rope(
            scratch.query.ptr,
            scratch.key.ptr,
            layer.weight("attn_q_norm").allocation("raw").tensor.ptr,
            layer.weight("attn_k_norm").allocation("raw").tensor.ptr,
            scratch.positions.ptr,
            scratch.query_rotated.ptr,
            scratch.key_rotated.ptr,
            config.rms_norm_eps,
            rows,
            heads,
            config.head_count_kv,
            config.key_length,
            rope,
            backend=self.backend,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        attention_ranges = self.prefill_chunk_policy.attention_ranges(rows)
        if stage_verifier_kv and len(attention_ranges) != 1:
            raise ValueError("Laguna staged verifier rows must fit one attention tile")
        for row_offset, attention_rows in attention_ranges:
            q_offset = row_offset * q_width * _F32_NBYTES
            kv_offset = row_offset * kv_width * _F32_NBYTES
            position_offset = row_offset * _I64_NBYTES
            sliced = row_offset != 0 or attention_rows != rows
            slice_kwargs = (
                {
                    "row_offset": row_offset,
                    "row_positions_ptr": scratch.positions.ptr + position_offset,
                }
                if sliced
                else {}
            )
            preappend = (
                self.prefill_kv_preappend
                and not stage_verifier_kv
                and self.kv_cache.can_preappend_prefill(
                    layer_id,
                    attention_rows,
                    row_offset=row_offset,
                )
            )
            if preappend:
                self.kv_cache.append_rows(
                    layer_id,
                    scratch.key_rotated.ptr + kv_offset,
                    scratch.value.ptr + kv_offset,
                    attention_rows,
                    stream=stream,
                    library=self.libraries.kv_attention,
                    **slice_kwargs,
                )
                self.kv_cache.attend_prefill_cached(
                    layer_id,
                    scratch.query_rotated.ptr + q_offset,
                    scratch.key_rotated.ptr + kv_offset,
                    scratch.value.ptr + kv_offset,
                    scratch.context.ptr + q_offset,
                    attention_rows,
                    stream=stream,
                    library=self.libraries.kv_attention,
                    **slice_kwargs,
                )
            else:
                self.kv_cache.attend_prefill(
                    layer_id,
                    scratch.query_rotated.ptr + q_offset,
                    scratch.key_rotated.ptr + kv_offset,
                    scratch.value.ptr + kv_offset,
                    scratch.context.ptr + q_offset,
                    attention_rows,
                    stream=stream,
                    library=self.libraries.kv_attention,
                    **slice_kwargs,
                )
                if not stage_verifier_kv:
                    self.kv_cache.append_rows(
                        layer_id,
                        scratch.key_rotated.ptr + kv_offset,
                        scratch.value.ptr + kv_offset,
                        attention_rows,
                        stream=stream,
                        library=self.libraries.kv_attention,
                        **slice_kwargs,
                    )
        if stage_verifier_kv:
            assert self.verifier_scratch is not None
            row_nbytes = rows * kv_width * _F32_NBYTES
            self.runtime.memcpy_async(
                self.verifier_scratch.key_ptr(layer_id),
                scratch.key_rotated.ptr,
                row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
            self.runtime.memcpy_async(
                self.verifier_scratch.value_ptr(layer_id),
                scratch.value.ptr,
                row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
        attention_gate = (
            self.kernel_plan.attention_gate_fp16_via_bf16
            if (
                self.f16_prefill_mode == "hipblaslt_range_direct"
                and self.fuse_f16_boundaries
            )
            else self.kernel_plan.attention_gate
        )
        attention_gate(
            scratch.context.ptr,
            scratch.gate_logits.ptr,
            scratch.gated_context.ptr,
            rows,
            heads,
            config.value_length,
            stream=stream,
            library=self.libraries.attention_gate,
            runtime=self.runtime,
        )
        self._launch_attention_output_rows(
            layer,
            scratch,
            rows=rows,
            q_width=q_width,
            stream=stream,
        )
        self.kernel_plan.add_rmsnorm(
            scratch.hidden.ptr,
            scratch.attention_output.ptr,
            layer.weight("ffn_norm").allocation("raw").tensor.ptr,
            scratch.norm.ptr,
            scratch.post_attention.ptr,
            rows,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        if layer.mlp_type == DENSE_MLP:
            self._run_dense_ffn_rows(layer, rows=rows, stream=stream)
        elif layer.mlp_type == SPARSE_MOE:
            self._run_sparse_ffn_rows(layer, rows=rows, stream=stream)
            if routing_capture is not None:
                assert self.rows_moe_scratch is not None
                capture_laguna_routing_rows(
                    self.rows_moe_scratch.selected_experts.ptr,
                    layer_id=layer_id,
                    leading_dense_layers=config.leading_dense_block_count,
                    sparse_layers=config.block_count - config.leading_dense_block_count,
                    rows=rows,
                    top_k=config.expert_used_count,
                    capture=routing_capture,
                    runtime=self.runtime,
                    stream=stream,
                )
        else:
            raise ValueError(f"unsupported Laguna MLP type {layer.mlp_type!r}")

    def _run_dense_ffn_rows(
        self,
        layer: LagunaGGUFResidentLayerWeights,
        *,
        rows: int,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.rows_scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        config = self.weights.config
        scratch = self.rows_scratch
        linear_libraries = self.libraries.linear
        for slot, output in (("ffn_gate", scratch.dense_gate), ("ffn_up", scratch.dense_up)):
            launch_gguf_linear(
                layer.weight(slot),
                scratch.norm.ptr,
                output.ptr,
                rows,
                config.hidden_size,
                config.feed_forward_length,
                backend=self.backend,
                stream=stream,
                libraries=linear_libraries,
                runtime=self.runtime,
                use_wmma_prefill=False,
                use_gemv_decode=rows == 1,
                use_q4_pack8_wmma=self.dense_q4_prefill_mode == "wmma_pack8",
            )
        self.kernel_plan.dense_silu(
            scratch.dense_gate.ptr,
            scratch.dense_up.ptr,
            scratch.dense_intermediate.ptr,
            rows,
            config.feed_forward_length,
            stream=stream,
            library=self.libraries.dense_silu,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            layer.weight("ffn_down"),
            scratch.dense_intermediate.ptr,
            scratch.dense_output.ptr,
            rows,
            config.feed_forward_length,
            config.hidden_size,
            backend=self.backend,
            stream=stream,
            libraries=linear_libraries,
            runtime=self.runtime,
            use_wmma_prefill=False,
            use_gemv_decode=rows == 1,
            use_q4_pack8_wmma=self.dense_q4_prefill_mode == "wmma_pack8",
        )
        self.kernel_plan.add(
            scratch.post_attention.ptr,
            scratch.dense_output.ptr,
            scratch.hidden.ptr,
            rows * config.hidden_size,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )

    def _run_sparse_ffn_rows(
        self,
        layer: LagunaGGUFResidentLayerWeights,
        *,
        rows: int,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.rows_scratch is not None
        assert self.rows_moe_scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        output = run_laguna_moe_rows(
            self.rows_scratch.norm.ptr,
            layer,
            self.rows_moe_scratch,
            rows=rows,
            selected_down_mode=self.selected_down_mode,
            selected_gate_up_mode=self.selected_gate_up_mode,
            router_logits_mode=self.router_logits_mode,
            dense_q4_prefill_mode=self.dense_q4_prefill_mode,
            group_compact_mode=self.group_compact_mode,
            fuse_selected_silu_pack=self.fuse_selected_silu_pack,
            shared_after_router=self.moe_shared_after_router,
            shared_stream=(
                self._moe_shared_stream
                if self.moe_branch_concurrency and rows > 1
                else 0
            ),
            shared_input_ready_event=(
                self._moe_shared_input_ready_event
                if self.moe_branch_concurrency and rows > 1
                else 0
            ),
            shared_output_ready_event=(
                self._moe_shared_output_ready_event
                if self.moe_branch_concurrency and rows > 1
                else 0
            ),
            stream=stream,
            runtime=self.runtime,
            libraries=self.libraries.moe,
        )
        self.kernel_plan.add(
            self.rows_scratch.post_attention.ptr,
            output.ptr,
            self.rows_scratch.hidden.ptr,
            rows * self.weights.config.hidden_size,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )

    def _project_rows_last(
        self,
        *,
        input_token_id: int,
        position: int,
        row_index: int,
        stream: int,
    ) -> LagunaEagerTokenResult:
        assert self.weights is not None
        assert self.scratch is not None
        assert self.rows_scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        config = self.weights.config
        scratch = self.rows_scratch
        hidden_nbytes = config.hidden_size * _BF16_NBYTES
        hidden_ptr = scratch.hidden.ptr + int(row_index) * hidden_nbytes
        self.kernel_plan.rmsnorm(
            hidden_ptr,
            self.weights.root("output_norm").allocation("raw").tensor.ptr,
            scratch.final_norm.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.weights.root("lm_head"),
            scratch.final_norm.ptr,
            scratch.logits.ptr,
            1,
            config.hidden_size,
            config.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            backend=self.backend,
            stream=stream,
            libraries=self.libraries.linear,
            runtime=self.runtime,
            use_wmma_prefill=False,
            use_gemv_decode=True,
        )
        self.kernel_plan.argmax(
            scratch.logits.ptr,
            self.scratch.argmax_block_values.ptr,
            self.scratch.argmax_block_indices.ptr,
            self.scratch.argmax_id.ptr,
            self.scratch.argmax_value.ptr,
            config.vocab_size,
            stream=stream,
            library=self.libraries.argmax,
            runtime=self.runtime,
        )
        if stream:
            self.runtime.stream_synchronize(stream)
        else:
            self.runtime.device_synchronize()
        result = LagunaEagerTokenResult(
            position=int(position),
            input_token_id=int(input_token_id),
            next_token_id=_read_i64(self.scratch.argmax_id, self.runtime),
            next_token_logit=_read_f32(self.scratch.argmax_value, self.runtime),
            logits=_buffer_view(scratch.logits, 0, config.vocab_size * _F32_NBYTES),
            final_hidden=_buffer_view(scratch.final_norm, 0, hidden_nbytes),
            post_layer_hidden=_buffer_view(scratch.hidden, int(row_index) * hidden_nbytes, hidden_nbytes),
        )
        self.last_result = result
        return result

    def _project_rows_all(
        self,
        *,
        input_token_ids: tuple[int, ...],
        start_position: int,
        stream: int,
    ) -> LagunaVerifierRowsResult:
        assert self.weights is not None
        assert self.rows_scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        config = self.weights.config
        scratch = self.rows_scratch
        rows = len(input_token_ids)
        self.kernel_plan.rmsnorm(
            scratch.hidden.ptr,
            self.weights.root("output_norm").allocation("raw").tensor.ptr,
            scratch.final_norm.ptr,
            rows,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.weights.root("lm_head"),
            scratch.final_norm.ptr,
            scratch.logits.ptr,
            rows,
            config.hidden_size,
            config.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            backend=self.backend,
            stream=stream,
            libraries=self.libraries.linear,
            runtime=self.runtime,
            use_wmma_prefill=False,
            use_gemv_decode=rows == 1,
        )
        if stream:
            self.runtime.stream_synchronize(stream)
        else:
            self.runtime.device_synchronize()
        hidden_nbytes = rows * config.hidden_size * _BF16_NBYTES
        return LagunaVerifierRowsResult(
            start_position=int(start_position),
            input_token_ids=input_token_ids,
            logits=_buffer_view(
                scratch.logits,
                0,
                rows * config.vocab_size * _F32_NBYTES,
            ),
            final_hidden=_buffer_view(scratch.final_norm, 0, hidden_nbytes),
            post_layer_hidden=_buffer_view(scratch.hidden, 0, hidden_nbytes),
            logits_row_stride=config.vocab_size,
        )

    def _run_layer(self, layer_id: int, *, stream: int) -> None:
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        assert self.full_rope is not None
        assert self.swa_rope is not None
        config = self.weights.config
        layer = self.weights.layer(layer_id)
        scratch = self.scratch
        heads = config.head_count(layer_id)
        q_width = heads * config.key_length
        kv_width = config.head_count_kv * config.key_length

        self.kernel_plan.rmsnorm(
            scratch.hidden.ptr,
            layer.weight("attn_norm").allocation("raw").tensor.ptr,
            scratch.norm.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        launch_laguna_attention_projections(
            layer.weight("attn_q"),
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            layer.weight("attn_gate"),
            scratch.norm.ptr,
            scratch.query.ptr,
            scratch.key.ptr,
            scratch.value.ptr,
            scratch.gate_logits.ptr,
            1,
            config.hidden_size,
            q_width,
            kv_width,
            kv_width,
            heads,
            backend=self.backend,
            stream=stream,
            libraries=self.libraries,
            runtime=self.runtime,
        )
        rope = self.full_rope if layer.attention_type == FULL_ATTENTION else self.swa_rope
        launch_laguna_head_rmsnorm_rope(
            scratch.query.ptr,
            scratch.key.ptr,
            layer.weight("attn_q_norm").allocation("raw").tensor.ptr,
            layer.weight("attn_k_norm").allocation("raw").tensor.ptr,
            scratch.position.ptr,
            scratch.query_rotated.ptr,
            scratch.key_rotated.ptr,
            config.rms_norm_eps,
            1,
            heads,
            config.head_count_kv,
            config.key_length,
            rope,
            backend=self.backend,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        self.kv_cache.append(
            layer_id,
            scratch.key_rotated.ptr,
            scratch.value.ptr,
            stream=stream,
            library=self.libraries.kv_attention,
        )
        self.kv_cache.attend(
            layer_id,
            scratch.query_rotated.ptr,
            scratch.context.ptr,
            stream=stream,
            library=self.libraries.kv_attention,
        )
        self.kernel_plan.attention_gate(
            scratch.context.ptr,
            scratch.gate_logits.ptr,
            scratch.gated_context.ptr,
            1,
            heads,
            config.value_length,
            stream=stream,
            library=self.libraries.attention_gate,
            runtime=self.runtime,
        )
        launch_laguna_weight_linear(
            layer.weight("attn_output"),
            scratch.gated_context.ptr,
            scratch.attention_output.ptr,
            1,
            q_width,
            config.hidden_size,
            backend=self.backend,
            stream=stream,
            libraries=self.libraries,
            runtime=self.runtime,
        )
        self.kernel_plan.add_rmsnorm(
            scratch.hidden.ptr,
            scratch.attention_output.ptr,
            layer.weight("ffn_norm").allocation("raw").tensor.ptr,
            scratch.norm.ptr,
            scratch.post_attention.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        if layer.mlp_type == DENSE_MLP:
            self._run_dense_ffn(layer, stream=stream)
        elif layer.mlp_type == SPARSE_MOE:
            self._run_sparse_ffn(layer, stream=stream)
        else:
            raise ValueError(f"unsupported Laguna MLP type {layer.mlp_type!r}")

    def _run_dense_ffn(
        self,
        layer: LagunaGGUFResidentLayerWeights,
        *,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        config = self.weights.config
        scratch = self.scratch
        linear_libraries = self.libraries.linear
        for slot, output in (
            ("ffn_gate", scratch.dense_gate),
            ("ffn_up", scratch.dense_up),
        ):
            launch_gguf_linear(
                layer.weight(slot),
                scratch.norm.ptr,
                output.ptr,
                1,
                config.hidden_size,
                config.feed_forward_length,
                backend=self.backend,
                stream=stream,
                libraries=linear_libraries,
                runtime=self.runtime,
                use_wmma_prefill=False,
                use_gemv_decode=True,
            )
        self.kernel_plan.dense_silu(
            scratch.dense_gate.ptr,
            scratch.dense_up.ptr,
            scratch.dense_intermediate.ptr,
            1,
            config.feed_forward_length,
            stream=stream,
            library=self.libraries.dense_silu,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            layer.weight("ffn_down"),
            scratch.dense_intermediate.ptr,
            scratch.dense_output.ptr,
            1,
            config.feed_forward_length,
            config.hidden_size,
            backend=self.backend,
            stream=stream,
            libraries=linear_libraries,
            runtime=self.runtime,
            use_wmma_prefill=False,
            use_gemv_decode=True,
        )
        self.kernel_plan.add(
            scratch.post_attention.ptr,
            scratch.dense_output.ptr,
            scratch.hidden.ptr,
            config.hidden_size,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )

    def _run_sparse_ffn(
        self,
        layer: LagunaGGUFResidentLayerWeights,
        *,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.scratch is not None
        assert self.moe_scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        output = run_laguna_moe_c1(
            self.scratch.norm.ptr,
            layer,
            self.moe_scratch,
            stream=stream,
            runtime=self.runtime,
            libraries=self.libraries.moe,
        )
        self.kernel_plan.add(
            self.scratch.post_attention.ptr,
            output.ptr,
            self.scratch.hidden.ptr,
            self.weights.config.hidden_size,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )

    def _project_and_sample(
        self,
        *,
        input_token_id: int,
        position: int,
        stream: int,
    ) -> LagunaEagerTokenResult:
        assert self.weights is not None
        assert self.scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        config = self.weights.config
        scratch = self.scratch
        self.kernel_plan.rmsnorm(
            scratch.hidden.ptr,
            self.weights.root("output_norm").allocation("raw").tensor.ptr,
            scratch.final_norm.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.weights.root("lm_head"),
            scratch.final_norm.ptr,
            scratch.logits.ptr,
            1,
            config.hidden_size,
            config.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            backend=self.backend,
            stream=stream,
            libraries=self.libraries.linear,
            runtime=self.runtime,
            use_wmma_prefill=False,
            use_gemv_decode=True,
        )
        self.kernel_plan.argmax(
            scratch.logits.ptr,
            scratch.argmax_block_values.ptr,
            scratch.argmax_block_indices.ptr,
            scratch.argmax_id.ptr,
            scratch.argmax_value.ptr,
            config.vocab_size,
            stream=stream,
            library=self.libraries.argmax,
            runtime=self.runtime,
        )
        if stream:
            self.runtime.stream_synchronize(stream)
        else:
            self.runtime.device_synchronize()
        next_id = _read_i64(scratch.argmax_id, self.runtime)
        next_value = _read_f32(scratch.argmax_value, self.runtime)
        return LagunaEagerTokenResult(
            position=position,
            input_token_id=input_token_id,
            next_token_id=next_id,
            next_token_logit=next_value,
            logits=scratch.logits,
            final_hidden=scratch.final_norm,
            post_layer_hidden=scratch.hidden,
        )

    def _validate_resident_weights(self) -> None:
        assert self.weights is not None
        config = self.weights.config
        if self.weights.backend != self.backend:
            raise ValueError("Laguna resident weights must share the session backend")
        if self.weights.q6_qmicro != self.q6_qmicro:
            raise ValueError(
                "Laguna resident Q6 qmicro layout must match the session kernel plan"
            )
        root_shapes = {
            "token_embedding": (config.vocab_size, config.hidden_size),
            "output_norm": (config.hidden_size,),
            "lm_head": (config.vocab_size, config.hidden_size),
        }
        for slot, shape in root_shapes.items():
            weight = self.weights.root(slot)
            _validate_laguna_weight_contract(
                weight,
                shape=shape,
                layouts_by_quant=_ROOT_LAYOUTS_BY_SLOT[slot],
                label=f"Laguna root {slot}",
            )
        if len(self.weights.layers) != config.block_count:
            raise ValueError("Laguna resident layer count does not match GGUF metadata")
        for layer_id, layer in enumerate(self.weights.layers):
            heads = config.head_count(layer_id)
            expected_attention = {
                "attn_norm": ((config.hidden_size,), {"f32": LAYOUT_DENSE_F32}),
                "attn_q": (
                    (heads * config.key_length, config.hidden_size),
                    _PROJECTION_LAYOUT_BY_QUANT,
                ),
                "attn_k": (
                    (config.head_count_kv * config.key_length, config.hidden_size),
                    _PROJECTION_LAYOUT_BY_QUANT,
                ),
                "attn_v": (
                    (config.head_count_kv * config.value_length, config.hidden_size),
                    _PROJECTION_LAYOUT_BY_QUANT,
                ),
                "attn_gate": ((heads, config.hidden_size), _PROJECTION_LAYOUT_BY_QUANT),
                "attn_q_norm": ((config.key_length,), {"f32": LAYOUT_DENSE_F32}),
                "attn_k_norm": ((config.key_length,), {"f32": LAYOUT_DENSE_F32}),
                "attn_output": (
                    (config.hidden_size, heads * config.value_length),
                    _PROJECTION_LAYOUT_BY_QUANT,
                ),
                "ffn_norm": ((config.hidden_size,), {"f32": LAYOUT_DENSE_F32}),
            }
            for slot, (shape, layouts_by_quant) in expected_attention.items():
                _validate_laguna_weight_contract(
                    layer.weight(slot),
                    shape=shape,
                    layouts_by_quant=layouts_by_quant,
                    label=f"Laguna layer {layer_id} {slot}",
                )
            if layer.weight("attn_gate").spec.source.shape != (
                heads,
                config.hidden_size,
            ):
                raise ValueError(
                    f"Laguna layer {layer_id} requires {PER_HEAD_GATE!r} attention gating"
                )
            if layer.mlp_type == DENSE_MLP:
                dense_shapes = {
                    "ffn_gate": (config.feed_forward_length, config.hidden_size),
                    "ffn_up": (config.feed_forward_length, config.hidden_size),
                    "ffn_down": (config.hidden_size, config.feed_forward_length),
                }
                for slot, shape in dense_shapes.items():
                    _validate_laguna_weight_contract(
                        layer.weight(slot),
                        shape=shape,
                        layouts_by_quant=_DENSE_MLP_LAYOUTS_BY_SLOT[slot],
                        label=f"Laguna dense layer {layer_id} {slot}",
                    )
            elif layer.mlp_type != SPARSE_MOE:
                raise ValueError(f"unsupported Laguna MLP type {layer.mlp_type!r}")
            else:
                continue
        assert self.moe_plan is not None
        for layer in self.weights.layers[config.leading_dense_block_count :]:
            validate_laguna_moe_layer(layer, self.moe_plan)

    def _close(self, *, suppress_errors: bool) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []

        def release(action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as exc:  # best-effort teardown after HIP failures
                errors.append(exc)

        if self._moe_shared_stream:
            stream = self._moe_shared_stream
            self._moe_shared_stream = 0
            release(lambda: self.runtime.stream_destroy(stream))
        if self._moe_shared_output_ready_event:
            event = self._moe_shared_output_ready_event
            self._moe_shared_output_ready_event = 0
            release(lambda: self.runtime.event_destroy(event))
        if self._moe_shared_input_ready_event:
            event = self._moe_shared_input_ready_event
            self._moe_shared_input_ready_event = 0
            release(lambda: self.runtime.event_destroy(event))
        if self.f16_hipblaslt is not None:
            route = self.f16_hipblaslt
            self.f16_hipblaslt = None
            release(route.close)
        if self.verifier_scratch is not None:
            scratch = self.verifier_scratch
            self.verifier_scratch = None
            release(lambda: scratch.free(runtime=self.runtime))
        if self.rows_moe_scratch is not None:
            scratch = self.rows_moe_scratch
            self.rows_moe_scratch = None
            release(lambda: scratch.free(runtime=self.runtime))
        if self.rows_scratch is not None:
            scratch = self.rows_scratch
            self.rows_scratch = None
            release(lambda: scratch.free(runtime=self.runtime))
        if self.moe_scratch is not None:
            scratch = self.moe_scratch
            self.moe_scratch = None
            release(lambda: scratch.free(runtime=self.runtime))
        if self.scratch is not None:
            scratch = self.scratch
            self.scratch = None
            release(lambda: scratch.free(runtime=self.runtime))
        if self.kv_cache is not None:
            cache = self.kv_cache
            self.kv_cache = None
            release(cache.free)
        if self.swa_rope is not None:
            tables = self.swa_rope
            self.swa_rope = None
            release(lambda: tables.free(runtime=self.runtime))
        if self.full_rope is not None:
            tables = self.full_rope
            self.full_rope = None
            release(lambda: tables.free(runtime=self.runtime))
        if self.weights is not None:
            weights = self.weights
            self.weights = None
            if self._owns_weights:
                release(lambda: weights.free(runtime=self.runtime))
        self.kernel_plan = None
        self.moe_plan = None
        self.libraries = None
        if errors and not suppress_errors:
            raise RuntimeError("one or more Laguna session resources failed to free") from errors[0]

    def _ensure_verifier_resources(self) -> None:
        if self.verifier_scratch is not None:
            return
        from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import (
            build_dflash_accept,
        )

        target_arch = hip_target_arch_for_backend(self.backend)
        with hip_target_arch_environment(target_arch):
            library = build_dflash_accept(
                load=True,
                compiler_version=self._compiler_version,
                require_cached=self._require_cached_build,
            )
        scratch = LagunaVerifierScratch.allocate(
            self.config,
            max_rows=self.prefill_chunk_size,
            runtime=self.runtime,
        )
        self._dflash_accept_library = library
        self.verifier_scratch = scratch

    def _check_no_staged_verifier(self) -> None:
        if self._staged_verifier_tokens is not None:
            raise RuntimeError("a Laguna verifier transaction is already staged")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Laguna GGUF resident session is closed")


def _validate_laguna_weight_contract(
    weight,
    *,
    shape: tuple[int, ...],
    layouts_by_quant: Mapping[str, str],
    label: str,
) -> None:
    source = weight.spec.source
    quant = weight.spec.quant_key
    expected_layout = layouts_by_quant.get(quant)
    if source.shape != shape or expected_layout is None or weight.spec.layout != expected_layout:
        raise ValueError(
            f"{label} resident contract mismatch: shape={source.shape} "
            f"layout/quant={weight.spec.layout}/{quant}"
        )


def _copy_i64(buffer: DeviceBuffer, value: int, runtime: HipRuntime) -> None:
    host = ctypes.c_int64(int(value))
    runtime.memcpy(
        buffer.ptr,
        ctypes.addressof(host),
        ctypes.sizeof(host),
        HipMemcpyKind.HOST_TO_DEVICE,
    )


def _copy_i64_rows(
    buffer: DeviceBuffer,
    values: Sequence[int],
    runtime: HipRuntime,
) -> None:
    parsed = tuple(int(value) for value in values)
    nbytes = len(parsed) * _I64_NBYTES
    if not parsed or nbytes > buffer.nbytes:
        raise ValueError("int64 row copy must be non-empty and fit the destination")
    host = (ctypes.c_int64 * len(parsed))(*parsed)
    runtime.memcpy(
        buffer.ptr,
        ctypes.addressof(host),
        nbytes,
        HipMemcpyKind.HOST_TO_DEVICE,
    )


def _copy_i32_rows(
    buffer: DeviceBuffer,
    values: Sequence[int],
    runtime: HipRuntime,
) -> None:
    parsed = tuple(int(value) for value in values)
    nbytes = len(parsed) * _I32_NBYTES
    if not parsed or nbytes > buffer.nbytes:
        raise ValueError("int32 row copy must be non-empty and fit the destination")
    host = (ctypes.c_int32 * len(parsed))(*parsed)
    runtime.memcpy(
        buffer.ptr,
        ctypes.addressof(host),
        nbytes,
        HipMemcpyKind.HOST_TO_DEVICE,
    )


def _copy_u8_rows(
    buffer: DeviceBuffer,
    values: Sequence[int],
    runtime: HipRuntime,
) -> None:
    parsed = tuple(int(value) for value in values)
    if not parsed or len(parsed) > buffer.nbytes or any(value not in {0, 1} for value in parsed):
        raise ValueError("uint8 row copy must contain boolean values and fit the destination")
    host = (ctypes.c_uint8 * len(parsed))(*parsed)
    runtime.memcpy(
        buffer.ptr,
        ctypes.addressof(host),
        len(parsed),
        HipMemcpyKind.HOST_TO_DEVICE,
    )


def _buffer_view(buffer: DeviceBuffer, offset: int, nbytes: int) -> DeviceBuffer:
    parsed_offset = int(offset)
    parsed_nbytes = int(nbytes)
    if parsed_offset < 0 or parsed_nbytes <= 0 or parsed_offset + parsed_nbytes > buffer.nbytes:
        raise ValueError("borrowed device-buffer view exceeds its owner")
    return DeviceBuffer(buffer.ptr + parsed_offset, parsed_nbytes)


def _read_i32_rows(
    buffer: DeviceBuffer,
    rows: int,
    runtime: HipRuntime,
) -> tuple[int, ...]:
    count = int(rows)
    nbytes = count * _I32_NBYTES
    if count <= 0 or nbytes > buffer.nbytes:
        raise ValueError("int32 row read must be non-empty and fit the source")
    host = (ctypes.c_int32 * count)()
    runtime.memcpy(
        ctypes.addressof(host),
        buffer.ptr,
        nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return tuple(int(value) for value in host)


def _read_f32_rows(
    buffer: DeviceBuffer,
    rows: int,
    runtime: HipRuntime,
) -> tuple[float, ...]:
    count = int(rows)
    nbytes = count * _F32_NBYTES
    if count <= 0 or nbytes > buffer.nbytes:
        raise ValueError("float32 row read must be non-empty and fit the source")
    host = (ctypes.c_float * count)()
    runtime.memcpy(
        ctypes.addressof(host),
        buffer.ptr,
        nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return tuple(float(value) for value in host)


def _read_i64(buffer: DeviceBuffer, runtime: HipRuntime) -> int:
    host = ctypes.c_int64()
    runtime.memcpy(
        ctypes.addressof(host),
        buffer.ptr,
        ctypes.sizeof(host),
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return int(host.value)


def _read_f32(buffer: DeviceBuffer, runtime: HipRuntime) -> float:
    host = ctypes.c_float()
    runtime.memcpy(
        ctypes.addressof(host),
        buffer.ptr,
        ctypes.sizeof(host),
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return float(host.value)


def _validate_laguna_dflash_accept_payload(
    *,
    tokens: tuple[int, ...],
    positions: tuple[int, ...],
    target_top1: tuple[int, ...],
    remaining_decode: int | None,
    payload: tuple[int, ...],
) -> None:
    """Cross-check the GPU's compact chain summary before canonical commit."""

    if len(tokens) < 2 or len(tokens) != len(positions) or len(tokens) != len(target_top1):
        raise ValueError("Laguna DFlash accept inputs must align root plus draft rows")
    if len(payload) != 7:
        raise ValueError("Laguna DFlash packed accept payload must contain seven fields")
    accepted = 0
    draft_count = len(tokens) - 1
    limit = draft_count if remaining_decode is None else min(draft_count, remaining_decode)
    while accepted < limit and tokens[accepted + 1] == target_top1[accepted]:
        accepted += 1
    expected_next = (
        -1
        if remaining_decode is not None and accepted >= remaining_decode
        else target_top1[accepted]
    )
    expected = (
        accepted,
        accepted,
        tokens[accepted],
        positions[accepted],
        expected_next,
        int(accepted == draft_count),
        accepted + 1,
    )
    if payload != expected:
        raise RuntimeError(
            "Laguna DFlash GPU accept summary disagrees with the CPU chain oracle: "
            f"expected={expected} actual={payload}"
        )


def _resolve_exact(key: KernelKey) -> Callable:
    if not is_registered(key):
        raise LookupError(f"required Laguna eager kernel is not registered: {key.display()}")
    function = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    assert function is not None
    return function


__all__ = [
    "LAGUNA_DFLASH_CAPTURE_DEPTHS",
    "LagunaDFlashVerifyResult",
    "LagunaEagerKernelPlan",
    "LagunaEagerLibraries",
    "LagunaEagerScratch",
    "LagunaEagerTokenResult",
    "LagunaGGUFResidentSession",
    "LagunaHiddenCaptureTargets",
    "LagunaRowsScratch",
    "LagunaVerifierRowsResult",
    "LagunaVerifierScratch",
    "capture_laguna_hidden_rows",
    "capture_laguna_hidden_tap",
    "load_laguna_eager_libraries",
    "resolve_laguna_eager_kernel_plan",
]
