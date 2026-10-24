"""Geometry and storage-policy planning for Qwen hybrid GGUF execution.

One scratch plan supplies device allocation and size-only capacity estimation.
No executor, device allocation, kernel selection or environment lookup belongs here.
Route-specific workspace costs are supplied by callers as sizing callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.runtime.qwen35_gguf_kv import (
    Qwen35GGUFKVChunkLayout,
    _qwen35_gguf_kv_page_bytes,
)


_PACKED_VERIFY_MIN_MAX_SEQUENCE = 1024
_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS = 8192
_Q8_1_BLOCK = 32
_Q8_1_BLOCK_BYTES = 36

# Scratch fields whose byte size grows with the declared context. Every other
# ``field_sizes`` entry is sized by slot count or model geometry alone.
_CONTEXT_SCALED_SCRATCH_FIELDS = frozenset(
    {"full_attn_split_partial", "full_attn_split_m", "full_attn_split_l"}
)

# Default transient coefficients for the packed slot-local server prefill route.
#
# The packed route materialises a BF16 full-prompt K/V image (the "oracle") for
# one full-attention layer at a time while attention runs, and keeps full-position
# hidden planes for the layer-outer executor. Measured on 2026-09-11 at 16,384
# declared context: 1,073,741,824 B of live oracle owners (16 owners = one
# full-prompt image, exactly 16 layers x 2 x 4 heads x 256 dims x 2 B =
# 64 KiB/token) plus 293,400,616 B of hidden owners (~10 KiB/token). The packed
# execution workspace itself is context-independent (~1.017 GiB at both 16,384
# and 32,768 declared), so it is a fixed term rather than a slope.
_GGUF_TRANSIENT_BYTES_PER_TOKEN_DEFAULT = 74 * 1024
_GGUF_TRANSIENT_FIXED_BYTES_DEFAULT = 1024**3
_GGUF_CAPACITY_RESERVE_MIB_DEFAULT = 512


def _qwen35_gguf_kv_chunk_layout(
    cfg: object,
    *,
    max_positions: int,
    kv_storage_dtype: str | DType,
    kv_storage_layout: str = "uniform",
    kv_scale_dtype: str | DType = DType.FP16,
    kv_scale_granularity: str = "per_token_head",
    int8_kv_value_bf16: bool = False,
    int8_bf16_full_attention_layer_indices: Sequence[int] = (),
    int8_kv_no_mirror_qualified: bool = False,
) -> Qwen35GGUFKVChunkLayout:
    """Build the resolved per-layer KV layout from policy inputs alone.

    This is the pure form of :func:`_qwen35_gguf_session_kv_chunk_layout`; the
    capacity estimator uses it to price KV pages before a session exists.
    """

    storage = DType.parse(kv_storage_dtype)
    bf16_full_indices = frozenset(int(index) for index in int8_bf16_full_attention_layer_indices)
    layer_storage: list[DType | None] = []
    full_attention_index = 0
    for layer_type in cfg.layer_types:
        if layer_type == LINEAR_ATTENTION:
            layer_storage.append(None)
            continue
        layer_storage.append(
            DType.BF16
            if storage == DType.BF16 or full_attention_index in bf16_full_indices
            else DType.INT8_PER_TOKEN_HEAD
        )
        full_attention_index += 1
    granularity = str(kv_scale_granularity or "per_token_head").strip().lower()
    mirror_layers = (
        tuple(
            layer_id
            for layer_id, layer in enumerate(layer_storage)
            if layer == DType.INT8_PER_TOKEN_HEAD
        )
        if storage == DType.INT8_PER_TOKEN_HEAD
        and granularity != "hadamard_group32"
        and not bool(int8_kv_no_mirror_qualified)
        and int(max_positions) <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS
        else ()
    )
    return Qwen35GGUFKVChunkLayout(
        storage_dtype=storage,
        storage_layout=str(kv_storage_layout or "uniform").strip().lower(),
        scale_dtype=DType.parse(kv_scale_dtype),
        scale_granularity=granularity,
        int8_kv_value_bf16=bool(int8_kv_value_bf16),
        layer_storage_dtypes=tuple(layer_storage),
        bf16_mirror_layer_indices=mirror_layers,
    )

def _align_prefill_scratch(value: int, alignment: int = 256) -> int:
    return (int(value) + alignment - 1) // alignment * alignment

@dataclass(frozen=True)
class _FullStackScratchPlan:
    """Pure sizing plan for one resident GGUF full-stack scratch stack.

    This is the single source of truth for how large a resident scratch stack is
    at a given declared context.  ``_FullStackScratch.allocate`` consumes the
    plan to drive real allocations, and the GGUF capacity estimator consumes the
    same plan to predict the footprint without touching the GPU.  Keeping the
    arithmetic in one place is what stops the estimator from drifting away from
    the allocator when a route, KV policy, or mirror threshold changes.

    ``owner_sizes`` is the ordered list of logical allocation sizes.  It is
    exactly the sequence ``allocate`` feeds to ``malloc``, so summing it is a
    byte-exact prediction of the dedicated-allocation footprint.
    """

    block_size: int
    max_positions: int
    block_count: int
    slot_count: int
    total_blocks: int
    total_positions: int
    full_attn_split_count: int
    full_attention_count: int
    bf16_full_attention_indices: tuple[int, ...]
    bf16_full_attention_index_set: frozenset[int]
    short_int8_bf16_mirror: bool
    scale_shape: tuple[int, ...]
    int8_cache_nbytes: int
    bf16_cache_nbytes: int
    mirror_bf16_nbytes: int
    scale_nbytes: int
    moe_top_k: int
    moe_experts: int
    kv_storage: DType
    scale_dtype: DType
    kv_storage_layout: str
    kv_scale_granularity: str
    int8_kv_value_bf16: bool
    kv_payload_bytes: int
    kv_mirror_bytes: int
    kv_scale_bytes: int
    linear_state_bytes: int
    metadata_bytes: int
    workspace_bytes: int
    context_scaled_bytes: int
    conv_zero_nbytes: int
    recurrent_zero_nbytes: int
    block_table_nbytes: int
    position_nbytes: int
    context_nbytes: int
    cos_nbytes: int
    sin_nbytes: int
    field_sizes: Mapping[str, int]
    owner_sizes: tuple[int, ...]
    conv_zero: np.ndarray | None
    recurrent_zero: np.ndarray | None
    block_table_arr: np.ndarray | None
    position_host: np.ndarray | None
    context_host: np.ndarray | None
    cos_arr: np.ndarray | None
    sin_arr: np.ndarray | None

    @property
    def dedicated_bytes(self) -> int:
        """Logical bytes allocated when every owner is a separate buffer."""

        return int(sum(self.owner_sizes))

    @property
    def fixed_bytes(self) -> int:
        """Plan bytes that do not grow with the declared context."""

        return int(self.dedicated_bytes) - int(self.context_scaled_bytes)

    def arena_bytes(self, *, alignment: int = 256) -> int:
        """Bytes the single-arena path would allocate for the same plan."""

        cursor = 0
        for size in self.owner_sizes:
            cursor = _align_prefill_scratch(cursor, alignment)
            cursor += int(size)
        return _align_prefill_scratch(cursor, alignment)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "max_positions": self.max_positions,
            "block_count": self.block_count,
            "slot_count": self.slot_count,
            "total_blocks": self.total_blocks,
            "total_positions": self.total_positions,
            "full_attn_split_count": self.full_attn_split_count,
            "full_attention_count": self.full_attention_count,
            "bf16_full_attention_indices": list(self.bf16_full_attention_indices),
            "short_int8_bf16_mirror": self.short_int8_bf16_mirror,
            "int8_cache_nbytes": self.int8_cache_nbytes,
            "bf16_cache_nbytes": self.bf16_cache_nbytes,
            "mirror_bf16_nbytes": self.mirror_bf16_nbytes,
            "scale_nbytes": self.scale_nbytes,
            "kv_storage": self.kv_storage.value,
            "kv_scale_dtype": self.scale_dtype.value,
            "kv_storage_layout": self.kv_storage_layout,
            "kv_scale_granularity": self.kv_scale_granularity,
            "int8_kv_value_bf16": self.int8_kv_value_bf16,
            "kv_payload_bytes": self.kv_payload_bytes,
            "kv_mirror_bytes": self.kv_mirror_bytes,
            "kv_scale_bytes": self.kv_scale_bytes,
            "linear_state_bytes": self.linear_state_bytes,
            "metadata_bytes": self.metadata_bytes,
            "workspace_bytes": self.workspace_bytes,
            "context_scaled_bytes": self.context_scaled_bytes,
            "conv_zero_nbytes": self.conv_zero_nbytes,
            "recurrent_zero_nbytes": self.recurrent_zero_nbytes,
            "block_table_nbytes": self.block_table_nbytes,
            "cos_nbytes": self.cos_nbytes,
            "sin_nbytes": self.sin_nbytes,
            "fixed_bytes": self.fixed_bytes,
            "dedicated_bytes": self.dedicated_bytes,
            "arena_bytes": self.arena_bytes(),
        }

def _full_stack_scratch_plan(
    cfg: object,
    *,
    hidden_size: int,
    ffn_size: int,
    q_width: int,
    kv_width: int,
    linear_qkv_width: int,
    fp16_recurrent_state: bool = False,
    max_sequence_length: int | None = None,
    max_batch_size: int = 1,
    kv_storage_dtype: str | DType = DType.BF16,
    kv_storage_layout: str = "uniform",
    kv_scale_dtype: str | DType = DType.FP16,
    kv_scale_granularity: str = "per_token_head",
    int8_kv_value_bf16: bool = False,
    int8_bf16_prefix_full_attention_layers: int = 0,
    int8_bf16_full_attention_layer_indices: tuple[int, ...] | None = None,
    retain_int8_bf16_mirrors: bool = True,
    allocate_kv_cache: bool = True,
    block_size: int = 256,
    materialize: bool = True,
) -> _FullStackScratchPlan:
    """Compute the resident scratch sizing plan without touching the GPU.

    The arithmetic here mirrors ``_FullStackScratch.allocate`` exactly; that
    method now allocates from this plan instead of repeating the formulas.

    ``materialize=False`` skips building the host-side arrays and returns their
    byte sizes only. The capacity estimator prices dozens of candidate contexts
    and must not allocate a 134 MiB rope table per candidate, so it always uses
    the size-only form.
    """

    block_size = int(block_size)
    kv_storage = DType.parse(kv_storage_dtype)
    if kv_storage not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
        raise ValueError("GGUF resident full-attention KV storage must be bf16 or int8_per_token_head")
    scale_dtype = DType.parse(kv_scale_dtype)
    if scale_dtype not in {DType.FP16, DType.FP32}:
        raise ValueError("GGUF INT8 KV scales must use fp16 or fp32")
    kv_storage_layout = str(kv_storage_layout or "uniform").strip().lower()
    if kv_storage_layout not in {"uniform", "tail4_hadamard_group32"}:
        raise ValueError(f"unsupported GGUF resident KV storage layout {kv_storage_layout!r}")
    kv_scale_granularity = str(kv_scale_granularity or "per_token_head").strip().lower()
    if kv_scale_granularity not in {"per_token_head", "block16", "hadamard_group32"}:
        raise ValueError(
            "GGUF INT8 KV scale granularity must be per_token_head, block16, or hadamard_group32"
        )
    if kv_storage_layout == "tail4_hadamard_group32" and (
        kv_storage != DType.INT8_PER_TOKEN_HEAD or kv_scale_granularity != "hadamard_group32"
    ):
        raise ValueError("tail4_hadamard_group32 requires Hadamard-group32 INT8 storage")
    int8_kv_value_bf16 = bool(int8_kv_value_bf16 and kv_storage == DType.INT8_PER_TOKEN_HEAD)
    if int8_kv_value_bf16 and kv_scale_granularity != "per_token_head":
        raise ValueError("GGUF grouped INT8 KV scales are not supported with the key-only diagnostic")
    requested_positions = block_size if max_sequence_length is None else int(max_sequence_length)
    if requested_positions <= 0:
        raise ValueError("max_sequence_length must be positive")
    if requested_positions > int(cfg.context_length):
        raise ValueError(
            f"max_sequence_length {requested_positions} exceeds GGUF context length {cfg.context_length}"
        )
    slot_count = int(max_batch_size)
    if slot_count <= 0:
        raise ValueError("max_batch_size must be positive")
    block_count = (requested_positions + block_size - 1) // block_size
    max_positions = min(int(cfg.context_length), block_count * block_size)
    total_blocks = slot_count * block_count
    total_positions = slot_count * max_positions
    hidden_bytes = slot_count * int(hidden_size) * 2
    hidden_fp32_bytes = int(hidden_size) * DType.FP32.itemsize
    ffn_bytes = slot_count * int(ffn_size) * 2
    moe_lane_count = max(1, int(cfg.expert_used_count)) if cfg.is_moe else 1
    moe_top_k = max(1, int(cfg.expert_used_count))
    moe_experts = max(1, int(cfg.expert_count))
    moe_shared_ffn = max(1, int(cfg.expert_shared_feed_forward_length or ffn_size or 1))
    q8_1_gate_blocks = (int(hidden_size) + _Q8_1_BLOCK - 1) // _Q8_1_BLOCK
    q8_1_down_blocks = moe_top_k * ((int(ffn_size) + _Q8_1_BLOCK - 1) // _Q8_1_BLOCK)
    q8_1_moe_bytes = max(q8_1_gate_blocks, q8_1_down_blocks) * _Q8_1_BLOCK_BYTES
    linear_qkv_bytes = slot_count * int(linear_qkv_width) * 2
    ssm_inner_bytes = slot_count * cfg.ssm_inner_size * 2
    alpha_bytes = slot_count * cfg.ssm_time_step_rank * 2
    q_proj_bytes = slot_count * 2 * int(q_width) * 2
    kv_bf16_bytes = slot_count * int(kv_width) * 2
    q_f32_bytes = slot_count * int(q_width) * 4
    kv_f32_bytes = slot_count * int(kv_width) * 4
    full_attn_split_count = (max_positions + block_size - 1) // block_size
    full_attn_split_partial_bytes = slot_count * int(q_width) * full_attn_split_count * 4
    full_attn_split_stat_bytes = slot_count * cfg.head_count * full_attn_split_count * 4
    conv_zero_nbytes = slot_count * int(linear_qkv_width) * cfg.ssm_conv_kernel * DType.FP32.itemsize
    recurrent_zero_nbytes = (
        slot_count
        * cfg.ssm_time_step_rank
        * cfg.ssm_state_size
        * (int(cfg.ssm_inner_size) // int(cfg.ssm_time_step_rank))
        * (DType.FP16.itemsize if bool(fp16_recurrent_state) else DType.FP32.itemsize)
    )
    block_table_nbytes = (
        block_count * DType.INT32.itemsize
        if slot_count == 1
        else total_blocks * DType.INT32.itemsize
    )
    position_nbytes = slot_count * DType.INT64.itemsize
    context_nbytes = slot_count * DType.INT64.itemsize
    cos_nbytes = max_positions * int(cfg.rope_dimension_count) * DType.FP32.itemsize
    sin_nbytes = cos_nbytes
    if materialize:
        conv_zero = np.zeros(
            (slot_count, int(linear_qkv_width), cfg.ssm_conv_kernel), dtype=np.float32
        )
        recurrent_zero = np.zeros(
            (
                slot_count,
                cfg.ssm_time_step_rank,
                cfg.ssm_state_size,
                int(cfg.ssm_inner_size) // int(cfg.ssm_time_step_rank),
            ),
            dtype=(np.float16 if bool(fp16_recurrent_state) else np.float32),
        )
        block_table_arr = (
            np.arange(block_count, dtype=np.int32)
            if slot_count == 1
            else np.arange(total_blocks, dtype=np.int32).reshape(slot_count, block_count)
        )
        position_host = np.zeros((slot_count,), dtype=np.int64)
        context_host = np.ones((slot_count,), dtype=np.int64)
        cos_arr, sin_arr = _rope_tables(
            max_positions=max_positions,
            rotary_dim=cfg.rope_dimension_count,
            base=cfg.rope_freq_base,
        )
        assert int(conv_zero.nbytes) == conv_zero_nbytes
        assert int(recurrent_zero.nbytes) == recurrent_zero_nbytes
        assert int(block_table_arr.nbytes) == block_table_nbytes
        assert int(position_host.nbytes) == position_nbytes
        assert int(context_host.nbytes) == context_nbytes
        assert int(cos_arr.nbytes) == cos_nbytes
        assert int(sin_arr.nbytes) == sin_nbytes
    else:
        conv_zero = None
        recurrent_zero = None
        block_table_arr = None
        position_host = None
        context_host = None
        cos_arr = None
        sin_arr = None
    int8_bf16_prefix_full_attention_layers = max(0, int(int8_bf16_prefix_full_attention_layers))
    if int8_bf16_full_attention_layer_indices is None:
        bf16_full_attention_indices = tuple(range(int8_bf16_prefix_full_attention_layers))
    else:
        bf16_full_attention_indices = tuple(sorted({int(idx) for idx in int8_bf16_full_attention_layer_indices}))
    full_attention_count = sum(1 for layer_type in cfg.layer_types if layer_type == FULL_ATTENTION)
    bad_bf16_indices = [idx for idx in bf16_full_attention_indices if idx < 0 or idx >= full_attention_count]
    if bad_bf16_indices:
        raise ValueError(
            f"GGUF INT8 BF16 full-attention layer indices {bad_bf16_indices} outside [0, {full_attention_count})"
        )
    bf16_full_attention_index_set = frozenset(bf16_full_attention_indices)
    int8_cache_nbytes = total_positions * cfg.head_count_kv * cfg.key_length * DType.INT8.itemsize
    bf16_cache_nbytes = total_positions * cfg.head_count_kv * cfg.key_length * DType.BF16.itemsize
    mirror_bf16_nbytes = bf16_cache_nbytes
    short_int8_bf16_mirror = (
        bool(retain_int8_bf16_mirrors)
        and kv_storage == DType.INT8_PER_TOKEN_HEAD
        and kv_scale_granularity != "hadamard_group32"
        and max_positions <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS
    )
    if kv_scale_granularity == "block16":
        scale_dim_blocks = int(cfg.key_length) // 16
        if int(cfg.key_length) % 16 != 0 or scale_dim_blocks != 16:
            raise ValueError("GGUF INT8 KV block16 scales require head_dim/key_length 256")
        scale_shape = (total_blocks, block_size, cfg.head_count_kv, scale_dim_blocks)
    elif kv_scale_granularity == "hadamard_group32":
        if int(cfg.key_length) % 32:
            raise ValueError("GGUF Hadamard-group32 KV requires head_dim/key_length divisible by 32")
        scale_shape = (total_blocks, block_size, cfg.head_count_kv, int(cfg.key_length) // 32)
    else:
        scale_shape = (total_blocks, block_size, cfg.head_count_kv)
    scale_nbytes = int(np.prod(scale_shape)) * scale_dtype.itemsize
    field_sizes = {
        "norm": hidden_bytes,
        "hidden_seed_fp32": hidden_fp32_bytes,
        "post_norm": hidden_bytes,
        "post_norm_f32": hidden_fp32_bytes,
        "residual": hidden_bytes,
        "attn_out": hidden_bytes,
        "linear_qkv": linear_qkv_bytes,
        "linear_z": ssm_inner_bytes,
        "linear_alpha": alpha_bytes,
        "linear_beta": alpha_bytes,
        "linear_alpha_beta": 2 * alpha_bytes,
        "conv_out": slot_count * int(linear_qkv_width) * 4,
        "recurrent_out": slot_count * cfg.ssm_inner_size * 4,
        "recurrent_bf16": ssm_inner_bytes,
        "linear_conv_state_tmp": conv_zero_nbytes,
        "linear_recurrent_state_tmp": recurrent_zero_nbytes,
        "full_q": q_proj_bytes,
        "full_k": kv_bf16_bytes,
        "full_v": kv_bf16_bytes,
        "full_query_raw": q_f32_bytes,
        "full_key_raw": kv_f32_bytes,
        "full_query": q_f32_bytes,
        "full_key": kv_f32_bytes,
        "full_gate": slot_count * int(q_width) * 2,
        "full_attn_context": q_f32_bytes,
        "full_attn_split_partial": full_attn_split_partial_bytes,
        "full_attn_split_m": full_attn_split_stat_bytes,
        "full_attn_split_l": full_attn_split_stat_bytes,
        "full_gated": slot_count * int(q_width) * 2,
        "ffn_gate_up": 2 * ffn_bytes * moe_lane_count,
        "ffn_intermediate": ffn_bytes * moe_lane_count,
        "ffn_intermediate_f32": (
            moe_top_k * int(ffn_size) * DType.FP32.itemsize
        ),
        "ffn_down": hidden_bytes,
        "moe_q8_1": q8_1_moe_bytes,
        "moe_router_logits": (
            slot_count * (moe_experts + 1) * DType.FP32.itemsize
        ),
        "moe_router_counter": DType.INT32.itemsize,
        "moe_selected_experts": slot_count * moe_top_k * DType.INT64.itemsize,
        "moe_routing_weights": slot_count * moe_top_k * DType.FP32.itemsize,
        "moe_down_out": moe_top_k * hidden_bytes,
        "moe_down_out_f32": (
            moe_top_k * int(hidden_size) * DType.FP32.itemsize
        ),
        "moe_group_counts": slot_count * moe_experts * DType.INT32.itemsize,
        "moe_padded_counts": slot_count * moe_experts * DType.INT32.itemsize,
        "moe_scatter_offsets": (
            slot_count * moe_experts * DType.INT32.itemsize
        ),
        "moe_expert_start_compact": (
            slot_count * (moe_experts + 1) * DType.INT64.itemsize
        ),
        "moe_total_compact": slot_count * DType.INT64.itemsize,
        "moe_sorted_lanes": slot_count * moe_top_k * DType.INT64.itemsize,
        "moe_sorted_experts": slot_count * moe_top_k * DType.INT64.itemsize,
        "moe_sorted_weights": slot_count * moe_top_k * DType.FP32.itemsize,
        "moe_lane_to_row": slot_count * moe_top_k * DType.INT64.itemsize,
        "moe_shared_gate": (
            slot_count * moe_shared_ffn * DType.BF16.itemsize
        ),
        "moe_shared_up": slot_count * moe_shared_ffn * DType.BF16.itemsize,
        "moe_shared_intermediate": (
            slot_count * moe_shared_ffn * DType.BF16.itemsize
        ),
        "moe_shared_out": hidden_bytes,
        "moe_shared_out_f32": hidden_fp32_bytes,
        "moe_shared_gate_logits": slot_count * DType.FP32.itemsize,
    }
    owner_sizes: list[int] = []
    kv_payload_bytes = 0
    kv_mirror_bytes = 0
    kv_scale_bytes = 0
    linear_state_bytes = 0
    full_attention_index = 0
    for layer_type in cfg.layer_types:
        if layer_type == LINEAR_ATTENTION:
            owner_sizes.extend((int(conv_zero_nbytes), int(recurrent_zero_nbytes)))
            linear_state_bytes += int(conv_zero_nbytes) + int(recurrent_zero_nbytes)
            continue
        if not allocate_kv_cache:
            full_attention_index += 1
            continue
        layer_uses_int8 = kv_storage == DType.INT8_PER_TOKEN_HEAD and (
            full_attention_index not in bf16_full_attention_index_set
        )
        key_cache_nbytes = (
            int8_cache_nbytes if layer_uses_int8 else bf16_cache_nbytes
        )
        value_cache_nbytes = (
            bf16_cache_nbytes
            if layer_uses_int8 and int8_kv_value_bf16
            else key_cache_nbytes
        )
        owner_sizes.extend((int(key_cache_nbytes), int(value_cache_nbytes)))
        kv_payload_bytes += int(key_cache_nbytes) + int(value_cache_nbytes)
        if short_int8_bf16_mirror and layer_uses_int8:
            owner_sizes.extend((int(mirror_bf16_nbytes), int(mirror_bf16_nbytes)))
            kv_mirror_bytes += 2 * int(mirror_bf16_nbytes)
        if layer_uses_int8:
            owner_sizes.extend((int(scale_nbytes), int(scale_nbytes)))
            kv_scale_bytes += 2 * int(scale_nbytes)
        full_attention_index += 1
    owner_sizes.extend(
        (
            int(block_table_nbytes),
            int(position_nbytes),
            int(context_nbytes),
            int(cos_nbytes),
            int(sin_nbytes),
            *(int(size) for size in field_sizes.values()),
        )
    )
    metadata_bytes = (
        int(block_table_nbytes)
        + int(position_nbytes)
        + int(context_nbytes)
        + int(cos_nbytes)
        + int(sin_nbytes)
    )
    workspace_bytes = sum(int(size) for size in field_sizes.values())
    context_scaled_bytes = (
        kv_payload_bytes
        + kv_mirror_bytes
        + kv_scale_bytes
        + int(block_table_nbytes)
        + int(cos_nbytes)
        + int(sin_nbytes)
        + sum(
            int(size)
            for name, size in field_sizes.items()
            if name in _CONTEXT_SCALED_SCRATCH_FIELDS
        )
    )
    return _FullStackScratchPlan(
        block_size=block_size,
        max_positions=max_positions,
        block_count=block_count,
        slot_count=slot_count,
        total_blocks=total_blocks,
        total_positions=total_positions,
        full_attn_split_count=full_attn_split_count,
        full_attention_count=full_attention_count,
        bf16_full_attention_indices=bf16_full_attention_indices,
        bf16_full_attention_index_set=bf16_full_attention_index_set,
        short_int8_bf16_mirror=short_int8_bf16_mirror,
        scale_shape=scale_shape,
        int8_cache_nbytes=int8_cache_nbytes,
        bf16_cache_nbytes=bf16_cache_nbytes,
        mirror_bf16_nbytes=mirror_bf16_nbytes,
        scale_nbytes=scale_nbytes,
        moe_top_k=moe_top_k,
        moe_experts=moe_experts,
        kv_storage=kv_storage,
        scale_dtype=scale_dtype,
        kv_storage_layout=kv_storage_layout,
        kv_scale_granularity=kv_scale_granularity,
        int8_kv_value_bf16=int8_kv_value_bf16,
        kv_payload_bytes=kv_payload_bytes,
        kv_mirror_bytes=kv_mirror_bytes,
        kv_scale_bytes=kv_scale_bytes,
        linear_state_bytes=linear_state_bytes,
        metadata_bytes=metadata_bytes,
        workspace_bytes=workspace_bytes,
        context_scaled_bytes=context_scaled_bytes,
        conv_zero_nbytes=conv_zero_nbytes,
        recurrent_zero_nbytes=recurrent_zero_nbytes,
        block_table_nbytes=block_table_nbytes,
        position_nbytes=position_nbytes,
        context_nbytes=context_nbytes,
        cos_nbytes=cos_nbytes,
        sin_nbytes=sin_nbytes,
        field_sizes=field_sizes,
        owner_sizes=tuple(int(size) for size in owner_sizes),
        conv_zero=conv_zero,
        recurrent_zero=recurrent_zero,
        block_table_arr=block_table_arr,
        position_host=position_host,
        context_host=context_host,
        cos_arr=cos_arr,
        sin_arr=sin_arr,
    )

@dataclass(frozen=True)
class Qwen35GGUFResidentBreakdown:
    """Exact resident footprint of one GGUF server configuration at a context."""

    context_tokens: int
    max_positions: int
    block_size: int
    page_bytes: int
    pages_per_request: int
    kv_pool_pages: int
    workspace_lease_pages: int
    scratch_bytes: int
    scratch_fixed_bytes: int
    scratch_context_bytes: int
    kv_pool_bytes: int
    workspace_lease_bytes: int
    transient_fixed_bytes: int
    transient_context_bytes: int
    private_workspace_bytes: int = 0
    # Prefill/packed execution scratch allocated after the free-memory reading
    # (bulk prefill scratch, packed verify state and scratch).
    prefill_workspace_bytes: int = 0
    # Speculative draft providers and the target's verify state rows.
    speculative_bytes: int = 0

    @property
    def retained_bytes(self) -> int:
        """Retained bytes live for the whole session at this context."""

        return (self.scratch_bytes + self.kv_pool_bytes + self.workspace_lease_bytes
                + self.private_workspace_bytes + self.prefill_workspace_bytes
                + self.speculative_bytes)

    @property
    def transient_bytes(self) -> int:
        return self.transient_fixed_bytes + self.transient_context_bytes

    @property
    def total_bytes(self) -> int:
        """Retained bytes plus the prefill transient this context can reach."""

        return self.retained_bytes + self.transient_bytes

    def to_json_dict(self) -> dict[str, object]:
        return {
            "context_tokens": self.context_tokens,
            "max_positions": self.max_positions,
            "block_size": self.block_size,
            "page_bytes": self.page_bytes,
            "pages_per_request": self.pages_per_request,
            "kv_pool_pages": self.kv_pool_pages,
            "workspace_lease_pages": self.workspace_lease_pages,
            "scratch_bytes": self.scratch_bytes,
            "scratch_fixed_bytes": self.scratch_fixed_bytes,
            "scratch_context_bytes": self.scratch_context_bytes,
            "kv_pool_bytes": self.kv_pool_bytes,
            "workspace_lease_bytes": self.workspace_lease_bytes,
            "private_workspace_bytes": self.private_workspace_bytes,
            "prefill_workspace_bytes": self.prefill_workspace_bytes,
            "speculative_bytes": self.speculative_bytes,
            "transient_fixed_bytes": self.transient_fixed_bytes,
            "transient_context_bytes": self.transient_context_bytes,
            "retained_bytes": self.retained_bytes,
            "transient_bytes": self.transient_bytes,
            "total_bytes": self.total_bytes,
        }

@dataclass(frozen=True)
class Qwen35GGUFKVCapacityEstimate:
    """Resident capacity estimate for the GGUF server path.

    Mirrors :class:`Qwen35ParoKVCapacityEstimate` in shape so server logging and
    ``/ready`` payloads can treat both routes the same way.
    """

    requested_context_tokens: int
    requested_context_tokens_rounded: int
    model_max_context_tokens: int
    allocatable_context_tokens: int
    available_bytes: int
    reserve_bytes: int
    usable_bytes: int
    retained_bytes: int
    transient_bytes: int
    requested_total_bytes: int
    marginal_bytes_per_token: int
    fixed_bytes: int
    scratch_bytes: int
    kv_pool_bytes: int
    workspace_lease_bytes: int
    page_bytes: int
    kv_pool_pages: int
    workspace_lease_pages: int
    block_size: int
    max_batch_size: int
    kv_storage_dtype: str
    kv_scale_dtype: str
    kv_scale_granularity: str
    int8_kv_no_mirror_qualified: bool
    workspace_lease_needed: bool
    private_workspace_bytes: int = 0
    prefill_workspace_bytes: int = 0
    speculative_bytes: int = 0

    @property
    def fits_requested(self) -> bool:
        return self.requested_total_bytes <= self.usable_bytes

    @property
    def fits_model_max(self) -> bool:
        return (
            self.model_max_context_tokens <= 0
            or self.allocatable_context_tokens >= self.model_max_context_tokens
        )

    # The server logs and /ready payloads were written against the PARO
    # estimate's field names. Expose the same names here so one reporting path
    # covers both resident routes instead of branching on the route.
    @property
    def bytes_per_token(self) -> int:
        return self.marginal_bytes_per_token

    @property
    def requested_kv_bytes(self) -> int:
        return self.kv_pool_bytes

    @property
    def requested_context_overhead_bytes(self) -> int:
        return (
            self.scratch_bytes
            + self.workspace_lease_bytes
            + self.private_workspace_bytes
            + self.prefill_workspace_bytes
            + self.speculative_bytes
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "requested_context_tokens": self.requested_context_tokens,
            "requested_context_tokens_rounded": self.requested_context_tokens_rounded,
            "model_max_context_tokens": self.model_max_context_tokens,
            "allocatable_context_tokens": self.allocatable_context_tokens,
            "available_bytes": self.available_bytes,
            "reserve_bytes": self.reserve_bytes,
            "usable_bytes": self.usable_bytes,
            "retained_bytes": self.retained_bytes,
            "transient_bytes": self.transient_bytes,
            "private_workspace_bytes": self.private_workspace_bytes,
            "prefill_workspace_bytes": self.prefill_workspace_bytes,
            "speculative_bytes": self.speculative_bytes,
            "requested_total_bytes": self.requested_total_bytes,
            "marginal_bytes_per_token": self.marginal_bytes_per_token,
            "fixed_bytes": self.fixed_bytes,
            "scratch_bytes": self.scratch_bytes,
            "kv_pool_bytes": self.kv_pool_bytes,
            "workspace_lease_bytes": self.workspace_lease_bytes,
            "page_bytes": self.page_bytes,
            "kv_pool_pages": self.kv_pool_pages,
            "workspace_lease_pages": self.workspace_lease_pages,
            "block_size": self.block_size,
            "max_batch_size": self.max_batch_size,
            "kv_storage_dtype": self.kv_storage_dtype,
            "kv_scale_dtype": self.kv_scale_dtype,
            "kv_scale_granularity": self.kv_scale_granularity,
            "int8_kv_no_mirror_qualified": self.int8_kv_no_mirror_qualified,
            "workspace_lease_needed": self.workspace_lease_needed,
            "fits_requested": self.fits_requested,
            "fits_model_max": self.fits_model_max,
        }

def qwen35_gguf_resident_breakdown(
    cfg: object,
    *,
    context_tokens: int,
    hidden_size: int,
    ffn_size: int,
    q_width: int,
    kv_width: int,
    linear_qkv_width: int,
    fp16_recurrent_state: bool = False,
    max_batch_size: int = 1,
    kv_storage_dtype: str | DType = DType.INT8_PER_TOKEN_HEAD,
    kv_storage_layout: str = "uniform",
    kv_scale_dtype: str | DType = DType.FP32,
    kv_scale_granularity: str = "per_token_head",
    int8_kv_value_bf16: bool = False,
    int8_bf16_full_attention_layer_indices: Sequence[int] = (),
    int8_kv_no_mirror_qualified: bool = False,
    int8_bf16_layer_indices_resolver: Callable[[int], Sequence[int]] | None = None,
    allocate_kv_cache: bool = False,
    workspace_lease_needed: bool = False,
    workspace_lease_slots: int | None = None,
    transient_bytes_per_token: int = _GGUF_TRANSIENT_BYTES_PER_TOKEN_DEFAULT,
    transient_fixed_bytes: int = _GGUF_TRANSIENT_FIXED_BYTES_DEFAULT,
    block_size: int = 256,
    kv_pool_floor_pages: int = 0,
    private_workspace_slots: int | None = None,
    transient_bytes_resolver: Callable[[int], int] | None = None,
    prefill_workspace_bytes_resolver: Callable[[int], int] | None = None,
    speculative_bytes_resolver: Callable[[int], int] | None = None,
    kv_request_slots: int | None = None,
) -> Qwen35GGUFResidentBreakdown:
    """Price one resident GGUF server configuration at a declared context.

    The retained terms come from the same sizing plan the allocator consumes, so
    this tracks KV policy, the INT8 mirror threshold, and per-layer BF16
    selection automatically. ``allocate_kv_cache`` must match the session: the
    server defers KV to the page pool and therefore plans scratch without it.
    ``workspace_lease_slots`` prices the independently pinned serving workspace;
    it defaults to ``max_batch_size`` for callers pricing a full batch.

    ``int8_bf16_layer_indices_resolver`` lets the caller resolve per-layer BF16
    selection from the *candidate* context, which is what the session does. It
    is called with the block-rounded ``max_positions`` of the priced context.

    The server route adds terms the session allocates after the free-memory
    reading: ``kv_pool_floor_pages`` is the pool's idle commit, which a short
    context still pays; ``private_workspace_slots`` prices full-context packed
    planes for that many long requests (default: every lease slot);
    the resolvers, each called with ``max_positions``, return the route's
    prefill transient (overriding the flat ``transient_bytes_per_token``),
    the prefill/packed execution scratch, and the speculative draft footprint.
    ``kv_request_slots`` prices request KV for fewer requests than
    ``max_batch_size`` sizes the resident scratch for: a deferred-KV server
    keeps per-slot recurrent state for its whole serving capacity while its
    elastic pool is priced for one full-context request.
    """

    block = int(block_size)
    plan = _full_stack_scratch_plan(
        cfg,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        q_width=q_width,
        kv_width=kv_width,
        linear_qkv_width=linear_qkv_width,
        fp16_recurrent_state=fp16_recurrent_state,
        max_sequence_length=int(context_tokens),
        max_batch_size=max_batch_size,
        kv_storage_dtype=kv_storage_dtype,
        kv_storage_layout=kv_storage_layout,
        kv_scale_dtype=kv_scale_dtype,
        kv_scale_granularity=kv_scale_granularity,
        int8_kv_value_bf16=int8_kv_value_bf16,
        int8_bf16_full_attention_layer_indices=tuple(int8_bf16_full_attention_layer_indices),
        retain_int8_bf16_mirrors=not bool(int8_kv_no_mirror_qualified),
        allocate_kv_cache=allocate_kv_cache,
        block_size=block,
        materialize=False,
    )
    resolved_layer_indices = (
        tuple(int8_bf16_full_attention_layer_indices)
        if int8_bf16_layer_indices_resolver is None
        else tuple(int8_bf16_layer_indices_resolver(int(plan.max_positions)))
    )
    layout = _qwen35_gguf_kv_chunk_layout(
        cfg,
        max_positions=plan.max_positions,
        kv_storage_dtype=kv_storage_dtype,
        kv_storage_layout=kv_storage_layout,
        kv_scale_dtype=kv_scale_dtype,
        kv_scale_granularity=kv_scale_granularity,
        int8_kv_value_bf16=int8_kv_value_bf16,
        int8_bf16_full_attention_layer_indices=resolved_layer_indices,
        int8_kv_no_mirror_qualified=int8_kv_no_mirror_qualified,
    )
    page_bytes = _qwen35_gguf_kv_page_bytes(cfg, layout)
    slots = max(1, int(max_batch_size))
    # ``kv_request_slots=0``: the scratch plan already owns the request KV
    # (DMS sessions allocate their own dense prefill pool; there is no global
    # pool to price on top of it).
    kv_slots = slots if kv_request_slots is None else max(0, int(kv_request_slots))
    pages_per_request = max(1, (plan.max_positions + block - 1) // block)
    kv_pool_pages = max(kv_slots * pages_per_request, max(0, int(kv_pool_floor_pages)))
    lease_slots = (
        max(1, kv_slots) if workspace_lease_slots is None else max(1, int(workspace_lease_slots))
    )
    floor_pages = (_PACKED_VERIFY_MIN_MAX_SEQUENCE + block - 1) // block
    workspace_lease_pages = lease_slots * floor_pages if workspace_lease_needed else 0
    # Private growth cannot reuse the undersized pinned lease. Price both
    # holdings, rather than making auto-context optimistic after bounding it.
    private_slots = (
        lease_slots if private_workspace_slots is None else max(0, int(private_workspace_slots))
    )
    private_workspace_bytes = (
        private_slots * pages_per_request * page_bytes
        if workspace_lease_needed and pages_per_request > floor_pages else 0
    )
    context = int(context_tokens)
    transient_context_bytes = (
        max(0, int(transient_bytes_resolver(int(plan.max_positions))))
        if transient_bytes_resolver is not None
        else max(0, int(transient_bytes_per_token)) * context
    )
    return Qwen35GGUFResidentBreakdown(
        context_tokens=context,
        max_positions=plan.max_positions,
        block_size=block,
        page_bytes=page_bytes,
        pages_per_request=pages_per_request,
        kv_pool_pages=kv_pool_pages,
        workspace_lease_pages=workspace_lease_pages,
        scratch_bytes=plan.dedicated_bytes,
        scratch_fixed_bytes=plan.fixed_bytes,
        scratch_context_bytes=plan.context_scaled_bytes,
        kv_pool_bytes=kv_pool_pages * page_bytes,
        workspace_lease_bytes=workspace_lease_pages * page_bytes,
        private_workspace_bytes=private_workspace_bytes,
        transient_fixed_bytes=max(0, int(transient_fixed_bytes)),
        transient_context_bytes=transient_context_bytes,
        prefill_workspace_bytes=(
            0
            if prefill_workspace_bytes_resolver is None
            else max(0, int(prefill_workspace_bytes_resolver(int(plan.max_positions))))
        ),
        speculative_bytes=(
            0
            if speculative_bytes_resolver is None
            else max(0, int(speculative_bytes_resolver(int(plan.max_positions))))
        ),
    )

def _gguf_largest_block_aligned_context(
    fits: Callable[[int], bool],
    *,
    low_tokens: int,
    high_tokens: int,
    block_size: int,
) -> int:
    """Largest block-aligned token count in ``[low, high]`` that ``fits``.

    Callers must guarantee ``fits`` is monotonic non-decreasing over the range.
    """

    block = max(1, int(block_size))
    low_blocks = max(1, int(low_tokens) // block)
    high_blocks = max(low_blocks, int(high_tokens) // block)
    best = 0
    while low_blocks <= high_blocks:
        mid = (low_blocks + high_blocks) // 2
        if fits(mid * block):
            best = mid * block
            low_blocks = mid + 1
        else:
            high_blocks = mid - 1
    return best

def estimate_qwen35_gguf_kv_capacity(
    cfg: object,
    *,
    available_bytes: int,
    requested_context_tokens: int,
    hidden_size: int,
    ffn_size: int,
    q_width: int,
    kv_width: int,
    linear_qkv_width: int,
    fp16_recurrent_state: bool = False,
    max_batch_size: int = 1,
    kv_storage_dtype: str | DType = DType.INT8_PER_TOKEN_HEAD,
    kv_storage_layout: str = "uniform",
    kv_scale_dtype: str | DType = DType.FP32,
    kv_scale_granularity: str = "per_token_head",
    int8_kv_value_bf16: bool = False,
    int8_bf16_full_attention_layer_indices: Sequence[int] = (),
    int8_kv_no_mirror_qualified: bool = False,
    int8_bf16_layer_indices_resolver: Callable[[int], Sequence[int]] | None = None,
    allocate_kv_cache: bool = False,
    workspace_lease_needed: bool = False,
    workspace_lease_slots: int | None = None,
    reserve_bytes: int = _GGUF_CAPACITY_RESERVE_MIB_DEFAULT * 1024**2,
    transient_bytes_per_token: int = _GGUF_TRANSIENT_BYTES_PER_TOKEN_DEFAULT,
    transient_fixed_bytes: int = _GGUF_TRANSIENT_FIXED_BYTES_DEFAULT,
    block_size: int = 256,
    kv_pool_floor_pages: int = 0,
    private_workspace_slots: int | None = None,
    transient_bytes_resolver: Callable[[int], int] | None = None,
    prefill_workspace_bytes_resolver: Callable[[int], int] | None = None,
    speculative_bytes_resolver: Callable[[int], int] | None = None,
    kv_request_slots: int | None = None,
) -> Qwen35GGUFKVCapacityEstimate:
    """Estimate the largest resident context that fits in ``available_bytes``.

    ``available_bytes`` is free HIP memory measured *after* resident weights
    load, so weights are excluded from the model rather than guessed at. The
    transient terms are represented by ``transient_fixed_bytes`` and
    ``transient_bytes_per_token``; the flat ``reserve_bytes`` covers allocator
    granularity and unaudited overhead.

    The footprint is not globally monotonic: below
    ``_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS`` the INT8 route retains BF16
    mirrors, so the curve steps down at the threshold. The search therefore
    covers the mirror-free region first and only then the mirror region, which
    keeps each bisection monotonic.
    """

    block = max(1, int(block_size))
    requested = int(requested_context_tokens)
    if requested <= 0:
        raise ValueError("requested_context_tokens must be positive")
    available = max(0, int(available_bytes))
    reserve = max(0, int(reserve_bytes))
    usable = max(0, available - reserve)
    model_max = int(getattr(cfg, "context_length", 0) or 0)

    def price(context: int) -> Qwen35GGUFResidentBreakdown:
        return qwen35_gguf_resident_breakdown(
            cfg,
            context_tokens=context,
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            q_width=q_width,
            kv_width=kv_width,
            linear_qkv_width=linear_qkv_width,
            fp16_recurrent_state=fp16_recurrent_state,
            max_batch_size=max_batch_size,
            kv_storage_dtype=kv_storage_dtype,
            kv_storage_layout=kv_storage_layout,
            kv_scale_dtype=kv_scale_dtype,
            kv_scale_granularity=kv_scale_granularity,
            int8_kv_value_bf16=int8_kv_value_bf16,
            int8_bf16_full_attention_layer_indices=int8_bf16_full_attention_layer_indices,
            int8_kv_no_mirror_qualified=int8_kv_no_mirror_qualified,
            int8_bf16_layer_indices_resolver=int8_bf16_layer_indices_resolver,
            allocate_kv_cache=allocate_kv_cache,
            workspace_lease_needed=workspace_lease_needed,
            workspace_lease_slots=workspace_lease_slots,
            transient_bytes_per_token=transient_bytes_per_token,
            transient_fixed_bytes=transient_fixed_bytes,
            block_size=block,
            kv_pool_floor_pages=kv_pool_floor_pages,
            private_workspace_slots=private_workspace_slots,
            transient_bytes_resolver=transient_bytes_resolver,
            prefill_workspace_bytes_resolver=prefill_workspace_bytes_resolver,
            speculative_bytes_resolver=speculative_bytes_resolver,
            kv_request_slots=kv_request_slots,
        )

    def fits(context: int) -> bool:
        return price(context).total_bytes <= usable

    requested_rounded = ((requested + block - 1) // block) * block
    if model_max > 0:
        requested_rounded = min(requested_rounded, model_max)
    ceiling = model_max if model_max > 0 else max(block, requested_rounded)
    if model_max > 0 and not fits(ceiling):
        # The INT8 mirror step makes the curve non-monotonic at the threshold:
        # a context *below* it carries BF16 mirrors and costs more than the first
        # context above it. Search the mirror-free region first, then the mirror
        # region, so each bisection is over a monotonic range.
        mirror_threshold = min(_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS, ceiling)
        mirror_free_floor = min(mirror_threshold + block, ceiling)
        if mirror_free_floor > mirror_threshold and fits(mirror_free_floor):
            allocatable = _gguf_largest_block_aligned_context(
                fits,
                low_tokens=mirror_free_floor,
                high_tokens=ceiling,
                block_size=block,
            )
        else:
            allocatable = _gguf_largest_block_aligned_context(
                fits,
                low_tokens=min(block, mirror_threshold),
                high_tokens=mirror_threshold,
                block_size=block,
            )
    elif model_max > 0:
        allocatable = model_max
    else:
        allocatable = _gguf_largest_block_aligned_context(
            fits,
            low_tokens=block,
            high_tokens=ceiling,
            block_size=block,
        )

    requested_breakdown = price(requested_rounded)
    marginal = 0
    if allocatable > block:
        # Probe downward: the upper end of the range may already sit on the
        # model maximum, where there is no larger context to price.
        lower = allocatable - block
        delta = price(allocatable).total_bytes - price(lower).total_bytes
        marginal = delta // (allocatable - lower)
    layout = DType.parse(kv_storage_dtype)
    scale = DType.parse(kv_scale_dtype)
    return Qwen35GGUFKVCapacityEstimate(
        requested_context_tokens=requested,
        requested_context_tokens_rounded=requested_rounded,
        model_max_context_tokens=model_max,
        allocatable_context_tokens=int(allocatable),
        available_bytes=available,
        reserve_bytes=reserve,
        usable_bytes=usable,
        retained_bytes=requested_breakdown.retained_bytes,
        transient_bytes=requested_breakdown.transient_bytes,
        requested_total_bytes=requested_breakdown.total_bytes,
        marginal_bytes_per_token=int(marginal),
        fixed_bytes=int(requested_breakdown.scratch_fixed_bytes)
        + int(requested_breakdown.transient_fixed_bytes),
        scratch_bytes=requested_breakdown.scratch_bytes,
        kv_pool_bytes=requested_breakdown.kv_pool_bytes,
        workspace_lease_bytes=requested_breakdown.workspace_lease_bytes,
        page_bytes=requested_breakdown.page_bytes,
        kv_pool_pages=requested_breakdown.kv_pool_pages,
        workspace_lease_pages=requested_breakdown.workspace_lease_pages,
        block_size=block,
        max_batch_size=max(1, int(max_batch_size)),
        kv_storage_dtype=layout.value,
        kv_scale_dtype=scale.value,
        kv_scale_granularity=str(kv_scale_granularity),
        int8_kv_no_mirror_qualified=bool(int8_kv_no_mirror_qualified),
        workspace_lease_needed=bool(workspace_lease_needed),
        private_workspace_bytes=requested_breakdown.private_workspace_bytes,
        prefill_workspace_bytes=requested_breakdown.prefill_workspace_bytes,
        speculative_bytes=requested_breakdown.speculative_bytes,
    )

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
