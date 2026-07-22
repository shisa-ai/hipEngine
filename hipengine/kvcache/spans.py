"""KV live-span value objects.

The engine and kernel wrappers pass KV layout through ``KVLiveSpans`` rather than
classic ``(block_table, context_len)`` tuples. Fixed-page policies use uniform
spans; DMS-like policies can later fill per-head-variable fields without changing
attention dispatch signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor

_SPAN_ROLES = {"prefill", "decode", "verify_chain", "verify_tree"}
_SCALE_GRANULARITIES = {"per_token_head", "block16", "hadamard_group32"}
_SCALE_DTYPES = {DType.FP16, DType.FP32}


@dataclass(frozen=True, slots=True)
class KVScaleMetadata:
    """Scale tensors associated with a quantized KV arena."""

    k_scale: Tensor
    v_scale: Tensor
    scale_dtype: DType = DType.FP16
    granularity: str = "per_token_head"

    def __post_init__(self) -> None:
        if self.k_scale.device != self.v_scale.device:
            raise ValueError("k_scale and v_scale must be on the same device")
        if self.k_scale.shape != self.v_scale.shape:
            raise ValueError("k_scale and v_scale must have the same shape")
        if self.k_scale.numel <= 0:
            raise ValueError("scale tensors must not be empty")
        parsed = DType.parse(self.scale_dtype)
        object.__setattr__(self, "scale_dtype", parsed)
        if parsed not in _SCALE_DTYPES:
            raise ValueError("scale_dtype must be fp16 or fp32")
        if self.k_scale.dtype != parsed or self.v_scale.dtype != parsed:
            raise ValueError("scale tensor dtypes must match scale_dtype")
        if self.granularity not in _SCALE_GRANULARITIES:
            raise ValueError(
                "scale granularity must be per_token_head, block16, or hadamard_group32"
            )

    @property
    def device(self):  # intentionally mirrors Tensor.device without importing Device here
        return self.k_scale.device


@dataclass(frozen=True, slots=True)
class KVLiveSpans:
    """Per-sequence/layer/head live K/V token-span metadata.

    ``base_offsets`` and ``live_counts`` are always present. Fixed-page spans
    carry the device page table and parent position/count tensor. Token-granular
    sliding rings additionally require absolute per-slot ``token_positions``,
    an ``evict_mask``, and the current absolute ``row_positions``. Public
    wrappers always receive one ``KVLiveSpans`` object, so callers do not depend
    on a backend's block-table or ring-slot API.
    """

    base_offsets: Tensor
    live_counts: Tensor
    max_live_count: int
    token_positions: Tensor | None
    evict_mask: Tensor | None
    storage_dtype: DType
    spans_mode: str = "uniform"
    request_ids: Tensor | None = None
    row_positions: Tensor | None = None
    span_role: str = "decode"
    scale_metadata: KVScaleMetadata | None = None

    def __post_init__(self) -> None:
        if self.base_offsets.device != self.live_counts.device:
            raise ValueError("base_offsets and live_counts must be on the same device")
        if (
            self.token_positions is not None
            and self.token_positions.device != self.base_offsets.device
        ):
            raise ValueError("token_positions must be on the same device as base_offsets")
        if self.token_positions is not None and self.token_positions.dtype not in {
            DType.INT32,
            DType.INT64,
        }:
            raise ValueError("token_positions must be int32 or int64")
        if self.evict_mask is not None and self.evict_mask.device != self.base_offsets.device:
            raise ValueError("evict_mask must be on the same device as base_offsets")
        if self.request_ids is not None and self.request_ids.device != self.base_offsets.device:
            raise ValueError("request_ids must be on the same device as base_offsets")
        if self.row_positions is not None and self.row_positions.device != self.base_offsets.device:
            raise ValueError("row_positions must be on the same device as base_offsets")
        if (
            self.scale_metadata is not None
            and self.scale_metadata.device != self.base_offsets.device
        ):
            raise ValueError("scale metadata must be on the same device as base_offsets")
        if self.base_offsets.dtype != DType.INT32:
            raise ValueError("base_offsets must be int32")
        if self.live_counts.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("live_counts must be int32 or int64")
        if self.evict_mask is not None and self.evict_mask.dtype != DType.BOOL:
            raise ValueError("evict_mask must be bool")
        if self.request_ids is not None and self.request_ids.dtype != DType.INT64:
            raise ValueError("request_ids must be int64")
        if self.row_positions is not None and self.row_positions.dtype not in {
            DType.INT32,
            DType.INT64,
        }:
            raise ValueError("row_positions must be int32 or int64")
        if self.request_ids is not None and self.request_ids.numel != self.live_counts.numel:
            raise ValueError("request_ids must have one entry per live_counts row")
        if self.row_positions is not None and self.row_positions.numel != self.live_counts.numel:
            raise ValueError("row_positions must have one entry per live_counts row")
        if self.max_live_count < 0:
            raise ValueError("max_live_count must be non-negative")
        storage = DType.parse(self.storage_dtype)
        object.__setattr__(self, "storage_dtype", storage)
        if self.spans_mode not in {"uniform", "per_head_variable", "sliding_ring"}:
            raise ValueError("spans_mode must be 'uniform', 'per_head_variable', or 'sliding_ring'")
        if self.spans_mode == "sliding_ring":
            if self.storage_dtype != DType.BF16:
                raise ValueError("sliding_ring spans require bf16 storage")
            if self.live_counts.dtype != DType.INT64 or self.live_counts.numel != 1:
                raise ValueError("sliding_ring live_counts must be one int64 scalar")
            if self.token_positions is None or self.token_positions.dtype != DType.INT64:
                raise ValueError("sliding_ring token_positions must be present and int64")
            if self.evict_mask is None:
                raise ValueError("sliding_ring evict_mask must be present")
            if self.row_positions is None or self.row_positions.dtype != DType.INT64:
                raise ValueError("sliding_ring row_positions must be present and int64")
            capacity = self.base_offsets.numel
            if capacity <= 0:
                raise ValueError("sliding_ring capacity must be positive")
            if self.max_live_count != capacity:
                raise ValueError("sliding_ring max_live_count must equal capacity")
            if self.token_positions.numel != capacity:
                raise ValueError(
                    "sliding_ring token_positions must have one entry per capacity slot"
                )
            if self.evict_mask.numel != capacity:
                raise ValueError("sliding_ring evict_mask must have one entry per capacity slot")
            if self.row_positions.numel != 1:
                raise ValueError("sliding_ring row_positions must be one int64 scalar")
        if self.span_role not in _SPAN_ROLES:
            raise ValueError("span_role must be one of prefill, decode, verify_chain, verify_tree")
        if storage == DType.INT8_PER_TOKEN_HEAD and self.scale_metadata is None:
            raise ValueError("int8_per_token_head spans require scale metadata")
        if storage != DType.INT8_PER_TOKEN_HEAD and self.scale_metadata is not None:
            raise ValueError("scale metadata is only valid for int8_per_token_head spans")

    @classmethod
    def paged_uniform(
        cls,
        *,
        block_table: Tensor,
        live_counts: Tensor,
        max_live_count: int,
        storage_dtype: str | DType,
        request_ids: Tensor | None = None,
        row_positions: Tensor | None = None,
        span_role: str = "decode",
        scale_metadata: KVScaleMetadata | None = None,
    ) -> "KVLiveSpans":
        """Build uniform fixed-page spans for parent paged kernels.

        ``block_table`` is carried as ``base_offsets`` because the parent gfx1100
        paged kernels already use a physical-block indirection table. The field
        name stays span-oriented at the public boundary.
        """

        return cls(
            base_offsets=block_table,
            live_counts=live_counts,
            max_live_count=max_live_count,
            token_positions=None,
            evict_mask=None,
            storage_dtype=DType.parse(storage_dtype),
            spans_mode="uniform",
            request_ids=request_ids,
            row_positions=row_positions,
            span_role=span_role,
            scale_metadata=scale_metadata,
        )

    @classmethod
    def paged_dense(
        cls,
        *,
        block_table: Tensor,
        live_counts: Tensor,
        token_positions: Tensor,
        evict_mask: Tensor,
        row_positions: Tensor,
        capacity: int,
        block_size: int,
        storage_dtype: str | DType = DType.BF16,
        request_ids: Tensor | None = None,
        span_role: str = "decode",
    ) -> "KVLiveSpans":
        """Build complete dense spans over a physical page table."""

        parsed_capacity = int(capacity)
        parsed_block = int(block_size)
        if parsed_capacity <= 0 or parsed_block <= 0:
            raise ValueError("paged_dense capacity and block_size must be positive")
        if block_table.numel * parsed_block < parsed_capacity:
            raise ValueError("paged_dense block table is too short for capacity")
        if live_counts.dtype != DType.INT64 or live_counts.numel != 1:
            raise ValueError("paged_dense live_counts must be one int64 scalar")
        if token_positions.dtype != DType.INT64:
            raise ValueError("paged_dense token_positions must be int64")
        if row_positions.dtype != DType.INT64 or row_positions.numel != 1:
            raise ValueError("paged_dense row_positions must be one int64 scalar")
        if token_positions.numel != parsed_capacity:
            raise ValueError("paged_dense token_positions must match capacity")
        if evict_mask.numel != parsed_capacity:
            raise ValueError("paged_dense evict_mask must match capacity")
        return cls(
            base_offsets=block_table,
            live_counts=live_counts,
            max_live_count=parsed_capacity,
            token_positions=token_positions,
            evict_mask=evict_mask,
            storage_dtype=DType.parse(storage_dtype),
            spans_mode="uniform",
            request_ids=request_ids,
            row_positions=row_positions,
            span_role=span_role,
        )

    @classmethod
    def sliding_ring(
        cls,
        *,
        base_offsets: Tensor,
        live_counts: Tensor,
        token_positions: Tensor,
        evict_mask: Tensor,
        row_positions: Tensor,
        capacity: int,
        storage_dtype: str | DType = DType.BF16,
        request_ids: Tensor | None = None,
        span_role: str = "decode",
    ) -> "KVLiveSpans":
        """Build token-granular sliding-window ring spans.

        ``base_offsets`` maps logical ring slots to physical cache slots.
        ``token_positions`` stores each slot's absolute position, while
        ``evict_mask`` marks invalid or explicitly evicted slots.
        ``row_positions`` is the current absolute query/append position.
        """

        parsed_capacity = int(capacity)
        if parsed_capacity <= 0:
            raise ValueError("sliding_ring capacity must be positive")
        for name, tensor in (
            ("base_offsets", base_offsets),
            ("token_positions", token_positions),
            ("evict_mask", evict_mask),
        ):
            if tensor.numel != parsed_capacity:
                raise ValueError(f"sliding_ring {name} must have one entry per capacity slot")
        return cls(
            base_offsets=base_offsets,
            live_counts=live_counts,
            max_live_count=parsed_capacity,
            token_positions=token_positions,
            evict_mask=evict_mask,
            storage_dtype=DType.parse(storage_dtype),
            spans_mode="sliding_ring",
            request_ids=request_ids,
            row_positions=row_positions,
            span_role=span_role,
        )

    @property
    def device(self):  # intentionally mirrors Tensor.device without importing Device here
        return self.base_offsets.device
