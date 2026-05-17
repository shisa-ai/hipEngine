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
_SCALE_GRANULARITIES = {"per_token_head"}
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
            raise ValueError("scale granularity must be per_token_head")

    @property
    def device(self):  # intentionally mirrors Tensor.device without importing Device here
        return self.k_scale.device


@dataclass(frozen=True, slots=True)
class KVLiveSpans:
    """Per-sequence/layer/head live K/V token-span metadata.

    ``base_offsets`` and ``live_counts`` are always present. For the current
    fixed-page gfx1100 bridge, ``base_offsets`` is the device page table used by
    the preserved parent paged kernels and ``live_counts`` is the device scalar
    position/count tensor consumed by the parent position-tensor writer. The
    public wrapper still receives one ``KVLiveSpans`` object so callers do not
    depend on the parent block-table API.
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
        if self.token_positions is not None and self.token_positions.device != self.base_offsets.device:
            raise ValueError("token_positions must be on the same device as base_offsets")
        if self.evict_mask is not None and self.evict_mask.device != self.base_offsets.device:
            raise ValueError("evict_mask must be on the same device as base_offsets")
        if self.request_ids is not None and self.request_ids.device != self.base_offsets.device:
            raise ValueError("request_ids must be on the same device as base_offsets")
        if self.row_positions is not None and self.row_positions.device != self.base_offsets.device:
            raise ValueError("row_positions must be on the same device as base_offsets")
        if self.scale_metadata is not None and self.scale_metadata.device != self.base_offsets.device:
            raise ValueError("scale metadata must be on the same device as base_offsets")
        if self.base_offsets.dtype != DType.INT32:
            raise ValueError("base_offsets must be int32")
        if self.live_counts.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("live_counts must be int32 or int64")
        if self.evict_mask is not None and self.evict_mask.dtype != DType.BOOL:
            raise ValueError("evict_mask must be bool")
        if self.request_ids is not None and self.request_ids.dtype != DType.INT64:
            raise ValueError("request_ids must be int64")
        if self.row_positions is not None and self.row_positions.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("row_positions must be int32 or int64")
        if self.request_ids is not None and self.request_ids.numel != self.live_counts.numel:
            raise ValueError("request_ids must have one entry per live_counts row")
        if self.row_positions is not None and self.row_positions.numel != self.live_counts.numel:
            raise ValueError("row_positions must have one entry per live_counts row")
        if self.max_live_count < 0:
            raise ValueError("max_live_count must be non-negative")
        if self.spans_mode not in {"uniform", "per_head_variable"}:
            raise ValueError("spans_mode must be 'uniform' or 'per_head_variable'")
        if self.span_role not in _SPAN_ROLES:
            raise ValueError("span_role must be one of prefill, decode, verify_chain, verify_tree")
        storage = DType.parse(self.storage_dtype)
        object.__setattr__(self, "storage_dtype", storage)
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

    @property
    def device(self):  # intentionally mirrors Tensor.device without importing Device here
        return self.base_offsets.device
