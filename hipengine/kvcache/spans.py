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

    def __post_init__(self) -> None:
        if self.base_offsets.device != self.live_counts.device:
            raise ValueError("base_offsets and live_counts must be on the same device")
        if self.token_positions is not None and self.token_positions.device != self.base_offsets.device:
            raise ValueError("token_positions must be on the same device as base_offsets")
        if self.evict_mask is not None and self.evict_mask.device != self.base_offsets.device:
            raise ValueError("evict_mask must be on the same device as base_offsets")
        if self.base_offsets.dtype != DType.INT32:
            raise ValueError("base_offsets must be int32")
        if self.live_counts.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("live_counts must be int32 or int64")
        if self.evict_mask is not None and self.evict_mask.dtype != DType.BOOL:
            raise ValueError("evict_mask must be bool")
        if self.max_live_count < 0:
            raise ValueError("max_live_count must be non-negative")
        if self.spans_mode not in {"uniform", "per_head_variable"}:
            raise ValueError("spans_mode must be 'uniform' or 'per_head_variable'")
        DType.parse(self.storage_dtype)

    @classmethod
    def paged_uniform(
        cls,
        *,
        block_table: Tensor,
        live_counts: Tensor,
        max_live_count: int,
        storage_dtype: str | DType,
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
        )

    @property
    def device(self):  # intentionally mirrors Tensor.device without importing Device here
        return self.base_offsets.device
