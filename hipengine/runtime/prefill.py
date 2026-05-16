"""Native prefill configuration objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefillConfig:
    """Configuration for Qwen3.5/PARO native bulk prefill.

    The defaults describe the final retained path: full-native prefill is
    required unless a caller explicitly opts into bring-up/oracle behavior.
    Chunk sizes of ``0`` mean unchunked, matching the parent environment-knob
    convention.
    """

    linear_chunk_size: int = 0
    full_attn_query_chunk_size: int = 0
    full_attn_post_chunk_size: int = 0
    full_attn_rope_chunk_size: int = 0
    moe_chunk_size: int = 0
    attn_aotriton_min_tokens: int = 0
    moe_grouped_device_gather: bool = True
    moe_stacked_compact: bool = True
    require_full_native: bool = True

    def __post_init__(self) -> None:
        for name in (
            "linear_chunk_size",
            "full_attn_query_chunk_size",
            "full_attn_post_chunk_size",
            "full_attn_rope_chunk_size",
            "moe_chunk_size",
            "attn_aotriton_min_tokens",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "moe_grouped_device_gather", bool(self.moe_grouped_device_gather))
        object.__setattr__(self, "moe_stacked_compact", bool(self.moe_stacked_compact))
        object.__setattr__(self, "require_full_native", bool(self.require_full_native))
