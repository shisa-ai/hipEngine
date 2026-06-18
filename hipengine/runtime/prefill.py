"""Native prefill configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, replace

_GIB = 1024 ** 3
_CHUNKED_PREFILL_MIN_TOKENS = 1025
_PREFILL_LINEAR_CHUNK = 1024
_PREFILL_MOE_CHUNK = 1024
_PREFILL_FULL_ATTN_QUERY_CHUNK = 4096
_PREFILL_FULL_ATTN_POST_CHUNK = 1024
_PREFILL_FULL_ATTN_ROPE_CHUNK = 1024
_PREFILL_LOW_MEMORY_CHUNK = 768
_PREFILL_LOW_MEMORY_FULL_ATTN_QUERY_CHUNK = 768
_LOW_MEMORY_MID_CONTEXT_FULL_ATTN_QUERY_CHUNK = 1024
_LOW_MEMORY_MID_CONTEXT_MIN_TOKENS = 52 * 1024
_LOW_MEMORY_FULL_CONTEXT_MIN_TOKENS = 131_072
_LOW_MEMORY_TOTAL_BYTES = 26 * _GIB
_DEFAULT_BUDGET_FRACTION = 0.55


@dataclass(frozen=True, slots=True)
class PrefillConfig:
    """Configuration for Qwen3.5/PARO native bulk prefill.

    The defaults describe the final retained path: full-native prefill is
    required unless a caller explicitly opts into bring-up/oracle behavior.
    Chunk sizes of ``0`` mean "no manual override", matching the parent
    environment-knob convention; with ``auto_tune_chunk_sizes`` enabled,
    prompts above 1K resolve those zeros to the retained 1024/4096 chunk
    policy, except 52K-class prompts on 24GB-class cards reduce the
    full-attention query chunk to 1024 rows and 128K-class prompts use
    conservative 768-token chunks to preserve transient scratch headroom while
    keeping AOTriton on for long-context full attention.  AOTriton is
    a baseline vendored runtime dependency for the gfx1100
    Qwen3.5/PARO path; the measured crossover policy uses AOTriton attention
    once prompts reach 512 tokens.
    """

    linear_chunk_size: int = 0
    full_attn_query_chunk_size: int = 0
    full_attn_post_chunk_size: int = 0
    full_attn_rope_chunk_size: int = 0
    moe_chunk_size: int = 0
    attn_aotriton_min_tokens: int = 512
    auto_tune_chunk_sizes: bool = True
    chunk_tune_min_tokens: int = _CHUNKED_PREFILL_MIN_TOKENS
    chunk_tune_memory_budget_gib: float = 0.0
    moe_grouped_device_gather: bool = True
    moe_stacked_compact: bool = True
    require_full_native: bool = True
    use_wmma_prefill: bool = False
    """Opt in to the GGUF WMMA batched prefill kernel family (P8).

    When ``True``, GGUF rows>1 dispatch in :mod:`hipengine.runtime.gguf_linear`
    rewrites the supported ``prefill_*`` variants (currently ``gguf_q8_0``)
    to the matching ``wmma_prefill_*`` registry keys. Otherwise the existing
    decode-shaped ``prefill_*`` aliases are used (see
    ``docs/GGUF.md`` "P8: real batched prefill GEMM"). Defaults to ``False``
    so the rollout can be correctness-bisected; the env var
    ``HIPENGINE_GGUF_WMMA_PREFILL=1`` provides an equivalent process-wide
    override that does not require a config change."""

    def __post_init__(self) -> None:
        for name in (
            "linear_chunk_size",
            "full_attn_query_chunk_size",
            "full_attn_post_chunk_size",
            "full_attn_rope_chunk_size",
            "moe_chunk_size",
            "attn_aotriton_min_tokens",
            "chunk_tune_min_tokens",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        budget = float(self.chunk_tune_memory_budget_gib)
        if budget < 0.0:
            raise ValueError("chunk_tune_memory_budget_gib must be non-negative")
        object.__setattr__(self, "chunk_tune_memory_budget_gib", budget)
        object.__setattr__(self, "auto_tune_chunk_sizes", bool(self.auto_tune_chunk_sizes))
        object.__setattr__(self, "moe_grouped_device_gather", bool(self.moe_grouped_device_gather))
        object.__setattr__(self, "moe_stacked_compact", bool(self.moe_stacked_compact))
        object.__setattr__(self, "require_full_native", bool(self.require_full_native))
        object.__setattr__(self, "use_wmma_prefill", bool(self.use_wmma_prefill))


def resolve_prefill_config_for_sequence(
    config: PrefillConfig,
    *,
    max_sequence_length: int,
    total_memory_bytes: int = 0,
) -> tuple[PrefillConfig, dict[str, object]]:
    """Resolve profile-derived chunk sizes for single-request prefill.

    Explicit non-zero chunk sizes are treated as manual overrides.  With the
    default auto policy, prompts up to 1K stay unchunked while prompts above 1K
    use the retained 1024/4096 policy across linear attention, MoE, full-attn
    query, post, and RoPE stages.  On 24GB-class cards, 52K-class prompts drop
    the full-attn query chunk to 1024 rows to avoid the 4096-row bulk-scratch
    cliff; 128K-class and longer prompts use 768-token chunks to keep the
    AOTriton path active and transient prefill scratch under the device limit.
    """

    max_sequence = int(max_sequence_length)
    if max_sequence <= 0:
        raise ValueError("max_sequence_length must be positive")
    tuning: dict[str, object] = {
        "enabled": bool(config.auto_tune_chunk_sizes),
        "applied": False,
        "reason": "disabled",
        "max_sequence_length": max_sequence,
        "source": "chunk_sweep_2026_05_midcontext_manual_long_equiv",
        "memory_budget_gib": 0.0,
    }
    if not config.auto_tune_chunk_sizes:
        return config, tuning
    if _has_manual_chunk_sizes(config):
        tuning["reason"] = "manual_chunk_sizes"
        return config, tuning
    if max_sequence < int(config.chunk_tune_min_tokens):
        tuning["reason"] = "below_min_tokens"
        return config, tuning

    budget_gib = _chunk_memory_budget_gib(config, total_memory_bytes=total_memory_bytes)
    low_memory_card = 0 < int(total_memory_bytes) <= _LOW_MEMORY_TOTAL_BYTES
    low_memory_full_context = low_memory_card and max_sequence >= _LOW_MEMORY_FULL_CONTEXT_MIN_TOKENS
    low_memory_mid_context = low_memory_card and max_sequence >= _LOW_MEMORY_MID_CONTEXT_MIN_TOKENS
    if low_memory_full_context:
        estimated_peak_gib = 23.4
        reason = "low_memory_full_context_24gb"
        tuned = replace(
            config,
            linear_chunk_size=_PREFILL_LOW_MEMORY_CHUNK,
            moe_chunk_size=_PREFILL_LOW_MEMORY_CHUNK,
            full_attn_query_chunk_size=_PREFILL_LOW_MEMORY_FULL_ATTN_QUERY_CHUNK,
            full_attn_post_chunk_size=_PREFILL_LOW_MEMORY_CHUNK,
            full_attn_rope_chunk_size=_PREFILL_LOW_MEMORY_CHUNK,
        )
    elif low_memory_mid_context:
        estimated_peak_gib = 23.0
        reason = "low_memory_mid_context_24gb"
        tuned = replace(
            config,
            linear_chunk_size=_PREFILL_LINEAR_CHUNK,
            moe_chunk_size=_PREFILL_MOE_CHUNK,
            full_attn_query_chunk_size=_LOW_MEMORY_MID_CONTEXT_FULL_ATTN_QUERY_CHUNK,
            full_attn_post_chunk_size=_PREFILL_FULL_ATTN_POST_CHUNK,
            full_attn_rope_chunk_size=_PREFILL_FULL_ATTN_ROPE_CHUNK,
        )
    else:
        estimated_peak_gib = 20.0
        reason = "manual_long_equiv_gt1k"
        tuned = replace(
            config,
            linear_chunk_size=_PREFILL_LINEAR_CHUNK,
            moe_chunk_size=_PREFILL_MOE_CHUNK,
            full_attn_query_chunk_size=_PREFILL_FULL_ATTN_QUERY_CHUNK,
            full_attn_post_chunk_size=_PREFILL_FULL_ATTN_POST_CHUNK,
            full_attn_rope_chunk_size=_PREFILL_FULL_ATTN_ROPE_CHUNK,
        )
    tuning.update(
        {
            "applied": True,
            "reason": reason,
            "memory_budget_gib": budget_gib,
            "estimated_peak_gib": estimated_peak_gib,
            "chunk_sizes": _chunk_sizes_dict(tuned),
        }
    )
    return tuned, tuning


def _has_manual_chunk_sizes(config: PrefillConfig) -> bool:
    return any(
        int(getattr(config, name)) > 0
        for name in (
            "linear_chunk_size",
            "moe_chunk_size",
            "full_attn_query_chunk_size",
            "full_attn_post_chunk_size",
            "full_attn_rope_chunk_size",
        )
    )


def _chunk_memory_budget_gib(config: PrefillConfig, *, total_memory_bytes: int) -> float:
    if config.chunk_tune_memory_budget_gib > 0.0:
        return float(config.chunk_tune_memory_budget_gib)
    if total_memory_bytes <= 0:
        return 0.0
    return (float(total_memory_bytes) / float(_GIB)) * _DEFAULT_BUDGET_FRACTION


def _chunk_sizes_dict(config: PrefillConfig) -> dict[str, int]:
    return {
        "linear": int(config.linear_chunk_size),
        "moe": int(config.moe_chunk_size),
        "full_attn_query": int(config.full_attn_query_chunk_size),
        "full_attn_post": int(config.full_attn_post_chunk_size),
        "full_attn_rope": int(config.full_attn_rope_chunk_size),
    }
