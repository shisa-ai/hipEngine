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
_ARCH_CHUNK_PROFILES: dict[str, dict[str, int]] = {
    "gfx1151": {
        "linear_chunk_size": 256,
        "moe_chunk_size": 256,
    },
}


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
    target_arch: str | None = None,
    geometry_chunk_sizes: tuple[int, int] | None = None,
) -> tuple[PrefillConfig, dict[str, object]]:
    """Resolve profile-derived chunk sizes for single-request prefill.

    This function is the **single owner** of the final linear/MoE chunk sizes.
    Callers must not rewrite the returned config afterwards; the recorded
    ``tuning["chunk_sizes"]`` always describes the config this call returned.

    Precedence is per field, highest first:

    1. an explicit non-zero field on the incoming ``config`` (a manual
       override, e.g. ``--prefill-linear-chunk-size``),
    2. ``geometry_chunk_sizes`` - a model+quant keyed measurement from the
       backend package (``GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY``),
    3. the architecture profile (``_ARCH_CHUNK_PROFILES[target_arch]``),
    4. the shape/memory policy: the retained 1024/4096 policy for prompts above
       1K, the 52K-class 1024-row full-attention query reduction on 24GB-class
       cards, and 768-token chunks for 128K-class and longer prompts.

    Layers 2 and 3 set only ``linear_chunk_size`` and ``moe_chunk_size``. A
    manual field is never overridden, and the remaining fields still receive
    the shape/memory policy, so a manual linear size no longer suppresses the
    long-context scratch-headroom policy.
    """

    max_sequence = int(max_sequence_length)
    if max_sequence <= 0:
        raise ValueError("max_sequence_length must be positive")
    manual_fields = _manual_chunk_fields(config)
    tuning: dict[str, object] = {
        "enabled": bool(config.auto_tune_chunk_sizes),
        "applied": False,
        "reason": "disabled",
        "max_sequence_length": max_sequence,
        "source": "chunk_sweep_2026_05_midcontext_manual_long_equiv",
        "memory_budget_gib": 0.0,
        "target_arch": None if target_arch is None else str(target_arch),
        "manual_chunk_sizes": list(manual_fields),
        "arch_profile_applied": False,
        "geometry_profile_applied": False,
    }
    if not config.auto_tune_chunk_sizes:
        # Auto-tuning off: the caller owns every size and no profile applies.
        tuning["chunk_sizes"] = _chunk_sizes_dict(config)
        return config, tuning
    if manual_fields:
        # An explicit chunk size means the caller owns the whole config, so no
        # profile layer applies - including the geometry policy, which used to
        # rewrite linear/MoE after this function returned. The record still
        # reports the sizes the session will actually run.
        tuning["reason"] = "manual_chunk_sizes"
        tuning["applied"] = False
        tuning["base_chunk_sizes"] = _chunk_sizes_dict(config)
        tuning["chunk_sizes"] = _chunk_sizes_dict(config)
        return config, tuning

    arch_key = str(target_arch or "").strip().lower()
    arch_profile = _ARCH_CHUNK_PROFILES.get(arch_key)
    geometry_profile = _geometry_profile_dict(geometry_chunk_sizes)

    if max_sequence < int(config.chunk_tune_min_tokens):
        tuning["reason"] = "below_min_tokens"
        base = config
    else:
        budget_gib = _chunk_memory_budget_gib(config, total_memory_bytes=total_memory_bytes)
        low_memory_card = 0 < int(total_memory_bytes) <= _LOW_MEMORY_TOTAL_BYTES
        low_memory_full_context = (
            low_memory_card and max_sequence >= _LOW_MEMORY_FULL_CONTEXT_MIN_TOKENS
        )
        low_memory_mid_context = (
            low_memory_card and max_sequence >= _LOW_MEMORY_MID_CONTEXT_MIN_TOKENS
        )
        if low_memory_full_context:
            estimated_peak_gib = 23.4
            reason = "low_memory_full_context_24gb"
            shape_policy = {
                "linear_chunk_size": _PREFILL_LOW_MEMORY_CHUNK,
                "moe_chunk_size": _PREFILL_LOW_MEMORY_CHUNK,
                "full_attn_query_chunk_size": _PREFILL_LOW_MEMORY_FULL_ATTN_QUERY_CHUNK,
                "full_attn_post_chunk_size": _PREFILL_LOW_MEMORY_CHUNK,
                "full_attn_rope_chunk_size": _PREFILL_LOW_MEMORY_CHUNK,
            }
        elif low_memory_mid_context:
            estimated_peak_gib = 23.0
            reason = "low_memory_mid_context_24gb"
            shape_policy = {
                "linear_chunk_size": _PREFILL_LINEAR_CHUNK,
                "moe_chunk_size": _PREFILL_MOE_CHUNK,
                "full_attn_query_chunk_size": _LOW_MEMORY_MID_CONTEXT_FULL_ATTN_QUERY_CHUNK,
                "full_attn_post_chunk_size": _PREFILL_FULL_ATTN_POST_CHUNK,
                "full_attn_rope_chunk_size": _PREFILL_FULL_ATTN_ROPE_CHUNK,
            }
        else:
            estimated_peak_gib = 20.0
            reason = "manual_long_equiv_gt1k"
            shape_policy = {
                "linear_chunk_size": _PREFILL_LINEAR_CHUNK,
                "moe_chunk_size": _PREFILL_MOE_CHUNK,
                "full_attn_query_chunk_size": _PREFILL_FULL_ATTN_QUERY_CHUNK,
                "full_attn_post_chunk_size": _PREFILL_FULL_ATTN_POST_CHUNK,
                "full_attn_rope_chunk_size": _PREFILL_FULL_ATTN_ROPE_CHUNK,
            }
        # The shape/memory policy is the lowest layer, so it also honors manual
        # fields: a manual linear size must not suppress the long-context
        # full-attention scratch-headroom reduction on the fields the caller
        # left unset.
        base = _apply_chunk_policy(config, shape_policy, manual_fields=manual_fields)
        tuning.update(
            {
                "reason": reason,
                "memory_budget_gib": budget_gib,
                "estimated_peak_gib": estimated_peak_gib,
            }
        )

    tuned = base
    if arch_profile:
        tuned = _apply_chunk_policy(tuned, arch_profile, manual_fields=manual_fields)
        tuning["arch_profile_applied"] = True
        tuning["arch_profile"] = {k: int(v) for k, v in arch_profile.items()}
        tuning["base_reason"] = str(tuning["reason"])
        tuning["base_source"] = str(tuning["source"])
        tuning["reason"] = f"{arch_key}_paro_prefill_recovery"
        tuning["source"] = f"sol_r1_{arch_key}_prefill_recovery_2026_07"
    if geometry_profile:
        tuned = _apply_chunk_policy(tuned, geometry_profile, manual_fields=manual_fields)
        tuning["geometry_profile_applied"] = True
        tuning["geometry_profile"] = {k: int(v) for k, v in geometry_profile.items()}
        # The geometry layer is the most specific, so it names the reason; the
        # arch layer's own naming stays visible above when both applied.
        if arch_profile:
            tuning["arch_reason"] = tuning["reason"]
            tuning["arch_source"] = tuning["source"]
        tuning["reason"] = "geometry_prefill_chunk_policy"
        tuning["source"] = "backend_geometry_chunk_policy"
    tuning["base_chunk_sizes"] = _chunk_sizes_dict(base)
    tuning["chunk_sizes"] = _chunk_sizes_dict(tuned)
    tuning["applied"] = bool(tuned != base)
    return tuned, tuning


_MANUAL_CHUNK_FIELDS: tuple[str, ...] = (
    "linear_chunk_size",
    "moe_chunk_size",
    "full_attn_query_chunk_size",
    "full_attn_post_chunk_size",
    "full_attn_rope_chunk_size",
)


def _manual_chunk_fields(config: PrefillConfig) -> tuple[str, ...]:
    """Return the fields the caller set explicitly, in declaration order."""

    return tuple(
        name for name in _MANUAL_CHUNK_FIELDS if int(getattr(config, name)) > 0
    )


def _apply_chunk_policy(
    config: PrefillConfig,
    policy: dict[str, int],
    *,
    manual_fields: tuple[str, ...],
) -> PrefillConfig:
    """Apply a chunk policy, skipping fields the caller set explicitly."""

    applied = {k: int(v) for k, v in policy.items() if k not in manual_fields}
    if not applied:
        return config
    return replace(config, **applied)


def _geometry_profile_dict(
    geometry_chunk_sizes: tuple[int, int] | None,
) -> dict[str, int] | None:
    """Normalize a backend geometry (linear, moe) chunk policy."""

    if geometry_chunk_sizes is None:
        return None
    linear, moe = (int(v) for v in geometry_chunk_sizes)
    profile: dict[str, int] = {}
    if linear > 0:
        profile["linear_chunk_size"] = linear
    if moe > 0:
        profile["moe_chunk_size"] = moe
    return profile or None


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
