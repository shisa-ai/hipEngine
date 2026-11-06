"""Resolved legacy cached-prefill choices, separate from package role maps."""

from dataclasses import dataclass

from hipengine.kernels.backends import backend_package_capability


@dataclass(frozen=True)
class LagunaCachedPrefillPolicy:
    kv_preappend: bool
    cached_meta: bool
    global_qrow6: bool
    dense_initial: bool


def resolve_laguna_cached_prefill_policy(
    backend: str, *, policy: LagunaCachedPrefillPolicy | None = None,
    kv_preappend: bool | None = None,
    cached_meta: bool | None = None, global_qrow6: bool | None = None,
    dense_initial: bool | None = None,
) -> LagunaCachedPrefillPolicy:
    """Resolve independent legacy choices without modifying role substitutions."""
    requested = {
        "kv_preappend": kv_preappend, "cached_meta": cached_meta,
        "global_qrow6": global_qrow6, "dense_initial": dense_initial,
    }
    if policy is not None:
        if not isinstance(policy, LagunaCachedPrefillPolicy):
            raise TypeError("cached prefill policy must be LagunaCachedPrefillPolicy")
        if any(value is not None for value in requested.values()):
            raise ValueError("cached prefill policy cannot be combined with legacy overrides")
        return policy
    return LagunaCachedPrefillPolicy(**{
        name: bool(backend_package_capability(
            backend, "LAGUNA_PREFILL_" + name.upper(), False,
        ) if value is None else value)
        for name, value in requested.items()
    })


@dataclass(frozen=True)
class LagunaLongAttentionPolicy:
    enabled: bool
    bounded: bool


def resolve_laguna_long_attention_policy(
    backend: str, *, policy: LagunaLongAttentionPolicy | None = None,
    enabled: bool | None = None, bounded: bool | None = None,
) -> LagunaLongAttentionPolicy:
    if policy is not None:
        if not isinstance(policy, LagunaLongAttentionPolicy):
            raise TypeError("long attention policy must be LagunaLongAttentionPolicy")
        if enabled is not None or bounded is not None:
            raise ValueError("long attention policy cannot be combined with legacy overrides")
        return policy
    return LagunaLongAttentionPolicy(
        bool(backend_package_capability(backend, "LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT", False)
             if enabled is None else enabled),
        bool(backend_package_capability(backend, "LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT", False)
             if bounded is None else bounded),
    )


def select_laguna_long_attention_route(
    *, enabled: bool, bounded: bool, short_route: bool, full_attention: bool,
    rows: int, start: int, context_length: int, q_heads: int, kv_heads: int,
    head_dim: int,
) -> str | None:
    """Select bounded or full-score execution without conflating their owners."""
    if (
        not enabled or short_route or not full_attention
        or rows not in {128, 256, 512, 1_024, 2_048}
        or (rows != 128 and not bounded)
        or (rows == 128 and start < 4_096)
        or start + rows > context_length
        or (q_heads, kv_heads, head_dim) != (48, 8, 128)
    ):
        return None
    return "bounded" if bounded else "full_score"
