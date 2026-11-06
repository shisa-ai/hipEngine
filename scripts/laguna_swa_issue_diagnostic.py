"""Scoped historical SWA issue-schedule policy for profiler comparisons."""

from contextlib import contextmanager


_SUFFIX = "_vstage64_vec16_direct_assume_exp_fixed512"
_STAGES = (
    ("compare_swa_assume_exp", "swa_gqa3"),
    ("compare_swa_mixed32", "swa_mixed32"),
    ("compare_swa_mixed32_exp4", "swa_mixed32_exp4"),
    ("compare_swa_mixed32_exp8", "swa_mixed32_exp8"),
    ("compare_swa_mixed32_exp16", "swa_mixed32_exp16"),
    ("compare_swa_mixed32_exp32", "swa_mixed32_exp32"),
)
_VARIANTS = (
    "swa_context_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_spans",
    *("swa_context_fused_exact_gated_" + prefix.removeprefix("swa_") + _SUFFIX + "_spans"
      for _, prefix in _STAGES),
)


def comparison_name(args):
    selected = [name for name, _ in _STAGES if getattr(args, name, False)]
    if len(selected) > 1:
        raise ValueError("only one SWA issue comparison may be active")
    return selected[0] if selected else None


@contextmanager
def swa_issue_arm(owner, args, *, candidate):
    """Scope one registered reference without changing production cache flags."""
    name = comparison_name(args)
    if name is None:
        yield
        return
    with swa_issue_comparison(owner, args):
        index = next(index for index, stage in enumerate(_STAGES) if stage[0] == name)
        owner.kv_cache._swa_issue_reference_variant = _VARIANTS[index + bool(candidate)]
        yield


@contextmanager
def swa_issue_comparison(owner, args):
    """Restore the control reference after nested or failed comparisons."""
    name = comparison_name(args)
    if name is None:
        yield
        return
    cache = owner.kv_cache
    if cache is None or not (
        cache.split_gate_fusion and cache.swa_split_wave_local
        and cache.swa_split_min_live is not None
        and cache.swa_split_tile16_min_live is not None
    ):
        raise ValueError("SWA issue comparison requires split attention, tile16, gated and wave-local reducers")
    index = next(index for index, stage in enumerate(_STAGES) if stage[0] == name)
    previous = cache._swa_issue_reference_variant
    try:
        cache._swa_issue_reference_variant = _VARIANTS[index]
        yield
    finally:
        cache._swa_issue_reference_variant = previous
