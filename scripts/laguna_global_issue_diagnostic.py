"""Scoped historical global-attention policy for profiler comparisons."""

from contextlib import contextmanager


_STAGES = (
    ("compare_global_assume_exp", "global_gqa2"),
    ("compare_global_exp32", "global_gqa2_exp32"),
    ("compare_global_mixed32", "global_mixed32_exp32"),
)


_VARIANTS = (
    "global_context_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_spans",
    "global_context_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_spans",
    "global_context_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_spans",
    "global_context_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_spans",
)


def comparison_name(args):
    selected = [name for name, _ in _STAGES if getattr(args, name, False)]
    if len(selected) > 1:
        raise ValueError("only one global issue comparison may be active")
    return selected[0] if selected else None


@contextmanager
def global_issue_arm(owner, args, *, candidate):
    """Select an explicit registered reference without changing production flags."""
    name = comparison_name(args)
    if name is None:
        yield
        return
    with global_issue_comparison(owner, args):
        index = next(index for index, stage in enumerate(_STAGES) if stage[0] == name)
        owner.kv_cache._global_issue_reference_variant = _VARIANTS[index + bool(candidate)]
        yield


@contextmanager
def global_issue_comparison(owner, args):
    """Scope the control reference while leaving production cache policy intact."""
    name = comparison_name(args)
    if name is None:
        yield
        return
    cache = owner.kv_cache
    if cache is None or not cache.split_gate_fusion or cache.global_split_min_live is None:
        raise ValueError("global issue comparison requires gated split attention")
    index = next(index for index, stage in enumerate(_STAGES) if stage[0] == name)
    previous = cache._global_issue_reference_variant
    try:
        cache._global_issue_reference_variant = _VARIANTS[index]
        yield
    finally:
        cache._global_issue_reference_variant = previous
