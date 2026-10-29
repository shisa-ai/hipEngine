"""Scoped historical global-attention policy for profiler comparisons."""

from contextlib import contextmanager


_SUFFIX = "_vstage64_vec16_direct_assume_exp_fixedshape"
_STAGES = (
    ("compare_global_assume_exp", "global_gqa2"),
    ("compare_global_exp32", "global_gqa2_exp32"),
    ("compare_global_mixed32", "global_mixed32_exp32"),
)
_SUCCESSORS = (
    "global_mixed32_exp32_producer_max",
    "global_mixed32_exp32_producer_max_dpp_qk",
    "global_mixed32_exp32_producer_max_dpp_qk_probability_vec4",
    "global_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm",
    "global_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm",
    "global_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm",
)
_PREREQUISITES = (
    "global_split_fixedshape_reduce", "global_fused_fixedshape",
    "global_gqa2_vstage64_fixedshape", "global_gqa2_vstage64_vec16_fixedshape",
    "global_gqa2_vstage64_vec16_direct_fixedshape",
)


def comparison_name(args):
    selected = [name for name, _ in _STAGES if getattr(args, name, False)]
    if len(selected) > 1:
        raise ValueError("only one global issue comparison may be active")
    return selected[0] if selected else None


@contextmanager
def global_issue_comparison(owner, args):
    """Hold predecessors on and successors off, restoring every touched field."""
    name = comparison_name(args)
    if name is None:
        yield
        return
    cache = owner.kv_cache
    if cache is None or not cache.split_gate_fusion or cache.global_split_min_live is None:
        raise ValueError("global issue comparison requires gated split attention")
    index = next(index for index, stage in enumerate(_STAGES) if stage[0] == name)
    policy = dict.fromkeys(_PREREQUISITES, True)
    policy.update({prefix + _SUFFIX: stage < index for stage, (_, prefix) in enumerate(_STAGES)})
    policy.update(dict.fromkeys((prefix + _SUFFIX for prefix in _SUCCESSORS), False))
    previous = {key: getattr(cache, key) for key in policy}
    try:
        for key, value in policy.items():
            setattr(cache, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(cache, key, value)
