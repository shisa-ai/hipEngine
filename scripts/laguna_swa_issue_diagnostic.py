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
_SUCCESSORS = (
    "swa_mixed32_exp32_producer_max",
    "swa_mixed32_exp32_producer_max_gate",
    "swa_mixed32_exp32_producer_max_gate_stage_pcache",
    "swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom",
    "swa_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability",
    "swa_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability",
    "swa_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability",
    "swa_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability",
    "swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability",
    "swa_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability",
)
_PREREQUISITES = (
    "swa_fused_fixed512", "swa_gqa3_local384_fixed512",
    "swa_gqa3_vstage64_fixed512", "swa_gqa3_vstage64_vec16_fixed512",
    "swa_gqa3_vstage64_vec16_direct_fixed512",
)


def comparison_name(args):
    selected = [name for name, _ in _STAGES if getattr(args, name, False)]
    if len(selected) > 1:
        raise ValueError("only one SWA issue comparison may be active")
    return selected[0] if selected else None


@contextmanager
def swa_issue_comparison(owner, args):
    """Hold prerequisites on and successors off, restoring every touched field."""
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
