"""Diagnose incumbent numerical drift with named-family ablations.

Full18 prefill-last rows plus the previously failing Japanese decode chain.
One numerical run per arm: not a promotion, determinism or task certificate.
"""

import argparse
from contextlib import ExitStack, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import RowDescriptor, compare_profile_logits
from hipengine.runtime.qwen4_exp_runner import gdn_peer_prefill, gdn_register_prefill, gdn_tile16_variant, qsa_h256_prefill_mode, production_moe_prefill_enabled, q8_down_variant, q8_mmq_prefill_enabled, q4_dp4a_decode, q4_pair_prefill, q8_wave_scale_prefill, gr_wave_scale, gr_wide, resolve_gdn_tile16_variant, resolve_production_moe_prefill, resolve_q8_down_variant, resolve_q8_mmq_prefill
from hipengine.runtime.qwen4_exp_runner import Q8_DOWN_WMMA_VARIANT
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_layer2_profile_gate import _make_generator, _strict_trajectory, _candidate_trajectory
from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity

PREFIX = "HIPENGINE_QWEN4_EXP_"
Q4_DP4A_ENV_KEYS = frozenset({
    PREFIX + "Q4_DP4A64", PREFIX + "Q4_DP4A64_LAYERS",
})
ARMS = {
    "bound": {},
    "gdn_strict": {PREFIX + "GDN_COLWARPS_PREFILL": "0"},
    "moe_strict": {PREFIX + "Q4_IU8_PREFILL": "0"},
    "dense_q8_gr_strict": {PREFIX + key: "0" for key in ("Q8_MMQ_PREFILL", "GR_IU8", "GR_IU8_DOWN")},
    # Q4 DP4A is owner state now; the arm loop binds it below instead of
    # writing a process key the bound generator no longer reads.
    "decode_strict": {},
}
ARMS["all_numerics_strict"] = {
    key: value for name, values in ARMS.items() if name != "bound"
    for key, value in values.items()
}
ARMS["production_conservative"] = dict(CANDIDATES["production_conservative"].environment)
for name in ("q8_blockscale", "q8_blockscale_mmq",
             "q8_blockscale_guarded", "q8_blockscale_guarded_mmq"):
    ARMS[name] = dict(CANDIDATES[name].environment)
# Candidate owner decisions are not environment keys any more. Keep each arm's
# former process value only as the owner decision bound in the arm loop; no
# process-wide switch is written.
_Q8_MMQ_ARM_VALUES: dict[str, bool] = {}
for name in tuple(ARMS):
    candidate = CANDIDATES.get(name)
    if candidate is not None and candidate.q8_mmq_prefill_enabled is not None:
        _Q8_MMQ_ARM_VALUES[name] = bool(candidate.q8_mmq_prefill_enabled)
ARMS["all_flags_strict"] = {}
ARMS["gdn_flags_strict"] = {}
ARMS["base_plus_moe_flags"] = {}
ARMS["base_plus_q8_gr_flags"] = {}
ARMS["base_plus_gdn_flags"] = {}
ARMS["q8_selected_down_strict"] = {PREFIX + "Q8_0_SELECTED_WMMA_DOWN": "0"}
ARMS["base_plus_q8_selected_down_strict"] = {
    **ARMS["all_numerics_strict"], **ARMS["q8_selected_down_strict"],
}
for name in ("strict_plus_gdn", "strict_plus_gdn_serial",
             "strict_plus_gdn_multi", "strict_prefill_production_decode",
             "production_prefill_strict_decode"):
    ARMS[name] = {}
OUTLIER = "heldout_general_ja_speculative"
FAMILY_PREFIXES = {
    "matrix": ("Q4_", "Q51_", "Q5_", "GROUPED_", "FORKB_", "PROFILE_Q5_1_",
               "PRODUCTION_MOE_", "Q8_0_"),
    "dense": ("Q8_MMQ_", "Q8_IU8_", "GR_"),
    "qsa": ("QSA_",),
}
for family in FAMILY_PREFIXES:
    ARMS["strict_plus_" + family] = {}
    ARMS["q8down_off_without_" + family] = {}
for flag in ("Q8_MMQ_PREFILL", "GR_IU8", "GR_IU8_DOWN"):
    ARMS["strict_plus_" + flag.lower()] = {}
    ARMS["q8down_off_without_" + flag.lower()] = {
        **ARMS["q8_selected_down_strict"], PREFIX + flag: "0"}

# Move every remaining Q8 MMQ process value onto the owner decision. This
# includes the strict families that inherited the historical ``0``.
_Q8_MMQ_ENV = PREFIX + "Q8_MMQ_PREFILL"
_Q8_MMQ_FALSE = frozenset({"", "0", "false", "False"})
for name in tuple(ARMS):
    raw = ARMS[name].pop(_Q8_MMQ_ENV, None)
    if raw is not None:
        _Q8_MMQ_ARM_VALUES[name] = raw not in _Q8_MMQ_FALSE
# These families are materialized later from ``all_numerics_strict`` and so
# inherited its historical Q8 MMQ ``0`` without carrying a key of their own.
for name in ("base_plus_moe_flags", "base_plus_q8_gr_flags",
             "base_plus_gdn_flags"):
    _Q8_MMQ_ARM_VALUES[name] = False


#: Arms whose former production-MoE process value was 0. The gate is owner
#: state now, so the arm binds it on the generator/runner instead. The generated
#: ``strict_plus_*`` family always inherited the strict binding's 0 through its
#: ``**strict_flags`` base, so every such arm is closed here too.
PRODUCTION_MOE_OFF_ARMS = frozenset({
    "moe_strict", "all_numerics_strict", "all_flags_strict",
    "base_plus_moe_flags", "base_plus_q8_gr_flags", "base_plus_gdn_flags",
    "base_plus_q8_selected_down_strict", "q8down_off_without_matrix",
    "production_conservative",
})
PRODUCTION_MOE_SPLIT_ARMS = frozenset({
    "strict_prefill_production_decode", "production_prefill_strict_decode",
})


def production_moe_arm_policy(arm):
    """Which production-MoE owner binding an arm's former process value meant."""

    if arm in PRODUCTION_MOE_SPLIT_ARMS:
        return "split"
    if arm in PRODUCTION_MOE_OFF_ARMS or arm.startswith("strict_plus_"):
        return "off"
    return "preserve"


def production_moe_arm_scope(generator, arm):
    """Bind the arm's production-MoE decision on the generator and runner."""

    if production_moe_arm_policy(arm) == "off":
        return production_moe_prefill_enabled(generator, False)
    return nullcontext()


def production_moe_split_scope(runner, arm, phase):
    """Bind one split phase's production-MoE decision on the measured runner."""

    if arm not in PRODUCTION_MOE_SPLIT_ARMS:
        return nullcontext()
    strict_phase = (
        (arm == "strict_prefill_production_decode" and phase == "prefill")
        or (arm == "production_prefill_strict_decode" and phase == "decode")
    )
    if not strict_phase:
        return nullcontext()
    return production_moe_prefill_enabled(runner, False)


def production_moe_arm_report(generator, arm):
    """Name the owner decision(s) that actually ran for this arm."""

    policy = production_moe_arm_policy(arm)
    if policy == "off":
        return False
    actual = resolve_production_moe_prefill(generator)
    if policy == "split":
        strict_prefill = arm == "strict_prefill_production_decode"
        return {
            "prefill": False if strict_prefill else actual,
            "decode": actual if strict_prefill else False,
        }
    return actual


#: Split arms bind Q8 MMQ per phase: the strict phase closes the workspace
#: owner, and the production phase leaves the production owner in place.
Q8_MMQ_SPLIT_ARMS = frozenset({
    "strict_prefill_production_decode", "production_prefill_strict_decode",
})


def q8_mmq_arm_enabled(arm):
    """The Q8 MMQ owner value this arm's former process key meant."""

    return _Q8_MMQ_ARM_VALUES.get(arm)


def q8_mmq_arm_scope(generator, arm):
    """Bind the arm's Q8 MMQ decision on the generator and its runner."""

    enabled = q8_mmq_arm_enabled(arm)
    if enabled is None:
        return nullcontext()
    return q8_mmq_prefill_enabled(generator, enabled)


def q8_mmq_split_scope(runner, arm, phase):
    """Bind one split phase's Q8 MMQ decision on the measured runner.

    The strict phase closes the owner. The production phase returns an
    unbound scope, so the production binder's owner decision is restored and
    used, including when that decision is enabled.
    """

    if arm not in Q8_MMQ_SPLIT_ARMS:
        return nullcontext()
    strict_phase = (
        (arm == "strict_prefill_production_decode" and phase == "prefill")
        or (arm == "production_prefill_strict_decode" and phase == "decode")
    )
    if not strict_phase:
        return nullcontext()
    return q8_mmq_prefill_enabled(runner, False)


def q8_mmq_arm_report(generator, arm):
    """Name the Q8 MMQ decision(s) that actually ran for this arm."""

    if arm in Q8_MMQ_SPLIT_ARMS:
        actual = resolve_q8_mmq_prefill(generator)
        strict_prefill = arm == "strict_prefill_production_decode"
        return {
            "prefill": False if strict_prefill else actual,
            "decode": actual if strict_prefill else False,
        }
    enabled = q8_mmq_arm_enabled(arm)
    return resolve_q8_mmq_prefill(generator) if enabled is None else enabled


#: Arms whose retired Q4_DP4A64 process key disabled DP4A without changing the
#: layer scope. They now bind the same decision on the owning generator/runner.
Q4_DP4A_CURRENT_ARMS = frozenset({
    "decode_strict", "all_numerics_strict", "base_plus_q8_gr_flags",
    "base_plus_gdn_flags", "base_plus_q8_selected_down_strict",
    "q8down_off_without_dense", "q8down_off_without_qsa",
})
#: Arms whose retired strict binding also published the empty layer scope.
Q4_DP4A_EMPTY_ARMS = frozenset({
    "all_flags_strict", "gdn_flags_strict", "base_plus_moe_flags",
    "q8down_off_without_matrix",
})
#: Split arms bind the pair per phase: the strict phase uses the empty scope,
#: and the production phase leaves the production scope in place.
Q4_DP4A_SPLIT_ARMS = frozenset({
    "strict_prefill_production_decode", "production_prefill_strict_decode",
})


def strict_environment(environ):
    """Capture the strict binding's process keys without the retired Q4 pair."""

    return {
        key: value for key, value in environ.items()
        if (key.startswith(PREFIX) or key == "HIPENGINE_GGUF_WMMA_PREFILL")
        and key not in Q4_DP4A_ENV_KEYS
        and key != _Q8_MMQ_ENV
    }


def q4_dp4a_arm_policy(arm):
    """Which Q4 DP4A owner binding an arm's retired process key implied.

    ``"empty"`` means the old strict key set the empty layer scope,
    ``"current"`` means it disabled DP4A but left the production layer scope,
    ``"split"`` means the phase loop binds it, and ``"preserve"`` means the arm
    never named the key.
    """

    if arm in Q4_DP4A_SPLIT_ARMS:
        return "split"
    if arm in Q4_DP4A_EMPTY_ARMS or arm.startswith("strict_plus_"):
        return "empty"
    if arm in Q4_DP4A_CURRENT_ARMS:
        return "current"
    return "preserve"


def q4_dp4a_arm_scope(generator, arm):
    """Bind the arm's Q4 DP4A decision on the generator and its runner."""

    policy = q4_dp4a_arm_policy(arm)
    if policy == "empty":
        return q4_dp4a_decode(generator, False, frozenset())
    if policy == "current":
        return q4_dp4a_decode(
            generator, False,
            getattr(generator, "q4_dp4a_decode_layers", None),
        )
    return nullcontext()


def q4_dp4a_split_scope(runner, arm, phase):
    """Bind one split phase's Q4 DP4A decision on the measured runner."""

    if arm not in Q4_DP4A_SPLIT_ARMS:
        return nullcontext()
    strict_phase = (
        (arm == "strict_prefill_production_decode" and phase == "prefill")
        or (arm == "production_prefill_strict_decode" and phase == "decode")
    )
    if not strict_phase:
        return nullcontext()
    return q4_dp4a_decode(runner, False, frozenset())


def clear_arm_graphs(runner):
    from hipengine.runtime.moe_graph import MoeGraphCache

    runner.runtime.device_synchronize()
    for name in ("moe_graph_cache", "layer_graph_cache"):
        cache = getattr(runner, name, None)
        if cache is not None:
            enabled = cache.enabled
            cache.close()
            setattr(runner, name, MoeGraphCache(runner.runtime, enabled=enabled))


def strict_family_overrides(strict_flags):
    families = {
        "base_plus_moe_flags": (
            "Q4_", "Q51_", "Q5_", "GROUPED_", "FORKB_", "PROFILE_Q5_1_",
            "Q8_0_SELECTED_WMMA_DOWN",
        ),
        "base_plus_q8_gr_flags": ("Q8_", "GR_"),
        "base_plus_gdn_flags": ("GDN_",),
    }
    return {
        name: {
            **ARMS["all_numerics_strict"],
            **{key: value for key, value in strict_flags.items()
               if any(key.startswith(PREFIX + prefix) for prefix in prefixes)},
        } for name, prefixes in families.items()
    }


def gdn_isolation_overrides(strict_flags, production_flags):
    production_gdn = {key: value for key, value in production_flags.items()
                      if key.startswith(PREFIX + "GDN_")}
    isolated = {**strict_flags, **production_gdn}
    return {
        "strict_plus_gdn": isolated,
        "strict_plus_gdn_serial": {**isolated, **ARMS["gdn_strict"]},
        "strict_plus_gdn_multi": {
            **isolated,
        },
    }


def serial_gdn(env):
    """Does this arm pin the serial GDN prefill owner?

    The peer owner's layer scope is request-local runner state now, so an arm
    that closes the columnwarp gate - the parent both owners sit inside - binds
    it off on the runner it measures instead of writing a process value. Before
    the migration the same arms zeroed the retired process gate; deriving the
    decision from the columnwarp value keeps a new strict aggregate covered
    without a second list to maintain.
    """

    return env.get(PREFIX + "GDN_COLWARPS_PREFILL") == "0"


def phase_peer_scope(runner, flags):
    """Bind the GDN peer prefill owner for one phase of a split trajectory.

    The split arms set the prefill and decode flag sets independently, so the
    peer owner's request-local scope has to follow the phase that is running: a
    strict phase binds it off, and a production phase leaves the runner's
    published scope in place. Decode runs one row and the mixer takes its decode
    branch before the prefill chain, so the owner is unreachable there either
    way. The prior scope is restored on exit, including on an exception.
    """

    if not serial_gdn(flags):
        return nullcontext()
    return gdn_peer_prefill(runner, False)


def set_flags(flags):
    for key, value in flags.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def h256_arm_mode(arm):
    """Keep strict QSA arms on the parent without process-wide selectors."""
    if arm in CANDIDATES:
        return CANDIDATES[arm].qsa_h256_prefill_mode
    if arm == "all_flags_strict" or arm == "q8down_off_without_qsa":
        return "parent"
    if arm.startswith("strict_plus_") and arm != "strict_plus_qsa":
        return "parent"
    return None


def register_arm_enabled(arm, published):
    """Preserve each arm's register-state choice after environment retirement."""
    if arm in {"all_flags_strict", "gdn_flags_strict", "base_plus_gdn_flags"}:
        return False
    if arm.startswith("strict_plus_"):
        return arm in {"strict_plus_gdn", "strict_plus_gdn_serial", "strict_plus_gdn_multi"}
    return published


def pair_arm_enabled(arm, published):
    """Preserve the strict matrix family's runner-local pair2 choice."""
    if arm in {"all_flags_strict", "base_plus_moe_flags", "q8down_off_without_matrix"}:
        return False
    if arm.startswith("strict_plus_"):
        return published if arm == "strict_plus_matrix" else False
    return published


def wave_arm_enabled(arm, published):
    """Preserve the former strict Q8 process-key overrides."""
    if arm in {"all_flags_strict", "base_plus_q8_gr_flags"} or arm.startswith("strict_plus_"):
        return False
    return published


def gr_wave_arm_enabled(arm, published):
    """Preserve the strict GR-family overrides after selector retirement."""
    if arm in {"all_flags_strict", "base_plus_q8_gr_flags", "q8down_off_without_dense"}:
        return False
    if arm.startswith("strict_plus_"):
        return published if arm == "strict_plus_dense" else False
    return published


#: The retired GR-up wide key. Both bindings wrote it - the strict bind
#: published the fused exact parent and the production bind the plan's route -
#: so the two snapshots this harness builds its arms from carried it too.
_GR_WIDE_ENV = PREFIX + "GR_WIDE"
_ENV_FALSE = frozenset({"", "0", "false", "False"})


def _env_flag(raw):
    """Parse one former process value the way the retired reader did."""

    return raw not in _ENV_FALSE


def gr_wide_snapshot(flags, *, published):
    """One former environment snapshot, including the retired wide key.

    ``flags`` is the snapshot the harness captures today and ``published`` is
    the route the binding that produced it publishes on its runner, which is
    exactly what the retired key carried at that point.
    """

    return {**flags, _GR_WIDE_ENV: "1" if published else "0"}


def declared_gr_wide():
    """Candidate arms whose declaration names an explicit GR-up wide route."""

    return {
        name: bool(candidate.gr_wide)
        for name, candidate in ((name, CANDIDATES.get(name)) for name in ARMS)
        if candidate is not None and candidate.gr_wide is not None
    }


#: Former effective GR-up wide route per arm, recorded from the same dicts that
#: produced each arm's former environment. Seeded with the candidate
#: declarations; ``main`` records the arms built from the two bindings' former
#: snapshots and defaults the rest to the production route.
_GR_WIDE_ARM_VALUES: dict[str, bool] = declared_gr_wide()


def migrate_gr_wide_arms(arms, *, production_value, declared=None):
    """Record each arm's former GR-up wide value, then drop the retired key.

    The retired key was a captured ``GR_*`` process key. An arm whose former
    environment named it ran that value; an arm whose candidate declaration
    names it ran the declaration, because the candidate environment carried it
    before retirement; and an arm that named it nowhere ran whatever the
    production bind had published - the arm loop's ``bound`` snapshot, which is
    ``production_value`` here.
    """

    declared = declared or {}
    values = dict(declared)
    for name, env in arms.items():
        raw = env.pop(_GR_WIDE_ENV, None)
        if name in values:
            continue
        values[name] = bool(production_value) if raw is None else _env_flag(raw)
    return values


def snapshot_arms(strict_snapshot, production_snapshot, *, arms=None):
    """Build the arms whose former environment came from the two snapshots.

    Both bindings wrote the GR-up wide key, so the strict snapshot carried the
    strict route and the production snapshot the production route. The arms are
    built from them exactly as they were from the live environments, so each
    arm keeps the value its former process key carried.
    """

    arms = ARMS if arms is None else arms
    arms["all_flags_strict"] = dict(strict_snapshot)
    arms["gdn_flags_strict"] = {key: value for key, value in strict_snapshot.items()
                                if key.startswith(PREFIX + "GDN_")}
    arms.update(strict_family_overrides(strict_snapshot))
    arms.update(gdn_isolation_overrides(strict_snapshot, production_snapshot))
    for family, prefixes in FAMILY_PREFIXES.items():
        keys = [key for key in strict_snapshot
                if any(key.startswith(PREFIX + prefix) for prefix in prefixes)]
        arms["strict_plus_" + family] = {
            **strict_snapshot, **{key: production_snapshot.get(key) for key in keys}}
        arms["q8down_off_without_" + family] = {
            **arms["q8_selected_down_strict"],
            **{key: strict_snapshot[key] for key in keys}}
    for flag in ("Q8_MMQ_PREFILL", "GR_IU8", "GR_IU8_DOWN"):
        arm = "strict_plus_" + flag.lower()
        arms[arm] = dict(strict_snapshot)
        if flag == "Q8_MMQ_PREFILL":
            _Q8_MMQ_ARM_VALUES[arm] = True
        else:
            arms[arm][PREFIX + flag] = "1"
    return arms


def gr_wide_arm_enabled(arm, published):
    """The GR-up wide route this arm's former process key implied."""

    return _GR_WIDE_ARM_VALUES.get(arm, bool(published))


def gr_wide_arm_report(generator, arm):
    """Name the GR-up wide route that actually ran for this arm."""

    return gr_wide_arm_enabled(
        arm, bool(getattr(generator.runner, "gr_wide", False)))


def declared_q8_down_variants():
    """Arms whose candidate declaration names a grouped Q8_0 down owner.

    The retired ``HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT`` key used to carry this
    decision. Each block-scale arm's candidate now declares it, so an arm that
    names one binds that leaf and every other arm keeps whatever the plan
    published.
    """

    return {
        name: candidate.q8_down_variant
        for name, candidate in ((name, CANDIDATES.get(name)) for name in ARMS)
        if candidate is not None and candidate.q8_down_variant is not None
    }


#: Grouped Q8_0 down owner each arm measures. Seeded with the candidate
#: declarations; a probe that counts or instruments one leaf names it here
#: through :func:`declare_q8_down_owner` instead of writing a process key.
_Q8_DOWN_ARM_VARIANTS: dict[str, str] = declared_q8_down_variants()


def declare_q8_down_owner(variant, *, arms=None):
    """Name the grouped Q8_0 down owner a probe's arms measure.

    A diagnostic probe that counts or samples one registered leaf declares it
    here, so its arms bind that leaf instead of the plan's published owner. The
    map is the same one the arm loop reads, so the declaration, the binding and
    the report cannot drift apart.
    """

    for arm in (ARMS if arms is None else arms):
        _Q8_DOWN_ARM_VARIANTS[arm] = variant


def q8_down_arm_variant(arm):
    """The grouped Q8_0 down owner this arm measures, or ``None`` for the plan's."""

    return _Q8_DOWN_ARM_VARIANTS.get(arm)


def q8_down_arm_scope(generator, arm):
    """Bind the grouped Q8_0 down owner this arm measures.

    ``None`` leaves the plan's published owner in place; a named leaf is bound
    for the arm's samples and the published owner comes back on exit.
    """

    variant = q8_down_arm_variant(arm)
    if variant is None:
        return nullcontext()
    return q8_down_variant(generator, variant)


def q8_down_arm_report(generator, arm):
    """Name the grouped Q8_0 down owner that actually ran for this arm."""

    return resolve_q8_down_variant(generator)


#: The tile-16 GDN prefill leaf this arm measures. The retired
#: ``HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT`` key used to carry it, so an arm
#: whose former environment named a leaf keeps that leaf here and the arm loop
#: binds it on the runner it measures. Seeded from the candidate declarations,
#: which is where the multi-column and DPP arms now name their leaf.
_GDN_TILE16_MULTI_VARIANT = "qwen4exp_gdn_tiled16_multi_prefill"


def declared_gdn_tile16_variants():
    """Arms whose candidate declaration names a tile-16 GDN prefill owner."""

    return {
        name: candidate.gdn_tile16_variant
        for name, candidate in ((name, CANDIDATES.get(name)) for name in ARMS)
        if candidate is not None and candidate.gdn_tile16_variant is not None
    }


_GDN_TILE16_ARM_VARIANTS: dict[str, str] = declared_gdn_tile16_variants()
#: ``strict_plus_gdn_multi`` is built by :func:`gdn_isolation_overrides` from
#: the production GDN flag set plus one named leaf, so it has no candidate
#: declaration to inherit from.
_GDN_TILE16_ARM_VARIANTS["strict_plus_gdn_multi"] = _GDN_TILE16_MULTI_VARIANT


def declare_gdn_tile16_owner(variant, *, arms=None):
    """Name the tile-16 GDN prefill leaf a probe's arms measure.

    A diagnostic probe that counts or samples one registered leaf declares it
    here, so its arms bind that leaf instead of the plan's published owner. The
    map is the same one the arm loop reads, so the declaration, the binding and
    the report cannot drift apart.
    """

    for arm in (ARMS if arms is None else arms):
        _GDN_TILE16_ARM_VARIANTS[arm] = variant


def gdn_tile16_arm_variant(arm):
    """The tile-16 GDN prefill owner this arm measures, or ``None``."""

    return _GDN_TILE16_ARM_VARIANTS.get(arm)


def gdn_tile16_arm_scope(generator, arm):
    """Bind the tile-16 GDN prefill owner this arm measures.

    ``None`` leaves the plan's published owner in place; a named leaf is bound
    for the arm's samples and the published owner comes back on exit. The
    columnwarp gate and the admission decision stay at the dispatch.
    """

    variant = gdn_tile16_arm_variant(arm)
    if variant is None:
        return nullcontext()
    return gdn_tile16_variant(generator, variant)


def gdn_tile16_arm_report(generator, arm):
    """Name the tile-16 GDN prefill owner that actually ran for this arm."""

    return resolve_gdn_tile16_variant(generator)


#: Split arms bind the GR-up wide route per phase: the strict phase ran the
#: strict snapshot's route, and the production phase kept the production one.
GR_WIDE_SPLIT_ARMS = frozenset({
    "strict_prefill_production_decode", "production_prefill_strict_decode",
})


def gr_wide_split_scope(runner, arm, phase):
    """Bind one split phase's GR-up wide route, as its former flag set did."""

    if arm not in GR_WIDE_SPLIT_ARMS:
        return nullcontext()
    strict_phase = (
        (arm == "strict_prefill_production_decode" and phase == "prefill")
        or (arm == "production_prefill_strict_decode" and phase == "decode")
    )
    if not strict_phase:
        return nullcontext()
    return gr_wide(runner, False)


def split_trajectory(runner, prompt_ids, forced_ids, prefill_flags, decode_flags,
                     *, arm=None, h256_mode=None, register_prefill=None,
                     pair_prefill=None, wave_prefill=None, gr_wave_prefill=None):
    # Reconstruct the complete prefix in one runner; snapshots omit KV data.
    clear_arm_graphs(runner)
    set_flags(prefill_flags)
    with phase_peer_scope(runner, prefill_flags), qsa_h256_prefill_mode(runner, h256_mode), gdn_register_prefill(
            runner, getattr(runner, "gdn_register_prefill", False)
            if register_prefill is None else register_prefill), q4_pair_prefill(
                runner, getattr(runner, "q4_pair_prefill", False)
                if pair_prefill is None else pair_prefill), q8_wave_scale_prefill(
                    runner, getattr(runner, "q8_wave_scale_prefill", False)
                    if wave_prefill is None else wave_prefill), gr_wave_scale(
                        runner, getattr(runner, "gr_wave_scale", False)
                        if gr_wave_prefill is None else gr_wave_prefill), q4_dp4a_split_scope(
                            runner, arm, "prefill"), q8_mmq_split_scope(
                            runner, arm, "prefill"), production_moe_split_scope(
                            runner, arm, "prefill"), gr_wide_split_scope(
                            runner, arm, "prefill"):
        runner.reset()
        result = runner.prefill(prompt_ids)
    logits = [np.array(result.logits, dtype=np.float32, copy=True)]
    clear_arm_graphs(runner)
    set_flags(decode_flags)
    with q4_dp4a_split_scope(runner, arm, "decode"), q8_mmq_split_scope(
            runner, arm, "decode"), production_moe_split_scope(
            runner, arm, "decode"), gr_wide_split_scope(runner, arm, "decode"):
        for token in forced_ids:
            result = runner.step(token)
            logits.append(np.array(result.logits, dtype=np.float32, copy=True))
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    args = parser.parse_args()
    check_host()
    args.max_sequence_length = 2051
    args.prefill_chunk_size = 1024
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        register_gfx1151_kernels(replace=True)
        register_qwen4_exp_gfx1151_profiles()
        prompt_rows = _load_suites(DEFAULT_PROMPTS)
        reset_memory_stats()
        identity = model_identity(args.model_root)
        strict_generator, strict_profile, _ = _make_generator(args, "strict")
        strict_flags = strict_environment(os.environ)
        # The retired GR-up wide key was a captured ``GR_*`` process key, so the
        # strict snapshot carried the strict bind's route. Reconstruct it from
        # the binding that published it and let ``snapshot_arms`` rebuild the
        # arms that named it.
        strict_snapshot = gr_wide_snapshot(
            strict_flags, published=bool(strict_generator.runner.gr_wide))
        strict = {}
        tokens = {}
        descriptors = []
        try:
            for row in prompt_rows:
                key = row["id"]
                tokens[key] = build_chat_prompt(strict_generator.tokenizer, row["prompt"])
                strict[key] = _strict_trajectory(strict_generator.runner, tokens[key], 32 if key == OUTLIER else 0)
                for step, sample in enumerate(strict[key]):
                    descriptors.append(RowDescriptor(
                        scenario_id="journey-family-localization", scenario_step=len(descriptors),
                        request_id=key, teacher_step=step, category=row["category"],
                        shape="prefill_last" if step == 0 else "c1",
                        transition="prefill_to_c1" if step == 0 else "steady",
                        teacher_token_id=sample["token_id"],
                    ))
                print("strict", key, flush=True)
        finally:
            strict_generator.close()
        strict_close = memory_stats()
        ARMS["all_flags_strict"] = dict(strict_snapshot)
        ARMS["gdn_flags_strict"] = {key: value for key, value in strict_snapshot.items()
                                   if key.startswith(PREFIX + "GDN_")}
        ARMS.update(strict_family_overrides(strict_snapshot))
        strict_logits = np.stack([r["logits"] for p in prompt_rows for r in strict[p["id"]]])
        report = {
            "kind": "journey_incumbent_family_localization", "status": "running",
            "performance_claim": False, "promotion_eligible": False,
            "command": sys.argv, "source": _git_metadata(ROOT), "host": _host_metadata(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_identity": identity,
            "strict_manifest_sha256": strict_profile.manifest_sha256,
            "protocol": {"prefill_prompts": 18, "outlier_decode_steps": 32,
                         "rows": len(descriptors), "chunk": 1024, "kv": "BF16",
                         "repeats": 1, "task_gate": "not_run",
                         "graph_cache_reset_per_arm": True},
            "arms": {}, "strict_after_close": strict_close,
        }
        generator, production_profile, _ = _make_generator(args, "production")
        production_snapshot = gr_wide_snapshot(
            os.environ, published=bool(generator.runner.gr_wide))
        snapshot_arms(strict_snapshot, production_snapshot)
        # Move every arm's former GR-up wide value onto the owner decision and
        # drop the retired key, so no arm writes it as a process value.
        _GR_WIDE_ARM_VALUES.update(migrate_gr_wide_arms(
            ARMS, production_value=bool(generator.runner.gr_wide),
            declared=declared_gr_wide()))
        # The strict binding historically published Q8 MMQ off. Bind that owner
        # decision explicitly now that the process key is gone.
        for name in tuple(ARMS):
            if (name.startswith("strict_plus_")
                    or name in {"all_flags_strict", "gdn_flags_strict",
                                "q8down_off_without_dense",
                                "q8down_off_without_qsa"}):
                _Q8_MMQ_ARM_VALUES.setdefault(name, False)
        controlled = set(key for values in ARMS.values() for key in values)
        bound = {key: os.environ.get(key) for key in controlled}
        try:
            for arm in args.arms:
                # Family overrides can affect captured nodes beyond the normal
                # production cache key. Never reuse an earlier arm's graphs.
                clear_arm_graphs(generator.runner)
                for key, value in bound.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                os.environ.update(ARMS[arm])
                candidate = []
                arm_scope = ExitStack()
                arm_scope.enter_context(production_moe_arm_scope(generator, arm))
                arm_scope.enter_context(q8_mmq_arm_scope(generator, arm))
                arm_scope.enter_context(q8_down_arm_scope(generator, arm))
                arm_scope.enter_context(gdn_tile16_arm_scope(generator, arm))
                if resolve_q8_mmq_prefill(generator):
                    # Explicit cold-path preparation for the arm's bound owner.
                    generator.runner.configure_mmq_prefill_resources()
                arm_scope.enter_context(q4_dp4a_arm_scope(generator, arm))
                arm_scope.enter_context(gr_wide(
                    generator.runner, gr_wide_arm_enabled(
                        arm, getattr(generator.runner, "gr_wide", False))))
                arm_scope.enter_context(gr_wave_scale(
                    generator.runner, gr_wave_arm_enabled(
                        arm, generator.runner.gr_wave_scale)))
                arm_scope.enter_context(q8_wave_scale_prefill(
                    generator.runner, wave_arm_enabled(
                        arm, generator.runner.q8_wave_scale_prefill)))
                arm_scope.enter_context(q4_pair_prefill(
                    generator.runner, pair_arm_enabled(
                        arm, generator.runner.q4_pair_prefill)))
                arm_scope.enter_context(gdn_register_prefill(
                    generator.runner, register_arm_enabled(
                        arm, generator.runner.gdn_register_prefill)))
                arm_scope.enter_context(qsa_h256_prefill_mode(
                    generator.runner, h256_arm_mode(arm)))
                if serial_gdn({**bound, **ARMS[arm]}):
                    arm_scope.enter_context(
                        gdn_peer_prefill(generator.runner, False))
                with arm_scope:
                    for row in prompt_rows:
                        key = row["id"]
                        forced = [r["token_id"] for r in strict[key][:-1]]
                        if arm in ("strict_prefill_production_decode",
                                   "production_prefill_strict_decode"):
                            pre, dec = ({**bound, **strict_flags}, bound)
                            if arm == "production_prefill_strict_decode":
                                pre, dec = dec, pre
                            candidate.extend(split_trajectory(
                                generator.runner, tokens[key], forced, pre, dec,
                                arm=arm,
                                register_prefill=arm != "strict_prefill_production_decode",
                                pair_prefill=arm != "strict_prefill_production_decode",
                                wave_prefill=arm != "strict_prefill_production_decode",
                                gr_wave_prefill=arm != "strict_prefill_production_decode",
                                h256_mode=("parent" if arm == "strict_prefill_production_decode"
                                           else None)))
                        else:
                            run = _candidate_trajectory(generator.runner, tokens[key], forced)
                            candidate.extend(r["logits"] for r in run)
                candidate_logits = np.stack(candidate)
                quality = compare_profile_logits(strict_logits, candidate_logits, descriptors)
                quality["all_rows"] = [
                    {**descriptor.to_dict(), **compare_profile_logits(
                        strict_logits[i:i + 1], candidate_logits[i:i + 1],
                        [descriptor])["summary"]}
                    for i, descriptor in enumerate(descriptors)
                ]
                report["arms"][arm] = {
                    "overrides": ARMS[arm],
                    "owner_overrides": {
                        "production_moe_prefill_enabled": (
                            production_moe_arm_report(generator, arm)),
                        "q8_mmq_prefill_enabled": q8_mmq_arm_report(
                            generator, arm),
                        "q8_down_variant": q8_down_arm_report(generator, arm),
                        "gdn_tile16_variant": gdn_tile16_arm_report(generator, arm),
                        "gr_wide": gr_wide_arm_report(generator, arm),
                    },
                    "quality": quality,
                }
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(arm, quality["summary"], flush=True)
            report["status"] = "completed"
        finally:
            for key, value in bound.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            generator.close()
            report["after_close"] = memory_stats()
            report["production_manifest_sha256"] = production_profile.manifest_sha256
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
