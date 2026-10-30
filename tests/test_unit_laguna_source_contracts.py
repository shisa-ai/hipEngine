"""RED/GREEN mutation coverage for the scoped Laguna source contracts (RF-11).

The three campaign source-default tests used to hash whole shared files. Those
pins failed on edits that could not affect the campaign and passed only after a
re-baseline, which is not evidence that behavior was preserved. Each contract in
``tests/_laguna_source_contracts.py`` is exercised here in both directions:

* an unrelated edit to the same file is accepted (the contract is not a hash);
* a mutation of the protected capability, dispatch boundary, ownership guard or
  RED coverage is rejected, naming the contract.

Mutations are applied to source *text* in memory. No runtime, kernel or test
file is written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import _laguna_source_contracts as contracts

_ROOT = Path(__file__).parents[1]


def _live(relative: str) -> str:
    return (_ROOT / relative).read_text()


_CONTRACTS = {
    "h8a-runner-owner": (contracts.assert_h8a_runner_owner, contracts.RUNNER),
    "h8b-shared-pack-scope": (
        contracts.assert_h8b_shared_pack_scope,
        contracts.MOE,
    ),
    "h7u-group-compact-resolver": (
        contracts.assert_h7u_group_compact_resolver,
        contracts.MOE,
    ),
    "h8a-gpu-red": (contracts.assert_h8a_gpu_red, contracts.H8A_GPU_RED),
    "h8b-unit-red": (contracts.assert_h8b_unit_red, contracts.H8B_UNIT_RED),
    "h7u-live-mode-red": (contracts.assert_h7u_live_mode_red, contracts.H7U_LIVE_RED),
}


@pytest.mark.parametrize("contract_id", sorted(_CONTRACTS))
def test_scoped_contract_accepts_live_source(contract_id: str) -> None:
    contract, relative = _CONTRACTS[contract_id]
    contract(_live(relative))


# An unrelated edit inside the same file. Each one touches a symbol or region
# the named contract does not own, so a whole-file hash would fail but the
# scoped contract must not.
_UNRELATED_EDITS = {
    "h8a-runner-owner": (
        '"""Resolve H8B source ownership from the backend capability."""',
        '"""Resolve the H8B pack-reuse policy; unrelated to the H8A cache."""',
    ),
    "h8b-shared-pack-scope": (
        '"""Resolve stable serial or architecture-qualified parallel compaction."""',
        '"""Resolve the H7U compaction mode; unrelated to pack reuse."""',
    ),
    "h7u-group-compact-resolver": (
        '"""Launch the independent always-on shared expert branch."""',
        '"""Launch the shared expert branch; unrelated to compaction mode."""',
    ),
    "h8a-gpu-red": (
        "_ROWS = (1, 7, 8, 9, 512)",
        "_ROWS = (\n    1,\n    7,\n    8,\n    9,\n    512,\n)",
    ),
    "h8b-unit-red": (
        "_NO_SALVAGE = (",
        "# Unrelated docstring/formatting churn below.\n_NO_SALVAGE = (",
    ),
    "h7u-live-mode-red": (
        "def test_laguna_grouped_exact_quant_miss_fails_closed_to_direct() -> None:",
        (
            "def test_laguna_grouped_exact_quant_miss_fails_closed_to_direct() -> None:\n"
            "    # Unrelated edit: the H7U contract owns only the mode test.\n"
            "    pass"
        ),
    ),
}


@pytest.mark.parametrize("contract_id", sorted(_CONTRACTS))
def test_scoped_contract_accepts_unrelated_edit(contract_id: str) -> None:
    contract, relative = _CONTRACTS[contract_id]
    source = _live(relative)
    old, new = _UNRELATED_EDITS[contract_id]
    assert source.count(old) == 1, f"unrelated-edit anchor moved: {old!r}"
    contract(source.replace(old, new))


# (contract, anchor, mutation, expected message fragment)
_PROTECTED_MUTATIONS = (
    pytest.param(
        "h8a-runner-owner",
        "        backend_package_capability(\n"
        "            backend,\n"
        '            "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE",\n'
        "            False,\n"
        "        )\n"
        "    )",
        "        backend_package_capability(\n"
        "            backend,\n"
        '            "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE",\n'
        "            False,\n"
        "        ) or True\n"
        "    )",
        "resolver must return the capability value",
        id="h8a-resolver-value-bypass",
    ),
    pytest.param(
        "h8a-runner-owner",
        "if len(targets) != 24 or len(raw_ptrs) != 24:",
        "if len(targets) == 24 or len(raw_ptrs) == 24:",
        "reject non-24 plane counts",
        id="h8a-plane-guard-inverted",
    ),
    pytest.param(
        "h8a-runner-owner",
        "            release(lambda: cache.free(runtime=self.runtime))\n"
        "        else:\n"
        "            self.q5_f32_resident_cache = None",
        "            pass\n"
        "        else:\n"
        "            self.q5_f32_resident_cache = None",
        "owned cache release must stay inside",
        id="h8a-owned-release-deleted",
    ),
    pytest.param(
        "h8b-shared-pack-scope",
        "    with activation_pack_reuse_scope(enabled=use_activation_pack_reuse):\n",
        "    with activation_pack_reuse_scope(enabled=use_activation_pack_reuse):\n"
        "        pass\n"
        "    if True:\n",
        "shared gate/up launches must stay inside",
        id="h8b-launches-outside-scope",
    ),
    pytest.param(
        "h7u-group-compact-resolver",
        '            "LAGUNA_MOE_GROUP_COMPACT_MODE",\n'
        "            _BASELINE_GROUP_COMPACT_MODE,\n"
        "        )\n        if requested is None\n"
        "        else str(requested)\n    )\n    parsed = str(selected)",
        '            "LAGUNA_MOE_GROUP_COMPACT_MODE",\n'
        "            _BASELINE_GROUP_COMPACT_MODE,\n"
        "        )\n        if requested is None\n"
        "        else 'serial'\n    )\n    parsed = str(selected)",
        "explicit requested mode must override",
        id="h7u-ignore-explicit-request",
    ),
    pytest.param(
        "h8a-runner-owner",
        '            "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE",\n'
        "            False,\n"
        "        ):\n"
        '            raise ValueError(\n'
        '                "resident global-Q5 F32 cache requires its source-qualified "',
        '            "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE_V2",\n'
        "            False,\n"
        "        ):\n"
        '            raise ValueError(\n'
        '                "resident global-Q5 F32 cache requires its source-qualified "',
        "must be read by the session resolver and the allocation precondition only",
        id="h8a-capability-swap",
    ),
    pytest.param(
        "h8a-runner-owner",
        "if len(targets) != 24 or len(raw_ptrs) != 24:",
        "if len(targets) != 12 or len(raw_ptrs) != 12:",
        "must stay pinned to 24 resident planes",
        id="h8a-plane-count",
    ),
    pytest.param(
        "h8a-runner-owner",
        "                elif (\n"
        "                    self.use_q5_f32_resident_global_cache\n"
        "                    and self._owns_weights\n"
        "                ):",
        "                elif (\n"
        "                    self.use_q5_f32_resident_global_cache\n"
        "                    or True\n"
        "                ):",
        "gated by the H8A policy and self._owns_weights",
        id="h8a-ownership-gate",
    ),
    pytest.param(
        "h8a-runner-owner",
        "        self.use_q5_f32_resident_global_cache = (\n"
        "            resolve_laguna_q5_f32_resident_global_cache(self.backend)\n"
        "        )",
        "        self.use_q5_f32_resident_global_cache = True",
        "must come from the capability resolver",
        id="h8a-session-policy-source",
    ),
    pytest.param(
        "h8a-runner-owner",
        "            self.q5_f32_resident_cache is not None\n"
        "            and self._owns_q5_f32_resident_cache\n",
        "            self.q5_f32_resident_cache is not None\n"
        "            and True\n",
        "_close must free the resident cache only when the session owns it",
        id="h8a-close-ownership",
    ),
    pytest.param(
        "h8b-shared-pack-scope",
        "with activation_pack_reuse_scope(enabled=use_activation_pack_reuse):",
        "with activation_pack_reuse_scope(enabled=True):",
        "the scope must follow use_activation_pack_reuse",
        id="h8b-scope-bit",
    ),
    pytest.param(
        "h8b-shared-pack-scope",
        "    use_activation_pack_reuse: bool = False,",
        "    use_activation_pack_reuse: bool = True,",
        "must default pack reuse off",
        id="h8b-default-flip",
    ),
    pytest.param(
        "h8b-shared-pack-scope",
        "            use_activation_pack_reuse=use_activation_pack_reuse,\n"
        "            stream=shared_stream,",
        "            stream=shared_stream,",
        "a shared launch site dropped its pack-reuse forwarding",
        id="h8b-forwarding-drop",
    ),
    pytest.param(
        "h7u-group-compact-resolver",
        '_BASELINE_GROUP_COMPACT_MODE = "serial"',
        '_BASELINE_GROUP_COMPACT_MODE = "parallel"',
        "serial rollback baseline must stay the default",
        id="h7u-baseline-flip",
    ),
    pytest.param(
        "h7u-group-compact-resolver",
        '            "LAGUNA_MOE_GROUP_COMPACT_MODE",\n'
        "            _BASELINE_GROUP_COMPACT_MODE,",
        '            "LAGUNA_MOE_GROUP_COMPACT_MODE_V2",\n'
        "            _BASELINE_GROUP_COMPACT_MODE,",
        "has no backend_package_capability read",
        id="h7u-capability-swap",
    ),
    pytest.param(
        "h7u-group-compact-resolver",
        "    if parsed not in _GROUP_COMPACT_MODES:\n"
        '        raise ValueError("unsupported Laguna MoE group compact mode")',
        "    if parsed not in _GROUP_COMPACT_MODES:\n        pass",
        "an unknown compaction mode must raise ValueError",
        id="h7u-fail-closed",
    ),
    pytest.param(
        "h8a-gpu-red",
        "_PLANE_COUNT = 24",
        "_PLANE_COUNT = 12",
        "no longer covers 24 resident planes",
        id="h8a-red-plane-count",
    ),
    pytest.param(
        "h8a-gpu-red",
        "def test_h8a_owner_failed_setup_rolls_back_without_publishing_partial_map(",
        "def test_h8a_owner_failed_setup_is_gone(",
        "missing",
        id="h8a-red-rollback-deleted",
    ),
    pytest.param(
        "h8b-unit-red",
        '_SOURCE_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE"',
        '_SOURCE_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE_V2"',
        "no longer targets",
        id="h8b-red-capability",
    ),
    pytest.param(
        "h8b-unit-red",
        "def test_h8b_scope_publishes_only_after_success_and_isolates_every_key_axis(",
        "def test_h8b_scope_is_gone(",
        "missing",
        id="h8b-red-isolation-deleted",
    ),
    pytest.param(
        "h8b-unit-red",
        '"full_attention_qkv": (12, 3, 24),',
        '"full_attention_qkv": (11, 3, 24),',
        "no longer covers the 95 recurrence groups",
        id="h8b-red-class-coverage",
    ),
    pytest.param(
        "h7u-live-mode-red",
        '    assert resolve_laguna_group_compact_mode("hip_gfx1100") == "parallel"',
        '    assert resolve_laguna_group_compact_mode("hip_gfx1100") == "serial"',
        "dropped backend-qualified mode assertions",
        id="h7u-live-default-flip",
    ),
    pytest.param(
        "h7u-live-mode-red",
        '        resolve_laguna_group_compact_mode("hip_gfx1151", "atomic")',
        '        resolve_laguna_group_compact_mode("hip_gfx1151", "serial")',
        "no longer rejects the 'atomic' mode",
        id="h7u-live-fail-closed",
    ),
)


@pytest.mark.parametrize(
    ("contract_id", "anchor", "mutation", "message"),
    _PROTECTED_MUTATIONS,
)
def test_scoped_contract_rejects_protected_mutation(
    contract_id: str,
    anchor: str,
    mutation: str,
    message: str,
) -> None:
    contract, relative = _CONTRACTS[contract_id]
    source = _live(relative)
    assert source.count(anchor) == 1, f"mutation anchor moved: {anchor!r}"
    mutated = source.replace(anchor, mutation)
    assert mutated != source
    with pytest.raises(AssertionError, match=message):
        contract(mutated)
