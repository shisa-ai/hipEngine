"""Guards for the disabled-selector ranking harness arm table.

The harness itself needs the GPU and the pinned model. These tests cover the
part that can be wrong silently: an arm that claims to enable a selector but
omits the layer companion the production binder would have supplied, or names a
key the child would reject. An arm like that measures a no-op and reports it as
a neutral selector.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "qwen4exp_disabled_selector_ranking.py"


def _load():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_disabled_selector_ranking", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load()


def test_every_override_key_is_a_valid_child_key(module):
    for name, overrides in module.ARMS.items():
        for key, value in overrides.items():
            assert key.startswith("HIPENGINE_"), (name, key)
            assert value != "", (name, key)


def test_layer_gated_selectors_carry_their_layer_companion(module):
    # The production binder blanks these layer lists when it disables the
    # selector, so enabling the selector alone would be a no-op.
    pairs = {
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL":
            "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS",
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL":
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS",
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL":
            "HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS",
    }
    for name, overrides in module.ARMS.items():
        for selector, layers in pairs.items():
            if selector in overrides:
                assert layers in overrides, (name, selector)
                assert overrides[layers], (name, layers)


def test_the_shadowed_selector_arm_disables_its_shadow(module):
    """The exact iu8-risk variant shadows Q4_IU8_PREFILL in the same chain.

    The chain's own gate is request-local runner state
    (``q4_exact_grouped_gate``), so an environment-only arm cannot clear it;
    ``q4_iu8_fast`` closes the iu8-risk *variant* instead, which leaves the chain
    on its own out4/expertgrid fallback.
    """
    shadow = "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT"
    for name, overrides in module.ARMS.items():
        if "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL" in overrides:
            assert overrides.get(shadow) == "0", name


def test_arms_cover_every_disabled_prefill_selector(module):
    from hipengine.generation.qwen4_exp_profiles import (
        PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
        PRODUCTION_Q8_QSA_RESTORED_FLAGS,
    )

    still_disabled = {
        flag for flag in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS
        if flag not in PRODUCTION_Q8_QSA_RESTORED_FLAGS
    }
    # Decode-only: it does not affect prefill wall time, so it is out of scope.
    still_disabled.discard("Q4_DP4A64")
    # The GDN peer owner is request-local runner state, not a recovery flag: the
    # columnwarp owner ahead of it is open on this lane, so no environment arm
    # could reach it. The layer2 gate's GDN candidates bind the scope instead.
    still_disabled.discard("GDN_PEER_PREFILL")
    covered = {
        key.replace("HIPENGINE_QWEN4_EXP_", "")
        for overrides in module.ARMS.values()
        for key in overrides
    }
    for owner in module.OWNER_ARMS.values():
        if "production_moe_prefill" in owner:
            covered.add("PRODUCTION_MOE_PREFILL")
    assert still_disabled <= covered, sorted(still_disabled - covered)

def test_the_restored_flags_are_not_in_the_disabled_set(module):
    """The tree's restored set is smaller than the disabled-selector narrative."""
    from hipengine.generation.qwen4_exp_profiles import (
        PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
        PRODUCTION_Q8_QSA_RESTORED_FLAGS,
    )

    recovery = set(PRODUCTION_ARITHMETIC_RECOVERY_FLAGS)
    restored = set(PRODUCTION_Q8_QSA_RESTORED_FLAGS)
    # H256 prefill, head grouping, ordered-v2 decode and GDN peer scope
    # are runner-owned, not recovery or restored environment values.
    assert len(PRODUCTION_ARITHMETIC_RECOVERY_FLAGS) == 10
    assert len(PRODUCTION_Q8_QSA_RESTORED_FLAGS) == 3
    # Two selectors are restored; Q8_DOWN_VARIANT is a companion value.
    # Q4 decode now keeps the strict owner after the long-context failure.
    assert "Q4_DP4A64" in recovery - restored
    assert len(restored & recovery) == 2
    assert restored - recovery == {"Q8_DOWN_VARIANT"}
    still_disabled = [flag for flag in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS
                      if flag not in restored]
    assert len(still_disabled) == 8


def test_production_moe_is_owner_state_not_an_environment_arm(module):
    key = "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL"
    assert all(key not in overrides for overrides in module.ARMS.values())
    assert module.OWNER_ARMS["moe"] == {"production_moe_prefill": "on"}
    assert module.OWNER_ARMS["all_prefill"] == {"production_moe_prefill": "on"}


def test_baseline_arm_enables_nothing(module):
    assert module.ARMS["baseline"] == {}
