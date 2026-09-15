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
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL":
            "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS",
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
    covered = {
        key.replace("HIPENGINE_QWEN4_EXP_", "")
        for overrides in module.ARMS.values()
        for key in overrides
    }
    assert still_disabled <= covered, sorted(still_disabled - covered)


def test_the_restored_flags_are_not_in_the_disabled_set(module):
    """The tree's restored set is smaller than the disabled-selector narrative."""
    from hipengine.generation.qwen4_exp_profiles import (
        PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
        PRODUCTION_Q8_QSA_RESTORED_FLAGS,
    )

    recovery = set(PRODUCTION_ARITHMETIC_RECOVERY_FLAGS)
    restored = set(PRODUCTION_Q8_QSA_RESTORED_FLAGS)
    assert len(PRODUCTION_ARITHMETIC_RECOVERY_FLAGS) == 15
    assert len(PRODUCTION_Q8_QSA_RESTORED_FLAGS) == 6
    # Five of the fifteen selectors are re-admitted; Q8_DOWN_VARIANT is the
    # companion value of one of them rather than a selector in its own right.
    assert len(restored & recovery) == 5
    assert restored - recovery == {"Q8_DOWN_VARIANT"}
    still_disabled = [flag for flag in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS
                      if flag not in restored]
    assert len(still_disabled) == 10


def test_baseline_arm_enables_nothing(module):
    assert module.ARMS["baseline"] == {}
