"""The retired Q8 MMQ attention-gate candidate fails loudly instead of being ignored."""
import inspect

import pytest

from hipengine.runtime import qwen4_exp_runner as runtime
from scripts import qwen4exp_candidate_ar as candidate_ar
from scripts import qwen4exp_profile_gap as gap

ENV = "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE"


def test_candidate_ar_rejects_the_retired_override():
    assert candidate_ar.split_owner_overrides(["HIPENGINE_X=1"]) == {"HIPENGINE_X": "1"}
    with pytest.raises(ValueError, match="retired"):
        candidate_ar.split_owner_overrides([f"{ENV}=1"])


def test_profile_gap_rejects_the_retired_override():
    with pytest.raises(ValueError, match="retired"):
        gap._parse_overrides([f"{ENV}=1"])


def test_runtime_has_no_attention_gate_owner():
    assert not hasattr(runtime, "q8_mmq_attn_gate_enabled")
    assert "attn_gate_enabled" not in inspect.signature(runtime._qwen4_exp_q8_mmq_policy).parameters
