"""CPU tests for AR batch admission eligibility and manifest binding.

The normative profile gate is 18 named prompts x (24 decode steps + 1
prefill-last row) = 450 teacher-forced rows. A shorter run can validate the
adapter protocol, but it must never be admitted as the production numerical
gate. These tests pin that distinction without loading a model.
"""

from __future__ import annotations

import pytest

from scripts.qwen4exp_layer2_profile_gate import (
    BATCH_NORMATIVE_DECODE_STEPS,
    BATCH_NORMATIVE_FREE_RUNS,
    BATCH_NORMATIVE_FREE_TOKENS,
    BATCH_NORMATIVE_REPEAT_RUNS,
    BATCH_NORMATIVE_TEACHER_ROWS,
    batch_admission_decision,
    batch_admission_eligibility,
)

NAMED_PROMPTS = tuple(f"prompt_{index:02d}" for index in range(18))
BOUND_MANIFEST = {"bound": True, "missing": [], "identities": []}


def eligibility_kwargs(**overrides):
    values = {
        "prompt_ids": NAMED_PROMPTS,
        "named_prompt_ids": NAMED_PROMPTS,
        "prompt_selection_unrestricted": True,
        "decode_steps": BATCH_NORMATIVE_DECODE_STEPS,
        "repeat_runs": BATCH_NORMATIVE_REPEAT_RUNS,
        "free_tokens": BATCH_NORMATIVE_FREE_TOKENS,
        "free_runs": BATCH_NORMATIVE_FREE_RUNS,
        "tracked_clean": True,
        "manifest_binding": BOUND_MANIFEST,
    }
    values.update(overrides)
    return values


# -- normative scope --------------------------------------------------------


def test_normative_scope_is_eligible():
    eligibility = batch_admission_eligibility(**eligibility_kwargs())
    assert eligibility["eligible"] is True
    assert eligibility["reasons"] == []
    assert eligibility["requirements"]["teacher_rows"] == 450
    assert eligibility["requirements"]["decode_steps"] == 24
    assert eligibility["requirements"]["repeat_runs"] == 3
    assert eligibility["requirements"]["free_tokens"] == 32
    assert eligibility["requirements"]["free_runs"] == 2


def test_eight_step_clean_tree_is_ineligible_and_cannot_be_admitted():
    eligibility = batch_admission_eligibility(**eligibility_kwargs(decode_steps=8))
    assert eligibility["eligible"] is False
    assert any("decode_steps" in reason for reason in eligibility["reasons"])

    # Defence in depth: even if a caller wrongly reported a valid measurement,
    # the admission decision still refuses the short run.
    decision = batch_admission_decision(
        eligibility=eligibility,
        measurement_valid=True,
        numerical_pass=True,
        task_pass=True,
    )
    assert decision["admitted"] is False
    assert any("decode_steps" in reason for reason in decision["reasons"])


def test_dirty_tree_is_ineligible():
    eligibility = batch_admission_eligibility(**eligibility_kwargs(tracked_clean=False))
    assert eligibility["eligible"] is False
    assert "tracked_tree_not_clean" in eligibility["reasons"]


def test_prompt_subset_is_ineligible():
    eligibility = batch_admission_eligibility(
        **eligibility_kwargs(prompt_ids=NAMED_PROMPTS[:4])
    )
    assert eligibility["eligible"] is False
    assert "prompt_set_not_named_fixture" in eligibility["reasons"]


def test_limited_prompt_selection_is_ineligible_even_with_all_prompts():
    eligibility = batch_admission_eligibility(
        **eligibility_kwargs(prompt_selection_unrestricted=False)
    )
    assert eligibility["eligible"] is False
    assert "prompt_selection_truncated" in eligibility["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("repeat_runs", 2, "repeat_runs"),
        ("free_tokens", 8, "free_tokens"),
        ("free_runs", 1, "free_runs"),
    ],
)
def test_short_protocol_parameters_are_ineligible(field, value, reason):
    eligibility = batch_admission_eligibility(**eligibility_kwargs(**{field: value}))
    assert eligibility["eligible"] is False
    assert any(reason in item for item in eligibility["reasons"])


# -- manifest binding -------------------------------------------------------


def test_structured_invalid_binding_blocks_admission_even_if_legacy_bound_is_true():
    binding = {
        "bound": True,
        "valid": False,
        "reasons": ["direct_identity_mismatch:gdn_conv_indexed"],
        "missing": [],
    }
    eligibility = batch_admission_eligibility(
        **eligibility_kwargs(manifest_binding=binding)
    )
    assert eligibility["eligible"] is False
    assert any(
        "batch_manifest:direct_identity_mismatch" in reason
        for reason in eligibility["reasons"]
    )
    decision = batch_admission_decision(
        eligibility=eligibility,
        measurement_valid=True,
        numerical_pass=True,
        task_pass=True,
    )
    assert decision["admitted"] is False


# -- decision ---------------------------------------------------------------


def test_normative_clean_full_run_is_admitted_only_when_all_gates_pass():
    eligibility = batch_admission_eligibility(**eligibility_kwargs())
    decision = batch_admission_decision(
        eligibility=eligibility,
        measurement_valid=True,
        numerical_pass=True,
        task_pass=True,
    )
    assert decision["admitted"] is True
    assert decision["reasons"] == []


def test_task_review_blocks_admission():
    eligibility = batch_admission_eligibility(**eligibility_kwargs())
    decision = batch_admission_decision(
        eligibility=eligibility,
        measurement_valid=True,
        numerical_pass=True,
        task_pass=False,
    )
    assert decision["admitted"] is False
    assert "task_gate_requires_review" in decision["reasons"]


def test_protocol_check_never_admits():
    eligibility = batch_admission_eligibility(**eligibility_kwargs())
    decision = batch_admission_decision(
        eligibility=eligibility,
        measurement_valid=True,
        numerical_pass=True,
        task_pass=True,
        protocol_check=True,
    )
    assert decision["admitted"] is False
    assert "protocol_check_only" in decision["reasons"]


def test_numerical_failure_blocks_admission():
    eligibility = batch_admission_eligibility(**eligibility_kwargs())
    decision = batch_admission_decision(
        eligibility=eligibility,
        measurement_valid=True,
        numerical_pass=False,
        task_pass=True,
    )
    assert decision["admitted"] is False
    assert "numerical_gate_failed" in decision["reasons"]
