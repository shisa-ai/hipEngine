"""Startup readiness guidance names the remedy the failure actually has.

``resident_prepare`` covers both capacity failures and configuration failures.
Telling an operator to lower ``--max-context-tokens`` when the real problem is a
width bound above the qualified maximum sends them to the wrong setting, so the
allocation hint is reserved for allocation errors.
"""

from __future__ import annotations

import pytest

from hipengine.server.api import _resident_prepare_guidance

_ALLOCATION_HINT = "Try a lower --max-context-tokens or --kv-storage int8_per_token_head."


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError("hipMalloc failed"),
        ValueError(
            "Qwen4Exp context 2048 at c2 needs 95801389408 bytes including reserve; "
            "available 29478170624"
        ),
        RuntimeError("workspace requires 12 GiB"),
        RuntimeError("insufficient device memory"),
        RuntimeError("OOM while preparing KV pool"),
    ],
)
def test_an_allocation_failure_keeps_the_capacity_hint(exc: BaseException) -> None:
    assert _resident_prepare_guidance(exc) == _ALLOCATION_HINT


@pytest.mark.parametrize(
    "exc",
    [
        ValueError(
            "HIPENGINE_QWEN4_EXP_AR_BATCH_SLOTS=9 exceeds the qualified physical width 4"
        ),
        ValueError("--batch-width 4 is unsupported by the long-context gate"),
        RuntimeError("Qwen4Exp AR batch decode requires model-owned resident weights"),
        RuntimeError("unknown backend hip_nope"),
    ],
)
def test_a_configuration_failure_points_at_the_setting(exc: BaseException) -> None:
    guidance = _resident_prepare_guidance(exc)

    assert guidance != _ALLOCATION_HINT
    assert "configuration error" in guidance


def test_a_resident_slot_refusal_names_both_fixes() -> None:
    """A slot bound at a long context is not fixed by a KV-storage change."""

    guidance = _resident_prepare_guidance(
        ValueError(
            "Qwen4Exp resident capacity must be within 1..2 for "
            "max_sequence_length=262144; the production bound is 2"
        )
    )

    assert guidance != _ALLOCATION_HINT
    assert "--max-active-requests" in guidance
    assert "--max-context-tokens" in guidance
