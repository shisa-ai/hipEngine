"""Startup readiness guidance names the remedy the failure actually has.

``resident_prepare`` covers both capacity failures and configuration failures.
Telling an operator to lower ``--max-context-tokens`` when the real problem is a
width bound above the qualified maximum sends them to the wrong setting, so the
allocation hint is reserved for allocation errors.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from hipengine.server import api as _api
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
            "Qwen4Exp active request count must be within 1..64 (the indexed batch "
            "decode kernels take at most 64 rows)"
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


@pytest.mark.parametrize("storage, suggests_int8", [("bf16", True), ("int8_per_token_head", False)])
@pytest.mark.parametrize("fits_requested", [True, False])
def test_capacity_warning_suggests_int8_only_when_it_would_help(
    caplog, storage, suggests_int8, fits_requested
) -> None:
    estimate = SimpleNamespace(
        kv_storage_dtype=storage, fits_requested=fits_requested, fits_model_max=False,
        model_max_context_tokens=262_144, requested_context_tokens=196_608,
        allocatable_context_tokens=0, max_batch_size=1,
    )
    engine = SimpleNamespace(kv_capacity_estimate=estimate)
    with caplog.at_level(logging.WARNING, logger=_api._LOGGER.name):
        _api._log_kv_capacity_summary(engine)
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings
    assert any("int8_per_token_head" in message for message in warnings) is suggests_int8
