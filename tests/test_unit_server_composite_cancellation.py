"""A coalesced MTP batch must not crash the engine tick on a cancellation probe.

Two concurrent MTP requests share one backend call, so the server hands the
generation a composite cancellation token. The serving MTP provider probes
``cancel_requested`` on that token from inside the driver tick; a token without
the attribute raised ``AttributeError``, which the engine service treats as a
driver failure and tears the resident engine down, failing every later request.
"""

import types

import pytest

from hipengine.generation.deadline import (
    FinishDetails,
    GenerationCancellationToken,
    generation_deadline_expired,
)
from hipengine.generation.qwen4_exp_serving_mtp import Qwen4ExpServingMTP
from hipengine.server.api import _CompositeGenerationCancellationToken


def test_composite_token_exposes_cancel_requested():
    first, second = GenerationCancellationToken(), GenerationCancellationToken()
    composite = _CompositeGenerationCancellationToken((first, second))
    assert composite.cancel_requested is False
    first.cancel()
    # The shared call is only abandoned once no live member remains.
    assert composite.cancel_requested is False
    second.cancel()
    assert composite.cancel_requested is True


def test_composite_token_matches_its_cancelled_view():
    first, second = GenerationCancellationToken(), GenerationCancellationToken()
    composite = _CompositeGenerationCancellationToken((first, second))
    assert composite.cancel_requested == composite.cancelled
    first.cancel(FinishDetails(reason="timeout", cancelled=True, deadline_exceeded=True))
    second.cancel(FinishDetails(reason="timeout", cancelled=True, deadline_exceeded=True))
    assert composite.cancel_requested == composite.cancelled is True
    assert composite.finish_details.reason == "timeout"


class _Row:
    def __init__(self, token):
        self.request = types.SimpleNamespace(
            deadline_at=None, cancellation_token=token
        )
        self.lifetime_request = None


def _provider(row):
    provider = object.__new__(Qwen4ExpServingMTP)
    provider._row = lambda request_id: row
    return provider


def test_request_cancel_reason_tolerates_a_token_without_cancel_requested():
    # A token that only exposes the base `cancelled` contract must not raise
    # inside the tick; that AttributeError killed the resident engine.
    provider = _provider(_Row(types.SimpleNamespace(cancelled=False)))
    assert provider.request_cancel_reason(0) is None


def test_request_cancel_reason_honors_a_requested_cancel():
    provider = _provider(
        _Row(
            types.SimpleNamespace(
                cancelled=False,
                cancel_requested=True,
                finish_details=FinishDetails(reason="disconnect", cancelled=True),
            )
        )
    )
    assert provider.request_cancel_reason(0) == "disconnect"


def test_request_cancel_reason_reports_a_cancelled_token():
    token = types.SimpleNamespace(cancelled=True, finish_details=FinishDetails(reason="disconnect", cancelled=True))
    provider = _provider(_Row(token))
    assert provider.request_cancel_reason(0) == "disconnect"


def test_request_cancel_reason_reports_a_composite_disconnect():
    first, second = GenerationCancellationToken(), GenerationCancellationToken()
    composite = _CompositeGenerationCancellationToken((first, second))
    first.cancel(FinishDetails(reason="disconnect", cancelled=True))
    second.cancel(FinishDetails(reason="disconnect", cancelled=True))
    provider = _provider(_Row(composite))
    assert provider.request_cancel_reason(0) == "disconnect"


def test_generation_deadline_helper_is_the_probe_contract():
    # Guards the import used by the provider above.
    assert generation_deadline_expired(None) is False
