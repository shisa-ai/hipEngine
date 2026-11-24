"""Cancellation intent reaches every owner without losing child-local details."""
from types import SimpleNamespace

import pytest

from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.engine_service import _ChildCancellationToken
from hipengine.generation.registry import FinishDetails


def test_parent_cancel_replays_to_an_owner_registered_after_its_snapshot():
    parent = GenerationCancellationToken()
    first = _ChildCancellationToken(parent)
    second = _ChildCancellationToken(parent)
    seen = []
    first_dispatch = lambda details: seen.append(('first', details.reason))
    second_dispatch = lambda details: seen.append(('second', details.reason))
    first.set_cancel_dispatch(first_dispatch)
    parent.cancel()
    assert seen == [('first', 'cancelled')]
    assert not parent.cancelled
    second.set_cancel_dispatch(second_dispatch)
    assert seen == [('first', 'cancelled'), ('second', 'cancelled')]
    first.clear_cancel_dispatch(first_dispatch)
    assert not parent.cancelled
    second.clear_cancel_dispatch(second_dispatch)
    assert parent.cancelled


def test_child_acknowledgement_preserves_its_deadline_details():
    parent = GenerationCancellationToken()
    child = _ChildCancellationToken(parent)
    details = FinishDetails(reason='deadline_exceeded', deadline_exceeded=True)
    child.acknowledge_cancel(details)
    assert child.cancelled
    assert child.finish_details == details
    assert not parent.cancel_requested and not parent.cancelled
    assert not parent.finish_details.deadline_exceeded


@pytest.mark.parametrize('cancelled', [False, True])
def test_child_view_accepts_observable_tokens_without_dispatch_methods(cancelled):
    parent = SimpleNamespace(cancelled=cancelled, finish_details=FinishDetails(
        reason='deadline_exceeded', deadline_exceeded=True,
    ))
    child = _ChildCancellationToken(parent)
    dispatch = lambda details: None
    child.set_cancel_dispatch(dispatch)
    assert child.cancel_requested == cancelled
    assert child.cancelled == cancelled
    assert child.finish_details == parent.finish_details
    child.clear_cancel_dispatch(dispatch)
