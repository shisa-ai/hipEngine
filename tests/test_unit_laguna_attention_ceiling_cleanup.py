"""Timing event ownership in maintained Laguna attention comparisons."""

import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("module", [
    "scripts.laguna_attention_long_context_hipblaslt_ceiling",
    "scripts.laguna_swa_attention_hipblaslt_ceiling",
])
@pytest.mark.parametrize("failure", ["second_create", "launch", "stop_destroy", None])
def test_attention_ceiling_events_drain_after_partial_setup_and_failure(module, failure):
    timer = importlib.import_module(module)._time_ms
    created, destroyed = [], []

    def create():
        if created and failure == "second_create":
            raise RuntimeError(failure)
        event = len(created) + 1
        created.append(event)
        return event

    def destroy(event):
        destroyed.append(event)
        if event == 2 and failure == "stop_destroy":
            raise RuntimeError(failure)

    def launch():
        if failure == "launch":
            raise RuntimeError(failure)

    runtime = SimpleNamespace(
        event_create=create, event_destroy=destroy,
        event_record=lambda event: None, event_synchronize=lambda event: None,
        event_elapsed_time_ms=lambda start, stop: 1.25,
    )
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            timer(runtime, launch)
    else:
        assert timer(runtime, launch) == 1.25
    assert destroyed == list(reversed(created))
