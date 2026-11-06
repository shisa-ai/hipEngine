"""Offline Qwen DMS capture uses the model owner, not serving collector ownership."""
from types import SimpleNamespace

import pytest

from hipengine.models.qwen35_dms import capture_qwen35_dms_prefill


def test_offline_capture_uses_bulk_teacher_path_without_closing_sink():
    calls = []
    sink = object()
    session = SimpleNamespace(prefill=lambda tokens, **kwargs: calls.append((tokens, kwargs)) or 'result')
    assert capture_qwen35_dms_prefill(session, (1, 2, 3, 4), sink=sink) == 'result'
    assert calls == [([1, 2, 3, 4], dict(use_bulk=True, bulk_attention_mode='bulk', return_logits=True, dms_capture=sink))]


def test_offline_capture_propagates_failure_without_taking_sink_ownership():
    sink = SimpleNamespace(close=lambda: pytest.fail('borrowed sink must not close'))
    def fail(*args, **kwargs):
        raise RuntimeError('capture failed')
    with pytest.raises(RuntimeError, match='capture failed'):
        capture_qwen35_dms_prefill(SimpleNamespace(prefill=fail), [1, 2, 3, 4], sink=sink)
