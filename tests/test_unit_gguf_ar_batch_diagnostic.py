"""Direct AR diagnostic retains borrowed-session failure cleanup."""
from types import SimpleNamespace

import pytest

from hipengine.generation.gguf_ar_batch_diagnostic import generate_ar_serving_slots


def test_ar_diagnostic_closes_acquired_slots_without_reuse_on_failure():
    slots = [object(), object()]
    closed = []

    def fail(*args):
        raise RuntimeError('decode failed')

    owner = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda prompt: [1, 2]),
        _open_ar_serving_slots=lambda *args: slots,
        _run_ar_serving_slots=fail,
        _close_ar_serving_slots=lambda acquired, **kwargs: closed.append((acquired, kwargs)),
    )
    request = SimpleNamespace(prompts=('first', 'second'), deadline_at=None)
    with pytest.raises(RuntimeError, match='decode failed'):
        generate_ar_serving_slots(owner, object(), request, plan=None)
    assert closed == [(slots, {'reuse': False})]
