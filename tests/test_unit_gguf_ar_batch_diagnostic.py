"""Direct AR diagnostic retains borrowed-session failure cleanup."""
from types import SimpleNamespace

import pytest

from hipengine.generation.gguf_ar_batch_diagnostic import (
    close_ar_serving_slots,
    generate_ar_serving_slots,
)


@pytest.mark.parametrize('reuse', [False, True])
def test_ar_diagnostic_close_destroys_stream_before_release(reuse):
    events = []
    session = SimpleNamespace(
        runtime=SimpleNamespace(stream_destroy=lambda stream: events.append(('stream', stream))),
        close=lambda: events.append(('close',)),
    )
    slot = SimpleNamespace(session=session, session_pool_key='ar', decode_stream=7)
    owner = SimpleNamespace(
        _release_shared_session=lambda key, session: events.append(('release', key)),
    )
    close_ar_serving_slots(owner, [slot], reuse=reuse)
    assert events == [('stream', 7), ('release', 'ar') if reuse else ('close',)]
    assert slot.decode_stream == 0


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
