"""The native-row diagnostic is independent of the public resident driver."""
from types import SimpleNamespace

import pytest

from hipengine.generation.gguf_native_batch_diagnostic import generate_native_greedy_batch


@pytest.mark.parametrize('capacity', [0, 1, 9])
def test_native_diagnostic_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError, match=r'capacity must be within \[2, 8\]'):
        generate_native_greedy_batch(None, {0: [1], 1: [2]}, None,
                                     capacity=capacity, tokenizer=None)


def test_native_diagnostic_closes_graph_on_replay_failure():
    closed = []

    def fail(tokens):
        raise RuntimeError('replay failure')

    graph = SimpleNamespace(step=fail, close=lambda: closed.append(True))
    session = SimpleNamespace(
        prefill_slot=lambda tokens, **kwargs: SimpleNamespace(token_id=1),
        step_rows_native=lambda tokens, **kwargs: SimpleNamespace(
            token_ids=(2, 2), execution_paths={}),
        capture_native_rows_graph=lambda **kwargs: graph,
    )
    request = SimpleNamespace(max_tokens=4, ignore_eos=True, deadline_at=None)
    with pytest.raises(RuntimeError, match='replay failure'):
        generate_native_greedy_batch(session, {0: [1], 1: [2]}, request,
                                     capacity=2, tokenizer=None)
    assert closed == [True]
