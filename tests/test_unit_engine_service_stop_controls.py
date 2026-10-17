"""Speculative terminal ownership must not bypass runtime stop suppression."""
from functools import partial
from types import SimpleNamespace

import pytest

from hipengine.generation.engine_loop import EngineLoopEvent
from hipengine.generation.engine_service import EngineService, OutputKind
from hipengine.generation.registry import FinishDetails, GenerationRequest, GenerationStreamChunk


@pytest.mark.parametrize('mode', ['ar', 'verify_chain', 'verify_tree'])
@pytest.mark.parametrize('case', ['single', 'multi', 'eos', 'unfinished_prefix'])
def test_stop_filter_sees_finish_before_speculative_terminal_metadata_is_removed(mode, case):
    make_request = partial(GenerationRequest, prompts=('x',), temperature=0.0, top_p=1.0, ignore_eos=False)
    if case == 'single':
        request = make_request(max_tokens=4, stop_token_ids=(2,))
        tokens = (1, 2)
        finish = FinishDetails(reason='stop', stop_sequence=(2,))
        expected = '<1>'
    elif case == 'multi':
        request = make_request(max_tokens=4, stop_token_sequences=((2, 3),))
        tokens = (1, 2, 3)
        finish = FinishDetails(reason='stop', stop_sequence=(2, 3))
        expected = '<1>'
    elif case == 'eos':
        request = make_request(max_tokens=4, eos_token_id=2)
        tokens = (1, 2)
        finish = FinishDetails(reason='eos', eos_token_id=2)
        expected = '<1>'
    else:
        request = make_request(max_tokens=2, stop_token_sequences=((2, 3),))
        tokens = (1, 2)
        finish = FinishDetails(reason='length', length_limit=2)
        expected = '<1><2>'
    published = []
    def publish(event):
        published.append(event)
        return True
    state = SimpleNamespace(
        terminal=False, request=request, token_ids=[], pending_stop_chunks=[],
        execution_mode=mode, service_request_id=7, collector=SimpleNamespace(publish=publish),
    )
    service = object.__new__(EngineService)
    service._canonical_token_events = True
    service._states_by_backend_id = {7: state}
    service._notify_output = lambda: None
    for index, token in enumerate(tokens):
        last = index == len(tokens) - 1
        service._route_events((EngineLoopEvent(
            kind='token', request_id=7, request_ids=(7,), token_id=token,
            stream_chunk=GenerationStreamChunk(
                text=f'<{token}>', finish_details=finish if last else None,
                generated_token_ids=tokens if last else None,
            ),
        ),))
    chunks = [event.stream_chunk for event in published if event.kind is OutputKind.CHUNK]
    assert ''.join(chunk.text for chunk in chunks) == expected
    assert state.pending_stop_chunks == []
    assert state.token_ids == list(tokens)  # Hidden stop tokens still count as generated.
    if mode != 'ar':
        assert all(chunk.finish_details is None for chunk in chunks)
        assert all(chunk.generated_token_ids is None for chunk in chunks)
    else:
        assert chunks[-1].finish_details == finish
