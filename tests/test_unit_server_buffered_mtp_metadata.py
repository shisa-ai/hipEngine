"""Buffered SSE summaries are per choice, including scheduler adapters."""
import json

import pytest

from hipengine.generation import DecodeState, GenerationOutput, GenerationTelemetry
from hipengine.server.api import (
    _chat_stream_scheduler_text_chunks,
    _chat_stream_scheduler_tool_call_chunks,
    _completion_stream,
)


def choices(events):
    return [choice for event in events if event.startswith('data: {')
            for choice in json.loads(event[6:]).get('choices', [])]


@pytest.mark.parametrize('include', [False, True])
def test_buffered_completion_summary_counts_each_row_once(include):
    details = [GenerationOutput(text='A', generated_token_ids=tuple(range(accepted + cycles + 1)), telemetry=GenerationTelemetry(
        decode_state=DecodeState(generated_tokens=accepted + cycles + 1, execution_path='qwen4exp_mtp_batched_state_rows_verify'),
        diagnostics={'specdec2_mtp2': dict(
            requested_budget=3, candidate_budget=3, cycles=cycles,
            generated_draft_tokens=proposed, accepted_draft_tokens=accepted,
            selected_depth_histogram={'3': cycles}, mtp_output_tokens=accepted + cycles,
            ar_output_tokens_in_cycles=0)},
        timing=dict(mtp_generated_draft_tokens=999, mtp_accepted_draft_tokens=888,
                    mtp_cycles_count=777)))
        for proposed, accepted, cycles in [(6, 2, 2), (9, 3, 3)]]
    events = list(_completion_stream('id', 0, 'model', ['A', 'A'],
        [dict(index=i, finish_reason='length') for i in range(2)], details=details,
        generation_shape=dict(route='speculative', route_decision=dict(
            requested_route='speculative', reason='explicit_speculative_request')),
        thinking_policy='hard', include_hipengine=include))
    terminal = [c for c in choices(events) if c.get('finish_reason')]
    assert len(terminal) == 2
    if not include:
        assert all('hipengine' not in c for c in choices(events))
        return
    for index, (proposed, accepted, cycles) in enumerate([(6, 2, 2), (9, 3, 3)]):
        extra = terminal[index]['hipengine']
        summary = extra['speculative_mtp']
        assert summary['draft_tokens'] == proposed
        assert summary['accepted_draft_tokens'] == accepted
        assert summary['rejected_draft_tokens'] == proposed - accepted
        assert summary['draft_cycles'] == cycles
        assert summary['mtp_output_tokens'] == accepted + cycles
        assert summary['ar_output_tokens'] == 1
        assert summary['output_accounting']['reconciled'] is True
        assert summary['selection_reason'] == 'explicit_speculative_request'
        assert summary['thinking_policy'] == 'hard'
        assert extra['generation_shape']['route'] == 'speculative'


@pytest.mark.parametrize('include', [False, True])
@pytest.mark.parametrize('adapter', ['text', 'tool'])
def test_scheduler_adapters_forward_terminal_summary_only_when_requested(include, adapter):
    metadata = dict(generation_shape=dict(route='speculative_mtp'),
                    speculative_mtp=dict(used=True, draft_tokens=3))
    kwargs = dict(final_hipengine=metadata, include_hipengine=include)
    if adapter == 'text':
        events = _chat_stream_scheduler_text_chunks('id', 0, 'model', [], 'length', **kwargs)
    else:
        events = _chat_stream_scheduler_tool_call_chunks('id', 0, 'model', [], **kwargs)
    terminal, = choices(events)
    if include:
        assert terminal['hipengine']['speculative_mtp'] == metadata['speculative_mtp']
        assert terminal['hipengine']['generation_shape'] == metadata['generation_shape']
    else:
        assert 'hipengine' not in terminal
