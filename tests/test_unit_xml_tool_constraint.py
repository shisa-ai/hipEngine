"""Regression coverage for Qwen's prompted XML tool protocol."""
import pytest

from hipengine.generation.constraints import ToolCallConstraintSpec, ToolCallConstraintState

CALL = '<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>'


@pytest.mark.parametrize('prefix', ['', 'I will inspect the repository.\n'])
@pytest.mark.parametrize('width', [1, 3, 17, 4096])
def test_xml_tool_constraint_accepts_prompted_format(prefix, width):
    state = ToolCallConstraintState(ToolCallConstraintSpec(
        tool_names=('bash',), envelope='xml'))
    text = prefix + CALL
    for offset in range(0, len(text), width):
        chunk = text[offset:offset + width]
        assert state.accepts_text(chunk), (offset, chunk)
        state.observe_text(chunk)
        assert not state.invalid
    assert state.complete and state.allows_eos


def test_xml_tool_constraint_rejects_unknown_name_and_early_eos():
    state = ToolCallConstraintState(ToolCallConstraintSpec(
        tool_names=('bash',), envelope='xml', mode='required'))
    assert not state.accepts_text('prose')
    state.observe_text('<tool_call>\n<function=')
    assert not state.accepts_text('unknown>')
    assert not state.allows_eos
    assert state.accepts_text('bash>\n')


def test_xml_tool_constraint_does_not_invent_argument_values():
    state = ToolCallConstraintState(ToolCallConstraintSpec(
        tool_names=('bash',), envelope='xml'))
    state.observe_text(CALL.split('pwd')[0] + 'unfinished')
    assert not state.invalid
    assert not state.allows_eos
    assert state.forced_close_suffix == ''


def test_xml_tool_constraint_safe_close_and_parallel_policy():
    spec = ToolCallConstraintSpec(tool_names=('bash',), envelope='xml')
    state = ToolCallConstraintState(spec).observe_text(CALL.removesuffix('</function>\n</tool_call>'))
    assert state.forced_close_suffix == '</function>\n</tool_call>'
    state.observe_text(state.forced_close_suffix)
    assert not state.accepts_text('\n' + CALL)
    parallel = ToolCallConstraintState(ToolCallConstraintSpec(
        tool_names=('bash',), envelope='xml', parallel_tool_calls=True))
    parallel.observe_text(CALL)
    assert parallel.accepts_text('\n' + CALL)
    parallel.observe_text('\n' + CALL)
    assert parallel.complete


def test_xml_tool_constraint_queues_only_a_complete_parameter_boundary():
    import numpy as np

    from hipengine import SamplingParams
    from hipengine.generation.sampling import RowSamplingState, select_token

    spec = ToolCallConstraintSpec(tool_names=('bash',), envelope='xml', mode='required')
    state = RowSamplingState(tool_call_constraint=spec)
    prefix = CALL.removesuffix('</function>\n</tool_call>')
    state.observe_selected_token_text(prefix, remaining_tokens=2, encode_text=lambda _text: (1, 2))
    assert state.forced_tokens == (1, 2)
    assert state.forced_token_reason == 'tool_call_close_forcing'
    decoder = {1: '</function>\n', 2: '</tool_call>'}.__getitem__
    params = SamplingParams(temperature=0, tool_call_constraint=spec)
    for token_id, remaining in ((1, 1), (2, 0)):
        sample = select_token(np.zeros(3, dtype=np.float32), params, state, token_text_for_id=decoder)
        assert sample.token_id == token_id and sample.forced
        state.observe_selected_token_text(decoder(token_id), remaining_tokens=remaining,
                                          encode_text=lambda _text: ())
    assert state.tool_call_constraint_state.complete

    for remaining in (1, 3):
        other = RowSamplingState(tool_call_constraint=spec)
        other.observe_selected_token_text(prefix, remaining_tokens=remaining,
                                         encode_text=lambda _text: (1, 2))
        assert not other.forced_tokens
    unfinished = RowSamplingState(tool_call_constraint=spec)
    unfinished.observe_selected_token_text(CALL.split('pwd')[0] + 'unfinished',
                                           remaining_tokens=2, encode_text=lambda _text: (1, 2))
    assert not unfinished.forced_tokens
    assert not unfinished.tool_call_constraint_state.allows_eos


def test_xml_tool_constraint_preserves_parameter_text():
    text = CALL.replace('pwd', 'echo "a < b"\ncat file.py')
    state = ToolCallConstraintState(ToolCallConstraintSpec(tool_names=('bash',), envelope='xml'))
    for char in text:
        assert state.accepts_text(char)
        state.observe_text(char)
    assert state.complete
