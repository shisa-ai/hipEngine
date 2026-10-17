"""The live control probe must distinguish actual MTP from a successful fallback."""
import copy
import json

import pytest

from scripts.qwen4exp_controlled_mtp_http_gate import EXPECTED, cases, structured, validate_control, replay_cases


def test_replay_includes_failed_comparison_even_when_rows_pass():
    rows = [dict(name=name, kind='schema', request={}, passed=True) for name in ('a', 'b', 'c')]
    previous = dict(rows=rows, comparisons=[dict(left='a', right='b', passed=False)])
    assert [r['name'] for r in replay_cases(previous)] == ['a', 'b']
    assert [r['name'] for r in replay_cases(previous, ['c'])] == ['c']
    with pytest.raises(AssertionError):
        replay_cases(previous, ['missing'])


def test_hard_cap_validator_checks_counter_and_consumed_delimiter():
    from scripts.qwen4exp_controlled_mtp_http_gate import validate_hard_thinking
    valid = dict(ids=[1, 2, 9, 3], finish=dict(phase='answer', reasoning_tokens=3, answer_tokens=1))
    validate_hard_thinking(valid, cap=2, close_ids=[9])
    for bad in (valid | dict(finish={}), valid | dict(ids=[1, 2, 8, 3]),
                valid | dict(finish=dict(phase='answer', reasoning_tokens=4, answer_tokens=1))):
        with pytest.raises(AssertionError):
            validate_hard_thinking(bad, cap=2, close_ids=[9])


def result():
    return dict(mtp=dict(used=True, draft_tokens=3, draft_cycles=1,
                         accepted_draft_tokens=2, rejected_draft_tokens=1),
                execution_path='qwen4exp_mtp_batched_state_rows_verify',
                sampling_algorithm='target_draw_match_deterministic_draft',
                usage=dict(completion_tokens=4), ids=[1, 2, 3, 4], finish_reason='stop',
                text=json.dumps(EXPECTED), calls=[], reasoning='', tokens=[])


def test_case_matrix_covers_structured_controls():
    rows = list(cases('model'))
    assert len(rows) == len({r['name'] for r in rows}) == 80
    for kind in ('schema', 'required_tool', 'auto_tool', 'auto_answer', 'hard_thinking'):
        group = [r for r in rows if r['kind'] == kind]
        assert len(group) == 16
        for row in group:
            body = row['request']
            assert (body.get('speculative') is True) != (body.get('speculative_mtp') is True)
            if body['stream']:
                assert body['stream_options'] == dict(include_usage=True, include_hipengine=True)


def test_actual_route_and_schema_pass():
    validate_control(result(), kind='schema', logs=False)


@pytest.mark.parametrize('patch', [
    {'mtp': dict(used=False)}, {'execution_path': 'autoregressive'},
    {'sampling_algorithm': 'wrong'}, {'usage': dict(completion_tokens=3)},
    {'text': '{}'}, {'finish_reason': 'length'}, {'reasoning': 'unexpected'},
])
def test_rejects_fallback_invalid_schema_or_accounting(patch):
    with pytest.raises(AssertionError):
        validate_control(result() | patch, kind='schema', logs=False)


def test_tool_fragments_reassemble_and_validate():
    body = [dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, id='call_1', function=dict(name='weather', arguments='{"ci'))]))]),
            dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, function=dict(arguments='ty":"Tokyo"}'))]))])]
    fields = structured(body)
    validate_control(result() | fields | dict(text='', finish_reason='tool_calls'), kind='required_tool', logs=False)
    wrong = copy.deepcopy(fields)
    wrong['calls'][0]['arguments'] = '{"city":"Paris"}'
    with pytest.raises(AssertionError):
        validate_control(result() | wrong | dict(finish_reason='tool_calls'), kind='required_tool', logs=False)


def test_thinking_requires_reasoning_and_answer():
    with pytest.raises(AssertionError):
        validate_control(result(), kind='hard_thinking', logs=False)
    validate_control(result() | dict(reasoning='Consider this.'), kind='hard_thinking', logs=False)
