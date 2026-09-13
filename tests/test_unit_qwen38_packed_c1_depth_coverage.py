"""Coverage must distinguish clipping from a verified candidate rejection."""
import pytest


@pytest.mark.parametrize('accepted,remaining,expected', [
    (0, 8, 'rejection'), (1, 8, 'rejection'), (2, 8, 'full_accept'),
    (0, 1, 'horizon_clip'), (1, 2, 'horizon_clip'),
])
def test_depth_outcome(accepted, remaining, expected):
    from scripts.qwen38_packed_c1_depth_coverage import summarize_depth
    row = fixture(accepted, remaining)
    result = summarize_depth({'complete': True, 'records': [row]}, budget=2)
    assert result['outcomes'] == {expected: 1}
    assert result['accepted_counts'] == {str(accepted): 1}


def test_requested_depth_must_actually_execute():
    from scripts.qwen38_packed_c1_depth_coverage import summarize_depth
    with pytest.raises(ValueError, match='requested depth'):
        summarize_depth(dict(complete=True, records=[fixture()]), budget=7)


def fixture(accepted=1, remaining=8):
    return dict(prompt_id='p', logical_rows=3, tokens=[1, 2, 3], remaining_decode=remaining,
        pre_accept_state_isolation=dict(passed=True, checked_buffers=131),
        selected_commit=dict(passed=True, checked_buffers=97, accepted=accepted),
        aux_commit=dict(passed=True, checked_buffers=3), kv_commit=dict(passed=True, checked_buffers=32))


@pytest.mark.parametrize('fault', ['empty', 'incomplete', 'error', 'missing_check', 'zero_buffers', 'bad_count', 'wrong_depth'])
def test_depth_coverage_fails_closed(fault):
    from scripts.qwen38_packed_c1_depth_coverage import summarize_depth
    row = fixture(); capture = dict(complete=True, records=[row])
    if fault == 'empty': capture['records'] = []
    if fault == 'incomplete': capture['complete'] = False
    if fault == 'error': row['commit_error'] = 'failure'
    if fault == 'missing_check': del row['aux_commit']
    if fault == 'zero_buffers': row['kv_commit']['checked_buffers'] = 0
    if fault == 'bad_count': row['selected_commit']['accepted'] = 3
    if fault == 'wrong_depth': row['logical_rows'] = 5
    with pytest.raises(ValueError): summarize_depth(capture, budget=2)
