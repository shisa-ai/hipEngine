"""Numerical aggregation must use actual rows and cover every declared scope."""
import numpy as np
import pytest

from scripts.qwen38_packed_c1_teacher_scopes import summarize_teacher_scopes


def rows():
    return [dict(prompt_id=str(i), category=category, suite=suite,
                 reference=np.array([[4., 0.], [0., 4.]], np.float32),
                 candidate=np.array([[4., 0.], [0., 4.]], np.float32))
            for i, (suite, category) in enumerate(
                (s, c) for s in ('canonical', 'category_heldout')
                for c in ('code', 'general_en', 'general_ja', 'mixed_ja_en'))]


def test_exact_scopes_and_counts():
    result = summarize_teacher_scopes(rows())
    assert result['numerical_envelope_passed']
    assert not result['full_profile_qualification']
    assert result['scopes']['all']['rows'] == 16
    assert len(result['scopes']) == 15
    assert result['scopes']['suite:canonical/category:code']['rows'] == 2


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'nan', 'shape', 'regression'])
def test_invalid_or_failed_scope_cannot_pass(fault):
    data = rows()
    if fault == 'missing':
        data.pop()
    elif fault == 'duplicate':
        data.append(data[0])
    elif fault == 'nan':
        data[0]['candidate'][0, 0] = np.nan
    elif fault == 'shape':
        data[0]['candidate'] = data[0]['candidate'][:1]
    else:
        data[0]['candidate'] = data[0]['candidate'][:, ::-1]
        result = summarize_teacher_scopes(data)
        assert not result['numerical_envelope_passed']
        assert not result['scopes']['suite:canonical/category:code']['numerical_envelope_passed']
        return
    with pytest.raises(ValueError):
        summarize_teacher_scopes(data)
