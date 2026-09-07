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


@pytest.mark.parametrize('mismatches,calibrated', [(1, True), (3, True), (4, False)])
def test_calibrated_scope_floor_preserves_stricter_screen(mismatches, calibrated):
    data = rows()
    for row in data:
        row['reference'] = np.tile(np.array([[0.001, 0.]], np.float32), (100, 1))
        row['candidate'] = row['reference'].copy()
    data[0]['candidate'][:mismatches] = [0., 0.001]
    result = summarize_teacher_scopes(data)
    scope = result['scopes']['suite:canonical/category:code']
    assert scope['calibrated_top1_min'] == 0.97
    assert scope['calibrated_numerical_envelope_passed'] is calibrated
    assert result['calibrated_numerical_envelope_passed'] is calibrated
    assert scope['numerical_envelope_passed'] is (mismatches == 1)
    assert result['scopes']['all']['calibrated_top1_min'] == 0.99
    assert not result['full_profile_qualification']


def test_calibrated_scope_floor_does_not_lower_global_floor():
    data = rows()
    for row in data:
        row['reference'] = np.tile(np.array([[0.001, 0.]], np.float32), (100, 1))
        row['candidate'] = row['reference'].copy()
        row['candidate'][:2] = [0., 0.001]
    result = summarize_teacher_scopes(data)
    assert not result['calibrated_numerical_envelope_passed']
    assert not result['scopes']['all']['calibrated_numerical_envelope_passed']
    assert result['scopes']['suite:canonical/category:code']['calibrated_numerical_envelope_passed']


@pytest.mark.parametrize('field,value', [('mean_kl', 0.0011), ('p95_kl', 0.0051),
                                       ('p99_kl', 0.0201), ('max_kl', 0.0501),
                                       ('review_rows', [0])])
def test_calibrated_verdict_preserves_all_kl_and_review_gates(monkeypatch, field, value):
    from scripts import qwen38_packed_c1_teacher_scopes as module
    original = module.compare_logits

    def compare(reference, candidate):
        result = original(reference, candidate)
        result[field] = value
        return result

    monkeypatch.setattr(module, 'compare_logits', compare)
    result = summarize_teacher_scopes(rows())
    assert not result['calibrated_numerical_envelope_passed']
    assert all(not scope['calibrated_numerical_envelope_passed']
               for scope in result['scopes'].values())


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
        assert not result['calibrated_numerical_envelope_passed']
        assert not result['scopes']['suite:canonical/category:code']['numerical_envelope_passed']
        return
    with pytest.raises(ValueError):
        summarize_teacher_scopes(data)
