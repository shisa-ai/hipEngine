"""Require actual bytes, coordinates and runtime identity across repeats."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.qwen38_packed_c1_teacher_repeat import assert_teacher_repeats


def runs():
    base = dict(complete=True, model_sha256='model', candidate_manifest_sha256='candidate',
                teacher_runtime_manifest_sha256='strict', teacher_manifest_sha256='teacher',
                capacity=1, budget=3, target_contexts={'active_slots': 1},
                execution_profile='production', kv_storage_dtype='DType.BF16',
                target_use_wmma_prefill=True, records=[dict(prompt_id='one', category='code',
                suite='canonical', windows=[dict(position=2, tokens=[1, 2], rows=2)],
                logits=np.zeros((2, 3), np.float32), prefill_logits=np.zeros(3, np.float32))])
    return [deepcopy(base) for _ in range(3)]


def test_three_bit_identical_runs():
    assert assert_teacher_repeats(runs()) == dict(repeats=3, prompts=1, decode_rows=2,
        prefill_rows=1, bit_identical=True, full_profile_qualification=False)


@pytest.mark.parametrize('key,value', [
    ('capacity', 2), ('budget', 4), ('target_contexts', {'active_slots': 2}),
    ('execution_profile', 'strict'), ('kv_storage_dtype', 'DType.FP32'),
    ('target_use_wmma_prefill', False), ('model_sha256', 'different'),
])
def test_runtime_scope_changes_cannot_count_as_repeat(key, value):
    data = runs()
    data[2][key] = value
    with pytest.raises(ValueError, match='provenance'):
        assert_teacher_repeats(data)


def test_prompt_reordering_is_a_different_schedule():
    data = runs()
    for run in data:
        second = deepcopy(run['records'][0])
        second['prompt_id'] = 'two'
        run['records'].append(second)
    data[2]['records'].reverse()
    with pytest.raises(ValueError, match='schedule'):
        assert_teacher_repeats(data)


@pytest.mark.parametrize('fault', ['two', 'partial', 'manifest', 'teacher', 'window',
                                    'bits', 'nan', 'dtype', 'missing', 'duplicate'])
def test_rejects_invalid_repeat(fault):
    data = runs()
    if fault == 'two':
        data.pop()
    elif fault == 'partial':
        data[1]['complete'] = False
    elif fault == 'manifest':
        data[1]['candidate_manifest_sha256'] = 'other'
    elif fault == 'teacher':
        data[1]['teacher_manifest_sha256'] = 'other'
    elif fault == 'window':
        data[1]['records'][0]['windows'][0]['position'] += 1
    elif fault == 'bits':
        data[1]['records'][0]['logits'][0, 0] = -0.0
    elif fault == 'nan':
        for run in data:
            run['records'][0]['logits'][0, 0] = np.nan
    elif fault == 'dtype':
        data[1]['records'][0]['logits'] = data[1]['records'][0]['logits'].astype(np.float64)
    elif fault == 'missing':
        data[1]['records'] = []
    else:
        for run in data:
            run['records'].append(deepcopy(run['records'][0]))
    with pytest.raises(ValueError):
        assert_teacher_repeats(data)
