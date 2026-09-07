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


def test_candidate_loader_checks_actual_arrays_and_windows(tmp_path):
    import hashlib
    import json
    from hipengine.execution_profiles import build_variant_manifest, manifest_sha256
    from scripts.qwen38_packed_c1_teacher_repeat import load_candidate_capture
    manifest = build_variant_manifest(profile='production', backend='hip_gfx1100',
        model='example', quant='gguf', kv_policy='paged_bf16', graph_policy='eager',
        selections=[dict(layer='linear', scope='all', selected_variant='strict',
                         strict_fallback_variant='strict')])
    teacher_manifest = dict(manifest, execution_profile='strict')
    data = runs()[0]
    record = data['records'][0]
    logits, prefill = record.pop('logits'), record.pop('prefill_logits')
    record['logits_file'] = 'one.npz'
    record['windows'][0].update(resident_slot=0, head_path='row_linear_f32_logits')
    data.update(candidate_manifest=manifest, candidate_manifest_sha256=manifest_sha256(manifest),
                teacher_runtime_manifest_sha256=manifest_sha256(teacher_manifest))
    for key, array in [('logits', logits), ('prefill_logits', prefill)]:
        record[key + '_sha256'] = hashlib.sha256(array.tobytes()).hexdigest()
    np.savez(tmp_path / 'one.npz', logits=logits, prefill_logits=prefill)
    fixture = dict(model_sha256='model', runtime_manifest=teacher_manifest,
                   runtime_manifest_sha256=manifest_sha256(teacher_manifest),
                   kv_storage_dtype='DType.BF16', records=[dict(prompt_id='one',
                   category='code', suite='canonical', prompt_ids=(0, 0), inputs=(1, 2),
                   logits=logits, prefill_logits=prefill)])
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps(data))
    result = load_candidate_capture(tmp_path, fixture, teacher_manifest_sha256='teacher')
    assert np.array_equal(result['records'][0]['logits'], logits)
    for fault in ('position', 'slot', 'capacity', 'hash', 'runtime', 'teacher', 'kv', 'file', 'shape'):
        broken = deepcopy(data)
        if fault == 'position':
            broken['records'][0]['windows'][0]['position'] += 1
        elif fault == 'slot':
            broken['records'][0]['windows'][0]['resident_slot'] = 1
        elif fault == 'capacity':
            broken['capacity'] = True
        elif fault == 'hash':
            broken['records'][0]['logits_sha256'] = 'wrong'
        elif fault == 'runtime':
            broken['candidate_manifest_sha256'] = 'wrong'
        elif fault == 'teacher':
            broken['teacher_runtime_manifest_sha256'] = 'wrong'
        elif fault == 'kv':
            broken['kv_storage_dtype'] = 'DType.FP32'
        elif fault == 'file':
            broken['records'][0]['logits_file'] = '../one.npz'
        else:
            wrong = logits.reshape(1, -1)
            np.savez(tmp_path / 'one.npz', logits=wrong, prefill_logits=prefill)
            broken['records'][0]['logits_sha256'] = hashlib.sha256(wrong.tobytes()).hexdigest()
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            load_candidate_capture(tmp_path, fixture, teacher_manifest_sha256='teacher')


def test_audit_cli_requires_three_captures():
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts/qwen38_packed_c1_teacher_repeat.py'
    help_result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert '--teacher' in help_result.stdout and '--capture' in help_result.stdout
    result = subprocess.run([sys.executable, str(script), '--teacher', 'missing',
        '--capture', 'one', '--capture', 'two', '--output', 'unused.json'],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert 'at least three captures' in result.stderr


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
