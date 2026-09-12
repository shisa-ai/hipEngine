"""Reject incomplete or internally inconsistent fixed-teacher fixtures."""
import hashlib
import json

import numpy as np
import pytest

from scripts.qwen38_packed_c1_teacher_fixture import load_teacher_fixture


MODEL = 'a' * 64
PROMPT = dict(id='example', category='code', suite='canonical', prompt_sha256='b' * 64)


def fixture(tmp_path, *, mutate=None, arrays=None):
    logits = np.eye(4, dtype=np.float32)[[2, 3, 0]]
    prefill = np.eye(4, dtype=np.float32)[1]
    if arrays:
        logits, prefill = arrays(logits, prefill)
    record = dict(prompt_id='example', category='code', suite='canonical',
                  prompt_sha256='b' * 64, prompt_ids=[0, 1], inputs=[1, 2, 3],
                  logits_file='example.npz',
                  logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                  prefill_logits_sha256=hashlib.sha256(prefill.tobytes()).hexdigest())
    manifest = dict(kind='packed_c1_strict_teacher', complete=True,
                    full_profile_qualification=False, execution_profile='strict',
                    model_sha256=MODEL, steps=3, records=[record])
    if mutate:
        mutate(manifest, record)
    np.savez(tmp_path / 'example.npz', logits=logits, prefill_logits=prefill)
    (tmp_path / 'teacher.json').write_text(json.dumps(manifest))
    return tmp_path


def load(path):
    return load_teacher_fixture(path, expected_model_sha256=MODEL, expected_prompts=[PROMPT])


def test_valid_fixture_returns_owned_arrays_and_teacher_coordinates(tmp_path):
    result = load(fixture(tmp_path))
    assert result['full_profile_qualification'] is False
    row = result['records'][0]
    assert row['inputs'] == (1, 2, 3)
    assert row['prompt_ids'] == (0, 1)
    assert row['logits'].shape == (3, 4)
    assert row['logits'].dtype == np.float32
    assert row['prefill_logits'].argmax() == 1


@pytest.mark.parametrize('key,value', [
    ('complete', False), ('complete', 1), ('execution_profile', 'production'),
    ('kind', 'conditional'), ('model_sha256', 'c' * 64),
    ('steps', 0), ('steps', True), ('steps', 3.0), ('steps', 4),
    ('records', []),
])
def test_rejects_bad_manifest(tmp_path, key, value):
    with pytest.raises(ValueError):
        load(fixture(tmp_path, mutate=lambda m, r: m.update({key: value})))


@pytest.mark.parametrize('key,value', [
    ('prompt_id', 'unknown'), ('category', 'general_en'), ('suite', 'category_heldout'),
    ('prompt_sha256', 'c' * 64), ('prompt_ids', []), ('prompt_ids', [-1]),
    ('prompt_ids', [4]), ('prompt_ids', [True]), ('prompt_ids', [1.0]),
    ('inputs', [1, 3, 2]), ('inputs', [2, 2, 3]), ('inputs', [1, 2]),
    ('inputs', ['1', 2, 3]), ('inputs', [1, 2, 4]),
    ('logits_sha256', 'c' * 64), ('prefill_logits_sha256', 'c' * 64),
    ('logits_file', '../example.npz'), ('logits_file', '/tmp/example.npz'),
    ('logits_file', 'missing.npz'),
])
def test_rejects_bad_record(tmp_path, key, value):
    with pytest.raises(ValueError):
        load(fixture(tmp_path, mutate=lambda m, r: r.update({key: value})))


@pytest.mark.parametrize('change', [
    lambda a, p: (a.astype(np.float64), p),
    lambda a, p: (a, p.astype(np.float64)),
    lambda a, p: (a[:, :0], p[:0]),
    lambda a, p: (a.reshape(-1), p),
    lambda a, p: (a, p[None, :]),
    lambda a, p: (a[:, :3], p),
    lambda a, p: (a * np.nan, p),
    lambda a, p: (a, p * np.inf),
    lambda a, p: (a[::-1].copy(), p),
])
def test_rejects_bad_arrays_even_with_recomputed_hashes(tmp_path, change):
    with np.errstate(invalid='ignore'), pytest.raises(ValueError):
        load(fixture(tmp_path, arrays=change))


def test_duplicate_record_is_not_suite_coverage(tmp_path):
    with pytest.raises(ValueError):
        load(fixture(tmp_path, mutate=lambda m, r: m['records'].append(dict(r))))


def test_qualification_flag_is_never_trusted(tmp_path):
    result = load(fixture(tmp_path, mutate=lambda m, r: m.update(full_profile_qualification=True)))
    assert result['full_profile_qualification'] is False


@pytest.mark.parametrize('digest', [None, '', 'abc', 'g' * 64])
def test_requires_independent_valid_model_digest(tmp_path, digest):
    with pytest.raises(ValueError):
        load_teacher_fixture(fixture(tmp_path), expected_model_sha256=digest,
                             expected_prompts=[PROMPT])


def test_missing_prefill_array_rejected(tmp_path):
    path = fixture(tmp_path)
    np.savez(path / 'example.npz', logits=np.eye(4, dtype=np.float32))
    with pytest.raises(ValueError, match='arrays'):
        load(path)


def test_cross_prompt_vocabulary_change_rejected(tmp_path):
    path = fixture(tmp_path)
    manifest = json.loads((path / 'teacher.json').read_text())
    row = dict(manifest['records'][0], prompt_id='other', logits_file='other.npz')
    logits = np.eye(5, dtype=np.float32)[[2, 3, 0]]
    prefill = np.eye(5, dtype=np.float32)[1]
    row['logits_sha256'] = hashlib.sha256(logits.tobytes()).hexdigest()
    row['prefill_logits_sha256'] = hashlib.sha256(prefill.tobytes()).hexdigest()
    manifest['records'].append(row)
    np.savez(path / 'other.npz', logits=logits, prefill_logits=prefill)
    (path / 'teacher.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='vocabulary changes'):
        load_teacher_fixture(path, expected_model_sha256=MODEL,
                             expected_prompts=[PROMPT, dict(PROMPT, id='other')])


@pytest.mark.parametrize('fault', [None, 'missing', 'hash', 'profile', 'kv'])
def test_required_runtime_provenance(tmp_path, fault):
    from hipengine.execution_profiles import build_variant_manifest, manifest_sha256
    path = fixture(tmp_path)
    data = json.loads((path / 'teacher.json').read_text())
    manifest = build_variant_manifest(profile='production' if fault == 'profile' else 'strict',
        backend='hip_gfx1100', model='example', quant='gguf', kv_policy='bf16',
        graph_policy='eager', selections=[dict(layer='linear', scope='all',
        selected_variant='strict', strict_fallback_variant='strict')])
    if fault != 'missing':
        data.update(runtime_manifest=manifest, runtime_manifest_sha256=(
            '0' * 64 if fault == 'hash' else manifest_sha256(manifest)),
            kv_storage_dtype='DType.FP32' if fault == 'kv' else 'DType.BF16')
    (path / 'teacher.json').write_text(json.dumps(data))
    def run():
        return load_teacher_fixture(path, expected_model_sha256=MODEL,
            expected_prompts=[PROMPT], require_runtime_provenance=True)
    if fault is None:
        assert run()['runtime_manifest_sha256'] == manifest_sha256(manifest)
    else:
        with pytest.raises(ValueError, match='provenance'):
            run()


def test_default_suite_requires_all_canonical_and_heldout_prompts(tmp_path):
    with pytest.raises(ValueError):
        load_teacher_fixture(fixture(tmp_path), expected_model_sha256=MODEL)
