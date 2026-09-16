"""Resident coverage uses real bulk capture and c1 head at every forced position."""
import ctypes
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from scripts import tp2_resident_control as control


def test_profile_manifest_is_normalized_for_json(monkeypatch):
    import hipengine.generation as generation
    import hipengine.execution_profiles as profiles
    manifest = profiles.build_variant_manifest(profile='production', backend='hip_gfx1100',
        model='qwen3_5_gguf', quant='gguf_q4_k_m', kv_policy='paged_bf16', graph_policy='test',
        selections=[profiles.VariantSelection(layer='test', scope='c1', selected_variant='v', strict_fallback_variant='v')])
    frozen = dict(manifest)
    frozen['selections'] = tuple(MappingProxyType(x) for x in manifest['selections'])
    bound = []
    resolved = SimpleNamespace(manifest=MappingProxyType(frozen),
        manifest_sha256=profiles.manifest_sha256(manifest), strict_manifest_sha256='strict',
        fell_back_to_strict=False, construct_generator=lambda factory: bound.append(factory()))
    monkeypatch.setattr(generation, 'register_builtin_generators', lambda: None)
    monkeypatch.setattr(profiles, 'resolve_runtime_profile', lambda **kw: resolved)
    payload = control.bind_resident_profile('production')
    json.dumps(payload, allow_nan=False)
    assert profiles.manifest_sha256(payload['manifest']) == payload['manifest_sha256']
    assert len(bound) == 1


def capture_fixture(tmp_path):
    import hashlib
    from hipengine.execution_profiles import build_variant_manifest, VariantSelection, manifest_sha256
    from scripts.tp2_teacher_coverage_broad import _row_hash
    array = np.array([[1., 0., -1.], [2., 0., -1.]], dtype=np.float32)
    array_path = tmp_path / 'logits.npy'
    np.save(array_path, array)
    events = []
    for sweep in range(3):
        for position, token in enumerate([0, 1]):
            events.append({'sweep': sweep, 'prompt_id': 'p', 'phase': 'head-complete',
                           'position': position, 'input_token': token})
        events.append({'sweep': sweep, 'prompt_id': 'p', 'phase': 'resident-state',
                       'device_token_rows': [0, 1], 'position_context': [2, 3]})
    path = tmp_path / 'controls.jsonl'
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    profile = build_variant_manifest(profile='production', backend='hip_gfx1100', model='qwen3_5_gguf',
        quant='gguf_q4_k_m', kv_policy='bf16', graph_policy='test',
        selections=[VariantSelection(layer='test', scope='c1', selected_variant='v', strict_fallback_variant='v')])
    scope = {'capacity': 8}
    data = {'arm': 'tp1-d0', 'status': 'complete', 'natural_teardown': True, 'first_bad_stage': None,
            'stages': [{'ok': True}], 'vocab_size': 3,
            'suite': {'ids': ['p'], 'tokens': [[0, 1]]},
            'determinism': {'sweeps': 3, 'per_row_match': True},
            'profile': {'manifest': profile, 'manifest_sha256': manifest_sha256(profile)},
            'scope_manifest': {'manifest': scope, 'sha256': hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(',', ':')).encode()).hexdigest()},
            'control_log': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            'arrays': [{'path': str(array_path), 'logit_sha256': _row_hash(array)}]}
    out = tmp_path / 'capture.json'
    out.write_text(json.dumps(data))
    return data, out


@pytest.mark.parametrize('fault', ['incomplete', 'manifest', 'shape', 'nan', 'hash', 'controls'])
def test_capture_reader_rejects_invalid_evidence(tmp_path, fault):
    import hashlib
    from scripts.tp2_teacher_coverage_broad import load_resident_capture
    data, path = capture_fixture(tmp_path)
    if fault == 'incomplete': data['natural_teardown'] = False
    if fault == 'manifest': data['profile']['manifest_sha256'] = 'wrong'
    if fault == 'shape': np.save(data['arrays'][0]['path'], np.ones((1, 3), dtype=np.float32))
    if fault == 'nan': np.save(data['arrays'][0]['path'], np.full((2, 3), np.nan, dtype=np.float32))
    if fault == 'hash': data['arrays'][0]['logit_sha256'] = 'wrong'
    if fault == 'controls':
        from pathlib import Path
        controls = Path(data['control_log']['path'])
        controls.write_text('')
        data['control_log']['sha256'] = hashlib.sha256(b'').hexdigest()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_resident_capture(path)


def test_capture_reader_accepts_complete_exact_controls(tmp_path):
    from scripts.tp2_teacher_coverage_broad import load_resident_capture
    _, path = capture_fixture(tmp_path)
    _, arrays = load_resident_capture(path)
    assert arrays[0].shape == (2, 3)


class FakeResident:
    def __init__(self):
        self.position = 0
        self.runtime = SimpleNamespace(get_device=lambda: 0, device_synchronize=lambda: None)
        self.runner = SimpleNamespace(hidden_size=2, vocab_size=3)
        self.scratch = SimpleNamespace(norm=SimpleNamespace(ptr=1))
        self.calls = []
    def reset(self): self.position = 0; self.calls.append('reset')
    def result(self, token):
        logits = np.array([[float(token), 0., -1.]], dtype=np.float32)
        return SimpleNamespace(token_id=0, logits=logits)
    def prefill(self, tokens, **kw):
        assert kw['use_bulk'] is None
        assert kw['return_logits'] is True
        self.calls.append(('bulk', tuple(tokens)))
        target = kw['capture_target_hidden_rows']
        data = np.repeat(np.asarray(tokens, dtype=np.uint16), 2)
        ctypes.memmove(target.ptr, data.ctypes.data, data.nbytes)
        self.position = len(tokens)
        return self.result(tokens[-1])
    def _run_output_norm_hidden(self, src, dst, **kw):
        return src
    def _sample_from_hidden(self, ptr, **kw):
        value = ctypes.c_uint16.from_address(ptr).value
        self.calls.append(('c1-head', value))
        return self.result(value)


def adapter(session, storage):
    return control.ResidentTP1Control(session,
        SimpleNamespace(ptr=storage.ctypes.data, nbytes=storage.nbytes, owner=storage), capacity=8)


def test_full_trajectory_uses_one_bulk_prefill_then_c1_heads():
    session = FakeResident()
    result = adapter(session, np.zeros(16, dtype=np.uint16)).teacher_forced_logits((1, 2, 3))
    np.testing.assert_array_equal(result, [[1., 0., -1.], [2., 0., -1.], [3., 0., -1.]])
    assert session.calls == ['reset', ('bulk', (1, 2, 3)), ('c1-head', 1), ('c1-head', 2), ('c1-head', 3)]
    assert session.position == 3


def test_nonfinite_position_stops_before_next_head():
    session = FakeResident()
    original = session._sample_from_hidden
    def bad(ptr, **kw):
        result = original(ptr, **kw)
        if len(session.calls) == 4:
            result.logits[0, 0] = np.nan
        return result
    session._sample_from_hidden = bad
    with pytest.raises(ValueError):
        adapter(session, np.zeros(16, dtype=np.uint16)).teacher_forced_logits((1, 2, 3))
    assert ('c1-head', 3) not in session.calls


def test_capture_last_row_must_match_normal_product_result():
    session = FakeResident()
    original = session.prefill
    def wrong_final(*args, **kw):
        result = original(*args, **kw)
        result.logits[0, 0] += 1
        return result
    session.prefill = wrong_final
    with pytest.raises(ValueError, match='final'):
        adapter(session, np.zeros(16, dtype=np.uint16)).teacher_forced_logits((1, 2, 3))
