"""The numerical gate releases its model owner on every post-construction exit."""
from types import SimpleNamespace

import pytest

from scripts import execution_profile_q8_wmma_prefill_layers_gate as gate


@pytest.mark.parametrize('failure', [
    'runner', 'warm', 'teacher', 'candidate', 'quality', 'provenance',
    'verdict', 'success',
])
def test_gate_closes_constructed_owner_and_restores_route(monkeypatch, tmp_path, failure):
    from hipengine.generation import qwen4_exp_profiles as profiles
    from hipengine.kernels import hip_gfx1151 as backend
    from hipengine.loading import gguf
    import hipengine.models as models

    class Injected(RuntimeError):
        pass

    def fail(*args, **kwargs):
        raise Injected(failure)

    closes = []
    owner = SimpleNamespace(runner=None if failure == 'runner' else SimpleNamespace(),
                            close=lambda: closes.append('closed'))
    profile = SimpleNamespace(construct_generator=lambda factory: owner)
    monkeypatch.setattr(gate, 'resolve_runtime_profile', lambda **kw: profile)
    monkeypatch.setattr(backend, 'register_gfx1151_kernels', lambda **kw: None)
    monkeypatch.setattr(profiles, 'register_qwen4_exp_gfx1151_profiles', lambda: None)
    # Owner diagnostics are independent of this explicit gate arm.
    monkeypatch.setattr(profiles, 'profile_ignored_overrides',
                        lambda owner: {'unrelated-owner': ('0', '1')})
    monkeypatch.setattr(gguf, 'discover_gguf_files', lambda path: [path / 'fake.gguf'])
    monkeypatch.setattr(gguf, 'load_gguf_index', lambda path: SimpleNamespace(architecture='fake'))
    monkeypatch.setattr(models, 'resolve_model', lambda architecture: object())
    monkeypatch.setattr(gate, '_load_fixture', lambda path: ({
        'cases': [{'id': 'test', 'category': 'code', 'prompt_tokens': 1,
                   'prompt_token_ids': [1]}], 'decode_transitions': 2,
    }, 'fixture'))
    monkeypatch.setattr(gate, '_warm', fail if failure == 'warm' else lambda *a: None)
    trajectory = lambda *a: [{'token_id': 1}]
    monkeypatch.setattr(gate, '_strict_trajectory', fail if failure == 'teacher' else trajectory)
    monkeypatch.setattr(gate, '_forced_trajectory', fail if failure == 'candidate' else trajectory)
    monkeypatch.setattr(gate, 'PromptCalibrationCapture', lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(gate, 'build_candidate_quality',
                        fail if failure == 'quality' else lambda *a, **kw: {
                            'quality': {}, 'repeat_determinism': {'passed': True},
                            'strict_logits_sha256': 'teacher',
                            'candidate_logits_sha256': 'candidate',
                        })
    monkeypatch.setattr(gate, 'build_variant_manifest', lambda **kw: {})
    monkeypatch.setattr(gate, 'collect_artifact_provenance',
                        fail if failure == 'provenance' else lambda **kw: {})
    monkeypatch.setattr(gate, '_classified_worktree_clean', lambda value: True)
    monkeypatch.setattr(gate, 'qualification_coverage', lambda **kw: {})
    monkeypatch.setattr(gate, 'qualification_verdict',
                        fail if failure == 'verdict' else lambda *a, **kw: {})
    profile.manifest_sha256 = 'profile'
    profile.strict_manifest_sha256 = 'strict'
    route = gate.SELECTORS['staged_f32']
    before = {}
    for key in route.env_keys:
        monkeypatch.setenv(key, 'original')
        before[key] = 'original'
    args = SimpleNamespace(require_cached_build=False, repeat_runs=3,
                           fixture='unused.json', case_id=None, limit=None,
                           decode_steps=0, prefill_chunk_size=73, layers='0',
                           route='staged_f32', model_root=tmp_path,
                           execution_profile='production')
    if failure == 'success':
        result = gate.run_gate(args)
        assert result['route']['name'] == 'staged_f32'
        assert result['route']['selector_restored_after_capture'] is True
        assert result['route']['rowtile_disabled_in_both_arms'] is True
    else:
        expected = gate.CalibrationError if failure == 'runner' else Injected
        with pytest.raises(expected):
            gate.run_gate(args)
    assert closes == ['closed'], 'constructed model owner leaked on gate failure'
    import os
    assert {key: os.environ.get(key) for key in before} == before
    assert not hasattr(owner, 'q8_wmma_layer_scope')
    if owner.runner is not None:
        assert not hasattr(owner.runner, 'q8_wmma_layer_scope')
