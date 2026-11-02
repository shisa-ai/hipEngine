"""Checked-row diagnostics select DPP on their owner without process writes."""
import importlib
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.parametrize('script', ['engine_ab', 'state_gate', 'natural_owner'])
@pytest.mark.parametrize('prior', [None, False, True])
def test_checked_diagnostic_binds_owner_and_restores_on_setup_failure(monkeypatch, tmp_path, script, prior):
    with patch.dict(os.environ):
        module = importlib.import_module('scripts.qwen4exp_dense_dpp_checked_full_rows_' + script)
    setup = module.gate if script == 'natural_owner' else module
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION', 'caller')
    owner = SimpleNamespace(q8_dpp_reduction_enabled=prior,
        runner=SimpleNamespace(q8_dpp_reduction_enabled=prior, runtime=SimpleNamespace()))
    case = {'id': 'code-p512', 'prompt_tokens': 512, 'prompt_token_ids': [1] * 512}
    profile = SimpleNamespace(construct_generator=lambda factory: owner)
    monkeypatch.setattr(setup, 'register_gfx1151_kernels', lambda **kw: None)
    monkeypatch.setattr(setup, 'register_qwen4_exp_gfx1151_profiles', lambda: None)
    monkeypatch.setattr(setup, 'load_fixture', lambda path: ({'cases': [case]}, 'fixture'))
    monkeypatch.setattr(setup, 'discover_gguf_files', lambda path: [tmp_path / 'model.gguf'])
    monkeypatch.setattr(setup, 'load_gguf_index', lambda path: SimpleNamespace(architecture='qwen4'))
    if script == 'state_gate':
        import hipengine.execution_profiles as profiles
        monkeypatch.setattr(profiles, 'resolve_runtime_profile', lambda **kw: profile)
    else:
        monkeypatch.setattr(setup, 'resolve_runtime_profile', lambda **kw: profile)
    if script == 'engine_ab':
        monkeypatch.setattr(module, 'resolve_model', lambda architecture: object())
    if script == 'natural_owner':
        monkeypatch.setattr(module, 'check_host', lambda: None)
    class Observed(Exception):
        pass
    def stop(**kwargs):
        assert os.environ['HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION'] == 'caller'
        assert owner.q8_dpp_reduction_enabled is True
        assert owner.runner.q8_dpp_reduction_enabled is True
        raise Observed
    monkeypatch.setattr(module, 'resolve', stop)
    argv = ['diagnostic', '--model-root', str(tmp_path), '--compiler-version-file',
            str(tmp_path / 'compiler.txt'), '--output', str(tmp_path / 'out.json'),
            '--case-id', 'code-p512']
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(Observed):
        module.main()
    assert owner.q8_dpp_reduction_enabled is prior
    assert owner.runner.q8_dpp_reduction_enabled is prior
    assert os.environ['HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION'] == 'caller'
