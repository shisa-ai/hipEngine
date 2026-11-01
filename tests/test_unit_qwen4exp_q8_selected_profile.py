"""Profiles publish the effective selected Q8 gate without process writes."""
import os
from types import SimpleNamespace
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation import qwen4_exp_profiles as profiles
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind, _isolate

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('ambient', ['0', '1'])
@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_selected_down_profile_isolation(monkeypatch, quant, ambient, order):
    monkeypatch.setenv(ENV, ambient)
    peers = []
    for production in order:
        resolved = _resolve(ExecutionProfile.PRODUCTION if production else ExecutionProfile.STRICT, quant=quant)
        owner, runner = _bind(resolved)
        expected = resolved.binder.__name__ == '_production_binder'
        peers.append((owner, runner, expected))
        assert os.environ[ENV] == ambient
        for peer, current, value in peers:
            assert peer.q8_selected_down_enabled is current.q8_selected_down_enabled is value
        assert _inherit_lane_decisions(owner, SimpleNamespace()).q8_selected_down_enabled is expected


def test_selected_down_recovery_restoration(monkeypatch):
    monkeypatch.setattr(profiles, 'PRODUCTION_Q8_QSA_RESTORED_FLAGS', {
        key: value for key, value in profiles.PRODUCTION_Q8_QSA_RESTORED_FLAGS.items()
        if key != 'Q8_0_SELECTED_WMMA_DOWN'})
    ud, _ = _bind(_resolve(ExecutionProfile.PRODUCTION))
    assert ud.q8_selected_down_enabled is False
    peer = SimpleNamespace(runner=SimpleNamespace())
    profiles._publish_selected_q8_down(peer, production=True, quant='gguf_q4_k_m')
    assert peer.q8_selected_down_enabled is peer.runner.q8_selected_down_enabled is True
    monkeypatch.setattr(profiles, 'PRODUCTION_ARITHMETIC_RECOVERY_FLAGS', ())
    ud, _ = _bind(_resolve(ExecutionProfile.PRODUCTION))
    assert ud.q8_selected_down_enabled is True


@pytest.mark.parametrize('effective', [None, False, True])
def test_ranking_reads_selected_owner_not_stale_route(monkeypatch, tmp_path, effective):
    import json
    from scripts import qwen4exp_disabled_selector_ranking as ranking
    def fake_run(command, **kwargs):
        output = ranking.Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'route_env': {ENV: 'stale'},
                                      'q8_selected_down_enabled': effective}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ranking.subprocess, 'run', fake_run)
    row = ranking._run_arm('selected', {ENV: '1'}, {}, model_root=tmp_path,
        case_id='test', fixture=tmp_path / 'fixture', chunk_size=512,
        repetitions=1, compiler_version_file=tmp_path / 'compiler', work_root=tmp_path)
    assert row['q8_selected_down_enabled'] is effective
    assert ENV not in row['effective_route']
    assert row['effective_route_missing_keys'] == ([ENV] if effective is None else [])
