"""GR recovery drivers attribute owner reports, never stale process values."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_disabled_family_sweep as sweep
from scripts import qwen4exp_disabled_selector_ranking as ranking

UP = 'HIPENGINE_QWEN4_EXP_GR_IU8'
DOWN = 'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN'


@pytest.mark.parametrize('requested,axis', [(UP, 'up'), (DOWN, 'down')])
@pytest.mark.parametrize('reported', [None, False, True])
def test_sweep_checks_gr_owner_not_ambient(requested, axis, reported):
    payload = {'bound_route_env': {requested: '0'}, 'route_env': {requested: '1'}}
    if reported is not None:
        payload['gr_iu8_policy'] = {'up': False, 'down': False, axis: reported}
    summary = sweep._summarize(payload, [])
    assert summary['gr_iu8_policy'] == payload.get('gr_iu8_policy')
    verified, _ = sweep._attribution(summary, {'family': 'GR_IU8', 'env': {}}, {requested: '1'})
    assert verified is reported


@pytest.mark.parametrize('policy,missing', [(None, [UP, DOWN]), ({'up': True, 'down': True}, [])])
def test_ranking_reads_gr_owner_and_excludes_process_keys(monkeypatch, tmp_path, policy, missing):
    def run(command, **kwargs):
        from contextlib import ExitStack
        from scripts.qwen4exp_profile_gap import bind_gr_iu8_override
        from hipengine.runtime.qwen4_exp_runner import GRIU8Policy
        output = Path(command[command.index('--output') + 1])
        overrides = dict(command[i + 1].split('=', 1) for i, token in enumerate(command) if token == '--override')
        owner = SimpleNamespace(gr_iu8_policy=GRIU8Policy(False, False))
        with ExitStack() as scopes:
            resolved = bind_gr_iu8_override(scopes, owner, overrides)['gr_iu8_policy']
            assert resolved == {'up': True, 'down': True}
        assert owner.gr_iu8_policy == GRIU8Policy(False, False)
        output.write_text(json.dumps({'wall_summary': {'tok_s': 1},
            'route_env': {UP: '0', DOWN: '0'}, 'bound_route_env': {UP: '0', DOWN: '0'},
            'gr_iu8_policy': policy}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ranking.subprocess, 'run', run)
    row = ranking._run_arm('gr_both', {UP: '1', DOWN: '1'}, {},
        model_root=tmp_path, case_id='test', fixture=tmp_path/'fixture', chunk_size=1024,
        repetitions=1, compiler_version_file=tmp_path/'compiler', work_root=tmp_path)
    assert row['gr_iu8_policy'] == policy
    assert row['effective_route_missing_keys'] == sorted(missing)
    assert UP not in row['effective_route'] and DOWN not in row['effective_route']
    assert UP not in row['baseline_route'] and DOWN not in row['baseline_route']


def test_sweep_mixed_owner_and_process_requirements():
    key = sweep.CONTROL_KEY
    payload = {'bound_route_env': {key: '0'}, 'route_env': {key: '1'},
               'gr_iu8_policy': {'up': True, 'down': False}}
    assert sweep._attribution(payload, {'family': 'mixed'}, {key: '1', UP: '1'})[0] is True
    assert sweep._attribution(payload, {'family': 'mixed'}, {key: '1', DOWN: '1'})[0] is False
    payload['route_env'][key] = '0'
    assert sweep._attribution(payload, {'family': 'mixed'}, {key: '1', UP: '1'})[0] is False
