"""Q4 multiplier report attribution follows current child owner records."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import qwen4exp_disabled_selector_ranking as ranking
from scripts import qwen4exp_disabled_family_sweep as sweep

KEY = 'HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT'
FIELD = 'q4_iu8_risk_mult'


@pytest.mark.parametrize('value', [0.5, 1.0, 4.0, None])
def test_ranking_reports_child_multiplier_not_process(monkeypatch, tmp_path, value):
    def run(command, **kwargs):
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'wall_summary': {}, FIELD: value,
            'route_env': {KEY: '8'}, 'bound_route_env': {KEY: '8'}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ranking.subprocess, 'run', run)
    row = ranking._run_arm('risk', {KEY: '1'}, {}, model_root=tmp_path,
        case_id='tiny', fixture=tmp_path / 'fixture', chunk_size=17, repetitions=1,
        compiler_version_file=tmp_path / 'compiler', work_root=tmp_path)
    assert row[FIELD] == value
    assert KEY not in row['effective_route'] and KEY not in row['baseline_route']
    assert row['effective_route_missing_keys'] == ([KEY] if value is None else [])


@pytest.mark.parametrize('value,expected', [(1.0, True), (4.0, False), (None, None),
    ('1', None), (True, None), (float('nan'), None)])
def test_family_multiplier_attribution_uses_effective_owner(value, expected):
    result = {FIELD: value, 'bound_route_env': {KEY: '8'}, 'route_env': {KEY: '8'}}
    assert sweep._env_request_applied(result, {KEY: '1.0'}, control=False)[0] is expected


def test_family_current_missing_is_unknown_and_legacy_still_works():
    result = {'q4_iu8_prefill_enabled': True,
              'bound_route_env': {KEY: '4'}, 'route_env': {KEY: '1'}}
    assert sweep._env_request_applied(result, {KEY: '1'}, control=False)[0] is None
    del result['q4_iu8_prefill_enabled']
    assert sweep._env_request_applied(result, {KEY: '1'}, control=False)[0] is True


def test_family_summary_and_main_preserve_multiplier():
    assert sweep._summarize({FIELD: 1.0}, [])[FIELD] == 1.0
    assert FIELD not in sweep._summarize({}, [])
    assert "'q4_iu8_risk_mult': result['q4_iu8_risk_mult']" in inspect.getsource(sweep.main)
