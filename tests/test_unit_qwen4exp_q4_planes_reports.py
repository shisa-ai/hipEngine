"""Q4 plane report attribution follows current child owner records."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import qwen4exp_disabled_selector_ranking as ranking
from scripts import qwen4exp_disabled_family_sweep as sweep

KEY = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES'
FIELD = 'q4_iu8_planes'


@pytest.mark.parametrize('value', [2, 3, None])
def test_ranking_reports_child_planes_not_process(monkeypatch, tmp_path, value):
    def run(command, **kwargs):
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'wall_summary': {}, FIELD: value,
            'route_env': {KEY: '3'}, 'bound_route_env': {KEY: '3'}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ranking.subprocess, 'run', run)
    row = ranking._run_arm('planes', {KEY: '2'}, {}, model_root=tmp_path,
        case_id='tiny', fixture=tmp_path / 'fixture', chunk_size=17, repetitions=1,
        compiler_version_file=tmp_path / 'compiler', work_root=tmp_path)
    assert row[FIELD] == value
    assert KEY not in row['effective_route'] and KEY not in row['baseline_route']
    assert row['effective_route_missing_keys'] == ([KEY] if value is None else [])


@pytest.mark.parametrize('value,raw,expected', [(2, '2', True), (3, '2', False),
    (3, 'bad', True), (3, '', True), (None, '2', None), ('2', '2', None),
    (True, '2', None), (4, '2', None)])
def test_family_plane_attribution_uses_effective_owner(value, raw, expected):
    result = {FIELD: value, 'bound_route_env': {KEY: '3'}, 'route_env': {KEY: '3'}}
    assert sweep._env_request_applied(result, {KEY: raw}, control=False)[0] is expected


def test_family_current_missing_is_unknown_and_legacy_still_works():
    result = {'q4_iu8_prefill_enabled': True,
              'bound_route_env': {KEY: '3'}, 'route_env': {KEY: '2'}}
    assert sweep._env_request_applied(result, {KEY: '2'}, control=False)[0] is None
    del result['q4_iu8_prefill_enabled']
    assert sweep._env_request_applied(result, {KEY: '2'}, control=False)[0] is True


def test_family_summary_and_main_preserve_planes():
    assert sweep._summarize({FIELD: 2}, [])[FIELD] == 2
    assert FIELD not in sweep._summarize({}, [])
    assert "'q4_iu8_planes': result['q4_iu8_planes']" in inspect.getsource(sweep.main)
