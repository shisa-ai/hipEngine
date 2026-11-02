"""Family sweep Q4 owner attribution and inert control records."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_disabled_family_sweep as sweep

KEY = sweep.CONTROL_KEY
FIELD = 'q4_iu8_prefill_enabled'


@pytest.mark.parametrize('selected', [False, True, None])
def test_control_prefers_effective_owner_over_ambient(selected):
    fallback = {'ok': True, 'route_env': {KEY: '0' if selected else '1'}, FIELD: selected}
    assert sweep._control_override(fallback) == (
        None if selected is None else {KEY: '1' if selected else '0'})


@pytest.mark.parametrize('selected,expected', [(True, True), (False, False), (None, None)])
def test_q4_binding_uses_owner_even_when_process_route_disagrees(selected, expected):
    result = {'bound_route_env': {KEY: '0'}, 'route_env': {KEY: '0'}, FIELD: selected}
    assert sweep._attribution(result, {'family': 'Q4_IU8_PREFILL'}, {KEY: '1'})[0] is expected


def test_control_detects_wrong_owner_with_unchanged_process_inputs():
    result = {'bound_route_env': {KEY: '0'}, 'route_env': {KEY: '0'}, FIELD: True}
    assert sweep._attribution(result, {'family': sweep.CONTROL_FAMILY}, {KEY: '0'})[0] is False


def test_summary_preserves_owner_record_and_distinguishes_legacy():
    assert sweep._summarize({FIELD: False, 'q4_iu8_layers': ''}, [])['q4_iu8_layers'] == ''
    assert sweep._summarize({FIELD: False}, [])[FIELD] is False
    assert FIELD not in sweep._summarize({}, [])


def test_main_reports_current_owner_control_and_family(monkeypatch, tmp_path):
    def run(command, **kwargs):
        output = Path(command[command.index('--output') + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        env = {}
        for index, token in enumerate(command):
            if token == '--override':
                key, _, value = command[index + 1].partition('=')
                env[key] = value
        # Match current child transport: both process maps stay unchanged.
        routes = {KEY: '0', 'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS': ','.join(map(str, range(35, 48)))}
        output.write_text(json.dumps({
            'wall_summary': {'median': 1.0}, 'wall_seconds': [1.0],
            'route_env': routes, 'bound_route_env': routes,
            FIELD: env.get(KEY, '1') == '1',
            'q4_iu8_risk_mult': 1.0,
            'q4_iu8_layers': env.get('HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS', '35-47'),
        }))
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(sweep.subprocess, 'run', run)
    output = tmp_path / 'report.json'
    assert sweep.main(['--model-root', str(tmp_path), '--fixture', str(tmp_path / 'fixture'),
        '--scratch', str(tmp_path / 'scratch'), '--output', str(output),
        '--only', sweep.CONTROL_FAMILY, '--only', 'Q4_IU8_PREFILL']) == 0
    report = json.loads(output.read_text())
    assert report['control_route_neutral'] is True
    assert len(report['families']) == 2
    for row in report['families']:
        assert row[FIELD] is True
        assert row['q4_iu8_risk_mult'] == 1.0
        assert row['q4_iu8_layers'] is not None
        assert row['attribution_verified'] is True
        assert row['env'][KEY] == '1'


@pytest.mark.parametrize('scope,requested,expected', [('', 'all', True), ('all', '', True),
    ('35-47', ','.join(map(str, range(35, 48))), True), ('999', 'all', False), (None, 'all', None)])
def test_layer_attribution_uses_owner_not_environment(scope, requested, expected):
    key = 'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS'
    result = {FIELD: True, 'q4_iu8_layers': scope,
              'bound_route_env': {key: '999'}, 'route_env': {key: '999'}}
    assert sweep._attribution(result, {'family': 'Q4_IU8_PREFILL'},
                              {KEY: '1', key: requested})[0] is expected


def test_missing_layer_record_does_not_claim_current_owner_binding():
    key = 'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS'
    result = {FIELD: True, 'bound_route_env': {key: '999'}, 'route_env': {key: 'all'}}
    assert sweep._attribution(result, {'family': 'Q4_IU8_PREFILL'},
                              {KEY: '1', key: 'all'})[0] is None
