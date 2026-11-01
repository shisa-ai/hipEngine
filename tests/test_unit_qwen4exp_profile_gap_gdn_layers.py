"""Profile-gap and sweep bind and verify GDN layer scope on the owner."""
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
from scripts import qwen4exp_profile_gap as gap
from scripts import qwen4exp_disabled_family_sweep as sweep
from hipengine.runtime.qwen4_exp_runner import resolve_gdn_colwarps_layers

ENV = 'HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS'


def test_parser_accepts_scope_and_classifies_it_as_diagnostic():
    args = gap.build_parser().parse_args([
        '--model-root', '/tmp/model', '--mode', 'decode', '--output', '/tmp/out.json',
        '--gdn-colwarps-layers', '0,13-47'])
    assert args.gdn_colwarps_layers == '0,13-47'
    assert not gap._named_profile_intact({}, None, gdn_layers_override='0')
    assert gap._configuration_class({}, None, gdn_layers_override='0') == 'diagnostic_post_binder_override'


@pytest.mark.parametrize('scope', [None, '', '0', '7-9'])
def test_scope_preserves_unspecified_owner_and_restores_on_failure(monkeypatch, scope):
    monkeypatch.setenv(ENV, 'all')
    owner = SimpleNamespace(gdn_colwarps_layers='13-47', runner=SimpleNamespace(gdn_colwarps_layers='13-47'))
    with pytest.raises(RuntimeError):
        with gap._gdn_layers_owner_override(owner, scope):
            assert resolve_gdn_colwarps_layers(owner) == ('13-47' if scope is None else scope)
            assert owner.runner.gdn_colwarps_layers == ('13-47' if scope is None else scope)
            raise RuntimeError('injected')
    assert owner.gdn_colwarps_layers == owner.runner.gdn_colwarps_layers == '13-47'


def test_sweep_declares_scope_on_owner_and_verifies_child_selection():
    entry = next(e for e in sweep.FAMILIES if e['family'] == 'GDN_COLWARPS_PREFILL')
    assert ENV not in entry['env']
    assert entry['owner']['gdn_colwarps_layers'] == ','.join(map(str, range(27, 48)))
    result = {'gdn_colwarps_prefill': True, 'gdn_colwarps_layers': entry['owner']['gdn_colwarps_layers']}
    assert sweep._attribution(result, entry, {})[0] is True
    result['gdn_colwarps_layers'] = '0'
    assert sweep._attribution(result, entry, {})[0] is False
    assert sweep._attribution({}, entry, {})[0] is None


def test_sweep_child_command_transports_gdn_scope(monkeypatch, tmp_path):
    output = tmp_path / 'result.json'
    seen = []
    def run(command, **kwargs):
        seen.extend(command)
        output.write_text(json.dumps({'gdn_colwarps_layers': '0', 'gdn_colwarps_prefill': True}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(sweep.subprocess, 'run', run)
    result = sweep._run('python', Path('/tmp/model'), Path('/tmp/fixture'),
                        'test', 1, {}, output,
                        owner={'gdn_colwarps_layers': '0', 'gdn_colwarps_prefill': 'on'})
    assert seen[seen.index('--gdn-colwarps-layers') + 1] == '0'
    assert result['gdn_colwarps_layers'] == '0'
    gap.build_parser().parse_args(seen[2:])
