"""CPU diagnostics exercise scoped columnwarp controls, never real launches."""
from types import SimpleNamespace
import json

import pytest

from scripts import qwen4exp_gdn_tiled16_multi_engagement_probe as engagement
from scripts import qwen4exp_gdn_owner_probe as owner_probe
from scripts import qwen4exp_halo_box_campaign_ab as halo
from hipengine.kernels.registry import KernelKey


@pytest.mark.parametrize('prior', [True, False])
@pytest.mark.parametrize('fail', [True, False])
def test_engagement_main_binds_both_owners_and_restores_on_failure(
        monkeypatch, tmp_path, prior, fail):
    class ProbeFailure(RuntimeError):
        pass

    runner = SimpleNamespace(gdn_colwarps_prefill=prior,
                             gdn_tile16_variant='published')
    owner = SimpleNamespace(runner=runner, gdn_colwarps_prefill=prior,
                            gdn_tile16_variant='published')
    observed = []

    def prefill(ids):
        observed.append((owner.gdn_colwarps_prefill, runner.gdn_colwarps_prefill,
                         owner.gdn_tile16_variant, runner.gdn_tile16_variant))
        if fail:
            raise ProbeFailure('measurement failed')
        return SimpleNamespace(token_id=7)

    runner.prefill = prefill
    closed = []
    owner.close = lambda: closed.append(True)
    profile = SimpleNamespace(construct_generator=lambda factory: owner,
                              manifest_sha256='manifest')
    monkeypatch.setattr(engagement, 'register_gfx1151_kernels', lambda **kw: None)
    monkeypatch.setattr(engagement, 'register_qwen4_exp_gfx1151_profiles', lambda: None)
    monkeypatch.setattr(engagement, 'resolve_runtime_profile', lambda **kw: profile)
    monkeypatch.setattr(engagement, 'discover_gguf_files', lambda path: ['fake'])
    monkeypatch.setattr(engagement, 'load_gguf_index', lambda path: SimpleNamespace(architecture='fake'))
    ids = ('code-p512', 'code-p1024', 'code-p4096')
    monkeypatch.setattr(engagement, 'load_fixture', lambda path: (
        {'cases': [{'id': key, 'prompt_token_ids': [1, 2]} for key in ids]}, 'fixture'))
    key = KernelKey('hip_gfx1151', 'gdn_recurrence_norm_gate', 'f32_state', engagement.VARIANT)
    monkeypatch.setattr(engagement, 'registered_keys', lambda: [key])
    original = lambda *a, **kw: None
    monkeypatch.setattr(engagement, 'resolve', lambda **kw: original)
    registrations = []
    monkeypatch.setattr(engagement, 'register', lambda key, fn, **kw: registrations.append(fn))
    monkeypatch.setattr(engagement, '_git_metadata', lambda path: {})
    monkeypatch.setattr(engagement, '_host_metadata', lambda: {})
    monkeypatch.setattr(engagement, 'memory_stats', lambda: {})
    args = ['--model-root', str(tmp_path), '--compiler-version-file', str(tmp_path / 'version'),
            '--output', str(tmp_path / 'report.json')]
    if fail:
        with pytest.raises(ProbeFailure):
            engagement.main(args)
    else:
        engagement.main(args)
    assert observed
    assert all(row == (True, True, engagement.VARIANT, engagement.VARIANT)
               for row in observed)
    assert owner.gdn_colwarps_prefill is prior
    assert runner.gdn_colwarps_prefill is prior
    assert owner.gdn_tile16_variant == runner.gdn_tile16_variant == 'published'
    assert registrations[-1] is original
    assert closed == [True]


@pytest.mark.parametrize('narrow', [True, False])
def test_serial_probe_restores_all_state_after_measurement_raises(narrow):
    runner = SimpleNamespace(gdn_colwarps_prefill=True,
                             gdn_prefill_variant='published')
    with pytest.raises(RuntimeError, match='measurement'):
        with owner_probe.serial_probe_scope(runner, narrow_index=narrow) as (_, _, report):
            assert runner.gdn_colwarps_prefill is False
            assert report['runner_field'] == 'gdn_colwarps_prefill'
            assert report['previous'] is True
            raise RuntimeError('measurement')
    assert runner.gdn_colwarps_prefill is True
    assert runner.gdn_prefill_variant == 'published'


@pytest.mark.parametrize('package', ['gdn-narrow', 'gdn-wave-norm', 'gdn-register'])
@pytest.mark.parametrize('mode', ['before', 'after'])
def test_halo_serial_scope_changes_only_measured_runner(package, mode):
    runner = SimpleNamespace(gdn_colwarps_prefill=True,
                             gdn_prefill_variant=None, gdn_register_prefill=True,
                             gdn_peer_prefill_layers=frozenset({27}))
    other = SimpleNamespace(gdn_colwarps_prefill=True)
    previous = vars(runner).copy()
    with pytest.raises(RuntimeError, match='measurement'):
        with halo.route_scope(package, mode, runner):
            assert runner.gdn_colwarps_prefill is False
            assert other.gdn_colwarps_prefill is True
            raise RuntimeError('measurement')
    assert vars(runner) == previous


@pytest.mark.parametrize('enabled', ['on', 'off'])
def test_profile_gap_accepts_owner_control_and_rejects_retired_override(enabled):
    from scripts import qwen4exp_profile_gap as gap
    args = gap.build_parser().parse_args([
        '--model-root', '/models/example', '--mode', 'prefill',
        '--output', '/tmp/report.json',
        '--gdn-colwarps-prefill', enabled])
    assert args.gdn_colwarps_prefill == enabled
    with pytest.raises(ValueError, match='--gdn-colwarps-prefill'):
        gap._parse_overrides(['HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL=1'])
    value = enabled == 'on'
    assert not gap._named_profile_intact({}, None, gdn_colwarps_override=value)
    assert gap._configuration_class({}, None, gdn_colwarps_override=value) == 'diagnostic_post_binder_override'


def test_ranking_forwards_gate_owner_to_child_and_keeps_actual_report(monkeypatch, tmp_path):
    from scripts import qwen4exp_disabled_selector_ranking as ranking
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        output = command[command.index('--output') + 1]
        from pathlib import Path
        Path(output).write_text(json.dumps({'gdn_colwarps_prefill': True}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ranking.subprocess, 'run', run)
    row = ranking._run_arm(
        name='gdn_colwarps', overrides=ranking.ARMS['gdn_colwarps'],
        owner=ranking.OWNER_ARMS['gdn_colwarps'], model_root=tmp_path,
        fixture=tmp_path / 'fixture.json', case_id='case', chunk_size=1024,
        repetitions=1, compiler_version_file=tmp_path / 'version', work_root=tmp_path)
    command = seen[0]
    assert command[command.index('--gdn-colwarps-prefill') + 1] == 'on'
    assert not any('HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL=' in arg for arg in command)
    assert row['gdn_colwarps_prefill'] is True


def test_family_sweep_forwards_the_gate_owner_to_child(monkeypatch, tmp_path):
    from scripts import qwen4exp_disabled_family_sweep as sweep
    seen = []
    output = tmp_path / 'child.json'

    def run(command, **kwargs):
        seen.append(command)
        output.write_text('{}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, 'run', run)
    monkeypatch.setattr(sweep, '_summarize', lambda payload, command: {'command': command})
    family = next(row for row in sweep.FAMILIES if row['family'] == 'GDN_COLWARPS_PREFILL')
    sweep._run(python='python', model_root=tmp_path, fixture=tmp_path / 'fixture',
               case_id='case', repetitions=1, overrides=family['env'], output=output,
               owner=family['owner'])
    command = seen[0]
    assert command[command.index('--gdn-colwarps-prefill') + 1] == 'on'
    assert not any('HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL=' in arg for arg in command)
