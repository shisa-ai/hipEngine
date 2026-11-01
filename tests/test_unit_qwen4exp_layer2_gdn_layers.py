"""Layer-2 GDN scope declarations bind their measured owner, not the process."""
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace

import pytest
from scripts import qwen4exp_layer2_profile_gate as gate
from scripts import qwen4exp_prefill_restore_ab as restore
from hipengine.runtime.qwen4_exp_runner import resolve_gdn_colwarps_layers

ENV = 'HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS'


def test_candidates_declare_layers_on_the_owner():
    expected = {
        'recovery_plan': ','.join(map(str, range(27, 48))),
        'recovery_plus_gdn_tiled_all_layers': ','.join(map(str, range(48))),
        'fast_composition': ','.join(map(str, range(48))),
        'gdn_isolated': ','.join(map(str, range(27, 48))),
    }
    for name, scope in expected.items():
        spec = gate.CANDIDATES[name]
        assert ENV not in spec.environment
        assert spec.gdn_colwarps_layers == scope
        assert gate.candidate_owner_changes(spec)['gdn_colwarps_layers'] == scope


@pytest.mark.parametrize('scope', ['', '0', '7-9'])
def test_shared_binder_restores_owner_on_error(monkeypatch, scope):
    monkeypatch.setenv(ENV, 'all')
    spec = replace(gate.CANDIDATES['gdn_isolated'], gdn_colwarps_layers=scope)
    owner = SimpleNamespace(gdn_colwarps_layers='13-47', runner=SimpleNamespace(gdn_colwarps_layers='13-47'))
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            selected = gate.bind_candidate_gdn_layers(stack, owner, spec)
            assert selected == resolve_gdn_colwarps_layers(owner) == scope
            assert owner.runner.gdn_colwarps_layers == scope
            raise RuntimeError('injected')
    assert owner.gdn_colwarps_layers == owner.runner.gdn_colwarps_layers == '13-47'


def test_restore_scope_and_shared_graph_inventory(monkeypatch):
    monkeypatch.setenv(ENV, 'all')
    owner = SimpleNamespace(gdn_colwarps_layers='13-47')
    with restore.arm_gdn_layers_scope(owner, 'gdn_isolated'):
        assert resolve_gdn_colwarps_layers(owner) == ','.join(map(str, range(27, 48)))
    assert owner.gdn_colwarps_layers == '13-47'
    assert 'gdn_colwarps_layers' in restore.SHARED_DECODE_SAFE_OWNER_FIELDS


def test_shared_candidate_owner_report_is_captured_after_binding(monkeypatch):
    monkeypatch.setenv(ENV, 'all')
    spec = replace(gate.CANDIDATES['production_baseline'], gdn_colwarps_layers='0')
    owner = SimpleNamespace(gdn_colwarps_layers='13-47', runner=SimpleNamespace())
    with ExitStack() as stack:
        record = gate.bind_candidate_owners(stack, owner, spec)
        assert record['gdn_colwarps_layers'] == '0'
    assert owner.gdn_colwarps_layers == '13-47'
