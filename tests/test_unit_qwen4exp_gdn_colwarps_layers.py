"""Explicit GDN columnwarp layer scopes outrank process diagnostics."""
import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

ENV = 'HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS'


def test_unbound_scope_preserves_legacy_diagnostics(monkeypatch):
    monkeypatch.setenv(ENV, '0,7-9')
    assert runtime.resolve_gdn_colwarps_layers(SimpleNamespace()) == '0,7-9'
    owner = SimpleNamespace(gdn_colwarps_layers='13-47')
    assert runtime.resolve_gdn_colwarps_layers(owner) == '13-47'
    monkeypatch.setenv(ENV, 'all')
    assert runtime.resolve_gdn_colwarps_layers(owner) == '13-47'


def test_scope_binds_both_owners_and_restores_missing_on_error(monkeypatch):
    monkeypatch.setenv(ENV, '27-47')
    runner = SimpleNamespace(gdn_colwarps_layers='all')
    owner = SimpleNamespace(runner=runner)
    with pytest.raises(RuntimeError, match='injected'):
        with runtime.gdn_colwarps_layers(owner, '0'):
            assert runtime.resolve_gdn_colwarps_layers(owner) == '0'
            assert runner.gdn_colwarps_layers == '0'
            with runtime.gdn_colwarps_layers(owner, '7-9'):
                assert runner.gdn_colwarps_layers == '7-9'
            assert runner.gdn_colwarps_layers == '0'
            raise RuntimeError('injected')
    assert not hasattr(owner, 'gdn_colwarps_layers')
    assert runner.gdn_colwarps_layers == 'all'


@pytest.mark.parametrize('scope,layer,expected', [
    ('0', 0, True), ('0', 7, False), ('7-9', 8, True),
    ('7-9', 12, False), ('', 31, True), ('all', 47, True),
])
def test_dispatch_scope_uses_explicit_value(monkeypatch, scope, layer, expected):
    monkeypatch.setenv(ENV, '2')
    weight = SimpleNamespace(spec=SimpleNamespace(slot_path=f'layers.{layer}.attn_qkv'))
    assert runtime.qwen4_exp_gdn_colwarps_layer_allowed(weight, published=scope) is expected


def test_late_runner_inherits_scope_and_all_forwarding_sites():
    owner = SimpleNamespace(gdn_colwarps_layers='0,13-47')
    clone = _inherit_lane_decisions(owner, SimpleNamespace())
    assert clone.gdn_colwarps_layers == owner.gdn_colwarps_layers
    for function in (runtime.run_qwen4_exp_gdn_layer,
                     runtime.run_qwen4_exp_gdn_token_mixer):
        assert inspect.signature(function).parameters['gdn_colwarps_layers'].default is None
    source = inspect.getsource(runtime.Qwen4ExpGGUFResidentModelRunner)
    assert 'gdn_colwarps_layers=self.gdn_colwarps_layers' in source


@pytest.mark.parametrize('scope,expected', [('27-47', 'parent'), ('0', 'serial')])
def test_real_mixer_consumes_explicit_scope_without_environment(monkeypatch, scope, expected):
    from tests.test_unit_qwen4exp_gdn_colwarps_owner import _mixer_launch
    record = _mixer_launch(
        monkeypatch, owner_kwargs={'gdn_colwarps_layers': scope},
        published=True, guard_environment=True)
    assert record['calls'] == [expected]
