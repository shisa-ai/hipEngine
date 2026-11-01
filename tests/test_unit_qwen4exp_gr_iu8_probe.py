"""The GR logits probe binds one projection axis without process writes."""
import inspect
import os
from types import SimpleNamespace
import pytest
from hipengine.runtime.qwen4_exp_runner import GRIU8Policy
from scripts import qwen4exp_gr_iu8_logits_probe as probe


@pytest.mark.parametrize('route', ['up', 'down'])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('other', [False, True])
def test_probe_binds_axis_and_restores_after_failure(monkeypatch, route, enabled, other):
    prior = GRIU8Policy(not enabled, other) if route == 'up' else GRIU8Policy(other, not enabled)
    owner = SimpleNamespace(gr_iu8_policy=prior)
    for key in probe.FLAGS:
        monkeypatch.setenv(key, '0' if enabled else '1')
    ambient = {key: os.environ[key] for key in probe.FLAGS}
    with pytest.raises(RuntimeError, match='restore'):
        with probe._route_scope(owner, route, enabled):
            expected = GRIU8Policy(enabled, other) if route == 'up' else GRIU8Policy(other, enabled)
            assert owner.gr_iu8_policy == expected
            assert {key: os.environ[key] for key in probe.FLAGS} == ambient
            raise RuntimeError('restore')
    assert owner.gr_iu8_policy is prior
    assert {key: os.environ[key] for key in probe.FLAGS} == ambient


def test_probe_teardown_does_not_rewrite_gr_inputs():
    source = inspect.getsource(probe.main)
    assert 'for flag in FLAGS:' not in source
    assert probe._route_record(SimpleNamespace(gr_iu8_policy=GRIU8Policy(True, False)), 'up') == {
        'owner_field': 'gr_iu8_policy', 'axis': 'up', 'incumbent': False,
        'candidate': True, 'other_axis': 'preserve',
        'published_value': {'up': True, 'down': False}}
    assert '"control": _route_record(generator, a.route)' in source
