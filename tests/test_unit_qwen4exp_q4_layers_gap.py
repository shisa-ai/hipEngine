"""Profile-gap layer scopes bind owners without rewriting process inputs."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS'


@pytest.mark.parametrize('scope', ['', 'all', '2,17-19', '35-47', '999'])
def test_layer_override_is_local_and_restoring(monkeypatch, scope):
    monkeypatch.setenv(ENV, 'ambient')
    runner = SimpleNamespace(q4_iu8_layers='runner')
    owner = SimpleNamespace(runner=runner, q4_iu8_layers='owner')
    peer = SimpleNamespace(q4_iu8_layers='peer')
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            assert gap.bind_q4_iu8_layers_override(stack, owner, {ENV: scope}) == {
                'q4_iu8_layers_override': scope, 'q4_iu8_layers': scope}
            gap._apply_post_binder_overrides({ENV: scope})
            assert owner.q4_iu8_layers == runner.q4_iu8_layers == scope
            assert peer.q4_iu8_layers == 'peer'
            assert os.environ[ENV] == 'ambient'
            raise RuntimeError('restore')
    assert owner.q4_iu8_layers == 'owner'
    assert runner.q4_iu8_layers == 'runner'


def test_layer_omission_preserves_owner_and_main_reports_it(monkeypatch):
    monkeypatch.setenv(ENV, 'ambient')
    owner = SimpleNamespace(q4_iu8_layers='')
    with ExitStack() as stack:
        assert gap.bind_q4_iu8_layers_override(stack, owner, {}) == {
            'q4_iu8_layers_override': None, 'q4_iu8_layers': ''}
    assert 'report.update(bind_q4_iu8_layers_override(' in inspect.getsource(gap.main)
