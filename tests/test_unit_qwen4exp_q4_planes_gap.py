"""Diagnostic Q4 plane overrides bind owners without process writes."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace

import pytest
from scripts import qwen4exp_profile_gap as gap
from scripts import qwen4exp_candidate_ar as ar

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES'


@pytest.mark.parametrize('module', [gap, ar])
@pytest.mark.parametrize('raw,expected', [('2', 2), ('3', 3), ('bad', 3), ('', 3)])
def test_plane_override_is_local_and_restoring(monkeypatch, module, raw, expected):
    monkeypatch.setenv(ENV, '2')
    owner = SimpleNamespace(q4_iu8_planes=3, runner=SimpleNamespace(q4_iu8_planes=2))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            assert module.bind_q4_iu8_planes_override(stack, owner, {ENV: raw}) == {
                'q4_iu8_planes_override': raw, 'q4_iu8_planes': expected}
            gap._apply_post_binder_overrides({ENV: raw})
            assert owner.q4_iu8_planes == owner.runner.q4_iu8_planes == expected
            assert os.environ[ENV] == '2'
            raise RuntimeError('restore')
    assert owner.q4_iu8_planes == 3 and owner.runner.q4_iu8_planes == 2


def test_plane_omission_and_both_entrypoints():
    owner = SimpleNamespace(q4_iu8_planes=2)
    with ExitStack() as stack:
        assert gap.bind_q4_iu8_planes_override(stack, owner, {}) == {
            'q4_iu8_planes_override': None, 'q4_iu8_planes': 2}
    assert 'report.update(bind_q4_iu8_planes_override(' in inspect.getsource(gap.main)
    source = inspect.getsource(ar.main)
    assert source.index('split_q4_iu8_planes_override(overrides)') < source.index('os.environ.update(overrides)')
    assert "artifact['owner_overrides'].update(bind_q4_iu8_planes_override(" in source
    assert ar.split_q4_iu8_planes_override({ENV: '2', 'OTHER': 'x'}) == (
        {'OTHER': 'x'}, {ENV: '2'})
