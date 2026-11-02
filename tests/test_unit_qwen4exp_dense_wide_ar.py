"""Candidate AR binds both dense-wide axes without process writes."""
from contextlib import ExitStack
from types import SimpleNamespace
import inspect
import os
import pytest
from hipengine.runtime.gguf_linear import Q8DenseWidePolicy
from scripts import qwen4exp_candidate_ar as ar

ENABLE = 'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE'
LAYERS = 'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS'


@pytest.mark.parametrize('overrides,expected', [
    ({ENABLE: '0'}, Q8DenseWidePolicy(False, frozenset({7}))),
    ({ENABLE: '1'}, Q8DenseWidePolicy(True, frozenset({7}))),
    ({LAYERS: ','.join(map(str, range(16, 48)))}, Q8DenseWidePolicy(True, frozenset(range(16, 48)))),
    ({ENABLE: '0', LAYERS: ''}, Q8DenseWidePolicy(False, None)),
    ({}, Q8DenseWidePolicy(True, frozenset({7}))),
])
def test_override_preserves_unspecified_axes_reports_and_restores(monkeypatch, overrides, expected):
    monkeypatch.setenv(ENABLE, 'ambient')
    monkeypatch.setenv(LAYERS, 'ambient')
    prior = Q8DenseWidePolicy(True, frozenset({7}))
    owner = SimpleNamespace(q8_dense_wide_policy=prior,
                            runner=SimpleNamespace(q8_dense_wide_policy=prior))
    with ExitStack() as stack:
        report = ar.bind_dense_wide_override(stack, owner, overrides)
        assert owner.q8_dense_wide_policy == owner.runner.q8_dense_wide_policy == expected
        assert report == {'q8_dense_wide_override': dict(overrides),
                          'q8_dense_wide_policy': expected.to_report()}
        assert os.environ[ENABLE] == os.environ[LAYERS] == 'ambient'
    assert owner.q8_dense_wide_policy is owner.runner.q8_dense_wide_policy is prior


def test_split_happens_before_both_process_write_stages():
    assert ar.split_dense_wide_overrides({ENABLE: '1', LAYERS: '16-47', 'OTHER': 'x'}) == (
        {'OTHER': 'x'}, {ENABLE: '1', LAYERS: '16-47'})
    source = inspect.getsource(ar.main)
    assert source.index('split_dense_wide_overrides(overrides)') < source.index('os.environ.update(overrides)')
    assert "artifact['owner_overrides'].update(bind_dense_wide_override(" in source


def test_invalid_layers_do_not_mutate_owner():
    prior = Q8DenseWidePolicy(True, None)
    owner = SimpleNamespace(q8_dense_wide_policy=prior)
    with ExitStack() as stack, pytest.raises(ValueError):
        ar.bind_dense_wide_override(stack, owner, {LAYERS: 'broken'})
    assert owner.q8_dense_wide_policy is prior
