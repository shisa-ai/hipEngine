"""Legacy plane CLI spelling binds explicit owner state in both diagnostics."""
from contextlib import ExitStack
from types import SimpleNamespace
import inspect
import os
import pytest
from scripts import qwen4exp_candidate_ar as ar
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES'


@pytest.mark.parametrize('module', [ar, gap])
@pytest.mark.parametrize('raw', ['2', '3'])
def test_override_binds_reports_and_restores(monkeypatch, module, raw):
    monkeypatch.setenv(ENV, str(5 - int(raw)))
    owner = SimpleNamespace(q8_mmq_f32_planes=None, runner=SimpleNamespace(q8_mmq_f32_planes=None))
    with ExitStack() as stack:
        report = module.bind_mmq_planes_override(stack, owner, {ENV: raw})
        assert report == {'q8_mmq_f32_planes_override': raw, 'q8_mmq_f32_planes': int(raw)}
        assert owner.q8_mmq_f32_planes == owner.runner.q8_mmq_f32_planes == int(raw)
        assert os.environ[ENV] == str(5 - int(raw))
    assert owner.q8_mmq_f32_planes is owner.runner.q8_mmq_f32_planes is None


@pytest.mark.parametrize('raw', ['', '1', '4', '2.5', 'two'])
def test_invalid_request_raises_before_binding(raw):
    owner = SimpleNamespace(q8_mmq_f32_planes=3)
    with ExitStack() as stack, pytest.raises(ValueError, match='2 or 3'):
        gap.bind_mmq_planes_override(stack, owner, {ENV: raw})
    assert owner.q8_mmq_f32_planes == 3


def test_absent_override_preserves_owner():
    with ExitStack() as stack:
        assert gap.bind_mmq_planes_override(stack, SimpleNamespace(q8_mmq_f32_planes=3), {}) == {
            'q8_mmq_f32_planes_override': None, 'q8_mmq_f32_planes': 3}


def test_candidate_splits_before_both_environment_write_stages():
    assert ar.split_mmq_planes_override({ENV: '2', 'OTHER': 'x'}) == ({'OTHER': 'x'}, {ENV: '2'})
    source = inspect.getsource(ar.main)
    assert source.index('split_mmq_planes_override(overrides)') < source.index('os.environ.update(overrides)')
    assert "artifact['owner_overrides'].update(bind_mmq_planes_override(" in source


def test_gap_does_not_write_plane_override(monkeypatch):
    monkeypatch.setenv(ENV, '3')
    gap._apply_post_binder_overrides({ENV: '2'})
    assert os.environ[ENV] == '3'
    assert 'report.update(bind_mmq_planes_override(' in inspect.getsource(gap.main)
