"""Generic diagnostic syntax binds DPP owner state without process writes."""
from contextlib import ExitStack
import importlib
import inspect
import os
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

KEY = 'HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION'


@pytest.mark.parametrize('interface', ['profile_gap', 'candidate_ar'])
@pytest.mark.parametrize('raw', ['0', '1', '', 'false', 'other'])
def test_dpp_override_restores_owner_and_never_writes_process(monkeypatch, interface, raw):
    script = importlib.import_module('scripts.qwen4exp_' + interface)
    monkeypatch.setenv(KEY, 'caller')
    owner = SimpleNamespace(q8_dpp_reduction_enabled=None,
                            runner=SimpleNamespace(q8_dpp_reduction_enabled=False))
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            report = script.bind_dpp_override(stack, owner, {KEY: raw})
            expected = raw != '0'  # Preserve legacy selector semantics, not generic flag parsing.
            assert owner.q8_dpp_reduction_enabled is expected
            assert owner.runner.q8_dpp_reduction_enabled is expected
            assert _inherit_lane_decisions(owner, SimpleNamespace()).q8_dpp_reduction_enabled is expected
            assert report == {'q8_dpp_reduction_override': raw,
                              'q8_dpp_reduction_enabled': expected}
            assert dict(os.environ) == before
            raise RuntimeError('restore')
    assert owner.q8_dpp_reduction_enabled is None
    assert owner.runner.q8_dpp_reduction_enabled is False


@pytest.mark.parametrize('interface', ['profile_gap', 'candidate_ar'])
@pytest.mark.parametrize('prior', [None, False, True])
def test_omitted_dpp_override_preserves_owner(interface, prior):
    script = importlib.import_module('scripts.qwen4exp_' + interface)
    owner = SimpleNamespace(q8_dpp_reduction_enabled=prior)
    with ExitStack() as stack:
        report = script.bind_dpp_override(stack, owner, {})
        assert owner.q8_dpp_reduction_enabled is prior
        assert report == {'q8_dpp_reduction_override': None,
                          'q8_dpp_reduction_enabled': prior}


def test_generic_mains_exclude_dpp_from_both_process_write_stages(monkeypatch):
    from scripts import qwen4exp_candidate_ar as ar, qwen4exp_profile_gap as gap
    overrides = {KEY: '0', 'HIPENGINE_HIP_ARCH': 'same'}
    monkeypatch.setenv(KEY, 'caller')
    monkeypatch.setenv('HIPENGINE_HIP_ARCH', 'prior')
    environment, selected = ar.split_dpp_override(overrides)
    assert environment == {'HIPENGINE_HIP_ARCH': 'same'}
    assert selected == {KEY: '0'}
    source = inspect.getsource(ar.main)
    assert source.index('split_dpp_override(overrides)') < source.index('os.environ.update(overrides)')
    assert "artifact['owner_overrides'].update(bind_dpp_override(" in source
    assert 'report.update(bind_dpp_override(' in inspect.getsource(gap.main)
    gap._apply_post_binder_overrides(overrides)
    assert os.environ[KEY] == 'caller'
    assert os.environ['HIPENGINE_HIP_ARCH'] == 'same'
