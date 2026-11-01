"""Layer2 GR projection choices are explicit candidate axes, not process keys."""
from contextlib import ExitStack
from dataclasses import replace
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_layer2_profile_gate as gate
from hipengine.runtime.qwen4_exp_runner import GRIU8Policy

UP = 'HIPENGINE_QWEN4_EXP_GR_IU8'
DOWN = 'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN'


def test_candidates_declare_gr_pair_not_process_input():
    expected = {'production_conservative': (False, False),
        'q8_blockscale_guarded_restore': (True, True),
        'production_gr_up_restore': (True, False),
        'production_gr_up_wide': (False, False),
        'production_gr_down_restore': (False, True)}
    for name, pair in expected.items():
        spec = gate.CANDIDATES[name]
        assert (spec.gr_iu8_up, spec.gr_iu8_down) == pair
    assert all(UP not in s.environment and DOWN not in s.environment for s in gate.CANDIDATES.values())


@pytest.mark.parametrize('up,down', [(None, None), (True, None), (None, False), (True, True), (False, False)])
def test_layer2_gr_pair_preserves_omitted_axes_and_restores(monkeypatch, up, down):
    monkeypatch.setenv(UP, 'ambient-up')
    monkeypatch.setenv(DOWN, 'ambient-down')
    prior = GRIU8Policy(False, True)
    runner_prior = GRIU8Policy(True, False)
    owner = SimpleNamespace(gr_iu8_policy=prior, runner=SimpleNamespace(gr_iu8_policy=runner_prior))
    spec = replace(gate.CANDIDATES['production_baseline'], gr_iu8_up=up, gr_iu8_down=down)
    expected = {'up': False if up is None else up, 'down': True if down is None else down}
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            assert gate.bind_candidate_gr_iu8(stack, owner, spec) == expected
            assert owner.gr_iu8_policy == GRIU8Policy(**expected)
            assert os.environ[UP] == 'ambient-up' and os.environ[DOWN] == 'ambient-down'
            changes = gate.candidate_owner_changes(spec)
            assert ('gr_iu8_up' in changes) is (up is not None)
            assert ('gr_iu8_down' in changes) is (down is not None)
            from scripts.qwen4exp_prefill_restore_ab import validate_shared_graphs
            validate_shared_graphs([], [changes])
            raise RuntimeError('restore')
    assert owner.gr_iu8_policy is prior
    assert owner.runner.gr_iu8_policy is runner_prior


def test_layer2_gr_pair_wiring():
    assert 'bind_candidate_gr_iu8(' in inspect.getsource(gate.bind_candidate_owners)
    assert 'bind_candidate_gr_iu8(' in inspect.getsource(gate._run)
    for name in ['qwen4exp_prefill_restore_ab', 'qwen4exp_conservative_cost']:
        module = __import__('scripts.' + name, fromlist=[name])
        assert 'candidate_gr_iu8_scope(' in inspect.getsource(module)
