"""Layer-2 dense-wide scopes bind the measured owner and restore it."""
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.runtime.gguf_linear import Q8DenseWidePolicy, resolve_q8_dense_wide_policy
from scripts import qwen4exp_layer2_profile_gate as gate

KEY = 'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS'


def test_candidate_scopes_leave_environment_transport():
    for name, spec in gate._f16_route_isolation_candidates().items():
        expected = tuple(range(48)) if name == 'recovery_plus_dense_wide_all_layers' else tuple(range(13, 48))
        assert KEY not in spec.environment
        assert spec.q8_dense_wide_layers == expected
    assert gate.CANDIDATES['fast_composition'].q8_dense_wide_layers == tuple(range(48))
    assert KEY not in gate.CANDIDATES['fast_composition'].environment


@pytest.mark.parametrize('enabled', [False, True])
def test_shared_candidate_binding_preserves_enablement_and_restores(enabled):
    prior = Q8DenseWidePolicy(enabled, frozenset({7}))
    runner = SimpleNamespace(q8_dense_wide_policy=prior)
    generator = SimpleNamespace(runner=runner)
    spec = replace(gate.CANDIDATES['production_baseline'], q8_dense_wide_layers=(0, 16, 47))
    with pytest.raises(RuntimeError, match='unwind'):
        with ExitStack() as stack:
            report = gate.bind_candidate_owners(stack, generator, spec)
            assert resolve_q8_dense_wide_policy(runner) == Q8DenseWidePolicy(enabled, frozenset({0, 16, 47}))
            assert report['q8_dense_wide_policy'] == {'enabled': enabled, 'layers': [0, 16, 47]}
            assert gate.candidate_owner_changes(spec)['q8_dense_wide_layers'] == [0, 16, 47]
            assert not gate._candidate_named_profile_intact(spec)
            raise RuntimeError('unwind')
    assert runner.q8_dense_wide_policy is prior


def test_restore_comparison_honors_shared_candidate_declaration(monkeypatch):
    from scripts import qwen4exp_prefill_restore_ab as restore
    prior = Q8DenseWidePolicy(True, frozenset({7}))
    runner = SimpleNamespace(q8_dense_wide_policy=prior)
    spec = replace(gate.CANDIDATES['production_baseline'], q8_dense_wide_layers=(0, 47))
    monkeypatch.setitem(restore.CANDIDATES, 'test_wide_scope', spec)
    with restore.arm_dense_wide_scope(runner, 'test_wide_scope'):
        assert resolve_q8_dense_wide_policy(runner) == Q8DenseWidePolicy(True, frozenset({0, 47}))
    assert runner.q8_dense_wide_policy is prior
    with restore.arm_dense_wide_scope(runner, 'before'):
        assert runner.q8_dense_wide_policy is prior


def test_unspecified_candidate_keeps_owner():
    prior = Q8DenseWidePolicy(True, frozenset({7}))
    runner = SimpleNamespace(q8_dense_wide_policy=prior)
    with ExitStack() as stack:
        policy = gate.bind_candidate_dense_wide(stack, SimpleNamespace(runner=runner), gate.CANDIDATES['production_baseline'])
        assert policy == prior.to_report()
        assert runner.q8_dense_wide_policy is prior
