"""Diagnostic CLI rejects invalid scope before constructing a GPU session."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/qwen38_packed_c1_teacher_run.py'


@pytest.mark.parametrize('fail', [False, True])
def test_contexts_use_resolved_adapter_flags_and_unwind(monkeypatch, fail):
    from contextlib import contextmanager
    from types import SimpleNamespace as NS
    from hipengine.generation import qwen35_gguf_mtp2 as m
    from scripts.qwen38_packed_c1_teacher_run import target_contexts
    entered, exited = [], []
    generator = object()
    def adapter(owner, **kwargs):
        assert owner.generator is generator and owner.capacity == 1
        assert kwargs['candidate_budget'] == 7
        return NS(production_physical_extra_rowtiles=True,
                  production_exact_target_row_counts=(5, 6, 7, 8),
                  production_physical_q6_rowtile=True)
    monkeypatch.setattr(m, 'Qwen35GGUFMTP2Adapter', adapter)
    names = ('target_verifier_active_slots_session', 'q4_t16_physical_extra_rowtiles_session',
             'physical_exact_rowtiles_session', 'q5_t16_physical_rowtile_session',
             'q6_t16_physical_rowtile_session', 'q6_t16_physical_mixed_rowtiles_session',
             'moe_physical_c2_numerics_session', 'moe_physical_c2_pairreuse_session',
             'moe_physical_c2_exact_linear_session', 'target_verifier_wide_q6_shared4_leaf_session')
    def context(name):
        @contextmanager
        def scoped(value):
            entered.append((name, value))
            try:
                yield
            finally:
                exited.append(name)
        return scoped
    for name in names:
        monkeypatch.setattr(m, name, context(name))
    try:
        with target_contexts(generator, 7) as flags:
            assert flags['exact_target_rows'] == [5, 6, 7, 8]
            assert flags['production_physical_extra_rowtiles'] is True
            assert flags['production_physical_q5_rowtile'] is False
            assert len(entered) == 10
            assert dict(entered)['target_verifier_active_slots_session'] == 1
            if fail:
                raise RuntimeError('injected')
    except RuntimeError:
        assert fail
    assert exited == list(reversed(names))


def test_help():
    result = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--teacher' in result.stdout


@pytest.mark.parametrize('extra', [['--budget', '8'], ['--budget', '0']])
def test_invalid_depth_fails_before_gpu_setup(extra):
    result = subprocess.run([sys.executable, str(SCRIPT), '--teacher', '/missing',
                             '--directory', '/missing', *extra], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr
