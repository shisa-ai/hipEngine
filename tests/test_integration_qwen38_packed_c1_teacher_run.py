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


@pytest.mark.parametrize('fault', [None, 'hash', 'backend', 'quant', 'model', 'kv_policy', 'profile'])
def test_candidate_provenance_matches_teacher_scope(fault):
    from types import SimpleNamespace as NS
    from hipengine.execution_profiles import build_variant_manifest, manifest_sha256
    from scripts.qwen38_packed_c1_teacher_run import candidate_provenance
    teacher = build_variant_manifest(profile='strict', backend='hip_gfx1100',
        model='example', quant='gguf', kv_policy='paged_bf16', graph_policy='eager',
        selections=[dict(layer='linear', scope='all', selected_variant='strict',
                         strict_fallback_variant='strict')])
    manifest = dict(teacher, execution_profile='production')
    if fault in ('backend', 'quant', 'model', 'kv_policy'):
        manifest[fault] = 'other'
    if fault == 'profile':
        manifest['execution_profile'] = 'strict'
    digest = manifest_sha256(manifest)
    llm = NS(execution_profile_manifest=manifest, execution_profile_manifest_sha256=digest)
    generator = NS(execution_profile_manifest_sha256='0' * 64 if fault == 'hash' else digest)
    fixture = dict(runtime_manifest=teacher, runtime_manifest_sha256=manifest_sha256(teacher))
    if fault:
        with pytest.raises(ValueError, match='provenance'):
            candidate_provenance(llm, generator, fixture)
    else:
        result = candidate_provenance(llm, generator, fixture)
        assert result['candidate_manifest'] == manifest
        assert result['teacher_runtime_manifest_sha256'] == fixture['runtime_manifest_sha256']


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
