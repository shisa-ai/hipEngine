"""Known failed profile plans cannot be selected just because they are registered."""
from types import SimpleNamespace

from hipengine.execution_profiles import (
    ExecutionProfile, RuntimeProfilePlan, VariantSelection,
    register_runtime_profile_plan, resolve_default_execution_profile,
    resolve_runtime_profile,
)
from hipengine.kernels.registry import KernelKey, register


def test_failed_production_plan_uses_strict_default_and_explicit_fallback():
    key = dict(model='quarantine_test', backend='quarantine_backend', quant='quarantine_quant')
    register(KernelKey(backend=key['backend'], layer='linear', quant=key['quant'], variant='strict'), lambda: None, replace=True)
    strict_selection = VariantSelection('linear', 'all', 'strict', 'strict')
    def strict_factory(**kwargs):
        return SimpleNamespace(owner='strict')
    register_runtime_profile_plan(**key, profile='strict', plan=RuntimeProfilePlan(
        selections=(strict_selection,), kv_policy='bf16', graph_policy='none', factory=strict_factory), replace=True)
    # Failed variant deliberately isn't registered. It must never be verified or bound.
    register_runtime_profile_plan(**key, profile='production', plan=RuntimeProfilePlan(
        selections=(VariantSelection('linear', 'all', 'failed', 'strict'),),
        kv_policy='bf16', graph_policy='none', factory=lambda **kw: (_ for _ in ()).throw(AssertionError()),
        qualification_failure='numerical envelope failed; evidence/test.json'), replace=True)
    assert resolve_default_execution_profile(**key) is ExecutionProfile.STRICT
    result = resolve_runtime_profile(**key, profile='production')
    assert result.source_profile is ExecutionProfile.STRICT
    assert result.fell_back_to_strict
    assert result.qualification_failure == 'numerical envelope failed; evidence/test.json'
    assert result.manifest['selections'][0]['selected_variant'] == 'strict'
    generator = result.construct_generator(lambda **kw: None)
    assert generator.owner == 'strict'
    assert generator.execution_profile_qualification_failure == result.qualification_failure
    from hipengine.server.api import ServerConfig, _server_execution_profile
    engine = SimpleNamespace(
        resolved_execution_profile='production',
        execution_profile_fell_back_to_strict=True,
        execution_profile_qualification_failure=result.qualification_failure,
    )
    metadata = _server_execution_profile(ServerConfig(model='test'), engine)
    assert metadata['qualification_failure'] == result.qualification_failure
    assert metadata['fell_back_to_strict'] is True


def _assert_qwen4exp_q4km_failed_plan_is_not_a_production_default():
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    register_qwen4_exp_gfx1151_profiles()
    key = dict(model='qwen4_exp_gguf', backend='hip_gfx1151', quant='gguf_q4_k_m')
    assert resolve_default_execution_profile(**key) is ExecutionProfile.STRICT
    strict = resolve_runtime_profile(**key, profile='strict')
    production = resolve_runtime_profile(**key, profile='production')
    assert production.fell_back_to_strict
    assert production.binder is strict.binder
    assert [(s['layer'], s['scope'], s['selected_variant']) for s in production.manifest['selections']] == [
        (s['layer'], s['scope'], s['selected_variant']) for s in strict.manifest['selections']]
    assert '2026-09-30-qwen4exp-gdn-multicolumn-q4km/gate.json' in production.qualification_failure
    assert resolve_default_execution_profile(**{**key, 'quant': 'gguf_ud_q4_k_xl'}) is ExecutionProfile.PRODUCTION


def test_qwen4exp_q4km_failed_plan_is_not_a_production_default():
    import subprocess
    import sys

    subprocess.run([sys.executable, '-c',
        "import runpy; ns = runpy.run_path(" + repr(__file__) + "); "
        "ns['_assert_qwen4exp_q4km_failed_plan_is_not_a_production_default']()"
    ], check=True, timeout=60)
