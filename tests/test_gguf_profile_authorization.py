"""F5: real headers through the public profile chain; no device/math claims."""
from types import SimpleNamespace
from dataclasses import replace
from tests._gguf_profile_fixture import profile_context
import os

import pytest

from hipengine import LLM
from hipengine.execution_profiles import resolve_runtime_profile
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_admission import (
    preflight_qwen35_gguf_artifact, qwen35_gguf_artifact_preset_key,
    GGUF_UNQUALIFIED_MANIFEST_PRESET,
)
from tests._qwen35_gguf_fixture import default_fixture_tensors, fixture_metadata, write_qwen35_gguf
from hipengine.quant.gguf import GGMLQuantizationType as Q
from hipengine.kernels.backends import load_backend_kernel_package

load_backend_kernel_package('hip_gfx1151')


@pytest.fixture
def unknown(tmp_path):
    tensors = default_fixture_tensors(1, alpha_beta_type=Q.BF16)
    tensors[0] = ('token_embd.weight', (64, 256), Q.Q8_0)
    tensors.append(('output.weight', (64, 256), Q.Q8_0))
    metadata = fixture_metadata(1)
    path = tmp_path / 'Qwen3.8-27B-Q4_K_M.gguf'
    write_qwen35_gguf(path, tensors, metadata)
    return GGUFReader(path).info


@pytest.fixture(autouse=True)
def profiles(monkeypatch):
    # Backend registration is metadata-only. Do not mock profile resolution/binders.
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.generation import register_builtin_generators
    register_gfx1151_kernels(replace=True)
    register_builtin_generators()
    from hipengine.generation.qwen38_gguf_profiles import register_qwen38_gguf_gfx1151_profiles
    from hipengine.generation.qwen36_gguf_profiles import register_qwen36_gguf_gfx1151_profiles
    from hipengine.generation.qwen36_gguf_gfx1100_profiles import (
        register_qwen36_dense_gguf_gfx1100_profiles, register_qwen36_moe_gguf_gfx1100_profiles,
    )
    register_qwen38_gguf_gfx1151_profiles()
    register_qwen36_gguf_gfx1151_profiles()
    register_qwen36_dense_gguf_gfx1100_profiles()
    register_qwen36_moe_gguf_gfx1100_profiles()
    from hipengine.generation import qwen36_gguf_gfx1100_profiles as gfx1100_profiles
    saved = dict(os.environ)
    bound = dict(gfx1100_profiles._PROFILE_BOUND_ENV)
    yield
    os.environ.clear()
    os.environ.update(saved)
    gfx1100_profiles._PROFILE_BOUND_ENV.clear()
    gfx1100_profiles._PROFILE_BOUND_ENV.update(bound)


def resolve(*, model='qwen3_5_gguf', backend='hip_gfx1151', profile='production', **kwargs):
    return resolve_runtime_profile(model=model, backend=backend,
                                   quant='gguf_q4_k_m', profile=profile, **kwargs)


def test_unknown_is_generic_ar_qualified_not_plain(unknown, monkeypatch):
    mapping = build_qwen35_gguf_tensor_map(unknown)
    report = preflight_qwen35_gguf_artifact(mapping, backend='hip_gfx1151', decode_repack=False)
    assert report.supported
    assert qwen35_gguf_artifact_preset_key(mapping) == GGUF_UNQUALIFIED_MANIFEST_PRESET
    from hipengine.runtime.qwen35_gguf_runner import _gguf_fp16_recurrent_state_enabled
    monkeypatch.delenv('HIPENGINE_GGUF_FP16_RECURRENT_STATE', raising=False)
    assert not _gguf_fp16_recurrent_state_enabled(
        backend='hip_gfx1151', file_type_name=unknown.file_type_name,
        artifact_preset_key=GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )


@pytest.mark.parametrize('quant', ['auto', 'gguf_q4_k_m'])
@pytest.mark.parametrize('profile', ['strict', 'production', 'batch_invariant'])
def test_public_llm_refuses_unknown_before_factory_or_device(unknown, monkeypatch, quant, profile):
    import hipengine.generation as generation
    from hipengine.loading import qwen35_gguf_materialize as loader
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('forbidden')
        pytest.fail('unqualified profile reached factory/payload/device')
    monkeypatch.setattr(generation, 'resolve_text_generator', lambda **_: forbidden)
    monkeypatch.setattr(loader, 'malloc', forbidden)
    monkeypatch.setattr(GGUFReader, 'tensor_data', forbidden)
    llm = LLM(str(unknown.path), backend='hip_gfx1151', quant=quant, execution_profile=profile)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='qualification'):
        llm.generate(['hello'])
    assert calls == []
    assert dict(os.environ) == before


def test_direct_unknown_construct_cannot_enable_fp16(unknown):
    before = dict(os.environ)
    with pytest.raises(ValueError, match='qualification'):
        resolved = resolve()
        resolved.construct_generator(lambda **kw: SimpleNamespace(**kw), weight_index=unknown)
    assert dict(os.environ) == before


@pytest.mark.parametrize('entry', ['construct', 'binder'])
@pytest.mark.parametrize('damage', ['missing', 'unknown', 'other_plain', 'wrong_path'])
def test_qualified_resolution_cannot_be_reused_for_another_artifact(unknown, entry, damage):
    context = profile_context()
    resolved = resolve(qualification_context=context)
    wrong = {'weight_index': None}
    if damage == 'unknown':
        wrong = {'weight_index': unknown}
    elif damage == 'other_plain':
        wrong = profile_context(control='Qwen3.6-27B-Q4_K_M.gguf')
    elif damage == 'wrong_path':
        wrong = {**context, 'model_path': str(unknown.path)}
    called = []
    def factory(**kw):
        called.append(True)
        return SimpleNamespace(**kw)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='qualification'):
        if entry == 'construct':
            resolved.construct_generator(factory, **wrong)
        else:
            resolved.binder(SimpleNamespace(**wrong), resolved)
    assert not called
    assert dict(os.environ) == before


@pytest.mark.parametrize('damage', ['missing', 'unknown', 'conflicting_metadata', 'immutable'])
def test_custom_factory_cannot_bypass_binder_qualification_or_mutate_env_on_failure(unknown, damage):
    context = profile_context()
    resolved = resolve(qualification_context=context)
    generator = SimpleNamespace(**context)
    if damage == 'missing':
        generator = SimpleNamespace()
    elif damage == 'unknown':
        generator = SimpleNamespace(weight_index=unknown)
    elif damage == 'conflicting_metadata':
        generator.execution_profile = 'strict'
    else:
        class Immutable:
            weight_index = context['weight_index']
            def __setattr__(self, *args):
                raise TypeError('immutable')
        generator = Immutable()
    before = dict(os.environ)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        resolved.construct_generator(lambda **_: generator, **context)
    assert dict(os.environ) == before


@pytest.mark.parametrize('quant', ['auto', 'gguf_q4_k_m'])
@pytest.mark.parametrize('control', ['Qwen3.8-27B-UD-Q4_K_M.gguf', 'Qwen3.8-27B-UD-Q4_K_S.gguf'])
def test_exact_ud_cannot_borrow_plain_profile(control, quant, monkeypatch):
    import hipengine.generation as generation
    context = profile_context(control=control)
    before = dict(os.environ)
    monkeypatch.setattr(generation, 'resolve_text_generator', lambda **_: lambda **kw: pytest.fail('UD factory called'))
    llm = LLM(context['model_path'], backend='hip_gfx1151', quant=quant, execution_profile='production')
    with pytest.raises(ValueError, match='qualification refused.*gguf_ud'):
        llm.generate(['hello'])
    assert dict(os.environ) == before


def test_same_stamp_role_swap_cannot_borrow_plain_profile():
    from hipengine.quant.gguf import nbytes_for_shape, quant_shape_to_byte_shape
    context = profile_context()
    info = context['weight_index']
    tensors = list(info.tensors)
    # Swap storage types between sensitive roles without changing the histogram,
    # geometry, stamp, path, or names; keep the replacement byte extents truthful.
    a = next(i for i, t in enumerate(tensors) if t.name.endswith('ffn_gate.weight') and t.ggml_type_name == 'Q4_K')
    b = next(i for i, t in enumerate(tensors) if t.name.endswith('ffn_down.weight') and t.ggml_type_name == 'Q6_K')
    ta, tb = tensors[a], tensors[b]
    for i, original, other in ((a, ta, tb), (b, tb, ta)):
        tensors[i] = replace(original, ggml_type=other.ggml_type, ggml_type_name=other.ggml_type_name,
                             nbytes=nbytes_for_shape(original.shape, other.ggml_type),
                             byte_shape=quant_shape_to_byte_shape(original.shape, other.ggml_type))
    swapped = replace(info, tensors=tuple(tensors))
    assert sorted(t.ggml_type for t in info.tensors) == sorted(t.ggml_type for t in swapped.tensors)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='qualification refused.*unqualified'):
        resolve(qualification_context={'weight_index': swapped})
    assert dict(os.environ) == before


@pytest.mark.parametrize('backend', ['hip_gfx1100', 'hip_gfx1151'])
@pytest.mark.parametrize('model,control', [
    ('qwen3_5_gguf', 'Qwen3.8-27B-Q4_K_M.gguf'),
    ('qwen3_5_gguf', 'Qwen3.6-27B-Q4_K_M.gguf'),
    ('qwen3_5_moe_gguf', 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'),
    ('qwen3_5_moe_gguf', 'Ornith-1.5-35B-A3B-Q4_K_M.gguf'),
])
@pytest.mark.parametrize('profile', ['strict', 'production', 'batch_invariant'])
def test_pinned_controls_keep_supported_profile_and_binder(backend, model, control, profile):
    context = profile_context(model, control=control)
    resolved = resolve(model=model, backend=backend, profile=profile, qualification_context=context)
    assert resolved.fell_back_to_strict == (profile == 'batch_invariant')
    generator = resolved.construct_generator(lambda **kw: SimpleNamespace(**kw), **context)
    assert generator.execution_profile_manifest_sha256 == resolved.manifest_sha256
    assert resolved.artifact_identity[1] is None


@pytest.mark.parametrize('control', ['Qwen3.5-0.8B-Q4_K_M.gguf', 'Qwen3.5-0.8B-Q8_0.gguf',
                                     'Qwen3.8-27B-Q4_K_S.gguf'])
def test_other_plain_controls_keep_admission_identity_not_uncertified_named_profile(control):
    from hipengine.loading.qwen35_gguf_admission import qwen35_gguf_artifact_identity_from_info
    context = profile_context(control=control)
    assert qwen35_gguf_artifact_identity_from_info(context['weight_index'])[1] is None
    before = dict(os.environ)
    with pytest.raises(ValueError, match='qualification does not cover'):
        resolve(qualification_context=context)
    assert dict(os.environ) == before


@pytest.mark.parametrize('quant', ['auto', 'gguf_q4_k_m'])
def test_actual_llm_pinned_plain_positive(quant, monkeypatch):
    import hipengine.generation as generation
    context = profile_context()
    calls = []
    class Generator(SimpleNamespace):
        def generate(self, request):
            return ['ok' for _ in request.prompts]
    def factory(**kw):
        calls.append(kw['weight_index'])
        return Generator(**kw)
    monkeypatch.setattr(generation, 'resolve_text_generator', lambda **_: factory)
    llm = LLM(context['model_path'], backend='hip_gfx1151', quant=quant, execution_profile='production')
    assert llm.generate(['hello']) == ['ok']
    assert len(calls) == 1
    assert llm.execution_profile_fell_back_to_strict is False
    assert os.environ['HIPENGINE_GGUF_FP16_RECURRENT_STATE'] == '1'
    llm.close()


def test_failed_actual_binder_restores_environment(monkeypatch):
    from hipengine.generation import qwen38_gguf_profiles as plugin
    original = plugin._binder
    def failing(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('binder application failed')
    monkeypatch.setattr(plugin, '_binder', failing)
    context = profile_context()
    resolved = resolve(qualification_context=context)
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match='binder application failed'):
        resolved.construct_generator(lambda **kw: SimpleNamespace(**kw), **context)
    assert dict(os.environ) == before


def test_profile_binding_preserves_outer_environment_context():
    from hipengine.generation.qwen35_gguf import _temporary_env
    env = 'HIPENGINE_GGUF_FP16_RECURRENT_STATE'
    context = profile_context()
    before = dict(os.environ)
    # Use all binder-owned keys in the existing caller's restore context.
    updates = {env: '0', 'HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN': '0',
               'HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE': '0',
               'HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256': ''}
    with _temporary_env(updates):
        resolve(qualification_context=context).construct_generator(lambda **kw: SimpleNamespace(**kw), **context)
        assert os.environ[env] == '1'
    assert dict(os.environ) == before


@pytest.mark.parametrize('value', ['0', '1'])
def test_explicit_fp16_override_is_not_profile_qualification(unknown, monkeypatch, value):
    from hipengine.runtime.qwen35_gguf_runner import _gguf_fp16_recurrent_state_enabled
    monkeypatch.setenv('HIPENGINE_GGUF_FP16_RECURRENT_STATE', value)
    assert _gguf_fp16_recurrent_state_enabled(
        backend='hip_gfx1151', file_type_name=unknown.file_type_name,
        artifact_preset_key=GGUF_UNQUALIFIED_MANIFEST_PRESET,
    ) == (value == '1')  # explicit diagnostic route, not named-profile consent
    before = dict(os.environ)
    with pytest.raises(ValueError, match='qualification refused'):
        resolve(qualification_context={'weight_index': unknown})
    assert dict(os.environ) == before


@pytest.mark.parametrize('context', [None, {}, {'weight_index': None},
                                   {'weight_index': SimpleNamespace(artifact_preset_key=None)}])
def test_direct_resolution_missing_qualification_fails_closed(context):
    before = dict(os.environ)
    with pytest.raises(ValueError, match='qualification'):
        resolve(qualification_context=context)
    assert dict(os.environ) == before


def test_public_binder_rejects_another_resolved_plan_before_mutation():
    context = profile_context()
    production = resolve(qualification_context=context)
    strict = resolve(profile='strict', qualification_context=context)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='qualification belongs to another plan'):
        production.binder(SimpleNamespace(**context), strict)
    assert dict(os.environ) == before
