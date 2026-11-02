"""The F32-only MMQ plane diagnostic can bind one generating owner."""
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import Q8MMQPrefillPolicy

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES'


@pytest.mark.parametrize('planes', [2, 3])
def test_explicit_planes_override_ambient_without_mutating_base(monkeypatch, planes):
    monkeypatch.setenv(ENV, str(5 - planes))
    base = Q8MMQPrefillPolicy({(2560, 10240): 64}, 2048, 0.0, 12288)
    chosen = runtime._qwen4_exp_q8_mmq_policy(base, f32_planes=planes)
    assert chosen.f32_planes == planes
    assert base.f32_planes == 3


@pytest.mark.parametrize('planes', [0, 1, 4, '2', 2.5])
def test_invalid_explicit_planes_raise(planes):
    with pytest.raises(ValueError, match='f32_planes'):
        with runtime.q8_mmq_f32_planes(SimpleNamespace(), planes):
            pass


def test_scope_restores_absence_nested_exception_and_clone():
    owner = SimpleNamespace(runner=SimpleNamespace())
    with pytest.raises(RuntimeError):
        with runtime.q8_mmq_f32_planes(owner, 2):
            assert owner.runner.q8_mmq_f32_planes == 2
            clone = _inherit_lane_decisions(owner, SimpleNamespace())
            assert clone.q8_mmq_f32_planes == 2
            with runtime.q8_mmq_f32_planes(owner, 3):
                assert owner.runner.q8_mmq_f32_planes == 3
            assert owner.runner.q8_mmq_f32_planes == 2
            raise RuntimeError('injected')
    assert not hasattr(owner, 'q8_mmq_f32_planes')
    assert not hasattr(owner.runner, 'q8_mmq_f32_planes')


@pytest.mark.parametrize('planes', [2, 3])
def test_session_policy_uses_explicit_owner(monkeypatch, planes):
    monkeypatch.setenv(ENV, str(5 - planes))
    calls = []
    monkeypatch.setattr(runtime, 'q8_mmq_prefill_session', lambda **kw: calls.append(kw))
    buffers = tuple(SimpleNamespace(ptr=i, nbytes=100) for i in (1, 2, 3))
    owner = SimpleNamespace(q8_mmq_prefill_enabled=True, q8_mmq_prepack_enabled=False,
        q8_mmq_f32_planes=planes, _q8_mmq_buffers=buffers,
        _q8_mmq_policy=Q8MMQPrefillPolicy({(2560, 10240): 64}, 2048, 0.0, 12288),
        _q8_mmq_library=object(), dense_mmq_token64_enabled=False,
        dense_mmq_raw_vector_enabled=False, dense_mmq_vec4_enabled=False,
        _q8_mmq_scopes=lambda *a, **kw: None)
    runtime.Qwen4ExpGGUFResidentModelRunner._q8_mmq_prefill_context(owner)
    assert calls[0]['policy'].f32_planes == planes


def test_late_constructor_carries_plane_choice(monkeypatch):
    from hipengine.generation import qwen4_exp_gguf as generation
    calls = []
    monkeypatch.setattr(generation, 'Qwen4ExpGGUFResidentModelRunner',
        lambda resident, **kw: calls.append(kw) or SimpleNamespace(**kw))
    owner = generation.Qwen4ExpGGUFTextGenerator(model_path='unused.gguf',
        weight_index=SimpleNamespace(), model_plugin=SimpleNamespace(), tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(max_sequence_length=8, prefill_chunk_size=4, close=lambda: None))
    assert owner.q8_mmq_f32_planes is None
    owner.q8_mmq_f32_planes = 2
    try:
        clone = owner._new_resident_runner(SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        owner.close()
    assert calls[0]['q8_mmq_f32_planes'] == clone.q8_mmq_f32_planes == 2
