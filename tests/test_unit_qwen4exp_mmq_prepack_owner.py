"""MMQ preparation and session entry share the explicit owner's prepack choice."""
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK'


@pytest.mark.parametrize('raw,expected', [('1', True), ('0', False), ('true', False), ('', False)])
def test_unbound_parser_and_explicit_precedence(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV, raw)
    assert runtime.resolve_q8_mmq_prepack(SimpleNamespace()) is expected
    assert runtime.resolve_q8_mmq_prepack(SimpleNamespace(q8_mmq_prepack_enabled=True)) is True
    assert runtime.resolve_q8_mmq_prepack(SimpleNamespace(q8_mmq_prepack_enabled=False)) is False


def test_scope_nested_exception_and_clone(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(q8_mmq_prepack_enabled=None, runner=SimpleNamespace(q8_mmq_prepack_enabled=None))
    with pytest.raises(RuntimeError):
        with runtime.q8_mmq_prepack_enabled(owner, False):
            assert owner.runner.q8_mmq_prepack_enabled is False
            clone = _inherit_lane_decisions(owner, SimpleNamespace())
            assert clone.q8_mmq_prepack_enabled is False
            with runtime.q8_mmq_prepack_enabled(owner, True):
                assert runtime.resolve_q8_mmq_prepack(owner) is True
            assert runtime.resolve_q8_mmq_prepack(owner) is False
            raise RuntimeError('injected')
    assert owner.q8_mmq_prepack_enabled is owner.runner.q8_mmq_prepack_enabled is None


@pytest.mark.parametrize('enabled', [False, True])
def test_session_uses_owner_not_opposite_environment(monkeypatch, enabled):
    monkeypatch.setenv(ENV, '0' if enabled else '1')
    buffers = tuple(SimpleNamespace(ptr=i, nbytes=100) for i in (1, 2, 3))
    mapping = {(100, 2560, 10240): (500, 200)}
    calls = []
    monkeypatch.setattr(runtime, 'q8_mmq_prefill_session', lambda **kw: calls.append(kw))
    monkeypatch.setattr(runtime, '_qwen4_exp_q8_mmq_policy', lambda policy: policy)
    owner = SimpleNamespace(q8_mmq_prefill_enabled=True, q8_mmq_prepack_enabled=enabled,
        _q8_mmq_buffers=buffers, _q8_mmq_weight_sidecars=SimpleNamespace(mapping=mapping),
        _q8_mmq_policy=object(), _q8_mmq_library=object(),
        dense_mmq_token64_enabled=False, dense_mmq_raw_vector_enabled=False,
        dense_mmq_vec4_enabled=False, _q8_mmq_scopes=lambda *a, **kw: None)
    runtime.Qwen4ExpGGUFResidentModelRunner._q8_mmq_prefill_context(owner)
    assert calls[0]['prepacked_weights'] == (mapping if enabled else None)


def test_constructor_and_late_runner_carry_the_choice(monkeypatch):
    from hipengine.generation import qwen4_exp_gguf as generation
    calls = []
    def make(resident, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)
    monkeypatch.setattr(generation, 'Qwen4ExpGGUFResidentModelRunner', make)
    owner = generation.Qwen4ExpGGUFTextGenerator(
        model_path='unused.gguf', weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(max_sequence_length=8, prefill_chunk_size=4, close=lambda: None))
    owner.q8_mmq_prepack_enabled = True
    try:
        clone = owner._new_resident_runner(SimpleNamespace(), max_sequence_length=8,
                                           prefill_chunk_size=4)
    finally:
        owner.close()
    assert calls[0]['q8_mmq_prepack_enabled'] is clone.q8_mmq_prepack_enabled is True


def test_prepack_requires_prepared_sidecars(monkeypatch):
    monkeypatch.setenv(ENV, '0')
    owner = SimpleNamespace(q8_mmq_prefill_enabled=True, q8_mmq_prepack_enabled=True,
        _q8_mmq_buffers=(1,2,3), _q8_mmq_weight_sidecars=None)
    with pytest.raises(RuntimeError, match='prepared sidecar ownership'):
        runtime.Qwen4ExpGGUFResidentModelRunner._q8_mmq_prefill_context(owner)
