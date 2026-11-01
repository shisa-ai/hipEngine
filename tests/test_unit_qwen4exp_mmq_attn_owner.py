"""MMQ attention-gate policy belongs to the generating owner."""
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import Q8MMQPrefillPolicy

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE'


def policy():
    return Q8MMQPrefillPolicy(min_rows={(2560, 10240): 64}, max_rows=2048,
                             risk_threshold=0.0, max_out_features=12288)


@pytest.mark.parametrize('raw,expected', [('1', True), ('true', True), ('False', False), ('0', False), ('', False)])
def test_owner_precedence_and_unbound_legacy_parser(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV, raw)
    base = policy()
    assert runtime._qwen4_exp_q8_mmq_policy(base)(73, 2560, 6144) is expected
    for enabled in (False, True):
        chosen = runtime._qwen4_exp_q8_mmq_policy(base, attn_gate_enabled=enabled)
        assert chosen(73, 2560, 6144) is enabled
        assert not chosen(63, 2560, 6144)
    assert base.min_rows == {(2560, 10240): 64}


def test_scope_restores_absence_nested_exception_and_clone(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    runner = SimpleNamespace()
    owner = SimpleNamespace(runner=runner)
    with pytest.raises(RuntimeError):
        with runtime.q8_mmq_attn_gate_enabled(owner, False):
            assert runner.q8_mmq_attn_gate_enabled is False
            clone = _inherit_lane_decisions(owner, SimpleNamespace())
            assert clone.q8_mmq_attn_gate_enabled is False
            with runtime.q8_mmq_attn_gate_enabled(owner, True):
                assert runner.q8_mmq_attn_gate_enabled is True
            assert runner.q8_mmq_attn_gate_enabled is False
            raise RuntimeError('injected')
    assert not hasattr(owner, 'q8_mmq_attn_gate_enabled')
    assert not hasattr(runner, 'q8_mmq_attn_gate_enabled')


@pytest.mark.parametrize('enabled', [False, True])
def test_session_policy_obeys_owner(monkeypatch, enabled):
    monkeypatch.setenv(ENV, '0' if enabled else '1')
    calls = []
    monkeypatch.setattr(runtime, 'q8_mmq_prefill_session', lambda **kw: calls.append(kw))
    buffers = tuple(SimpleNamespace(ptr=i, nbytes=100) for i in (1, 2, 3))
    owner = SimpleNamespace(q8_mmq_prefill_enabled=True, q8_mmq_prepack_enabled=False,
        q8_mmq_attn_gate_enabled=enabled, _q8_mmq_buffers=buffers, _q8_mmq_policy=policy(),
        _q8_mmq_library=object(), dense_mmq_token64_enabled=False,
        dense_mmq_raw_vector_enabled=False, dense_mmq_vec4_enabled=False,
        _q8_mmq_scopes=lambda *a, **kw: None)
    runtime.Qwen4ExpGGUFResidentModelRunner._q8_mmq_prefill_context(owner)
    assert calls[0]['policy'](73, 2560, 6144) is enabled


def test_late_runner_constructor_receives_owner(monkeypatch):
    from hipengine.generation import qwen4_exp_gguf as generation
    calls = []
    monkeypatch.setattr(generation, 'Qwen4ExpGGUFResidentModelRunner',
        lambda resident, **kw: calls.append(kw) or SimpleNamespace(**kw))
    owner = generation.Qwen4ExpGGUFTextGenerator(model_path='unused.gguf',
        weight_index=SimpleNamespace(), model_plugin=SimpleNamespace(), tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(max_sequence_length=8, prefill_chunk_size=4, close=lambda: None))
    assert owner.q8_mmq_attn_gate_enabled is None
    owner.q8_mmq_attn_gate_enabled = True
    try:
        clone = owner._new_resident_runner(SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        owner.close()
    assert calls[0]['q8_mmq_attn_gate_enabled'] is clone.q8_mmq_attn_gate_enabled is True
