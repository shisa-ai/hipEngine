"""Raw-prefill geometry is carried by the executing owner, not its peers."""
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import gguf_linear as linear
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions


@pytest.mark.parametrize('rowbatch,variant', [(0, 'rowbatch'), (8, 'coltile16'), (32, 'coltile8')])
def test_explicit_policy_overrides_ambient_and_drives_chunk(monkeypatch, rowbatch, variant):
    policy = runtime.RawPrefillPolicy(rowbatch, variant)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_RAW_ROWBATCH', 'invalid')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_RAW_VARIANT', 'invalid')
    monkeypatch.setattr(runtime, '_qwen4_exp_q8_tile_scope', lambda _: nullcontext())
    seen = []
    def chunk(tokens, **kwargs):
        seen.append((linear.raw_k_prefill_rowbatch(), linear.raw_k_prefill_variant()))
        raise RuntimeError('observed')
    owner = SimpleNamespace(raw_prefill_policy=policy, q8_wave_scale_prefill=False,
        max_sequence_length=4096, prefill_chunk_size=1024, position=0, reset=lambda: None,
        _prefill_chunk=chunk, _q8_mmq_prefill_context=nullcontext,
        _wide_f16_activation_context=nullcontext)
    with pytest.raises(RuntimeError, match='observed'):
        next(runtime.Qwen4ExpGGUFResidentModelRunner._prefill_chunked_steps(owner, [3] * 73))
    assert seen == [(rowbatch, variant)]


@pytest.mark.parametrize('rowbatch,variant', [(1, 'coltile8'), (True, 'coltile8'), (32, 'bad')])
def test_invalid_policy_rejected(rowbatch, variant):
    with pytest.raises(ValueError):
        runtime.RawPrefillPolicy(rowbatch, variant)


def test_scope_isolates_restores_missing_nested_failure_and_late_runner():
    owner = SimpleNamespace(runner=SimpleNamespace())
    peer = SimpleNamespace()
    policy = runtime.RawPrefillPolicy(8, 'coltile16')
    with pytest.raises(RuntimeError):
        with runtime.raw_prefill_policy(owner, policy):
            assert owner.raw_prefill_policy is owner.runner.raw_prefill_policy is policy
            assert not hasattr(peer, 'raw_prefill_policy')
            assert _inherit_lane_decisions(owner, SimpleNamespace()).raw_prefill_policy is policy
            with runtime.raw_prefill_policy(owner, runtime.RawPrefillPolicy(0, 'rowbatch')):
                assert owner.raw_prefill_policy.rowbatch == 0
            assert owner.raw_prefill_policy is policy
            raise RuntimeError('injected')
    assert not hasattr(owner, 'raw_prefill_policy')
    assert not hasattr(owner.runner, 'raw_prefill_policy')


def test_late_constructor_carries_policy(monkeypatch):
    from hipengine.generation import qwen4_exp_gguf as generation
    calls = []
    monkeypatch.setattr(generation, 'Qwen4ExpGGUFResidentModelRunner',
        lambda resident, **kw: calls.append(kw) or SimpleNamespace(**kw))
    owner = generation.Qwen4ExpGGUFTextGenerator(model_path='unused.gguf',
        weight_index=SimpleNamespace(), model_plugin=SimpleNamespace(), tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(max_sequence_length=8, prefill_chunk_size=4, close=lambda: None))
    policy = runtime.RawPrefillPolicy(8, 'coltile16')
    owner.raw_prefill_policy = policy
    try:
        clone = owner._new_resident_runner(SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        owner.close()
    assert calls[0]['raw_prefill_policy'] is clone.raw_prefill_policy is policy
