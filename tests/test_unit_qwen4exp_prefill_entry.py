"""The public reset-prefill entry owns the chunked implementation."""
from types import SimpleNamespace

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner


def test_public_prefill_forwards_every_option_to_chunked_implementation():
    calls = []
    result = object()
    runner = SimpleNamespace(_prefill_chunked_impl=lambda tokens, **kw: (
        calls.append((tokens, kw)), result)[1])
    tokens = [1, 2, 3]
    overrides, positions, consumer = object(), object(), object()
    actual = Qwen4ExpGGUFResidentModelRunner.prefill(
        runner, tokens, capture_hidden_seeds=True, capture_logits=False,
        capture_target_hidden=False, embedding_overrides=overrides,
        mrope_positions=positions, hidden_seed_consumer=consumer)
    assert actual is result
    assert calls == [(tokens, dict(
        capture_hidden_seeds=True, capture_logits=False,
        capture_target_hidden=False, embedding_overrides=overrides,
        mrope_positions=positions, hidden_seed_consumer=consumer))]
    assert not hasattr(Qwen4ExpGGUFResidentModelRunner, 'prefill_chunked')
    assert callable(Qwen4ExpGGUFResidentModelRunner.prefill_serial)
