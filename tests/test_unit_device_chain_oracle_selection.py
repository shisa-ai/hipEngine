"""Device-chain oracle selection belongs to a diagnostic adapter instance."""

from types import SimpleNamespace

from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter


def test_explicit_oracle_delegates_to_qualification_adapter(monkeypatch):
    from hipengine.benchmark import device_chain_oracle
    received = []
    def qualify(*args, **kwargs):
        received.append((args, kwargs))
        return 0.25
    monkeypatch.setattr(device_chain_oracle, 'qualify_device_accept', qualify)
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = object()
    result = adapter._qualify_target_batch_device_accept(
        'frontier', 'proposal', ['target'], 'summary', [3],
        provider='provider', runtime='runtime',
    )
    assert result == 0.25
    assert received[0][0] == (adapter.owner, 'frontier', 'proposal', ['target'], 'summary', [3])
    assert received[0][1]['provider'] == 'provider'
    assert callable(received[0][1]['trace_rows'])


def test_device_chain_oracle_is_explicit_and_instance_local(monkeypatch):
    monkeypatch.setenv("HIPENGINE_SPECDEC2_DEVICE_CHAIN_ORACLE", "1")
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100", execution_profile="production"),
        capacity=2, _shared_runner=SimpleNamespace(hidden_size=4),
    )
    args = dict(enabled=True, target_verify_mode="native", candidate_budget=2)
    default = Qwen35GGUFMTP2Adapter(owner, **args)
    diagnostic = None
    try:
        assert default.device_chain_qualification_oracle is False
        diagnostic = Qwen35GGUFMTP2Adapter(owner, device_chain_qualification_oracle=True, **args)
        assert diagnostic.device_chain_qualification_oracle is True
        assert default.device_chain_qualification_oracle is False
    finally:
        default.close()
        if diagnostic is not None:
            diagnostic.close()
