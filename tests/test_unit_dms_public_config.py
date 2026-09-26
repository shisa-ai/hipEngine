from pathlib import Path
import pytest
from hipengine import DMSConfig, LLM


def test_dms_config_normalizes_path_and_sets_compatible_defaults():
    cfg = DMSConfig(Path('sidecar.json'))
    llm = LLM('not-loaded.gguf', dms=cfg)
    assert cfg.metadata_path == 'sidecar.json'
    assert llm.max_active_requests == 1
    assert llm.kv_storage == 'bf16'
    assert llm.prefix_cache == llm.speculative_mtp_serving == 'off'


@pytest.mark.parametrize('kwargs', [dict(metadata_path=''),
    dict(metadata_path='x', prefill_mode='unknown'),
    dict(metadata_path='x', decision_mode='diagnostic')])
def test_dms_config_rejects_invalid_fields(kwargs):
    with pytest.raises(ValueError):
        DMSConfig(**kwargs)


@pytest.mark.parametrize('kwargs,reason', [
    (dict(max_active_requests=2), 'max_active_requests'),
    (dict(kv_storage='int8_per_token_head'), 'BF16'),
    (dict(prefix_cache='radix'), 'prefix-cache'),
    (dict(speculative_mtp_serving='enabled'), 'speculative'),
])
def test_dms_rejects_unsupported_compositions_before_loading(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        LLM('not-loaded.gguf', dms=DMSConfig('sidecar.json'), **kwargs)


def test_server_requires_explicit_unsupported_compositions_to_be_disabled():
    from hipengine.server.api import ServerConfig
    with pytest.raises(ValueError, match='prefix-cache'):
        ServerConfig(model='unused', dms=DMSConfig('sidecar.json'))
    config = ServerConfig(model='unused', dms=DMSConfig('sidecar.json'),
                          prefix_cache='off', speculative_mtp_serving='off')
    assert config.max_active_requests == 1
    assert config.kv_storage == 'bf16'


def test_dms_generator_factory_selects_adapter_before_loading(monkeypatch):
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    from hipengine.generation import qwen35_gguf_dms
    from types import SimpleNamespace
    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator.configure_dms(DMSConfig('sidecar.json'))
    monkeypatch.setattr(qwen35_gguf_dms, 'Qwen35GGUFDMSModelRunner',
                        lambda owner, capacity: SimpleNamespace(owner=owner, capacity=capacity))
    runner = generator.create_resident_model_runner()
    assert runner.capacity == 1 and runner.owner is generator
    assert not generator.supports_speculative_mtp
    with pytest.raises(ValueError, match='max_active_requests'):
        generator.create_resident_model_runner(capacity=4)
