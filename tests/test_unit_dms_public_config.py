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
    (dict(kv_storage='unsupported_codec'), 'compact storage'),
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


@pytest.mark.parametrize('storage', ['bf16', 'int8_per_token_head'])
def test_public_dms_preserves_requested_codec_and_layer_outer_route(monkeypatch, storage):
    from hipengine.server.api import ServerConfig
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    from hipengine.generation import qwen35_gguf_dms
    from types import SimpleNamespace

    cfg = DMSConfig('sidecar.json', prefill_mode='layer_outer')
    llm = LLM('unused.gguf', dms=cfg, kv_storage=storage)
    assert llm.kv_storage == storage
    server = ServerConfig(model='unused.gguf', dms=cfg, kv_storage=storage,
                          prefix_cache='off', speculative_mtp_serving='off')
    assert server.kv_storage == storage
    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator.configure_dms(cfg)
    assert generator.dms_metadata_path == cfg.metadata_path
    generator._prepared_kv_signature = (storage, 'uniform', 'fp32', 'per_token_head')
    kwargs = generator._prepared_session_kv_kwargs()
    assert kwargs['dms_backend_factory'].__name__ == ('create_dms_int8_backend' if storage == 'int8_per_token_head' else 'create_dms_bf16_backend')
    captured = []
    from hipengine.generation import qwen35_gguf
    monkeypatch.setattr(qwen35_gguf, 'Qwen35GGUFResidentSession',
                        lambda *args, **kw: captured.append(kw) or SimpleNamespace(max_sequence_length=kw.get('max_sequence_length')))
    monkeypatch.setattr(generator, '_configure_session', lambda session: None, raising=False)
    runner = object.__new__(qwen35_gguf_dms.Qwen35GGUFDMSModelRunner)
    runner.generator = generator
    runner._shared_runner = SimpleNamespace(runtime=None)
    runner._max_sequence_length = 262144
    runner._available = []
    generator.model_path = 'unused.gguf'
    generator.backend = 'hip_gfx1100'
    runner._reserve_sessions()
    assert captured[0]['dms_prefill_mode'] == 'layer_outer'
    assert captured[0]['dms_backend_factory'] is kwargs['dms_backend_factory']
    assert captured[0]['max_sequence_length'] == 262144
    monkeypatch.setattr(qwen35_gguf_dms.Qwen35GGUFResidentModelRunner,
                        '_execution_metadata', lambda self, row: {})
    retention = runner._execution_metadata(SimpleNamespace())['retention']
    assert retention['storage'] == storage
    assert retention['prefill_mode'] == 'layer_outer'


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
