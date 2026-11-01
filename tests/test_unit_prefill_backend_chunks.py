"""Backend-owned fallback chunks are shared without borrowing GGUF geometry."""
import pytest

from hipengine.runtime.prefill import PrefillConfig, resolve_prefill_config_for_sequence


@pytest.mark.parametrize('length', [37, 255, 256, 257, 1537])
def test_backend_fallback_preserves_sizes_and_record(length):
    config, record = resolve_prefill_config_for_sequence(
        PrefillConfig(), max_sequence_length=length, target_arch='gfx1151',
        backend='hip_gfx1151')
    assert (config.linear_chunk_size, config.moe_chunk_size) == (256, 256)
    assert record['arch_profile_applied'] is True
    assert record['geometry_profile_applied'] is False
    assert record['chunk_sizes']['linear'] == config.linear_chunk_size
    assert record['chunk_sizes']['moe'] == config.moe_chunk_size


def test_backend_override_is_live_and_geometry_wins(monkeypatch):
    from hipengine.kernels import hip_gfx1151
    monkeypatch.setattr(hip_gfx1151, 'PREFILL_CHUNK_SIZES', (320, 384), raising=False)
    base, record = resolve_prefill_config_for_sequence(
        PrefillConfig(), max_sequence_length=1537, target_arch='gfx1151',
        backend='hip_gfx1151')
    assert (base.linear_chunk_size, base.moe_chunk_size) == (320, 384)
    specific, _ = resolve_prefill_config_for_sequence(
        PrefillConfig(), max_sequence_length=1537, target_arch='gfx1151',
        backend='hip_gfx1151', geometry_chunk_sizes=(512, 640))
    assert (specific.linear_chunk_size, specific.moe_chunk_size) == (512, 640)
    monkeypatch.setattr(hip_gfx1151, 'PREFILL_CHUNK_SIZES', None)
    absent, record = resolve_prefill_config_for_sequence(
        PrefillConfig(), max_sequence_length=1537, target_arch='gfx1151',
        backend='hip_gfx1151')
    assert (absent.linear_chunk_size, absent.moe_chunk_size) == (1024, 1024)
    assert record['arch_profile_applied'] is False


@pytest.mark.parametrize('config', [PrefillConfig(linear_chunk_size=73),
                                  PrefillConfig(auto_tune_chunk_sizes=False)])
def test_manual_config_does_not_read_backend_policy(monkeypatch, config):
    from hipengine.kernels import backends
    def unexpected(*args, **kwargs):
        raise AssertionError('manual config consulted automatic backend policy')
    monkeypatch.setattr(backends, 'backend_package_capability', unexpected)
    actual, record = resolve_prefill_config_for_sequence(
        config, max_sequence_length=1537, backend='hip_gfx1151', target_arch='gfx1151')
    assert actual is config
    assert record['arch_profile_applied'] is False


@pytest.mark.parametrize('backend', ['cpu_reference', 'auto', 'unknown_backend'])
def test_nonhardware_backend_keeps_shape_policy(backend):
    config, report = resolve_prefill_config_for_sequence(
        PrefillConfig(), max_sequence_length=1537, backend=backend)
    assert (config.linear_chunk_size, config.moe_chunk_size) == (1024, 1024)
    assert report['arch_profile_applied'] is False


def test_unknown_arch_without_backend_keeps_shape_policy():
    config, record = resolve_prefill_config_for_sequence(
        PrefillConfig(), max_sequence_length=1537, target_arch='gfx9999')
    assert (config.linear_chunk_size, config.moe_chunk_size) == (1024, 1024)
    assert record['arch_profile_applied'] is False
