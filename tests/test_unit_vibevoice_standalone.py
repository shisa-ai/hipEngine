"""Standalone assets fail closed and preserve source bytes without HF access."""
import numpy as np
import pytest


def test_embedded_assets_roundtrip_and_tamper(tmp_path):
    from hipengine.loading.vibevoice_assets import asset_metadata, read_assets
    files = {'config.json': '{"model_type":"vibevoice_asr"}',
             'tokenizer.json': '{"version":"1.0"}',
             'tokenizer_config.json': '{}', 'processor_config.json': '{}',
             'generation_config.json': '{}', 'chat_template.jinja': '{{ messages }}'}
    for name, value in files.items():
        (tmp_path / name).write_text(value)
    metadata = asset_metadata(tmp_path)
    assert read_assets(metadata) == files
    metadata['vibevoice.assets.config.json'] += ' '
    with pytest.raises(ValueError, match='hash'):
        read_assets(metadata)


def test_missing_assets_rejected(tmp_path):
    from hipengine.loading.vibevoice_assets import asset_metadata, read_assets
    with pytest.raises(ValueError, match='missing'):
        asset_metadata(tmp_path)
    with pytest.raises(ValueError, match='standalone'):
        read_assets({})


def test_gguf_frontend_reads_bf16_and_reports_missing(tmp_path):
    gguf = pytest.importorskip('gguf')
    from hipengine.loading.vibevoice_frontend_gguf import GGUFFrontendIndex
    p = tmp_path / 'tiny.gguf'
    w = gguf.GGUFWriter(str(p), 'vibevoice-asr')
    name = 'ate.stem.conv.conv.weight'
    bits = np.array([0x3f80, 0xc000], dtype=np.uint16).reshape(2,1,1)
    w.add_tensor(name, bits, raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    i = GGUFFrontendIndex(p)
    original = 'model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight'
    assert i.require((original,))[0].shape == (2,1,1)
    np.testing.assert_array_equal(i.read(original).reshape(-1), [1.,-2.])
    from hipengine.loading.safetensors import MissingTensorError
    with pytest.raises(MissingTensorError):
        i.require(('model.acoustic_connector.fc1.weight',))


def test_repackage_preserves_tensor_bits_and_refreshes_assets(tmp_path):
    gguf = pytest.importorskip('gguf')
    from scripts.vibevoice_asr_to_gguf import repackage
    from hipengine.loading.gguf import GGUFReader, scan_gguf
    from hipengine.loading.vibevoice_assets import ASSET_FILES, read_assets
    source = tmp_path / 'source.gguf'
    destination = tmp_path / 'standalone.gguf'
    for name in ASSET_FILES:
        (tmp_path / name).write_text('{}' if name.endswith('.json') else 'template')
    w = gguf.GGUFWriter(str(source), 'vibevoice-asr')
    bits = np.array([[0x3f80,0xc000]], dtype=np.uint16)
    w.add_tensor('ate.example', bits, raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    repackage(source, destination, tmp_path)
    np.testing.assert_array_equal(GGUFReader(destination).tensor_data('ate.example'), bits)
    assert read_assets(scan_gguf(destination).metadata)['chat_template.jinja'] == 'template'
    with pytest.raises(ValueError, match='new file'):
        repackage(source, source, tmp_path)


def test_gguf_frontend_rejects_wrong_precision(tmp_path):
    gguf = pytest.importorskip('gguf')
    from hipengine.loading.vibevoice_frontend_gguf import GGUFFrontendIndex
    p = tmp_path / 'f16.gguf'
    w = gguf.GGUFWriter(str(p), 'vibevoice-asr')
    w.add_tensor('mmp.acoustic_linear_1.weight', np.ones((2,2),dtype=np.float16))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    with pytest.raises(ValueError, match='BF16'):
        GGUFFrontendIndex(p).read('model.acoustic_connector.fc1.weight')


def test_hf_backbone_tensor_reader_still_accepts_weight_index(monkeypatch):
    from types import SimpleNamespace
    from hipengine.loading import vibevoice_asr as loader
    info = SimpleNamespace(dtype='BF16', shape=(2,))
    index = SimpleNamespace(require=lambda names: (info,))
    monkeypatch.setattr(loader, 'read_tensor_storage_bytes',
                        lambda t: np.array([0x3f80,0xc000],dtype=np.uint16).tobytes())
    np.testing.assert_array_equal(loader._load_tensor(index,'language_model.example',None),[1.,-2.])


def test_fresh_export_embeds_assets(tmp_path, monkeypatch):
    pytest.importorskip('gguf')
    from scripts import vibevoice_asr_to_gguf as exporter
    from hipengine.loading.vibevoice_assets import ASSET_FILES, read_assets
    from hipengine.loading.gguf import scan_gguf
    import sys
    for name in ASSET_FILES:
        (tmp_path / name).write_text('{}' if name.endswith('.json') else 'template')
    out = tmp_path / 'export.gguf'
    monkeypatch.setattr(exporter,'resolve_snapshot',lambda model:tmp_path)
    monkeypatch.setattr(exporter,'add_metadata',lambda writer, config:None)
    monkeypatch.setattr(exporter,'iter_tensors',lambda snapshot, dtype:iter([
        ('acoustic_tokenizer_encoder.example',np.array([0x3f80],dtype=np.uint16))]))
    monkeypatch.setattr(sys,'argv',['export','--out',str(out)])
    assert exporter.main() == 0
    assert read_assets(scan_gguf(out).metadata)['tokenizer.json'] == '{}'
