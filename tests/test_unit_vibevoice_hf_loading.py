"""Namespace mapping must select one complete artifact for ASR."""
import pytest

from hipengine.loading.vibevoice_asr import hf_frontend_tensor_name


@pytest.mark.parametrize('source,target', [
    ('model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight',
     'acoustic_tokenizer_encoder.stem.conv.conv.weight'),
    ('model.semantic_tokenizer.encoder.stages.6.7.mixer.conv.conv.conv.bias',
     'semantic_tokenizer_encoder.conv_layers.5.stage.7.mixer.conv.bias'),
    ('model.acoustic_tokenizer.encoder.head.conv.conv.weight',
     'acoustic_tokenizer_encoder.head.conv.weight'),
    ('model.semantic_connector.fc1.weight', 'multi_modal_projector.semantic_linear_1.weight'),
    ('model.acoustic_connector.norm.weight', 'multi_modal_projector.acoustic_norm.weight'),
])
def test_frontend_namespace(source, target):
    assert hf_frontend_tensor_name(source) == target
