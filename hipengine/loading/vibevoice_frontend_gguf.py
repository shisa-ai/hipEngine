"""BF16 audio frontend loading from hipEngine's standalone VibeVoice GGUF."""
from hipengine.loading.gguf import GGUFReader, MissingGGUFTensorError
from hipengine.loading.safetensors import MissingTensorError
from hipengine.loading.vibevoice_asr import (
    hf_frontend_tensor_name, _bf16_bytes_to_f32,
    _load_encoder_from_index, _load_connector_from_index,
)


class GGUFFrontendIndex:
    def __init__(self, path):
        self.reader = GGUFReader(path)

    @staticmethod
    def _name(name):
        name = hf_frontend_tensor_name(name)
        for src, dst in (('acoustic_tokenizer_encoder.', 'ate.'),
                         ('semantic_tokenizer_encoder.', 'ste.'),
                         ('multi_modal_projector.', 'mmp.')):
            if name.startswith(src):
                return dst + name[len(src):]
        raise ValueError(f'unknown frontend tensor: {name}')

    def require(self, names):
        try:
            infos = tuple(self.reader.tensor_info(self._name(n)) for n in names)
        except MissingGGUFTensorError as exc:
            raise MissingTensorError(str(exc)) from exc
        if any(t.ggml_type != 30 for t in infos):
            raise ValueError('standalone frontend tensors must be BF16')
        return infos

    def read(self, name):
        info = self.require((name,))[0]
        data = self.reader.tensor_data(info.name)
        return _bf16_bytes_to_f32(data.tobytes(), info.shape)


def load_gguf_frontend(path):
    index = GGUFFrontendIndex(path)
    specs = {k: _load_encoder_from_index(index, path, k) for k in ('acoustic', 'semantic')}
    connectors = {k: _load_connector_from_index(index, path, k) for k in specs}
    return specs, connectors
