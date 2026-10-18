"""Single-file Q4 ASR session using the existing evaluated Q4 kernels.

This packaging adapter does not certify a new production arithmetic profile.
"""
import json
import math
from threading import RLock
from hipengine.generation.vibevoice_asr import VibeVoiceASRGenerator
from hipengine.generation.vibevoice_protocol import AUDIO_TOKEN, AUDIO_BOS, AUDIO_EOS, AUDIO_TOKEN_ID, IM_END_ID


class VibeVoiceASRQ4Generator(VibeVoiceASRGenerator):
    def __init__(self, model_path, *, max_sequence_length=4096, backend='auto'):
        from tokenizers import Tokenizer
        from hipengine.loading.gguf import scan_gguf
        from hipengine.loading.vibevoice_assets import read_assets
        from hipengine.loading.vibevoice_frontend_gguf import load_gguf_frontend
        from hipengine.loading.vibevoice_asr_gguf import load_vibevoice_qwen2_q4
        from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
        from hipengine.runtime.vibevoice_qwen2_q4 import VibevoiceQwen2Q4Runtime
        self._lock = RLock()
        self._closed = False
        self.max_context = max_sequence_length
        info = scan_gguf(model_path)
        if info.architecture != 'vibevoice-asr':
            raise ValueError('expected hipEngine vibevoice-asr GGUF')
        assets = read_assets(info.metadata)
        config = json.loads(assets['config.json'])
        processor = json.loads(assets['processor_config.json'])
        self.tokenizer = Tokenizer.from_str(assets['tokenizer.json'])
        for token, expected in ((AUDIO_TOKEN, AUDIO_TOKEN_ID), (AUDIO_BOS,151646),
                                (AUDIO_EOS,151647), ('<|im_end|>',IM_END_ID)):
            if self.tokenizer.token_to_id(token) != expected:
                raise ValueError(f'unsupported tokenizer control ID for {token}')
        for key, token in (('audio_token',AUDIO_TOKEN), ('audio_bos_token',AUDIO_BOS),
                           ('audio_eos_token',AUDIO_EOS)):
            if processor.get(key) != token:
                raise ValueError(f'unsupported processor {key}')
        feature = processor['feature_extractor']
        if feature.get('sampling_rate') != 24000:
            raise ValueError('unsupported audio sample rate')
        self.audio_config = {k: feature[k] for k in ('normalize_audio','target_dB_FS','eps')}
        self.vae_std = config['acoustic_tokenizer_encoder_config']['vae_std']
        if not isinstance(self.vae_std, (int, float)) or not math.isfinite(self.vae_std) or self.vae_std < 0:
            raise ValueError('invalid acoustic noise scale')
        specs, connectors = load_gguf_frontend(model_path)
        self.noise_width = specs['acoustic'][0].hidden_size
        self.frontend = VibevoiceFrontendRuntime(*specs['acoustic'], *specs['semantic'],
            connectors['acoustic'], connectors['semantic'], frontend_variant='wmma', backend=backend)
        self.weights = None
        try:
            self.weights = load_vibevoice_qwen2_q4(model_path)
            self.runner = VibevoiceQwen2Q4Runtime(self.weights, max_context=self.max_context, backend=backend)
        except BaseException:
            self.frontend.close()
            if self.weights is not None:
                self.weights.close()
            raise

    def close(self):
        with self._lock:
            if not self._closed:
                try:
                    super().close()
                finally:
                    self.weights.close()
