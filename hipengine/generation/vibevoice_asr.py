"""Serialized, torch-free ASR requests using the registered strict primitives.

Production arithmetic candidates remain available in the low-level benchmark
runtimes. This adapter does not certify them without the production task gate.
"""
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import json
import numpy as np
from hipengine.generation.registry import register_text_generator
from hipengine.generation.vibevoice_protocol import (
    AUDIO_TOKEN_ID, IM_END_ID, build_prompt, parse_transcript, preprocess_audio,
)

@dataclass(frozen=True)
class TranscriptionOutput:
    text: str
    segments: list[dict] | None
    generated_token_ids: tuple[int, ...]
    finish_reason: str
    prompt_tokens: int
    audio_seconds: float

class VibeVoiceASRGenerator:
    supports_audio = True
    max_active_requests = 1

    def __init__(self, model_path, *, max_sequence_length=4096):
        from tokenizers import Tokenizer
        from hipengine.loading.vibevoice_asr import load_vibevoice_encoder, load_vibevoice_connector, load_vibevoice_qwen2
        from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime
        from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime
        self._lock = RLock()
        self._closed = False
        self.max_context = max_sequence_length or 4096
        path = Path(model_path)
        self.tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
        config = json.loads((path / 'processor_config.json').read_text())['feature_extractor']
        self.audio_config = {k: config[k] for k in ('normalize_audio','target_dB_FS','eps') if k in config}
        specs = {k: load_vibevoice_encoder(str(path),k) for k in ('acoustic','semantic')}
        self.noise_width = specs['acoustic'][0].hidden_size
        self.vae_std = specs['acoustic'][0].vae_std
        self.frontend = VibevoiceFrontendRuntime(*specs['acoustic'], *specs['semantic'],
            load_vibevoice_connector(str(path),'acoustic'), load_vibevoice_connector(str(path),'semantic'),
            frontend_variant='strict')
        try:
            self.runner = VibevoiceQwen2Runtime(load_vibevoice_qwen2(str(path)),
                max_context=self.max_context, prefill_variant='strict')
        except BaseException:
            self.frontend.close()
            raise

    def generate(self, request):
        raise ValueError('VibeVoice requires audio; use LLM.transcribe(audio, ...)')

    def transcribe(self, audio, *, sample_rate=24000, context='', max_new_tokens=256, seed=20260914):
        from numbers import Integral
        from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
        from hipengine.runtime.vibevoice_qwen2 import greedy_generate
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens < 0:
            raise ValueError('max_new_tokens must be a nonnegative integer')
        if not isinstance(context, str):
            raise ValueError('context must be text')
        pcm = preprocess_audio(audio, sample_rate=sample_rate, **self.audio_config)
        duration = len(pcm) / 24000
        frames = (len(pcm) + 3199) // 3200
        ids = self.tokenizer.encode(build_prompt(duration,frames,context=context), add_special_tokens=False).ids
        positions = [i for i,t in enumerate(ids) if t == AUDIO_TOKEN_ID]
        if len(positions) != frames:
            raise ValueError('audio token count differs from frame count')
        if len(ids) + max(max_new_tokens-1,0) > self.max_context:
            raise ValueError('audio prompt and generation exceed max_sequence_length')
        def bf16(x):
            return (f32_to_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)
        rng = np.random.default_rng(seed)
        noise = bf16(rng.standard_normal((frames,self.noise_width)))
        scale = bf16(bf16(rng.standard_normal(1)) * self.vae_std)[0]
        # One request owns frontend scratch, KV and RNG operands through decode.
        with self._lock:
            if self._closed:
                raise RuntimeError('transcription engine is closed')
            generated = []
            if max_new_tokens:
                embeds = self.frontend.forward(pcm, noise=noise, noise_scale=scale)
                rows = [self.runner.embed_row(t) for t in ids]
                for i,row in zip(positions,embeds):
                    rows[i] = row
                generated = greedy_generate(self.runner,rows,max_new_tokens=max_new_tokens,eos_token_id=IM_END_ID)
            text = self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        reason = 'eos' if generated and generated[-1] == IM_END_ID else 'length'
        return TranscriptionOutput(text,parse_transcript(text),tuple(generated),reason,len(ids),duration)

    def close(self):
        with self._lock:
            if not self._closed:
                self.frontend.close()
                self.runner.close()
                self._closed = True


def make_vibevoice_generator(*, model_path, weight_index, model_plugin, max_sequence_length=None):
    return VibeVoiceASRGenerator(model_path,max_sequence_length=max_sequence_length)

for _backend in ('hip_gfx1100','hip_gfx1151'):
    register_text_generator(model='vibevoice_asr',backend=_backend,quant='bf16',factory=make_vibevoice_generator)
