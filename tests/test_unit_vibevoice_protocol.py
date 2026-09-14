import numpy as np
import pytest
from hipengine.generation.vibevoice_protocol import build_prompt, preprocess_audio, parse_transcript

@pytest.mark.parametrize('audio', [[], [np.nan], [[0.,1.]]])
def test_invalid_audio(audio):
    with pytest.raises(ValueError):
        preprocess_audio(audio)

def test_audio_contract():
    x = np.zeros(3201,dtype=np.float32)
    np.testing.assert_array_equal(preprocess_audio(x),x)
    with pytest.raises(ValueError):
        preprocess_audio(x,sample_rate=16000)
    y = np.ones(3201,dtype=np.float32)
    out = preprocess_audio(y)
    assert len(out) == 3201 and np.all(out > .05) and np.all(out < .06)
    np.testing.assert_array_equal(y,np.ones_like(y))

def test_prompt_context_and_generation_boundary():
    prompt = build_prompt(11.07,84,context='Kyoto')
    assert '11.07 seconds' in prompt and 'extra info: Kyoto' in prompt
    assert prompt.endswith('<|im_end|>\n')
    assert '<|im_start|>assistant' not in prompt

def test_parser_keeps_assistant_content():
    text = 'assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"my assistant said hello"}]'
    assert parse_transcript(text)[0]['Content'] == 'my assistant said hello'
    assert parse_transcript(text + ' trailing') is None

def test_public_transcribe_delegates():
    from hipengine import LLM
    from types import SimpleNamespace
    llm = object.__new__(LLM)
    seen = {}
    def transcribe(audio,**kwargs):
        seen.update(kwargs)
        return audio
    llm._get_text_generator = lambda: SimpleNamespace(transcribe=transcribe)
    assert llm.transcribe('pcm',context='Kyoto') == 'pcm'
    assert seen['context'] == 'Kyoto'

def test_generator_serializes_and_closes(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import RLock
    from types import SimpleNamespace
    import time
    from hipengine.generation.vibevoice_asr import VibeVoiceASRGenerator
    import hipengine.runtime.vibevoice_qwen2 as lm
    owner = object.__new__(VibeVoiceASRGenerator)
    owner._lock, owner._closed = RLock(), False
    owner.audio_config, owner.max_context = {}, 1000
    owner.noise_width, owner.vae_std = 2, .625
    owner.tokenizer = SimpleNamespace(encode=lambda *a,**k: SimpleNamespace(ids=[151648]),decode=lambda *a,**k:'[]')
    active = 0
    seen = []
    def forward(pcm,**kwargs):
        nonlocal active
        active += 1
        assert active == 1
        seen.append(kwargs['noise'].copy())
        time.sleep(.01)
        return np.zeros((1,2),dtype=np.float32)
    def generate(*a,**k):
        nonlocal active
        active -= 1
        return [151645]
    monkeypatch.setattr(lm,'greedy_generate',generate)
    owner.frontend = SimpleNamespace(forward=forward,close=lambda:None)
    owner.runner = SimpleNamespace(embed_row=lambda _:np.zeros(2),close=lambda:None)
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda _:owner.transcribe(np.zeros(3200),seed=1),range(2)))
    assert all(r.finish_reason == 'eos' for r in results)
    np.testing.assert_array_equal(*seen)
    owner.close(); owner.close()
    with pytest.raises(RuntimeError,match='closed'):
        owner.transcribe(np.zeros(3200))
