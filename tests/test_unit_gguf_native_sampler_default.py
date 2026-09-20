"""GGUF native placement defaults on without widening request support."""
import pytest
from hipengine.generation.qwen35_gguf import _gguf_sampler_plan, _native_gpu_sampler_requested
from hipengine.generation.registry import GenerationRequest
from hipengine.generation.sampling import SamplingMode

@pytest.mark.parametrize('value,enabled', [(None,True),('1',True),('0',False),('false',False),('no',False),('off',False),('',False)])
def test_default_and_explicit_rollback(monkeypatch,value,enabled):
    if value is None:
        monkeypatch.delenv('HIPENGINE_QWEN35_NATIVE_SAMPLER',raising=False)
    else:
        monkeypatch.setenv('HIPENGINE_QWEN35_NATIVE_SAMPLER',value)
    assert _native_gpu_sampler_requested() is enabled
    req=GenerationRequest(prompts=((1,2),),max_tokens=3,ignore_eos=True,temperature=.7,top_p=.95)
    assert _gguf_sampler_plan(req,native_gpu_available=True).mode is (SamplingMode.GPU_SAMPLE if enabled else SamplingMode.HOST_LOGITS_SAMPLE)

@pytest.mark.parametrize('available,top_k', [(False,0),(True,65)])
def test_default_keeps_unavailable_or_unsupported_requests_on_host(monkeypatch,available,top_k):
    monkeypatch.delenv('HIPENGINE_QWEN35_NATIVE_SAMPLER',raising=False)
    req=GenerationRequest(prompts=((1,2),),max_tokens=3,ignore_eos=True,temperature=.7,top_p=.95,top_k=top_k)
    assert _gguf_sampler_plan(req,native_gpu_available=available).mode is SamplingMode.HOST_LOGITS_SAMPLE
