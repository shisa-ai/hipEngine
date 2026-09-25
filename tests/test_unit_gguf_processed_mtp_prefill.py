"""Processed tool constraints retain MTP priming across radix boundaries."""
from collections import Counter
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.sampling import SamplingMode
from hipengine.generation.engine_loop import SubmitPollTextGenerator


@pytest.mark.parametrize('present', [False, True])
def test_staged_hooks_do_not_override_missing_model_mtp_weights(present):
    wrapper = object.__new__(SubmitPollTextGenerator)
    wrapper._inner = SimpleNamespace(supports_speculative_mtp=present)
    wrapper._has_resident_runner = True
    wrapper._runner = SimpleNamespace(
        speculative_capability=lambda: None,
        execute_target_frontier=lambda: None,
    )
    assert wrapper.supports_speculative_mtp is present


@pytest.mark.parametrize('prompt_length,reused', [(513,0),(777,0),(1383,0),(777,512)])
def test_processed_prefill_streams_every_new_hidden_row(monkeypatch, prompt_length, reused):
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    session = SimpleNamespace(position=reused)
    prompt = tuple(range(prompt_length))
    row = SimpleNamespace(slot=None, lease=SimpleNamespace(session=session),
        sampler_plan=SimpleNamespace(mode=SamplingMode.PROCESSED_ARGMAX),
        prompt_ids=prompt, prefix_reused_tokens=reused, request_id=7,
        mtp2_candidate_budget=3, prefill_ms=0, prefill_chunk_count=0)
    sink = object()
    calls, closes, finishes, snapshots = [], [], [], []
    def prefill(prompts, **kwargs):
        calls.append((prompts[0], kwargs))
        session.position += len(prompts[0])
        return [SimpleNamespace(logits=[1.,2.])]
    monkeypatch.setattr(runner,'_prepare_sampled_prefill',lambda row:(None,None))
    monkeypatch.setattr(runner,'_begin_mtp2_prompt_streaming',lambda rows:(sink,))
    monkeypatch.setattr(runner,'_packed_execution_owner',lambda session:SimpleNamespace(prefill_batch_native=prefill))
    monkeypatch.setattr(runner,'_finish_mtp2_prompt_streaming',lambda *args,**kw:closes.append(kw))
    monkeypatch.setattr(runner,'_refresh_prefix_cache_at_prompt_boundary',lambda *a:None)
    monkeypatch.setattr(runner,'_refresh_prefix_cache',lambda row:snapshots.append(session.position))
    monkeypatch.setattr(runner,'_finish_sampled_prefill',lambda *a,**kw:finishes.append(kw))
    monkeypatch.setattr(runner,'_prefix_phase_add',lambda *a:None)
    runner._route_counts=Counter()
    runner._prefill_processed_argmax_chunk(row,prompt[reused:],final_chunk=True)
    assert tuple(t for tokens,_ in calls for t in tokens)==prompt[reused:]
    offset=0
    for tokens,kw in calls:
        assert kw['target_hidden_chunk_sinks']==(sink,)
        assert kw['target_hidden_chunk_starts']==(offset,)
        assert kw['finish_target_hidden_sinks'] is False
        offset+=len(tokens)
    assert calls[-1][1]['return_logits'] is True
    assert closes==[{'success':True}]
    assert finishes==[{'native_compact_prefill':True}]
    assert session.position==prompt_length
