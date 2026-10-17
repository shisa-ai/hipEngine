"""Flash-Next must consume sampled IDs, not the runner's raw argmax."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.registry import GenerationRequest


class Tokenizer:
    eos_token_id = 9

    def encode(self, text):
        return [1]

    def decode(self, ids, *, skip_special=False):
        return ''.join(chr(65 + i) if i != 9 else '' for i in ids)


class Runner:
    max_sequence_length = 128
    prefill_chunk_size = 8
    runtime = None

    def __init__(self, scores=None):
        self.scores = np.array([0, 0, 0, 0] + [-np.inf] * 6 if scores is None else scores, dtype=np.float32)
        self.calls = []
        self.position = 0

    def reset(self):
        self.position = 0

    def prefill_append(self, ids, **kw):
        self.position += len(ids)
        self.calls.append(('prefill', tuple(ids), kw))
        return SimpleNamespace(token_id=int(np.argmax(self.scores)), logits=self.scores.copy())

    def prefill(self, ids, **kw):
        self.reset()
        return self.prefill_append(ids, **kw)

    def step(self, token, **kw):
        self.position += 1
        self.calls.append(('step', token, kw))
        return SimpleNamespace(token_id=int(np.argmax(self.scores)), logits=self.scores.copy())

    def close(self):
        pass


def generator(runner=None):
    return Qwen4ExpGGUFTextGenerator(model_path='unused', weight_index=SimpleNamespace(),
                                   model_plugin=SimpleNamespace(), tokenizer=Tokenizer(), runner=runner or Runner())


def request(**kw):
    return replace(GenerationRequest(prompts=((1,),), max_tokens=12, temperature=1,
                                     top_p=1, ignore_eos=False, seed=41, logprobs=True, top_logprobs=4), **kw)


@pytest.fixture(params=[False, True], ids=['direct', 'resident'])
def owner(request):
    model = generator()
    api = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1)) if request.param else model
    yield model, api
    api.close()


def test_uniform_sample_is_seeded_non_greedy_and_logprobs_are_probabilities(owner):
    model, api = owner
    first = api.generate_detailed(request())[0]
    again = api.generate_detailed(request())[0]
    changed = api.generate_detailed(request(seed=42))[0]
    assert first.generated_token_ids == again.generated_token_ids
    assert first.token_logprobs == again.token_logprobs
    assert first.generated_token_ids != changed.generated_token_ids
    assert len(set(first.generated_token_ids)) > 1
    assert len(first.token_logprobs) == 12
    assert first.finish_details.sampler_mode == 'host_logits_sample'
    for token, chosen in zip(first.token_logprobs, first.generated_token_ids, strict=True):
        assert token.token_id == chosen
        assert token.logprob == pytest.approx(-np.log(4))
        assert [i for i, _, _ in token.top_logprobs] == [0, 1, 2, 3]
        assert sum(np.exp(lp) for _, _, lp in token.top_logprobs) == pytest.approx(1)
    steps = [c for c in model.runner.calls if c[0] == 'step'][:11]
    assert [c[1] for c in steps] == list(first.generated_token_ids[:-1])
    assert all(c[2]['capture_logits'] and not c[2]['token_id_resident'] for c in steps)


@pytest.mark.parametrize('filters,allowed', [({'top_k': 2}, [0, 1]), ({'top_p': .5}, [0, 1]),
                                           ({'suppress_token_ids': (0, 1)}, [2, 3])])
def test_filtered_logprobs_describe_the_retained_distribution(owner, filters, allowed):
    _, api = owner
    output = api.generate_detailed(request(**filters))[0]
    assert set(output.generated_token_ids) <= set(allowed)
    for token in output.token_logprobs:
        assert [i for i, _, _ in token.top_logprobs] == allowed
        assert token.logprob == pytest.approx(-np.log(2))


def test_penalties_use_prompt_and_generated_history(owner):
    _, api = owner
    output = api.generate_detailed(request(temperature=0, frequency_penalty=1, max_tokens=3))[0]
    assert output.generated_token_ids == (0, 2, 3)
    for token in output.token_logprobs:
        assert np.isfinite(token.logprob)


def test_min_p_and_bias_have_correct_probability(owner):
    _, api = owner
    output = api.generate_detailed(request(logit_bias={0: np.log(4)}, min_p=.5))[0]
    assert output.generated_token_ids == (0,) * 12
    assert all(t.logprob == 0 and t.top_logprobs == ((0, 'A', 0.0),) for t in output.token_logprobs)


def test_greedy_inert_filters_keep_compact_argmax(owner):
    model, api = owner
    output = api.generate_detailed(request(temperature=0, top_k=40, top_p=.2, min_p=.1,
                                          logprobs=False, top_logprobs=0))[0]
    assert output.generated_token_ids == (0,) * 12
    assert not output.token_logprobs
    assert all(not c[2]['capture_logits'] for c in model.runner.calls)


def test_native_stream_and_blocking_logprobs_agree():
    model = generator()
    api = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1))
    try:
        output = api.generate_detailed(request())[0]
        chunks = list(api.stream_detailed(request()))
        assert ''.join(c.text for c in chunks) == output.text
        assert tuple(t for c in chunks for t in c.token_logprobs) == output.token_logprobs
        assert chunks[-1].generated_token_ids == output.generated_token_ids
    finally:
        api.close()


def test_explicit_row_seeds_are_isolated_from_neighbors_slots_and_request_ids(monkeypatch):
    model = generator()
    model._resident = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr('hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner', lambda *a, **k: Runner())
    api = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=2))
    try:
        alone = api.generate_detailed(request(row_seeds=(123,)))[0]
        pair = api.generate_detailed(request(prompts=((2,), (1,)), row_seeds=(99, 123)))
        after = api.generate_detailed(request(row_seeds=(123,)))[0]
        assert alone.generated_token_ids == pair[1].generated_token_ids == after.generated_token_ids
        assert alone.token_logprobs == pair[1].token_logprobs == after.token_logprobs
        assert pair[0].generated_token_ids != pair[1].generated_token_ids
    finally:
        api.close()


def test_bad_row_seeds_fail_before_prefill(owner):
    model, api = owner
    with pytest.raises(ValueError, match='row_seeds'):
        api.generate_detailed(request(prompts=((1,), (2,)), row_seeds=(123,)))
    assert not model.runner.calls


def test_min_tokens_suppresses_eos_until_requested_count(owner):
    model, api = owner
    model.runner.scores[9] = 100
    output = api.generate_detailed(request(temperature=0, eos_token_id=9, min_tokens=2))[0]
    assert output.generated_token_ids == (0, 0, 9)
    assert output.finish_details.reason == 'eos'
    assert all(9 not in [i for i, _, _ in t.top_logprobs] for t in output.token_logprobs[:2])


def test_alternate_eos_and_ignore_eos(owner):
    _, api = owner
    kwargs = dict(temperature=0, logit_bias={3: 10}, eos_token_id=3)
    output = api.generate_detailed(request(**kwargs))[0]
    assert output.generated_token_ids == (3,)
    assert output.finish_details.eos_token_id == 3
    assert api.generate_detailed(request(**kwargs, ignore_eos=True))[0].generated_token_ids == (3,) * 12


def test_forced_token_sequence_stops_with_real_model_consumption(owner):
    model, api = owner
    output = api.generate_detailed(request(forced_tokens_pending=(2, 3), stop_token_sequences=((2, 3),)))[0]
    assert output.generated_token_ids == (2, 3)
    assert output.finish_details.reason == 'stop'
    assert output.finish_details.stop_sequence == (2, 3)
    assert [c[1] for c in model.runner.calls if c[0] == 'step'] == [2]


def test_zero_tokens_has_no_model_work_or_logprobs(owner):
    model, api = owner
    output = api.generate_detailed(request(max_tokens=0))[0]
    assert output.generated_token_ids == ()
    assert output.token_logprobs == ()
    assert not model.runner.calls


def test_grammar_masks_before_sampling_and_logprob_normalization(owner):
    model, api = owner
    class Grammar:
        done = False
        def mask_logits(self, scores):
            out = scores.copy()
            out[:2] = -np.inf
            return out
        def accept_token(self, token):
            assert token in (2, 3)
    model._grammar_session = lambda request: Grammar()
    output = api.generate_detailed(request(grammar={'format': 'choice', 'value': ['C', 'D']}))[0]
    assert set(output.generated_token_ids) <= {2, 3}
    for token in output.token_logprobs:
        assert [i for i, _, _ in token.top_logprobs] == [2, 3]
        assert token.logprob == pytest.approx(-np.log(2))


def test_cancelled_work_does_not_publish_sampled_token_metadata():
    from hipengine.generation.deadline import GenerationCancellationToken
    token = GenerationCancellationToken()
    class CancellingRunner(Runner):
        def step(self, token_id, **kwargs):
            result = super().step(token_id, **kwargs)
            token.cancel()
            return result
    model = generator(CancellingRunner())
    api = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1))
    try:
        output = api.generate_detailed(request(cancellation_token=token))[0]
        assert output.generated_token_ids == ()
        assert output.token_logprobs == ()
    finally:
        api.close()


@pytest.mark.parametrize('controls', [dict(temperature=1), dict(temperature=0, logprobs=True),
                                      dict(temperature=0, frequency_penalty=1)])
def test_unimplemented_multimodal_controls_reject_before_vision_work(controls):
    model = generator()
    model._vision_runner = SimpleNamespace(close=lambda: None)
    model._encode_multimodal_input = lambda _: pytest.fail('vision work before admission')
    params = request(**(dict(logprobs=False, top_logprobs=0) | controls))
    try:
        with pytest.raises(NotImplementedError, match='multimodal sampling and logprobs'):
            model.generate_multimodal_detailed('prompt', None, params)
    finally:
        model.close()
