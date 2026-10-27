"""Thinking controls must select real tokens through the Flash-Next owner."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.engine_loop import GenerationExecutionFailed

from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.registry import GenerationRequest


@pytest.fixture(autouse=True)
def _request_owned_c1_path(monkeypatch):
    """These cases drive the request-owned c1 path with a stand-in resident.

    The shared Flash-Next decode batch is the default; its own suite is
    ``tests/test_unit_qwen4exp_ar_batch_decode.py``.
    """

    from tests._qwen4_exp_c1_control import select_request_owned_c1
    select_request_owned_c1(monkeypatch)


class Tokenizer:
    eos_token_id = 9

    def encode(self, text):
        return [1]

    def decode(self, ids, *, skip_special=False):
        return "".join({2: "thought", 3: "</", 4: "think>", 5: "answer", 9: ""}.get(i, "x") for i in ids)


class Runner:
    max_sequence_length = 64
    prefill_chunk_size = 8
    runtime = None

    def __init__(self, *, early_eos=False, natural_close=False):
        self.early_eos = early_eos
        self.natural_close = natural_close
        self.calls = []
        self.position = 0

    def reset(self):
        self.position = 0

    def result(self, token):
        scores = np.full(10, -100, dtype=np.float32)
        scores[token] = 20
        scores[2] = max(scores[2], 10)  # best non-EOS alternative
        return SimpleNamespace(token_id=int(np.argmax(scores)), logits=scores)

    def prefill_append(self, tokens, **kw):
        self.position += len(tokens)
        self.calls.append(('prefill', tuple(tokens), kw))
        return self.result(9 if self.early_eos else 3 if self.natural_close else 2)

    def prefill(self, tokens, **kw):
        self.reset()
        return self.prefill_append(tokens, **kw)

    def step(self, token, **kw):
        self.position += 1
        self.calls.append(('step', token, kw))
        return self.result({3: 4, 4: 5, 5: 9}.get(token, 2))

    def close(self):
        pass


def generator(runner=None):
    return Qwen4ExpGGUFTextGenerator(
        model_path='unused', weight_index=SimpleNamespace(), model_plugin=SimpleNamespace(),
        tokenizer=Tokenizer(), runner=runner or Runner(),
    )


def request(**kw):
    return replace(GenerationRequest(prompts=((1,),), temperature=0, top_p=1, ignore_eos=False, max_tokens=12,
                   thinking_close_token_ids=(3, 4), thinking_hard_token_cap=2), **kw)


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('cap', [0, 2, 4])
def test_hard_close_is_model_consumed_and_finish_state_is_truthful(resident, cap):
    model = generator()
    owner = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1)) if resident else model
    try:
        output = owner.generate_detailed(request(thinking_hard_token_cap=cap))[0]
        assert output.generated_token_ids == (2,) * cap + (3, 4, 5, 9)
        details = output.finish_details
        assert details.reason == 'eos'
        assert details.forced_close
        assert details.phase == 'answer'
        assert details.reasoning_tokens == cap + 2  # shared counter includes close tokens
        assert details.answer_tokens == 2
        steps = [c for c in model.runner.calls if c[0] == 'step']
        assert [c[1] for c in steps] == list(output.generated_token_ids[:-1])
        assert all(c[2]['capture_logits'] for c in model.runner.calls)
        assert all(not c[2]['token_id_resident'] for c in steps)
    finally:
        owner.close()


@pytest.mark.parametrize('resident', [False, True])
def test_thinking_suppresses_default_eos_until_answer(resident):
    model = generator(Runner(early_eos=True))
    owner = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1)) if resident else model
    try:
        output = owner.generate_detailed(request())[0]
        assert output.generated_token_ids == (2, 2, 3, 4, 5, 9)
    finally:
        owner.close()


@pytest.mark.parametrize('resident', [False, True])
def test_natural_close_does_not_force_an_extra_close(resident):
    model = generator(Runner(natural_close=True))
    owner = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1)) if resident else model
    try:
        output = owner.generate_detailed(request(thinking_hard_token_cap=8))[0]
        assert output.generated_token_ids == (3, 4, 5, 9)
        assert not output.finish_details.forced_close
        assert output.finish_details.reasoning_tokens == 2
    finally:
        owner.close()


def test_budget_state_is_per_row_and_resets_on_reuse(monkeypatch):
    model = generator()
    model._resident = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr('hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner',
                        lambda *a, **kw: Runner())
    owner = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=2))
    try:
        outputs = owner.generate_detailed(request(prompts=((1,), (1,))))
        assert [o.generated_token_ids for o in outputs] == [(2, 2, 3, 4, 5, 9)] * 2
        output = owner.generate_detailed(request(thinking_hard_token_cap=0))[0]
        assert output.generated_token_ids == (3, 4, 5, 9)
    finally:
        owner.close()


def test_budget_state_survives_native_streaming():
    model = generator()
    owner = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1))
    try:
        chunks = list(owner.stream_detailed(request()))
        assert ''.join(c.text for c in chunks) == 'thoughtthought</think>answer'
        terminal = [c for c in chunks if c.finish_details is not None][-1]
        assert terminal.finish_details.forced_close
        assert terminal.generated_token_ids == (2, 2, 3, 4, 5, 9)
    finally:
        owner.close()


@pytest.mark.parametrize('resident', [False, True])
def test_soft_close_and_post_thinking_queue_use_the_same_state(resident):
    class SoftRunner(Runner):
        def result(self, token):
            value = super().result(token)
            if token == 2:
                value.logits[3] = 19  # sparse soft-close bias can select it
            return value

    model = generator(SoftRunner())
    owner = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1)) if resident else model
    try:
        output = owner.generate_detailed(request(
            thinking_hard_token_cap=4, thinking_soft_close_window=2,
            post_thinking_forced_tokens_pending=(6,),
            post_thinking_forced_token_reason='tool_choice_required',
            max_tokens=6,
        ))[0]
        assert output.generated_token_ids == (2, 2, 3, 4, 6, 2)
        assert output.finish_details.reasoning_tokens == 4
        assert output.finish_details.answer_tokens == 2
        assert not output.finish_details.forced_close
    finally:
        owner.close()


@pytest.mark.parametrize('resident', [False, True])
def test_partial_forced_close_reports_reasoning_not_answer(resident):
    model = generator()
    owner = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1)) if resident else model
    try:
        output = owner.generate_detailed(request(max_tokens=3))[0]
        assert output.generated_token_ids == (2, 2, 3)
        assert output.finish_details.reason == 'thinking_budget_exhausted'
        assert output.finish_details.phase == 'closing_think'
        assert output.finish_details.answer_tokens == 0
    finally:
        owner.close()


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('allow_close', [False, True])
def test_thinking_close_must_also_satisfy_output_grammar(resident, allow_close):
    class Mask:
        done = False

        def mask_logits(self, logits):
            scores = logits.copy()
            if not allow_close:
                scores[3] = -np.inf
            return scores

        def accept_token(self, token):
            assert allow_close or token != 3

    model = generator()
    model._grammar_session = lambda req: Mask()
    owner = SubmitPollTextGenerator(model, config=EngineLoopConfig(max_active_requests=1)) if resident else model
    try:
        if allow_close:
            assert owner.generate_detailed(request())[0].generated_token_ids == (2, 2, 3, 4, 5, 9)
        else:
            # The engine loop reports an uncontained runner error to the failed
            # request's caller as GenerationExecutionFailed, keeping the cause
            # type and message; the bare model generator raises it unchanged.
            expected = GenerationExecutionFailed if resident else ValueError
            with pytest.raises(expected, match='thinking close token violates output grammar') as caught:
                owner.generate_detailed(request())
            if resident:
                assert caught.value.cause_type == 'ValueError'
    finally:
        owner.close()
