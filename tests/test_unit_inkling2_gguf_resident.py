from contextlib import ExitStack, closing
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from hipengine.generation import GenerationRequest
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.inkling2_gguf import Inkling2GGUFGenerator


def _generator():
    calls = []
    sessions = []

    class Session:
        def __init__(self, capacity):
            self.identity = len(sessions)
            self.closed = False
            self.position = 0
            self.capacity = capacity
            self.first = None
            sessions.append(self)

        def prefill_block(self, tokens, *, logits=True):
            assert not self.closed
            if self.position == 0:
                self.first = tokens[0]
            self.position += len(tokens)
            assert self.position <= self.capacity
            calls.append((self.identity, tuple(tokens), logits))
            row = np.zeros(16, np.float32)
            row[self.first + self.position % 3] = 10
            return row if logits else None

        def step(self, token, *, logits=True):
            return self.prefill_block((token,), logits=logits)

        def close(self):
            assert not self.closed
            self.closed = True

    inner = Inkling2GGUFGenerator.__new__(Inkling2GGUFGenerator)
    inner._owned = ExitStack()
    inner.model = SimpleNamespace(session=Session, ffn_device_output=True,
                                  decode_rows=lambda rows: tuple(s.step(t) for s, t in rows),
                                  projection=SimpleNamespace(supports_device_activation=False))
    inner.tokenizer = Tokenizer(WordLevel({f'x{i}': i for i in range(16)}))
    inner.eos_token_id = 15
    inner.max_sequence_length = 24
    inner.prefill_block_width = 3
    return inner, calls, sessions


def test_inkling_gguf_resident_prefill_yields_and_cancel_does_not_close_sibling():
    inner, calls, sessions = _generator()
    params = dict(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True)
    long = GenerationRequest(prompts=((1, 1, 1, 1, 1, 1, 1, 1),), **params)
    short = GenerationRequest(prompts=((8, 8),), **params)
    config = EngineLoopConfig(max_active_requests=2, max_prefill_chunk_tokens=3,
                              round_prefill_token_budget=3, prefix_cache='off', speculative_mtp_serving='off')
    with closing(SubmitPollTextGenerator(inner, config=config)) as driver:
        assert driver.supports_controlled_streaming
        long_sub = driver.submit_detailed(long)
        driver.poll(max_ticks=1)
        assert calls == [(0, (1, 1, 1), False)]
        assert sessions[0].position == 3 and not sessions[0].closed
        short_sub = driver.submit_detailed(short)
        driver.poll(max_ticks=1)
        driver.abort_submission(long_sub, reason='disconnect')
        assert sessions[0].closed
        for _ in range(12):
            if driver.generation_complete(short_sub):
                break
            driver.poll(max_ticks=1)
        assert driver.generation_complete(short_sub)
        out, = driver.take_result(short_sub)
        assert out.generated_token_ids == (10, 8)
        assert out.prompt_tokens == 2 and out.finish_details.reason == 'length'
        assert all(s.closed for s in sessions)
        assert all(len(part) <= 3 for _, part, _ in calls)


@pytest.mark.parametrize("options", [
    dict(max_tokens=0, temperature=0.0),
    dict(max_tokens=4, temperature=0.0, stop_token_sequences=((10, 8),)),
    dict(max_tokens=4, temperature=3.0, row_seeds=(453, 197)),
])
def test_resident_keeps_prefill_chunk_arithmetic_and_sampling_contract(options):
    inner, calls, sessions = _generator()
    request = GenerationRequest(prompts=((3,) * 8, (8, 8)), top_p=1.0, ignore_eos=True, **options)
    expected = inner.generate_detailed(request)
    direct_schedule = [list((tokens, logits) for identity, tokens, logits in calls if identity == session.identity)
                       for session in sessions]
    calls.clear()
    sessions.clear()
    config = EngineLoopConfig(max_active_requests=2, max_prefill_chunk_tokens=2,
                              round_prefill_token_budget=2, prefix_cache='off', speculative_mtp_serving='off')
    with closing(SubmitPollTextGenerator(inner, config=config)) as driver:
        actual = driver.generate_detailed(request)
    assert [(o.generated_token_ids, o.text, o.finish_details) for o in actual] == [
        (o.generated_token_ids, o.text, o.finish_details) for o in expected]
    schedule = [list((tokens, logits) for identity, tokens, logits in calls if identity == session.identity)
                for session in sessions]
    assert sorted(schedule) == sorted(direct_schedule)
    assert all(session.closed for session in sessions)


@pytest.mark.parametrize('fast', [False, True])
def test_gguf_prompt_block_is_collected_and_computed_in_one_round(fast):
    """The scheduler hands over a whole block per round; a narrower budget would idle the row."""
    width, defaults = Inkling2GGUFGenerator._prefill_schedule(fast)
    assert width > 0
    assert defaults['max_prefill_chunk_tokens'] == width
    assert defaults['round_prefill_token_budget'] == width


def test_gguf_fast_reads_wider_prompt_blocks_than_strict():
    """Fast blocks are 256 rows; strict keeps the 64 rows where it is faster to the first token."""
    strict_width, strict_defaults = Inkling2GGUFGenerator._prefill_schedule(False)
    fast_width, fast_defaults = Inkling2GGUFGenerator._prefill_schedule(True)
    assert (strict_width, fast_width) == (64, 256)
    assert strict_defaults is Inkling2GGUFGenerator.engine_loop_config_defaults
    assert fast_defaults is Inkling2GGUFGenerator.fast_engine_loop_config_defaults


def test_resident_waits_for_a_whole_block_when_the_budget_splits_it():
    """A block runs only after the scheduler has handed over its whole width."""
    inner, calls, _sessions = _generator()
    request = GenerationRequest(prompts=((1,) * 7,), max_tokens=1, temperature=0.0, ignore_eos=True, top_p=1.0)

    def blocks_per_tick(chunk):
        calls.clear()
        config = EngineLoopConfig(max_active_requests=1, max_prefill_chunk_tokens=chunk,
                                  round_prefill_token_budget=chunk, prefix_cache='off', speculative_mtp_serving='off')
        ticks = []
        with closing(SubmitPollTextGenerator(inner, config=config)) as driver:
            driver.submit_detailed(request)
            for _ in range(3):
                driver.poll(max_ticks=1)
                ticks.append([part for _, part, _ in calls])
        return ticks

    # One three-token block per round, then the one-token tail that yields logits.
    assert blocks_per_tick(3) == [[(1, 1, 1)], [(1, 1, 1), (1, 1, 1)], [(1, 1, 1), (1, 1, 1), (1,)]]
    # Two tokens per round: the first block waits for the second round and the row idles between blocks.
    assert blocks_per_tick(2) == [[], [(1, 1, 1)], [(1, 1, 1), (1, 1, 1)]]
