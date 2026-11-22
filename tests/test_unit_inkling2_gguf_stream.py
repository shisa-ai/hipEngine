from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models

from hipengine.generation.deadline import GenerationCancellationToken, GenerationCancelled
from hipengine.generation.inkling2_gguf import Inkling2GGUFGenerator
from hipengine.generation.registry import GenerationRequest


def _setup():
    calls = []

    class Session:
        def __init__(self):
            self.tokens = iter((1, 2, 3, 4))

        def step(self, token, *, logits=True):
            calls.append(token)
            scores = np.zeros(5, dtype=np.float32)
            scores[next(self.tokens)] = 5
            return scores if logits else None

        def close(self):
            calls.append('close')

    gen = Inkling2GGUFGenerator.__new__(Inkling2GGUFGenerator)
    gen.model = SimpleNamespace(session=lambda capacity: Session(), ffn_device_output=False,
                                projection=SimpleNamespace(supports_device_activation=False))
    gen.tokenizer = Tokenizer(models.BPE(vocab={'x': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}, merges=[]))
    gen.tokenizer.decoder = decoders.ByteLevel()
    gen.max_sequence_length = 10
    gen.eos_token_id = 0
    return gen, calls, GenerationRequest(prompts=((0,),), max_tokens=4, temperature=0, top_p=1, ignore_eos=False)


def test_stream_yields_before_the_second_decode_step_and_agrees_with_blocking():
    gen, calls, request = _setup()
    stream = gen.stream_detailed(request)
    first = next(stream)
    assert first.text == 'a'
    assert calls == [0]
    rest = list(stream)
    output, = gen.generate_detailed(request)
    assert ''.join(chunk.text for chunk in [first, *rest]) == output.text == 'abcd'
    assert rest[-1].generated_token_ids == output.generated_token_ids == (1, 2, 3, 4)
    assert rest[-1].finish_details == output.finish_details
    assert rest[-1].telemetry.diagnostics['expert_mix'] == 'ordered_host'
    assert rest[-1].telemetry.diagnostics['expert_activation'] == 'split_host'


def test_stream_holds_a_possible_stop_prefix_and_reports_terminal_ids():
    gen, calls, request = _setup()
    chunks = list(gen.stream_detailed(replace(request, stop_token_sequences=((2, 3),))))
    assert ''.join(chunk.text for chunk in chunks) == 'a'
    assert calls == [0, 1, 2, 'close']
    assert chunks[-1].generated_token_ids == (1, 2, 3)
    assert chunks[-1].finish_details.stop_sequence == (2, 3)


def test_stream_cancellation_closes_only_the_owned_session():
    gen, calls, request = _setup()
    cancellation = GenerationCancellationToken()
    stream = gen.stream_detailed(replace(request, cancellation_token=cancellation))
    assert next(stream).text == 'a'
    cancellation.cancel()
    with pytest.raises(GenerationCancelled):
        next(stream)
    assert calls == [0, 'close']
