"""CPU checks for the dense-Llama verification protocol."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.llama_dense_verify import build_cases, measured_request


def test_context_cases_keep_exact_lengths_and_retrieval_key():
    class Tokenizer:
        @property
        def tokenizer(self):
            return self

        def format_chat_prompt(self, text, *, enable_thinking):
            assert enable_thinking is False
            return '<|im_start|>user\n' + text + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

        def tokenize(self, text):
            return tuple(map(ord, text))

        def tokenize_chat(self, text, *, enable_thinking):
            assert enable_thinking is False
            return self.tokenize(text)

    cases = build_cases(Tokenizer())
    assert [len(c['prompt_ids']) for c in cases[4:]] == [1023, 1025, 4097, 8193]
    for case in cases[4:]:
        assert ''.join(map(chr, case['prompt_ids'])).count('LANTERN-5831') == 1


@pytest.mark.parametrize('nonfinite', [False, True])
def test_measurement_restores_runner_and_counts_decode(nonfinite):
    @dataclasses.dataclass
    class Finish:
        reason: str = 'stop'

    def prefill(ids):
        return np.array([[float('nan') if nonfinite else 1.0]])

    def decode_step(token):
        return np.array([1.0])

    runner = SimpleNamespace(prefill=prefill, decode_step=decode_step,
                             runtime=SimpleNamespace(device_synchronize=lambda: None),
                             _attention_is_tiled=True)
    generator = SimpleNamespace(_ensure_runner=lambda: runner, strict_arithmetic=False)
    class Public:
        def generate_detailed(self, ids, params):
            runner.prefill(ids)
            runner.decode_step(1)
            return [SimpleNamespace(generated_token_ids=(1, 2), text='OK', finish_details=Finish())]
    case = dict(name='fake', prompt_ids=[1, 2, 3], required='OK')
    if nonfinite:
        with pytest.raises(RuntimeError, match='nonfinite'):
            measured_request(Public(), generator, case, 4)
    else:
        row = measured_request(Public(), generator, case, 4)
        assert row['completed'] and row['required_text_present']
        assert len(row['decode_step_seconds']) == 1
        assert len(row['logit_hashes']) == 2
    assert runner.prefill is prefill
    assert runner.decode_step is decode_step
