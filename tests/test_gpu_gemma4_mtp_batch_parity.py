"""Real-artifact packed target and assistant parity at identical contexts."""
from contextlib import contextmanager

import numpy as np
import pytest

from tests.test_gpu_gemma4_assistant_forward import _Session, _needs, _to_bf16, TEMPLATED


@pytest.fixture(scope="module")
def session():
    with _Session(TEMPLATED) as value:
        yield value


@contextmanager
def _batch(session):
    from hipengine.runtime.gemma4 import Gemma4Runner

    owners = [Gemma4Runner(weights=session.backbone_weights, capacity=128,
                          max_logits_rows=5) for _ in range(4)]
    workspace = Gemma4Runner(weights=session.backbone_weights, capacity=128,
                            max_logits_rows=20)
    prompts = [list(TEMPLATED) + [17] * i for i in range(4)]
    try:
        first = [int(np.argmax(owner.forward(prompt))) for owner, prompt in zip(owners, prompts)]
        yield owners, workspace, prompts, first
    finally:
        for owner in owners:
            owner.close()
        workspace.close()


@_needs
def test_real_target_batch_matches_independent_requests(session):
    with _batch(session) as (owners, workspace, prompts, first):
        candidate_rows = [[token] + [17] * i for i, token in enumerate(first)]
        expected = []
        for prompt, row in zip(prompts, candidate_rows):
            session.runner.reset()
            session.runner.forward(prompt)
            expected.append(np.asarray(session.runner.forward(row, logits_rows=len(row), verification=True)).reshape(len(row), -1))
        actual = workspace.forward_batch(owners, candidate_rows, verification=True)
        for reference, result in zip(expected, actual):
            np.testing.assert_array_equal(result, reference)


@_needs
def test_real_assistant_batch_matches_independent_recurrent_chains(session):
    with _batch(session) as (owners, workspace, prompts, first):
        shared = [{i: owner.shared_kv(i) for i in range(owner.layer_count)} for owner in owners]
        head = session.head
        expected_logits = [[] for _ in owners]
        expected_states = [[] for _ in owners]
        for i, owner in enumerate(owners):
            head.prime(owner.hidden_state())
            token = first[i]
            for _ in range(4):
                logits, state = head.forward(token, position=owner.position, shared_kv=shared[i])
                expected_logits[i].append(logits.copy())
                expected_states[i].append(_to_bf16(state))
                token = int(np.argmax(logits))
        states = head.prime_batch([owner.hidden_state() for owner in owners])
        tokens = list(first)
        for depth in range(4):
            logits, states = head.forward_batch(tokens, positions=[owner.position for owner in owners],
                                                shared_kv=shared, recurrent_states=states)
            for i in range(4):
                np.testing.assert_array_equal(logits[i], expected_logits[i][depth])
                np.testing.assert_array_equal(states[i], expected_states[i][depth])
            tokens = np.argmax(logits, axis=1).tolist()
