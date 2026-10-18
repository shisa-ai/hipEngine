"""Strict packed verification against independent real-GPU request forwards."""
import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available
from tests.test_gpu_gemma4_runner_verify_rows import runner

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("prefix_length,uniform_rows", [(3, 0), (19, 0), (53, 0), (19, 9)])
def test_packed_requests_match_independent_logits_and_rewind(runner, width, prefix_length, uniform_rows):
    from hipengine.runtime.gemma4 import Gemma4Runner

    workspace = Gemma4Runner(weights=runner.weights, capacity=64, max_block=40,
                             max_logits_rows=36, verify_dense_exact=True)
    owners = [Gemma4Runner(weights=runner.weights, capacity=64, max_block=runner.max_block,
                          max_logits_rows=9, verify_dense_exact=True) for _ in range(width)]
    prompts = [[(3 + i * 7 + j) % 60 for j in range(prefix_length + i)] for i in range(width)]
    candidates = [[(11 + i + j) % 60 for j in range(uniform_rows or 1 + i)] for i in range(width)]
    try:
        expected = []
        for prompt, tokens in zip(prompts, candidates):
            runner.reset()
            runner.forward(prompt)
            expected.append(np.stack([runner.forward([token], verification=True) for token in tokens]))
        for owner, prompt in zip(owners, prompts):
            owner.forward(prompt)
        actual = workspace.forward_batch(owners, candidates, verification=True)
        for owner, prompt, tokens, reference, result in zip(owners, prompts, candidates, expected, actual):
            np.testing.assert_array_equal(result, reference)
            assert owner.position == len(prompt) + len(tokens)
        # Reuse the same independent KV slots after dropping speculative tails.
        for owner, prompt in zip(owners, prompts):
            owner.rewind(len(prompt))
        repeated = workspace.forward_batch(owners, candidates, verification=True)
        for reference, result in zip(actual, repeated):
            np.testing.assert_array_equal(result, reference)
        # Replace the last neighbor's prompt while holding width and the other
        # rows fixed. Nothing about its data may affect the first request.
        if width > 1:
            for owner, prompt in zip(owners, prompts):
                owner.reset()
                owner.forward(prompt if owner is not owners[-1] else [8] * len(prompt))
            substituted = workspace.forward_batch(owners, candidates, verification=True)
            np.testing.assert_array_equal(substituted[0], actual[0])
    finally:
        for owner in owners:
            owner.close()
        workspace.close()
