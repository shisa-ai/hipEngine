import numpy as np
from types import SimpleNamespace

from hipengine.generation import GenerationRequest
from hipengine.generation.inkling2_mini import Inkling2ReferenceGenerator

from hipengine.runtime.inkling2_mini_dspark import markov_step


def test_markov_head_carries_state_and_adds_target_unembed_correction_before_softcap():
    state = np.array([0.5], dtype=np.float32)
    latent = np.arctanh(0.75)
    joint_w = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    joint_b = np.zeros(3, dtype=np.float32)
    out_w = np.array([[2.0]], dtype=np.float32)
    out_b = np.array([0.5], dtype=np.float32)
    head = np.array([[1], [-1]], dtype=np.float32)
    logits, new_state = markov_step(state, np.array([-latent]), np.array([latent]), np.zeros(1), np.array([1, 0]),
                                   joint_w, joint_b, out_w, out_b, head, 2)
    np.testing.assert_allclose(new_state, [-0.125], atol=1e-7)
    np.testing.assert_allclose(logits, 2 * np.tanh([1.5, -1]), rtol=1e-6)
    np.testing.assert_array_equal(state, [0.5])


def test_serial_verification_drops_failed_suffix_and_only_commits_target_tokens():
    seen = []
    draft_seen = []

    class Target:
        def step(self, token, *, logits=True, capture=None):
            seen.append(token)
            if capture is not None:
                capture(0, token)
            scores = np.full(8, -10.0, dtype=np.float32)
            scores[token + 1] = 10
            return scores if logits else None

    gen = Inkling2ReferenceGenerator.__new__(Inkling2ReferenceGenerator)
    gen.model = SimpleNamespace(session=lambda _: Target())
    gen.max_sequence_length = 6
    gen.eos_token_id = 7
    gen.detokenize = lambda tokens: str(tokens)
    draft = SimpleNamespace(append=lambda f: draft_seen.append(f[0]),
                            propose=lambda anchor, depth: (2, 7, 7) if anchor == 1 else (4,))
    drafter = SimpleNamespace(taps=(0,), cfg={"block_size": 4}, session=lambda _: draft)
    request = GenerationRequest(prompts=((0,),), max_tokens=4, temperature=0, top_p=1, ignore_eos=False)
    output, = gen._generate_batch(request, drafter=drafter)
    assert output.generated_token_ids == (1, 2, 3, 4)
    assert seen == draft_seen == [0, 1, 2, 3]
    assert output.telemetry.diagnostics["speculative_mtp"] == {
        "provider": "dspark", "verification": "serial_target", "proposed_tokens": 4, "accepted_tokens": 2,
    }
    seen.clear()
    baseline, = gen.generate_detailed(request)
    assert baseline.generated_token_ids == output.generated_token_ids
