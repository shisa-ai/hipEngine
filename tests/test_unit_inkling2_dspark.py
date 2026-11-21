import numpy as np

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
