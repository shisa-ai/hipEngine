"""Completed-layer suspension preserves a request beside another device owner."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL('libamdhip64.so')
    HIP_AVAILABLE = True
except OSError:
    HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HIP_AVAILABLE, reason='HIP runtime is not available')

from tests.test_gpu_qwen4exp_batch_runner import model, _c1, _download
from hipengine.runtime.gguf_linear import raw_k_prefill_rowbatch


def _state(runner):
    snapshot = runner.snapshot()
    arrays = dict(snapshot.decode_state.buffers)
    for layer, attention in enumerate(runner.attention_states):
        rows = ((attention.max_positions + attention.block_size - 1) // attention.block_size) * attention.block_size
        live_bytes = runner.position * (attention.key_cache.nbytes // rows)
        for name in ('key_cache', 'value_cache'):
            arrays[f'attention{layer}.{name}'] = _download(
                getattr(attention, name), (live_bytes,), np.uint8, runner.runtime,
            )
    for layer, index in enumerate(runner.index_states):
        for name, rows in (('raw_keys', index.count), ('pooled_keys', index.pooled_count)):
            nbytes = rows * index.index_dim * 4
            arrays[f'index{layer}.{name}'] = _download(
                getattr(index, name), (nbytes,), np.uint8, runner.runtime,
            )
    return snapshot, arrays


@pytest.mark.parametrize('length', (3, 5, 11))
def test_segmented_prefill_matches_identical_chunks_with_peer_steps(model, length):
    reference, candidate, peer = (_c1(model) for _ in range(3))
    tokens = tuple((row * 7 + 1) % 60 for row in range(length))
    try:
        peer.prefill((1, 2, 3), capture_logits=False)
        segments = 0
        for start in range(0, length, 4):
            chunk = tokens[start:start + 4]
            expected = reference.prefill_append(chunk)
            while True:
                actual = candidate.prefill_append_resumable(chunk, time_budget_seconds=1e-12)
                assert raw_k_prefill_rowbatch() == 0
                if actual is not None:
                    break
                segments += 1
                peer.step(11, capture_logits=False)
            np.testing.assert_array_equal(actual.logits, expected.logits)
        assert segments > 0
        left, arrays = _state(reference)
        right, other = _state(candidate)
        assert left.position == right.position == length
        assert dict(left.ple_hash_states) == dict(right.ple_hash_states)
        assert arrays.keys() == other.keys()
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], other[name], err_msg=name)
        for _ in range(3):
            np.testing.assert_array_equal(reference.step(7).logits, candidate.step(7).logits)
    finally:
        peer.close()
        candidate.close()
        reference.close()


def test_reset_closes_a_suspended_context_before_reuse(model):
    runner = _c1(model)
    try:
        assert runner.prefill_append_resumable((1, 2, 3), time_budget_seconds=1e-12) is None
        runner.reset()
        assert runner._pending_prefill_append is None
        assert raw_k_prefill_rowbatch() == 0
        recovered = runner.prefill((4, 5, 6))
        runner.reset()
        np.testing.assert_array_equal(runner.prefill((4, 5, 6)).logits, recovered.logits)
    finally:
        runner.close()
