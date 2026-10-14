"""Prefill QSA selection: causal scorer and direct-mask top-k are exact.

``qwen4_exp_qsa_score_rows_f32`` stages each 16-query tile once and scores
only the complete blocks a row can see; ``qwen4_exp_qsa_topk_mask_rows_f32``
writes the WMMA selection bitmask straight from the scores. Contract: exact
parity with the tiled scorer on every score a top-k kernel reads, and a mask
bit-identical to the expand kernel followed by the selection-mask kernel,
including threshold ties.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_score_f32,
    qwen4_exp_qsa_score_rows_f32,
    qwen4_exp_qsa_topk_expand_rows_f32_i64,
    qwen4_exp_qsa_topk_mask_rows_f32,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa_wmma import (
    qwen4_exp_qsa_wmma_selection_mask,
)

RATIO = 4
BUDGET = 512
SENTINEL = np.float32(-321.0)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


class _Device:
    def __init__(self):
        self.runtime = get_hip_runtime()
        self.buffers = []

    def put(self, array):
        host = np.ascontiguousarray(array)
        dev = malloc(max(host.nbytes, 4), runtime=self.runtime)
        copy_host_to_device(dev, host_array_ptr(host), runtime=self.runtime)
        self.buffers.append(dev)
        return dev

    def get(self, dev, like):
        out = np.empty_like(like)
        self.runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), dev, out.nbytes, runtime=self.runtime)
        return out

    def close(self):
        for dev in reversed(self.buffers):
            free(dev, runtime=self.runtime)


@pytest.mark.parametrize("first_position,rows", [(2051, 70), (2051, 16), (6000, 33)])
def test_causal_scorer_matches_tiled_scorer_on_visible_blocks(first_position, rows):
    rng = np.random.default_rng(first_position + rows)
    blocks = (first_position + rows) // RATIO
    queries = rng.normal(0.0, 0.2, size=(rows, 4, 128)).astype(np.float32)
    keys = rng.normal(0.0, 0.2, size=(blocks, 128)).astype(np.float32)
    dev = _Device()
    try:
        q, k = dev.put(queries), dev.put(keys)
        sentinel = np.full((rows, blocks), SENTINEL, dtype=np.float32)
        tiled, causal, full = dev.put(sentinel), dev.put(sentinel), dev.put(sentinel)
        qwen4_exp_qsa_score_f32(q.ptr, k.ptr, tiled.ptr, rows, blocks, 4, 128)
        qwen4_exp_qsa_score_rows_f32(q.ptr, k.ptr, causal.ptr, rows, blocks,
                                     first_position, RATIO)
        qwen4_exp_qsa_score_rows_f32(q.ptr, k.ptr, full.ptr, rows, blocks, -1, RATIO)
        want, got, every = (dev.get(b, sentinel) for b in (tiled, causal, full))
    finally:
        dev.close()
    assert np.array_equal(every.view(np.uint32), want.view(np.uint32))
    for row in range(rows):
        visible = (first_position + row + 1) // RATIO
        assert np.array_equal(got[row, :visible].view(np.uint32),
                              want[row, :visible].view(np.uint32))
        assert np.all(got[row, visible:] == SENTINEL)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("first_position,rows", [(2051, 40), (8190, 9)])
def test_direct_mask_matches_expand_then_mask(ties, first_position, rows):
    rng = np.random.default_rng(7 + rows + int(ties))
    stride = (first_position + rows) // RATIO
    scores = rng.random((rows, stride), dtype=np.float32)
    if ties:
        # Few distinct values: the threshold falls inside a run of equal scores.
        scores = np.floor(scores * 6.0).astype(np.float32) / 6.0
    mask_words = (stride + 1 + 31) // 32
    capacity = BUDGET * RATIO + RATIO - 1
    positions = np.arange(first_position, first_position + rows, dtype=np.int64)
    dev = _Device()
    try:
        s = dev.put(scores)
        pos = dev.put(positions)
        selected = dev.put(np.zeros((rows, capacity), dtype=np.int64))
        counts = dev.put(np.zeros(rows, dtype=np.int32))
        zero_mask = np.zeros((rows, mask_words), dtype=np.uint32)
        parent_mask, direct_mask = dev.put(zero_mask), dev.put(zero_mask)
        qwen4_exp_qsa_topk_expand_rows_f32_i64(
            s.ptr, pos.ptr, selected.ptr, counts.ptr, rows, stride, capacity,
            RATIO, BUDGET)
        qwen4_exp_qsa_wmma_selection_mask(
            selected.ptr, counts.ptr, parent_mask.ptr, rows=rows,
            selected_stride=capacity, mask_words=mask_words)
        qwen4_exp_qsa_topk_mask_rows_f32(
            s.ptr, direct_mask.ptr, rows, first_position, stride, mask_words,
            RATIO, BUDGET)
        want = dev.get(parent_mask, zero_mask)
        got = dev.get(direct_mask, zero_mask)
    finally:
        dev.close()
    assert np.array_equal(got, want)
    bits = np.unpackbits(want.view(np.uint8), bitorder="little").reshape(rows, -1)
    # BUDGET complete blocks plus the partial trailing block when there is one.
    for row in range(rows):
        position = first_position + row
        partial = int((position + 1) % RATIO != 0)
        assert bits[row].sum() == BUDGET + partial
