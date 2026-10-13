"""The tiled QSA block-score kernel is bit-identical to its per-pair parent.

``qwen4_exp_qsa_score_f32`` runs a 16-query x 16-block tiled kernel for four
index heads of 128 dimensions with at least 16 queries. It changes launch
geometry and operand staging only; each score keeps the parent's summation
order. The contract is exact parent parity, plus the CPU reference as the
outer check.
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
from hipengine.kernels.cpu_reference.qwen4_exp import qsa_index_scores
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    build_qwen4_exp_qsa,
    qwen4_exp_qsa_score_f32,
)

HEADS = 4
INDEX_DIM = 128
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


def _scores(queries, keys, *, parent):
    runtime = get_hip_runtime()
    library = build_qwen4_exp_qsa(load=True)
    query_count, blocks = queries.shape[0], keys.shape[0]
    out = np.full(query_count * blocks + 64, SENTINEL, dtype=np.float32)
    buffers = []
    try:
        devs = []
        for arr in (queries, keys, out):
            host = np.ascontiguousarray(arr)
            dev = malloc(host.nbytes, runtime=runtime)
            buffers.append(dev)
            copy_host_to_device(dev, host_array_ptr(host), runtime=runtime)
            devs.append(dev)
        qwen4_exp_qsa_score_f32(
            devs[0].ptr, devs[1].ptr, devs[2].ptr,
            query_count, blocks, HEADS, INDEX_DIM,
            library=library, runtime=runtime, parent=parent,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), devs[2], runtime=runtime)
    finally:
        for dev in reversed(buffers):
            free(dev, runtime=runtime)
    assert np.all(out[query_count * blocks:] == SENTINEL)
    return out[: query_count * blocks].reshape(query_count, blocks)


def _operands(rng, query_count, blocks, *, scale=0.2, shift=0.0):
    queries = rng.normal(shift, scale, size=(query_count, HEADS, INDEX_DIM))
    keys = rng.normal(0.0, scale, size=(blocks, INDEX_DIM))
    return queries.astype(np.float32), keys.astype(np.float32)


@pytest.mark.parametrize(
    "query_count,blocks",
    [(16, 16), (16, 1), (17, 33), (64, 6), (100, 257), (512, 1024), (4096, 100)],
)
def test_tiled_score_is_bit_identical_to_parent(query_count, blocks):
    rng = np.random.default_rng(query_count * 4099 + blocks)
    queries, keys = _operands(rng, query_count, blocks)
    tiled = _scores(queries, keys, parent=False)
    parent = _scores(queries, keys, parent=True)
    assert np.isfinite(parent).all()
    np.testing.assert_array_equal(tiled.view(np.uint32), parent.view(np.uint32))


def test_tiled_score_matches_parent_on_relu_and_zero_edges():
    rng = np.random.default_rng(77)
    queries, keys = _operands(rng, 48, 40, scale=3.0)
    # Heads that are all negative, exactly zero, and one-hot against the keys.
    queries[0] = -np.abs(queries[0])
    keys[0] = np.abs(keys[0])
    queries[1] = 0.0
    queries[2, 1:] = 0.0
    queries[3, :, 1:] = 0.0
    keys[5] = 0.0
    keys[6] = -0.0
    tiled = _scores(queries, keys, parent=False)
    parent = _scores(queries, keys, parent=True)
    np.testing.assert_array_equal(tiled.view(np.uint32), parent.view(np.uint32))
    assert tiled[0, 0] == 0.0
    assert np.all(tiled[1] == 0.0)
    assert np.all(tiled[:, 5] == 0.0)


def test_tiled_score_matches_cpu_reference():
    rng = np.random.default_rng(5)
    queries, keys = _operands(rng, 80, 72)
    tiled = _scores(queries, keys, parent=False)
    expected = qsa_index_scores(queries, keys)
    np.testing.assert_allclose(tiled, expected, rtol=2e-5, atol=2e-5)


def test_fewer_than_sixteen_queries_keep_the_parent_kernel():
    rng = np.random.default_rng(9)
    queries, keys = _operands(rng, 15, 50)
    np.testing.assert_array_equal(
        _scores(queries, keys, parent=False).view(np.uint32),
        _scores(queries, keys, parent=True).view(np.uint32),
    )
