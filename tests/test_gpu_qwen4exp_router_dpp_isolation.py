"""Same-width row isolation and buffer ownership for the ordered router leaf."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.moe import router


@pytest.fixture(scope="module")
def library():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime unavailable")
    return router.build_qwen35_router()


@pytest.mark.parametrize("rows", [3, 5, 17, 513])
def test_router_dpp_neighbor_substitution_permutation_and_guards(library, rows):
    """Neighbors and slot moves cannot alter a fixed row's logits or routes."""
    hidden, experts, top_k, guard = 2560, 33, 10, 32
    rng = np.random.default_rng(20261006 + rows)
    x = rng.normal(0, .1, (rows, hidden)).astype(np.float32)
    weight = rng.normal(0, .1, (experts, hidden)).astype(np.float32)
    weight[1] = weight[0]  # Deterministic ties remain part of every scenario.
    shapes = ((rows, experts), (rows, top_k), (rows, top_k))
    dtypes = (np.float32, np.int64, np.float32)
    arrays = [x, weight]
    output_bytes = [int(np.prod(shape)) * np.dtype(dtype).itemsize
                    for shape, dtype in zip(shapes, dtypes, strict=True)]
    sentinels = [np.full(nbytes + 2 * guard, 0xA5, np.uint8)
                 for nbytes in output_bytes]
    buffers = []
    try:
        for array in [*arrays, *sentinels]:
            buffers.append(malloc(array.nbytes))
        copy_host_to_device(buffers[1], host_array_ptr(weight), weight.nbytes)

        def capture(inputs, producer):
            copy_host_to_device(buffers[0], host_array_ptr(inputs), inputs.nbytes)
            for sentinel, buffer in zip(sentinels, buffers[2:], strict=True):
                copy_host_to_device(buffer, host_array_ptr(sentinel), sentinel.nbytes)
            output_ptrs = [buffer.ptr + guard for buffer in buffers[2:]]
            producer(buffers[0].ptr, buffers[1].ptr, output_ptrs[0],
                     rows, hidden, experts, library=library)
            router.qwen35_router_select(*output_ptrs, rows, experts, experts,
                                       top_k, library=library)
            outputs = []
            for shape, dtype, nbytes, sentinel, buffer in zip(
                    shapes, dtypes, output_bytes, sentinels, buffers[2:], strict=True):
                raw = np.empty_like(sentinel)
                copy_device_to_host(host_array_ptr(raw), buffer, raw.nbytes)
                np.testing.assert_array_equal(raw[:guard], sentinel[:guard])
                np.testing.assert_array_equal(raw[guard + nbytes:], sentinel[guard + nbytes:])
                outputs.append(raw[guard:guard + nbytes].copy().view(dtype).reshape(shape))
            # Neither the producer nor selection may overwrite read-only inputs.
            for expected, buffer in zip((inputs, weight), buffers[:2], strict=True):
                observed = np.empty_like(expected)
                copy_device_to_host(host_array_ptr(observed), buffer, observed.nbytes)
                assert observed.tobytes() == expected.tobytes()
            return outputs

        parent = capture(x, router.qwen35_router_logits_f32_f32w_token_tile4_dense_exact)
        candidate = capture(x, router.qwen35_router_logits_f32_f32w_token_tile4_dpp_exact)
        for left, right in zip(parent, candidate, strict=True):
            assert left.tobytes() == right.tobytes()
            assert np.isfinite(right).all()
        # Change all neighbors, including rows in the same four-token tile.
        replaced = rng.normal(0, 10., x.shape).astype(np.float32)
        replaced[0] = x[0]
        neighbor = capture(replaced, router.qwen35_router_logits_f32_f32w_token_tile4_dpp_exact)
        for left, right in zip(candidate, neighbor, strict=True):
            assert left[0].tobytes() == right[0].tobytes()
        order = rng.permutation(rows)
        moved = capture(np.ascontiguousarray(x[order]),
                        router.qwen35_router_logits_f32_f32w_token_tile4_dpp_exact)
        for left, right in zip(candidate, moved, strict=True):
            assert left[order].tobytes() == right.tobytes()
        # Reuse the same allocations after both unrelated scenarios.
        again = capture(x, router.qwen35_router_logits_f32_f32w_token_tile4_dpp_exact)
        for left, right in zip(candidate, again, strict=True):
            assert left.tobytes() == right.tobytes()
    finally:
        for buffer in reversed(buffers):
            free(buffer)
