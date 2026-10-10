"""The packed-hidden gather writes the same bytes in its row form and its element form.

``hipengine_qwen35_moe_gather_packed_hidden_lowp`` picks a row-block kernel that
moves eight BF16 elements per thread when the row width is a whole number of
128-bit spans and both buffers are 16-byte aligned, and keeps the per-element
kernel otherwise. Both are plain copies, so the contract is byte equality with a
host gather for every shape, including dead lanes and out-of-range tokens, and
bytes outside the written range must be left alone.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)

_GUARD = 0xBEEF


def _require_hip() -> None:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:  # pragma: no cover - depends on local ROCm install
        pytest.skip(f"HIP runtime unavailable: {exc}")


def _expected(hidden: np.ndarray, lanes: np.ndarray, tokens: int, top_k: int) -> np.ndarray:
    out = np.zeros((lanes.size, hidden.shape[1]), dtype=np.uint16)
    for row, lane in enumerate(lanes):
        token = int(lane) // top_k if lane >= 0 else -1
        if 0 <= token < tokens:
            out[row] = hidden[token]
    return out


@pytest.mark.parametrize(
    ("tokens", "top_k", "hidden_size", "byte_offset"),
    [
        # Row form: Gemma 4's own shape and two neighbours either side of the
        # 128-thread span count (2816 / 8 = 352 spans).
        (1024, 8, 2816, 0),
        (513, 8, 2816, 0),
        (7, 8, 2816, 0),
        (64, 10, 2560, 0),
        (33, 4, 8, 0),      # one span per row
        (33, 4, 1032, 0),   # 129 spans: one more than the block width
        # Element form: a width that is not a whole number of spans, and an
        # aligned width on a buffer that is not 16-byte aligned.
        (33, 4, 2814, 0),
        (33, 4, 2816, 2),
    ],
)
def test_gather_matches_a_host_gather(tokens, top_k, hidden_size, byte_offset) -> None:
    _require_hip()
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_gather_packed_hidden_lowp,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(tokens * 131 + hidden_size + byte_offset)
    hidden = rng.integers(0, 1 << 16, size=(tokens, hidden_size), dtype=np.uint16)
    rows = tokens * top_k
    lanes = rng.permutation(rows).astype(np.int64)
    # Dead lanes and lanes naming a token past the end both gather zeros.
    lanes[rng.choice(rows, size=max(1, rows // 16), replace=False)] = -1
    lanes[rng.choice(rows, size=max(1, rows // 32), replace=False)] = rows + 5 * top_k

    pad = byte_offset // 2
    guard = 8
    packed_host = np.full(pad + rows * hidden_size + guard, _GUARD, dtype=np.uint16)
    hidden_host = np.concatenate(
        [np.full(pad, _GUARD, dtype=np.uint16), hidden.reshape(-1)]
    )
    allocations = []
    try:
        d_hidden = malloc(hidden_host.nbytes, runtime=runtime)
        allocations.append(d_hidden)
        copy_host_to_device(d_hidden, host_array_ptr(hidden_host), runtime=runtime)
        d_lanes = malloc(lanes.nbytes, runtime=runtime)
        allocations.append(d_lanes)
        copy_host_to_device(d_lanes, host_array_ptr(lanes), runtime=runtime)
        d_packed = malloc(packed_host.nbytes, runtime=runtime)
        allocations.append(d_packed)
        copy_host_to_device(d_packed, host_array_ptr(packed_host), runtime=runtime)
        assert d_hidden.ptr % 16 == 0 and d_packed.ptr % 16 == 0

        qwen35_moe_gather_packed_hidden_lowp(
            d_hidden.ptr + byte_offset, d_lanes.ptr, d_packed.ptr + byte_offset,
            rows * hidden_size, tokens, top_k, hidden_size, runtime=runtime,
        )
        runtime.device_synchronize()
        result = np.empty_like(packed_host)
        copy_device_to_host(host_array_ptr(result), d_packed, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    body = result[pad:pad + rows * hidden_size].reshape(rows, hidden_size)
    np.testing.assert_array_equal(body, _expected(hidden, lanes, tokens, top_k))
    # Nothing before the first row or past the last one is written.
    assert np.all(result[:pad] == _GUARD)
    assert np.all(result[pad + rows * hidden_size:] == _GUARD)
