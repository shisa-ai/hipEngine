"""Raw-coordinate and grid bounds for the plain Gemma 4 prefill kernel.

The plain block kernel is reachable through the prefill symbol with
``tokens == 1``: the Python wrapper routes a one-token launch to the decode
symbol, and multi-token launches into the decode launcher, so these tests drive
``hipengine_gemma4_attention_prefill_f32`` directly. That is also the surface a
diagnostic script or a raw ctypes caller reaches.

The contract is the staged family's, applied to the strict kernel:

* An extreme ``window`` or ``row_offset`` narrows the walk into ``[0, keys]``.
  It never wraps into a negative column index. A window that does not intersect
  the key range is an empty walk and writes NaN, the same result an all-masked
  row produces.
* An unrepresentable ``tokens * num_heads`` grid is refused with
  ``hipErrorInvalidValue`` before a launch.

RED before the 2026-10-26 fix: the bounds were narrowed through a 32-bit cast
without clamping, so a huge coordinate could land ``first_kept`` and
``walk_end`` on opposite sides of zero and walk backwards over the allocation
(the recorded reproducer faulted the GPU queue), and the unsigned grid product
could wrap. GREEN after it. The RED reproduction is the recorded raw-launch
case behind the R0053 finding; it is deliberately not re-run here.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

from hipengine.core.ctypes_cache import signed_kernel_fn  # noqa: E402
from hipengine.core.hip import HIP_SUCCESS  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (  # noqa: E402
    _ARGTYPES_PREFILL,
    _SYMBOL_PREFILL_F32,
    build_gemma4_attention,
)
from tests.test_unit_gemma4_attention_staged import staged_cpu_oracle  # noqa: E402

# HIP's ``hipErrorInvalidValue``. The launcher's shape and grid checks return it
# before any kernel is enqueued.
_HIP_ERROR_INVALID_VALUE = 1


@pytest.fixture(scope="module")
def attention_library():
    return build_gemma4_attention(load=True)


# (tokens, keys, num_heads, num_kv_heads, head_dim, window, row_offset, expected)
_HUGE_COORDINATE_SHAPES = [
    # One past INT32_MAX: the old cast wrapped the row's own position negative.
    (1, 8, 2, 1, 256, 4, 2**31, "empty"),
    # A saturated sum, not a wrapped one.
    (1, 8, 2, 1, 256, 4, 2**62, "empty"),
    # Far below zero: the clamped window is empty at the other end.
    (1, 8, 2, 1, 256, 4, -(2**62), "empty"),
    # 2**32 leaves the narrowed bounds on opposite sides of zero, which is the
    # shape that walked backwards over the allocation before the clamp.
    (1, 8, 2, 1, 256, 4, 2**32, "empty"),
    # A window as wide as the offset: the intersection is [1, 8), so this is a
    # real attention result rather than a NaN.
    (1, 8, 2, 1, 256, 2**40, 2**40, "clamped"),
    # A window this close to the int64 limit keeps `own - window` exact while
    # `own` alone is already saturated.
    (1, 8, 2, 1, 256, 2**63 - 1, 2**63 - 1, "clamped"),
    # head_dim 6 takes the block-sum path, which computes the same bounds.
    (1, 8, 2, 1, 6, 2**40, 2**40, "clamped"),
]


def _run_plain(
    library,
    *,
    tokens,
    keys,
    num_heads,
    num_kv_heads,
    head_dim,
    window,
    row_offset,
    mask,
):
    """Launch the raw prefill symbol once and return its output and inputs."""

    rng = np.random.default_rng(0)
    query = rng.standard_normal((tokens, num_heads, head_dim), dtype=np.float32)
    key = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32)
    value = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32)
    out = np.empty_like(query)
    buffers = []
    try:
        for array in (query, key, value, mask, out):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        fn = signed_kernel_fn(
            library, _SYMBOL_PREFILL_F32, _ARGTYPES_PREFILL, ctypes.c_int
        )
        err = fn(
            *(buffer.ptr for buffer in buffers),
            tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            ctypes.c_float(1.0),
            0,
            keys,
            window,
            row_offset,
        )
        assert int(err) == HIP_SUCCESS, f"prefill returned {err}"
        copy_device_to_host(host_array_ptr(out), buffers[-1], out.nbytes)
        return out.copy(), query, key, value
    finally:
        for buffer in buffers:
            free(buffer)


@pytest.mark.parametrize(
    "tokens,keys,num_heads,num_kv_heads,head_dim,window,row_offset,expected",
    _HUGE_COORDINATE_SHAPES,
)
def test_huge_coordinates_stay_inside_the_key_range(
    attention_library,
    tokens,
    keys,
    num_heads,
    num_kv_heads,
    head_dim,
    window,
    row_offset,
    expected,
):
    """An extreme window or row offset clamps; it never indexes past the keys."""

    if expected == "empty":
        mask = np.zeros((tokens, keys), dtype=np.uint8)
    else:
        # Python integers, so building the mask cannot overflow either.
        mask = np.array(
            [
                [int(row_offset + t - window < j <= row_offset + t) for j in range(keys)]
                for t in range(tokens)
            ],
            dtype=np.uint8,
        )

    out, query, key, value = _run_plain(
        attention_library,
        tokens=tokens,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        window=window,
        row_offset=row_offset,
        mask=mask,
    )

    if expected == "empty":
        assert np.isnan(out).all()
        return

    oracle = staged_cpu_oracle(
        query,
        key,
        value,
        mask,
        num_kv_heads=num_kv_heads,
        scale=1.0,
        window=window,
        row_offset=row_offset,
    )
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, oracle, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("tokens,num_heads", [(2**31, 2), (2, 2**31 + 1)])
def test_an_unrepresentable_grid_is_refused_before_launch(
    attention_library, tokens, num_heads
):
    """A grid past INT32_MAX returns hipErrorInvalidValue, not a wrapped launch."""

    fn = signed_kernel_fn(
        attention_library, _SYMBOL_PREFILL_F32, _ARGTYPES_PREFILL, ctypes.c_int
    )
    # Null pointers are safe: the grid check returns before the launch, and the
    # kernel never runs.
    err = fn(
        0,
        0,
        0,
        0,
        0,
        tokens,
        num_heads,
        1,
        256,
        ctypes.c_float(1.0),
        0,
        8,
        0,
        0,
    )
    assert int(err) == _HIP_ERROR_INVALID_VALUE, f"expected invalid value, got {err}"

    # The Python wrapper is not a second surface for this check: it selects the
    # decode symbol for every row count, and the decode launcher is a separate
    # path. The raw prefill symbol is the only caller of the launcher this unit
    # validated, so it is the one the test drives.
