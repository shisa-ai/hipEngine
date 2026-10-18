"""Attention reduction geometry regressions on gfx1100."""

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")


@pytest.mark.parametrize("head_dim", [6, 48, 64, 96, 192, 256, 512])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("keys", [3, 8192, 15616])
def test_uniform_attention_preserves_unit_values(head_dim, dtype, keys):
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32, gemma4_attention_prefill_bf16,
    )

    storage = np.float32 if dtype == "f32" else np.uint16
    one = 1.0 if dtype == "f32" else 0x3F80
    query = np.zeros((1, 1, head_dim), dtype=storage)
    key = np.zeros((keys, 1, head_dim), dtype=storage)
    value = np.full_like(key, one)
    mask = np.ones((1, keys), dtype=np.uint8)
    output = np.empty_like(query)
    buffers = []
    try:
        for array in (query, key, value, mask, output):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        launch = gemma4_attention_prefill_f32 if dtype == "f32" else gemma4_attention_prefill_bf16
        launch(*(b.ptr for b in buffers), tokens=1, keys=keys, num_heads=1,
               num_kv_heads=1, head_dim=head_dim, scale=1.0)
        copy_device_to_host(host_array_ptr(output), buffers[-1], output.nbytes)
        # Zero scores give equal weights. Their weighted average must be one,
        # independently of reduction block width or output representation.
        np.testing.assert_array_equal(output, np.full_like(output, one))
    finally:
        for buffer in buffers:
            free(buffer)


@pytest.mark.parametrize("head_dim", [6, 256])
@pytest.mark.parametrize("window", [1, 4])
def test_a_narrowed_prefill_walk_is_bit_identical_to_a_full_walk(head_dim, window):
    """Skipping a row's leading masked columns must not change the output.

    A sliding row's mask is zero for every column below ``max(0, row_offset +
    row - window + 1)``, so the kernel may start its walk there. The dropped
    columns contribute ``expf(-inf) == 0`` to both reductions and the surviving
    terms keep their lane assignment and their order, so the result is
    bit-identical rather than merely close. That is what lets the per-row bound
    ship without a numerical gate, and it is the whole safety argument.

    ``head_dim`` 256 takes the warp-per-key path and 6 takes the block-sum path,
    which start their walks differently and have to agree independently.
    """

    from hipengine.core.memory import (
        malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32,
    )

    tokens, keys = 8, 16
    rng = np.random.default_rng(0)
    query = rng.standard_normal((tokens, 1, head_dim), dtype=np.float32)
    key = rng.standard_normal((keys, 1, head_dim), dtype=np.float32)
    value = rng.standard_normal((keys, 1, head_dim), dtype=np.float32)
    # A sliding causal mask, in the shape ``_keep_mask`` builds: column 0 is key
    # ``key_begin`` (here 0), so row t's own bound is max(0, t - window + 1).
    mask = np.array(
        [[np.uint8(c <= t and t - c < window) for c in range(keys)] for t in range(tokens)],
        dtype=np.uint8,
    )
    assert not mask[0, :].any() is None  # mask is real, not all-zero

    def run(narrowed):
        output = np.empty_like(query)
        buffers = []
        try:
            for array in (query, key, value, mask, output):
                buffer = malloc(array.nbytes)
                buffers.append(buffer)
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
            gemma4_attention_prefill_f32(
                *(b.ptr for b in buffers), tokens=tokens, keys=keys, num_heads=1,
                num_kv_heads=1, head_dim=head_dim, scale=1.0,
                window=window if narrowed else 0, row_offset=0,
            )
            copy_device_to_host(host_array_ptr(output), buffers[-1], output.nbytes)
            return output.copy()
        finally:
            for buffer in buffers:
                free(buffer)

    full = run(narrowed=False)
    narrowed = run(narrowed=True)
    # Bit-identical, not approximately: assert_array_equal compares raw floats.
    np.testing.assert_array_equal(narrowed, full)
