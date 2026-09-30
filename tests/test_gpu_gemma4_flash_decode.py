"""Flash-decoding battery for the Gemma 4 sliding-layer decode path (D2).

The flash route changes arithmetic *association*: each key slice softmaxes
against its own slice maximum and a combine kernel rescales the partials, so
the contract is agreement with the incumbent single kernel to float precision
plus the production teacher-forced bars at promotion -- not the bit equality
`test_gpu_gemma4_attention_decode_parity.py` pins for the split. What the
battery pins here:

- the launcher selects ``"flash"`` for the admitted sliding geometry and
  only there (capability: head_dim 256, GQA ratio 2, one token, workspace),
  including through the production wrapper the engine registers -- not just
  the raw ABI;
- outputs match the single kernel to a measured float-precision bound over
  keep / holes / window masks and both dtypes;
- the all-masked row propagates NaN exactly as the incumbent does;
- a poisoned workspace cannot leak into the result (every partial slot is
  written before the combine reads it);
- every non-admitted geometry falls through to the incumbent chain.
"""

import ctypes

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

# The artifact's real sliding-layer geometry: q=16, kv=8, head_dim=256.
_SLIDING = dict(num_heads=16, num_kv_heads=8, head_dim=256)
# Capability misses that must keep the incumbent routing.
_NON_FLASH = [
    (16, 2, 512, "global-layer head_dim is a different template"),
    (16, 2, 256, "GQA ratio 8 is outside the kernel's ratio-2 packing"),
    (16, 16, 256, "MHA ratio 1 is outside the kernel's ratio-2 packing"),
]


def _bf16_bits(array: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(array, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _as_float(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "f32":
        return array
    return (array.astype(np.uint32) << 16).view(np.float32)


@pytest.fixture(scope="module")
def attention_library():
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        build_gemma4_attention,
    )

    return build_gemma4_attention(load=True)


def _capability_surface():
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        flash_admits,
        flash_slices,
        flash_workspace_bytes,
    )

    return flash_admits, flash_slices, flash_workspace_bytes


def test_capability_surface_and_slice_formula(attention_library):
    """RED surface: admission, slice policy, and workspace sizing exist.

    The slice formula is Python's (the single source of truth the host
    wrapper trusts through the negative-slices request), so it is pinned
    here: 32 at the 1024 entry threshold, capped at 64, never below 4.
    """
    flash_admits, flash_slices, flash_workspace_bytes = _capability_surface()
    assert flash_admits(tokens=1, head_dim=256, num_heads=16, num_kv_heads=8)
    assert not flash_admits(tokens=2, head_dim=256, num_heads=16, num_kv_heads=8)
    for num_heads, num_kv_heads, head_dim, _ in _NON_FLASH:
        assert not flash_admits(
            tokens=1, head_dim=head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads
        )
    assert flash_slices(1024) == 32
    assert flash_slices(4096) == 64
    assert flash_slices(1 << 18) == 64
    assert flash_slices(1025) == 33
    # (tokens, heads, head_dim, slices) -> tokens*heads*slices*(2+head_dim)*4
    assert flash_workspace_bytes(1, 16, 256, 32) == 1 * 16 * 32 * (2 + 256) * 4


@pytest.mark.parametrize("keys", [1024, 2055, 4096])
def test_flash_selected_for_admitted_sliding_geometry(attention_library, keys):
    out_single, out_flash, slices = _flash_ab(
        attention_library, dtype="f32", mask_mode="holes", keys=keys, **_SLIDING
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_decode_variant,
    )

    assert slices >= 2
    assert gemma4_attention_decode_variant(attention_library) == "flash"
    assert out_flash.shape == out_single.shape


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("mask_mode", ["keep", "holes", "window"])
@pytest.mark.parametrize("keys", [1024, 2055, 4096])
def test_flash_matches_single_kernel_to_float_precision(
    attention_library, dtype, mask_mode, keys
):
    out_single, out_flash, _ = _flash_ab(
        attention_library, dtype=dtype, mask_mode=mask_mode, keys=keys, **_SLIDING
    )
    reference = _as_float(out_single, dtype)
    candidate = _as_float(out_flash, dtype)
    # f32: the rescale identity is exact in reals; only f32 rounding differs.
    # bf16: both arms round to bf16, so the bound is one bf16 quantum of head.
    if dtype == "f32":
        np.testing.assert_allclose(candidate, reference, rtol=1e-4, atol=1e-4, equal_nan=True)
    else:
        np.testing.assert_allclose(candidate, reference, rtol=6e-3, atol=6e-3, equal_nan=True)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_flash_all_masked_row_propagates_nan(attention_library, dtype):
    # The incumbent row_max is -inf on an all-masked row, so expf(-inf - -inf)
    # makes every weight NaN and the output NaN. Flash must agree, NaN for NaN.
    out_single, out_flash, _ = _flash_ab(
        attention_library, dtype=dtype, mask_mode="none", keys=1024, **_SLIDING
    )
    reference = _as_float(out_single, dtype)
    candidate = _as_float(out_flash, dtype)
    assert np.isnan(reference).all()
    np.testing.assert_array_equal(candidate, reference)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("keys", [1024, 2055])
def test_flash_ignores_poisoned_workspace(attention_library, dtype, keys):
    out_single, out_flash, _ = _flash_ab(
        attention_library, dtype=dtype, mask_mode="keep", keys=keys, poison=True, **_SLIDING
    )
    reference = _as_float(out_single, dtype)
    candidate = _as_float(out_flash, dtype)
    assert np.isfinite(candidate).all() or np.isnan(reference).all()
    if dtype == "f32":
        np.testing.assert_allclose(candidate, reference, rtol=1e-4, atol=1e-4, equal_nan=True)
    else:
        np.testing.assert_allclose(candidate, reference, rtol=6e-3, atol=6e-3, equal_nan=True)


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,reason", _NON_FLASH)
def test_non_admitted_geometry_keeps_incumbent_routing(
    attention_library, num_heads, num_kv_heads, head_dim, reason
):
    _, _, slices = _flash_ab(
        attention_library,
        dtype="f32",
        mask_mode="keep",
        keys=1024,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_decode_variant,
    )

    # A negative-slices flash request the host does not admit must fall
    # through to the incumbent split, never silently to a third behaviour.
    assert gemma4_attention_decode_variant(attention_library) == "split"
    assert slices >= 2


def test_production_wrapper_selects_flash_for_admitted_decode(attention_library):
    """The registered attention entry -- what the engine calls for a decode
    step -- must take the flash route for the admitted sliding geometry.

    The raw-ABI selection the other tests pin is not the route a request
    takes; this runs the wrapper's own admission (workspace sizing, negative
    slices request) and introspects the launcher's selection after a real
    launch.
    """
    from hipengine.core.memory import (
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        decode_selection,
        gemma4_attention_prefill_f32,
    )

    keys = 1024
    num_heads, num_kv_heads, head_dim = 16, 8, 256
    rng = np.random.default_rng(20260930)
    arrays = [
        np.ascontiguousarray(rng.standard_normal((1, num_heads, head_dim)) * 0.7, dtype=np.float32),
        np.ascontiguousarray(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7, dtype=np.float32),
        np.ascontiguousarray(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7, dtype=np.float32),
        np.ones((1, keys), dtype=np.uint8),
        np.zeros((1, num_heads, head_dim), dtype=np.float32),
    ]
    buffers = []
    try:
        for array in arrays:
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            if array is not arrays[-1]:
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        gemma4_attention_prefill_f32(
            *(b.ptr for b in buffers),
            tokens=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=1.0,
            keys=keys,
            library=attention_library,
        )
        assert decode_selection(attention_library) == 3, (
            "the production wrapper must select flash for the admitted "
            "sliding geometry"
        )
    finally:
        for buffer in buffers:
            free(buffer)


def _flash_ab(
    library,
    *,
    dtype,
    mask_mode,
    keys,
    num_heads,
    num_kv_heads,
    head_dim,
    seed=20260930,
    scale=1.0,
    poison=False,
):
    """Run the incumbent single-kernel path and the flash route on one input."""
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        _SYMBOL_DECODE_BF16,
        _SYMBOL_DECODE_F32,
        flash_slices,
        flash_workspace_bytes,
    )
    from tests.test_gpu_gemma4_attention_decode_parity import _raw_launch

    slices = flash_slices(keys)
    rng = np.random.default_rng(seed)
    storage = np.float32 if dtype == "f32" else np.uint16

    def cast(array):
        if dtype == "f32":
            return np.ascontiguousarray(array, dtype=np.float32)
        return _bf16_bits(array)

    query = cast(rng.standard_normal((1, num_heads, head_dim)) * 0.7)
    key = cast(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7)
    value = cast(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7)
    if mask_mode == "keep":
        mask = np.ones((1, keys), dtype=np.uint8)
    elif mask_mode == "none":
        mask = np.zeros((1, keys), dtype=np.uint8)
    elif mask_mode == "window":
        span = np.arange(keys, dtype=np.int64)
        mask = ((span >= keys // 4) & (span <= keys - 1 - keys // 8)).astype(np.uint8)
        assert mask.any(), "window mask must keep at least one key"
    else:
        mask = (rng.random((1, keys)) < 0.7).astype(np.uint8)
        mask[0, 0] = 1

    out_single = np.zeros((1, num_heads, head_dim), dtype=storage)
    out_flash = np.zeros_like(out_single)
    arrays = [query, key, value, mask, out_single, out_flash]
    symbol = _SYMBOL_DECODE_F32 if dtype == "f32" else _SYMBOL_DECODE_BF16
    shape = dict(
        tokens=1,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
    )
    buffers = []
    workspace = None
    try:
        for array in arrays:
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            if array is not out_single and array is not out_flash:
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)

        _raw_launch(library, symbol, buffers[:5], split_slices=1, **shape)
        copy_device_to_host(host_array_ptr(out_single), buffers[4], out_single.nbytes)

        workspace = malloc(flash_workspace_bytes(1, num_heads, head_dim, slices, library=library))
        if poison:
            sentinel = np.full(workspace.nbytes // 4, np.nan, dtype=np.float32)
            copy_host_to_device(workspace, host_array_ptr(sentinel), sentinel.nbytes)
        _raw_launch(
            library,
            symbol,
            [buffers[0], buffers[1], buffers[2], buffers[3], buffers[5]],
            split_workspace=workspace.ptr,
            split_slices=-slices,
            **shape,
        )
        copy_device_to_host(host_array_ptr(out_flash), buffers[5], out_flash.nbytes)
        return out_single, out_flash, slices
    finally:
        if workspace is not None:
            free(workspace)
        for buffer in buffers:
            free(buffer)