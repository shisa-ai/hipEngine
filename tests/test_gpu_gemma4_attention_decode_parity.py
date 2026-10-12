"""Decode-kernel parity with the block kernel on gfx1100.

The warp-32 decode kernel must be *bit-exact* with the original block kernel:
same products, same stride-tree order, same per-lane logit partitions, same
weighted-sum order. These tests launch both exported symbols on identical
inputs and compare outputs bitwise (uint16 for BF16, float equality for F32).
"""

import ctypes

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

@pytest.mark.parametrize("live_keys", [257, 4096, None], ids=["short-span", "window-4k", "full-context"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("tokens,keys", [(1, 15857), (3, 16640), (9, 16640), (2, 32771)])
def test_global_logits_preserve_strict_output_and_cpu_oracle(
    attention_library, dtype, head_dim, tokens, keys, live_keys,
):
    """Masked padding crosses LDS capacity without changing any live term.

    Holes, a displaced window, and an empty row exercise the exact mask ABI
    used by the layer's live-span materializer; poisoned masked K/V must not
    influence output. Short and long paths use the same 256-lane denominator.
    """
    from hipengine.core.memory import (
        malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        _SYMBOL_PREFILL_F32, _SYMBOL_PREFILL_BF16,
        gemma4_attention_prefill_f32, gemma4_attention_prefill_bf16,
        gemma4_attention_decode_variant,
    )
    rng = np.random.default_rng(4711)
    heads, kv_heads = 2, 1
    live_keys = keys if live_keys is None else live_keys
    q = rng.normal(0, .1, (tokens, heads, head_dim)).astype(np.float32)
    k = rng.normal(0, .1, (keys, kv_heads, head_dim)).astype(np.float32)
    v = rng.normal(0, .5, k.shape).astype(np.float32)
    mask = np.zeros((tokens, keys), np.uint8)
    mask[:, 32:live_keys] = 1
    mask[:, 47:73] = 0
    if tokens > 1:
        mask[-1] = 0
    # Masked tails contain NaNs: the strict skip must prevent propagation.
    k[live_keys:] = np.nan
    v[live_keys:] = np.nan
    cast = (lambda a: np.ascontiguousarray(a)) if dtype == "f32" else _bf16_bits
    q, k, v = map(cast, (q, k, v))
    launch = gemma4_attention_prefill_f32 if dtype == "f32" else gemma4_attention_prefill_bf16
    buffers = []

    def run(nkeys):
        arrays = [q, np.ascontiguousarray(k[:nkeys]), np.ascontiguousarray(v[:nkeys]),
                  np.ascontiguousarray(mask[:, :nkeys]), np.zeros_like(q)]
        local = [malloc(a.nbytes) for a in arrays]
        buffers.extend(local)
        for buf, array in zip(local[:4], arrays[:4]):
            copy_host_array_to_device(buf, array)
        if nkeys < keys:
            # Compare to the strict block oracle, not automatic decode's flash
            # variant (which changes arithmetic on short shape-admitted rows).
            symbol = _SYMBOL_PREFILL_F32 if dtype == "f32" else _SYMBOL_PREFILL_BF16
            _raw_launch(attention_library, symbol, local, tokens=tokens, keys=nkeys,
                        num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, scale=1.)
        else:
            launch(*(b.ptr for b in local), tokens=tokens, keys=nkeys, num_heads=heads,
                   num_kv_heads=kv_heads, head_dim=head_dim, scale=1., library=attention_library)
        copy_device_to_host(host_array_ptr(arrays[-1]), local[-1], arrays[-1].nbytes)
        return arrays[-1]

    try:
        short = run(live_keys) if live_keys < keys else None
        long = run(keys)
        assert gemma4_attention_decode_variant(attention_library) == "global_class"
        if short is not None:
            np.testing.assert_array_equal(short.view(np.uint32 if dtype == "f32" else np.uint16),
                                          long.view(np.uint32 if dtype == "f32" else np.uint16))
        def f32(a):
            return a if dtype == "f32" else (a.astype(np.uint32) << 16).view(np.float32)
        reference = np.zeros(q.shape, np.float32)
        for row in range(tokens):
            active = np.flatnonzero(mask[row, :live_keys])
            if not len(active):
                continue
            for head in range(heads):
                logits = f32(k)[active, 0] @ f32(q)[row, head]
                weights = np.exp(logits - logits.max())
                reference[row, head] = weights @ f32(v)[active, 0] / weights.sum()
        # The incumbent represents an entirely masked row as NaNs; exact
        # parity above pins that contract separately from finite live rows.
        live_rows = mask[:, :live_keys].any(axis=1)
        np.testing.assert_allclose(f32(long)[live_rows], reference[live_rows],
                                   rtol=.02 if dtype == "bf16" else 2e-5,
                                   atol=5e-4 if dtype == "bf16" else 1e-6)
        if not live_rows.all():
            assert np.isnan(f32(long)[~live_rows]).all()
        actual_rows = f32(long)[live_rows].reshape(-1, head_dim).astype(np.float64)
        expected_rows = reference[live_rows].reshape(-1, head_dim).astype(np.float64)
        def probabilities(rows):
            exp = np.exp(rows - rows.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
        teacher, candidate = probabilities(expected_rows), probabilities(actual_rows)
        kl = np.sum(teacher * np.log(teacher / candidate), axis=1)
        assert float(kl.max()) <= .05
        top_reference = expected_rows
        if dtype == "bf16":
            # CPU oracle at the declared BF16 output boundary: RNE ties select
            # the same first index as the device output, not an unrounded F32
            # winner hidden inside that tie.
            bits = reference[live_rows].copy().view(np.uint32)
            rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            top_reference = f32(rounded).reshape(-1, head_dim)
        assert np.mean(actual_rows.argmax(axis=1) == top_reference.argmax(axis=1)) >= .9
    finally:
        for buf in buffers:
            free(buf)

_SHAPES = [
    # (num_heads, num_kv_heads, head_dim, keys)
    (1, 1, 128, 1),  # single live key
    (1, 1, 128, 1040),  # metric-shape decode context
    (4, 2, 128, 1152),  # GQA
    (8, 4, 64, 300),  # head_dim 64 -> two chains
    (2, 1, 192, 64),  # non-power-of-two head_dim (M = 256, zero leaves)
    (2, 2, 512, 96),  # head_dim 512 -> kTile edge over the 256 thread cap
    (1, 1, 768, 64),  # head_dim above the 256-wide thread cap -> multi-term dots
    (1, 1, 6, 40),  # head_dim < 32 -> sub-wave workgroup (shfl-only tree)
    # Warp-per-head path (head_dim exactly one or two 256-lane tree widths).
    (16, 2, 256, 1024),  # the artifact's sliding-layer geometry
    (16, 8, 512, 1024),  # the artifact's full-layer geometry
    (16, 2, 256, 2055),  # odd key count past the kKeysPerTile tile edge
    (16, 2, 256, 8192),  # context cap: one logit row per block
]


# Split A/B shapes: the artifact's two real layer geometries plus a longer
# context and an odd key count at the slice boundary.
_SPLIT_SHAPES = [
    (16, 2, 256, 1024),  # sliding-layer geometry at the metric context
    (16, 8, 512, 1024),  # full-layer geometry at the metric context
    (16, 8, 512, 2055),  # odd key count, 4 slices, last slice short
    (16, 2, 256, 8192),  # context cap: 4 slices of 2048 keys
]


def _bf16_bits(array: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(array, dtype=np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )


@pytest.fixture(scope="module")
def attention_library():
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        build_gemma4_attention,
    )

    return build_gemma4_attention(load=True)


def _raw_launch(library, symbol, buffers, *, tokens, keys, num_heads, num_kv_heads,
                head_dim, scale, split_workspace=0, split_slices=1, live_extent=0,
                stream=0):
    from hipengine.core.ctypes_cache import signed_kernel_fn
    from hipengine.core.hip import HIP_SUCCESS
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        _ARGTYPES_DECODE,
        _ARGTYPES_PREFILL,
        _DECODE_SYMBOLS,
    )

    decode = symbol in _DECODE_SYMBOLS
    argtypes = _ARGTYPES_DECODE if decode else _ARGTYPES_PREFILL
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    args = [
        *(b.ptr for b in buffers[:5]),
        tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        ctypes.c_float(scale),
        stream,
        keys,
    ]
    if decode:
        # split_slices <= 1 keeps the single-kernel path; the pointer is only
        # dereferenced when the split actually runs. The live-extent slot is
        # the flash route's optional device pair; null keeps the scalar
        # partitioning these parity tests pin.
        args += [ctypes.c_void_p(split_workspace), ctypes.c_int(split_slices),
                 ctypes.c_void_p(live_extent)]
    else:
        # Prefill's tail is (window, row_offset) instead of decode's
        # (split_workspace, split_slices, live_extent). window 0 makes
        # first_kept 0, so the prefill symbol visits every key exactly as
        # decode's keep_mask does -- which is the parity these tests assert.
        args += [0, 0]
    err = fn(*args)
    assert int(err) == HIP_SUCCESS, f"{symbol} returned {err}"


def _run_both_symbols(library, *, dtype, num_heads, num_kv_heads, head_dim, keys,
                      mask_mode="keep", scale=1.0, seed=20260924):
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
        _SYMBOL_PREFILL_BF16,
        _SYMBOL_PREFILL_F32,
    )

    rng = np.random.default_rng(seed)
    storage = np.float32 if dtype == "f32" else np.uint16

    def cast(array):
        return np.ascontiguousarray(array, dtype=np.float32) if dtype == "f32" else _bf16_bits(array)

    query = cast(rng.standard_normal((1, num_heads, head_dim)) * 0.7)
    key = cast(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7)
    value = cast(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7)
    if mask_mode == "keep":
        mask = np.ones((1, keys), dtype=np.uint8)
    elif mask_mode == "holes":
        mask = (rng.random((1, keys)) < 0.7).astype(np.uint8)
        mask[0, 0] = 1  # at least one live key, as decode always keeps position 0's row
    elif mask_mode == "window":
        # A contiguous live span that starts after key 0 and ends before the
        # last key, so *both* of the class kernel's mask-derived bounds are
        # non-trivial. This is the mask the pass-3 lower bound reads:
        # ``keep`` begins at key 0 and ``holes`` is never a contiguous span,
        # so neither exercises a first kept key above 0.
        span = np.arange(keys, dtype=np.int64)
        mask = ((span >= keys // 4) & (span <= keys - 1 - keys // 8)).astype(np.uint8)
        assert mask.any(), "window mask must keep at least one key"
    elif mask_mode == "none":
        mask = np.zeros((1, keys), dtype=np.uint8)
    else:  # pragma: no cover - defensive
        raise ValueError(mask_mode)

    arrays = [query, key, value, mask]
    out_prefill = np.zeros_like(query, dtype=storage)
    out_decode = np.zeros_like(query, dtype=storage)
    arrays += [out_prefill, out_decode]

    buffers = []
    try:
        for array in arrays:
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            if array is not out_prefill and array is not out_decode:
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)

        shape = dict(
            tokens=1,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
        )
        prefill_symbol = _SYMBOL_PREFILL_F32 if dtype == "f32" else _SYMBOL_PREFILL_BF16
        decode_symbol = _SYMBOL_DECODE_F32 if dtype == "f32" else _SYMBOL_DECODE_BF16

        _raw_launch(library, prefill_symbol, buffers[:5], **shape)
        copy_device_to_host(host_array_ptr(out_prefill), buffers[4], out_prefill.nbytes)

        _raw_launch(library, decode_symbol,
                    [buffers[0], buffers[1], buffers[2], buffers[3], buffers[5]], **shape)
        copy_device_to_host(host_array_ptr(out_decode), buffers[5], out_decode.nbytes)

        return out_prefill, out_decode
    finally:
        for buffer in buffers:
            free(buffer)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("mask_mode", ["keep", "holes", "window"])
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,keys", _SHAPES)
def test_decode_symbol_bit_matches_prefill_symbol(
    attention_library, dtype, mask_mode, num_heads, num_kv_heads, head_dim, keys
):
    reference, candidate = _run_both_symbols(
        attention_library,
        dtype=dtype,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        keys=keys,
        mask_mode=mask_mode,
    )
    if dtype == "bf16":
        np.testing.assert_array_equal(reference, candidate)
    else:
        np.testing.assert_array_equal(reference, candidate)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_all_masked_row_matches(attention_library, dtype):
    # No live key: row_max is -inf and both kernels propagate NaN the same way.
    reference, candidate = _run_both_symbols(
        attention_library, dtype=dtype, num_heads=2, num_kv_heads=2, head_dim=128,
        keys=64, mask_mode="none",
    )
    np.testing.assert_array_equal(reference, candidate)


@pytest.mark.parametrize("scale", [1.0, 0.375])
def test_scaled_decode_matches(attention_library, scale):
    reference, candidate = _run_both_symbols(
        attention_library, dtype="f32", num_heads=1, num_kv_heads=1, head_dim=128,
        keys=517, mask_mode="holes", scale=scale,
    )
    np.testing.assert_array_equal(reference, candidate)


def test_public_wrapper_tokens_one_routes_to_decode_result(attention_library):
    """The wrapper a caller reaches must produce the block kernel's exact output.

    The unit tier proves ``tokens == 1`` selects the decode symbol; here we
    confirm the launched result through that wrapper is still bit-identical to
    the original block kernel on the same inputs.
    """
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        _SYMBOL_PREFILL_F32,
        _ARGTYPES_PREFILL,
        gemma4_attention_prefill_f32,
    )
    from hipengine.core.ctypes_cache import signed_kernel_fn
    from hipengine.core.hip import HIP_SUCCESS

    rng = np.random.default_rng(7)
    keys, heads, head_dim = 777, 2, 128
    query = np.ascontiguousarray(rng.standard_normal((1, heads, head_dim)), dtype=np.float32)
    key = np.ascontiguousarray(rng.standard_normal((keys, heads, head_dim)), dtype=np.float32)
    value = np.ascontiguousarray(rng.standard_normal((keys, heads, head_dim)), dtype=np.float32)
    mask = np.ones((1, keys), dtype=np.uint8)
    out_reference = np.zeros((1, heads, head_dim), dtype=np.float32)
    out_wrapper = np.zeros_like(out_reference)

    buffers = []
    try:
        for array in (query, key, value, mask, out_reference, out_wrapper):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)

        shape = dict(tokens=1, keys=keys, num_heads=heads, num_kv_heads=heads,
                     head_dim=head_dim, scale=1.0)
        fn = signed_kernel_fn(attention_library, _SYMBOL_PREFILL_F32, _ARGTYPES_PREFILL,
                              ctypes.c_int)
        # The prefill symbol's tail is (window, row_offset); 0/0 is what the
        # Python wrapper below defaults to, which is what this test compares
        # the raw symbol against.
        err = fn(*(b.ptr for b in buffers[:5]), 1, heads, heads, head_dim,
                 ctypes.c_float(1.0), 0, keys, 0, 0)
        assert int(err) == HIP_SUCCESS
        copy_device_to_host(host_array_ptr(out_reference), buffers[4], out_reference.nbytes)

        gemma4_attention_prefill_f32(buffers[0].ptr, buffers[1].ptr, buffers[2].ptr,
                                     buffers[3].ptr, buffers[5].ptr, **shape)
        copy_device_to_host(host_array_ptr(out_wrapper), buffers[5], out_wrapper.nbytes)

        np.testing.assert_array_equal(out_reference, out_wrapper)
    finally:
        for buffer in buffers:
            free(buffer)

@pytest.mark.parametrize(
    "head_dim,expected",    [
        (512, "class"),  # two 256-lane tree widths per row
        (256, "class"),  # the artifact's sliding-layer geometry
        (128, "block"),  # sub-256 geometry is outside the key-class kernel
        (768, "block"),  # three tree widths: no instantiation
    ],
)
def test_decode_variant_selection_is_by_geometry(attention_library, head_dim, expected):
    """The launcher's kernel choice is observable, not inferred from timings.

    All three paths are bit-identical, so a parity test cannot tell them apart; this
    asserts which one the launcher selected for the shape a caller reaches.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_decode_variant,
    )

    _run_both_symbols(
        attention_library,
        dtype="bf16",
        num_heads=16,
        num_kv_heads=2,
        head_dim=head_dim,
        keys=64,
        mask_mode="keep",
    )
    assert gemma4_attention_decode_variant(attention_library) == expected


# --- Two-phase split --------------------------------------------------------
#
# The split changes arithmetic *association*, not the summands: every weight it
# multiplies is the same f32 value the single-kernel path uses (there is no
# per-slice max and no second expf), so the only perturbation is the order of an
# f32 sum. Its contract is therefore agreement to float precision rather than
# the bit-equality the rest of this file asserts, and whether that perturbation
# stays inside the production kl_max bar is decided by the campaign gate, not
# here.


def _split_ab(library, *, dtype, num_heads, num_kv_heads, head_dim, keys, mask_mode,
              seed=20260925, split_slices=None, poison=False):
    """Run the incumbent single-kernel path and the split on identical inputs."""

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
        decode_slices,
        split_workspace_bytes,
    )

    slices = decode_slices(keys, head_dim) if split_slices is None else split_slices
    assert slices > 1, "shape is below the split's context threshold"

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
    else:
        mask = (rng.random((1, keys)) < 0.7).astype(np.uint8)
        mask[0, 0] = 1

    out_single = np.zeros((1, num_heads, head_dim), dtype=storage)
    out_split = np.zeros_like(out_single)
    arrays = [query, key, value, mask, out_single, out_split]

    symbol = _SYMBOL_DECODE_F32 if dtype == "f32" else _SYMBOL_DECODE_BF16
    shape = dict(
        tokens=1,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=1.0,
    )
    buffers = []
    workspace = None
    try:
        for array in arrays:
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            if array is not out_single and array is not out_split:
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)

        _raw_launch(library, symbol, buffers[:5], split_slices=1, **shape)
        copy_device_to_host(host_array_ptr(out_single), buffers[4], out_single.nbytes)

        workspace = malloc(
            split_workspace_bytes(1, num_heads, head_dim, keys, slices, library=library)
        )
        if poison:
            sentinel = np.full(workspace.nbytes // 4, np.nan, dtype=np.float32)
            copy_host_to_device(workspace, host_array_ptr(sentinel), sentinel.nbytes)
        _raw_launch(
            library,
            symbol,
            [buffers[0], buffers[1], buffers[2], buffers[3], buffers[5]],
            split_workspace=workspace.ptr,
            split_slices=slices,
            **shape,
        )
        copy_device_to_host(host_array_ptr(out_split), buffers[5], out_split.nbytes)

        def as_float(array):
            if dtype == "f32":
                return array
            return (array.astype(np.uint32) << 16).view(np.float32)

        return as_float(out_single), as_float(out_split), slices
    finally:
        if workspace is not None:
            free(workspace)
        for buffer in buffers:
            free(buffer)


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("slices", [2, 4, 32])
def test_dimension_workspace_has_no_key_slice_partials(attention_library, head_dim, slices):
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import split_workspace_bytes
    assert split_workspace_bytes(1, 4, head_dim, 1057, slices, library=attention_library) == 4 * (1057 + 2) * 4


@pytest.mark.parametrize("head_dim,keys,slices", [(256, 1057, 4), (512, 2055, 8), (256, 17, 32), (256, 1023, 2), (512, 1025, 4), (512, 777, 2)])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_dimension_partition_preserves_single_accumulation(attention_library, head_dim, keys, slices, dtype):
    reference, candidate, _ = _split_ab(
        attention_library, dtype=dtype, num_heads=4, num_kv_heads=2,
        head_dim=head_dim, keys=keys, mask_mode="holes", split_slices=slices,
        poison=True,
    )
    np.testing.assert_array_equal(candidate, reference)
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_decode_variant
    assert gemma4_attention_decode_variant(attention_library) == "split"


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,keys", _SPLIT_SHAPES)
def test_split_matches_single_kernel_within_float_precision(
    attention_library, num_heads, num_kv_heads, head_dim, keys
):
    reference, candidate, slices = _split_ab(
        attention_library,
        dtype="f32",
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        keys=keys,
        mask_mode="holes",
    )
    assert slices == decode_slices_for(keys, head_dim)
    np.testing.assert_allclose(candidate, reference, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,keys", _SPLIT_SHAPES)
def test_split_bf16_matches_single_kernel_exactly(
    attention_library, num_heads, num_kv_heads, head_dim, keys
):
    """Dimension partitioning preserves the single-kernel accumulation order."""

    reference, candidate, _ = _split_ab(
        attention_library,
        dtype="bf16",
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        keys=keys,
        mask_mode="keep",
    )
    np.testing.assert_array_equal(candidate, reference)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("keys,slices", [(1, 4), (5, 4), (17, 8)])
def test_short_dimension_pass_ignores_poisoned_workspace(attention_library, dtype, head_dim, keys, slices):
    reference, candidate, _ = _split_ab(
        attention_library, dtype=dtype, num_heads=2, num_kv_heads=1,
        head_dim=head_dim, keys=keys, mask_mode="keep",
        split_slices=slices, poison=True,
    )
    assert np.isfinite(candidate).all()
    np.testing.assert_allclose(candidate, reference, rtol=1e-2, atol=1e-3)


def decode_slices_for(keys, head_dim):
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import decode_slices

    return decode_slices(keys, head_dim)
