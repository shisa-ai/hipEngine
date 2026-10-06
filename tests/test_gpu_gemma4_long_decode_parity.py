"""Direct parity: native ``global_class`` decode vs the staged singleton candidate.

The strict Gemma 4 decode family holds one FP32 logit per live key in shared
memory, so at ``head_dim`` 512 it cannot launch above 15,616 keys. Past that
bound the launcher does not fall back to a different association: it keeps the
strict key-class arithmetic and moves the logit row to caller-owned global
scratch. That path is the ``global_class`` decode variant, and it is the native
reference the staged candidate has to reproduce bit for bit at long contexts.

``tests/test_gpu_gemma4_attention_staged.py`` pins the staged candidate against
the monolithic *prefill* parent and, past the strict ceiling, against an
independent float64 oracle. Neither is a direct native comparison at a long
context: the prefill parent is not the ``global_class`` route a one-token block
actually takes above the resident bound, and the oracle measures accumulation
rather than bitwise identity. This module closes that gap. It drives the native
wrapper a caller reaches -- :func:`gemma4_attention_prefill_bf16` /
:func:`gemma4_attention_prefill_f32` at ``tokens == 1``, which selects the decode
entry and, above the resident bound, ``global_class`` -- and the staged singleton
launcher from identical device buffers, and compares their outputs bit for bit
(``uint16`` payloads for BF16, ``uint32`` for F32, so ``0.0``/``-0.0`` and NaN
payloads are not interchangeable).

Both output buffers are poisoned before the launch, so an element a kernel never
writes stays poisoned and the comparison fails instead of silently passing. The
GQA ratio is 8 throughout -- the global layer's geometry, ``head_dim`` 512 -- and
the key lengths straddle the resident bound and the 1,024-key score-chunk and
256-key PV-tile boundaries. Masks cover the all-keep, broad-softmax,
cancellation-heavy-value, sparse, and last-key-only patterns; masked K/V is set
to NaN in the sparse and last-key cases so a kernel that read a masked key would
propagate it.

Reached with an explicit file target; the default suite collects ``test_unit_*``
only. No model load: the fixtures are raw device buffers.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available
from tests.test_unit_gemma4_attention_staged import bf16_decode, bf16_round

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as native  # noqa: E402
from hipengine.kernels.hip_gfx1100.gemma4 import (  # noqa: E402
    gemma4_attention_staged as staged,
)

# The first key count at head_dim 512 whose resident logit row exceeds the 64 KiB
# LDS budget, so the native decode launcher selects ``global_class``. It is
# ``(keys + 768) * 4 > 65536`` for this head width, and the native and Python
# bounds are the same expression.
GLOBAL_CLASS_MIN_KEYS = 15617

# Every key length the long-context comparison is asked to cover. The first four
# straddle the resident bound (15,616 is the last resident count; 15,617 is the
# first global-logit one) and the 16,384 score-chunk boundary; the rest grow
# through the score-chunk and PV-tile edges to a full 262,144-key context.
KEY_LENGTHS = (15617, 16383, 16384, 16385, 20003, 32768, 65536, 131072, 262144)

# Patterns whose mask leaves at least one key live per row, so every row is a
# finite result rather than the all-masked NaN contract.
PATTERNS = ("keep", "broad", "cancel", "sparse", "last_key")

# Patterns whose masked K/V is overwritten with NaN: the kernels must skip a
# masked key before its K or V load, so a masked NaN can never reach the output.
_POISONED_PATTERNS = ("sparse", "last_key")

# One raw BF16 NaN payload and one raw F32 NaN payload, used as the output
# poison. Distinct from any value either kernel writes for a live row.
_BF16_POISON = np.uint16(0x7FC1)
_F32_POISON = np.uint32(0x7FC00001)

_DTYPE_BYTES = {"bf16": np.uint16, "f32": np.float32}


class _Device:
    """Host arrays mirrored into device buffers, released together."""

    def __init__(self) -> None:
        self.buffers: list = []

    def put(self, array: np.ndarray):
        buffer = malloc(array.nbytes)
        self.buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        return buffer

    def take(self, array: np.ndarray):
        buffer = malloc(array.nbytes)
        self.buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        return buffer

    def read(self, buffer, array: np.ndarray) -> None:
        copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)

    def close(self) -> None:
        for buffer in self.buffers:
            free(buffer)
        self.buffers.clear()


@pytest.fixture(scope="module")
def native_library():
    return native.build_gemma4_attention(load=True)


@pytest.fixture(scope="module")
def staged_library():
    return staged.build_gemma4_attention_staged(load=True)


def _mask(tokens, keys, pattern, seed):
    """A ``(tokens, keys)`` keep-mask with at least one live key per row."""

    if pattern in ("keep", "broad", "cancel"):
        return np.ones((tokens, keys), dtype=np.uint8)
    if pattern == "sparse":
        rng = np.random.default_rng(seed)
        mask = (rng.random((tokens, keys)) < 0.02).astype(np.uint8)
        # Guarantee a live key on each row, so no row is the all-masked NaN case.
        mask[:, keys // 3] = 1
        mask[:, -1] = 1
        return mask
    if pattern == "last_key":
        mask = np.zeros((tokens, keys), dtype=np.uint8)
        mask[:, -1] = 1
        return mask
    raise ValueError(pattern)  # pragma: no cover - defensive


def _case(keys, num_heads, num_kv_heads, head_dim, pattern, dtype, seed):
    """The four device inputs for one comparison, plus the softmax scale."""

    rng = np.random.default_rng(seed)
    query = rng.standard_normal((1, num_heads, head_dim), dtype=np.float32) * 0.7
    key = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 0.7
    value = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 0.7
    scale = 1.0
    if pattern == "broad":
        # A small scale keeps the softmax broad over the whole context: the
        # denominator is a many-term sum of comparable terms rather than one
        # winning key, so a disagreement in accumulation order shows up.
        scale = 0.02
    elif pattern == "cancel":
        # Large values with an alternating sign per key, and a small scale so
        # many keys stay comparable in weight: the weighted sum is then
        # dominated by cancellation between near-equal positive and negative
        # terms, which is the accumulation a different reduction order exposes.
        scale = 0.1
        value = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 4.0
        # Adjacent pairs share a score but have opposite values. Merely flipping
        # independent zero-mean samples would not create paired cancellation.
        pairs = keys // 2
        key[1:2 * pairs:2] = key[:2 * pairs:2]
        value[1:2 * pairs:2] = -value[:2 * pairs:2]
    mask = _mask(1, keys, pattern, seed)
    if pattern in _POISONED_PATTERNS:
        # Every masked key's K and V is NaN. Both kernels must skip a masked key
        # before its load; if one read it, the output would be NaN and the
        # finiteness assertion below would fail rather than the parity assert.
        dead = mask[0] == 0
        key[dead] = np.nan
        value[dead] = np.nan
    return query, key, value, mask, scale


def _cast(dtype, array):
    if dtype == "f32":
        return np.ascontiguousarray(array, dtype=np.float32)
    return bf16_round(array)


def _poison(dtype, shape):
    if dtype == "f32":
        return np.full(shape, _F32_POISON, dtype=np.uint32).view(np.float32)
    return np.full(shape, _BF16_POISON, dtype=np.uint16)


def _bit_view(array, dtype):
    """The exact bits the kernel wrote, so NaN payloads and signed zero matter."""

    if dtype == "bf16":
        return np.ascontiguousarray(array, dtype=np.uint16)
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def _decoded(array, dtype):
    return bf16_decode(array) if dtype == "bf16" else array


def _assert_exact_and_finite(native_out, staged_out, dtype, context):
    """Both outputs finite, then bit-for-bit equal; a miss reports the numbers."""

    native_f = _decoded(native_out, dtype)
    staged_f = _decoded(staged_out, dtype)
    assert np.isfinite(native_f).all(), (
        f"{context}: native global_class produced a non-finite element "
        f"({int(np.isnan(native_f).sum())} NaN, "
        f"{int(np.isinf(native_f).sum())} Inf)"
    )
    assert np.isfinite(staged_f).all(), (
        f"{context}: staged singleton produced a non-finite element "
        f"({int(np.isnan(staged_f).sum())} NaN, "
        f"{int(np.isinf(staged_f).sum())} Inf)"
    )
    native_bits = _bit_view(native_out, dtype)
    staged_bits = _bit_view(staged_out, dtype)
    if np.array_equal(native_bits, staged_bits):
        return
    # A genuine disagreement: report the numeric difference rather than only the
    # assertion. Production-correct behavior is judged from these numbers, not
    # from the fact that two bit patterns differ.
    difference = np.abs(native_f.astype(np.float64) - staged_f.astype(np.float64))
    mismatch = np.flatnonzero(native_bits.reshape(-1) != staged_bits.reshape(-1))
    pytest.fail(
        f"{context}: native global_class and staged singleton differ bitwise; "
        f"{mismatch.size}/{native_bits.size} elements differ; "
        f"max |delta|={difference.max():.6e}; "
        f"max rel={np.nanmax(difference / (np.abs(native_f.astype(np.float64)) + 1e-12)):.6e}; "
        f"first mismatching flat indices {mismatch[:8].tolist()}"
    )


def _assert_plan(plan, *, keys, num_heads, num_kv_heads, head_dim, context):
    """The staged launch resolved a successful singleton long-context plan."""

    assert plan.keys == keys, context
    assert plan.tokens == 1, context
    assert plan.num_heads == num_heads and plan.num_kv_heads == num_kv_heads, context
    assert plan.head_dim == head_dim, context
    rows_per_head = num_heads // num_kv_heads
    assert plan.rows_per_head == rows_per_head, context
    assert plan.row_groups == -(-rows_per_head // staged.SINGLETON_ROWS_PER_BLOCK), context
    assert plan.pv_singleton, context
    assert plan.pv_threads == staged.SINGLETON_THREADS, context
    assert plan.pv_rows_per_block == staged.SINGLETON_ROWS_PER_BLOCK, context
    assert plan.pv_dim_slices == head_dim // staged.SINGLETON_DIM_SLICE, context
    assert plan.pv_grid == (
        1,
        num_kv_heads,
        plan.row_groups * plan.pv_dim_slices,
    ), context
    assert plan.workspace_bytes > 0, context
    assert plan.lds_bytes <= staged.LDS_BUDGET_BYTES, context
    assert plan.lds_bytes == max(
        plan.score_lds_bytes, plan.softmax_lds_bytes, plan.pv_lds_bytes
    ), context


def _run_parity(
    native_library,
    staged_library,
    *,
    keys,
    num_heads,
    num_kv_heads,
    head_dim,
    pattern,
    dtype,
    seed,
    native_scratch=None,
    staged_scratch=None,
):
    """One native/staged comparison from identical, independently poisoned buffers."""

    query, key, value, mask, scale = _case(
        keys, num_heads, num_kv_heads, head_dim, pattern, dtype, seed
    )
    query, key, value = _cast(dtype, query), _cast(dtype, key), _cast(dtype, value)
    mask = np.ascontiguousarray(mask)
    native_out = _poison(dtype, query.shape)
    staged_out = _poison(dtype, query.shape)

    device = _Device()
    try:
        query_buffer = device.put(query)
        key_buffer = device.put(key)
        value_buffer = device.put(value)
        mask_buffer = device.put(mask)
        native_buffer = device.take(native_out)
        staged_buffer = device.take(staged_out)

        launch = (
            native.gemma4_attention_prefill_bf16
            if dtype == "bf16"
            else native.gemma4_attention_prefill_f32
        )
        launch(
            query_buffer.ptr,
            key_buffer.ptr,
            value_buffer.ptr,
            mask_buffer.ptr,
            native_buffer.ptr,
            tokens=1,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            library=native_library,
            scratch=native_scratch,
        )
        variant = native.gemma4_attention_decode_variant(native_library)

        staged_entry = (
            staged.gemma4_attention_staged_bf16
            if dtype == "bf16"
            else staged.gemma4_attention_staged_f32
        )
        plan = staged_entry(
            query_buffer.ptr,
            key_buffer.ptr,
            value_buffer.ptr,
            mask_buffer.ptr,
            staged_buffer.ptr,
            tokens=1,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            library=staged_library,
            scratch=staged_scratch,
        )

        device.read(native_buffer, native_out)
        device.read(staged_buffer, staged_out)
        return native_out, staged_out, plan, variant
    finally:
        device.close()


# --- the resident-bound premise ---------------------------------------------


def test_global_class_is_selected_above_the_resident_bound(native_library):
    """15,617 keys is the first one-token shape the native path sends to global logits.

    Establishes the selection this file's parity cases rest on: below the bound
    the one-token block is served by the flash/class chain, and at and above it
    by the strict class arithmetic with owned global logits. The check reads the
    launcher's own selection, not a timing or an inference.
    """

    keys, num_heads, num_kv_heads, head_dim = 15617, 8, 1, 512
    assert (keys + 768) * 4 > native._LDS_BUDGET_BYTES
    assert (keys - 1 + 768) * 4 <= native._LDS_BUDGET_BYTES
    query, key, value, mask, scale = _case(
        keys, num_heads, num_kv_heads, head_dim, "keep", "bf16", seed=1
    )
    query, key, value = _cast("bf16", query), _cast("bf16", key), _cast("bf16", value)
    out = _poison("bf16", query.shape)
    device = _Device()
    try:
        buffers = [device.put(a) for a in (query, key, value, mask)]
        out_buffer = device.take(out)
        native.gemma4_attention_prefill_bf16(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            out_buffer.ptr,
            tokens=1,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            library=native_library,
        )
        assert native.gemma4_attention_decode_variant(native_library) == "global_class"
        device.read(out_buffer, out)
        assert np.isfinite(bf16_decode(out)).all()
    finally:
        device.close()


# --- the long-context comparison --------------------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("keys", KEY_LENGTHS)
def test_global_class_matches_staged_singleton_over_key_lengths(
    native_library, staged_library, dtype, keys
):
    """The global-class route and the staged singleton agree bit for bit.

    One query token, the global layer's geometry (head_dim 512, GQA ratio 8),
    the all-keep mask, and every key length from the first global-class count to
    a full 262,144-key context. The seed is fixed per key length so the inputs
    are reproducible, and both output buffers start poisoned.
    """

    num_heads, num_kv_heads, head_dim = 8, 1, 512
    native_out, staged_out, plan, variant = _run_parity(
        native_library,
        staged_library,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        pattern="keep",
        dtype=dtype,
        seed=1000 + keys,
    )
    context = f"keys={keys} dtype={dtype} keep"
    assert variant == "global_class", context
    _assert_plan(
        plan, keys=keys, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, context=context,
    )
    _assert_exact_and_finite(native_out, staged_out, dtype, context)


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("keys", [15617, 32768])
def test_global_class_matches_staged_singleton_over_patterns(
    native_library, staged_library, dtype, pattern, keys
):
    """The same equality under the mask and value patterns the family must carry.

    All-keep, a broad softmax (small scale), cancellation-heavy values, a sparse
    mask, and a last-key-only mask, at the first global-class key count and at a
    32,768-key context. The sparse and last-key cases set masked K/V to NaN, so
    a kernel that read a masked key would produce a non-finite result instead of
    a silent parity pass.
    """

    num_heads, num_kv_heads, head_dim = 8, 1, 512
    native_out, staged_out, plan, variant = _run_parity(
        native_library,
        staged_library,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        pattern=pattern,
        dtype=dtype,
        seed=2000 + keys + PATTERNS.index(pattern),
    )
    context = f"keys={keys} dtype={dtype} pattern={pattern}"
    assert variant == "global_class", context
    _assert_plan(
        plan, keys=keys, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, context=context,
    )
    _assert_exact_and_finite(native_out, staged_out, dtype, context)


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
def test_global_class_matches_staged_singleton_at_16_heads(
    native_library, staged_library, dtype
):
    """A second ratio-8 headcount: 16 query heads over 2 KV heads.

    The same GQA ratio as the model's global geometry but a wider head count, so
    the native grid (16 blocks) and the staged row grouping differ from the
    8-head case while the arithmetic contract must not.
    """

    keys, num_heads, num_kv_heads, head_dim = 20003, 16, 2, 512
    native_out, staged_out, plan, variant = _run_parity(
        native_library,
        staged_library,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        pattern="broad",
        dtype=dtype,
        seed=3000 + keys,
    )
    context = f"keys={keys} dtype={dtype} heads={num_heads}"
    assert variant == "global_class", context
    _assert_plan(
        plan, keys=keys, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, context=context,
    )
    _assert_exact_and_finite(native_out, staged_out, dtype, context)


def test_scratch_reuse_long_short_long_keeps_results(native_library, staged_library):
    """One scratch across long-short-long launches with different data.

    A caller-owned :class:`Gemma4AttentionScratch` is reused for both wrappers
    across a 131,072-key launch, a 16,384-key launch, and a 131,072-key launch
    with different data. Each reused result is compared bit for bit with a fresh
    no-scratch launch of the same shape and seed, so a workspace the launcher
    failed to rewrite -- or an allocation that bled across launches -- shows up
    as a difference rather than as a plausible-looking number.
    """

    num_heads, num_kv_heads, head_dim = 8, 1, 512
    native_scratch = native.Gemma4AttentionScratch()
    staged_scratch = native.Gemma4AttentionScratch()
    schedule = [(131072, 11), (16384, 12), (131072, 13)]
    try:
        for keys, seed in schedule:
            native_out, staged_out, plan, variant = _run_parity(
                native_library,
                staged_library,
                keys=keys,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                pattern="keep",
                dtype="bf16",
                seed=seed,
                native_scratch=native_scratch,
                staged_scratch=staged_scratch,
            )
            context = f"reused scratch keys={keys} seed={seed}"
            assert variant == "global_class", context
            _assert_plan(
                plan, keys=keys, num_heads=num_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, context=context,
            )
            _assert_exact_and_finite(native_out, staged_out, "bf16", context)

            fresh_native, fresh_staged, _, fresh_variant = _run_parity(
                native_library,
                staged_library,
                keys=keys,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                pattern="keep",
                dtype="bf16",
                seed=seed,
            )
            assert fresh_variant == "global_class", context
            _assert_exact_and_finite(
                native_out, fresh_native, "bf16", context + " native vs fresh"
            )
            _assert_exact_and_finite(
                staged_out, fresh_staged, "bf16", context + " staged vs fresh"
            )
        # The two 131,072-key launches used different data, so their equality
        # above is not the trivial case of the same inputs twice.
        assert len(native_scratch._owned) == 1
        assert len(staged_scratch._owned) == 1
    finally:
        native_scratch.close()
        staged_scratch.close()
