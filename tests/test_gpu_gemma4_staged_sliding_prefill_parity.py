"""Direct GPU parity for the exact staged *sliding-prefill* substitution.

The proposed production route replaces the native/exact prefill entry at the one
call site a sliding layer falls to when AOTriton refuses it. The incumbent and
the candidate must produce the same bits for the same real mask. This file
drives **three** arms from identical device buffers and compares raw bytes:

* **native (incumbent, untrimmed).** ``strict._launch_prefill`` with the
  ``_SYMBOL_PREFILL_*`` entry and ``window=0, row_offset=0``. The C++ launcher
  routes a multi-token block into the key-class family
  (``launch_gemma4_attention_decode``), which is bit-identical to the block
  kernel by construction; the old block kernel is only the fallback when the
  class family refuses the geometry. The class family takes no
  ``window``/``row_offset`` for multi-token blocks, so this arm always walks the
  full presented range. Its pass 3 already skips masked (zero-weight) columns,
  so it already trims PV by mask -- what it does not do is trim the leading
  masked run by window.
* **staged, trimmed.** ``gemma4_attention_staged_bf16/f32`` with the layer's
  true sliding ``window``/``row_offset``.
* **staged, untrimmed.** The same staged entry with ``window=0, row_offset=0``.

Comparing all three separates the two claims the substitution rests on:
``staged_untrimmed == native`` shows the staged kernel equals the native on the
same untrimmed call, and ``staged_trimmed == staged_untrimmed`` shows the window
trim changes nothing on a genuine sliding mask (the softmax partition is
preserved because the thread-to-column map keeps ``j mod 256`` and the reduction
tree is unchanged, conditional on the caller's valid window promise).

Each output buffer is independently pre-filled with NaN before the launch, so a
row a kernel fails to write is a NaN rather than a plausible zero. The test
asserts every decoded output is finite *and* the three are byte-identical. BF16
and F32 are both run; the F32 arm exposes rounding that a trim error could hide
in BF16.

A mismatch here is a numerical change in the staged path, to be reviewed as
such -- the route is a proposed exact substitution, and a failure is evidence
about the substitution, not an automatic rejection of the candidate.

Reached with an explicit file target; the default suite collects ``test_unit_*``
only. No device work happens unless the module is actually run.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable"
)

from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as strict  # noqa: E402
from hipengine.kernels.hip_gfx1100.gemma4 import (  # noqa: E402
    gemma4_attention_staged as staged,
)
from tests.test_gpu_gemma4_attention_staged import (  # noqa: E402
    _Device,
    _assert_bit_identical,
    _mask,
    bf16_decode,
    bf16_round,
)


# --- input construction -----------------------------------------------------


def _base_inputs(tokens, keys, num_heads, num_kv_heads, head_dim, *, seed):
    """FP32 inputs, before the dtype rounding the buffers carry."""

    rng = np.random.default_rng(seed)
    query = rng.standard_normal((tokens, num_heads, head_dim), dtype=np.float32) * 0.7
    key = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 0.7
    value = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 0.7
    return query, key, value


def _sliding_holes_mask(tokens, keys, window, row_offset, *, seed):
    """A sliding-causal mask with extra holes inside the band.

    The sliding part is zero outside ``[first_kept, walk_end)``, so the trim's
    promise still holds; the holes only add more masked columns inside it.
    """

    base = _mask(tokens, keys, "sliding", window, row_offset, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    holes = (rng.random((tokens, keys)) < 0.6).astype(np.uint8)
    holes[:, 0] = 1
    return (base & holes).astype(np.uint8)


def _cancellation_pairs(tokens, keys, window, row_offset, *, limit=64):
    """Stable, disjoint adjacent column pairs kept by every row of the block.

    The intersection of the rows' kept ranges is
    ``[row_offset + tokens - 1 - window + 1, row_offset]``. Pairs are laid out
    inside it on a stride of two, so no pair overlaps the next and each pair's
    two columns are kept by every row -- and therefore survive the trim.
    """

    own_min = row_offset
    own_max = row_offset + tokens - 1
    lo = max(0, own_max - window + 1)
    hi = min(keys - 1, own_min)
    pairs = []
    column = lo
    while column + 1 <= hi and len(pairs) < limit:
        pairs.append((column, column + 1))
        column += 2
    return pairs


def _apply_cancellation(key, value, pairs):
    """Identical K and opposite V on each pair, so its numerator sum is 0.

    Equal keys make the two logits bit-identical, so the two weights are equal
    and the pair's real-arithmetic contributions cancel. FP32 FMA rounding can
    leave a residual; the contract here is the same ascending-key chain in all
    three arms, not a zero result. Dropping either column changes that chain.
    """

    for c0, c1 in pairs:
        key[c1] = key[c0]
        value[c0] = 6.0
        value[c1] = -6.0


def _apply_last_kept_impulse(key, value, *, tokens, keys, row_offset):
    """Large, distinctive values on each row's own (last kept) column edges."""

    for token in range(tokens):
        own = min(row_offset + token, keys - 1)
        value[own] = 8.0
        if own + 1 < keys:
            key[own + 1] = key[own]
            value[own + 1] = -8.0
        if own - 1 >= 0:
            value[own - 1] = 4.0


def _encode(query, key, value, dtype):
    if dtype == "bf16":
        return bf16_round(query), bf16_round(key), bf16_round(value)
    return (
        np.ascontiguousarray(query),
        np.ascontiguousarray(key),
        np.ascontiguousarray(value),
    )


def _poison(shape, dtype):
    """A host output buffer pre-filled with NaN in the target's own bit format."""

    if dtype == "bf16":
        return bf16_round(np.full(shape, np.nan, dtype=np.float32))
    return np.full(shape, np.nan, dtype=np.float32)


# --- the three-arm launch ---------------------------------------------------


def _run_triple(
    shape,
    *,
    dtype,
    mask,
    scale,
    window,
    row_offset,
    staged_library,
    native_library,
    seed,
    impulse=None,
):
    """Native (window=0), staged-trimmed, and staged-untrimmed, one buffer set.

    Returns ``(native_out, trimmed_out, untrimmed_out, trimmed_plan,
    untrimmed_plan)`` on the host. All three output buffers were poisoned with
    NaN before the launch, so an unwritten element reads back as NaN.
    """

    tokens, keys, num_heads, num_kv_heads, head_dim = shape
    query, key, value = _base_inputs(
        tokens, keys, num_heads, num_kv_heads, head_dim, seed=seed
    )
    pairs = None
    if impulse == "paired_cancellation":
        pairs = _cancellation_pairs(tokens, keys, window, row_offset)
        _apply_cancellation(key, value, pairs)
        # The mask must keep both columns of every tested pair, and the pairs
        # were chosen inside the rows' common kept band, so forcing them is a
        # guard rather than a change.
        mask = mask.copy()
        for c0, c1 in pairs:
            mask[:, c0] = 1
            mask[:, c1] = 1
    elif impulse == "last_kept_impulse":
        _apply_last_kept_impulse(
            key, value, tokens=tokens, keys=keys, row_offset=row_offset
        )
    elif impulse is not None:  # pragma: no cover - defensive
        raise ValueError(f"unknown impulse mode {impulse!r}")
    query, key, value = _encode(query, key, value, dtype)

    out_shape = (tokens, num_heads, head_dim)
    native_out = _poison(out_shape, dtype)
    trimmed_out = _poison(out_shape, dtype)
    untrimmed_out = _poison(out_shape, dtype)

    device = _Device()
    try:
        query_buffer = device.put(query)
        key_buffer = device.put(key)
        value_buffer = device.put(value)
        mask_buffer = device.put(mask)
        native_buffer = device.put(native_out)
        trimmed_buffer = device.put(trimmed_out)
        untrimmed_buffer = device.put(untrimmed_out)

        kwargs = dict(
            tokens=tokens,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
        )
        # The incumbent: the native prefill entry with no window trim.
        strict._launch_prefill(
            strict._SYMBOL_PREFILL_BF16 if dtype == "bf16" else strict._SYMBOL_PREFILL_F32,
            query_buffer.ptr, key_buffer.ptr, value_buffer.ptr, mask_buffer.ptr,
            native_buffer.ptr, stream=0, library=native_library, runtime=None,
            window=0, row_offset=0, **kwargs,
        )
        staged_entry = (
            staged.gemma4_attention_staged_bf16
            if dtype == "bf16"
            else staged.gemma4_attention_staged_f32
        )
        trimmed_plan = staged_entry(
            query_buffer.ptr, key_buffer.ptr, value_buffer.ptr, mask_buffer.ptr,
            trimmed_buffer.ptr, library=staged_library, window=window,
            row_offset=row_offset, **kwargs,
        )
        untrimmed_plan = staged_entry(
            query_buffer.ptr, key_buffer.ptr, value_buffer.ptr, mask_buffer.ptr,
            untrimmed_buffer.ptr, library=staged_library, window=0,
            row_offset=0, **kwargs,
        )
        device.read(native_buffer, native_out)
        device.read(trimmed_buffer, trimmed_out)
        device.read(untrimmed_buffer, untrimmed_out)
        return native_out, trimmed_out, untrimmed_out, trimmed_plan, untrimmed_plan
    finally:
        device.close()


def _decode(out, dtype):
    return bf16_decode(out) if dtype == "bf16" else out


# --- the curated matrix -----------------------------------------------------
#
# (name, tokens, keys, num_heads, num_kv_heads, head_dim, mask_mode, window,
#  row_offset, scale, impulse)
#
# Curated, not a Cartesian product: the measured chunk, row counts unrelated to
# the window boundary, the boundary itself on both sides, the 256-lane
# first_kept boundary under holed sliding masks, both head widths, a holed mask,
# a broad and a peaked softmax, and two boundary-impulse constructions. Every
# row keeps at least one key, so all three arms are finite.
_CASES = [
    # The measured chunk from the 8K attribution: 1024 rows over the
    # window-bounded 2047 presented keys (window 1024 + rows 1024 - 1).
    ("measured_1024x2047", 1024, 2047, 16, 8, 256, "sliding", 1024, 1023, 1.0, None),
    # Row counts unrelated to the boundary, at the measured key count.
    ("rows_2", 2, 2047, 16, 8, 256, "sliding", 1024, 1023, 1.0, None),
    ("rows_3", 3, 2047, 16, 8, 256, "sliding", 1024, 1023, 1.0, None),
    ("rows_127", 127, 2047, 16, 8, 256, "sliding", 1024, 1023, 1.0, None),
    ("rows_257", 257, 2047, 16, 8, 256, "sliding", 1024, 1023, 1.0, None),
    # The window boundary: keys == window + rows - 1 exactly, and one past it.
    ("window_exact_boundary", 64, 191, 16, 8, 256, "sliding", 128, 127, 1.0, None),
    ("window_one_past", 64, 192, 16, 8, 256, "sliding", 128, 128, 1.0, None),
    # A window narrower than the block: rows straddle their own windows.
    ("window_narrower_than_block", 64, 95, 16, 8, 256, "sliding", 32, 31, 1.0, None),
    # A window wider than the presented range: the trim is a no-op.
    ("window_wider_than_keys", 64, 128, 16, 8, 256, "causal", 4096, 0, 1.0, None),
    # head_dim 512, the global width, driven with a sliding mask too.
    ("head512_sliding", 128, 383, 16, 2, 512, "sliding", 256, 255, 1.0, None),
    ("head512_measured_keys", 128, 2047, 16, 2, 512, "sliding", 1024, 1023, 1.0, None),
    # A holed mask with no window promise: both arms walk every column.
    ("holed_no_trim", 64, 1024, 16, 8, 256, "holes", 0, 0, 1.0, None),
    # Softmax shape: broad (many comparable terms) and peaked (one winner).
    ("softmax_broad", 64, 2047, 16, 8, 256, "sliding", 1024, 1023, 0.05, None),
    ("softmax_peaked", 64, 2047, 16, 8, 256, "sliding", 1024, 1023, 8.0, None),
    # first_kept crossing the 256-lane softmax partition boundary: the offsets
    # make row 0's first_kept exactly 255, 256 and 257, under holed sliding
    # masks. A thread-to-column map that shifted ``j mod 256`` would show here.
    ("first_kept_255", 8, 1286, 16, 8, 256, "sliding_holes", 1024, 1278, 1.0, None),
    ("first_kept_256", 8, 1287, 16, 8, 256, "sliding_holes", 1024, 1279, 1.0, None),
    ("first_kept_257", 8, 1288, 16, 8, 256, "sliding_holes", 1024, 1280, 1.0, None),
    # Boundary impulses: the trim's first_kept / walk_end edges.
    ("paired_cancellation", 64, 2047, 16, 8, 256, "sliding", 1024, 1023, 1.0,
     "paired_cancellation"),
    ("last_kept_impulse", 64, 2047, 16, 8, 256, "sliding", 1024, 1023, 1.0,
     "last_kept_impulse"),
]


@pytest.fixture(scope="module")
def staged_library():
    return staged.build_gemma4_attention_staged(load=True)


@pytest.fixture(scope="module")
def native_library():
    return strict.build_gemma4_attention(load=True)


def _case_mask(case):
    _, tokens, keys, _nh, _nkv, _hd, mode, window, row_offset, _scale, _impulse = case
    if mode == "sliding_holes":
        return _sliding_holes_mask(tokens, keys, window, row_offset, seed=3)
    return _mask(tokens, keys, mode, window, row_offset, seed=3)


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_native_and_both_staged_arms_are_byte_identical(
    staged_library, native_library, dtype, case,
):
    (name, tokens, keys, num_heads, num_kv_heads, head_dim, mode, window,
     row_offset, scale, impulse) = case
    shape = (tokens, keys, num_heads, num_kv_heads, head_dim)
    mask = _case_mask(case)

    native_out, trimmed_out, untrimmed_out, trimmed_plan, untrimmed_plan = _run_triple(
        shape, dtype=dtype, mask=mask, scale=scale, window=window,
        row_offset=row_offset, staged_library=staged_library,
        native_library=native_library, seed=17, impulse=impulse,
    )

    # Every row keeps at least one key, so a finite result is required and the
    # NaN poison only proves a write happened.
    for label, out in (
        ("native", native_out), ("trimmed", trimmed_out), ("untrimmed", untrimmed_out)
    ):
        assert np.isfinite(_decode(out, dtype)).all(), f"{name}: {label} not finite"

    assert native_out.shape == trimmed_out.shape == untrimmed_out.shape
    # The two claims, separated: staged-untrimmed reproduces the native kernel,
    # and the window trim changes nothing on the sliding mask.
    _assert_bit_identical(native_out, untrimmed_out, dtype)
    _assert_bit_identical(untrimmed_out, trimmed_out, dtype)

    # The launch is the one the shape selects, so the equality above is about
    # the decomposition that ran rather than a fallback that quietly served it.
    assert trimmed_plan.rows == untrimmed_plan.rows == tokens * num_heads
    assert trimmed_plan.keys == untrimmed_plan.keys == keys
    if impulse == "paired_cancellation":
        pairs = _cancellation_pairs(tokens, keys, window, row_offset)
        assert pairs, f"{name}: no disjoint kept pairs to cancel"


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
def test_the_trim_is_a_noop_on_the_dense_first_chunk(staged_library, native_library, dtype):
    """Chunk 0 presents ``keys == rows``; the window does not bind.

    A sliding layer's first chunk has ``key_begin == 0``, so the incumbent and
    the candidate both see the same dense causal range and the candidate's trim
    drops nothing.
    """

    tokens = 128
    shape = (tokens, tokens, 16, 8, 256)
    mask = _mask(tokens, tokens, "causal", 0, 0, seed=5)
    native_out, trimmed_out, untrimmed_out, trimmed_plan, _ = _run_triple(
        shape, dtype=dtype, mask=mask, scale=1.0, window=1024, row_offset=0,
        staged_library=staged_library, native_library=native_library, seed=23,
    )
    for out in (native_out, trimmed_out, untrimmed_out):
        assert np.isfinite(_decode(out, dtype)).all()
    _assert_bit_identical(native_out, untrimmed_out, dtype)
    _assert_bit_identical(untrimmed_out, trimmed_out, dtype)
    assert trimmed_plan.keys == tokens
