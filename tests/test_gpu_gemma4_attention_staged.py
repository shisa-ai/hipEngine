"""Bitwise parent parity and oracle agreement for staged Gemma 4 attention.

The bitwise oracle is the monolithic strict block entry point, including for
singleton queries. The adaptive decode wrapper also selects split/flash
implementations, which do not declare this parent accumulation-order contract.
The staged candidate is checked against its original parent, not those routes.

Two independent checks:

* **Parity.** The staged launcher and monolithic strict block entry are driven from
  identical device buffers with identical arguments, and their outputs are
  compared bit for bit -- the ``uint16`` payloads for BF16 and the ``uint32``
  payloads for F32, so ``0.0`` and ``-0.0`` are different results and one NaN is
  not interchangeable with another. That is
  the strongest available statement of the candidate's contract, and it covers
  both the one-token decode routing and multi-token blocks, causal, sliding and
  holed masks, nonzero ``window``/``row_offset``, and both head widths. A single
  query token takes the singleton PV decomposition and every other shape takes
  the 256-thread one, so the parity table is also the check that the two
  decompositions agree bit for bit on the shapes they share.
* **Oracle.** Above the monolithic block's resident-logit LDS ceiling,
  results are compared against
  the independent float64 oracle in
  ``tests/test_unit_gemma4_attention_staged.py``, with the *exact* inputs the
  kernel was given (BF16-decoded where the buffers are BF16), so the tolerance
  measures the kernel's accumulation and not its input rounding. The oracle
  itself is first checked against the strict kernel on a shape both can run, so
  the comparison is evidence about the kernel rather than about the oracle.

Reached with an explicit file target; the default suite collects ``test_unit_*``
only.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available
from tests.test_unit_gemma4_attention_staged import (
    bf16_decode,
    bf16_round,
    staged_cpu_oracle,
)

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as strict  # noqa: E402
from hipengine.kernels.hip_gfx1100.gemma4 import (  # noqa: E402
    gemma4_attention_staged as staged,
)

# The strict wrapper's shared-memory ceiling, per head width: it stores one FP32
# logit per live key, so `keys` is what pushes it past 64 KB.
STRICT_KEY_CEILING = {256: 15872, 512: 15616}

# (tokens, keys, num_heads, num_kv_heads, head_dim, mask mode, window, row_offset, scale)
_PARITY_SHAPES = [
    (1, 1, 16, 2, 512, "keep", 0, 0, 1.0),  # a single live key
    (2, 7, 16, 8, 256, "holes", 0, 0, 1.0),  # sub-warp key count
    (1, 31, 16, 8, 256, "keep", 0, 0, 1.0),
    (1, 257, 16, 2, 512, "keep", 0, 0, 1.0),
    (1, 1025, 16, 8, 256, "keep", 0, 0, 1.0),
    (1, 2055, 16, 8, 256, "holes", 0, 0, 0.375),
    (1, 1025, 16, 2, 512, "holes", 0, 0, 1.0),
    (1, 4096, 16, 2, 512, "keep", 0, 0, 1.0),
    (1, 1023, 4, 4, 256, "keep", 0, 0, 1.0),  # GQA ratio 1: one row per PV CTA
    (1, 1023, 2, 1, 256, "keep", 0, 0, 1.0),  # 2q/1kv: GQA ratio 2, one row group
    (1, 512, 32, 2, 512, "keep", 0, 0, 1.0),  # GQA ratio 16: two full row groups
    (3, 257, 32, 2, 512, "causal", 0, 0, 1.0),  # the same geometry, 3-token block
    (2, 1025, 24, 2, 256, "holes", 0, 0, 1.0),  # ratio 12: a full group, then 4 rows
    (3, 257, 16, 2, 512, "causal", 0, 0, 1.0),
    (3, 33, 6, 2, 512, "holes", 0, 0, 0.375),  # paired score, odd GQA tail
    (2, 1023, 10, 2, 256, "sliding", 129, 900, 0.375),
    (3, 1024, 6, 2, 512, "keep", 0, 0, 0.05),
    (2, 1025, 10, 2, 512, "holes", 0, 0, 0.05),
    (7, 257, 16, 2, 512, "causal", 0, 0, 1.0),
    (5, 1025, 16, 8, 256, "causal", 0, 0, 1.0),
    (4, 512, 4, 1, 512, "causal", 0, 0, 1.0),  # GQA ratio 4
    (1, 1025, 16, 2, 512, "sliding", 96, 1024, 1.0),
    (3, 1025, 16, 8, 256, "sliding", 64, 900, 1.0),
    (2, 2055, 16, 2, 512, "sliding", 128, 1500, 1.0),
    (3, 17003, 6, 2, 512, "holes", 0, 0, 0.05),  # global parent, odd paired tail
    (2, 1025, 16, 2, 512, "holes", 0, 0, 1.0),
    # Single-token shapes, which take the 32-thread singleton PV decomposition:
    # both head widths, the GQA ratios 1/2/4/8 and the odd ratios whose last row
    # group holds a single row, a weight tile narrower than the block, and key
    # counts that land on and past a tile boundary.
    (1, 1, 16, 8, 256, "keep", 0, 0, 1.0),  # one live key, one-element tile
    (1, 7, 6, 2, 512, "keep", 0, 0, 1.0),  # ratio 3, tile narrower than 32 threads
    (1, 33, 10, 2, 256, "holes", 0, 0, 1.0),  # ratio 5, a second partial tile
    (1, 257, 4, 4, 512, "causal", 0, 0, 1.0),  # ratio 1: one resident row
    (1, 257, 32, 2, 512, "causal", 0, 0, 1.0),  # ratio 16: eight row groups
    (1, 1025, 6, 2, 256, "keep", 0, 0, 1.0),  # ratio 3: a one-row tail group
    (1, 1025, 10, 2, 256, "holes", 0, 0, 1.0),  # ratio 5: two full, one partial
    (1, 2055, 24, 2, 256, "keep", 0, 0, 1.0),  # ratio 12: every group full
    (1, 4096, 16, 8, 256, "keep", 0, 0, 1.0),  # 256 dims, four full tiles
    (1, 4096, 16, 2, 512, "holes", 0, 0, 1.0),  # 512 dims, sixteen full tiles
    (1, 1025, 16, 8, 256, "sliding", 96, 1024, 1.0),  # the sliding trim too
]


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
        return buffer

    def read(self, buffer, array: np.ndarray) -> None:
        copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)

    def close(self) -> None:
        for buffer in self.buffers:
            free(buffer)
        self.buffers.clear()


@pytest.fixture(scope="module")
def staged_library():
    return staged.build_gemma4_attention_staged(load=True)


def _mask(tokens, keys, mode, window, row_offset, seed):
    index = np.arange(keys)[None, :]
    own = row_offset + np.arange(tokens)[:, None]
    if mode == "keep":
        return np.ones((tokens, keys), dtype=np.uint8)
    if mode == "causal":
        return (index <= own).astype(np.uint8)
    if mode == "sliding":
        return ((index <= own) & (index > own - window)).astype(np.uint8)
    if mode == "holes":
        rng = np.random.default_rng(seed)
        mask = (rng.random((tokens, keys)) < 0.6).astype(np.uint8)
        # Every row keeps at least one key, so no row is all-masked and the
        # bitwise comparisons below are not comparing NaN payloads.
        mask[:, 0] = 1
        return mask
    raise ValueError(mode)  # pragma: no cover - defensive


def _inputs(tokens, keys, num_heads, num_kv_heads, head_dim, *, dtype, seed):
    rng = np.random.default_rng(seed)
    query = rng.standard_normal((tokens, num_heads, head_dim), dtype=np.float32) * 0.7
    key = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 0.7
    value = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 0.7
    if dtype == "bf16":
        return bf16_round(query), bf16_round(key), bf16_round(value)
    return (np.ascontiguousarray(query), np.ascontiguousarray(key), np.ascontiguousarray(value))


def _exact(query, key, value, dtype):
    """The oracle's inputs: the same values the kernel reads, in float64."""

    if dtype == "bf16":
        return (
            bf16_decode(query).astype(np.float64),
            bf16_decode(key).astype(np.float64),
            bf16_decode(value).astype(np.float64),
        )
    return query.astype(np.float64), key.astype(np.float64), value.astype(np.float64)


def _bit_view(array: np.ndarray, dtype: str) -> np.ndarray:
    """The comparison view: the exact bits the kernel wrote.

    ``np.testing.assert_array_equal`` compares values, so it treats ``0.0`` and
    ``-0.0`` as equal and every NaN payload as equal to every other. The
    candidate's contract is bitwise, so the comparisons below compare bits.
    """

    if dtype == "bf16":
        return np.ascontiguousarray(array, dtype=np.uint16)
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def _assert_bit_identical(left: np.ndarray, right: np.ndarray, dtype: str) -> None:
    np.testing.assert_array_equal(_bit_view(left, dtype), _bit_view(right, dtype))


def _launch_kwargs(shape, *, scale, window, row_offset):
    tokens, keys, num_heads, num_kv_heads, head_dim = shape
    return dict(
        tokens=tokens,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        window=window,
        row_offset=row_offset,
    )


def _run_staged(
    shape,
    *,
    dtype,
    mask,
    scale,
    window,
    row_offset,
    library,
    scratch=None,
    runtime=None,
    seed=17,
    query=None,
    key=None,
    value=None,
):
    """One staged launch; returns (output, plan, exact inputs, mask)."""

    tokens, keys, num_heads, num_kv_heads, head_dim = shape
    if query is None:
        query, key, value = _inputs(
            tokens, keys, num_heads, num_kv_heads, head_dim, dtype=dtype, seed=seed
        )
    out = np.zeros_like(query)
    device = _Device()
    try:
        query_buffer = device.put(query)
        key_buffer = device.put(key)
        value_buffer = device.put(value)
        mask_buffer = device.put(mask)
        out_buffer = device.take(out)
        entry = (
            staged.gemma4_attention_staged_bf16
            if dtype == "bf16"
            else staged.gemma4_attention_staged_f32
        )
        plan = entry(
            query_buffer.ptr,
            key_buffer.ptr,
            value_buffer.ptr,
            mask_buffer.ptr,
            out_buffer.ptr,
            library=library,
            scratch=scratch,
            runtime=runtime,
            **_launch_kwargs(shape, scale=scale, window=window, row_offset=row_offset),
        )
        device.read(out_buffer, out)
        return out, plan, _exact(query, key, value, dtype), mask
    finally:
        device.close()


def _run_both(
    shape, *, dtype, mask, scale, window, row_offset, library, scratch=None, seed=17
):
    """The same buffers through the monolithic parent and staged candidate."""

    tokens, keys, num_heads, num_kv_heads, head_dim = shape
    query, key, value = _inputs(
        tokens, keys, num_heads, num_kv_heads, head_dim, dtype=dtype, seed=seed
    )
    out_strict = np.zeros_like(query)
    out_staged = np.zeros_like(query)
    device = _Device()
    try:
        query_buffer = device.put(query)
        key_buffer = device.put(key)
        value_buffer = device.put(value)
        mask_buffer = device.put(mask)
        strict_buffer = device.take(out_strict)
        staged_buffer = device.take(out_staged)
        kwargs = _launch_kwargs(shape, scale=scale, window=window, row_offset=row_offset)
        # Pin the declared monolithic parent even at tokens==1; the public
        # wrapper may otherwise select the incoming split/flash decode route.
        strict._launch_prefill(
            strict._SYMBOL_PREFILL_BF16 if dtype == "bf16" else strict._SYMBOL_PREFILL_F32,
            query_buffer.ptr, key_buffer.ptr, value_buffer.ptr, mask_buffer.ptr,
            strict_buffer.ptr, stream=0, library=None, runtime=None, **kwargs,
        )
        staged_entry = (
            staged.gemma4_attention_staged_bf16
            if dtype == "bf16"
            else staged.gemma4_attention_staged_f32
        )
        plan = staged_entry(
            query_buffer.ptr, key_buffer.ptr, value_buffer.ptr, mask_buffer.ptr,
            staged_buffer.ptr, library=library, scratch=scratch, **kwargs,
        )
        device.read(strict_buffer, out_strict)
        device.read(staged_buffer, out_staged)
        return out_strict, out_staged, plan
    finally:
        device.close()


# --- bitwise parity with the shipped kernel ---------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize(
    "tokens,keys,num_heads,num_kv_heads,head_dim,mode,window,row_offset,scale",
    _PARITY_SHAPES,
)
def test_staged_is_bit_identical_to_strict(
    staged_library, dtype, tokens, keys, num_heads, num_kv_heads, head_dim, mode, window,
    row_offset, scale,
):
    shape = (tokens, keys, num_heads, num_kv_heads, head_dim)
    mask = _mask(tokens, keys, mode, window, row_offset, seed=3)
    reference, candidate, plan = _run_both(
        shape, dtype=dtype, mask=mask, scale=scale, window=window, row_offset=row_offset,
        library=staged_library,
    )
    # The plan is checked as well as the values: a parity failure on a shape
    # whose decomposition the launcher got wrong is easier to read next to the
    # grid it actually launched.
    assert reference.shape == candidate.shape
    _assert_bit_identical(reference, candidate, dtype)
    # Every parity mask keeps at least one key per row, so both sides are
    # finite: the bitwise equality above is a comparison of real results, not of
    # two NaNs that happen to share a payload.
    for values in (reference, candidate):
        decoded = bf16_decode(values) if dtype == "bf16" else values
        assert np.isfinite(decoded).all()
    assert plan.rows == tokens * num_heads
    assert plan.keys == keys
    # The shape ran the decomposition its token count selects, so the bitwise
    # equality above is a statement about that kernel rather than about a
    # fallback that quietly served the launch.
    if tokens == 1:
        assert plan.pv_threads == staged.SINGLETON_THREADS
        assert plan.pv_dim_slices * staged.SINGLETON_DIM_SLICE == head_dim
    else:
        assert plan.pv_threads == staged.THREADS


def test_all_masked_rows_are_nan_in_both(staged_library):
    """No live key: both paths propagate NaN, and neither invents a value.

    The NaN payload is not part of the contract -- the strict kernel's comes out
    of ``expf(NaN)`` and this candidate's denominator is NaN as well -- so this
    asserts NaN-ness rather than bitwise equality, unlike the parity cases above.
    """

    shape = (2, 64, 4, 4, 256)
    mask = np.zeros((2, 64), dtype=np.uint8)
    reference, candidate, _ = _run_both(
        shape, dtype="bf16", mask=mask, scale=1.0, window=0, row_offset=0,
        library=staged_library,
    )
    assert np.isnan(bf16_decode(reference)).all()
    assert np.isnan(bf16_decode(candidate)).all()


def test_launch_accepts_an_explicit_runtime_and_library(staged_library):
    """The keyword surface a caller reaches must work, not just the defaults."""

    shape = (1, 512, 8, 4, 256)
    mask = _mask(1, 512, "keep", 0, 0, seed=1)
    runtime = get_hip_runtime()
    out, plan, _, _ = _run_staged(
        shape, dtype="bf16", mask=mask, scale=1.0, window=0, row_offset=0,
        library=staged_library, runtime=runtime,
    )
    reference, candidate, _ = _run_both(
        shape, dtype="bf16", mask=mask, scale=1.0, window=0, row_offset=0,
        library=staged_library,
    )
    _assert_bit_identical(reference, candidate, "bf16")
    assert out.shape == candidate.shape
    assert np.isfinite(bf16_decode(out)).all()
    # The single-token shape took the singleton decomposition, so the launch
    # under test is the one whose reservation this asserts.
    assert plan.pv_singleton
    assert plan.lds_bytes == max(
        plan.score_lds_bytes, plan.softmax_lds_bytes, plan.pv_lds_bytes
    )
    assert plan.pv_lds_bytes == staged.SINGLETON_ROWS_PER_BLOCK * staged.PV_TILE_KEYS * 4


def test_caller_owned_scratch_survives_repeated_launches(staged_library):
    """One scratch across calls: same result, no growth, no leak."""

    shape = (1, 2048, 16, 2, 512)
    mask = _mask(1, 2048, "keep", 0, 0, seed=2)
    scratch = strict.Gemma4AttentionScratch()
    try:
        outputs = []
        for _ in range(3):
            out, plan, _, _ = _run_staged(
                shape, dtype="bf16", mask=mask, scale=1.0, window=0, row_offset=0,
                library=staged_library, scratch=scratch,
            )
            outputs.append(out)
        reference, candidate, _ = _run_both(
            shape, dtype="bf16", mask=mask, scale=1.0, window=0, row_offset=0,
            library=staged_library,
        )
        _assert_bit_identical(reference, candidate, "bf16")
        for out in outputs:
            _assert_bit_identical(candidate, out, "bf16")
        # The launcher sized the workspace from the plan, and the scratch kept
        # one buffer for the stream rather than one per launch.
        assert len(scratch._owned) == 1
        assert scratch._owned[0].nbytes >= plan.workspace_bytes
    finally:
        scratch.close()


def test_kernel_exports_match_the_python_planner(staged_library):
    """The workspace and shared-memory definitions cannot drift.

    The launcher sizes the allocation in Python and the kernel indexes it in
    C++; a disagreement between the two is memory corruption rather than a slow
    path, so both sides are read here and compared.
    """

    for tokens, num_heads, keys in ((1, 16, 4096), (3, 5, 1025), (2, 8, 2048), (1, 16, 17000),
                                    (65535, 1, 64), (65536, 1, 1), (17000, 16, 1)):
        assert staged.staged_workspace_bytes_from_kernel(
            tokens, num_heads, keys, library=staged_library
        ) == staged.staged_workspace_bytes(tokens, num_heads, keys)
    for head_dim in staged.SCORE_TREE_HEAD_DIMS:
        assert staged.staged_lds_bytes_from_kernel(
            head_dim, library=staged_library
        ) == staged.staged_lds_bytes(head_dim)
    # The bounded-layout property, from the kernel's own numbers.
    assert staged.staged_lds_bytes_from_kernel(512, library=staged_library) == 8 * 256 * 4


def test_returned_plan_names_the_decomposition_that_ran(staged_library):
    """A fallback or a wrong geometry must be visible, not inferred from timing."""

    shape = (3, 4096, 16, 2, 512)
    mask = _mask(3, 4096, "causal", 0, 0, seed=4)
    _, plan, _, _ = _run_staged(
        shape, dtype="bf16", mask=mask, scale=1.0, window=0, row_offset=0,
        library=staged_library,
    )
    assert plan.score_grid == (24, 4)
    assert plan.softmax_grid == (48, 1)
    assert plan.pv_grid == (3, 2, 1)
    assert plan.rows_per_head == 8
    assert plan.lds_bytes == 8 * 256 * 4
    assert plan.workspace_bytes == staged.staged_workspace_bytes(3, 16, 4096)
    assert "pv_grid=3x2x1" in plan.describe()


# --- the PV variant dispatch -------------------------------------------------

# The variant the PV launcher selects is a host-visible fact: the kernel exports
# the resident row count it dispatches on. Declared here rather than in the
# wrapper module because it is a diagnostic for this test, not part of the
# launcher's calling contract.
_SYMBOL_PV_RESIDENT_ROWS = "hipengine_gemma4_attention_staged_pv_resident_rows"


def _pv_resident_rows(library, num_heads: int, num_kv_heads: int) -> int:
    """The resident row count the PV launcher would use for this geometry."""

    import ctypes

    from hipengine.core.ctypes_cache import signed_kernel_fn

    fn = signed_kernel_fn(
        library,
        _SYMBOL_PV_RESIDENT_ROWS,
        (ctypes.c_int64, ctypes.c_int64),
        ctypes.c_int64,
    )
    return int(fn(int(num_heads), int(num_kv_heads)))


# (num_heads, num_kv_heads, the resident row count the launcher must select)
_PV_VARIANT_TABLE = [
    (4, 4, 1),  # ratio 1: one row per CTA, the degenerate exact case
    (16, 16, 1),
    (16, 8, 2),  # the sliding layer's geometry, whole group resident
    (2, 1, 2),
    (4, 1, 4),
    (16, 2, 8),  # the full layer's geometry, whole group resident
    (8, 1, 8),
    (32, 2, 8),  # ratio 16: two full groups, generic kernel
    (24, 2, 8),  # ratio 12: a full group then a partial one, generic kernel
    (6, 2, 8),  # ratio 3: a single partial group, generic kernel
    (20, 2, 8),  # ratio 10, generic kernel
]


def test_pv_variant_dispatch_is_pinned(staged_library):
    """Which PV kernel runs is read from the kernel, not inferred from timing.

    The GQA ratios 1, 2, 4 and 8 fill the resident tile exactly, so their whole
    group is in one CTA, their row count is a compile-time constant and the
    per-key ``r >= rows`` predicate is dead. Every other ratio keeps the generic
    kernel with its runtime row count. The launcher dispatches on the same table
    this export reads, so the variant a caller names here is the variant that
    ran -- and ``resident == rows_per_head`` is the test for the exact kind.
    """

    for num_heads, num_kv_heads, resident in _PV_VARIANT_TABLE:
        rows_per_head = num_heads // num_kv_heads
        assert _pv_resident_rows(staged_library, num_heads, num_kv_heads) == resident
        # The table is the 256-thread decomposition's, which is the one a launch
        # with more than one query token uses: the plan below is driven with two
        # tokens so the row grouping it reports is that decomposition's.
        plan = staged.staged_plan(
            tokens=2, keys=64, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=256,
        )
        # The variant changes the kernel body only: the grid the plan reports is
        # the one the launch uses either way.
        assert plan.rows_per_head == rows_per_head
        assert plan.row_groups == -(-rows_per_head // staged.MAX_ROWS_PER_BLOCK)
        assert plan.pv_grid == (2, num_kv_heads, plan.row_groups)
        assert plan.pv_threads == staged.THREADS
        assert plan.pv_lds_bytes == staged.staged_pv_lds_bytes()
        assert (resident == rows_per_head) == (rows_per_head in (1, 2, 4, 8))
    # A geometry the launcher refuses answers 0, like the workspace export's
    # invalid sentinel, instead of naming a variant that cannot run.
    for num_heads, num_kv_heads in ((3, 2), (0, 1), (2, 0), (-2, 1), (2**31, 1)):
        assert _pv_resident_rows(staged_library, num_heads, num_kv_heads) == 0
    # Every exact variant has a bitwise parity case above, and at least one
    # generic ratio does too, so no dispatch path is without a reference.
    parity_ratios = {
        num_heads // num_kv_heads for _, _, num_heads, num_kv_heads, *_ in _PARITY_SHAPES
    }
    assert {1, 2, 4, 8} <= parity_ratios
    assert parity_ratios - {1, 2, 4, 8}


def test_exact_and_generic_variants_agree_on_one_row_group(staged_library):
    """The specialization is a code shape, not a different result.

    A 16q/2kv launch (ratio 8) runs the exact variant over its single row group.
    A 32q/2kv launch (ratio 16) runs the generic kernel over two full groups,
    whose runtime row count is 8 as well, so its per-key ``r >= rows`` predicate
    never fires either. Query heads 0..7 of both launches read KV head 0 (the
    ratio is wider than the group in one and not the other, but the first group
    starts at head 0 either way) and the same Q, K and V values, so the two
    variants must produce the same bits for them. That makes "only the predicate
    differs" a checked statement about the two compiled kernels rather than a
    claim about the source they share.
    """

    tokens, keys, head_dim, kv_heads = 2, 257, 512, 2
    exact_heads, generic_heads = 16, 32
    # The premise: one launch is exact and the other is not.
    assert _pv_resident_rows(staged_library, exact_heads, kv_heads) == exact_heads // kv_heads
    assert _pv_resident_rows(staged_library, generic_heads, kv_heads) == staged.MAX_ROWS_PER_BLOCK
    assert generic_heads // kv_heads != staged.MAX_ROWS_PER_BLOCK

    rng = np.random.default_rng(41)
    key = rng.standard_normal((keys, kv_heads, head_dim), dtype=np.float32) * 0.7
    value = rng.standard_normal((keys, kv_heads, head_dim), dtype=np.float32) * 0.7
    query = rng.standard_normal((tokens, generic_heads, head_dim), dtype=np.float32) * 0.7
    mask = _mask(tokens, keys, "holes", 0, 0, seed=8)

    exact, exact_plan, _, _ = _run_staged(
        (tokens, keys, exact_heads, kv_heads, head_dim), dtype="f32", mask=mask,
        scale=1.0, window=0, row_offset=0, library=staged_library,
        query=query[:, :exact_heads].copy(), key=key, value=value,
    )
    generic, generic_plan, _, _ = _run_staged(
        (tokens, keys, generic_heads, kv_heads, head_dim), dtype="f32", mask=mask,
        scale=1.0, window=0, row_offset=0, library=staged_library,
        query=query, key=key, value=value,
    )
    assert exact_plan.pv_grid == (tokens, kv_heads, 1)
    assert generic_plan.pv_grid == (tokens, kv_heads, 2)
    assert np.isfinite(generic[:, :exact_heads]).all()
    assert np.isfinite(exact).all()
    # The shared row group: heads 0..7, KV head 0, in both launches.
    _assert_bit_identical(exact[:, :exact_heads // kv_heads],
                          generic[:, :exact_heads // kv_heads], "f32")


# --- the singleton (T0) PV decomposition ------------------------------------

# The decomposition a shape selects, read from the kernel's own resolver. It is
# declared here rather than in the wrapper module because it is a diagnostic for
# this test, not part of the launcher's calling contract.
_SYMBOL_PV_THREADS = "hipengine_gemma4_attention_staged_pv_threads"
_SYMBOL_PV_ROWS_PER_BLOCK = "hipengine_gemma4_attention_staged_pv_rows_per_block"
_SYMBOL_PV_GRID_Z = "hipengine_gemma4_attention_staged_pv_grid_z"


def _pv_decomposition(library, tokens: int, num_heads: int, num_kv_heads: int,
                      head_dim: int) -> tuple[int, int, int]:
    """(block width, rows per block, grid Z) the PV launcher would use."""

    import ctypes

    from hipengine.core.ctypes_cache import signed_kernel_fn

    answers = []
    for symbol in (_SYMBOL_PV_THREADS, _SYMBOL_PV_ROWS_PER_BLOCK, _SYMBOL_PV_GRID_Z):
        fn = signed_kernel_fn(
            library, symbol, (ctypes.c_int64,) * 4, ctypes.c_int64
        )
        answers.append(
            int(fn(int(tokens), int(num_heads), int(num_kv_heads), int(head_dim)))
        )
    return (answers[0], answers[1], answers[2])


# (tokens, num_heads, num_kv_heads, head_dim, block width, rows per block, grid Z)
_PV_DECOMPOSITION_TABLE = [
    # A single query token: 32 threads, two rows, and one 32-dimension slice per
    # CTA, so Z is the row-group count times head_dim / 32.
    (1, 16, 2, 512, 32, 2, 4 * 16),
    (1, 16, 8, 256, 32, 2, 1 * 8),
    (1, 4, 4, 256, 32, 1, 1 * 8),  # ratio 1: no second row to pair
    (1, 24, 2, 256, 32, 2, 6 * 8),  # ratio 12, every group full
    (1, 6, 2, 256, 32, 2, 2 * 8),  # ratio 3: a full group and a one-row tail
    (1, 10, 2, 512, 32, 2, 3 * 16),  # ratio 5: two full groups, one partial
    # The combined Z bound, both sides of it: 16,382 rows at head_dim 256 is
    # 8,191 row groups x 8 slices, and 8,192 x 8 does not fit one grid dimension.
    (1, 16382, 1, 256, 32, 2, 65528),
    (1, 16384, 1, 256, 256, 8, 2048),
    (1, 16383, 1, 256, 256, 8, 2048),  # an odd ratio past the bound, same fallback
    (1, 8190, 1, 512, 32, 2, 65520),
    (1, 8192, 1, 512, 256, 8, 1024),
    # More than one query token keeps the 256-thread decomposition, whose
    # resident row count comes from the variant table rather than the tile.
    (2, 16, 2, 512, 256, 8, 1),
    (7, 16, 2, 512, 256, 8, 1),
    (3, 32, 2, 512, 256, 8, 2),
    (2, 16, 8, 256, 256, 2, 1),  # an exact two-row variant
    (2, 4, 1, 512, 256, 4, 1),
    (1, 524280, 1, 256, 256, 8, 65535),  # the widest ratio the family serves
    # Not shapes this family serves at all: no ratio, a width the score tree does
    # not implement, and a ratio past the row-group grid.
    (0, 16, 2, 512, 0, 0, 0),
    (1, 3, 2, 256, 0, 0, 0),
    (1, 16, 2, 128, 0, 0, 0),
    (1, 524290, 1, 256, 0, 0, 0),
    (1, 2**31, 1, 256, 0, 0, 0),
]


def test_pv_decomposition_exports_match_the_planner(staged_library):
    """The kernel's own resolver and the Python planner agree, shape by shape.

    The launcher dispatches on the C++ resolver and the plan reports what Python
    computed; a disagreement would mean the plan named a kernel that did not run,
    so both sides are read here for the same table -- including the shapes that
    sit on and one past the combined Z bound, and the shapes neither side serves.
    """

    for tokens, num_heads, num_kv_heads, head_dim, threads, rows, grid_z in (
        _PV_DECOMPOSITION_TABLE
    ):
        assert _pv_decomposition(
            staged_library, tokens, num_heads, num_kv_heads, head_dim
        ) == (threads, rows, grid_z)
        if threads == 0:
            with pytest.raises((ValueError, NotImplementedError)):
                staged.staged_plan(
                    tokens=tokens, keys=64, num_heads=num_heads,
                    num_kv_heads=num_kv_heads, head_dim=head_dim,
                )
            continue
        plan = staged.staged_plan(
            tokens=tokens, keys=64, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        assert (plan.pv_threads, plan.pv_rows_per_block) == (threads, rows)
        assert plan.pv_grid[2] == grid_z
        # The selection is what keeps every launch inside the grid: the
        # singleton decomposition's packed Z fits, and so does the Z of the
        # decomposition a shape falls back to.
        assert plan.pv_grid[2] <= staged.MAX_GRID_DIM
        assert plan.pv_singleton == (threads == staged.SINGLETON_THREADS)
        assert plan.pv_lds_bytes == (
            rows * staged.PV_TILE_KEYS * 4
            if plan.pv_singleton
            else staged.staged_pv_lds_bytes()
        )


# The same comparison as a sweep, so the parity above is not only true of the
# named rows: every ratio either boundary can turn on, at both head widths, for a
# single token and for several.
_PV_DECOMPOSITION_SWEEP_RATIOS = (
    1, 2, 3, 4, 5, 8, 12, 16, 17, 255, 4095, 4096, 8190, 8191, 8192, 16382, 16383,
    16384, 16385, 65534 * 8, 65535 * 8, 65535 * 8 + 1,
)


def test_pv_decomposition_parity_holds_across_the_ratio_sweep(staged_library):
    """The launcher's resolver and the planner agree shape by shape, not by luck.

    The launcher dispatches on the C++ resolver this export reads, so an equality
    here is an equality between the grid the planner reports and the grid the
    launch uses -- for the singleton decomposition, for the eight-row fallback a
    packed-Z overflow takes, and for the shapes neither serves.
    """

    for tokens in (1, 2, 7):
        for head_dim in (256, 512):
            for ratio in _PV_DECOMPOSITION_SWEEP_RATIOS:
                kernel = _pv_decomposition(staged_library, tokens, ratio, 1, head_dim)
                try:
                    plan = staged.staged_plan(
                        tokens=tokens, keys=64, num_heads=ratio, num_kv_heads=1,
                        head_dim=head_dim,
                    )
                except NotImplementedError:
                    assert kernel == (0, 0, 0), (tokens, ratio, head_dim)
                    continue
                assert kernel == (
                    plan.pv_threads,
                    plan.pv_rows_per_block,
                    plan.pv_grid[2],
                ), (tokens, ratio, head_dim)
                assert plan.pv_grid[2] <= staged.MAX_GRID_DIM
                # The singleton decomposition is only ever the single-token one;
                # a packed-Z overflow keeps the eight-row grid instead.
                assert not plan.pv_singleton or tokens == 1


def test_singleton_decomposition_is_the_one_that_ran(staged_library):
    """Which PV kernel ran is read back from the launch, not inferred from time.

    The 256-thread decomposition holds a whole query row, so a single-token
    launch has one PV CTA per KV head. The singleton decomposition pairs the
    query rows and slices the head width instead, so the same launch has
    ``ceil(rows_per_head / 2) * head_dim / 32`` CTAs. Both are launched below
    and the decomposition that ran is asserted against the plan the launcher
    returned and against the kernel's own resolver.
    """

    for tokens, num_heads, num_kv_heads, head_dim, threads, rows, slices in (
        (1, 16, 2, 512, 32, 2, 16),  # the full-attention geometry
        (1, 16, 8, 256, 32, 2, 8),  # the sliding geometry
        (1, 4, 4, 256, 32, 1, 8),  # ratio 1: one resident row
        (1, 6, 2, 256, 32, 2, 8),  # ratio 3: a one-row tail group
        (2, 16, 2, 512, 256, 8, 1),  # two tokens: the 256-thread decomposition
    ):
        shape = (tokens, 512, num_heads, num_kv_heads, head_dim)
        mask = _mask(tokens, 512, "keep", 0, 0, seed=9)
        out, plan, _, _ = _run_staged(
            shape, dtype="bf16", mask=mask, scale=1.0, window=0, row_offset=0,
            library=staged_library,
        )
        assert np.isfinite(bf16_decode(out)).all()
        assert (plan.pv_threads, plan.pv_rows_per_block, plan.pv_dim_slices) == (
            threads,
            rows,
            slices,
        )
        assert plan.pv_grid == (tokens, num_kv_heads, plan.row_groups * slices)
        assert _pv_decomposition(
            staged_library, tokens, num_heads, num_kv_heads, head_dim
        ) == (threads, rows, plan.pv_grid[2])
        # A single-token launch has more independent PV CTAs than the
        # 256-thread decomposition would launch for the same shape, which is one
        # CTA per (token, KV head).
        if tokens == 1:
            assert plan.pv_singleton
            rows_per_head = num_heads // num_kv_heads
            multi_ctas = num_kv_heads * (
                -(-rows_per_head // staged.MAX_ROWS_PER_BLOCK)
            )
            assert num_kv_heads * plan.pv_grid[2] > multi_ctas


def test_singleton_grid_bound_launches_and_the_fallback_matches(staged_library):
    """Both sides of the combined Z bound, against the strict kernel.

    16,382 query rows at head_dim 256 is the last single-token shape whose
    row-group-by-slice product (65,528) fits one grid dimension, and 16,384 is
    one past it: the first launches the singleton decomposition with 65,528 CTAs
    in Z, the second keeps the 256-thread one. Both are compared bitwise with the
    strict kernel, which is what makes "the fallback is not a different result"
    a checked statement rather than a claim about the source the two share.
    """

    for num_heads, threads, grid_z in ((16382, 32, 65528), (16384, 256, 2048)):
        shape = (1, 64, num_heads, 1, 256)
        mask = _mask(1, 64, "keep", 0, 0, seed=10)
        reference, candidate, plan = _run_both(
            shape, dtype="bf16", mask=mask, scale=1.0, window=0, row_offset=0,
            library=staged_library,
        )
        assert (plan.pv_threads, plan.pv_grid[2]) == (threads, grid_z)
        assert plan.pv_grid[2] <= staged.MAX_GRID_DIM
        assert np.isfinite(bf16_decode(candidate)).all()
        _assert_bit_identical(reference, candidate, "bf16")


# --- the oracle, past the strict kernel's ceiling ---------------------------


def test_oracle_agrees_with_the_strict_kernel(staged_library):
    """Validate the oracle against the shipped kernel before trusting it alone.

    Every parity case above compares the candidate against the strict kernel;
    this one compares the *oracle* against the strict kernel, so the
    past-the-ceiling checks below are anchored to the same arithmetic.
    """

    shape = (2, 1025, 16, 2, 512)
    mask = _mask(2, 1025, "sliding", 128, 900, seed=5)
    reference, candidate, _ = _run_both(
        shape, dtype="f32", mask=mask, scale=1.0, window=128, row_offset=900,
        library=staged_library,
    )
    query, key, value = _inputs(*shape, dtype="f32", seed=17)
    oracle = staged_cpu_oracle(
        query, key, value, mask, num_kv_heads=2, scale=1.0, window=128, row_offset=900
    )
    np.testing.assert_allclose(reference, oracle, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(candidate, oracle, rtol=1e-5, atol=1e-6)


def test_strict_global_fallback_and_staged_plan_serve_long_contexts():
    """Both bounded-LDS implementations admit the repaired head geometries.

    Admission is host-only: never pass null pointers to a path that can launch.
    """
    for head_dim, keys in ((512, 17000), (256, 17000)):
        assert strict.gemma4_attention_shared_bytes(head_dim=head_dim, keys=keys) <= 65536
        plan = staged.staged_plan(
            tokens=1, keys=keys, num_heads=16, num_kv_heads=2, head_dim=head_dim
        )
        assert plan.keys == keys
        assert plan.lds_bytes <= staged.LDS_BUDGET_BYTES
        assert keys > STRICT_KEY_CEILING[head_dim]


@pytest.mark.parametrize("dtype,rtol,atol", [("f32", 1e-4, 1e-5), ("bf16", 1e-2, 1e-3)])
def test_staged_matches_the_oracle_past_the_strict_ceiling(staged_library, dtype, rtol, atol):
    """17,000 keys at head_dim 512: 1,384 past the strict ceiling.

    The reference here is the float64 oracle, given the same values the kernel
    reads. ``scale`` is small on purpose: it keeps the softmax broad over the
    whole context, so the denominator is a 17,000-term sum of comparable terms
    and the tolerance measures the accumulation order rather than a single
    winning key.
    """

    shape = (1, 17000, 16, 2, 512)
    mask = _mask(1, 17000, "keep", 0, 0, seed=6)
    out, plan, (query, key, value), _ = _run_staged(
        shape, dtype=dtype, mask=mask, scale=0.05, window=0, row_offset=0,
        library=staged_library,
    )
    oracle = staged_cpu_oracle(
        query, key, value, mask, num_kv_heads=2, scale=0.05
    )
    decoded = bf16_decode(out) if dtype == "bf16" else out
    assert np.isfinite(decoded).all()
    # The softmax really is broad: no single key holds half of head 0's mass, so
    # the tolerance above is a statement about accumulating 17,000 terms rather
    # than about reproducing one winning key.
    logits = key[:, 0, :] @ query[0, 0, :] * 0.05
    weights = np.exp(logits - logits.max())
    assert (weights / weights.sum()).max() < 0.5
    np.testing.assert_allclose(decoded, oracle, rtol=rtol, atol=atol)
    # Independent CPU-reference outer safety floor, distinct from strict parity.
    from scripts.gemma4_teacher_forced_gate import row_kl_divergence
    reference_rows = oracle.reshape(-1, shape[-1])
    candidate_rows = decoded.reshape(-1, shape[-1])
    assert max(row_kl_divergence(b, c) for b, c in
               zip(reference_rows, candidate_rows)) <= 0.05
    assert np.mean(np.argmax(reference_rows, axis=1) ==
                   np.argmax(candidate_rows, axis=1)) >= 0.90
    assert plan.chunks == 17
    assert plan.pv_singleton
    assert plan.pv_lds_bytes == staged.SINGLETON_ROWS_PER_BLOCK * staged.PV_TILE_KEYS * 4
    assert plan.lds_bytes == max(
        plan.score_lds_bytes, plan.softmax_lds_bytes, plan.pv_lds_bytes
    )


def test_staged_sliding_band_matches_the_oracle_past_the_ceiling(staged_library):
    """A sliding layer at 17,000 keys keeps a 1,024-key band, not the context.

    head_dim 256's strict ceiling is 15,872 keys, so this shape is unlaunchable
    there; it is also the case the score stage's chunk skip exists for, since
    only one of the seventeen chunks overlaps the band.
    """

    shape = (1, 17000, 16, 8, 256)
    window, row_offset = 1024, 15976
    mask = _mask(1, 17000, "sliding", window, row_offset, seed=7)
    out, plan, (query, key, value), _ = _run_staged(
        shape, dtype="f32", mask=mask, scale=1.0, window=window, row_offset=row_offset,
        library=staged_library,
    )
    oracle = staged_cpu_oracle(
        query, key, value, mask, num_kv_heads=8, scale=1.0, window=window,
        row_offset=row_offset,
    )
    np.testing.assert_allclose(out, oracle, rtol=1e-4, atol=1e-5)
    assert plan.chunks == 17
    assert int(mask.sum()) == window


# (num_heads, num_kv_heads, head_dim, keys, row groups, slices)
_SINGLETON_LONG_KEY_SHAPES = [
    # The sliding geometry's ratio 2 at head_dim 256 (strict ceiling 15,872): one
    # row group and eight slices, so 8 CTAs of 32 threads walk 20,000 keys in
    # 79 tiles each.
    (16, 8, 256, 20000, 1, 8),
    # Ratio 3: a full group and a one-row tail, which is the generic variant.
    (6, 2, 256, 17000, 2, 8),
    # Ratio 5 past the head_dim 512 ceiling (15,616): two full groups, one partial.
    (10, 2, 512, 16000, 3, 16),
]


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,keys,groups,slices", _SINGLETON_LONG_KEY_SHAPES
)
def test_singleton_long_keys_match_the_oracle_past_the_strict_ceiling(
    staged_library, num_heads, num_kv_heads, head_dim, keys, groups, slices
):
    """A single token over a context the strict kernel cannot launch.

    Every shape here is past the strict kernel's shared-memory ceiling, so the
    reference is the independent float64 oracle. The key counts are long enough
    that each CTA walks many tiles of the shared weight tile, and the GQA ratios
    cover the full-group and the partial-tail variants of the decomposition.
    """

    shape = (1, keys, num_heads, num_kv_heads, head_dim)
    mask = _mask(1, keys, "keep", 0, 0, seed=12)
    out, plan, (query, key, value), _ = _run_staged(
        shape, dtype="f32", mask=mask, scale=0.05, window=0, row_offset=0,
        library=staged_library,
    )
    assert plan.pv_singleton
    assert (plan.row_groups, plan.pv_dim_slices) == (groups, slices)
    assert plan.pv_grid == (1, num_kv_heads, groups * slices)
    assert keys > STRICT_KEY_CEILING[head_dim]
    assert np.isfinite(out).all()
    oracle = staged_cpu_oracle(query, key, value, mask, num_kv_heads=num_kv_heads, scale=0.05)
    np.testing.assert_allclose(out, oracle, rtol=1e-4, atol=1e-5)


def test_row_zero_is_identical_across_token_decompositions(staged_library):
    """Batch-composition invariance at a context the strict kernel cannot launch.

    Row 0's inputs, mask and arithmetic are the same in both launches; only the
    number of rows sharing the device changes. If the row grouping or the
    tiling leaked into a row's accumulation, this would disagree even though
    each launch looked self-consistent.
    """

    keys, num_heads, num_kv_heads, head_dim = 17000, 16, 2, 512
    single_shape = (1, keys, num_heads, num_kv_heads, head_dim)
    rng = np.random.default_rng(23)
    query = rng.standard_normal((3, num_heads, head_dim), dtype=np.float32) * 0.7
    key = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 0.7
    value = rng.standard_normal((keys, num_kv_heads, head_dim), dtype=np.float32) * 0.7
    mask = np.ones((3, keys), dtype=np.uint8)
    # Rows 1 and 2 keep a narrow band; row 0 keeps everything, in both runs.
    index = np.arange(keys)[None, :]
    mask[1] = ((index <= 9000) & (index > 8000)).astype(np.uint8)
    mask[2] = ((index <= 15000) & (index > 14000)).astype(np.uint8)

    single, single_plan, _, _ = _run_staged(
        single_shape, dtype="f32", mask=mask[:1], scale=1.0, window=0, row_offset=0,
        library=staged_library, query=query[:1], key=key, value=value,
    )
    batched, plan, _, _ = _run_staged(
        (3, keys, num_heads, num_kv_heads, head_dim), dtype="f32", mask=mask, scale=1.0,
        window=0, row_offset=0, library=staged_library, query=query, key=key, value=value,
    )
    assert plan.pv_grid == (3, 2, 1)
    # The two launches take different PV decompositions -- the single token one
    # and the multi-token one -- so the equality below is the statement that the
    # decomposition is a code shape and not a different result.
    assert single_plan.pv_singleton and not plan.pv_singleton
    _assert_bit_identical(single[0], batched[0], "f32")
    # Rows 1 and 2 are not degenerate, so the equality above is not a
    # comparison of two empty rows.
    assert np.isfinite(batched[1]).all() and np.isfinite(batched[2]).all()
    assert not np.array_equal(batched[0], batched[1])


# --- launch bounds at the edges of the coordinate space ----------------------

# (tokens, keys, num_heads, num_kv_heads, head_dim, window, row_offset, expected)
_HUGE_COORDINATE_SHAPES = [
    # 2**31: one past INT32_MAX, so narrowing the row's own position to an int
    # would wrap it negative and index the key range from the wrong end.
    (1, 8, 2, 1, 256, 4, 2**31, "empty"),
    # 2**62: the sum with the window saturates before it is clamped.
    (1, 8, 2, 1, 256, 4, 2**62, "empty"),
    # A row offset far below zero: the clamped window is empty at the other end.
    (1, 8, 2, 1, 256, 4, -(2**62), "empty"),
    # A window as wide as the offset: the intersection is the tail of the key
    # range, [1, 8), so this row is a real attention result rather than a NaN.
    (1, 8, 2, 1, 256, 2**40, 2**40, "clamped"),
    # own can overflow int64, while own-window remains a small exact bound.
    (2, 8, 2, 1, 256, 2**63-1, 2**63-1, "clamped"),
]


@pytest.mark.parametrize(
    "tokens,keys,num_heads,num_kv_heads,head_dim,window,row_offset,expected",
    _HUGE_COORDINATE_SHAPES,
)
def test_huge_coordinates_stay_inside_the_key_range(
    staged_library, tokens, keys, num_heads, num_kv_heads, head_dim, window, row_offset,
    expected,
):
    """An extreme window or row offset narrows the walk; it never indexes past it.

    The strict kernel narrows both bounds through a 32-bit cast, so these
    coordinates are not a parity case -- they are a statement about this
    candidate's own bounds arithmetic, checked against the independent oracle
    where the clamped walk is non-empty and against NaN where it is empty.
    """

    shape = (tokens, keys, num_heads, num_kv_heads, head_dim)
    if expected == "empty":
        # No column is inside the clamped window, so the row has no live key.
        mask = np.zeros((tokens, keys), dtype=np.uint8)
    else:
        # Use Python integers so constructing the mask cannot overflow either.
        mask = np.array([[int(row_offset + t - window < j <= row_offset + t)
                          for j in range(keys)] for t in range(tokens)], dtype=np.uint8)
    out, plan, (query, key, value), _ = _run_staged(
        shape, dtype="f32", mask=mask, scale=1.0, window=window, row_offset=row_offset,
        library=staged_library,
    )
    assert plan.keys == keys
    if expected == "empty":
        assert np.isnan(out).all()
        return
    oracle = staged_cpu_oracle(
        query, key, value, mask, num_kv_heads=num_kv_heads, scale=1.0, window=window,
        row_offset=row_offset,
    )
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, oracle, rtol=1e-4, atol=1e-5)


def test_workspace_export_returns_its_invalid_sentinel(staged_library):
    """The kernel's export never wraps: a shape it cannot serve returns 0.

    The Python planner raises a named reason for the same shapes, so "a size
    exists" means the same thing on both sides of the ABI -- which is what lets
    the launcher size the allocation from either one.
    """

    unservable = [
        (0, 1, 1),  # not a shape at all
        (2**31, 1, 64),  # one past score/PV X
        (1, 2**31, 64),  # one past the score row grid in heads
        (2**62, 4, 1),  # a row product that would overflow int64 if formed
        (1, 1, staged.MAX_KEYS + 1),  # one past the score grid's chunk dimension
    ]
    for tokens, num_heads, keys in unservable:
        assert staged.staged_workspace_bytes_from_kernel(
            tokens, num_heads, keys, library=staged_library
        ) == 0
        with pytest.raises((ValueError, NotImplementedError)):
            staged.staged_workspace_bytes(tokens, num_heads, keys)
    # The sentinel is not a size any launchable shape has, and the widest row
    # grid the shape serves still answers with its exact byte count.
    edge = staged.staged_workspace_bytes(65536, 1, 64)
    assert edge > 0
    assert staged.staged_workspace_bytes_from_kernel(
        65536, 1, 64, library=staged_library
    ) == edge


@pytest.mark.parametrize("heads,kv_heads", [(65536, 65536), (524281, 1)])
def test_raw_launcher_rejects_pv_y_z_before_dereferencing(staged_library, heads, kv_heads):
    import ctypes
    from hipengine.core.ctypes_cache import signed_kernel_fn

    fn = signed_kernel_fn(staged_library, staged.SYMBOL_STAGED_F32,
                          staged._ARGTYPES_STAGED, ctypes.c_int)
    # Invalid shape must be distinguishable from missing workspace. All pointers
    # stay null so a removed grid guard still cannot reach a GPU launch.
    assert fn(None, None, None, None, None, 1, heads, kv_heads, 256,
              ctypes.c_float(1.0), None, 1, 0, 0, None) == 1
    assert fn(None, None, None, None, None, 1, 2, 1, 256,
              ctypes.c_float(1.0), None, 1, 0, 0, None) == 17
