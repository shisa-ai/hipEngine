"""Bitwise parity and oracle agreement for the staged Gemma 4 attention candidate.

Two independent checks, because the candidate exists to serve a shape the
reference cannot launch:

* **Parity.** The staged launcher and the shipped strict wrapper are driven from
  identical device buffers with identical arguments, and their outputs are
  compared bit for bit -- the ``uint16`` payloads for BF16 and the ``uint32``
  payloads for F32, so ``0.0`` and ``-0.0`` are different results and one NaN is
  not interchangeable with another. That is
  the strongest available statement of the candidate's contract, and it covers
  both the one-token decode routing and multi-token blocks, causal, sliding and
  holed masks, nonzero ``window``/``row_offset``, and both head widths.
* **Oracle.** Past 15,872 keys at head_dim 256 (15,616 at 512) the strict
  kernel's shared-memory requirement exceeds the 64 KB budget and it refuses to
  launch, so no bitwise reference exists there. Those cases are compared against
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
    (7, 257, 16, 2, 512, "causal", 0, 0, 1.0),
    (5, 1025, 16, 8, 256, "causal", 0, 0, 1.0),
    (4, 512, 4, 1, 512, "causal", 0, 0, 1.0),  # GQA ratio 4
    (1, 1025, 16, 2, 512, "sliding", 96, 1024, 1.0),
    (3, 1025, 16, 8, 256, "sliding", 64, 900, 1.0),
    (2, 2055, 16, 2, 512, "sliding", 128, 1500, 1.0),
    (2, 1025, 16, 2, 512, "holes", 0, 0, 1.0),
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
    """The same launch through the strict wrapper and the staged candidate."""

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
        strict_entry = (
            strict.gemma4_attention_prefill_bf16
            if dtype == "bf16"
            else strict.gemma4_attention_prefill_f32
        )
        strict_entry(
            query_buffer.ptr, key_buffer.ptr, value_buffer.ptr, mask_buffer.ptr,
            strict_buffer.ptr, **kwargs,
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
    assert plan.lds_bytes == staged.staged_lds_bytes(256)


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
    assert plan.score_grid == (48, 4)
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
        plan = staged.staged_plan(
            tokens=1, keys=64, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=256,
        )
        # The variant changes the kernel body only: the grid the plan reports is
        # the one the launch uses either way.
        assert plan.rows_per_head == rows_per_head
        assert plan.row_groups == -(-rows_per_head // staged.MAX_ROWS_PER_BLOCK)
        assert plan.pv_grid == (1, num_kv_heads, plan.row_groups)
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


def test_strict_kernel_refuses_the_context_the_candidate_serves():
    """The reason this candidate exists, asserted rather than assumed.

    No buffers are passed: the strict wrapper validates the shared-memory
    requirement before it builds or launches anything, so this costs nothing and
    documents the exact ceiling the staged path lifts.
    """

    for head_dim, keys in ((512, 17000), (256, 17000)):
        with pytest.raises(NotImplementedError, match="shared memory"):
            strict.gemma4_attention_prefill_bf16(
                0, 0, 0, 0, 0,
                tokens=1, keys=keys, num_heads=16, num_kv_heads=2, head_dim=head_dim,
                scale=1.0,
            )
        # ...and the candidate plans the same shape without complaint.
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
    assert plan.lds_bytes == staged.staged_lds_bytes(512)


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

    single, _, _, _ = _run_staged(
        single_shape, dtype="f32", mask=mask[:1], scale=1.0, window=0, row_offset=0,
        library=staged_library, query=query[:1], key=key, value=value,
    )
    batched, plan, _, _ = _run_staged(
        (3, keys, num_heads, num_kv_heads, head_dim), dtype="f32", mask=mask, scale=1.0,
        window=0, row_offset=0, library=staged_library, query=query, key=key, value=value,
    )
    assert plan.pv_grid == (3, 2, 1)
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
