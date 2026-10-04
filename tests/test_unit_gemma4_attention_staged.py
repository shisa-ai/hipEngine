"""Planner, workspace-ownership and oracle contracts for staged Gemma 4 attention.

CPU only: no HIP, no build, no device. The GPU half of this candidate's contract
-- bitwise parity with the shipped strict wrapper, and agreement with the
independent oracle below on a context the strict kernel cannot launch -- lives in
``tests/test_gpu_gemma4_attention_staged.py`` and is reached with an explicit
file target.

The oracle in this file is deliberately *not* the kernel's algorithm. It
computes masked ungated attention from the definition, in float64, in dimension
order, with no tree and no partition; the tests below pin it against hand-computed
values so that a GPU comparison against it is evidence about the kernel rather
than about a shared implementation.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as strict
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_staged as staged


class Runtime:
    """A fake device runtime: the ownership contract needs no GPU to check."""

    def __init__(self) -> None:
        self.live: dict[int, int] = {}
        self.events: list[tuple] = []
        self.next_ptr = 4096
        self.device = 0

    def current_device(self) -> int:
        return self.device

    def malloc(self, size: int) -> int:
        self.next_ptr += 4096
        self.live[self.next_ptr] = size
        self.events.append(("malloc", self.next_ptr))
        return self.next_ptr

    def free(self, ptr: int) -> None:
        self.events.append(("free", ptr))
        del self.live[ptr]

    def stream_synchronize(self, stream: int) -> None:
        self.events.append(("sync", stream))


class FailingLibrary:
    """A fake kernel library whose entry points report a launch failure.

    ``signed_kernel_fn`` assigns ``argtypes``/``restype`` onto whatever
    ``getattr`` returns and then calls it, so the fake hands out one callable
    object per symbol that accepts those attributes and returns a non-success
    HIP error code.
    """

    class _Entry:
        argtypes = None
        restype = None

        def __call__(self, *args):
            return 1

    def __getattr__(self, name: str) -> "FailingLibrary._Entry":
        entry = FailingLibrary._Entry()
        object.__setattr__(self, name, entry)
        return entry


class FailingRuntime(Runtime):
    """The fake device runtime, with the launch-failure check a real one has."""

    def check(self, code: int) -> None:
        raise RuntimeError(f"fake launch failure {code}")


# --- the launch interface ---------------------------------------------------


def test_launcher_exposes_the_strict_wrapper_keywords():
    """The staged launcher must be call-compatible with the strict wrapper.

    The two are driven from one call site -- same buffers, same shape and mask
    arguments -- so a caller can compare them without an adapter. The five
    pointers stay positional and every other argument stays keyword-only, which
    is the strict wrapper's shape as well.
    """

    signature = inspect.signature(staged.gemma4_attention_staged_bf16)
    positional = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["query_ptr", "key_ptr", "value_ptr", "keep_mask_ptr", "out_ptr"]
    keywords = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert keywords == {
        "tokens",
        "keys",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "scale",
        "window",
        "row_offset",
        "stream",
        "scratch",
        "library",
        "runtime",
    }
    strict_signature = inspect.signature(strict.gemma4_attention_prefill_bf16)
    strict_keywords = {
        name
        for name, parameter in strict_signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert strict_keywords <= keywords
    for name in ("keys", "window", "row_offset", "stream", "scratch", "library", "runtime"):
        assert signature.parameters[name].default in (None, 0), name
    assert signature.parameters["keys"].default is None
    assert signature.parameters["window"].default == 0
    assert signature.parameters["row_offset"].default == 0


def test_f32_entry_point_shares_the_shape():
    bf16 = inspect.signature(staged.gemma4_attention_staged_bf16)
    f32 = inspect.signature(staged.gemma4_attention_staged_f32)
    assert list(bf16.parameters) == list(f32.parameters)
    assert [p.default for p in bf16.parameters.values()] == [
        p.default for p in f32.parameters.values()
    ]


# --- workspace sizing and layout --------------------------------------------


def test_workspace_layout_is_three_regions_and_matches_the_bytes():
    tokens, heads, keys = 1, 16, 4096
    layout = staged._staged_workspace_layout(tokens, heads, keys)
    rows = tokens * heads
    chunks = staged.staged_score_chunks(keys)
    assert layout.scores_floats == rows * keys
    assert layout.chunk_max_floats == rows * chunks
    assert layout.denominator_floats == rows
    assert layout.total_floats == rows * keys + rows * chunks + rows
    assert layout.total_bytes == layout.total_floats * 4
    assert staged.staged_workspace_bytes(tokens, heads, keys) == layout.total_bytes
    assert chunks == 4


def test_workspace_grows_with_every_dimension_and_rounds_chunks_up():
    base = staged.staged_workspace_bytes(1, 16, 1024)
    assert staged.staged_workspace_bytes(2, 16, 1024) > base
    assert staged.staged_workspace_bytes(1, 32, 1024) > base
    assert staged.staged_workspace_bytes(1, 16, 2048) > base
    # A key count that is not a multiple of the score chunk still allocates the
    # chunk maxima for the partial chunk, or the last chunk's maximum would be
    # written past the end of the workspace.
    assert staged.staged_score_chunks(1025) == 2
    assert staged.staged_workspace_bytes(1, 1, 1025) == (
        1 * 1025 + 1 * 2 + 1
    ) * 4
    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            staged.staged_workspace_bytes(bad, 16, 1024)


def test_workspace_is_exposed_for_the_shapes_the_model_uses():
    """The two real layer geometries, so the size is known before a launch."""

    # Sliding layer: 16 query heads, 8 KV heads, head_dim 256.
    sliding = staged.staged_plan(
        tokens=1, keys=262144, num_heads=16, num_kv_heads=8, head_dim=256
    )
    # Full layer: 16 query heads, 2 KV heads, head_dim 512.
    full = staged.staged_plan(
        tokens=1, keys=262144, num_heads=16, num_kv_heads=2, head_dim=512
    )
    for plan in (sliding, full):
        assert plan.workspace_bytes == staged.staged_workspace_bytes(
            plan.tokens, plan.num_heads, plan.keys
        )
        assert plan.workspace_bytes > 16 * 262144 * 4  # the scores region alone


# --- the bounded shared-memory property -------------------------------------


def test_shared_footprint_does_not_grow_with_the_context():
    """The candidate's whole reason to exist, stated as an assertion.

    The strict family's resident requirement is a function of ``keys`` and, once
    it passes the 64 KiB budget at 15,616 keys for head_dim 512, stops growing
    and moves its logits to request-owned global scratch. The staged candidate's
    requirement is a constant 8 KB.
    """

    assert staged.staged_lds_bytes(256) == 8 * 256 * 4
    assert staged.staged_lds_bytes(512) == 8 * 256 * 4
    assert staged.staged_lds_bytes(256) == staged.staged_lds_bytes(512)
    assert staged.staged_lds_bytes(512) <= staged.LDS_BUDGET_BYTES
    # The plan's own figure is the same constant at 31 keys and at 262,144,
    # and it is the reservation of the decomposition that plan selected.
    small = staged.staged_plan(
        tokens=1, keys=31, num_heads=16, num_kv_heads=2, head_dim=512
    )
    large = staged.staged_plan(
        tokens=1, keys=262144, num_heads=16, num_kv_heads=2, head_dim=512
    )
    assert small.lds_bytes == large.lds_bytes
    assert small.lds_bytes <= staged.staged_lds_bytes(512)
    assert small.lds_bytes == max(
        small.score_lds_bytes, small.softmax_lds_bytes, small.pv_lds_bytes
    )
    # Contrast: the strict requirement grows with the context up to its resident
    # budget, and past the budget it stops growing -- the 256/512 class kernel
    # moves its logits to request-owned global scratch rather than refusing. The
    # staged candidate's own figure is the constant above at every key count.
    assert strict.gemma4_attention_shared_bytes(head_dim=512, keys=8192) == (512 + 8192 + 256) * 4
    resident = strict.gemma4_attention_shared_bytes(head_dim=512, keys=16384)
    assert resident == strict.gemma4_attention_shared_bytes(head_dim=512, keys=262144)
    assert resident < staged.LDS_BUDGET_BYTES
    # ...and the same context is a normal plan for the staged candidate.
    assert large.keys == 262144


# --- capability and shape validation ----------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128, 192, 384, 768, 1024])
def test_a_width_the_score_tree_does_not_implement_is_named(head_dim):
    """A capability miss is raised, not approximated or silently narrowed."""

    with pytest.raises(NotImplementedError) as failure:
        staged.staged_plan(
            tokens=1, keys=64, num_heads=2, num_kv_heads=1, head_dim=head_dim
        )
    message = str(failure.value)
    assert "256" in message and "512" in message and str(head_dim) in message
    assert "score tree" in message


def test_key_ceiling_names_the_grid_limit():
    with pytest.raises(NotImplementedError) as failure:
        staged.staged_plan(
            tokens=1,
            keys=staged.MAX_KEYS + staged.SCORE_CHUNK_KEYS,
            num_heads=2,
            num_kv_heads=1,
            head_dim=256,
        )
    assert str(staged.MAX_KEYS) in str(failure.value)
    # Exactly at the ceiling is a plan, not a refusal.
    plan = staged.staged_plan(
        tokens=1, keys=staged.MAX_KEYS, num_heads=2, num_kv_heads=1, head_dim=256
    )
    assert plan.chunks == staged.MAX_KEY_CHUNKS


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(tokens=0, keys=8, num_heads=2, num_kv_heads=1, head_dim=256), "tokens"),
        (dict(tokens=1, keys=0, num_heads=2, num_kv_heads=1, head_dim=256), "keys"),
        (dict(tokens=1, keys=8, num_heads=0, num_kv_heads=1, head_dim=256), "num_heads"),
        (dict(tokens=1, keys=8, num_heads=2, num_kv_heads=0, head_dim=256), "num_kv_heads"),
        (dict(tokens=1, keys=8, num_heads=2, num_kv_heads=1, head_dim=0), "head_dim"),
        (dict(tokens=1, keys=8, num_heads=3, num_kv_heads=2, head_dim=256), "multiple"),
    ],
)
def test_shape_validation_rejects_nonsense_before_any_device_work(kwargs, match):
    with pytest.raises(ValueError, match=match):
        staged.staged_plan(**kwargs)


@pytest.mark.parametrize(
    "kwargs, named",
    [
        # PV Y is limited independently from the larger score/PV X dimension.
        (dict(tokens=1, keys=64, num_heads=65536, num_kv_heads=65536, head_dim=256), "65536"),
        (dict(tokens=2**31, keys=64, num_heads=1, num_kv_heads=1, head_dim=256), str(2**31)),
        # GQA ratio 524,281 would need ceil(524281 / 8) = 65,536 PV row groups.
        (
            dict(tokens=1, keys=64, num_heads=1048562, num_kv_heads=2, head_dim=256),
            "1048562",
        ),
        # A token count whose row product overflows int64 if it is multiplied
        # before it is bounded.
        (dict(tokens=2**62, keys=64, num_heads=4, num_kv_heads=1, head_dim=256), str(2**62)),
    ],
)
def test_a_shape_past_a_grid_dimension_is_named(kwargs, named):
    """Every launch dimension is bounded, and a miss names the dimension.

    The score/PV X dimension allows 2**31-1 blocks; Y/Z allow 65535.
    All products must be bounded before allocation or raw ctypes conversion.
    """

    with pytest.raises(NotImplementedError) as failure:
        staged.staged_plan(**kwargs)
    message = str(failure.value)
    assert "grid" in message
    assert named in message
    # Workspace sizing knows no KV-head count, so Y/Z validation is necessarily
    # performed only by the full planner/launcher.
    if kwargs["tokens"] > 2**31 - 1:
        with pytest.raises(NotImplementedError):
            staged.staged_workspace_bytes(kwargs["tokens"], kwargs["num_heads"], kwargs["keys"])


def test_the_grid_boundaries_are_plans_not_refusals():
    """Exactly at a limit is a plan; the tests above are one past each limit."""

    edge = staged.staged_plan(tokens=65535, keys=64, num_heads=1, num_kv_heads=1, head_dim=256)
    assert edge.rows == staged.MAX_GRID_DIM
    assert edge.score_grid == (65535, 1)
    assert edge.pv_grid == (65535, 1, 1)
    assert staged.staged_workspace_bytes(65535, 1, 64) == (
        65535 * 64 + 65535 * 1 + 65535
    ) * 4
    # X is not limited to 65535, including a non-threshold 17K-query shape.
    for tokens, heads in ((65536, 1), (17000, 16)):
        plan = staged.staged_plan(tokens=tokens, keys=1, num_heads=heads,
                                  num_kv_heads=1, head_dim=256)
        assert plan.score_grid == (tokens * heads, 1)
    x_edge = staged.staged_plan(tokens=2**31-1, keys=1, num_heads=1,
                               num_kv_heads=1, head_dim=256)
    assert x_edge.rows == 2**31-1
    z_edge = staged.staged_plan(tokens=1, keys=1, num_heads=65535*8,
                               num_kv_heads=1, head_dim=256)
    assert z_edge.pv_grid == (1, 1, 65535)


def test_workspace_products_past_representability_are_named():
    """The layout helper is where the products are formed, so it checks them.

    ``staged_plan`` and :func:`staged.staged_workspace_bytes` bound the shape
    first, so this is the direct-caller path -- and the definition the kernel's
    own export mirrors when it returns its 0 sentinel instead.
    """

    with pytest.raises(ValueError, match="representable"):
        staged._staged_workspace_layout(2**62, 4, 1)
    # A layout that fits: the largest shape the launch serves still has room.
    largest = staged._staged_workspace_layout(
        staged.MAX_GRID_DIM, 1, staged.MAX_KEYS
    )
    assert largest.total_floats > 0


def test_plan_reports_the_grid_and_the_row_grouping():
    # The artifact's full-layer geometry: 8 query heads per KV head, so one PV
    # CTA covers the whole GQA group and the grid is (tokens, 2, 1).
    full = staged.staged_plan(tokens=3, keys=4096, num_heads=16, num_kv_heads=2, head_dim=512)
    assert (full.rows, full.rows_per_head, full.row_groups) == (48, 8, 1)
    assert full.pv_grid == (3, 2, 1)
    assert full.score_grid == (48, 4)
    assert full.softmax_grid == (48, 1)
    # Sliding geometry: 2 query heads per KV head.
    sliding = staged.staged_plan(tokens=3, keys=1024, num_heads=16, num_kv_heads=8, head_dim=256)
    assert (sliding.rows_per_head, sliding.row_groups) == (2, 1)
    assert sliding.pv_grid == (3, 8, 1)
    # A GQA ratio wider than one CTA's row budget splits into row groups.
    wide = staged.staged_plan(tokens=3, keys=1024, num_heads=32, num_kv_heads=2, head_dim=512)
    assert (wide.rows_per_head, wide.row_groups) == (16, 2)
    assert wide.pv_grid == (3, 2, 2)
    # A row per CTA, like the strict kernel, when the ratio is one.
    plain = staged.staged_plan(tokens=3, keys=64, num_heads=4, num_kv_heads=4, head_dim=256)
    assert (plain.rows_per_head, plain.row_groups, plain.pv_grid) == (1, 1, (3, 4, 1))


# --- the T0 singleton PV decomposition ---------------------------------------


def test_single_token_selects_the_32_thread_pv_decomposition():
    """One query token gets the T0 singleton PV: 32 threads, 2 rows, 32-dim slices.

    The 256-thread decomposition holds every dimension of a query row in one
    CTA, so a single-token launch has as few PV CTAs as the layer has KV heads
    (two, for the full-attention geometry). The singleton decomposition pairs
    the query rows that share a KV head and slices the output dimensions
    instead, so the same launch has ``ceil(rows_per_head / 2) * head_dim / 32``
    independent CTAs -- and every output element is still one ascending-key FP32
    FMA chain, because a slice decides which dimensions a CTA owns, never which
    keys contribute to one.
    """

    for head_dim, slices in ((256, 8), (512, 16)):
        plan = staged.staged_plan(
            tokens=1, keys=4096, num_heads=16, num_kv_heads=2, head_dim=head_dim
        )
        assert plan.pv_threads == staged.SINGLETON_THREADS == 32
        assert plan.pv_rows_per_block == staged.SINGLETON_ROWS_PER_BLOCK == 2
        assert plan.pv_dim_slices == slices
        assert plan.pv_dim_slices * staged.SINGLETON_DIM_SLICE == head_dim
        # 8 query heads per KV head, two resident rows per group.
        assert plan.row_groups == 4
        assert plan.pv_grid == (1, 2, 4 * slices)
        assert plan.pv_singleton
        # The score and softmax stages keep the strict 256-lane shapes.
        assert plan.threads == staged.THREADS == 256
        assert plan.score_grid == (16, 4)
        assert plan.softmax_grid == (16, 1)
        assert plan.pv_lds_bytes == 2 * staged.PV_TILE_KEYS * 4
        assert plan.lds_bytes == max(
            plan.score_lds_bytes, plan.softmax_lds_bytes, plan.pv_lds_bytes
        )


def test_multi_token_launches_keep_the_256_thread_decomposition():
    """More than one query token is the shipped decomposition, unchanged."""

    for num_heads, num_kv_heads, resident in (
        (16, 2, 8),  # the full-attention geometry
        (16, 8, 2),  # the sliding geometry: an exact variant with two rows
        (4, 1, 4),
        (24, 2, 8),  # ratio 12: the generic variant
    ):
        for tokens in (2, 3, 17):
            plan = staged.staged_plan(
                tokens=tokens, keys=1024, num_heads=num_heads,
                num_kv_heads=num_kv_heads, head_dim=512,
            )
            assert plan.pv_threads == staged.THREADS == 256
            assert plan.pv_rows_per_block == resident
            # One CTA holds the whole head width: the dimensions are not sliced.
            assert plan.pv_dim_slices == 1
            assert plan.row_groups == -(-(num_heads // num_kv_heads) // staged.MAX_ROWS_PER_BLOCK)
            assert plan.pv_grid == (tokens, num_kv_heads, plan.row_groups)
            assert not plan.pv_singleton
            # The 256-thread launcher reserves its eight-row tile for every
            # variant, so that -- not the resident row count -- is the footprint.
            assert plan.pv_lds_bytes == staged.staged_pv_lds_bytes()


def test_singleton_geometry_covers_every_gqa_ratio_shape():
    """Ratio 1, an even ratio, and odd ratios whose last group holds one row."""

    # A row per CTA when the ratio is one: no second row to pair with.
    one = staged.staged_plan(tokens=1, keys=64, num_heads=4, num_kv_heads=4, head_dim=256)
    assert (one.pv_rows_per_block, one.row_groups, one.pv_grid) == (1, 1, (1, 4, 8))
    # An even ratio fills every group with two rows.
    even = staged.staged_plan(tokens=1, keys=64, num_heads=24, num_kv_heads=2, head_dim=256)
    assert (even.pv_rows_per_block, even.row_groups) == (2, 6)
    assert even.pv_grid == (1, 2, 48)
    # An odd ratio keeps a partial tail group; the row count is the group's own.
    for ratio, groups in ((3, 2), (5, 3), (9, 5)):
        odd = staged.staged_plan(
            tokens=1, keys=64, num_heads=2 * ratio, num_kv_heads=2, head_dim=256
        )
        assert (odd.pv_rows_per_block, odd.row_groups) == (2, groups)
        assert odd.pv_grid == (1, 2, groups * 8)


def test_singleton_grid_z_is_the_row_group_times_slice_product():
    """Z packs the row group and the dimension slice, so their product is the bound.

    A single-token launch whose combined Z does not fit one grid dimension keeps
    the 256-thread decomposition -- the same code path a multi-token launch uses
    -- rather than refusing a shape this family has always served. The three
    boundaries below are the last ratio that fits the packed grid, the first that
    does not, and the last ratio the eight-row grid itself admits.
    """

    # head_dim 256 slices the output eight ways: 8,191 row groups (16,382 query
    # rows) is the last shape whose product fits 65,535.
    fits = staged.staged_plan(
        tokens=1, keys=64, num_heads=16382, num_kv_heads=1, head_dim=256
    )
    assert fits.pv_singleton
    assert (fits.pv_rows_per_block, fits.row_groups) == (2, 8191)
    assert fits.pv_grid == (1, 1, 8191 * 8)
    assert fits.pv_grid[2] == 65528 <= staged.MAX_GRID_DIM
    # One more row group is 65,536: past the packed grid, so the launch is the
    # 256-thread decomposition on its own grid, whose Z is the row-group count.
    past = staged.staged_plan(
        tokens=1, keys=64, num_heads=16384, num_kv_heads=1, head_dim=256
    )
    assert not past.pv_singleton
    assert (past.pv_rows_per_block, past.pv_dim_slices) == (staged.MAX_ROWS_PER_BLOCK, 1)
    assert past.pv_grid == (1, 1, 2048)
    assert past.pv_lds_bytes == staged.staged_pv_lds_bytes()
    # An odd ratio past the bound takes the same fallback: 16,383 rows is 8,192
    # row groups once they are paired, and 8,192 * 8 is one past the grid.
    odd_past = staged.staged_plan(
        tokens=1, keys=64, num_heads=16383, num_kv_heads=1, head_dim=256
    )
    assert not odd_past.pv_singleton
    assert odd_past.pv_grid == (1, 1, 2048)
    # head_dim 512 slices the output sixteen ways: 4,095 row groups (8,190 rows).
    fits_512 = staged.staged_plan(
        tokens=1, keys=64, num_heads=8190, num_kv_heads=1, head_dim=512
    )
    assert fits_512.pv_singleton
    assert fits_512.pv_grid == (1, 1, 4095 * 16)
    assert fits_512.pv_grid[2] == 65520 <= staged.MAX_GRID_DIM
    past_512 = staged.staged_plan(
        tokens=1, keys=64, num_heads=8192, num_kv_heads=1, head_dim=512
    )
    assert not past_512.pv_singleton
    assert past_512.pv_grid == (1, 1, 1024)
    # The last ratio the eight-row grid admits is 8 * 65,535 = 524,280, which the
    # packed grid cannot hold at either width -- and it is still a plan, on the
    # grid the 256-thread decomposition has always launched.
    old_limit = staged.staged_plan(
        tokens=1, keys=64, num_heads=524280, num_kv_heads=1, head_dim=256
    )
    assert not old_limit.pv_singleton
    assert old_limit.pv_grid == (1, 1, staged.MAX_GRID_DIM)
    assert old_limit.pv_grid[2] == 65535
    # One row group past that limit is refused with the message it always had.
    with pytest.raises(NotImplementedError) as failure:
        staged.staged_plan(
            tokens=1, keys=64, num_heads=524288, num_kv_heads=1, head_dim=256
        )
    assert "524288" in str(failure.value)
    # Whatever the decomposition, no plan reports a Z past the grid.
    for num_heads, num_kv_heads, head_dim in (
        (16382, 1, 256),
        (16383, 1, 256),
        (16384, 1, 256),
        (8190, 1, 512),
        (8192, 1, 512),
        (524280, 1, 256),
        (524280, 1, 512),
        (16, 2, 512),
        (16, 8, 256),
    ):
        plan = staged.staged_plan(
            tokens=1, keys=64, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        assert plan.pv_grid[2] <= staged.MAX_GRID_DIM


def test_no_shape_the_eight_row_grid_serves_is_refused():
    """The packed grid is the tighter one, so an overflow falls back, not out.

    A single-token shape whose row-group-by-slice product does not fit one grid
    dimension keeps the 256-thread decomposition, so the acceptance rule is the
    eight-row grid's: a shape is a plan exactly when its eight-row row groups fit
    one grid dimension. This sweeps ratios across both boundaries at both head
    widths -- the last packed fit, the first packed overflow, and the last ratio
    the eight-row grid itself admits -- so "no previously runnable shape is
    refused" is checked rather than argued.
    """

    accepted = refused = 0
    for head_dim in (256, 512):
        for ratio in (
            1, 2, 3, 4, 5, 8, 12, 16, 17, 4095, 4096, 8190, 8191, 8192, 8193,
            16382, 16383, 16384, 16385, 32768, 65534 * 8, 65535 * 8, 65535 * 8 + 1,
        ):
            legacy_groups = -(-ratio // staged.MAX_ROWS_PER_BLOCK)
            try:
                plan = staged.staged_plan(
                    tokens=1, keys=64, num_heads=ratio, num_kv_heads=1,
                    head_dim=head_dim,
                )
            except NotImplementedError:
                assert legacy_groups > staged.MAX_GRID_DIM, (head_dim, ratio)
                refused += 1
                continue
            assert legacy_groups <= staged.MAX_GRID_DIM, (head_dim, ratio)
            assert plan.pv_grid[2] <= staged.MAX_GRID_DIM, (head_dim, ratio)
            accepted += 1
    assert accepted and refused


def test_plan_describe_names_the_decomposition():
    plan = staged.staged_plan(tokens=3, keys=2048, num_heads=16, num_kv_heads=2, head_dim=512)
    described = plan.describe()
    assert "score_grid=48x2" in described
    assert "pv_grid=3x2x1" in described
    assert "pv=multi(256t,8r,1s)" in described
    assert f"lds={plan.lds_bytes}B" in described
    assert f"workspace={plan.workspace_bytes}B" in described
    # A single token names the other decomposition, with its own grid and tile.
    singleton = staged.staged_plan(
        tokens=1, keys=2048, num_heads=16, num_kv_heads=2, head_dim=512
    )
    singleton_described = singleton.describe()
    assert "pv_grid=1x2x64" in singleton_described
    assert "pv=singleton(32t,2r,16s)" in singleton_described
    assert f"lds={singleton.lds_bytes}B" in singleton_described


# --- workspace ownership ----------------------------------------------------


def test_workspace_buffer_is_per_stream_and_freed_after_the_stream_quiets():
    runtime = Runtime()
    scratch = strict.Gemma4AttentionScratch()
    plan = staged.staged_plan(tokens=1, keys=4096, num_heads=16, num_kv_heads=2, head_dim=512)
    first = staged.staged_workspace_buffer(scratch, plan, stream=0, runtime=runtime)
    assert first.nbytes >= plan.workspace_bytes
    # Reuse, not reallocation, on the same stream.
    assert staged.staged_workspace_buffer(scratch, plan, stream=0, runtime=runtime) is first
    # A second stream gets its own buffer, because the two launches can overlap.
    other = staged.staged_workspace_buffer(scratch, plan, stream=7, runtime=runtime)
    assert other.ptr != first.ptr
    assert len(runtime.live) == 2
    scratch.close()
    assert runtime.live == {}
    first_free = next(i for i, event in enumerate(runtime.events) if event[0] == "free")
    assert ("sync", 0) in runtime.events[:first_free]
    assert ("sync", 7) in runtime.events[:first_free]
    with pytest.raises(RuntimeError, match="closed"):
        staged.staged_workspace_buffer(scratch, plan, stream=0, runtime=runtime)


def test_workspace_buffer_refuses_a_second_device_or_runtime():
    runtime = Runtime()
    scratch = strict.Gemma4AttentionScratch()
    plan = staged.staged_plan(tokens=1, keys=1024, num_heads=2, num_kv_heads=1, head_dim=256)
    staged.staged_workspace_buffer(scratch, plan, stream=0, runtime=runtime)
    runtime.device = 1
    with pytest.raises(ValueError, match="device"):
        staged.staged_workspace_buffer(scratch, plan, stream=0, runtime=runtime)
    runtime.device = 0
    with pytest.raises(ValueError, match="runtime"):
        staged.staged_workspace_buffer(scratch, plan, stream=0, runtime=Runtime())


def test_a_failed_launch_releases_the_temporary_workspace():
    """A launch failure must not leak the scratch the launcher allocated.

    The kernel reports the failure through its return code, so the release path
    is the launcher's own -- no device is involved and no kernel runs.
    """

    runtime = FailingRuntime()
    with pytest.raises(RuntimeError, match="fake launch failure"):
        staged.gemma4_attention_staged_f32(
            0, 0, 0, 0, 0,
            tokens=1, keys=8, num_heads=2, num_kv_heads=1, head_dim=256, scale=1.0,
            library=FailingLibrary(), runtime=runtime,
        )
    assert runtime.live == {}
    assert any(event[0] == "sync" for event in runtime.events)


# --- the independent oracle -------------------------------------------------


@pytest.mark.parametrize("name", ["window", "row_offset"])
@pytest.mark.parametrize("value", [-(2**63)-1, 2**63])
def test_window_coordinates_must_fit_the_raw_abi(name, value):
    runtime = FailingRuntime()
    with pytest.raises(ValueError, match=f"{name}.*int64"):
        staged.gemma4_attention_staged_f32(
            0, 0, 0, 0, 0, tokens=1, keys=8, num_heads=2,
            num_kv_heads=1, head_dim=256, scale=1.0,
            library=FailingLibrary(), runtime=runtime, **{name: value},
        )
    assert runtime.events == []


def bf16_round(values: np.ndarray) -> np.ndarray:
    """FP32 -> BF16 bits, round-to-nearest-even, as the kernel's bit trick does."""

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    return ((bits + np.uint32(0x7FFF) + lsb) >> 16).astype(np.uint16)


def bf16_decode(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def staged_cpu_oracle(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    keep_mask: np.ndarray,
    *,
    num_kv_heads: int,
    scale: float,
    window: int = 0,
    row_offset: int = 0,
) -> np.ndarray:
    """Masked ungated attention from the definition, in float64.

    Dimension order, no tree, no lane partition, no tiling: this is what the
    kernel's result is compared against when the strict kernel cannot be
    launched (a context past its shared-memory ceiling). ``window > 0`` applies
    the same trim the kernels apply, which is sound only because a caller that
    passes ``window`` promises the mask is zero outside it.
    """

    tokens, num_heads, _ = query.shape
    keys = key.shape[0]
    kv_groups = num_heads // num_kv_heads
    positions = np.arange(keys)
    out = np.zeros((tokens, num_heads, query.shape[2]), dtype=np.float64)
    for token in range(tokens):
        for head in range(num_heads):
            kv_head = head // kv_groups
            logits = (
                key[:, kv_head, :].astype(np.float64) @ query[token, head, :].astype(np.float64)
            ) * scale
            kept = keep_mask[token].astype(bool)
            if window > 0:
                own = row_offset + token
                kept = kept & (positions > own - window) & (positions <= own)
            logits = np.where(kept, logits, -np.inf)
            row_max = logits.max()
            weights = np.exp(logits - row_max)
            denominator = weights.sum()
            out[token, head] = (
                weights @ value[:, kv_head, :].astype(np.float64)
            ) / denominator
    return out


def test_oracle_matches_a_hand_computed_row():
    """Two keys, one head: the weights are exact and the result is analytic."""

    query = np.array([[[2.0, 0.0]]], dtype=np.float32)
    key = np.array([[[1.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32)
    value = np.array([[[4.0, 8.0]], [[1.0, 2.0]]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=np.uint8)
    out = staged_cpu_oracle(query, key, value, mask, num_kv_heads=1, scale=1.0)
    # logits 2 and 0; exp(2), exp(0) -> weights 0.8808 and 0.1192.
    w0 = np.exp(2.0) / (np.exp(2.0) + 1.0)
    w1 = 1.0 / (np.exp(2.0) + 1.0)
    assert out[0, 0, 0] == pytest.approx(w0 * 4.0 + w1 * 1.0, rel=1e-12)
    assert out[0, 0, 1] == pytest.approx(w0 * 8.0 + w1 * 2.0, rel=1e-12)


def test_oracle_honours_the_mask_and_the_window_trim():
    rng = np.random.default_rng(11)
    query = rng.standard_normal((1, 1, 8), dtype=np.float32)
    key = rng.standard_normal((16, 1, 8), dtype=np.float32)
    value = rng.standard_normal((16, 1, 8), dtype=np.float32)

    uniform = np.ones((1, 16), dtype=np.uint8)
    # scale 0 makes every logit 0, so every weight is 1 and the row is the
    # plain mean of V -- the one case with a closed form regardless of the data.
    all_kept = staged_cpu_oracle(query, key, value, uniform, num_kv_heads=1, scale=0.0)
    assert np.allclose(all_kept[0, 0], value[:, 0, :].astype(np.float64).mean(0), atol=1e-12)
    # With real logits the weights are not uniform, so the mean is not the answer:
    # the oracle must not be collapsing to it.
    peaked = staged_cpu_oracle(query, key, value, uniform, num_kv_heads=1, scale=8.0)
    assert not np.allclose(peaked[0, 0], value[:, 0, :].astype(np.float64).mean(0), atol=1e-6)

    single = np.zeros((1, 16), dtype=np.uint8)
    single[0, 5] = 1
    only_five = staged_cpu_oracle(query, key, value, single, num_kv_heads=1, scale=1.0)
    assert np.allclose(only_five[0, 0], value[5, 0, :].astype(np.float64), atol=1e-12)

    # A sliding band around position 9 with window 4 keeps columns 6..9, and the
    # oracle's trim must agree with the mask it is given.
    band = np.zeros((1, 16), dtype=np.uint8)
    band[0, 6:10] = 1
    trimmed = staged_cpu_oracle(
        query, key, value, band, num_kv_heads=1, scale=1.0, window=4, row_offset=9
    )
    masked_only = staged_cpu_oracle(query, key, value, band, num_kv_heads=1, scale=1.0)
    assert np.allclose(trimmed, masked_only, atol=1e-12)
    # ...and a window that would trim a column the mask keeps must not be used
    # that way: the oracle follows the promise, so it would drop column 6.
    too_narrow = staged_cpu_oracle(
        query, key, value, band, num_kv_heads=1, scale=1.0, window=2, row_offset=9
    )
    assert not np.allclose(too_narrow, masked_only, atol=1e-12)


def test_bf16_rounding_rule_is_round_to_nearest_even():
    """Pinned on hand-computed values, so the oracle's decode is not self-referential."""

    values = np.array(
        [
            0.0,
            1.0,
            -1.0,
            # Exactly between 1.0 and the next BF16 value (1 + 2**-7): ties to
            # the even mantissa, which is 1.0.
            1.0 + 2.0**-8,
            # Halfway between 1 + 2**-7 (odd mantissa) and 1 + 2**-6 (even):
            # ties to the even one, which is the upper value.
            1.0 + 3.0 * 2.0**-8,
            # Between 1 + 2**-6 and 1 + 3 * 2**-7: ties to the even mantissa.
            1.0 + 5.0 * 2.0**-8,
        ],
        dtype=np.float32,
    )
    bits = bf16_round(values)
    assert bits[0] == 0x0000
    assert bits[1] == 0x3F80
    assert bits[2] == 0xBF80
    assert bits[3] == 0x3F80
    assert bits[4] == 0x3F82
    assert bits[5] == 0x3F82
    decoded = bf16_decode(bits)
    assert decoded[1] == np.float32(1.0)
    assert decoded[3] == np.float32(1.0)
