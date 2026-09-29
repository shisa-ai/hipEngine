"""Gemma 4 MMQ dual-route row-alignment guard.

Regression: ``_mmq_dual_route`` selected the 32-row MMQ leaf whenever the
quant, shape and lane-count guards passed, but the leaf sizes its tile plan as
``compact_rows + 31 * num_experts`` and ``_check_mmq32_common`` raises
``ValueError: mmq_total_rows must be a multiple of 32`` on a total that is not
aligned. The route was therefore taken for row counts the leaf cannot accept,
and the projection raised mid-prefill -- reachable through the public surface
as ``LLM.generate()`` failing with ``GenerationExecutionFailed``.

Observed before the guard, on the shipped 128-expert stack with top_k = 8:

* a single prefill block of N tokens builds ``compact_rows = N * 8``, so the
  total is aligned only when ``N % 4 == 0``;
* N in [256, 511] with ``N % 4 != 0`` raised;
* N > 512 raised when ``N % 512`` landed in that same region;
* N < 256 did not raise, because a smaller guard keeps it off the MMQ path.

For 128 experts the ``31 * 128`` term is 3968, itself a multiple of 32, so the
condition collapses to ``compact_rows % 32``. It must not be *written* that
way: with any expert count that is not a multiple of 32 the constant is not
aligned, and the two forms disagree. ``test_alignment_follows_the_plan_formula``
pins that.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    _DS4_BLOCK_VALUES,
    _MMQ_DUAL_QUANT_KEY,
    _mmq_dual_route,
)


class _Spec:
    def __init__(self, quant_key: str) -> None:
        self.quant_key = quant_key


class _Weight:
    """Minimal stand-in: the guard reads only ``weight.spec.quant_key``."""

    def __init__(self, quant_key: str = _MMQ_DUAL_QUANT_KEY) -> None:
        self.spec = _Spec(quant_key)


def _route(compact_rows: int, num_experts: int, *, quant_key: str = _MMQ_DUAL_QUANT_KEY) -> bool:
    in_features = 7 * _DS4_BLOCK_VALUES  # aligned: 7 blocks of 128
    out_features = 704  # 704 % 32 == 0
    return _mmq_dual_route(
        _Weight(quant_key), compact_rows, in_features, out_features, num_experts
    )


def test_alignment_follows_the_plan_formula() -> None:
    """The guard must test ``compact_rows + 31 * num_experts``, not rows alone.

    num_experts=16 makes ``31 * 16 = 496``, and 496 % 32 == 16, so the constant
    itself is unaligned. The total is then aligned only for compact_rows that
    are 16 (mod 32) -- the opposite residue from the 128-expert case, where the
    constant is aligned and rows must be 0 (mod 32). A guard written as
    ``compact_rows % 32`` would accept 1024 and reject 1040 here, backwards.
    """
    assert (31 * 16) % 32 == 16, "precondition: constant is unaligned for 16 experts"

    assert _route(1040, 16) is True, "1040 % 32 == 16, so rows + 496 is aligned"
    assert _route(1008, 16) is True, "1008 % 32 == 16 too: 1008 + 496 = 1504 = 47*32"
    assert _route(1024, 16) is False, "1024 % 32 == 0, so rows + 496 is 16 (mod 32)"

    # The residue flips with the expert count, which a ``compact_rows % 32``
    # guard could not express: 16 experts need rows == 16 (mod 32), while the
    # shipped 128 experts need rows == 0 (mod 32).
    assert _route(1024, 128) is True, "same rows, aligned for 128 experts"
    assert _route(1040, 128) is False, "same rows, unaligned for 128 experts"


def test_shipped_128_expert_geometry_refuses_unaligned_token_counts() -> None:
    """The observed failure: tokens * 8 rows must land on a 32-row boundary.

    31 * 128 = 3968 is a multiple of 32, so for the shipped stack the check
    reduces to compact_rows % 32 -- which, at eight routed lanes per token, is
    ``tokens % 4``.
    """
    num_experts = 128

    for tokens in (257, 301, 501, 845, 961, 1001):
        rows = tokens * 8
        assert rows % 32 != 0, f"precondition: {tokens} tokens is unaligned"
        assert _route(rows, num_experts) is False, (
            f"{tokens} tokens -> {rows} rows must fall back to the grouped owner"
        )

    for tokens in (256, 504, 512):
        rows = tokens * 8
        assert rows % 32 == 0, f"precondition: {tokens} tokens is aligned"
        assert _route(rows, num_experts) is True, (
            f"{tokens} tokens -> {rows} rows may keep the MMQ leaf"
        )


def test_row_count_matches_the_measured_probe() -> None:
    """Pin the numbers the instrumented probe printed, so the fix and the
    measurement refer to the same quantity."""
    # width 501 -> compact_rows 4008, mmq_total_rows 7976 (7976 % 32 == 8)
    assert 4008 + 31 * 128 == 7976
    assert 7976 % 32 == 8
    # width 504 -> 4032 / 8000 ; width 512 -> 4096 / 8064
    assert 4032 + 31 * 128 == 8000 and 8000 % 32 == 0
    assert 4096 + 31 * 128 == 8064 and 8064 % 32 == 0


def test_other_guards_still_refuse_first() -> None:
    """The new guard must not shadow the pre-existing ones."""
    assert _route(4096, 128, quant_key="gguf_q8_0") is False, "wrong quant key"
    assert _mmq_dual_route(0, 4096, 896, 704, 128) is False, "int weight sentinel"
    assert _route(32, 128) is False, "below the minimum lanes-per-expert floor"


def test_lane_floor_still_applies_to_aligned_rows() -> None:
    """Aligned rows below the floor stay off the path, so the two guards are
    independent rather than one subsuming the other."""
    assert _route(480, 128) is False, "480 is aligned but below 4 * 128 lanes"
    assert _route(512, 128) is True, "512 clears 4 * 128 exactly and is aligned"
    assert _route(2048, 128) is True, "2048 rows clears 4 * 128 and is aligned"