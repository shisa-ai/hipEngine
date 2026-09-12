"""Static route mix for a GGUF artifact's rank-2 matmul weights.

The Phase 1 MTP attribution uses this to quantify the rows 2-7 dead zone: UD
assigns IQ-family quants to 38.5% (K_M) and 60.6% (K_S) of its rank-2 MACs,
while both plain controls are ~0%.
"""
from __future__ import annotations

from scripts.gguf_route_mix import (
    IQ_PREFILL_SOURCE_TYPES,
    ROWTILE_SOURCE_TYPES,
    classify,
)


def test_rowtile_types_are_the_amortized_small_row_owner() -> None:
    for name in ("Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        assert classify(name) == "rowtile_rows2_8", name


def test_iq_family_is_the_dead_zone() -> None:
    for name in ("IQ4_XS", "IQ4_NL", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "Q3_K"):
        assert classify(name) == "iq_dead_zone_rows2_7", name


def test_non_linear_and_float_types_are_other() -> None:
    for name in ("F32", "BF16", "Q4_0", "Q8_K"):
        assert classify(name) == "other", name


def test_the_two_sets_are_disjoint() -> None:
    assert not (ROWTILE_SOURCE_TYPES & IQ_PREFILL_SOURCE_TYPES)
