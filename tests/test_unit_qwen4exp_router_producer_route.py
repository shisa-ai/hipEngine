"""Router logits producer selection contract.

The warp-per-token reduction candidate is screened through an env switch that
two modules have to agree on: the profile declares the variant and hashes it
into the manifest, and the runner resolves the variant it actually launches. A
disagreement is silent, and it produces a measurement that looks like a null
result rather than a broken arm. These tests pin the agreement.
"""

from __future__ import annotations

import pytest

from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.runtime import qwen4_exp_runner as runner

_ENV = "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"


def test_route_env_key_matches_between_profile_and_runner() -> None:
    """Both modules must read the same variable name."""

    assert profiles._ROUTER_VARIANT_ENV == runner._ROUTER_PRODUCER_VARIANT_ENV == _ENV


def test_shipped_default_agrees_between_profile_and_runner() -> None:
    """The promoted default lives in two modules and must not drift."""

    assert profiles._ROUTER_TILE4_VARIANT == runner._ROUTER_PRODUCER_TILE4_VARIANT
    assert (
        profiles._ROUTER_WARP_REDUCE_VARIANT
        == runner._ROUTER_PRODUCER_WARP_REDUCE_VARIANT
    )


def test_profile_selects_shipped_tile_by_default(monkeypatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert profiles._selected_router_producer_variant() == profiles._ROUTER_TILE4_VARIANT


def test_profile_follows_the_candidate_switch(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, profiles._ROUTER_WARP_REDUCE_VARIANT)
    assert (
        profiles._selected_router_producer_variant()
        == profiles._ROUTER_WARP_REDUCE_VARIANT
    )


def test_runner_uses_the_linear_path_below_two_rows(monkeypatch) -> None:
    """Rows 0 and 1 have no multirow producer, whatever the switch says."""

    for rows in (0, 1):
        for value in ("", profiles._ROUTER_WARP_REDUCE_VARIANT):
            monkeypatch.setenv(_ENV, value)
            assert runner._qwen4_exp_router_producer_variant(rows) is None


def test_runner_defaults_to_the_shipped_tile(monkeypatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert (
        runner._qwen4_exp_router_producer_variant(2)
        == runner._ROUTER_PRODUCER_TILE4_VARIANT
    )


def test_runner_selects_the_candidate_switch(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, runner._ROUTER_PRODUCER_WARP_REDUCE_VARIANT)
    assert (
        runner._qwen4_exp_router_producer_variant(1024)
        == runner._ROUTER_PRODUCER_WARP_REDUCE_VARIANT
    )


def test_runner_rejects_an_unrecognized_variant(monkeypatch) -> None:
    """An unrecognized value raises instead of silently running the default.

    A silently inert arm is the failure this route already produced once, so
    the selector fails loudly rather than falling back.
    """

    monkeypatch.setenv(_ENV, "f32_hidden_not_a_variant")
    with pytest.raises(ValueError, match=_ENV):
        runner._qwen4_exp_router_producer_variant(1024)


def _router_selection(selections) -> object:
    matches = [s for s in selections if s.layer == "router_logits"]
    assert len(matches) == 1, f"expected one router_logits selection, got {len(matches)}"
    return matches[0]


def test_strict_stays_pinned_to_the_exact_parent() -> None:
    """The strict profile is the debugging oracle and must not follow the switch.

    The candidate is not bit-identical to the four-token tile, so letting the
    screening switch reach the strict plan would quietly replace the exact
    reference with the candidate.
    """

    strict = _router_selection(profiles._strict_selections())
    assert strict.selected_variant == profiles._ROUTER_TILE4_VARIANT
    assert strict.selected_variant != profiles._ROUTER_WARP_REDUCE_VARIANT


def test_production_follows_the_switch_and_keeps_the_exact_fallback() -> None:
    production = _router_selection(
        profiles._production_selections(recovery=True)
    )
    assert production.selected_variant == profiles._ROUTER_LOGITS_SELECTED_VARIANT
    assert production.strict_fallback_variant == profiles._ROUTER_TILE4_VARIANT
