"""Router producer selection and strict-fallback contract."""

from __future__ import annotations

import pytest

from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.runtime import qwen4_exp_runner as runner

_ENV = "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"


def test_route_env_key_matches_between_profile_and_runner() -> None:
    assert profiles._ROUTER_VARIANT_ENV == runner._ROUTER_PRODUCER_VARIANT_ENV == _ENV


def test_named_variants_agree_between_profile_and_runner() -> None:
    assert profiles._ROUTER_TILE4_VARIANT == runner._ROUTER_PRODUCER_TILE4_VARIANT
    assert profiles._ROUTER_WARP_REDUCE_VARIANT == runner._ROUTER_PRODUCER_WARP_REDUCE_VARIANT
    assert profiles._ROUTER_DPP_VARIANT == runner._ROUTER_PRODUCER_DPP_VARIANT


def test_profile_selects_shipped_tile_by_default(monkeypatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert profiles._selected_router_producer_variant() == profiles._ROUTER_TILE4_VARIANT


def test_profile_follows_the_candidate_switch(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, profiles._ROUTER_WARP_REDUCE_VARIANT)
    assert profiles._selected_router_producer_variant() == profiles._ROUTER_WARP_REDUCE_VARIANT


def test_profile_follows_the_dpp_candidate_switch(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, profiles._ROUTER_DPP_VARIANT)
    assert profiles._selected_router_producer_variant() == profiles._ROUTER_DPP_VARIANT


def test_runner_uses_the_linear_path_below_two_rows(monkeypatch) -> None:
    for rows in (0, 1):
        for value in ("", profiles._ROUTER_WARP_REDUCE_VARIANT, profiles._ROUTER_DPP_VARIANT):
            monkeypatch.setenv(_ENV, value)
            assert runner._qwen4_exp_router_producer_variant(rows) is None


def test_runner_defaults_to_the_shipped_tile(monkeypatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert runner._qwen4_exp_router_producer_variant(2) == runner._ROUTER_PRODUCER_TILE4_VARIANT


def test_runner_selects_the_candidate_switch(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, runner._ROUTER_PRODUCER_WARP_REDUCE_VARIANT)
    assert runner._qwen4_exp_router_producer_variant(1024) == runner._ROUTER_PRODUCER_WARP_REDUCE_VARIANT


def test_runner_selects_the_dpp_candidate_switch(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, runner._ROUTER_PRODUCER_DPP_VARIANT)
    assert runner._qwen4_exp_router_producer_variant(1024) == runner._ROUTER_PRODUCER_DPP_VARIANT


def test_runner_rejects_an_unrecognized_variant(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, "f32_hidden_not_a_variant")
    with pytest.raises(ValueError, match=_ENV):
        runner._qwen4_exp_router_producer_variant(1024)


def _router_selection(selections) -> object:
    matches = [s for s in selections if s.layer == "router_logits"]
    assert len(matches) == 1
    return matches[0]


def test_strict_stays_pinned_to_the_exact_parent() -> None:
    strict = _router_selection(profiles._strict_selections())
    assert strict.selected_variant == profiles._ROUTER_TILE4_VARIANT
    assert strict.selected_variant != profiles._ROUTER_WARP_REDUCE_VARIANT
    assert strict.selected_variant != profiles._ROUTER_DPP_VARIANT


def test_production_follows_the_switch_and_keeps_the_exact_fallback() -> None:
    production = _router_selection(profiles._production_selections(recovery=True))
    assert production.selected_variant == profiles._ROUTER_LOGITS_SELECTED_VARIANT
    assert production.strict_fallback_variant == profiles._ROUTER_TILE4_VARIANT
