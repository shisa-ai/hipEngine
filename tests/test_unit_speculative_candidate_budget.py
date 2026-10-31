"""Explicit speculative candidate-depth setting and its compatibility alias."""

from __future__ import annotations

import warnings

import pytest

from hipengine.speculative.candidate_budget import (
    LEGACY_GGUF_MTP_CANDIDATE_BUDGET_ENV,
    SPECULATIVE_CANDIDATE_BUDGET_ENV,
    resolve_explicit_candidate_budget_env,
)

CANONICAL = SPECULATIVE_CANDIDATE_BUDGET_ENV
LEGACY = LEGACY_GGUF_MTP_CANDIDATE_BUDGET_ENV


@pytest.fixture(autouse=True)
def _unset_both_names(monkeypatch) -> None:
    monkeypatch.delenv(CANONICAL, raising=False)
    monkeypatch.delenv(LEGACY, raising=False)


def _resolve_with_captured_warnings() -> tuple[int | None, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = resolve_explicit_candidate_budget_env()
    return value, list(caught)


def test_neither_name_set_resolves_to_none_without_warning() -> None:
    value, caught = _resolve_with_captured_warnings()

    assert value is None
    assert caught == []


def test_canonical_name_resolves_without_warning(monkeypatch) -> None:
    monkeypatch.setenv(CANONICAL, "2")

    value, caught = _resolve_with_captured_warnings()

    assert value == 2
    assert caught == []


def test_legacy_only_warns_and_preserves_the_requested_depth(monkeypatch) -> None:
    monkeypatch.setenv(LEGACY, "3")

    with pytest.warns(FutureWarning) as recorded:
        value = resolve_explicit_candidate_budget_env()

    assert value == 3
    assert len(recorded) == 1
    message = str(recorded[0].message)
    assert LEGACY in message
    assert CANONICAL in message


def test_agreeing_names_resolve_without_warning(monkeypatch) -> None:
    monkeypatch.setenv(CANONICAL, "4")
    monkeypatch.setenv(LEGACY, "4")

    value, caught = _resolve_with_captured_warnings()

    assert value == 4
    assert caught == []


def test_conflicting_names_raise_before_warning(monkeypatch) -> None:
    monkeypatch.setenv(CANONICAL, "4")
    monkeypatch.setenv(LEGACY, "3")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="conflicts"):
            resolve_explicit_candidate_budget_env()

    assert caught == []


@pytest.mark.parametrize("name", [CANONICAL, LEGACY])
@pytest.mark.parametrize("raw", ["not-an-int", "1.5"])
def test_malformed_values_raise_before_warning(monkeypatch, name, raw) -> None:
    monkeypatch.setenv(name, raw)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="positive integer"):
            resolve_explicit_candidate_budget_env()

    assert caught == []


@pytest.mark.parametrize("name", [CANONICAL, LEGACY])
@pytest.mark.parametrize("raw", ["0", "-2"])
def test_non_positive_values_raise_before_warning(monkeypatch, name, raw) -> None:
    monkeypatch.setenv(name, raw)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=">= 1"):
            resolve_explicit_candidate_budget_env()

    assert caught == []


def test_blank_legacy_value_is_unset_and_does_not_warn(monkeypatch) -> None:
    monkeypatch.setenv(LEGACY, "   ")

    value, caught = _resolve_with_captured_warnings()

    assert value is None
    assert caught == []
