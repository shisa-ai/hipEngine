"""Explicit speculative candidate-depth setting and its removed legacy name."""

from __future__ import annotations

import warnings

import pytest

from hipengine.speculative.candidate_budget import (
    REMOVED_GGUF_MTP_CANDIDATE_BUDGET_ENV,
    SPECULATIVE_CANDIDATE_BUDGET_ENV,
    resolve_explicit_candidate_budget_env,
)

CANONICAL = SPECULATIVE_CANDIDATE_BUDGET_ENV
REMOVED = REMOVED_GGUF_MTP_CANDIDATE_BUDGET_ENV


@pytest.fixture(autouse=True)
def _unset_both_names(monkeypatch) -> None:
    monkeypatch.delenv(CANONICAL, raising=False)
    monkeypatch.delenv(REMOVED, raising=False)


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


@pytest.mark.parametrize("canonical", [None, "3", "4"])
def test_removed_name_raises_naming_its_replacement(monkeypatch, canonical) -> None:
    monkeypatch.setenv(REMOVED, "3")
    if canonical is not None:
        monkeypatch.setenv(CANONICAL, canonical)

    with pytest.raises(ValueError, match="has been removed") as raised:
        resolve_explicit_candidate_budget_env()

    message = str(raised.value)
    assert REMOVED in message
    assert f"{CANONICAL}=3" in message


@pytest.mark.parametrize("raw", ["not-an-int", "1.5"])
def test_malformed_values_raise(monkeypatch, raw) -> None:
    monkeypatch.setenv(CANONICAL, raw)

    with pytest.raises(ValueError, match="positive integer"):
        resolve_explicit_candidate_budget_env()


@pytest.mark.parametrize("raw", ["0", "-2"])
def test_non_positive_values_raise(monkeypatch, raw) -> None:
    monkeypatch.setenv(CANONICAL, raw)

    with pytest.raises(ValueError, match=">= 1"):
        resolve_explicit_candidate_budget_env()


def test_blank_removed_value_is_unset(monkeypatch) -> None:
    monkeypatch.setenv(REMOVED, "   ")

    value, caught = _resolve_with_captured_warnings()

    assert value is None
    assert caught == []
