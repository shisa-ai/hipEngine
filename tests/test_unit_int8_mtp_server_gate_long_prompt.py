"""Cover the long-prompt rows and the long-prompt multi-choice check."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.int8_mtp_server_gate import (
    LONG_PROMPT_SOURCES,
    build_parser,
    check_long_multichoice,
    long_prompt_rows,
    long_prompt_text,
)


def _words(text):
    """Deterministic stand-in for a tokenizer: one token per whitespace word."""

    return text.split()


def test_long_prompt_text_is_the_longest_prefix_within_the_budget():
    text = " ".join(f"word{index}" for index in range(500))

    body, count = long_prompt_text(text, 120, _words)

    assert count <= 120
    assert len(_words(body)) == count
    assert body == text[:len(body)]
    # One more character either keeps the count or crosses the budget; either way
    # no longer prefix fits, which is the property the row's length rests on.
    assert len(_words(text[: len(body) + 1])) > count


def test_long_prompt_text_returns_everything_when_the_source_is_short():
    text = "short source text"

    body, count = long_prompt_text(text, 10_000, _words)

    assert body == text
    assert count == len(_words(text))


def test_long_prompt_rows_record_a_reproducible_prefix_of_committed_prose():
    rows = long_prompt_rows([64, 96], _words, sources=("docs/PLAN.md",))

    text = (Path(__file__).resolve().parents[1] / "docs/PLAN.md").read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert [row["id"] for row in rows] == ["long_64", "long_96"]
    for row in rows:
        content = row["messages"][0]["content"]
        assert text.startswith(content)
        assert row["prompt_tokens"] == len(_words(content))
        assert row["prompt_tokens"] <= row["target_prompt_tokens"]
        assert row["source_sha256"] == digest
    # The longer row is a strict extension, so the two rows are not the same
    # prompt under two names.
    assert len(rows[1]["messages"][0]["content"]) > len(rows[0]["messages"][0]["content"])


def test_long_prompt_sources_are_files_that_exist():
    root = Path(__file__).resolve().parents[1]

    assert LONG_PROMPT_SOURCES
    for name in LONG_PROMPT_SOURCES:
        assert (root / name).is_file(), name


class _Response:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def raise_for_status(self):
        assert self.status_code == 200, self.status_code

    def json(self):
        return self._body


class _Client:
    """Minimal stand-in that answers one multi-choice request."""

    def __init__(self, choices):
        self.choices = choices
        self.payloads = []

    def post(self, url, json=None):
        self.payloads.append((url, json))
        return _Response({"choices": self.choices})


def _choice(ids, cycles, *, decline=None, failures=None):
    spec = {
        "plan_reason": "no_provider" if decline or failures else "speculative_qualified",
        "provider_decline_reason": decline,
        "failure_reasons": failures or [],
    }
    return {
        "finish_reason": "length",
        "hipengine": {
            "generated_token_ids": ids,
            "timing": {"mtp_cycles_count": cycles},
            "diagnostics": {"specdec2_mtp2": spec},
        },
    }


ARGS = SimpleNamespace(model="int8-mtp", max_tokens=24)
ROW = {"id": "long_4096", "messages": [{"role": "user", "content": "prompt"}]}


def test_long_multichoice_accepts_a_group_that_matches_the_autoregressive_ids():
    client = _Client([_choice([1, 2], 3), _choice([1, 2], 3)])

    result = check_long_multichoice(client, ARGS, ROW, {"max_packed_rows": 4}, [1, 2])

    assert result["speculates"] is True
    assert result["cycles"] == [3, 3]
    assert client.payloads[0][1]["n"] == 2


def test_long_multichoice_rejects_a_choice_that_drifted_from_the_autoregressive_ids():
    client = _Client([_choice([1, 2], 3), _choice([1, 9], 3)])

    with pytest.raises(AssertionError) as excinfo:
        check_long_multichoice(client, ARGS, ROW, {"max_packed_rows": 4}, [1, 2])

    assert excinfo.value.args[0]["long_multichoice"]["choice_ids"] == [[1, 2], [1, 9]]


def test_long_multichoice_requires_speculation_when_the_group_ran_packed():
    client = _Client([_choice([1, 2], 0), _choice([1, 2], 0)])

    with pytest.raises(AssertionError) as excinfo:
        check_long_multichoice(client, ARGS, ROW, {"max_packed_rows": 4}, [1, 2])

    assert excinfo.value.args[0] == {
        "long_multichoice_silent_decline": {"id": "long_4096", "cycles": [0, 0]}
    }


def test_long_multichoice_accepts_a_named_fallback_with_identical_ids():
    """Past the packed path's admitted context the group must name its cause."""

    failure = {
        "category": "precommit_failure_ar_fallback",
        "detail": (
            "NotImplementedError:long-context packed AR decode requires "
            "a row-sized split-K workspace"
        ),
    }
    client = _Client([
        _choice([1, 2], 0, failures=[failure]),
        _choice([1, 2], 0, failures=[failure]),
    ])

    result = check_long_multichoice(client, ARGS, ROW, {"max_packed_rows": 4}, [1, 2])

    assert result["speculates"] is False
    assert result["decline_reasons"] == [failure["detail"]]


def test_long_multichoice_accepts_a_planner_decline_with_identical_ids():
    client = _Client([
        _choice([1, 2], 0, decline="request 168 unregistered or disabled"),
        _choice([1, 2], 0, decline="request 168 unregistered or disabled"),
    ])

    result = check_long_multichoice(client, ARGS, ROW, {"max_packed_rows": 4}, [1, 2])

    assert result["speculates"] is False
    assert result["decline_reasons"] == ["request 168 unregistered or disabled"]


def test_long_multichoice_rejects_rows_that_took_different_routes():
    client = _Client([_choice([1, 2], 3), _choice([1, 2], 0)])

    with pytest.raises(AssertionError) as excinfo:
        check_long_multichoice(client, ARGS, ROW, {"max_packed_rows": 4}, [1, 2])

    assert excinfo.value.args[0] == {
        "long_multichoice_route_split": {"id": "long_4096", "cycles": [3, 0]}
    }


def test_long_multichoice_rejects_a_group_of_the_wrong_size():
    client = _Client([_choice([1, 2], 3)])

    with pytest.raises(AssertionError) as excinfo:
        check_long_multichoice(client, ARGS, ROW, {"max_packed_rows": 4}, [1, 2])

    assert excinfo.value.args[0] == {"long_multichoice": {"choices": 1}}


def test_long_prompt_rows_are_opt_in_and_parsed_as_lengths():
    """The committed category evidence is the short suite; long rows are opt-in."""

    parser = build_parser()

    assert parser.parse_args(["--json", "report.json"]).long_prompt_tokens == ""
    requested = parser.parse_args(["--json", "report.json", "--long-prompt-tokens", "2048,4096"])
    assert [int(value) for value in requested.long_prompt_tokens.split(",")] == [2048, 4096]
