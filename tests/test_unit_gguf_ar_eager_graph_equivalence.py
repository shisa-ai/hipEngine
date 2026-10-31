"""Contract tests for the eager/graph true-AR equivalence comparison.

``scripts/gguf_ar_eager_graph_equivalence.py`` is an operator-invoked GPU check:
it loads the artifact twice and compares the two decode paths' token ids. The
device half cannot run in CI, but the comparison half is pure and is what
decides the verdict, so it is pinned here.

The failure that matters is a silent one. If this comparison reported
``matched`` for payloads that disagree -- because it zipped to the shorter
sequence, ignored a missing row, or compared the wrong keys -- the script would
certify a decode graph that decodes different tokens than the eager path, and
every published AR rate would rest on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import gguf_ar_eager_graph_equivalence as checker  # noqa: E402
from scripts.gguf_ar_eager_graph_equivalence import (  # noqa: E402
    ar_rows_by_key,
    compare_ar_token_sequences,
)


def payload(rows: list[dict]) -> dict:
    return {"rows": {"true_ar": rows}}


def test_matching_sequences_compare_clean() -> None:
    rows = [
        {"id": "code-1", "run": 0, "token_ids": [10, 11, 12]},
        {"id": "ja-2", "run": 0, "token_ids": [40, 41]},
    ]
    report = compare_ar_token_sequences(payload(rows), payload(rows))
    assert report["matched"] is True
    assert report["problems"] == []
    assert report["compared_rows"] == 2
    assert report["compared_transitions"] == 5
    assert report["divergences"] == []


def test_divergence_is_named_by_prompt_run_and_transition() -> None:
    eager = payload([{"id": "code-1", "run": 0, "token_ids": [10, 11, 12]}])
    graph = payload([{"id": "code-1", "run": 0, "token_ids": [10, 99, 12]}])
    report = compare_ar_token_sequences(eager, graph)
    assert report["matched"] is False
    assert report["divergences"] == [
        {
            "prompt": "code-1",
            "run": 0,
            "transition": 1,
            "eager_token": 11,
            "graph_token": 99,
        }
    ]
    assert "code-1#run0" in report["problems"][-1]


def test_a_length_mismatch_is_a_problem_even_when_the_prefix_agrees() -> None:
    """A truncated graph replay must not pass by zipping to the shorter list."""

    eager = payload([{"id": "code-1", "run": 0, "token_ids": [10, 11, 12]}])
    graph = payload([{"id": "code-1", "run": 0, "token_ids": [10, 11]}])
    report = compare_ar_token_sequences(eager, graph)
    assert report["matched"] is False
    assert any("length" in problem for problem in report["problems"])
    assert report["compared_transitions"] == 2


def test_rows_present_in_only_one_payload_are_reported() -> None:
    eager = payload(
        [
            {"id": "code-1", "run": 0, "token_ids": [1]},
            {"id": "ja-2", "run": 0, "token_ids": [2]},
        ]
    )
    graph = payload([{"id": "code-1", "run": 0, "token_ids": [1]}])
    report = compare_ar_token_sequences(eager, graph)
    assert report["matched"] is False
    assert any("only in eager" in problem for problem in report["problems"])


def test_empty_payloads_do_not_vacuously_match() -> None:
    """Two payloads that recorded nothing must not certify as equivalent."""

    report = compare_ar_token_sequences(payload([]), payload([]))
    assert report["matched"] is False
    assert any("missing true-AR rows" in problem for problem in report["problems"])


@pytest.mark.parametrize("missing_key", ["rows", "true_ar"])
def test_absent_structure_is_missing_rows_not_a_crash(missing_key: str) -> None:
    if missing_key == "rows":
        report = compare_ar_token_sequences({}, payload([]))
    else:
        report = compare_ar_token_sequences({"rows": {}}, payload([]))
    assert report["matched"] is False


def test_rows_are_indexed_by_id_and_run() -> None:
    indexed = ar_rows_by_key(
        payload(
            [
                {"id": "a", "run": 0, "token_ids": [1]},
                {"id": "a", "run": 1, "token_ids": [2]},
            ]
        )
    )
    assert sorted(indexed) == [("a", 0), ("a", 1)]
    assert indexed[("a", 1)]["token_ids"] == [2]


def test_empty_token_rows_do_not_certify_any_transitions() -> None:
    rows = [{"id": "code-1", "run": 0, "token_ids": []}]
    report = compare_ar_token_sequences(payload(rows), payload(rows))
    assert report["matched"] is False
    assert report["compared_transitions"] == 0
    assert any("empty token" in problem for problem in report["problems"])


def test_duplicate_row_keys_do_not_hide_a_divergent_sequence() -> None:
    eager = payload([
        {"id": "code-1", "run": 0, "token_ids": [99]},
        {"id": "code-1", "run": 0, "token_ids": [10]},
    ])
    graph = payload([{"id": "code-1", "run": 0, "token_ids": [10]}])
    report = compare_ar_token_sequences(eager, graph)
    assert report["matched"] is False
    assert any("duplicate" in problem for problem in report["problems"])


def test_indexing_rejects_duplicate_row_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ar_rows_by_key(payload([
            {"id": "a", "token_ids": [1]},
            {"id": "a", "run": 0, "token_ids": [2]},
        ]))


def test_failed_suite_cannot_reuse_a_previous_payload(monkeypatch, tmp_path) -> None:
    from scripts import qwen36_dense_gguf_suite as suite

    output = tmp_path / "ar-eager.json"
    original = '{"rows": {"true_ar": [{"id": "old", "token_ids": [1]}]}}'
    output.write_text(original)
    previous_argv = sys.argv
    monkeypatch.setattr(suite, "main", lambda: 1)

    with pytest.raises(SystemExit, match="suite failed with exit code 1"):
        checker._run_suite([checker.SUITE, "--output", str(output)], output)

    assert sys.argv is previous_argv
    assert output.read_text() == original


def test_successful_suite_loads_its_new_payload_and_restores_argv(
    monkeypatch, tmp_path,
) -> None:
    from scripts import qwen36_dense_gguf_suite as suite

    output = tmp_path / "ar-graph.json"
    output.write_text('{"old": true}')
    previous_argv = sys.argv
    argv = [checker.SUITE, "--output", str(output)]

    def run_suite():
        assert sys.argv == argv
        output.write_text('{"fresh": true}')
        return 0

    monkeypatch.setattr(suite, "main", run_suite)
    assert checker._run_suite(argv, output) == {"fresh": True}
    assert sys.argv is previous_argv


def test_a_run_key_defaults_to_zero_when_absent() -> None:
    indexed = ar_rows_by_key(payload([{"id": "a", "token_ids": [1]}]))
    assert list(indexed) == [("a", 0)]
