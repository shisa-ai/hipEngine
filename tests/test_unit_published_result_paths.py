import pytest
import ast

from scripts._published_results import cited_result_paths
from scripts.check_published_command_drift import _call_bindings


def test_nested_and_flat_result_paths_do_not_collide():
    text = """
    [a](results/one/artifact.json)
    [b](results/two/artifact.json)
    [c](results/flat.json)
    [d](https://github.com/org/repo/blob/main/benchmarks/results/one/artifact.json)
    corpus.jsonl
    """
    assert cited_result_paths(text) == ["flat.json", "one/artifact.json", "two/artifact.json"]


def test_result_citations_cannot_escape_root():
    with pytest.raises(ValueError):
        cited_result_paths("[bad](results/../../secret.json)")


def test_cli_helper_literal_defaults_are_bound_without_guessing_overrides():
    target = ast.parse("def options(parser, flags=(), *, aliases=('--storage',)): pass").body[0]
    assert _call_bindings(target, ast.parse("options(parser)").body[0].value) == {
        "flags": (), "aliases": ("--storage",),
    }
    assert _call_bindings(target, ast.parse("options(parser, aliases=dynamic)").body[0].value) == {
        "flags": (),
    }
    assert _call_bindings(target, ast.parse("options(parser, **dynamic)").body[0].value) == {}
