from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/docs/check_refactor.py"


def _tool():
    spec = importlib.util.spec_from_file_location("check_refactor_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(identifier="R0001", *, title="Migrate caller ownership", owner="hipengine/runtime/", scope="`owner_state`", done="Caller isolation and failure cleanup tests pass."):
    return f"### {identifier} — {title}\n\nOwner: `{owner}`.\nScope: {scope}.\nDone when: {done}\n\nMove the live caller to its request owner.\n"


def _root(tmp_path):
    (tmp_path / "hipengine/runtime").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    return tmp_path


def _errors(text, tmp_path):
    return _tool().validate_ledger(text, root=_root(tmp_path), ledger=Path("docs/REFACTOR.md"))


def test_valid_current_entry_and_wrapped_completion_condition(tmp_path):
    assert not _errors(_entry(done="Caller isolation and failure cleanup\n  tests pass."), tmp_path)


def test_distinct_derived_ids_and_shared_original_referral_are_not_duplicates(tmp_path):
    text = _entry("R0281-ATTN") + _entry("R0281-GDN / R0377")
    text += _entry("RF-1 / R0187") + _entry("RF-3 / R0187")
    assert not _errors(text, tmp_path)


@pytest.mark.parametrize("identifiers", [("R0001", "R0001"), ("R0001 / R0002", "R0002"), ("R0343–R0348", "R0346")])
def test_duplicate_primary_alias_and_range_member_fail(tmp_path, identifiers):
    errors = _errors(_entry(identifiers[0]) + _entry(identifiers[1]), tmp_path)
    assert any("duplicate active ID" in error for error in errors)


@pytest.mark.parametrize("field", ["Owner", "Scope", "Done when"])
def test_required_field_cannot_be_omitted(tmp_path, field):
    text = "\n".join(line for line in _entry().splitlines() if not line.startswith(field + ":"))
    assert any(field in error for error in _errors(text, tmp_path))


@pytest.mark.parametrize("field,value", [("owner", "TODO"), ("scope", "TBD"), ("done", "<required>")])
def test_placeholder_is_not_a_contract(tmp_path, field, value):
    assert _errors(_entry(**{field: value}), tmp_path)


def test_owner_must_resolve_inside_repository(tmp_path):
    errors = _errors(_entry(owner="missing/subsystem.py"), tmp_path)
    assert any("owner path" in error for error in errors)


@pytest.mark.parametrize("status", ["completed", "closed", "resolved", "superseded", "retired"])
def test_closed_status_cannot_live_in_backlog(tmp_path, status):
    errors = _errors(_entry() + f"\nStatus: {status}\n", tmp_path)
    assert any("closed entry" in error for error in errors)


def test_completed_heading_fails_without_rejecting_future_completion_conditions(tmp_path):
    assert any("closed entry" in error for error in _errors(_entry(title="Ownership repair (fixed)"), tmp_path))


def test_missing_local_link_and_anchor_fail(tmp_path):
    root = _root(tmp_path)
    (root / "docs/target.md").write_text("# Present heading\n")
    text = _entry() + "\n[missing](missing.md) [bad anchor](target.md#absent)\n"
    errors = _tool().validate_ledger(text, root=root, ledger=Path("docs/REFACTOR.md"))
    assert any("missing.md" in error for error in errors)
    assert any("#absent" in error for error in errors)


def test_existing_relative_link_anchor_and_external_links_pass(tmp_path):
    root = _root(tmp_path)
    (root / "docs/target.md").write_text("# Present `owner_state`\n")
    text = _entry() + "\n[local](target.md#present-owner_state) [web](https://example.org/not-local)\n"
    assert not _tool().validate_ledger(text, root=root, ledger=Path("docs/REFACTOR.md"))


def test_fenced_diagnostic_text_does_not_create_entries_statuses_or_links(tmp_path):
    text = _entry() + "\n```text\n### R0001 — Fake\nStatus: completed\n[example](absent.md)\n```\n"
    assert not _errors(text, tmp_path)


def test_repository_ledger_satisfies_the_same_contract():
    root = SCRIPT.parents[2]
    assert not _tool().validate_ledger(
        (root / "docs/REFACTOR.md").read_text(), root=root, ledger=Path("docs/REFACTOR.md")
    )


def test_gate_is_part_of_documentation_entrypoint():
    path = SCRIPT.with_name("check_all.py")
    spec = importlib.util.spec_from_file_location("check_all_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert any(command == ["scripts/docs/check_refactor.py"] for _, command, _ in module.GATES)


def test_empty_ledger_and_invalid_heading_fail(tmp_path):
    assert _errors("# Backlog\n", tmp_path)
    root = tmp_path
    errors = _tool().validate_ledger(_entry("unknown"), root=root, ledger=Path("docs/REFACTOR.md"))
    assert any("identifier" in error for error in errors)
