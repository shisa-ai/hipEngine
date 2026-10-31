"""Consumer contract for the pure server owner modules.

``hipengine.server.schemas``, ``hipengine.server.json_schema``,
``hipengine.server.chat_output``, ``hipengine.server.chat_result_policy``,
``hipengine.server.chat_sessions``,
``hipengine.server.chat_validation``, ``hipengine.server.model_lifecycle``,
``hipengine.server.metrics``,
``hipengine.server.readiness``,
``hipengine.server.session_routes``, ``hipengine.server.session_snapshots`` and
``hipengine.server.session_snapshot_routes`` are owners under the HTTP boundary.
``hipengine.server.api`` imports the names its own application code references;
it is not a compatibility facade for the rest.

These checks pin the two properties that keep that true:

* every owner name imported into ``api.py`` is referenced there, so no unused
  re-export survives the migration;
* application code, tests and scripts import owner names from the owner module
  instead of reaching them through ``hipengine.server.api`` or
  ``hipengine.server``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "hipengine" / "server" / "api.py"
OWNER_PATHS = {
    "hipengine.server.schemas": REPO_ROOT / "hipengine" / "server" / "schemas.py",
    "hipengine.server.json_schema": REPO_ROOT / "hipengine" / "server" / "json_schema.py",
    "hipengine.server.chat_output": REPO_ROOT / "hipengine" / "server" / "chat_output.py",
    "hipengine.server.chat_result_policy": (
        REPO_ROOT / "hipengine" / "server" / "chat_result_policy.py"
    ),
    "hipengine.server.chat_sessions": REPO_ROOT / "hipengine" / "server" / "chat_sessions.py",
    "hipengine.server.chat_validation": REPO_ROOT / "hipengine" / "server" / "chat_validation.py",
    "hipengine.server.model_lifecycle": (
        REPO_ROOT / "hipengine" / "server" / "model_lifecycle.py"
    ),
    "hipengine.server.metrics": REPO_ROOT / "hipengine" / "server" / "metrics.py",
    "hipengine.server.session_routes": REPO_ROOT / "hipengine" / "server" / "session_routes.py",
    "hipengine.server.session_snapshots": (
        REPO_ROOT / "hipengine" / "server" / "session_snapshots.py"
    ),
    "hipengine.server.session_snapshot_routes": (
        REPO_ROOT / "hipengine" / "server" / "session_snapshot_routes.py"
    ),
    "hipengine.server.readiness": REPO_ROOT / "hipengine" / "server" / "readiness.py",
    "hipengine.server.stream_events": (
        REPO_ROOT / "hipengine" / "server" / "stream_events.py"
    ),
}
SCANNED_ROOTS = ("hipengine", "tests", "scripts")
API_MODULES = ("hipengine.server.api", "hipengine.server")


def _defined_names(tree: ast.Module) -> set[str]:
    """Module-level names, including definitions under a top-level if/try."""

    names: set[str] = set()

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.If, ast.Try)):
                walk(node.body)
                walk(node.orelse)
                for handler in getattr(node, "handlers", ()):
                    walk(handler.body)

    walk(tree.body)
    return names


def _owner_names() -> dict[str, set[str]]:
    return {
        module: _defined_names(ast.parse(path.read_text(encoding="utf-8")))
        for module, path in OWNER_PATHS.items()
    }


def _api_imports() -> dict[str, set[str]]:
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"))
    imported: dict[str, set[str]] = {module: set() for module in OWNER_PATHS}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in imported:
            imported[node.module].update(alias.name for alias in node.names)
    return imported


def _loaded_names(tree: ast.Module) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _api_aliases(tree: ast.Module) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hipengine.server":
            for alias in node.names:
                if alias.name == "api":
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "hipengine.server.api":
                    aliases.add(alias.asname or "hipengine")
    return aliases


@pytest.mark.parametrize("module", sorted(OWNER_PATHS))
def test_api_imports_only_the_owner_names_it_uses(module: str) -> None:
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"))
    imported = _api_imports()[module]
    assert imported, module
    assert imported <= _owner_names()[module], sorted(imported - _owner_names()[module])
    assert imported <= _loaded_names(tree), sorted(imported - _loaded_names(tree))


def test_application_tests_and_scripts_import_owners_directly() -> None:
    owners = _owner_names()
    violations: list[str] = []
    for root in SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if path == API_PATH:
                continue
            text = path.read_text(encoding="utf-8")
            if "hipengine.server" not in text:
                continue
            tree = ast.parse(text)
            aliases = _api_aliases(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in API_MODULES:
                    for alias in node.names:
                        for module, names in owners.items():
                            if alias.name in names:
                                violations.append(
                                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
                                    f"{alias.name} imported through {node.module}"
                                )
                elif isinstance(node, ast.Attribute):
                    dotted = _dotted(node)
                    if dotted is None:
                        continue
                    prefix, _, name = dotted.rpartition(".")
                    if prefix == "hipengine.server.api" or prefix in aliases:
                        for module, names in owners.items():
                            if name in names:
                                violations.append(
                                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
                                    f"{dotted} reaches {module} through the application module"
                                )
    assert not violations, "\n".join(violations)
