"""Contract checks for the RF-12 server API integration split.

The server API tests were one 25,807-line module. They now live in
behavior-named ``test_integration_server_*`` modules with shared doubles and
builders under ``tests.support``. These checks keep the split from regressing
into a re-export chain: support modules stay test-free, split modules do not
import other test modules, and pytest collects each identity exactly once.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SPLIT_MODULES = (
    "test_integration_server_routing.py",
    "test_integration_server_capabilities.py",
    "test_integration_server_startup.py",
    "test_integration_server_sessions.py",
    "test_integration_server_context.py",
    "test_integration_server_batcher.py",
    "test_integration_server_errors.py",
    "test_integration_server_mtp_accounting.py",
    "test_integration_server_continuation.py",
    "test_integration_server_structured_output.py",
    "test_integration_server_tools.py",
    "test_integration_server_logprobs.py",
    "test_integration_server_streaming.py",
    "test_integration_server_request_semantics.py",
    "test_integration_server_metrics.py",
    "test_integration_server_replay.py",
)
SUPPORT_MODULES = (
    "tests/support/server_api_fakes.py",
    "tests/support/server_api_support.py",
)


def _tree(path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"), filename=path)


def test_split_modules_do_not_import_test_modules() -> None:
    problems = []
    for name in SPLIT_MODULES:
        for node in ast.walk(_tree(f"tests/{name}")):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("tests.test_") or any(
                    alias.name.startswith("test_") for alias in node.names
                ):
                    problems.append(f"tests/{name}:{node.lineno}: {module}")
            elif isinstance(node, ast.Import):
                problems.extend(
                    f"tests/{name}:{node.lineno}: {alias.name}"
                    for alias in node.names
                    if alias.name.startswith("tests.test_")
                )
    assert problems == [], "split modules must not re-export or collect other test modules"


def test_support_modules_are_test_free_and_fixture_free() -> None:
    for name in SUPPORT_MODULES:
        tree = _tree(name)
        for node in tree.body:
            assert not (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            ), f"{name} defines a test"
            decorators = getattr(node, "decorator_list", ())
            assert not any(
                isinstance(decorator, ast.Attribute) and decorator.attr == "fixture"
                for decorator in decorators
            ), f"{name} defines a pytest fixture"
            assert not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"), (
                f"{name} defines a test class"
            )


def test_no_test_or_script_imports_the_legacy_module() -> None:
    offenders = []
    for root in (REPO_ROOT / "tests", REPO_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if path.name == "test_integration_server_split_contract.py":
                continue
            if "tests.test_integration_server_api" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], "test helpers belong to tests.support, not a test module"
    assert not (REPO_ROOT / "tests/test_integration_server_api.py").exists()


def test_split_modules_collect_each_identity_once() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *(f"tests/{name}" for name in SPLIT_MODULES),
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    identities = [
        line.split("::", 1)[1]
        for line in result.stdout.splitlines()
        if "::test_" in line
    ]
    assert identities
    duplicates = sorted(
        {identity for identity in identities if identities.count(identity) > 1}
    )
    assert duplicates == []
    for name in SPLIT_MODULES:
        assert any(line.startswith(f"tests/{name}::test_") for line in result.stdout.splitlines()), (
            f"{name} collected no tests"
        )


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_split_module_filename_matches_integration_tier(name: str) -> None:
    assert name.startswith("test_integration_")
    assert (REPO_ROOT / "tests" / name).is_file()
