from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from scripts.check_test_tiers import inspect_tests


@pytest.mark.parametrize("root", ["tests", "audit/tests"])
def test_repository_uses_tier_names_and_isolates_unit_imports(root: str) -> None:
    assert inspect_tests(Path(__file__).resolve().parents[1] / root) == []


def test_product_and_audit_contract_tests_are_in_default_discovery() -> None:
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    config = tomllib.loads(project.read_text())["tool"]["pytest"]["ini_options"]
    assert {"tests", "audit/tests"} <= set(config["testpaths"])


def test_explicit_audit_collection_respects_nonunit_suite() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "audit/tests", "--suite", "integration",
         "--collect-only", "-q"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    # All audit contracts are unit tests: an integration-only selection is empty,
    # not an unknown-option error from suite hooks living in the product subtree.
    assert result.returncode == 5, result.stdout + result.stderr


def test_product_test_imports_resolve_to_product_tree() -> None:
    import tests._rocm_guard as guard

    assert Path(guard.__file__).resolve().parent == Path(__file__).resolve().parent


def test_audit_first_collection_preserves_product_helper_imports() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "audit/tests",
         "tests/test_unit_test_tier_names.py::test_product_test_imports_resolve_to_product_tree",
         "-o", "addopts=", "-q"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_execution_tiers_are_accepted(tmp_path: Path) -> None:
    for tier in ("unit", "integration", "gpu", "benchmark", "live", "slow"):
        (tmp_path / f"test_{tier}_example.py").write_text("def test_example(): pass\n")
    assert inspect_tests(tmp_path) == []


def test_legacy_filename_is_reported(tmp_path: Path) -> None:
    (tmp_path / "test_example.py").write_text("")
    assert inspect_tests(tmp_path) == ["test_example.py: missing execution-tier prefix"]


def test_nested_legacy_filename_is_reported(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "test_example.py").write_text("")
    assert inspect_tests(tmp_path) == [
        "nested/test_example.py: missing execution-tier prefix"
    ]


def test_unit_cannot_import_an_expensive_test_module(tmp_path: Path) -> None:
    (tmp_path / "test_unit_example.py").write_text(
        "from tests.test_gpu_kernel import helper\n"
        "from tests import test_live_model as model\n"
        "import tests.test_integration_service\n"
    )
    problems = inspect_tests(tmp_path)
    assert len(problems) == 3
    assert "unit imports test_gpu_kernel" in problems[0]
    assert "unit imports test_live_model" in problems[1]
    assert "unit imports test_integration_service" in problems[2]


@pytest.mark.parametrize("statement", [
    "from .test_gpu_kernel import helper",
    "from . import test_gpu_kernel",
    "from tests.nested.test_gpu_kernel import helper",
    "import tests.nested.test_gpu_kernel",
    "from tests.nested import test_gpu_kernel",
])
def test_nested_unit_imports_cannot_bypass_isolation(tmp_path: Path, statement: str) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "test_unit_example.py").write_text(statement + "\n")
    problems = inspect_tests(tmp_path)
    assert len(problems) == 1
    assert "nested/test_unit_example.py:1: unit imports test_gpu_kernel" in problems[0]


def test_unit_helpers_and_unit_modules_are_allowed(tmp_path: Path) -> None:
    (tmp_path / "_helpers.py").write_text("def test_gpu_named_function(): pass\n")
    (tmp_path / "test_unit_example.py").write_text(
        "from tests._synthetic_weights import helper\n"
        "from tests.test_unit_other import other\n"
        "from .test_unit_other import test_gpu_named_function\n"
        "from . import _synthetic_weights\n"
        "from tests._helpers import test_gpu_named_function\n"
        "from ._helpers import test_gpu_named_function\n"
    )
    assert inspect_tests(tmp_path) == []
