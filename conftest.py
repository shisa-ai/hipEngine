"""Repository-wide test-tier selection, shared by product and audit tests."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--suite",
        choices=("unit", "integration", "gpu", "benchmark", "live", "slow", "all"),
        default="unit",
        help="Discover test_<suite>_*.py; all explicitly includes legacy tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    suite = config.getoption("--suite")
    pattern = "test_*.py" if suite == "all" else f"test_{suite}_*.py"
    config.getini("python_files")[:] = [pattern]
