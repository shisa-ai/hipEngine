"""Host-only import boundary for the B1 verifier owner-map script.

The script is executed directly from a checkout that need not be installed:
``main`` inserts the repository root on ``sys.path`` before importing
``hipengine``. A resolver import at module scope would break that contract, so
these tests run the module with ``-S`` (no site-packages, no editable install)
and a temporary working directory, and separately pin the import order.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/qwen38_b1_verifier_owner_map.py"


def test_script_module_imports_without_site_packages(tmp_path: Path) -> None:
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import runpy\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='rf8_boundary')\n"
        "print('imported')\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-S", str(driver)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_resolver_import_is_deferred_until_after_path_setup() -> None:
    tree = ast.parse(SCRIPT.read_text())

    module_hipengine_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_hipengine_imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "hipengine" or alias.name.startswith("hipengine.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "hipengine" or node.module.startswith("hipengine."):
                module_hipengine_imports.append(node.module)
    assert module_hipengine_imports == []

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    path_insert = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "path"
    )
    resolver_import = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.ImportFrom)
        and node.module == "hipengine.kernels.backends"
        and any(
            alias.name == "backend_package_capability" for alias in node.names
        )
    )
    assert resolver_import.lineno > path_insert.lineno
