"""Wheel ownership must not depend on the source-only scripts workspace."""
import ast
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_wheel_excludes_research_scripts():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["hipengine"]


def test_product_has_no_scripts_imports():
    violations = []
    for path in (ROOT / "hipengine").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            if any(name == "scripts" or name.startswith("scripts.") for name in names):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations
