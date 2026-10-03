"""Keep generic live HTTP parity coverage on the detected device backend."""

from __future__ import annotations

import ast
from pathlib import Path


def test_mtp_http_surface_uses_auto_backend() -> None:
    """The HTTP contract does not require a particular GPU architecture."""
    source = Path(__file__).with_name("test_live_mtp_http_surface.py")
    module = ast.parse(source.read_text(encoding="utf-8"))
    client = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "client"
    )
    command = next(
        node.value
        for node in ast.walk(client)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "command"
            for target in node.targets
        )
    )
    assert isinstance(command, ast.List)
    arguments = [
        node.value if isinstance(node, ast.Constant) else None
        for node in command.elts
    ]
    assert arguments[arguments.index("--backend") + 1] == "auto", (
        "Generic HTTP parity tests must not launch code for a fixed foreign GPU"
    )
