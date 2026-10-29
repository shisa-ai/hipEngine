"""Execute the documented route A/B shell example without loading a model."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("inherited", [None, "0", "1"])
def test_documented_slot_local_ab_uses_explicit_off_and_on(inherited):
    source = Path(__file__).resolve().parents[1] / "scripts/gguf_prefill_route_ab.py"
    doc = ast.get_docstring(ast.parse(source.read_text()))
    start = doc.index("    for mode in off on; do")
    end = doc.index("    done", start) + len("    done")
    example = doc[start:end]
    environment = {"PATH": os.defpath}
    key = "HIPENGINE_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON"
    environment.pop(key, None)
    if inherited is not None:
        environment[key] = inherited
    # Replace the expensive model command with a shell function. Run the actual
    # documented assignments/branches, including their inherited environment.
    command = f'python() {{ printf "%s\\n" "${key}"; }}\n' + example
    result = subprocess.run(
        ["bash", "-eu", "-c", command], env=environment,
        text=True, capture_output=True, check=True, timeout=10,
    )
    assert result.stdout.splitlines() == ["0", "1"]
