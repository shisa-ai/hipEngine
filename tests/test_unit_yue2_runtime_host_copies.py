"""Guard the host-copy lifetime rule in the YuE2 AR runtime.

``host_array_ptr(f(x))`` is unsafe: the temporary array is released as soon as
``host_array_ptr`` returns, while the pointer-only ``copy_host_to_device`` keeps
using it. That produced silent garbage hidden rows in ``push_token`` (the H2D copy
read freed memory) and it is easy to reintroduce. ``copy_host_array_to_device``
takes the array itself and retains the owner through the synchronous copy.

The check is structural because the failure is invisible without a GPU: the copy
succeeds and the destination simply holds whatever now lives at that address.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULES = (
    REPO / "hipengine/runtime/yue2_ar.py",
    REPO / "hipengine/loading/yue2.py",
)
SAFE_ARGUMENTS = (ast.Name, ast.Attribute, ast.Subscript, ast.Constant)


def temporary_pointer_uses(source: str) -> list[int]:
    """Line numbers where ``host_array_ptr`` receives a freshly built array."""
    tree = ast.parse(source)
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Name) or function.id != "host_array_ptr":
            continue
        if not node.args:
            continue
        if not isinstance(node.args[0], SAFE_ARGUMENTS):
            offenders.append(node.lineno)
    return offenders


def test_checker_flags_a_temporary_and_allows_a_named_array():
    """The guard itself must fail on the pattern it exists to catch."""
    assert temporary_pointer_uses("copy_host_to_device(buf, host_array_ptr(f32_to_bf16_bits(row)))") == [1]
    assert temporary_pointer_uses("copy_host_to_device(buf, host_array_ptr(np.ascontiguousarray(x)))") == [1]
    assert temporary_pointer_uses("copy_host_to_device(buf, host_array_ptr(bits))") == []
    assert temporary_pointer_uses("copy_host_to_device(buf, host_array_ptr(self._ctx_len_host[branch]))") == []
    assert temporary_pointer_uses("copy_host_array_to_device(buf, f32_to_bf16_bits(row))") == []


@pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
def test_no_host_pointer_to_a_temporary(module: Path):
    offenders = temporary_pointer_uses(module.read_text())
    assert offenders == [], f"{module}: host_array_ptr receives a temporary at line(s) {offenders}"


def test_push_token_uploads_through_the_owner_retaining_helper():
    source = (REPO / "hipengine/runtime/yue2_ar.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "push_token"
        or isinstance(node, ast.FunctionDef) and node.name == "push_token"
    ]
    assert calls, "push_token disappeared from the runtime"
    body = next(node for node in calls if isinstance(node, ast.FunctionDef))
    uploads = [
        ast.unparse(node)
        for node in ast.walk(body)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "copy_host_array_to_device"
    ]
    assert any("f32_to_bf16_bits" in text for text in uploads), (
        "push_token must upload the bf16 hidden row with copy_host_array_to_device"
    )
