"""Architectural invariants from AGENTS.md, checked against the tree.

These are the rules the project says define what hipEngine is. Each finding here
is a concrete edit, not a judgement call.
"""

from __future__ import annotations

import re

from ..core import REPO_ROOT, Row
from ..inventory import corpus
from . import finding, register

#  A module-level `import torch`, not one inside a function or a TYPE_CHECKING block.
TORCH_IMPORT = re.compile(r"^(?:import torch|from torch\b)", re.M)
#  Branching on a registry axis instead of registering against a key.
AXIS_BRANCH = re.compile(
    r"""\bif\s+[\w.\[\]"']*\b(backend|quant)\b[\w.\[\]"']*\s*(?:==|!=|\bin\b)\s*[\[(]?["']""")
#  Tests that drive the GPU need a runtime-availability guard or no-ROCm CI fails.
HIP_USE = re.compile(r"libamdhip64|hipcc|rocprofv3|hipModuleLaunch|HIP_VISIBLE_DEVICES")
HIP_GUARD = re.compile(r"pytest\.skip|pytest\.mark\.skipif|importorskip|requires_hip|_hip_available")


def _module_level(text: str, pattern: re.Pattern) -> list[int]:
    """Line numbers where `pattern` matches at indent zero, outside TYPE_CHECKING."""
    out = []
    in_type_checking = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("if TYPE_CHECKING"):
            in_type_checking = True
            continue
        if in_type_checking and line and not line[0].isspace():
            in_type_checking = False
        if in_type_checking or (line[:1].isspace() if line else True):
            continue
        if pattern.match(line):
            out.append(number)
    return out


@register("torch-hot-path")
def torch_hot_path() -> tuple[list[Row], dict]:
    """`import torch` is not allowed in any module reached by `LLM.generate()`."""
    rows = []
    for path, text in sorted(corpus().items()):
        if not path.startswith("hipengine/") or not path.endswith(".py"):
            continue
        #  The dlpack bridge at the user boundary is the sanctioned exception.
        if "/torch" in path or path.endswith("torch_bridge.py"):
            continue
        for line in _module_level(text, TORCH_IMPORT):
            rows.append(finding(
                "torch-hot-path", f"{path}:{line}",
                f"module-level torch import in {path}", f"{path}:{line}",
                fix="Move the import behind the optional `hipengine[torch]` extra, or into the "
                    "function that needs it. A torch import on the hot path is an architectural "
                    "change, not a refactor.",
                why="AGENTS.md 'Torch-free runtime' forbids torch in any module reached by LLM.generate()",
            ))
    return rows, {"scanned": "hipengine/**/*.py"}


@register("axis-branch")
def axis_branch() -> tuple[list[Row], dict]:
    """`if backend == ...` / `if quant == ...` belongs in a registry key, not a branch."""
    rows = []
    for path, text in sorted(corpus().items()):
        if not path.startswith("hipengine/") or not path.endswith(".py"):
            continue
        #  The registry and its resolution code legitimately compare axis values.
        if "/registry" in path or path.endswith(("backends.py", "policy.py")):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if AXIS_BRANCH.search(line):
                rows.append(finding(
                    "axis-branch", f"{path}:{number}",
                    f"branches on a registry axis: {line.strip()[:90]}", f"{path}:{number}",
                    fix="Register the behaviour against a `(backend, layer, quant, variant)` key "
                        "and resolve it, instead of branching on the axis value here.",
                    why="AGENTS.md 'Four-axis plugin registry' forbids axis branches in "
                        "dispatch/engine/model code",
                ))
    return rows, {"scanned": "hipengine/**/*.py"}


@register("unguarded-hip-test")
def unguarded_hip_test() -> tuple[list[Row], dict]:
    """A test that drives ROCm must skip, not fail, where there is no GPU."""
    rows = []
    for path, text in sorted(corpus().items()):
        if not path.startswith("tests/") or not path.endswith(".py"):
            continue
        if not HIP_USE.search(text) or HIP_GUARD.search(text):
            continue
        line = next((n for n, l in enumerate(text.splitlines(), 1) if HIP_USE.search(l)), 1)
        rows.append(finding(
            "unguarded-hip-test", path,
            f"{path.rsplit('/', 1)[-1]} touches the ROCm runtime with no availability guard", f"{path}:{line}",
            fix='Add an explicit guard — `ctypes.CDLL("libamdhip64.so")` in a try/except with '
                "`pytest.skip` — so a no-ROCm runner skips instead of failing release validation.",
            why="AGENTS.md 'During Work' requires a HIP-availability guard on GPU tests",
        ))
    return rows, {"scanned": "tests/**/*.py"}
