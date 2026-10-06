"""Re-execute a torch-absence test in a clean interpreter.

Torch-free architectural claims of the form "importing/executing this path
never imports torch" are asserted against ``sys.modules``, which is
interpreter-global. Several unit tests in the default tier legitimately import
torch when it is available (``qwen38_dms_*`` at module level, maple and the
quant quality harness inside test bodies), so in a full suite run they poison
the session and the claim fails *order-dependently* - the code under test may
be perfectly torch-free.

The claim is about a fresh process importing that code, so each such test
re-executes its own node in a child ``pytest`` that starts from a clean
``sys.modules``. The parent skips its body when the child passed; inside the
child the helper returns ``True`` and the original assertions run unchanged,
including the full fixture path.

Usage::

    def test_something_torch_free(request):
        if not run_in_clean_interpreter(request.node.nodeid):
            return
        ...original body...

The child re-runs only the one node, so collection and conftest loading match
an isolated manual run of that test. Recursion is prevented by an environment
flag keyed on the node id.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_FLAG = "HIPENGINE_TORCH_ABSENCE_CHILD"


def run_in_clean_interpreter(nodeid: str) -> bool:
    """Run ``nodeid`` in a fresh interpreter and report whether this is that child.

    Returns ``True`` inside the clean child (caller proceeds with the real test
    body). In the parent, re-executes the node in a child pytest, asserts the
    child passed - surfacing its full output - and returns ``False`` so the
    caller skips its body in the already-polluted session.
    """
    if os.environ.get(ENV_FLAG) == nodeid:
        return True
    env = dict(os.environ, **{ENV_FLAG: nodeid})
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "-q", "--tb=short"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"torch-free claim failed in a clean interpreter for {nodeid} "
        f"(exit {proc.returncode}):\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    return False