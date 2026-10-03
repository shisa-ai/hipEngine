"""The assistant-forward GPU module must guard every node on its artifacts.

``tests/test_gpu_gemma4_assistant_forward.py`` loads a 17 GB backbone and its MTP
head. Its first three nodes are wrapped in the module's artifact guard, but the
adapter-cycle node was not: on a box whose artifacts live at the moved location
it ran anyway and failed with ``FileNotFoundError`` at module scope instead of
skipping. That put it first in the all-suite failure list, ahead of every real
GPU failure, because the stale hard-coded ``/models/gguf`` root no longer
resolves.

This pins the two invariants that failure violated: every node in the module is
artifact-guarded, and the module resolves the backbone and head from the known
locations rather than a single stale path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("test_gpu_gemma4_assistant_forward.py")

# The locations the gemma-4 artifact has lived at. Kept local rather than read
# off the module under test so a stale pin in that module is caught instead of
# silently agreeing with itself.
_KNOWN_ROOTS = (
    Path("/models/gguf/gemma-4-26B-A4B-it-GGUF"),
    Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF"),
)
_BACKBONE_NAME = "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"


def _assistant_forward_module():
    spec = importlib.util.spec_from_file_location(
        "_assistant_forward_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_node_in_the_assistant_forward_module_is_artifact_guarded() -> None:
    """A node without the guard turns a missing artifact into a hard failure."""

    module = _assistant_forward_module()
    nodes = [
        (name, value)
        for name, value in vars(module).items()
        if name.startswith("test_") and callable(value)
    ]
    assert nodes, "no test nodes found in the assistant-forward module"
    unguarded = [
        name
        for name, func in nodes
        if not any(
            getattr(mark, "name", "") == "skipif"
            for mark in getattr(func, "pytestmark", [])
        )
    ]
    assert not unguarded, (
        f"{unguarded} run without the artifact guard; a box without the gemma-4 "
        "artifacts would fail these nodes with FileNotFoundError instead of "
        "skipping them"
    )


def test_the_assistant_forward_artifact_resolves_from_a_known_root() -> None:
    """The stale ``/models/gguf`` pin is what skipped the three guarded nodes."""

    module = _assistant_forward_module()
    present = [root for root in _KNOWN_ROOTS if (root / _BACKBONE_NAME).is_file()]
    if not present:
        pytest.skip("no gemma-4 backbone at a known root on this box")
    assert module.BACKBONE.is_file(), (
        f"the backbone is present at {present[0]} but the module resolved "
        f"{module.BACKBONE}; the artifact pin is stale"
    )
    assert module.HEAD.is_file(), (
        f"the module resolved no existing head ({module.HEAD}); the MTP/ sidecar "
        "beside the backbone was not discovered"
    )
    assert module.HEAD.parent.name == "MTP"
