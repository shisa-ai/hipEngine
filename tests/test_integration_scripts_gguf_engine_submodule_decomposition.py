"""The decomposition packet must describe itself.

This script produced the C1-C8 admission/decode decomposition whose numbers kept being quoted as
current after grouped prefill shipped, because nothing in the packet said where or from what code it
was measured. These tests pin the stamp, in both the normal and the GPU-free fallback path.
"""

from __future__ import annotations

import importlib.util
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_engine_submodule_decomposition.py"


def _load():
    spec = importlib.util.spec_from_file_location("gesd_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provenance_is_derived_from_the_machine_not_written_by_hand() -> None:
    module = _load()
    prov = module._provenance(["--arm", "ar", "--output", "x.json"])
    assert prov["host_name"] == socket.gethostname()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert prov["hipengine_commit"] == head
    assert prov["command"] == ["--arm", "ar", "--output", "x.json"]


def test_provenance_reports_a_real_dirty_axis() -> None:
    """`dirty` must be a boolean, not an absent field, so a packet can never read as clean by default."""
    prov = _load()._provenance(["--arm", "ar"])
    assert isinstance(prov["dirty"], bool)


def test_provenance_falls_back_without_the_runtime(tmp_path, monkeypatch) -> None:
    """A no-HIP runner must still get a stamped packet, and must be able to tell it is a fallback."""
    import hipengine.benchmark.provenance as provenance

    def boom(**kwargs):
        raise RuntimeError("no device here")

    monkeypatch.setattr(provenance, "collect_artifact_provenance", boom)
    prov = _load()._provenance(["--arm", "mtp"])
    assert "provenance_error" in prov
    assert "no device here" in prov["provenance_error"]
    assert prov["host_name"] == socket.gethostname()
    assert re.fullmatch(r"[0-9a-f]{40}", str(prov["hipengine_commit"])), prov["hipengine_commit"]
    assert prov["command"] == ["--arm", "mtp"]
