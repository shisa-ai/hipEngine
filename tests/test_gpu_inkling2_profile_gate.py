"""End-to-end smoke test for scripts/inkling2_profile_gate.py on the real Inkling 2 GGUF model.

The model is a multi-gigabyte local file, so the test runs only when
``HIPENGINE_INKLING2_MINI_GGUF`` points at it and a HIP runtime loads. It runs the
gate as a limited screen (which can never pass) and checks that every section
ran and that the deterministic sections held.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE = REPO_ROOT / "scripts" / "inkling2_profile_gate.py"
DETERMINISTIC_GATES = ("control_semantics", "repeat_determinism", "isolation", "public_route")


def _skip_unless_runnable() -> str:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    model = os.environ.get("HIPENGINE_INKLING2_MINI_GGUF", "")
    if not model or not Path(model).is_file():
        pytest.skip("HIPENGINE_INKLING2_MINI_GGUF must point at the Inkling 2 GGUF model")
    return model


@pytest.mark.parametrize("candidate", ["fast", "fast_graph"])
def test_profile_gate_screen_runs_every_section(tmp_path, candidate):
    model = _skip_unless_runnable()
    output = tmp_path / f"{candidate}.json"
    command = [
        sys.executable, "-B", str(GATE), model,
        "--backend", os.environ.get("HIPENGINE_INKLING2_TEST_BACKEND", "auto"),
        "--candidate", candidate, "--limit", "2", "--repeat-runs", "3", "--json", str(output),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    # A limited screen reports through the normal exit code and never passes.
    assert completed.returncode == 1, completed.stdout[-2000:] + completed.stderr[-3000:]
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["kind"] == "hipengine_execution_profile_inkling2_gguf_gate"
    assert artifact["status"] == "failed_or_screen_only" and not artifact["measurement_valid"]
    assert artifact["candidate"]["candidate_owner"]["routes"] == {
        "device_routes": candidate != "fast",
        "resident_decode": candidate != "fast",
        "decode_graph": candidate == "fast_graph",
    }
    assert artifact["protocol"]["teacher_forced_rows"] == 48
    assert artifact["protocol"]["coverage"]["prompts"] == 2
    assert artifact["quality"]["section_2_11"]["global"]["rows"] == 48
    assert artifact["task_quality"]["cases"] == 12
    for section in DETERMINISTIC_GATES:
        assert artifact["gates"][section] is True, section
    assert artifact["repeat_determinism"]["runs_per_request"] == {
        row["id"]: 3 for row in artifact["prompts"]
    }
    assert artifact["isolation"]["bit_identical_requests"] == 2
    assert artifact["public_route"]["generated_tokens"] > 0
    assert artifact["decision"] == {"passed": False}
    worker = json.loads((output.parent / f"{output.stem}.isolation" / "worker.json").read_text())
    assert worker["route"] == artifact["candidate"]["candidate_owner"]
