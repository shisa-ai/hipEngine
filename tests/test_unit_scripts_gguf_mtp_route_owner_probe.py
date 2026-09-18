"""Guards for ``scripts/gguf_mtp_route_owner_probe.py``.

The probe is the instrument a published long-context verifier packet cites, so
its two contracts are worth pinning without a GPU: the recorded command must
resolve to a script that lives in the repository (the drift gate reads the
artifact's own ``command``), and the diagnostic arm that forces the scalar
row-wise owner must stay behind its environment switch instead of becoming the
default route the probe reports.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_mtp_route_owner_probe.py"
ARTIFACT = (
    ROOT
    / "benchmarks"
    / "results"
    / "2026-09-18-gfx1151-qwen38-long-context-eager-verifier-staged-chain.json"
)
FORCE_SCALAR_ENV = "HIPENGINE_LC_PROBE_FORCE_SCALAR"


def _tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def test_probe_source_is_repo_relative():
    """A probe that only exists under /tmp cannot back a published row."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "/tmp/" not in source
    assert "/home/lhl" not in source
    assert 'REPO_ROOT = Path(__file__).resolve().parents[1]' in source


def test_forced_scalar_arm_is_env_gated():
    """Forcing the scalar owner changes which route runs, so it is opt-in."""
    tree = _tree()
    guard = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "_PROBE_FORCE_SCALAR_ENV" in ast.dump(node.test)
    ]
    assert guard, "the forced-scalar arm is not behind its environment switch"
    forced = {
        node.targets[0].attr
        for node in ast.walk(guard[0])
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_staged_full_attention_rows_ready"
    }
    assert forced == {"_staged_full_attention_rows_ready"}
    # ... and the switch is the documented environment variable, not a private
    # flag that a re-runner cannot find.
    assert f'"{FORCE_SCALAR_ENV}"' in SCRIPT.read_text(encoding="utf-8")


def test_probe_report_is_self_describing():
    """A probe report has to say whether its own diagnostic arm was active."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"forced_scalar"' in source
    for field in ("staged_chain_total", "scalar_attn_total", "split_plan_total"):
        assert f'"{field}"' in source


def test_probe_passes_the_gate_flags_through():
    """The wrapper must expose the gate's own CLI, not a private subset."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    for flag in ("--model", "--cycle-ends", "--candidate-budgets", "--acceptance-budget"):
        assert flag in completed.stdout


@pytest.mark.parametrize("key", ["command", "command_template"])
def test_recorded_packet_command_names_a_committed_script(key: str):
    """The retained packet must re-run from the repository as published."""
    import json

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    commands = [
        value
        for value in _walk_values(payload, key)
        if isinstance(value, str) and "gguf_mtp" in value
    ]
    assert commands, f"no {key} in the retained packet"
    for command in commands:
        assert "scripts/gguf_mtp_route_owner_probe.py" in command
    assert (ROOT / "scripts" / "gguf_mtp_route_owner_probe.py").is_file()


def _walk_values(node, key: str):
    """Every value stored under ``key`` anywhere in a nested artifact."""
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                yield value
            yield from _walk_values(value, key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_values(item, key)
