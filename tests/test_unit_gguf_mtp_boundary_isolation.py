"""CPU contracts for the potentially destructive MTP context-crossing gate."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tests import _gguf_mtp_boundary_isolation as isolation


def test_crossing_body_is_in_a_child_before_fresh_health(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(isolation.subprocess, "run", run)
    isolation.run_context_crossing_isolated(timeout_seconds=23)
    assert len(calls) == 2
    assert "gate._run_dense_mtp_context_crossing()" in calls[0][0][2]
    assert "gate._run_dense_mtp_fresh_health()" in calls[1][0][2]
    assert all(call[1]["timeout"] == 23 and call[1]["cwd"] == isolation.ROOT for call in calls)


@pytest.mark.parametrize("code", [1, -11, 77])
def test_failed_or_skipped_crossing_never_runs_health(monkeypatch, code):
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=code, stdout="child explanation", stderr="details")
    monkeypatch.setattr(isolation.subprocess, "run", run)
    expected = pytest.skip.Exception if code == 77 else AssertionError
    with pytest.raises(expected):
        isolation.run_context_crossing_isolated()
    assert len(calls) == 1


def test_timed_out_crossing_never_runs_health(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(isolation.subprocess, "run", run)
    with pytest.raises(pytest.fail.Exception, match="child retired"):
        isolation.run_context_crossing_isolated(timeout_seconds=1)
    assert len(calls) == 1


def test_live_node_uses_isolation_instead_of_skipping_original_body():
    path = Path(__file__).with_name("test_live_qwen35_gguf_mtp_e2e.py")
    tree = ast.parse(path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    node = functions["test_dense_q4_k_m_mtp_crosses_1024_via_prelaunch_eager_fallback"]
    assert "run_context_crossing_isolated()" in ast.unparse(node)
    assert not any(isinstance(mark, ast.Call) and ast.unparse(mark.func) == "pytest.mark.skip" for mark in node.decorator_list)
    body = functions["_run_dense_mtp_context_crossing"]
    assert "range(1018)" in ast.unparse(body)
    assert "candidate_budget=3" in ast.unparse(body)
    assert "target.reset()" in ast.unparse(body)
    assert "_DENSE_MODEL" in ast.unparse(body)
    assert "_run_dense_mtp_fresh_health" in functions
