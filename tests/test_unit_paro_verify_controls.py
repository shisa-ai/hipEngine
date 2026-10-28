"""PARO verifier controls preserve defaults and explicit reference paths."""
from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
from hipengine.speculative.interfaces import TargetVerifyBatch


@pytest.mark.parametrize("method,env", [
    ("_verify_chain_linear_tloop_enabled", "HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP"),
    ("_verify_gpu_accept_enabled", "HIPENGINE_VERIFY_GPU_ACCEPT"),
])
@pytest.mark.parametrize("value,expected", [
    (None, True), ("", True), ("1", True), ("validate", True),
    ("0", False), ("FALSE", False), (" no ", False), ("off", False),
])
def test_verifier_controls_default_on_and_preserve_explicit_reference(monkeypatch, method, env, value, expected):
    if value is None:
        monkeypatch.delenv(env, raising=False)
    else:
        monkeypatch.setenv(env, value)
    session = object.__new__(Qwen35ParoResidentSession)
    assert getattr(session, method)() is expected


def _chain(depth):
    rows = depth + 1
    return TargetVerifyBatch(
        request_ids=(7,), tokens=tuple(range(rows)), positions=tuple(range(5, 5 + rows)),
        row_to_request=(7,) * rows, parent_rows=(-1, *range(depth)), root_rows=(0,),
        candidate_rows=tuple(range(1, rows)), draft_depths=tuple(range(rows)),
        active_mask=(True,) * rows,
    )


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_chain_tloop_selection_is_topology_owned_not_historical_b1_gate(monkeypatch, depth):
    monkeypatch.delenv("HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP", raising=False)
    session = object.__new__(Qwen35ParoResidentSession)
    batch = _chain(depth)
    assert session._should_use_chain_tloop_linear_verify(batch, rows=batch.rows, graph_mode="off")
    assert not session._should_use_chain_tloop_linear_verify(
        replace(batch, mode="verify_tree"), rows=batch.rows, graph_mode="off",
    )
    assert not session._should_use_chain_tloop_linear_verify(
        replace(batch, active_mask=(True,) * depth + (False,)), rows=batch.rows, graph_mode="off",
    )
    monkeypatch.setenv("HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP", "0")
    assert not session._should_use_chain_tloop_linear_verify(batch, rows=batch.rows, graph_mode="off")


def test_branching_parent_layout_keeps_unfused_verifier_fallback(monkeypatch):
    monkeypatch.delenv("HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP", raising=False)
    session = object.__new__(Qwen35ParoResidentSession)
    batch = replace(_chain(3), parent_rows=(-1, 0, 0, 1))
    assert not session._should_use_chain_tloop_linear_verify(batch, rows=batch.rows, graph_mode="off")
