"""Reject invalid qualification provenance before model/GPU setup."""

from types import SimpleNamespace

import pytest

from scripts import qwen4exp_layer2_profile_gate as gate


def test_dirty_source_fails_before_model_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "_git_metadata",
                        lambda path: {"tracked_clean": False})
    args = SimpleNamespace(model_root=tmp_path)
    with pytest.raises(gate.GateError, match="clean committed"):
        gate.run(args, command=[])
