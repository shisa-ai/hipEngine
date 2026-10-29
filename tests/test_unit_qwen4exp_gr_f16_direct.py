"""Direct GR launches keep geometry guards and execution-local diagnostics."""
import contextvars
import os

import pytest
from hipengine.runtime import qwen4_exp_runner as runner


@pytest.mark.parametrize("stale", ["0", "1", "false"])
@pytest.mark.parametrize("rows,rank,expected", [(1, 320, False), (256, 320, False),
                                               (257, 320, True), (566, 320, True),
                                               (4096, 321, False)])
def test_direct_selection_ignores_environment(monkeypatch, stale, rows, rank, expected):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_F16_DIRECT", stale)
    assert runner._qwen4_exp_gr_f16_direct_enabled(rows, rank) is expected


def test_direct_diagnostic_restores_on_exception(monkeypatch):
    key = "HIPENGINE_QWEN4_EXP_GR_F16_DIRECT"
    monkeypatch.setenv(key, "stale")
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        with runner.gr_f16_direct(False):
            assert not runner._qwen4_exp_gr_f16_direct_enabled(566, 320)
            assert contextvars.Context().run(runner._qwen4_exp_gr_f16_direct_enabled, 566, 320)
            with runner.gr_f16_direct(True):
                assert runner._qwen4_exp_gr_f16_direct_enabled(566, 320)
            assert not runner._qwen4_exp_gr_f16_direct_enabled(566, 320)
            raise RuntimeError("diagnostic failed")
    assert runner._qwen4_exp_gr_f16_direct_enabled(566, 320)
    assert os.environ[key] == "stale"


def test_historical_arms_use_explicit_direct_control():
    from scripts.qwen4exp_layer2_profile_gate import _f16_route_isolation_candidates
    for arm in _f16_route_isolation_candidates().values():
        assert arm.gr_f16_direct is False
        assert "HIPENGINE_QWEN4_EXP_GR_F16_DIRECT" not in arm.environment
