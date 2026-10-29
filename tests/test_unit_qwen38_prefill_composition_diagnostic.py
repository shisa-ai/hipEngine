"""The prefill composition diagnostic changes only its owned sessions."""
from types import SimpleNamespace

import pytest

from scripts.qwen38_prefill_composition_diagnostic import MODES, _configure_mode_sessions


@pytest.mark.parametrize("mode,staging,mmq", (("no_staging", False, True), ("no_mmq", True, False), ("neither", False, False)))
def test_prefill_opt_outs_are_session_local(mode, staging, mmq):
    sessions = [SimpleNamespace(use_q6_integer_mmq=True, use_prefill_f16_staging=True)
                for _ in range(3)]
    unrelated = SimpleNamespace(use_q6_integer_mmq=True, use_prefill_f16_staging=True)
    _configure_mode_sessions(sessions, mode)
    assert [session.use_q6_integer_mmq for session in sessions] == [mmq] * 3
    assert [session.use_prefill_f16_staging for session in sessions] == [staging] * 3
    assert unrelated.use_q6_integer_mmq is True
    assert unrelated.use_prefill_f16_staging is True
    assert mode in MODES


def test_q6_mmq_diagnostic_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown prefill composition mode"):
        _configure_mode_sessions([], "unknown")
