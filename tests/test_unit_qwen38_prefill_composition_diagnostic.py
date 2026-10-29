"""The prefill composition diagnostic changes only its owned sessions."""
from types import SimpleNamespace

import pytest

from scripts.qwen38_prefill_composition_diagnostic import MODES, _configure_mode_sessions


@pytest.mark.parametrize("mode,expected", (("no_staging", True), ("no_mmq", False), ("neither", False)))
def test_q6_mmq_opt_out_is_session_local(mode, expected):
    sessions = [SimpleNamespace(use_q6_integer_mmq=True) for _ in range(3)]
    unrelated = SimpleNamespace(use_q6_integer_mmq=True)
    _configure_mode_sessions(sessions, mode)
    assert [session.use_q6_integer_mmq for session in sessions] == [expected] * 3
    assert unrelated.use_q6_integer_mmq is True
    assert all("Q6_INTEGER_MMQ" not in name for name in MODES[mode])


def test_q6_mmq_diagnostic_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown prefill composition mode"):
        _configure_mode_sessions([], "unknown")
