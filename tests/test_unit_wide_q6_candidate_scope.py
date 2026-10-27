"""Request-local ownership of the wide-Q6 diagnostic candidate."""

import pytest

from hipengine.runtime.gguf_linear import (
    target_verifier_wide_q6_shared4_policy_enabled,
    target_verifier_wide_q6_shared4_session,
)


def test_wide_q6_candidate_scope_ignores_retired_environment(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_WIDE_Q6_SHARED4", "1")
    assert not target_verifier_wide_q6_shared4_policy_enabled()
    with target_verifier_wide_q6_shared4_session(True):
        assert target_verifier_wide_q6_shared4_policy_enabled()
        with pytest.raises(RuntimeError):
            with target_verifier_wide_q6_shared4_session(False):
                assert not target_verifier_wide_q6_shared4_policy_enabled()
                raise RuntimeError("diagnostic failed")
        assert target_verifier_wide_q6_shared4_policy_enabled()
    assert not target_verifier_wide_q6_shared4_policy_enabled()
