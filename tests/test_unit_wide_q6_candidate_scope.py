"""The W1 wide-Q6 shared4 verifier candidate is retired, not silently selectable."""

from hipengine.kernels.backends import backend_package_capability
from hipengine.runtime import gguf_linear


def test_wide_q6_candidate_selection_is_retired(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_WIDE_Q6_SHARED4", "1")
    for name in (
        "target_verifier_wide_q6_shared4_session",
        "target_verifier_wide_q6_shared4_leaf_session",
        "target_verifier_wide_q6_shared4_policy_enabled",
        "_target_verifier_wide_q6_shared4_variant",
    ):
        assert not hasattr(gguf_linear, name)
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS", None
    ) is None
