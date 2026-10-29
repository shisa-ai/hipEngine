"""Register-state GDN selection is owned by the measured runner."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions


@pytest.mark.parametrize("rows", [1, 2, 17, 65])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("stale", ["0", "1"])
def test_register_choice_ignores_process_state(monkeypatch, rows, published, stale):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL", stale)
    monkeypatch.setattr(runtime, "is_registered", lambda key: key.backend == "present")
    shape = dict(backend="present", rows=rows, num_k_heads=16,
                 num_v_heads=48, head_dim=128, published=published)
    select = runtime.qwen4_exp_gdn_register_prefill_selected
    assert select(**shape) is (published and rows >= 2)
    for override in ({"backend": "missing"}, {"num_k_heads": 8},
                     {"num_v_heads": 32}, {"head_dim": 64}):
        assert not select(**(shape | override))


def test_register_scope_restores_nested_failure():
    owner = SimpleNamespace(gdn_register_prefill=True)
    with pytest.raises(RuntimeError, match="failed"):
        with runtime.gdn_register_prefill(owner, False):
            with runtime.gdn_register_prefill(owner, True):
                assert owner.gdn_register_prefill is True
            assert owner.gdn_register_prefill is False
            raise RuntimeError("failed")
    assert owner.gdn_register_prefill is True


@pytest.mark.parametrize("published", [False, True])
def test_register_choice_reaches_late_clone(published):
    child = _inherit_lane_decisions(
        SimpleNamespace(gdn_register_prefill=published), SimpleNamespace())
    assert child.gdn_register_prefill is published


@pytest.mark.parametrize("mode, enabled", [("before", False), ("after", True)])
def test_campaign_scope_isolates_and_restores_serial_owner(mode, enabled):
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope
    owner = SimpleNamespace(gdn_register_prefill=True,
                            gdn_prefill_variant="qwen4exp_sigmoid_wave_norm_prefill",
                            gdn_peer_prefill_layers="all")
    prior = vars(owner).copy()
    env = {"HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1"}
    with pytest.raises(RuntimeError, match="failed"):
        with route_scope("gdn-register", mode, owner, environment=env):
            assert owner.gdn_register_prefill is enabled
            assert owner.gdn_prefill_variant is None
            assert owner.gdn_peer_prefill_layers is None
            assert env["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "0"
            raise RuntimeError("failed")
    assert vars(owner) == prior
    assert env == {"HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1"}


def test_isolated_candidate_requests_register_owner():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    candidate = CANDIDATES["gdn_isolated"]
    assert candidate.gdn_register_prefill is True
    assert "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL" not in candidate.environment


@pytest.mark.parametrize("arm, expected", [
    ("bound", True), ("all_flags_strict", False), ("gdn_flags_strict", False),
    ("base_plus_gdn_flags", False), ("strict_plus_matrix", False),
    ("strict_plus_gdn", True), ("strict_plus_gdn_serial", True),
    ("strict_plus_gdn_multi", True),
])
def test_journey_register_arm_preserves_profile_intent(arm, expected):
    from scripts.qwen4exp_journey_localize import register_arm_enabled
    assert register_arm_enabled(arm, True) is expected
