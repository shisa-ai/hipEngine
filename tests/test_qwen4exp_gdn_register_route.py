import pytest

from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_register_prefill_route_requires_flag_shape_and_registration(monkeypatch):
    select = runner.qwen4_exp_gdn_register_prefill_selected
    shape = {
        "backend": "hip_gfx1151",
        "rows": 512,
        "num_k_heads": 16,
        "num_v_heads": 48,
        "head_dim": 128,
    }
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL", raising=False)
    assert not select(**shape)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL", "1")
    assert select(**shape)
    for override in ({"rows": 1}, {"num_k_heads": 8}, {"num_v_heads": 32}, {"head_dim": 64}):
        assert not select(**(shape | override))
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert not select(**shape)


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_register_ab_changes_only_its_flag(mode, value):
    environment = {"other": "unchanged"}
    _apply_mode(mode, environment=environment, route_package="gdn-register")
    assert environment == {
        "other": "unchanged",
        "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL": value,
    }
