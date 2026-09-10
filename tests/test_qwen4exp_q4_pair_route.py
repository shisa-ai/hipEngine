import pytest

from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_q4_pair_route_requires_flag_shape_and_registration(monkeypatch):
    select = runner.qwen4_exp_q4_pair_prefill_selected
    shape = {"backend": "hip_gfx1151", "quant": "gguf_q4_k", "rows": 512, "in_features": 2560}
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL", raising=False)
    assert not select(**shape)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL", "1")
    assert select(**shape)
    assert select(**(shape | {"rows": 64, "in_features": 4096}))
    for override in ({"rows": 63}, {"in_features": 4352}, {"in_features": 0}, {"in_features": 257}):
        assert not select(**(shape | override))
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert not select(**shape)


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_q4_pair_ab_changes_only_its_flag(mode, value):
    environment = {"other": "unchanged"}
    _apply_mode(mode, environment=environment, route_package="q4-pair")
    assert environment == {
        "other": "unchanged",
        "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL": value,
    }
