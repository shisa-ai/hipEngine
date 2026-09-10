import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_bundle_requires_row4_scope(monkeypatch):
    select = runner._qwen4_exp_q8_down_bundle_key
    key = KernelKey("hip_gfx1151","linear","gguf_q8_0",
                    "selected_grouped_row4_gemv_bf16_bf16_out")
    monkeypatch.setattr(runner,"is_registered",lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL",raising=False)
    assert select(key,rows=512) == key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL","1")
    assert select(key,rows=512).variant == "selected_grouped_row4_bundle_gemv_bf16_bf16_out"
    assert select(key,rows=511) == key
    other = KernelKey(key.backend,key.layer,key.quant,"selected_grouped_gemv_bf16_bf16_out")
    assert select(other,rows=512) == other
    monkeypatch.setattr(runner,"is_registered",lambda key: False)
    assert select(key,rows=512) == key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_bundle_ab_only_one_flag(mode,value):
    env={"other":"unchanged"}
    _apply_mode(mode,environment=env,route_package="q8-down-bundle")
    assert env == {"other":"unchanged","HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL":value}
