import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_register_cache_requires_folded_large_k640(monkeypatch):
    key = KernelKey("hip_gfx1151","moe_linear","gguf_q5_1",
                    "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out")
    select = runner._qwen4_exp_q51_register_cache_key
    monkeypatch.setattr(runner,"is_registered",lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE",raising=False)
    assert select(key,rows=512,in_features=640) == key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE","1")
    assert "register_cache" in select(key,rows=512,in_features=640).variant
    assert select(key,rows=511,in_features=640) == key
    assert select(key,rows=512,in_features=4096) == key
    other = KernelKey(key.backend,key.layer,key.quant,"selected_grouped_prefill_pair2_bf16_bf16_out")
    assert select(other,rows=512,in_features=640) == other
    monkeypatch.setattr(runner,"is_registered",lambda key: False)
    assert select(key,rows=512,in_features=640) == key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_register_cache_ab_one_flag(mode,value):
    env = {"other":"keep"}
    _apply_mode(mode,environment=env,route_package="q51-register-cache")
    assert env == {"other":"keep","HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE":value}
