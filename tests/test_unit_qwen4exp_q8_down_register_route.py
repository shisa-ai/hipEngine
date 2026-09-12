import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_register_down_requires_existing_bundle_and_k640(monkeypatch):
    key=KernelKey("hip_gfx1151","linear","gguf_q8_0","selected_grouped_row4_bundle_gemv_bf16_bf16_out")
    select=runner._qwen4_exp_q8_down_register_key
    monkeypatch.setattr(runner,"is_registered",lambda key:True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER",raising=False)
    assert select(key,rows=1024,in_features=640)==key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER","1")
    assert "register" in select(key,rows=1024,in_features=640).variant
    assert select(key,rows=511,in_features=640)==key
    assert select(key,rows=1024,in_features=512)==key
    other=KernelKey(key.backend,key.layer,key.quant,"selected_gemv_bf16_bf16_out")
    assert select(other,rows=1024,in_features=640)==other
    monkeypatch.setattr(runner,"is_registered",lambda key:False)
    assert select(key,rows=1024,in_features=640)==key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_register_down_one_flag(mode,value):
    env={"other":"keep"}
    _apply_mode(mode,environment=env,route_package="q8-down-register")
    assert env=={"other":"keep","HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER":value}


@pytest.mark.parametrize("tokens,chunk,expected",[(512,1024,5),(1024,1024,5),
                                                (4096,1024,20),(511,1024,0),(1536,1024,10)])
def test_register_down_counts(tokens,chunk,expected):
    from scripts.qwen4exp_halo_box_campaign_ab import q8_down_register_expected_calls
    assert q8_down_register_expected_calls(tokens,chunk)==expected
