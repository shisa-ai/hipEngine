import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode,q5k_bundle_expected_calls


def test_q5k_bundle_scope(monkeypatch):
    select=runner._qwen4_exp_q5k_bundle_key
    key=KernelKey("hip_gfx1151","linear","gguf_q5_k","selected_grouped_row4_gemv_bf16_bf16_out")
    monkeypatch.setattr(runner,"is_registered",lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q5K_BUNDLE_PREFILL",raising=False)
    assert select(key,rows=512)==key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q5K_BUNDLE_PREFILL","1")
    assert select(key,rows=64).variant=="selected_grouped_row4_bundle_gemv_bf16_bf16_out"
    assert select(key,rows=63)==key
    other=KernelKey(key.backend,key.layer,key.quant,"selected_gemv_bf16_bf16_out")
    assert select(other,rows=512)==other
    monkeypatch.setattr(runner,"is_registered",lambda key: False)
    assert select(key,rows=512)==key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_q5k_bundle_ab_one_flag(mode,value):
    env={"other":"unchanged"}
    _apply_mode(mode,environment=env,route_package="q5k-bundle")
    assert env=={"other":"unchanged","HIPENGINE_QWEN4_EXP_Q5K_BUNDLE_PREFILL":value}


@pytest.mark.parametrize("prompt,chunk,calls",[(64,512,2),(512,512,2),(4096,512,16),
    (576,512,4),(63,512,0),(512,32,0)])
def test_q5k_bundle_counts(prompt,chunk,calls):
    assert q5k_bundle_expected_calls(prompt,chunk)==calls
