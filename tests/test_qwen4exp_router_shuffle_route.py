import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_router_shuffle_requires_parent_and_prefill(monkeypatch):
    key = KernelKey("hip_gfx1151","router_logits","f32","f32_hidden_token_tile4_dense_exact")
    select = runner._qwen4_exp_router_shuffle_key
    monkeypatch.setattr(runner,"is_registered",lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_ROUTER_SHUFFLE",raising=False)
    assert select(key,rows=1024) == key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_ROUTER_SHUFFLE","1")
    assert select(key,rows=1) == key
    assert select(key,rows=2).variant == "f32_hidden_token_tile4_shuffle_exact"
    other = KernelKey(key.backend,key.layer,key.quant,"other")
    assert select(other,rows=1024) == other
    monkeypatch.setattr(runner,"is_registered",lambda key: False)
    assert select(key,rows=1024) == key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_router_ab_flag(mode,value):
    env = {"other":"keep"}
    _apply_mode(mode,environment=env,route_package="router-shuffle")
    assert env == {"other":"keep","HIPENGINE_QWEN4_EXP_ROUTER_SHUFFLE":value}


@pytest.mark.parametrize("tokens,chunk,calls",[(512,1024,48),(1024,1024,48),
                                             (4096,1024,192),(1025,1024,48),(1,1024,0)])
def test_router_call_counts(tokens,chunk,calls):
    from scripts.qwen4exp_halo_box_campaign_ab import router_shuffle_expected_calls
    assert router_shuffle_expected_calls(tokens,chunk)==calls
