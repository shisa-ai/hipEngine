import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_parent_and_geometry(monkeypatch):
    key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv",
                  "strict_h256_page256_wave_rows_spans")
    select=runner._qwen4_exp_qsa_head_pair_key
    monkeypatch.setattr(runner,"is_registered",lambda key:True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR",raising=False)
    assert select(key,query_heads=24,kv_heads=2)==key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR","1")
    assert select(key,query_heads=24,kv_heads=2).variant=="strict_h256_head_pair_rows_spans"
    assert select(key,query_heads=6,kv_heads=2)==key
    other=KernelKey(key.backend,key.layer,key.quant,"strict_h256_wave_rows_spans")
    assert select(other,query_heads=24,kv_heads=2)==other
    monkeypatch.setattr(runner,"is_registered",lambda key:False)
    assert select(key,query_heads=24,kv_heads=2)==key


@pytest.mark.parametrize("mode,value",[("before","0"),("after","1")])
def test_one_flag(mode,value):
    env={"other":"keep"}
    _apply_mode(mode,route_package="qsa-head-pair",environment=env)
    assert env=={"other":"keep","HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR":value}


def test_quad_mode_explicit_and_parent_only(monkeypatch):
    key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv",
                  "strict_h256_page256_wave_rows_spans")
    monkeypatch.setattr(runner,"is_registered",lambda key:True)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR","quad")
    assert runner._qwen4_exp_qsa_head_pair_key(key,query_heads=24,kv_heads=2).variant=="strict_h256_head_quad_rows_spans"
    assert runner._qwen4_exp_qsa_head_pair_key(key,query_heads=12,kv_heads=2)==key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR","invalid")
    assert runner._qwen4_exp_qsa_head_pair_key(key,query_heads=24,kv_heads=2)==key


def test_quad_harness_modes():
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode
    for enabled in ("0","1"):
        env={}
        _apply_mode("after" if enabled=="1" else "before",route_package="qsa-head-quad",environment=env)
        assert env=={"HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR":"quad" if enabled=="1" else "0"}
        other={}
        apply_state_gate_mode(None,"qsa-head-quad",enabled,"flag",environment=other)
        assert other["flag"]==("quad" if enabled=="1" else "0")
