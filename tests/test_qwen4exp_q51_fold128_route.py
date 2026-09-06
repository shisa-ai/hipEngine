import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, q51_fold128_expected_calls


def test_fold128_scoped_to_pair_prefill(monkeypatch):
    key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                    "selected_grouped_prefill_pair2_bf16_bf16_out")
    select = runner._qwen4_exp_q51_fold128_key
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL", raising=False)
    assert select(key, rows=512) == key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL", "1")
    assert select(key, rows=64).variant == "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"
    assert select(key, rows=63) == key
    other = KernelKey(key.backend, key.layer, key.quant, "other")
    assert select(other, rows=512) == other
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert select(key, rows=512) == key


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_fold128_ab_changes_one_flag(mode, value):
    env = {"other": "unchanged"}
    _apply_mode(mode, environment=env, route_package="q51-fold128")
    assert env == {"other": "unchanged", "HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL": value}


@pytest.mark.parametrize("prompt,chunk,calls", [(512,512,25),(4096,512,200),
    (513,512,25),(576,512,50),(63,512,0),(64,512,25),(512,32,0)])
def test_fold128_expected_calls(prompt,chunk,calls):
    assert q51_fold128_expected_calls(prompt,chunk) == calls
