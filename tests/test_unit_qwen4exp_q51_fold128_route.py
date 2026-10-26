import pytest
from hipengine.kernels import registry
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, q51_fold128_expected_calls
from tests._qwen4exp_expert_down_fixtures import expert_down_runner


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


@pytest.mark.parametrize("prompt,chunk,calls", [(512,512,43),(4096,512,344),
    (513,512,43),(576,512,86),(63,512,0),(64,512,43)])
def test_fold128_expected_calls(monkeypatch, prompt,chunk,calls):
    # The count is the loaded artifact's Q5_1 down layers times the chunks that
    # clear the leaf's own rows>=64 guard - never a layer constant.
    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    assert q51_fold128_expected_calls(expert_down_runner(), prompt,chunk) == calls


def test_a_chunk_below_the_row_guard_is_a_named_failure(monkeypatch):
    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    with pytest.raises(ValueError, match="rows>=64.*chunk is 32"):
        q51_fold128_expected_calls(expert_down_runner(), 512, 32)
