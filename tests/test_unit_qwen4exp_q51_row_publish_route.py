import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, q51_fold_pair_expected_calls


def test_parent_geometry_capability(monkeypatch):
    key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                    "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out")
    select = runner._qwen4_exp_q51_row_publish_key
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH", raising=False)
    assert select(key, rows=1024, in_features=640) == key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH", "1")
    assert "row_publish" in select(key, rows=1024, in_features=640).variant
    assert select(key, rows=511, in_features=640) == key
    assert select(key, rows=1024, in_features=256) == key
    other = KernelKey(key.backend, key.layer, key.quant, "strict")
    assert select(other, rows=1024, in_features=640) == other
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert select(key, rows=1024, in_features=640) == key


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_one_flag(mode, value):
    env = {"other": "keep"}
    _apply_mode(mode, route_package="q51-row-publish", environment=env)
    assert env == {"other": "keep", "HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH": value}


@pytest.mark.parametrize("tokens,expected", [(512,25),(1024,25),(4096,100),(511,0)])
def test_current_chunk_counts(tokens, expected):
    assert q51_fold_pair_expected_calls(tokens, 1024) == expected
