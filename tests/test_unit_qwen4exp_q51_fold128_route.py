from types import SimpleNamespace

import pytest
from hipengine.kernels import registry
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, q51_fold128_expected_calls
from tests._qwen4exp_expert_down_fixtures import expert_down_runner


FOLD128 = "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"


def _pair2_key():
    return KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                     "selected_grouped_prefill_pair2_bf16_bf16_out")


def test_fold128_is_the_runners_published_decision(monkeypatch):
    key = _pair2_key()
    select = runner._qwen4_exp_q51_fold128_key
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    assert select(key, rows=64, published=False) == key
    assert select(key, rows=64, published=True).variant == FOLD128
    assert select(key, rows=63, published=True) == key
    other = KernelKey(key.backend, key.layer, key.quant, "other")
    assert select(other, rows=64, published=True) == other
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert select(key, rows=64, published=True) == key


def test_the_scope_binds_and_restores_the_published_value():
    owner = SimpleNamespace(q51_fold128_prefill=False)
    with runner.q51_fold128_prefill(owner, True) as prior:
        assert owner.q51_fold128_prefill is True
        assert prior is False
    assert owner.q51_fold128_prefill is False
    # Leaving the argument unset keeps whatever the plan published.
    owner.q51_fold128_prefill = True
    with runner.q51_fold128_prefill(owner):
        assert owner.q51_fold128_prefill is True
    assert owner.q51_fold128_prefill is True


def test_the_scope_restores_on_an_exception():
    owner = SimpleNamespace(q51_fold128_prefill=False)
    with pytest.raises(RuntimeError):
        with runner.q51_fold128_prefill(owner, True):
            raise RuntimeError("measured prefill failed")
    assert owner.q51_fold128_prefill is False


@pytest.mark.parametrize("mode", ["before", "after"])
def test_the_arm_writes_no_process_key(mode):
    env = {"other": "unchanged"}
    _apply_mode(mode, environment=env, route_package="q51-fold128")
    assert env == {"other": "unchanged"}


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
