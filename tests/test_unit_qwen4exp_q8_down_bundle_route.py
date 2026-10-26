from types import SimpleNamespace

import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


BUNDLE = "selected_grouped_row4_bundle_gemv_bf16_bf16_out"


def _row4_key():
    return KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                     "selected_grouped_row4_gemv_bf16_bf16_out")


def test_bundle_requires_row4_scope(monkeypatch):
    select = runner._qwen4_exp_q8_down_bundle_key
    key = _row4_key()
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    assert select(key, rows=512, published=False) == key
    assert select(key, rows=512, published=True).variant == BUNDLE
    assert select(key, rows=511, published=True) == key
    other = KernelKey(key.backend, key.layer, key.quant,
                      "selected_grouped_gemv_bf16_bf16_out")
    assert select(other, rows=512, published=True) == other
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert select(key, rows=512, published=True) == key


def test_the_scope_binds_and_restores_the_published_value():
    owner = SimpleNamespace(q8_down_bundle_prefill=False)
    with runner.q8_down_bundle_prefill(owner, True) as prior:
        assert owner.q8_down_bundle_prefill is True
        assert prior is False
    assert owner.q8_down_bundle_prefill is False
    with runner.q8_down_bundle_prefill(owner):
        assert owner.q8_down_bundle_prefill is False
    assert owner.q8_down_bundle_prefill is False


def test_the_scope_restores_on_an_exception():
    owner = SimpleNamespace(q8_down_bundle_prefill=True)
    with pytest.raises(RuntimeError):
        with runner.q8_down_bundle_prefill(owner, False):
            raise RuntimeError("measured prefill failed")
    assert owner.q8_down_bundle_prefill is True


@pytest.mark.parametrize("mode", ["before", "after"])
def test_the_arm_writes_no_process_key(mode):
    env = {"other": "unchanged"}
    _apply_mode(mode, environment=env, route_package="q8-down-bundle")
    assert env == {"other": "unchanged"}
