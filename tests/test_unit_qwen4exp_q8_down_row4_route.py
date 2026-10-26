from types import SimpleNamespace

import pytest

from hipengine.kernels import registry
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, q8_down_row4_expected_calls
from tests._qwen4exp_expert_down_fixtures import expert_down_runner


ROW4 = "selected_grouped_row4_gemv_bf16_bf16_out"


def _sorted_parent():
    return KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                     "selected_grouped_gemv_bf16_bf16_out")


def test_q8_down_row4_scope(monkeypatch):
    parent = _sorted_parent()
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    select = runner._qwen4_exp_q8_down_row4_key
    assert select(parent, rows=512, published=False) == parent
    assert select(parent, rows=512, published=True).variant == ROW4
    for rows in (1, 64, 511):
        assert select(parent, rows=rows, published=True) == parent
    other = KernelKey(parent.backend, parent.layer, parent.quant,
                      "selected_gemv_bf16_bf16_out")
    assert select(other, rows=512, published=True) == other
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert select(parent, rows=512, published=True) == parent


def test_the_scope_binds_and_restores_the_published_value():
    owner = SimpleNamespace(q8_down_row4_prefill=False)
    with runner.q8_down_row4_prefill(owner, True) as prior:
        assert owner.q8_down_row4_prefill is True
        assert prior is False
    assert owner.q8_down_row4_prefill is False
    owner.q8_down_row4_prefill = True
    with runner.q8_down_row4_prefill(owner):
        assert owner.q8_down_row4_prefill is True
    assert owner.q8_down_row4_prefill is True


def test_the_scope_restores_on_an_exception():
    owner = SimpleNamespace(q8_down_row4_prefill=False)
    with pytest.raises(RuntimeError):
        with runner.q8_down_row4_prefill(owner, True):
            raise RuntimeError("measured prefill failed")
    assert owner.q8_down_row4_prefill is False


@pytest.mark.parametrize("mode", ["before", "after"])
def test_the_arm_writes_no_process_key(mode):
    env = {"other": "unchanged"}
    _apply_mode(mode, environment=env, route_package="q8-down-row4")
    assert env == {"other": "unchanged"}


@pytest.mark.parametrize("prompt,chunk,calls", [(512,512,5), (4096,512,40),
    (513,512,5), (511,512,0), (1024,1024,5)])
def test_q8_down_expected_calls(monkeypatch, prompt, chunk, calls):
    # Five Q8_0 expert-down layers in the shipped artifact, once per chunk that
    # clears the row4 leaf's rows>=512 guard.
    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    assert q8_down_row4_expected_calls(expert_down_runner(), prompt, chunk) == calls


def test_a_chunk_below_the_row_guard_is_a_named_failure(monkeypatch):
    # A plan whose chunk can never clear the leaf's guard measures nothing, so it
    # must not report zero calls as a measured zero.
    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    with pytest.raises(ValueError, match="rows>=512.*chunk is 256"):
        q8_down_row4_expected_calls(expert_down_runner(), 4096, 256)
