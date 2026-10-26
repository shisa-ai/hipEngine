from types import SimpleNamespace

import pytest
from hipengine.kernels import registry
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, q51_fold_pair_expected_calls
from tests._qwen4exp_expert_down_fixtures import expert_down_runner


FOLD_PAIR = "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out"


def _fold128_key():
    return KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                     "selected_grouped_prefill_pair2_fold128_bf16_bf16_out")


def test_fold_pair_requires_large_folded_scope(monkeypatch):
    select = runner._qwen4_exp_q51_fold_pair_key
    key = _fold128_key()
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    assert select(key, rows=512, published=False) == key
    assert select(key, rows=512, published=True).variant == FOLD_PAIR
    for rows in (1, 64, 511):
        assert select(key, rows=rows, published=True) == key
    other = KernelKey(key.backend, key.layer, key.quant,
                      "selected_grouped_prefill_pair2_bf16_bf16_out")
    assert select(other, rows=512, published=True) == other
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert select(key, rows=512, published=True) == key


def test_the_scope_binds_and_restores_the_published_value():
    owner = SimpleNamespace(q51_fold_pair_prefill=True)
    with runner.q51_fold_pair_prefill(owner, False) as prior:
        assert owner.q51_fold_pair_prefill is False
        assert prior is True
    assert owner.q51_fold_pair_prefill is True
    with runner.q51_fold_pair_prefill(owner):
        assert owner.q51_fold_pair_prefill is True
    assert owner.q51_fold_pair_prefill is True


def test_the_scope_restores_on_an_exception():
    owner = SimpleNamespace(q51_fold_pair_prefill=True)
    with pytest.raises(RuntimeError):
        with runner.q51_fold_pair_prefill(owner, False):
            raise RuntimeError("measured prefill failed")
    assert owner.q51_fold_pair_prefill is True


@pytest.mark.parametrize("mode", ["before", "after"])
def test_the_arm_writes_no_process_key(mode):
    env = {"other": "unchanged"}
    _apply_mode(mode, environment=env, route_package="q51-fold-pair")
    assert env == {"other": "unchanged"}


@pytest.mark.parametrize("prompt,chunk,calls",[(512,512,43),(4096,512,344),
    (576,512,43),(511,512,0),(1024,1024,43)])
def test_large_chunk_counts(monkeypatch, prompt,chunk,calls):
    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    assert q51_fold_pair_expected_calls(expert_down_runner(),prompt,chunk) == calls


def test_a_chunk_below_the_row_guard_is_a_named_failure(monkeypatch):
    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    with pytest.raises(ValueError, match="rows>=512.*chunk is 256"):
        q51_fold_pair_expected_calls(expert_down_runner(), 4096, 256)
