"""Q5_1 pair2 admission belongs to the runner, not the process."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

ENV = "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL"


@pytest.mark.parametrize("rows", [17, 63, 64, 65, 131])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("stale", ["0", "1"])
def test_pair_admission_uses_explicit_publication(monkeypatch, rows, published, stale):
    monkeypatch.setenv(ENV, stale)
    monkeypatch.setattr(runner, "is_registered", lambda key: key.backend == "present")
    selected = runner.qwen4_exp_q51_pair_prefill_selected
    assert selected("present", "gguf_q5_1", rows=rows, in_features=640,
                    published=published) is (published and rows >= 64)
    assert not selected("missing", "gguf_q5_1", rows=rows, in_features=640,
                        published=published)
    assert not selected("present", "gguf_q5_1", rows=rows, in_features=8192,
                        published=published)


def test_pair_scope_restores_after_nested_failure():
    owner = SimpleNamespace(q51_pair_prefill=True)
    other = SimpleNamespace(q51_pair_prefill=True)
    with pytest.raises(RuntimeError, match="failed"):
        with runner.q51_pair_prefill(owner, False):
            assert owner.q51_pair_prefill is False
            with runner.q51_pair_prefill(owner, True):
                assert owner.q51_pair_prefill is True
            assert owner.q51_pair_prefill is False
            assert other.q51_pair_prefill is True
            raise RuntimeError("failed")
    assert owner.q51_pair_prefill is True


@pytest.mark.parametrize("published", [False, True])
def test_pair_choice_survives_late_runner_construction(published):
    child = _inherit_lane_decisions(
        SimpleNamespace(q51_pair_prefill=published), SimpleNamespace())
    assert child.q51_pair_prefill is published


def test_disabled_pair_keeps_m1_parent_even_when_fold_children_are_on(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "64")
    key = runner._qwen4_exp_q51_down_key(
        backend="test", quant="gguf_q5_1", rows=512, in_features=640,
        q51_down_m1=True, q51_pair_prefill=False, q51_fold128_prefill=True,
        q51_fold_pair_prefill=True, q51_register_cache=True, q51_row_publish=True)
    assert key.variant == (
        "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out")
