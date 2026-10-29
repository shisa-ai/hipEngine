"""Q5_K row4 admission is owned by the measured runner, not the process."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as module
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, route_scope

LEGACY_ENV = "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"


def _weights(quant="gguf_q5_k"):
    weight = SimpleNamespace(spec=SimpleNamespace(quant_key=quant))
    return {"expert_gate": weight, "expert_up": weight}


@pytest.mark.parametrize("rows", [1, 63, 64, 65, 131, 512])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("stale", ["0", "1"])
def test_dispatch_uses_published_choice_with_unchanged_guards(monkeypatch, rows, published, stale):
    monkeypatch.setenv(LEGACY_ENV, stale)
    monkeypatch.setattr(module, "is_registered", lambda key: True)
    assert module.qwen4_exp_grouped_row4_prefill_selected(
        _weights(), rows=rows, backend="fake", published=published,
    ) is (published and rows >= 64)


def test_missing_either_registration_keeps_the_parent(monkeypatch):
    monkeypatch.setattr(module, "is_registered", lambda key: key.quant == "gguf_q5_k")
    weights = _weights()
    weights["expert_up"] = _weights("missing")["expert_up"]
    assert not module.qwen4_exp_grouped_row4_prefill_selected(
        weights, rows=131, backend="fake", published=True)


def test_nested_scopes_restore_each_owner_after_failure():
    owner = SimpleNamespace(q5k_row4_prefill=True)
    neighbor = SimpleNamespace(q5k_row4_prefill=False)
    with pytest.raises(RuntimeError, match="test failure"):
        with module.q5k_row4_prefill(owner, False):
            assert owner.q5k_row4_prefill is False
            with module.q5k_row4_prefill(owner, True):
                assert owner.q5k_row4_prefill is True
                assert neighbor.q5k_row4_prefill is False
            assert owner.q5k_row4_prefill is False
            raise RuntimeError("test failure")
    assert owner.q5k_row4_prefill is True
    assert neighbor.q5k_row4_prefill is False


@pytest.mark.parametrize("mode", ["before", "after"])
def test_campaign_scope_binds_runner_and_leaves_legacy_env_alone(mode):
    owner = SimpleNamespace(q5k_row4_prefill=True)
    env = {LEGACY_ENV: "stale"}
    _apply_mode(mode, environment=env, route_package="q5k-row4")
    assert env == {LEGACY_ENV: "stale"}
    with route_scope("q5k-row4", mode, owner, environment=env):
        assert owner.q5k_row4_prefill is (mode == "after")
        assert env == {LEGACY_ENV: "stale"}
    assert owner.q5k_row4_prefill is True


def test_campaign_scope_requires_owner():
    with pytest.raises(ValueError, match="runner"):
        with route_scope("q5k-row4", "after"):
            pass
