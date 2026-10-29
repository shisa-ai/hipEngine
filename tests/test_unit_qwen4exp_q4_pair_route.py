from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, route_scope, Q4_PAIR_ENV


def test_q4_pair_route_requires_publication_shape_and_registration(monkeypatch):
    select = runner.qwen4_exp_q4_pair_prefill_selected
    shape = {"backend": "hip_gfx1151", "quant": "gguf_q4_k", "rows": 512, "in_features": 2560}
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL", "1")
    assert not select(**shape)
    shape["published"] = True
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL", "0")
    assert select(**shape)
    assert select(**(shape | {"rows": 64, "in_features": 4096}))
    for override in ({"rows": 63}, {"in_features": 4352}, {"in_features": 0}, {"in_features": 257}):
        assert not select(**(shape | override))
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert not select(**shape)


@pytest.mark.parametrize("mode", ["before", "after"])
def test_q4_pair_ab_scopes_its_owner_and_restores_prerequisites(mode):
    environment = {"other": "unchanged", "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL": "stale"}
    prior_env = dict(environment)
    owner = SimpleNamespace(q4_pair_prefill=True, q4_bundle_prefill=False,
                            q4_exact_grouped_gate=False, q4_exact_grouped_all=False)
    prior_owner = vars(owner).copy()
    _apply_mode(mode, environment=environment, route_package="q4-pair")
    assert environment == prior_env
    with pytest.raises(RuntimeError, match="failure"):
        with route_scope("q4-pair", mode, owner, environment=environment):
            assert owner.q4_pair_prefill is (mode == "after")
            assert owner.q4_bundle_prefill is True
            assert owner.q4_exact_grouped_gate is True
            assert owner.q4_exact_grouped_all is True
            assert environment == prior_env | Q4_PAIR_ENV
            raise RuntimeError("failure")
    assert environment == prior_env
    assert vars(owner) == prior_owner


def test_q4_pair_ab_requires_owning_runner():
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("q4-pair", "after")
