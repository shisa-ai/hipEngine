from types import SimpleNamespace

from hipengine.runtime import qwen4_exp_runner as runner


def test_bundle_selection_requires_publication_and_registered_owner(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: key.backend == "present")
    assert runner.qwen4_exp_q4_bundle_prefill_selected("present", "q4", published=True)
    assert not runner.qwen4_exp_q4_bundle_prefill_selected("missing", "q4", published=True)
    assert not runner.qwen4_exp_q4_bundle_prefill_selected("present", "q4", published=False)


def test_bundle_arm_does_not_toggle_other_candidates():
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, route_scope

    env = {"HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL": "0",
           "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL": "1"}
    original = dict(env)
    owner = SimpleNamespace(q4_bundle_prefill=True, qsa_h256_prefill_mode="wave")
    for mode in ("after", "before"):
        _apply_mode(mode, environment=env, route_package="q4-bundle")
        with route_scope("q4-bundle", mode, owner, environment=env):
            assert owner.q4_bundle_prefill is (mode == "after")
            assert owner.qsa_h256_prefill_mode == "wave"
            assert env == original
        assert owner.q4_bundle_prefill is True
