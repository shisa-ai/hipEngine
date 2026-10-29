from types import SimpleNamespace

from hipengine.runtime import qwen4_exp_runner as runner


def test_pair_admission(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL", "1")
    monkeypatch.setattr(runner, "is_registered", lambda key: key.backend == "present")
    select = runner.qwen4_exp_q51_pair_prefill_selected
    assert select("present", "q5", rows=64, in_features=640, published=True)
    assert not select("present", "q5", rows=63, in_features=640, published=True)
    assert not select("present", "q5", rows=64, in_features=8192, published=True)
    assert not select("missing", "q5", rows=64, in_features=640, published=True)
    assert not select("present", "q5", rows=64, in_features=640)


def test_pair_arm_retains_other_defaults():
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, route_scope

    env = {"HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL": "page256",
           "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL": "1",
           "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL": "0"}
    before = dict(env)
    owner = SimpleNamespace(q51_pair_prefill=True, q4_bundle_prefill=True)
    for mode in ("after", "before"):
        _apply_mode(mode, environment=env, route_package="q51-pair")
        with route_scope("q51-pair", mode, owner, environment=env):
            assert owner.q51_pair_prefill is (mode == "after")
            assert owner.q4_bundle_prefill is True
        assert owner.q51_pair_prefill is True
        assert env == before
