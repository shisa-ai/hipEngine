from hipengine.runtime import qwen4_exp_runner as runner


def test_bundle_selection_requires_flag_and_registered_owner(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL","1")
    monkeypatch.setattr(runner,"is_registered",lambda key:key.backend=="present")
    assert runner.qwen4_exp_q4_bundle_prefill_selected("present","q4")
    assert not runner.qwen4_exp_q4_bundle_prefill_selected("missing","q4")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL","0")
    assert not runner.qwen4_exp_q4_bundle_prefill_selected("present","q4")


def test_bundle_arm_does_not_toggle_other_candidates():
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, Q4_BUNDLE_ENV, QSA_H256_ENV, ROW4_ENV
    env={QSA_H256_ENV:"0",ROW4_ENV:"1"}
    _apply_mode("after",environment=env,route_package="q4-bundle")
    assert env=={QSA_H256_ENV:"0",ROW4_ENV:"1",Q4_BUNDLE_ENV:"1"}
    _apply_mode("before",environment=env,route_package="q4-bundle")
    assert env[Q4_BUNDLE_ENV]=="0"
