from hipengine.runtime import qwen4_exp_runner as runner


def test_pair_admission(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL","1")
    monkeypatch.setattr(runner,"is_registered",lambda key:key.backend=="present")
    select=runner.qwen4_exp_q51_pair_prefill_selected
    assert select("present","q5",rows=64,in_features=640)
    assert not select("present","q5",rows=63,in_features=640)
    assert not select("present","q5",rows=64,in_features=8192)
    assert not select("missing","q5",rows=64,in_features=640)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL","0")
    assert not select("present","q5",rows=64,in_features=640)


def test_pair_arm_retains_other_defaults():
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode,Q51_PAIR_ENV,QSA_H256_ENV,Q4_BUNDLE_ENV
    env={QSA_H256_ENV:"page256",Q4_BUNDLE_ENV:"1"}
    _apply_mode("after",environment=env,route_package="q51-pair")
    assert env=={QSA_H256_ENV:"page256",Q4_BUNDLE_ENV:"1",Q51_PAIR_ENV:"1"}
    _apply_mode("before",environment=env,route_package="q51-pair")
    assert env[Q51_PAIR_ENV]=="0"
