from hipengine.runtime import qwen4_exp_runner as runner
import pytest


def test_qsa_benchmark_modes_and_engagement():
    from scripts.qwen4exp_halo_box_campaign_ab import (
        _apply_mode, QSA_H256_ENV, ROW4_ENV, validate_qsa_h256_engagement,
    )
    env = {ROW4_ENV: "1"}
    _apply_mode("before", environment=env, route_package="qsa-h256-wave")
    assert env == {ROW4_ENV: "1", QSA_H256_ENV: "0"}
    _apply_mode("after", environment=env, route_package="qsa-h256-wave")
    assert env[QSA_H256_ENV] == "1"
    validate_qsa_h256_engagement("after", 48, 4096)
    for shape in (512, 1024, 4096):
        validate_qsa_h256_engagement("before", 0, shape)
    validate_qsa_h256_engagement("after", 0, 512)
    validate_qsa_h256_engagement("after", 0, 1024)
    with pytest.raises(ValueError):
        validate_qsa_h256_engagement("after", 0, 4096)
    with pytest.raises(ValueError):
        validate_qsa_h256_engagement("after", 48, 512)
    with pytest.raises(ValueError):
        validate_qsa_h256_engagement("before", 48, 4096)


def test_subset_keeps_original_case_counterbalance():
    from scripts.qwen4exp_halo_box_campaign_ab import fixture_case_index

    cases = [{"id": "code-p512"}, {"id": "english-p4096"}]
    assert fixture_case_index(cases, cases[1]) == 1


def test_h256_sparse_prefill_admission(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL", "1")
    monkeypatch.setattr(runner, "is_registered", lambda key: key.backend == "registered")
    select = runner.qwen4_exp_qsa_h256_wave_prefill_selected
    assert select(head_dim=256, rows=509, backend="registered")
    assert not select(head_dim=128, rows=509, backend="registered")
    assert not select(head_dim=256, rows=15, backend="registered")
    assert not select(head_dim=256, rows=509, backend="missing")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL", "0")
    assert not select(head_dim=256, rows=509, backend="registered")
