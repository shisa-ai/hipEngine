from hipengine.runtime import qwen4_exp_runner as runner
import pytest


def test_qsa_benchmark_modes_and_engagement():
    from scripts.qwen4exp_halo_box_campaign_ab import (
        _apply_mode, route_scope, validate_qsa_h256_engagement,
    )
    stale_row4 = "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"
    from types import SimpleNamespace
    owner = SimpleNamespace(qsa_h256_prefill_mode="page256")
    env = {stale_row4: "1"}
    for package, selected in (("qsa-h256-wave", "wave"),
                              ("qsa-h256-page256", "page256")):
        for mode in ("before", "after"):
            _apply_mode(mode, environment=env, route_package=package)
            assert env == {stale_row4: "1"}
            with route_scope(package, mode, owner):
                assert owner.qsa_h256_prefill_mode == (
                    selected if mode == "after" else "parent")
            assert owner.qsa_h256_prefill_mode == "page256"
    # The head leaf is request-local: neither head package writes an env key.
    for package in ("qsa-head-pair", "qsa-head-quad"):
        for mode in ("before", "after"):
            _apply_mode(mode, environment=env, route_package=package)
            assert env == {stale_row4: "1"}
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
    def select(**kwargs):
        return runner.qwen4_exp_qsa_h256_wave_prefill_selected(published="wave", **kwargs)
    assert select(head_dim=256, rows=509, backend="registered")
    assert not select(head_dim=128, rows=509, backend="registered")
    assert not select(head_dim=256, rows=15, backend="registered")
    assert not select(head_dim=256, rows=509, backend="missing")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL", "0")
    assert select(head_dim=256, rows=509, backend="registered")
    assert not runner.qwen4_exp_qsa_h256_wave_prefill_selected(
        head_dim=256, rows=509, backend="registered", published="parent")


def test_page256_registry_choice(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL", "page256")
    monkeypatch.setattr(runner, "is_registered", lambda key: "page256" in key.variant)
    assert runner.qwen4_exp_qsa_h256_wave_prefill_selected(
        head_dim=256, rows=509, backend="test", block_size=256, published="page256")
    assert not runner.qwen4_exp_qsa_h256_wave_prefill_selected(
        head_dim=256, rows=509, backend="test", block_size=128, published="page256")
