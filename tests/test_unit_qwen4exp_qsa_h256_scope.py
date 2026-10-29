"""H256 prefill choice belongs to the runner; decode and kernels stay separate."""

import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as module
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, route_scope

LEGACY_ENV = "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"


@pytest.mark.parametrize("mode", ["parent", "wave", "page256"])
@pytest.mark.parametrize("stale", ["0", "1", "page256"])
@pytest.mark.parametrize("rows", [1, 15, 16, 17, 131])
def test_selection_uses_runner_choice_not_environment(monkeypatch, mode, stale, rows):
    monkeypatch.setenv(LEGACY_ENV, stale)
    monkeypatch.setattr(module, "is_registered", lambda key: True)
    assert module.qwen4_exp_qsa_h256_wave_prefill_selected(
        head_dim=256, rows=rows, backend="test", published=mode,
    ) is (mode != "parent" and rows >= 16)


@pytest.mark.parametrize("mode", ["parent", "wave", "page256"])
def test_geometry_registration_and_block_size_guards(monkeypatch, mode):
    monkeypatch.setattr(module, "is_registered", lambda key: key.backend == "present")
    select = module.qwen4_exp_qsa_h256_wave_prefill_selected
    assert not select(head_dim=128, rows=131, backend="present", published=mode)
    assert not select(head_dim=256, rows=131, backend="missing", published=mode)
    assert select(head_dim=256, rows=131, backend="present", block_size=128,
                  published=mode) is (mode == "wave")


def test_nested_scope_restores_the_owner_on_failure():
    owner = SimpleNamespace(qsa_h256_prefill_mode="page256")
    neighbor = SimpleNamespace(qsa_h256_prefill_mode="parent")
    with pytest.raises(RuntimeError, match="arm failed"):
        with module.qsa_h256_prefill_mode(owner, "wave"):
            with module.qsa_h256_prefill_mode(owner, "parent"):
                assert owner.qsa_h256_prefill_mode == "parent"
            assert owner.qsa_h256_prefill_mode == "wave"
            assert neighbor.qsa_h256_prefill_mode == "parent"
            raise RuntimeError("arm failed")
    assert owner.qsa_h256_prefill_mode == "page256"
    with pytest.raises(ValueError, match="H256"):
        with module.qsa_h256_prefill_mode(owner, "bogus"):
            pass
    assert owner.qsa_h256_prefill_mode == "page256"


@pytest.mark.parametrize("mode", ["parent", "wave", "page256"])
def test_clones_inherit_the_profile_choice(mode):
    assert _inherit_lane_decisions(
        SimpleNamespace(qsa_h256_prefill_mode=mode), SimpleNamespace(),
    ).qsa_h256_prefill_mode == mode
    assert _inherit_lane_decisions(SimpleNamespace(), SimpleNamespace()).qsa_h256_prefill_mode == "parent"


@pytest.mark.parametrize("package,selected", [
    ("qsa-h256-wave", "wave"), ("qsa-h256-page256", "page256"),
])
@pytest.mark.parametrize("arm", ["before", "after"])
def test_campaign_binds_and_restores_runner_without_writing_env(package, selected, arm):
    owner = SimpleNamespace(qsa_h256_prefill_mode="page256")
    env = {LEGACY_ENV: "stale"}
    _apply_mode(arm, route_package=package, environment=env)
    with route_scope(package, arm, owner, environment=env):
        assert owner.qsa_h256_prefill_mode == (selected if arm == "after" else "parent")
        assert env == {LEGACY_ENV: "stale"}
    assert owner.qsa_h256_prefill_mode == "page256"


def test_prefill_layer_and_runner_forward_choice_but_decode_does_not_read_it():
    layer = inspect.getsource(module.run_qwen4_exp_qsa_prefill_layer)
    prefill = inspect.getsource(module.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps)
    mixer = inspect.getsource(module.run_qwen4_exp_qsa_prefill_token_mixer)
    assert "h256_mode=h256_mode" in layer
    assert "h256_mode=self.qsa_h256_prefill_mode" in prefill
    assert "published=h256_mode" in mixer
    assert LEGACY_ENV not in mixer
    assert "h256_mode" not in inspect.getsource(module.run_qwen4_exp_dense_qsa_token_mixer)
