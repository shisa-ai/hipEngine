"""The production grouped-MoE prefill gate is owner state.

The historical process key remains the direct/unbound fallback. A bound owner
never reads it, the row>=16 guard is resolved at the same consumer point as
before, and a captured batch segment carries the decision only when the active
row scope can select it. These are CPU contract tests; no GPU and no kernels.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine import execution_profiles as profiles
from hipengine.generation import qwen4_exp_profiles as profile_module
from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation import qwen4_exp_gguf
from hipengine.generation.qwen4_exp_gguf import (
    Qwen4ExpGGUFTextGenerator,
    _inherit_lane_decisions,
)
from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_MOE_PREFILL_ENV,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_batch_runner as batch_module
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGGUFResidentModelRunner,
    _qwen4_exp_production_moe_prefill_enabled,
    production_moe_prefill_enabled,
    resolve_production_moe_prefill,
)
from tests.test_unit_qwen4exp_batch_runner import _fake_runner, _install_step_fakes

ENV = PRODUCTION_MOE_PREFILL_ENV
UD = "gguf_ud_q4_k_xl"
Q4KM = "gguf_q4_k_m"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    profile_module._owner_binder_discarded.clear()
    profile_module._last_prebinder_conflicts.clear()
    monkeypatch.delenv(ENV, raising=False)
    yield
    profile_module._owner_binder_discarded.clear()
    profile_module._last_prebinder_conflicts.clear()
    clear_runtime_profile_registry_for_tests()


def _enable_historical_q4km_plan() -> None:
    key = profiles.RuntimeProfileKey(
        QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, Q4KM,
        ExecutionProfile.PRODUCTION,
    )
    plan = profiles._RUNTIME_PROFILE_PLANS[key]
    if plan.qualification_failure is None:
        return
    profiles.register_runtime_profile_plan(
        model=key.model, backend=key.backend, quant=key.quant,
        profile=key.profile,
        plan=replace(plan, qualification_failure=None), replace=True,
    )


def _resolve(profile: ExecutionProfile, *, quant: str = UD):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    if quant == Q4KM and profile is ExecutionProfile.PRODUCTION:
        _enable_historical_q4km_plan()
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=quant,
        profile=profile,
    )


def _bind(resolved):
    runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)
    return generator, runner


def _weight(layer: int = 27, *, backend: str = "hip_gfx1151"):
    return SimpleNamespace(
        backend=backend,
        spec=SimpleNamespace(slot_path=f"layers.{layer}.expert_gate"),
    )


def _batch_owner(enabled=None):
    owner = object.__new__(Qwen4ExpGGUFBatchRunner)
    owner.production_moe_prefill_enabled = enabled
    return owner


def _recording_layers(monkeypatch):
    _install_step_fakes(monkeypatch)
    seen: list[tuple] = []

    def layer(name):
        def run(residual_ptr, weights, *, rows, **kwargs):
            seen.append(
                (name, int(rows),
                 kwargs.get("production_moe_prefill_enabled"),
                 kwargs.get("graph_key"))
            )
            return SimpleNamespace(ptr=residual_ptr + 0x10)

        return run

    monkeypatch.setattr(
        batch_module, "run_qwen4_exp_batch_gdn_layer", layer("gdn"))
    monkeypatch.setattr(
        batch_module, "run_qwen4_exp_batch_qsa_layer", layer("qsa"))
    return seen


def _step(runner, rows: int):
    owners = [runner.admit(slot) for slot in range(rows)]
    try:
        return runner.step(
            [1] * rows,
            slots=list(range(rows)),
            owners=owners,
            capture_logits=False,
        )
    finally:
        for slot, owner in enumerate(owners):
            runner.release(slot, owner)


def test_default_selection_is_unchanged_for_all_three_plans():
    ud_production, ud_runner = _bind(
        _resolve(ExecutionProfile.PRODUCTION, quant=UD))
    assert ud_production.production_moe_prefill_enabled is False
    assert ud_runner.production_moe_prefill_enabled is False

    strict, strict_runner = _bind(
        _resolve(ExecutionProfile.STRICT, quant=UD))
    assert strict.production_moe_prefill_enabled is False
    assert strict_runner.production_moe_prefill_enabled is False

    q4km, q4km_runner = _bind(
        _resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))
    assert q4km.production_moe_prefill_enabled is True
    assert q4km_runner.production_moe_prefill_enabled is True


def test_two_owners_keep_independent_decisions_in_both_bind_orders():
    production = _resolve(ExecutionProfile.PRODUCTION, quant=Q4KM)
    strict = _resolve(ExecutionProfile.STRICT, quant=UD)
    first_runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    second_runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    first = SimpleNamespace(runner=first_runner)
    second = SimpleNamespace(runner=second_runner)

    production.binder(first, production)
    strict.binder(second, strict)
    assert first.production_moe_prefill_enabled is True
    assert first_runner.production_moe_prefill_enabled is True
    assert second.production_moe_prefill_enabled is False
    assert second_runner.production_moe_prefill_enabled is False

    strict.binder(first, strict)
    assert first.production_moe_prefill_enabled is False
    assert second.production_moe_prefill_enabled is False
    production.binder(second, production)
    assert first.production_moe_prefill_enabled is False
    assert second.production_moe_prefill_enabled is True


def test_binder_no_longer_writes_the_key_and_reports_the_discarded_value(
        monkeypatch):
    monkeypatch.setenv(ENV, "0")
    with pytest.warns(RuntimeWarning, match=ENV):
        generator, runner = _bind(
            _resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))
    assert os.environ[ENV] == "0"
    assert generator.production_moe_prefill_enabled is True
    assert runner.production_moe_prefill_enabled is True
    # A later bind of the same owner does not re-report the untouched value.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _bind(_resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_clone_and_constructor_carry_the_owner_decision(monkeypatch):
    calls: list[dict] = []

    def fake_runner(resident, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        qwen4_exp_gguf, "Qwen4ExpGGUFResidentModelRunner", fake_runner)
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        runner=SimpleNamespace(
            max_sequence_length=8, prefill_chunk_size=4, close=lambda: None),
    )
    generator.production_moe_prefill_enabled = True
    try:
        clone = generator._new_resident_runner(
            SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        generator.close()
    assert calls[0]["production_moe_prefill_enabled"] is True
    assert clone.production_moe_prefill_enabled is True

    inherited = _inherit_lane_decisions(
        generator, SimpleNamespace(production_moe_prefill_enabled=False))
    assert inherited.production_moe_prefill_enabled is True


def test_runner_constructor_exposes_the_owner_input():
    parameters = inspect.signature(
        Qwen4ExpGGUFResidentModelRunner.__init__).parameters
    assert parameters["production_moe_prefill_enabled"].default is None


@pytest.mark.parametrize("raw,expected", [
    ("", False), ("0", False), ("false", False), ("False", False),
    ("1", True), ("true", True), ("FALSE", True), (" 0 ", True),
    ("off", True),
])
def test_raw_boolean_parser_preserves_the_historical_false_set(
        monkeypatch, raw, expected):
    monkeypatch.setenv(ENV, raw)
    assert resolve_production_moe_prefill(
        SimpleNamespace(production_moe_prefill_enabled=None)) is expected


def test_owner_wins_over_ambient_and_override_wins_over_owner(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert resolve_production_moe_prefill(
        SimpleNamespace(production_moe_prefill_enabled=None)) is True
    assert resolve_production_moe_prefill(
        SimpleNamespace(production_moe_prefill_enabled=False)) is False

    owner = SimpleNamespace(
        production_moe_prefill_enabled=False,
        runner=SimpleNamespace(production_moe_prefill_enabled=False),
    )
    with production_moe_prefill_enabled(owner, True):
        assert owner.production_moe_prefill_enabled is True
        assert owner.runner.production_moe_prefill_enabled is True
    assert owner.production_moe_prefill_enabled is False
    assert owner.runner.production_moe_prefill_enabled is False
    with production_moe_prefill_enabled(owner, None):
        assert owner.production_moe_prefill_enabled is None
        assert owner.runner.production_moe_prefill_enabled is None


def test_override_restores_after_failure_without_allocation():
    generator = SimpleNamespace(
        production_moe_prefill_enabled=True,
        runner=SimpleNamespace(production_moe_prefill_enabled=True),
    )
    with pytest.raises(RuntimeError, match="failed arm"):
        with production_moe_prefill_enabled(generator, False):
            assert generator.production_moe_prefill_enabled is False
            assert generator.runner.production_moe_prefill_enabled is False
            raise RuntimeError("failed arm")
    assert generator.production_moe_prefill_enabled is True
    assert generator.runner.production_moe_prefill_enabled is True


def test_guarded_lookup_keeps_the_rows_guard_before_any_owner_or_ambient_read(
        monkeypatch):
    class ExplodingWeight:
        backend = "hip_gfx1151"

        @property
        def spec(self):
            raise AssertionError("slot path read below the row guard")

    monkeypatch.setenv(ENV, "1")
    assert _qwen4_exp_production_moe_prefill_enabled(
        ExplodingWeight(), rows=15) is False
    assert _qwen4_exp_production_moe_prefill_enabled(
        ExplodingWeight(), rows=16, enabled=False) is False
    with pytest.raises(AssertionError, match="row guard"):
        _qwen4_exp_production_moe_prefill_enabled(
            ExplodingWeight(), rows=16, enabled=True)


def test_layer_and_capability_guards_are_unchanged():
    register_gfx1151_kernels(replace=True)
    assert not _qwen4_exp_production_moe_prefill_enabled(
        _weight(26), rows=256, enabled=True)
    assert _qwen4_exp_production_moe_prefill_enabled(
        _weight(27), rows=256, enabled=True)
    assert _qwen4_exp_production_moe_prefill_enabled(
        _weight(47), rows=16, enabled=True)
    assert not _qwen4_exp_production_moe_prefill_enabled(
        _weight(47), rows=15, enabled=True)
    assert not _qwen4_exp_production_moe_prefill_enabled(
        SimpleNamespace(
            backend="hip_gfx1151",
            spec=SimpleNamespace(slot_path="root.expert_gate")),
        rows=256, enabled=True)


def test_run_moe_does_not_resolve_at_entry():
    source = inspect.getsource(runtime.run_qwen4_exp_moe)
    assert "resolve_production_moe_prefill(" not in source
    assert "enabled=production_moe_prefill_enabled" in source


def test_batch_policy_reads_ambient_only_in_the_active_row_scope(monkeypatch):
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            if key == ENV:
                raise AssertionError("ambient read below the active row scope")
            return super().get(key, default)

    monkeypatch.setattr(runtime.os, "environ", GuardedEnvironment())
    assert _batch_owner(None)._production_moe_policy(1) is False
    assert _batch_owner(None)._production_moe_policy(15) is False
    assert _batch_owner(False)._production_moe_policy(16) is False
    with pytest.raises(AssertionError, match="active row scope"):
        _batch_owner(None)._production_moe_policy(16)


@pytest.mark.parametrize("rows", [1, 15, 16, 64])
def test_batch_step_forwards_the_owner_and_keys_only_the_active_scope(
        monkeypatch, rows):
    monkeypatch.setenv(ENV, "1")
    seen = _recording_layers(monkeypatch)
    runner = _fake_runner(slots=rows)
    runner.production_moe_prefill_enabled = False

    _step(runner, rows)
    assert seen
    for name, width, enabled, _key in seen:
        assert width == rows, (name, width)
        assert enabled is False, name
    first = [key for _name, _width, _enabled, key in seen]
    seen.clear()
    runner.production_moe_prefill_enabled = True
    _step(runner, rows)
    second = [key for _name, _width, _enabled, key in seen]

    if rows < 16:
        assert first == second
    else:
        assert first != second


def test_ar_batch_carrier_forwards_the_owner_field(monkeypatch):
    from hipengine.generation import qwen4_exp_ar_batch as ar

    class FakeBatch:
        pass

    monkeypatch.setattr(
        batch_module, "Qwen4ExpGGUFBatchRunner",
        lambda *args, **kwargs: FakeBatch())
    generator = SimpleNamespace(
        _resident=object(),
        runner=SimpleNamespace(
            max_sequence_length=8, runtime=None),
        backend="hip_gfx1151",
        production_moe_prefill_enabled=False,
    )
    batch = ar.make_qwen4_exp_ar_batch_runner(generator, state_slots=2)
    assert batch.production_moe_prefill_enabled is False


def test_mixed_target_forwards_the_owner_field():
    from hipengine.runtime import qwen4_exp_mixed_target as mixed

    source = inspect.getsource(mixed)
    assert "production_moe_prefill_enabled=getattr(" in source
    assert 'batch, "production_moe_prefill_enabled", None' in source
