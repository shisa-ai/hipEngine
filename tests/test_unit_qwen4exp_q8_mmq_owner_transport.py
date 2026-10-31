"""The Q8 MMQ prefill workspace gate is owner state.

The historical process key remains the direct/unbound fallback. A bound owner
never reads it, resource preparation stays an explicit cold-path operation, and
an enabled prefill without a prepared workspace fails its precondition instead
of silently measuring the scalar owner. These are CPU contract tests; no GPU
and no kernels.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine import execution_profiles as profiles
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
    Q8_MMQ_PREFILL_ENV,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGGUFResidentModelRunner,
    _parse_q8_mmq_prefill_enabled,
    _resolve_q8_mmq_prefill_enabled,
    q8_mmq_prefill_enabled,
    resolve_q8_mmq_prefill,
)

ENV = Q8_MMQ_PREFILL_ENV
UD = "gguf_ud_q4_k_xl"
Q4KM = "gguf_q4_k_m"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES", raising=False)
    yield
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


def _owner(**overrides):
    owner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    owner.runtime = object()
    owner.prefill_chunk_size = 512
    owner._buffers = []
    owner._q8_mmq_policy = None
    owner._q8_mmq_library = None
    owner._q8_mmq_buffers = ()
    owner._q8_mmq_weight_sidecars = None
    owner.dense_mmq_token64_enabled = False
    owner.dense_mmq_raw_vector_enabled = False
    owner.dense_mmq_vec4_enabled = False
    owner.q8_mmq_prefill_enabled = None
    for key, value in overrides.items():
        setattr(owner, key, value)
    return owner


def _policy():
    return SimpleNamespace(
        max_rows=1024,
        min_rows=((2560, 10240),),
        f32_planes=3,
        risk_indices_nbytes=lambda rows: 16,
    )


def _patch_allocation(monkeypatch, *, fail_at: int | None = None, error=None):
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_mmq_prefill as kernel

    freed: list[object] = []
    allocated: list[object] = []

    def allocate(nbytes, *, runtime):
        del runtime
        if fail_at is not None and len(allocated) == fail_at:
            raise error if error is not None else MemoryError("allocation failed")
        buffer = SimpleNamespace(ptr=0x1000 + len(allocated), nbytes=nbytes)
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(runtime, "malloc", allocate)
    monkeypatch.setattr(runtime, "free", lambda buffer, *, runtime: freed.append(buffer))
    monkeypatch.setattr(
        runtime, "resolve_q8_mmq_prefill_policy", lambda quant: _policy())
    monkeypatch.setattr(
        kernel, "q8_mmq_d4x3_nbytes", lambda rows, hidden: 123)
    monkeypatch.setattr(
        kernel, "build_gguf_q8_0_mmq_prefill", lambda load: "library")
    return allocated, freed


@pytest.mark.parametrize("raw,expected", [
    ("", False), ("0", False), ("false", False), ("False", False),
    ("1", True), ("true", True), ("FALSE", True), (" 0 ", True),
    ("off", True),
])
def test_raw_boolean_parser_preserves_the_historical_false_set(raw, expected):
    assert _parse_q8_mmq_prefill_enabled(raw) is expected


def test_owner_wins_over_ambient_and_override_wins_over_owner(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert resolve_q8_mmq_prefill(
        SimpleNamespace(q8_mmq_prefill_enabled=None)) is True
    assert resolve_q8_mmq_prefill(
        SimpleNamespace(q8_mmq_prefill_enabled=False)) is False

    owner = SimpleNamespace(
        q8_mmq_prefill_enabled=False,
        runner=SimpleNamespace(q8_mmq_prefill_enabled=False),
    )
    with q8_mmq_prefill_enabled(owner, True):
        assert owner.q8_mmq_prefill_enabled is True
        assert owner.runner.q8_mmq_prefill_enabled is True
    assert owner.q8_mmq_prefill_enabled is False
    assert owner.runner.q8_mmq_prefill_enabled is False
    with q8_mmq_prefill_enabled(owner, None):
        assert owner.q8_mmq_prefill_enabled is None
        assert owner.runner.q8_mmq_prefill_enabled is None


def test_override_restores_after_failure_without_allocation():
    generator = SimpleNamespace(
        q8_mmq_prefill_enabled=True,
        runner=SimpleNamespace(q8_mmq_prefill_enabled=True),
    )
    with pytest.raises(RuntimeError, match="failed arm"):
        with q8_mmq_prefill_enabled(generator, False):
            assert generator.q8_mmq_prefill_enabled is False
            assert generator.runner.q8_mmq_prefill_enabled is False
            raise RuntimeError("failed arm")
    assert generator.q8_mmq_prefill_enabled is True
    assert generator.runner.q8_mmq_prefill_enabled is True


def test_default_selection_is_unchanged_for_all_three_plans():
    ud_production, ud_runner = _bind(
        _resolve(ExecutionProfile.PRODUCTION, quant=UD))
    assert ud_production.q8_mmq_prefill_enabled is False
    assert ud_runner.q8_mmq_prefill_enabled is False

    strict, strict_runner = _bind(
        _resolve(ExecutionProfile.STRICT, quant=UD))
    assert strict.q8_mmq_prefill_enabled is False
    assert strict_runner.q8_mmq_prefill_enabled is False

    q4km, q4km_runner = _bind(
        _resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))
    assert q4km.q8_mmq_prefill_enabled is True
    assert q4km_runner.q8_mmq_prefill_enabled is True


def test_two_owners_keep_independent_decisions_in_both_bind_orders():
    production = _resolve(ExecutionProfile.PRODUCTION, quant=Q4KM)
    strict = _resolve(ExecutionProfile.STRICT, quant=UD)
    first_runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    second_runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    first = SimpleNamespace(runner=first_runner)
    second = SimpleNamespace(runner=second_runner)

    production.binder(first, production)
    strict.binder(second, strict)
    assert first.q8_mmq_prefill_enabled is True
    assert first_runner.q8_mmq_prefill_enabled is True
    assert second.q8_mmq_prefill_enabled is False
    assert second_runner.q8_mmq_prefill_enabled is False

    strict.binder(first, strict)
    assert first.q8_mmq_prefill_enabled is False
    assert second.q8_mmq_prefill_enabled is False
    production.binder(second, production)
    assert first.q8_mmq_prefill_enabled is False
    assert second.q8_mmq_prefill_enabled is True


def test_binder_no_longer_writes_the_key(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    with pytest.warns(RuntimeWarning, match=ENV):
        generator, runner = _bind(
            _resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))
    assert os.environ[ENV] == "0"
    assert generator.q8_mmq_prefill_enabled is True
    assert runner.q8_mmq_prefill_enabled is True


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
    generator.q8_mmq_prefill_enabled = True
    try:
        clone = generator._new_resident_runner(
            SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4)
    finally:
        generator.close()
    assert calls[0]["q8_mmq_prefill_enabled"] is True
    assert clone.q8_mmq_prefill_enabled is True

    inherited = _inherit_lane_decisions(
        generator, SimpleNamespace(q8_mmq_prefill_enabled=False))
    assert inherited.q8_mmq_prefill_enabled is True


def test_runner_constructor_exposes_the_owner_input():
    parameters = inspect.signature(
        Qwen4ExpGGUFResidentModelRunner.__init__).parameters
    assert parameters["q8_mmq_prefill_enabled"].default is None


def test_prepared_resources_short_circuit_before_any_ambient_lookup(monkeypatch):
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            if key == ENV:
                raise AssertionError("ambient read despite owned resources")
            return super().get(key, default)

    owned = (object(),)
    owner = _owner(_q8_mmq_buffers=owned)
    monkeypatch.setattr(runtime.os, "environ", GuardedEnvironment())
    owner._configure_q8_mmq_prefill_resources()
    assert owner._q8_mmq_buffers is owned


def test_bound_false_does_not_read_ambient_or_allocate(monkeypatch):
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            if key == ENV:
                raise AssertionError("ambient read below a bound false")
            return super().get(key, default)

    allocated, _freed = _patch_allocation(monkeypatch)
    owner = _owner(q8_mmq_prefill_enabled=False)
    monkeypatch.setattr(runtime.os, "environ", GuardedEnvironment())
    owner._configure_q8_mmq_prefill_resources()
    assert allocated == []
    assert owner._q8_mmq_buffers == ()


def test_unbound_true_ambient_allocates_and_records_ownership(monkeypatch):
    allocated, _freed = _patch_allocation(monkeypatch)
    owner = _owner()
    monkeypatch.setenv(ENV, "1")
    owner._configure_q8_mmq_prefill_resources()
    assert owner._q8_mmq_buffers == tuple(allocated)
    assert owner._buffers == allocated
    assert owner._q8_mmq_library == "library"
    assert owner._q8_mmq_policy is not None


def test_bound_true_allocates_without_ambient(monkeypatch):
    allocated, _freed = _patch_allocation(monkeypatch)
    owner = _owner(q8_mmq_prefill_enabled=True)
    monkeypatch.delenv(ENV, raising=False)
    owner._configure_q8_mmq_prefill_resources()
    assert owner._q8_mmq_buffers == tuple(allocated)


def test_partial_allocation_failure_frees_every_buffer_and_leaves_no_owner(monkeypatch):
    allocated, freed = _patch_allocation(monkeypatch, fail_at=1)
    owner = _owner(q8_mmq_prefill_enabled=True)
    with pytest.raises(MemoryError, match="allocation failed"):
        owner._configure_q8_mmq_prefill_resources()
    assert allocated == freed
    assert owner._q8_mmq_buffers == ()
    assert owner._buffers == []
    assert owner._q8_mmq_policy is None
    assert owner._q8_mmq_library is None


def test_a_failing_rollback_notes_cleanup_and_keeps_releasing(monkeypatch):
    original = MemoryError("allocation failed")
    allocated, _freed = _patch_allocation(
        monkeypatch, fail_at=2, error=original)
    freed = []

    def failing_free(buffer, *, runtime):
        del runtime
        freed.append(buffer)
        if len(freed) == 1:
            raise RuntimeError("free failed")

    monkeypatch.setattr(runtime, "free", failing_free)
    owner = _owner(q8_mmq_prefill_enabled=True)
    with pytest.raises(MemoryError) as info:
        owner._configure_q8_mmq_prefill_resources()
    assert info.value is original
    assert freed == list(reversed(allocated))
    notes = getattr(info.value, "__notes__", ())
    assert any("Q8 MMQ prefill allocation cleanup failed" in note
               and "free failed" in note for note in notes)
    assert owner._q8_mmq_buffers == ()
    assert owner._buffers == []
    assert owner._q8_mmq_policy is None
    assert owner._q8_mmq_library is None


def test_enabled_prefill_without_a_prepared_workspace_fails_loudly():
    owner = _owner(q8_mmq_prefill_enabled=True)
    with pytest.raises(ValueError, match="no device workspace is prepared"):
        with owner._q8_mmq_prefill_context():
            raise AssertionError("the session must not open")


def test_inactive_prefill_does_not_read_ambient(monkeypatch):
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            if key == ENV:
                raise AssertionError("ambient read below a bound false")
            return super().get(key, default)

    captured: dict = {}

    @contextlib.contextmanager
    def fake_session(**kwargs):
        captured.update(kwargs)
        yield

    owner = _owner(q8_mmq_prefill_enabled=False)
    monkeypatch.setattr(runtime, "q8_mmq_prefill_session", fake_session)
    monkeypatch.setattr(runtime.os, "environ", GuardedEnvironment())
    with owner._q8_mmq_prefill_context():
        pass
    assert captured["policy"] is None
    assert captured["workspace_ptr"] == 0


def test_enabled_prefill_uses_the_owned_workspace(monkeypatch):
    captured: dict = {}

    @contextlib.contextmanager
    def fake_session(**kwargs):
        captured.update(kwargs)
        yield

    workspace = SimpleNamespace(ptr=0x11, nbytes=0x22)
    risk_count = SimpleNamespace(ptr=0x33, nbytes=4)
    risk_indices = SimpleNamespace(ptr=0x44, nbytes=0x55)
    policy = _policy()
    owner = _owner(
        q8_mmq_prefill_enabled=True,
        _q8_mmq_policy=policy,
        _q8_mmq_library="library",
        _q8_mmq_buffers=(workspace, risk_count, risk_indices),
    )
    monkeypatch.setattr(runtime, "q8_mmq_prefill_session", fake_session)
    with owner._q8_mmq_prefill_context():
        pass
    assert captured["workspace_ptr"] == 0x11
    assert captured["workspace_nbytes"] == 0x22
    assert captured["risk_count_ptr"] == 0x33
    assert captured["risk_indices_ptr"] == 0x44
    assert captured["library"] == "library"
    assert captured["policy"] is policy


def test_q8_candidate_migrations_bind_the_owner():
    from scripts.qwen4exp_layer2_profile_gate import (
        CANDIDATES,
        _bind_candidate_q8_mmq_prefill,
        _candidate_named_profile_intact,
    )

    for name in ("q8_blockscale_mmq", "q8_blockscale_guarded_mmq"):
        candidate = CANDIDATES[name]
        assert candidate.q8_mmq_prefill_enabled is True
        assert ENV not in candidate.environment
        assert not _candidate_named_profile_intact(candidate)

    conservative = CANDIDATES["production_conservative"]
    assert conservative.q8_mmq_prefill_enabled is False
    assert ENV not in conservative.environment

    generator = SimpleNamespace(
        q8_mmq_prefill_enabled=False,
        runner=SimpleNamespace(q8_mmq_prefill_enabled=False),
    )
    stack = contextlib.ExitStack()
    try:
        actual = _bind_candidate_q8_mmq_prefill(
            stack, generator,
            SimpleNamespace(q8_mmq_prefill_enabled=True),
        )
        assert actual is True
        assert generator.q8_mmq_prefill_enabled is True
        assert generator.runner.q8_mmq_prefill_enabled is True
    finally:
        stack.close()
    assert generator.q8_mmq_prefill_enabled is False
    assert generator.runner.q8_mmq_prefill_enabled is False


def test_profile_gap_rejects_the_dead_key_and_names_the_owner_override():
    from scripts.qwen4exp_profile_gap import (
        _configuration_class,
        _named_profile_intact,
        _parse_overrides,
        build_parser,
    )

    with pytest.raises(ValueError, match="Q8 MMQ prefill gate is owner state"):
        _parse_overrides([f"{ENV}=1"])

    assert _named_profile_intact({}, None) is True
    assert _named_profile_intact({}, None, False) is False
    assert _configuration_class({}, None, False) == (
        "diagnostic_post_binder_override")

    args = build_parser().parse_args([
        "--model-root", "/tmp/model",
        "--mode", "prefill",
        "--output", "/tmp/out.json",
        "--q8-mmq-prefill", "on",
    ])
    assert args.q8_mmq_prefill == "on"


def test_journey_arms_keep_the_owner_decision_out_of_the_environment():
    from scripts import qwen4exp_journey_localize as journey

    key = "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"
    assert all(key not in flags for flags in journey.ARMS.values())
    assert journey.q8_mmq_arm_enabled("q8_blockscale_mmq") is True
    assert journey.q8_mmq_arm_enabled("production_conservative") is False
    assert journey.q8_mmq_arm_enabled("dense_q8_gr_strict") is False
    assert journey.q8_mmq_arm_enabled("base_plus_q8_gr_flags") is False
    assert journey.q8_mmq_arm_enabled("strict") is None
