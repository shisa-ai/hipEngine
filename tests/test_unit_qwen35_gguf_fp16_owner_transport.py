from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation.qwen35_dense_gfx1151_profiles import (
    FP16_RECURRENT_STATE_ENV,
    QWEN38_GGUF_BACKEND,
    QWEN38_GGUF_MODEL,
    QWEN38_GGUF_QUANT,
    register_qwen38_gguf_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    monkeypatch.delenv(FP16_RECURRENT_STATE_ENV, raising=False)
    yield
    clear_runtime_profile_registry_for_tests()


def _new_generator():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    return object.__new__(Qwen35GGUFBringupGenerator)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        context_length=32_768,
        layer_types=(LINEAR_ATTENTION,),
        expert_used_count=0,
        is_moe=False,
        expert_count=0,
        expert_shared_feed_forward_length=0,
        ssm_inner_size=6_144,
        ssm_conv_kernel=4,
        ssm_time_step_rank=48,
        ssm_state_size=128,
        ssm_group_count=16,
        head_count_kv=4,
        key_length=256,
        rope_dimension_count=64,
        rope_freq_base=10_000_000.0,
        head_count=24,
    )


def test_fp16_resolver_prefers_owner_and_reads_default_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    calls: list[dict[str, object]] = []

    def fake_enabled(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(runner_mod, "_gguf_fp16_recurrent_state_enabled", fake_enabled)

    assert (
        runner_mod._resolve_fp16_recurrent_state(
            False,
            backend=QWEN38_GGUF_BACKEND,
            file_type_name="mostly_q4_k_s",
            artifact_preset_key=None,
        )
        is False
    )
    assert (
        runner_mod._resolve_fp16_recurrent_state(
            True,
            backend=QWEN38_GGUF_BACKEND,
            file_type_name="mostly_q4_k_s",
            artifact_preset_key=None,
        )
        is True
    )
    assert calls == []

    assert (
        runner_mod._resolve_fp16_recurrent_state(
            None,
            backend=QWEN38_GGUF_BACKEND,
            file_type_name="mostly_q4_k_s",
            artifact_preset_key=None,
        )
        is True
    )
    assert len(calls) == 1


def test_fp16_runner_freezes_resolution_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    calls: list[dict[str, object]] = []

    def fake_enabled(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(runner_mod, "_gguf_fp16_recurrent_state_enabled", fake_enabled)
    runner = object.__new__(runner_mod.Qwen35GGUFFullStackRunner)
    runner.backend = QWEN38_GGUF_BACKEND
    runner.target_fp16_recurrent_state = None
    runner.fp16_recurrent_state = False
    runner._fp16_recurrent_state_frozen = False

    assert (
        runner._freeze_fp16_recurrent_state(
            file_type_name="mostly_q4_k_s",
            artifact_preset_key=None,
        )
        is True
    )
    assert (
        runner._freeze_fp16_recurrent_state(
            file_type_name="other",
            artifact_preset_key="preset",
        )
        is True
    )
    assert len(calls) == 1

    bound = object.__new__(runner_mod.Qwen35GGUFFullStackRunner)
    bound.backend = QWEN38_GGUF_BACKEND
    bound.target_fp16_recurrent_state = False
    bound.fp16_recurrent_state = False
    bound._fp16_recurrent_state_frozen = False
    monkeypatch.setattr(
        runner_mod,
        "_gguf_fp16_recurrent_state_enabled",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("ambient read")),
    )
    assert (
        bound._freeze_fp16_recurrent_state(
            file_type_name="mostly_q4_k_s",
            artifact_preset_key=None,
        )
        is False
    )


def test_fp16_actual_post_init_uses_one_resolution_for_preflight_and_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.loading import qwen35_gguf_execution as execution_mod
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    class StopAfterFinalFreeze(Exception):
        pass

    resolutions: list[bool | None] = []

    def fake_resolve(
        requested: bool | None,
        *,
        backend: str | None,
        file_type_name: str | None,
        artifact_preset_key: str | None,
    ) -> bool:
        resolutions.append(requested)
        return True

    monkeypatch.setattr(runner_mod, "_resolve_fp16_recurrent_state", fake_resolve)
    real_freeze = runner_mod.Qwen35GGUFFullStackRunner._freeze_fp16_recurrent_state
    freezes: list[tuple[str | None, bool]] = []

    def wrapped_freeze(self, *, file_type_name, artifact_preset_key):
        value = real_freeze(
            self,
            file_type_name=file_type_name,
            artifact_preset_key=artifact_preset_key,
        )
        freezes.append((file_type_name, value))
        if len(freezes) == 2:
            raise StopAfterFinalFreeze
        return value

    monkeypatch.setattr(
        runner_mod.Qwen35GGUFFullStackRunner,
        "_freeze_fp16_recurrent_state",
        wrapped_freeze,
    )
    monkeypatch.setattr(runner_mod, "resolve_backend", lambda backend: "hip_gfx1151")
    monkeypatch.setattr(
        runner_mod,
        "hip_target_arch_for_backend",
        lambda backend: "gfx1151",
    )
    monkeypatch.setattr(runner_mod, "load_backend_kernel_package", lambda backend: None)
    monkeypatch.setattr(runner_mod, "resolve", lambda **kwargs: object())
    monkeypatch.setattr(runner_mod, "get_hip_runtime", lambda: object())
    monkeypatch.setattr(
        execution_mod,
        "execution_operations",
        lambda routes: ("eager", "ar_decode_native_rows"),
    )
    monkeypatch.setattr(
        runner_mod,
        "GGUFReader",
        lambda path: SimpleNamespace(
            info=SimpleNamespace(file_type_name="mostly_q4_k_s")
        ),
    )
    monkeypatch.setattr(
        runner_mod,
        "_gguf_model_info_artifact_preset_key",
        lambda info: None,
    )
    materialized: dict[str, object] = {}

    def fake_materialize(path, **kwargs):
        materialized.update(kwargs)
        return SimpleNamespace(
            file_type_name="mostly_q4_k_s",
            artifact_preset_key=None,
            backend="hip_gfx1151",
            config=None,
        )

    monkeypatch.setattr(
        runner_mod,
        "materialize_qwen35_gguf_weights",
        fake_materialize,
    )

    runner = object.__new__(runner_mod.Qwen35GGUFFullStackRunner)
    runner.model_path = "unused"
    runner.backend = "auto"
    runner.runtime = None
    runner.require_cached_build = False
    runner.token_embedding_placement = "device"
    runner.resident_weights = None
    runner.execution_routes = ("eager", "native_rows")
    runner.dense_flag_overrides = {}
    runner.use_selective_weight_arena = False
    runner.target_fp16_recurrent_state = None
    runner._fp16_recurrent_state_frozen = False

    with pytest.raises(StopAfterFinalFreeze):
        runner_mod.Qwen35GGUFFullStackRunner.__post_init__(runner)

    assert resolutions == [None]
    assert len(freezes) == 2
    assert freezes[0] == ("mostly_q4_k_s", True)
    assert freezes[1] == ("mostly_q4_k_s", True)
    assert materialized["recurrent_state_dtype"] == "fp16"
    assert runner.fp16_recurrent_state is True
    assert runner._fp16_recurrent_state_frozen is True


def test_fp16_shared_runner_transports_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(gguf, "Qwen35GGUFFullStackRunner", FakeRunner)
    generator = _new_generator()
    generator.model_path = "unused"
    generator.backend = QWEN38_GGUF_BACKEND
    generator.native_batch_decode = False
    generator._serving_capacity_bound = None
    generator.target_fp16_recurrent_state = True

    generator._get_shared_runner()

    assert captured["target_fp16_recurrent_state"] is True


def test_fp16_shared_session_transports_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    captured: dict[str, object] = {}

    class FakeSession:
        max_sequence_length = None

        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _new_generator()
    generator.model_path = "unused"
    generator.backend = QWEN38_GGUF_BACKEND
    generator.target_fp16_recurrent_state = False
    generator._prepared_kv_signature = None
    generator.dms_metadata_path = None
    generator._auto_context_estimate = None

    generator._construct_shared_session(
        SimpleNamespace(runtime=None),
        max_sequence_length=None,
        max_batch_size=1,
        defer_kv_allocation=False,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )

    assert captured["target_fp16_recurrent_state"] is False


def test_fp16_direct_session_scope_transports_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    captured: dict[str, object] = {}

    class FakeSession:
        runtime = None

        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

    monkeypatch.setattr(gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _new_generator()
    generator.model_path = "unused"
    generator.backend = QWEN38_GGUF_BACKEND
    generator.target_fp16_recurrent_state = True
    generator._prepared_kv_signature = None
    generator.dms_metadata_path = None
    monkeypatch.setattr(generator, "_configure_session", lambda session: None)

    with generator._resident_session_scope(
        shared_runner=None,
        pool_name="test",
    ) as (session, reused):
        assert session is not None
        assert reused is False

    assert captured["target_fp16_recurrent_state"] is True


@pytest.mark.parametrize("requested", [True, False, None])
def test_fp16_nextn_transports_owner(
    monkeypatch: pytest.MonkeyPatch,
    requested: bool | None,
) -> None:
    from hipengine.runtime import qwen35_gguf_nextn as nextn

    runner_kwargs: dict[str, object] = {}
    session_kwargs: dict[str, object] = {}

    class StopNextN(Exception):
        pass

    class FakeWeights:
        backend = QWEN38_GGUF_BACKEND

        def as_full_stack_weights(self):
            return object()

    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            runner_kwargs.update(kwargs)
            self.hidden_size = 4
            self.vocab_size = 8
            self.runtime = kwargs.get("runtime")

    class FakeSession:
        def __init__(self, *args, **kwargs) -> None:
            session_kwargs.update(kwargs)
            raise StopNextN

    monkeypatch.setattr(nextn, "register_gguf_q3_k_gemv_kernels", lambda: None)
    monkeypatch.setattr(nextn, "Qwen35GGUFFullStackRunner", FakeRunner)
    monkeypatch.setattr(nextn, "Qwen35GGUFResidentSession", FakeSession)

    with pytest.raises(StopNextN):
        nextn.Qwen35GGUFNextNDraftProvider.from_model(
            "unused",
            max_positions=8,
            max_requests=1,
            runtime=object(),
            resident_weights=FakeWeights(),
            target_fp16_recurrent_state=requested,
        )

    assert runner_kwargs["target_fp16_recurrent_state"] is requested
    assert session_kwargs["target_fp16_recurrent_state"] is requested


@pytest.mark.parametrize("requested", [None, False, True])
@pytest.mark.parametrize("live", [False, True])
def test_fp16_shared_session_validates_live_allocation_before_scratch(
    monkeypatch: pytest.MonkeyPatch, requested: bool | None, live: bool
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    class ReachedScratchBoundary(Exception):
        pass

    shared = SimpleNamespace(
        backend="hip_gfx1151", runtime=object(), fp16_recurrent_state=live
    )
    monkeypatch.setattr(runner_mod, "resolve_backend", lambda backend: "hip_gfx1151")
    monkeypatch.setattr(
        runner_mod, "backend_package_capability", lambda backend, name, default=None: default
    )
    monkeypatch.setattr(
        runner_mod, "_resolve_gguf_private_c1_small_weight_arena",
        lambda **kwargs: (False, "test"),
    )
    monkeypatch.setattr(
        runner_mod, "_resolve_gguf_private_c1_decode_scratch_arena",
        lambda **kwargs: (False, "test"),
    )

    def stop_before_scratch(value):
        raise ReachedScratchBoundary

    monkeypatch.setattr(
        runner_mod, "_resolve_target_verifier_production_q4_rowtile", stop_before_scratch
    )
    mismatch = requested is not None and requested is not live
    expected_error = RuntimeError if mismatch else ReachedScratchBoundary
    with pytest.raises(expected_error) as caught:
        runner_mod.Qwen35GGUFResidentSession(
            "unused", shared_runner=shared, runtime=object(),
            target_fp16_recurrent_state=requested,
        )
    if mismatch:
        assert "recurrent-state allocation" in str(caught.value)
    assert shared.fp16_recurrent_state is live


def test_fp16_configure_session_rejects_abi_change() -> None:
    generator = _new_generator()
    generator.backend = QWEN38_GGUF_BACKEND
    generator.target_fp16_recurrent_state = True
    session = SimpleNamespace(
        _native_sampler_workspace=None,
        runner=SimpleNamespace(fp16_recurrent_state=False),
    )

    with pytest.raises(RuntimeError, match="recurrent-state"):
        generator._configure_session(session)


def test_fp16_scratch_pricing_uses_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod
    from hipengine.runtime.qwen35_gguf_prefill_plan import (
        build_prefill_scratch_plan,
    )

    monkeypatch.setenv(FP16_RECURRENT_STATE_ENV, "1")
    cfg = _cfg()
    common = {
        "cfg": cfg,
        "hidden_size": 5_120,
        "ffn_size": 17_408,
        "q_width": 6_144,
        "kv_width": 1_024,
        "linear_qkv_width": 10_240,
        "max_sequence_length": 256,
        "max_batch_size": 1,
        "allocate_kv_cache": False,
    }
    f32 = runner_mod._full_stack_scratch_plan(
        **common,
        fp16_recurrent_state=False,
    )
    f16 = runner_mod._full_stack_scratch_plan(
        **common,
        fp16_recurrent_state=True,
    )
    assert f16.recurrent_zero_nbytes * 2 == f32.recurrent_zero_nbytes
    assert f16.conv_zero_nbytes == f32.conv_zero_nbytes

    prefill_common = {
        "cfg": cfg,
        "hidden_size": 5_120,
        "ffn_size": 17_408,
        "q_width": 6_144,
        "kv_width": 1_024,
        "linear_qkv_width": 10_240,
        "ssm_value_dim": 128,
        "rows": 4,
        "capacity": 4,
        "allocate_kv_cache": False,
        "normalized_qk_heads": 24,
    }
    p32 = build_prefill_scratch_plan(
        **prefill_common,
        fp16_recurrent_state=False,
    )
    p16 = build_prefill_scratch_plan(
        **prefill_common,
        fp16_recurrent_state=True,
    )
    assert (
        p16.field_sizes["linear_recurrent_state_tmp"] * 2
        == p32.field_sizes["linear_recurrent_state_tmp"]
    )
    assert (
        p16.field_sizes["linear_conv_state_tmp"]
        == p32.field_sizes["linear_conv_state_tmp"]
    )

    runner = SimpleNamespace(
        weights=SimpleNamespace(config=cfg),
        linear_qkv_width=10_240,
        ssm_value_dim=128,
        fp16_recurrent_state=False,
    )
    packed_f32 = runner_mod._gguf_packed_state_linear_bytes(runner, slot_count=1)
    runner.fp16_recurrent_state = True
    packed_f16 = runner_mod._gguf_packed_state_linear_bytes(runner, slot_count=1)
    assert packed_f16 < packed_f32


def test_fp16_dense_and_moe_gfx1151_owners_keep_distinct_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation.qwen35_moe_gfx1151_profiles import (
        QWEN36_GGUF_BACKEND,
        QWEN36_GGUF_MODEL,
        QWEN36_GGUF_QUANT,
        register_qwen36_gguf_gfx1151_profiles,
    )
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    register_qwen36_gguf_gfx1151_profiles()
    monkeypatch.setenv(FP16_RECURRENT_STATE_ENV, "1")

    dense = resolve_runtime_profile(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    ).construct_generator(_new_generator)
    moe = resolve_runtime_profile(
        model=QWEN36_GGUF_MODEL,
        backend=QWEN36_GGUF_BACKEND,
        quant=QWEN36_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    ).construct_generator(_new_generator)

    assert dense.target_fp16_recurrent_state is False
    assert moe.target_fp16_recurrent_state is None
    assert (
        runner_mod._resolve_fp16_recurrent_state(
            dense.target_fp16_recurrent_state,
            backend=QWEN38_GGUF_BACKEND,
            file_type_name="mostly_q4_k_s",
            artifact_preset_key=None,
        )
        is False
    )
    assert (
        runner_mod._resolve_fp16_recurrent_state(
            moe.target_fp16_recurrent_state,
            backend=QWEN36_GGUF_BACKEND,
            file_type_name="mostly_q4_k_s",
            artifact_preset_key=None,
        )
        is True
    )
