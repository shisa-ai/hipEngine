from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation.qwen35_dense_gfx1151_profiles import (
    FP16_RECURRENT_STATE_ENV,
    PRODUCTION_Q4_VERIFIER_ROWTILE_ENV,
    QWEN38_GGUF_BACKEND,
    QWEN38_GGUF_MODEL,
    QWEN38_GGUF_QUANT,
    VERIFY_CAPTURE_PREFILL_GDN_ENV,
    qwen38_gguf_gfx1151_plans_registered,
    register_qwen38_gguf_gfx1151_profiles,
)


@pytest.fixture(autouse=True)
def _isolate_profile_registry(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    for name in (
        FP16_RECURRENT_STATE_ENV,
        PRODUCTION_Q4_VERIFIER_ROWTILE_ENV,
        VERIFY_CAPTURE_PREFILL_GDN_ENV,
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    clear_runtime_profile_registry_for_tests()


def _resolve(profile: ExecutionProfile):
    return resolve_runtime_profile(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=profile,
    )


def _selection_map(resolved):
    return {
        (row["layer"], row["scope"]): row
        for row in resolved.manifest["selections"]
    }


@pytest.mark.parametrize(
    "order",
    [(ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION),
     (ExecutionProfile.PRODUCTION, ExecutionProfile.STRICT)],
)
@pytest.mark.parametrize("ambient", [None, "unrelated-manifest"])
def test_manifest_identity_stays_on_each_generator(order, ambient, monkeypatch):
    name = "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256"
    if ambient is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, ambient)
    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    owners = []
    for profile in order:
        resolved = _resolve(profile)
        generator = resolved.construct_generator(SimpleNamespace)
        owners.append((generator, resolved))
        assert os.environ.get(name) == ambient
        for owner, expected in reversed(owners):
            assert owner.execution_profile_manifest_sha256 == expected.manifest_sha256
            assert owner.execution_profile_manifest == expected.manifest
            assert owner.execution_profile == expected.profile.value


def test_qwen38_strict_profile_resolves_and_disables_fp16_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FP16_RECURRENT_STATE_ENV, "1")
    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    assert qwen38_gguf_gfx1151_plans_registered()

    resolved = _resolve(ExecutionProfile.STRICT)
    generator = object.__new__(type("Generator", (), {}))
    assert resolved.binder is not None
    resolved.binder(generator, resolved)

    assert os.environ[FP16_RECURRENT_STATE_ENV] == "1"
    assert generator.target_fp16_recurrent_state is False
    assert generator.target_verifier_prefill_gdn_capture is True
    assert VERIFY_CAPTURE_PREFILL_GDN_ENV not in os.environ
    assert generator.target_verifier_production_q4_rowtile is False
    assert PRODUCTION_Q4_VERIFIER_ROWTILE_ENV not in os.environ
    assert resolved.profile is ExecutionProfile.STRICT
    assert resolved.manifest_sha256 == resolved.strict_manifest_sha256
    assert resolved.manifest["graph_policy"] == "specdec2_eager_c1"
    assert resolved.manifest["kv_policy"] == "paged_bf16"
    selections = _selection_map(resolved)
    assert selections[
        ("gdn_chain_recurrent_rmsnorm_gate", "specdec2_mtp2_target_state_rows")
    ]["selected_variant"] == "bf16_c1_exact_state_rows_tloop"
    for budget, physical_rows in ((1, 6), (2, 9), (3, 12)):
        assert selections[
            ("linear", f"specdec2_mtp2_c3_k{budget}_r{physical_rows}_q4_single")
        ]["selected_variant"] == "t16_wmma_prefill_smallm_bf16_bf16_out"
        assert selections[
            (
                "linear_pair_silu",
                f"specdec2_mtp2_c3_k{budget}_r{physical_rows}_q4_gate_up",
            )
        ]["selected_variant"] == "dense_dual_wmma_prefill_bf16_bf16_out"


def test_qwen38_production_profile_keeps_fp32_state_and_q4_rowtiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FP16_RECURRENT_STATE_ENV, "1")
    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    generator = object.__new__(type("Generator", (), {}))
    assert resolved.binder is not None
    resolved.binder(generator, resolved)

    assert resolved.profile is ExecutionProfile.PRODUCTION
    assert not resolved.fell_back_to_strict
    assert resolved.manifest_sha256 != resolved.strict_manifest_sha256
    assert os.environ[FP16_RECURRENT_STATE_ENV] == "1"
    assert generator.target_fp16_recurrent_state is False
    assert generator.target_verifier_prefill_gdn_capture is True
    assert VERIFY_CAPTURE_PREFILL_GDN_ENV not in os.environ
    assert generator.target_verifier_production_q4_rowtile is True
    assert PRODUCTION_Q4_VERIFIER_ROWTILE_ENV not in os.environ
    selections = _selection_map(resolved)
    expected = {
        ("linear", "specdec2_mtp2_c2_k3_r8_q4_single"): (
            "dense_rowtile_bf16_bf16_out",
            "t16_wmma_prefill_smallm_bf16_bf16_out",
        ),
        ("linear_pair_silu", "specdec2_mtp2_c2_k3_r8_q4_gate_up"): (
            "dense_dual_rowtile_bf16_bf16_out",
            "dense_dual_wmma_prefill_bf16_bf16_out",
        ),
        **{
            ("linear", f"specdec2_mtp2_c3_k{budget}_r{rows}_q4_single"): (
                "dense_rowtile_bf16_bf16_out",
                "t16_wmma_prefill_smallm_bf16_bf16_out",
            )
            for budget, rows in ((1, 6), (2, 9), (3, 12))
        },
        **{
            (
                "linear_pair_silu",
                f"specdec2_mtp2_c3_k{budget}_r{rows}_q4_gate_up",
            ): (
                "dense_dual_rowtile_bf16_bf16_out",
                "dense_dual_wmma_prefill_bf16_bf16_out",
            )
            for budget, rows in ((1, 6), (2, 9), (3, 12))
        },
        ("gdn_chain_recurrent_rmsnorm_gate", "specdec2_mtp2_target_state_rows"): (
            "bf16_c1_exact_state_rows_tloop",
            "bf16_c1_exact_state_rows_tloop",
        ),
    }
    for key, variants in expected.items():
        assert (
            selections[key]["selected_variant"],
            selections[key]["strict_fallback_variant"],
        ) == variants


def _new_generator():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    return object.__new__(Qwen35GGUFBringupGenerator)


@pytest.mark.parametrize(
    "order",
    [
        (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION),
        (ExecutionProfile.PRODUCTION, ExecutionProfile.STRICT),
    ],
)
def test_qwen38_construction_order_keeps_per_owner_rowtile(order) -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    generators = {}
    for profile in order:
        resolved = _resolve(profile)
        generators[profile] = resolved.construct_generator(_new_generator)

    strict = generators[ExecutionProfile.STRICT]
    production = generators[ExecutionProfile.PRODUCTION]
    assert strict.target_verifier_production_q4_rowtile is False
    assert production.target_verifier_production_q4_rowtile is True
    assert strict.target_fp16_recurrent_state is False
    assert production.target_fp16_recurrent_state is False
    assert strict._target_verifier_rowtile_session_kwargs() == {
        "target_verifier_production_q4_rowtile": False
    }
    assert production._target_verifier_rowtile_session_kwargs() == {
        "target_verifier_production_q4_rowtile": True
    }
    assert strict._target_fp16_recurrent_state_session_kwargs() == {
        "target_fp16_recurrent_state": False
    }
    assert production._target_fp16_recurrent_state_session_kwargs() == {
        "target_fp16_recurrent_state": False
    }


@pytest.mark.parametrize(
    ("profile", "ambient", "expected"),
    [
        (ExecutionProfile.PRODUCTION, "0", True),
        (ExecutionProfile.STRICT, "1", False),
    ],
)
def test_qwen38_profile_decision_overrides_ambient_without_rewriting_it(
    monkeypatch: pytest.MonkeyPatch,
    profile: ExecutionProfile,
    ambient: str,
    expected: bool,
) -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    monkeypatch.setenv(PRODUCTION_Q4_VERIFIER_ROWTILE_ENV, ambient)

    generator = _resolve(profile).construct_generator(_new_generator)

    assert generator.target_verifier_production_q4_rowtile is expected
    assert os.environ[PRODUCTION_Q4_VERIFIER_ROWTILE_ENV] == ambient


def test_qwen38_unbound_rowtile_uses_ambient_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime.qwen35_gguf_runner import (
        _resolve_target_verifier_production_q4_rowtile,
    )

    monkeypatch.setenv(PRODUCTION_Q4_VERIFIER_ROWTILE_ENV, "1")
    assert _resolve_target_verifier_production_q4_rowtile(None) is True
    assert _resolve_target_verifier_production_q4_rowtile(False) is False

    monkeypatch.setenv(PRODUCTION_Q4_VERIFIER_ROWTILE_ENV, "0")
    assert _resolve_target_verifier_production_q4_rowtile(None) is False
    assert _resolve_target_verifier_production_q4_rowtile(True) is True

    monkeypatch.delenv(PRODUCTION_Q4_VERIFIER_ROWTILE_ENV, raising=False)
    assert _resolve_target_verifier_production_q4_rowtile(None) is False


def test_qwen38_shared_session_construction_transports_rowtile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    generator.target_verifier_production_q4_rowtile = True
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

    assert captured["target_verifier_production_q4_rowtile"] is True


def test_qwen38_direct_session_scope_transports_rowtile(
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
    generator.target_verifier_production_q4_rowtile = False
    generator._prepared_kv_signature = None
    generator.dms_metadata_path = None
    monkeypatch.setattr(generator, "_configure_session", lambda session: None)

    with generator._resident_session_scope(
        shared_runner=None,
        pool_name="test",
    ) as (session, reused):
        assert session is not None
        assert reused is False

    assert captured["target_verifier_production_q4_rowtile"] is False


def test_qwen38_configure_session_preserves_interleaved_owner_rowtile_against_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime.gguf_linear import (
        _target_verifier_production_q4_rowtile_session_enabled as session_rowtile,
        target_verifier_production_q4_rowtile_session,
    )

    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    owners = [
        (_resolve(ExecutionProfile.STRICT).construct_generator(_new_generator), False),
        (
            _resolve(ExecutionProfile.PRODUCTION).construct_generator(_new_generator),
            True,
        ),
        (_resolve(ExecutionProfile.STRICT).construct_generator(_new_generator), False),
    ]
    for generator, _ in owners:
        generator.backend = QWEN38_GGUF_BACKEND
    sessions = [SimpleNamespace(_native_sampler_workspace=None) for _ in owners]

    # Hostile ambient values alternate between every owner. A bound owner's
    # choice must survive both a fresh session and a session reused in
    # strict -> production -> strict order.
    for ambient in ("1", "0"):
        monkeypatch.setenv(PRODUCTION_Q4_VERIFIER_ROWTILE_ENV, ambient)
        for (generator, expected), session in zip(owners, sessions):
            generator._configure_session(session)
            assert session.target_verifier_production_q4_rowtile is expected
            with target_verifier_production_q4_rowtile_session(
                session.target_verifier_production_q4_rowtile
            ):
                assert session_rowtile.get() is expected
            assert session_rowtile.get() is False

    reused = sessions[0]
    for generator, expected in owners:
        monkeypatch.setenv(
            PRODUCTION_Q4_VERIFIER_ROWTILE_ENV,
            "0" if expected else "1",
        )
        generator._configure_session(reused)
        assert reused.target_verifier_production_q4_rowtile is expected
        with target_verifier_production_q4_rowtile_session(
            reused.target_verifier_production_q4_rowtile
        ):
            assert session_rowtile.get() is expected
        assert session_rowtile.get() is False

    # An unbound/direct owner leaves the session's construction-time diagnostic
    # decision alone instead of retargeting it from ambient input.
    monkeypatch.setenv(PRODUCTION_Q4_VERIFIER_ROWTILE_ENV, "0")
    unbound = _new_generator()
    unbound.backend = QWEN38_GGUF_BACKEND
    assert unbound.target_verifier_production_q4_rowtile is None
    diagnostic_session = SimpleNamespace(
        _native_sampler_workspace=None,
        target_verifier_production_q4_rowtile=True,
    )
    unbound._configure_session(diagnostic_session)
    assert diagnostic_session.target_verifier_production_q4_rowtile is True


@pytest.mark.parametrize("requested", [True, False, None])
def test_qwen38_nextn_transports_rowtile_to_batch_session(
    monkeypatch: pytest.MonkeyPatch,
    requested: bool | None,
) -> None:
    from hipengine.runtime import qwen35_gguf_nextn as nextn

    captured: dict[str, object] = {}

    class StopNextN(Exception):
        pass

    class FakeWeights:
        backend = QWEN38_GGUF_BACKEND

        def as_full_stack_weights(self):
            return object()

    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            self.hidden_size = 4
            self.vocab_size = 8
            self.runtime = kwargs.get("runtime")

    class FakeSession:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)
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
            target_verifier_production_q4_rowtile=requested,
        )

    assert captured["target_verifier_production_q4_rowtile"] is requested
