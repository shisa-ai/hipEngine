"""R0137 owner transport for the verifier prefill-GDN capture selector.

``HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN`` selects the captured-row producer
for the target verifier (prefill-shaped no-copy Conv/GDN state rows versus the
decode-order chain) and is a hard precondition of the packed target verifier.
The process environment remains the unbound/direct diagnostic input, but it is
not the carrier between owners: a profile binds the decision on the generator,
sessions freeze it at construction, and verifier operations that require the
capture mode state it explicitly instead of wrapping the call in an environment
write.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)

CAPTURE_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    monkeypatch.delenv(CAPTURE_ENV, raising=False)
    yield
    clear_runtime_profile_registry_for_tests()
    os.environ.pop(CAPTURE_ENV, None)


def _new_generator():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    return object.__new__(Qwen35GGUFBringupGenerator)


def _bare_runner(requested: bool | None):
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    runner = object.__new__(runner_mod.Qwen35GGUFFullStackRunner)
    runner.target_verifier_prefill_gdn_capture = requested
    runner.verifier_prefill_gdn_capture = False
    runner._verifier_prefill_gdn_capture_frozen = False
    return runner


# --------------------------------------------------------------- resolution


def test_capture_prefill_gdn_resolver_prefers_owner_and_reads_ambient_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    calls: list[int] = []

    def fake_enabled() -> bool:
        calls.append(1)
        return True

    monkeypatch.setattr(
        runner_mod, "_gguf_verify_capture_prefill_gdn_enabled", fake_enabled
    )

    assert runner_mod._resolve_verifier_prefill_gdn_capture(False) is False
    assert runner_mod._resolve_verifier_prefill_gdn_capture(True) is True
    assert calls == []

    assert runner_mod._resolve_verifier_prefill_gdn_capture(None) is True
    assert len(calls) == 1


def test_capture_prefill_gdn_runner_freezes_ambient_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    calls: list[int] = []

    def fake_enabled() -> bool:
        calls.append(1)
        return True

    monkeypatch.setattr(
        runner_mod, "_gguf_verify_capture_prefill_gdn_enabled", fake_enabled
    )
    runner = _bare_runner(None)

    assert runner._freeze_verifier_prefill_gdn_capture() is True
    assert runner._freeze_verifier_prefill_gdn_capture() is True
    assert len(calls) == 1
    assert runner.verifier_prefill_gdn_capture is True
    assert runner._verifier_prefill_gdn_capture_frozen is True


def test_capture_prefill_gdn_bound_runner_never_reads_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "_gguf_verify_capture_prefill_gdn_enabled",
        lambda: (_ for _ in ()).throw(AssertionError("ambient read")),
    )
    monkeypatch.setenv(CAPTURE_ENV, "1")
    runner = _bare_runner(False)

    assert runner._freeze_verifier_prefill_gdn_capture() is False


def test_capture_prefill_gdn_operation_override_wins_over_frozen_owner() -> None:
    runner = _bare_runner(True)
    runner._freeze_verifier_prefill_gdn_capture()

    assert runner._prefill_gdn_capture_for_call(None) is True
    assert runner._prefill_gdn_capture_for_call(False) is False
    assert runner._prefill_gdn_capture_for_call(True) is True


# --------------------------------------------- construction order / interleave


def test_capture_prefill_gdn_construction_order_does_not_retarget_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unbound = _bare_runner(None)
    monkeypatch.delenv(CAPTURE_ENV, raising=False)
    assert unbound._freeze_verifier_prefill_gdn_capture() is False

    bound = _bare_runner(True)
    assert bound._freeze_verifier_prefill_gdn_capture() is True

    # A later ambient mutation cannot retarget either resolved owner.
    monkeypatch.setenv(CAPTURE_ENV, "1")
    assert unbound._prefill_gdn_capture_for_call(None) is False
    assert bound._prefill_gdn_capture_for_call(None) is True

    # Reverse construction order, with the ambient value flipped between them.
    first = _bare_runner(True)
    assert first._freeze_verifier_prefill_gdn_capture() is True
    monkeypatch.delenv(CAPTURE_ENV)
    second = _bare_runner(None)
    assert second._freeze_verifier_prefill_gdn_capture() is False
    assert first._prefill_gdn_capture_for_call(None) is True
    assert second._prefill_gdn_capture_for_call(None) is False


def test_capture_prefill_gdn_session_call_ignores_later_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.verifier_prefill_gdn_capture = False
    monkeypatch.setenv(CAPTURE_ENV, "1")

    assert session._capture_prefill_gdn_for_call(None) is False
    assert session._capture_prefill_gdn_for_call(True) is True
    assert session._capture_prefill_gdn_for_call(False) is False


# ------------------------------------------------------- generator transport


def test_capture_prefill_gdn_generator_session_kwargs() -> None:
    generator = _new_generator()
    generator.target_verifier_prefill_gdn_capture = None
    assert generator._target_verifier_prefill_gdn_capture_session_kwargs() == {}

    generator.target_verifier_prefill_gdn_capture = False
    assert generator._target_verifier_prefill_gdn_capture_session_kwargs() == {
        "target_verifier_prefill_gdn_capture": False
    }

    generator.target_verifier_prefill_gdn_capture = True
    assert generator._target_verifier_prefill_gdn_capture_session_kwargs() == {
        "target_verifier_prefill_gdn_capture": True
    }


def test_capture_prefill_gdn_generator_transports_to_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(gguf, "Qwen35GGUFFullStackRunner", FakeRunner)
    generator = _new_generator()
    generator.model_path = "unused"
    generator.backend = "hip_gfx1151"
    generator.native_batch_decode = False
    generator._serving_capacity_bound = None
    generator.target_verifier_prefill_gdn_capture = True

    generator._get_shared_runner()

    assert captured["target_verifier_prefill_gdn_capture"] is True


def test_capture_prefill_gdn_generator_transports_to_shared_session(
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
    generator.backend = "hip_gfx1151"
    generator.target_verifier_prefill_gdn_capture = False
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

    assert captured["target_verifier_prefill_gdn_capture"] is False


# --------------------------------------------------------------- NextN seam


@pytest.mark.parametrize("requested", [True, False, None])
def test_capture_prefill_gdn_nextn_transports_owner(
    monkeypatch: pytest.MonkeyPatch,
    requested: bool | None,
) -> None:
    from hipengine.runtime import qwen35_gguf_nextn as nextn

    runner_kwargs: dict[str, object] = {}
    session_kwargs: dict[str, object] = {}

    class StopNextN(Exception):
        pass

    class FakeWeights:
        backend = "hip_gfx1151"

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
            target_verifier_prefill_gdn_capture=requested,
        )

    assert runner_kwargs["target_verifier_prefill_gdn_capture"] is requested
    assert session_kwargs["target_verifier_prefill_gdn_capture"] is requested


# ------------------------------------------------------- shared-session ABI


@pytest.mark.parametrize("requested", [None, False, True])
@pytest.mark.parametrize("live", [False, True])
def test_capture_prefill_gdn_shared_session_validates_live_mode(
    monkeypatch: pytest.MonkeyPatch, requested: bool | None, live: bool
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    class ReachedScratchBoundary(Exception):
        pass

    shared = SimpleNamespace(
        backend="hip_gfx1151",
        runtime=object(),
        fp16_recurrent_state=False,
        verifier_prefill_gdn_capture=live,
    )
    monkeypatch.setattr(runner_mod, "resolve_backend", lambda backend: "hip_gfx1151")
    monkeypatch.setattr(
        runner_mod,
        "backend_package_capability",
        lambda backend, name, default=None: default,
    )
    monkeypatch.setattr(
        runner_mod,
        "_resolve_gguf_private_c1_small_weight_arena",
        lambda **kwargs: (False, "test"),
    )
    monkeypatch.setattr(
        runner_mod,
        "_resolve_gguf_private_c1_decode_scratch_arena",
        lambda **kwargs: (False, "test"),
    )

    def stop_before_scratch(value):
        raise ReachedScratchBoundary

    monkeypatch.setattr(
        runner_mod,
        "_resolve_target_verifier_production_q4_rowtile",
        stop_before_scratch,
    )
    mismatch = requested is not None and requested is not live
    expected_error = RuntimeError if mismatch else ReachedScratchBoundary
    with pytest.raises(expected_error) as caught:
        runner_mod.Qwen35GGUFResidentSession(
            "unused",
            shared_runner=shared,
            runtime=object(),
            target_verifier_prefill_gdn_capture=requested,
        )
    if mismatch:
        assert "verifier capture" in str(caught.value)
    assert shared.verifier_prefill_gdn_capture is live


# ------------------------------------------------------------ verifier seam


def test_capture_prefill_gdn_verifier_entry_points_declare_explicit_mode() -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    for name in ("verify_target_block", "verify_target_blocks_batch"):
        parameter = inspect.signature(
            getattr(runner_mod.Qwen35GGUFResidentSession, name)
        ).parameters["capture_prefill_gdn"]
        assert parameter.default is None

    for owner, name in (
        (
            runner_mod.Qwen35GGUFResidentSession,
            "_capture_verify_layer_boundary_rows",
        ),
        (
            runner_mod.Qwen35GGUFFullStackRunner,
            "_run_linear_attention_prefill_layer_rows",
        ),
    ):
        parameter = inspect.signature(getattr(owner, name)).parameters[
            "prefill_gdn_capture"
        ]
        assert parameter.default is None


def test_capture_prefill_gdn_verify_target_block_resolves_before_closed_check() -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.verifier_prefill_gdn_capture = False
    session.runner = None
    seen: list[bool | None] = []
    session._capture_prefill_gdn_for_call = (
        lambda requested: seen.append(requested) or False
    )

    with pytest.raises(RuntimeError, match="closed"):
        session.verify_target_block([1, 2], capture_prefill_gdn=True)

    assert seen == [True]


def test_capture_prefill_gdn_packed_verify_resolves_before_closed_check() -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.verifier_prefill_gdn_capture = False
    session.runner = None
    seen: list[bool | None] = []
    session._capture_prefill_gdn_for_call = (
        lambda requested: seen.append(requested) or False
    )

    with pytest.raises(RuntimeError, match="closed"):
        session.verify_target_blocks_batch([{}], capture_prefill_gdn=True)

    assert seen == [True]


# ------------------------------------------------------ profile owner binding


def test_capture_prefill_gdn_dense_gfx1100_profile_binds_owner() -> None:
    from hipengine.generation.qwen35_dense_gfx1100_profiles import (
        QWEN36_DENSE_GGUF_BACKEND,
        QWEN36_DENSE_GGUF_MODEL,
        QWEN36_DENSE_GGUF_QUANT,
        register_qwen36_dense_gguf_gfx1100_profiles,
    )

    register_qwen36_dense_gguf_gfx1100_profiles()
    generator = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    ).construct_generator(_new_generator)

    assert generator.target_verifier_prefill_gdn_capture is True
    assert CAPTURE_ENV not in os.environ


def test_capture_prefill_gdn_moe_gfx1100_profile_inherits_dense_owner() -> None:
    from hipengine.generation.qwen35_dense_gfx1100_profiles import (
        QWEN36_DENSE_GGUF_BACKEND,
        QWEN36_DENSE_GGUF_QUANT,
        register_qwen36_dense_gguf_gfx1100_profiles,
    )
    from hipengine.generation.qwen35_moe_gfx1100_profiles import (
        QWEN36_MOE_GGUF_MODEL,
        register_qwen36_moe_gguf_gfx1100_profiles,
    )

    register_qwen36_dense_gguf_gfx1100_profiles()
    register_qwen36_moe_gguf_gfx1100_profiles()
    generator = resolve_runtime_profile(
        model=QWEN36_MOE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    ).construct_generator(_new_generator)

    assert generator.target_verifier_prefill_gdn_capture is True
    assert CAPTURE_ENV not in os.environ


def test_capture_prefill_gdn_dense_gfx1151_profile_binds_owner() -> None:
    from hipengine.generation.qwen35_dense_gfx1151_profiles import (
        QWEN38_GGUF_BACKEND,
        QWEN38_GGUF_MODEL,
        QWEN38_GGUF_QUANT,
        register_qwen38_gguf_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    generator = resolve_runtime_profile(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
    ).construct_generator(_new_generator)

    assert generator.target_verifier_prefill_gdn_capture is True
    assert CAPTURE_ENV not in os.environ


def test_capture_prefill_gdn_moe_gfx1151_profile_stays_unbound() -> None:
    from hipengine.generation.qwen35_moe_gfx1151_profiles import (
        QWEN36_GGUF_BACKEND,
        QWEN36_GGUF_MODEL,
        QWEN36_GGUF_QUANT,
        register_qwen36_gguf_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    register_qwen36_gguf_gfx1151_profiles()
    generator = resolve_runtime_profile(
        model=QWEN36_GGUF_MODEL,
        backend=QWEN36_GGUF_BACKEND,
        quant=QWEN36_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    ).construct_generator(_new_generator)

    # The MoE gfx1151 owner declares no verifier capture decision; its
    # verifier routes state the requirement per operation instead.
    assert generator.target_verifier_prefill_gdn_capture is None
    assert CAPTURE_ENV not in os.environ


# --------------------------------------------------- generation route scopes


def test_capture_prefill_gdn_llama_compat_route_does_not_publish_env() -> None:
    from hipengine.generation import qwen35_gguf as gguf

    assert CAPTURE_ENV not in gguf._LLAMA_COMPAT_MTP_ENV


def test_capture_prefill_gdn_staged_environment_leaves_ambient_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    monkeypatch.delenv(CAPTURE_ENV, raising=False)
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner.generator = SimpleNamespace(target_arch="gfx1151")

    with runner._staged_adapter_arch_environment():
        assert CAPTURE_ENV not in os.environ

    assert CAPTURE_ENV not in os.environ


def test_capture_prefill_gdn_complete_cycle_leaves_ambient_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    observed: list[str | None] = []

    def execute(plan, commit):
        observed.append(os.environ.get(CAPTURE_ENV))
        return plan, commit

    adapter = SimpleNamespace(staged_frontier=False, execute_cycle=execute)
    runner = object.__new__(gguf.Qwen35GGUFResidentModelRunner)
    runner._execution_committed_request_ids = set()
    runner._mtp2_adapter = adapter
    runner._mtp2_adapter_resolved = True
    runner.generator = SimpleNamespace(target_arch="gfx1151")
    monkeypatch.delenv(CAPTURE_ENV, raising=False)

    assert runner.execute_speculative_cycle("plan", commit=True) == ("plan", True)
    assert observed == [None]
    assert CAPTURE_ENV not in os.environ


# ---------------------------------------------------------- spec-cycle plumbing


def test_capture_prefill_gdn_spec_cycle_resolver_prefers_explicit_and_session() -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    session = SimpleNamespace(verifier_prefill_gdn_capture=True)
    assert cycle_mod._resolve_capture_prefill_gdn(session, True) is True
    assert cycle_mod._resolve_capture_prefill_gdn(session, False) is False
    assert cycle_mod._resolve_capture_prefill_gdn(session, None) is True

    seen: list[bool | None] = []
    session._capture_prefill_gdn_for_call = (
        lambda requested: seen.append(requested) or False
    )
    assert cycle_mod._resolve_capture_prefill_gdn(session, None) is False
    assert seen == [None]


def test_capture_prefill_gdn_spec_cycle_entries_take_optional_mode() -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    for name in (
        "capture_qwen35_gguf_native_b2_target_graph",
        "verify_qwen35_gguf_native_b2_target",
        "verify_qwen35_gguf_native_target_from_device_proposal",
        "run_qwen35_gguf_native_mtp_cycle",
    ):
        parameter = inspect.signature(getattr(cycle_mod, name)).parameters[
            "capture_prefill_gdn"
        ]
        assert parameter.default is None


def test_capture_prefill_gdn_graph_identity_separates_capture_modes() -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    common = dict(
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    )
    decode_order = cycle_mod._native_target_configuration_key(
        capture_prefill_gdn=False, **common
    )
    prefill_order = cycle_mod._native_target_configuration_key(
        capture_prefill_gdn=True, **common
    )

    assert decode_order != prefill_order


def test_capture_prefill_gdn_graph_rejects_mismatched_capture_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    session = SimpleNamespace(position=8)
    monkeypatch.setattr(
        cycle_mod, "_native_target_binding_signature", lambda _session: (7,)
    )
    graph = cycle_mod.Qwen35GGUFNativeB2TargetGraph.__new__(
        cycle_mod.Qwen35GGUFNativeB2TargetGraph
    )
    graph.closed = False
    graph.session = session
    graph.rows = 3
    graph.context_limit = 4096
    graph.device_accept_commit = True
    graph.configuration_key = cycle_mod._native_target_configuration_key(
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_prefill_gdn=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    )
    graph.binding_signature = (7,)
    common = dict(
        position=8,
        rows=3,
        remaining_decode=8,
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    )

    assert (
        graph.launch_ineligibility_reason(
            session, capture_prefill_gdn=True, **common
        )
        is None
    )
    # A decode-order graph must never serve a capture-order call.
    assert (
        graph.launch_ineligibility_reason(
            session, capture_prefill_gdn=False, **common
        )
        == "target_graph_configuration_miss"
    )
    assert graph.can_launch(session, capture_prefill_gdn=False, **common) is False


@pytest.mark.parametrize("frozen", [True, False])
def test_capture_prefill_gdn_device_proposal_eligibility_states_mode(
    monkeypatch: pytest.MonkeyPatch, frozen: bool
) -> None:
    from hipengine.runtime import qwen35_gguf_mtp as mtp

    recorded: list[dict[str, object]] = []

    class FakeGraph:
        context_limit = 4096

        def launch_ineligibility_reason(self, _session, **kwargs):
            recorded.append(kwargs)
            return "stop"

    target = SimpleNamespace(
        position=10,
        verifier_prefill_gdn_capture=frozen,
        _native_spec_b2_target_graph_n2=FakeGraph(),
    )
    verifier = object.__new__(mtp.Qwen35GGUFTransactionalVerifier)
    verifier.closed = False
    verifier.target_verify_mode = "native"
    verifier.backend = "hip_gfx1151"
    verifier.target = target
    verifier.capture_prefill_gdn = None
    verifier.last_device_proposal_fallback_reason = None
    monkeypatch.setattr(
        mtp, "_effective_target_verify_mode", lambda *args, **kwargs: "native"
    )

    assert verifier.device_proposal_ready(2) is False
    assert recorded[0]["capture_prefill_gdn"] is frozen
    assert verifier.last_device_proposal_fallback_reason == "stop"


# ----------------------------------------------------------- source hygiene

_CAPTURE_AWARE_CALLS = frozenset(
    {
        "native_cycle",
        "prepare_native_spec_target_graph",
        "prepare_target",
        "run_native_spec_mtp_cycle",
        "verify_batch",
        "verify_target_block",
        "verify_target_blocks_batch",
        "warm",
        "warm_target",
    }
)


def _capture_aware_calls(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name in _CAPTURE_AWARE_CALLS:
            yield node, name


def test_capture_prefill_gdn_generation_calls_state_the_mode() -> None:
    from hipengine.generation import (
        qwen35_gguf,
        qwen35_gguf_moe_mtp2,
        qwen35_gguf_mtp2,
    )

    offenders: list[str] = []
    for module in (qwen35_gguf, qwen35_gguf_moe_mtp2, qwen35_gguf_mtp2):
        path = pathlib.Path(module.__file__)
        for node, name in _capture_aware_calls(path):
            keywords = {keyword.arg for keyword in node.keywords}
            if "capture_prefill_gdn" not in keywords and None not in keywords:
                offenders.append(f"{path.name}:{node.lineno}:{name}")

    assert offenders == []


def test_capture_prefill_gdn_generation_publishes_no_env_scope() -> None:
    from hipengine.generation import qwen35_gguf

    source = pathlib.Path(qwen35_gguf.__file__).read_text()

    assert CAPTURE_ENV not in source


# ------------------------------------------------ generator -> NextN seam


def _nextn_generator(requested: bool | None):
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator.model_path = "unused"
    generator.target_verifier_prefill_gdn_capture = requested
    generator._mtp_hot_vocab_path = lambda: None
    generator._shared_dense_mtp_nextn_weights = lambda *args, **kwargs: object()
    return generator


def _fake_nextn_provider(monkeypatch: pytest.MonkeyPatch, calls: list[dict]):
    from hipengine.runtime import qwen35_gguf_nextn as nextn

    class FakeProvider:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        @classmethod
        def from_model(cls, model, **kwargs):
            calls.append({"model": model, **kwargs})
            return cls()

    monkeypatch.setattr(nextn, "Qwen35GGUFNextNDraftProvider", FakeProvider)
    return FakeProvider


@pytest.mark.parametrize("requested", [True, False, None])
def test_capture_prefill_gdn_dense_provider_transports_owner_to_nextn(
    monkeypatch: pytest.MonkeyPatch, requested: bool | None
) -> None:
    calls: list[dict] = []
    _fake_nextn_provider(monkeypatch, calls)
    generator = _nextn_generator(requested)
    target = SimpleNamespace(
        runtime=object(), require_cached_build=False, runner=object()
    )

    provider, key, reused = generator._acquire_dense_mtp_draft_provider(
        target, max_positions=64, pool_enabled=False
    )

    assert provider is not None and reused is False and key is None
    # The generator owns the decision; the NextN runner must not resolve the
    # unbound default from an ambient value the dense owner never published.
    assert calls[0]["target_verifier_prefill_gdn_capture"] is requested


def test_capture_prefill_gdn_dense_provider_pool_reuse_keeps_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    _fake_nextn_provider(monkeypatch, calls)
    generator = _nextn_generator(True)
    target = SimpleNamespace(
        runtime=object(), require_cached_build=False, runner=object()
    )

    first, key, reused = generator._acquire_dense_mtp_draft_provider(
        target, max_positions=64, pool_enabled=True
    )
    assert reused is False and key is not None
    generator._release_mtp_draft_runner(key, first)

    second, second_key, second_reused = generator._acquire_dense_mtp_draft_provider(
        target, max_positions=64, pool_enabled=True
    )

    assert second is first and second_reused is True and second_key == key
    assert len(calls) == 1
    assert calls[0]["target_verifier_prefill_gdn_capture"] is True



# ------------------------------------------- transactional verifier owner seam


class _StopPrepare(Exception):
    pass


class _PrepareJournal:
    initial_state_only = False
    producer_capture_initial_state = False

    def bind_request(self, request_id: int) -> None:
        pass

    def capture_initial(self, *, stream: int, force_consumer_state: bool) -> None:
        pass

    def restore_initial(self, *, stream: int) -> None:
        pass


def _transactional_batch():
    from hipengine.speculative.interfaces import TargetVerifyBatch

    return TargetVerifyBatch(
        request_ids=(7,),
        tokens=(5, 6, 7),
        positions=(10, 11, 12),
        row_to_request=(7, 7, 7),
        parent_rows=(-1, 0, 1),
        root_rows=(0,),
        candidate_rows=(1, 2),
        draft_depths=(0, 1, 2),
        active_mask=(True, True, True),
        mode="verify_chain",
    )


def _transactional_verifier(
    monkeypatch: pytest.MonkeyPatch,
    recorded: list[dict],
    *,
    declared: bool | None,
):
    from hipengine.runtime import qwen35_gguf_mtp as mtp

    class FakeTarget:
        position = 10
        verifier_prefill_gdn_capture = False

        def verify_target_block_native_cycle(self, tokens, **kwargs):
            recorded.append(kwargs)
            raise _StopPrepare

        def verify_target_block(self, tokens, **kwargs):
            recorded.append(kwargs)
            raise _StopPrepare

    target = FakeTarget()
    verifier = object.__new__(mtp.Qwen35GGUFTransactionalVerifier)
    verifier.closed = False
    verifier._prepared = None
    verifier.target = target
    verifier.backend = "hip_gfx1151"
    verifier.target_verify_mode = "native"
    verifier.max_candidate_budget = 2
    verifier.capture_prefill_gdn = declared
    verifier._primary_journal = _PrepareJournal()
    verifier._serial_journal = None
    verifier.journal = verifier._primary_journal
    verifier._publish_position = lambda *args, **kwargs: None
    verifier._synchronize = lambda *args, **kwargs: None
    monkeypatch.setattr(
        mtp, "_effective_target_verify_mode", lambda *args, **kwargs: "native"
    )
    return verifier, target


def _prepare_recorded(
    monkeypatch: pytest.MonkeyPatch,
    *,
    declared: bool | None,
    cached_mode: bool | None,
    allow_graph: bool = True,
) -> dict:
    recorded: list[dict] = []
    verifier, target = _transactional_verifier(monkeypatch, recorded, declared=declared)
    if cached_mode is not None:
        target._native_spec_b2_target_graph_n2 = SimpleNamespace(
            capture_prefill_gdn=cached_mode
        )
    bucket = SimpleNamespace(
        owner=SimpleNamespace(
            spec=SimpleNamespace(mode="verify_chain", max_rows=3)
        ),
        replay_count=0,
    )
    with pytest.raises(_StopPrepare):
        verifier.prepare(
            _transactional_batch(),
            transaction_id=1,
            graph_bucket=bucket,
            remaining_decode=(5,),
            allow_graph=allow_graph,
        )
    assert len(recorded) == 1
    return recorded[0]


def _capture_keyed_graph(target, capture_prefill_gdn: bool):
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    graph = cycle_mod.Qwen35GGUFNativeB2TargetGraph.__new__(
        cycle_mod.Qwen35GGUFNativeB2TargetGraph
    )
    graph.closed = False
    graph.session = target
    graph.rows = 3
    graph.context_limit = 4096
    graph.device_accept_commit = True
    graph.configuration_key = cycle_mod._native_target_configuration_key(
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_prefill_gdn=capture_prefill_gdn,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    )
    graph.binding_signature = (7,)
    return graph


def test_capture_prefill_gdn_transactional_constructor_defaults_to_session_owner() -> None:
    import inspect

    from hipengine.runtime import qwen35_gguf_mtp as mtp

    signature = inspect.signature(mtp.Qwen35GGUFTransactionalVerifier.__init__)

    assert signature.parameters["capture_prefill_gdn"].default is None


def test_capture_prefill_gdn_public_verifier_states_declared_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Public-MTP declares True even though this target session is unbound
    # (owner False), so the staged adapter's capture-order graph is replayed.
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    recorded: list[dict] = []
    verifier, target = _transactional_verifier(monkeypatch, recorded, declared=True)
    monkeypatch.setattr(
        cycle_mod, "_native_target_binding_signature", lambda _session: (7,)
    )
    target._native_spec_b2_target_graph_n2 = _capture_keyed_graph(target, True)

    assert verifier.device_proposal_ready(2, remaining_decode=5) is True
    assert verifier.last_device_proposal_fallback_reason is None

    kwargs = _prepare_recorded(
        monkeypatch, declared=True, cached_mode=True
    )
    assert kwargs["capture_prefill_gdn"] is True
    assert kwargs["capture_linear_state_rows"] is True


def test_capture_prefill_gdn_direct_verifier_keeps_session_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A standalone diagnostic declares nothing: the session's frozen decision
    # wins even when a capture-order graph is cached, so the mismatch rejects.
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    recorded: list[dict] = []
    verifier, target = _transactional_verifier(monkeypatch, recorded, declared=None)
    monkeypatch.setattr(
        cycle_mod, "_native_target_binding_signature", lambda _session: (7,)
    )
    target._native_spec_b2_target_graph_n2 = _capture_keyed_graph(target, True)

    assert verifier.device_proposal_ready(2, remaining_decode=5) is False
    assert (
        verifier.last_device_proposal_fallback_reason
        == "target_graph_configuration_miss"
    )

    kwargs = _prepare_recorded(monkeypatch, declared=None, cached_mode=True)
    assert kwargs["capture_prefill_gdn"] is False


def test_capture_prefill_gdn_eviction_keeps_declared_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[dict] = []
    verifier, target = _transactional_verifier(monkeypatch, recorded, declared=True)

    assert verifier.device_proposal_ready(2, remaining_decode=5) is False
    assert verifier.last_device_proposal_fallback_reason == "target_graph_not_cached"

    kwargs = _prepare_recorded(monkeypatch, declared=True, cached_mode=None)
    assert kwargs["capture_prefill_gdn"] is True


def test_capture_prefill_gdn_eager_and_graph_paths_share_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_kwargs = _prepare_recorded(monkeypatch, declared=True, cached_mode=None)
    eager_kwargs = _prepare_recorded(
        monkeypatch, declared=True, cached_mode=None, allow_graph=False
    )

    assert graph_kwargs["capture_prefill_gdn"] is True
    assert eager_kwargs["capture_prefill_gdn"] is True
