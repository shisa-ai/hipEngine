"""Q6 workspace scopes own selection after the runner resolves its policy."""
from __future__ import annotations

import importlib

import pytest

q6 = importlib.import_module("hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv")
q4 = importlib.import_module("hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv")
ENV = "HIPENGINE_C8_Q6_DP4A_GROUPED"


@pytest.fixture
def launches(monkeypatch):
    seen = []
    monkeypatch.setattr(q6, "_launch", lambda symbol, *args, **kwargs: seen.append((symbol, args, kwargs)))
    monkeypatch.setattr(q4, "gguf_q4_k_quantize_bf16_q8_1",
                        lambda *args, **kwargs: seen.append(("quantize", args, kwargs)))
    # The primitive fallback, not a profile-specific mixed-row policy, is under test.
    monkeypatch.setattr(q6, "q6_t16_physical_rowtile_enabled", lambda: False)
    return seen


def launch(rows=24, in_features=512):
    q6.gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
        11, 22, 33, rows, in_features, 256, stream=9, library="library", runtime="runtime",
    )


@pytest.mark.parametrize("value", (None, "0", "1", "invalid"))
def test_explicit_workspace_scope_is_not_reselected_by_environment(monkeypatch, launches, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    with q6.q6_dp4a_grouped_target_session(workspace_ptr=44, workspace_nbytes=100_000):
        launch()
    assert [item[0] for item in launches] == [
        "quantize", q6._Q6_T16_QMICRO_PLANAR_Q8_1_DP4A_GROUPED_BF16_BF16,
    ]
    assert launches[0][1] == (11, 44, 24, 512)
    assert launches[1][1] == (44, 22, 33, 24, 512, 256)
    for _, _, kwargs in launches:
        assert kwargs["stream"] == 9 and kwargs["runtime"] == "runtime"
    assert launches[1][2]["library"] == "library"
    assert q6._Q6_DP4A_GROUPED_SESSION.get() is None


@pytest.mark.parametrize("rows,eligible", ((7, False), (8, True), (16, True),
                                           (24, True), (40, True), (64, True), (72, False)))
def test_scope_preserves_kernel_row_capability(launches, rows, eligible):
    with q6.q6_dp4a_grouped_target_session(workspace_ptr=44, workspace_nbytes=100_000):
        launch(rows=rows)
    assert (launches[0][0] == "quantize") is eligible
    if not eligible:
        assert all("dp4a" not in item[0] for item in launches)


def test_unsupported_input_width_uses_strict_primitive(launches):
    with q6.q6_dp4a_grouped_target_session(workspace_ptr=44, workspace_nbytes=100_000):
        launch(in_features=384)
    assert launches[0][0] != "quantize"
    assert all("dp4a" not in item[0] for item in launches)


def test_disabled_nested_scope_and_exception_restore_workspace_owner(launches):
    with q6.q6_dp4a_grouped_target_session(workspace_ptr=44, workspace_nbytes=100_000):
        outer = q6._Q6_DP4A_GROUPED_SESSION.get()
        with q6.q6_dp4a_grouped_target_session(enabled=False):
            assert q6._Q6_DP4A_GROUPED_SESSION.get() is None
            launch()
        assert q6._Q6_DP4A_GROUPED_SESSION.get() is outer
        with pytest.raises(RuntimeError, match="fixture failure"):
            with q6.q6_dp4a_grouped_target_session(workspace_ptr=55, workspace_nbytes=100_000):
                assert q6._Q6_DP4A_GROUPED_SESSION.get().workspace_ptr == 55
                raise RuntimeError("fixture failure")
        assert q6._Q6_DP4A_GROUPED_SESSION.get() is outer
    assert q6._Q6_DP4A_GROUPED_SESSION.get() is None
    assert all("dp4a" not in item[0] for item in launches)


def test_workspace_capacity_error_happens_before_any_launch(launches):
    required = 24 * (512 // 32) * 36
    with q6.q6_dp4a_grouped_target_session(workspace_ptr=44, workspace_nbytes=required - 1):
        with pytest.raises(ValueError, match="workspace is too small"):
            launch()
    assert launches == []
    assert q6._Q6_DP4A_GROUPED_SESSION.get() is None


@pytest.mark.parametrize("ptr,nbytes", ((0, 1), (1, 0), (-1, 1)))
def test_invalid_workspace_cannot_bind(ptr, nbytes):
    with pytest.raises(ValueError, match="requires a device workspace"):
        with q6.q6_dp4a_grouped_target_session(workspace_ptr=ptr, workspace_nbytes=nbytes):
            pytest.fail("invalid workspace entered")
    assert q6._Q6_DP4A_GROUPED_SESSION.get() is None


@pytest.mark.parametrize("value,enabled", ((None, False), ("0", False), ("1", True)))
def test_runner_policy_resolves_rollback_before_workspace_binding(monkeypatch, value, enabled):
    from hipengine.runtime.qwen35_gguf_runner import _gguf_c8_q6_dp4a_grouped_enabled

    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    kwargs = dict(request_count=8)
    assert _gguf_c8_q6_dp4a_grouped_enabled("hip_gfx1100", **kwargs) is enabled
    assert _gguf_c8_q6_dp4a_grouped_enabled("hip_gfx1151", **kwargs) is False
    for request_count in (1, 7, 9, 13):
        assert _gguf_c8_q6_dp4a_grouped_enabled(
            "hip_gfx1100", request_count=request_count,
        ) is False


def test_runner_uses_backend_declared_target_policy(monkeypatch):
    from hipengine.runtime import qwen35_gguf_runner as runner

    calls = []
    def capability(backend, name, default):
        calls.append((backend, name, default))
        return frozenset({3, 8})

    monkeypatch.setattr(runner, "backend_package_capability", capability)
    assert runner._gguf_c8_q6_dp4a_grouped_enabled(
        "test_backend", request_count=3, enabled=True,
    ) is True
    assert calls == [("test_backend", "GGUF_Q6_DP4A_GROUPED_TARGET_REQUEST_COUNTS", ())]
    assert runner._gguf_c8_q6_dp4a_grouped_enabled(
        "test_backend", request_count=4, enabled=True,
    ) is False


def test_backend_declares_automatic_target_width_without_identity():
    from hipengine.kernels.backends import backend_package_capability

    assert backend_package_capability(
        "hip_gfx1100", "GGUF_Q6_DP4A_GROUPED_TARGET_REQUEST_COUNTS", (),
    ) == frozenset({8})
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_Q6_DP4A_GROUPED_TARGET_REQUEST_COUNTS", (),
    ) == ()


@pytest.mark.parametrize("model_name,file_type_name", (
    (None, None), ("Unfamiliar hybrid", "MOSTLY_Q4_K_S"),
    ("Qwen3.6-27B", "MOSTLY_Q5_K_M"),
))
def test_compatible_session_does_not_select_on_artifact_identity(monkeypatch, model_name, file_type_name):
    from types import SimpleNamespace
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    session = object.__new__(Qwen35GGUFResidentSession)
    session.use_q6_dp4a_grouped = True
    session.runner = SimpleNamespace(
        backend="hip_gfx1100",
        weights=SimpleNamespace(model_name=model_name, file_type_name=file_type_name),
    )
    scratch = SimpleNamespace(moe_q8_1=SimpleNamespace(ptr=44, nbytes=100_000))
    monkeypatch.delenv(ENV, raising=False)
    with session._q6_dp4a_grouped_target_context(scratch, request_count=8):
        assert q6._Q6_DP4A_GROUPED_SESSION.get().workspace_ptr == 44
    assert q6._Q6_DP4A_GROUPED_SESSION.get() is None


@pytest.mark.parametrize("selected", (True, False))
def test_profile_session_choice_is_not_reselected_by_other_owner_env(monkeypatch, selected):
    from types import SimpleNamespace
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    session = object.__new__(Qwen35GGUFResidentSession)
    session.use_q6_dp4a_grouped = selected
    session.runner = SimpleNamespace(
        backend="hip_gfx1100",
        weights=SimpleNamespace(model_name="Qwen3.8-27B", file_type_name="MOSTLY_Q4_K_M"),
    )
    scratch = SimpleNamespace(moe_q8_1=SimpleNamespace(ptr=44, nbytes=100_000))
    monkeypatch.setenv(ENV, "0" if selected else "1")
    with session._q6_dp4a_grouped_target_context(scratch, request_count=8):
        state = q6._Q6_DP4A_GROUPED_SESSION.get()
        assert (state is not None) is selected
    assert q6._Q6_DP4A_GROUPED_SESSION.get() is None


@pytest.mark.parametrize("selected", (None, False, True))
def test_generator_configuration_carries_owner_choice_to_each_session(monkeypatch, selected):
    from types import SimpleNamespace
    from hipengine.generation import qwen35_gguf as generation

    cls = generation.Qwen35GGUFBringupGenerator
    generator = object.__new__(cls)
    generator.use_q6_dp4a_grouped = selected
    generator._configure_native_sampler = lambda session: None
    for name in ("_prefill_f16_staging_for", "_wide_f16_activation_for", "_q6_integer_mmq_for"):
        monkeypatch.setattr(generation, name, lambda owner: False)
    sessions = [SimpleNamespace(), SimpleNamespace()]
    for session in sessions:
        generator._configure_session(session)
        assert session.use_q6_dp4a_grouped is selected
    # Reacquisition restores this generator's choice, not another owner's env.
    sessions[0].use_q6_dp4a_grouped = not bool(selected)
    generator._configure_session(sessions[0])
    assert sessions[0].use_q6_dp4a_grouped is selected


def test_without_scope_ambient_true_cannot_select_changed_arithmetic(monkeypatch, launches):
    monkeypatch.setenv(ENV, "1")
    launch()
    assert launches and all("dp4a" not in item[0] for item in launches)
