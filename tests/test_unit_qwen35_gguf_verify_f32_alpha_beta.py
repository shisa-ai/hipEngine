from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


class _Layer:
    def weight(self, name: str) -> str:
        return name


def _runner() -> qgr.Qwen35GGUFFullStackRunner:
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=16,
            is_moe=True,
            ssm_time_step_rank=4,
        )
    )
    runner._cast_library_handle = object()
    return runner


def _scratch() -> SimpleNamespace:
    return SimpleNamespace(
        linear_alpha=SimpleNamespace(ptr=300),
        linear_beta=SimpleNamespace(ptr=400),
    )


def _scratch_with_f32() -> SimpleNamespace:
    scratch = _scratch()
    scratch.linear_alpha_f32 = SimpleNamespace(ptr=500)
    scratch.linear_beta_f32 = SimpleNamespace(ptr=600)
    return scratch


def test_verify_f32_alpha_beta_flag_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA", raising=False)
    assert qgr._gguf_verify_f32_alpha_beta_enabled() is False


def test_linear_attention_alpha_beta_f32_uses_dp4a_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA", "1")
    calls: list[tuple[str, tuple, dict]] = []

    def dp4a_pair(*args, **kwargs):
        calls.append(("dp4a_pair_f32", args, kwargs))
        return True

    monkeypatch.setattr(qgr, "_try_launch_dense_q8_pair_dp4a_f32", dp4a_pair)
    monkeypatch.setattr(qgr, "launch_gguf_linear", lambda *args, **kwargs: pytest.fail("fallback should not run"))

    route = _runner()._run_linear_attention_alpha_beta_rows(
        _Layer(),
        norm_ptr=100,
        norm_f32_ptr=200,
        scratch=_scratch(),
        rows=2,
        stream=7,
        runtime=SimpleNamespace(),
    )

    assert route == "dense_q8_dp4a_f32"
    assert [name for name, _args, _kwargs in calls] == ["dp4a_pair_f32"]
    assert calls[0][1][:5] == ("ssm_alpha", "ssm_beta", 200, 300, 400)
    assert calls[0][2]["rows"] == 2
    assert calls[0][2]["in_features"] == 16
    assert calls[0][2]["out_features_a"] == 4
    assert calls[0][2]["out_features_b"] == 4
    assert calls[0][2]["stream"] == 7


def test_linear_attention_alpha_beta_f32_projection_outputs_cast_mirrors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS", "1")
    calls: list[tuple[str, tuple, dict]] = []

    def dp4a_pair(*args, **kwargs):
        calls.append(("dp4a_pair_f32_out", args, kwargs))
        return True

    def cast(*args, **kwargs):
        calls.append(("f32_to_bf16", args, kwargs))

    monkeypatch.setattr(qgr, "_try_launch_dense_q8_pair_dp4a_f32_out", dp4a_pair)
    monkeypatch.setattr(qgr, "_gguf_linear_supports_f32_activation_f32_output", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "f32_to_bf16", cast)
    monkeypatch.setattr(qgr, "_try_launch_dense_q8_pair_dp4a_f32", lambda *args, **kwargs: pytest.fail("old route"))
    monkeypatch.setattr(qgr, "launch_gguf_linear", lambda *args, **kwargs: pytest.fail("fallback should not run"))

    route = _runner()._run_linear_attention_alpha_beta_rows(
        _Layer(),
        norm_ptr=100,
        norm_f32_ptr=200,
        scratch=_scratch_with_f32(),
        rows=2,
        stream=7,
        runtime=SimpleNamespace(),
    )

    assert route == "dense_q8_dp4a_f32_out"
    assert calls[0][0] == "dp4a_pair_f32_out"
    assert calls[0][1][:5] == ("ssm_alpha", "ssm_beta", 200, 500, 600)
    assert [call[1][:3] for call in calls[1:]] == [(500, 300, 8), (600, 400, 8)]


def test_linear_attention_alpha_beta_f32_projection_outputs_use_dense_f32_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS", "1")
    calls: list[tuple[str, tuple, dict]] = []

    def linear(*args, **kwargs):
        calls.append(("linear", args, kwargs))

    def cast(*args, **kwargs):
        calls.append(("f32_to_bf16", args, kwargs))

    monkeypatch.setattr(qgr, "_gguf_linear_supports_f32_activation_f32_output", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        qgr,
        "_try_launch_dense_q8_pair_dp4a_f32_out",
        lambda *args, **kwargs: pytest.fail("dense-F32 route should win"),
    )
    monkeypatch.setattr(qgr, "launch_gguf_linear", linear)
    monkeypatch.setattr(qgr, "f32_to_bf16", cast)

    route = _runner()._run_linear_attention_alpha_beta_rows(
        _Layer(),
        norm_ptr=100,
        norm_f32_ptr=200,
        scratch=_scratch_with_f32(),
        rows=2,
        stream=7,
        runtime=SimpleNamespace(),
    )

    assert route == "f32_singletons_f32_out"
    assert [call[0] for call in calls] == ["linear", "linear", "f32_to_bf16", "f32_to_bf16"]
    assert [call[1][0] for call in calls[:2]] == ["ssm_alpha", "ssm_beta"]
    assert [call[1][1] for call in calls[:2]] == [200, 200]
    assert [call[1][2] for call in calls[:2]] == [500, 600]
    assert all(call[2]["activation_dtype"] == qgr.GGUF_ACTIVATION_F32 for call in calls[:2])
    assert all(call[2]["output_dtype"] == qgr.GGUF_OUTPUT_F32 for call in calls[:2])
    assert [call[1][:3] for call in calls[2:]] == [(500, 300, 8), (600, 400, 8)]


def test_linear_attention_alpha_beta_f32_route_predicate_keeps_dense_f32_scratch() -> None:
    assert qgr._linear_attention_alpha_beta_f32_outputs_ready("dense_q8_dp4a_f32_out")
    assert qgr._linear_attention_alpha_beta_f32_outputs_ready("f32_singletons_f32_out")
    assert not qgr._linear_attention_alpha_beta_f32_outputs_ready("dense_q8_dp4a_f32")
    assert not qgr._linear_attention_alpha_beta_f32_outputs_ready("singletons")


def test_linear_attention_alpha_beta_f32_falls_back_to_f32_singletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA", "1")
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(qgr, "_try_launch_dense_q8_pair_dp4a_f32", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "_gguf_linear_supports_f32_activation", lambda weight: True)

    def linear(*args, **kwargs):
        calls.append(("linear", args, kwargs))

    monkeypatch.setattr(qgr, "launch_gguf_linear", linear)

    route = _runner()._run_linear_attention_alpha_beta_rows(
        _Layer(),
        norm_ptr=100,
        norm_f32_ptr=200,
        scratch=_scratch(),
        rows=2,
        stream=7,
        runtime=SimpleNamespace(),
    )

    assert route == "f32_singletons"
    assert [call[1][0] for call in calls] == ["ssm_alpha", "ssm_beta"]
    assert [call[1][1] for call in calls] == [200, 200]
    assert [call[1][2] for call in calls] == [300, 400]
    assert all(call[2]["activation_dtype"] == qgr.GGUF_ACTIVATION_F32 for call in calls)


def test_linear_attention_alpha_beta_uses_bf16_mirror_without_f32_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA", "1")
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(
        qgr,
        "_try_launch_dense_q8_pair_dp4a_f32",
        lambda *args, **kwargs: pytest.fail("F32 route requires norm_f32_ptr"),
    )

    def linear(*args, **kwargs):
        calls.append(("linear", args, kwargs))

    monkeypatch.setattr(qgr, "launch_gguf_linear", linear)

    route = _runner()._run_linear_attention_alpha_beta_rows(
        _Layer(),
        norm_ptr=100,
        norm_f32_ptr=None,
        scratch=_scratch(),
        rows=2,
        stream=7,
        runtime=SimpleNamespace(),
    )

    assert route == "singletons"
    assert [call[1][0] for call in calls] == ["ssm_alpha", "ssm_beta"]
    assert [call[1][1] for call in calls] == [100, 100]
    assert all("activation_dtype" not in call[2] for call in calls)
