from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


def _runner() -> qgr.Qwen35GGUFFullStackRunner:
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.weights = SimpleNamespace(config=SimpleNamespace(hidden_size=16))
    runner._cast_library = lambda: "cast-library"
    return runner


def test_verify_f32_attention_norm_flag_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM", raising=False)
    assert qgr._gguf_verify_f32_attention_norm_enabled() is False


def test_attention_norm_f32_diagnostic_materializes_f32_and_bf16_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM", "1")
    calls: list[tuple[str, tuple, dict]] = []

    def rmsnorm_out_f32(*args, **kwargs):
        calls.append(("rmsnorm_out_f32", args, kwargs))

    def cast_f32_to_bf16(*args, **kwargs):
        calls.append(("f32_to_bf16", args, kwargs))

    monkeypatch.setattr(qgr, "gguf_rmsnorm_f32_f32_weight_out_f32", rmsnorm_out_f32)
    monkeypatch.setattr(
        qgr,
        "gguf_rmsnorm_f32_f32_weight",
        lambda *args, **kwargs: pytest.fail("BF16-only F32 RMSNorm should not run"),
    )
    monkeypatch.setattr(qgr, "f32_to_bf16", cast_f32_to_bf16)

    out_f32_ptr = _runner()._run_attention_norm_rows(
        hidden_ptr=100,
        hidden_f32_ptr=200,
        weight_ptr=300,
        out_ptr=400,
        out_f32_ptr=500,
        rows=2,
        eps=1.0e-6,
        stream=7,
        runtime=SimpleNamespace(),
    )

    assert out_f32_ptr == 500
    assert [name for name, _args, _kwargs in calls] == ["rmsnorm_out_f32", "f32_to_bf16"]
    assert calls[0][1][:3] == (200, 300, 500)
    assert calls[0][2]["rows"] == 2
    assert calls[0][2]["hidden_size"] == 16
    assert calls[0][2]["stream"] == 7
    assert calls[1][1][:3] == (500, 400, 32)
    assert calls[1][2]["library"] == "cast-library"
    assert calls[1][2]["stream"] == 7


def test_attention_norm_f32_diagnostic_falls_back_without_f32_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM", "1")
    calls: list[tuple[str, tuple, dict]] = []

    def rmsnorm_bf16_out(*args, **kwargs):
        calls.append(("rmsnorm_bf16_out", args, kwargs))

    monkeypatch.setattr(
        qgr,
        "gguf_rmsnorm_f32_f32_weight_out_f32",
        lambda *args, **kwargs: pytest.fail("F32 output requires out_f32_ptr"),
    )
    monkeypatch.setattr(qgr, "gguf_rmsnorm_f32_f32_weight", rmsnorm_bf16_out)
    monkeypatch.setattr(qgr, "f32_to_bf16", lambda *args, **kwargs: pytest.fail("cast should not run"))

    out_f32_ptr = _runner()._run_attention_norm_rows(
        hidden_ptr=100,
        hidden_f32_ptr=200,
        weight_ptr=300,
        out_ptr=400,
        out_f32_ptr=None,
        rows=2,
        eps=1.0e-6,
        stream=7,
        runtime=SimpleNamespace(),
    )

    assert out_f32_ptr is None
    assert [name for name, _args, _kwargs in calls] == ["rmsnorm_bf16_out"]
    assert calls[0][1][:3] == (200, 300, 400)
