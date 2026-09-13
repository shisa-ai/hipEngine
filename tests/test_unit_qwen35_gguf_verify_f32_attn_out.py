from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


class _Layer:
    def weight(self, name: str):
        assert name == "post_attention_norm"
        return SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=300)))


class _Weights:
    config = SimpleNamespace(
        hidden_size=16,
        is_moe=True,
        rms_norm_eps=1.0e-6,
    )

    def layer(self, layer_id: int):
        assert layer_id == 0
        return _Layer()


def _runner() -> qgr.Qwen35GGUFFullStackRunner:
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.weights = _Weights()
    runner.runtime = SimpleNamespace()
    return runner


def _scratch() -> SimpleNamespace:
    return SimpleNamespace(post_norm=SimpleNamespace(ptr=400))


def test_verify_f32_attn_out_flag_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT", raising=False)
    assert qgr._gguf_verify_f32_attn_out_enabled() is False


def test_post_attention_f32_attn_out_uses_f32_add_norm_when_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT", "1")
    calls: list[tuple[str, tuple, dict]] = []

    def add_norm_f32_f32(*args, **kwargs):
        calls.append(("add_norm_f32_f32", args, kwargs))

    def moe_rows(self, *args, **kwargs):
        calls.append(("moe_rows", args, kwargs))

    monkeypatch.setattr(qgr, "gguf_add_rmsnorm_f32_f32_f32_weight", add_norm_f32_f32)
    monkeypatch.setattr(
        qgr,
        "gguf_add_rmsnorm_f32_bf16_f32_weight",
        lambda *args, **kwargs: pytest.fail("BF16 add path should not run"),
    )
    monkeypatch.setattr(qgr.Qwen35GGUFFullStackRunner, "_run_post_attention_moe_rows", moe_rows)

    _runner()._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=500,
        scratch=_scratch(),
        rows=2,
        hidden_f32_ptr=600,
        out_f32_ptr=700,
        attn_out_f32_ptr=800,
    )

    assert [name for name, _args, _kwargs in calls] == ["add_norm_f32_f32", "moe_rows"]
    assert calls[0][1][:5] == (600, 800, 300, 400, 700)
    assert calls[0][2]["rows"] == 2
    assert calls[0][2]["hidden_size"] == 16
    assert calls[1][2]["residual_f32_ptr"] == 700
    assert calls[1][2]["out_f32_ptr"] == 700


def test_post_attention_c1_moe_closes_device_stage_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    scratch = SimpleNamespace(
        post_norm=SimpleNamespace(ptr=400),
        residual=SimpleNamespace(ptr=401),
    )
    marks: list[str] = []
    monkeypatch.setattr(
        qgr,
        "_gguf_norm_residual_decode_kernel",
        lambda *args, **kwargs: lambda *args, **kwargs: None,
    )
    def fake_moe(*args, **kwargs):
        recorder = kwargs["gpu_stage_recorder"]
        prefix = kwargs["stage_prefix"]
        recorder.mark(f"{prefix}_router")
        recorder.mark(f"{prefix}_combine")

    monkeypatch.setattr(runner, "_run_post_attention_moe_c1", fake_moe)
    recorder = SimpleNamespace(mark=lambda name: marks.append(name))

    runner._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=500,
        scratch=scratch,
        rows=1,
        stage_prefix="decode_ffn",
        gpu_stage_recorder=recorder,
    )

    assert marks == [
        "decode_ffn_post_norm_residual",
        "decode_ffn_moe_router",
        "decode_ffn_moe_combine",
    ]


def test_post_attention_f32_attn_out_falls_back_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT", raising=False)
    calls: list[tuple[str, tuple, dict]] = []

    def add_norm_f32_bf16(*args, **kwargs):
        calls.append(("add_norm_f32_bf16", args, kwargs))

    def moe_rows(self, *args, **kwargs):
        calls.append(("moe_rows", args, kwargs))

    monkeypatch.setattr(
        qgr,
        "gguf_add_rmsnorm_f32_f32_f32_weight",
        lambda *args, **kwargs: pytest.fail("F32 add path requires flag"),
    )
    monkeypatch.setattr(qgr, "gguf_add_rmsnorm_f32_bf16_f32_weight", add_norm_f32_bf16)
    monkeypatch.setattr(qgr.Qwen35GGUFFullStackRunner, "_run_post_attention_moe_rows", moe_rows)

    _runner()._run_post_attention_ffn_rows(
        0,
        hidden_ptr=100,
        attn_out_ptr=200,
        out_ptr=500,
        scratch=_scratch(),
        rows=2,
        hidden_f32_ptr=600,
        out_f32_ptr=700,
        attn_out_f32_ptr=800,
    )

    assert [name for name, _args, _kwargs in calls] == ["add_norm_f32_bf16", "moe_rows"]
    assert calls[0][1][:5] == (600, 200, 300, 400, 700)
