from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


class _Layer:
    def weight(self, name: str):
        assert name == "attn_output"
        return _weight()


def _weight(*, quant_key: str = "gguf_q8_0_t16_v1", raw: bool = True):
    allocations = {"raw": SimpleNamespace(tensor=SimpleNamespace(ptr=200))}
    if not raw:
        allocations = {}

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(layout="gguf_q8_0_t16", quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def _runner() -> qgr.Qwen35GGUFFullStackRunner:
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.weights = SimpleNamespace(config=SimpleNamespace(hidden_size=16, head_count=2, key_length=4))
    runner._cast_library = lambda: "cast-library"
    return runner


def _scratch() -> SimpleNamespace:
    return SimpleNamespace(post_norm_f32=SimpleNamespace(ptr=500))


def test_full_attention_output_f32_diagnostic_materializes_f32_and_bf16_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT", "1")
    calls: list[tuple[str, tuple, dict]] = []

    def launch_f32(*args, **kwargs):
        calls.append(("linear_bf16_f32", args, kwargs))
        return True

    def cast_f32_to_bf16(*args, **kwargs):
        calls.append(("f32_to_bf16", args, kwargs))

    monkeypatch.setattr(qgr, "_try_launch_gguf_linear_bf16_f32_output", launch_f32)
    monkeypatch.setattr(qgr, "f32_to_bf16", cast_f32_to_bf16)
    monkeypatch.setattr(qgr, "launch_gguf_linear", lambda *args, **kwargs: pytest.fail("fallback should not run"))

    attn_out_f32_ptr = _runner()._run_full_attention_output_rows(
        _Layer(),
        gated_ptr=100,
        attn_out_ptr=300,
        scratch=_scratch(),
        rows=3,
        hidden_f32_ptr=600,
        out_f32_ptr=700,
        stream=9,
        runtime=SimpleNamespace(),
    )

    assert attn_out_f32_ptr == 500
    assert [name for name, _args, _kwargs in calls] == ["linear_bf16_f32", "f32_to_bf16"]
    assert calls[0][1][0].spec.quant_key == "gguf_q8_0_t16_v1"
    assert calls[0][1][1:3] == (100, 500)
    assert calls[0][2]["rows"] == 3
    assert calls[0][2]["in_features"] == 8
    assert calls[0][2]["out_features"] == 16
    assert calls[0][2]["stream"] == 9
    assert calls[1][1][:3] == (500, 300, 48)
    assert calls[1][2]["library"] == "cast-library"


def test_full_attention_output_f32_diagnostic_falls_back_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT", raising=False)
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(
        qgr,
        "_try_launch_gguf_linear_bf16_f32_output",
        lambda *args, **kwargs: pytest.fail("F32 output requires flag"),
    )

    def linear(*args, **kwargs):
        calls.append(("linear", args, kwargs))

    monkeypatch.setattr(qgr, "launch_gguf_linear", linear)

    attn_out_f32_ptr = _runner()._run_full_attention_output_rows(
        _Layer(),
        gated_ptr=100,
        attn_out_ptr=300,
        scratch=_scratch(),
        rows=3,
        hidden_f32_ptr=600,
        out_f32_ptr=700,
        stream=9,
        runtime=SimpleNamespace(),
    )

    assert attn_out_f32_ptr is None
    assert [name for name, _args, _kwargs in calls] == ["linear"]
    assert calls[0][1][0].spec.quant_key == "gguf_q8_0_t16_v1"
    assert calls[0][1][1:3] == (100, 300)
    assert calls[0][2]["rows"] == 3


def test_full_attention_output_raw_q8_sidecar_uses_rowtile_bf16_f32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    def rowtile(*args, **kwargs):
        calls.append(("rowtile_bf16_f32", args, kwargs))

    def unsupported_dispatch(*args, **kwargs):
        raise ValueError

    monkeypatch.setattr(qgr, "resolve_gguf_linear_dispatch", unsupported_dispatch)
    monkeypatch.setattr(qgr, "launch_gguf_linear", lambda *args, **kwargs: pytest.fail("generic dispatch is unsupported"))
    monkeypatch.setattr(qgr, "gguf_q8_0_gemv_rowtile_bf16_f32_out", rowtile)
    monkeypatch.setattr(qgr, "gguf_q8_0_gemv_bf16_f32_out", lambda *args, **kwargs: pytest.fail("rowtile should handle rows=3"))

    assert qgr._try_launch_gguf_linear_bf16_f32_output(
        _weight(),
        100,
        300,
        rows=3,
        in_features=2048,
        out_features=2048,
        stream=9,
        runtime=SimpleNamespace(),
    )

    assert [name for name, _args, _kwargs in calls] == ["rowtile_bf16_f32"]
    assert calls[0][1][:6] == (100, 200, 300, 3, 2048, 2048)
    assert calls[0][2]["stream"] == 9
