from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.qwen35_gguf_runner as runner_mod


def _session_with_lm_head_x8() -> SimpleNamespace:
    weight = SimpleNamespace(
        allocation=lambda name="raw": SimpleNamespace(tensor=SimpleNamespace(ptr=0x2200))
        if name == "x8"
        else (_ for _ in ()).throw(KeyError(name))
    )
    return SimpleNamespace(
        runner=SimpleNamespace(
            hidden_size=64,
            vocab_size=128,
            weights=SimpleNamespace(root=lambda slot: weight),
        ),
        runtime=SimpleNamespace(),
        compiler_version=None,
        require_cached_build=False,
        _verify_lm_q8_1=SimpleNamespace(ptr=0x1000),
        _verify_lm_block_values=SimpleNamespace(ptr=0x1100),
        _verify_lm_block_indices_i32=SimpleNamespace(ptr=0x1200),
        _verify_lm_out_indices_i32=SimpleNamespace(ptr=0x1300),
        _verify_lm_out_values=SimpleNamespace(ptr=0x1400),
        _q6_pack8_library=SimpleNamespace(name="q6lib"),
    )


def test_verify_lm_head_q6_top1_dp4a_launches_x8_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session_with_lm_head_x8()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(runner_mod, "_gguf_verify_lm_head_q6_top1_dp4a_enabled", lambda: True)
    monkeypatch.setattr(
        runner_mod,
        "gguf_q4_k_quantize_bf16_q8_1",
        lambda *args, **kwargs: calls.append(("quant", args, kwargs)),
    )
    monkeypatch.setattr(
        runner_mod,
        "gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_gather_f32",
        lambda *args, **kwargs: calls.append(("top1", args, kwargs)),
    )

    launched = runner_mod.Qwen35GGUFResidentSession._verify_lm_head_q6_top1_dp4a(
        session,
        0x9000,
        3,
        stream=7,
        runtime=session.runtime,
    )

    assert launched is True
    assert calls[0] == ("quant", (0x9000, 0x1000, 3, 64), {"stream": 7, "runtime": session.runtime})
    assert calls[1][0] == "top1"
    assert calls[1][1][:11] == (
        0x1000,
        0x2200,
        0x1100,
        0x1200,
        0x1300,
        0x1400,
        None,
        None,
        3,
        64,
        128,
    )
    assert calls[1][1][11] == 0
    assert calls[1][2] == {"stream": 7, "library": session._q6_pack8_library, "runtime": session.runtime}


def test_verify_lm_head_q6_top1_dp4a_requires_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session_with_lm_head_x8()
    session.runner.weights.root = lambda slot: SimpleNamespace(allocation=lambda name="raw": (_ for _ in ()).throw(KeyError(name)))

    monkeypatch.setattr(runner_mod, "_gguf_verify_lm_head_q6_top1_dp4a_enabled", lambda: True)

    with pytest.raises(RuntimeError, match="LM_HEAD_Q6_X8_SIDECAR"):
        runner_mod.Qwen35GGUFResidentSession._verify_lm_head_q6_top1_dp4a(
            session,
            0x9000,
            3,
            stream=7,
            runtime=session.runtime,
        )
