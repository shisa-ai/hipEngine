from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.qwen35_gguf_runner as runner_mod


def test_small_b_rowtile_chunks_avoid_single_row_tail() -> None:
    assert runner_mod._small_b_rowtile_chunks(6) == (6,)
    assert runner_mod._small_b_rowtile_chunks(7) == (5, 2)
    assert runner_mod._small_b_rowtile_chunks(8) == (6, 2)
    assert runner_mod._small_b_rowtile_chunks(12) == (6, 6)
    assert runner_mod._small_b_rowtile_chunks(13) == (6, 5, 2)


def test_verify_lm_head_rowtile_chunked_splits_large_packed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=64, vocab_size=128, backend="hip_gfx1100")
    runtime = SimpleNamespace()
    calls: list[tuple[int, int, int, int, object]] = []

    def fake_rowtile(self, hidden_ptr, out_ptr, rows, *, stream=0, runtime=None):
        calls.append((int(hidden_ptr), int(out_ptr), int(rows), int(stream), runtime))
        return True

    monkeypatch.setattr(runner_mod.Qwen35GGUFResidentSession, "_verify_lm_head_rowtile", fake_rowtile)
    monkeypatch.setattr(
        runner_mod.Qwen35GGUFResidentSession,
        "_verify_lm_head_rowtile_max_rows",
        lambda self: 4,
    )

    handled = session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 12, stream=3, runtime=runtime)

    assert handled is True
    hidden_stride = 64 * runner_mod.DType.BF16.itemsize
    logits_stride = 128 * runner_mod.DType.FP32.itemsize
    # The planar-qmicro lm_head f32 rowtile owner accepts rows in [2, 4], so
    # the chunk planner caps configured chunks to 4 and rows=12 splits 4+4+4.
    assert calls == [
        (0x100000, 0x200000, 4, 3, runtime),
        (0x100000 + 4 * hidden_stride, 0x200000 + 4 * logits_stride, 4, 3, runtime),
        (0x100000 + 8 * hidden_stride, 0x200000 + 8 * logits_stride, 4, 3, runtime),
    ]


def test_verify_lm_head_rowtile_chunked_uses_gfx1151_chunk8_and_env_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=64, vocab_size=128, backend="hip_gfx1151")
    calls: list[int] = []

    def fake_rowtile(self, hidden_ptr, out_ptr, rows, *, stream=0, runtime=None):
        calls.append(int(rows))
        return True

    monkeypatch.delenv("HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK", raising=False)
    monkeypatch.setattr(
        runner_mod.Qwen35GGUFResidentSession,
        "_verify_lm_head_rowtile",
        fake_rowtile,
    )
    monkeypatch.setattr(
        runner_mod.Qwen35GGUFResidentSession,
        "_verify_lm_head_rowtile_max_rows",
        lambda self: 8,
    )

    # c8 now has a native rows-8 owner: single direct launch, no partition.
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 8)
    assert calls == [8]
    calls.clear()
    # rows > 8 chunk at the default 8.
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 12)
    assert calls == [8, 4]
    calls.clear()
    # Env overrides remain real rollback controls even when the primitive can
    # execute the complete row count directly.
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK", "6")
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 12)
    assert calls == [6, 6]
    calls.clear()
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 8)
    assert calls == [6, 2]
    calls.clear()
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK", "4")
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 8)
    assert calls == [4, 4]


def test_verify_lm_head_rowtile_chunked_honors_gfx1100_package_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=64, vocab_size=128, backend="hip_gfx1100")
    calls: list[int] = []

    def fake_rowtile(self, hidden_ptr, out_ptr, rows, **kwargs):
        calls.append(int(rows))
        return True

    monkeypatch.delenv("HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK", raising=False)
    monkeypatch.setattr(
        runner_mod.Qwen35GGUFResidentSession,
        "_verify_lm_head_rowtile",
        fake_rowtile,
    )
    monkeypatch.setattr(
        runner_mod.Qwen35GGUFResidentSession,
        "_verify_lm_head_rowtile_max_rows",
        lambda self: 8,
    )

    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 8)
    assert calls == [6, 2]


def test_verify_lm_head_rowtile_chunked_falls_back_when_chunk_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=64, vocab_size=128, backend="hip_gfx1100")

    def fake_rowtile(self, hidden_ptr, out_ptr, rows, *, stream=0, runtime=None):
        return False

    monkeypatch.setattr(runner_mod.Qwen35GGUFResidentSession, "_verify_lm_head_rowtile", fake_rowtile)
    monkeypatch.setattr(
        runner_mod.Qwen35GGUFResidentSession,
        "_verify_lm_head_rowtile_max_rows",
        lambda self: 6,
    )

    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 12) is False


def test_verify_lm_head_rowtile_resolves_planar_qmicro_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_key = "gguf_q6_k_t16_qmicro_planar_v1"
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=quant_key, quant_key=quant_key),
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=0x2200))
        if name == "tiles"
        else (_ for _ in ()).throw(KeyError(name)),
    )
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        backend="hip_gfx1100",
        hidden_size=5120,
        vocab_size=248320,
        weights=SimpleNamespace(root=lambda slot: weight),
    )
    runtime = SimpleNamespace()
    expected_key = runner_mod.KernelKey(
        "hip_gfx1100",
        "linear",
        quant_key,
        "t16_gemv_rowtile_bf16_f32_out",
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(runner_mod, "is_registered", lambda key: key == expected_key)

    def fake_resolve(**kwargs):
        assert runner_mod.KernelKey(**kwargs) == expected_key

        def kernel(*args, **kernel_kwargs):
            calls.append((args, kernel_kwargs))

        setattr(kernel, "_hipengine_max_rows", 8)
        return kernel

    monkeypatch.setattr(runner_mod, "resolve", fake_resolve)

    handled = session._verify_lm_head_rowtile(
        0x1000,
        0x2000,
        8,
        stream=7,
        runtime=runtime,
    )

    assert handled is True
    assert session._verify_lm_head_rowtile_max_rows() == 8
    assert calls == [
        (
            (0x1000, 0x2200, 0x2000, 8, 5120, 248320),
            {"stream": 7, "runtime": runtime},
        )
    ]


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
        1,
        stream=7,
        runtime=session.runtime,
    )

    assert launched is True
    assert calls[0] == ("quant", (0x9000, 0x1000, 1, 64), {"stream": 7, "runtime": session.runtime})
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
        1,
        64,
        128,
    )
    assert calls[1][1][11] == 0
    assert calls[1][2] == {"stream": 7, "library": session._q6_pack8_library, "runtime": session.runtime}


def test_verify_lm_head_q6_top1_dp4a_rejects_multirow_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session_with_lm_head_x8()
    monkeypatch.setattr(
        runner_mod,
        "_gguf_verify_lm_head_q6_top1_dp4a_enabled",
        lambda: True,
    )

    assert (
        runner_mod.Qwen35GGUFResidentSession._verify_lm_head_q6_top1_dp4a(
            session,
            0x9000,
            2,
            stream=7,
            runtime=session.runtime,
        )
        is False
    )


def test_verify_lm_head_q6_top1_dp4a_quantizes_f32_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session_with_lm_head_x8()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(runner_mod, "_gguf_verify_lm_head_q6_top1_dp4a_enabled", lambda: True)
    monkeypatch.setattr(
        runner_mod,
        "gguf_q4_k_quantize_bf16_q8_1",
        lambda *args, **kwargs: pytest.fail("BF16 quantizer should not be used for FP32 hidden rows"),
    )
    monkeypatch.setattr(
        runner_mod,
        "gguf_q4_k_quantize_f32_q8_1",
        lambda *args, **kwargs: calls.append(("quant_f32", args, kwargs)),
    )
    monkeypatch.setattr(
        runner_mod,
        "gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_gather_f32",
        lambda *args, **kwargs: calls.append(("top1", args, kwargs)),
    )

    launched = runner_mod.Qwen35GGUFResidentSession._verify_lm_head_q6_top1_dp4a(
        session,
        0x9800,
        1,
        activation_dtype=runner_mod.GGUF_ACTIVATION_F32,
        stream=5,
        runtime=session.runtime,
    )

    assert launched is True
    assert calls[0] == ("quant_f32", (0x9800, 0x1000, 1, 64), {"stream": 5, "runtime": session.runtime})
    assert calls[1][0] == "top1"


def test_verify_lm_head_q6_top1_dp4a_requires_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session_with_lm_head_x8()
    session.runner.weights.root = lambda slot: SimpleNamespace(allocation=lambda name="raw": (_ for _ in ()).throw(KeyError(name)))

    monkeypatch.setattr(runner_mod, "_gguf_verify_lm_head_q6_top1_dp4a_enabled", lambda: True)

    with pytest.raises(RuntimeError, match="LM_HEAD_Q6_X8_SIDECAR"):
        runner_mod.Qwen35GGUFResidentSession._verify_lm_head_q6_top1_dp4a(
            session,
            0x9000,
            1,
            stream=7,
            runtime=session.runtime,
        )
