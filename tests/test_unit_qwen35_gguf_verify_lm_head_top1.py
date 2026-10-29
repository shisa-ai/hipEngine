from __future__ import annotations

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
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


def test_verify_lm_head_rowtile_chunked_ignores_retired_env(
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
    # Retired overrides cannot replace the backend's chunk policy.
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK", "6")
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 12)
    assert calls == [8, 4]
    calls.clear()
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 8)
    assert calls == [8]
    calls.clear()
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK", "4")
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 8)
    assert calls == [8]


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

    # The W7900 package advertises its native R8 owner.
    assert session._verify_lm_head_rowtile_chunked(0x100000, 0x200000, 8)
    assert calls == [8]


@pytest.mark.parametrize("limit", [1, 9])
def test_verify_lm_head_rejects_invalid_backend_chunk_policy(monkeypatch, limit) -> None:
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(backend="hip_gfx1100")
    session._verify_lm_head_rowtile_max_rows = lambda: 8
    monkeypatch.setattr(runner_mod, "backend_package_capability", lambda *args: limit)
    with pytest.raises(ValueError, match="capability must be in"):
        session._verify_lm_head_rowtile_chunked(1, 2, 8)


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


@pytest.mark.parametrize("rows", [2, 3, 7, 9])
def test_verify_lm_head_q6_top1_dp4a_rejects_multirow_shortcut(
    monkeypatch: pytest.MonkeyPatch, rows: int,
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
            rows,
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


@pytest.mark.parametrize("rows,enqueue_only", [(1, False), (2, False), (3, False), (7, False), (9, False), (2, True), (3, True), (7, True)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("capture_logits", [False, True])
def test_block_verifier_lm_head_dtype_matches_singleton_shortcut(
    monkeypatch: pytest.MonkeyPatch, rows: int, enqueue_only: bool, enabled: bool,
    capture_logits: bool,
) -> None:
    """Run the real block caller and head dispatcher with host-only device stubs."""
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    for name, value in vars(_session_with_lm_head_x8()).items():
        setattr(session, name, value)
    session.runner.weights.config = SimpleNamespace(
        layer_types=(), is_moe=False, rms_norm_eps=1e-6,
    )
    lm_head = session.runner.weights.root("lm_head")
    session.runner.weights.root = lambda slot: (
        lm_head if slot == "lm_head"
        else SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0x2300)))
    )
    session.runtime = SimpleNamespace(
        memcpy_async=lambda *args: None,
        device_synchronize=lambda: None,
        stream_synchronize=lambda stream: None,
    )
    session._position = 0
    session.scratch = SimpleNamespace(
        max_positions=128, hidden_seed_fp32=SimpleNamespace(ptr=0x7000),
        position_host=[0], context_host=[1],
        position_buf=SimpleNamespace(ptr=0x7100), context_buf=SimpleNamespace(ptr=0x7200),
    )
    for name, ptr in {
        "_prefill_token_buf": 0x3000, "_prefill_hidden_a": 0x4000,
        "_prefill_hidden_b": 0x5000, "_verify_hidden_seed_buf": 0x6000,
        "_verify_token_ids_i64": 0x8000, "_verify_token_counter_i64": 0x8100,
        "_verify_logits_buf": 0x9000, "_lm_out_index": 0xA000,
    }.items():
        setattr(session, name, SimpleNamespace(ptr=ptr, nbytes=16 * 64 * 4))
    session._bulk_prefill_scratch = SimpleNamespace(
        rows=16, for_chunk=lambda *args, **kwargs: SimpleNamespace(norm=SimpleNamespace(ptr=0xB000)),
    )
    session._verify_hidden_f32_a = None
    session._verify_hidden_f32_b = None
    session._runtime_state_library = object()
    session._lm_head_library = object()
    session._lm_head_threads = 256
    session.use_wmma_prefill = False
    session.use_gemv_decode = True
    session._ensure_verify_block_buffers = lambda *args, **kwargs: None
    session._ensure_verify_lm_head_buffers = lambda *args, **kwargs: None
    session._device_token_embedding_weight = lambda **kwargs: object()
    session._iq_dense_mmq_verify_context = lambda **kwargs: nullcontext()
    calls = []
    session._verify_lm_head_rowtile_chunked = lambda *args, **kwargs: calls.append(("rowtile", args)) or True
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_ROW_LM_HEAD", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A", str(int(enabled)))
    monkeypatch.setattr(runner_mod, "_gguf_verify_f32_residual_enabled", lambda: False)
    for name in (
        "copy_host_array_to_device", "launch_gguf_embedding",
        "gguf_rmsnorm_bf16_f32_weight", "gguf_rmsnorm_bf16_f32_weight_out_f32",
        "set_decode_position_i64", "set_i64_scalar", "copy_i32_to_i64",
    ):
        monkeypatch.setattr(runner_mod, name, lambda *args, **kwargs: None)
    readbacks = []

    def readback(ptr, buf, nbytes, **kwargs):
        readbacks.append((buf.ptr, nbytes, kwargs))
        ctypes.memset(ptr, 0, nbytes)

    monkeypatch.setattr(runner_mod, "copy_device_to_host", readback)
    for name, label in (
        ("gguf_q4_k_quantize_f32_q8_1", "quant_f32"),
        ("gguf_q4_k_quantize_bf16_q8_1", "quant_bf16"),
        ("gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_gather_f32", "top1"),
        ("argmax_f32_rows_i32", "argmax"),
    ):
        monkeypatch.setattr(runner_mod, name, lambda *args, _label=label, **kwargs: calls.append((_label, args)))

    if enqueue_only and capture_logits:
        with pytest.raises(ValueError, match="does not support host diagnostic outputs"):
            session.verify_target_block(
                [1] * rows, stream=7, capture_lm_head_logits=True,
                _enqueue_only=True, _pre_staged_token_ids_ptr=0xC000, _target_top1_i32_ptr=0xD000,
            )
        assert calls == []
        assert readbacks == []
        assert session.position == 0
        return

    result = session.verify_target_block(
        [1] * rows, stream=7 if enqueue_only else 0,
        capture_lm_head_logits=capture_logits,
        _enqueue_only=enqueue_only,
        _pre_staged_token_ids_ptr=0xC000 if enqueue_only else None,
        _target_top1_i32_ptr=0xD000 if enqueue_only else None,
    )

    direct = enabled and rows == 1
    assert [label for label, args in calls] == (["quant_f32", "top1"] if direct else ["rowtile", "argmax"])
    assert calls[0][1][0] == (0x6000 if direct else 0xB000)
    assert session._last_packed_lm_head_decode_path == ("direct_top1_rows" if direct else "q6_rowtile_f32_logits")
    assert session._last_packed_sampler_decode_path == ("fused_top1_i32_rows" if direct else "argmax_i32_rows")
    if enqueue_only:
        assert result is None
        assert session.position == 0
        assert readbacks == []
    else:
        assert result.token_ids == [0] * rows
        np.testing.assert_array_equal(result.hidden_seeds, np.zeros((rows, 64), dtype=np.float32))
        logits_readbacks = [item for item in readbacks if item[0] == 0x9000]
        if direct or not capture_logits:
            assert result.lm_head_logits_f32 is None
            assert logits_readbacks == []
        else:
            np.testing.assert_array_equal(result.lm_head_logits_f32, np.zeros((rows, 128), dtype=np.float32))
            assert result.lm_head_logits_f32.flags.c_contiguous
            assert logits_readbacks == [(0x9000, rows * 128 * 4, {"runtime": session.runtime})]
        assert session.position == rows


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("repack", [False, True])
@pytest.mark.parametrize("runtime_top1", [False, True])
def test_verifier_x8_sidecar_plan_is_independent_of_runtime_and_draft_controls(
    monkeypatch, sidecar: bool, repack: bool, runtime_top1: bool,
) -> None:
    from hipengine.loading.gguf import GGUFTensorInfo
    from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_weight_spec
    from hipengine.quant.gguf import GGMLQuantizationType

    monkeypatch.setenv("HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR", str(int(sidecar)))
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A", str(int(runtime_top1)))
    monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE", "x8")
    tensor = GGUFTensorInfo(
        name="output.weight", shape=(248_320, 5_120), ggml_shape=(5_120, 248_320),
        ggml_type=int(GGMLQuantizationType.Q6_K), ggml_type_name="Q6_K",
        n_elements=248_320 * 5_120, nbytes=248_320 * 20 * 210,
        offset=0, data_offset=0, byte_shape=(248_320, 20 * 210),
    )
    spec = plan_qwen35_gguf_weight_spec(
        "root.lm_head", tensor, decode_repack=repack, dense_q6_qmicro_planar=True,
    )
    if not repack:
        assert spec.quant_key == "gguf_q6_k"
        assert spec.allocation_names == ("raw",)
    elif sidecar:
        assert spec.quant_key == "gguf_q6_k_t16_v1"
        assert spec.allocation_names == ("tiles", "x8")
    else:
        assert spec.quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
        assert spec.allocation_names == ("tiles",)


def test_gfx1100_lm_head_rowtile_width_bound_comes_from_the_resolved_quant() -> None:
    """The width-8 root head depends on which Q6 quant the lm_head resolves to.

    ``_verify_lm_head_rowtile_max_rows()`` reads ``_hipengine_max_rows`` off the
    registered rowtile primitive, and the effective chunk is
    ``min(GGUF_Q6_LM_HEAD_MAX_CHUNK, that bound)``. gfx1100 registers the
    variant under two Q6 quants with different bounds, so the package constant
    alone does not establish that a physical width-8 verifier head is one
    launch. Pin all three values together.
    """

    from hipengine.kernels.backends import backend_package_capability
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        register_gguf_q6_k_t16_gemv_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_gguf_q6_k_t16_gemv_kernels()

    bounds = {}
    for quant in ("gguf_q6_k_t16_v1", "gguf_q6_k_t16_qmicro_planar_v1"):
        fn = resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant=quant,
            variant="t16_gemv_rowtile_bf16_f32_out",
        )
        bounds[quant] = int(getattr(fn, "_hipengine_max_rows", 0))

    assert bounds["gguf_q6_k_t16_v1"] == 6
    assert bounds["gguf_q6_k_t16_qmicro_planar_v1"] == 8

    package_chunk = int(
        backend_package_capability("hip_gfx1100", "GGUF_Q6_LM_HEAD_MAX_CHUNK", 0)
    )
    assert package_chunk == 8

    # Planar resolution: one launch at width 8. Non-planar resolution: 6+2.
    assert min(package_chunk, bounds["gguf_q6_k_t16_qmicro_planar_v1"]) == 8
    assert runner_mod._small_b_rowtile_chunks(8, max_chunk=6) == (6, 2)
