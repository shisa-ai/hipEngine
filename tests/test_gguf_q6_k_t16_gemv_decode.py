"""Correctness fixtures for dense GGUF Q6_K T16 GEMV decode (P9.H3)."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_t16_gemv as t16_mod
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_gemv_decode_bf16_f32_out,
    gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1,
    plan_gguf_q6_k_t16_gemv_build,
    register_gguf_q6_k_t16_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16
from tests._gguf_synthetic_weights import make_q6_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def q6_t16_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q6_k_t16_gemv(load=True)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _run_single(fn, x, tiles, rows, in_features, out_features, out_dtype, library):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    w_buf = malloc(tiles.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(tiles), tiles.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(x_buf.ptr, w_buf.ptr, out_buf.ptr, rows, in_features, out_features, library=library)
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, w_buf, out_buf):
            free(b)


def test_p9_h3_q6_t16_registry_key_resolves() -> None:
    register_gguf_q6_k_t16_gemv_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_v1",
        variant="t16_gemv_decode_bf16_f32_out",
    ) is not None
    assert resolve(
        backend="hip_gfx1100",
        layer="linear+argmax",
        quant="gguf_q6_k_t16_v1",
        variant="proposal_top1_exact_bf16",
    ) is not None


def test_q6_t16_proposal_top1_adapter_uses_tiles_and_half_vocab_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}
    runtime = object()
    t16_library = object()
    pack8_library = object()
    weight = SimpleNamespace(
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=0xA000))
        if name == "tiles"
        else (_ for _ in ()).throw(KeyError(name))
    )

    monkeypatch.setattr(
        t16_mod,
        "gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1",
        lambda *args, **kwargs: calls.__setitem__("stage1", (args, kwargs)),
    )
    monkeypatch.setattr(
        t16_mod,
        "gguf_q6_k_pack8_top1_stage2_gather_f32",
        lambda *args, **kwargs: calls.__setitem__("stage2", (args, kwargs)),
        raising=False,
    )

    t16_mod.gguf_q6_k_t16_proposal_top1_exact_bf16(
        weight,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        1,
        512,
        1024,
        stream=7,
        libraries={"q6_t16": t16_library, "q6_pack8": pack8_library},
        runtime=runtime,
    )

    assert calls["stage1"] == (
        (0x1000, 0xA000, 0x2000, 0x3000, 0x4000, 512, 1024),
        {"stream": 7, "library": t16_library, "runtime": runtime},
    )
    assert calls["stage2"] == (
        (0x3000, 0x4000, 0x5000, 0x6000, None, None, 1, 64, 0, 1024),
        {"stream": 7, "library": pack8_library, "runtime": runtime},
    )


def test_p9_h3_q6_t16_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_q6_k_t16_gemv_build()
    assert plan.output_path.name == "gguf_q6_k_t16_gemv.so"
    assert plan.sources[0].name == "gguf_q6_k_t16_gemv.hip"


def test_p9_h3_q6_t16_wrappers_validate_args() -> None:
    with pytest.raises(ValueError, match="positive multiple of 256"):
        gguf_q6_k_t16_gemv_decode_bf16_f32_out(0, 0, 0, 1, 128, 16)
    with pytest.raises(ValueError, match="positive multiple of 16"):
        gguf_q6_k_t16_gemv_decode_bf16_f32_out(0, 0, 0, 1, 256, 8)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 16),
        (1, 512, 256),
        (2, 512, 128),
    ],
)
def test_p9_h3_q6_t16_bf16_f32_matches_cpu_oracle(rows, in_features, out_features, q6_t16_library) -> None:
    rng = np.random.default_rng(rows * 23 + in_features + out_features)
    qweight = make_q6_k_weight(out_features, in_features)
    tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual = _run_single(
        gguf_q6_k_t16_gemv_decode_bf16_f32_out,
        x_bf16,
        tiles,
        rows,
        in_features,
        out_features,
        np.float32,
        q6_t16_library,
    )

    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q6_K)
    np.testing.assert_allclose(actual, expected, atol=1.0e-3, rtol=5.0e-3)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_t16_producer_top1_preserves_logits_ties_and_control(q6_t16_library) -> None:
    from hipengine.kernels.hip_gfx1100.linear.lm_head import (
        argmax_tile_stage2_i32_publish_control,
        build_lm_head,
    )

    rows, in_features, out_features = 1, 512, 256
    rng = np.random.default_rng(0x1151)
    qweight = make_q6_k_weight(out_features, in_features)
    # Force an exact equal-logit pair in separate producer tiles.
    qweight[80] = qweight[17]
    tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    tile_count = out_features // 16

    buffers = []
    try:
        x_d = malloc(x_bf16.nbytes)
        tiles_d = malloc(tiles.nbytes)
        logits_d = malloc(out_features * 4)
        tile_values_d = malloc(tile_count * 4)
        tile_indices_d = malloc(tile_count * 4)
        out_index_d = malloc(8)
        out_value_d = malloc(4)
        next_token_d = malloc(8)
        scratch_position_d = malloc(8)
        kv_position_d = malloc(8)
        buffers.extend(
            (
                x_d,
                tiles_d,
                logits_d,
                tile_values_d,
                tile_indices_d,
                out_index_d,
                out_value_d,
                next_token_d,
                scratch_position_d,
                kv_position_d,
            )
        )
        copy_host_to_device(x_d, host_array_ptr(x_bf16), x_bf16.nbytes)
        copy_host_to_device(tiles_d, host_array_ptr(tiles), tiles.nbytes)

        expected_logits = _run_single(
            gguf_q6_k_t16_gemv_decode_bf16_f32_out,
            x_bf16,
            tiles,
            rows,
            in_features,
            out_features,
            np.float32,
            q6_t16_library,
        )
        gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1(
            x_d.ptr,
            tiles_d.ptr,
            logits_d.ptr,
            tile_values_d.ptr,
            tile_indices_d.ptr,
            in_features,
            out_features,
            library=q6_t16_library,
        )
        argmax_tile_stage2_i32_publish_control(
            tile_values_d.ptr,
            tile_indices_d.ptr,
            out_index_d.ptr,
            out_value_d.ptr,
            next_token_d.ptr,
            scratch_position_d.ptr,
            kv_position_d.ptr,
            tile_count,
            513,
            library=build_lm_head(load=True),
        )

        actual_logits = np.empty(out_features, dtype=np.float32)
        out_index = np.empty(1, dtype=np.int64)
        out_value = np.empty(1, dtype=np.float32)
        next_token = np.empty(1, dtype=np.int64)
        scratch_position = np.empty(1, dtype=np.int64)
        kv_position = np.empty(1, dtype=np.int64)
        for host, device in (
            (actual_logits, logits_d),
            (out_index, out_index_d),
            (out_value, out_value_d),
            (next_token, next_token_d),
            (scratch_position, scratch_position_d),
            (kv_position, kv_position_d),
        ):
            copy_device_to_host(host_array_ptr(host), device, host.nbytes)

        np.testing.assert_array_equal(actual_logits, expected_logits[0])
        expected_index = int(np.argmax(expected_logits[0]))
        assert int(out_index[0]) == expected_index
        assert out_value[0].view(np.uint32) == expected_logits[0, expected_index].view(np.uint32)
        assert int(next_token[0]) == expected_index
        assert int(scratch_position[0]) == 513
        assert int(kv_position[0]) == 513
    finally:
        for buffer in reversed(buffers):
            free(buffer)
