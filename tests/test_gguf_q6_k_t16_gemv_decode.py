"""Correctness fixtures for dense GGUF Q6_K T16 GEMV decode (P9.H3)."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.specdec2_scope import (
    physical_exact_rowtiles_session,
    q6_t16_physical_mixed_rowtiles_session,
    q6_t16_physical_rowtile_session,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_t16_gemv as t16_mod
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_gemv_decode_bf16_f32_out,
    gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1,
    gguf_q6_k_t16_gemv_rowtile_bf16_f32_out,
    gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out,
    gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out,
    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out,
    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows6_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows8_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out,
    plan_gguf_q6_k_t16_gemv_build,
    register_gguf_q6_k_t16_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import (
    repack_gguf_q6_k_tile16,
    repack_gguf_q6_k_tile16_qmicro_planar,
)
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


@pytest.fixture(scope="module")
def q6_t16_selected_prefill_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_k_t16_selected_prefill(load=True)


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


def _run_residual(fn, x, tiles, residual, rows, in_features, out_features, library):
    buffers = []
    try:
        device = []
        for value in (x, tiles, residual):
            buffer = malloc(value.nbytes)
            buffers.append(buffer)
            device.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(value), value.nbytes)
        out = np.zeros((rows, out_features), dtype=np.uint16)
        out_buffer = malloc(out.nbytes)
        buffers.append(out_buffer)
        fn(
            device[0].ptr,
            device[1].ptr,
            device[2].ptr,
            out_buffer.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buffer, out.nbytes)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer)


@pytest.mark.parametrize(
    "chunk_rows,total_rows,grouped",
    [
        (
            6,
            12,
            gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows6_bf16_bf16_out,
        ),
        (
            8,
            16,
            gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows8_bf16_bf16_out,
        ),
        (
            8,
            24,
            gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows8_bf16_bf16_out,
        ),
    ],
)
def test_q6_planar_grouped_rowtiles_match_repeated_chunk_bits(
    chunk_rows: int,
    total_rows: int,
    grouped,
    q6_t16_library,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x6A20 + total_rows)
    raw = make_q6_k_weight(out_features, in_features)
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(total_rows, in_features)).astype(np.float32)
    )
    tiles = repack_gguf_q6_k_tile16_qmicro_planar(raw[None, ...]).tiles
    repeated = np.concatenate(
        [
            _run_single(
                gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out,
                x[row_base : row_base + chunk_rows],
                tiles,
                chunk_rows,
                in_features,
                out_features,
                np.uint16,
                q6_t16_library,
            )
            for row_base in range(0, total_rows, chunk_rows)
        ],
        axis=0,
    )
    candidate = _run_single(
        grouped,
        x,
        tiles,
        total_rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )

    np.testing.assert_array_equal(candidate, repeated)


def test_q6_planar_decode_uses_request_scoped_physical_rowtile(monkeypatch) -> None:
    symbols: list[str] = []
    monkeypatch.setattr(
        t16_mod,
        "_launch",
        lambda symbol, *args, **kwargs: symbols.append(symbol),
    )

    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
        1, 2, 3, 6, 5_120, 10_240
    )
    with q6_t16_physical_rowtile_session(True):
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
            1, 2, 3, 6, 5_120, 10_240
        )
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
            1, 2, 3, 8, 5_120, 10_240
        )
        with physical_exact_rowtiles_session(True):
            gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
                1, 2, 3, 8, 5_120, 10_240
            )
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
            1, 2, 3, 3, 5_120, 10_240
        )

    assert symbols == [
        t16_mod._Q6_T16_QMICRO_PLANAR_BF16_BF16,
        t16_mod._Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_BF16,
        t16_mod._Q6_T16_QMICRO_PLANAR_BF16_BF16,
        t16_mod._Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_BF16,
        t16_mod._Q6_T16_QMICRO_PLANAR_BF16_BF16,
    ]


def test_q6_planar_mixed_r8_chunks_are_candidate_and_shape_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setenv(
        "HIPENGINE_GGUF_Q6_T16_GROUPED_TARGET_ROWTILES",
        "0",
    )
    monkeypatch.setattr(
        t16_mod,
        "_launch",
        lambda _symbol, x, _tiles, out, rows, *_args, **_kwargs: calls.append(
            (int(rows), int(x), int(out))
        ),
    )
    launch = gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out
    x_ptr = 0x100_000
    out_ptr = 0x200_000

    with q6_t16_physical_rowtile_session(True):
        launch(x_ptr, 2, out_ptr, 24, 5_120, 10_240)
        assert [call[0] for call in calls] == [6, 6, 6, 6]
        calls.clear()

        with q6_t16_physical_mixed_rowtiles_session(True):
            launch(x_ptr, 2, out_ptr, 18, 5_120, 10_240)
            assert [call[0] for call in calls] == [6, 6, 6]
            calls.clear()

            launch(x_ptr, 2, out_ptr, 24, 5_120, 10_240)
            assert [call[0] for call in calls] == [8, 8, 8]
            assert [call[1] for call in calls] == [
                x_ptr,
                x_ptr + 8 * 5_120 * 2,
                x_ptr + 16 * 5_120 * 2,
            ]
            assert [call[2] for call in calls] == [
                out_ptr,
                out_ptr + 8 * 10_240 * 2,
                out_ptr + 16 * 10_240 * 2,
            ]
            calls.clear()

            launch(x_ptr, 2, out_ptr, 30, 5_120, 1_024)
            assert [call[0] for call in calls] == [8, 8, 8, 6]
            calls.clear()

            launch(x_ptr, 2, out_ptr, 36, 17_408, 5_120)
            assert [call[0] for call in calls] == [8, 8, 8, 6, 6]
            calls.clear()

            launch(x_ptr, 2, out_ptr, 24, 512, 256)
            assert [call[0] for call in calls] == [6, 6, 6, 6]


def test_q6_planar_grouped_mixed_chunks_default_on_for_identical_prefix_and_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int, int]] = []
    monkeypatch.delenv(
        "HIPENGINE_GGUF_Q6_T16_GROUPED_TARGET_ROWTILES",
        raising=False,
    )
    monkeypatch.setattr(
        t16_mod,
        "_launch",
        lambda symbol, x, _tiles, out, rows, *_args, **_kwargs: calls.append(
            (str(symbol), int(rows), int(x), int(out))
        ),
    )
    launch = gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out
    x_ptr = 0x100_000
    out_ptr = 0x200_000
    grouped8 = (
        "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_"
        "grouped_rows8_bf16_bf16_out"
    )
    grouped6 = (
        "hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_"
        "grouped_rows6_bf16_bf16_out"
    )

    with (
        q6_t16_physical_rowtile_session(True),
        q6_t16_physical_mixed_rowtiles_session(True),
    ):
        launch(x_ptr, 2, out_ptr, 24, 5_120, 10_240)
        assert calls == [(grouped8, 24, x_ptr, out_ptr)]
        calls.clear()

        launch(x_ptr, 2, out_ptr, 30, 5_120, 1_024)
        assert calls == [
            (grouped8, 24, x_ptr, out_ptr),
            (
                t16_mod._Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_BF16,
                6,
                x_ptr + 24 * 5_120 * 2,
                out_ptr + 24 * 1_024 * 2,
            ),
        ]
        calls.clear()

        launch(x_ptr, 2, out_ptr, 36, 17_408, 5_120)
        assert calls == [
            (grouped8, 24, x_ptr, out_ptr),
            (
                grouped6,
                12,
                x_ptr + 24 * 17_408 * 2,
                out_ptr + 24 * 5_120 * 2,
            ),
        ]
        calls.clear()

        launch(x_ptr, 2, out_ptr, 28, 5_120, 10_240)
        assert calls == [
            (grouped8, 24, x_ptr, out_ptr),
            (
                t16_mod._Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_BF16_BF16,
                4,
                x_ptr + 24 * 5_120 * 2,
                out_ptr + 24 * 10_240 * 2,
            ),
        ]
        calls.clear()

        launch(x_ptr, 2, out_ptr, 32, 17_408, 5_120)
        assert calls == [(grouped8, 32, x_ptr, out_ptr)]


def test_q6_planar_grouped_mixed_chunks_have_explicit_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setenv(
        "HIPENGINE_GGUF_Q6_T16_GROUPED_TARGET_ROWTILES",
        "0",
    )
    monkeypatch.setattr(
        t16_mod,
        "_launch",
        lambda _symbol, _x, _tiles, _out, rows, *_args, **_kwargs: calls.append(
            int(rows)
        ),
    )

    with (
        q6_t16_physical_rowtile_session(True),
        q6_t16_physical_mixed_rowtiles_session(True),
    ):
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
            1, 2, 3, 24, 5_120, 10_240
        )

    assert calls == [8, 8, 8]


def test_q6_planar_exact_prefill_selects_measured_gfx1100_bands(monkeypatch) -> None:
    calls: list[str] = []
    for name, label in (
        ("gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out", "parent"),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_bf16_out",
            "row64",
        ),
        (
            "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_gfx1100_bf16_bf16_out",
            "shared256",
        ),
    ):
        monkeypatch.setattr(
            t16_mod,
            name,
            lambda *args, _label=label, **kwargs: calls.append(_label),
        )
    monkeypatch.delenv("HIPENGINE_GGUF_Q6_PLANAR_EXACT_PREFILL", raising=False)
    monkeypatch.setattr(t16_mod, "_Q6_PLANAR_EXACT_PREFILL_RESOLVED", None)

    for rows, shape in (
        (32, (17_408, 5_120)),
        (33, (17_408, 5_120)),
        (64, (17_408, 5_120)),
        (65, (17_408, 5_120)),
        (128, (17_408, 5_120)),
        (129, (17_408, 5_120)),
        (511, (17_408, 5_120)),
        (512, (17_408, 5_120)),
        (35, (5_120, 10_240)),
        (20, (17_408, 5_120)),
        (28, (17_408, 5_120)),
    ):
        t16_mod.gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1100_bf16_bf16_out(
            1, 2, 3, rows, *shape
        )
    assert calls == [
        "parent",
        "row64",
        "row64",
        "row64",
        "row64",
        "shared256",
        "shared256",
        "parent",
        "parent",
        "parent",
        "parent",
    ]

    calls.clear()
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_PLANAR_EXACT_PREFILL", "0")
    monkeypatch.setattr(t16_mod, "_Q6_PLANAR_EXACT_PREFILL_RESOLVED", None)
    t16_mod.gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1100_bf16_bf16_out(
        1, 2, 3, 35, 17_408, 5_120
    )
    assert calls == ["parent"]


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
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_v1",
        variant="t16_gemv_rowtile_bf16_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_gemv_rowtile_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_v1",
        variant="t16_gemv_rowtile_col8_bf16_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_v1",
        variant="t16_gemv_rowtile_col8_bf16_f32_out",
    ) is t16_mod.gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_decode_bf16_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_rowtile_bf16_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_rowtile_col8_bf16_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_rowtile_col8_grouped_rows8_bf16_bf16_out",
    ) is gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows8_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_rowtile_col8_grouped_rows6_bf16_bf16_out",
    ) is gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows6_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear+residual",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_rowtile_bf16_residual_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_residual_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_wmma_prefill_bf16_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1100_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_wmma_prefill_single_wave_bf16_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_wmma_prefill_shared4_row64_bf16_bf16_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_decode_bf16_f32_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_rowtile_bf16_f32_out",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear+argmax",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="t16_gemv_decode_bf16_f32_top1_stage1",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1
    assert resolve(
        backend="hip_gfx1100",
        layer="linear+argmax",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="proposal_top1_exact_bf16",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_proposal_top1_exact_bf16
    assert resolve(
        backend="hip_gfx1100",
        layer="linear+argmax",
        quant="gguf_q6_k_t16_qmicro_planar_v1",
        variant="proposal_top1_mapped_bf16",
    ) is t16_mod.gguf_q6_k_t16_qmicro_planar_proposal_top1_mapped_bf16
    dense_wmma = getattr(
        t16_mod,
        "gguf_q6_k_t16_wmma_prefill_bf16_bf16_out",
        None,
    )
    assert dense_wmma is not None
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k_t16_v1",
        variant="t16_wmma_prefill_bf16_bf16_out",
    ) is dense_wmma


@pytest.mark.parametrize(
    ("adapter", "stage1_name"),
    [
        (
            t16_mod.gguf_q6_k_t16_proposal_top1_exact_bf16,
            "gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1",
        ),
        (
            t16_mod.gguf_q6_k_t16_qmicro_planar_proposal_top1_exact_bf16,
            "gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1",
        ),
    ],
)
def test_q6_t16_proposal_top1_adapter_uses_tiles_and_half_vocab_blocks(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
    stage1_name,
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
        stage1_name,
        lambda *args, **kwargs: calls.__setitem__("stage1", (args, kwargs)),
    )
    monkeypatch.setattr(
        t16_mod,
        "gguf_q6_k_pack8_top1_stage2_gather_f32",
        lambda *args, **kwargs: calls.__setitem__("stage2", (args, kwargs)),
        raising=False,
    )

    adapter(
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


def test_q6_t16_mapped_proposal_top1_maps_compact_winner(
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
        "gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1",
        lambda *args, **kwargs: calls.__setitem__("stage1", (args, kwargs)),
    )
    monkeypatch.setattr(
        t16_mod,
        "gguf_q6_k_pack8_top1_stage2_gather_mapped_f32",
        lambda *args, **kwargs: calls.__setitem__("stage2", (args, kwargs)),
    )

    t16_mod.gguf_q6_k_t16_qmicro_planar_proposal_top1_mapped_bf16(
        weight,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
        1,
        512,
        256,
        1024,
        stream=7,
        libraries={"q6_t16": t16_library, "q6_pack8": pack8_library},
        runtime=runtime,
    )

    assert calls["stage1"] == (
        (0x1000, 0xA000, 0x2000, 0x3000, 0x4000, 512, 256),
        {"stream": 7, "library": t16_library, "runtime": runtime},
    )
    assert calls["stage2"] == (
        (0x3000, 0x4000, 0x7000, 0x5000, 0x6000, 1, 16, 256, 1024),
        {"stream": 7, "library": pack8_library, "runtime": runtime},
    )


def test_p9_h3_q6_t16_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_q6_k_t16_gemv_build()
    assert plan.output_path.name == "gguf_q6_k_t16_gemv.so"
    assert plan.sources[0].name == "gguf_q6_k_t16_gemv.hip"


def test_p9_h3_q6_t16_wrappers_validate_args() -> None:
    assert getattr(gguf_q6_k_t16_gemv_rowtile_bf16_f32_out, "_hipengine_max_rows") == 6
    assert (
        getattr(
            gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out,
            "_hipengine_max_rows",
        )
        == 8
    )
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
@pytest.mark.parametrize(
    "candidate,reference,out_dtype",
    [
        (
            gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out,
            t16_mod.gguf_q6_k_t16_gemv_decode_bf16_bf16_out,
            np.uint16,
        ),
        (
            gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out,
            t16_mod.gguf_q6_k_t16_gemv_decode_bf16_f32_out,
            np.float32,
        ),
    ],
)
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_q6_t16_rowtile_col8_is_bit_exact_to_t16_decode(
    candidate,
    reference,
    out_dtype,
    rows,
    q6_t16_library,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0xC018 + rows)
    qweight = make_q6_k_weight(out_features, in_features)
    tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    expected = _run_single(
        reference,
        x,
        tiles,
        rows,
        in_features,
        out_features,
        out_dtype,
        q6_t16_library,
    )
    actual = _run_single(
        candidate,
        x,
        tiles,
        rows,
        in_features,
        out_features,
        out_dtype,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_t16_qmicro_planar_c1_is_bit_exact_to_legacy(
    q6_t16_library,
) -> None:
    rows, in_features, out_features = 1, 512, 256
    rng = np.random.default_rng(0x6A10)
    qweight = make_q6_k_weight(out_features, in_features)
    legacy_tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    expected = _run_single(
        t16_mod.gguf_q6_k_t16_gemv_decode_bf16_bf16_out,
        x,
        legacy_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    actual = _run_single(
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_q6_t16_qmicro_planar_rowtile_col8_is_bit_exact_to_legacy(
    rows,
    q6_t16_library,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x6A11 + rows)
    qweight = make_q6_k_weight(out_features, in_features)
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    # Reference is the per-row planar decode (bit-exact source of truth); the
    # legacy raw col8 rowtile only covers rows 2-6, so per-row is used for the
    # full 2-8 range.
    expected = _run_single(
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    actual = _run_single(
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_q6_t16_qmicro_planar_down_residual_is_bit_exact(
    rows,
    q6_t16_library,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x6A18 + rows)
    qweight = make_q6_k_weight(out_features, in_features)
    tiles = repack_gguf_q6_k_tile16_qmicro_planar(qweight[None, ...]).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )
    residual = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, out_features)).astype(np.float32)
    )
    projected = _run_single(
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out,
        x,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    expected = _f32_to_bf16_u16(
        _bf16_u16_to_f32(residual) + _bf16_u16_to_f32(projected)
    )
    candidate = _run_residual(
        t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_residual_bf16_out,
        x,
        tiles,
        residual,
        rows,
        in_features,
        out_features,
        q6_t16_library,
    )

    np.testing.assert_array_equal(candidate, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_t16_qmicro_planar_c1_down_residual_is_bit_exact(
    q6_t16_library,
) -> None:
    rows, in_features, out_features = 1, 512, 256
    rng = np.random.default_rng(0x6A19)
    qweight = make_q6_k_weight(out_features, in_features)
    tiles = repack_gguf_q6_k_tile16_qmicro_planar(qweight[None, ...]).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )
    residual = _f32_to_bf16_u16(
        rng.normal(0.0, 0.4, size=(rows, out_features)).astype(np.float32)
    )
    projected = _run_single(
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
        x,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    expected = _f32_to_bf16_u16(
        _bf16_u16_to_f32(residual) + _bf16_u16_to_f32(projected)
    )
    candidate = _run_residual(
        t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_residual_bf16_out,
        x,
        tiles,
        residual,
        rows,
        in_features,
        out_features,
        q6_t16_library,
    )

    np.testing.assert_array_equal(candidate, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_t16_qmicro_planar_c1_f32_is_bit_exact_to_legacy(
    q6_t16_library,
) -> None:
    rows, in_features, out_features = 1, 512, 256
    rng = np.random.default_rng(0x6A12)
    qweight = make_q6_k_weight(out_features, in_features)
    legacy_tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    expected = _run_single(
        gguf_q6_k_t16_gemv_decode_bf16_f32_out,
        x,
        legacy_tiles,
        rows,
        in_features,
        out_features,
        np.float32,
        q6_t16_library,
    )
    actual = _run_single(
        t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.float32,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_q6_t16_qmicro_planar_rowtile_col8_f32_is_bit_exact_to_legacy(
    rows,
    q6_t16_library,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x6A13 + rows)
    qweight = make_q6_k_weight(out_features, in_features)
    legacy_tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    expected = _run_single(
        gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out,
        x,
        legacy_tiles,
        rows,
        in_features,
        out_features,
        np.float32,
        q6_t16_library,
    )
    actual = _run_single(
        t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_f32_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.float32,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_q6_t16_qmicro_planar_rowtile_f32_matches_generic_legacy_tree(
    rows,
    q6_t16_library,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x6A16)
    qweight = make_q6_k_weight(out_features, in_features)
    legacy_tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    expected = _run_single(
        t16_mod.gguf_q6_k_t16_gemv_rowtile_bf16_f32_out,
        x,
        legacy_tiles,
        rows,
        in_features,
        out_features,
        np.float32,
        q6_t16_library,
    )
    actual = _run_single(
        getattr(
            t16_mod,
            "gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out",
        ),
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.float32,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)


def test_q6_t16_qmicro_planar_rowtile_f32_rows8_matches_two_rows4_bits(
    q6_t16_library,
) -> None:
    rows, in_features, out_features = 8, 512, 256
    rng = np.random.default_rng(0x6A1608)
    qweight = make_q6_k_weight(out_features, in_features)
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )
    fn = t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out
    control = np.concatenate(
        [
            _run_single(
                fn,
                chunk,
                qmicro_tiles,
                4,
                in_features,
                out_features,
                np.float32,
                q6_t16_library,
            )
            for chunk in np.split(x, 2)
        ],
        axis=0,
    )
    candidate = _run_single(
        fn,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.float32,
        q6_t16_library,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [17, 33])
def test_q6_t16_qmicro_planar_wmma_is_bit_exact_to_legacy_wmma(
    q6_t16_library,
    rows: int,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x6A17)
    qweight = make_q6_k_weight(out_features, in_features)
    legacy_tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    expected = _run_single(
        t16_mod.gguf_q6_k_t16_wmma_prefill_bf16_bf16_out,
        x,
        legacy_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    actual = _run_single(
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [17, 257])
def test_q6_t16_standard_shared4_wmma_is_bit_exact_to_retained_wmma(
    q6_t16_library,
    rows: int,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x3627 if rows == 17 else 0x6A19 + rows)
    qweight = make_q6_k_weight(out_features, in_features)
    tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )
    candidate = getattr(
        t16_mod,
        "gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out",
        None,
    )
    assert candidate is not None

    expected = _run_single(
        t16_mod.gguf_q6_k_t16_wmma_prefill_bf16_bf16_out,
        x,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    actual = _run_single(
        candidate,
        x,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)
    if rows == 17:
        x_ref = _bf16_u16_to_f32(x)
        cpu_expected = gguf_quant_gemv(
            x_ref,
            qweight,
            GGMLQuantizationType.Q6_K,
        )
        actual_f32 = _bf16_u16_to_f32(actual)
        np.testing.assert_allclose(
            actual_f32,
            cpu_expected,
            atol=3.0e-1,
            rtol=1.2e-2,
        )
        kls = [
            _stable_kl(cpu_expected[row], actual_f32[row])
            for row in range(rows)
        ]
        top1 = np.mean(
            np.argmax(cpu_expected, axis=1) == np.argmax(actual_f32, axis=1)
        )
        assert max(kls) <= 0.05
        assert top1 >= 0.90


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", list(range(1, 37)) + [48, 64, 65])
def test_q6_t16_qmicro_planar_shared4_row64_wmma_is_bit_exact_to_retained_wmma(
    q6_t16_library,
    rows: int,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x64A0 + rows)
    qweight = make_q6_k_weight(out_features, in_features)
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )
    candidate = getattr(
        t16_mod,
        "gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_bf16_out",
        None,
    )
    assert candidate is not None

    expected = _run_single(
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    actual = _run_single(
        candidate,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [17, 257])
def test_q6_t16_qmicro_planar_shared4_wmma_is_bit_exact_to_retained_wmma(
    q6_t16_library,
    rows: int,
) -> None:
    in_features, out_features = 512, 256
    rng = np.random.default_rng(0x3627 if rows == 17 else 0x6A18 + rows)
    qweight = make_q6_k_weight(out_features, in_features)
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )

    expected = _run_single(
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    actual = _run_single(
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out,
        x,
        qmicro_tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )

    np.testing.assert_array_equal(actual, expected)
    if rows == 17:
        x_ref = _bf16_u16_to_f32(x)
        cpu_expected = gguf_quant_gemv(
            x_ref,
            qweight,
            GGMLQuantizationType.Q6_K,
        )
        actual_f32 = _bf16_u16_to_f32(actual)
        np.testing.assert_allclose(
            actual_f32,
            cpu_expected,
            atol=3.0e-1,
            rtol=1.2e-2,
        )
        kls = [
            _stable_kl(cpu_expected[row], actual_f32[row])
            for row in range(rows)
        ]
        top1 = np.mean(
            np.argmax(cpu_expected, axis=1) == np.argmax(actual_f32, axis=1)
        )
        assert max(kls) <= 0.05
        assert top1 >= 0.90


def _run_selected_q6_t16_direct(
    x: np.ndarray,
    tiles: np.ndarray,
    rows: int,
    in_features: int,
    out_features: int,
    library,
) -> np.ndarray:
    padded_rows = ((rows + 15) // 16) * 16
    host_metadata = (
        np.array([0, rows], dtype=np.int64),
        np.array([0, padded_rows], dtype=np.int64),
        np.zeros(padded_rows // 16, dtype=np.int64),
    )
    buffers = []
    try:
        x_buf = malloc(x.nbytes)
        tiles_buf = malloc(tiles.nbytes)
        out = np.zeros((rows, out_features), dtype=np.uint16)
        out_buf = malloc(out.nbytes)
        buffers.extend((x_buf, tiles_buf, out_buf))
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        copy_host_to_device(tiles_buf, host_array_ptr(tiles), tiles.nbytes)
        metadata = []
        for host in host_metadata:
            device = malloc(host.nbytes)
            buffers.append(device)
            metadata.append(device)
            copy_host_to_device(device, host_array_ptr(host), host.nbytes)
        gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out(
            x_buf.ptr,
            metadata[0].ptr,
            metadata[1].ptr,
            metadata[2].ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            1,
            padded_rows,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_t16_dense_wmma_prefill_passes_cpu_quality_gate(
    q6_t16_library,
    q6_t16_selected_prefill_library,
) -> None:
    rows, in_features, out_features = 17, 512, 256
    rng = np.random.default_rng(0x3627)
    qweight = make_q6_k_weight(out_features, in_features)
    tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    dense_wmma = getattr(
        t16_mod,
        "gguf_q6_k_t16_wmma_prefill_bf16_bf16_out",
        None,
    )
    assert dense_wmma is not None

    actual_bits = _run_single(
        dense_wmma,
        x_bf16,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q6_t16_library,
    )
    selected_bits = _run_selected_q6_t16_direct(
        x_bf16,
        tiles,
        rows,
        in_features,
        out_features,
        q6_t16_selected_prefill_library,
    )
    actual = _bf16_u16_to_f32(actual_bits)
    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q6_K)

    np.testing.assert_array_equal(actual_bits, selected_bits)
    np.testing.assert_allclose(actual, expected, atol=3.0e-1, rtol=1.2e-2)
    kls = [_stable_kl(expected[row], actual[row]) for row in range(rows)]
    top1 = np.mean(np.argmax(expected, axis=1) == np.argmax(actual, axis=1))
    assert max(kls) <= 0.05
    assert top1 >= 0.90


def _stable_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    got = candidate.astype(np.float64)
    ref -= np.max(ref)
    got -= np.max(got)
    ref_lse = np.log(np.sum(np.exp(ref)))
    got_lse = np.log(np.sum(np.exp(got)))
    probabilities = np.exp(ref - ref_lse)
    return float(np.sum(probabilities * ((ref - ref_lse) - (got - got_lse))))


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_t16_qmicro_planar_top1_stage1_is_bit_exact_to_legacy(
    q6_t16_library,
) -> None:
    rows, in_features, out_features = 1, 512, 256
    rng = np.random.default_rng(0x6A14)
    qweight = make_q6_k_weight(out_features, in_features)
    qweight[80] = qweight[17]
    legacy_tiles = repack_gguf_q6_k_tile16(qweight[None, ...]).tiles
    qmicro_tiles = repack_gguf_q6_k_tile16_qmicro_planar(
        qweight[None, ...]
    ).tiles
    x = _f32_to_bf16_u16(
        rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    )
    tile_count = out_features // 16

    def run_stage1(fn, tiles):
        buffers = []
        try:
            x_d = malloc(x.nbytes)
            tiles_d = malloc(tiles.nbytes)
            logits_d = malloc(out_features * 4)
            tile_values_d = malloc(tile_count * 4)
            tile_indices_d = malloc(tile_count * 4)
            buffers.extend((x_d, tiles_d, logits_d, tile_values_d, tile_indices_d))
            copy_host_to_device(x_d, host_array_ptr(x), x.nbytes)
            copy_host_to_device(tiles_d, host_array_ptr(tiles), tiles.nbytes)
            fn(
                x_d.ptr,
                tiles_d.ptr,
                logits_d.ptr,
                tile_values_d.ptr,
                tile_indices_d.ptr,
                in_features,
                out_features,
                library=q6_t16_library,
            )
            logits = np.empty(out_features, dtype=np.float32)
            tile_values = np.empty(tile_count, dtype=np.float32)
            tile_indices = np.empty(tile_count, dtype=np.int32)
            for host, device in (
                (logits, logits_d),
                (tile_values, tile_values_d),
                (tile_indices, tile_indices_d),
            ):
                copy_device_to_host(host_array_ptr(host), device, host.nbytes)
            return logits, tile_values, tile_indices
        finally:
            for buffer in reversed(buffers):
                free(buffer)

    expected = run_stage1(
        gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1,
        legacy_tiles,
    )
    actual = run_stage1(
        t16_mod.gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1,
        qmicro_tiles,
    )
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


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
