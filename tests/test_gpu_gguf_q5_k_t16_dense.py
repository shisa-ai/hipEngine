"""Q5 T16 dense dual fused-SiLU prefill kernel tests (UD impact-list task 4).

The dual is the Q5 port of the Q4 gate/up owner: it must reproduce the
unfused production chain (Q5 T16 single WMMA prefill x2 +
silu_mul_separate_out_bf16) bit-exactly - the same K16 WMMA association and
the same BF16 projection boundary before SiLU - while the 256-thread decode
spreads the heavier Q5 element decode off the WMMA critical path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q5_k_t16_dense_dual_wmma_prefill_row128_silu_bf16_bf16_out,
    gguf_q5_k_t16_dense_dual_wmma_prefill_row32_silu_bf16_bf16_out,
    gguf_q5_k_t16_dense_dual_wmma_prefill_row48_silu_bf16_bf16_out,
    gguf_q5_k_t16_dense_dual_wmma_prefill_row64_silu_bf16_bf16_out,
    gguf_q5_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_gfx1100_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_shared8r2_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as t16_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    gguf_q5_k_t16_dense_dual_silu_gemv_bf16_bf16_out,
    gguf_q5_k_t16_gemv_decode_bf16_bf16_out,
)
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16


@pytest.fixture(autouse=True)
def _require_hip(hip_test_target_arch):
    """These tests launch kernels rather than only inspecting dispatch."""


def _bf16_bits(a: np.ndarray) -> np.ndarray:
    u32 = a.astype(np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def make_q5_k_raw(out_features: int, in_features: int, seed: int) -> np.ndarray:
    """Synthesize raw GGUF Q5_K bytes: finite fp16 d/dmin, random payload."""

    assert in_features % 256 == 0 and out_features % 32 == 0
    blocks_per_row = in_features // 256
    rng = np.random.default_rng(seed)
    raw = np.zeros((out_features, blocks_per_row, 176), dtype=np.uint8)
    d = np.array([0.010, 0.020, 0.031, 0.008], dtype=np.float16)
    dmin = np.array([0.004, 0.009, 0.001, 0.016], dtype=np.float16)
    raw[..., 0:2] = d[rng.integers(0, 4, raw.shape[:2])].view(np.uint8).reshape(
        raw.shape[0], raw.shape[1], 2
    )
    raw[..., 2:4] = dmin[
        rng.integers(0, 4, raw.shape[:2])
    ].view(np.uint8).reshape(raw.shape[0], raw.shape[1], 2)
    raw[..., 4:16] = rng.integers(0, 256, (*raw.shape[:2], 12), dtype=np.uint8)
    raw[..., 16:176] = rng.integers(0, 256, (*raw.shape[:2], 160), dtype=np.uint8)
    return raw.reshape(out_features, blocks_per_row * 176)


_DUAL_VARIANTS = {
    96: gguf_q5_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
    128: gguf_q5_k_t16_dense_dual_wmma_prefill_row128_silu_bf16_bf16_out,
    64: gguf_q5_k_t16_dense_dual_wmma_prefill_row64_silu_bf16_bf16_out,
    48: gguf_q5_k_t16_dense_dual_wmma_prefill_row48_silu_bf16_bf16_out,
    32: gguf_q5_k_t16_dense_dual_wmma_prefill_row32_silu_bf16_bf16_out,
}


@pytest.mark.parametrize("rows", (96, 128, 64, 48, 32))
def test_dual_matches_unfused_chain(rows: int) -> None:
    runtime = get_hip_runtime()
    in_features = 512
    out_features = 64
    raw_a = make_q5_k_raw(out_features, in_features, seed=0x5A17)
    raw_b = make_q5_k_raw(out_features, in_features, seed=0x5A18)
    tiles_a = repack_gguf_q5_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q5_k_tile16(raw_b[None, ...]).tiles
    rng = np.random.default_rng(0x38D52)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    expected_bits = np.zeros((rows, out_features), dtype=np.uint16)
    actual_bits = np.zeros_like(expected_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        gate_dev = malloc(expected_bits.nbytes, runtime=runtime)
        up_dev = malloc(expected_bits.nbytes, runtime=runtime)
        control_dev = malloc(expected_bits.nbytes, runtime=runtime)
        candidate_dev = malloc(expected_bits.nbytes, runtime=runtime)
        buffers.extend(
            (x_dev, tiles_a_dev, tiles_b_dev, gate_dev, up_dev, control_dev, candidate_dev)
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            tiles_a_dev, host_array_ptr(tiles_a), runtime=runtime
        )
        copy_host_to_device(
            tiles_b_dev, host_array_ptr(tiles_b), runtime=runtime
        )
        library = build_gguf_k_t16_selected_prefill(load=True)
        silu_library = build_paro_silu(load=True)
        for tiles_dev, out_dev in (
            (tiles_a_dev, gate_dev),
            (tiles_b_dev, up_dev),
        ):
            gguf_q5_k_t16_wmma_prefill_bf16_bf16_out(
                x_dev.ptr,
                tiles_dev.ptr,
                out_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        silu_mul_separate_out_bf16(
            gate_dev.ptr,
            up_dev.ptr,
            control_dev.ptr,
            rows,
            out_features,
            library=silu_library,
            runtime=runtime,
        )
        _DUAL_VARIANTS[rows](
            x_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            candidate_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(expected_bits), control_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(actual_bits), candidate_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_bits, expected_bits)
    assert np.isfinite(
        (actual_bits.astype(np.uint32) << 16).view(np.float32)
    ).all()


def test_shared8r2_bulk_owner_matches_plain_single() -> None:
    """The row-qualified bulk owner must be bit-exact with the plain single.

    The selector routes rows >= 257 to the eight-wave two-row-tile
    shared-LDS owner; the shared kernel family preserves the plain owner's
    K16 WMMA association, so every output bit must match.
    """

    runtime = get_hip_runtime()
    in_features = 512
    out_features = 64
    raw = make_q5_k_raw(out_features, in_features, seed=0x5A27)
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles
    rng = np.random.default_rng(0x38D53)
    library = build_gguf_k_t16_selected_prefill(load=True)
    for rows in (512, 384, 257, 256):
        x_bits = _bf16_bits(
            rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
        )
        expected_bits = np.zeros((rows, out_features), dtype=np.uint16)
        actual_bits = np.zeros_like(expected_bits)
        buffers = []
        try:
            x_dev = malloc(x_bits.nbytes, runtime=runtime)
            tiles_dev = malloc(tiles.nbytes, runtime=runtime)
            plain_dev = malloc(expected_bits.nbytes, runtime=runtime)
            owner_dev = malloc(expected_bits.nbytes, runtime=runtime)
            buffers.extend((x_dev, tiles_dev, plain_dev, owner_dev))
            copy_host_to_device(
                x_dev, host_array_ptr(x_bits), runtime=runtime
            )
            copy_host_to_device(
                tiles_dev, host_array_ptr(tiles), runtime=runtime
            )
            for out_dev, fn in (
                (plain_dev, gguf_q5_k_t16_wmma_prefill_bf16_bf16_out),
                (owner_dev, gguf_q5_k_t16_wmma_prefill_shared8r2_bf16_bf16_out),
                (owner_dev, gguf_q5_k_t16_wmma_prefill_gfx1100_bf16_bf16_out),
            ):
                fn(
                    x_dev.ptr,
                    tiles_dev.ptr,
                    out_dev.ptr,
                    rows,
                    in_features,
                    out_features,
                    library=library,
                    runtime=runtime,
                )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(expected_bits), plain_dev, runtime=runtime
            )
            copy_device_to_host(
                host_array_ptr(actual_bits), owner_dev, runtime=runtime
            )
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
        np.testing.assert_array_equal(actual_bits, expected_bits)


def test_dense_dual_silu_gemv_decode_matches_unfused_chain() -> None:
    """The Q5 T16 decode dual must be bit-exact with the unfused chain.

    The dual mirrors the resident direct-GEMV decode owner exactly - the
    same 128-thread block, strided K ownership, weight expression, wave32
    tree and serial wave-0..3 reduction - with gate and up independently
    rounded to BF16 before the fused SiLU product, so every output bit must
    match single GEMV x2 + silu_mul_separate.
    """

    runtime = get_hip_runtime()
    in_features = 512
    out_features = 64
    raw_a = make_q5_k_raw(out_features, in_features, seed=0x5A29)
    raw_b = make_q5_k_raw(out_features, in_features, seed=0x5A2A)
    tiles_a = repack_gguf_q5_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q5_k_tile16(raw_b[None, ...]).tiles
    rng = np.random.default_rng(0x38D54)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.2, size=(1, in_features)).astype(np.float32)
    )
    expected_bits = np.zeros((1, out_features), dtype=np.uint16)
    actual_bits = np.zeros_like(expected_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        gate_dev = malloc(expected_bits.nbytes, runtime=runtime)
        up_dev = malloc(expected_bits.nbytes, runtime=runtime)
        control_dev = malloc(expected_bits.nbytes, runtime=runtime)
        candidate_dev = malloc(expected_bits.nbytes, runtime=runtime)
        buffers.extend(
            (x_dev, tiles_a_dev, tiles_b_dev, gate_dev, up_dev, control_dev, candidate_dev)
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            tiles_a_dev, host_array_ptr(tiles_a), runtime=runtime
        )
        copy_host_to_device(
            tiles_b_dev, host_array_ptr(tiles_b), runtime=runtime
        )
        library = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        silu_library = build_paro_silu(load=True)
        for tiles_dev, out_dev in (
            (tiles_a_dev, gate_dev),
            (tiles_b_dev, up_dev),
        ):
            gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
                x_dev.ptr,
                tiles_dev.ptr,
                out_dev.ptr,
                1,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        silu_mul_separate_out_bf16(
            gate_dev.ptr,
            up_dev.ptr,
            control_dev.ptr,
            1,
            out_features,
            library=silu_library,
            runtime=runtime,
        )
        gguf_q5_k_t16_dense_dual_silu_gemv_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            candidate_dev.ptr,
            1,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(expected_bits), control_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(actual_bits), candidate_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_bits, expected_bits)
    assert np.isfinite(
        (actual_bits.astype(np.uint32) << 16).view(np.float32)
    ).all()


def test_dense_pair_variants_are_registered() -> None:
    from hipengine.kernels.registry import KernelKey, is_registered
    from hipengine.runtime.gguf_linear import _q5_t16_dense_pair_silu_variant

    # Below one full row tile the pair declines and the two singles keep the
    # verifier-row t16 rowtile owner. Every variant here is a WMMA prefill
    # owner whose cost is its block's 32/48/64/128/256-row tile, so at the
    # native verifier envelope (2-8 rows) the row32 entry would run a 32-row
    # WMMA tile to produce at most 8 rows. Measured at rows 3 on
    # Qwen3.8-27B-UD-Q4_K_M: the pair path cost 5.719 ms/step for nine
    # gate/up pairs (193 GB/s) against 2.609 ms/step for the same pairs as
    # two singles through q5_k_t16_dense_rowtile_gemv (410 GB/s), so the
    # owner that fires at verifier rows is the single.
    for rows, expected in (
        (512, "dense_dual_wmma_prefill_bf16_bf16_out"),
        (257, "dense_dual_wmma_prefill_bf16_bf16_out"),
        (256, "dense_dual_wmma_prefill_row128_bf16_bf16_out"),
        (129, "dense_dual_wmma_prefill_row128_bf16_bf16_out"),
        (128, "dense_dual_wmma_prefill_row64_bf16_bf16_out"),
        (65, "dense_dual_wmma_prefill_row64_bf16_bf16_out"),
        (64, "dense_dual_wmma_prefill_row48_bf16_bf16_out"),
        (48, "dense_dual_wmma_prefill_row48_bf16_bf16_out"),
        (33, "dense_dual_wmma_prefill_row48_bf16_bf16_out"),
        (32, "dense_dual_wmma_prefill_row32_bf16_bf16_out"),
        (31, None),
        (8, None),
        (4, None),
        (3, None),
        (2, None),
        (1, None),
    ):
        assert _q5_t16_dense_pair_silu_variant(rows) == expected, rows

    for variant in (
        "dense_dual_wmma_prefill_bf16_bf16_out",
        "dense_dual_wmma_prefill_row32_bf16_bf16_out",
        "dense_dual_wmma_prefill_row48_bf16_bf16_out",
        "dense_dual_wmma_prefill_row64_bf16_bf16_out",
        "dense_dual_wmma_prefill_row128_bf16_bf16_out",
    ):
        assert is_registered(
            KernelKey(
                "hip_gfx1100", "linear_pair_silu", "gguf_q5_k_t16_v1", variant
            )
        ), variant


@pytest.mark.parametrize(
    "in_features,out_features",
    [(512, 64), (1024, 32), (512, 128)],
)
def test_q5_t16_local32_single_matches_direct_gemv(
    in_features: int, out_features: int
) -> None:
    """The Q5 local32 decode single must agree with the direct GEMV.

    Same contract as the IQ4 local32 owners: per-element products identical,
    only the summation order differs (one superblock of eight contiguous K
    per lane, wave32 shuffle tree, lane-0 serial store). The tiles come from
    the real repack path (repack_gguf_q5_k_tile16 over synthetic raw Q5_K
    blocks), so the layout arithmetic - the hoisted d/dmin/scale/min unpack
    plus the u32 nibble window and u8 high-bit row - is exercised exactly as
    production materializes it.
    """

    runtime = get_hip_runtime()
    raw = make_q5_k_raw(out_features, in_features, seed=0x6B31)
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles
    rng = np.random.default_rng(0x71C1A)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.2, size=(1, in_features)).astype(np.float32)
    )
    ref_bits = np.zeros((1, out_features), dtype=np.uint16)
    got_bits = np.zeros_like(ref_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        ref_dev = malloc(ref_bits.nbytes, runtime=runtime)
        got_dev = malloc(got_bits.nbytes, runtime=runtime)
        buffers.extend((x_dev, tiles_dev, ref_dev, got_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            tiles_dev, host_array_ptr(tiles), runtime=runtime
        )
        library = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            ref_dev.ptr,
            1,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        t16_gemv.gguf_q5_k_t16_dense_single_local32_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            got_dev.ptr,
            1,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(ref_bits), ref_dev, ref_bits.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(got_bits), got_dev, got_bits.nbytes, runtime=runtime
        )
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)

    ref = (ref_bits.astype(np.uint32) << 16).view(np.float32)
    got = (got_bits.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(got).all()
    scale = max(float(np.abs(ref).max()), 1e-30)
    rel = float(np.abs(ref - got).max()) / scale
    assert rel <= 5e-4, "Q5 local32 diverged from the direct GEMV"
    assert float(np.corrcoef(ref.ravel(), got.ravel())[0, 1]) >= 0.9999


def test_q5_t16_local32_registered_and_routed_at_c1() -> None:
    """The local32 single is registered and the C1 table routes production
    shapes at rows == 1 while rows > 1 keeps the direct family."""

    from hipengine.kernels.backends import (
        backend_package_capability,
        load_backend_kernel_package,
    )
    from hipengine.kernels.registry import KernelKey, is_registered

    load_backend_kernel_package("hip_gfx1100")

    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        )
    )
    c1 = backend_package_capability(
        "hip_gfx1100", "GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE", {}
    )
    for shape in (
        (5_120, 6_144),
        (5_120, 17_408),
        (17_408, 5_120),
        (6_144, 5_120),
        (1_024, 5_120),
        (5_120, 10_240),
        (5_120, 12_288),
    ):
        assert c1["gguf_q5_k_t16_v1"][shape] == "dense_single_local32_bf16_bf16_out"
