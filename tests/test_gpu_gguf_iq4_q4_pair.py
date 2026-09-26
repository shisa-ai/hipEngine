"""E6b-1 GPU gates: the (IQ4_XS gate, Q4_K up) mixed pair + SiLU owner.

The fused mixed pair must be bit-identical to the production unfused chain
at rows==1: IQ4_XS local32 single + Q4_K dense single + the production
silu_mul_separate_out kernel, all on device so the SiLU's expf is the same
libm the fused kernel calls. HIP-guarded so no-ROCm runners skip.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
except OSError:
    pytest.skip("HIP runtime unavailable", allow_module_level=True)

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused import silu_mul_separate_out_bf16  # noqa: E402
from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (  # noqa: E402
    gguf_iq4_q4_pair_silu_bf16_bf16_out,
    gguf_iq4_q5_pair_silu_bf16_bf16_out,
    gguf_q4_iq4_pair_silu_bf16_bf16_out,
    gguf_q4_q5_pair_silu_bf16_bf16_out,
    gguf_q3_iq4_pair_silu_bf16_bf16_out,
    gguf_q5_q4_pair_silu_bf16_bf16_out,
    gguf_iq4_q3_pair_silu_bf16_bf16_out,
    gguf_iq3s_iq4_pair_silu_bf16_bf16_out,
    gguf_iq4nl_q5_pair_silu_bf16_bf16_out,
    gguf_iq4nl_nl_pair_silu_bf16_bf16_out,
    gguf_q5_q6_pair_silu_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (  # noqa: E402
    gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out,
)
from hipengine.quant.gguf_t16 import (  # noqa: E402
    repack_gguf_q5_k_tile16,
    repack_gguf_q6_k_tile16_qmicro_planar,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense  # noqa: E402
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (  # noqa: E402
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant import (  # noqa: E402
    gguf_t16_selected_gemv as t16_gemv,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (  # noqa: E402
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, is_registered  # noqa: E402
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16  # noqa: E402
from tests._gguf_synthetic_weights import (  # noqa: E402
    make_q3_k_weight,
    make_q4_k_weight,
    make_q5_k_weight,
    make_q6_k_weight,
)

FIXTURE = Path(__file__).parent / "fixtures/gguf_ud"


def bf16(x: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(x, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)


def test_iq4_q4_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_iq4_xs+gguf_q4_k_t16_v1",
            "iq4_q4_pair_silu_bf16_bf16_out",
        )
    )


def test_iq4_q4_mixed_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"IQ4_XS fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6B1)
    n = 16
    # gate: IQ4_XS local32 rows (raw packed bytes, as the local32 path eats)
    # Each raw IQ4_XS row is one output column's packed local32 bytes
    # (K/256 * 136), exactly what launch_local32 consumes; rows repeat to N.
    wa = np.ascontiguousarray(
        source[rng.integers(0, len(source), n) % len(source)]
    )
    if wa.shape[0] != n or n % 8:
        pytest.skip("IQ4_XS fixture rows are not (N, 8-compatible)")
    # up: synthetic Q4_K in T16 tiles, same K and N
    raw_b = make_q4_k_weight(n, k)
    tiles_b = np.ascontiguousarray(repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles)

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        wa_b = dev(wa)
        tb_b = dev(tiles_b)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        q4lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)

        # unfused chain: IQ4_XS local32 single + Q4_K dense single + silu_mul
        gguf_iq_dense.launch_local32(
            x_b.ptr, wa_b.ptr, ga_b.ptr, 1, k, n, library=silulib, runtime=runtime
        )
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
            x_b.ptr, tb_b.ptr, ub_b.ptr, 1, k, n, library=q4lib, runtime=runtime
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused mixed pair
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_iq4_q4_pair_silu_bf16_bf16_out(
            x_b.ptr,
            wa_b.ptr,
            tb_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "mixed IQ4_XS/Q4_K pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )

def test_q4_iq4_mirror_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1+gguf_iq4_xs",
            "q4_iq4_pair_silu_bf16_bf16_out",
        )
    )


def test_q4_iq4_mirror_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6b-2: (Q4_K gate, IQ4_XS up) fused owner vs the unfused chain.

    Same gates as E6b-1 with the roles exchanged: the Q4_K single writes
    the gate buffer, the IQ4_XS local32 single writes the up buffer, and
    the reference is those two outputs through silu_mul - all on device.
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"IQ4_XS fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6B2)
    n = 16
    w_up = np.ascontiguousarray(
        source[rng.integers(0, len(source), n) % len(source)]
    )
    if w_up.shape[0] != n or n % 8:
        pytest.skip("IQ4_XS fixture rows are not (N, 8-compatible)")
    raw_gate = make_q4_k_weight(n, k)
    tiles_gate = np.ascontiguousarray(
        repack_gguf_q4_k_tile16(raw_gate[None, ...]).tiles
    )

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        w_up_b = dev(w_up)
        t_gate_b = dev(tiles_gate)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        q4lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)

        # unfused chain: Q4_K dense single (gate) + IQ4_XS local32 single
        # (up) + silu_mul
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
            x_b.ptr, t_gate_b.ptr, ga_b.ptr, 1, k, n, library=q4lib, runtime=runtime
        )
        gguf_iq_dense.launch_local32(
            x_b.ptr, w_up_b.ptr, ub_b.ptr, 1, k, n, library=silulib, runtime=runtime
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused mirror pair: gate-first (Q4 tiles, IQ4 raw)
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_q4_iq4_pair_silu_bf16_bf16_out(
            x_b.ptr,
            t_gate_b.ptr,
            w_up_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "mirror Q4_K/IQ4_XS pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )


def test_iq4_q5_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_iq4_xs+gguf_q5_k_t16_v1",
            "iq4_q5_pair_silu_bf16_bf16_out",
        )
    )


def test_iq4_q5_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6b-3: (IQ4_XS gate, Q5_K up) fused owner vs the unfused chain.

    Chain: IQ4_XS local32 single (gate) + Q5_K tile8 single (up) +
    silu_mul, all on device. The pair's side B must reproduce the
    tile8 single's 4-group chain exactly (wave-0 emulation).
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"IQ4_XS fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6B3)
    n = 16  # one resident Q5_T16 tile
    wa = np.ascontiguousarray(
        source[rng.integers(0, len(source), n) % len(source)]
    )
    if wa.shape[0] != n or n % 8:
        pytest.skip("IQ4_XS fixture rows are not (N, 8-compatible)")
    raw_up = make_q5_k_weight(n, k)
    tiles_up = np.ascontiguousarray(
        repack_gguf_q5_k_tile16(raw_up[None, ...]).tiles
    )

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        wa_b = dev(wa)
        tb_b = dev(tiles_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        q5lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)

        # unfused chain: IQ4_XS local32 single (gate) + Q5_K tile8 single
        # (up) + silu_mul
        gguf_iq_dense.launch_local32(
            x_b.ptr, wa_b.ptr, ga_b.ptr, 1, k, n, library=silulib, runtime=runtime
        )
        gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out(
            x_b.ptr, tb_b.ptr, ub_b.ptr, 1, k, n, library=q5lib, runtime=runtime
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: gate first (IQ4 raw), up second (Q5 tiles)
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_iq4_q5_pair_silu_bf16_bf16_out(
            x_b.ptr,
            wa_b.ptr,
            tb_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "IQ4_XS/Q5_K pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )


def test_q4_q5_mixed_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k_t16_v1+gguf_q5_k_t16_v1",
            "q4_q5_pair_silu_bf16_bf16_out",
        )
    )


def test_q4_q5_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6b-4: (Q4_K gate, Q5_K up) fused owner vs the unfused chain.

    Neither side is IQ4: gate = Q4_K dense single (tiles), up = Q5_K
    tile8 single (tiles), reference = those two through silu_mul, all
    on device. Side A must reproduce the dense single's chain exactly
    under A_IS_CHAIN (wave-0 published value), side B the tile8 single's
    4-group chain exactly (wave-distributed emulation).
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6B4)
    n = 16  # one resident T16 tile column pair
    raw_gate = make_q4_k_weight(n, k)
    tiles_gate = np.ascontiguousarray(
        repack_gguf_q4_k_tile16(raw_gate[None, ...]).tiles
    )
    raw_up = make_q5_k_weight(n, k)
    tiles_up = np.ascontiguousarray(
        repack_gguf_q5_k_tile16(raw_up[None, ...]).tiles
    )

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        t_gate_b = dev(tiles_gate)
        t_up_b = dev(tiles_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        q4lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        q5lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)

        # unfused chain: Q4_K dense single (gate) + Q5_K tile8 single
        # (up) + silu_mul
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
            x_b.ptr, t_gate_b.ptr, ga_b.ptr, 1, k, n, library=q4lib, runtime=runtime
        )
        gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out(
            x_b.ptr, t_up_b.ptr, ub_b.ptr, 1, k, n, library=q5lib, runtime=runtime
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: gate first (Q4 tiles), up second (Q5 tiles)
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_q4_q5_pair_silu_bf16_bf16_out(
            x_b.ptr,
            t_gate_b.ptr,
            t_up_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "Q4_K/Q5_K pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )


def test_q3_iq4_mixed_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q3_k+gguf_iq4_xs",
            "q3_iq4_pair_silu_bf16_bf16_out",
        )
    )


def test_q3_iq4_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6b-5: (Q3_K gate, IQ4_XS up) fused owner vs the unfused chain.

    Chain: Q3_K strict per-row GEMV (gate) + IQ4_XS local32 single
    (up) + silu_mul, all on device. Side B must reproduce the strict
    128-thread tile exactly (phase-A hoist, k-ascending accumulation,
    wave32 tree, serial wave-0..3 total), side A the unchanged IQ4
    split-K chain; the wrapper reorders route (gate, up) to the C
    geometry (IQ4 first).
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"IQ4_XS fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6B5)
    n = 16
    w_up = np.ascontiguousarray(
        source[rng.integers(0, len(source), n) % len(source)]
    )
    if w_up.shape[0] != n or n % 8:
        pytest.skip("IQ4_XS fixture rows are not (N, 8-compatible)")
    w_gate = make_q3_k_weight(n, k)

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        w_gate_b = dev(w_gate)
        w_up_b = dev(w_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)

        # unfused chain: Q3_K strict single (gate) + IQ4_XS local32
        # single (up) + silu_mul
        gguf_iq_dense.launch(
            x_b.ptr, w_gate_b.ptr, ga_b.ptr, 1, k, n,
            quant="gguf_q3_k", output="bf16",
            library=silulib, runtime=runtime,
        )
        gguf_iq_dense.launch_local32(
            x_b.ptr, w_up_b.ptr, ub_b.ptr, 1, k, n,
            library=silulib, runtime=runtime,
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: route order gate first (Q3 raw), up second (IQ4 raw);
        # the wrapper reorders to the C geometry (IQ4 first).
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_q3_iq4_pair_silu_bf16_bf16_out(
            x_b.ptr,
            w_gate_b.ptr,
            w_up_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "Q3_K/IQ4_XS pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )

def test_q5_q4_mixed_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q5_k_t16_v1+gguf_q4_k_t16_v1",
            "q5_q4_pair_silu_bf16_bf16_out",
        )
    )


def test_q5_q4_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6b-6: (Q5_K gate, Q4_K up) fused owner vs the unfused chain.

    Exact role-swap of the E6b-4 test: gate = Q5_K tile8 single
    (tiles), up = Q4_K dense single (tiles), reference = those two
    through silu_mul, all on device. Side A must reproduce the dense
    single's chain exactly under A_IS_CHAIN (wave-0 published value),
    side B the tile8 single's 4-group chain exactly; the epilogue takes
    the gate from side B (GATE_IS_Q4=true) since gate/up roles are
    exchanged relative to E6b-4.
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6B6)
    n = 16  # one resident T16 tile column pair
    raw_gate = make_q5_k_weight(n, k)
    tiles_gate = np.ascontiguousarray(
        repack_gguf_q5_k_tile16(raw_gate[None, ...]).tiles
    )
    raw_up = make_q4_k_weight(n, k)
    tiles_up = np.ascontiguousarray(
        repack_gguf_q4_k_tile16(raw_up[None, ...]).tiles
    )

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        t_gate_b = dev(tiles_gate)
        t_up_b = dev(tiles_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        q4lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        q5lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)

        # unfused chain: Q5_K tile8 single (gate) + Q4_K dense single
        # (up) + silu_mul
        gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out(
            x_b.ptr, t_gate_b.ptr, ga_b.ptr, 1, k, n, library=q5lib, runtime=runtime
        )
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
            x_b.ptr, t_up_b.ptr, ub_b.ptr, 1, k, n, library=q4lib, runtime=runtime
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: gate first (Q5 tiles), up second (Q4 tiles) -
        # the wrapper reorders to C geometry order internally.
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_q5_q4_pair_silu_bf16_bf16_out(
            x_b.ptr,
            t_gate_b.ptr,
            t_up_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "Q5_K/Q4_K pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )

def test_iq4_q3_mixed_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_iq4_xs+gguf_q3_k",
            "iq4_q3_pair_silu_bf16_bf16_out",
        )
    )


def test_iq4_q3_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6b-7: (IQ4_XS gate, Q3_K up) fused owner vs the unfused chain.

    Opposite epilogue roles to the dormant E6b-5 instance with identical
    side arithmetic: chain = IQ4_XS local32 single (gate) + Q3_K strict
    per-row GEMV (up) + silu_mul, all on device. Side A must reproduce
    the local32 chain exactly, side B the strict 128-thread tile exactly
    (phase-A hoist, k-ascending accumulation, wave32 tree, serial
    wave-0..3 total); no wrapper reorder (route order == C geometry).
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"IQ4_XS fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6B7)
    n = 16
    w_gate = np.ascontiguousarray(
        source[rng.integers(0, len(source), n) % len(source)]
    )
    if w_gate.shape[0] != n or n % 8:
        pytest.skip("IQ4_XS fixture rows are not (N, 8-compatible)")
    w_up = make_q3_k_weight(n, k)

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        w_gate_b = dev(w_gate)
        w_up_b = dev(w_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)

        # unfused chain: IQ4_XS local32 single (gate) + Q3_K strict
        # single (up) + silu_mul
        gguf_iq_dense.launch_local32(
            x_b.ptr, w_gate_b.ptr, ga_b.ptr, 1, k, n,
            library=silulib, runtime=runtime,
        )
        gguf_iq_dense.launch(
            x_b.ptr, w_up_b.ptr, ub_b.ptr, 1, k, n,
            quant="gguf_q3_k", output="bf16",
            library=silulib, runtime=runtime,
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: route order gate first (IQ4 raw), up second (Q3
        # raw) - geometry order, no wrapper reorder.
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_iq4_q3_pair_silu_bf16_bf16_out(
            x_b.ptr,
            w_gate_b.ptr,
            w_up_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "IQ4_XS/Q3_K pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )


def test_iq3s_iq4_mixed_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_iq3_s+gguf_iq4_xs",
            "iq3s_iq4_pair_silu_bf16_bf16_out",
        )
    )


def test_iq3s_iq4_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6 closeout: (IQ3_S gate, IQ4_XS up) fused owner vs unfused chain.

    Chain = IQ3_S local32 single (gate) + IQ4_XS local32 single (up) +
    silu_mul. Side A must reproduce the IQ4_XS local32 split-K exactly;
    side B the IQ3_S local32 owner's Q==2 split-K path exactly (blk
    wave-stride, lane-grouped XV[8], sd * mag * sgn, plain +=,
    cross-wave sum in the epilogue). Wrapper takes gate-first (IQ3_S)
    and reorders internally to C geometry (IQ4 side A).
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    e_iq3s = next((e for e in entries if e["type"] == "IQ3_S"), None)
    e_iq4 = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if e_iq3s is None or e_iq4 is None:
        pytest.skip("no IQ3_S/IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        src_iq3s = data[e_iq3s["key"] + "_raw"]
        src_iq4 = data[e_iq4["key"] + "_raw"]
        k3 = data[e_iq3s["key"] + "_f32"].shape[1]
        k4 = data[e_iq4["key"] + "_f32"].shape[1]
    # The two fixture tensors carry different K (5120 vs 17408). Align
    # on the smaller by taking a block-aligned prefix of the IQ4_XS raw
    # rows: raw IQ rows are K/256 packed blocks, so the first k3/256
    # blocks are themselves a valid K=k3 IQ4_XS weight, and both sides
    # then read exactly the bytes their singles read.
    if k3 % 256 or k4 < k3 or (k4 % 256):
        pytest.skip(f"fixture K unusable: IQ3_S {k3} vs IQ4_XS {k4}")
    k = k3
    nblocks = k // 256
    rowbytes = (k4 // 256) * 136  # IQ4_XS local32 row size at fixture K
    if rowbytes < nblocks * 136:
        pytest.skip("IQ4_XS fixture row shorter than prefix requirement")

    rng = np.random.default_rng(0xE6C1)
    n = 16
    w_gate = np.ascontiguousarray(
        src_iq3s[rng.integers(0, len(src_iq3s), n) % len(src_iq3s)]
    )
    w_up = np.ascontiguousarray(
        src_iq4[rng.integers(0, len(src_iq4), n) % len(src_iq4), : nblocks * 136]
    )
    if w_gate.shape[0] != n or w_up.shape[0] != n or n % 8:
        pytest.skip("fixture rows are not (N, 8-compatible)")

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        w_gate_b = dev(w_gate)
        w_up_b = dev(w_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)

        # unfused chain: IQ3_S local32 single (gate) + IQ4_XS local32
        # single (up) + silu_mul
        gguf_iq_dense.launch_local32(
            x_b.ptr, w_gate_b.ptr, ga_b.ptr, 1, k, n,
            quant="gguf_iq3_s", library=silulib, runtime=runtime,
        )
        gguf_iq_dense.launch_local32(
            x_b.ptr, w_up_b.ptr, ub_b.ptr, 1, k, n,
            library=silulib, runtime=runtime,
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: gate-first (IQ3_S raw, IQ4_XS raw); the wrapper
        # reorders to C geometry order (IQ4 side A).
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_iq3s_iq4_pair_silu_bf16_bf16_out(
            x_b.ptr,
            w_gate_b.ptr,
            w_up_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "IQ3_S/IQ4_XS pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )


def test_iq4nl_q5_mixed_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_iq4_nl+gguf_q5_k_t16_v1",
            "iq4nl_q5_pair_silu_bf16_bf16_out",
        )
    )


def test_iq4nl_q5_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6 closeout: (IQ4_NL gate, Q5_K up) fused owner vs unfused chain.

    Chain = IQ4_NL local32 single (gate) + Q5_K tile8 single (up) +
    silu_mul. Side A must reproduce the IQ4_NL local32 split-K exactly
    (18-byte block stride, plain f16 scale); side B the tile8 single's
    exact 4-group emulation. Route order already matches C geometry
    (IQ4 side A first), so no wrapper reorder.
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_NL"), None)
    if entry is None:
        pytest.skip("no IQ4_NL fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"IQ4_NL fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6C2)
    n = 16  # one resident Q5_T16 tile
    w_gate = np.ascontiguousarray(
        source[rng.integers(0, len(source), n) % len(source)]
    )
    if w_gate.shape[0] != n or n % 8:
        pytest.skip("IQ4_NL fixture rows are not (N, 8-compatible)")
    raw_up = make_q5_k_weight(n, k)
    tiles_up = np.ascontiguousarray(repack_gguf_q5_k_tile16(raw_up[None, ...]).tiles)

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        w_gate_b = dev(w_gate)
        tb_b = dev(tiles_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)
        q5lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)

        # unfused chain: IQ4_NL local32 single (gate) + Q5_K tile8
        # single (up) + silu_mul
        gguf_iq_dense.launch_local32(
            x_b.ptr, w_gate_b.ptr, ga_b.ptr, 1, k, n,
            quant="gguf_iq4_nl", library=silulib, runtime=runtime,
        )
        gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out(
            x_b.ptr, tb_b.ptr, ub_b.ptr, 1, k, n, library=q5lib, runtime=runtime
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: gate-first (IQ4_NL raw) already matches geometry.
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_iq4nl_q5_pair_silu_bf16_bf16_out(
            x_b.ptr,
            w_gate_b.ptr,
            tb_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "IQ4_NL/Q5_K pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )


def test_q5_q6_mixed_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q5_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1",
            "q5_q6_pair_silu_bf16_bf16_out",
        )
    )


def test_q5_q6_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6 closeout: (Q5_K gate, Q6_K planar up) fused owner vs chain.

    Chain = Q5_K tile8 single (gate) + Q6_K qmicro-planar decode single
    (up) + silu_mul. Side A must reproduce the planar single's exact
    4-wave chain (emulated across the block's waves, thread-0 serial
    wave total); side B the tile8 single's exact 4-group emulation. The
    wrapper takes gate-first (Q5) and reorders internally to C geometry
    (Q6 planar side A).
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE6C3)
    n = 16  # one resident T16 tile
    raw_gate = make_q5_k_weight(n, k)
    tiles_gate = np.ascontiguousarray(
        repack_gguf_q5_k_tile16(raw_gate[None, ...]).tiles
    )
    raw_up = make_q6_k_weight(n, k)
    tiles_up = np.ascontiguousarray(
        repack_gguf_q6_k_tile16_qmicro_planar(raw_up[None, ...]).tiles
    )

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        tg_b = dev(tiles_gate)
        tu_b = dev(tiles_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        q5lib = t16_gemv.build_gguf_t16_selected_gemv(load=True)
        q6lib = build_gguf_q6_k_t16_gemv(load=True)

        # unfused chain: Q5_K tile8 single (gate) + Q6 planar decode
        # single (up) + silu_mul
        gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out(
            x_b.ptr, tg_b.ptr, ga_b.ptr, 1, k, n, library=q5lib, runtime=runtime
        )
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
            x_b.ptr, tu_b.ptr, ub_b.ptr, 1, k, n, library=q6lib, runtime=runtime
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: gate-first (Q5 tiles, Q6 tiles); the wrapper
        # reorders to C geometry order (Q6 planar side A).
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_q5_q6_pair_silu_bf16_bf16_out(
            x_b.ptr,
            tg_b.ptr,
            tu_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "Q5_K/Q6_K pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )


def test_iq4nl_nl_same_quant_pair_registered_as_pair_silu_owner() -> None:
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_iq4_nl+gguf_iq4_nl",
            "iq4nl_nl_pair_silu_bf16_bf16_out",
        )
    )


def test_iq4nl_nl_pair_silu_is_bit_exact_with_singles_and_silu_mul() -> None:
    """E6 same-quant closeout: (IQ4_NL gate, IQ4_NL up) fused vs chain.

    Chain = IQ4_NL local32 single (gate) + IQ4_NL local32 single (up) +
    silu_mul. Side A must reproduce the IQ4_NL local32 split-K exactly
    (A_KIND=2: 18-byte block stride, plain f16 scale); side B the same
    body mirrored into side B (B_KIND=4: wb instead of wa, acc_b
    instead of acc_a - staging only, element-for-element identical
    math). Route order already matches C geometry (gate first), so no
    wrapper reorder.
    """
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_NL"), None)
    if entry is None:
        pytest.skip("no IQ4_NL fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    if k % 256:
        pytest.skip(f"IQ4_NL fixture K={k} is not block-aligned")

    rng = np.random.default_rng(0xE650)
    n = 16  # one resident tile
    w_gate = np.ascontiguousarray(
        source[rng.integers(0, len(source), n) % len(source)]
    )
    w_up = np.ascontiguousarray(
        source[rng.integers(0, len(source), n) % len(source)]
    )
    if w_gate.shape[0] != n or w_up.shape[0] != n or n % 8:
        pytest.skip("IQ4_NL fixture rows are not (N, 8-compatible)")

    x_bits = bf16(rng.normal(0.0, 0.1, size=(1, k)))

    runtime = get_hip_runtime()
    bufs = []

    def dev(a: np.ndarray):
        b = malloc(a.nbytes, runtime=runtime)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes, runtime=runtime)
        return b

    try:
        x_b = dev(x_bits)
        w_gate_b = dev(w_gate)
        w_up_b = dev(w_up)
        ga_b = malloc(n * 2, runtime=runtime); bufs.append(ga_b)
        ub_b = malloc(n * 2, runtime=runtime); bufs.append(ub_b)
        ref_b = malloc(n * 2, runtime=runtime); bufs.append(ref_b)
        got_b = malloc(n * 2, runtime=runtime); bufs.append(got_b)
        for b in (ga_b, ub_b, ref_b, got_b):
            copy_host_to_device(
                b,
                host_array_ptr(np.zeros(n, dtype=np.uint16)),
                n * 2,
                runtime=runtime,
            )

        silulib = gguf_iq_dense.build_gguf_iq_dense(load=True)

        # unfused chain: NL local32 single (gate) + NL local32 single
        # (up) + silu_mul
        gguf_iq_dense.launch_local32(
            x_b.ptr, w_gate_b.ptr, ga_b.ptr, 1, k, n,
            quant="gguf_iq4_nl", library=silulib, runtime=runtime,
        )
        gguf_iq_dense.launch_local32(
            x_b.ptr, w_up_b.ptr, ub_b.ptr, 1, k, n,
            quant="gguf_iq4_nl", library=silulib, runtime=runtime,
        )
        from hipengine.kernels.hip_gfx1100.fused import (
            build_paro_silu,
        )

        silulib2 = build_paro_silu(load=True)
        silu_mul_separate_out_bf16(
            ga_b.ptr, ub_b.ptr, ref_b.ptr, 1, n, library=silulib2, runtime=runtime
        )
        # fused pair: gate-first matches geometry.
        from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import (
            build_gguf_iq4_q4_pair,
        )

        pairlib = build_gguf_iq4_q4_pair(load=True)
        gguf_iq4nl_nl_pair_silu_bf16_bf16_out(
            x_b.ptr,
            w_gate_b.ptr,
            w_up_b.ptr,
            got_b.ptr,
            1,
            k,
            n,
            library=pairlib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        ref = np.zeros(n, dtype=np.uint16)
        got = np.zeros(n, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(ref), ref_b, n * 2, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_b, n * 2, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)

    assert np.array_equal(got, ref), (
        "IQ4_NL/IQ4_NL pair+SiLU diverged from single/single/silu_mul: "
        f"{int((got != ref).sum())}/{ref.size} bf16 outputs differ"
    )
