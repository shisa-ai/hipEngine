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
)
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense  # noqa: E402
from hipengine.kernels.hip_gfx1100.quant import (  # noqa: E402
    gguf_t16_selected_gemv as t16_gemv,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (  # noqa: E402
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, is_registered  # noqa: E402
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16  # noqa: E402
from tests._gguf_synthetic_weights import make_q4_k_weight  # noqa: E402

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