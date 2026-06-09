"""One-shot launch of the B1 fused selected-expert GGUF Q4_K MoE FFN megakernel.

Intended for ``rocprofv3 --kernel-trace`` smoke: builds (or loads cached) the
kernel, runs it once on a tiny synthetic input, and synchronizes. Pass
``--compiler-version-file`` + ``--require-cached-build`` so the profiled process
never spawns hipcc.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--x-rows", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn-len", type=int, default=512)
    ap.add_argument("--num-experts", type=int, default=256)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_moe_ffn_fused import (
        build_gguf_q4_k_moe_ffn_fused,
        gguf_q4_k_selected_ffn_fused_bf16_bf16_out,
    )
    from tests._gguf_synthetic_weights import make_q4_k_weight

    library = build_gguf_q4_k_moe_ffn_fused(load=True, require_cached=args.require_cached_build)
    runtime = get_hip_runtime()

    rng = np.random.default_rng(7)
    hidden, ffn_len, E = args.hidden, args.ffn_len, args.num_experts
    x = (_f32_to_bf16_u16((rng.standard_normal((args.x_rows, hidden)) * 1e-3).astype(np.float32)))
    selected = np.ascontiguousarray((np.arange(args.rows) % E).astype(np.int64))
    gate = np.ascontiguousarray(np.stack([make_q4_k_weight(ffn_len, hidden)] * E))
    up = gate
    down = np.ascontiguousarray(np.stack([make_q4_k_weight(hidden, ffn_len)] * E))
    out = np.zeros((args.rows, hidden), dtype=np.uint16)

    bufs = []
    try:
        x_buf = malloc(x.nbytes); copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes); bufs.append(x_buf)
        s_buf = malloc(selected.nbytes); copy_host_to_device(s_buf, host_array_ptr(selected), selected.nbytes); bufs.append(s_buf)
        g_buf = malloc(gate.nbytes); copy_host_to_device(g_buf, host_array_ptr(gate), gate.nbytes); bufs.append(g_buf)
        u_buf = malloc(up.nbytes); copy_host_to_device(u_buf, host_array_ptr(up), up.nbytes); bufs.append(u_buf)
        d_buf = malloc(down.nbytes); copy_host_to_device(d_buf, host_array_ptr(down), down.nbytes); bufs.append(d_buf)
        o_buf = malloc(out.nbytes); bufs.append(o_buf)
        gguf_q4_k_selected_ffn_fused_bf16_bf16_out(
            x_buf.ptr, s_buf.ptr, g_buf.ptr, u_buf.ptr, d_buf.ptr, o_buf.ptr,
            args.x_rows, args.rows, E, hidden, ffn_len, threads=256, library=library,
        )
        runtime.device_synchronize()
        print(f"fused MoE FFN launched: rows={args.rows} hidden={hidden} ffn_len={ffn_len} experts={E}")
    finally:
        for buf in bufs:
            free(buf)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return (((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)).reshape(f32.shape)


if __name__ == "__main__":
    main()
