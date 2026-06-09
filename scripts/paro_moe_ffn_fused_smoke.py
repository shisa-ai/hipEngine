"""One-shot launch of the B3 fused selected-expert PARO MoE FFN megakernel.

Intended for ``rocprofv3 --kernel-trace`` smoke: builds (or loads cached) the
kernel, runs it once on a tiny synthetic input (one block per selected
(token, expert) row), and synchronizes. Pass ``--compiler-version-file`` +
``--require-cached-build`` so the profiled process never spawns hipcc.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return (((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)).reshape(f32.shape)


def _awq_stack(rng, out_f: int, in_f: int, E: int, group_size: int):
    out_packed = out_f // 8
    groups = in_f // group_size
    qw = rng.integers(0, 2**32, size=(E, out_packed, in_f), dtype=np.uint64).astype(np.uint32).view(np.int32)
    qz = rng.integers(0, 2**32, size=(E, groups, out_packed), dtype=np.uint64).astype(np.uint32).view(np.int32)
    sc = rng.uniform(0.001, 0.04, size=(E, groups, out_f)).astype(np.float32)
    return qw, qz, sc


def _calib(rng, dim: int, group_size: int, krot: int):
    half = group_size // 2
    pairs = np.zeros((krot, dim), np.int16)
    for r in range(krot):
        for g in range(dim // group_size):
            for lane in range(half):
                pairs[r, g * group_size + 2 * lane] = 2 * lane
                pairs[r, g * group_size + 2 * lane + 1] = 2 * lane + 1
    theta = rng.uniform(-1, 1, (krot, dim // 2)).astype(np.float32)
    cscale = rng.uniform(0.5, 1.5, dim).astype(np.float32)
    return pairs, theta, cscale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--x-rows", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn-len", type=int, default=768)
    ap.add_argument("--num-experts", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--krot", type=int, default=1)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.paro_moe_ffn_fused import (
        build_paro_moe_ffn_fused,
        paro_selected_ffn_fused_bf16_bf16_out,
    )

    library = build_paro_moe_ffn_fused(load=True, require_cached=args.require_cached_build)
    runtime = get_hip_runtime()

    rng = np.random.default_rng(7)
    H, F, E, gs, kr = args.hidden, args.ffn_len, args.num_experts, args.group_size, args.krot
    x = _f32_to_bf16_u16((rng.standard_normal((args.x_rows, H)) * 0.1).astype(np.float32))
    selected = np.ascontiguousarray((np.arange(args.rows) % E).astype(np.int64))
    gqw, gqz, gsc = _awq_stack(rng, F, H, E, gs)
    uqw, uqz, usc = _awq_stack(rng, F, H, E, gs)
    dqw, dqz, dsc = _awq_stack(rng, H, F, E, gs)
    r1p, r1t, r1s = _calib(rng, H, gs, kr)
    drp, drt, drs = _calib(rng, F, gs, kr)
    out = np.zeros((args.rows, H), dtype=np.uint16)

    def up(arr, cast=None):
        a = cast(arr) if cast is not None else np.ascontiguousarray(arr)
        b = malloc(a.nbytes)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes)
        bufs.append(b)
        held.append(a)
        return b.ptr

    bufs: list = []
    held: list = []
    try:
        ptrs = [
            up(x),
            up(selected),
            up(gqw, lambda a: np.ascontiguousarray(a, np.int32)),
            up(gqz, lambda a: np.ascontiguousarray(a, np.int32)),
            up(gsc, _f32_to_bf16_u16),
            up(uqw, lambda a: np.ascontiguousarray(a, np.int32)),
            up(uqz, lambda a: np.ascontiguousarray(a, np.int32)),
            up(usc, _f32_to_bf16_u16),
            up(dqw, lambda a: np.ascontiguousarray(a, np.int32)),
            up(dqz, lambda a: np.ascontiguousarray(a, np.int32)),
            up(dsc, _f32_to_bf16_u16),
            up(r1p, lambda a: np.ascontiguousarray(a, np.int16)),
            up(r1t, _f32_to_bf16_u16),
            up(r1s, _f32_to_bf16_u16),
            up(drp, lambda a: np.ascontiguousarray(a, np.int16)),
            up(drt, _f32_to_bf16_u16),
            up(drs, _f32_to_bf16_u16),
        ]
        o_buf = malloc(out.nbytes)
        bufs.append(o_buf)
        paro_selected_ffn_fused_bf16_bf16_out(
            *ptrs, o_buf.ptr,
            args.x_rows, args.rows, E, H, F, gs, kr,
            threads=256, library=library,
        )
        runtime.device_synchronize()
        print(f"fused PARO MoE FFN launched: rows={args.rows} hidden={H} ffn_len={F} experts={E} krot={kr}")
    finally:
        for buf in bufs:
            free(buf)


if __name__ == "__main__":
    main()
