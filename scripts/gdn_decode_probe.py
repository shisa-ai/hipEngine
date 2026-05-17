#!/usr/bin/env python3
"""Synthetic Qwen3.5 linear-attention GDN decode-shape probe.

This helper launches ``qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel`` at the
Qwen3.5/PARO c=1 decode shape used by the resident runner:

* num_k_heads=16, head_k_dim=128
* num_v_heads=32, head_v_dim=128
* one block per value head, 128 threads/block from the C ABI wrapper

Run it under ``rocprofv3 --kernel-trace`` and group rows by kernel name to audit
median duration and resource metadata (VGPR, scratch, LDS, workgroup size).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear_attn import (
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--reps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument(
        "--dtypes",
        default="bf16,fp16",
        help="Comma-separated lowp input variants to run: bf16, fp16.",
    )
    args = parser.parse_args()

    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    dtypes = [value.strip() for value in args.dtypes.split(",") if value.strip()]
    if not dtypes or any(value not in {"bf16", "fp16"} for value in dtypes):
        raise ValueError("--dtypes must contain only bf16 and/or fp16")

    compiler_version = args.compiler_version_file.read_text() if args.compiler_version_file is not None else None
    library = build_qwen35_linear_attn_gdn(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()

    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    eps = 1.0e-6
    key_dim = num_k_heads * head_k_dim
    conv_dim = 2 * key_dim + num_v_heads * head_v_dim

    conv_out = np.asarray([((idx * 7) % 31 - 15) * 0.002 for idx in range(conv_dim)], dtype=np.float32)
    gate_f32 = np.asarray([((idx * 5) % 17 - 8) * 0.01 for idx in range(num_v_heads * head_v_dim)], dtype=np.float32)
    a_f32 = np.asarray([((idx * 3) % 11 - 5) * 0.01 for idx in range(num_v_heads)], dtype=np.float32)
    b_f32 = np.asarray([((idx * 7) % 13 - 6) * 0.01 for idx in range(num_v_heads)], dtype=np.float32)
    dt_bias = np.asarray([((idx * 11) % 19 - 9) * 0.005 for idx in range(num_v_heads)], dtype=np.float32)
    a_log = np.asarray([-1.0 + (idx % 5) * 0.01 for idx in range(num_v_heads)], dtype=np.float32)
    norm_weight = np.asarray([1.0 + ((idx % 7) - 3) * 0.001 for idx in range(head_v_dim)], dtype=np.float32)
    recurrent_state = np.asarray(
        [((idx * 13) % 23 - 11) * 0.001 for idx in range(num_v_heads * head_k_dim * head_v_dim)],
        dtype=np.float32,
    )

    conv_out_d = _to_device(conv_out, runtime)
    dt_bias_d = _to_device(dt_bias, runtime)
    a_log_d = _to_device(a_log, runtime)
    norm_weight_d = _to_device(norm_weight, runtime)

    gate_bf16_d = _to_device(_float32_to_bf16_bits(gate_f32), runtime)
    a_bf16_d = _to_device(_float32_to_bf16_bits(a_f32), runtime)
    b_bf16_d = _to_device(_float32_to_bf16_bits(b_f32), runtime)
    gate_fp16_d = _to_device(gate_f32.astype(np.float16), runtime)
    a_fp16_d = _to_device(a_f32.astype(np.float16), runtime)
    b_fp16_d = _to_device(b_f32.astype(np.float16), runtime)

    state_bf16_d = _to_device(recurrent_state, runtime)
    state_fp16_d = _to_device(recurrent_state, runtime)
    out_bf16_d = malloc(num_v_heads * head_v_dim * np.dtype(np.float32).itemsize, runtime=runtime)
    out_fp16_d = malloc(num_v_heads * head_v_dim * np.dtype(np.float32).itemsize, runtime=runtime)

    cases: list[dict[str, object]] = []

    def launch(dtype: str) -> None:
        if dtype == "bf16":
            qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
                conv_out_d.ptr,
                gate_bf16_d.ptr,
                a_bf16_d.ptr,
                b_bf16_d.ptr,
                dt_bias_d.ptr,
                a_log_d.ptr,
                norm_weight_d.ptr,
                state_bf16_d.ptr,
                out_bf16_d.ptr,
                eps,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                library=library,
                runtime=runtime,
            )
        else:
            qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16(
                conv_out_d.ptr,
                gate_fp16_d.ptr,
                a_fp16_d.ptr,
                b_fp16_d.ptr,
                dt_bias_d.ptr,
                a_log_d.ptr,
                norm_weight_d.ptr,
                state_fp16_d.ptr,
                out_fp16_d.ptr,
                eps,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                library=library,
                runtime=runtime,
            )

    for dtype in dtypes:
        for _ in range(args.warmup):
            launch(dtype)
    runtime.device_synchronize()

    start = time.perf_counter()
    for dtype in dtypes:
        for _ in range(args.reps):
            launch(dtype)
        cases.append(
            {
                "case": f"gdn_recurrent_{dtype}",
                "reps": args.reps,
                "num_k_heads": num_k_heads,
                "num_v_heads": num_v_heads,
                "head_k_dim": head_k_dim,
                "head_v_dim": head_v_dim,
                "threads": 128,
                "dynamic_shared_bytes": int(((2 * head_k_dim) + (3 * 128) + head_v_dim) * 4),
            }
        )
    runtime.device_synchronize()
    elapsed = time.perf_counter() - start

    print(json.dumps({"elapsed_s": elapsed, "cases": cases}, indent=2))
    return 0


def _to_device(array: np.ndarray, runtime) -> object:
    arr = np.ascontiguousarray(array)
    buf = malloc(arr.nbytes, runtime=runtime)
    copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


def _float32_to_bf16_bits(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb.astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


if __name__ == "__main__":
    raise SystemExit(main())
