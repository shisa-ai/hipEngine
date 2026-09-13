"""Q5 T16 dual fused-SiLU prefill gate (UD impact-list task 4).

Compares the new dense dual gate/up WMMA+SiLU kernel against the unfused
production chain (Q5 T16 single WMMA prefill x2 + silu_mul_separate_out_bf16)
on a real Q5_K tensor pair from the UD-K_M artifact. The Q4 dual precedent
requires bit-exactness: the dual preserves the single owner's K16 WMMA
association and the unfused BF16 projection boundary, so every output bit
must match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

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
    gguf_q5_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
    gguf_q5_k_t16_dense_dual_wmma_prefill_row128_silu_bf16_bf16_out,
    gguf_q5_k_t16_dense_dual_wmma_prefill_row32_silu_bf16_bf16_out,
    gguf_q5_k_t16_dense_dual_wmma_prefill_row48_silu_bf16_bf16_out,
    gguf_q5_k_t16_dense_dual_wmma_prefill_row64_silu_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16


def _bf16_bits(a: np.ndarray) -> np.ndarray:
    u32 = a.astype(np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 128, 64, 32])
    parser.add_argument("--json", default="")
    parser.add_argument(
        "--compiler-version-file",
        default="/tmp/ud-hipcc-version.txt",
    )
    args = parser.parse_args()

    reader = GGUFReader(args.model)
    gate_info = reader.tensor_info(f"blk.{args.layer}.ffn_gate.weight")
    up_info = reader.tensor_info(f"blk.{args.layer}.ffn_up.weight")
    assert gate_info.ggml_type_name == "Q5_K", gate_info.ggml_type_name
    assert up_info.ggml_type_name == "Q5_K", up_info.ggml_type_name
    n, k = int(gate_info.shape[0]), int(gate_info.shape[1])
    assert (int(up_info.shape[0]), int(up_info.shape[1])) == (n, k)

    gate_tiles = repack_gguf_q5_k_tile16(
        np.asarray(reader.tensor_data(gate_info.name))[None, ...]
    ).tiles
    up_tiles = repack_gguf_q5_k_tile16(
        np.asarray(reader.tensor_data(up_info.name))[None, ...]
    ).tiles

    runtime = get_hip_runtime()
    prefill_library = build_gguf_k_t16_selected_prefill(
        load=True, compiler_version=Path(args.compiler_version_file).read_text()
    )
    silu_library = build_paro_silu(load=True)

    variants = {
        512: gguf_q5_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
        128: gguf_q5_k_t16_dense_dual_wmma_prefill_row128_silu_bf16_bf16_out,
        64: gguf_q5_k_t16_dense_dual_wmma_prefill_row64_silu_bf16_bf16_out,
        48: gguf_q5_k_t16_dense_dual_wmma_prefill_row48_silu_bf16_bf16_out,
        32: gguf_q5_k_t16_dense_dual_wmma_prefill_row32_silu_bf16_bf16_out,
    }

    results = {}
    for rows in args.rows:
        rng = np.random.default_rng(0xC0FFEE + rows)
        x_bits = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, k)))
        expected_bits = np.zeros((rows, n), dtype=np.uint16)
        actual_bits = np.zeros_like(expected_bits)
        buffers = []
        try:
            x_dev = malloc(x_bits.nbytes, runtime=runtime)
            gate_dev = malloc(gate_tiles.nbytes, runtime=runtime)
            up_dev = malloc(up_tiles.nbytes, runtime=runtime)
            gate_out = malloc(expected_bits.nbytes, runtime=runtime)
            up_out = malloc(expected_bits.nbytes, runtime=runtime)
            control_dev = malloc(expected_bits.nbytes, runtime=runtime)
            candidate_dev = malloc(expected_bits.nbytes, runtime=runtime)
            buffers.extend(
                (x_dev, gate_dev, up_dev, gate_out, up_out, control_dev, candidate_dev)
            )
            copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
            copy_host_to_device(
                gate_dev, host_array_ptr(gate_tiles), runtime=runtime
            )
            copy_host_to_device(up_dev, host_array_ptr(up_tiles), runtime=runtime)

            for tiles_dev, out_dev in ((gate_dev, gate_out), (up_dev, up_out)):
                gguf_q5_k_t16_wmma_prefill_bf16_bf16_out(
                    x_dev.ptr,
                    tiles_dev.ptr,
                    out_dev.ptr,
                    rows,
                    k,
                    n,
                    library=prefill_library,
                    runtime=runtime,
                )
            silu_mul_separate_out_bf16(
                gate_out.ptr,
                up_out.ptr,
                control_dev.ptr,
                rows,
                n,
                library=silu_library,
                runtime=runtime,
            )
            variants[rows](
                x_dev.ptr,
                gate_dev.ptr,
                up_dev.ptr,
                candidate_dev.ptr,
                rows,
                k,
                n,
                library=prefill_library,
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

        mismatches = int(np.count_nonzero(actual_bits != expected_bits))
        results[rows] = {
            "bf16_mismatches": mismatches,
            "exact": mismatches == 0,
            "finite": bool(
                np.isfinite(
                    (actual_bits.astype(np.uint32) << 16).view(np.float32)
                ).all()
            ),
        }
        print(f"rows={rows}: mismatches={mismatches} exact={mismatches == 0}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
