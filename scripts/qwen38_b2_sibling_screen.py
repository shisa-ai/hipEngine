"""B2 item-3 screen: time F16 siblings vs BF16 owners at prefill shapes.

Physical gfx1151, warm JIT cache, deterministic synthetic weights. For each
(family, shape, rows) the driver alternates owner/sibling warmup then times
~60 iterations per arm with device synchronization. No throughput claim;
the screen decides whether the serving integration is worth building.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/lhl/hipEngine")
sys.path.insert(0, str(ROOT))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (  # noqa: E402
    build_gguf_k_t16_selected_prefill,
    gguf_q4_k_t16_wmma_prefill_bf16_bf16_out as q4_owner,
    gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out as q4_sibling,
    gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out as q4s_owner,
    gguf_q4_k_t16_wmma_prefill_shared_b_fp16_in_bf16_out as q4s_sibling,
    gguf_q5_k_t16_wmma_prefill_bf16_bf16_out as q5_owner,
    gguf_q5_k_t16_wmma_prefill_fp16_in_bf16_out as q5_sibling,
)
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16  # noqa: E402
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16  # noqa: E402
from tests._gguf_synthetic_weights import (  # noqa: E402
    make_q4_k_weight,
    make_q5_k_weight,
)

OUT = Path("/tmp/q38-b2-run/sibling-screen.json")
ITERS = 60
WARMUP = 10


def _float_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    u32 = values.astype(np.float32).view(np.uint32)
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def _time(fn, ptrs, rows, in_f, out_f, runtime):
    for _ in range(WARMUP):
        fn(*ptrs, rows, in_f, out_f, runtime=runtime)
    runtime.device_synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(*ptrs, rows, in_f, out_f, runtime=runtime)
    runtime.device_synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def main() -> None:
    ctypes.CDLL("libamdhip64.so")
    runtime = get_hip_runtime()
    build_gguf_k_t16_selected_prefill(load=True)
    rng = np.random.default_rng(7)

    cases = [
        ("q4_plain", q4_owner, q4_sibling, make_q4_k_weight,
         repack_gguf_q4_k_tile16, (5_120, 6_144), (72, 288)),
        ("q4_plain", q4_owner, q4_sibling, make_q4_k_weight,
         repack_gguf_q4_k_tile16, (5_120, 17_408), (72, 288)),
        ("q4_plain", q4_owner, q4_sibling, make_q4_k_weight,
         repack_gguf_q4_k_tile16, (17_408, 5_120), (72, 288)),
        ("q4_shared_b", q4s_owner, q4s_sibling, make_q4_k_weight,
         repack_gguf_q4_k_tile16, (5_120, 6_144), (72, 288)),
        ("q4_shared_b", q4s_owner, q4s_sibling, make_q4_k_weight,
         repack_gguf_q4_k_tile16, (5_120, 17_408), (72, 288)),
        ("q5", q5_owner, q5_sibling, make_q5_k_weight,
         repack_gguf_q5_k_tile16, (6_144, 5_120), (72, 288)),
    ]
    results = []
    for name, owner, sibling, make_w, repack, (in_f, out_f), rows_list in cases:
        raw = make_w(out_f, in_f)
        tiles = repack(raw[np.newaxis, :, :]).tiles
        for rows in rows_list:
            x_values = rng.standard_normal((rows, in_f)).astype(np.float32)
            x_bf16 = _float_to_bf16_bits(x_values)
            x_f16 = x_values.astype(np.float16)
            bufs = []
            try:
                x_dev = malloc(x_bf16.nbytes, runtime=runtime)
                x16_dev = malloc(x_f16.nbytes, runtime=runtime)
                tiles_dev = malloc(tiles.nbytes, runtime=runtime)
                o1 = malloc(rows * out_f * 2, runtime=runtime)
                o2 = malloc(rows * out_f * 2, runtime=runtime)
                bufs = [x_dev, x16_dev, tiles_dev, o1, o2]
                copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
                copy_host_to_device(
                    x16_dev, host_array_ptr(x_f16.view(np.uint16)), runtime=runtime
                )
                copy_host_to_device(
                    tiles_dev, host_array_ptr(tiles), runtime=runtime
                )
                ms_owner = _time(
                    owner, (x_dev.ptr, tiles_dev.ptr, o1.ptr), rows, in_f, out_f, runtime
                )
                ms_sib = _time(
                    sibling, (x16_dev.ptr, tiles_dev.ptr, o2.ptr), rows, in_f, out_f, runtime
                )
            finally:
                for b in bufs:
                    free(b, runtime=runtime)
            entry = {
                "family": name,
                "shape": [in_f, out_f],
                "rows": rows,
                "owner_ms": round(ms_owner, 4),
                "sibling_ms": round(ms_sib, 4),
                "ratio_sibling_over_owner": round(ms_sib / ms_owner, 4),
            }
            results.append(entry)
            print(entry, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"kind": "b2_sibling_screen", "results": results}, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
