#!/usr/bin/env python3
"""Cached-only rocprofv3 smoke for the q8_1-DP4A Q4_K load-reuse kernels.

Prebuilt by the pytest run outside rocprof; this child loads the exact
cached DSO with ``require_cached=True`` (no compiler subprocess can be
launched from the profiled process), launches the q8_1 producer, the
unamortized control kernel and the VDR kernel once each on a small
fixture, and exits. Run under ``rocprofv3 --kernel-trace`` and confirm
``q4_k_q8_1_dp4a_ctl_gemv_kernel`` and ``q4_k_q8_1_dp4a_vdr_gemv_kernel``
appear with plausible durations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_dp4a_vdr_gemv import (
    build_gguf_q4_k_q8_1_dp4a_vdr_gemv,
    gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out,
    gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out,
)


def main() -> int:
    runtime = get_hip_runtime()
    q4_library = build_gguf_q4_k_gemv(load=True, require_cached=True)
    vdr_library = build_gguf_q4_k_q8_1_dp4a_vdr_gemv(load=True, require_cached=True)

    rng = np.random.default_rng(0x6A34C0DE)
    rows, in_features, out_features = 1, 512, 256
    from tests.test_gpu_gguf_x8_selected_gemv import _weights

    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )[0]
    x_bits = (
        (rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32) + 0.002)
        .astype(np.float32)
        .view(np.uint32)
    )
    x_bits = ((x_bits + np.uint32(0x7FFF) + ((x_bits >> 16) & np.uint32(1))) >> 16).astype(np.uint16)

    def upload(host: np.ndarray):
        host = np.ascontiguousarray(host)
        buffer = malloc(host.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
        return buffer

    buffers = []
    try:
        x_buf = upload(x_bits)
        w_buf = upload(qweight)
        xq_buf = malloc(rows * (in_features // 32) * 36, runtime=runtime)
        ctl_buf = malloc(rows * out_features * 4, runtime=runtime)
        vdr_buf = malloc(rows * out_features * 4, runtime=runtime)
        buffers.extend((x_buf, w_buf, xq_buf, ctl_buf, vdr_buf))

        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr, xq_buf.ptr, rows, in_features, library=q4_library, runtime=runtime
        )
        common = dict(
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            library=vdr_library,
            runtime=runtime,
        )
        gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out(xq_buf.ptr, w_buf.ptr, ctl_buf.ptr, **common)
        gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out(xq_buf.ptr, w_buf.ptr, vdr_buf.ptr, **common)
        runtime.device_synchronize()

        ctl_out = np.empty((rows, out_features), dtype=np.float32)
        vdr_out = np.empty((rows, out_features), dtype=np.float32)
        copy_device_to_host(host_array_ptr(ctl_out), ctl_buf, ctl_out.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(vdr_out), vdr_buf, vdr_out.nbytes, runtime=runtime)
        if not np.array_equal(ctl_out, vdr_out):
            print("ERROR: ctl/vdr outputs differ", file=sys.stderr)
            return 1
        print(f"smoke ok: ctl/vdr bit-identical, finite={bool(np.isfinite(ctl_out).all())}")
        return 0
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


if __name__ == "__main__":
    raise SystemExit(main())
