#!/usr/bin/env python3
"""Cached-only rocprofv3 smoke for the raw-Q4_K x DS4-Q8_1 MMQ leaf kernels.

Prebuilt by the pytest run outside rocprof; this child loads the exact
cached DSOs with ``require_cached=True`` (no compiler subprocess can be
launched from the profiled process), packs a small DS4 activation fixture
and launches all four screen kernels (mmq32/wmma32 x ctl/vdr), then exits.
Run under ``rocprofv3 --kernel-trace`` and confirm the
``q4_k_q8_1_mmq32_dense_kernel`` and ``q4_k_q8_1_wmma32_dense_kernel``
instances appear with plausible durations.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    build_gguf_q4_k_q8_1_selected_prefill,
    gguf_q8_1_mmq_ds4_pack_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_mmq_prefill import (
    build_gguf_q4_k_q8_1_mmq_prefill,
    gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out,
    gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out,
    gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out,
    gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out,
)


def main() -> int:
    runtime = get_hip_runtime()
    pack_library = build_gguf_q4_k_q8_1_selected_prefill(
        load=True, require_cached=True
    )
    mmq_library = build_gguf_q4_k_q8_1_mmq_prefill(load=True, require_cached=True)

    rng = np.random.default_rng(0x6A34C1DE)
    rows, in_features, out_features = 32, 512, 256
    from tests.test_gpu_gguf_x8_selected_gemv import _weights

    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )[0]
    x_bits = (
        (rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32) + 0.002)
        .astype(np.float32)
        .view(np.uint32)
    )
    x_bits = (
        (x_bits + np.uint32(0x7FFF) + ((x_bits >> 16) & np.uint32(1))) >> 16
    ).astype(np.uint16)

    def upload(host: np.ndarray):
        host = np.ascontiguousarray(host)
        buffer = malloc(host.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
        return buffer

    buffers = []
    try:
        x_buf = upload(x_bits)
        w_buf = upload(qweight)
        xq_buf = malloc(rows * (in_features // 128) * 144, runtime=runtime)
        outs = {
            name: malloc(rows * out_features * 4, runtime=runtime)
            for name in (
                "mmq32_ctl",
                "mmq32_vdr",
                "wmma32_ctl",
                "wmma32_vdr",
            )
        }
        buffers.extend((x_buf, w_buf, xq_buf, *outs.values()))

        gguf_q8_1_mmq_ds4_pack_bf16(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=pack_library,
            runtime=runtime,
        )
        common = dict(
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            library=mmq_library,
            runtime=runtime,
        )
        gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, outs["mmq32_ctl"].ptr, **common
        )
        gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, outs["mmq32_vdr"].ptr, **common
        )
        gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, outs["wmma32_ctl"].ptr, **common
        )
        gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, outs["wmma32_vdr"].ptr, **common
        )
        runtime.device_synchronize()

        def download(name):
            host = np.empty((rows, out_features), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(host), outs[name], host.nbytes, runtime=runtime
            )
            return host

        mmq32_ctl = download("mmq32_ctl")
        mmq32_vdr = download("mmq32_vdr")
        wmma32_ctl = download("wmma32_ctl")
        wmma32_vdr = download("wmma32_vdr")
        if not np.array_equal(mmq32_ctl, mmq32_vdr):
            print("ERROR: mmq32 ctl/vdr outputs differ", file=sys.stderr)
            return 1
        if not np.array_equal(wmma32_ctl, wmma32_vdr):
            print("ERROR: wmma32 ctl/vdr outputs differ", file=sys.stderr)
            return 1
        if not np.allclose(mmq32_ctl, wmma32_ctl, rtol=1e-4, atol=1e-4):
            print("ERROR: mmq32/wmma32 classes disagree", file=sys.stderr)
            return 1
        print(
            "smoke ok: ctl/vdr bit-identical per class, classes agree, "
            f"finite={bool(np.isfinite(mmq32_ctl).all())}"
        )
        return 0
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


if __name__ == "__main__":
    raise SystemExit(main())
