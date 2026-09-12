"""Correctness gate for the Q6_K T16 small-B rowtile GEMV (verify lm-head path).

The rowtile kernel (reads each weight tile once, accumulates ROW_TILE rows) must
be BIT-IDENTICAL to the per-row t16 decode kernel. The direct rowtile kernel
handles rows 2-8; packed serving verifies larger row counts by chunking into
2-8-row launches. The per-row decode kernel is the exact reference. Skips
without HIP or the local 35B GGUF fixture.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_q6_k_t16_rowtile_matches_per_row_decode(
    monkeypatch: pytest.MonkeyPatch,
    hip_test_target_arch: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", hip_test_target_arch)
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")

    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
    from hipengine.core.memory import malloc, free, host_array_ptr
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession, _small_b_rowtile_chunks
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        gguf_q6_k_t16_gemv_decode_bf16_f32_out as decode,
        gguf_q6_k_t16_gemv_rowtile_bf16_f32_out as rowtile,
    )

    rt = get_hip_runtime()
    failures = []
    with Qwen35GGUFResidentSession(MODEL, use_wmma_prefill=True, use_gemv_decode=True) as s:
        w = s.runner.weights.root("lm_head").allocation("tiles").tensor
        H = s.runner.hidden_size
        V = s.runner.vocab_size
        rng = np.random.default_rng(0)
        for rows in (2, 3, 4, 5, 6, 7, 8, 12):
            xh = (rng.standard_normal(rows * H).astype(np.float32) * 0.2).astype(np.float16).view(np.uint16)
            x = malloc(rows * H * 2, runtime=rt)
            rt.memcpy(x.ptr, host_array_ptr(np.ascontiguousarray(xh)), xh.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            ref = malloc(rows * V * 4, runtime=rt)
            got = malloc(rows * V * 4, runtime=rt)
            for r in range(rows):
                decode(x.ptr + r * H * 2, w.ptr, ref.ptr + r * V * 4, 1, H, V, runtime=rt)
            row_offset = 0
            for chunk_rows in _small_b_rowtile_chunks(rows, max_chunk=8):
                rowtile(
                    x.ptr + row_offset * H * 2,
                    w.ptr,
                    got.ptr + row_offset * V * 4,
                    chunk_rows,
                    H,
                    V,
                    runtime=rt,
                )
                row_offset += chunk_rows
            rt.device_synchronize()
            a = np.empty((rows * V,), dtype=np.float32)
            b = np.empty((rows * V,), dtype=np.float32)
            rt.memcpy(host_array_ptr(a), ref.ptr, a.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            rt.memcpy(host_array_ptr(b), got.ptr, b.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            max_abs = float(np.max(np.abs(a - b)))
            if max_abs != 0.0:
                failures.append(f"rows={rows} max_abs={max_abs}")
            for buf in (x, ref, got):
                free(buf, runtime=rt)
    assert not failures, "rowtile != per-row decode: " + "; ".join(failures)
