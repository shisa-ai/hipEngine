"""Native DP-2 loan shares one cast without changing either consumer output."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import free, memory_stats
from tests.test_gpu_qwen4_exp_gr_hip import _upload, _download, _alloc

try:
    ctypes.CDLL("libamdhip64.so")
    HIP_AVAILABLE = True
except OSError:
    HIP_AVAILABLE = False
pytestmark = pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime unavailable")


@pytest.mark.parametrize("rows,roles,ns", [
    (1024, ("attn_qkv", "attn_gate"), (10240, 6144)),
    (4096, ("shared_gate", "shared_up"), (640, 640)),
])
def test_native_pair_nondefault_stream_exact_and_bounded(monkeypatch, rows, roles, ns):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
    from hipengine.runtime import gguf_linear as linear
    from hipengine.runtime import qwen4_exp_runner as runner

    runtime = get_hip_runtime()
    register_gfx1151_kernels(replace=True)
    monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", ",".join(map(str, range(13, 48))))
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS", raising=False)
    k = 2560
    rng = np.random.default_rng(3307)
    x = rng.normal(0, .3, (rows, k)).astype(np.float32)
    x[0, :4] = np.array([0, -0., 2**-25, 1 + 2**-11], np.float32)
    blocks = []
    for n in ns:
        w = np.empty((n, k // 32, 34), np.uint8)
        w[:, :, :2] = np.array([.01], np.float16).view(np.uint8)
        w[:, :, 2:] = rng.integers(-127, 128, (n, k // 32, 32), dtype=np.int8).view(np.uint8)
        blocks.append(w)
    allocations = []
    before = memory_stats()
    stream = runtime.stream_create()
    casts = []
    cast = wide.f32_to_f16
    monkeypatch.setattr(wide, "f32_to_f16", lambda *a, **kw: casts.append(a) or cast(*a, **kw))
    try:
        xd = _upload(x, runtime, allocations)
        wd = [_upload(w, runtime, allocations) for w in blocks]
        od = [_alloc((rows, n), np.float32, runtime, allocations) for n in ns]
        hd = _alloc((rows, k), np.float16, runtime, allocations)
        weights = [SimpleNamespace(backend="hip_gfx1151", spec=SimpleNamespace(
            layout="raw_gguf", quant_key="gguf_q8_0", slot_path=f"layers.24.{role}",
            source=SimpleNamespace(shape=(n, k))), allocation=lambda _, d=d: SimpleNamespace(
                tensor=SimpleNamespace(ptr=d.ptr), buffer=d)) for role, n, d in zip(roles, ns, wd)]
        snapshots = []
        with linear.raw_k_prefill_variant_session("coltile8"), linear.wide_f16_activation_session(
                workspace_ptr=hd.ptr, workspace_nbytes=hd.nbytes):
            for enabled in ("0", "1", "1", "0"):
                monkeypatch.setenv(linear.WIDE_F16_PAIR_ENV, enabled)
                start = len(casts)
                live = memory_stats()
                runner._launch_qwen4_exp_dense_pair(*weights, xd.ptr, od[0].ptr, od[1].ptr,
                    rows, k, *ns, stream=stream, runtime=runtime)
                runtime.stream_synchronize(stream)
                assert len(casts) - start == (1 if enabled == "1" else 2)
                assert all(c[2] == rows * k for c in casts[start:])
                assert memory_stats()["current_allocated_bytes"] == live["current_allocated_bytes"]
                snapshots.append([_download(d, (rows, n), np.float32, runtime).tobytes()
                                  for d, n in zip(od, ns)])
                staged = _download(hd, (rows, k), np.float16, runtime)
                assert staged.tobytes() == x.astype(np.float16).tobytes()
        assert all(s == snapshots[0] for s in snapshots)
        for bits, n in zip(snapshots[0], ns):
            assert np.isfinite(np.frombuffer(bits, dtype=np.float32)).all()
        assert _download(xd, x.shape, np.float32, runtime).tobytes() == x.tobytes()
        for d, w in zip(wd, blocks):
            assert _download(d, w.shape, np.uint8, runtime).tobytes() == w.tobytes()
    finally:
        runtime.stream_destroy(stream)
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    for field in ("active_allocations", "current_allocated_bytes"):
        assert memory_stats()[field] == before[field]
