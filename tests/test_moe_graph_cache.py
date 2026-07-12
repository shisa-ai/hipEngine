"""Unit gate for the per-layer MoE graph capture/replay cache (task #15).

``MoeGraphCache`` wraps a stateless capturable unit (the rows==1 MoE FFN) with
capture-on-first-use, a self-validating bit-exact parity check against an eager
reference run, replay thereafter, and an eager fallback on any
capture/instantiate failure or parity mismatch.  This test pins that contract on
a model-free stateless 2-kernel "ffn" (f32->bf16->f32) so the cache logic is
exercised without the full runner: first call captures, subsequent calls replay
bit-exactly while reading FRESH input written between replays, disabled mode is a
pure passthrough, and a key marked eager-only never captures.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="requires ROCm/libamdhip64.so")


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def _bf16_round(x: np.ndarray) -> np.ndarray:
    return (_to_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def _make_ffn():
    """Return (ffn, set_input, read_out, cleanup, out_ptr, out_nbytes, runtime).

    ``ffn(stream)`` runs a stateless f32->bf16->f32 round-trip reading the fixed
    input pointer and writing the fixed output pointer, mirroring the rows==1 MoE
    FFN's "recompute from fresh inputs each token" shape.
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.convert.cast import (
        bf16_to_f32,
        build_cast,
        f32_to_bf16,
    )

    rt = get_hip_runtime()
    cast_lib = build_cast(load=True)
    n = 4096
    x = malloc(n * 4)
    xb = malloc(n * 2)
    out = malloc(n * 4)
    bufs = [x, xb, out]

    # Warm (force JIT build/load) before any capture happens.
    warm = np.zeros(n, dtype=np.float32)
    copy_host_to_device(x, host_array_ptr(warm), warm.nbytes)
    f32_to_bf16(x.ptr, xb.ptr, n, library=cast_lib, runtime=rt)
    bf16_to_f32(xb.ptr, out.ptr, n, library=cast_lib, runtime=rt)
    rt.device_synchronize()

    def ffn(stream):
        f32_to_bf16(x.ptr, xb.ptr, n, stream=stream, library=cast_lib, runtime=rt)
        bf16_to_f32(xb.ptr, out.ptr, n, stream=stream, library=cast_lib, runtime=rt)

    def set_input(arr):
        copy_host_to_device(x, host_array_ptr(np.ascontiguousarray(arr, dtype=np.float32)), arr.nbytes)

    def read_out():
        dst = np.empty(n, dtype=np.float32)
        copy_device_to_host(host_array_ptr(dst), out, dst.nbytes)
        return dst

    def cleanup():
        for b in reversed(bufs):
            free(b)

    return ffn, set_input, read_out, cleanup, out.ptr, n * 4, rt, n


def test_disabled_cache_is_passthrough() -> None:
    from hipengine.runtime.moe_graph import MoeGraphCache

    ffn, set_input, read_out, cleanup, out_ptr, out_nbytes, rt, n = _make_ffn()
    cache = MoeGraphCache(rt, enabled=False)
    try:
        rng = np.random.default_rng(1)
        inp = rng.standard_normal(n).astype(np.float32)
        set_input(inp)
        kind = cache.run(("L", 0), eager=ffn, out_ptr=out_ptr, out_nbytes=out_nbytes, stream=0)
        rt.device_synchronize()
        assert kind == "eager"
        assert np.array_equal(read_out(), _bf16_round(inp))
        assert cache.stats["capture"] == 0
    finally:
        cache.close()
        cleanup()


def test_capture_then_replay_reads_fresh_input() -> None:
    from hipengine.runtime.moe_graph import MoeGraphCache

    ffn, set_input, read_out, cleanup, out_ptr, out_nbytes, rt, n = _make_ffn()
    cache = MoeGraphCache(rt, enabled=True)
    try:
        rng = np.random.default_rng(20260628)
        kinds = []
        for it in range(4):
            inp = rng.standard_normal(n).astype(np.float32) * (it + 1)
            set_input(inp)
            kind = cache.run(("L", 0), eager=ffn, out_ptr=out_ptr, out_nbytes=out_nbytes, stream=0)
            rt.device_synchronize()
            kinds.append(kind)
            assert np.array_equal(read_out(), _bf16_round(inp)), f"iter {it} diverged"
        assert kinds[0] == "capture"
        assert kinds[1:] == ["replay", "replay", "replay"]
        assert cache.stats["capture"] == 1
        assert cache.stats["replay"] == 3
    finally:
        cache.close()
        cleanup()


def test_eager_only_key_never_captures() -> None:
    from hipengine.runtime.moe_graph import MoeGraphCache

    ffn, set_input, read_out, cleanup, out_ptr, out_nbytes, rt, n = _make_ffn()
    cache = MoeGraphCache(rt, enabled=True)
    try:
        cache.mark_eager_only(("L", 7))
        inp = np.random.default_rng(3).standard_normal(n).astype(np.float32)
        set_input(inp)
        kind = cache.run(("L", 7), eager=ffn, out_ptr=out_ptr, out_nbytes=out_nbytes, stream=0)
        rt.device_synchronize()
        assert kind == "eager"
        assert np.array_equal(read_out(), _bf16_round(inp))
        assert cache.stats["capture"] == 0
    finally:
        cache.close()
        cleanup()
