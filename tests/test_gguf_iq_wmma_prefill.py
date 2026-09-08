"""W4A16 prefill for raw dense GGUF IQ weights.

The route expands raw IQ blocks into fp16 WMMA operands and reads activations
straight from the caller's bf16 buffer, so unlike the integer-MMQ route it has
no quantized activation plane. Measured on real tensors it lands at the strict
GEMV's accuracy (both at the bf16 output floor) while the integer MMQ sits
~3.4x above; it is 1.4-2.1x slower than the integer MMQ and 5-8x faster than
the GEMV, so it is registered but not routed by any dispatch policy.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.backends import backend_package_capability, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_wmma_prefill as w4a16
from hipengine.kernels.registry import KernelKey, is_registered

FIXTURE = Path(__file__).parent / "fixtures/gguf_ud"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


def bf16(x):
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + 0x7fff + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_f32(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


@pytest.fixture(scope="module")
def library():
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    return w4a16.build_gguf_iq_wmma_prefill(
        load=True,
        compiler_version=Path(version_file).read_text() if version_file else None,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1")


# ---------------------------------------------------------------- registration


@pytest.mark.parametrize("backend", ("hip_gfx1100", "hip_gfx1151"))
@pytest.mark.parametrize("quant", sorted(w4a16.QUANTS))
def test_registered_on_both_hip_backends(backend, quant):
    load_backend_kernel_package(backend)
    assert is_registered(KernelKey(backend, "linear", quant, w4a16._VARIANT))


def test_is_the_gfx1151_dense_iq_prefill_default():
    """W4A16 owns dense raw-IQ prefill on gfx1151.

    Scored against hipEngine strict, as EXECUTION-PROFILES.md section 6
    specifies, it passes every threshold in the calibrated envelope
    (mean 0.000827, p95 0.004547, p99 0.012475, max 0.023513, top-1 100%)
    while the integer-MMQ alternative breaches the 5e-2 absolute maximum-row
    ceiling at 0.170390. It costs 12% throughput: 150.9 against 171.9 tok/s.
    """
    load_backend_kernel_package("hip_gfx1151")
    policy = backend_package_capability(
        "hip_gfx1151", "GGUF_IQ_DENSE_PREFILL_POLICY", {})
    assert set(policy) == {"gguf_iq4_xs", "gguf_iq3_xxs", "gguf_iq3_s", "gguf_iq4_nl"}
    for quant, entry in policy.items():
        assert entry["variant"] == w4a16._VARIANT, quant


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize("kwargs", [
    dict(quant="gguf_q4_k"),                      # unsupported quant
    dict(output="f32"),                           # kernel writes bf16 only
    dict(rows=0),
    dict(in_features=255),                        # K not block-aligned
    dict(out_features=0),
    dict(x_ptr=0),
])
def test_rejects_invalid_arguments_before_building(kwargs, monkeypatch):
    monkeypatch.setattr(w4a16, "_default_library",
                        lambda: pytest.fail("built before validation"))
    args = dict(x_ptr=1, qweight_ptr=2, out_ptr=3, rows=8, in_features=256,
                out_features=128, quant="gguf_iq4_xs")
    args.update(kwargs)
    with pytest.raises(ValueError):
        w4a16.launch(**args)


def test_iq4_nl_block_alignment_is_32_not_256():
    """IQ4_NL stores 32 elements per block, so K=32 is valid for it alone."""
    assert w4a16._BLOCK_ELEMENTS["gguf_iq4_nl"] == 32
    with pytest.raises(ValueError):
        w4a16.launch(1, 2, 3, rows=8, in_features=32, out_features=128,
                     quant="gguf_iq4_xs")


# ------------------------------------------------------------------- numerics


@pytest.mark.parametrize("quant", ("IQ4_XS", "IQ4_NL", "IQ3_S", "Q3_K"))
@pytest.mark.parametrize("rows", (16, 64))
def test_matches_the_strict_gemv_on_real_rows(library, quant, rows):
    """W4A16 must agree with the strict dense GEMV, the exact reference path.

    Both write bf16, so the comparison floor is the output format rather than
    either kernel's arithmetic.
    """
    import json
    from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                       free, host_array_ptr, malloc)
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import (
        build_gguf_iq_dense, launch as gemv)

    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == quant), None)
    if entry is None:
        pytest.skip(f"no {quant} fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        weights = data[entry["key"] + "_f32"]
    # The fixture holds a couple of real rows; repeat them to a tile-aligned
    # output width so the comparison exercises the real codec at real K.
    raw = np.ascontiguousarray(source[np.arange(64) % len(source)])
    n, k = len(raw), weights.shape[1]
    if k % (32 if quant == "IQ4_NL" else 256):
        pytest.skip(f"{quant} fixture K={k} is not block-aligned")

    x = bf16(np.random.default_rng(17).normal(0, 0.1, (rows, k)))
    dense_lib = build_gguf_iq_dense()
    got = np.zeros((rows, n), dtype=np.uint16)
    ref = np.zeros((rows, n), dtype=np.uint16)
    bufs = []
    try:
        def dev(a):
            b = malloc(a.nbytes); bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes); return b
        x_b, w_b = dev(x), dev(raw)
        o_b = malloc(got.nbytes); bufs.append(o_b)
        r_b = malloc(ref.nbytes); bufs.append(r_b)
        qkey = "gguf_" + quant.lower()
        gemv(x_b.ptr, w_b.ptr, r_b.ptr, rows, k, n, quant=qkey,
             output="bf16", library=dense_lib)
        w4a16.launch(x_b.ptr, w_b.ptr, o_b.ptr, rows, k, n,
                     quant=qkey, library=library)
        copy_device_to_host(host_array_ptr(ref), r_b, ref.nbytes)
        copy_device_to_host(host_array_ptr(got), o_b, got.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    a = bf16_f32(ref).astype(np.float64)
    c = bf16_f32(got).astype(np.float64)
    assert np.isfinite(c).all()
    scale = max(float(np.abs(a).max()), 1e-30)
    assert float(np.abs(a - c).max()) / scale <= 0.02, "W4A16 diverged from the GEMV"
    assert float(np.corrcoef(a.ravel(), c.ravel())[0, 1]) >= 0.9999
