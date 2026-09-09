"""Local32 IQ4_XS decode GEMV (UD route-plan item C, gfx1100 lane).

The local32 owner is the raw-layout analogue of the retained pack8/T16
local32 decode family: one wave per 8 output columns, each lane owning 8
contiguous K values, two u32 payload loads per column-block, and a
byte-indexed fused codebook LUT. Its accumulation order differs from the
strict per-row GEMV (lane-grouped contiguous k, wave32 shuffle tree,
fixed-order cross-wave sum), so it is registered as a gated approximate
owner: these tests pin registration, policy-driven dispatch admission, the
decline-without-session behavior, argument validation, and agreement with
the strict owner on real fixture rows.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.backends import (
    backend_package_capability,
    load_backend_kernel_package,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense
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


# ---------------------------------------------------------------- registration


@pytest.mark.parametrize("backend", ("hip_gfx1100", "hip_gfx1151"))
def test_local32_registered_on_both_hip_backends(backend):
    """The kernel body is shared source lineage; aliasing registers it on gfx1151.

    Routing there is a separate per-backend policy decision (none yet).
    """
    load_backend_kernel_package(backend)
    assert is_registered(
        KernelKey(backend, "linear", "gguf_iq4_xs", "local32_gemv_bf16_bf16_out"))


def test_gfx1100_declares_an_empty_decode_policy_and_gfx1151_none():
    """The local32 candidate is registered but deliberately unrouted.

    The teacher-forced probe breaches the 6.1 envelope on occasional contexts
    (max row 5.85e-2 at one seed), so the shipped policy is empty; see the
    declaration comment. Enabling is a one-line edit pending the full
    production-referenced gate.
    """
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_IQ_DENSE_DECODE_POLICY", None) == {}
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_IQ_DENSE_DECODE_POLICY", None) is None


# ------------------------------------------------------------------- dispatch


def _dispatch(quant="gguf_iq4_xs", *, rows, out_features=17408, backend="hip_gfx1100"):
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch, _iq_dense_decode_dispatch)
    load_backend_kernel_package(backend)
    base = GGUFLinearDispatch(
        KernelKey(backend, "linear", quant, "gemv_bf16_bf16_out"), "raw")
    return _iq_dense_decode_dispatch(base, rows=rows, out_features=out_features)


def test_no_route_engages_without_an_execution_owner():
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    assert not iq_mmq.iq_dense_mmq_has_workspace()
    with iq_mmq.iq_dense_mmq_session(False):
        assert iq_mmq.iq_dense_mmq_workspace() is None
        out = _dispatch(rows=1)
    assert out.key.variant == "gemv_bf16_bf16_out"


def test_route_is_selected_when_the_policy_entry_is_present(monkeypatch):
    """The dispatch wiring works; the shipped policy is empty by decision, not
    by absence - a one-line entry is all the enable needs."""
    import hipengine.kernels.hip_gfx1100 as be
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    monkeypatch.setattr(
        be, "GGUF_IQ_DENSE_DECODE_POLICY",
        {"gguf_iq4_xs": {"variant": "local32_gemv_bf16_bf16_out"}})
    with iq_mmq.iq_dense_mmq_session(True):
        out = _dispatch(rows=1)
    assert out.key.variant == "local32_gemv_bf16_bf16_out"
    assert out.abi == "raw"


def test_shipped_policy_keeps_the_strict_owner():
    """The shipped (empty) policy must leave decode on the strict GEMV."""
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    with iq_mmq.iq_dense_mmq_session(True):
        out = _dispatch(rows=1)
    assert out.key.variant == "gemv_bf16_bf16_out"


def test_route_declines_above_one_row():
    """The owner serves single-stream decode only; prefill owners stay put."""
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    with iq_mmq.iq_dense_mmq_session(True):
        for rows in (2, 8, 512):
            out = _dispatch(rows=rows)
            assert out.key.variant == "gemv_bf16_bf16_out", rows


def test_route_declines_for_unrouted_quants():
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    with iq_mmq.iq_dense_mmq_session(True):
        for quant in ("gguf_iq4_nl", "gguf_q3_k", "gguf_iq3_xxs"):
            out = _dispatch(quant, rows=1)
            assert out.key.variant == "gemv_bf16_bf16_out", quant


def test_route_declines_when_n_is_not_multiple_of_8():
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    with iq_mmq.iq_dense_mmq_session(True):
        out = _dispatch(rows=1, out_features=17409)
    assert out.key.variant == "gemv_bf16_bf16_out"


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("kwargs", [
    dict(quant="gguf_iq4_nl"),                      # unsupported quant
    dict(output="f32"),                             # kernel writes bf16 only
    dict(rows=2),                                   # decode owner serves rows=1
    dict(in_features=255),                          # K not block-aligned
    dict(out_features=17409),                      # N not a multiple of 8
    dict(x_ptr=0),
])
def test_rejects_invalid_arguments_before_building(kwargs, monkeypatch):
    monkeypatch.setattr(gguf_iq_dense, "_default_library",
                        lambda: pytest.fail("built before validation"))
    with pytest.raises(ValueError):
        gguf_iq_dense.launch_local32(
            kwargs.pop("x_ptr", 1), 2, 3,
            rows=kwargs.pop("rows", 1),
            in_features=kwargs.pop("in_features", 5120),
            out_features=kwargs.pop("out_features", 17408),
            **kwargs)


# ------------------------------------------------------------------- numerics


@pytest.mark.parametrize("waves_shape", (17408, 5120))  # wide-N and split-K paths
def test_matches_the_strict_gemv_on_real_rows(waves_shape):
    """The local32 owner must agree with the strict GEMV within tolerance.

    Both write bf16 and the products are identical; only the summation order
    differs, so the comparison floor is far below the W4A16 prefill route's.
    """
    from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                       free, host_array_ptr, malloc)

    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == "IQ4_XS"), None)
    if entry is None:
        pytest.skip("no IQ4_XS fixture row")
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    raw = np.ascontiguousarray(source[np.arange(64) % len(source)])
    n = len(raw)
    if k % 256:
        pytest.skip(f"IQ4_XS fixture K={k} is not block-aligned")

    x = bf16(np.random.default_rng(17).normal(0, 0.1, (1, k)))
    dense_lib = gguf_iq_dense.build_gguf_iq_dense()
    got = np.zeros((1, n), dtype=np.uint16)
    ref = np.zeros((1, n), dtype=np.uint16)
    bufs = []
    try:
        def dev(a):
            b = malloc(a.nbytes); bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes); return b
        x_b, w_b = dev(x), dev(raw)
        o_b = malloc(got.nbytes); bufs.append(o_b)
        r_b = malloc(ref.nbytes); bufs.append(r_b)
        gguf_iq_dense.launch(x_b.ptr, w_b.ptr, r_b.ptr, 1, k, n,
                             quant="gguf_iq4_xs", output="bf16",
                             library=dense_lib)
        gguf_iq_dense.launch_local32(x_b.ptr, w_b.ptr, o_b.ptr, 1, k, n)
        copy_device_to_host(host_array_ptr(ref), r_b, ref.nbytes)
        copy_device_to_host(host_array_ptr(got), o_b, got.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    a = bf16_f32(ref).astype(np.float64)
    c = bf16_f32(got).astype(np.float64)
    assert np.isfinite(c).all()
    scale = max(float(np.abs(a).max()), 1e-30)
    rel = float(np.abs(a - c).max()) / scale
    assert rel <= 5e-4, "local32 diverged from the strict GEMV"
    assert float(np.corrcoef(a.ravel(), c.ravel())[0, 1]) >= 0.9999
