"""Local32 IQ verifier sibling: one block owning 2-4 prompt rows (gfx1100 lane).

The native target verifier reaches the same raw-IQ tensors at rows 2-4 that
single-stream decode reaches at row 1. The rows == 1 local32 owner is a
one-row-per-block kernel, so those rows ran the strict per-row GEMV while
decode ran the local32 owner - the verifier was paying the slow population for
the fast one's tensors.

The sibling adds a ``ROWS`` template parameter to the same kernel bodies: one
block owns ``ROWS`` prompt rows, decodes each weight block once and applies it
to every row. Each row keeps the rows == 1 owner's per-(row, column) k
ownership, FMA order, shuffle tree and cross-wave sum, so each row's output is
**bit-identical** to the rows == 1 owner's output for that row. That is the
contract these tests pin, and it is what makes the sibling a pure
execution-geometry change rather than a new arithmetic owner.

Because the underlying owner is still approximate relative to the strict
per-row GEMV, the sibling carries the decode owner's gate obligations: an
opened dense-IQ session, a backend ``GGUF_IQ_DENSE_VERIFY_POLICY`` entry, the
per-slot strict pin, and the production-referenced accuracy gate on any route
admitting it. The strict per-row GEMV stays registered and remains the owner
for rows 1, rows 5 and up, and wherever the policy is absent.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _iq_dense_decode_dispatch,
)

FIXTURE = Path(__file__).parent / "fixtures/gguf_ud"

QUANT_TYPES = {
    "IQ4_XS": "gguf_iq4_xs",
    "IQ4_NL": "gguf_iq4_nl",
    "IQ3_S": "gguf_iq3_s",
    "IQ3_XXS": "gguf_iq3_xxs",
    "IQ2_S": "gguf_iq2_s",
    "IQ2_XS": "gguf_iq2_xs",
}


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def bf16(x):
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + 0x7fff + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_f32(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _parent_variant(rows):
    """The strict owner's registered variant at this row count.

    Above one row the strict per-row GEMV is reached through the prefill
    alias until the dense-IQ prefill policy (which starts at 8 rows) claims
    it, so verifier row counts arrive under ``prefill_bf16_bf16_out``.
    """
    return "gemv_bf16_bf16_out" if rows == 1 else "prefill_bf16_bf16_out"


def _dispatch(quant="gguf_iq4_xs", *, rows, out_features=17408):
    base = GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", quant, _parent_variant(rows)), "raw")
    return _iq_dense_decode_dispatch(base, rows=rows, out_features=out_features)


# ---------------------------------------------------------------- registration


@pytest.mark.parametrize("backend", ("hip_gfx1100", "hip_gfx1151"))
def test_verify_sibling_registered_on_both_hip_backends(backend):
    """The sibling is shared source lineage; aliasing registers it on gfx1151.

    Routing there is a separate per-backend policy decision (none yet).
    """
    load_backend_kernel_package(backend)
    for quant in QUANT_TYPES.values():
        key = KernelKey(backend, "linear", quant,
                        "local32_rows_gemv_bf16_bf16_out")
        assert is_registered(key), (backend, quant)


def test_strict_per_row_gemv_stays_registered_as_the_fallback():
    """The sibling must not displace the strict owner it is gated against."""
    load_backend_kernel_package("hip_gfx1100")
    for quant in QUANT_TYPES.values():
        key = KernelKey("hip_gfx1100", "linear", quant, "gemv_bf16_bf16_out")
        assert is_registered(key), quant


# --------------------------------------------------------------------- routing


def test_verify_route_declines_without_an_execution_owner():
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    with iq_mmq.iq_dense_mmq_session(False):
        assert iq_mmq.iq_dense_mmq_workspace() is None
        for rows in (2, 3, 4):
            out = _dispatch(rows=rows)
            assert out.key.variant == _parent_variant(rows), rows


def test_verify_route_declines_without_a_policy_entry(monkeypatch):
    import hipengine.kernels.hip_gfx1100 as be
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    monkeypatch.setattr(be, "GGUF_IQ_DENSE_VERIFY_POLICY", {})
    with iq_mmq.iq_dense_mmq_session(True):
        for rows in (2, 3, 4):
            out = _dispatch(rows=rows)
            assert out.key.variant == _parent_variant(rows), rows


def test_verify_route_declines_for_a_quant_without_a_policy_entry(monkeypatch):
    import hipengine.kernels.hip_gfx1100 as be
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    monkeypatch.setattr(be, "GGUF_IQ_DENSE_VERIFY_POLICY",
                        {"gguf_iq4_xs": {"variant": "local32_rows_gemv_bf16_bf16_out"}})
    with iq_mmq.iq_dense_mmq_session(True):
        assert _dispatch("gguf_iq4_xs", rows=2).key.variant == \
            "local32_rows_gemv_bf16_bf16_out"
        # Q3_K has no local32 owner at any row count.
        assert _dispatch("gguf_q3_k", rows=2).key.variant == \
            _parent_variant(2)


def test_verify_route_declines_for_an_unregistered_variant(monkeypatch):
    """A policy naming a variant that is not registered must not route."""
    import hipengine.kernels.hip_gfx1100 as be
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    monkeypatch.setattr(be, "GGUF_IQ_DENSE_VERIFY_POLICY",
                        {"gguf_iq4_xs": {"variant": "not_a_kernel"}})
    with iq_mmq.iq_dense_mmq_session(True):
        out = _dispatch(rows=2)
    assert out.key.variant == _parent_variant(2)


def test_verify_route_declines_for_decode_strict_slots():
    """A slot pinned to the strict decode owner keeps the strict verify owner.

    The pin exists because a specific artifact's local32 accumulation-order
    tail compounds past the gate ceiling; routing the verifier rows of that
    same tensor to the same accumulation order would reintroduce it.
    """
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    base = GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_iq4_xs",
                  _parent_variant(2)), "raw")
    with iq_mmq.iq_dense_mmq_session(True, decode_strict_slots={"pinned"}):
        pinned = _iq_dense_decode_dispatch(base, rows=2, out_features=17408,
                                           slot_path="pinned")
        free_slot = _iq_dense_decode_dispatch(base, rows=2, out_features=17408,
                                              slot_path="free")
    assert pinned.key.variant == _parent_variant(2)
    assert free_slot.key.variant == "local32_rows_gemv_bf16_bf16_out"


def test_verify_route_declines_when_n_is_not_a_multiple_of_8():
    """The local32 grid is N/8 blocks; an unaligned N keeps the strict GEMV."""
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    with iq_mmq.iq_dense_mmq_session(True):
        for n in (17405, 1):
            out = _dispatch(rows=2, out_features=n)
            assert out.key.variant == _parent_variant(2), n


def test_verify_route_declines_above_four_rows():
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    with iq_mmq.iq_dense_mmq_session(True):
        for rows in (5, 8, 512):
            out = _dispatch(rows=rows)
            assert out.key.variant == _parent_variant(rows), rows


def test_verify_route_requires_the_prefill_alias_parent():
    """Rows 2-4 must not be claimed through the decode parent variant.

    The verifier reaches the strict owner under ``prefill_bf16_bf16_out``;
    accepting the decode name here would route a caller that is not the
    verifier (an ad-hoc multi-row launch on the decode alias) into the
    sibling without the prefill policy's own row accounting.
    """
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    base = GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_iq4_xs",
                  "gemv_bf16_bf16_out"), "raw")
    with iq_mmq.iq_dense_mmq_session(True):
        out = _iq_dense_decode_dispatch(base, rows=2, out_features=17408)
    assert out.key.variant == "gemv_bf16_bf16_out"


def test_verify_route_does_not_engage_for_prefill_or_non_raw_dispatch():
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
    with iq_mmq.iq_dense_mmq_session(True):
        packed = GGUFLinearDispatch(
            KernelKey("hip_gfx1100", "linear", "gguf_iq4_xs", "gemv_bf16_bf16_out"),
            "pack8")
        assert _iq_dense_decode_dispatch(packed, rows=2,
                                         out_features=17408).abi == "pack8"
        other = GGUFLinearDispatch(
            KernelKey("hip_gfx1100", "linear", "gguf_iq4_xs", "wmma_bf16_bf16_out"),
            "raw")
        assert _iq_dense_decode_dispatch(other, rows=2,
                                         out_features=17408).key.variant == \
            "wmma_bf16_bf16_out"


# --------------------------------------------------------------- host contract


@pytest.mark.parametrize("kwargs", (
    {"rows": 1},
    {"rows": 5},
    {"rows": 8},
    {"in_features": 5000},
    {"out_features": 17405},
    {"output": "f32"},
    {"quant": "gguf_q3_k"},
    {"x_ptr": 0},
    {"out_ptr": 0},
))
def test_verify_launch_rejects_invalid_arguments_before_building(kwargs, monkeypatch):
    monkeypatch.setattr(gguf_iq_dense, "_default_library",
                        lambda: pytest.fail("built before validation"))
    with pytest.raises(ValueError):
        gguf_iq_dense.launch_local32_rows(
            kwargs.pop("x_ptr", 1), 2, kwargs.pop("out_ptr", 3),
            rows=kwargs.pop("rows", 2),
            in_features=kwargs.pop("in_features", 5120),
            out_features=kwargs.pop("out_features", 17408),
            **kwargs)


class _VerifyContextStub:
    """The attributes ``_iq_dense_mmq_verify_context`` reads off a session."""

    use_iq_dense_mmq = True
    backend = "hip_gfx1100"

    def _iq_dense_mmq_strict_slots(self):
        return ()

    def _iq_dense_decode_strict_slots(self):
        return ()

    # The real policy logic, so the tests exercise it rather than a
    # re-implementation.
    def _iq_dense_policy_present(self):
        return Qwen35GGUFResidentSession._iq_dense_policy_present(self)

    def _iq_dense_verify_binding_needed(self, rows):
        return Qwen35GGUFResidentSession._iq_dense_verify_binding_needed(self, rows)


def test_verify_context_binds_the_execution_owner_without_a_workspace():
    """The verifier's execution-owner binding must be allocation-free.

    The binding is entered inside stream capture for the native target graph,
    so it may not allocate a dense-IQ workspace. Only the strict-slot pins and
    the row-regime policies matter at verifier row counts anyway.
    """
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq

    assert iq_mmq.iq_dense_mmq_workspace() is None
    ctx = Qwen35GGUFResidentSession._iq_dense_mmq_verify_context(
        _VerifyContextStub(), rows=3)
    with ctx:
        assert iq_mmq.iq_dense_mmq_workspace() is not None
        assert not iq_mmq.iq_dense_mmq_has_workspace()
        out = _dispatch(rows=3)
    assert out.key.variant == "local32_rows_gemv_bf16_bf16_out"
    assert iq_mmq.iq_dense_mmq_workspace() is None


def test_verify_context_stays_closed_outside_the_declared_row_window():
    """Rows a verifier policy does not declare must keep their previous owner.

    Binding unconditionally would also open the dense-IQ prefill policy
    (min_rows 8) inside the verifier, which is a different owner with a
    different arithmetic and was measured slower than the strict GEMV at
    exactly 8 rows.
    """
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq

    stub = _VerifyContextStub()
    assert stub._iq_dense_verify_binding_needed(2)
    assert stub._iq_dense_verify_binding_needed(4)
    assert not stub._iq_dense_verify_binding_needed(1)
    assert not stub._iq_dense_verify_binding_needed(5)
    assert not stub._iq_dense_verify_binding_needed(8)
    with Qwen35GGUFResidentSession._iq_dense_mmq_verify_context(stub, rows=8):
        assert iq_mmq.iq_dense_mmq_workspace() is None


def test_verify_context_declines_when_the_backend_declares_no_policy(monkeypatch):
    import hipengine.kernels.hip_gfx1100 as be
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq

    for name in ("GGUF_IQ_DENSE_PREFILL_POLICY", "GGUF_IQ_DENSE_DECODE_POLICY",
                 "GGUF_IQ_DENSE_VERIFY_POLICY"):
        monkeypatch.setattr(be, name, {})
    with Qwen35GGUFResidentSession._iq_dense_mmq_verify_context(
            _VerifyContextStub(), rows=3):
        assert iq_mmq.iq_dense_mmq_workspace() is None
        assert _dispatch(rows=3).key.variant == _parent_variant(3)


def test_block_verifier_enters_the_dense_iq_execution_owner():
    """The eager block verifier must bind the session around its layer loop.

    Without the binding the verifier's raw-IQ projections keep the strict
    per-row GEMV while the AR route alongside them uses the local32 owner,
    which is exactly the verification-specific residual the gate measures.
    This is a structural assertion because exercising the loop needs a model.
    """
    import inspect

    src = inspect.getsource(Qwen35GGUFResidentSession.verify_target_block)
    assert "self._iq_dense_mmq_verify_context(rows=rows)," in src
    assert "wmma_prefill_session(block_wmma_prefill)," in src


def test_verify_wave_split_is_the_shared_rows_one_rule():
    """The sibling must pick the same split-K wave count as the decode owner.

    The bit-exactness claim is only true if the cross-wave reduction order is
    identical, so the split-K rule is one shared function rather than a
    copied constant in each launcher.
    """
    import inspect
    for fn in (gguf_iq_dense.launch_local32, gguf_iq_dense.launch_local32_rows):
        assert "_local32_waves(in_features, out_features)" in inspect.getsource(fn)
    assert gguf_iq_dense._local32_waves(5120, 17408) == 2
    assert gguf_iq_dense._local32_waves(17408, 5120) == 4
    assert gguf_iq_dense._local32_waves(256, 5120) == 1


# ------------------------------------------------------------------- numerics


def _load_fixture(quant_type):
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    entry = next((e for e in entries if e["type"] == quant_type), None)
    if entry is None:
        return None
    with np.load(FIXTURE / "real_rows.npz") as data:
        source = data[entry["key"] + "_raw"]
        k = data[entry["key"] + "_f32"].shape[1]
    return np.ascontiguousarray(source[np.arange(64) % len(source)]), k


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("quant_type", tuple(QUANT_TYPES))
@pytest.mark.parametrize("rows", (2, 3, 4))
def test_verify_rows_are_bit_exact_with_the_rows_one_owner(quant_type, rows):
    """Each row of the sibling must be bit-identical to the one-row owner.

    Not a tolerance: the sibling shares the rows == 1 kernel body and the
    rows == 1 split-K rule, so any difference is a real arithmetic change and
    the test is the RED contract for the execution-geometry claim.
    """
    from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                       free, host_array_ptr, malloc)

    loaded = _load_fixture(quant_type)
    if loaded is None:
        pytest.skip(f"no {quant_type} fixture row")
    raw, k = loaded
    n = len(raw)
    if k % 256:
        pytest.skip(f"{quant_type} fixture K={k} is not window-aligned")
    quant = QUANT_TYPES[quant_type]

    rng = np.random.default_rng(41)
    x = bf16(rng.normal(0, 0.1, (rows, k)))
    dense_lib = gguf_iq_dense.build_gguf_iq_dense()
    ref = np.zeros((rows, n), dtype=np.uint16)
    got = np.zeros((rows, n), dtype=np.uint16)
    bufs = []
    try:
        def dev(a):
            b = malloc(a.nbytes); bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes); return b
        x_b, w_b = dev(x), dev(raw)
        o_b = malloc(got.nbytes); bufs.append(o_b)
        r_b = malloc(ref.nbytes); bufs.append(r_b)
        for r in range(rows):
            gguf_iq_dense.launch_local32(
                x_b.ptr + r * k * 2, w_b.ptr, r_b.ptr + r * n * 2, 1, k, n,
                quant=quant, library=dense_lib)
        gguf_iq_dense.launch_local32_rows(
            x_b.ptr, w_b.ptr, o_b.ptr, rows, k, n, quant=quant,
            library=dense_lib)
        copy_device_to_host(host_array_ptr(ref), r_b, ref.nbytes)
        copy_device_to_host(host_array_ptr(got), o_b, got.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    assert np.isfinite(bf16_f32(got)).all()
    assert (got == ref).all(), (
        f"{quant_type} rows={rows} sibling is not bit-exact with the one-row "
        f"owner: {(got != ref).sum()} of {got.size} elements differ")


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("quant_type", tuple(QUANT_TYPES))
def test_verify_rows_agree_with_the_strict_gemv_on_real_rows(quant_type):
    """The sibling inherits the decode owner's accuracy class, not a new one."""
    from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                       free, host_array_ptr, malloc)

    loaded = _load_fixture(quant_type)
    if loaded is None:
        pytest.skip(f"no {quant_type} fixture row")
    raw, k = loaded
    n = len(raw)
    if k % 256:
        pytest.skip(f"{quant_type} fixture K={k} is not window-aligned")
    quant = QUANT_TYPES[quant_type]

    rows = 3
    x = bf16(np.random.default_rng(53).normal(0, 0.1, (rows, k)))
    dense_lib = gguf_iq_dense.build_gguf_iq_dense()
    ref = np.zeros((rows, n), dtype=np.uint16)
    got = np.zeros((rows, n), dtype=np.uint16)
    bufs = []
    try:
        def dev(a):
            b = malloc(a.nbytes); bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes); return b
        x_b, w_b = dev(x), dev(raw)
        o_b = malloc(got.nbytes); bufs.append(o_b)
        r_b = malloc(ref.nbytes); bufs.append(r_b)
        for r in range(rows):
            gguf_iq_dense.launch(
                x_b.ptr + r * k * 2, w_b.ptr, r_b.ptr + r * n * 2, 1, k, n,
                quant=quant, output="bf16", library=dense_lib)
        gguf_iq_dense.launch_local32_rows(
            x_b.ptr, w_b.ptr, o_b.ptr, rows, k, n, quant=quant,
            library=dense_lib)
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
    assert rel <= 5e-4, f"{quant_type} sibling diverged from the strict GEMV"
    assert float(np.corrcoef(a.ravel(), c.ravel())[0, 1]) >= 0.9999
