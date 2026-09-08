"""Dense raw-IQ integer-MMQ prefill route (UD item 3).

The selected/MoE kernel reads weights as
``qweight + expert*expert_bytes + out_row*weight_row_bytes``, so at expert 0 it
consumes the dense raw GGUF layout unchanged. These tests pin that the dense
route is registered on both HIP backends, stays inert without a bound
workspace, is admitted only inside its declared policy, and produces the same
answer as the strict per-row GEMV owner within the Q8_1 activation tolerance.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.backends import backend_package_capability, load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
from hipengine.kernels.registry import KernelKey, is_registered

_DENSE_VARIANT = iq_mmq._DENSE_VARIANT


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


# ---------------------------------------------------------------- registration


@pytest.mark.parametrize("backend", ("hip_gfx1100", "hip_gfx1151"))
@pytest.mark.parametrize("quant", ("gguf_iq4_xs", "gguf_iq3_xxs", "gguf_iq4_nl"))
def test_dense_route_is_registered_on_both_hip_backends(backend, quant):
    load_backend_kernel_package(backend)
    assert is_registered(KernelKey(backend, "linear", quant, _DENSE_VARIANT))


@pytest.mark.parametrize("quant", ("gguf_iq4_xs", "gguf_iq3_xxs"))
def test_selected_moe_keys_stay_gfx1100_only(quant):
    """Adding the dense route must not alias the selected/MoE keys to gfx1151."""
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        load_backend_kernel_package(backend)
    assert is_registered(KernelKey("hip_gfx1100", "moe_linear", quant, iq_mmq._VARIANT))
    assert not is_registered(
        KernelKey("hip_gfx1151", "moe_linear", quant, iq_mmq._VARIANT)
    )


def test_gfx1151_declares_the_policy_and_gfx1100_does_not_yet():
    """gfx1151 is this unit's target; gfx1100 is a separate follow-up."""
    assert isinstance(
        backend_package_capability("hip_gfx1151", "GGUF_IQ_DENSE_MMQ_PREFILL_POLICY", None),
        dict,
    )
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_IQ_DENSE_MMQ_PREFILL_POLICY", None
    ) is None


# -------------------------------------------------------------------- dispatch


def _dispatch(quant, *, rows, in_features, out_features, variant="prefill_bf16_bf16_out"):
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch, _iq_dense_mmq_prefill_dispatch)
    load_backend_kernel_package("hip_gfx1151")
    base = GGUFLinearDispatch(KernelKey("hip_gfx1151", "linear", quant, variant), "raw")
    return _iq_dense_mmq_prefill_dispatch(
        base, rows=rows, in_features=in_features, out_features=out_features)


def test_route_is_inert_without_a_bound_workspace():
    out = _dispatch("gguf_iq4_xs", rows=512, in_features=5120, out_features=17408)
    assert out.key.variant == "prefill_bf16_bf16_out"


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (dict(rows=4, in_features=5120, out_features=17408), "below min_rows"),
        (dict(rows=512, in_features=5121, out_features=17408), "K not 256-aligned"),
        (dict(rows=512, in_features=5120, out_features=17400), "N not 128-aligned"),
        (dict(rows=512, in_features=5120, out_features=17408,
              variant="prefill_bf16_f32_out"), "f32 output has no dense MMQ owner"),
        (dict(rows=1, in_features=5120, out_features=17408), "decode stays on the GEMV"),
    ],
)
def test_route_declines_outside_its_policy(kwargs, reason):
    with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=1 << 20, workspace_nbytes=1 << 24):
        out = _dispatch("gguf_iq4_xs", **kwargs)
    assert out.key.variant == kwargs.get("variant", "prefill_bf16_bf16_out"), reason


@pytest.mark.parametrize("rows", (8, 16, 32, 512))
def test_route_is_selected_inside_its_policy(rows):
    with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=1 << 20, workspace_nbytes=1 << 24):
        out = _dispatch("gguf_iq4_xs", rows=rows, in_features=5120, out_features=17408)
    assert out.key.variant == _DENSE_VARIANT
    assert out.abi == "raw"


@pytest.mark.parametrize("rows", (1, 2, 4))
def test_route_declines_below_the_measured_crossover(rows):
    """Below 8 rows the GEMV still wins on at least one real shape.

    Swept in scripts/gguf_iq_dense_mmq_crossover.py: worst case over the six
    distinct IQ4_XS dense shapes is 0.59x at 2 rows and 0.68x at 4, against
    1.58x at 8. The policy floor must not drop under that.
    """
    with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=1 << 20, workspace_nbytes=1 << 24):
        out = _dispatch("gguf_iq4_xs", rows=rows, in_features=5120, out_features=17408)
    assert out.key.variant == "prefill_bf16_bf16_out"


def test_policy_floor_matches_the_measured_crossover():
    policy = backend_package_capability(
        "hip_gfx1151", "GGUF_IQ_DENSE_MMQ_PREFILL_POLICY", {})
    for quant, entry in policy.items():
        assert entry["min_rows"] == 8, (
            f"{quant} min_rows drifted from the swept crossover")


def test_unsupported_quants_keep_the_strict_owner():
    """Quants whose 32-element groups need finer scales must not be routed.

    The MMQ expansion returns one float scale plus 32 signed int8 per K32
    group. Q3_K, IQ2_S and IQ2_XS carry a scale per 16 elements, so under a
    per-32 scale their residual integers reach +/-128 and +/-1333 respectively
    and do not fit int8. IQ3_S does fit (+/-15) but needs its 2 KB grid table
    ported into this translation unit, so it is not wired yet.
    """
    with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=1 << 20, workspace_nbytes=1 << 24):
        for quant in ("gguf_iq2_xs", "gguf_iq3_s", "gguf_q3_k", "gguf_iq2_s"):
            out = _dispatch(quant, rows=512, in_features=5120, out_features=17408)
            assert out.key.variant == "prefill_bf16_bf16_out", quant


def test_every_policy_quant_has_a_registered_owner():
    """A policy entry without a registered kernel would silently do nothing."""
    load_backend_kernel_package("hip_gfx1151")
    policy = backend_package_capability(
        "hip_gfx1151", "GGUF_IQ_DENSE_MMQ_PREFILL_POLICY", {})
    assert set(policy) == {"gguf_iq4_xs", "gguf_iq3_xxs"}
    for quant, entry in policy.items():
        assert is_registered(
            KernelKey("hip_gfx1151", "linear", quant, entry["variant"]))


def test_iq4_nl_has_kernel_support_but_is_held_out_of_the_default():
    """IQ4_NL is implemented and correct, but not routed by default.

    Enabling it measured prefill 127.2 -> 155.2 tok/s and moved the teacher
    gate mean 0.0010796 -> 0.0013457, p95 +39%, top-1 162/162 -> 161/162. The
    median per-position delta is 0.000000 while a few positions diverge
    100-300x, so a small number of sensitive tensors have to be identified
    before it can be the default.
    """
    load_backend_kernel_package("hip_gfx1151")
    assert is_registered(
        KernelKey("hip_gfx1151", "linear", "gguf_iq4_nl", _DENSE_VARIANT))
    policy = backend_package_capability(
        "hip_gfx1151", "GGUF_IQ_DENSE_MMQ_PREFILL_POLICY", {})
    assert "gguf_iq4_nl" not in policy
    with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=1 << 20, workspace_nbytes=1 << 24):
        out = _dispatch("gguf_iq4_nl", rows=512, in_features=5120, out_features=17408)
    assert out.key.variant == "prefill_bf16_bf16_out"


# ------------------------------------------------------------------- workspace


def test_workspace_sizing_covers_activations_plus_metadata():
    rows, hidden = 512, 17_408
    total = iq_mmq.iq_dense_mmq_nbytes(rows, hidden)
    assert total > iq_mmq.iq_dense_mmq_activation_nbytes(rows, hidden)
    assert total - iq_mmq.iq_dense_mmq_activation_nbytes(rows, hidden) == iq_mmq._METADATA_NBYTES


def test_session_rejects_a_workspace_that_cannot_hold_metadata():
    with pytest.raises(ValueError):
        with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=1 << 20, workspace_nbytes=8):
            pass
    with pytest.raises(ValueError):
        with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=0, workspace_nbytes=1 << 24):
            pass


def test_launch_without_a_session_raises_rather_than_reading_stale_state():
    with pytest.raises(RuntimeError, match="workspace session"):
        iq_mmq.gguf_iq4_xs_dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out(
            1, 2, 3, 512, 5120, 17408)
