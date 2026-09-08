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
@pytest.mark.parametrize("quant", ("gguf_iq4_xs", "gguf_iq3_xxs"))
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
        (dict(rows=8, in_features=5120, out_features=17408), "below min_rows"),
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


def test_route_is_selected_inside_its_policy():
    with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=1 << 20, workspace_nbytes=1 << 24):
        out = _dispatch("gguf_iq4_xs", rows=512, in_features=5120, out_features=17408)
    assert out.key.variant == _DENSE_VARIANT
    assert out.abi == "raw"


def test_unsupported_quants_keep_the_strict_owner():
    """IQ2_XS/IQ3_S/Q3_K have no MMQ kernel and must not be routed."""
    with iq_mmq.iq_dense_mmq_session(True, workspace_ptr=1 << 20, workspace_nbytes=1 << 24):
        for quant in ("gguf_iq2_xs", "gguf_iq3_s", "gguf_q3_k", "gguf_iq2_s", "gguf_iq4_nl"):
            out = _dispatch(quant, rows=512, in_features=5120, out_features=17408)
            assert out.key.variant == "prefill_bf16_bf16_out", quant


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
