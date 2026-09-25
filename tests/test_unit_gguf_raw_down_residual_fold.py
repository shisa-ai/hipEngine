"""E6c-2 raw-IQ FFN-down residual fold: launcher, siblings, decode owner.

RED first: none of these resolve until the raw residual launcher lands in
`_LAUNCH_RESIDUAL_ABI`, the per-quant `linear+residual` siblings register,
the rows-1 residual path replays the production decode-owner decision, and
the raw quants declare a rows-1 residual limit.

Production ground truth (re-derived 2026-09-25, session e6c-2): the resident
artifact's 31 raw-layout ffn_down slots resolve contract variant
`gemv_bf16_bf16_out` under `abi='raw'`; with the dense-IQ session bound the
launch chain redirects unpinned iq4_xs/iq4_nl slots to the local32 owner,
while the artifact's pinned down slots (iq3_s) and the policy-less q3_k slot
keep the strict owner. The composite must pick the same parent, or the fold
would change arithmetic on the redirected layers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import so the registry has the real kernels to resolve against.
import hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense  # noqa: F401
from hipengine.kernels.backends import (
    backend_package_capability,
    load_backend_kernel_package,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_source_mmq_prefill import (
    iq_dense_mmq_session,
)
from hipengine.kernels.registry import is_registered
from hipengine.kernels.registry import KernelKey
from hipengine.runtime.gguf_linear import (
    _LAUNCH_RESIDUAL_ABI,
    _linear_residual_variant,
    _resolve_registered_linear_residual,
    resolve_gguf_linear_dispatch,
)

BACKEND = "hip_gfx1151"
RAW_DOWN_QUANTS = ("gguf_iq4_xs", "gguf_iq4_nl", "gguf_iq3_s", "gguf_q3_k")
# The four raw-layout down slots the resident artifact's pin table keeps on
# the strict owner (GGUF_IQ_DENSE_DECODE_STRICT_SLOTS).
PINNED_DOWN_SLOTS = ("layers.14.ffn_down", "layers.15.ffn_down",
                     "layers.17.ffn_down")


def _weight(quant_key: str, slot_path: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(
            quant_key=quant_key,
            layout="raw_gguf",
            slot_path=slot_path,
        ),
        backend=BACKEND,
    )


def _dispatch(weight: SimpleNamespace, rows: int = 1):
    return resolve_gguf_linear_dispatch(weight, rows=rows)


@pytest.fixture(autouse=True)
def _backend_package() -> object:
    load_backend_kernel_package(BACKEND)
    yield


def test_raw_residual_abi_has_launcher() -> None:
    """The rows-1 residual path fails closed today: no raw ABI entry."""

    assert "raw" in _LAUNCH_RESIDUAL_ABI


def test_raw_down_contract_is_strict_parent() -> None:
    for quant in RAW_DOWN_QUANTS:
        dispatch = _dispatch(_weight(quant))
        assert dispatch.abi == "raw", quant
        assert dispatch.key.variant == "gemv_bf16_bf16_out", quant


def test_strict_residual_sibling_registered_for_raw_down_quants() -> None:
    """Strict parent composite resolves for every raw down quant (pins and
    session-closed production both run strict for some of them)."""

    for quant in RAW_DOWN_QUANTS:
        dispatch = _dispatch(_weight(quant))
        resolved = _resolve_registered_linear_residual(dispatch.key, rows=1)
        assert resolved is not None, f"{quant} strict residual must resolve"
        fused_key, _fn = resolved
        assert fused_key.layer == "linear+residual"
        assert fused_key.variant == "gemv_bf16_residual_bf16_out"
        assert is_registered(fused_key), quant


def test_local32_residual_sibling_registered_for_redirected_quants() -> None:
    """iq4_xs / iq4_nl unpinned production runs the local32 owner, so its
    composite sibling must exist for the fold to stay exact there."""

    for quant in ("gguf_iq4_xs", "gguf_iq4_nl"):
        key = KernelKey(BACKEND, "linear", quant,
                        "local32_gemv_bf16_bf16_out")
        assert is_registered(key), quant
        resolved = _resolve_registered_linear_residual(key, rows=1)
        assert resolved is not None, f"{quant} local32 residual must resolve"
        fused_key, _fn = resolved
        assert fused_key.layer == "linear+residual"
        assert fused_key.variant == "local32_gemv_bf16_residual_bf16_out"
        assert is_registered(fused_key), quant


def test_linear_residual_variant_maps_raw_parents() -> None:
    assert _linear_residual_variant("gemv_bf16_bf16_out") == (
        "gemv_bf16_residual_bf16_out")
    assert _linear_residual_variant("local32_gemv_bf16_bf16_out") == (
        "local32_gemv_bf16_residual_bf16_out")


def test_raw_quants_declare_rows1_residual_limit() -> None:
    limits = backend_package_capability(
        BACKEND, "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT", {}
    )
    for quant in RAW_DOWN_QUANTS:
        assert limits.get(quant) == 1, quant


def test_residual_path_replays_session_redirect() -> None:
    """Session-bound, unpinned: the residual resolution must follow the
    production local32 owner, not the contract strict key."""

    from hipengine.runtime.gguf_linear import _residual_rows1_decode_dispatch

    weight = _weight("gguf_iq4_xs", slot_path="layers.0.ffn_down")
    contract = _dispatch(weight)

    # Session closed: production runs strict -> dispatch unchanged.
    closed = _residual_rows1_decode_dispatch(contract, weight, out_features=5120)
    assert closed.key.variant == "gemv_bf16_bf16_out"

    # Session bound, slot unpinned: production runs local32 -> redirect.
    with iq_dense_mmq_session(True):
        opened = _residual_rows1_decode_dispatch(contract, weight,
                                                 out_features=5120)
        assert opened.key.variant == "local32_gemv_bf16_bf16_out"

    # Session bound, slot pinned (artifact admission): strict stays.
    pinned = _weight("gguf_iq3_s", slot_path=PINNED_DOWN_SLOTS[0])
    pinned_contract = _dispatch(pinned)
    with iq_dense_mmq_session(True,
                              decode_strict_slots=PINNED_DOWN_SLOTS):
        kept = _residual_rows1_decode_dispatch(pinned_contract, pinned,
                                               out_features=5120)
        assert kept.key.variant == "gemv_bf16_bf16_out"

    # q3_k has no decode-policy entry: strict stays even unpinned.
    q3 = _weight("gguf_q3_k", slot_path="layers.0.ffn_down")
    q3_contract = _dispatch(q3)
    with iq_dense_mmq_session(True):
        kept = _residual_rows1_decode_dispatch(q3_contract, q3,
                                               out_features=5120)
        assert kept.key.variant == "gemv_bf16_bf16_out"


def test_non_raw_dispatch_is_not_touched() -> None:
    from hipengine.runtime.gguf_linear import _residual_rows1_decode_dispatch

    weight = _weight("gguf_q5_k_t16_v1", slot_path="layers.0.ffn_down")
    weight.spec.layout = "gguf_q5_k_t16_v1"
    dispatch = resolve_gguf_linear_dispatch(weight, rows=1)
    with iq_dense_mmq_session(True):
        out = _residual_rows1_decode_dispatch(dispatch, weight,
                                              out_features=5120)
    assert out.key is dispatch.key