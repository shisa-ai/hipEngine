"""E6c rows-1 FFN-down residual fold: admission, capability, siblings.

RED first: none of these resolve until the Q5 sibling, the residual
max-rows row, and the unqualified-lane admission row land.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import so the registry has the real kernels to resolve against.
import hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv  # noqa: F401
from hipengine.kernels.backends import (
    backend_package_capability,
    load_backend_kernel_package,
)
from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY
from hipengine.kernels.registry import is_registered
from hipengine.runtime.gguf_linear import (
    _resolve_registered_linear_residual,
    resolve_gguf_linear_dispatch,
)
from hipengine.runtime.qwen35_gguf_runner import _gguf_policy_identity

BACKEND = "hip_gfx1151"
# The runtime policy identity for the resident Qwen3.8-27B UD-Q4_K_M
# artifact: H5120 dense geometry, plain-lane GGUF stamp, unqualified-manifest
# preset key (this file's manifest is not a pinned UD preset).
UD_IDENTITY = (
    QWEN35_DENSE_H5120_GEOMETRY,
    "MOSTLY_Q4_K_M",
    "gguf-unqualified-manifest",
)
DOWN_DECODE_SHAPE = (1, 17_408, 5_120)


def _weight(quant_key: str, layout: str) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(quant_key=quant_key, layout=layout),
        backend=BACKEND,
    )


@pytest.fixture(autouse=True)
def _backend_package() -> object:
    load_backend_kernel_package(BACKEND)
    yield


def test_q5_down_rows1_resolves_exact_residual_sibling() -> None:
    dispatch = resolve_gguf_linear_dispatch(
        _weight("gguf_q5_k_t16_v1", "gguf_q5_k_t16_v1"), rows=1
    )
    assert dispatch.abi == "t16"
    assert dispatch.key.variant == "t16_gemv_decode_bf16_bf16_out"
    resolved = _resolve_registered_linear_residual(dispatch.key, rows=1)
    assert resolved is not None, "Q5 down residual sibling must resolve"
    fused_key, _fn = resolved
    assert fused_key.layer == "linear+residual"
    assert fused_key.variant == "t16_gemv_decode_bf16_residual_bf16_out"
    assert is_registered(fused_key)


def test_q4_down_rows1_residual_sibling_stays_registered() -> None:
    dispatch = resolve_gguf_linear_dispatch(
        _weight("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1"), rows=1
    )
    assert dispatch.key.variant == "dense_single_local32_bf16_bf16_out"
    resolved = _resolve_registered_linear_residual(dispatch.key, rows=1)
    assert resolved is not None
    fused_key, _fn = resolved
    assert is_registered(fused_key)


def test_residual_max_rows_declares_q5_down_decode() -> None:
    limits = backend_package_capability(
        BACKEND, "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT", {}
    )
    assert limits.get("gguf_q5_k_t16_v1") == 1


def test_runtime_identity_matches_unqualified_lane_row() -> None:
    weights = SimpleNamespace(
        geometry=QWEN35_DENSE_H5120_GEOMETRY,
        config=SimpleNamespace(is_moe=False),
        file_type_name="MOSTLY_Q4_K_M",
        artifact_preset_key="gguf-unqualified-manifest",
    )
    identity = _gguf_policy_identity(weights)
    policies = backend_package_capability(
        BACKEND, "GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES", {}
    )
    assert identity == UD_IDENTITY
    assert policies.get(identity, {}).get(DOWN_DECODE_SHAPE) is True


def test_certificate_bound_ud_preset_identity_is_admitted() -> None:
    # The resident artifact's runtime identity as observed under the
    # decode-graph driver: the load binds the certificate preset
    # gguf_ud_q4_k_m, which extends the policy identity to a 3-tuple.
    weights = SimpleNamespace(
        geometry=QWEN35_DENSE_H5120_GEOMETRY,
        config=SimpleNamespace(is_moe=False),
        file_type_name="MOSTLY_Q4_K_M",
        artifact_preset_key="gguf_ud_q4_k_m",
    )
    identity = _gguf_policy_identity(weights)
    policies = backend_package_capability(
        BACKEND, "GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES", {}
    )
    assert identity == (
        QWEN35_DENSE_H5120_GEOMETRY,
        "MOSTLY_Q4_K_M",
        "gguf_ud_q4_k_m",
    )
    assert policies.get(identity, {}).get(DOWN_DECODE_SHAPE) is True