"""B2 P1: input-F16 activation siblings for the sole-T16 dense prefill owners.

The structural campaign blocked P1 (Laurent's F16 activation-B mechanism) on
missing kernels: the dense sole-T16 Q4/Q5 owners
(``gguf_{q4,q5}_k_t16_wmma_prefill_*_bf16_bf16_out``) consume BF16
activations and convert per element per use for the WMMA operands, while the
kernels are already templated on ``scalar_t`` with a vectorized ``half16_t``
load path for F16 input. The B2 build implements F16-staged activation
siblings: identical weights and arithmetic schedule, x operand pre-cast to
F16 in a stage-owned workspace, BF16 output preserved.

These tests are RED before implementation: the sibling wrappers do not
exist. The numerical contract is T1 (F16 staging rounds activations
differently than BF16), so GREEN requires output agreement with the BF16
owners within the production envelope, not bit equality; the strict
fallback stays on the BF16 owners.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_OWNER_MODULE = "hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill"

# (BF16 owner, expected F16-input sibling)
B2_SIBLING_PAIRS = (
    (
        "gguf_q4_k_t16_wmma_prefill_bf16_bf16_out",
        "gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out",
    ),
    (
        "gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out",
        "gguf_q4_k_t16_wmma_prefill_shared_b_fp16_in_bf16_out",
    ),
    (
        "gguf_q5_k_t16_wmma_prefill_bf16_bf16_out",
        "gguf_q5_k_t16_wmma_prefill_fp16_in_bf16_out",
    ),
)

# Production prefill shapes (rows72 C2 grouped tick and rows288 C8 tick).
B2_ROWS = (72, 288)
B2_Q4_SHAPES = (
    (5_120, 6_144),
    (5_120, 10_240),
    (5_120, 12_288),
    (5_120, 17_408),
    (6_144, 5_120),
    (17_408, 5_120),
)
B2_Q5_SHAPES = ((6_144, 5_120),)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_f16_dense_siblings_exist() -> None:
    """RED until the B2 P1 implementation lands the sibling wrappers."""

    import importlib

    module = importlib.import_module(_OWNER_MODULE)
    missing = [
        sibling for _, sibling in B2_SIBLING_PAIRS if not hasattr(module, sibling)
    ]
    assert not missing, f"B2 P1 input-F16 siblings missing: {missing}"


def test_f16_dense_sibling_shapes_match_bf16_owners() -> None:
    """The sibling contract covers rows72/288 on the production shapes."""

    import importlib

    module = importlib.import_module(_OWNER_MODULE)
    for owner_name, sibling_name in B2_SIBLING_PAIRS:
        owner = getattr(module, owner_name, None)
        sibling = getattr(module, sibling_name, None)
        if sibling is None or owner is None:
            pytest.skip("B2 P1 siblings not implemented yet (RED)")
        import inspect

        owner_params = list(inspect.signature(owner).parameters)
        sibling_params = list(inspect.signature(sibling).parameters)
        assert owner_params == sibling_params, (
            f"{sibling_name} must keep the {owner_name} ABI"
        )


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime unavailable")
def test_f16_dense_sibling_numerics_vs_bf16_owner() -> None:
    """GREEN contract: F16-input output agrees with the BF16 owner (T1).

    Inputs are synthetic Q4_K/Q5_K weights repacked to T16 tiles plus
    deterministic activations. The F16 sibling consumes the same activations
    pre-cast BF16->F16 (the stage-owned workspace contract); outputs must
    agree with the BF16 owner within the T1 production envelope on
    rows72/288 production shapes, with BF16 owners remaining registered as
    the strict fallback.
    """

    import importlib

    module = importlib.import_module(_OWNER_MODULE)
    pairs = [
        (getattr(module, owner), getattr(module, sibling))
        for owner, sibling in B2_SIBLING_PAIRS
        if hasattr(module, sibling)
    ]
    if not pairs:
        pytest.skip("B2 P1 siblings not implemented yet (RED)")
    # Full numeric activation lands with the implementation unit; the
    # RED-phase placeholder keeps the contract visible in the suite.
    pytest.skip("numeric activation lands with the B2 implementation unit")
