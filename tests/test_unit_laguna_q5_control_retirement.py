from __future__ import annotations

import inspect

import pytest

from hipengine.runtime.laguna_gguf_runner import (
    LagunaGGUFResidentSession,
    resolve_laguna_q5_wave32x2_variants,
)


@pytest.mark.parametrize(
    "name", ("use_q5_wave32x2_output", "use_q5_wave32x2_query_gate")
)
def test_legacy_q5_constructor_opt_out_is_retired(name: str) -> None:
    signature = inspect.signature(LagunaGGUFResidentSession.__init__)
    assert name not in signature.parameters
    with pytest.raises(TypeError):
        signature.bind(None, **{name: False})


def test_q5_resolver_keeps_backend_defaults_and_independent_exact_controls() -> None:
    fixed = (
        "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
        "wave32x2_fixed_meta_gemv_decode_bf16_f32_out",
    )
    assert resolve_laguna_q5_wave32x2_variants("hip_gfx1100") == fixed
    assert resolve_laguna_q5_wave32x2_variants("hip_gfx1151") == (None, None)
    assert resolve_laguna_q5_wave32x2_variants(
        "hip_gfx1100", output=False
    ) == (None, fixed[1])
    assert resolve_laguna_q5_wave32x2_variants(
        "hip_gfx1100", query_gate=False
    ) == (fixed[0], None)
    assert resolve_laguna_q5_wave32x2_variants(
        "hip_gfx1100", fixed_meta_output=False, fixed_meta_query_gate=False
    ) == (
        "wave32x2_gemv_decode_bf16_bf16_out",
        "wave32x2_gemv_decode_bf16_f32_out",
    )
