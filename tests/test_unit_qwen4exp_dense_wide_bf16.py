"""CPU-only registration/export contract for the opt-in BF16 dense-wide probe."""

from __future__ import annotations

import re

from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide


SYMBOL = "hipengine_gguf_q8_0_dense_wide256_bf16in_f32_f32_out"
VARIANT = "dense_wide256_bf16in_f32_f32_out"


def test_bf16_probe_has_a_registered_wrapper_and_export() -> None:
    assert VARIANT in wide._VARIANTS
    assert wide._VARIANTS[VARIANT] == SYMBOL
    assert callable(getattr(wide, "gguf_q8_0_dense_wide256_bf16in_f32_f32_out"))

    source = wide._SOURCE.read_text()
    assert re.search(
        rf"HIPENGINE_DENSE_WIDE_EXPORT_BF16\(\s*{SYMBOL}", source
    )
