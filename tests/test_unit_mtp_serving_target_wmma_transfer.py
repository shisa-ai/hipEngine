"""MTP serving target verify owner routing follows the execution profile.

Production uses the prefill band owners; strict, absent profile context and
strict fallback keep the GEMV verifier. Stale values of the retired environment
override cannot change either choice. Tests are host-only.
"""

from __future__ import annotations

import pytest

from hipengine.runtime.gguf_linear import (
    mtp_serving_target_use_wmma_prefill,
)


_RETIRED_ENV = "HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL"


def test_no_profile_context_keeps_gemv_owners() -> None:
    assert mtp_serving_target_use_wmma_prefill() is False
    assert mtp_serving_target_use_wmma_prefill(None) is False


def test_production_profile_serves_transferred_owners() -> None:
    assert mtp_serving_target_use_wmma_prefill(
        "production", profile_fell_back_to_strict=False,
    ) is True


def test_strict_profile_keeps_gemv_oracle() -> None:
    assert mtp_serving_target_use_wmma_prefill("strict") is False
    assert mtp_serving_target_use_wmma_prefill("legacy_exact") is False


def test_production_fallback_to_strict_keeps_gemv_owners() -> None:
    assert mtp_serving_target_use_wmma_prefill(
        "production", profile_fell_back_to_strict=True,
    ) is False


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "1", "true", "yes", "on", "invalid"])
def test_retired_env_cannot_change_profile_selection(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv(_RETIRED_ENV, raising=False)
    else:
        monkeypatch.setenv(_RETIRED_ENV, value)
    assert mtp_serving_target_use_wmma_prefill("production") is True
    assert mtp_serving_target_use_wmma_prefill(_ProfileLike()) is True
    assert mtp_serving_target_use_wmma_prefill("strict") is False
    assert mtp_serving_target_use_wmma_prefill("legacy_exact") is False
    assert mtp_serving_target_use_wmma_prefill() is False
    assert mtp_serving_target_use_wmma_prefill(None) is False
    assert mtp_serving_target_use_wmma_prefill(
        "production", profile_fell_back_to_strict=True,
    ) is False


class _ProfileLike:
    value = "production"


def test_profile_enum_objects_unwrap() -> None:
    assert mtp_serving_target_use_wmma_prefill(_ProfileLike()) is True


def test_serving_modules_consume_the_switch() -> None:
    """Both MTP serving entry points route verify jobs through the switch."""

    import inspect

    import hipengine.generation.qwen35_gguf as serving
    import hipengine.generation.qwen35_gguf_mtp2 as mtp2

    serving_source = inspect.getsource(serving)
    assert "_mtp_serving_target_wmma_for(self)" in serving_source
    assert "_MTP_SERVING_TARGET_USE_WMMA_PREFILL" not in serving_source
    mtp2_source = inspect.getsource(mtp2)
    assert '"use_wmma_prefill": mtp_serving_target_use_wmma_prefill(' in (
        mtp2_source
    )


def test_prefill_band_variants_are_registered_for_verifier_quants() -> None:
    """The rewrite target variants must resolve on the gfx1151 backend."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey, is_registered

    register_gfx1151_kernels()
    for quant in (
        "gguf_q4_k_t16_v1",
        "gguf_q5_k_t16_v1",
        "gguf_q6_k_t16_v1",
        "gguf_q6_k_t16_qmicro_planar_v1",
    ):
        key = KernelKey(
            "hip_gfx1151",
            "linear",
            quant,
            "t16_wmma_prefill_bf16_bf16_out",
        )
        assert is_registered(key), f"missing prefill band variant for {quant}"
