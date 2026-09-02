"""B1 mechanism-A transfer switch: MTP serving target verify owner routing.

The July-2026 small-B decision (9cceedbcc) pinned the packed MTP serving
target verifier to per-row GEMV owners. The B1 build campaign re-opens that
routing behind a default-off env so the verifier can use the same retained
exact prefill band owners the prefill path uses, with the current owners as
the strict fallback. These tests pin the switch and the registry surface the
rewrite depends on; they are host-only and skip nothing.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from hipengine.runtime.gguf_linear import (
    MTP_SERVING_TARGET_WMMA_PREFILL_ENV,
    mtp_serving_target_use_wmma_prefill,
)


def test_transfer_defaults_off_preserving_strict_fallback() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MTP_SERVING_TARGET_WMMA_PREFILL_ENV, None)
        assert mtp_serving_target_use_wmma_prefill() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_transfer_env_enables(value: str) -> None:
    with mock.patch.dict(
        os.environ, {MTP_SERVING_TARGET_WMMA_PREFILL_ENV: value}
    ):
        assert mtp_serving_target_use_wmma_prefill() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_transfer_env_negative_values_keep_fallback(value: str) -> None:
    with mock.patch.dict(
        os.environ, {MTP_SERVING_TARGET_WMMA_PREFILL_ENV: value}
    ):
        assert mtp_serving_target_use_wmma_prefill() is False


def test_serving_modules_consume_the_switch() -> None:
    """Both MTP serving entry points route verify jobs through the switch."""

    import inspect

    import hipengine.generation.qwen35_gguf as serving
    import hipengine.generation.qwen35_gguf_mtp2 as mtp2

    serving_source = inspect.getsource(serving)
    assert "_mtp_serving_target_use_wmma_prefill()" in serving_source
    assert "_MTP_SERVING_TARGET_USE_WMMA_PREFILL" not in serving_source
    mtp2_source = inspect.getsource(mtp2)
    assert '"use_wmma_prefill": mtp_serving_target_use_wmma_prefill(),' in (
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
