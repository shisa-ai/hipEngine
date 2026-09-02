"""B1 mechanism-A transfer switch: MTP serving target verify owner routing.

The July-2026 small-B decision (9cceedbcc) pinned the packed MTP serving
target verifier to per-row GEMV owners. The B1 build campaign retained the
transfer: the production execution profile serves verify rows>1 through the
same retained exact prefill band owners the prefill path uses, while strict
(and any profile fallback) keeps the GEMV verifier oracle unchanged. The env
remains an explicit override for bisection and diagnostics. These tests pin
the switch, its profile scoping, and the registry surface the rewrite
depends on; they are host-only.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from hipengine.runtime.gguf_linear import (
    MTP_SERVING_TARGET_WMMA_PREFILL_ENV,
    mtp_serving_target_use_wmma_prefill,
    mtp_serving_target_wmma_prefill_allows_rows,
)


def test_no_profile_context_keeps_gemv_owners() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MTP_SERVING_TARGET_WMMA_PREFILL_ENV, None)
        assert mtp_serving_target_use_wmma_prefill() is False
        assert mtp_serving_target_use_wmma_prefill(None) is False


def test_production_profile_serves_transferred_owners() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MTP_SERVING_TARGET_WMMA_PREFILL_ENV, None)
        assert (
            mtp_serving_target_use_wmma_prefill(
                "production", profile_fell_back_to_strict=False
            )
            is True
        )


def test_strict_profile_keeps_gemv_oracle() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MTP_SERVING_TARGET_WMMA_PREFILL_ENV, None)
        assert mtp_serving_target_use_wmma_prefill("strict") is False
        assert mtp_serving_target_use_wmma_prefill("legacy_exact") is False


def test_production_fallback_to_strict_keeps_gemv_owners() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MTP_SERVING_TARGET_WMMA_PREFILL_ENV, None)
        assert (
            mtp_serving_target_use_wmma_prefill(
                "production", profile_fell_back_to_strict=True
            )
            is False
        )


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_env_forces_transfer_on_any_profile(value: str) -> None:
    with mock.patch.dict(
        os.environ, {MTP_SERVING_TARGET_WMMA_PREFILL_ENV: value}
    ):
        assert mtp_serving_target_use_wmma_prefill() is True
        assert mtp_serving_target_use_wmma_prefill("strict") is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_env_forces_gemv_off_override(value: str) -> None:
    with mock.patch.dict(
        os.environ, {MTP_SERVING_TARGET_WMMA_PREFILL_ENV: value}
    ):
        assert (
            mtp_serving_target_use_wmma_prefill(
                "production", profile_fell_back_to_strict=False
            )
            is False
        )


def test_small_packed_rows_floor_keeps_gemv_owners() -> None:
    """R4 (width-1 K3) measured broken under the transferred owner.

    The 2026-09-03 current-head refresh measured production explicit C1 K3
    collapsing to 8.556 tok/s (0.743x AR) with draft acceptance 0.1523 and
    0/10 AR equality under the B1 transferred owner at the 4-row packed
    shape, while the per-row GEMV owners restore 18.168 tok/s (1.582x,
    10/10). The dispatch floor keeps small packed shapes on the GEMV owner.
    """
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MTP_SERVING_TARGET_WMMA_PREFILL_ENV, None)
        assert (
            mtp_serving_target_use_wmma_prefill(
                "production", profile_fell_back_to_strict=False, packed_rows=4
            )
            is False
        )
        assert mtp_serving_target_wmma_prefill_allows_rows(4) is False
        assert mtp_serving_target_wmma_prefill_allows_rows(2) is False


@pytest.mark.parametrize("rows", [8, 12, 16, 20, 24, 28, 32])
def test_measured_healthy_packed_rows_keep_transfer(rows: int) -> None:
    """R8+ shapes measured healthy (C2-C8, acceptance 0.78+, 10/10 equality)."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MTP_SERVING_TARGET_WMMA_PREFILL_ENV, None)
        assert mtp_serving_target_wmma_prefill_allows_rows(rows) is True
        assert (
            mtp_serving_target_use_wmma_prefill(
                "production", profile_fell_back_to_strict=False, packed_rows=rows
            )
            is True
        )


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_explicit_env_on_defeats_the_floor(value: str) -> None:
    """The registered opt-in remains the escape hatch at every shape."""
    with mock.patch.dict(
        os.environ, {MTP_SERVING_TARGET_WMMA_PREFILL_ENV: value}
    ):
        assert mtp_serving_target_wmma_prefill_allows_rows(4) is True
        assert (
            mtp_serving_target_use_wmma_prefill(
                "production", profile_fell_back_to_strict=False, packed_rows=4
            )
            is True
        )


def test_runner_clamps_both_verify_entry_points() -> None:
    """Packed and single-block verifiers floor the transfer by packed rows."""

    import inspect

    import hipengine.runtime.qwen35_gguf_runner as runner

    source = inspect.getsource(runner)
    assert source.count("mtp_serving_target_wmma_prefill_allows_rows") >= 2


class _ProfileLike:
    value = "production"


def test_profile_enum_objects_unwrap() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MTP_SERVING_TARGET_WMMA_PREFILL_ENV, None)
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
