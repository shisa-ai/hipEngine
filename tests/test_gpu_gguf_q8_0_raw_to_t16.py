"""SH6-P1 byte-oracle tests for GPU raw-Q8_0 to Q8T16 repacking."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_raw_to_t16 import (
    build_gguf_q8_0_raw_to_t16,
    gguf_q8_0_raw_pair_to_t16,
    gguf_q8_0_raw_pair_to_t16_nbytes,
    plan_gguf_q8_0_raw_to_t16_build,
    register_gguf_q8_0_raw_to_t16_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()
_PRODUCTION_IN_FEATURES = 2048
_PRODUCTION_OUT_FEATURES_A = 8192
_PRODUCTION_OUT_FEATURES_B = 4096
_PRODUCTION_PAIR_BYTES = 26_738_688


@pytest.fixture(scope="module")
def raw_to_t16_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q8_0_raw_to_t16(load=True)


def test_sh6_p1_build_and_registry_surface() -> None:
    plan = plan_gguf_q8_0_raw_to_t16_build()
    assert plan.output_path.name == "gguf_q8_0_raw_to_t16.so"
    assert plan.sources[0].name == "gguf_q8_0_raw_to_t16.hip"

    register_gguf_q8_0_raw_to_t16_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="layout_transform",
        quant="gguf_q8_0",
        variant="raw_pair_to_t16",
    ) is gguf_q8_0_raw_pair_to_t16


def test_sh6_p1_wrapper_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="in_features must be divisible by 32"):
        gguf_q8_0_raw_pair_to_t16(0, 0, 0, 0, 31, 16, 16)
    with pytest.raises(ValueError, match="out_features.*divisible by 16"):
        gguf_q8_0_raw_pair_to_t16(0, 0, 0, 0, 32, 15, 16)
    with pytest.raises(ValueError, match="threads must be one of"):
        gguf_q8_0_raw_pair_to_t16(0, 0, 0, 0, 32, 16, 16, threads=96)


def test_sh6_p1_production_pair_uses_one_bounded_scratch_owner() -> None:
    assert gguf_q8_0_raw_pair_to_t16_nbytes(
        _PRODUCTION_IN_FEATURES,
        _PRODUCTION_OUT_FEATURES_A,
        _PRODUCTION_OUT_FEATURES_B,
    ) == _PRODUCTION_PAIR_BYTES


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("threads", [64, 128, 256])
def test_sh6_p1_gpu_pair_repack_matches_host_packer_on_both_production_shapes(
    threads: int,
    raw_to_t16_library,
) -> None:
    rng = np.random.default_rng(0x5A61 + threads)
    blocks_per_row = _PRODUCTION_IN_FEATURES // 32
    raw_a = rng.integers(
        0,
        256,
        size=(_PRODUCTION_OUT_FEATURES_A, blocks_per_row * 34),
        dtype=np.uint8,
    )
    raw_b = rng.integers(
        0,
        256,
        size=(_PRODUCTION_OUT_FEATURES_B, blocks_per_row * 34),
        dtype=np.uint8,
    )
    expected_a = repack_gguf_q8_0_tile16(raw_a).tiles.reshape(-1)
    expected_b = repack_gguf_q8_0_tile16(raw_b).tiles.reshape(-1)
    scratch_nbytes = gguf_q8_0_raw_pair_to_t16_nbytes(
        _PRODUCTION_IN_FEATURES,
        _PRODUCTION_OUT_FEATURES_A,
        _PRODUCTION_OUT_FEATURES_B,
    )
    actual = np.empty((scratch_nbytes,), dtype=np.uint8)

    raw_a_buf = malloc(raw_a.nbytes)
    raw_b_buf = malloc(raw_b.nbytes)
    scratch = malloc(scratch_nbytes)
    try:
        copy_host_to_device(raw_a_buf, host_array_ptr(raw_a), raw_a.nbytes)
        copy_host_to_device(raw_b_buf, host_array_ptr(raw_b), raw_b.nbytes)
        gguf_q8_0_raw_pair_to_t16(
            raw_a_buf.ptr,
            raw_b_buf.ptr,
            scratch.ptr,
            scratch.ptr + raw_a.nbytes,
            _PRODUCTION_IN_FEATURES,
            _PRODUCTION_OUT_FEATURES_A,
            _PRODUCTION_OUT_FEATURES_B,
            threads=threads,
            library=raw_to_t16_library,
        )
        copy_device_to_host(host_array_ptr(actual), scratch, actual.nbytes)
    finally:
        free(scratch)
        free(raw_b_buf)
        free(raw_a_buf)

    np.testing.assert_array_equal(actual[: raw_a.nbytes], expected_a)
    np.testing.assert_array_equal(actual[raw_a.nbytes :], expected_b)
