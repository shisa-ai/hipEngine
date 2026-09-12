"""Exact RED/GREEN gate for device-owned Laguna decode control publication."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


requires_hip = pytest.mark.skipif(
    not _hip_available(),
    reason="HIP runtime is not available",
)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


@pytest.fixture(scope="module")
def _library(hip_test_target_arch: str):
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = (
        Path(compiler_file).read_text(encoding="utf-8")
        if compiler_file
        else None
    )
    require_cached = os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"
    with hip_target_arch_environment(hip_test_target_arch):
        return build_lm_head(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        )


@requires_hip
def test_argmax_publish_control_matches_top1_and_publishes_next_state(
    _library,
    _runtime,
) -> None:
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.linear.lm_head import (
        argmax_f32,
        argmax_f32_publish_control,
        lm_head_argmax_stage1_blocks,
    )

    logits = np.asarray(
        [-7.0, 2.0, 9.5, 1.0, 9.5, -3.0, 8.0, 0.0] * 513,
        dtype=np.float32,
    )
    # Exercise the stable minimum-index tie break outside the first chunk.
    logits[2] = 8.5
    logits[2049] = 11.0
    logits[3073] = 11.0
    blocks = lm_head_argmax_stage1_blocks(int(logits.size))
    buffers = []
    try:
        logits_d = malloc(logits.nbytes, runtime=_runtime)
        buffers.append(logits_d)
        copy_host_to_device(logits_d, host_array_ptr(logits), runtime=_runtime)
        block_values_d = malloc(blocks * 4, runtime=_runtime)
        block_indices_d = malloc(blocks * 8, runtime=_runtime)
        control_block_values_d = malloc(blocks * 4, runtime=_runtime)
        control_block_indices_d = malloc(blocks * 8, runtime=_runtime)
        out_index_d = malloc(8, runtime=_runtime)
        out_value_d = malloc(4, runtime=_runtime)
        control_index_d = malloc(8, runtime=_runtime)
        control_value_d = malloc(4, runtime=_runtime)
        token_d = malloc(8, runtime=_runtime)
        scratch_position_d = malloc(8, runtime=_runtime)
        kv_position_d = malloc(8, runtime=_runtime)
        buffers.extend(
            (
                block_values_d,
                block_indices_d,
                control_block_values_d,
                control_block_indices_d,
                out_index_d,
                out_value_d,
                control_index_d,
                control_value_d,
                token_d,
                scratch_position_d,
                kv_position_d,
            )
        )

        argmax_f32(
            logits_d.ptr,
            block_values_d.ptr,
            block_indices_d.ptr,
            out_index_d.ptr,
            out_value_d.ptr,
            int(logits.size),
            library=_library,
            runtime=_runtime,
        )
        argmax_f32_publish_control(
            logits_d.ptr,
            control_block_values_d.ptr,
            control_block_indices_d.ptr,
            control_index_d.ptr,
            control_value_d.ptr,
            token_d.ptr,
            scratch_position_d.ptr,
            kv_position_d.ptr,
            int(logits.size),
            639,
            library=_library,
            runtime=_runtime,
        )
        _runtime.device_synchronize()

        baseline_index = np.empty(1, dtype=np.int64)
        baseline_value = np.empty(1, dtype=np.float32)
        control_index = np.empty(1, dtype=np.int64)
        control_value = np.empty(1, dtype=np.float32)
        token = np.empty(1, dtype=np.int64)
        scratch_position = np.empty(1, dtype=np.int64)
        kv_position = np.empty(1, dtype=np.int64)
        for host, device in (
            (baseline_index, out_index_d),
            (baseline_value, out_value_d),
            (control_index, control_index_d),
            (control_value, control_value_d),
            (token, token_d),
            (scratch_position, scratch_position_d),
            (kv_position, kv_position_d),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=_runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=_runtime)

    np.testing.assert_array_equal(control_index, baseline_index)
    np.testing.assert_array_equal(control_value.view(np.uint32), baseline_value.view(np.uint32))
    np.testing.assert_array_equal(token, baseline_index)
    np.testing.assert_array_equal(scratch_position, np.asarray([639], dtype=np.int64))
    np.testing.assert_array_equal(kv_position, np.asarray([639], dtype=np.int64))


def test_prepublished_kv_position_advances_without_a_host_copy() -> None:
    from hipengine.runtime.laguna_kv import LagunaKVCache

    cache = object.__new__(LagunaKVCache)
    cache._closed = False
    cache._pending_positions = ()
    cache.context_length = 1024
    cache.position = 637
    cache.runtime = SimpleNamespace(memcpy=lambda *args, **kwargs: pytest.fail("copy"))

    cache.adopt_prepublished_position(638)

    assert cache.position == 638


def test_forward_token_reuses_only_the_exact_previous_argmax_control() -> None:
    from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession

    source = __import__("inspect").getsource(LagunaGGUFResidentSession.forward_token)
    assert "_prepublished_control_position" in source
    assert "_prepublished_control_token" in source
    assert "adopt_prepublished_position" in source
    assert "token == self._prepublished_control_token" in source
