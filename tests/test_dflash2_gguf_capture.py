"""DFlash2 GGUF target-tap capture plumbing tests.

Covers the DFlash2HiddenCaptureTargets ABI used to retain the full-prompt
post-layer hidden at the DFlash2 tap depths during GGUF bulk prefill. Device
execution of the capture path is covered by the D1 end-to-end driver
(scripts/dflash2_gguf_cycle.py) on gfx1151; this file validates the
host-side contract only.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.runtime.qwen35_gguf_runner import (
    DFLASH2_TAP_DEPTHS,
    DFLASH2_TAP_LAYER_IDS,
    DFlash2HiddenCaptureTargets,
)

HAS_HIP = False
try:
    import ctypes

    ctypes.CDLL("libamdhip64.so")
    HAS_HIP = True
except Exception:  # pragma: no cover - no-ROCm runners
    HAS_HIP = False

pytestmark = pytest.mark.skipif(not HAS_HIP, reason="requires ROCm runtime")


def _device_buffers(rows: int, hidden_size: int, depths: tuple[int, ...]) -> dict[int, object]:
    runtime = get_hip_runtime()
    from hipengine.core.memory import DeviceBuffer

    buffers = {}
    nbytes = rows * hidden_size * DType.BF16.itemsize
    for depth in depths:
        buffers[depth] = DeviceBuffer(ptr=runtime.malloc(nbytes), nbytes=nbytes)
    return buffers


@pytest.fixture

def runtime():
    return get_hip_runtime()


def _free_buffers(buffers) -> None:
    runtime = get_hip_runtime()
    for buf in buffers.values():
        runtime.free(buf.ptr)


def test_tap_depth_mapping() -> None:
    # The capture depths must be the post-layer (layer_id + 1) depths for the
    # reference drafter's target_layer_ids.
    assert DFLASH2_TAP_LAYER_IDS == (5, 19, 33, 47, 61)
    assert DFLASH2_TAP_DEPTHS == tuple(layer + 1 for layer in DFLASH2_TAP_LAYER_IDS)


def test_capture_targets_valid(runtime) -> None:
    buffers = _device_buffers(rows=16, hidden_size=5120, depths=DFLASH2_TAP_DEPTHS)
    try:
        targets = DFlash2HiddenCaptureTargets(hidden_size=5120, rows=16, buffers=buffers)
        assert targets.rows == 16
        assert set(targets.buffers) == set(DFLASH2_TAP_DEPTHS)
    finally:
        _free_buffers(buffers)


def test_capture_targets_reject_unknown_depth(runtime) -> None:
    buffers = _device_buffers(rows=4, hidden_size=5120, depths=(6, 20, 34, 48, 62, 63))
    try:
        with pytest.raises(ValueError, match="tap depths"):
            DFlash2HiddenCaptureTargets(hidden_size=5120, rows=4, buffers=buffers)
    finally:
        _free_buffers(buffers)


def test_capture_targets_reject_wrong_size(runtime) -> None:
    buffers = _device_buffers(rows=4, hidden_size=5120, depths=DFLASH2_TAP_DEPTHS)
    try:
        # Wrong hidden_size contract: buffer holds rows*hidden_size BF16.
        with pytest.raises(ValueError, match="must hold exactly"):
            DFlash2HiddenCaptureTargets(hidden_size=5130, rows=4, buffers=buffers)
    finally:
        _free_buffers(buffers)


def test_capture_targets_reject_non_devicebuffer(runtime) -> None:
    with pytest.raises(TypeError, match="DeviceBuffer"):
        DFlash2HiddenCaptureTargets(
            hidden_size=5120,
            rows=4,
            buffers={6: np.zeros(4 * 5120, dtype=np.float16)},
        )


def test_capture_targets_reject_nonpositive_rows() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        DFlash2HiddenCaptureTargets(hidden_size=5120, rows=0, buffers={})
