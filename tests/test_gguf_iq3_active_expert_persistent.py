"""WPF-H5Q exact active-expert persistent IQ3 traversal contract."""

from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_gguf_iq_gemv import (
    _bf16_u16_to_f32,
    _f32_to_bf16_u16,
    _make_x,
    _selected_reference,
)

_PARTITIONS = (64,)
_IN_FEATURES = 1024
_OUT_FEATURES = 3072
_NUM_EXPERTS = 256
_IQ3_BLOCK_BYTES = 98


def _variant(partition: int) -> str:
    return (
        "selected_grouped_prefill_compact_k1024_active_expert_p"
        f"{partition}_resident_rowbatch8_bf16_bf16_out"
    )


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def grouped_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return module.build_gguf_iq_selected_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


def test_h5q_partition_registry_preflight_and_gfx1151_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )
    candidates = getattr(module, "GGUF_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITIONS")
    assert tuple(candidates) == _PARTITIONS
    for partition, function in candidates.items():
        key = KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq3_xxs",
            _variant(partition),
        )
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is function

    load_backend_kernel_package("hip_gfx1151")
    for partition in _PARTITIONS:
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                "moe_linear",
                "gguf_iq3_xxs",
                _variant(partition),
            )
        )

    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H5Q shape reached the HIP loader")

    monkeypatch.setattr(module, "build_gguf_iq_selected_prefill", fail_if_loaded)
    candidate = candidates[64]
    common = dict(
        compact_rows=9,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=_NUM_EXPERTS,
    )
    for changed, message in (
        ({"in_features": 768}, "exactly 1024"),
        ({"out_features": 1024}, "exactly 3072"),
        ({"num_experts": 255}, "exactly 256"),
    ):
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, 4, 5, 6, **(common | changed))
    assert load_attempts == 0


def _device_buffer(array: np.ndarray, buffers: list[Any]):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
    buffers.append(buffer)
    return buffer


def _run_h5j_or_h5q(
    function,
    library,
    *,
    x_bf16: np.ndarray,
    starts: np.ndarray,
    active_experts: np.ndarray,
    active_count: np.ndarray,
    qweight: np.ndarray,
    initial: np.ndarray,
    persistent: bool,
) -> np.ndarray:
    buffers: list[Any] = []
    out = np.ascontiguousarray(initial.copy())
    try:
        x_buf = _device_buffer(x_bf16, buffers)
        starts_buf = _device_buffer(starts, buffers)
        active_buf = _device_buffer(active_experts, buffers)
        count_buf = _device_buffer(active_count, buffers)
        weight_buf = _device_buffer(qweight, buffers)
        out_buf = _device_buffer(out, buffers)
        args = [x_buf.ptr, starts_buf.ptr]
        if persistent:
            args.extend((active_buf.ptr, count_buf.ptr))
        args.extend((weight_buf.ptr, out_buf.ptr))
        function(
            *args,
            compact_rows=x_bf16.shape[0],
            in_features=_IN_FEATURES,
            out_features=_OUT_FEATURES,
            num_experts=_NUM_EXPERTS,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)
    return out


def _make_metadata(*, tail: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active_limit = 129 if tail else 128
    pattern = (1, 2, 7, 8, 9)
    for expert in range(active_limit):
        counts[expert] = pattern[expert % len(pattern)]
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active[:active_limit] = np.arange(active_limit, dtype=np.int64)
    return starts, active, np.asarray([active_limit], dtype=np.int64)


def _make_iq3_weight(experts: int) -> np.ndarray:
    row_bytes = (_IN_FEATURES // 256) * _IQ3_BLOCK_BYTES
    rng = np.random.default_rng(0xA503)
    weight = rng.integers(
        0,
        256,
        size=(experts, _OUT_FEATURES, row_bytes),
        dtype=np.uint8,
    )
    scale = np.asarray([np.float16(0.00390625)], dtype=np.float16).view(np.uint8)
    for block in range(_IN_FEATURES // 256):
        start = block * _IQ3_BLOCK_BYTES
        weight[:, :, start : start + 2] = scale
    return weight


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h5q_partitions_match_h5j_and_cpu_at_boundary_and_tail(grouped_library) -> None:
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )
    candidates = module.GGUF_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITIONS
    control = (
        module.gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out
    )
    qweight = _make_iq3_weight(129)

    for tail in (False, True):
        starts, active, active_count = _make_metadata(tail=tail)
        compact_rows = int(starts[-1])
        x_bf16 = _f32_to_bf16_u16(_make_x(compact_rows, _IN_FEATURES))
        initial = np.full((compact_rows, _OUT_FEATURES), 0x7FC0, dtype=np.uint16)
        expected = _run_h5j_or_h5q(
            control,
            grouped_library,
            x_bf16=x_bf16,
            starts=starts,
            active_experts=active,
            active_count=active_count,
            qweight=qweight,
            initial=initial,
            persistent=False,
        )
        for partition, candidate in candidates.items():
            actual = _run_h5j_or_h5q(
                candidate,
                grouped_library,
                x_bf16=x_bf16,
                starts=starts,
                active_experts=active,
                active_count=active_count,
                qweight=qweight,
                initial=initial,
                persistent=True,
            )
            np.testing.assert_array_equal(
                actual,
                expected,
                err_msg=f"partition={partition} tail={tail}",
            )

        sample_rows = np.asarray([0, compact_rows // 2, compact_rows - 1])
        selected = np.searchsorted(starts[1:], sample_rows, side="right").astype(
            np.int64
        )
        sample_cols = np.asarray([0, 1535, 3071])
        cpu = _selected_reference(
            x_bf16[sample_rows],
            selected,
            qweight[:, sample_cols, :],
            GGMLQuantizationType.IQ3_XXS,
        )
        np.testing.assert_array_equal(expected[np.ix_(sample_rows, sample_cols)], cpu)
        assert np.isfinite(_bf16_u16_to_f32(cpu)).all()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h5q_empty_active_list_preserves_output(grouped_library) -> None:
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active_count = np.zeros(1, dtype=np.int64)
    x_bf16 = _f32_to_bf16_u16(_make_x(1, _IN_FEATURES))
    qweight = _make_iq3_weight(1)
    initial = np.full((1, _OUT_FEATURES), 0x3F80, dtype=np.uint16)
    for partition in _PARTITIONS:
        actual = _run_h5j_or_h5q(
            module.GGUF_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITIONS[partition],
            grouped_library,
            x_bf16=x_bf16,
            starts=starts,
            active_experts=active,
            active_count=active_count,
            qweight=qweight,
            initial=initial,
            persistent=True,
        )
        np.testing.assert_array_equal(actual, initial)
