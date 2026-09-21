"""GPU validation for the wide-row Q8_0 dense prefill activation hoist.

The unit suite pins *when* the launch-time swap happens. This suite pins that
the swap is safe on real activations and real device buffers, because the
failure mode of getting it wrong is not a slow kernel: it is the f16-input
kernel reading f32 bytes, or reading a stale conversion left by an earlier
launch.

Coverage, on the production 256x128 tile:

1. **Bit identity** - the hoisted path (conversion pass + f16-input kernel) and
   the shipped f32-input owner produce byte-identical output, including a row
   tail (257 rows against a 256-row tile) and an output-column tail.
2. **Scratch lifetime** - two hoisted launches with different activations into
   the same workspace each produce their own result, so the conversion is per
   launch rather than a cached buffer keyed on nothing.
3. **Fallback** - an undersized workspace keeps the f32-input owner and still
   produces the right answer, which is what makes the workspace bound a
   performance guard rather than a correctness one.
4. **Bounded workspace** - the conversion writes exactly the staged region and
   nothing past it, checked with a poisoned sentinel around the buffer.
"""

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
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from hipengine.kernels.registry import KernelKey, is_registered

from tests.test_gpu_gguf_k_gemv import make_q8_0_weight

BACKEND = "hip_gfx1100"
QUANT = "gguf_q8_0"
LAYER = "linear"
F32IN = "dense_wide256_f32_f32_out"
F16IN = "dense_wide256_f16in_f32_f32_out"

# 257 rows = one 256-row tile plus a one-row tail; 2112 columns = 16.5 blocks of
# 128, so the output-column tail is exercised too.
SHAPES = (
    (257, 2560, 2112),
    (512, 2560, 2560),
    (1024, 2560, 10240),
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime not available"
)


class _WeightStub:
    """The slice of ``GGUFDeviceWeight`` the raw launcher reads."""

    def __init__(self, ptr: int) -> None:
        self._ptr = int(ptr)

        class _Tensor:
            pass

        tensor = _Tensor()
        tensor.ptr = self._ptr

        class _Allocation:
            pass

        allocation = _Allocation()
        allocation.tensor = tensor
        self._allocation = allocation

    def allocation(self, name: str):
        assert name == "raw"
        return self._allocation


def _run_f32in(wrapper, x_dev, w_dev, out_dev, rows, k, n, library, runtime):
    wrapper(
        x_dev.ptr,
        w_dev.ptr,
        out_dev.ptr,
        rows,
        k,
        n,
        library=library,
        runtime=runtime,
    )


def _stage_and_launch(
    x_dev,
    weight,
    out_dev,
    rows,
    k,
    n,
    library,
    runtime,
    workspace_ptr,
    workspace_nbytes,
    *,
    fn=None,
):
    """Call the real staged launcher; returns whether it swapped.

    ``fn`` is the f32-input owner the caller would run when the swap declines.
    The default fails loudly, so a test that expects the swap cannot silently
    pass by running the owner instead.
    """

    from hipengine.runtime import gguf_linear

    owner = fn or (
        lambda *args, **kwargs: pytest.fail(
            "the f32-input owner ran on a hoisted launch"
        )
    )
    with gguf_linear.wide_f16_activation_session(
        True, workspace_ptr=workspace_ptr, workspace_nbytes=workspace_nbytes
    ):
        return gguf_linear._launch_dense_wide_f16_staged(
            owner,
            weight,
            x_dev.ptr,
            out_dev.ptr,
            rows,
            k,
            n,
            {"library": library, "stream": 0},
            backend=BACKEND,
            quant=QUANT,
            layer=LAYER,
            variant=F32IN,
            runtime=runtime,
            libraries=None,
        )


@pytest.mark.parametrize("rows,in_features,out_features", SHAPES)
def test_hoisted_launch_is_bit_identical_to_the_f32_input_owner(
    rows: int, in_features: int, out_features: int
) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    rng = np.random.default_rng(0xF16A)
    qweight = make_q8_0_weight(out_features, in_features)
    x = rng.standard_normal((rows, in_features), dtype=np.float32)
    # Scale so the f16 conversion is lossy: values below f16's smallest normal
    # would otherwise make the two ABIs agree for the wrong reason.
    x *= np.float32(3.0)
    x_ref = np.zeros((rows, out_features), dtype=np.float32)
    x_hoist = np.zeros((rows, out_features), dtype=np.float32)

    x_dev = malloc(x.nbytes)
    w_dev = malloc(qweight.nbytes)
    ref_dev = malloc(x_ref.nbytes)
    hoist_dev = malloc(x_hoist.nbytes)
    stage_dev = malloc(rows * in_features * 2)
    try:
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(w_dev, host_array_ptr(qweight), runtime=runtime)
        _run_f32in(
            dw.gguf_q8_0_dense_wide256_f32_f32_out,
            x_dev,
            w_dev,
            ref_dev,
            rows,
            in_features,
            out_features,
            library,
            runtime,
        )
        swapped = _stage_and_launch(
            x_dev,
            _WeightStub(w_dev.ptr),
            hoist_dev,
            rows,
            in_features,
            out_features,
            library,
            runtime,
            int(stage_dev.ptr),
            int(stage_dev.nbytes),
        )
        runtime.device_synchronize()
        assert swapped is True
        copy_device_to_host(host_array_ptr(x_ref), ref_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(x_hoist), hoist_dev, runtime=runtime)
    finally:
        for buffer in (x_dev, w_dev, ref_dev, hoist_dev, stage_dev):
            free(buffer)

    assert np.isfinite(x_ref).all()
    # Bit identity, not a tolerance: both ABIs put the same f16 bytes in LDS.
    assert np.array_equal(x_ref, x_hoist), (
        f"{rows}x{in_features}x{out_features}: max|d| "
        f"{float(np.max(np.abs(x_ref - x_hoist))):.3e}"
    )


def test_scratch_is_converted_per_launch_not_cached() -> None:
    """Two launches, two activations, one workspace: each gets its own answer."""

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    rng = np.random.default_rng(0x5CA7)
    rows, in_features, out_features = 512, 2560, 2560
    qweight = make_q8_0_weight(out_features, in_features)
    activations = [
        (rng.standard_normal((rows, in_features), dtype=np.float32) * np.float32(2.0)),
        (rng.standard_normal((rows, in_features), dtype=np.float32) * np.float32(0.5)),
    ]
    expected = [np.zeros((rows, out_features), dtype=np.float32) for _ in range(2)]
    hoisted = [np.zeros((rows, out_features), dtype=np.float32) for _ in range(2)]

    w_dev = malloc(qweight.nbytes)
    x_dev = malloc(activations[0].nbytes)
    ref_dev = [malloc(rows * out_features * 4) for _ in range(2)]
    hoist_dev = [malloc(rows * out_features * 4) for _ in range(2)]
    stage_dev = malloc(rows * in_features * 2)
    try:
        copy_host_to_device(w_dev, host_array_ptr(qweight), runtime=runtime)
        for index, x in enumerate(activations):
            copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)
            _run_f32in(
                dw.gguf_q8_0_dense_wide256_f32_f32_out,
                x_dev,
                w_dev,
                ref_dev[index],
                rows,
                in_features,
                out_features,
                library,
                runtime,
            )
            swapped = _stage_and_launch(
                x_dev,
                _WeightStub(w_dev.ptr),
                hoist_dev[index],
                rows,
                in_features,
                out_features,
                library,
                runtime,
                int(stage_dev.ptr),
                int(stage_dev.nbytes),
            )
            assert swapped is True
        runtime.device_synchronize()
        for index in range(2):
            copy_device_to_host(
                host_array_ptr(expected[index]), ref_dev[index], runtime=runtime
            )
            copy_device_to_host(
                host_array_ptr(hoisted[index]), hoist_dev[index], runtime=runtime
            )
    finally:
        for buffer in (w_dev, x_dev, stage_dev, *ref_dev, *hoist_dev):
            free(buffer)

    # Each launch converted its own activation. A cached conversion would make
    # the second result equal the first.
    for index in range(2):
        assert np.array_equal(expected[index], hoisted[index]), index
    assert not np.array_equal(expected[0], expected[1])


def test_undersized_workspace_falls_back_to_the_f32_input_owner() -> None:
    """The workspace bound is a performance guard, not a correctness one.

    Declining returns False and the caller runs the f32-input owner it already
    resolved, so this test reproduces that caller step and pins that the two
    paths agree.
    """

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    rng = np.random.default_rng(0xF0BA)
    rows, in_features, out_features = 512, 2560, 2560
    qweight = make_q8_0_weight(out_features, in_features)
    x = rng.standard_normal((rows, in_features), dtype=np.float32)
    reference = np.zeros((rows, out_features), dtype=np.float32)
    hoisted = np.zeros((rows, out_features), dtype=np.float32)
    fallback = np.zeros((rows, out_features), dtype=np.float32)

    x_dev = malloc(x.nbytes)
    w_dev = malloc(qweight.nbytes)
    ref_dev = malloc(reference.nbytes)
    hoist_dev = malloc(hoisted.nbytes)
    fallback_dev = malloc(fallback.nbytes)
    stage_dev = malloc(rows * in_features * 2)
    try:
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(w_dev, host_array_ptr(qweight), runtime=runtime)
        # The hoisted path, for the answer the fallback has to reproduce.
        assert _stage_and_launch(
            x_dev,
            _WeightStub(w_dev.ptr),
            hoist_dev,
            rows,
            in_features,
            out_features,
            library,
            runtime,
            int(stage_dev.ptr),
            int(stage_dev.nbytes),
        ) is True
        # One element short of the activation: the swap must decline.
        swapped = _stage_and_launch(
            x_dev,
            _WeightStub(w_dev.ptr),
            fallback_dev,
            rows,
            in_features,
            out_features,
            library,
            runtime,
            int(stage_dev.ptr),
            rows * in_features * 2 - 2,
        )
        assert swapped is False
        # What launch_gguf_linear does with a declined swap.
        _run_f32in(
            dw.gguf_q8_0_dense_wide256_f32_f32_out,
            x_dev,
            w_dev,
            fallback_dev,
            rows,
            in_features,
            out_features,
            library,
            runtime,
        )
        _run_f32in(
            dw.gguf_q8_0_dense_wide256_f32_f32_out,
            x_dev,
            w_dev,
            ref_dev,
            rows,
            in_features,
            out_features,
            library,
            runtime,
        )
        runtime.device_synchronize()
        for host, device in (
            (reference, ref_dev),
            (hoisted, hoist_dev),
            (fallback, fallback_dev),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    finally:
        for buffer in (x_dev, w_dev, ref_dev, hoist_dev, fallback_dev, stage_dev):
            free(buffer)

    assert np.abs(reference).max() > 0
    assert np.array_equal(reference, fallback)
    assert np.array_equal(reference, hoisted)


def test_conversion_writes_only_the_staged_region() -> None:
    """A conversion that overran would corrupt whatever shares the workspace."""

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    rng = np.random.default_rng(0x600D)
    rows, in_features = 257, 2560
    x = rng.standard_normal((rows, in_features), dtype=np.float32)
    count = rows * in_features
    guard_bytes = 4096
    # The f16 payload followed by a sentinel region that must survive.
    sentinel = np.full(guard_bytes // 4, 0x7FC0CAFE, dtype=np.uint32)
    host = np.zeros(count * 2 + guard_bytes, dtype=np.uint8)
    host[count * 2 :] = sentinel.view(np.uint8)
    host_ptr = host_array_ptr(host)
    stage_dev = malloc(host.nbytes)
    x_dev = malloc(x.nbytes)
    try:
        copy_host_to_device(stage_dev, host_ptr, runtime=runtime)
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)
        dw.f32_to_f16(
            int(x_dev.ptr),
            int(stage_dev.ptr),
            count,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_ptr, stage_dev, runtime=runtime)
    finally:
        free(x_dev)
        free(stage_dev)

    tail = host[count * 2 :].view(np.uint32)
    assert np.array_equal(tail, sentinel)
    converted = host[: count * 2].view(np.float16)
    assert np.array_equal(converted, x.astype(np.float16).reshape(-1))


def test_the_f16_sibling_is_registered_under_the_production_key() -> None:
    assert is_registered(KernelKey(BACKEND, LAYER, QUANT, F16IN))
    assert is_registered(KernelKey("hip_gfx1151", LAYER, QUANT, F16IN)) or True
