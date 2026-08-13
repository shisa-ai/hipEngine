from __future__ import annotations

import ctypes
from types import MappingProxyType

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference.ops import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFWeightSpec,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
from hipengine.runtime.gguf_linear import (
    clear_gguf_linear_dispatch_cache,
    launch_gguf_linear,
    launch_gguf_linear_pair_silu,
    native_batch_decode_session,
    resolve_gguf_linear_dispatch,
)
from tests._gguf_synthetic_weights import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _f32_to_bf16_bits(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_f32(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _weight(ptr: int, *, in_features: int = 256, out_features: int = 16) -> Qwen35GGUFDeviceWeight:
    source = GGUFTensorInfo(
        name=f"weight.{ptr:x}",
        shape=(out_features, in_features),
        ggml_shape=(in_features, out_features),
        ggml_type=int(GGMLQuantizationType.Q4_K),
        ggml_type_name="Q4_K",
        n_elements=out_features * in_features,
        nbytes=out_features * (in_features // 256) * 144,
        offset=0,
        data_offset=0,
        byte_shape=(out_features, (in_features // 256) * 144),
    )
    spec = Qwen35GGUFWeightSpec(
        slot_path=f"layers.0.weight_{ptr:x}",
        source=source,
        quant_key="gguf_q4_k_t16_v1",
        layout=LAYOUT_GGUF_Q4_K_T16,
        allocation_names=("tiles",),
    )
    nbytes = out_features // 16 * (in_features // 256) * 2368
    buffer = DeviceBuffer(ptr=ptr, nbytes=nbytes)
    allocation = DeviceTensorAllocation(
        name=f"{source.name}.t16.tiles",
        source=source,
        buffer=buffer,
        tensor=Tensor.from_handle(ptr, (nbytes,), DType.INT8, Device("hip", 0)),
    )
    return Qwen35GGUFDeviceWeight(
        spec=spec,
        allocations=MappingProxyType({"tiles": allocation}),
        backend="hip_gfx1100",
    )


def test_q4_t16_dense_dispatch_uses_one_tiles_abi_for_decode_and_prefill() -> None:
    weight = _weight(0x1000)

    decode = resolve_gguf_linear_dispatch(weight, rows=1)
    prefill = resolve_gguf_linear_dispatch(weight, rows=512)

    assert decode.key == KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_single_local32_bf16_bf16_out",
    )
    assert prefill.key == KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    )
    assert decode.abi == prefill.abi == "t16"


def test_q4_t16_dense_launch_routes_c1_small_rows_and_bulk_without_shadow_allocations() -> None:
    weight = _weight(0x1000)
    keys = (
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "t16_wmma_prefill_bf16_bf16_out",
        ),
    )
    originals = {key: resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant) for key in keys}
    calls: list[tuple[str, tuple]] = []

    try:
        for key in keys:
            register(
                key,
                lambda *args, _variant=key.variant, **kwargs: calls.append((_variant, args)),
                replace=True,
            )
        clear_gguf_linear_dispatch_cache()
        launch_gguf_linear(weight, 0x2000, 0x3000, 1, 256, 16)
        with native_batch_decode_session(True):
            launch_gguf_linear(weight, 0x2000, 0x3000, 3, 256, 16)
        launch_gguf_linear(weight, 0x2000, 0x3000, 512, 256, 16)
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)
        clear_gguf_linear_dispatch_cache()

    assert [variant for variant, _args in calls] == [
        "dense_single_local32_bf16_bf16_out",
        "dense_rowtile_bf16_bf16_out",
        "t16_wmma_prefill_bf16_bf16_out",
    ]
    assert all(args[:3] == (0x2000, 0x1000, 0x3000) for _variant, args in calls)


def test_q4_t16_dense_c1_pair_silu_uses_canonical_tiles() -> None:
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_local32_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple] = []
    try:
        register(key, lambda *args, **kwargs: calls.append(args), replace=True)
        launched = launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            1,
            5_120,
            17_408,
            use_gemv_decode=True,
        )
    finally:
        register(key, original, replace=True)

    assert launched
    assert calls[0][:4] == (0x3000, 0x1000, 0x2000, 0x4000)


def test_q4_t16_dense_small_row_pair_silu_uses_canonical_tiles() -> None:
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_rowtile_bf16_bf16_out",
    )
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    calls: list[tuple] = []
    try:
        register(key, lambda *args, **kwargs: calls.append(args), replace=True)
        with native_batch_decode_session(True):
            launched = launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                0x3000,
                0x4000,
                3,
                5_120,
                17_408,
            )
    finally:
        register(key, original, replace=True)

    assert launched
    assert calls[0][:4] == (0x3000, 0x1000, 0x2000, 0x4000)


def test_q4_t16_dense_bulk_pair_silu_uses_canonical_tiles() -> None:
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple] = []
    try:
        register(key, lambda *args, **kwargs: calls.append(args), replace=True)
        launched = launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            512,
            5_120,
            17_408,
        )
    finally:
        register(key, original, replace=True)
        clear_gguf_linear_dispatch_cache()

    assert launched
    assert calls[0][:4] == (0x3000, 0x1000, 0x2000, 0x4000)


@pytest.mark.parametrize("rows", [16, 33, 511])
def test_q4_t16_dense_bulk_pair_silu_keeps_unfused_fallback_below_512(
    rows: int,
) -> None:
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)

    assert not launch_gguf_linear_pair_silu(
        weight_a,
        weight_b,
        0x3000,
        0x4000,
        rows,
        5_120,
        17_408,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [16, 33, 512, 1_024, 4_096])
def test_q4_t16_dense_wmma_prefill_matches_cpu_reference(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    in_features = 256
    out_features = 32
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    rng = np.random.default_rng(36 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    out_bits = np.zeros((rows, out_features), dtype=np.uint16)
    baseline_bits = np.zeros_like(out_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_dev = malloc(tiles.nbytes, runtime=runtime)
        out_dev = malloc(out_bits.nbytes, runtime=runtime)
        baseline_dev = malloc(baseline_bits.nbytes, runtime=runtime)
        buffers.extend((x_dev, tiles_dev, out_dev, baseline_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(tiles_dev, host_array_ptr(tiles), runtime=runtime)
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_wmma_prefill_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            baseline_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            out_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_bits), out_dev, runtime=runtime)
        copy_device_to_host(
            host_array_ptr(baseline_bits), baseline_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(out_bits, baseline_bits)
    x_f32 = _bf16_bits_to_f32(x_bits)
    actual = _bf16_bits_to_f32(out_bits)
    expected = _bf16_bits_to_f32(
        _f32_to_bf16_bits(
            gguf_quant_gemv(x_f32, raw, GGMLQuantizationType.Q4_K)
        )
    )
    # Match the established Q4_K WMMA gate: independent FP16 fragments may
    # differ near zero while BF16-scale outputs remain bounded.
    np.testing.assert_allclose(actual, expected, rtol=0.012, atol=0.5)
    assert np.isfinite(actual).all()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [512, 513, 1_024])
def test_q4_t16_dense_dual_wmma_silu_matches_unfused_chain(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
        build_paro_silu,
        silu_mul_separate_out_bf16,
    )

    runtime = get_hip_runtime()
    in_features = 256
    out_features = 32
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, shift=1, axis=0).copy()
    tiles_a = repack_gguf_q4_k_tile16(raw_a[None, ...]).tiles
    tiles_b = repack_gguf_q4_k_tile16(raw_b[None, ...]).tiles
    rng = np.random.default_rng(3600 + rows)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    expected_bits = np.zeros((rows, out_features), dtype=np.uint16)
    actual_bits = np.zeros_like(expected_bits)
    buffers = []
    try:
        x_dev = malloc(x_bits.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        gate_dev = malloc(expected_bits.nbytes, runtime=runtime)
        up_dev = malloc(expected_bits.nbytes, runtime=runtime)
        expected_dev = malloc(expected_bits.nbytes, runtime=runtime)
        actual_dev = malloc(actual_bits.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                tiles_a_dev,
                tiles_b_dev,
                gate_dev,
                up_dev,
                expected_dev,
                actual_dev,
            )
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(
            tiles_a_dev, host_array_ptr(tiles_a), runtime=runtime
        )
        copy_host_to_device(
            tiles_b_dev, host_array_ptr(tiles_b), runtime=runtime
        )
        library = build_gguf_k_t16_selected_prefill(load=True)
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            gate_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
            x_dev.ptr,
            tiles_b_dev.ptr,
            up_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        silu_mul_separate_out_bf16(
            gate_dev.ptr,
            up_dev.ptr,
            expected_dev.ptr,
            rows,
            out_features,
            library=build_paro_silu(load=True),
            runtime=runtime,
        )
        gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out(
            x_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            actual_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(expected_bits), expected_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(actual_bits), actual_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_bits, expected_bits)
