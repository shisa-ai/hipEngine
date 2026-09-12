"""Exact gate for native host batching of existing Laguna decode launches."""

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
from hipengine.kernels.backends import (
    backend_package_capability,
    load_backend_kernel_package,
)
from hipengine.kernels.hip_gfx1100.runtime.laguna_launch_batch import (
    build_laguna_launch_batch,
    laguna_q4_shared_down_tail_batch,
    laguna_q4_t16_shared_down_tail_batch,
    laguna_q6_shared_down_tail_batch,
    plan_laguna_launch_batch_build,
)
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf_q4_k import (
    repack_gguf_q4_k_pack8,
    repack_gguf_q4_k_tile16,
)
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q6_k_weight


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


def test_laguna_launch_batch_build_plan_is_dry_run_safe() -> None:
    plan = plan_laguna_launch_batch_build(compiler_version="test-compiler")
    assert plan.output_path.name == "laguna_launch_batch.so"
    assert plan.sources[0].name == "laguna_launch_batch.hip"
    dry = build_laguna_launch_batch(
        dry_run=True,
        compiler_version="test-compiler",
    )
    assert dry.output_path == plan.output_path


def test_laguna_launch_batch_exports_shared_down_abis() -> None:
    assert callable(laguna_q4_shared_down_tail_batch)
    assert callable(laguna_q4_t16_shared_down_tail_batch)
    assert callable(laguna_q6_shared_down_tail_batch)


def test_gfx1151_registers_both_native_launch_batch_keys() -> None:
    load_backend_kernel_package("hip_gfx1151")
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_SHARED_DOWN_MOE_TAIL_HOST_BATCH",
        False,
    )
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear+moe_tail+next_rmsnorm_host_batch",
            "gguf_q4_k",
            "pack8_bf16_bf16_out",
        )
    )
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear+moe_tail+next_rmsnorm_host_batch",
            "gguf_q6_k",
            "pack8_gemv_decode_bf16_bf16_out",
        )
    )
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear+moe_tail+next_rmsnorm_host_batch",
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        )
    )


def _upload(array: np.ndarray):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(max(4, contiguous.nbytes))
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
    return buffer


def _download_u16(buffer, shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes)
    return out


@requires_hip
@pytest.mark.parametrize("quant", ["q4", "q4_t16", "q6"])
def test_native_shared_down_tail_batch_preserves_existing_launches(
    quant: str,
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
        build_paro_combine,
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_gemv
    from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_pack8_gemv

    rows, in_features, hidden = 1, 1024, 3072
    quant_seed = {"q4": 4, "q4_t16": 5, "q6": 6}[quant]
    rng = np.random.default_rng(0xBA7C + quant_seed)
    x = float_array_to_bf16_bits(
        rng.standard_normal((rows, in_features)).astype(np.float32) * 0.2
    )
    routed = float_array_to_bf16_bits(
        rng.standard_normal(hidden).astype(np.float32) * 0.4
    )
    post_attention = float_array_to_bf16_bits(
        rng.standard_normal(hidden).astype(np.float32) * 0.4
    )
    norm_weight = rng.uniform(0.25, 1.75, size=hidden).astype(np.float32)
    runtime = get_hip_runtime()
    tail_library = build_paro_combine(load=True)
    batch_library = build_laguna_launch_batch(load=True)
    tail_function = getattr(
        tail_library,
        "hipengine_laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out",
    )

    if quant == "q4":
        packed = repack_gguf_q4_k_pack8(make_q4_k_weight(hidden, in_features))
        weight_arrays = (packed.qweight, packed.scales, packed.mins)
        projection_library = gguf_q4_k_gemv.build_gguf_q4_k_gemv(load=True)
        projection_function = getattr(
            projection_library,
            "hipengine_gguf_q4_k_pack8_gemv_bf16_bf16_out",
        )
    elif quant == "q4_t16":
        from hipengine.kernels.hip_gfx1100.quant import (
            gguf_t16_selected_gemv,
        )

        tiles = repack_gguf_q4_k_tile16(
            make_q4_k_weight(hidden, in_features)[None, ...]
        ).tiles
        weight_arrays = (tiles,)
        projection_library = (
            gguf_t16_selected_gemv.build_gguf_t16_selected_gemv(load=True)
        )
        projection_function = getattr(
            projection_library,
            "hipengine_gguf_q4_k_t16_dense_single_local32_gemv_"
            "bf16_bf16_out",
        )
    else:
        weight_arrays = (make_q6_k_weight(hidden, in_features),)
        projection_library = (
            gguf_q6_k_pack8_gemv.build_gguf_q6_k_pack8_gemv(load=True)
        )
        projection_function = getattr(
            projection_library,
            "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_bf16_out",
        )

    device_inputs = [
        _upload(array)
        for array in (x, *weight_arrays, routed, post_attention, norm_weight)
    ]
    output_nbytes = hidden * np.dtype(np.uint16).itemsize
    device_outputs = [malloc(output_nbytes) for _ in range(6)]
    try:
        x_d = device_inputs[0]
        routed_d, post_d, norm_weight_d = device_inputs[-3:]
        (
            control_shared_d,
            control_norm_d,
            control_hidden_d,
            batch_shared_d,
            batch_norm_d,
            batch_hidden_d,
        ) = device_outputs
        if quant == "q4":
            qweight_d, scales_d, mins_d = device_inputs[1:4]
            gguf_q4_k_gemv.gguf_q4_k_pack8_gemv_bf16_bf16_out(
                x_d.ptr,
                qweight_d.ptr,
                scales_d.ptr,
                mins_d.ptr,
                control_shared_d.ptr,
                rows,
                in_features,
                hidden,
                threads=32,
                library=projection_library,
                runtime=runtime,
            )
        elif quant == "q4_t16":
            (tiles_d,) = device_inputs[1:2]
            (
                gguf_t16_selected_gemv
                .gguf_q4_k_t16_dense_single_local32_bf16_bf16_out
            )(
                x_d.ptr,
                tiles_d.ptr,
                control_shared_d.ptr,
                rows,
                in_features,
                hidden,
                library=projection_library,
                runtime=runtime,
            )
        else:
            (qweight_d,) = device_inputs[1:2]
            gguf_q6_k_pack8_gemv.gguf_q6_k_pack8_gemv_decode_bf16_bf16_out(
                x_d.ptr,
                qweight_d.ptr,
                control_shared_d.ptr,
                rows,
                in_features,
                hidden,
                library=projection_library,
                runtime=runtime,
            )
        laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out(
            routed_d.ptr,
            control_shared_d.ptr,
            post_d.ptr,
            norm_weight_d.ptr,
            control_norm_d.ptr,
            control_hidden_d.ptr,
            hidden,
            library=tail_library,
            runtime=runtime,
        )

        if quant == "q4":
            laguna_q4_shared_down_tail_batch(
                projection_function,
                tail_function,
                x_d.ptr,
                qweight_d.ptr,
                scales_d.ptr,
                mins_d.ptr,
                batch_shared_d.ptr,
                routed_d.ptr,
                post_d.ptr,
                norm_weight_d.ptr,
                batch_norm_d.ptr,
                batch_hidden_d.ptr,
                rows,
                in_features,
                hidden,
                library=batch_library,
                runtime=runtime,
            )
        elif quant == "q4_t16":
            laguna_q4_t16_shared_down_tail_batch(
                projection_function,
                tail_function,
                x_d.ptr,
                tiles_d.ptr,
                batch_shared_d.ptr,
                routed_d.ptr,
                post_d.ptr,
                norm_weight_d.ptr,
                batch_norm_d.ptr,
                batch_hidden_d.ptr,
                rows,
                in_features,
                hidden,
                library=batch_library,
                runtime=runtime,
            )
        else:
            laguna_q6_shared_down_tail_batch(
                projection_function,
                tail_function,
                x_d.ptr,
                qweight_d.ptr,
                batch_shared_d.ptr,
                routed_d.ptr,
                post_d.ptr,
                norm_weight_d.ptr,
                batch_norm_d.ptr,
                batch_hidden_d.ptr,
                rows,
                in_features,
                hidden,
                library=batch_library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        control = [
            _download_u16(buffer, (hidden,)) for buffer in device_outputs[:3]
        ]
        candidate = [
            _download_u16(buffer, (hidden,)) for buffer in device_outputs[3:]
        ]
    finally:
        for buffer in reversed([*device_outputs, *device_inputs]):
            free(buffer)

    for actual, expected in zip(candidate, control, strict=True):
        np.testing.assert_array_equal(actual, expected)
