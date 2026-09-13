"""gfx1151 exact output-to-router any-order continuation contract."""

from __future__ import annotations

import ctypes
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.materialize import float_array_to_bf16_bits


_LINEAR_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "linear"
    / "laguna_f16_projection.hip"
)
_ROUTER_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "moe"
    / "router.hip"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"


def test_anyorder_output_and_router_symbols_are_separate_continuations() -> None:
    linear = _LINEAR_SOURCE.read_text(encoding="utf-8")
    router = _ROUTER_SOURCE.read_text(encoding="utf-8")

    assert (
        "hipengine_laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_signal_bf16"
        in linear
    )
    assert (
        "hipengine_qwen35_router_logits_bf16_f32w_wave0_tree_anyorder"
        in router
    )
    assert "hipExtAnyOrderLaunch" not in linear
    assert "hipExtAnyOrderLaunch" in router
    assert "completion_counter + 1" in linear
    assert "completion_counter + 1" in router
    assert "completion_counter + 2" in router


def test_anyorder_output_and_router_wrappers_are_exposed() -> None:
    import hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection as linear
    import hipengine.kernels.hip_gfx1100.moe.router as router

    assert callable(
        getattr(
            linear,
            "laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_signal_bf16",
            None,
        )
    )
    assert callable(
        getattr(
            router,
            "qwen35_router_logits_bf16_f32w_wave0_tree_anyorder",
            None,
        )
    )


def test_anyorder_runtime_is_gfx1151_available_and_default_on() -> None:
    import hipengine.kernels.hip_gfx1151 as gfx1151
    import hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection as linear
    import hipengine.kernels.hip_gfx1100.moe.router as router
    from hipengine.kernels.registry import KernelKey, is_registered
    from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession

    linear.register_laguna_f16_projection_kernels(replace=True)
    router.register_qwen35_router_kernels(replace=True)
    gfx1151.register_gfx1151_kernels(replace=True)
    assert gfx1151.LAGUNA_OUTPUT_ROUTER_ANYORDER_DECODE_AVAILABLE is True
    assert gfx1151.LAGUNA_OUTPUT_ROUTER_ANYORDER_DECODE is True
    assert "use_output_router_anyorder_decode" in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear+add+rmsnorm",
            "fp16_weight+gguf_f32_weight",
            "fixedk_nontemporal_signal_bf16_out",
        )
    )
    assert is_registered(
        KernelKey(
            "hip_gfx1151",
            "router_logits",
            "f32",
            "bf16_hidden_wave0_tree_anyorder",
        )
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_anyorder_output_router_chain_is_exact_and_reusable() -> None:
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        build_laguna_f16_projection,
        laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_signal_bf16,
        laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_bf16,
    )
    from hipengine.kernels.hip_gfx1100.moe.router import (
        build_qwen35_router,
        qwen35_router_logits_bf16_f32w_wave0_tree,
        qwen35_router_logits_bf16_f32w_wave0_tree_anyorder,
    )

    rng = np.random.default_rng(0xA11)
    hidden_size = 3072
    in_features = 6144
    experts = 256
    x = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=in_features).astype(np.float32)
    )
    residual = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=hidden_size).astype(np.float32)
    )
    output_weight = rng.normal(
        0.0, 0.01, size=(hidden_size, in_features)
    ).astype(np.float16)
    norm_weight = rng.normal(1.0, 0.05, size=hidden_size).astype(np.float32)
    router_weight = rng.normal(
        0.0, 0.02, size=(experts, hidden_size)
    ).astype(np.float32)

    runtime = get_hip_runtime()
    linear = build_laguna_f16_projection(
        load=True, require_cached=_require_cached_build()
    )
    router = build_qwen35_router(
        load=True, require_cached=_require_cached_build()
    )
    buffers = []

    def upload(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(
            buffer,
            host_array_ptr(np.ascontiguousarray(array)),
            array.nbytes,
            runtime=runtime,
        )
        return buffer

    def allocate(nbytes: int):
        buffer = malloc(nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    def download(buffer, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(
            host_array_ptr(out),
            buffer,
            out.nbytes,
            runtime=runtime,
        )
        return out

    try:
        dx = upload(x)
        dx_source = upload(x)
        doutput_weight = upload(output_weight)
        dresidual = upload(residual)
        dnorm_weight = upload(norm_weight)
        drouter_weight = upload(router_weight)
        control_projection = allocate(hidden_size * 2)
        control_norm = allocate(hidden_size * 2)
        control_residual = allocate(hidden_size * 2)
        control_logits = allocate(experts * 4)
        candidate_projection = allocate(hidden_size * 2)
        candidate_norm = allocate(hidden_size * 2)
        candidate_residual = allocate(hidden_size * 2)
        candidate_logits = allocate(experts * 4)
        control_counters = allocate(3 * 4)
        candidate_counters = allocate(3 * 4)
        runtime.memset(control_counters.ptr, 0, control_counters.nbytes)
        runtime.memset(candidate_counters.ptr, 0, candidate_counters.nbytes)

        laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_bf16(
            dx.ptr,
            doutput_weight.ptr,
            control_projection.ptr,
            dresidual.ptr,
            dnorm_weight.ptr,
            control_norm.ptr,
            control_residual.ptr,
            control_counters.ptr,
            1,
            in_features,
            hidden_size,
            1.0e-6,
            library=linear,
            runtime=runtime,
        )
        qwen35_router_logits_bf16_f32w_wave0_tree(
            control_norm.ptr,
            drouter_weight.ptr,
            control_logits.ptr,
            1,
            hidden_size,
            experts,
            library=router,
            runtime=runtime,
        )

        for _ in range(4):
            runtime.memset(dx.ptr, 0, dx.nbytes)
            runtime.memcpy_async(
                dx.ptr,
                dx_source.ptr,
                dx.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
            laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_signal_bf16(
                dx.ptr,
                doutput_weight.ptr,
                candidate_projection.ptr,
                dresidual.ptr,
                dnorm_weight.ptr,
                candidate_norm.ptr,
                candidate_residual.ptr,
                candidate_counters.ptr,
                1,
                in_features,
                hidden_size,
                1.0e-6,
                library=linear,
                runtime=runtime,
            )
            qwen35_router_logits_bf16_f32w_wave0_tree_anyorder(
                candidate_norm.ptr,
                drouter_weight.ptr,
                candidate_logits.ptr,
                candidate_counters.ptr,
                1,
                hidden_size,
                experts,
                library=router,
                runtime=runtime,
            )
        runtime.device_synchronize()

        for control, candidate, shape, dtype in (
            (
                control_projection,
                candidate_projection,
                (hidden_size,),
                np.uint16,
            ),
            (control_norm, candidate_norm, (hidden_size,), np.uint16),
            (
                control_residual,
                candidate_residual,
                (hidden_size,),
                np.uint16,
            ),
            (control_logits, candidate_logits, (experts,), np.float32),
        ):
            np.testing.assert_array_equal(
                download(candidate, shape, dtype),
                download(control, shape, dtype),
            )
        np.testing.assert_array_equal(
            download(candidate_counters, (3,), np.int32),
            np.zeros(3, dtype=np.int32),
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
