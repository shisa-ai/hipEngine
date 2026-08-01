"""WPF-H6C exact special-IQ3 expert-major fused-SiLU contract."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import (
    build_gguf_iq_gemv,
    gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_gguf_iq_gemv import (
    HIP_AVAILABLE,
    IQ3_XXS_BLOCK_BYTES,
    _bf16_u16_to_f32,
    _f32_to_bf16_u16,
    _make_x,
    _selected_reference,
)

_IN_FEATURES = 3072
_OUT_FEATURES = 1024
_NUM_EXPERTS = 256
_QK_K = 256
_ROW_BATCH = 4
_WRAPPER_NAME = (
    "gguf_iq3_xxs_selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
_SYMBOL = f"hipengine_{_WRAPPER_NAME}"
_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
_H5Z_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_rowbatch8_bf16_bf16_out"
)
_H6D_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out"
)
_H6F_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_paired_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6I_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6P_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_triple_output_rowbatch8_bf16_bf16_out"
)
_H5Q_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "resident_rowbatch8_bf16_bf16_out"
)
_H5J_IQ4_VARIANT = (
    "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
)
_H6Q_RUNTIME_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_compact_shuffle_loop_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6R_RUNTIME_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_dpp_peer_exchange_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6T_RUNTIME_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
_ACTIVE_EXPERT_ABI = "grouped_raw_iq_active_experts"


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )


def _candidate():
    return getattr(_module(), _WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq3_xxs", _VARIANT)


@pytest.fixture(scope="module")
def libraries():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    require_cached = os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"
    return (
        _module().build_gguf_iq_selected_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
        build_gguf_iq_gemv(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
    )


def test_h6c_registry_source_scope_and_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    assert hip_gfx1100.LAGUNA_SELECTED_GATE_UP_MODE == "grouped_pair16"
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {
        "gguf_iq3_xxs": _H6R_RUNTIME_VARIANT,
        "gguf_iq4_xs": _H5J_IQ4_VARIANT,
    }
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {
        _H5Q_VARIANT: _ACTIVE_EXPERT_ABI,
        _H5Z_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6D_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6F_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6I_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6P_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6Q_RUNTIME_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6R_RUNTIME_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6T_RUNTIME_VARIANT: _ACTIVE_EXPERT_ABI,
    }

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    candidate = _candidate()
    assert candidate.__name__ == _WRAPPER_NAME
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_VARIANT,
    ) is candidate

    source = Path(_module().__file__).with_suffix(".hip").read_text()
    assert source.count(_SYMBOL) == 1
    assert (
        "gguf_iq3_xxs_selected_dual_grouped_prefill_compact_"
        "rowbatch_kernel<4, true>" in source
    )


def test_h6c_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    candidate = _candidate()
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H6C shape reached the HIP loader")

    monkeypatch.setattr(module, "build_gguf_iq_selected_prefill", fail_if_loaded)
    common = dict(
        compact_rows=17,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=_NUM_EXPERTS,
    )
    for changed, message in (
        ({"compact_rows": 0}, "compact_rows must be positive"),
        ({"in_features": 2816}, "exactly 3072"),
        ({"out_features": 1023}, "exactly 1024"),
        ({"num_experts": 255}, "exactly 256"),
    ):
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, 4, 5, **(common | changed))
    assert load_attempts == 0


def _make_full_iq3_weight(seed: int) -> np.ndarray:
    """Build a deterministic full E256/N1024/K3072 IQ3 fixture efficiently."""

    blocks = _IN_FEATURES // _QK_K
    rng = np.random.default_rng(seed)
    template = rng.integers(
        0,
        256,
        size=(blocks, IQ3_XXS_BLOCK_BYTES),
        dtype=np.uint8,
    )
    weight = np.empty(
        (_NUM_EXPERTS, _OUT_FEATURES, blocks, IQ3_XXS_BLOCK_BYTES),
        dtype=np.uint8,
    )
    weight[...] = template

    for block in range(blocks):
        scale = np.asarray(
            [np.float16(0.0009765625 * (1 + block % 5))],
            dtype=np.float16,
        ).view(np.uint8)
        weight[:, :, block, :2] = scale
    for residue in range(16):
        weight[:, residue::16, :, 2:] ^= np.uint8((29 * residue + seed) & 0xFF)
    for expert in (0, 1, 3, 7, 63, 64, 127, 128, 254, 255):
        weight[expert, :, :, 2:] ^= np.uint8((37 * expert + seed) & 0xFF)
    return weight.reshape(_NUM_EXPERTS, _OUT_FEATURES, -1)


@pytest.fixture(scope="module")
def full_weights() -> tuple[np.ndarray, np.ndarray]:
    return _make_full_iq3_weight(17), _make_full_iq3_weight(91)


def _reordered_tail_route() -> np.ndarray:
    counts = {
        0: 1,
        1: 3,
        3: 4,
        7: 5,
        63: 7,
        64: 8,
        127: 9,
        128: 2,
        254: 6,
        255: 11,
    }
    sorted_route = np.concatenate(
        [np.full(count, expert, dtype=np.int64) for expert, count in counts.items()]
    )
    return np.random.default_rng(0x6C).permutation(sorted_route)


def _compact_metadata(
    route_experts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(route_experts, kind="stable")
    sorted_experts = route_experts[order]
    counts = np.bincount(sorted_experts, minlength=_NUM_EXPERTS)
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts, dtype=np.int64)
    return order, sorted_experts, starts


def _device_buffer(array: np.ndarray, buffers: list[Any], runtime):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(
        buffer,
        host_array_ptr(contiguous),
        contiguous.nbytes,
        runtime=runtime,
    )
    buffers.append(buffer)
    return buffer


def _run_control_then_h6c(
    grouped_library,
    direct_library,
    *,
    route_x: np.ndarray,
    route_experts: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    order, sorted_experts, starts = _compact_metadata(route_experts)
    compact_x = np.ascontiguousarray(route_x[order], dtype=np.uint16)
    route_expected = np.full(
        (route_experts.size, _OUT_FEATURES), 0x7FC0, dtype=np.uint16
    )
    actual = np.full_like(route_expected, 0x7FC0)
    before = memory_stats()
    buffers: list[Any] = []
    try:
        route_x_device = _device_buffer(route_x, buffers, runtime)
        compact_x_device = _device_buffer(compact_x, buffers, runtime)
        route_experts_device = _device_buffer(route_experts, buffers, runtime)
        starts_device = _device_buffer(starts, buffers, runtime)
        gate_device = _device_buffer(gate, buffers, runtime)
        up_device = _device_buffer(up, buffers, runtime)
        expected_device = _device_buffer(route_expected, buffers, runtime)
        actual_device = _device_buffer(actual, buffers, runtime)

        gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out(
            route_x_device.ptr,
            route_experts_device.ptr,
            gate_device.ptr,
            up_device.ptr,
            expected_device.ptr,
            x_rows=route_experts.size,
            rows=route_experts.size,
            num_experts=_NUM_EXPERTS,
            in_features=_IN_FEATURES,
            out_features=_OUT_FEATURES,
            threads=256,
            library=direct_library,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(route_expected),
            expected_device,
            route_expected.nbytes,
            runtime=runtime,
        )

        candidate = _candidate()
        candidate(
            compact_x_device.ptr,
            starts_device.ptr,
            gate_device.ptr,
            up_device.ptr,
            actual_device.ptr,
            compact_rows=route_experts.size,
            in_features=_IN_FEATURES,
            out_features=_OUT_FEATURES,
            num_experts=_NUM_EXPERTS,
            library=grouped_library,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual),
            actual_device,
            actual.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        after = memory_stats()
        assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
        assert after["active_allocations"] == before["active_allocations"]
    return route_expected[order], actual, compact_x, sorted_experts


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h6c_complete_reordered_rowbatch4_tails_match_route_major_and_cpu(
    libraries,
    full_weights: tuple[np.ndarray, np.ndarray],
) -> None:
    grouped_library, direct_library = libraries
    gate, up = full_weights
    route_experts = _reordered_tail_route()
    route_x = _f32_to_bf16_u16(_make_x(route_experts.size, _IN_FEATURES))
    expected, actual, compact_x, sorted_experts = _run_control_then_h6c(
        grouped_library,
        direct_library,
        route_x=route_x,
        route_experts=route_experts,
        gate=gate,
        up=up,
    )
    np.testing.assert_array_equal(actual, expected)

    sample_rows = np.unique(
        np.asarray([0, 1, 3, 4, 5, 7, 8, actual.shape[0] // 2, actual.shape[0] - 1])
    )
    sample_cols = np.asarray([0, 1, 255, 256, 511, 1023])
    gate_cpu = _selected_reference(
        compact_x[sample_rows],
        sorted_experts[sample_rows],
        gate[:, sample_cols, :],
        GGMLQuantizationType.IQ3_XXS,
    )
    up_cpu = _selected_reference(
        compact_x[sample_rows],
        sorted_experts[sample_rows],
        up[:, sample_cols, :],
        GGMLQuantizationType.IQ3_XXS,
    )
    gate_f32 = _bf16_u16_to_f32(gate_cpu)
    up_f32 = _bf16_u16_to_f32(up_cpu)
    cpu = _f32_to_bf16_u16(
        gate_f32
        * (
            np.float32(1.0)
            / (np.float32(1.0) + np.exp(-gate_f32).astype(np.float32))
        )
        * up_f32
    )
    sampled = actual[np.ix_(sample_rows, sample_cols)]
    cpu_f32 = _bf16_u16_to_f32(cpu)
    sampled_f32 = _bf16_u16_to_f32(sampled)
    max_rel = float(
        np.max(np.abs(sampled_f32 - cpu_f32) / np.maximum(np.abs(cpu_f32), 1.0))
    )
    assert max_rel <= 0.05
    assert evaluate_logits(cpu_f32, sampled_f32).passed


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h6c_empty_compact_route_preserves_output_and_recovers_allocations(
    libraries,
) -> None:
    from hipengine.core.hip import get_hip_runtime

    grouped_library, _ = libraries
    module = _module()
    runtime = get_hip_runtime()
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    x = _f32_to_bf16_u16(_make_x(1, _IN_FEATURES))
    tiny_weight = np.zeros(1, dtype=np.uint8)
    control = np.full((1, 2 * _OUT_FEATURES), 0x3F80, dtype=np.uint16)
    actual = np.full((1, _OUT_FEATURES), 0x3F80, dtype=np.uint16)
    before = memory_stats()
    buffers: list[Any] = []
    try:
        x_device = _device_buffer(x, buffers, runtime)
        starts_device = _device_buffer(starts, buffers, runtime)
        gate_device = _device_buffer(tiny_weight, buffers, runtime)
        up_device = _device_buffer(tiny_weight, buffers, runtime)
        control_device = _device_buffer(control, buffers, runtime)
        actual_device = _device_buffer(actual, buffers, runtime)

        module.gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
            x_device.ptr,
            starts_device.ptr,
            gate_device.ptr,
            up_device.ptr,
            control_device.ptr,
            compact_rows=1,
            in_features=_IN_FEATURES,
            out_features=_OUT_FEATURES,
            num_experts=_NUM_EXPERTS,
            library=grouped_library,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(control),
            control_device,
            control.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(control, np.full_like(control, 0x3F80))

        candidate = _candidate()
        candidate(
            x_device.ptr,
            starts_device.ptr,
            gate_device.ptr,
            up_device.ptr,
            actual_device.ptr,
            compact_rows=1,
            in_features=_IN_FEATURES,
            out_features=_OUT_FEATURES,
            num_experts=_NUM_EXPERTS,
            library=grouped_library,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual),
            actual_device,
            actual.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(actual, np.full_like(actual, 0x3F80))
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        after = memory_stats()
        assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
        assert after["active_allocations"] == before["active_allocations"]
