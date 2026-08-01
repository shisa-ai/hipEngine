"""WPF-H6Y exact IQ3 packed-prefix b32-load contract."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import re
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import memory_stats
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.laguna_moe import (
    laguna_moe_scratch_nbytes,
    resolve_laguna_moe_plan,
)
from tests._laguna_synthetic import make_laguna_info
from tests.test_gguf_iq3_active_expert_persistent import (
    HIP_AVAILABLE,
    _IN_FEATURES,
    _NUM_EXPERTS,
    _OUT_FEATURES,
    _make_iq3_weight,
    _run_h5j_or_h5q,
)
from tests.test_gguf_iq_gemv import (
    _bf16_u16_to_f32,
    _f32_to_bf16_u16,
    _make_x,
    _selected_reference,
)
from tests.test_laguna_h6p_iq3_staged_wave_publication import (
    _partition_boundary_metadata,
    _single_expert_metadata,
)

_H6Y_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_packed_prefix_b32_"
    "rowbatch8_bf16_bf16_out"
)
_H6Y_WRAPPER_NAME = "gguf_iq3_xxs_" + _H6Y_VARIANT
_H6Y_SYMBOL = "hipengine_" + _H6Y_WRAPPER_NAME
_H6Y_KERNEL_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_"
    "p64_activation_resident_output_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_packed_prefix_b32_"
    "rowbatch8_kernel"
)
_H6Y_LOAD_HELPER = "load_iq3_segment_packed_prefix_b32"
_H6Y_FP16_HELPER = "h6y_fp16_bits_to_float"
_H6T_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
_H6T_WRAPPER_NAME = "gguf_iq3_xxs_" + _H6T_VARIANT
_H6T_KERNEL_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_"
    "p64_activation_resident_output_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_kernel"
)
_H6T_PUBLISH_HELPER = (
    "publish_local128_wave_sums_batched_dpp_peer_fused_add_no_barrier"
)
_H6T_ADD_HELPER = "h6t_dpp_add_row_shl1_f32"
_H6R_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_dpp_peer_exchange_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6Q_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_compact_shuffle_loop_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6P_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_triple_output_rowbatch8_bf16_bf16_out"
)
_H6I_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6F_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_paired_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6D_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out"
)
_H5Z_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_rowbatch8_bf16_bf16_out"
)
_H5Q_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "resident_rowbatch8_bf16_bf16_out"
)
_H5J_IQ4_VARIANT = "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
_ACTIVE_EXPERT_ABI = "grouped_raw_iq_active_experts"
_PRODUCTION_MOE_SCRATCH_BYTES = 104_370_208
_IQ3_GRID_BYTES_SHA256 = (
    "46e35f5a997efdee6c99ce57854c8a0d4f0ff8ca57e5e8a60c0793ea580acf5d"
)
_H6T_LOAD_DECL_SHA256 = (
    "3587f01bf37e87c15aec767bf6f666114f9ac6f27d771c7db5d7e1b7c6268ca1"
)
_H6T_ADD_DECL_SHA256 = (
    "8ebcb4fb644a744450cccc683ba75f21729c255482d0142fd6a6b32ae41c31ce"
)
_H6T_PUBLISH_DECL_SHA256 = (
    "7e16097a9228bd023322686b5a371db0329d56ff88624575572c121db05d2ef0"
)
_H6T_KERNEL_DECL_SHA256 = (
    "36f941e39aa27767a35582d99a84dfb9d5f8d001626da9b4855f3f4d4ed618eb"
)
_H6T_PYTHON_WRAPPER_SHA256 = (
    "afcb504afababd006face7b600cdf0b22e4138828633179f6b33c133342492e8"
)
_SUM_HELPER = "sum_local128_wave_sums_serial"
_DOT_HELPER = "dot_iq3_segment_rowbatch8_interleaved"
_ACTUAL_LAYER_IDS = tuple(range(1, 46))
_PHYSICAL_CONTRACT = {
    "source_global_load_b128": 8,
    "source_global_load_b32": 9,
    "source_global_load_d16_b16": 6,
    "source_global_loads": 23,
    "candidate_global_load_b128": 8,
    "candidate_global_load_b32": 12,
    "candidate_global_load_d16_b16": 0,
    "candidate_global_loads": 20,
    "source_prefix_loads_per_three_scopes": 6,
    "candidate_prefix_loads_per_three_scopes": 3,
    "source_static_barriers": 2,
    "candidate_static_barriers": 2,
    "source_metadata_lds_bytes": 384,
    "source_runtime_lds_bytes": 512,
    "candidate_metadata_lds_bytes": 384,
    "candidate_runtime_lds_bytes_max": 512,
    "source_metadata_vgpr": 101,
    "source_runtime_vgpr": 104,
    "candidate_metadata_vgpr_max": 101,
    "candidate_runtime_vgpr_max": 104,
    "ds_load_b128": 24,
    "ds_store_2addr_b32": 12,
    "fma_operations": 216,
    "permlanex16": 24,
    "dpp_adds": 96,
    "local_size": 128,
    "grid_x": 32_768,
    "grid_y": 64,
    "private_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "runtime_scratch_bytes": 0,
}
_TRACE_CONTRACT = {
    "kernel_name": _H6Y_KERNEL_NAME,
    "require_cached_build": True,
    "new_compiler_processes": 0,
    "local_size": 128,
    "grid_x": 32_768,
    "grid_y": 64,
    "positive_duration": True,
}
_TIMING_CONTRACT = {
    "warmups": 5,
    "repetitions": 15,
    "launches_per_sample": 5,
    "order": "counter_rotated",
    "clocks": ("hip_event", "synchronized_wall"),
    "actual_layer_ids": _ACTUAL_LAYER_IDS,
    "require_every_layer_both_clocks": True,
    "require_aggregate_both_clocks": True,
    "one_shot_only": True,
}
_REJECTION_SURFACES = (
    "gfx1100 HIP helper/kernel/export",
    "gfx1100 Python symbol/wrapper/registry key",
    "gfx1151 explicit exclusion",
    "H6Y tests",
)
_REJECT_RULE = (
    "Any correctness, physical, resource, cached-trace, compiler, lifecycle, "
    "per-layer both-clock, or aggregate both-clock miss removes every H6Y "
    "surface without tuning or rerun."
)
_FINITE_FP16_BIT_CLASSES = {
    "positive_zero": 0x0000,
    "negative_zero": 0x8000,
    "positive_min_subnormal": 0x0001,
    "negative_min_subnormal": 0x8001,
    "positive_max_subnormal": 0x03FF,
    "negative_max_subnormal": 0x83FF,
    "positive_min_normal": 0x0400,
    "negative_min_normal": 0x8400,
    "positive_normal": 0x3C00,
    "negative_normal": 0xBC00,
    "positive_max_finite": 0x7BFF,
    "negative_max_finite": 0xFBFF,
}
_SAMPLE_COLS = np.asarray(
    [
        0,
        255,
        256,
        511,
        512,
        767,
        768,
        1023,
        1024,
        1279,
        1280,
        1535,
        1536,
        1791,
        1792,
        2047,
        2048,
        2303,
        2304,
        2559,
        2560,
        2815,
        2816,
        3071,
    ],
    dtype=np.int64,
)


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )


def _candidate():
    return getattr(_module(), _H6Y_WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq3_xxs", _H6Y_VARIANT)


def _declaration(source: str, anchor: str) -> str:
    start = source.index(anchor)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated declaration: {anchor}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _allocation_lifecycle() -> tuple[int, int]:
    stats = memory_stats()
    return stats["current_allocated_bytes"], stats["active_allocations"]


@pytest.fixture(scope="module")
def grouped_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return _module().build_gguf_iq_selected_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


@pytest.fixture(scope="module")
def iq3_weights() -> dict[int, np.ndarray]:
    return {1: _make_iq3_weight(1), 65: _make_iq3_weight(65)}


def test_h6y_frozen_physical_trace_timing_and_rejection_contract() -> None:
    assert _ACTUAL_LAYER_IDS == tuple(range(1, 46))
    assert _PHYSICAL_CONTRACT == {
        "source_global_load_b128": 8,
        "source_global_load_b32": 9,
        "source_global_load_d16_b16": 6,
        "source_global_loads": 23,
        "candidate_global_load_b128": 8,
        "candidate_global_load_b32": 12,
        "candidate_global_load_d16_b16": 0,
        "candidate_global_loads": 20,
        "source_prefix_loads_per_three_scopes": 6,
        "candidate_prefix_loads_per_three_scopes": 3,
        "source_static_barriers": 2,
        "candidate_static_barriers": 2,
        "source_metadata_lds_bytes": 384,
        "source_runtime_lds_bytes": 512,
        "candidate_metadata_lds_bytes": 384,
        "candidate_runtime_lds_bytes_max": 512,
        "source_metadata_vgpr": 101,
        "source_runtime_vgpr": 104,
        "candidate_metadata_vgpr_max": 101,
        "candidate_runtime_vgpr_max": 104,
        "ds_load_b128": 24,
        "ds_store_2addr_b32": 12,
        "fma_operations": 216,
        "permlanex16": 24,
        "dpp_adds": 96,
        "local_size": 128,
        "grid_x": 32_768,
        "grid_y": 64,
        "private_bytes": 0,
        "vgpr_spills": 0,
        "sgpr_spills": 0,
        "runtime_scratch_bytes": 0,
    }
    assert _TRACE_CONTRACT == {
        "kernel_name": _H6Y_KERNEL_NAME,
        "require_cached_build": True,
        "new_compiler_processes": 0,
        "local_size": 128,
        "grid_x": 32_768,
        "grid_y": 64,
        "positive_duration": True,
    }
    assert _TIMING_CONTRACT == {
        "warmups": 5,
        "repetitions": 15,
        "launches_per_sample": 5,
        "order": "counter_rotated",
        "clocks": ("hip_event", "synchronized_wall"),
        "actual_layer_ids": tuple(range(1, 46)),
        "require_every_layer_both_clocks": True,
        "require_aggregate_both_clocks": True,
        "one_shot_only": True,
    }
    assert len(_REJECTION_SURFACES) == 4
    assert "H6Y tests" in _REJECTION_SURFACES
    assert _REJECT_RULE.endswith("without tuning or rerun.")
    assert "every H6Y surface" in _REJECT_RULE
    assert "packed_prefix_b32" in _H6Y_VARIANT
    assert all(term not in _H6Y_VARIANT for term in ("prompt", "token", "layer"))

    bits = np.asarray(list(_FINITE_FP16_BIT_CLASSES.values()), dtype="<u2")
    values = bits.view("<f2")
    assert np.isfinite(values).all()
    assert set(_FINITE_FP16_BIT_CLASSES) == {
        "positive_zero",
        "negative_zero",
        "positive_min_subnormal",
        "negative_min_subnormal",
        "positive_max_subnormal",
        "negative_max_subnormal",
        "positive_min_normal",
        "negative_min_normal",
        "positive_normal",
        "negative_normal",
        "positive_max_finite",
        "negative_max_finite",
    }
    assert not np.signbit(values[0]) and np.signbit(values[1])
    assert np.all(np.abs(values[2:6]) < np.float16(2**-14))
    assert np.all(np.abs(values[6:8]) == np.float16(2**-14))
    assert np.all(np.abs(values[10:12]) == np.finfo(np.float16).max)
    selector0 = np.arange(256, dtype=np.uint32)
    selector1 = np.arange(255, -1, -1, dtype=np.uint32)
    prefixes = (
        bits[np.arange(256) % bits.size].astype(np.uint32)
        | (selector0 << np.uint32(16))
        | (selector1 << np.uint32(24))
    )
    np.testing.assert_array_equal(
        (prefixes & np.uint32(0xFFFF)).astype(np.uint16),
        bits[np.arange(256) % bits.size],
    )
    np.testing.assert_array_equal(
        ((prefixes >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8),
        selector0.astype(np.uint8),
    )
    np.testing.assert_array_equal(
        (prefixes >> np.uint32(24)).astype(np.uint8), selector1.astype(np.uint8)
    )


def test_h6y_registry_table_source_policy_and_h6t_immutability() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    expected_variants = {
        "gguf_iq3_xxs": _H6T_VARIANT,
        "gguf_iq4_xs": _H5J_IQ4_VARIANT,
    }
    expected_abis = {
        _H5Q_VARIANT: _ACTIVE_EXPERT_ABI,
        _H5Z_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6D_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6F_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6I_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6P_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6Q_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6R_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6T_VARIANT: _ACTIVE_EXPERT_ABI,
    }
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == expected_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == expected_abis

    config = laguna_gguf_config_from_metadata(make_laguna_info())
    production = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert production.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _H6T_VARIANT
    )
    assert production.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        _ACTIVE_EXPERT_ABI
    )
    assert (
        laguna_moe_scratch_nbytes(production, max_rows=512)
        == _PRODUCTION_MOE_SCRATCH_BYTES
    )

    module = _module()
    source = Path(module.__file__).with_suffix(".hip").read_text()
    gemv_source = Path(module.__file__).with_name("gguf_iq_gemv.hip").read_text()
    initializer = gemv_source.split("IQ3_XXS_GRID[256] = {", 1)[1].split(
        "};", 1
    )[0]
    table = np.asarray(
        [
            int(value, 16)
            for value in re.findall(r"0x([0-9a-fA-F]{8})u", initializer)
        ],
        dtype="<u4",
    )
    assert table.size == 256
    assert table.nbytes == 1_024
    assert hashlib.sha256(table.tobytes()).hexdigest() == _IQ3_GRID_BYTES_SHA256

    h6t_load = _declaration(source, "__device__ inline IQ3Segment load_iq3_segment(")
    h6t_add = _declaration(
        source, f"__device__ inline float {_H6T_ADD_HELPER}("
    )
    h6t_publish = _declaration(
        source,
        f"template <int ROW_BATCH>\n__device__ inline void {_H6T_PUBLISH_HELPER}(",
    )
    h6t_kernel = _declaration(source, f"__global__ void {_H6T_KERNEL_NAME}(")
    assert _sha256(h6t_load) == _H6T_LOAD_DECL_SHA256
    assert _sha256(h6t_add) == _H6T_ADD_DECL_SHA256
    assert _sha256(h6t_publish) == _H6T_PUBLISH_DECL_SHA256
    assert _sha256(h6t_kernel) == _H6T_KERNEL_DECL_SHA256
    h6t_wrapper = getattr(module, _H6T_WRAPPER_NAME)
    assert _sha256(inspect.getsource(h6t_wrapper)) == _H6T_PYTHON_WRAPPER_SHA256
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H6T_VARIANT,
    ) is h6t_wrapper

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    # Intentional RED: retained source/table/policy facts pass before the only
    # missing boundary, the separately named H6Y Python wrapper.
    candidate = _candidate()
    assert candidate.__name__ == _H6Y_WRAPPER_NAME
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H6Y_VARIANT,
    ) is candidate
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == expected_variants
    # Standalone H6Y is registry-only; runtime/source promotion stays separate.
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == expected_abis
    assert source.count(_H6Y_SYMBOL) == 1
    assert source.count(_H6Y_KERNEL_NAME) == 2
    assert source.count(f"__device__ inline IQ3Segment {_H6Y_LOAD_HELPER}(") == 1
    assert source.count(f"{_H6Y_LOAD_HELPER}(") == 4
    assert source.count(f"__device__ inline float {_H6Y_FP16_HELPER}(") == 1
    assert source.count(f"{_H6Y_FP16_HELPER}(") == 2

    candidate_body = _declaration(source, f"__global__ void {_H6Y_KERNEL_NAME}(")
    assert "constexpr int expert_partitions = 64;" in candidate_body
    assert "constexpr int row_batch = 8;" in candidate_body
    assert "__shared__ uint32_t" not in candidate_body
    assert candidate_body.count("__syncthreads();") == 2
    assert candidate_body.count(_H6Y_LOAD_HELPER) == 3
    assert candidate_body.count(_H6T_PUBLISH_HELPER) == 3
    assert candidate_body.count(_SUM_HELPER) == 3
    assert candidate_body.count(_DOT_HELPER) == 3
    assert candidate_body.count("float acc[row_batch] = {};") == 3
    assert "active_index += expert_partitions" in candidate_body
    assert "out_col += 3 * OUTPUT_PARTITIONS" in candidate_body
    assert "float_to_bf16_bits(value_a)" in candidate_body
    assert "float_to_bf16_bits(value_b)" in candidate_body
    assert "float_to_bf16_bits(value_c)" in candidate_body

    candidate_load = _declaration(
        source, f"__device__ inline IQ3Segment {_H6Y_LOAD_HELPER}("
    )
    expected_unpack = (
        "const int selector_offset = group32 * 8 + 2 * local8;",
        "const uint32_t packed_prefix = load_u32_le(block + selector_offset);",
        "const uint32_t block_prefix = __shfl(packed_prefix, 0);",
        "const uint16_t scale_bits = static_cast<uint16_t>(block_prefix);",
        "const uint32_t selector_pair = packed_prefix >> 16;",
        "IQ3_XXS_GRID[selector_pair & 255U]",
        "IQ3_XXS_GRID[(selector_pair >> 8) & 255U]",
        f"{_H6Y_FP16_HELPER}(scale_bits)",
    )
    assert all(step in candidate_load for step in expected_unpack)
    assert "const uint32_t aux = load_u32_le" in candidate_load
    assert "const uint32_t signs = iq3_sign_byte" in candidate_load
    assert "fp16_bytes_to_float" not in candidate_load
    assert candidate_load.count("IQ3_XXS_GRID[") == 2
    assert candidate_load.count("segment.magnitude[") == 2

    fp16_unpack = _declaration(
        source, f"__device__ inline float {_H6Y_FP16_HELPER}("
    )
    assert "uint16_t u16;" in fp16_unpack
    assert "half_t f16;" in fp16_unpack
    assert "value.u16 = bits;" in fp16_unpack
    assert "return static_cast<float>(value.f16);" in fp16_unpack

    wrapper = inspect.getsource(candidate)
    assert _H6Y_SYMBOL in wrapper
    gfx1151_source = inspect.getsource(hip_gfx1151)
    assert gfx1151_source.count(_H6Y_VARIANT) == 1


def test_h6y_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    control = getattr(module, _H6T_WRAPPER_NAME)
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid IQ3 packed-prefix shape reached the HIP loader")

    monkeypatch.setattr(module, "build_gguf_iq_selected_prefill", fail_if_loaded)
    common = dict(
        compact_rows=9,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=_NUM_EXPERTS,
    )
    invalid = (
        ({"compact_rows": 0}, "compact_rows must be positive"),
        ({"in_features": 768}, "exactly 1024"),
        ({"out_features": 1024}, "exactly 3072"),
        ({"num_experts": 255}, "exactly 256"),
    )
    for changed, message in invalid:
        with pytest.raises(ValueError, match=message):
            control(1, 2, 3, 4, 5, 6, **(common | changed))
    assert load_attempts == 0

    # Intentional RED only after retained H6T preflight is proven.
    candidate = _candidate()
    for changed, message in invalid:
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, 4, 5, 6, **(common | changed))
    assert load_attempts == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("case", "value"),
    [
        pytest.param("rows", 1, id="rows1"),
        pytest.param("rows", 7, id="rows7"),
        pytest.param("rows", 8, id="rows8"),
        pytest.param("rows", 9, id="rows9"),
        pytest.param("rows", 512, id="rows512"),
        pytest.param("partitions", 64, id="reversed-p64"),
        pytest.param("partitions", 65, id="reversed-p65"),
    ],
)
def test_h6y_complete_outputs_match_h6t_and_cpu_at_staged_boundaries(
    grouped_library,
    iq3_weights: dict[int, np.ndarray],
    case: str,
    value: int,
) -> None:
    module = _module()
    control = getattr(module, _H6T_WRAPPER_NAME)
    if case == "rows":
        starts, active, active_count, selected = _single_expert_metadata(value)
        qweight = iq3_weights[1]
    else:
        starts, active, active_count, selected = _partition_boundary_metadata(value)
        qweight = iq3_weights[65]
    compact_rows = int(starts[-1])
    x_bf16 = _f32_to_bf16_u16(_make_x(compact_rows, _IN_FEATURES))
    initial = np.full((compact_rows, _OUT_FEATURES), 0x7FC0, dtype=np.uint16)
    baseline = _allocation_lifecycle()

    expected = _run_h5j_or_h5q(
        control,
        grouped_library,
        x_bf16=x_bf16,
        starts=starts,
        active_experts=active,
        active_count=active_count,
        qweight=qweight,
        initial=initial,
        persistent=True,
    )
    assert _allocation_lifecycle() == baseline
    assert not np.any(expected == np.uint16(0x7FC0))
    sample_rows = np.unique(
        np.asarray([0, 1, 7, 8, compact_rows // 2, compact_rows - 1]).clip(
            0, compact_rows - 1
        )
    )
    cpu = _selected_reference(
        x_bf16[sample_rows],
        selected[sample_rows],
        qweight[:, _SAMPLE_COLS, :],
        GGMLQuantizationType.IQ3_XXS,
    )
    np.testing.assert_array_equal(expected[np.ix_(sample_rows, _SAMPLE_COLS)], cpu)
    assert np.isfinite(_bf16_u16_to_f32(expected)).all()

    # Intentional RED only after complete H6T/CPU bytes and lifecycle pass.
    candidate = _candidate()
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
    assert _allocation_lifecycle() == baseline
    np.testing.assert_array_equal(actual, expected, err_msg=f"{case}={value}")
    assert not np.any(actual == np.uint16(0x7FC0))
    assert np.isfinite(_bf16_u16_to_f32(actual)).all()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h6y_all_finite_fp16_bit_classes_and_selector_bytes_match_h6t(
    grouped_library,
) -> None:
    module = _module()
    control = getattr(module, _H6T_WRAPPER_NAME)
    qweight = _make_iq3_weight(1)
    scale_bits = np.asarray(list(_FINITE_FP16_BIT_CLASSES.values()), dtype="<u2")
    output_rows = np.arange(_OUT_FEATURES, dtype=np.int64)
    selector_values: set[int] = set()
    for block_index in range(_IN_FEATURES // 256):
        block_start = block_index * 98
        row_scale_bits = scale_bits[(output_rows + block_index) % scale_bits.size]
        qweight[0, :, block_start : block_start + 2] = row_scale_bits.reshape(
            -1, 1
        ).view(np.uint8).reshape(_OUT_FEATURES, 2)
        selectors = (
            output_rows.reshape(-1, 1) * 64
            + np.arange(64, dtype=np.int64).reshape(1, -1)
            + 17 * block_index
        ).astype(np.uint8)
        qweight[0, :, block_start + 2 : block_start + 66] = selectors
        selector_values.update(int(value) for value in np.unique(selectors))
    assert selector_values == set(range(256))
    recovered_scale_bits = np.concatenate(
        [
            qweight[0, :, block_index * 98 : block_index * 98 + 2]
            .copy()
            .reshape(-1, 2)
            .view("<u2")
            .reshape(-1)
            for block_index in range(_IN_FEATURES // 256)
        ]
    )
    assert set(int(value) for value in recovered_scale_bits) == set(
        _FINITE_FP16_BIT_CLASSES.values()
    )

    starts, active, active_count, selected = _single_expert_metadata(1)
    x_bf16 = _f32_to_bf16_u16(_make_x(1, _IN_FEATURES))
    initial = np.full((1, _OUT_FEATURES), 0x7FC0, dtype=np.uint16)
    baseline = _allocation_lifecycle()
    expected = _run_h5j_or_h5q(
        control,
        grouped_library,
        x_bf16=x_bf16,
        starts=starts,
        active_experts=active,
        active_count=active_count,
        qweight=qweight,
        initial=initial,
        persistent=True,
    )
    assert _allocation_lifecycle() == baseline
    assert not np.any(expected == np.uint16(0x7FC0))
    assert np.isfinite(_bf16_u16_to_f32(expected)).all()
    cpu = _selected_reference(
        x_bf16,
        selected,
        qweight[:, _SAMPLE_COLS, :],
        GGMLQuantizationType.IQ3_XXS,
    )
    np.testing.assert_array_equal(expected[:, _SAMPLE_COLS], cpu)

    # Intentional RED only after all finite FP16 classes, all selector bytes,
    # complete H6T output, sampled CPU bytes, and lifecycle are proven.
    candidate = _candidate()
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
    assert _allocation_lifecycle() == baseline
    np.testing.assert_array_equal(actual, expected)
    assert not np.any(actual == np.uint16(0x7FC0))
    assert np.isfinite(_bf16_u16_to_f32(actual)).all()
