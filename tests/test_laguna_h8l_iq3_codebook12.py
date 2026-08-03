"""RED contracts for WPF-H8L exact IQ3 lossless codebook packing."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import memory_stats
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.quant.gguf import GGMLQuantizationType, _IQ3_XXS_GRID
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

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-03-gfx1100-laguna-q2-xl-post-h8k-iq3-codebook12-target.json"
)
_TARGET_SHA256 = "4573d12cace03da5b3d70370018924ebde548aea52dbda3bcbe3b9d85efe340d"
_H6T_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
_H8L_VARIANT = _H6T_VARIANT.replace(
    "rowbatch8_bf16",
    "rowbatch8_packed_codebook12_bf16",
)
_H6T_WRAPPER_NAME = "gguf_iq3_xxs_" + _H6T_VARIANT
_H8L_WRAPPER_NAME = "gguf_iq3_xxs_" + _H8L_VARIANT
_H6T_SYMBOL = "hipengine_" + _H6T_WRAPPER_NAME
_H8L_SYMBOL = "hipengine_" + _H8L_WRAPPER_NAME
_H6T_PY_SYMBOL = (
    "_SYMBOL_IQ3_ACTIVATION_RESIDENT_FUSED_DPP_ADD_STAGED_TRIPLE_OUTPUT"
)
_H8L_PY_SYMBOL = _H6T_PY_SYMBOL + "_PACKED_CODEBOOK12"
_H6T_KERNEL = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_"
    "p64_activation_resident_output_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_kernel"
)
_H8L_KERNEL = _H6T_KERNEL.replace(
    "rowbatch8_kernel",
    "rowbatch8_packed_codebook12_kernel",
)
_RAW_TABLE = "IQ3_XXS_GRID"
_PACKED_TABLE = "IQ3_XXS_GRID_PACKED_CODEBOOK12"
_H6T_LOAD = "load_iq3_segment"
_H8L_LOAD = "load_iq3_segment_packed_codebook12"
_H6T_DOT = "dot_iq3_segment_rowbatch8_interleaved"
_H6T_PUBLISH = "publish_local128_wave_sums_batched_dpp_peer_fused_add_no_barrier"
_H6T_ADD = "h6t_dpp_add_row_shl1_f32"
_SUM_HELPER = "sum_local128_wave_sums_serial"
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
_RAW_TABLE_SHA256 = "bc20f469f077bf27bd0bc9f607245cd0b0c41636d8573e4a9c7730a3d17ecf9f"
_H6T_LOAD_SHA256 = "3587f01bf37e87c15aec767bf6f666114f9ac6f27d771c7db5d7e1b7c6268ca1"
_H6T_DOT_SHA256 = "8edfb174a36747da4506cbe62d3fe6daebce942cb63396749b02e0a9c997dd3b"
_H6T_ADD_SHA256 = "8ebcb4fb644a744450cccc683ba75f21729c255482d0142fd6a6b32ae41c31ce"
_H6T_PUBLISH_SHA256 = "7e16097a9228bd023322686b5a371db0329d56ff88624575572c121db05d2ef0"
_H6T_KERNEL_SHA256 = "36f941e39aa27767a35582d99a84dfb9d5f8d001626da9b4855f3f4d4ed618eb"
_H6T_EXPORT_SHA256 = "e10a77579ac6705a3cc5ce3207ccbbe126394a1526bcbfceb8308dbbb99cc904"
_H6T_WRAPPER_SHA256 = "afcb504afababd006face7b600cdf0b22e4138828633179f6b33c133342492e8"
_EXPECTED_MAGNITUDES = (4, 12, 20, 28, 36, 44, 52, 62)
_EXPECTED_RAW_TABLE_SHA256 = (
    "46e35f5a997efdee6c99ce57854c8a0d4f0ff8ca57e5e8a60c0793ea580acf5d"
)
_EXPECTED_PACKED_TABLE_SHA256 = (
    "c9678803c61b2210fbd8cbbb3f965a835409cd7d88a04d086d236261bd221d11"
)
_ACTUAL_LAYER_IDS = tuple(range(1, 46))
_PHYSICAL_CONTRACT = {
    "launch_bounds": (128, 1),
    "local_size": 128,
    "runtime_grid_x": 32_768,
    "runtime_grid_y": 64,
    "metadata_lds_bytes_max": 384,
    "runtime_lds_bytes_max": 512,
    "metadata_vgpr_max": 128,
    "runtime_vgpr_max": 128,
    "code_bytes_max": 10_000,
    "instruction_slots_max": 1_800,
    "private_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "runtime_scratch_bytes": 0,
    "global_loads": 23,
    "global_load_b128": 8,
    "global_load_b32": 3,
    "global_load_d16": 12,
    "useful_fmas": 216,
    "permlanex16": 24,
    "dpp_adds": 96,
    "lds_load_b128": 24,
    "lds_stores": 12,
    "barriers": 2,
}
_TRACE_CONTRACT = {
    "candidate_calls": 45,
    "unchanged_iq4_calls": 2,
    "request_dispatches": 2_155,
    "queues": 1,
    "streams": 1,
    "compiler_processes": 0,
    "local_size": 128,
    "runtime_grid_x": 32_768,
    "runtime_grid_y": 64,
    "positive_duration": True,
}
_TIMING_CONTRACT = {
    "warmups": 5,
    "samples": 15,
    "launch_repeats": 5,
    "order": "counter_rotated",
    "required_clocks": ("hip_event", "synchronized_wall"),
    "actual_layer_ids": _ACTUAL_LAYER_IDS,
    "require_every_layer_both_clocks": True,
    "require_aggregate_both_clocks": True,
    "allow_layer_subset": False,
    "allow_table_width_sweep": False,
    "allow_decode_formula_change": False,
    "allow_load_source_change": False,
    "allow_body_rewrite": False,
    "allow_recompile": False,
    "allow_favorable_rerun": False,
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
    return getattr(_module(), _H8L_WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq3_xxs", _H8L_VARIANT)


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack_codebook() -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(_IQ3_XXS_GRID, dtype="<u4").view(np.uint8).reshape(256, 4)
    assert tuple(sorted(int(value) for value in np.unique(raw))) == (
        _EXPECTED_MAGNITUDES
    )
    value_to_code = {value: code for code, value in enumerate(_EXPECTED_MAGNITUDES)}
    packed = np.zeros(256, dtype="<u2")
    reconstructed = np.zeros_like(raw)
    for index, row in enumerate(raw):
        for coordinate, raw_value in enumerate(row):
            code = value_to_code[int(raw_value)]
            packed[index] |= np.uint16(code << (3 * coordinate))
            decoded = (int(packed[index]) >> (3 * coordinate)) & 7
            reconstructed[index, coordinate] = 4 + 8 * decoded + 2 * (decoded == 7)
    np.testing.assert_array_equal(reconstructed, raw)
    assert len(set(int(value) for value in packed)) == 256
    return raw, packed


def _expected_packed_load(h6t_load: str) -> str:
    old_grids = (
        "  const uint32_t grid1 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8]];\n"
        "  const uint32_t grid2 = "
        "IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8 + 1]];"
    )
    new_grids = (
        "  const uint16_t grid1 = "
        "IQ3_XXS_GRID_PACKED_CODEBOOK12[qs[group32 * 8 + 2 * local8]];\n"
        "  const uint16_t grid2 = "
        "IQ3_XXS_GRID_PACKED_CODEBOOK12[qs[group32 * 8 + 2 * local8 + 1]];"
    )
    old_decode = (
        "    const float magnitude1 = "
        "static_cast<float>((grid1 >> (8 * j)) & 255U);\n"
        "    const float magnitude2 = "
        "static_cast<float>((grid2 >> (8 * j)) & 255U);"
    )
    new_decode = (
        "    const uint32_t code1 = "
        "(static_cast<uint32_t>(grid1) >> (3 * j)) & 7U;\n"
        "    const uint32_t code2 = "
        "(static_cast<uint32_t>(grid2) >> (3 * j)) & 7U;\n"
        "    const float magnitude1 = static_cast<float>(\n"
        "        4U + 8U * code1 + 2U * (code1 == 7U));\n"
        "    const float magnitude2 = static_cast<float>(\n"
        "        4U + 8U * code2 + 2U * (code2 == 7U));"
    )
    assert old_grids in h6t_load
    assert old_decode in h6t_load
    return (
        h6t_load.replace(_H6T_LOAD, _H8L_LOAD, 1)
        .replace(old_grids, new_grids)
        .replace(old_decode, new_decode)
    )


def _table_values(declaration: str) -> tuple[int, ...]:
    return tuple(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]+)u", declaration))


def _allocation_lifecycle() -> tuple[int, int]:
    stats = memory_stats()
    return stats["current_allocated_bytes"], stats["active_allocations"]


def _uneven_reordered_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_experts = (7, 0, 11, 3, 5)
    counts_by_expert = {0: 1, 3: 3, 5: 4, 7: 5, 11: 9}
    counts = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    for expert, count in counts_by_expert.items():
        counts[expert] = count
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active[: len(active_experts)] = active_experts
    active_count = np.asarray([len(active_experts)], dtype=np.int64)
    selected = np.repeat(np.arange(_NUM_EXPERTS, dtype=np.int64), counts)
    return starts, active, active_count, selected


def _empty_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(_NUM_EXPERTS + 1, dtype=np.int64),
        np.zeros(_NUM_EXPERTS, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
    )


def _load_grouped_library():
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return _module().build_gguf_iq_selected_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


def test_h8l_target_codebook_physical_trace_and_timing_contract() -> None:
    assert _sha256_file(_TARGET) == _TARGET_SHA256
    artifact = json.loads(_TARGET.read_text(encoding="utf-8"))
    assert artifact["status"] == "target_selected_no_candidate_no_speed_result"
    assert artifact["kind"] == "gfx1100_laguna_q2_xl_post_h8k_iq3_codebook12_target"
    assert artifact["performance_claim"] is False
    assert artifact["decision"] == {
        "candidate_implemented": False,
        "next_action": (
            "Commit this target-only packet, then freeze RED before any "
            "executable source change."
        ),
        "performance_measured": False,
        "production_changed": False,
        "target_selected": True,
    }

    raw, packed = _pack_codebook()
    codebook = artifact["codebook_audit"]
    assert codebook["entries"] == 256
    assert codebook["coordinates_per_entry"] == 4
    assert tuple(codebook["unique_magnitudes"]) == _EXPECTED_MAGNITUDES
    assert codebook["decode_formula"] == (
        "magnitude = 4 + 8*code + 2*(code == 7)"
    )
    assert codebook["bits_per_magnitude"] == 3
    assert codebook["payload_bits_per_entry"] == 12
    assert codebook["storage_bits_per_entry"] == 16
    assert codebook["control_storage_bytes"] == raw.nbytes == 1_024
    assert codebook["candidate_storage_bytes"] == packed.nbytes == 512
    assert codebook["storage_reduction_percent"] == 50.0
    assert codebook["all_256_entries_reconstruct_exactly"] is True
    assert codebook["all_256_packed_words_unique"] is True
    assert hashlib.sha256(raw.tobytes()).hexdigest() == _EXPECTED_RAW_TABLE_SHA256
    assert hashlib.sha256(packed.tobytes()).hexdigest() == (
        _EXPECTED_PACKED_TABLE_SHA256
    )
    assert codebook["raw_table_sha256"] == _EXPECTED_RAW_TABLE_SHA256
    assert codebook["packed_table_sha256"] == _EXPECTED_PACKED_TABLE_SHA256

    operation = artifact["operation_contract"]
    assert operation["id"] == "WPF-H8L"
    assert operation["candidate_name_suffix"] == "packed_codebook12"
    assert operation["decode_formula"] == codebook["decode_formula"]
    assert operation["raw_pointer_abi"] is True
    assert operation["gfx1151_fail_closed"] is True
    assert operation["no_allocation_workspace_dispatch_or_policy_change"] is True
    assert "registered H6T source owner" in operation["unfused_fallback"]

    model = artifact["operation_model"]
    assert model["actual_iq3_layers"] == model["h6t_calls"] == 45
    assert model["natural_segment_decodes"] == 103_056_384
    assert model["control_table_wave_loads"] == 824_451_072
    assert model["candidate_table_wave_loads"] == 824_451_072
    assert model["control_table_logical_bytes"] == 105_529_737_216
    assert model["candidate_table_logical_bytes"] == 52_764_868_608
    assert model["table_logical_bytes_removed"] == 52_764_868_608
    assert model["table_logical_byte_reduction_percent"] == 50.0
    assert model["static_load_shape_control"] == "8 b128 + 9 b32 + 6 d16"
    assert model["static_load_shape_candidate"] == "8 b128 + 3 b32 + 12 d16"
    assert model["candidate_additional_code_extracts"] == 24
    assert model["not_a_cache_traffic_or_speed_claim"] is True

    admission = artifact["admission"]
    assert admission["red_first"] is True
    assert admission["all_45_layers_inseparable"] is True
    assert "one uint16" in admission["no_sweep"]
    assert "every layer and aggregate" in admission["timing"]
    assert "no layer/expert/routing" in admission["forbidden_salvage"]
    assert _PHYSICAL_CONTRACT["metadata_vgpr_max"] == 128
    assert _PHYSICAL_CONTRACT["runtime_vgpr_max"] == 128
    assert _PHYSICAL_CONTRACT["metadata_lds_bytes_max"] == 384
    assert _PHYSICAL_CONTRACT["runtime_lds_bytes_max"] == 512
    assert _PHYSICAL_CONTRACT["runtime_scratch_bytes"] == 0
    assert _PHYSICAL_CONTRACT["global_loads"] == 23
    assert _PHYSICAL_CONTRACT["global_load_b128"] == 8
    assert _PHYSICAL_CONTRACT["global_load_b32"] == 3
    assert _PHYSICAL_CONTRACT["global_load_d16"] == 12
    assert _PHYSICAL_CONTRACT["useful_fmas"] == 216
    assert _TRACE_CONTRACT["candidate_calls"] == len(_ACTUAL_LAYER_IDS) == 45
    assert _TRACE_CONTRACT["request_dispatches"] == 2_155
    assert _TIMING_CONTRACT["actual_layer_ids"] == tuple(range(1, 46))
    assert _TIMING_CONTRACT["required_clocks"] == (
        "hip_event",
        "synchronized_wall",
    )
    assert _TIMING_CONTRACT["require_every_layer_both_clocks"] is True
    assert _TIMING_CONTRACT["require_aggregate_both_clocks"] is True
    assert not any(
        _TIMING_CONTRACT[name]
        for name in (
            "allow_layer_subset",
            "allow_table_width_sweep",
            "allow_decode_formula_change",
            "allow_load_source_change",
            "allow_body_rewrite",
            "allow_recompile",
            "allow_favorable_rerun",
        )
    )
    assert artifact["lineage"]["source_sha256"] == {
        "hipengine/kernels/hip_gfx1100/__init__.py": (
            "3638a8fb56d7f87b928bd4f9c8f533f3923381db4d7d7a6e1929ae283b37968d"
        ),
        "hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.hip": (
            "b78198db6edee39099378d0adce3629dce020edddc67378b93716be90db6c87a"
        ),
        "hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_prefill.hip": (
            "e7b616995c6381a0db6868349c5e407c985de7176e4abb297592fccd19849e8f"
        ),
        "hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_prefill.py": (
            "ec05d2a7da321dcedfc996d3b09da61eb6c02d45fd9f5f1f85f048d6f9aa3be2"
        ),
        "hipengine/kernels/hip_gfx1151/__init__.py": (
            "a5838ffc8fd8df367cd828f397e701f94f2268c7992d0a5e143c8d7e2b8ba3b3"
        ),
    }


def test_h8l_registry_source_policy_and_exact_h6t_body_delta() -> None:
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
    assert laguna_moe_scratch_nbytes(production, max_rows=512) == (
        _PRODUCTION_MOE_SCRATCH_BYTES
    )

    module = _module()
    hip_source = Path(module.__file__).with_suffix(".hip").read_text(encoding="utf-8")
    parent_source = (
        Path(module.__file__).with_name("gguf_iq_gemv.hip").read_text(encoding="utf-8")
    )
    raw_table = _declaration(
        parent_source,
        f"__device__ __constant__ uint32_t {_RAW_TABLE}[256] = {{",
    )
    h6t_load = _declaration(hip_source, f"__device__ inline IQ3Segment {_H6T_LOAD}(")
    h6t_dot = _declaration(hip_source, f"__device__ inline void {_H6T_DOT}(")
    h6t_add = _declaration(hip_source, f"__device__ inline float {_H6T_ADD}(")
    h6t_publish = _declaration(
        hip_source,
        f"template <int ROW_BATCH>\n__device__ inline void {_H6T_PUBLISH}(",
    )
    h6t_kernel = _declaration(hip_source, f"__global__ void {_H6T_KERNEL}(")
    h6t_export = _declaration(hip_source, f'extern "C" int {_H6T_SYMBOL}(')
    h6t_wrapper = inspect.getsource(getattr(module, _H6T_WRAPPER_NAME))
    assert _sha256_text(raw_table) == _RAW_TABLE_SHA256
    assert _sha256_text(h6t_load) == _H6T_LOAD_SHA256
    assert _sha256_text(h6t_dot) == _H6T_DOT_SHA256
    assert _sha256_text(h6t_add) == _H6T_ADD_SHA256
    assert _sha256_text(h6t_publish) == _H6T_PUBLISH_SHA256
    assert _sha256_text(h6t_kernel) == _H6T_KERNEL_SHA256
    assert _sha256_text(h6t_export) == _H6T_EXPORT_SHA256
    assert _sha256_text(h6t_wrapper) == _H6T_WRAPPER_SHA256
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H6T_VARIANT,
    ) is getattr(module, _H6T_WRAPPER_NAME)

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    # Intentional RED: retained production/source facts pass before the first
    # absent H8L surface, its separately named Python wrapper.
    candidate = _candidate()
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H8L_VARIANT,
    ) is candidate
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == expected_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == expected_abis
    assert hip_source.count(_H8L_SYMBOL) == 1
    assert hip_source.count(_H8L_KERNEL) == 2
    assert hip_source.count(_PACKED_TABLE) == 3
    assert hip_source.count(_H8L_LOAD) == 4

    packed_table = _declaration(
        hip_source,
        f"__device__ __constant__ uint16_t {_PACKED_TABLE}[256] = {{",
    )
    _, expected_packed = _pack_codebook()
    assert _table_values(packed_table) == tuple(int(value) for value in expected_packed)
    h8l_load = _declaration(hip_source, f"__device__ inline IQ3Segment {_H8L_LOAD}(")
    assert h8l_load == _expected_packed_load(h6t_load)
    h8l_kernel = _declaration(hip_source, f"__global__ void {_H8L_KERNEL}(")
    expected_kernel = h6t_kernel.replace(_H6T_KERNEL, _H8L_KERNEL).replace(
        f"{_H6T_LOAD}(",
        f"{_H8L_LOAD}(",
    )
    assert h8l_kernel == expected_kernel
    h8l_anchor = hip_source.index(f"__global__ void {_H8L_KERNEL}(")
    h8l_prefix = hip_source[max(0, h8l_anchor - 96) : h8l_anchor]
    assert "__launch_bounds__(128, 1)" in h8l_prefix
    assert "constexpr int row_batch = 8;" in h8l_kernel
    assert h8l_kernel.count(_H8L_LOAD) == 3
    assert h8l_kernel.count(_H6T_DOT) == 3
    assert h8l_kernel.count(_H6T_PUBLISH) == 3
    assert h8l_kernel.count(_SUM_HELPER) == 3
    assert h8l_kernel.count("__syncthreads();") == 2

    h8l_export = _declaration(hip_source, f'extern "C" int {_H8L_SYMBOL}(')
    expected_export = h6t_export.replace(_H6T_SYMBOL, _H8L_SYMBOL).replace(
        _H6T_KERNEL,
        _H8L_KERNEL,
    )
    assert h8l_export == expected_export
    assert h8l_export.count(f"{_H8L_KERNEL}<256>") == 1
    assert "dim3(256, 64)" in h8l_export
    assert "dim3(128)" in h8l_export
    candidate_wrapper = inspect.getsource(candidate)
    expected_wrapper = h6t_wrapper.replace(_H6T_WRAPPER_NAME, _H8L_WRAPPER_NAME).replace(
        _H6T_PY_SYMBOL,
        _H8L_PY_SYMBOL,
    )
    assert candidate_wrapper == expected_wrapper
    assert tuple(inspect.signature(candidate).parameters) == tuple(
        inspect.signature(getattr(module, _H6T_WRAPPER_NAME)).parameters
    )
    assert not is_registered(_candidate_key("hip_gfx1151"))
    gfx1151_source = inspect.getsource(hip_gfx1151)
    assert gfx1151_source.count("# H8L") == 1
    assert gfx1151_source.count("packed_codebook12") == 1


def test_h8l_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    module = _module()
    control = getattr(module, _H6T_WRAPPER_NAME)
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H8L shape reached the HIP loader")

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
    for function in (control, candidate):
        for changed, message in invalid:
            with pytest.raises(ValueError, match=message):
                function(1, 2, 3, 4, 5, 6, **(common | changed))
    assert load_attempts == 0


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h8l_complete_outputs_match_h6t_cpu_poison_and_lifecycle() -> None:
    candidate = _candidate()
    module = _module()
    control = getattr(module, _H6T_WRAPPER_NAME)
    grouped_library = _load_grouped_library()
    iq3_weights = {
        1: _make_iq3_weight(1),
        12: _make_iq3_weight(12),
        65: _make_iq3_weight(65),
    }
    cases = (
        ("rows", 1),
        ("rows", 7),
        ("rows", 8),
        ("rows", 9),
        ("rows", 512),
        ("partitions", 64),
        ("partitions", 65),
        ("uneven", 0),
        ("empty", 0),
    )
    for case, value in cases:
        if case == "rows":
            starts, active, active_count, selected = _single_expert_metadata(value)
            qweight = iq3_weights[1]
            compact_rows = value
        elif case == "partitions":
            starts, active, active_count, selected = _partition_boundary_metadata(value)
            qweight = iq3_weights[65]
            compact_rows = int(starts[-1])
        elif case == "uneven":
            starts, active, active_count, selected = _uneven_reordered_metadata()
            qweight = iq3_weights[12]
            compact_rows = int(starts[-1])
        else:
            starts, active, active_count, selected = _empty_metadata()
            qweight = iq3_weights[1]
            compact_rows = 1

        x_bf16 = _f32_to_bf16_u16(_make_x(compact_rows, _IN_FEATURES))
        poison = np.uint16(0x7FC0)
        initial_value = np.uint16(0x3F80) if case == "empty" else poison
        initial = np.full(
            (compact_rows, _OUT_FEATURES),
            initial_value,
            dtype=np.uint16,
        )
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
        if case == "empty":
            np.testing.assert_array_equal(expected, initial)
        else:
            assert not np.any(expected == poison)
            sample_rows = np.unique(
                np.asarray(
                    [0, 1, 7, 8, compact_rows // 2, compact_rows - 1]
                ).clip(0, compact_rows - 1)
            )
            cpu = _selected_reference(
                x_bf16[sample_rows],
                selected[sample_rows],
                qweight[:, _SAMPLE_COLS, :],
                GGMLQuantizationType.IQ3_XXS,
            )
            np.testing.assert_array_equal(
                expected[np.ix_(sample_rows, _SAMPLE_COLS)],
                cpu,
            )
        assert np.isfinite(_bf16_u16_to_f32(expected)).all()

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
        repeat = _run_h5j_or_h5q(
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
        np.testing.assert_array_equal(repeat, actual, err_msg=f"repeat {case}={value}")
        assert np.isfinite(_bf16_u16_to_f32(actual)).all()
        if case == "empty":
            np.testing.assert_array_equal(actual, initial)
        else:
            assert not np.any(actual == poison)
