"""RED contracts for WPF-H8K exact IQ3 uniform-rowbatch4 ownership."""

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

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-03-gfx1100-laguna-q2-xl-post-h8j-"
    "iq3-rowbatch4-triple-output-target.json"
)
_TARGET_SHA256 = "999d0f92a631aa8c5b4f2ba36d969bae72970bb1725d8883848c0bb91dd32aa8"
_H6T_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
_H8K_VARIANT = _H6T_VARIANT.replace("rowbatch8_bf16", "rowbatch4_bf16")
_H6T_WRAPPER_NAME = "gguf_iq3_xxs_" + _H6T_VARIANT
_H8K_WRAPPER_NAME = "gguf_iq3_xxs_" + _H8K_VARIANT
_H6T_SYMBOL = "hipengine_" + _H6T_WRAPPER_NAME
_H8K_SYMBOL = "hipengine_" + _H8K_WRAPPER_NAME
_H6T_PY_SYMBOL = (
    "_SYMBOL_IQ3_ACTIVATION_RESIDENT_FUSED_DPP_ADD_STAGED_TRIPLE_OUTPUT"
)
_H8K_PY_SYMBOL = _H6T_PY_SYMBOL + "_ROWBATCH4"
_H6T_KERNEL = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_"
    "p64_activation_resident_output_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_kernel"
)
_H8K_KERNEL = _H6T_KERNEL.replace("rowbatch8_kernel", "rowbatch4_kernel")
_H6T_PUBLISH = "publish_local128_wave_sums_batched_dpp_peer_fused_add_no_barrier"
_H6T_ADD = "h6t_dpp_add_row_shl1_f32"
_DOT8_HELPER = "dot_iq3_segment_rowbatch8_interleaved"
_DOT4_HELPER = "dot_iq3_segment_rowbatch4_interleaved"
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
_H6T_LOAD_SHA256 = "3587f01bf37e87c15aec767bf6f666114f9ac6f27d771c7db5d7e1b7c6268ca1"
_H6T_ADD_SHA256 = "8ebcb4fb644a744450cccc683ba75f21729c255482d0142fd6a6b32ae41c31ce"
_H6T_PUBLISH_SHA256 = "7e16097a9228bd023322686b5a371db0329d56ff88624575572c121db05d2ef0"
_H6T_KERNEL_SHA256 = "36f941e39aa27767a35582d99a84dfb9d5f8d001626da9b4855f3f4d4ed618eb"
_H6T_EXPORT_SHA256 = "e10a77579ac6705a3cc5ce3207ccbbe126394a1526bcbfceb8308dbbb99cc904"
_H6T_WRAPPER_SHA256 = "afcb504afababd006face7b600cdf0b22e4138828633179f6b33c133342492e8"
_ACTUAL_LAYER_IDS = tuple(range(1, 46))
_PHYSICAL_CONTRACT = {
    "launch_bounds": (128, 1),
    "local_size": 128,
    "runtime_grid_x": 32_768,
    "runtime_grid_y": 64,
    "metadata_lds_bytes_max": 192,
    "runtime_lds_bytes_max": 256,
    "metadata_vgpr_max": 96,
    "runtime_vgpr_max": 96,
    "code_bytes_max": 7_920,
    "instruction_slots_max": 1_384,
    "private_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "runtime_scratch_bytes": 0,
    "global_loads": 19,
    "useful_fmas": 108,
    "permlanex16": 12,
    "dpp_adds": 48,
    "lds_load_b128": 12,
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
    "allow_rowbatch_sweep": False,
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
    return getattr(_module(), _H8K_WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq3_xxs", _H8K_VARIANT)


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


def _expected_rowbatch4_helper(h6t_helper: str) -> str:
    lines = []
    for line in h6t_helper.splitlines():
        if re.search(r"sum[4-7]|x\[[4-7]\]|acc\[[4-7]\]", line):
            continue
        lines.append(line)
    return (
        "\n".join(lines)
        .replace(_DOT8_HELPER, _DOT4_HELPER)
        .replace("const ushort8_t (&x)[8]", "const ushort8_t (&x)[4]")
        .replace("float (&acc)[8]", "float (&acc)[4]")
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_h8k_target_physical_trace_and_timing_contract() -> None:
    assert _sha256_file(_TARGET) == _TARGET_SHA256
    artifact = json.loads(_TARGET.read_text(encoding="utf-8"))
    assert artifact["status"] == "target_selected_no_candidate_no_speed_result"
    assert artifact["kind"] == (
        "gfx1100_laguna_q2_xl_post_h8j_iq3_rowbatch4_triple_output_target"
    )
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
    operation = artifact["operation_contract"]
    assert operation["control_row_batch"] == 8
    assert operation["candidate_row_batch"] == 4
    assert operation["candidate_name_suffix"] == "triple_output_rowbatch4"
    assert operation["raw_pointer_abi"] is True
    assert operation["gfx1151_fail_closed"] is True
    assert operation["no_allocation_workspace_or_request_dispatch_change"] is True

    model = artifact["operation_model"]
    assert model["actual_iq3_layers"] == 45
    assert model["active_experts"] == 9_844
    assert model["useful_rows"] == 230_400
    assert model["h6t_calls"] == 45
    assert model["control"]["row_batch"] == 8
    assert model["control"]["total_batches"] == 33_547
    assert model["control"]["compute_slots"] == 268_376
    assert model["control"]["padded_rows"] == 37_976
    assert model["control"]["barrier_epochs"] == 68_704_256
    assert model["candidate"]["row_batch"] == 4
    assert model["candidate"]["total_batches"] == 61_546
    assert model["candidate"]["compute_slots"] == 246_184
    assert model["candidate"]["padded_rows"] == 15_784
    assert model["candidate"]["barrier_epochs"] == 126_046_208
    assert model["delta"]["compute_slots"] == -22_192
    assert model["delta"]["barrier_epochs"] == 57_341_952
    assert model["delta"]["modeled_removed_fma_wave_operations"] == 2_454_257_664
    assert model["delta"]["modeled_removed_exchange_wave_operations"] == 1_363_476_480
    assert model["register_liveness"]["explicit_long_lived_dword_reduction"] == 20
    assert model["register_liveness"]["candidate_metadata_vgpr_ceiling"] == 96
    assert model["register_liveness"]["candidate_runtime_vgpr_ceiling"] == 96
    assert model["not_a_speed_claim"] is True

    admission = artifact["admission"]
    assert admission["red_first"] is True
    assert admission["all_45_layers_inseparable"] is True
    assert "rowbatch2/3/5/6/7" in admission["no_sweep"]
    assert "every layer and aggregate" in admission["timing"]
    assert "favorable-rerun subset" in admission["forbidden_salvage"]
    assert _PHYSICAL_CONTRACT["metadata_vgpr_max"] == 96
    assert _PHYSICAL_CONTRACT["runtime_vgpr_max"] == 96
    assert _PHYSICAL_CONTRACT["metadata_lds_bytes_max"] == 192
    assert _PHYSICAL_CONTRACT["runtime_lds_bytes_max"] == 256
    assert _PHYSICAL_CONTRACT["runtime_scratch_bytes"] == 0
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
            "allow_rowbatch_sweep",
            "allow_body_rewrite",
            "allow_recompile",
            "allow_favorable_rerun",
        )
    )
    assert artifact["lineage"]["source_sha256"] == {
        "hipengine/kernels/hip_gfx1100/__init__.py": (
            "3638a8fb56d7f87b928bd4f9c8f533f3923381db4d7d7a6e1929ae283b37968d"
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


def test_h8k_registry_source_policy_and_exact_h6t_body_delta() -> None:
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
    assert production.grouped_exact_down_keys["gguf_iq3_xxs"].variant == _H6T_VARIANT
    assert production.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        _ACTIVE_EXPERT_ABI
    )
    assert laguna_moe_scratch_nbytes(production, max_rows=512) == (
        _PRODUCTION_MOE_SCRATCH_BYTES
    )

    module = _module()
    hip_source = Path(module.__file__).with_suffix(".hip").read_text(encoding="utf-8")
    h6t_load = _declaration(
        hip_source,
        "__device__ inline IQ3Segment load_iq3_segment(",
    )
    h6t_add = _declaration(
        hip_source,
        f"__device__ inline float {_H6T_ADD}(",
    )
    h6t_publish = _declaration(
        hip_source,
        f"template <int ROW_BATCH>\n__device__ inline void {_H6T_PUBLISH}(",
    )
    h6t_dot = _declaration(
        hip_source,
        f"__device__ inline void {_DOT8_HELPER}(",
    )
    h6t_kernel = _declaration(hip_source, f"__global__ void {_H6T_KERNEL}(")
    h6t_export = _declaration(hip_source, f'extern "C" int {_H6T_SYMBOL}(')
    h6t_wrapper = inspect.getsource(getattr(module, _H6T_WRAPPER_NAME))
    assert _sha256_text(h6t_load) == _H6T_LOAD_SHA256
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

    # Intentional RED: all retained source facts pass before the first absent
    # H8K surface, its separately named Python wrapper.
    candidate = _candidate()
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H8K_VARIANT,
    ) is candidate
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == expected_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == expected_abis
    assert hip_source.count(_H8K_SYMBOL) == 1
    assert hip_source.count(_H8K_KERNEL) == 2

    h8k_dot = _declaration(hip_source, f"__device__ inline void {_DOT4_HELPER}(")
    assert h8k_dot == _expected_rowbatch4_helper(h6t_dot)
    h8k_kernel = _declaration(hip_source, f"__global__ void {_H8K_KERNEL}(")
    expected_kernel = (
        h6t_kernel.replace(_H6T_KERNEL, _H8K_KERNEL)
        .replace("constexpr int row_batch = 8;", "constexpr int row_batch = 4;")
        .replace(_DOT8_HELPER, _DOT4_HELPER)
    )
    assert h8k_kernel == expected_kernel
    h8k_anchor = hip_source.index(f"__global__ void {_H8K_KERNEL}(")
    h8k_prefix = hip_source[max(0, h8k_anchor - 96) : h8k_anchor]
    assert "__launch_bounds__(128, 1)" in h8k_prefix
    assert "__launch_bounds__(128, 4)" not in h8k_prefix
    assert "constexpr int row_batch = 4;" in h8k_kernel
    assert "ushort8_t activation[row_batch] = {};" in h8k_kernel
    assert h8k_kernel.count("float acc[row_batch] = {};") == 3
    assert h8k_kernel.count(_DOT4_HELPER) == 3
    assert h8k_kernel.count(_H6T_PUBLISH) == 3
    assert h8k_kernel.count(_SUM_HELPER) == 3
    assert h8k_kernel.count("__syncthreads();") == 2
    assert "live_tail" not in h8k_kernel

    h8k_export = _declaration(hip_source, f'extern "C" int {_H8K_SYMBOL}(')
    expected_export = h6t_export.replace(_H6T_SYMBOL, _H8K_SYMBOL).replace(
        _H6T_KERNEL,
        _H8K_KERNEL,
    )
    assert h8k_export == expected_export
    assert h8k_export.count(f"{_H8K_KERNEL}<256>") == 1
    assert "dim3(256, 64)" in h8k_export
    assert "dim3(128)" in h8k_export
    candidate_wrapper = inspect.getsource(candidate)
    expected_wrapper = h6t_wrapper.replace(_H6T_WRAPPER_NAME, _H8K_WRAPPER_NAME).replace(
        _H6T_PY_SYMBOL,
        _H8K_PY_SYMBOL,
    )
    assert candidate_wrapper == expected_wrapper
    assert tuple(inspect.signature(candidate).parameters) == tuple(
        inspect.signature(getattr(module, _H6T_WRAPPER_NAME)).parameters
    )
    assert not is_registered(_candidate_key("hip_gfx1151"))
    gfx1151_source = inspect.getsource(hip_gfx1151)
    assert gfx1151_source.count("# H8K") == 1
    assert gfx1151_source.count("triple_output_rowbatch4") == 1


def test_h8k_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    module = _module()
    control = getattr(module, _H6T_WRAPPER_NAME)
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H8K shape reached the HIP loader")

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
def test_h8k_complete_outputs_match_h6t_cpu_poison_and_lifecycle() -> None:
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
        ("rows", 3),
        ("rows", 4),
        ("rows", 5),
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
                np.asarray([0, 1, 3, 4, 7, 8, compact_rows // 2, compact_rows - 1])
                .clip(0, compact_rows - 1)
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
