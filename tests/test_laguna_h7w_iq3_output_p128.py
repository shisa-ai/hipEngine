"""WPF-H7W exact H6T output-partition P128 RED contract."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
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
_TARGET_ARTIFACT = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-post-h7v-iq3-output-p128-target.json"
)
_TARGET_ARTIFACT_SHA256 = (
    "f0f694361e4ebd24bd0c02d76d3182b6754bdda6b4e795bfa59c415088de17de"
)
_H7W_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p128_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
_H7W_WRAPPER_NAME = "gguf_iq3_xxs_" + _H7W_VARIANT
_H7W_SYMBOL = "hipengine_" + _H7W_WRAPPER_NAME
_H7W_PY_SYMBOL_CONSTANT = (
    "_SYMBOL_IQ3_ACTIVATION_RESIDENT_FUSED_DPP_ADD_STAGED_"
    "TRIPLE_OUTPUT_P128"
)
_H6T_VARIANT = _H7W_VARIANT.replace("out_p128", "out_p256")
_H6T_WRAPPER_NAME = "gguf_iq3_xxs_" + _H6T_VARIANT
_H6T_SYMBOL = "hipengine_" + _H6T_WRAPPER_NAME
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
_H6T_EXPORT_DECL_SHA256 = (
    "e10a77579ac6705a3cc5ce3207ccbbe126394a1526bcbfceb8308dbbb99cc904"
)
_H6T_PYTHON_WRAPPER_SHA256 = (
    "afcb504afababd006face7b600cdf0b22e4138828633179f6b33c133342492e8"
)
_DOT_HELPER = "dot_iq3_segment_rowbatch8_interleaved"
_SUM_HELPER = "sum_local128_wave_sums_serial"
_ACTUAL_LAYER_IDS = tuple(range(1, 46))
_PHYSICAL_CONTRACT = {
    "template_argument": 128,
    "local_size": 128,
    "runtime_grid_x": 16_384,
    "runtime_grid_y": 64,
    "workgroups_per_launch": 8_192,
    "metadata_lds_bytes": 384,
    "runtime_lds_bytes": 512,
    "metadata_vgpr_max": 101,
    "runtime_vgpr_max": 104,
    "metadata_sgpr_max": 78,
    "useful_fmas": 216,
    "permlanex16": 24,
    "dpp_adds": 96,
    "lds_load_b128": 24,
    "lds_stores": 12,
    "barriers": 2,
    "code_bytes_max": 7_920,
    "instruction_slots_max": 1_384,
    "private_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "runtime_scratch_bytes": 0,
}
_TRACE_CONTRACT = {
    "h7w_calls": 45,
    "unchanged_iq4_calls": 2,
    "request_dispatches": 2_286,
    "queues": 1,
    "compiler_processes": 0,
    "local_size": 128,
    "runtime_grid_x": 16_384,
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
    "allow_subset_salvage": False,
    "allow_partition_retune": False,
    "allow_body_rewrite": False,
    "allow_recompile": False,
    "allow_favorable_rerun": False,
}
_REJECTION_SURFACES = (
    "gfx1100 HIP export",
    "gfx1100 Python symbol/wrapper/registry key",
    "gfx1151 explicit exclusion",
    "H7W RED",
)
_SAMPLE_COLS = np.asarray(
    [
        0,
        127,
        128,
        255,
        256,
        383,
        384,
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
        2943,
        2944,
        3071,
    ],
    dtype=np.int64,
)


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )


def _candidate():
    return getattr(_module(), _H7W_WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq3_xxs", _H7W_VARIANT)


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


def _allocation_lifecycle() -> tuple[int, int]:
    stats = memory_stats()
    return stats["current_allocated_bytes"], stats["active_allocations"]


def _uneven_reordered_metadata(
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_experts = (7, 0, 11, 3, 5)
    counts_by_expert = {0: 1, 3: 2, 5: 7, 7: 8, 11: 9}
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
    return {
        1: _make_iq3_weight(1),
        12: _make_iq3_weight(12),
        65: _make_iq3_weight(65),
    }


def test_h7w_target_physical_trace_timing_and_rejection_contract() -> None:
    artifact_bytes = _TARGET_ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _TARGET_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "accepted_target_only_no_candidate_run"
    assert artifact["target"]["id"] == "WPF-H7W"
    assert artifact["target"]["implementation_present"] is False
    assert artifact["target"]["candidate_executed"] is False
    assert artifact["target"]["candidate_speed_result_exists"] is False
    assert artifact["decision"] == {
        "candidate_executed": False,
        "candidate_implemented": False,
        "next_action": (
            "commit this target-only packet, then freeze RED before "
            "implementation or candidate execution"
        ),
        "production_changed": False,
        "speed_result_exists": False,
        "target_selected": True,
    }
    production = artifact["production"]
    assert production["revision"].startswith("f9a538206")
    assert production["all_fresh_exact"] is True
    assert production["fresh_fixed_wall_tok_s"] == pytest.approx(437.8362401645228)
    assert production["fresh_generic_wall_tok_s"] == pytest.approx(434.611248008627)
    assert production["kernel_sum_ms"] == pytest.approx(1_153.3470579999982)
    assert production["dispatches"] == 2_286
    assert production["matched_llamacpp_hip_tok_s"] == 690.791

    current = artifact["current_h6t"]
    target = artifact["target"]
    assert current["calls"] == target["grid_workgroups_across_45_layers"] // 8_192
    assert current["calls"] == 45
    assert current["output_partitions"] == 256
    assert target["output_partitions"] == 128
    assert current["expert_partitions"] == target["expert_partitions"] == 64
    assert current["grid_workgroups_per_launch"] == 16_384
    assert target["grid_workgroups_per_launch"] == 8_192
    assert current["grid_workgroups_across_45_layers"] == 737_280
    assert target["grid_workgroups_across_45_layers"] == 368_640
    assert current["outputs_per_workgroup"] == 12
    assert target["outputs_per_workgroup"] == 24
    assert current["triple_epochs_per_workgroup"] == 4
    assert target["triple_epochs_per_workgroup"] == 8
    assert target["current_activation_b128_records"] == 68_704_256
    assert target["candidate_activation_b128_records"] == 34_352_128
    assert target["unchanged_weight_records"] == 515_281_920
    assert target["request_dispatch_delta"] == 0
    assert target["expected_request_dispatches"] == 2_286
    assert target["new_device_allocation_bytes"] == 0
    assert target["new_workspace_bytes"] == 0
    assert target["workgroup_reduction_percent"] == 50.0
    assert target["activation_record_reduction_percent"] == 50.0
    assert target["modeled_total_record_reduction_percent"] == pytest.approx(
        100.0 * (583_986_176 - 549_634_048) / 583_986_176
    )
    assert artifact["historical_h5z_p128"]["both_clock_positive_layers"] == 39
    assert artifact["historical_h5z_p128"]["p128_over_p256_event"] == pytest.approx(
        0.9708660778254004
    )
    assert artifact["historical_h5z_p128"]["p128_over_p256_wall"] == pytest.approx(
        0.9697126896999166
    )

    assert _PHYSICAL_CONTRACT == {
        "template_argument": 128,
        "local_size": 128,
        "runtime_grid_x": 16_384,
        "runtime_grid_y": 64,
        "workgroups_per_launch": 8_192,
        "metadata_lds_bytes": 384,
        "runtime_lds_bytes": 512,
        "metadata_vgpr_max": 101,
        "runtime_vgpr_max": 104,
        "metadata_sgpr_max": 78,
        "useful_fmas": 216,
        "permlanex16": 24,
        "dpp_adds": 96,
        "lds_load_b128": 24,
        "lds_stores": 12,
        "barriers": 2,
        "code_bytes_max": 7_920,
        "instruction_slots_max": 1_384,
        "private_bytes": 0,
        "vgpr_spills": 0,
        "sgpr_spills": 0,
        "runtime_scratch_bytes": 0,
    }
    assert _TRACE_CONTRACT == {
        "h7w_calls": 45,
        "unchanged_iq4_calls": 2,
        "request_dispatches": 2_286,
        "queues": 1,
        "compiler_processes": 0,
        "local_size": 128,
        "runtime_grid_x": 16_384,
        "runtime_grid_y": 64,
        "positive_duration": True,
    }
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
            "allow_subset_salvage",
            "allow_partition_retune",
            "allow_body_rewrite",
            "allow_recompile",
            "allow_favorable_rerun",
        )
    )
    assert len(_REJECTION_SURFACES) == 4
    assert "H7W RED" in _REJECTION_SURFACES
    assert (
        "every H7W export/wrapper/key/RED/gfx1151-exclusion surface"
        in artifact["admission"]["no_salvage"]
    )
    assert "all-45-layer" in artifact["admission"]["timing"]
    assert "exactly 45 H7W P128" in artifact["admission"]["trace"]

    # Candidate files are intentionally allowed to gain only the separately
    # checked H7W surface. All unrelated source owners remain byte-frozen.
    for relative in (
        "hipengine/kernels/hip_gfx1100/__init__.py",
        "hipengine/runtime/laguna_moe.py",
        "tests/test_laguna_h6t_iq3_fused_dpp_add.py",
        "tests/test_laguna_h6t_runtime_policy.py",
        "tests/test_laguna_h6t_source_default.py",
    ):
        assert _sha256_file(_ROOT / relative) == artifact["source_sha256"][relative]


def test_h7w_registry_source_policy_and_h6t_immutability() -> None:
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
    python_source = Path(module.__file__).read_text()
    hip_source = Path(module.__file__).with_suffix(".hip").read_text()
    h6t_load = _declaration(
        hip_source,
        "__device__ inline IQ3Segment load_iq3_segment(",
    )
    h6t_add = _declaration(
        hip_source,
        f"__device__ inline float {_H6T_ADD_HELPER}(",
    )
    h6t_publish = _declaration(
        hip_source,
        f"template <int ROW_BATCH>\n__device__ inline void {_H6T_PUBLISH_HELPER}(",
    )
    h6t_kernel = _declaration(
        hip_source,
        f"__global__ void {_H6T_KERNEL_NAME}(",
    )
    h6t_export = _declaration(hip_source, f'extern "C" int {_H6T_SYMBOL}(')
    assert _sha256_text(h6t_load) == _H6T_LOAD_DECL_SHA256
    assert _sha256_text(h6t_add) == _H6T_ADD_DECL_SHA256
    assert _sha256_text(h6t_publish) == _H6T_PUBLISH_DECL_SHA256
    assert _sha256_text(h6t_kernel) == _H6T_KERNEL_DECL_SHA256
    assert _sha256_text(h6t_export) == _H6T_EXPORT_DECL_SHA256
    h6t_wrapper = getattr(module, _H6T_WRAPPER_NAME)
    assert _sha256_text(inspect.getsource(h6t_wrapper)) == _H6T_PYTHON_WRAPPER_SHA256
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H6T_VARIANT,
    ) is h6t_wrapper
    assert h6t_kernel.count("float acc[row_batch] = {};") == 3
    assert h6t_kernel.count(_DOT_HELPER) == 3
    assert h6t_kernel.count(_H6T_PUBLISH_HELPER) == 3
    assert h6t_kernel.count(_SUM_HELPER) == 3
    assert h6t_kernel.count("__syncthreads();") == 2
    assert h6t_export.count(f"{_H6T_KERNEL_NAME}<256>") == 1
    assert "dim3(256, 64)" in h6t_export

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    # Intentional RED: all retained source/default/ABI facts pass before the
    # sole missing boundary, the separately named H7W P128 Python wrapper.
    candidate = _candidate()
    assert candidate.__name__ == _H7W_WRAPPER_NAME
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H7W_VARIANT,
    ) is candidate
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == expected_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == expected_abis
    assert hip_source.count(_H7W_SYMBOL) == 1
    assert hip_source.count(_H6T_KERNEL_NAME) == 3
    h7w_export = _declaration(hip_source, f'extern "C" int {_H7W_SYMBOL}(')
    assert h7w_export.count(f"{_H6T_KERNEL_NAME}<128>") == 1
    assert "dim3(128, 64)" in h7w_export
    assert "dim3(128)" in h7w_export
    assert "in_features != 1024" in h7w_export
    assert "out_features != 3072" in h7w_export
    assert "num_experts != 256" in h7w_export
    wrapper = inspect.getsource(candidate)
    assert _H7W_PY_SYMBOL_CONSTANT in wrapper
    assert (
        python_source.count(
            "active_expert_p64_activation_resident_out_p128_row_interleaved_vopd_"
        )
        == 5
    )
    assert python_source.count(_H7W_PY_SYMBOL_CONSTANT) >= 2
    assert not is_registered(_candidate_key("hip_gfx1151"))
    gfx1151_source = inspect.getsource(hip_gfx1151)
    assert gfx1151_source.count("# H7W") == 1
    assert (
        gfx1151_source.count(
            "activation_resident_out_p128_row_interleaved_vopd_staged_wave_"
        )
        == 1
    )


def test_h7w_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    control = getattr(module, _H6T_WRAPPER_NAME)
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H7W P128 shape reached the HIP loader")

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

    # Intentional RED only after retained H6T rejects before HIP loading.
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
        pytest.param("uneven", 0, id="uneven-reordered"),
        pytest.param("empty", 0, id="empty-active"),
    ],
)
def test_h7w_complete_output_matches_h6t_and_cpu_at_p128_boundaries(
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
            np.asarray([0, 1, 7, 8, compact_rows // 2, compact_rows - 1]).clip(
                0,
                compact_rows - 1,
            )
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

    # Intentional RED only after complete H6T bytes, sampled independent CPU
    # bytes, poison/finite behavior, and allocation recovery all pass.
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
