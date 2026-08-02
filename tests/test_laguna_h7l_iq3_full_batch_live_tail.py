"""WPF-H7L exact IQ3 full-batch/live-tail RED contract."""

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
    / "benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7k-matched-iq3-live-tail-target.json"
)
_TARGET_ARTIFACT_SHA256 = (
    "fb6d1d64ffd335dc0649478eec12bd0d9809b435ec0b4da66ec1abc016ec51a5"
)
_H7L_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_full_batch_live_tail_"
    "triple_output_rowbatch8_bf16_bf16_out"
)
_H7L_WRAPPER_NAME = "gguf_iq3_xxs_" + _H7L_VARIANT
_H7L_SYMBOL = "hipengine_" + _H7L_WRAPPER_NAME
_H7L_KERNEL_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_"
    "p64_activation_resident_output_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_full_batch_live_tail_"
    "triple_output_rowbatch8_kernel"
)
_H7L_DOT_TAIL_HELPER = "dot_iq3_segment_live_tail"
_H7L_PUBLISH_TAIL_HELPER = (
    "publish_local128_wave_sums_live_tail_dpp_peer_fused_add_no_barrier"
)
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
_ROUTING_CONTRACT = {
    "rows": 230_400,
    "active_experts": 9_844,
    "full_batches": 24_650,
    "tail_batches": 8_897,
    "total_batches": 33_547,
    "full_rows": 197_200,
    "tail_rows": 33_200,
    "padded_tail_rows": 37_976,
    "modeled_inactive_fma_wave_operations": 4_199_841_792,
    "modeled_inactive_exchange_wave_operations": 2_333_245_440,
}
_REMAINDER_HISTOGRAM = {
    "0": 947,
    "1": 1_623,
    "2": 1_356,
    "3": 1_311,
    "4": 1_258,
    "5": 1_204,
    "6": 1_135,
    "7": 1_010,
}
_PHYSICAL_CONTRACT = {
    "local_size": 128,
    "grid_x": 32_768,
    "grid_y": 64,
    "metadata_vgpr_max": 101,
    "runtime_vgpr_max": 104,
    "metadata_lds_bytes": 384,
    "runtime_lds_bytes": 512,
    "code_bytes_max": 14_000,
    "instruction_slots_max": 2_400,
    "private_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "runtime_scratch_bytes": 0,
}
_TRACE_CONTRACT = {
    "kernel_name": _H7L_KERNEL_NAME,
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
_REJECT_RULE = (
    "Any correctness, physical, cached-trace, compiler, lifecycle, per-layer "
    "both-clock, or aggregate both-clock miss removes every H7L surface "
    "without layer/tail/prompt subset, tuning, recompile, or favorable rerun."
)
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
    return getattr(_module(), _H7L_WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq3_xxs", _H7L_VARIANT)


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


def test_h7l_frozen_target_physical_trace_timing_and_rejection_contract() -> None:
    artifact = json.loads(_TARGET_ARTIFACT.read_text())
    assert _sha256_file(_TARGET_ARTIFACT) == _TARGET_ARTIFACT_SHA256
    assert artifact["status"] == (
        "accepted_matched_production_rerank_and_exact_h7l_target"
    )
    assert artifact["target"]["id"] == "WPF-H7L"
    assert artifact["target"]["implementation_absent"] is True
    assert artifact["decision"]["candidate_implemented"] is False
    assert artifact["decision"]["production_changed"] is False
    assert artifact["production"]["wall_tok_s"] == pytest.approx(
        431.31016450993457
    )
    assert artifact["production"]["matched_llamacpp_hip_tok_s"] == 690.791
    assert artifact["production"]["kernel_sum_ms"] == pytest.approx(
        1_172.2412389999988
    )

    dynamic = artifact["routing"]["dynamic_model"]
    for key, value in _ROUTING_CONTRACT.items():
        assert dynamic[key] == value
    assert artifact["routing"]["remainder_histogram"] == _REMAINDER_HISTOGRAM
    assert dynamic["full_batches"] + dynamic["tail_batches"] == dynamic[
        "total_batches"
    ]
    assert dynamic["full_rows"] + dynamic["tail_rows"] == dynamic["rows"]
    assert dynamic["padded_tail_rows"] == (
        8 * dynamic["tail_batches"] - dynamic["tail_rows"]
    )
    assert dynamic["full_batch_fraction"] == pytest.approx(24_650 / 33_547)
    assert dynamic["full_row_fraction"] == pytest.approx(197_200 / 230_400)
    assert dynamic["wasted_tail_compute_fraction"] == pytest.approx(
        37_976 / 268_376
    )
    assert dynamic["modeled_inactive_fma_wave_operations"] == (
        37_976 * 1_024 * 27 * 4
    )
    assert dynamic["modeled_inactive_exchange_wave_operations"] == (
        37_976 * 1_024 * 15 * 4
    )

    gate = artifact["target"]["physical_gate"]
    assert gate["local_size"] == _PHYSICAL_CONTRACT["local_size"]
    assert gate["grid"] == [
        _PHYSICAL_CONTRACT["grid_x"],
        _PHYSICAL_CONTRACT["grid_y"],
    ]
    for key in (
        "metadata_vgpr_max",
        "runtime_vgpr_max",
        "metadata_lds_bytes",
        "runtime_lds_bytes",
        "code_bytes_max",
        "instruction_slots_max",
    ):
        assert gate[key] == _PHYSICAL_CONTRACT[key]
    assert gate["private_spill_scratch_bytes"] == 0
    assert _TRACE_CONTRACT["kernel_name"] == _H7L_KERNEL_NAME
    assert _TRACE_CONTRACT["new_compiler_processes"] == 0
    assert _TIMING_CONTRACT["actual_layer_ids"] == tuple(range(1, 46))
    assert _TIMING_CONTRACT["require_every_layer_both_clocks"] is True
    assert _TIMING_CONTRACT["require_aggregate_both_clocks"] is True
    assert _TIMING_CONTRACT["one_shot_only"] is True
    assert "every H7L surface" in _REJECT_RULE
    assert "layer/tail/prompt subset" in _REJECT_RULE
    assert all(term not in _H7L_VARIANT for term in ("prompt", "token", "layer"))


def test_h7l_registry_source_policy_and_h6t_immutability() -> None:
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
    h6t_load = _declaration(source, "__device__ inline IQ3Segment load_iq3_segment(")
    h6t_add = _declaration(
        source, f"__device__ inline float {_H6T_ADD_HELPER}("
    )
    h6t_publish = _declaration(
        source,
        f"template <int ROW_BATCH>\n__device__ inline void {_H6T_PUBLISH_HELPER}(",
    )
    h6t_kernel = _declaration(source, f"__global__ void {_H6T_KERNEL_NAME}(")
    assert _sha256_text(h6t_load) == _H6T_LOAD_DECL_SHA256
    assert _sha256_text(h6t_add) == _H6T_ADD_DECL_SHA256
    assert _sha256_text(h6t_publish) == _H6T_PUBLISH_DECL_SHA256
    assert _sha256_text(h6t_kernel) == _H6T_KERNEL_DECL_SHA256
    h6t_wrapper = getattr(module, _H6T_WRAPPER_NAME)
    assert _sha256_text(inspect.getsource(h6t_wrapper)) == _H6T_PYTHON_WRAPPER_SHA256
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H6T_VARIANT,
    ) is h6t_wrapper
    assert h6t_kernel.count("if (row < end)") == 2
    assert h6t_kernel.count(_DOT_HELPER) == 3
    assert h6t_kernel.count(_H6T_PUBLISH_HELPER) == 3
    assert h6t_kernel.count(_SUM_HELPER) == 3
    assert h6t_kernel.count("__syncthreads();") == 2

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    # Intentional RED: all retained source/default/ABI facts pass before the
    # only missing boundary, the separately named H7L Python wrapper.
    candidate = _candidate()
    assert candidate.__name__ == _H7L_WRAPPER_NAME
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H7L_VARIANT,
    ) is candidate
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == expected_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == expected_abis
    assert source.count(_H7L_SYMBOL) == 1
    assert source.count(_H7L_KERNEL_NAME) == 2

    candidate_body = _declaration(source, f"__global__ void {_H7L_KERNEL_NAME}(")
    assert "constexpr int expert_partitions = 64;" in candidate_body
    assert "constexpr int row_batch = 8;" in candidate_body
    assert "const int64_t full_end" in candidate_body
    assert "((end - begin) / row_batch) * row_batch" in candidate_body
    assert "row_base < full_end" in candidate_body
    assert "const int tail_rows = static_cast<int>(end - full_end);" in candidate_body
    assert "if (tail_rows > 0)" in candidate_body
    assert candidate_body.count(_DOT_HELPER) == 3
    assert candidate_body.count(_H6T_PUBLISH_HELPER) == 3
    assert candidate_body.count(_H7L_DOT_TAIL_HELPER) == 3
    assert candidate_body.count(_H7L_PUBLISH_TAIL_HELPER) == 3
    assert candidate_body.count(_SUM_HELPER) == 6
    assert candidate_body.count("__syncthreads();") == 4
    assert candidate_body.count("if (row < end)") == 0
    assert candidate_body.count("out_col += 3 * OUTPUT_PARTITIONS") == 2

    tail_dot = _declaration(
        source, f"__device__ inline void {_H7L_DOT_TAIL_HELPER}("
    )
    assert "for (int row = 0; row < live_rows; ++row)" in tail_dot
    assert "acc[row] += dot_iq3_segment(segment, x[row]);" in tail_dot
    tail_publish = _declaration(
        source, f"__device__ inline void {_H7L_PUBLISH_TAIL_HELPER}("
    )
    assert tail_publish.count("row < live_rows") >= 5
    assert "h6r_permlanex16_f32(value[row])" in tail_publish
    assert "h6t_dpp_add_row_shl1_f32(value[row])" in tail_publish
    assert "wave_sums[row * 4 + wave] = value[row];" in tail_publish
    assert "__syncthreads();" not in tail_publish

    wrapper = inspect.getsource(candidate)
    assert _H7L_SYMBOL in wrapper
    gfx1151_source = inspect.getsource(hip_gfx1151)
    assert gfx1151_source.count(_H7L_VARIANT) == 1


def test_h7l_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    control = getattr(module, _H6T_WRAPPER_NAME)
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H7L live-tail shape reached HIP loader")

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
        *(pytest.param("rows", rows, id=f"rows{rows}") for rows in range(1, 10)),
        pytest.param("rows", 512, id="rows512"),
        pytest.param("partitions", 64, id="reversed-p64"),
        pytest.param("partitions", 65, id="reversed-p65"),
    ],
)
def test_h7l_complete_outputs_match_h6t_and_cpu_for_every_tail(
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
