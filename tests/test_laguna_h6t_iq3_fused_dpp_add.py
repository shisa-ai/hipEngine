"""WPF-H6T exact fused-DPP-add staged-wave IQ3 contract."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

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

_H6T_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
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
_H6R_WRAPPER_NAME = "gguf_iq3_xxs_" + _H6R_VARIANT
_H6R_KERNEL_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_output_row_interleaved_vopd_staged_wave_publication_"
    "dpp_peer_exchange_triple_output_rowbatch8_kernel"
)
_H6R_PUBLISH_HELPER = "publish_local128_wave_sums_batched_dpp_peer_no_barrier"
_H6R_PERMLANE_HELPER = "h6r_permlanex16_f32"
_H6R_MOVE_HELPER = "h6r_dpp_move_f32"
_SUM_HELPER = "sum_local128_wave_sums_serial"
_DOT_HELPER = "dot_iq3_segment_rowbatch8_interleaved"
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
_PRODUCTION_MOE_SCRATCH_BYTES = 104_370_976
_H6R_PERMLANE_DECL_SHA256 = (
    "c2dfa7a1735338ec3a7f7c1a5dfebe04d696c00475e253df64459454d7d35d47"
)
_H6R_MOVE_DECL_SHA256 = (
    "d73cff3f5aac847e53fce9775672fb7ddfd66a34b897f6e0bd032010857be71c"
)
_H6R_PUBLISH_DECL_SHA256 = (
    "85c2ca7f4df1f35792654c3561fdc6c5af54e66fb8a490d0570b91d52365bb6b"
)
_H6R_KERNEL_DECL_SHA256 = (
    "323a003060ff3a98ff3a4ecab63ce536167e9180967d45e97896934f23c77d53"
)
_H6R_PYTHON_WRAPPER_SHA256 = (
    "392cb6041a75d9b8c06dba2993e8a8d743eb44d7512db84546818d1fbd21c5f3"
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
    return getattr(_module(), _H6T_WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq3_xxs", _H6T_VARIANT)


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


def test_h6t_registry_source_policy_and_h6r_immutability() -> None:
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
    h6r_permlane = _declaration(
        source, f"__device__ inline float {_H6R_PERMLANE_HELPER}("
    )
    h6r_move = _declaration(
        source,
        f"template <int DPP_CTRL>\n__device__ inline float {_H6R_MOVE_HELPER}(",
    )
    h6r_publish = _declaration(
        source,
        f"template <int ROW_BATCH>\n__device__ inline void {_H6R_PUBLISH_HELPER}(",
    )
    h6r_kernel = _declaration(source, f"__global__ void {_H6R_KERNEL_NAME}(")
    assert _sha256(h6r_permlane) == _H6R_PERMLANE_DECL_SHA256
    assert _sha256(h6r_move) == _H6R_MOVE_DECL_SHA256
    assert _sha256(h6r_publish) == _H6R_PUBLISH_DECL_SHA256
    assert _sha256(h6r_kernel) == _H6R_KERNEL_DECL_SHA256
    h6r_wrapper = getattr(module, _H6R_WRAPPER_NAME)
    assert _sha256(inspect.getsource(h6r_wrapper)) == _H6R_PYTHON_WRAPPER_SHA256
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H6R_VARIANT,
    ) is h6r_wrapper

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    # Intentional RED: all retained production/source facts pass before the
    # only missing boundary, the separately named H6T Python wrapper.
    candidate = _candidate()
    assert candidate.__name__ == _H6T_WRAPPER_NAME
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_H6T_VARIANT,
    ) is candidate
    assert source.count(_H6T_SYMBOL) == 1
    assert source.count(_H6T_KERNEL_NAME) == 2
    assert source.count(f"__device__ inline void {_H6T_PUBLISH_HELPER}") == 1
    assert source.count(f"__device__ inline float {_H6T_ADD_HELPER}") == 1

    candidate_body = _declaration(source, f"__global__ void {_H6T_KERNEL_NAME}(")
    assert "constexpr int expert_partitions = 64;" in candidate_body
    assert "constexpr int row_batch = 8;" in candidate_body
    assert "active_index += expert_partitions" in candidate_body
    assert "out_col += 3 * OUTPUT_PARTITIONS" in candidate_body
    assert "const int second_out_col = out_col + OUTPUT_PARTITIONS;" in candidate_body
    assert (
        "const int third_out_col = second_out_col + OUTPUT_PARTITIONS;"
        in candidate_body
    )
    assert candidate_body.count("float acc[row_batch] = {};") == 3
    assert candidate_body.count(_H6T_PUBLISH_HELPER) == 3
    assert candidate_body.count(_SUM_HELPER) == 3
    assert candidate_body.count(_DOT_HELPER) == 3
    assert candidate_body.count("__syncthreads();") == 2
    assert "float_to_bf16_bits(value_a)" in candidate_body
    assert "float_to_bf16_bits(value_b)" in candidate_body
    assert "float_to_bf16_bits(value_c)" in candidate_body

    publish = _declaration(
        source,
        f"template <int ROW_BATCH>\n__device__ inline void {_H6T_PUBLISH_HELPER}(",
    )
    expected_steps = [
        f"value[row] += {_H6R_PERMLANE_HELPER}(value[row]);",
        f"value[row] += {_H6R_MOVE_HELPER}<0x108>(value[row]);",
        f"value[row] += {_H6R_MOVE_HELPER}<0x104>(value[row]);",
        f"value[row] += {_H6R_MOVE_HELPER}<0x102>(value[row]);",
        f"value[row] = {_H6T_ADD_HELPER}(value[row]);",
    ]
    offsets = [publish.index(step) for step in expected_steps]
    assert offsets == sorted(offsets)
    assert f"{_H6R_MOVE_HELPER}<0x101>" not in publish
    assert "__shfl_down" not in publish
    assert "__syncthreads();" not in publish
    assert "wave_sums[row * 4 + wave] = value[row];" in publish

    direct_add = _declaration(
        source, f"__device__ inline float {_H6T_ADD_HELPER}("
    )
    assert "asm volatile" in direct_add
    assert "v_add_f32_dpp %0, %1, %1 row_shl:1" in direct_add
    assert "row_mask:0xf bank_mask:0xf bound_ctrl:1" in direct_add
    assert '"=v"(result)' in direct_add
    assert '"v"(value)' in direct_add
    assert "__builtin_amdgcn_mov_dpp" not in direct_add


def test_h6t_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    control = getattr(module, _H6R_WRAPPER_NAME)
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid fused-DPP-add shape reached the HIP loader")

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

    # Intentional RED only after the retained H6R preflight is proven.
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
def test_h6t_complete_outputs_match_h6r_and_cpu_at_staged_boundaries(
    grouped_library,
    iq3_weights: dict[int, np.ndarray],
    case: str,
    value: int,
) -> None:
    module = _module()
    control = getattr(module, _H6R_WRAPPER_NAME)
    if case == "rows":
        starts, active, active_count, selected = _single_expert_metadata(value)
        qweight = iq3_weights[1]
    else:
        starts, active, active_count, selected = _partition_boundary_metadata(value)
        qweight = iq3_weights[65]
    compact_rows = int(starts[-1])
    x_bf16 = _f32_to_bf16_u16(_make_x(compact_rows, _IN_FEATURES))
    initial = np.full((compact_rows, _OUT_FEATURES), 0x7FC0, dtype=np.uint16)

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
    assert np.isfinite(_bf16_u16_to_f32(cpu)).all()

    # Intentional RED only after complete H6R and sampled CPU bytes pass.
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
    np.testing.assert_array_equal(actual, expected, err_msg=f"{case}={value}")
