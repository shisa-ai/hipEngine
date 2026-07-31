"""WPF-H6P exact staged-wave-publication triple-output IQ3 contract."""

from __future__ import annotations

import hashlib
import importlib
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

_EXPERT_PARTITIONS = 64
_OUTPUT_PARTITIONS = 256
_ROW_BATCH = 8
_WRAPPER_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_triple_output_rowbatch8_bf16_bf16_out"
)
_SYMBOL = f"hipengine_{_WRAPPER_NAME}"
_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_triple_output_rowbatch8_bf16_bf16_out"
)
_KERNEL_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_output_row_interleaved_vopd_"
    "staged_wave_publication_triple_output_rowbatch8_kernel"
)
_DOT_HELPER_NAME = "dot_iq3_segment_rowbatch8_interleaved"
_PUBLISH_HELPER_NAME = "publish_local128_wave_sums_batched_no_barrier"
_SUM_HELPER_NAME = "sum_local128_wave_sums_serial"
_H6I_KERNEL_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_output_row_interleaved_vopd_triple_output_"
    "rowbatch8_kernel"
)
_H6I_REDUCE_HELPER_NAME = "reduce_local128_triples_batched"
_H6I_KERNEL_BODY_SHA256 = (
    "c5c09756a7ba62bdddbb102936e9f19365f5d8e504e1a44c969a94484a71a535"
)
_H6I_REDUCE_BODY_SHA256 = (
    "8becc8a7ba1e6c57f9725b1933210310873eb9fdc8d65b5e053afab92b2739a1"
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
    return getattr(_module(), _WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq3_xxs", _VARIANT)


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


def _function_body(source: str, declaration: str) -> str:
    start = source.index(declaration)
    body_start = source.index("{", start)
    depth = 0
    for offset in range(body_start, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1 : offset]
    raise AssertionError(f"unterminated function: {declaration}")


def _body_sha256(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def test_h6p_registry_source_schedule_and_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    expected_variants = {
        "gguf_iq3_xxs": _H6I_VARIANT,
        "gguf_iq4_xs": _H5J_IQ4_VARIANT,
    }
    expected_abis = {
        _H5Q_VARIANT: _ACTIVE_EXPERT_ABI,
        _H5Z_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6D_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6F_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6I_VARIANT: _ACTIVE_EXPERT_ABI,
    }
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == expected_variants
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == expected_abis

    config = laguna_gguf_config_from_metadata(make_laguna_info())
    production = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert production.grouped_exact_down_keys["gguf_iq3_xxs"].variant == (
        _H6I_VARIANT
    )
    assert production.grouped_exact_down_routes["gguf_iq3_xxs"].abi == (
        _ACTIVE_EXPERT_ABI
    )
    assert (
        laguna_moe_scratch_nbytes(production, max_rows=512)
        == _PRODUCTION_MOE_SCRATCH_BYTES
    )

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    source = Path(_module().__file__).with_suffix(".hip").read_text()
    h6i_body = _function_body(source, f"__global__ void {_H6I_KERNEL_NAME}")
    h6i_reduce_body = _function_body(
        source,
        f"__device__ inline void {_H6I_REDUCE_HELPER_NAME}",
    )
    assert _body_sha256(h6i_body) == _H6I_KERNEL_BODY_SHA256
    assert _body_sha256(h6i_reduce_body) == _H6I_REDUCE_BODY_SHA256

    candidate = _candidate()
    assert candidate.__name__ == _WRAPPER_NAME
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_VARIANT,
    ) is candidate
    assert source.count(_SYMBOL) == 1
    assert source.count(_KERNEL_NAME) == 2
    assert source.count(
        f"__device__ inline void {_PUBLISH_HELPER_NAME}"
    ) == 1
    assert source.count(
        f"__device__ inline float {_SUM_HELPER_NAME}"
    ) == 1

    candidate_body = _function_body(source, f"__global__ void {_KERNEL_NAME}")
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
    assert "float acc_a[row_batch]" not in candidate_body
    assert "float acc_b[row_batch]" not in candidate_body
    assert "float acc_c[row_batch]" not in candidate_body
    publish_offsets = []
    for suffix in ("a", "b", "c"):
        assert f"const IQ3Segment segment_{suffix}" in candidate_body
        assert candidate_body.count(
            f"{_DOT_HELPER_NAME}(segment_{suffix}, activation, acc);"
        ) == 1
        publication = f"{_PUBLISH_HELPER_NAME}(acc, wave_sums_{suffix});"
        assert candidate_body.count(publication) == 1
        publish_offsets.append(candidate_body.index(publication))
        assert candidate_body.count(
            f"{_SUM_HELPER_NAME}(wave_sums_{suffix}, row_idx)"
        ) == 1
    assert publish_offsets == sorted(publish_offsets)
    first_barrier = candidate_body.index("__syncthreads();")
    assert publish_offsets[-1] < first_barrier
    assert candidate_body.count("__syncthreads();") == 2
    assert candidate_body.count(_H6I_REDUCE_HELPER_NAME) == 0
    assert "float_to_bf16_bits(value_a)" in candidate_body
    assert "float_to_bf16_bits(value_b)" in candidate_body
    assert "float_to_bf16_bits(value_c)" in candidate_body
    assert "out[row * out_features + out_col]" in candidate_body
    assert "out[row * out_features + second_out_col]" in candidate_body
    assert "out[row * out_features + third_out_col]" in candidate_body

    publish_body = _function_body(
        source,
        f"__device__ inline void {_PUBLISH_HELPER_NAME}",
    )
    assert publish_body.count("__syncthreads();") == 0
    assert "for (int offset = 16; offset > 0; offset >>= 1)" in publish_body
    assert "value[row] += __shfl_down(value[row], offset);" in publish_body
    assert "wave_sums[row * 4 + wave] = value[row];" in publish_body

    sum_body = _function_body(
        source,
        f"__device__ inline float {_SUM_HELPER_NAME}",
    )
    assert sum_body.count("__syncthreads();") == 0
    assert "float total = 0.0f;" in sum_body
    assert "for (int wave_idx = 0; wave_idx < 4; ++wave_idx)" in sum_body
    assert "total += wave_sums[row * 4 + wave_idx];" in sum_body
    assert "return total;" in sum_body


def test_h6p_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    candidate = _candidate()
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H6P shape reached the HIP loader")

    monkeypatch.setattr(module, "build_gguf_iq_selected_prefill", fail_if_loaded)
    common = dict(
        compact_rows=9,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=_NUM_EXPERTS,
    )
    for changed, message in (
        ({"compact_rows": 0}, "compact_rows must be positive"),
        ({"in_features": 768}, "exactly 1024"),
        ({"out_features": 1024}, "exactly 3072"),
        ({"num_experts": 255}, "exactly 256"),
    ):
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, 4, 5, 6, **(common | changed))
    assert load_attempts == 0


def _single_expert_metadata(
    compact_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    counts[0] = compact_rows
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active_count = np.asarray([1], dtype=np.int64)
    selected = np.zeros(compact_rows, dtype=np.int64)
    return starts, active, active_count, selected


def _partition_boundary_metadata(
    active_expert_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    counts[:active_expert_count] = 1
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    # Reverse active traversal while expert starts remain ascending. The leaf
    # must still visit all P64/P65 experts exactly once.
    active[:active_expert_count] = np.arange(active_expert_count - 1, -1, -1)
    active_count = np.asarray([active_expert_count], dtype=np.int64)
    selected = np.arange(active_expert_count, dtype=np.int64)
    return starts, active, active_count, selected


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("case", "value"),
    [
        pytest.param("rows", 1, id="rows1"),
        pytest.param("rows", 7, id="rows7"),
        pytest.param("rows", 8, id="rows8"),
        pytest.param("rows", 9, id="rows9"),
        pytest.param("rows", 512, id="rows512"),
        pytest.param("partitions", 64, id="p64"),
        pytest.param("partitions", 65, id="p65"),
    ],
)
def test_h6p_complete_outputs_match_h6i_and_cpu_at_staged_boundaries(
    grouped_library,
    iq3_weights: dict[int, np.ndarray],
    case: str,
    value: int,
) -> None:
    module = _module()
    candidate = _candidate()
    control = getattr(
        module,
        "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_"
        "p64_activation_resident_out_p256_row_interleaved_vopd_triple_output_"
        "rowbatch8_bf16_bf16_out",
    )
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
