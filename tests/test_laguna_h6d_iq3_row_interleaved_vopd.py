"""WPF-H6D exact row-interleaved IQ3 VOPD contract."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.quant.gguf import GGMLQuantizationType
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
    "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out"
)
_SYMBOL = f"hipengine_{_WRAPPER_NAME}"
_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out"
)
_KERNEL_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_output_row_interleaved_vopd_rowbatch8_kernel"
)
_HELPER_NAME = "dot_iq3_segment_rowbatch8_interleaved"
_H5Z_KERNEL_NAME = (
    "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_output_rowbatch8_kernel"
)
_H5Z_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_rowbatch8_bf16_bf16_out"
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


def test_h6d_registry_schedule_and_h6q_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    candidate = _candidate()
    assert candidate.__name__ == _WRAPPER_NAME
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant=_VARIANT,
    ) is candidate

    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {
        "gguf_iq3_xxs": _H6Q_RUNTIME_VARIANT,
        "gguf_iq4_xs": _H5J_IQ4_VARIANT,
    }
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {
        _H5Q_VARIANT: _ACTIVE_EXPERT_ABI,
        _H5Z_VARIANT: _ACTIVE_EXPERT_ABI,
        _VARIANT: _ACTIVE_EXPERT_ABI,
        _H6F_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6I_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6P_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6Q_RUNTIME_VARIANT: _ACTIVE_EXPERT_ABI,
        _H6R_RUNTIME_VARIANT: _ACTIVE_EXPERT_ABI,
    }
    assert _VARIANT in hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANTS == {}
    assert hip_gfx1151.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS == {}

    source = Path(_module().__file__).with_suffix(".hip").read_text()
    assert source.count(_SYMBOL) == 1
    assert source.count(_KERNEL_NAME) == 2
    assert source.count(f"__device__ inline void {_HELPER_NAME}") == 1

    helper = _function_body(
        source,
        f"__device__ inline void {_HELPER_NAME}",
    )
    assert "for (int j = 0; j < 4; ++j)" in helper
    schedule: list[int] = []
    for high_half in (False, True):
        magnitude = "j + 4" if high_half else "j"
        activation = "j + 4" if high_half else "j"
        for row in range(_ROW_BATCH):
            statement = (
                f"sum{row} += segment.magnitude[{magnitude}] * "
                f"bf16_bits_to_float(x[{row}][{activation}]);"
            )
            assert helper.count(statement) == 1
            schedule.append(helper.index(statement))
    assert schedule == sorted(schedule)
    for row in range(_ROW_BATCH):
        assert helper.count(f"float sum{row} = 0.0f;") == 1
        assert helper.count(f"acc[{row}] += segment.scale * sum{row};") == 1

    candidate_body = _function_body(source, f"__global__ void {_KERNEL_NAME}")
    assert "constexpr int expert_partitions = 64;" in candidate_body
    assert "constexpr int row_batch = 8;" in candidate_body
    assert "active_index += expert_partitions" in candidate_body
    assert "out_col += OUTPUT_PARTITIONS" in candidate_body
    assert candidate_body.count(
        f"{_HELPER_NAME}(segment, activation, acc);"
    ) == 1
    assert "reduce_block_batched(acc, wave_sums);" in candidate_body
    assert "float_to_bf16_bits(acc[row_idx])" in candidate_body
    assert "dot_iq3_segment(segment, activation[row_idx])" not in candidate_body

    h5z_body = _function_body(source, f"__global__ void {_H5Z_KERNEL_NAME}")
    assert "out_col += OUTPUT_PARTITIONS" in h5z_body
    assert (
        "acc[row_idx] += dot_iq3_segment(segment, activation[row_idx]);"
        in h5z_body
    )


def test_h6d_strict_shape_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    candidate = _candidate()
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H6D shape reached the HIP loader")

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


def _metadata(
    active_experts: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    pattern = (1, 2, 7, 8, 9)
    for expert in active_experts:
        counts[expert] = pattern[expert % len(pattern)]
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active[: len(active_experts)] = active_experts
    active_count = np.asarray([len(active_experts)], dtype=np.int64)
    selected = np.repeat(np.arange(_NUM_EXPERTS, dtype=np.int64), counts)
    return starts, active, active_count, selected


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h6d_complete_outputs_match_h5z_and_cpu_at_p64_boundary_and_tails(
    grouped_library,
) -> None:
    module = _module()
    control = module.GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS[
        _OUTPUT_PARTITIONS
    ]
    candidate = _candidate()

    for active_expert_count in (_EXPERT_PARTITIONS, _EXPERT_PARTITIONS + 1):
        active_experts = tuple(reversed(range(active_expert_count)))
        starts, active, active_count, selected = _metadata(active_experts)
        compact_rows = int(starts[-1])
        x_bf16 = _f32_to_bf16_u16(_make_x(compact_rows, _IN_FEATURES))
        qweight = _make_iq3_weight(active_expert_count)
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
        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=f"active_expert_count={active_expert_count}",
        )

        sample_rows = np.unique(
            np.asarray([0, 1, 7, 8, compact_rows // 2, compact_rows - 1])
        )
        sample_cols = np.asarray([0, 255, 256, 1535, 2815, 2816, 3071])
        cpu = _selected_reference(
            x_bf16[sample_rows],
            selected[sample_rows],
            qweight[:, sample_cols, :],
            GGMLQuantizationType.IQ3_XXS,
        )
        np.testing.assert_array_equal(actual[np.ix_(sample_rows, sample_cols)], cpu)
        assert np.isfinite(_bf16_u16_to_f32(cpu)).all()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h6d_empty_active_list_preserves_output(grouped_library) -> None:
    module = _module()
    candidate = _candidate()
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active_count = np.zeros(1, dtype=np.int64)
    x_bf16 = _f32_to_bf16_u16(_make_x(1, _IN_FEATURES))
    qweight = _make_iq3_weight(1)
    initial = np.full((1, _OUT_FEATURES), 0x3F80, dtype=np.uint16)

    expected = _run_h5j_or_h5q(
        module.GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS[
            _OUTPUT_PARTITIONS
        ],
        grouped_library,
        x_bf16=x_bf16,
        starts=starts,
        active_experts=active,
        active_count=active_count,
        qweight=qweight,
        initial=initial,
        persistent=True,
    )
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
    np.testing.assert_array_equal(expected, initial)
    np.testing.assert_array_equal(actual, initial)
