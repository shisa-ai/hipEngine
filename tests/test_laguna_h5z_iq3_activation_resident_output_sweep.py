"""WPF-H5Z exact IQ3 activation-resident output-column sweep contract."""

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

_OUTPUT_PARTITIONS = (32, 64, 128, 256, 512)
_EXPERT_PARTITIONS = 64


def _variant(output_partition: int) -> str:
    return (
        "selected_grouped_prefill_compact_k1024_active_expert_p64_"
        f"activation_resident_out_p{output_partition}_rowbatch8_bf16_bf16_out"
    )


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )


def _candidates():
    return _module().GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS


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


def test_h5z_registry_preflight_and_gfx1151_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    candidates = _candidates()
    assert tuple(candidates) == _OUTPUT_PARTITIONS
    assert len({id(function) for function in candidates.values()}) == len(candidates)

    for output_partition, function in candidates.items():
        assert function.__name__ == (
            "gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_"
            "active_expert_p64_activation_resident_out_p"
            f"{output_partition}_rowbatch8_bf16_bf16_out"
        )
        key = KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_iq3_xxs",
            _variant(output_partition),
        )
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is function

    load_backend_kernel_package("hip_gfx1151")
    for output_partition in _OUTPUT_PARTITIONS:
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                "moe_linear",
                "gguf_iq3_xxs",
                _variant(output_partition),
            )
        )

    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H5Z shape reached the HIP loader")

    monkeypatch.setattr(module, "build_gguf_iq_selected_prefill", fail_if_loaded)
    candidate = candidates[32]
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
    counts_by_expert: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _assert_all_candidates_match_h5q_and_cpu(
    grouped_library,
    *,
    starts: np.ndarray,
    active: np.ndarray,
    active_count: np.ndarray,
    selected: np.ndarray,
    weight_experts: int,
) -> None:
    module = _module()
    control = module.GGUF_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITIONS[
        _EXPERT_PARTITIONS
    ]
    compact_rows = int(starts[-1])
    x_bf16 = _f32_to_bf16_u16(_make_x(compact_rows, _IN_FEATURES))
    qweight = _make_iq3_weight(weight_experts)
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

    for output_partition, candidate in _candidates().items():
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
            err_msg=f"output_partition={output_partition}",
        )

    sample_rows = np.unique(
        np.asarray([0, compact_rows // 3, compact_rows // 2, compact_rows - 1])
    )
    sample_cols = np.asarray([0, 31, 32, 1535, 3071])
    cpu = _selected_reference(
        x_bf16[sample_rows],
        selected[sample_rows],
        qweight[:, sample_cols, :],
        GGMLQuantizationType.IQ3_XXS,
    )
    np.testing.assert_array_equal(expected[np.ix_(sample_rows, sample_cols)], cpu)
    assert np.isfinite(_bf16_u16_to_f32(cpu)).all()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h5z_uneven_active_expert_order_matches_h5q_and_cpu(grouped_library) -> None:
    active = (7, 0, 11, 3, 5)
    counts = {0: 1, 3: 2, 5: 7, 7: 8, 11: 9}
    starts, active_array, active_count, selected = _metadata(active, counts)
    _assert_all_candidates_match_h5q_and_cpu(
        grouped_library,
        starts=starts,
        active=active_array,
        active_count=active_count,
        selected=selected,
        weight_experts=12,
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("active_expert_count", (_EXPERT_PARTITIONS, 65))
def test_h5z_p64_boundary_and_tail_match_h5q_and_cpu(
    grouped_library,
    active_expert_count: int,
) -> None:
    active = tuple(range(active_expert_count))
    pattern = (1, 2, 7, 8, 9)
    counts = {
        expert: pattern[expert % len(pattern)] for expert in active
    }
    starts, active_array, active_count, selected = _metadata(active, counts)
    _assert_all_candidates_match_h5q_and_cpu(
        grouped_library,
        starts=starts,
        active=active_array,
        active_count=active_count,
        selected=selected,
        weight_experts=active_expert_count,
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h5z_empty_active_list_preserves_output(grouped_library) -> None:
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active_count = np.zeros(1, dtype=np.int64)
    x_bf16 = _f32_to_bf16_u16(_make_x(1, _IN_FEATURES))
    qweight = _make_iq3_weight(1)
    initial = np.full((1, _OUT_FEATURES), 0x3F80, dtype=np.uint16)

    for output_partition, candidate in _candidates().items():
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
            initial,
            err_msg=f"output_partition={output_partition}",
        )
