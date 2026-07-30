"""WPF-H5T exact one-wave IQ3 K-partition ownership contract."""

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
    _OUT_FEATURES,
    _f32_to_bf16_u16,
    _make_iq3_weight,
    _make_x,
    _run_h5j_or_h5q,
)
from tests.test_gguf_iq_gemv import _selected_reference

_IN_FEATURES = 1024
_NUM_EXPERTS = 256
_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_wave32x4_"
    "resident_rowbatch8_bf16_bf16_out"
)
_FUNCTION = f"gguf_iq3_xxs_{_VARIANT}"
_H5Q_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "resident_rowbatch8_bf16_bf16_out"
)


def _candidate(module):
    return getattr(module, _FUNCTION)


@pytest.fixture(scope="module")
def h5t_grouped_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return module.build_gguf_iq_selected_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


def test_h5t_registry_preflight_backend_scope_and_policy_immutability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )
    candidate = _candidate(module)
    key = KernelKey("hip_gfx1100", "moe_linear", "gguf_iq3_xxs", _VARIANT)
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is candidate

    package = importlib.import_module("hipengine.kernels.hip_gfx1100")
    assert package.LAGUNA_GROUPED_IQ_DOWN_VARIANTS["gguf_iq3_xxs"] == _H5Q_VARIANT
    assert _VARIANT not in package.LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(
        KernelKey("hip_gfx1151", "moe_linear", "gguf_iq3_xxs", _VARIANT)
    )

    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H5T shape reached the HIP loader")

    monkeypatch.setattr(module, "build_gguf_iq_selected_prefill", fail_if_loaded)
    common = dict(
        compact_rows=9,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=_NUM_EXPERTS,
    )
    for changed, message in (
        ({"in_features": 768}, "exactly 1024"),
        ({"out_features": 1024}, "exactly 3072"),
        ({"num_experts": 255}, "exactly 256"),
    ):
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, 4, 5, 6, **(common | changed))
    assert load_attempts == 0


def _metadata(active_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    counts[:active_count] = 1
    counts[0] = 9
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    active[:active_count] = np.arange(active_count, dtype=np.int64)
    return starts, active, np.asarray([active_count], dtype=np.int64)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("active_count", (64, 65))
def test_h5t_matches_h5q_and_cpu_across_p64_boundary_and_rowbatch_tail(
    h5t_grouped_library,
    active_count: int,
) -> None:
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )
    candidate = _candidate(module)
    control = module.GGUF_IQ3_ACTIVE_EXPERT_PERSISTENT_PARTITIONS[64]
    starts, active, count = _metadata(active_count)
    compact_rows = int(starts[-1])
    x_bf16 = _f32_to_bf16_u16(_make_x(compact_rows, _IN_FEATURES))
    qweight = _make_iq3_weight(active_count)
    initial = np.full((compact_rows, _OUT_FEATURES), 0x7FC0, dtype=np.uint16)
    expected = _run_h5j_or_h5q(
        control,
        h5t_grouped_library,
        x_bf16=x_bf16,
        starts=starts,
        active_experts=active,
        active_count=count,
        qweight=qweight,
        initial=initial,
        persistent=True,
    )
    actual = _run_h5j_or_h5q(
        candidate,
        h5t_grouped_library,
        x_bf16=x_bf16,
        starts=starts,
        active_experts=active,
        active_count=count,
        qweight=qweight,
        initial=initial,
        persistent=True,
    )
    np.testing.assert_array_equal(actual, expected)

    sample_rows = np.asarray([0, 8, compact_rows - 1])
    selected = np.searchsorted(starts[1:], sample_rows, side="right").astype(np.int64)
    sample_cols = np.asarray([0, 1535, 3071])
    cpu = _selected_reference(
        x_bf16[sample_rows],
        selected,
        qweight[:, sample_cols, :],
        GGMLQuantizationType.IQ3_XXS,
    )
    np.testing.assert_array_equal(actual[np.ix_(sample_rows, sample_cols)], cpu)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_h5t_empty_active_list_preserves_output(h5t_grouped_library) -> None:
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    active = np.zeros(_NUM_EXPERTS, dtype=np.int64)
    count = np.zeros(1, dtype=np.int64)
    x_bf16 = _f32_to_bf16_u16(_make_x(1, _IN_FEATURES))
    qweight = _make_iq3_weight(1)
    initial = np.full((1, _OUT_FEATURES), 0x3F80, dtype=np.uint16)
    actual = _run_h5j_or_h5q(
        _candidate(module),
        h5t_grouped_library,
        x_bf16=x_bf16,
        starts=starts,
        active_experts=active,
        active_count=count,
        qweight=qweight,
        initial=initial,
        persistent=True,
    )
    np.testing.assert_array_equal(actual, initial)
