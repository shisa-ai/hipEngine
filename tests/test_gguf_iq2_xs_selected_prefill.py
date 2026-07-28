"""Compact grouped/WMMA prefill for raw Laguna-shaped IQ2_XS experts."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import (
    build_gguf_iq_gemv,
    gguf_iq2_xs_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill import (
    build_gguf_iq_selected_prefill,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_auto_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.registry import resolve
from tests.test_gguf_iq2_xs_gemv import _make_iq2_xs_weight
from tests.test_gguf_iq_gemv import _f32_to_bf16_u16, _make_x, _run_selected
from tests.test_gguf_iq_selected_prefill import (
    _bf16_u16_to_f32,
    _compact_meta,
    _distinct_up,
    _run_dual_grouped,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def libraries():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    require_cached = os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"
    return (
        build_gguf_iq_selected_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
        build_gguf_iq_gemv(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
    )


def _weights(num_experts: int, out_features: int, in_features: int):
    gate = _make_iq2_xs_weight(
        num_experts, out_features, in_features, seed=0x12A500 + out_features
    )
    return gate, _distinct_up(gate, 74)


def test_iq2_xs_prefill_registry_contract() -> None:
    expected = {
        "selected_dual_grouped_prefill_compact_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        ),
        "selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out
        ),
        "selected_dual_grouped_prefill_compact_rowbatch8_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch8_bf16_bf16_out
        ),
        "selected_dual_silu_grouped_prefill_compact_rowbatch4_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_rowbatch4_bf16_bf16_out
        ),
        "selected_dual_silu_grouped_prefill_compact_rowbatch8_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_rowbatch8_bf16_bf16_out
        ),
        "selected_dual_silu_grouped_prefill_compact_auto_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_auto_bf16_bf16_out
        ),
        "selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out
        ),
        "selected_dual_grouped_prefill_compact_auto_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out
        ),
        "selected_dual_wmma_prefill_compact_bf16_bf16_out": (
            gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out
        ),
    }
    for variant, fn in expected.items():
        assert resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_iq2_xs",
            variant=variant,
        ) is fn


@pytest.mark.parametrize(
    ("in_features", "compact_rows", "num_experts", "expected"),
    [
        (2048, 11, 3, "base"),
        (2048, 12, 3, "rowbatch4"),
        (3072, 1023, 256, "adaptive"),
        (3072, 1024, 256, "rowbatch4"),
    ],
)
def test_iq2_xs_auto_policy_uses_laguna_width_and_short_k_crossover(
    monkeypatch: pytest.MonkeyPatch,
    in_features: int,
    compact_rows: int,
    num_experts: int,
    expected: str,
) -> None:
    calls: list[str] = []
    module = "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    monkeypatch.setattr(
        f"{module}.gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("base"),
    )
    monkeypatch.setattr(
        f"{module}.gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("rowbatch4"),
    )
    monkeypatch.setattr(
        f"{module}.gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("adaptive"),
    )
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out(
        1,
        2,
        3,
        4,
        5,
        compact_rows=compact_rows,
        in_features=in_features,
        out_features=1024,
        num_experts=num_experts,
    )
    assert calls == [expected]


def test_iq2_xs_grouped_scalar_is_exact_at_k3072(libraries) -> None:
    grouped_library, direct_library = libraries
    meta = _compact_meta([0, 1, 15, 16, 17])
    in_features = 3072
    out_features = 19
    x = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    gate, up = _weights(meta.num_experts, out_features, in_features)
    expected_gate = _run_selected(
        gguf_iq2_xs_selected_gemv_bf16_bf16_out,
        direct_library,
        x_bf16=x,
        selected=meta.selected,
        qweight=gate,
        threads=256,
    )
    expected_up = _run_selected(
        gguf_iq2_xs_selected_gemv_bf16_bf16_out,
        direct_library,
        x_bf16=x,
        selected=meta.selected,
        qweight=up,
        threads=256,
    )
    actual = _run_dual_grouped(
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    np.testing.assert_array_equal(actual, np.concatenate((expected_gate, expected_up), axis=1))


def test_iq2_xs_rowbatch_variants_adaptive_and_auto_are_exact_at_k3072(libraries) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta([0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31])
    in_features = 3072
    out_features = 23
    x = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    gate, up = _weights(meta.num_experts, out_features, in_features)
    expected = _run_dual_grouped(
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    for wrapper in (
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out,
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out,
    ):
        actual = _run_dual_grouped(
            wrapper,
            grouped_library,
            x_bf16=x,
            meta=meta,
            gate=gate,
            up=up,
            wmma=False,
        )
        np.testing.assert_array_equal(actual, expected)


def test_iq2_xs_grouped_fused_silu_preserves_projection_boundaries(libraries) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta([0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17])
    in_features = 3072
    out_features = 19
    x = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    gate, up = _weights(meta.num_experts, out_features, in_features)
    projected = _run_dual_grouped(
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    gate_f32 = _bf16_u16_to_f32(projected[:, :out_features])
    up_f32 = _bf16_u16_to_f32(projected[:, out_features:])
    expected = _f32_to_bf16_u16(
        gate_f32
        * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_f32)))
        * up_f32
    )
    for wrapper in (
        gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
        gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
        gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_auto_bf16_bf16_out,
    ):
        actual = _run_dual_grouped(
            wrapper,
            grouped_library,
            x_bf16=x,
            meta=meta,
            gate=gate,
            up=up,
            wmma=False,
            fused_silu=True,
        )
        np.testing.assert_array_equal(actual, expected)


def test_iq2_xs_wmma_passes_k3072_quality_gate(libraries) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta([1, 15, 16, 17])
    in_features = 3072
    out_features = 32
    x = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    gate, up = _weights(meta.num_experts, out_features, in_features)
    expected = _run_dual_grouped(
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    actual = _run_dual_grouped(
        gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x,
        meta=meta,
        gate=gate,
        up=up,
        wmma=True,
    )
    expected_f32 = _bf16_u16_to_f32(expected)
    actual_f32 = _bf16_u16_to_f32(actual)
    max_rel = float(
        np.max(np.abs(actual_f32 - expected_f32) / np.maximum(np.abs(expected_f32), 1.0))
    )
    assert max_rel <= 0.05
    result = evaluate_logits(expected_f32, actual_f32)
    assert result.passed, result


def test_iq2_xs_grouped_scalar_covers_laguna_k3072_n1024(libraries) -> None:
    grouped_library, direct_library = libraries
    meta = _compact_meta([1, 4])
    in_features = 3072
    out_features = 1024
    x = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    gate, up = _weights(meta.num_experts, out_features, in_features)
    expected_gate = _run_selected(
        gguf_iq2_xs_selected_gemv_bf16_bf16_out,
        direct_library,
        x_bf16=x,
        selected=meta.selected,
        qweight=gate,
        threads=256,
    )
    expected_up = _run_selected(
        gguf_iq2_xs_selected_gemv_bf16_bf16_out,
        direct_library,
        x_bf16=x,
        selected=meta.selected,
        qweight=up,
        threads=256,
    )
    actual = _run_dual_grouped(
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out,
        grouped_library,
        x_bf16=x,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    np.testing.assert_array_equal(actual, np.concatenate((expected_gate, expected_up), axis=1))


def test_iq2_xs_prefill_wrapper_rejects_width_above_laguna() -> None:
    with pytest.raises(ValueError, match="at most 3072"):
        gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            compact_rows=1,
            in_features=3328,
            out_features=1024,
            num_experts=256,
        )
