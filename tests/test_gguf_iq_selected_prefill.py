"""Grouped scalar and compact-WMMA prefill gates for raw GGUF IQ experts.

The kernels consume the existing compact-MoE ABI.  Scalar grouped kernels use
``x[compact_rows, K]`` plus ``expert_start_compact[E+1]`` and must be BF16-bit
exact to the selected-single fallback.  Compact WMMA kernels additionally use
``expert_start_wmma`` and ``tile_expert``; they are checked against the exact
scalar route under the repository KL/top-1 gate and explicit relative bounds.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import (
    build_gguf_iq_gemv,
    gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
    gguf_iq4_xs_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill import (
    build_gguf_iq_selected_prefill,
    gguf_iq3_xxs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out,
    gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
    gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
    gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out,
    gguf_iq3_xxs_selected_grouped_prefill_compact_bf16_bf16_out,
    gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out,
    gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
    gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
    gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out,
    gguf_iq4_xs_selected_grouped_prefill_compact_auto_bf16_bf16_out,
    gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out,
    gguf_iq4_xs_selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out,
    gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out,
    gguf_iq4_xs_selected_wmma_prefill_compact_bf16_bf16_out,
    plan_gguf_iq_selected_prefill_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_gguf_iq_gemv import (
    _bf16_u16_to_f32,
    _f32_to_bf16_u16,
    _make_iq3_weight,
    _make_iq4_weight,
    _make_x,
    _run_selected,
    _selected_reference,
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


@dataclass(frozen=True)
class CompactMeta:
    counts: tuple[int, ...]
    selected: np.ndarray
    expert_start_compact: np.ndarray
    expert_start_wmma: np.ndarray
    tile_expert: np.ndarray
    compact_rows: int
    wmma_total_rows: int
    num_experts: int


def _compact_meta(counts: list[int]) -> CompactMeta:
    if not counts or sum(counts) <= 0:
        raise ValueError("counts must contain at least one routed row")
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    starts[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    padded = [((count + 15) // 16) * 16 for count in counts]
    wmma_starts = np.zeros(len(counts) + 1, dtype=np.int64)
    wmma_starts[1:] = np.cumsum(np.asarray(padded, dtype=np.int64))
    tile_expert = np.asarray(
        [expert for expert, rows in enumerate(padded) for _ in range(rows // 16)],
        dtype=np.int64,
    )
    selected = np.asarray(
        [expert for expert, count in enumerate(counts) for _ in range(count)],
        dtype=np.int64,
    )
    return CompactMeta(
        counts=tuple(counts),
        selected=selected,
        expert_start_compact=starts,
        expert_start_wmma=wmma_starts,
        tile_expert=tile_expert,
        compact_rows=int(starts[-1]),
        wmma_total_rows=int(wmma_starts[-1]),
        num_experts=len(counts),
    )


def _device_buffer(array: np.ndarray, buffers: list[Any]):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes)
    buffers.append(buffer)
    return buffer


def _run_dual_grouped(
    wrapper,
    library,
    *,
    x_bf16: np.ndarray,
    meta: CompactMeta,
    gate: np.ndarray,
    up: np.ndarray,
    wmma: bool,
    fused_silu: bool = False,
) -> np.ndarray:
    x_bf16 = np.ascontiguousarray(x_bf16, dtype=np.uint16)
    gate = np.ascontiguousarray(gate, dtype=np.uint8)
    up = np.ascontiguousarray(up, dtype=np.uint8)
    out_features = int(gate.shape[1])
    out_columns = out_features if fused_silu else 2 * out_features
    out = np.zeros((meta.compact_rows, out_columns), dtype=np.uint16)
    buffers: list[Any] = []
    try:
        x_buf = _device_buffer(x_bf16, buffers)
        start_buf = _device_buffer(meta.expert_start_compact, buffers)
        gate_buf = _device_buffer(gate, buffers)
        up_buf = _device_buffer(up, buffers)
        out_buf = malloc(out.nbytes)
        buffers.append(out_buf)
        args: list[int] = [x_buf.ptr, start_buf.ptr]
        if wmma:
            wmma_start_buf = _device_buffer(meta.expert_start_wmma, buffers)
            tile_buf = _device_buffer(meta.tile_expert, buffers)
            args.extend((wmma_start_buf.ptr, tile_buf.ptr))
        args.extend((gate_buf.ptr, up_buf.ptr, out_buf.ptr))
        kwargs = dict(
            compact_rows=meta.compact_rows,
            in_features=x_bf16.shape[1],
            out_features=out_features,
            num_experts=meta.num_experts,
            library=library,
        )
        if wmma:
            kwargs["wmma_total_rows"] = meta.wmma_total_rows
        wrapper(*args, **kwargs)
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)
    return out


def _run_single_grouped(
    wrapper,
    library,
    *,
    x_bf16: np.ndarray,
    meta: CompactMeta,
    qweight: np.ndarray,
    wmma: bool,
) -> np.ndarray:
    x_bf16 = np.ascontiguousarray(x_bf16, dtype=np.uint16)
    qweight = np.ascontiguousarray(qweight, dtype=np.uint8)
    out = np.zeros((meta.compact_rows, qweight.shape[1]), dtype=np.uint16)
    buffers: list[Any] = []
    try:
        x_buf = _device_buffer(x_bf16, buffers)
        start_buf = _device_buffer(meta.expert_start_compact, buffers)
        weight_buf = _device_buffer(qweight, buffers)
        out_buf = malloc(out.nbytes)
        buffers.append(out_buf)
        args: list[int] = [x_buf.ptr, start_buf.ptr]
        if wmma:
            wmma_start_buf = _device_buffer(meta.expert_start_wmma, buffers)
            tile_buf = _device_buffer(meta.tile_expert, buffers)
            args.extend((wmma_start_buf.ptr, tile_buf.ptr))
        args.extend((weight_buf.ptr, out_buf.ptr))
        kwargs = dict(
            compact_rows=meta.compact_rows,
            in_features=x_bf16.shape[1],
            out_features=qweight.shape[1],
            num_experts=meta.num_experts,
            library=library,
        )
        if wmma:
            kwargs["wmma_total_rows"] = meta.wmma_total_rows
        wrapper(*args, **kwargs)
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)
    return out


def _distinct_up(weight: np.ndarray, block_bytes: int) -> np.ndarray:
    out = weight.copy()
    for block_start in range(0, out.shape[-1], block_bytes):
        payload = out[..., block_start + 2 : block_start + block_bytes]
        out[..., block_start + 2 : block_start + block_bytes] = np.roll(
            payload, shift=17, axis=-1
        )
    return out


def test_iq_selected_prefill_registry_and_build_plan() -> None:
    expected = {
        ("gguf_iq3_xxs", "selected_dual_grouped_prefill_compact_bf16_bf16_out"): (
            gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        ),
        (
            "gguf_iq3_xxs",
            "selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out",
        ): gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
        (
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_bf16_bf16_out",
        ): gguf_iq3_xxs_selected_grouped_prefill_compact_bf16_bf16_out,
        (
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
        ): gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
        (
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out",
        ): gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out,
        (
            "gguf_iq3_xxs",
            "selected_dual_grouped_prefill_compact_auto_bf16_bf16_out",
        ): gguf_iq3_xxs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out,
        ("gguf_iq3_xxs", "selected_dual_wmma_prefill_compact_bf16_bf16_out"): (
            gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out
        ),
        ("gguf_iq4_xs", "selected_dual_grouped_prefill_compact_bf16_bf16_out"): (
            gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        ),
        ("gguf_iq4_xs", "selected_dual_wmma_prefill_compact_bf16_bf16_out"): (
            gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out
        ),
        ("gguf_iq4_xs", "selected_grouped_prefill_compact_bf16_bf16_out"): (
            gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out
        ),
        (
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out",
        ): gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out,
        (
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out",
        ): gguf_iq4_xs_selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out,
        ("gguf_iq4_xs", "selected_grouped_prefill_compact_auto_bf16_bf16_out"): (
            gguf_iq4_xs_selected_grouped_prefill_compact_auto_bf16_bf16_out
        ),
        ("gguf_iq4_xs", "selected_wmma_prefill_compact_bf16_bf16_out"): (
            gguf_iq4_xs_selected_wmma_prefill_compact_bf16_bf16_out
        ),
    }
    for (quant, variant), fn in expected.items():
        assert resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant=quant,
            variant=variant,
        ) is fn

    artifact = plan_gguf_iq_selected_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_iq_selected_prefill.so"
    assert artifact.profile.name == "prefill"
    assert any(path.name == "gguf_iq_selected_prefill.hip" for path in artifact.sources)
    dry_run = build_gguf_iq_selected_prefill(
        dry_run=True, compiler_version="test-compiler"
    )
    assert dry_run.output_path == artifact.output_path


def test_iq_selected_prefill_wrappers_validate_before_loading() -> None:
    scalar = dict(
        x_ptr=1,
        expert_start_compact_ptr=2,
        gate_weight_ptr=3,
        up_weight_ptr=4,
        out_ptr=5,
        compact_rows=17,
        in_features=512,
        out_features=16,
        num_experts=4,
    )
    with pytest.raises(ValueError, match="compact_rows"):
        gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
            **{**scalar, "compact_rows": 0}
        )
    with pytest.raises(ValueError, match="divisible by 256"):
        gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
            **{**scalar, "in_features": 511}
        )
    with pytest.raises(ValueError, match="at most 3072"):
        gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
            **{**scalar, "in_features": 3328}
        )
    with pytest.raises(ValueError, match="exactly 512"):
        gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out(
            scalar["x_ptr"],
            scalar["expert_start_compact_ptr"],
            scalar["gate_weight_ptr"],
            scalar["out_ptr"],
            compact_rows=scalar["compact_rows"],
            in_features=256,
            out_features=scalar["out_features"],
            num_experts=scalar["num_experts"],
        )
    for wrapper in (
        gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out,
        gguf_iq4_xs_selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out,
    ):
        with pytest.raises(ValueError, match="exactly 1024"):
            wrapper(
                scalar["x_ptr"],
                scalar["expert_start_compact_ptr"],
                scalar["gate_weight_ptr"],
                scalar["out_ptr"],
                compact_rows=scalar["compact_rows"],
                in_features=512,
                out_features=scalar["out_features"],
                num_experts=scalar["num_experts"],
            )

    wmma = dict(
        **scalar,
        expert_start_wmma_ptr=6,
        tile_expert_ptr=7,
        wmma_total_rows=32,
    )
    with pytest.raises(ValueError, match="multiple of 16"):
        gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out(
            **{**wmma, "wmma_total_rows": 31}
        )


_COUNTS = [
    pytest.param([0, 1, 15, 16, 17], id="empty-and-1-15-16-17"),
    pytest.param([16, 16, 16, 16], id="uniform-full-tiles"),
    pytest.param([0, 2, 3, 33, 5], id="hot-repeated-partial"),
]


@pytest.mark.parametrize("counts", _COUNTS)
@pytest.mark.parametrize("quant", ["gguf_iq3_xxs", "gguf_iq4_xs"])
def test_grouped_scalar_dual_is_bit_exact_to_selected_single_fallback(
    libraries, counts: list[int], quant: str
) -> None:
    grouped_library, direct_library = libraries
    meta = _compact_meta(counts)
    in_features = 512
    out_features = 19
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    if quant == "gguf_iq3_xxs":
        make_weight = _make_iq3_weight
        block_bytes = 98
        direct = gguf_iq3_xxs_selected_gemv_bf16_bf16_out
        grouped = gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        threads = 256
    else:
        make_weight = _make_iq4_weight
        block_bytes = 136
        direct = gguf_iq4_xs_selected_gemv_bf16_bf16_out
        grouped = gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        threads = 128
    gate = make_weight(meta.num_experts, out_features, in_features)
    up = _distinct_up(gate, block_bytes)
    expected_gate = _run_selected(
        direct,
        direct_library,
        x_bf16=x_bf16,
        selected=meta.selected,
        qweight=gate,
        threads=threads,
    )
    expected_up = _run_selected(
        direct,
        direct_library,
        x_bf16=x_bf16,
        selected=meta.selected,
        qweight=up,
        threads=threads,
    )
    actual = _run_dual_grouped(
        grouped,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    np.testing.assert_array_equal(actual[:, :out_features], expected_gate)
    np.testing.assert_array_equal(actual[:, out_features:], expected_up)


@pytest.mark.parametrize(
    ("compact_rows", "num_experts", "expected"),
    [(11, 3, "rt1"), (12, 3, "rowbatch4")],
)
def test_grouped_iq3_auto_uses_measured_four_rows_per_expert_crossover(
    monkeypatch: pytest.MonkeyPatch,
    compact_rows: int,
    num_experts: int,
    expected: str,
) -> None:
    calls: list[str] = []
    module = "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    monkeypatch.setattr(
        f"{module}.gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("rt1"),
    )
    monkeypatch.setattr(
        f"{module}.gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("rowbatch4"),
    )
    gguf_iq3_xxs_selected_dual_grouped_prefill_compact_auto_bf16_bf16_out(
        1,
        2,
        3,
        4,
        5,
        compact_rows=compact_rows,
        in_features=2048,
        out_features=512,
        num_experts=num_experts,
    )
    assert calls == [expected]


@pytest.mark.parametrize("in_features", [2048, 3072])
def test_grouped_iq3_rowbatch_variants_are_bit_exact_across_batch_boundaries(
    libraries, in_features: int
) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta([0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33])
    out_features = 19
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    gate = _make_iq3_weight(meta.num_experts, out_features, in_features)
    up = _distinct_up(gate, 98)
    expected = _run_dual_grouped(
        gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    actual = _run_dual_grouped(
        gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    np.testing.assert_array_equal(actual, expected)


def test_grouped_iq3_down_variants_are_bit_exact_across_batch_boundaries(
    libraries,
) -> None:
    grouped_library, direct_library = libraries
    meta = _compact_meta([0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33])
    in_features = 1024
    out_features = 23
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    qweight = _make_iq3_weight(meta.num_experts, out_features, in_features)
    expected = _run_selected(
        gguf_iq3_xxs_selected_gemv_bf16_bf16_out,
        direct_library,
        x_bf16=x_bf16,
        selected=meta.selected,
        qweight=qweight,
        threads=128,
    )
    for wrapper in (
        gguf_iq3_xxs_selected_grouped_prefill_compact_bf16_bf16_out,
        gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
    ):
        actual = _run_single_grouped(
            wrapper,
            grouped_library,
            x_bf16=x_bf16,
            meta=meta,
            qweight=qweight,
            wmma=False,
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("quant", ["gguf_iq3_xxs", "gguf_iq4_xs"])
def test_h5j_k1024_candidates_match_retained_bits_and_cpu_oracle(
    libraries, quant: str
) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta([0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65])
    in_features = 1024
    out_features = 19
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    if quant == "gguf_iq3_xxs":
        qweight = _make_iq3_weight(meta.num_experts, out_features, in_features)
        control = gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out
        candidate = gguf_iq3_xxs_selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out
        qtype = GGMLQuantizationType.IQ3_XXS
    else:
        qweight = _make_iq4_weight(meta.num_experts, out_features, in_features)
        control = gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out
        candidate = gguf_iq4_xs_selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out
        qtype = GGMLQuantizationType.IQ4_XS
    expected = _run_single_grouped(
        control,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=False,
    )
    actual = _run_single_grouped(
        candidate,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=False,
    )
    np.testing.assert_array_equal(actual, expected)

    cpu = _selected_reference(x_bf16, meta.selected, qweight, qtype)
    actual_f32 = _bf16_u16_to_f32(actual)
    cpu_f32 = _bf16_u16_to_f32(cpu)
    max_rel = float(
        np.max(np.abs(actual_f32 - cpu_f32) / np.maximum(np.abs(cpu_f32), 1.0))
    )
    assert max_rel <= 0.05
    assert evaluate_logits(cpu_f32, actual_f32).passed


@pytest.mark.parametrize("counts", _COUNTS)
def test_grouped_scalar_iq4_down_is_bit_exact_to_selected_single_fallback(
    libraries, counts: list[int]
) -> None:
    grouped_library, direct_library = libraries
    meta = _compact_meta(counts)
    in_features = 512
    out_features = 23
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    qweight = _make_iq4_weight(meta.num_experts, out_features, in_features)
    expected = _run_selected(
        gguf_iq4_xs_selected_gemv_bf16_bf16_out,
        direct_library,
        x_bf16=x_bf16,
        selected=meta.selected,
        qweight=qweight,
        threads=128,
    )
    actual = _run_single_grouped(
        gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=False,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("counts", _COUNTS)
def test_grouped_wave32_iq4_down_is_bit_exact_to_local128_fallback(
    libraries, counts: list[int]
) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta(counts)
    in_features = 512
    out_features = 23
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    qweight = _make_iq4_weight(meta.num_experts, out_features, in_features)
    expected = _run_single_grouped(
        gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=False,
    )
    actual = _run_single_grouped(
        gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=False,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("in_features", [1024, 3072])
def test_grouped_iq4_auto_keeps_local128_for_general_k(
    libraries, in_features: int
) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta([0, 2, 3, 5])
    out_features = 23
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    qweight = _make_iq4_weight(meta.num_experts, out_features, in_features)
    expected = _run_single_grouped(
        gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=False,
    )
    actual = _run_single_grouped(
        gguf_iq4_xs_selected_grouped_prefill_compact_auto_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=False,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("counts", _COUNTS)
@pytest.mark.parametrize("quant", ["gguf_iq3_xxs", "gguf_iq4_xs"])
def test_compact_wmma_dual_passes_scalar_quality_gate(
    libraries, counts: list[int], quant: str
) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta(counts)
    in_features = 512
    out_features = 32
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    if quant == "gguf_iq3_xxs":
        make_weight = _make_iq3_weight
        block_bytes = 98
        scalar = gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        wmma = gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out
    else:
        make_weight = _make_iq4_weight
        block_bytes = 136
        scalar = gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        wmma = gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out
    gate = make_weight(meta.num_experts, out_features, in_features)
    up = _distinct_up(gate, block_bytes)
    expected = _run_dual_grouped(
        scalar,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    actual = _run_dual_grouped(
        wmma,
        grouped_library,
        x_bf16=x_bf16,
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
    assert max_rel <= 0.04
    result = evaluate_logits(expected_f32, actual_f32)
    assert result.passed, result


@pytest.mark.parametrize("in_features", [2048, 3072])
@pytest.mark.parametrize("quant", ["gguf_iq3_xxs", "gguf_iq4_xs"])
def test_grouped_and_wmma_dual_cover_full_gate_up_shapes(
    libraries, quant: str, in_features: int
) -> None:
    """Exercise Qwen K=2048 and Laguna K=3072 gate/up shapes."""

    grouped_library, direct_library = libraries
    meta = _compact_meta([1, 15, 16, 17])
    out_features = 16
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    if quant == "gguf_iq3_xxs":
        make_weight = _make_iq3_weight
        block_bytes = 98
        direct = gguf_iq3_xxs_selected_gemv_bf16_bf16_out
        scalar = gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        wmma = gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out
        threads = 256
    else:
        make_weight = _make_iq4_weight
        block_bytes = 136
        direct = gguf_iq4_xs_selected_gemv_bf16_bf16_out
        scalar = gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out
        wmma = gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out
        threads = 128
    gate = make_weight(meta.num_experts, out_features, in_features)
    up = _distinct_up(gate, block_bytes)
    expected_gate = _run_selected(
        direct,
        direct_library,
        x_bf16=x_bf16,
        selected=meta.selected,
        qweight=gate,
        threads=threads,
    )
    expected_up = _run_selected(
        direct,
        direct_library,
        x_bf16=x_bf16,
        selected=meta.selected,
        qweight=up,
        threads=threads,
    )
    expected = np.concatenate((expected_gate, expected_up), axis=1)
    grouped = _run_dual_grouped(
        scalar,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
    )
    np.testing.assert_array_equal(grouped, expected)

    actual = _run_dual_grouped(
        wmma,
        grouped_library,
        x_bf16=x_bf16,
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
    max_rel_limit = 0.05 if in_features == 3072 else 0.04
    assert max_rel <= max_rel_limit
    result = evaluate_logits(expected_f32, actual_f32)
    assert result.passed, result


@pytest.mark.parametrize("counts", _COUNTS)
def test_compact_wmma_iq4_down_passes_scalar_quality_gate(
    libraries, counts: list[int]
) -> None:
    grouped_library, _ = libraries
    meta = _compact_meta(counts)
    in_features = 512
    out_features = 32
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, in_features))
    qweight = _make_iq4_weight(meta.num_experts, out_features, in_features)
    expected = _run_single_grouped(
        gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=False,
    )
    actual = _run_single_grouped(
        gguf_iq4_xs_selected_wmma_prefill_compact_bf16_bf16_out,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        qweight=qweight,
        wmma=True,
    )
    expected_f32 = _bf16_u16_to_f32(expected)
    actual_f32 = _bf16_u16_to_f32(actual)
    max_rel = float(
        np.max(np.abs(actual_f32 - expected_f32) / np.maximum(np.abs(expected_f32), 1.0))
    )
    assert max_rel <= 0.04
    result = evaluate_logits(expected_f32, actual_f32)
    assert result.passed, result
