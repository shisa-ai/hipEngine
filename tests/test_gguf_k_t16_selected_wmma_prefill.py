"""Correctness tests for selected GGUF Q4_K / Q5_K / Q6_K T16 WMMA prefill.

The Q5T16 / Q6T16 selected single-output WMMA prefill kernels consume the
T16 tile layout defined in ``hipengine/quant/gguf_t16.py`` (Q5T16 = 2880
bytes per 16-col x 256-K tile, Q6T16 = 3360 bytes). They keep the
compact-MoE scheduler ABI from P8.5 / P10.B1, just swapping raw rank-3
GGUF expert bytes for the resident tile bytes:

* x[compact_rows, in_features]
* expert_start_compact[num_experts + 1]
* expert_start_wmma[num_experts + 1]
* tile_expert[wmma_total_rows / 16]
* tiles[E, out_tiles16, blocks_per_row, tile_bytes]
* out[compact_rows, out_features]

CPU oracle is the same per-expert ``gguf_quant_gemv(..., Q5_K / Q6_K)``
chain used by the raw-Q5/Q6 selected fixture, since T16 is bit-lossless.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
    gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
    gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
    gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
    gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
    gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
    plan_gguf_k_t16_selected_prefill_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import GGUF_Q4_K_TILE16_BLOCK_BYTES, repack_gguf_q4_k_tile16
from hipengine.quant.gguf_t16 import (
    GGUF_Q5_K_T16_BLOCK_BYTES,
    GGUF_Q6_K_T16_BLOCK_BYTES,
    repack_gguf_q5_k_tile16,
    repack_gguf_q6_k_tile16,
)
from tests._gguf_synthetic_weights import make_q4_k_weight
from tests.test_gguf_k_gemv import make_q5_k_weight, make_q6_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


_QUANT_INFO: dict[str, tuple[GGMLQuantizationType, Callable[[int, int], np.ndarray], Callable[[Any], Any], int]] = {
    "gguf_q4_k_t16_v1": (
        GGMLQuantizationType.Q4_K,
        make_q4_k_weight,
        repack_gguf_q4_k_tile16,
        GGUF_Q4_K_TILE16_BLOCK_BYTES,
    ),
    "gguf_q5_k_t16_v1": (
        GGMLQuantizationType.Q5_K,
        make_q5_k_weight,
        repack_gguf_q5_k_tile16,
        GGUF_Q5_K_T16_BLOCK_BYTES,
    ),
    "gguf_q6_k_t16_v1": (
        GGMLQuantizationType.Q6_K,
        make_q6_k_weight,
        repack_gguf_q6_k_tile16,
        GGUF_Q6_K_T16_BLOCK_BYTES,
    ),
}

_WRAPPERS: dict[tuple[str, str], Any] = {
    ("gguf_q4_k_t16_v1", "bf16"): gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
    ("gguf_q4_k_t16_v1", "fp16"): gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
    ("gguf_q5_k_t16_v1", "bf16"): gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
    ("gguf_q5_k_t16_v1", "fp16"): gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
    ("gguf_q6_k_t16_v1", "bf16"): gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
    ("gguf_q6_k_t16_v1", "fp16"): gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
}


# ---------------------------------------------------------------------------
# No-GPU surface checks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant", list(_QUANT_INFO))
def test_gguf_k_t16_selected_wmma_registry_and_build_plan(
    quant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS", raising=False)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant=quant,
            variant="selected_wmma_prefill_compact_bf16_bf16_out",
        )
        is _WRAPPERS[(quant, "bf16")]
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant=quant,
            variant="selected_wmma_prefill_compact_fp16_fp16_out",
        )
        is _WRAPPERS[(quant, "fp16")]
    )

    artifact = plan_gguf_k_t16_selected_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_k_t16_selected_prefill.so"
    assert any(path.name == "gguf_k_t16_selected_prefill.hip" for path in artifact.sources)
    assert "-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS=4" not in artifact.flags

    dry_run = build_gguf_k_t16_selected_prefill(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path

    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS", "4")
    lb4 = plan_gguf_k_t16_selected_prefill_build(compiler_version="test-compiler")
    assert "-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS=4" in lb4.flags
    assert lb4.cache_key != artifact.cache_key


@pytest.mark.parametrize("wrapper", list(_WRAPPERS.values()))
def test_gguf_k_t16_selected_wmma_wrapper_validates_common_contract(wrapper: Any) -> None:
    kwargs = dict(
        x_ptr=1,
        expert_start_compact_ptr=2,
        expert_start_wmma_ptr=3,
        tile_expert_ptr=4,
        tiles_ptr=5,
        out_ptr=6,
        compact_rows=17,
        in_features=256,
        out_features=32,
        num_experts=2,
        wmma_total_rows=32,
    )

    with pytest.raises(ValueError, match="compact_rows"):
        wrapper(**{**kwargs, "compact_rows": 0})
    with pytest.raises(ValueError, match="block size 256"):
        wrapper(**{**kwargs, "in_features": 128})
    with pytest.raises(ValueError, match="out_features.*multiple of 16"):
        wrapper(**{**kwargs, "out_features": 24})
    with pytest.raises(ValueError, match="num_experts"):
        wrapper(**{**kwargs, "num_experts": 0})
    with pytest.raises(ValueError, match="wmma_total_rows.*multiple of 16"):
        wrapper(**{**kwargs, "wmma_total_rows": 31})


# ---------------------------------------------------------------------------
# Compact selected-MoE fixture helpers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactT16Fixture:
    quant: str
    dtype: str
    x_host: np.ndarray
    expert_start_compact: np.ndarray
    expert_start_wmma: np.ndarray
    tile_expert: np.ndarray
    qweight: np.ndarray
    tiles: np.ndarray
    reference: np.ndarray
    compact_rows: int
    wmma_total_rows: int
    in_features: int
    out_features: int
    num_experts: int


def _float_array_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _make_activation(rows: int, in_features: int, seed: int = 0) -> np.ndarray:
    values = (
        np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features)
        + seed
    )
    return ((values % 13) - 6) / 64.0


def _prepare_input(x_f32: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return _float_array_to_bf16_bits(x_f32)
    if dtype == "fp16":
        return x_f32.astype(np.float16)
    raise ValueError(dtype)


def _decode_input_for_reference(host: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        decoded = _bf16_bits_to_float32(host)
    elif dtype == "fp16":
        decoded = host.astype(np.float32)
    else:
        raise ValueError(dtype)
    return decoded.astype(np.float16).astype(np.float32)


def _decode_output(host: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return _bf16_bits_to_float32(host)
    if dtype == "fp16":
        return host.astype(np.float32)
    raise ValueError(dtype)


def _make_expert_weights(
    *, quant: str, num_experts: int, out_features: int, in_features: int
) -> np.ndarray:
    _, make_weight, _, _ = _QUANT_INFO[quant]
    base = make_weight(out_features, in_features)
    return np.ascontiguousarray(
        np.stack([np.roll(base, shift=expert, axis=0) for expert in range(num_experts)], axis=0)
    )


def _build_compact_t16_fixture(
    *,
    quant: str,
    counts: list[int],
    in_features: int,
    out_features: int,
    dtype: str,
    seed: int = 0,
) -> CompactT16Fixture:
    qtype, _, repack, tile_bytes = _QUANT_INFO[quant]
    num_experts = len(counts)
    compact_rows = int(sum(counts))
    assert compact_rows > 0
    expert_start_compact = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start_compact[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))

    padded_counts = [((count + 15) // 16) * 16 for count in counts]
    expert_start_wmma = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start_wmma[1:] = np.cumsum(np.asarray(padded_counts, dtype=np.int64))
    wmma_total_rows = int(expert_start_wmma[-1])
    tile_expert = np.asarray(
        [expert for expert, padded in enumerate(padded_counts) for _ in range(padded // 16)],
        dtype=np.int64,
    )
    assert tile_expert.size == wmma_total_rows // 16

    x_f32 = _make_activation(compact_rows, in_features, seed=seed)
    x_host = _prepare_input(x_f32, dtype)
    x_ref = _decode_input_for_reference(x_host, dtype)
    qweight = _make_expert_weights(
        quant=quant,
        num_experts=num_experts,
        out_features=out_features,
        in_features=in_features,
    )
    tiles = repack(qweight).tiles
    assert tiles.shape[-1] == tile_bytes

    reference = np.zeros((compact_rows, out_features), dtype=np.float32)
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        start = int(expert_start_compact[expert])
        stop = start + count
        reference[start:stop] = gguf_quant_gemv(x_ref[start:stop], qweight[expert], qtype)

    return CompactT16Fixture(
        quant=quant,
        dtype=dtype,
        x_host=np.ascontiguousarray(x_host),
        expert_start_compact=expert_start_compact,
        expert_start_wmma=expert_start_wmma,
        tile_expert=tile_expert,
        qweight=qweight,
        tiles=np.ascontiguousarray(tiles),
        reference=reference,
        compact_rows=compact_rows,
        wmma_total_rows=wmma_total_rows,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
    )


def _run_selected_t16_gpu(fixture: CompactT16Fixture) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_k_t16_selected_prefill(load=True)
    out_dtype = np.uint16 if fixture.dtype == "bf16" else np.float16
    host_out = np.zeros((fixture.compact_rows, fixture.out_features), dtype=out_dtype)
    wrapper = _WRAPPERS[(fixture.quant, fixture.dtype)]

    bufs = []
    try:
        x_dev = malloc(fixture.x_host.nbytes, runtime=runtime)
        start_compact_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_wmma_dev = malloc(fixture.expert_start_wmma.nbytes, runtime=runtime)
        tile_expert_dev = malloc(fixture.tile_expert.nbytes, runtime=runtime)
        tiles_dev = malloc(fixture.tiles.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend(
            (
                x_dev,
                start_compact_dev,
                start_wmma_dev,
                tile_expert_dev,
                tiles_dev,
                out_dev,
            )
        )
        for dev, arr in (
            (x_dev, fixture.x_host),
            (start_compact_dev, fixture.expert_start_compact),
            (start_wmma_dev, fixture.expert_start_wmma),
            (tile_expert_dev, fixture.tile_expert),
            (tiles_dev, fixture.tiles),
        ):
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime)

        wrapper(
            x_dev.ptr,
            start_compact_dev.ptr,
            start_wmma_dev.ptr,
            tile_expert_dev.ptr,
            tiles_dev.ptr,
            out_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            fixture.out_features,
            fixture.num_experts,
            fixture.wmma_total_rows,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    return _decode_output(host_out, fixture.dtype)


_SELECTED_CASES = [
    pytest.param([4, 0, 5], 256, 16, id="empty-middle-out16"),
    pytest.param([16, 17, 31], 256, 32, id="exact-plus-padding-out32"),
    pytest.param([0, 33, 1, 16], 512, 64, id="empty-first-multi-block-out64"),
    pytest.param([7, 18, 0, 33], 512, 32, id="empty-third-aligned-out32"),
    pytest.param([32, 0, 0, 17], 768, 48, id="multi-block-empty-tail-out48"),
]

_TOLERANCES = {
    ("gguf_q4_k_t16_v1", "bf16"): {"rtol": 2.0e-2, "atol": 5.0e-1},
    ("gguf_q4_k_t16_v1", "fp16"): {"rtol": 7.5e-3, "atol": 1.5e-1},
    ("gguf_q5_k_t16_v1", "bf16"): {"rtol": 2.0e-2, "atol": 5.0e-1},
    ("gguf_q6_k_t16_v1", "bf16"): {"rtol": 1.2e-2, "atol": 3.0e-1},
    ("gguf_q5_k_t16_v1", "fp16"): {"rtol": 7.5e-3, "atol": 1.5e-1},
    ("gguf_q6_k_t16_v1", "fp16"): {"rtol": 6.0e-3, "atol": 1.0e-1},
}


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("quant", ["gguf_q4_k_t16_v1", "gguf_q5_k_t16_v1", "gguf_q6_k_t16_v1"])
@pytest.mark.parametrize(("counts", "in_features", "out_features"), _SELECTED_CASES)
def test_p10_b2_b3_k_t16_selected_wmma_bf16_matches_cpu_selected_reference(
    quant: str, counts: list[int], in_features: int, out_features: int
) -> None:
    fixture = _build_compact_t16_fixture(
        quant=quant,
        counts=counts,
        in_features=in_features,
        out_features=out_features,
        dtype="bf16",
    )
    actual = _run_selected_t16_gpu(fixture)
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCES[(quant, "bf16")])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("quant", "counts", "in_features", "out_features"),
    [
        pytest.param("gguf_q4_k_t16_v1", [5, 11, 0, 23], 256, 48, id="q4-fp16-uneven-out48"),
        pytest.param("gguf_q4_k_t16_v1", [0, 16, 17], 512, 64, id="q4-fp16-empty-first-out64"),
        pytest.param("gguf_q5_k_t16_v1", [5, 11, 0, 23], 256, 48, id="q5-fp16-uneven-out48"),
        pytest.param("gguf_q5_k_t16_v1", [0, 16, 17], 512, 64, id="q5-fp16-empty-first-out64"),
        pytest.param("gguf_q6_k_t16_v1", [5, 11, 0, 23], 256, 48, id="q6-fp16-uneven-out48"),
        pytest.param("gguf_q6_k_t16_v1", [0, 16, 17], 512, 64, id="q6-fp16-empty-first-out64"),
    ],
)
def test_p10_b2_b3_k_t16_selected_wmma_fp16_matches_cpu_selected_reference(
    quant: str, counts: list[int], in_features: int, out_features: int
) -> None:
    fixture = _build_compact_t16_fixture(
        quant=quant,
        counts=counts,
        in_features=in_features,
        out_features=out_features,
        dtype="fp16",
        seed=5,
    )
    actual = _run_selected_t16_gpu(fixture)
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCES[(quant, "fp16")])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("quant", ["gguf_q4_k_t16_v1", "gguf_q5_k_t16_v1", "gguf_q6_k_t16_v1"])
def test_p10_b2_b3_k_t16_selected_wmma_runs_exported_bf16_symbol_on_w7900(quant: str) -> None:
    """Tiny launches confirm both T16 selected WMMA kernel instantiations run."""

    fixture = _build_compact_t16_fixture(
        quant=quant,
        counts=[1, 0],
        in_features=256,
        out_features=16,
        dtype="bf16",
        seed=9,
    )
    actual = _run_selected_t16_gpu(fixture)
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCES[(quant, "bf16")])
