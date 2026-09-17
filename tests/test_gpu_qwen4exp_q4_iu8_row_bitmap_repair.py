"""The row-bitmap repair must be bit-identical to the per-slot repair.

``gguf_q4_k_selected_dual_sparse_exact_repair_bf16`` is the shipped exact
repair for the promoted ``q4_iu8_exact`` gate/up route: one 128-thread block
per queued output element, recomputing that element with the grouped pair2
parent's arithmetic. It is correct but reads one full activation row per
element, which at the measured 0.712% repair rate is 78% of its DRAM traffic
(``benchmarks/results/2026-09-17-qwen4exp-iu8-repair-cost-structure``).

``gguf_q4_k_selected_dual_row_bitmap_repair_bf16`` repairs the same elements
with the same instructions, but stages each compact row's activations in LDS
once and walks that row's set bits. The arithmetic must be unchanged: the
whole point of the route is that the repaired value *is* the parent's value,
not a close one. These tests pin that with a realistic queue on real Q4_K
blocks -- bit comparison, not tolerance -- and pin the traffic claim's
preconditions (the bitmap is idempotent and a row is touched once).

RED before the kernels existed: the symbols are absent and the fixture
comparison cannot run.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hipengine.core.hip")

REPO_ROOT = Path(__file__).resolve().parents[1]

Q4_K_BLOCK_BYTES = 144
QK_K = 256
EXPERTS = 8
IN_FEATURES = 512
OUT_FEATURES = 128  # per tensor; the dual output is 256 wide
ROWS = 96
SEED = 20260917


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def _q4_k_weights(rng: np.random.Generator, experts: int, out_features: int, in_features: int):
    row_bytes = (in_features // QK_K) * Q4_K_BLOCK_BYTES
    raw = rng.integers(0, 256, size=(experts, out_features, row_bytes), dtype=np.uint8)
    # Keep the fp16 d/dmin fields in a normal range so the fixture exercises
    # real arithmetic instead of denormal soup.
    words = raw.view(np.uint16)
    words[..., 0] = (rng.integers(1, 64, size=words[..., 0].shape) * 256).astype(
        np.uint16
    )
    words[..., 1] = (rng.integers(1, 64, size=words[..., 1].shape) * 256).astype(
        np.uint16
    )
    return np.ascontiguousarray(raw)


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(SEED)
    row_bytes = (IN_FEATURES // QK_K) * Q4_K_BLOCK_BYTES
    weights_a = _q4_k_weights(rng, EXPERTS, OUT_FEATURES, IN_FEATURES)
    weights_b = _q4_k_weights(rng, EXPERTS, OUT_FEATURES, IN_FEATURES)
    # Activations: bf16-representable values, which is what the route stages.
    x = (rng.standard_normal((ROWS, IN_FEATURES)) * 0.75).astype(np.float32)
    x_bf16 = (
        x.astype(np.float16).view(np.uint16).astype(np.uint32) << 16
    ).astype(np.uint32).view(np.float32).astype(np.float32)
    x_bits = x_bf16.view(np.uint32) >> 16
    # Expert rows: consecutive blocks per expert, the layout the tile map makes.
    per_expert = ROWS // EXPERTS
    expert_start = np.arange(EXPERTS + 1, dtype=np.int64) * per_expert
    return {
        "row_bytes": row_bytes,
        "weights_a": weights_a,
        "weights_b": weights_b,
        "x_bits": x_bits.astype(np.uint16),
        "expert_start": expert_start,
    }


def _risk_queue(rng: np.random.Generator, count: int, *, clustered: bool):
    total = OUT_FEATURES * 2
    if not clustered:
        rows = rng.integers(0, ROWS, size=count)
        cols = rng.integers(0, total, size=count)
    else:
        rows = rng.integers(0, max(1, ROWS // 8), size=count)
        cols = rng.integers(0, total, size=count)
    return np.ascontiguousarray(rows * total + cols, dtype=np.int32)


def _run_both(queue: np.ndarray, fixture: dict, *, grid_blocks: int = 64):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as gu

    gu.build_gguf_q4_k_selected_prefill(load=True)
    runtime = get_hip_runtime()

    total = OUT_FEATURES * 2
    capacity = max(1, int(queue.size))
    weights_a = fixture["weights_a"]
    weights_b = fixture["weights_b"]
    # A repaired value can legitimately be zero, so both outputs start at a
    # sentinel and ownership is asserted against it rather than against zero.
    sentinel = np.uint16(0x7FFF)

    buffers = []
    try:
        def dev(array: np.ndarray):
            buf = malloc(array.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=runtime)
            return buf

        x_buf = dev(fixture["x_bits"])
        a_buf = dev(weights_a)
        b_buf = dev(weights_b)
        starts = dev(fixture["expert_start"])
        count_buf = dev(np.asarray([queue.size], dtype=np.int32))
        queue_buf = dev(queue if queue.size else np.zeros(1, dtype=np.int32))
        per_slot_out = dev(np.full(ROWS * total, sentinel, dtype=np.uint16))
        bitmap_out = dev(np.full(ROWS * total, sentinel, dtype=np.uint16))
        words = gu.risk_bitmap_words(total)
        bitmap = dev(np.zeros(ROWS * words, dtype=np.uint32))

        gu.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
            x_buf.ptr,
            starts.ptr,
            a_buf.ptr,
            b_buf.ptr,
            per_slot_out.ptr,
            count_buf.ptr,
            queue_buf.ptr,
            capacity,
            ROWS,
            IN_FEATURES,
            OUT_FEATURES,
            OUT_FEATURES,
            EXPERTS,
            grid_blocks=grid_blocks,
            runtime=runtime,
        )
        gu.gguf_q4_k_selected_dual_risk_bitmap(
            count_buf.ptr,
            queue_buf.ptr,
            bitmap.ptr,
            capacity,
            ROWS,
            total,
            runtime=runtime,
        )
        gu.gguf_q4_k_selected_dual_row_bitmap_repair_bf16(
            x_buf.ptr,
            starts.ptr,
            a_buf.ptr,
            b_buf.ptr,
            bitmap_out.ptr,
            bitmap.ptr,
            ROWS,
            IN_FEATURES,
            OUT_FEATURES,
            OUT_FEATURES,
            EXPERTS,
            runtime=runtime,
        )
        runtime.device_synchronize()

        def read(buf):
            host = np.zeros(ROWS * total, dtype=np.uint16)
            copy_device_to_host(host_array_ptr(host), buf, host.nbytes, runtime=runtime)
            return host

        return read(per_slot_out), read(bitmap_out), sentinel
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


def _unique_targets(queue: np.ndarray) -> np.ndarray:
    """The set of elements the queue actually repairs, deduplicated."""

    return np.unique(np.asarray(queue, dtype=np.int64))


@pytest.mark.parametrize("count", [1, 37, 512, 4096])
def test_row_bitmap_repair_is_bit_identical(fixture, count: int) -> None:
    rng = np.random.default_rng(SEED + count)
    queue = _risk_queue(rng, count, clustered=False)
    per_slot, bitmap, sentinel = _run_both(queue, fixture)

    targets = _unique_targets(queue)
    assert np.array_equal(per_slot[targets], bitmap[targets]), (
        "row-bitmap repair changed a repaired value"
    )
    # Both kernels own exactly the queued elements: every other element still
    # holds the sentinel, in both buffers.
    untouched = np.setdiff1d(np.arange(per_slot.size), targets, assume_unique=False)
    assert np.all(per_slot[untouched] == sentinel)
    assert np.all(bitmap[untouched] == sentinel), "bitmap repair wrote an unqueued element"


def test_row_bitmap_repair_is_bit_identical_when_clustered(fixture) -> None:
    """Rows carrying most of the queue are the case the bucketing exists for."""

    rng = np.random.default_rng(SEED + 11)
    queue = _risk_queue(rng, 2048, clustered=True)
    per_slot, bitmap, _ = _run_both(queue, fixture)
    targets = _unique_targets(queue)
    assert np.array_equal(per_slot[targets], bitmap[targets])


def test_row_bitmap_repair_is_idempotent_under_duplicates(fixture) -> None:
    """A duplicated queue entry must not change the value or the bitmap."""

    rng = np.random.default_rng(SEED + 7)
    queue = _risk_queue(rng, 256, clustered=False)
    duplicated = np.ascontiguousarray(np.concatenate([queue, queue]), dtype=np.int32)
    per_slot, bitmap, _ = _run_both(duplicated, fixture)
    _, once, _ = _run_both(queue, fixture)
    targets = _unique_targets(queue)
    assert np.array_equal(bitmap, once)
    assert np.array_equal(per_slot[targets], bitmap[targets])


def test_bitmap_layout_is_pinned(fixture) -> None:
    """The bitmap is 32-bit words per row, and the launch reads that layout."""

    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as gu

    assert gu.risk_bitmap_words(256) == 8
    assert gu.risk_bitmap_words(1) == 1
    assert gu.risk_bitmap_words(33) == 2
    assert gu.risk_bitmap_bytes(ROWS, OUT_FEATURES * 2) == ROWS * 8 * 4
    for bad in (0, -1):
        with pytest.raises(ValueError):
            gu.risk_bitmap_words(bad)
        with pytest.raises(ValueError):
            gu.risk_bitmap_bytes(bad, 256)


def test_wrappers_reject_missing_buffers(fixture) -> None:
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as gu

    with pytest.raises(ValueError):
        gu.gguf_q4_k_selected_dual_row_bitmap_repair_bf16(
            1, 1, 1, 1, 1, 0, ROWS, IN_FEATURES, OUT_FEATURES, OUT_FEATURES, EXPERTS
        )
    with pytest.raises(ValueError):
        gu.gguf_q4_k_selected_dual_row_bitmap_repair_bf16(
            1, 1, 1, 1, 1, 1, ROWS, 0, OUT_FEATURES, OUT_FEATURES, EXPERTS
        )
    with pytest.raises(ValueError):
        gu.gguf_q4_k_selected_dual_risk_bitmap(0, 1, 1, 16, ROWS, 256)


def test_kernels_are_registered() -> None:
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as gu
    from hipengine.kernels.registry import can_resolve

    # The session conftest restores a baseline registry per test, so re-run the
    # module's own registration rather than relying on import order.
    gu.register_gguf_q4_k_selected_prefill_kernels(replace=True)

    for variant in (
        "selected_dual_row_bitmap_repair_bf16",
        "selected_dual_risk_bitmap",
        "selected_dual_sparse_exact_repair_bf16",
    ):
        assert can_resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant=variant,
        ), variant


def test_cost_structure_artifact_pins_the_traffic_claim() -> None:
    """The design note's numbers come from the retained artifact."""

    artifact = json.loads(
        (
            REPO_ROOT
            / "benchmarks/results/2026-09-17-qwen4exp-iu8-repair-cost-structure"
            / "artifact.json"
        ).read_text()
    )
    gate = artifact["grids"]["gate_up"]
    shipped = [e for e in gate if e["grid"] == 17920 and e["risks"] == 93342]
    assert shipped, "the shipped grid and the measured incidence must both be present"
    entry = shipped[0]
    # Bandwidth-bound, not latency-bound: a 3.7x larger grid does not help.
    bigger = [e for e in gate if e["grid"] == 65536 and e["risks"] == 93342][0]
    assert abs(bigger["ms"] - entry["ms"]) / entry["ms"] < 0.10
    assert entry["gb_per_s"] > 150.0
    # The activation row dominates the per-element traffic.
    shape = artifact["shapes"]["gate_up"]
    assert shape["activation_bytes_per_element"] > 3 * shape["weight_bytes_per_element"]
