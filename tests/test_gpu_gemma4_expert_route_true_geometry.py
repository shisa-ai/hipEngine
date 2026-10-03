"""The MMQ gate/up route at the true model geometry, not a toy stand-in.

``tests/test_unit_gemma4_expert_route.py`` covers the route's tile walk over 128
experts, but at ``in_features=256`` and ``intermediate=64``. The real projection
is ``2816 -> 2 * 704`` per expert, which is 285 MB of Q4_K per layer - the figure
the prefill profile attributes a third of prefill device time to. A route can hold
at a toy geometry and fail at the real one whenever a stride, a block count or a
column tile is derived from a shape that the toy makes trivial: ``256`` input
features is one Q4_K block per row and ``88`` is a full row of Q8_1 blocks, while
the real row is eleven Q4_K blocks and 88 Q8_1 blocks with an odd block count.

The route also quantizes activations to DS4 Q8_1, which derives one int8 scale per
128-element group. A single large value inside a group therefore sets the scale for
the whole group and crushes its neighbours toward zero. The unit tests use
unit-variance gaussian rows, which have no such value; real hidden states do. So
this file holds the route at the true shape under two activation profiles: the
gaussian rows the rest of the coverage uses, and rows carrying the outlier
channels and near-zero rows a real short prefill produces. A gap that opens only
in the second profile is the quantization, not the kernel.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    Gemma4ExpertScratch,
    gemma4_moe_expert_route_counts,
    gemma4_project_experts_down_mmq,
    gemma4_project_experts_gate_up_mmq,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q4_k_weight
from tests._rocm_guard import hip_runtime_available

_needs_hip = pytest.mark.skipif(
    not hip_runtime_available(),
    reason="requires the HIP runtime and a gfx1100/gfx1151 device",
)

# The artifact's real projection: 2816 input features, 704 expert intermediate,
# 128 experts, top-8 routing.
_IN_FEATURES = 2816
_INTERMEDIATE = 704
_NUM_EXPERTS = 128

# Counts a real router produces at a short prefill: every expert hit, most of
# them with fewer rows than one 32-row MMQ tile, and a few empty. Note the
# largest count is exactly 32, i.e. exactly one tile, so this pattern never
# makes an expert span two tiles.
_SHORT_PREFILL = np.asarray(
    [0, 1, 2, 3, 5, 8, 13, 21, 2, 1, 4, 7, 11, 17, 31, 32] * 8, dtype=np.int64
)


def _skewed_prefill(lanes: int, seed: int) -> np.ndarray:
    """Counts a real router produces: a heavy head and a long light tail.

    Router weights are close to a power law, so at a given lane count a few
    experts take far more than the mean while most take a handful. With 128
    experts and 512 lanes the mean is 4 rows per expert, which is well under one
    32-row tile, so a uniform or mildly fragmented pattern never exercises an
    expert whose rows cross a tile boundary. The head here is what does.
    """

    rng = np.random.default_rng(seed)
    rank = np.arange(_NUM_EXPERTS, dtype=np.float64) + 1.0
    weights = rank**-1.25
    return rng.multinomial(lanes, weights / weights.sum()).astype(np.int64)


# The lane count a 64-id prefill with top_k 8 produces, which is a length the
# real probe diverges at.
_SKEWED_LANES = 512

# The envelope the MMQ family already asserts against its strict fp32 owner.
_MAX_ENVELOPE = 2e-2
_MEAN_ENVELOPE = 2e-3


def _to_bf16_bits(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _activations(profile: str, rows: int, seed: int) -> np.ndarray:
    """Rows of the shape a real short prefill feeds the expert projection."""

    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((rows, _IN_FEATURES)).astype(np.float32)
    if profile == "gaussian":
        return hidden
    if profile != "outlier":
        raise ValueError(f"unknown activation profile {profile!r}")
    # Persistent outlier channels: a handful of feature dimensions carry a value
    # orders of magnitude above the row's median, which is the pattern Gemma's
    # large activations are recorded as having. Each one sits inside a different
    # 128-element Q8_1 group, so it sets that group's scale.
    for offset, magnitude in ((0, 420.0), (17, 260.0), (64, 900.0), (191, 130.0)):
        hidden[::3, offset] = magnitude * np.sign(hidden[::3, offset] + 1e-3)
    # A few rows near the bottom of the range rather than centred on it, so the
    # profile also exercises a group whose scale comes out very small.
    hidden[-4:-1] *= 1e-3
    return hidden


@pytest.fixture(scope="module")
def _weights() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One fused ``(num_experts, 2 * intermediate, hidden)`` Q4_K block.

    Module-scoped because building and dequantizing 285 MB of synthetic Q4_K is
    the expensive part of this file, and both activation profiles read the same
    weights.
    """

    gate_raw = np.concatenate(
        [make_q4_k_weight(_INTERMEDIATE, _IN_FEATURES) for _ in range(_NUM_EXPERTS)],
        axis=0,
    )
    up_raw = np.concatenate(
        [make_q4_k_weight(_INTERMEDIATE, _IN_FEATURES) for _ in range(_NUM_EXPERTS)],
        axis=0,
    )
    # Make the two halves distinguishable so a wrong fused stride cannot pass by
    # reading the right values out of the other half.
    up_raw = up_raw.copy()
    up_raw[:, 1::2] ^= np.uint8(0x04)
    fused_raw = np.concatenate(
        [
            gate_raw.reshape(_NUM_EXPERTS, _INTERMEDIATE, -1),
            up_raw.reshape(_NUM_EXPERTS, _INTERMEDIATE, -1),
        ],
        axis=1,
    ).reshape(_NUM_EXPERTS * 2 * _INTERMEDIATE, -1)
    return fused_raw, gate_raw, up_raw


def _reference(
    hidden_bits: np.ndarray,
    gate_raw: np.ndarray,
    up_raw: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    from hipengine.quant.gguf import dequantize_gguf_data

    rounded = _from_bf16_bits(hidden_bits)
    expected = np.zeros((int(counts.sum()), 2 * _INTERMEDIATE), dtype=np.float32)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        gate = np.asarray(
            dequantize_gguf_data(
                gate_raw[expert * _INTERMEDIATE : (expert + 1) * _INTERMEDIATE],
                GGMLQuantizationType.Q4_K,
            ),
            dtype=np.float32,
        )
        up = np.asarray(
            dequantize_gguf_data(
                up_raw[expert * _INTERMEDIATE : (expert + 1) * _INTERMEDIATE],
                GGMLQuantizationType.Q4_K,
            ),
            dtype=np.float32,
        )
        block = rounded[start : start + count]
        expected[start : start + count, :_INTERMEDIATE] = block @ gate.T
        expected[start : start + count, _INTERMEDIATE:] = block @ up.T
        start += int(count)
    assert start == int(counts.sum())
    return expected


def _run_route(
    fused_raw: np.ndarray,
    hidden_bits: np.ndarray,
    counts: np.ndarray,
    capacity_factor: int = 1,
    *,
    tiles: bool = False,
):
    """Run the route with the live row count and a scratch of its own width.

    ``capacity_factor`` sizes the scratch above the live lane count, which is the
    documented way it is used: the caller sizes one scratch for the widest block
    it will run and then runs narrower blocks through it. A factor of 1 makes
    capacity and live width equal, which is what the rest of the coverage does.

    ``tiles`` gives the weight the Q4T16 gate and up allocations that the T16 WMMA
    route reads, which is how the loader hands a stacked Q4_K expert tensor to it
    in production. Without them the route declines and the DS4 MMQ32 route serves
    the call, so both arms are reachable from this one harness.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

    rows = int(counts.sum())
    fused_width = 2 * _INTERMEDIATE
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    hidden_buf = malloc(hidden_bits.nbytes)
    weights_buf = malloc(fused_raw.nbytes)
    starts_buf = malloc(starts.nbytes)
    out_buf = malloc(rows * fused_width * 2)
    tile_bufs: list = []
    scratch = None
    try:
        copy_host_array_to_device(hidden_buf, hidden_bits)
        copy_host_array_to_device(weights_buf, fused_raw)
        copy_host_array_to_device(starts_buf, starts)

        allocations = {
            "raw": SimpleNamespace(buffer=SimpleNamespace(ptr=weights_buf.ptr))
        }
        if tiles:
            stacked = fused_raw.reshape(_NUM_EXPERTS, fused_width, -1)
            for name, start in (("t16_gate", 0), ("t16_up", _INTERMEDIATE)):
                packed = repack_gguf_q4_k_tile16(
                    stacked[:, start : start + _INTERMEDIATE, :]
                ).tiles
                buf = malloc(packed.nbytes)
                tile_bufs.append(buf)
                copy_host_array_to_device(buf, packed)
                allocations[name] = SimpleNamespace(
                    buffer=SimpleNamespace(ptr=buf.ptr)
                )

        weight = SimpleNamespace(
            spec=SimpleNamespace(quant_key="gguf_q4_k"),
            allocations=allocations,
            allocation=lambda name: allocations[name],
        )
        scratch = Gemma4ExpertScratch(
            tokens=rows * capacity_factor,
            top_k=1,
            hidden_size=_IN_FEATURES,
            intermediate=_INTERMEDIATE,
            num_experts=_NUM_EXPERTS,
        )
        served = gemma4_project_experts_gate_up_mmq(
            weight,
            hidden_buf.ptr,
            out_buf.ptr,
            SimpleNamespace(ptr=starts_buf.ptr),
            rows,
            _NUM_EXPERTS,
            _IN_FEATURES,
            _INTERMEDIATE,
            scratch=scratch,
        )
        assert served is True, "the fused MMQ gate_up route declined a Q4_K weight"

        got = np.empty((rows, fused_width), dtype=np.uint16)
        copy_device_to_host(
            int(got.ctypes.data),
            DeviceBuffer(ptr=out_buf.ptr, nbytes=got.nbytes),
            got.nbytes,
        )
        return _from_bf16_bits(got)
    finally:
        if scratch is not None:
            scratch.free()
        for buffer in (hidden_buf, weights_buf, starts_buf, out_buf, *tile_bufs):
            free(buffer)


@_needs_hip
@pytest.mark.parametrize("profile", ["gaussian", "outlier"])
@pytest.mark.parametrize("capacity_factor", [1, 8])
@pytest.mark.parametrize("counts_kind", ["fragmented", "skewed"])
def test_fused_gate_up_mmq_route_holds_at_the_true_projection_geometry(
    _weights: tuple[np.ndarray, np.ndarray, np.ndarray],
    profile: str,
    capacity_factor: int,
    counts_kind: str,
) -> None:
    fused_raw, gate_raw, up_raw = _weights
    if counts_kind == "fragmented":
        counts = _SHORT_PREFILL
    else:
        counts = _skewed_prefill(_SKEWED_LANES, 20260927)
        # The point of this pattern is an expert spanning several tiles. If the
        # head is not heavy enough to cross a tile boundary the case is not
        # being exercised at all, so fail loudly rather than pass vacuously.
        assert counts.max() > 32, (
            f"skewed pattern peaked at {counts.max()} rows, which is one 32-row "
            "tile: it does not exercise a multi-tile expert"
        )
    assert len(counts) == _NUM_EXPERTS
    rows = int(counts.sum())

    hidden_bits = _to_bf16_bits(_activations(profile, rows, 20260927))
    expected = _reference(hidden_bits, gate_raw, up_raw, counts)
    got = _run_route(fused_raw, hidden_bits, counts, capacity_factor)

    scale = float(np.abs(expected).max())
    assert scale > 0
    difference = np.abs(got - expected)
    normalized_max = float(difference.max()) / scale
    normalized_mean = float(difference.mean()) / scale
    worst_row = int(np.argmax(difference.max(axis=1)))
    context = (
        f"profile {profile!r}, {counts_kind} counts at the true geometry with "
        f"capacity factor {capacity_factor} (live {rows} rows, peak "
        f"{int(counts.max())} rows on one expert): normalized max "
        f"{normalized_max:.4g} at compact row {worst_row}, normalized mean "
        f"{normalized_mean:.4g}, against scale {scale:.4g}"
    )
    assert normalized_max < _MAX_ENVELOPE, (
        f"fused MMQ gate_up exceeded the envelope ({context})"
    )
    assert normalized_mean < _MEAN_ENVELOPE, (
        f"fused MMQ gate_up exceeded the envelope ({context})"
    )


@_needs_hip
@pytest.mark.parametrize("profile", ["gaussian", "outlier"])
@pytest.mark.parametrize("counts_kind", ["fragmented", "skewed"])
def test_fused_gate_up_t16_route_holds_at_the_true_projection_geometry(
    _weights: tuple[np.ndarray, np.ndarray, np.ndarray],
    profile: str,
    counts_kind: str,
) -> None:
    """The Q4T16 WMMA gate_up route holds the same envelope as the MMQ32 one.

    This is the route the loader hands a stacked Q4_K expert tensor to once it
    carries the Q4T16 gate and up allocations, and it is a different arithmetic
    path: it quantizes no activations, so its error is the bf16 input rounding
    and the Q4_K weight rounding rather than an activation step. A route that
    quantizes nothing must not be the less accurate of the two, so it is held to
    the same envelope rather than a looser one.
    """

    fused_raw, gate_raw, up_raw = _weights
    counts = (
        _SHORT_PREFILL if counts_kind == "fragmented"
        else _skewed_prefill(_SKEWED_LANES, 20260927)
    )
    assert len(counts) == _NUM_EXPERTS
    rows = int(counts.sum())

    hidden_bits = _to_bf16_bits(_activations(profile, rows, 20260927))
    expected = _reference(hidden_bits, gate_raw, up_raw, counts)
    before = gemma4_moe_expert_route_counts().get("gate_up_t16", 0)
    got = _run_route(fused_raw, hidden_bits, counts, tiles=True)
    # Without this the arm passes whenever *either* route serves the call, and the
    # MMQ32 arm would satisfy it on its own.
    after = gemma4_moe_expert_route_counts().get("gate_up_t16", 0)
    assert after == before + 1, (
        "the Q4T16 route did not serve the call; the tiled weight fell through to "
        "another route, so this arm does not cover the route it names"
    )

    scale = float(np.abs(expected).max())
    assert scale > 0
    difference = np.abs(got - expected)
    normalized_max = float(difference.max()) / scale
    normalized_mean = float(difference.mean()) / scale
    worst_row = int(np.argmax(difference.max(axis=1)))
    context = (
        f"profile {profile!r}, {counts_kind} counts at the true geometry with "
        f"Q4T16 tiles (live {rows} rows, peak {int(counts.max())} rows on one "
        f"expert): normalized max {normalized_max:.4g} at compact row "
        f"{worst_row}, normalized mean {normalized_mean:.4g}, against scale "
        f"{scale:.4g}"
    )
    assert normalized_max < _MAX_ENVELOPE, (
        f"fused T16 gate_up exceeded the envelope ({context})"
    )
    assert normalized_mean < _MEAN_ENVELOPE, (
        f"fused T16 gate_up exceeded the envelope ({context})"
    )


# The scratch is documented as "a capacity, not an identity: the caller sizes it
# for the widest block it will run and then runs narrower blocks through it."
# The cases below are that property, which the envelope cases above do not
# exercise: they hold the live width and the capacity equal (factor 1), so a
# route that read its capacity instead of its live width would pass them all.
#
# The real call always runs with capacity far above live -- the expert scratch is
# sized for the decode block's 64 tokens times top_k, while a prefill chunk or a
# speculative verify runs a handful of lanes through it -- so this is the regime
# production is actually in.

_EQUAL_WIDTH_COUNTS = np.ones(_NUM_EXPERTS, dtype=np.int64)
_ONE_WIDER_COUNTS = np.concatenate(
    [np.asarray([2], dtype=np.int64), np.ones(_NUM_EXPERTS - 1, dtype=np.int64)]
)


def test_mmq_route_does_not_read_its_scratch_capacity(
    _weights: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Widening the scratch must not change the result for the same live rows."""

    fused_raw, _, _ = _weights
    counts = _EQUAL_WIDTH_COUNTS
    rows = int(counts.sum())
    hidden_bits = _to_bf16_bits(_activations("gaussian", rows, 20260927))

    narrow = _run_route(fused_raw, hidden_bits, counts, capacity_factor=1)
    wide = _run_route(fused_raw, hidden_bits, counts, capacity_factor=8)

    difference = float(np.abs(narrow - wide).max())
    scale = float(np.abs(narrow).max())
    assert difference == 0.0, (
        "the MMQ gate_up route read its scratch capacity rather than its live "
        f"width: the same {rows} live rows differ by {difference:.4g} "
        f"(scale {scale:.4g}) when the scratch is widened from 1x to 8x"
    )


def test_mmq_route_row_zero_is_invariant_under_an_appended_row(
    _weights: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """One extra row must not move the rows that were already there.

    ``_activations`` fills row-major from a fixed seed, so row 0 is the same row
    in both calls; expert 0 takes the extra row, and compaction places expert 0's
    rows first, so compact row 0 is that same source row in both.
    """

    fused_raw, _, _ = _weights
    narrow_counts = _EQUAL_WIDTH_COUNTS
    wide_counts = _ONE_WIDER_COUNTS
    narrow_rows = int(narrow_counts.sum())
    wide_rows = int(wide_counts.sum())

    narrow_bits = _to_bf16_bits(_activations("gaussian", narrow_rows, 20260927))
    wide_bits = _to_bf16_bits(_activations("gaussian", wide_rows, 20260927))
    assert np.array_equal(narrow_bits[0], wide_bits[0]), (
        "this case assumes row 0 is identical across the two row counts; the "
        "activation helper no longer guarantees it, so the case is vacuous"
    )

    narrow = _run_route(fused_raw, narrow_bits, narrow_counts)
    wide = _run_route(fused_raw, wide_bits, wide_counts)

    difference = float(np.abs(narrow[0] - wide[0]).max())
    scale = float(np.abs(narrow[0]).max())
    assert difference == 0.0, (
        f"compact row 0 moved by {difference:.4g} (scale {scale:.4g}) when one "
        f"row was appended at the true geometry ({narrow_rows} -> {wide_rows} "
        "live rows)"
    )


# The down projection's route disposition is the same shape as the gate/up's, not
# the general path: the fixture's `ffn_down_exps` is Q8_0, and
# `gemma4_project_experts_down_mmq` handles Q5_1 or Q8_0, so this artifact takes
# the down MMQ accelerator. It is measured here at the real 2816 out / 704 in
# shape, which Q8_0's 32-element blocks divide exactly (704 = 22 x 32), so no
# substitution is needed.
#
# The expert count is small so the weight is a few megabytes rather than 142, and
# both row counts are kept at or above the expert count so both take the MMQ
# route rather than crossing the grouped/selected crossover.

_DOWN_EXPERTS = 4
_DOWN_IN = _INTERMEDIATE
_DOWN_OUT = _IN_FEATURES


def _down_counts(extra: int) -> np.ndarray:
    counts = np.ones(_DOWN_EXPERTS, dtype=np.int64)
    counts[0] += extra
    return counts


def _run_down_mmq(down_raw: np.ndarray, hidden_bits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Run the down MMQ route over ``counts`` and return the F32 output."""

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )

    rows = int(counts.sum())
    starts = np.zeros(_DOWN_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    hidden_buf = malloc(hidden_bits.nbytes)
    weights_buf = malloc(down_raw.nbytes)
    starts_buf = malloc(starts.nbytes)
    out_buf = malloc(rows * _DOWN_OUT * 2)
    scratch = None
    try:
        copy_host_array_to_device(hidden_buf, hidden_bits)
        copy_host_array_to_device(weights_buf, down_raw)
        copy_host_array_to_device(starts_buf, starts)

        weight = SimpleNamespace(
            backend="hip_gfx1100",
            spec=SimpleNamespace(quant_key="gguf_q8_0"),
            allocation=lambda name: SimpleNamespace(
                buffer=SimpleNamespace(ptr=weights_buf.ptr)
            ),
        )
        scratch = Gemma4ExpertScratch(
            tokens=rows,
            top_k=1,
            hidden_size=_DOWN_IN,
            intermediate=_DOWN_OUT,
            num_experts=_DOWN_EXPERTS,
        )
        served = gemma4_project_experts_down_mmq(
            weight,
            hidden_buf.ptr,
            out_buf.ptr,
            SimpleNamespace(ptr=starts_buf.ptr),
            rows,
            _DOWN_EXPERTS,
            _DOWN_IN,
            _DOWN_OUT,
            scratch=scratch,
        )
        assert served is True, "the down MMQ route declined a Q8_0 weight"

        got = np.empty((rows, _DOWN_OUT), dtype=np.uint16)
        copy_device_to_host(
            int(got.ctypes.data),
            DeviceBuffer(ptr=out_buf.ptr, nbytes=got.nbytes),
            got.nbytes,
        )
        return _from_bf16_bits(got)
    finally:
        if scratch is not None:
            scratch.free()
        for buffer in (hidden_buf, weights_buf, starts_buf, out_buf):
            free(buffer)


@_needs_hip
def test_down_mmq_route_row_zero_is_invariant_under_an_appended_row() -> None:
    """The down MMQ accelerator must not move row 0 when a row is appended."""

    from tests._gguf_synthetic_weights import make_q8_0_weight

    down_raw = np.concatenate(
        [make_q8_0_weight(_DOWN_OUT, _DOWN_IN) for _ in range(_DOWN_EXPERTS)], axis=0
    )
    narrow_counts = _down_counts(0)
    wide_counts = _down_counts(1)
    narrow_rows = int(narrow_counts.sum())
    wide_rows = int(wide_counts.sum())

    # standard_normal fills row-major from a fixed seed, so row 0 is the same row
    # at both row counts.
    narrow_bits = _to_bf16_bits(
        np.random.default_rng(20260927)
        .standard_normal((narrow_rows, _DOWN_IN))
        .astype(np.float32)
    )
    wide_bits = _to_bf16_bits(
        np.random.default_rng(20260927)
        .standard_normal((wide_rows, _DOWN_IN))
        .astype(np.float32)
    )
    assert np.array_equal(narrow_bits[0], wide_bits[0]), (
        "this case assumes row 0 is identical across the two row counts; the "
        "activation helper no longer guarantees it, so the case is vacuous"
    )

    narrow = _run_down_mmq(down_raw, narrow_bits, narrow_counts)
    wide = _run_down_mmq(down_raw, wide_bits, wide_counts)

    difference = float(np.abs(narrow[0] - wide[0]).max())
    scale = float(np.abs(narrow[0]).max())
    assert difference == 0.0, (
        f"compact row 0 moved by {difference:.4g} (scale {scale:.4g}) when one "
        f"row was appended at the true down shape ({narrow_rows} -> {wide_rows} "
        "live rows, down MMQ route)"
    )
