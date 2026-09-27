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
# them with fewer rows than one 32-row MMQ tile, and a few empty.
_SHORT_PREFILL = np.asarray(
    [0, 1, 2, 3, 5, 8, 13, 21, 2, 1, 4, 7, 11, 17, 31, 32] * 8, dtype=np.int64
)

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
):
    """Run the route with the live row count and a scratch of its own width.

    ``capacity_factor`` sizes the scratch above the live lane count, which is the
    documented way it is used: the caller sizes one scratch for the widest block
    it will run and then runs narrower blocks through it. A factor of 1 makes
    capacity and live width equal, which is what the rest of the coverage does.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )

    rows = int(counts.sum())
    fused_width = 2 * _INTERMEDIATE
    starts = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    hidden_buf = malloc(hidden_bits.nbytes)
    weights_buf = malloc(fused_raw.nbytes)
    starts_buf = malloc(starts.nbytes)
    out_buf = malloc(rows * fused_width * 2)
    scratch = None
    try:
        copy_host_array_to_device(hidden_buf, hidden_bits)
        copy_host_array_to_device(weights_buf, fused_raw)
        copy_host_array_to_device(starts_buf, starts)

        weight = SimpleNamespace(
            spec=SimpleNamespace(quant_key="gguf_q4_k"),
            allocation=lambda name: SimpleNamespace(
                buffer=SimpleNamespace(ptr=weights_buf.ptr)
            ),
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
        for buffer in (hidden_buf, weights_buf, starts_buf, out_buf):
            free(buffer)


@_needs_hip
@pytest.mark.parametrize("profile", ["gaussian", "outlier"])
@pytest.mark.parametrize("capacity_factor", [1, 8])
def test_fused_gate_up_mmq_route_holds_at_the_true_projection_geometry(
    _weights: tuple[np.ndarray, np.ndarray, np.ndarray],
    profile: str,
    capacity_factor: int,
) -> None:
    fused_raw, gate_raw, up_raw = _weights
    counts = _SHORT_PREFILL
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
        f"profile {profile!r} at the true geometry with capacity factor "
        f"{capacity_factor} (live {rows} rows): normalized max "
        f"{normalized_max:.4g} at compact row {worst_row}, normalized mean "
        f"{normalized_mean:.4g}, against scale {scale:.4g}"
    )
    assert normalized_max < _MAX_ENVELOPE, (
        f"fused MMQ gate_up exceeded the envelope ({context})"
    )
    assert normalized_mean < _MEAN_ENVELOPE, (
        f"fused MMQ gate_up exceeded the envelope ({context})"
    )
