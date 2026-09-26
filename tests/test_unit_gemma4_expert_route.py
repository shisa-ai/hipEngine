"""Gemma 4 routed-expert projection routing.

The routed-expert block projects one block of compact rows through each expert's
weights, and there are three ways to do it. The selected-GEMV family launches
one block per (out_col, lane), so a prefill block re-reads each expert's weight
matrix once per lane. A grouped-prefill family launches one block per (expert,
out_col) and reuses that expert's weight row across its rows. The per-expert
offset path launches one projection per non-empty expert and costs a
host-visible read of the row counts.

Which one runs is decided by the block's lane count and by whether the weight's
quant key registers a family -- never by a model name, an artifact path, or a
list of known-good combinations. These tests pin that decision, its fallback
order, the grouped kernel's arithmetic against a dequantized reference, and the
bit-exactness that lets the grouped route replace the per-lane one without a
numerical gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    _GROUPED_PREFILL_VARIANT,
    gemma4_moe_expert_route_counts,
    gemma4_moe_prefill_route_enabled,
    gemma4_project_experts_grouped_prefill,
    gemma4_project_experts_rows,
)
from hipengine.kernels.registry import KernelKey, is_registered, register
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q5_1_weight
from tests._rocm_guard import hip_runtime_available

_needs_hip = pytest.mark.skipif(
    not hip_runtime_available(),
    reason="HIP runtime unavailable; skipping grouped-prefill parity test",
)

# The registry key the prefill route resolves against. Pinned as a literal, so
# renaming the variant in the registry without updating the route fails here
# instead of silently falling back.
_GROUPED_KEY = KernelKey(
    "hip_gfx1100",
    "moe_linear",
    "gguf_q5_1",
    "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
)

# The route the grouped family has to beat, and to be bit-exact against.
_SELECTED_KEY = KernelKey(
    "hip_gfx1100",
    "linear",
    "gguf_q5_1",
    "selected_gemv_bf16_bf16_out",
)

# Quants this tree may register a grouped family for. The fallback case is about
# the quant's declared capability, so it asks which quant has no family instead
# of naming one: a quant that later gains a family moves the test to the next
# candidate rather than launching a real kernel against the fake pointers below.
_GROUPED_CANDIDATES = ("gguf_q4_k", "gguf_q6_k", "gguf_q8_0", "gguf_q3_k")


def _quant_without_a_grouped_family() -> str:
    """Return a quant whose registry key has no grouped-prefill family.

    The registry is asked directly, after the same lazy registration the route
    performs, so the answer is what the route will see. A real kernel launched
    from these tests would dereference a placeholder pointer, so proving the
    family is absent is what keeps the fallback tests off the device.
    """

    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    for quant in _GROUPED_CANDIDATES:
        key = KernelKey("hip_gfx1100", "moe_linear", quant, _GROUPED_PREFILL_VARIANT)
        _ensure_linear_kernel_registered(key)
        if not is_registered(key):
            return quant
    raise AssertionError(
        "every candidate quant registers a grouped family; the fallback case "
        "needs one that does not"
    )


class _ResidentWeight:
    """The attributes the grouped route reads off a resident quantized weight."""

    def __init__(self, *, backend: str, quant_key: str, ptr: int = 0x5A0000) -> None:
        self.backend = backend
        self.spec = SimpleNamespace(quant_key=quant_key)
        self.ptr = ptr

    def allocation(self, name: str) -> SimpleNamespace:
        assert name == "raw", "the grouped route reads the artifact's raw bytes"
        return SimpleNamespace(buffer=SimpleNamespace(ptr=self.ptr))


def _grouped_backend(quant: str) -> str:
    """Resolve this host's backend with one quant's grouped family registered.

    The session restores a collection-time registry snapshot after every test,
    which drops lazily registered quant families. Re-loading the backend
    package cannot rebuild them on its own: its modules are already cached, so
    the alias pass finds no source keys to copy. Registering the family first
    and the backend package second is what makes these tests independent of run
    order and of whichever backend this host resolves.
    """

    from hipengine.kernels.backends import load_backend_kernel_package, resolve_backend

    if quant == "gguf_q5_1":
        from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
            register_qwen4_exp_q5_1_kernels,
        )

        register_qwen4_exp_q5_1_kernels(replace=True)
    else:
        from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
            register_gguf_q4_k_gemv_kernels,
        )

        register_gguf_q4_k_gemv_kernels(replace=True)
    backend = resolve_backend("auto")
    load_backend_kernel_package(backend)
    return backend


_WEIGHT_MAKERS = {"gguf_q5_1": make_q5_1_weight, "gguf_q4_k": make_q4_k_weight}
_GGML_TYPES = {
    "gguf_q5_1": GGMLQuantizationType.Q5_1,
    "gguf_q4_k": GGMLQuantizationType.Q4_K,
}

# Both quants are exercised because they are separate kernels against separate
# incumbent reductions: the Q5_1 grouped family reduces with a 256-wide shared
# tree at 256 threads, and the Q4_K one has to reproduce ``reduce_block_sum`` at
# the selected path's 128 threads, which is also its k-stride. The geometries
# are per quant because Q4_K's block is 256 wide, so it cannot represent an
# in_features that Q5_1 can.
_GROUPED_GEOMETRIES = {
    "gguf_q5_1": [(2816, 704), (1408, 2816), (64, 128), (37, 96), (16, 32)],
    "gguf_q4_k": [(1408, 2816), (2816, 2816), (64, 256), (37, 512), (16, 256)],
}

_GROUPED_CASES = [
    (quant, out_features, in_features)
    for quant, geometry in _GROUPED_GEOMETRIES.items()
    for out_features, in_features in geometry
]

_REFERENCE_GEOMETRY = {"gguf_q5_1": (64, 128), "gguf_q4_k": (64, 256)}


@pytest.mark.parametrize(
    "lanes, num_experts, expected",
    [
        # The crossover itself: one lane per expert.
        (8, 8, True),
        (7, 8, False),
        (128, 128, True),
        (127, 128, False),
        # Values unrelated to the crossover, so this is not only checking the
        # two sides of one threshold.
        (4096, 128, True),
        (1, 128, False),
        (1024, 4, True),
        (300, 256, True),
        (2, 1, True),
    ],
)
def test_prefill_route_switches_at_one_lane_per_expert(lanes, num_experts, expected) -> None:
    assert gemma4_moe_prefill_route_enabled(lanes=lanes, num_experts=num_experts) is expected


def test_prefill_route_rejects_a_degenerate_expert_count() -> None:
    with pytest.raises(ValueError):
        gemma4_moe_prefill_route_enabled(lanes=8, num_experts=0)


def test_a_bf16_expert_weight_never_reaches_the_grouped_family() -> None:
    """A bare device pointer carries no quant key to resolve a family against."""

    assert gemma4_project_experts_grouped_prefill(0x1000, 1, 2, 3, 4, 5, 6, 7) is False


def test_grouped_prefill_declines_a_quant_without_the_family() -> None:
    """A quant with no grouped family falls back instead of failing."""

    quant = _quant_without_a_grouped_family()
    weight = _ResidentWeight(backend="hip_gfx1100", quant_key=quant)
    assert gemma4_project_experts_grouped_prefill(weight, 1, 2, 3, 4, 5, 6, 7) is False


def test_grouped_prefill_launches_the_registered_family() -> None:
    """The route resolves by quant key and passes the documented ABI."""

    calls: list[tuple[tuple, dict]] = []

    def stub(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    register(_GROUPED_KEY, stub, replace=True)
    assert is_registered(_GROUPED_KEY)

    weight = _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1")
    served = gemma4_project_experts_grouped_prefill(
        weight,
        0x11000,
        0x22000,
        0x33000,
        96,
        128,
        704,
        2816,
        stream=7,
    )

    assert served is True
    assert calls == [
        (
            (0x11000, 0x22000, 0x5A0000, 0x33000, 96, 128, 704, 2816),
            {"stream": 7},
        )
    ]


def test_route_counts_hand_back_a_snapshot() -> None:
    """The diagnostic must not let a caller rewrite the counters."""

    snapshot = gemma4_moe_expert_route_counts()
    snapshot["grouped_prefill"] = 10**9
    assert gemma4_moe_expert_route_counts().get("grouped_prefill", 0) != 10**9


def test_rows_prefers_the_grouped_family_over_the_selected_gemv() -> None:
    """The cheaper route wins, and the other one is not launched anyway."""

    grouped_calls: list[tuple] = []
    selected_calls: list[tuple] = []
    register(_GROUPED_KEY, lambda *args, **kwargs: grouped_calls.append(args), replace=True)
    register(_SELECTED_KEY, lambda *args, **kwargs: selected_calls.append(args), replace=True)

    route = gemma4_project_experts_rows(
        _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1"),
        0x1000,
        0x2000,
        SimpleNamespace(ptr=0x3000),
        0x4000,
        compact_rows=256,
        num_experts=128,
        in_features=704,
        out_features=2816,
    )

    assert route == "grouped_prefill"
    assert len(grouped_calls) == 1
    assert selected_calls == []


def test_rows_falls_back_to_the_selected_gemv_without_a_grouped_family(monkeypatch) -> None:
    """A quant with no grouped family keeps the arithmetic it had before.

    The selected projection is replaced on the module rather than in the
    registry. A registry fixture kernel at an unrelated key does not survive
    this call: the grouped lookup finds its key absent, so the route's lazy
    registration runs the whole GGUF family set, and that pass re-registers
    over a caller's fixture kernel. Replacing the collaborator keeps the test
    on the route decision, which is what it is about, and keeps it off the
    device -- the pointers below are placeholders.
    """

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts

    quant = _quant_without_a_grouped_family()
    weight = _ResidentWeight(backend="hip_gfx1100", quant_key=quant)
    assert gemma4_project_experts_grouped_prefill(weight, 1, 2, 3, 4, 5, 6, 7) is False

    selected_calls: list[tuple] = []
    monkeypatch.setattr(
        gemma4_experts,
        "gemma4_project_experts_selected",
        lambda *args, **kwargs: selected_calls.append(args) or True,
    )

    route = gemma4_project_experts_rows(
        weight,
        0x1000,
        0x2000,
        SimpleNamespace(ptr=0x3000),
        0x4000,
        compact_rows=256,
        num_experts=128,
        in_features=704,
        out_features=2816,
    )

    assert route == "selected_gemv"
    assert len(selected_calls) == 1


def test_rows_keeps_the_selected_gemv_below_one_lane_per_expert() -> None:
    """A decode block must not pay for a grid that walks every expert."""

    grouped_calls: list[tuple] = []
    register(_GROUPED_KEY, lambda *args, **kwargs: grouped_calls.append(args), replace=True)
    register(_SELECTED_KEY, lambda *args, **kwargs: None, replace=True)

    route = gemma4_project_experts_rows(
        _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1"),
        0x1000,
        0x2000,
        SimpleNamespace(ptr=0x3000),
        0x4000,
        compact_rows=127,
        num_experts=128,
        in_features=704,
        out_features=2816,
    )

    assert route == "selected_gemv"
    assert grouped_calls == []


@_needs_hip
@pytest.mark.parametrize("quant", tuple(_REFERENCE_GEOMETRY))
def test_grouped_prefill_matches_a_dequantized_reference(quant) -> None:
    """The grouped kernel reproduces an empty-expert-aware quantized reference.

    The kernel walks ``expert_start`` itself, so this covers three things a
    shape-only check would miss: that expert *e*'s rows read expert *e*'s
    weights, that an expert with no rows is skipped without shifting the rows
    after it, and that the compact row order is the output row order.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.quant.gguf import dequantize_gguf_data

    backend = _grouped_backend(quant)
    num_experts = 4
    out_features, in_features = _REFERENCE_GEOMETRY[quant]
    counts = np.asarray([3, 0, 5, 2], dtype=np.int64)
    rows = int(counts.sum())

    rng = np.random.default_rng(20260927)
    hidden = rng.standard_normal((rows, in_features)).astype(np.float32)
    hidden_bits = _to_bf16_bits(hidden)
    raw = np.concatenate(
        [_WEIGHT_MAKERS[quant](out_features, in_features) for _ in range(num_experts)], axis=0
    )

    # The kernel reads bf16 activations, so the reference must read the rounded
    # values rather than the float32 source.
    rounded = _from_bf16_bits(hidden_bits)
    expected = np.zeros((rows, out_features), dtype=np.float32)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        block = raw[expert * out_features : (expert + 1) * out_features]
        weights = np.asarray(
            dequantize_gguf_data(block, _GGML_TYPES[quant]), dtype=np.float32
        )
        expected[start : start + count] = rounded[start : start + count] @ weights.T
        start += int(count)
    assert start == rows

    starts = np.zeros(num_experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    hidden_buf = malloc(hidden_bits.nbytes)
    weights_buf = malloc(raw.nbytes)
    starts_buf = malloc(starts.nbytes)
    out_buf = malloc(rows * out_features * 2)
    try:
        copy_host_array_to_device(hidden_buf, hidden_bits)
        copy_host_array_to_device(weights_buf, raw)
        copy_host_array_to_device(starts_buf, starts)

        weight = _ResidentWeight(
            backend=backend,
            quant_key=quant,
            ptr=weights_buf.ptr,
        )
        served = gemma4_project_experts_grouped_prefill(
            weight,
            hidden_buf.ptr,
            starts_buf.ptr,
            out_buf.ptr,
            rows,
            num_experts,
            in_features,
            out_features,
        )
        assert served is True, f"the {quant} grouped family did not resolve on this backend"

        got = np.empty((rows, out_features), dtype=np.uint16)
        copy_device_to_host(
            int(got.ctypes.data),
            DeviceBuffer(ptr=out_buf.ptr, nbytes=got.nbytes),
            got.nbytes,
        )
        got = _from_bf16_bits(got)
    finally:
        for buffer in (hidden_buf, weights_buf, starts_buf, out_buf):
            free(buffer)

    scale = float(np.abs(expected).max())
    assert scale > 0
    assert np.allclose(got, expected, rtol=5e-3, atol=5e-3 * scale), (
        f"grouped prefill diverged from the reference: "
        f"max abs diff {np.abs(got - expected).max():.4g} against scale {scale:.4g}"
    )
    # The empty expert is load-bearing: expert 1 has no rows, so rows 3-7 must
    # carry expert 2's weights rather than a shifted copy of expert 1's.
    assert not np.allclose(got[3], got[0]), "row order did not follow expert_start"


def _to_bf16_bits(array: np.ndarray) -> np.ndarray:
    """Pack float32 values into the bfloat16 bit patterns the kernels read."""

    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def _from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    """Widen bfloat16 bit patterns back to float32."""

    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


@_needs_hip
@pytest.mark.parametrize("quant, out_features, in_features", _GROUPED_CASES)
def test_grouped_prefill_is_bit_exact_against_the_selected_gemv(
    quant, out_features, in_features
) -> None:
    """Weight reuse must not cost a single bit against the per-lane GEMV.

    The selected family launches one block per (out_col, lane) and the grouped
    family launches one block per (expert, out_col) that walks the expert's
    rows. Each row accumulates as ``column = t, t + blockDim.x, ...`` and
    reduces through the same tree the selected path uses, so the two are
    expected to agree exactly rather than approximately. A grouped kernel with
    a different association would still pass a tolerance check while changing
    every prefill logit in the model, so the comparison here is on bits.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        gemma4_project_experts_selected,
    )

    num_experts = 4
    counts = np.asarray([3, 0, 5, 2], dtype=np.int64)
    rows = int(counts.sum())
    backend = _grouped_backend(quant)
    rng = np.random.default_rng(20260928)
    hidden_bits = _to_bf16_bits(rng.standard_normal((rows, in_features)).astype(np.float32))
    raw = np.concatenate(
        [_WEIGHT_MAKERS[quant](out_features, in_features) for _ in range(num_experts)], axis=0
    )
    selected = np.repeat(np.arange(num_experts, dtype=np.int64), counts)
    starts = np.zeros(num_experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    hidden_buf = malloc(hidden_bits.nbytes)
    weights_buf = malloc(raw.nbytes)
    starts_buf = malloc(starts.nbytes)
    selected_buf = malloc(selected.nbytes)
    out_grouped = malloc(rows * out_features * 2)
    out_selected = malloc(rows * out_features * 2)
    try:
        copy_host_array_to_device(hidden_buf, hidden_bits)
        copy_host_array_to_device(weights_buf, raw)
        copy_host_array_to_device(starts_buf, starts)
        copy_host_array_to_device(selected_buf, selected)

        weight = _ResidentWeight(
            backend=backend,
            quant_key=quant,
            ptr=weights_buf.ptr,
        )
        assert gemma4_project_experts_grouped_prefill(
            weight,
            hidden_buf.ptr,
            starts_buf.ptr,
            out_grouped.ptr,
            rows,
            num_experts,
            in_features,
            out_features,
        )
        assert gemma4_project_experts_selected(
            weight,
            hidden_buf.ptr,
            selected_buf.ptr,
            out_selected.ptr,
            rows,
            rows,
            num_experts,
            in_features,
            out_features,
        )

        def read_bits(buffer) -> np.ndarray:
            bits = np.empty((rows, out_features), dtype=np.uint16)
            copy_device_to_host(
                int(bits.ctypes.data),
                DeviceBuffer(ptr=buffer.ptr, nbytes=bits.nbytes),
                bits.nbytes,
            )
            return bits

        grouped = read_bits(out_grouped)
        selected_bits = read_bits(out_selected)
    finally:
        for buffer in (
            hidden_buf,
            weights_buf,
            starts_buf,
            selected_buf,
            out_grouped,
            out_selected,
        ):
            free(buffer)

    differing = int((grouped != selected_bits).sum())
    assert differing == 0, (
        f"grouped prefill differs from the selected GEMV in {differing} of "
        f"{grouped.size} bf16 outputs; max abs diff "
        f"{np.abs(_from_bf16_bits(grouped) - _from_bf16_bits(selected_bits)).max():.4g}"
    )
