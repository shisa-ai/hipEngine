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
    _GROUPED_PREFILL_VARIANTS,
    Gemma4ExpertScratch,
    gemma4_moe_expert_route_counts,
    gemma4_moe_gate_up_mmq_enabled,
    gemma4_moe_grouped_variant_counts,
    gemma4_moe_prefill_route_enabled,
    gemma4_project_experts_gate_up_mmq,
    gemma4_project_experts_grouped_prefill,
    gemma4_project_experts_rows,
    gemma4_project_experts_selected,
)
from hipengine.kernels.registry import KernelKey, is_registered, register, unregister
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import (
    make_q4_k_weight,
    make_q5_1_weight,
    make_q5_k_weight,
    make_q8_0_weight,
)
from tests._rocm_guard import hip_runtime_available

_needs_hip = pytest.mark.skipif(
    not hip_runtime_available(),
    reason="HIP runtime unavailable; skipping grouped-prefill parity test",
)

# The registry key the prefill route resolves against: the variant the
# preference order tries first, pinned as a literal so renaming it in the
# registry without updating the route fails here instead of silently falling
# back. It has to track the order rather than a fixed variant, because the route
# tries the variants in that order and a stub on a later one would never be
# called - the real kernel would run against the test's fake pointers instead.
_GROUPED_KEY = KernelKey(
    "hip_gfx1100",
    "moe_linear",
    "gguf_q5_1",
    "selected_grouped_prefill_staged_out4_bf16_bf16_out",
)


def test_grouped_key_matches_the_preference_order() -> None:
    """The stub key above must be the variant the route tries first."""

    assert _GROUPED_KEY.variant == _GROUPED_PREFILL_VARIANTS[0]

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
        registered = False
        for variant in _GROUPED_PREFILL_VARIANTS:
            key = KernelKey("hip_gfx1100", "moe_linear", quant, variant)
            _ensure_linear_kernel_registered(key)
            registered = registered or is_registered(key)
        if not registered:
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

    def has_allocation(self, name: str) -> bool:
        # This double models a weight carrying only the raw blocks, which is what
        # a rank-2 tensor or a non-Q4_K expert tensor plans. The pack8 route must
        # decline it, so the ladder falls through to the raw selected route
        # rather than looking up a packed allocation that was never planned.
        return name == "raw"


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
    elif quant == "gguf_q4_k":
        from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
            register_gguf_q4_k_gemv_kernels,
        )

        register_gguf_q4_k_gemv_kernels(replace=True)
    else:
        from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
            register_gguf_k_gemv_kernels,
        )

        register_gguf_k_gemv_kernels(replace=True)
    backend = resolve_backend("auto")
    load_backend_kernel_package(backend)
    return backend


_WEIGHT_MAKERS = {
    "gguf_q5_1": make_q5_1_weight,
    "gguf_q4_k": make_q4_k_weight,
    "gguf_q5_k": make_q5_k_weight,
    "gguf_q8_0": make_q8_0_weight,
}
_GGML_TYPES = {
    "gguf_q5_1": GGMLQuantizationType.Q5_1,
    "gguf_q4_k": GGMLQuantizationType.Q4_K,
    "gguf_q5_k": GGMLQuantizationType.Q5_K,
    "gguf_q8_0": GGMLQuantizationType.Q8_0,
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
    # Q8_0's block is 32 wide, so its in_features only has to be a multiple of
    # 32. The 704 -> 2816 case is the artifact's own Q8_0 expert down shape.
    "gguf_q8_0": [(2816, 704), (704, 2816), (64, 32), (37, 96)],
}

_GROUPED_CASES = [
    (quant, out_features, in_features)
    for quant, geometry in _GROUPED_GEOMETRIES.items()
    for out_features, in_features in geometry
]

_REFERENCE_GEOMETRY = {
    "gguf_q5_1": (64, 128),
    "gguf_q4_k": (64, 256),
    "gguf_q8_0": (64, 32),
}


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
    """The route resolves by quant key and passes the documented ABI.

    The stub goes on every variant in the preference order, not just the first.
    The route walks that order and calls the first one it resolves, so a stub on
    a single variant would let the real kernel run against this test's fake
    pointers the moment the order changes; stubbing all of them keeps the test
    about the ABI instead of about which variant currently wins.
    """

    calls: list[tuple[tuple, dict]] = []

    def stub(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    # Import the runtime module before stubbing. Its first import registers the
    # real GGUF families, and the route's own import of it happens after the
    # stubs would be installed, which would overwrite them.
    from hipengine.runtime import gguf_linear as _gguf_linear  # noqa: F401

    for variant in _GROUPED_PREFILL_VARIANTS:
        register(
            KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1", variant),
            stub,
            replace=True,
        )
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


def test_grouped_prefill_prefers_the_earlier_variant_in_the_order() -> None:
    """When several grouped variants exist, the route takes the preferred one.

    The variants are the same reduction with different fetch strategies, so the
    order is a cost ranking and the route must follow it rather than whichever
    key happens to resolve first.
    """

    preferred, fallback = _GROUPED_PREFILL_VARIANTS[0], _GROUPED_PREFILL_VARIANTS[1]
    calls: list[str] = []
    for variant, label in ((preferred, "preferred"), (fallback, "fallback")):
        register(
            KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1", variant),
            (lambda *args, label=label, **kwargs: calls.append(label)),
            replace=True,
        )

    weight = _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1")
    assert gemma4_project_experts_grouped_prefill(weight, 1, 2, 3, 4, 5, 6, 7) is True
    assert calls == ["preferred"]


def test_grouped_prefill_falls_through_to_the_next_variant() -> None:
    """A quant that registers only a later variant still runs it."""

    preferred, fallback = _GROUPED_PREFILL_VARIANTS[0], _GROUPED_PREFILL_VARIANTS[1]
    calls: list[str] = []
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1", fallback),
        (lambda *args, **kwargs: calls.append("fallback")),
        replace=True,
    )
    # This test is about a quant that registers *only* the later variant, so
    # clear the preferred one rather than asserting it happens to be absent.
    # Asserting would make the test depend on whether an earlier test or a lazy
    # registration sweep already put the preferred variant in the registry.
    unregister(KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1", preferred))

    weight = _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1")
    assert gemma4_project_experts_grouped_prefill(weight, 1, 2, 3, 4, 5, 6, 7) is True
    assert calls == ["fallback"]


def test_route_counts_hand_back_a_snapshot() -> None:
    """The diagnostic must not let a caller rewrite the counters."""

    snapshot = gemma4_moe_expert_route_counts()
    snapshot["grouped_prefill"] = 10**9
    assert gemma4_moe_expert_route_counts().get("grouped_prefill", 0) != 10**9


def test_variant_counts_hand_back_a_snapshot() -> None:
    """The variant diagnostic is a snapshot too."""

    snapshot = gemma4_moe_grouped_variant_counts()
    snapshot["selected_grouped_prefill_staged_out4_bf16_bf16_out"] = 10**9
    assert (
        gemma4_moe_grouped_variant_counts().get(
            "selected_grouped_prefill_staged_out4_bf16_bf16_out", 0
        )
        != 10**9
    )


def test_grouped_prefill_records_the_variant_it_resolved() -> None:
    """The route reports the family; this reports which fetch strategy ran.

    Every variant in the family is bit-exact, so a cost change is only confirmed
    by naming the one that actually launched. The recorded name has to be the one
    the preference order selected, not merely a registered one.
    """

    preferred = _GROUPED_PREFILL_VARIANTS[0]
    register(_GROUPED_KEY, lambda *args, **kwargs: None, replace=True)
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1", preferred),
        lambda *args, **kwargs: None,
        replace=True,
    )
    before = gemma4_moe_grouped_variant_counts().get(preferred, 0)
    assert gemma4_project_experts_grouped_prefill(
        _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1"),
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )
    assert gemma4_moe_grouped_variant_counts().get(preferred, 0) == before + 1


def test_rows_prefers_the_grouped_family_over_the_selected_gemv() -> None:
    """The cheaper route wins, and the other one is not launched anyway."""

    grouped_calls: list[tuple] = []
    selected_calls: list[tuple] = []
    pack8_calls: list[tuple] = []
    register(_GROUPED_KEY, lambda *args, **kwargs: grouped_calls.append(args), replace=True)
    register(_SELECTED_KEY, lambda *args, **kwargs: selected_calls.append(args), replace=True)
    # Both selected variant keys, not just the scalar one. The dispatcher prefers
    # the pack8 sibling wherever it is registered, so a fixture that pins only
    # the scalar key stops pinning the route the moment a quant gains the
    # sibling -- and these pointers are placeholders, so an unpinned route is a
    # real launch on garbage rather than a recorded call.
    register(
        _SELECTED_PACK8_KEY, lambda *args, **kwargs: pack8_calls.append(args), replace=True
    )

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
    assert pack8_calls == []


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
    # Both selected variant keys, for the reason the test above gives. This test
    # used to reach the real pack8 kernel once Q5_1 registered a sibling, and
    # faulted on the placeholder pointers below; the route decision it is about
    # is the same either way, so both variants are recorded and neither runs.
    register(_SELECTED_PACK8_KEY, lambda *args, **kwargs: None, replace=True)

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


@_needs_hip
@pytest.mark.parametrize("out_features, in_features", [(2816, 2816), (1408, 2816)])
def test_grouped_row4_is_bit_exact_against_the_selected_gemv(
    out_features, in_features
) -> None:
    """A quant with no grouped prefill family still gets weight reuse.

    Q5_K is the case this artifact has: one MoE layer carries Q5_K expert
    weights while the rest of the model is Q4_K. It has no grouped *prefill*
    kernel, so the dispatcher falls past that family to the grouped row4 GEMV,
    which reuses an expert's weight rows across four rows instead of one. The
    geometry is the fused gate/up width, because that is what the caller asks
    for: both halves are projected in one call.
    """

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )

    num_experts = 8
    counts = np.asarray([100, 0, 90, 40, 0, 82, 100, 100], dtype=np.int64)
    rows = int(counts.sum())
    rng = np.random.default_rng(20260929)
    hidden_bits = _to_bf16_bits(
        rng.standard_normal((rows, in_features)).astype(np.float32)
    )
    raw = np.concatenate(
        [
            make_q5_k_weight(out_features, in_features)
            for _ in range(num_experts)
        ],
        axis=0,
    )
    selected = np.repeat(np.arange(num_experts, dtype=np.int64), counts)
    starts = np.zeros(num_experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    hidden_buf = malloc(hidden_bits.nbytes)
    weights_buf = malloc(raw.nbytes)
    starts_buf = malloc(starts.nbytes)
    selected_buf = malloc(selected.nbytes)
    out_row4 = malloc(rows * out_features * 2)
    out_selected = malloc(rows * out_features * 2)
    try:
        for buffer, array in (
            (hidden_buf, hidden_bits),
            (weights_buf, raw),
            (starts_buf, starts),
            (selected_buf, selected),
        ):
            copy_host_array_to_device(buffer, array)

        weight = _ResidentWeight(
            backend=_grouped_backend("gguf_q5_k"),
            quant_key="gguf_q5_k",
            ptr=weights_buf.ptr,
        )
        route = gemma4_project_experts_rows(
            weight,
            hidden_buf.ptr,
            out_row4.ptr,
            starts_buf,
            selected_buf.ptr,
            rows,
            num_experts,
            in_features,
            out_features,
        )
        assert route == "grouped_row4", route
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

        row4_bits = np.empty(rows * out_features, dtype=np.uint16)
        selected_bits = np.empty(rows * out_features, dtype=np.uint16)
        copy_device_to_host(row4_bits.ctypes.data, out_row4, row4_bits.nbytes)
        copy_device_to_host(
            selected_bits.ctypes.data, out_selected, selected_bits.nbytes
        )
    finally:
        for buffer in (
            hidden_buf,
            weights_buf,
            starts_buf,
            selected_buf,
            out_row4,
            out_selected,
        ):
            free(buffer)

    differing = int((row4_bits != selected_bits).sum())
    assert differing == 0, (
        f"grouped row4 differs from the selected GEMV in {differing} of "
        f"{row4_bits.size} bf16 outputs at out={out_features} in={in_features}"
    )


# The bit-exact sibling the selected route prefers when it is registered. Eight
# output columns per block share one x read and one block reduction; the k walk,
# the dequant and the reduction tree are per output and unchanged.
_SELECTED_PACK8_KEY = KernelKey(
    "hip_gfx1100",
    "linear",
    "gguf_q5_1",
    "selected_pack8_gemv_bf16_bf16_out",
)


def _run_selected(weight, out_features: int, in_features: int = 704):
    return gemma4_project_experts_selected(
        weight,
        0x1000,
        0x2000,
        0x3000,
        8,
        8,
        128,
        in_features,
        out_features,
    )


def test_selected_prefers_the_pack8_sibling_over_the_single_output_owner() -> None:
    """Eight columns per block is the same arithmetic, so it is preferred."""

    pack8_calls: list[tuple] = []
    selected_calls: list[tuple] = []
    register(_SELECTED_PACK8_KEY, lambda *a, **k: pack8_calls.append(a), replace=True)
    register(_SELECTED_KEY, lambda *a, **k: selected_calls.append(a), replace=True)
    try:
        ran = _run_selected(
            _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1"), 2816
        )
    finally:
        unregister(_SELECTED_PACK8_KEY)

    assert ran is True
    assert len(pack8_calls) == 1
    assert selected_calls == []


def test_selected_falls_back_when_the_width_is_not_eight_wide() -> None:
    """A pack8 block has to write eight columns, so an odd width declines it.

    2812 is the shape half the artifact's widths are not: it is a multiple of
    four, so it is not obviously malformed, but it leaves a partial block at the
    end. The route must not reach for the pack8 kernel and then rely on the
    kernel to drop the tail.
    """

    pack8_calls: list[tuple] = []
    selected_calls: list[tuple] = []
    register(_SELECTED_PACK8_KEY, lambda *a, **k: pack8_calls.append(a), replace=True)
    register(_SELECTED_KEY, lambda *a, **k: selected_calls.append(a), replace=True)
    try:
        ran = _run_selected(
            _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1"), 2812
        )
    finally:
        unregister(_SELECTED_PACK8_KEY)

    assert ran is True
    assert pack8_calls == []
    assert len(selected_calls) == 1


def test_selected_falls_back_when_no_pack8_sibling_is_registered() -> None:
    """A quant that declares no pack8 sibling keeps the arithmetic it had.

    Q5_1 used to be that quant in this tree, and this test used to borrow it.
    Q5_1 registers a pack8 sibling on the linear layer now, so the state is
    constructed here instead. That is what the test was asserting all along: a
    quant with a scalar selected route and no pack8 sibling takes the scalar
    route rather than reaching for a kernel that is not there.
    """

    quant = "gguf_q4_0"
    scalar_key = KernelKey(
        "hip_gfx1100", "linear", quant, "selected_gemv_bf16_bf16_out"
    )
    pack8_key = KernelKey(
        "hip_gfx1100", "linear", quant, "selected_pack8_gemv_bf16_bf16_out"
    )
    assert not is_registered(pack8_key), (
        "this test needs a quant with no pack8 sibling on the linear layer"
    )

    selected_calls: list[tuple] = []
    register(scalar_key, lambda *a, **k: selected_calls.append(a), replace=True)
    try:
        ran = _run_selected(
            _ResidentWeight(backend="hip_gfx1100", quant_key=quant), 2816
        )
    finally:
        unregister(scalar_key)

    assert ran is True
    assert len(selected_calls) == 1


def test_selected_reports_a_quant_that_registers_neither_variant() -> None:
    """The boolean still means what it meant: this quant has no selected GEMV."""

    unregister(_SELECTED_PACK8_KEY)
    unregister(_SELECTED_KEY)

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts

    original = gemma4_experts._SELECTED_VARIANT
    gemma4_experts._SELECTED_VARIANT = "selected_gemv_variant_that_is_not_registered"
    try:
        ran = _run_selected(
            _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_1"), 2816
        )
    finally:
        gemma4_experts._SELECTED_VARIANT = original

    assert ran is False


# The pack8 sibling has to be bit-exact with the single-output owner, because the
# route prefers it without a numerical gate. These are the artifact's own expert
# shapes: Q4_K and Q5_K gate/up at 1408 out, Q4_K up at 2816 out, the odd layer's
# Q8_0 down at 2816 out, and the Q5_1 down projection at its 704-wide input -- the
# shape the pack8 route exists for, since 704 is a multiple of 32 but not of 256.
_PACK8_SELECTED_CASES = [
    ("gguf_q4_k", 1408, 2816),
    ("gguf_q4_k", 2816, 2816),
    ("gguf_q5_k", 1408, 2816),
    ("gguf_q8_0", 2816, 704),
    ("gguf_q5_1", 2816, 704),
]


@_needs_hip
@pytest.mark.parametrize("quant, out_features, in_features", _PACK8_SELECTED_CASES)
def test_pack8_selected_is_bit_exact_against_the_single_output_owner(
    quant, out_features, in_features
) -> None:
    """Weight reuse must not cost a single bit against the per-lane GEMV.

    Both kernels are launched on the same weights and the same rows and their
    bf16 outputs are compared as bit patterns. The route prefers the pack8
    sibling, so a difference here is a silent numerical change on the default
    decode path rather than a candidate that a gate would catch.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.registry import resolve

    backend = _grouped_backend(quant)
    num_experts = 4
    rows = 8
    rng = np.random.default_rng(20260930)
    hidden_bits = _to_bf16_bits(
        rng.standard_normal((rows, in_features)).astype(np.float32)
    )
    raw = np.concatenate(
        [_WEIGHT_MAKERS[quant](out_features, in_features) for _ in range(num_experts)],
        axis=0,
    )
    selected = np.asarray([3, 0, 2, 1, 1, 3, 0, 2], dtype=np.int64)

    hidden_buf = malloc(hidden_bits.nbytes)
    weights_buf = malloc(raw.nbytes)
    selected_buf = malloc(selected.nbytes)
    out_pack8 = malloc(rows * out_features * 2)
    out_owner = malloc(rows * out_features * 2)
    try:
        for buffer, array in (
            (hidden_buf, hidden_bits),
            (weights_buf, raw),
            (selected_buf, selected),
        ):
            copy_host_array_to_device(buffer, array)

        weight = _ResidentWeight(backend=backend, quant_key=quant, ptr=weights_buf.ptr)
        args = (
            hidden_buf.ptr,
            selected_buf.ptr,
            weights_buf.ptr,
            rows,
            rows,
            num_experts,
            in_features,
            out_features,
        )
        resolved = {}
        for variant, out_buf in (
            ("selected_pack8_gemv_bf16_bf16_out", out_pack8),
            ("selected_gemv_bf16_bf16_out", out_owner),
        ):
            fn = resolve(
                backend=backend, layer="linear", quant=quant, variant=variant
            )
            resolved[variant] = fn
            fn(*args[:3], out_buf.ptr, *args[3:])

        # Two keys that resolve to one function would make this test compare a
        # route with itself and report agreement. The Q4_K gate/up landing hit
        # exactly that shape when its harnesses selected both arms through the
        # same helper, so the identity is asserted rather than assumed.
        assert (
            resolved["selected_pack8_gemv_bf16_bf16_out"]
            is not resolved["selected_gemv_bf16_bf16_out"]
        ), (
            f"the {quant} pack8 and single-output variants resolved to the same "
            f"kernel, so this comparison is vacuous"
        )

        pack8_bits = np.empty(rows * out_features, dtype=np.uint16)
        owner_bits = np.empty(rows * out_features, dtype=np.uint16)
        copy_device_to_host(pack8_bits.ctypes.data, out_pack8, pack8_bits.nbytes)
        copy_device_to_host(owner_bits.ctypes.data, out_owner, owner_bits.nbytes)
    finally:
        for buffer in (
            hidden_buf,
            weights_buf,
            selected_buf,
            out_pack8,
            out_owner,
        ):
            free(buffer)

    assert np.abs(_from_bf16_bits(owner_bits)).max() > 0, "the owner wrote nothing"
    differing = int((pack8_bits != owner_bits).sum())
    assert differing == 0, (
        f"the {quant} pack8 selected GEMV differs from the single-output owner "
        f"in {differing} of {pack8_bits.size} bf16 outputs at "
        f"out={out_features} in={in_features}"
    )


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
    extra_outputs: list[tuple[str, object]] = []
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
        # Every grouped variant this quant registers has to agree, not just the
        # one the route prefers: they are the same reduction with different
        # fetch strategies, and the route picks between them on cost alone.
        extra_outputs = []
        for variant in _GROUPED_PREFILL_VARIANTS[1:]:
            from hipengine.kernels.registry import resolve as resolve_kernel
            from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

            key = KernelKey(backend, "moe_linear", quant, variant)
            _ensure_linear_kernel_registered(key)
            try:
                fn = resolve_kernel(
                    backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
                )
            except Exception:
                continue
            buffer = malloc(rows * out_features * 2)
            fn(
                hidden_buf.ptr,
                starts_buf.ptr,
                weights_buf.ptr,
                buffer.ptr,
                rows,
                num_experts,
                in_features,
                out_features,
                stream=0,
            )
            extra_outputs.append((variant, buffer))
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
        for variant, buffer in extra_outputs:
            other = read_bits(buffer)
            assert int((other != selected_bits).sum()) == 0, (
                f"the {quant} grouped variant {variant} differs from the selected GEMV"
            )
    finally:
        for buffer in (
            hidden_buf,
            weights_buf,
            starts_buf,
            selected_buf,
            out_grouped,
            out_selected,
            *[buffer for _, buffer in extra_outputs],
        ):
            free(buffer)

    differing = int((grouped != selected_bits).sum())
    assert differing == 0, (
        f"grouped prefill differs from the selected GEMV in {differing} of "
        f"{grouped.size} bf16 outputs; max abs diff "
        f"{np.abs(_from_bf16_bits(grouped) - _from_bf16_bits(selected_bits)).max():.4g}"
    )


@_needs_hip
def test_fused_gate_up_mmq_route_matches_a_dequantized_reference() -> None:
    """The MMQ gate_up route reproduces the fused layout's dequantized reference.

    Gemma 4 stores ``ffn_gate_up_exps`` as ``(num_experts, 2 * intermediate,
    hidden)`` with the gate rows first *per expert*, so an expert's stride is the
    fused width rather than one half's width. The route reads both halves from
    that one buffer and emits the fused ``gate | up`` row block the GeGLU
    consumer expects. This pins three things a shape-only check would miss: that
    expert *e*'s two halves are addressed with the fused stride, that the output
    halves land where that consumer reads them, and that quantizing the block's
    activations to DS4 Q8_1 keeps the result inside the family's
    production-variant envelope.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.quant.gguf import dequantize_gguf_data

    backend = _grouped_backend("gguf_q4_k")
    num_experts = 4
    in_features = 256
    intermediate = 64
    fused_width = 2 * intermediate
    counts = np.asarray([3, 0, 5, 2], dtype=np.int64)
    rows = int(counts.sum())

    rng = np.random.default_rng(20260927)
    hidden = rng.standard_normal((rows, in_features)).astype(np.float32)
    hidden_bits = _to_bf16_bits(hidden)
    gate_raw = np.concatenate(
        [make_q4_k_weight(intermediate, in_features) for _ in range(num_experts)],
        axis=0,
    )
    up_raw = np.concatenate(
        [make_q4_k_weight(intermediate, in_features) for _ in range(num_experts)],
        axis=0,
    )
    # The synthetic fixture is deterministic, so gate and up would otherwise be
    # byte-identical and a fused-stride error would read the right *values* from
    # the wrong half. Flipping the up half's scale exponents makes the two halves
    # distinguishable while keeping both legal Q4_K.
    up_raw = up_raw.copy()
    up_raw[:, 1::2] ^= np.uint8(0x04)
    # (num_experts, 2 * intermediate, hidden) with the gate first, per expert.
    fused_raw = np.concatenate(
        [
            gate_raw.reshape(num_experts, intermediate, -1),
            up_raw.reshape(num_experts, intermediate, -1),
        ],
        axis=1,
    ).reshape(num_experts * fused_width, -1)

    # The route reads bf16 activations, so the reference reads the rounded values.
    rounded = _from_bf16_bits(hidden_bits)
    expected = np.zeros((rows, fused_width), dtype=np.float32)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        gate = np.asarray(
            dequantize_gguf_data(
                gate_raw[expert * intermediate : (expert + 1) * intermediate],
                GGMLQuantizationType.Q4_K,
            ),
            dtype=np.float32,
        )
        up = np.asarray(
            dequantize_gguf_data(
                up_raw[expert * intermediate : (expert + 1) * intermediate],
                GGMLQuantizationType.Q4_K,
            ),
            dtype=np.float32,
        )
        block = rounded[start : start + count]
        expected[start : start + count, :intermediate] = block @ gate.T
        expected[start : start + count, intermediate:] = block @ up.T
        start += int(count)
    assert start == rows

    starts = np.zeros(num_experts + 1, dtype=np.int64)
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

        weight = _ResidentWeight(
            backend=backend, quant_key="gguf_q4_k", ptr=weights_buf.ptr
        )
        scratch = Gemma4ExpertScratch(
            tokens=rows,
            top_k=1,
            hidden_size=in_features,
            intermediate=intermediate,
            num_experts=num_experts,
        )
        served = gemma4_project_experts_gate_up_mmq(
            weight,
            hidden_buf.ptr,
            out_buf.ptr,
            SimpleNamespace(ptr=starts_buf.ptr),
            rows,
            num_experts,
            in_features,
            intermediate,
            scratch=scratch,
        )
        assert served is True, "the fused MMQ gate_up route declined a Q4_K weight"

        got = np.empty((rows, fused_width), dtype=np.uint16)
        copy_device_to_host(
            int(got.ctypes.data),
            DeviceBuffer(ptr=out_buf.ptr, nbytes=got.nbytes),
            got.nbytes,
        )
        got = _from_bf16_bits(got)
    finally:
        if scratch is not None:
            scratch.free()
        for buffer in (hidden_buf, weights_buf, starts_buf, out_buf):
            free(buffer)

    scale = float(np.abs(expected).max())
    assert scale > 0
    difference = np.abs(got - expected)
    # DS4 Q8_1 activation quantization is a changed-arithmetic path, so the
    # bound is the envelope the MMQ family already asserts against its strict
    # owner rather than bit equality with the fp32 grouped route.
    assert float(difference.max()) < 2e-2 * scale, (
        f"fused MMQ gate_up exceeded the envelope: normalized max "
        f"{float(difference.max()) / scale:.4g} against scale {scale:.4g}"
    )
    assert float(difference.mean()) < 2e-3 * scale, (
        f"fused MMQ gate_up exceeded the envelope: normalized mean "
        f"{float(difference.mean()) / scale:.4g} against scale {scale:.4g}"
    )
    # The two halves are separate reads of one buffer, so a wrong fused stride
    # would show up as the up half carrying the gate half's expert.
    assert not np.allclose(got[:, :intermediate], got[:, intermediate:]), (
        "the gate and up halves matched, so the second half read the first"
    )


@_needs_hip
def test_fused_gate_up_mmq_route_takes_its_min_correction_from_one_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Q4_K min-offset correction is a property of the activation, not a plane.

    The leaf's correction is ``-dmin * xsum``, where ``xsum`` is the sum stored
    beside the activation plane. For the Q4_K weight ``d*scale*q - dmin*min``
    that term accounts for ``dmin * min * sum_k x[k]``, so it needs the
    activation's own sum -- which is exactly what the *first* plane stores.
    Later planes store their residuals' sums, so accumulating the term once per
    plane over-counts it by about the size of the error the extra planes exist to
    remove. That is why three planes measured 1.08x better than one rather than
    22x on the model, and why the route's residual error looked like a
    non-activation source at 0.61 percent relative.

    The contract this pins is that the later planes' stored sums are not an
    input. Two three-plane runs that differ only in those sums must agree
    bit-for-bit.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q4_k_q8_1_selected_prefill as leaf_module,
    )
    from hipengine.quant.gguf import dequantize_gguf_data

    backend = _grouped_backend("gguf_q4_k")
    num_experts = 4
    in_features = 256
    intermediate = 64
    fused_width = 2 * intermediate
    counts = np.asarray([3, 0, 5, 2], dtype=np.int64)
    rows = int(counts.sum())

    rng = np.random.default_rng(20260928)
    hidden = rng.standard_normal((rows, in_features)).astype(np.float32)
    hidden_bits = _to_bf16_bits(hidden)
    gate_raw = np.concatenate(
        [make_q4_k_weight(intermediate, in_features) for _ in range(num_experts)],
        axis=0,
    )
    up_raw = np.concatenate(
        [make_q4_k_weight(intermediate, in_features) for _ in range(num_experts)],
        axis=0,
    )
    up_raw = up_raw.copy()
    up_raw[:, 1::2] ^= np.uint8(0x04)
    fused_raw = np.concatenate(
        [
            gate_raw.reshape(num_experts, intermediate, -1),
            up_raw.reshape(num_experts, intermediate, -1),
        ],
        axis=1,
    ).reshape(num_experts * fused_width, -1)

    rounded = _from_bf16_bits(hidden_bits)
    expected = np.zeros((rows, fused_width), dtype=np.float32)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        gate = np.asarray(
            dequantize_gguf_data(
                gate_raw[expert * intermediate : (expert + 1) * intermediate],
                GGMLQuantizationType.Q4_K,
            ),
            dtype=np.float32,
        )
        up = np.asarray(
            dequantize_gguf_data(
                up_raw[expert * intermediate : (expert + 1) * intermediate],
                GGMLQuantizationType.Q4_K,
            ),
            dtype=np.float32,
        )
        block = rounded[start : start + count]
        expected[start : start + count, :intermediate] = block @ gate.T
        expected[start : start + count, intermediate:] = block @ up.T
        start += int(count)
    assert start == rows

    starts = np.zeros(num_experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    real_pack = leaf_module.gguf_q8_1_mmq_ds4_pack_bf16_d4x3
    block_bytes = 144

    def pack_without_later_sums(
        x_bf16_ptr: int, out_q8_ptr: int, n_rows: int, hidden_size: int, **kwargs
    ) -> None:
        """Pack three planes, then blank the stored sums of planes 1 and 2."""
        real_pack(x_bf16_ptr, out_q8_ptr, n_rows, hidden_size, **kwargs)
        per_plane = n_rows * (hidden_size // 128) * block_bytes
        for plane in (1, 2):
            view = np.empty(per_plane, dtype=np.uint8)
            buffer = DeviceBuffer(ptr=out_q8_ptr + plane * per_plane, nbytes=per_plane)
            copy_device_to_host(host_array_ptr(view), buffer, per_plane)
            view = view.reshape(-1, block_bytes).copy()
            view[:, :16].view(np.uint16).reshape(-1, 8)[:, 1::2] = 0
            copy_host_array_to_device(buffer, view.reshape(-1))

    def run(zero_later_sums: bool) -> np.ndarray:
        monkeypatch.setattr(gemma4_experts, "_MMQ_ACTIVATION_PASSES", 3)
        monkeypatch.setattr(
            leaf_module, "gguf_q8_1_mmq_ds4_pack_bf16", real_pack
        )
        monkeypatch.setattr(
            leaf_module,
            "gguf_q4_k_selected_dual_q8_1_ds4_mmq32_fused_prefill_compact32_bf16_bf16_out",
            leaf_module.gguf_q4_k_selected_dual_q8_1_ds4x3_fused_mmq32_prefill_compact32_bf16_bf16_out,
        )
        if zero_later_sums:
            monkeypatch.setattr(
                leaf_module, "gguf_q8_1_mmq_ds4_pack_bf16", pack_without_later_sums
            )

        hidden_buf = malloc(hidden_bits.nbytes)
        weights_buf = malloc(fused_raw.nbytes)
        starts_buf = malloc(starts.nbytes)
        out_buf = malloc(rows * fused_width * 2)
        scratch = None
        try:
            copy_host_array_to_device(hidden_buf, hidden_bits)
            copy_host_array_to_device(weights_buf, fused_raw)
            copy_host_array_to_device(starts_buf, starts)
            weight = _ResidentWeight(
                backend=backend, quant_key="gguf_q4_k", ptr=weights_buf.ptr
            )
            scratch = Gemma4ExpertScratch(
                tokens=rows,
                top_k=1,
                hidden_size=in_features,
                intermediate=intermediate,
                num_experts=num_experts,
            )
            served = gemma4_project_experts_gate_up_mmq(
                weight,
                hidden_buf.ptr,
                out_buf.ptr,
                SimpleNamespace(ptr=starts_buf.ptr),
                rows,
                num_experts,
                in_features,
                intermediate,
                scratch=scratch,
            )
            assert served is True, "the fused MMQ gate_up route declined a Q4_K weight"
            got = np.empty((rows, fused_width), dtype=np.uint16)
            copy_device_to_host(
                int(got.ctypes.data),
                DeviceBuffer(ptr=out_buf.ptr, nbytes=got.nbytes),
                got.nbytes,
            )
        finally:
            if scratch is not None:
                scratch.free()
            for buffer in (hidden_buf, weights_buf, starts_buf, out_buf):
                free(buffer)
        return got.copy()

    natural = run(zero_later_sums=False)
    blanked = run(zero_later_sums=True)
    differing = int((natural != blanked).sum())
    assert differing == 0, (
        "the three-plane leaf read the later planes' stored sums, so its Q4_K "
        "min-offset correction was applied once per plane: "
        f"{differing} of {natural.size} output elements changed when those sums "
        "were blanked"
    )


@_needs_hip
def test_fused_gate_up_mmq_route_holds_at_the_model_geometry_and_every_fragmentation() -> None:
    """The route's own tile walk, over the count patterns a real router produces.

    The route's only other coverage is four experts over ten rows, which cannot
    reach the model's 128 experts or the fragmented counts a short prefill
    produces. This sweeps both, reusing one scratch across calls the way the real
    path does across layers and chunks, and holds a canary after the output
    because the tile walk pads every expert up to a whole 32-row tile and reports
    the padded total as its row count - so a leaf that wrote its padding would
    run past the buffer the caller sized for compact rows only.
    """

    from types import SimpleNamespace

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        Gemma4ExpertScratch,
        gemma4_project_experts_gate_up_mmq,
    )
    from hipengine.quant.gguf import dequantize_gguf_data

    backend = _grouped_backend("gguf_q4_k")
    num_experts = 128
    in_features = 256
    intermediate = 64
    fused_width = 2 * intermediate
    canary_rows = 32 * num_experts
    canary_fill = 0xABAB
    rng = np.random.default_rng(20260927)

    gate_raw = np.concatenate(
        [make_q4_k_weight(intermediate, in_features) for _ in range(num_experts)]
    )
    up_raw = np.concatenate(
        [make_q4_k_weight(intermediate, in_features) for _ in range(num_experts)]
    ).copy()
    up_raw[:, 1::2] ^= np.uint8(0x04)
    fused_raw = np.concatenate(
        [
            gate_raw.reshape(num_experts, intermediate, -1),
            up_raw.reshape(num_experts, intermediate, -1),
        ],
        axis=1,
    ).reshape(num_experts * fused_width, -1)
    weights_buf = malloc(fused_raw.nbytes)
    copy_host_array_to_device(weights_buf, fused_raw)
    weight = _ResidentWeight(
        backend=backend, quant_key="gguf_q4_k", ptr=weights_buf.ptr
    )

    patterns = {
        "one lane per expert": np.full(num_experts, 1, dtype=np.int64),
        "three lanes per expert": np.full(num_experts, 3, dtype=np.int64),
        "a whole tile per expert": np.full(num_experts, 32, dtype=np.int64),
        "one expert active": np.concatenate(
            [np.ones(1, dtype=np.int64), np.zeros(num_experts - 1, dtype=np.int64)]
        ),
        "thirty-two experts active": np.concatenate(
            [np.full(32, 8, dtype=np.int64), np.zeros(num_experts - 32, dtype=np.int64)]
        ),
    }
    # One scratch for every pattern: the real path reuses one across layers and
    # across chunks, so a stale identity or a stale tile total has to show here.
    scratch = Gemma4ExpertScratch(
        tokens=int(max(int(c.sum()) for c in patterns.values())),
        top_k=1,
        hidden_size=in_features,
        intermediate=intermediate,
        num_experts=num_experts,
    )
    try:
        for name, counts in patterns.items():
            rows = int(counts.sum())
            hidden = rng.standard_normal((rows, in_features)).astype(np.float32)
            hidden_bits = _to_bf16_bits(hidden)
            rounded = _from_bf16_bits(hidden_bits)
            expected = np.zeros((rows, fused_width), dtype=np.float32)
            start = 0
            for expert, count in enumerate(counts):
                if count == 0:
                    continue
                gate = np.asarray(
                    dequantize_gguf_data(
                        gate_raw[expert * intermediate : (expert + 1) * intermediate],
                        GGMLQuantizationType.Q4_K,
                    ),
                    dtype=np.float32,
                )
                up = np.asarray(
                    dequantize_gguf_data(
                        up_raw[expert * intermediate : (expert + 1) * intermediate],
                        GGMLQuantizationType.Q4_K,
                    ),
                    dtype=np.float32,
                )
                block = rounded[start : start + count]
                expected[start : start + count, :intermediate] = block @ gate.T
                expected[start : start + count, intermediate:] = block @ up.T
                start += int(count)
            starts = np.zeros(num_experts + 1, dtype=np.int64)
            starts[1:] = np.cumsum(counts)

            hidden_buf = malloc(hidden_bits.nbytes)
            starts_buf = malloc(starts.nbytes)
            out_bytes = rows * fused_width * 2
            out_buf = malloc(out_bytes + canary_rows * fused_width * 2)
            try:
                copy_host_array_to_device(hidden_buf, hidden_bits)
                copy_host_array_to_device(starts_buf, starts)
                canary = np.full((canary_rows, fused_width), canary_fill, dtype=np.uint16)
                copy_host_array_to_device(
                    DeviceBuffer(ptr=out_buf.ptr + out_bytes, nbytes=canary.nbytes),
                    canary,
                )
                served = gemma4_project_experts_gate_up_mmq(
                    weight,
                    hidden_buf.ptr,
                    out_buf.ptr,
                    SimpleNamespace(ptr=starts_buf.ptr),
                    rows,
                    num_experts,
                    in_features,
                    intermediate,
                    scratch=scratch,
                )
                assert served is True, f"{name}: the route declined a Q4_K weight"
                got = np.empty((rows, fused_width), dtype=np.uint16)
                copy_device_to_host(
                    int(got.ctypes.data),
                    DeviceBuffer(ptr=out_buf.ptr, nbytes=got.nbytes),
                )
                tail = np.empty((canary_rows, fused_width), dtype=np.uint16)
                copy_device_to_host(
                    int(tail.ctypes.data),
                    DeviceBuffer(ptr=out_buf.ptr + out_bytes, nbytes=tail.nbytes),
                )
                got = _from_bf16_bits(got)
            finally:
                for buffer in (hidden_buf, starts_buf, out_buf):
                    free(buffer)

            clobbered = int(np.count_nonzero(tail != canary_fill))
            assert clobbered == 0, (
                f"{name}: the route wrote {clobbered} canary words past the "
                f"{rows}-row output buffer"
            )
            scale = float(np.abs(expected).max())
            difference = np.abs(got - expected)
            assert float(difference.max()) < 2e-2 * scale, (
                f"{name}: normalized max {float(difference.max()) / scale:.4g} "
                f"against scale {scale:.4g}"
            )
            assert float(difference.mean()) < 2e-3 * scale, (
                f"{name}: normalized mean {float(difference.mean()) / scale:.4g}"
            )
    finally:
        scratch.free()
        free(weights_buf)


@_needs_hip
@pytest.mark.parametrize(
    "quant_key, make_weight",
    [("gguf_q5_1", make_q5_1_weight), ("gguf_q8_0", make_q8_0_weight)],
)
def test_expert_down_mmq_route_holds_at_gemma4_s_own_down_geometry(
    quant_key, make_weight
) -> None:
    """The DS4 DP4A down route at 704 -> 2816 for both quants it accepts.

    Gemma 4 26B-A4B UD-Q4_K_XL quantizes 29 of its 30 expert down projections
    Q5_1 and the last one Q8_0, so a route that accepts only Q5_1 sends exactly
    one layer of thirty to the fp32 grouped family -- measured at 43.85 ms of a
    873 ms 512-token prefill against 5.30 ms for a sibling at the same shape.
    Both consumers share the DS4 activation pack and differ only in the weight
    decode, so both are held here to the same envelope the Q4_K gate/up MMQ
    route runs at: this route quantizes the block's activations, so its error is
    bounded by that step rather than by reassociation. The oracle is an
    independent dequantized reference rather than another of our own kernels,
    which is what ``gemma4_project_experts_rows`` falls back to for these two
    quants -- neither registers a ``grouped_row4`` owner.

    704 is the width that makes this shape interesting. It is 5 x 128 + 64, so
    the trailing DS4 block is partial, and flooring the block count instead of
    rounding it up silently drops those last 64 columns.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        gemma4_project_experts_down_mmq,
    )
    from hipengine.quant.gguf import dequantize_gguf_data

    backend = _grouped_backend(quant_key)
    num_experts = 8
    in_features = 704
    out_features = 2816
    counts = np.full(num_experts, 4, dtype=np.int64)
    rows = int(counts.sum())

    rng = np.random.default_rng(20260928)
    raw = np.concatenate(
        [make_weight(out_features, in_features) for _ in range(num_experts)]
    )
    weights_buf = malloc(raw.nbytes)
    hidden = rng.standard_normal((rows, in_features)).astype(np.float32)
    hidden_bits = _to_bf16_bits(hidden)
    hidden_buf = malloc(hidden_bits.nbytes)
    starts = np.zeros(num_experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    starts_buf = malloc(starts.nbytes)
    out_bytes = rows * out_features * 2
    route_buf = malloc(out_bytes)
    scratch = Gemma4ExpertScratch(
        tokens=rows,
        top_k=1,
        hidden_size=out_features,
        intermediate=in_features,
        num_experts=num_experts,
    )
    try:
        copy_host_array_to_device(weights_buf, raw)
        copy_host_array_to_device(hidden_buf, hidden_bits)
        copy_host_array_to_device(starts_buf, starts)
        weight = _ResidentWeight(
            backend=backend, quant_key=quant_key, ptr=weights_buf.ptr
        )
        ggml_type = (
            GGMLQuantizationType.Q5_1 if quant_key == "gguf_q5_1"
            else GGMLQuantizationType.Q8_0
        )

        served = gemma4_project_experts_down_mmq(
            weight,
            hidden_buf.ptr,
            route_buf.ptr,
            SimpleNamespace(ptr=starts_buf.ptr),
            rows,
            num_experts,
            in_features,
            out_features,
            scratch=scratch,
        )
        assert served is True, (
            f"the {quant_key} DS4 down route declined Gemma 4's own down "
            f"geometry, so that layer runs the fp32 grouped fallback"
        )

        # The reference reads the bf16-rounded activations the kernel reads, and
        # each expert's own rows, dequantized exactly. A second reference drops
        # the last 64 contraction columns -- the partial trailing DS4 block --
        # so the route can be shown to be closer to the full one than to that.
        rounded = _from_bf16_bits(hidden_bits)
        expected = np.zeros((rows, out_features), dtype=np.float32)
        truncated = np.zeros((rows, out_features), dtype=np.float32)
        tail_start = (in_features // 128) * 128
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            dequantized = np.asarray(
                dequantize_gguf_data(
                    raw[expert * out_features : (expert + 1) * out_features],
                    ggml_type,
                ),
                dtype=np.float32,
            )
            block = rounded[start : start + count]
            expected[start : start + count] = block @ dequantized.T
            truncated[start : start + count] = (
                block[:, :tail_start] @ dequantized[:, :tail_start].T
            )
            start += int(count)

        def read(buf):
            out = np.empty((rows, out_features), dtype=np.uint16)
            copy_device_to_host(
                int(out.ctypes.data), DeviceBuffer(ptr=buf.ptr, nbytes=out.nbytes)
            )
            return _from_bf16_bits(out)

        got = read(route_buf)
        scale = float(np.abs(expected).max())
        difference = np.abs(got - expected)
        assert float(difference.max()) < 2e-2 * scale, (
            f"{quant_key}: normalized max "
            f"{float(difference.max()) / scale:.4g} against scale {scale:.4g}"
        )
        assert float(difference.mean()) < 2e-3 * scale, (
            f"{quant_key}: normalized mean "
            f"{float(difference.mean()) / scale:.4g}"
        )
        # 704 is 5 * 128 + 64, so flooring the DS4 block count instead of
        # rounding it up drops the last 64 of the contraction. This is the
        # failure the Q5_1 route already hit once, so the test proves it can see
        # it: the route must be closer to the full reference than to the one
        # with the tail removed.
        full_error = float(np.abs(got - expected).max())
        tail_error = float(np.abs(got - truncated).max())
        assert tail_error > full_error, (
            f"{quant_key}: the route is as close to a reference with the "
            f"trailing {in_features - tail_start} contraction columns removed "
            f"({tail_error:.4g}) as to the full one ({full_error:.4g}), so the "
            f"partial trailing DS4 block is not being read"
        )
    finally:
        scratch.free()
        for buffer in (weights_buf, hidden_buf, starts_buf, route_buf):
            free(buffer)


@_needs_hip
@pytest.mark.parametrize(
    "quant_key, make_weight",
    [("gguf_q5_1", make_q5_1_weight), ("gguf_q8_0", make_q8_0_weight)],
)
def test_expert_down_mmq_route_holds_past_the_fp16_scale_range(
    quant_key, make_weight
) -> None:
    """The DS4 down route must stay accurate where an fp16 scale cannot.

    A DS4 block stores its per-32 scale and its block sum as fp16, and the scale
    is ``amax/127``, so the format runs out at ``amax`` of 65504 * 127, about
    8.3e6. Q5_1 runs out far sooner, because its extra ``m * sum`` term needs
    the *block sum* to fit fp16 as well: 32 values of 2048 already overflow it.

    A down projection's input is a GeGLU output, and this is not hypothetical.
    The gemma4 test fixture's own activations reach 7.6e7, which is 9x past the
    Q8_0 limit, and before the pack clamped its scale the route turned that into
    NaN. A NaN there is not local: it flows through the residual stream and the
    next layer's router returns its -1 no-expert sentinel for every lane, which
    makes the following gate/up call see an empty expert range and raise.

    The fp32 layout is exact here rather than merely finite, so this asserts the
    same envelope the in-range case does. Finiteness alone would pass a route
    that saturates: an fp16 pack with its scale clamped to 65504 stays finite
    and is wrong by 90 percent, which the 5 percent tolerance the runner tests
    use against a fp32 reference rejects.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        gemma4_project_experts_down_mmq,
    )
    from hipengine.quant.gguf import dequantize_gguf_data

    backend = _grouped_backend(quant_key)
    num_experts = 8
    in_features = 704
    out_features = 2816
    counts = np.full(num_experts, 4, dtype=np.int64)
    rows = int(counts.sum())

    rng = np.random.default_rng(20260928)
    raw = np.concatenate(
        [make_weight(out_features, in_features) for _ in range(num_experts)]
    )
    weights_buf = malloc(raw.nbytes)
    # 7.6e7 is the measured peak of the fixture's own GeGLU output at its tiny
    # geometry, and it is 9x past what the DS4 fp16 scale can hold.
    hidden = (rng.standard_normal((rows, in_features)) * 2.5e7).astype(np.float32)
    hidden_bits = _to_bf16_bits(hidden)
    hidden_buf = malloc(hidden_bits.nbytes)
    starts = np.zeros(num_experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    starts_buf = malloc(starts.nbytes)
    route_buf = malloc(rows * out_features * 2)
    scratch = Gemma4ExpertScratch(
        tokens=rows,
        top_k=1,
        hidden_size=out_features,
        intermediate=in_features,
        num_experts=num_experts,
    )
    try:
        copy_host_array_to_device(weights_buf, raw)
        copy_host_array_to_device(hidden_buf, hidden_bits)
        copy_host_array_to_device(starts_buf, starts)
        weight = _ResidentWeight(
            backend=backend, quant_key=quant_key, ptr=weights_buf.ptr
        )
        served = gemma4_project_experts_down_mmq(
            weight,
            hidden_buf.ptr,
            route_buf.ptr,
            SimpleNamespace(ptr=starts_buf.ptr),
            rows,
            num_experts,
            in_features,
            out_features,
            scratch=scratch,
        )
        assert served is True

        out = np.empty((rows, out_features), dtype=np.uint16)
        copy_device_to_host(
            int(out.ctypes.data), DeviceBuffer(ptr=route_buf.ptr, nbytes=out.nbytes)
        )
        got = _from_bf16_bits(out)
        assert np.isfinite(got).all(), (
            f"{quant_key}: {int(np.isnan(got).sum())} NaN and "
            f"{int(np.isinf(got).sum())} inf of {got.size} outputs past the fp16 "
            f"scale range, which poisons the residual stream"
        )
        ggml_type = (
            GGMLQuantizationType.Q5_1 if quant_key == "gguf_q5_1"
            else GGMLQuantizationType.Q8_0
        )
        rounded = _from_bf16_bits(hidden_bits)
        expected = np.zeros((rows, out_features), dtype=np.float32)
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            dequantized = np.asarray(
                dequantize_gguf_data(
                    raw[expert * out_features : (expert + 1) * out_features],
                    ggml_type,
                ),
                dtype=np.float32,
            )
            expected[start : start + count] = (
                rounded[start : start + count] @ dequantized.T
            )
            start += int(count)
        scale = float(np.abs(expected).max())
        difference = np.abs(got - expected)
        assert float(difference.max()) < 2e-2 * scale, (
            f"{quant_key}: normalized max {float(difference.max()) / scale:.4g} "
            f"past the fp16 scale range, where an fp16 activation layout can only "
            f"saturate"
        )
        assert float(difference.mean()) < 2e-3 * scale
    finally:
        scratch.free()
        for buffer in (weights_buf, hidden_buf, starts_buf, route_buf):
            free(buffer)


@_needs_hip
def test_fused_gate_up_iu8_route_holds_at_gemma4_s_own_half_width() -> None:
    """The Q5_K iu8-WMMA gate/up route at the half width this artifact ships.

    Gemma 4 26B-A4B's expert intermediate is 704, which is 5 x 128 + 64. The
    Q5_K iu8 route used to decline on ``intermediate % 128`` because its leaf
    walks output columns as ``blockIdx.x * 128``, so 704 would put the gate/up
    seam in the middle of a column block. The seam is not a problem the kernel
    has to solve at the block level -- it resolves the half per *column*, so a
    straddling block already reads the right weight for each of its columns.

    What the guard was hiding is the expert stride. The Q4_K route and this one
    both pass ``weight_ptr`` and ``weight_ptr + intermediate * row_bytes``, but
    Gemma 4 stores ``ffn_gate_up_exps`` as ``(num_experts, 2 * intermediate,
    hidden)`` with the gate rows first *per expert*, so an expert's stride is
    the fused width. A leaf that strides by its own half width reads the wrong
    expert's rows from the second expert on, which the 128 guard kept out of
    reach rather than surfacing.

    This holds the route to the strict ``grouped_row4`` owner at the artifact's
    own width, bit for bit, which is the contract the route's docstring claims.
    """

    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        gemma4_project_experts_grouped_row4,
    )

    backend = _grouped_backend("gguf_q5_k")
    num_experts = 8
    in_features = 2816
    # Gemma 4 26B-A4B's own expert intermediate: 704 = 5 * 128 + 64.
    intermediate = 704
    fused_width = 2 * intermediate
    counts = np.full(num_experts, 4, dtype=np.int64)
    rows = int(counts.sum())

    rng = np.random.default_rng(20260928)
    gate_raw = np.concatenate(
        [make_q5_k_weight(intermediate, in_features) for _ in range(num_experts)]
    )
    up_raw = np.concatenate(
        [make_q5_k_weight(intermediate, in_features) for _ in range(num_experts)]
    ).copy()
    # The synthetic fixture is deterministic, so the two halves would otherwise
    # be byte-identical and a wrong-half read would return the right values.
    up_raw[:, 1::2] ^= np.uint8(0x04)
    fused_raw = np.concatenate(
        [
            gate_raw.reshape(num_experts, intermediate, -1),
            up_raw.reshape(num_experts, intermediate, -1),
        ],
        axis=1,
    ).reshape(num_experts * fused_width, -1)

    weights_buf = malloc(fused_raw.nbytes)
    hidden = rng.standard_normal((rows, in_features)).astype(np.float32)
    hidden_bits = _to_bf16_bits(hidden)
    hidden_buf = malloc(hidden_bits.nbytes)
    starts = np.zeros(num_experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    starts_buf = malloc(starts.nbytes)
    out_bytes = rows * fused_width * 2
    route_buf = malloc(out_bytes)
    owner_buf = malloc(out_bytes)
    scratch = Gemma4ExpertScratch(
        tokens=rows,
        top_k=1,
        hidden_size=in_features,
        intermediate=intermediate,
        num_experts=num_experts,
    )
    try:
        copy_host_array_to_device(weights_buf, fused_raw)
        copy_host_array_to_device(hidden_buf, hidden_bits)
        copy_host_array_to_device(starts_buf, starts)
        weight = _ResidentWeight(
            backend=backend, quant_key="gguf_q5_k", ptr=weights_buf.ptr
        )

        owner_served = gemma4_project_experts_grouped_row4(
            weight,
            hidden_buf.ptr,
            starts_buf.ptr,
            owner_buf.ptr,
            rows,
            num_experts,
            in_features,
            fused_width,
        )
        assert owner_served is True, "the strict row4 owner declined a Q5_K weight"

        served = gemma4_project_experts_gate_up_mmq(
            weight,
            hidden_buf.ptr,
            route_buf.ptr,
            SimpleNamespace(ptr=starts_buf.ptr),
            rows,
            num_experts,
            in_features,
            intermediate,
            scratch=scratch,
        )
        assert served is True, (
            "the Q5_K iu8 gate/up route declined the artifact's own 704-wide "
            "expert half, so this layer runs the fp32 grouped fallback"
        )

        def read(buf):
            out = np.empty((rows, fused_width), dtype=np.uint16)
            copy_device_to_host(
                int(out.ctypes.data), DeviceBuffer(ptr=buf.ptr, nbytes=out.nbytes)
            )
            return _from_bf16_bits(out)

        got, want = read(route_buf), read(owner_buf)
        differing = int(np.count_nonzero(got != want))
        scale = float(np.abs(want).max())
        assert differing == 0, (
            f"the iu8 route differs from the strict row4 owner at "
            f"{differing} of {got.size} outputs; normalized max "
            f"{float(np.abs(got - want).max()) / scale:.4g} against scale "
            f"{scale:.4g}. A non-zero difference that grows with the expert "
            f"index is the fused-stride read."
        )
    finally:
        scratch.free()
        for buffer in (weights_buf, hidden_buf, starts_buf, route_buf, owner_buf):
            free(buffer)


@pytest.mark.parametrize(
    "quant_key, in_features, intermediate",
    [
        # The DS4 pack reads 128-element groups and the tile walk reads 32-wide
        # output columns, so a geometry that does not divide declines rather
        # than launching a kernel that would read past its rows.
        ("gguf_q4_k", 96, 64),
        ("gguf_q4_k", 256, 48),
        # Only the Q4_K family has a DS4 MMQ32 leaf; another quant key declines
        # so the caller keeps its fp32 grouped route.
        ("gguf_q5_1", 256, 64),
    ],
)
def test_fused_gate_up_mmq_route_declines_what_it_cannot_execute(
    quant_key: str, in_features: int, intermediate: int
) -> None:
    """The route's capability test is the geometry and the quant key, not a name.

    Only declining geometries are listed: a qualifying weight would launch the
    real leaf against these placeholder pointers, and the positive case is
    covered by the dequantized-reference test above.
    """

    weight = _ResidentWeight(backend="hip_gfx1100", quant_key=quant_key)
    scratch = Gemma4ExpertScratch(
        tokens=8,
        top_k=1,
        hidden_size=in_features,
        intermediate=intermediate,
        num_experts=4,
    )
    try:
        served = gemma4_project_experts_gate_up_mmq(
            weight,
            0x1000,
            0x2000,
            SimpleNamespace(ptr=0x3000),
            8,
            4,
            in_features,
            intermediate,
            scratch=scratch,
        )
    finally:
        scratch.free()
    assert served is False


def test_fused_gate_up_mmq_route_declines_a_bf16_weight() -> None:
    """A bf16 expert weight has no quantized bytes to read, so the route declines."""

    scratch = Gemma4ExpertScratch(
        tokens=8, top_k=1, hidden_size=256, intermediate=64, num_experts=4
    )
    try:
        served = gemma4_project_experts_gate_up_mmq(
            0x5A0000, 0x1000, 0x2000, SimpleNamespace(ptr=0x3000), 8, 4, 256, 64,
            scratch=scratch,
        )
    finally:
        scratch.free()
    assert served is False


def test_fused_gate_up_mmq_route_ships_on_and_the_variable_rolls_it_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route is the default, and the variable that once enabled it now disables it.

    The route is the measured faster path at 1.27x on the 512/128 prefill, and the
    decision that held it back has cleared, so it ships on. Both halves are pinned
    because the campaign's harnesses select their arms through this function: when
    the sense of the variable changed, popping it stopped meaning "fp32" and would
    have had the probes compare the route with itself while reporting agreement.
    """

    monkeypatch.delenv("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", raising=False)
    assert gemma4_moe_gate_up_mmq_enabled() is True

    for value in ("1", "true", "yes", "on", "  ON  "):
        monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", value)
        assert gemma4_moe_gate_up_mmq_enabled() is True, value

    for value in ("0", "false", "no", "off", "disable", "disabled", " 0 "):
        monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", value)
        assert gemma4_moe_gate_up_mmq_enabled() is False, value

    # An unrecognised value must not silently downgrade the default path.
    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", "banana")
    assert gemma4_moe_gate_up_mmq_enabled() is True
