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
list of known-good combinations. These tests pin that decision and its fallback
order on the CPU, with every kernel either stubbed or reached only through a
route that declines before launching. The device-execution parity checks -- the
grouped kernel against a dequantized reference, and the bit-exactness that lets
the grouped route replace the per-lane one without a numerical gate -- run in
``test_gpu_gemma4_expert_route.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

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
from tests._gemma4_expert_route_fixtures import _ResidentWeight

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


@pytest.fixture
def strict_selected_route(monkeypatch):
    """Pin the strict selected chain for tests that stub its owners.

    These tests register stand-ins for the selected owners and call with
    placeholder pointers. The unpinned production route would first offer the
    real eight-input-run sibling, which is not stubbed, and launching it on the
    placeholders faults the device; the next synchronizing call then never
    returns. The chain under test is the strict one, so the route is pinned.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        _PREFILL_MODE_ENV,
        _SELECTED_ROUTE_CACHE,
    )

    monkeypatch.setenv(_PREFILL_MODE_ENV, "grouped")
    _SELECTED_ROUTE_CACHE.clear()
    yield
    _SELECTED_ROUTE_CACHE.clear()


@pytest.mark.usefixtures("strict_selected_route")
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


@pytest.mark.usefixtures("strict_selected_route")
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


@pytest.mark.usefixtures("strict_selected_route")
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


@pytest.mark.usefixtures("strict_selected_route")
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


@pytest.mark.usefixtures("strict_selected_route")
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


@pytest.mark.usefixtures("strict_selected_route")
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


@pytest.mark.usefixtures("strict_selected_route")
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


# ---------------------------------------------------------------------------
# Route accounting
#
# Every expert projection moves exactly one route counter, under the name of the
# leaf that ran it. The fused gate/up wrapper is where that name is decided: it
# is the only place that knows which of its leaves served the call, and two of
# those leaves -- the Q4T16 tile route and the Q5_K iu8 route -- are not the
# MMQ32 route the wrapper is named after.
# ---------------------------------------------------------------------------


class _TiledResidentWeight(_ResidentWeight):
    """A resident Q4_K expert stack that also carries its Q4T16 tiles."""

    def __init__(self, *, backend: str) -> None:
        super().__init__(backend=backend, quant_key="gguf_q4_k")
        self.allocations = {
            "raw": SimpleNamespace(buffer=SimpleNamespace(ptr=self.ptr)),
            "t16_gate": SimpleNamespace(buffer=SimpleNamespace(ptr=self.ptr + 1)),
            "t16_up": SimpleNamespace(buffer=SimpleNamespace(ptr=self.ptr + 2)),
        }


class _AccountingScratch:
    """A scratch stand-in for the block's accounting test.

    The subject is which route counter moves, not what the kernels compute, so
    every kernel the block calls is stubbed and the buffers only have to carry a
    pointer.
    """

    def __init__(
        self, *, tokens: int, top_k: int, hidden_size: int, intermediate: int, num_experts: int
    ) -> None:
        self.tokens = tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate = intermediate
        self.num_experts = num_experts

    def buffer(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(ptr=0x10000 + 8 * len(name), nbytes=1 << 20)


def _route_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0)}


@pytest.mark.parametrize(
    ("route", "leaf"),
    [
        ("gate_up_t16", "_gemma4_project_experts_gate_up_wmma_t16"),
        ("gate_up_iu8", "_gemma4_project_experts_gate_up_wmma_iu8"),
    ],
)
def test_the_fused_gate_up_wrapper_counts_the_leaf_that_served_it(
    monkeypatch: pytest.MonkeyPatch, route: str, leaf: str
) -> None:
    """A served projection is counted under its own leaf's name, exactly once.

    The wrapper dispatches to leaves whose names differ from its own, and the
    counter has to follow the leaf that ran. Counting the family instead
    attributes a Q4T16 tile launch and a Q5_K iu8 launch to ``gate_up_mmq32``,
    a route neither of them ran.
    """

    import hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts as experts

    served: list[str] = []

    def stub(*args, **kwargs) -> bool:
        served.append(route)
        return True

    monkeypatch.setattr(experts, leaf, stub)
    # A Q5_K expert tensor is not a Q4_K one, so the iu8 case is a weight with no
    # tiles at all; the tile route's case is the stack the loader plans both
    # layouts for.
    weight = (
        _ResidentWeight(backend="hip_gfx1100", quant_key="gguf_q5_k")
        if route == "gate_up_iu8"
        else _TiledResidentWeight(backend="hip_gfx1100")
    )
    scratch = Gemma4ExpertScratch(
        tokens=8, top_k=1, hidden_size=256, intermediate=64, num_experts=4
    )
    before = gemma4_moe_expert_route_counts()
    try:
        served_route = gemma4_project_experts_gate_up_mmq(
            weight,
            0x1000,
            0x2000,
            SimpleNamespace(ptr=0x3000),
            8,
            4,
            256,
            64,
            scratch=scratch,
        )
    finally:
        scratch.free()

    assert served_route is True
    assert served == [route], "the wrapper did not reach the leaf this case names"
    delta = _route_delta(before, gemma4_moe_expert_route_counts())
    assert delta == {route: 1}, (
        f"the {route} leaf served one projection but the counters moved by {delta}"
    )


def test_the_expert_block_counts_one_gate_up_route_per_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block records one gate/up name per projection, never two.

    The wrapper records the leaf that served the projection, so the caller must
    not add a second name for the same launch. With the Q4T16 tiles live in
    production that double count made ``gate_up_mmq32`` report every tile-route
    launch, so the counters read as twice the projections the block ran.
    """

    import hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts as experts

    def no_op(*args, **kwargs) -> None:
        return None

    def gate_up_leaf(*args, **kwargs) -> bool:
        # What the wrapper does once the Q4T16 leaf reports that it served the
        # call: the leaf's own name, and nothing from the caller on top of it.
        experts._record_moe_route("gate_up_t16")
        return True

    monkeypatch.setattr(experts, "_zero", no_op)
    monkeypatch.setattr(experts, "qwen35_moe_group_compact_active", no_op)
    monkeypatch.setattr(experts, "qwen35_moe_gather_packed_hidden_lowp", no_op)
    monkeypatch.setattr(experts, "gemma4_gelu_tanh_mul_bf16", no_op)
    monkeypatch.setattr(experts, "gemma4_moe_lane_to_row_i32", no_op)
    monkeypatch.setattr(experts, "gemma4_moe_weighted_accumulate_bf16", no_op)
    monkeypatch.setattr(experts, "gemma4_project_experts_gate_up_mmq", gate_up_leaf)
    monkeypatch.setattr(experts, "gemma4_project_experts_down_mmq", lambda *a, **k: True)
    monkeypatch.setattr(experts, "gemma4_project_experts_rows", lambda *a, **k: "selected_gemv")

    scratch = _AccountingScratch(
        tokens=8, top_k=1, hidden_size=256, intermediate=64, num_experts=4
    )
    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_DOWN_MMQ", "1")
    before = gemma4_moe_expert_route_counts()
    experts.gemma4_experts_forward_bf16(
        0x1000,
        0x2000,
        0x3000,
        SimpleNamespace(),
        SimpleNamespace(),
        0x4000,
        scratch=scratch,
        rows=8,
    )
    delta = _route_delta(before, gemma4_moe_expert_route_counts())
    assert delta == {"gate_up_t16": 1, "down_mmq32": 1}, (
        "one gate/up projection and one down projection ran, so exactly one name "
        f"per projection is expected; the counters moved by {delta}"
    )


def test_the_expert_block_counts_the_ladder_route_when_the_mmq_leaves_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declining MMQ leaf leaves the accounting to the ladder route.

    The caller records the ladder route only when no MMQ-family leaf served the
    projection, so a refusal must not consume a counter of its own.
    """

    import hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts as experts

    def no_op(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(experts, "_zero", no_op)
    monkeypatch.setattr(experts, "qwen35_moe_group_compact_active", no_op)
    monkeypatch.setattr(experts, "qwen35_moe_gather_packed_hidden_lowp", no_op)
    monkeypatch.setattr(experts, "gemma4_gelu_tanh_mul_bf16", no_op)
    monkeypatch.setattr(experts, "gemma4_moe_lane_to_row_i32", no_op)
    monkeypatch.setattr(experts, "gemma4_moe_weighted_accumulate_bf16", no_op)
    monkeypatch.setattr(experts, "gemma4_project_experts_gate_up_mmq", lambda *a, **k: False)
    monkeypatch.setattr(experts, "gemma4_project_experts_down_mmq", lambda *a, **k: False)
    monkeypatch.setattr(experts, "gemma4_project_experts_rows", lambda *a, **k: "grouped_prefill")

    scratch = _AccountingScratch(
        tokens=8, top_k=1, hidden_size=256, intermediate=64, num_experts=4
    )
    before = gemma4_moe_expert_route_counts()
    experts.gemma4_experts_forward_bf16(
        0x1000,
        0x2000,
        0x3000,
        SimpleNamespace(),
        SimpleNamespace(),
        0x4000,
        scratch=scratch,
        rows=8,
    )
    delta = _route_delta(before, gemma4_moe_expert_route_counts())
    assert delta == {"grouped_prefill": 2}, (
        "both projections fell to the ladder, so its route is the only name "
        f"that may move; the counters moved by {delta}"
    )


@pytest.mark.parametrize(
    ("mode", "lever", "lanes", "expected_down"),
    [
        # The unpinned default with at least 16 lanes per expert: WMMA down.
        ("auto", None, 64, "down_wmma"),
        ("auto", "1", 64, "down_wmma"),
        # The lever restores the ladder's exact grouped owner.
        ("auto", "0", 64, "grouped_prefill"),
        # A pinned exact mode never takes it, whatever the lever says.
        ("grouped", "1", 64, "grouped_prefill"),
        # Too few lanes for a tile plan: the ladder serves the block.
        ("auto", None, 8, "grouped_prefill"),
    ],
)
def test_the_unpinned_default_runs_the_down_through_wmma(
    monkeypatch: pytest.MonkeyPatch, mode, lever, lanes, expected_down,
) -> None:
    """The default down projection is the WMMA owner behind its own lever."""

    import hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts as experts

    def no_op(*args, **kwargs) -> None:
        return None

    plans: list[int] = []

    def tile_plan(scratch, expert_start_ptr, lane_count, **kwargs) -> int:
        plans.append(lane_count)
        return lane_count

    def wmma_down(*args, **kwargs) -> bool:
        experts._record_moe_route("down_wmma")
        return True

    for name in (
        "_zero", "qwen35_moe_group_compact_active", "qwen35_moe_gather_packed_hidden_lowp",
        "gemma4_gelu_tanh_mul_bf16", "gemma4_moe_lane_to_row_i32",
        "gemma4_moe_weighted_accumulate_bf16",
    ):
        monkeypatch.setattr(experts, name, no_op)
    monkeypatch.setattr(experts, "_build_wmma_tile_plan", tile_plan)
    monkeypatch.setattr(experts, "_fused_wmma_owner", lambda *a, **k: None)
    monkeypatch.setattr(experts, "gemma4_project_experts_gate_up_mmq", lambda *a, **k: False)
    monkeypatch.setattr(experts, "gemma4_project_experts_down_mmq", lambda *a, **k: False)
    monkeypatch.setattr(experts, "gemma4_project_experts_grouped", lambda *a, **k: False)
    monkeypatch.setattr(experts, "gemma4_project_experts_grouped_dual", lambda *a, **k: False)
    monkeypatch.setattr(experts, "_gemma4_project_experts_down_wmma_t16", lambda *a, **k: False)
    monkeypatch.setattr(experts, "gemma4_project_experts_wmma", wmma_down)
    monkeypatch.setattr(experts, "gemma4_project_experts_rows", lambda *a, **k: "grouped_prefill")
    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_PREFILL", mode)
    if lever is None:
        monkeypatch.delenv("HIPENGINE_GEMMA4_MOE_DOWN_WMMA", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_DOWN_WMMA", lever)

    scratch = _AccountingScratch(
        tokens=lanes, top_k=1, hidden_size=256, intermediate=64, num_experts=4
    )
    before = gemma4_moe_expert_route_counts()
    experts.gemma4_experts_forward_bf16(
        0x1000, 0x2000, 0x3000, SimpleNamespace(), SimpleNamespace(), 0x4000,
        scratch=scratch, rows=lanes,
    )
    delta = _route_delta(before, gemma4_moe_expert_route_counts())
    assert delta.get(expected_down, 0) >= 1, delta
    if expected_down == "down_wmma":
        assert plans == [lanes], "the down builds its own tile plan when the gate/up built none"
        assert delta == {"grouped_prefill": 1, "down_wmma": 1}, delta
    else:
        assert "down_wmma" not in delta
        assert not plans


@pytest.mark.parametrize("enabled", [True, False])
def test_auto_t16_uses_default_mmq_family_and_honors_rollback(monkeypatch, enabled):
    """A tiled allocation must not pin the one-plane diagnostic MMQ owner."""
    import hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts as experts

    calls = []
    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_PREFILL", "auto")
    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", "1" if enabled else "0")
    # Isolate gate/up dispatch from the independent down-owner default.
    monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_DOWN_MMQ", "1")
    for name in (
        "_zero", "qwen35_moe_group_compact_active",
        "qwen35_moe_gather_packed_hidden_lowp", "gemma4_gelu_tanh_mul_bf16",
        "gemma4_moe_lane_to_row_i32", "gemma4_moe_weighted_accumulate_bf16",
    ):
        monkeypatch.setattr(experts, name, lambda *a, **k: None)
    monkeypatch.setattr(experts, "_build_mmq_tile_plan", lambda *a, **k: 256)
    monkeypatch.setattr(experts, "_build_wmma_tile_plan", lambda *a, **k: 256)
    monkeypatch.setattr(experts, "_mmq_dual_route", lambda *a, **k: True)
    monkeypatch.setattr(experts, "gemma4_project_experts_mmq_dual",
                        lambda *a, **k: calls.append("diagnostic_mmq") or True)
    monkeypatch.setattr(experts, "gemma4_project_experts_gate_up_mmq",
                        lambda *a, **k: calls.append("default_mmq") or True)
    monkeypatch.setattr(experts, "gemma4_project_experts_rows",
                        lambda *a, **k: calls.append("ladder") or "selected_gemv")
    monkeypatch.setattr(experts, "gemma4_project_experts_wmma", lambda *a, **k: True)
    monkeypatch.setattr(experts, "gemma4_project_experts_down_mmq", lambda *a, **k: True)
    scratch = _AccountingScratch(
        tokens=128, top_k=1, hidden_size=256, intermediate=64, num_experts=4
    )
    weight = SimpleNamespace(spec=SimpleNamespace(layout="gguf_q4_k_t16_v1"))
    experts.gemma4_experts_forward_bf16(
        0x1000, 0x2000, 0x3000, weight, SimpleNamespace(), 0x4000,
        scratch=scratch, rows=128,
    )
    assert calls == ["default_mmq" if enabled else "ladder"]


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_down_mmq_defaults_to_exact_after_observed_continuation_failure(monkeypatch, value, expected):
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import gemma4_moe_down_mmq_enabled
    if value is None:
        monkeypatch.delenv("HIPENGINE_GEMMA4_MOE_DOWN_MMQ", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GEMMA4_MOE_DOWN_MMQ", value)
    assert gemma4_moe_down_mmq_enabled() is expected


# --- production eight-input-run siblings on the unpinned route --------------


@pytest.mark.parametrize(
    ("quant", "tiles_quant", "has_tiles", "out_features", "production_variant", "strict_variant"),
    [
        # A stack planned in the T16 layout carries the tiles quant as its own key.
        ("gguf_q4_k_t16_v1", None, True, 1408,
         "selected_chunk8_gemv_bf16_bf16_out", "selected_gemv_bf16_bf16_out"),
        # A raw stack with a T16 side copy resolves the sibling under the tiles key.
        ("gguf_q4_k", "gguf_q4_k_t16_v1", True, 1408,
         "selected_chunk8_gemv_bf16_bf16_out", "selected_gemv_bf16_bf16_out"),
        # The Q5_1 down stack's pack8 owner and its sibling.
        ("gguf_q5_1", None, False, 2816,
         "selected_pack8_chunk8_gemv_bf16_bf16_out", "selected_pack8_gemv_bf16_bf16_out"),
    ],
)
@pytest.mark.parametrize("mode", ["", "auto", "grouped", "selected"])
def test_the_unpinned_route_takes_the_production_sibling(
    monkeypatch, quant, tiles_quant, has_tiles, out_features, production_variant,
    strict_variant, mode,
):
    from types import SimpleNamespace

    from hipengine.kernels import registry
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts
    from hipengine.runtime import gguf_linear

    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key=quant, tiles_quant_key=tiles_quant),
        backend="hip_gfx1100",
        has_allocation=lambda name: has_tiles and name == "tiles",
    )
    registered = {
        (tiles_quant or quant, production_variant),
        (tiles_quant or quant, strict_variant),
    }
    monkeypatch.setattr(registry, "is_registered", lambda key: (key.quant, key.variant) in registered)
    monkeypatch.setattr(registry, "resolve", lambda *, backend, layer, quant, variant: variant)
    monkeypatch.setattr(gguf_linear, "_ensure_linear_kernel_registered", lambda key: None)
    monkeypatch.setenv(gemma4_experts._PREFILL_MODE_ENV, mode)
    gemma4_experts._SELECTED_ROUTE_CACHE.clear()
    try:
        fn, _allocation = gemma4_experts._selected_route(weight, out_features)
    finally:
        gemma4_experts._SELECTED_ROUTE_CACHE.clear()

    assert fn == (production_variant if mode in ("", "auto") else strict_variant)
