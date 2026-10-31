"""CPU-only selected-expert caller contracts, shared by admission and runtime.

Call kind is an invocation interface, not a registry axis. Singleton, dual,
dual-SiLU and weighted-down have different operands/output ownership. The
runtime's existing primitive selection/variant policy stays below these
interfaces. Certificates are not numerical/profile qualification.

A call that consumes a Q8_1 operand -- the mandatory X8 layouts and the
optional DP4A routes over an otherwise bf16-input resident -- binds the
conversion it executes, the operand it consumes, and the plan-owned workspace
that holds the converted activation, so a record never describes arithmetic
the runtime does not run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import os
from types import MappingProxyType
from typing import Mapping

from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_RAW_GGUF, LAYOUT_GGUF_Q4_K_T16, LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_X8, LAYOUT_GGUF_Q5_K_QMICRO_T16, LAYOUT_GGUF_Q5_K_X8,
    LAYOUT_GGUF_Q6_K_T16, LAYOUT_GGUF_Q6_K_X8,
)
from hipengine.quant.gguf_q4_k import GGUF_Q8_1_BLOCK

# Existing concrete resident qualification rows, no inferred variant substrings.
RAW_SELECTED_CONSUMERS = (
    ("Q3_K", "moe_linear", "gguf_q3_k", "selected_gemv_decode_bf16_bf16_out"),
    ("Q4_K", "linear", "gguf_q4_k", "selected_gemv_bf16_bf16_out"),
    ("Q5_K", "linear", "gguf_q5_k", "selected_gemv_bf16_bf16_out"),
    ("Q6_K", "linear", "gguf_q6_k", "selected_gemv_bf16_bf16_out"),
    ("IQ2_XS", "moe_linear", "gguf_iq2_xs", "selected_gemv_decode_bf16_bf16_out"),
    ("IQ3_XXS", "moe_linear", "gguf_iq3_xxs", "selected_gemv_decode_bf16_bf16_out"),
    ("IQ4_XS", "moe_linear", "gguf_iq4_xs", "selected_gemv_decode_bf16_bf16_out"),
)
REPACKED_SELECTED_CONSUMERS = (
    (LAYOUT_GGUF_Q4_K_T16, "Q4_K", "moe_linear", "gguf_q4_k_t16_v1", "selected_t16_gemv_decode_bf16_bf16_out"),
    (LAYOUT_GGUF_Q4_K_QMICRO_T16, "Q4_K", "moe_linear", "gguf_q4_k_qmicro_t16_v1", "selected_dual_t16_gemv_decode_bf16_bf16_out"),
    (LAYOUT_GGUF_Q4_K_X8, "Q4_K", "moe_linear", "gguf_q4_k_x8_v1", "selected_dual_x8_q8_1_dp4a_gemv_decode_bf16_bf16_out"),
    (LAYOUT_GGUF_Q5_K_QMICRO_T16, "Q5_K", "moe_linear", "gguf_q5_k_qmicro_t16_v1", "selected_t16_gemv_decode_bf16_bf16_out"),
    (LAYOUT_GGUF_Q5_K_X8, "Q5_K", "moe_linear", "gguf_q5_k_x8_v1", "selected_x8_q8_1_dp4a_gemv_decode_bf16_bf16_out"),
    (LAYOUT_GGUF_Q6_K_T16, "Q6_K", "moe_linear", "gguf_q6_k_t16_v1", "selected_t16_gemv_decode_bf16_bf16_out"),
    (LAYOUT_GGUF_Q6_K_X8, "Q6_K", "moe_linear", "gguf_q6_k_x8_v1", "selected_x8_q8_1_dp4a_gemv_decode_bf16_bf16_out"),
)

SELECTED_VARIANTS = MappingProxyType({
    "single": "selected_gemv_decode_bf16_bf16_out",
    "dual_silu": "selected_dual_silu_gemv_decode_bf16_bf16_out",
    "weighted_down": "selected_weighted_down_gemv_decode_bf16_bf16_out",
})
# Exact generic entries exposed by the production registrars. Other resident
# families use the existing direct wrappers, including dual->two-single fallback.
EXACT_SELECTED_MODES = MappingProxyType({
    "gguf_q3_k": ("single", "dual_silu"),
    "gguf_iq2_xs": ("single", "dual_silu"),
    "gguf_iq3_xxs": ("single", "dual_silu", "weighted_down"),
    "gguf_iq4_xs": ("single", "weighted_down"),
    "gguf_q4_k_qmicro_t16_v1": ("dual_silu",),
    "gguf_q5_k_qmicro_t16_v1": ("single",),
})
_X8_SINGLE = frozenset({"gguf_q5_k_x8_v1", "gguf_q6_k_x8_v1"})
_X8_DUAL = "gguf_q4_k_x8_v1"
_PAIR_ONLY = frozenset({_X8_DUAL, "gguf_q4_k_qmicro_t16_v1"})
_ALLOCATION = MappingProxyType({
    **{row[2]: "raw" for row in RAW_SELECTED_CONSUMERS},
    **{row[3]: "tiles" for row in REPACKED_SELECTED_CONSUMERS},
    "gguf_q5_k_t16_v1": "tiles",
    # Raw Q5_1 expert weights (legacy qwen4_exp_q5_1 selected GEMV and the
    # compact pack8 down GEMV) stay in their GGUF row-major layout.
    "gguf_q5_1": "raw",
})
_ENTRY = MappingProxyType({
    "single": "_launch_selected_raw_gguf_moe_linear",
    "dual": "_launch_selected_raw_gguf_moe_dual",
    "dual_silu": "_launch_selected_raw_gguf_moe_pair_silu",
    "weighted_down": "_launch_weighted_selected_raw_gguf_moe_linear",
})


def selected_adapter_enabled(name: str) -> bool:
    """Existing selected-adapter env semantics, shared with production."""
    raw = os.environ.get(name, "").strip()
    return bool(raw) and raw.lower() not in {"0", "false", "off", "no"}


# The Q8_1 operand every selected q8_1 caller consumes lives in ONE plan-owned
# resident scratch slot. The scratch plan prices it for the widest selected
# call (``max`` of the gate/up and down block counts); each call rewrites it in
# place, so no selected caller allocates, resizes or frees it. ``owner`` names
# that slot, ``lifetime`` records that the plan owns it for the whole session.
SELECTED_Q8_1_WORKSPACE_OWNER = "moe_q8_1"
SELECTED_WORKSPACE_LIFETIME = "scratch_plan"
#: Workspace owners this contract can bind. The runtime consumes exactly these
#: slots, so a bound record can never name a workspace nothing executes.
SELECTED_WORKSPACE_OWNERS = frozenset({SELECTED_Q8_1_WORKSPACE_OWNER})


def selected_workspace_owner(owner: str | None) -> str:
    """Resolve one declared (or mandatory) Q8_1 workspace owner.

    ``None`` is the mandatory X8 case, which has no choice to make. A declared
    owner must be a slot the runtime actually reads; there is no owner mapping,
    so any other name is refused instead of being recorded.
    """
    resolved = SELECTED_Q8_1_WORKSPACE_OWNER if owner is None else owner
    if resolved not in SELECTED_WORKSPACE_OWNERS:
        raise ValueError(
            f"selected workspace owner {resolved!r} is not an executed Q8_1 "
            f"workspace ({', '.join(sorted(SELECTED_WORKSPACE_OWNERS))})")
    return resolved


# Optional selected DP4A invocation: convert the call's activation to Q8_1 in
# the caller-owned workspace, then run a dp4a GEMV whose resident allocation
# and arithmetic differ from the bf16-input consumer the same quant key serves
# by default. The env selectors below decide whether a call takes the route;
# this table says what taking it means, so the intent can name the conversion,
# the kernel and the workspace instead of borrowing a BF16 or X8 record.
SELECTED_Q8_1_DP4A = "q8_1_dp4a"


@dataclass(frozen=True)
class SelectedOptionalAdapter:
    """One optional selected-DP4A route over an otherwise bf16-input resident."""

    adapter: str
    kind: str
    quant: str
    allocation: str
    kernel: str
    flags: tuple[str, ...]
    out_features_multiple: int | None = None

    def engaged(self, shape: tuple[int, ...]) -> bool:
        """Whether the live selectors take this route at this geometry.

        Route availability is a resident/kernel fact; engagement is the
        selectors plus the geometry the kernel is registered for.
        """
        multiple = self.out_features_multiple
        if multiple is not None and int(shape[1]) % int(multiple):
            return False
        return any(selected_adapter_enabled(flag) for flag in self.flags)


SELECTED_OPTIONAL_ADAPTERS = (
    SelectedOptionalAdapter(
        SELECTED_Q8_1_DP4A, "single", "gguf_q5_k", "raw",
        "gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out",
        ("HIPENGINE_GGUF_RAW_SELECTED_DP4A",), 8),
    SelectedOptionalAdapter(
        SELECTED_Q8_1_DP4A, "single", "gguf_q6_k", "raw",
        "gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out",
        ("HIPENGINE_GGUF_RAW_SELECTED_DP4A",), 8),
    SelectedOptionalAdapter(
        SELECTED_Q8_1_DP4A, "single", "gguf_q5_k_t16_v1", "tiles",
        "gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out",
        ("HIPENGINE_GGUF_T16_SELECTED_DP4A",)),
    SelectedOptionalAdapter(
        SELECTED_Q8_1_DP4A, "dual", "gguf_q4_k", "raw",
        "gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out",
        ("HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A", "HIPENGINE_GGUF_RAW_SELECTED_DP4A")),
    SelectedOptionalAdapter(
        SELECTED_Q8_1_DP4A, "dual", "gguf_q4_k_t16_v1", "tiles",
        "gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out",
        ("HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A", "HIPENGINE_GGUF_T16_SELECTED_DP4A")),
)
SELECTED_OPTIONAL_ADAPTER_NAMES = frozenset(
    route.adapter for route in SELECTED_OPTIONAL_ADAPTERS)


def selected_optional_adapter(kind: str, quant: str) -> SelectedOptionalAdapter | None:
    """The optional adapter route this call kind/quant can take, if any."""
    for route in SELECTED_OPTIONAL_ADAPTERS:
        if route.kind == kind and route.quant == quant:
            return route
    return None


def selected_input_adapter(input_dtype: str) -> str:
    """Actual activation converter symbol used by the production caller."""
    return {"bf16": "gguf_q4_k_quantize_bf16_q8_1", "f32": "gguf_q4_k_quantize_f32_q8_1"}[input_dtype]


def selected_f32_output_supported(quant: str) -> bool:
    return quant in _X8_SINGLE


def selected_allocation(quant: str) -> str:
    try:
        return _ALLOCATION[quant]
    except KeyError as exc:
        raise ValueError(f"unsupported selected resident quant {quant!r}") from exc


@lru_cache(maxsize=128)
def selected_ffn_modes(gate_quant: str, up_quant: str, down_quant: str,
                       *, allow_legacy_silu: bool, f32_gate: bool = False,
                       f32_intermediate: bool = False, f32_down: bool = False,
                       weighted_down: bool = True) -> tuple[str, str]:
    """Production call topology; no load filter is an input to this decision."""
    fused = (gate_quant == up_quant and not f32_gate and not f32_intermediate
             and ("dual_silu" in EXACT_SELECTED_MODES.get(gate_quant, ())
                  or (allow_legacy_silu and gate_quant == "gguf_q4_k_t16_v1")))
    weighted = weighted_down and not f32_down and "weighted_down" in EXACT_SELECTED_MODES.get(down_quant, ())
    return ("dual_silu" if fused else "dual", "weighted_down" if weighted else "single")


@dataclass(frozen=True)
class SelectedCallIntent:
    operation: str
    kind: str
    weight_slots: tuple[str, ...]
    input_dtype: str = "bf16"
    output_dtype: str = "bf16"
    selected_owner: str = "selected_experts"
    routing_owner: str = "routing_weights"
    input_owner: str = "selected_input"
    output_owner: str = "selected_output"
    lanes_per_token: int = 1
    input_rows_per_token: int | None = None
    # Optional selected DP4A route this call may take, and the caller-owned
    # workspace its conversion writes. Declaring the route is not engaging it:
    # the live env selectors decide that at bind time, and the bound record
    # names the conversion, kernel and workspace actually executed.
    optional_adapter: str | None = None
    workspace_owner: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "weight_slots", tuple(self.weight_slots))
        count = 2 if self.kind in {"dual", "dual_silu"} else 1
        if self.kind not in _ENTRY or len(self.weight_slots) != count:
            raise ValueError("selected intent kind/ordered weight count mismatch")
        if not all(isinstance(slot, str) and slot for slot in self.weight_slots):
            raise ValueError("selected weight owners must be nonempty slot paths")
        if len(set(self.weight_slots)) != count:
            raise ValueError("selected intent requires distinct named weight owners")
        if self.input_dtype not in {"bf16", "f32"} or self.output_dtype not in {"bf16", "f32"}:
            raise ValueError("unsupported selected input/output dtype")
        lanes = int(self.lanes_per_token)
        if lanes != self.lanes_per_token or lanes < 1:
            raise ValueError("selected lanes per token must be a positive integer")
        object.__setattr__(self, "lanes_per_token", lanes)
        inputs = (lanes if self.kind == "weighted_down" else 1) if self.input_rows_per_token is None else self.input_rows_per_token
        if int(inputs) != inputs or inputs < 1 or lanes % inputs:
            raise ValueError("selected input rows must divide selected lanes per token")
        if self.kind == "weighted_down" and inputs != lanes:
            raise ValueError("weighted-down consumes one input row per selected expert lane")
        object.__setattr__(self, "input_rows_per_token", int(inputs))
        if not all(isinstance(owner, str) and owner for owner in (
            self.selected_owner, self.input_owner, self.output_owner, self.routing_owner,
        )):
            raise ValueError("selected intent operand owners must be named")
        if self.optional_adapter is not None and self.optional_adapter not in SELECTED_OPTIONAL_ADAPTER_NAMES:
            raise ValueError(f"unknown optional selected adapter {self.optional_adapter!r}")
        if self.workspace_owner is not None:
            if not isinstance(self.workspace_owner, str) or not self.workspace_owner:
                raise ValueError("selected workspace owner must be a nonempty name")
            if self.optional_adapter is None:
                raise ValueError("selected workspace owner requires an optional adapter intent")
            if self.workspace_owner in {
                self.selected_owner, self.routing_owner, self.input_owner,
                self.output_owner, *self.weight_slots,
            }:
                raise ValueError("selected q8_1 workspace must not alias an operand or weight owner")
            selected_workspace_owner(self.workspace_owner)


@dataclass(frozen=True)
class SelectedWorkspace:
    """The caller-owned Q8_1 workspace one bound selected call consumes.

    ``owner`` is the resident scratch slot holding it and ``lifetime`` records
    that the scratch plan owns it for the whole session. The geometry is the
    per-token pair the runtime sizes that slot with: the call's activation rows
    per token, and the resident's K.
    """

    owner: str
    lifetime: str
    rows_per_token: int
    in_features: int


@dataclass(frozen=True)
class SelectedABI:
    kind: str
    operands: tuple[tuple[str, str, str], ...]
    dimensions: tuple[str, ...]

    def call(self, fn, pointers: tuple[int, ...], dimensions: Mapping[str, int], *, positional_dimensions: bool = False, **kwargs):
        if len(pointers) != len(self.operands):
            raise ValueError("selected launch operand count mismatch")
        # Generic selected wrappers require keyword geometry. All legacy
        # selected wrappers accept the same named geometry parameters.
        if positional_dimensions:
            return fn(*pointers, *(dimensions[name] for name in self.dimensions), **kwargs)
        return fn(*pointers, **{name: dimensions[name] for name in self.dimensions}, **kwargs)


@lru_cache(maxsize=32)
def selected_abi(kind: str, *, input_dtype: str = "bf16", output_dtype: str = "bf16") -> SelectedABI:
    x = (("x_ptr", input_dtype, "read"), ("selected_ptr", "i64", "read"))
    weights = (("weight_a_ptr", "i8", "read"), ("weight_b_ptr", "i8", "read")) if kind in {"dual", "dual_silu"} else (("weight_ptr", "i8", "read"),)
    outputs = (("out_a_ptr", output_dtype, "write"), ("out_b_ptr", output_dtype, "write")) if kind == "dual" else (("out_ptr", output_dtype, "write"),)
    if kind == "weighted_down":
        x += (("routing_weights_ptr", "f32", "read"),)
    if kind not in _ENTRY:
        raise ValueError(f"unknown selected call kind {kind!r}")
    dims = ("tokens", "top_k") if kind == "weighted_down" else ("x_rows", "rows")
    return SelectedABI(kind, x + weights + outputs, dims + ("num_experts", "in_features", "out_features"))


@dataclass(frozen=True)
class BoundSelectedCall:
    intent: SelectedCallIntent
    backend: str
    weight_bindings: tuple[tuple[str, str], ...]
    shape: tuple[int, ...]
    operands: tuple[tuple[str, str, str], ...]
    entry: str
    allocations: tuple[str, ...]
    adapters: tuple[str, ...]
    row_limits: tuple[int, int | None]
    output_rows_per_token: int
    adapter_kernels: tuple[str, ...] = ()
    workspace: SelectedWorkspace | None = None

    def canonical_record(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def bind_selected_call(intent: SelectedCallIntent, specs: Mapping, records: Mapping[str, str],
                       *, backend: str, row_limits: tuple[int, int | None]) -> BoundSelectedCall:
    """Bind EVERY ordered partner; a filtered or refused owner is not optional.

    A Q8_1-consuming call also binds the conversion, kernel and plan-owned
    workspace it executes, so the record and the runtime agree on which
    arithmetic ran.
    """
    missing = tuple(slot for slot in intent.weight_slots if slot not in specs or slot not in records)
    if missing:
        raise ValueError(f"selected {intent.kind} requires resident partner(s): {', '.join(missing)}")
    weights = tuple(specs[slot] for slot in intent.weight_slots)
    shapes = tuple(tuple(int(d) for d in weight.source.shape) for weight in weights)
    if any(len(shape) != 3 for shape in shapes) or len(set(shapes)) != 1:
        raise ValueError("selected partners must have equal rank-3 (experts,N,K) geometry")
    quants = tuple(weight.quant_key for weight in weights)
    allocations = tuple(selected_allocation(quant) for quant in quants)
    if any(allocation not in weight.allocation_names for allocation, weight in zip(allocations, weights)):
        raise ValueError("selected resident lacks its actual consumed allocation")
    kind = intent.kind
    same = len(set(quants)) == 1
    if kind == "single" and quants[0] in _PAIR_ONLY:
        raise ValueError("selected resident has no singleton consumer")
    if kind == "dual_silu" and not (same and ("dual_silu" in EXACT_SELECTED_MODES.get(quants[0], ()) or quants[0] == "gguf_q4_k_t16_v1")):
        raise ValueError("selected partners have no dual-SiLU consumer")
    if kind == "weighted_down" and "weighted_down" not in EXACT_SELECTED_MODES.get(quants[0], ()):
        raise ValueError("selected resident has no weighted-down consumer")
    if kind == "dual" and any(q in _X8_SINGLE for q in quants):
        # The existing two-single fallback does not pass a Q8_1 workspace.
        raise ValueError("dual X8 single-format fallback lacks an executed input adapter")
    if kind == "dual" and any(q in _PAIR_ONLY for q in quants) and not (same and quants[0] == _X8_DUAL):
        raise ValueError("selected partners have no dual or two-single consumer")
    q8 = (kind == "single" and quants[0] in _X8_SINGLE) or (kind == "dual" and same and quants[0] == _X8_DUAL)
    if intent.lanes_per_token > shapes[0][0]:
        raise ValueError("selected lanes per token exceed the expert inventory")
    # The optional DP4A routes are a different conversion contract over the same
    # residents: the call converts its activation to Q8_1 in the plan-owned
    # workspace and runs a dp4a kernel. A route needs one shared resident layout
    # (ordered partners are read by one kernel) and the intent must name the
    # conversion and the workspace, otherwise the record would describe a
    # different arithmetic than the runtime takes. Selectors still decide
    # whether the route runs; this only makes that run representable.
    route = selected_optional_adapter(kind, quants[0]) if same else None
    if route is not None and route.allocation != allocations[0]:
        raise ValueError("optional selected adapter route disagrees with the resident allocation")
    if intent.optional_adapter is not None:
        if route is None or intent.optional_adapter != route.adapter:
            raise ValueError("declared optional selected adapter has no route for this selected call")
        if not intent.workspace_owner:
            raise ValueError("optional selected DP4A adapter requires a named q8_1 workspace owner")
    adapter = route if route is not None and route.engaged(shapes[0]) else None
    if adapter is not None and intent.optional_adapter != adapter.adapter:
        raise ValueError("optional selected DP4A adapter requires a separately qualified invocation intent")
    q8_1 = bool(q8 or adapter)
    if intent.input_dtype == "f32" and not q8_1:
        raise ValueError("supplied F32 selected input requires an executed Q8_1 adapter")
    if intent.output_dtype == "f32" and not (kind == "single" and quants[0] in _X8_SINGLE):
        raise ValueError("selected consumer has no F32 output")
    abi = selected_abi(kind, input_dtype=intent.input_dtype, output_dtype=intent.output_dtype)
    operands = abi.operands
    workspace = None
    if q8_1:
        if intent.input_dtype == "f32":
            operands = (("x_f32_ptr", "f32", "read"),) + operands[1:]
        operands += (("q8_1_workspace_ptr", "q8_1", "read_write"),)
        workspace = SelectedWorkspace(
            selected_workspace_owner(intent.workspace_owner if adapter is not None else None),
            SELECTED_WORKSPACE_LIFETIME, intent.input_rows_per_token, shapes[0][2])
        if workspace.in_features % GGUF_Q8_1_BLOCK:
            raise ValueError(
                f"q8_1 workspace requires in_features divisible by {GGUF_Q8_1_BLOCK}")
    return BoundSelectedCall(intent, backend, tuple((slot, records[slot]) for slot in intent.weight_slots),
        shapes[0], operands, "hipengine.runtime.qwen35_gguf_runner:" + _ENTRY[kind],
        allocations, ((selected_input_adapter(intent.input_dtype),) if q8_1 else ()), row_limits,
        1 if kind == "weighted_down" else intent.lanes_per_token,
        (adapter.kernel,) if adapter is not None else (), workspace)


def _optional_adapter_declaration(kind: str, quants: tuple[str, ...]) -> dict[str, str]:
    """The shared plan declares the optional route this call can take.

    Whether the route is taken is the live selector at bind time, not this
    declaration: one plan binds both arithmetic contracts, and the bound record
    says which one ran. A pair with unequal partners has no single-layout route,
    so it declares nothing.
    """
    if len(set(quants)) != 1:
        return {}
    route = selected_optional_adapter(kind, quants[0])
    if route is None:
        return {}
    return {"optional_adapter": route.adapter,
            "workspace_owner": SELECTED_Q8_1_WORKSPACE_OWNER}


def default_selected_call_intents(slot_quants: Mapping[str, str], operations, *, lanes_per_token: int = 1) -> tuple[SelectedCallIntent, ...]:
    """The full-model caller plan; never reinterpret a paired call by filtering.

    The logical dual boundary includes the existing ordered two-single
    fallback. Fused SiLU and weighted-down remain distinct interfaces.
    """
    calls = []
    for gate in sorted(slot for slot in slot_quants if slot.endswith(".ffn_gate_exps")):
        prefix = gate.rsplit(".", 1)[0]
        up, down = prefix + ".ffn_up_exps", prefix + ".ffn_down_exps"
        for op in operations:
            if op not in {"ar_decode_c1", "ar_decode_rows", "ar_prefill", "ar_decode_native_rows"}:
                continue
            gate_kind, down_kind = selected_ffn_modes(slot_quants[gate], slot_quants.get(up, ""), slot_quants.get(down, ""), allow_legacy_silu=op == "ar_decode_c1")
            calls.extend((
                SelectedCallIntent(op, gate_kind, (gate, up), lanes_per_token=lanes_per_token,
                    input_owner="post_norm", selected_owner="moe_selected_experts",
                    output_owner="ffn_intermediate" if gate_kind == "dual_silu" else "ffn_gate_up",
                    **_optional_adapter_declaration(
                        gate_kind, (slot_quants[gate], slot_quants.get(up, "")))),
                SelectedCallIntent(op, down_kind, (down,), lanes_per_token=lanes_per_token,
                    input_rows_per_token=lanes_per_token, input_owner="ffn_intermediate", selected_owner="moe_selected_experts",
                    routing_owner="moe_routing_weights", output_owner="moe_down_out",
                    **_optional_adapter_declaration(
                        down_kind, (slot_quants.get(down, ""),))),
            ))
    return tuple(calls)
