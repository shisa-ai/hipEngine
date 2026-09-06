"""CPU-only selected-expert caller contracts, shared by admission and runtime.

Call kind is an invocation interface, not a registry axis. Singleton, dual,
dual-SiLU and weighted-down have different operands/output ownership. The
runtime's existing primitive selection/variant policy stays below these
interfaces. Certificates are not numerical/profile qualification.
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

# Existing concrete resident qualification rows, no inferred variant substrings.
RAW_SELECTED_CONSUMERS = (
    ("Q3_K", "moe_linear", "gguf_q3_k", "selected_gemv_decode_bf16_bf16_out", None, None),
    ("Q4_K", "linear", "gguf_q4_k", "selected_gemv_bf16_bf16_out",
     "hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv", "gguf_q4_k_selected_gemv_bf16_bf16_out"),
    ("Q5_K", "linear", "gguf_q5_k", "selected_gemv_bf16_bf16_out", None, None),
    ("Q6_K", "linear", "gguf_q6_k", "selected_gemv_bf16_bf16_out", None, None),
    ("IQ2_XS", "moe_linear", "gguf_iq2_xs", "selected_gemv_decode_bf16_bf16_out", None, None),
    ("IQ3_XXS", "moe_linear", "gguf_iq3_xxs", "selected_gemv_decode_bf16_bf16_out", None, None),
    ("IQ4_XS", "moe_linear", "gguf_iq4_xs", "selected_gemv_decode_bf16_bf16_out", None, None),
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

    def canonical_record(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def bind_selected_call(intent: SelectedCallIntent, specs: Mapping, records: Mapping[str, str],
                       *, backend: str, row_limits: tuple[int, int | None]) -> BoundSelectedCall:
    """Bind EVERY ordered partner; a filtered or refused owner is not optional."""
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
    # Optional legacy DP4A knobs select a different conversion contract. The
    # default qualified selected interface does not authorize those experiments;
    # mandatory X8 adapters are explicit below. Runtime choices are unchanged.
    raw_dp4a = selected_adapter_enabled("HIPENGINE_GGUF_RAW_SELECTED_DP4A")
    t16_dp4a = selected_adapter_enabled("HIPENGINE_GGUF_T16_SELECTED_DP4A")
    q4_dp4a = selected_adapter_enabled("HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A")
    optional_adapter = (
        kind == "single" and ((quants[0] in {"gguf_q5_k", "gguf_q6_k"} and shapes[0][1] % 8 == 0 and raw_dp4a)
                              or (quants[0] == "gguf_q5_k_t16_v1" and t16_dp4a))
        or kind == "dual" and same and ((quants[0] == "gguf_q4_k" and (raw_dp4a or q4_dp4a))
                                       or (quants[0] == "gguf_q4_k_t16_v1" and (t16_dp4a or q4_dp4a)))
    )
    if optional_adapter:
        raise ValueError("optional selected DP4A adapter requires a separately qualified invocation intent")
    if intent.input_dtype == "f32" and not q8:
        raise ValueError("supplied F32 selected input requires an executed Q8_1 adapter")
    if intent.output_dtype == "f32" and not (kind == "single" and quants[0] in _X8_SINGLE):
        raise ValueError("selected consumer has no F32 output")
    abi = selected_abi(kind, input_dtype=intent.input_dtype, output_dtype=intent.output_dtype)
    operands = abi.operands
    if q8:
        if intent.input_dtype == "f32":
            operands = (("x_f32_ptr", "f32", "read"),) + operands[1:]
        operands += (("q8_1_workspace_ptr", "q8_1", "read_write"),)
    return BoundSelectedCall(intent, backend, tuple((slot, records[slot]) for slot in intent.weight_slots),
        shapes[0], operands, "hipengine.runtime.qwen35_gguf_runner:" + _ENTRY[kind],
        allocations, ((selected_input_adapter(intent.input_dtype),) if q8 else ()), row_limits,
        1 if kind == "weighted_down" else intent.lanes_per_token)


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
            if op not in {"ar_decode_c1", "ar_decode_rows", "ar_prefill"}:
                continue
            gate_kind, down_kind = selected_ffn_modes(slot_quants[gate], slot_quants.get(up, ""), slot_quants.get(down, ""), allow_legacy_silu=op == "ar_decode_c1")
            calls.extend((
                SelectedCallIntent(op, gate_kind, (gate, up), lanes_per_token=lanes_per_token,
                    input_owner="post_norm", selected_owner="moe_selected_experts",
                    output_owner="ffn_intermediate" if gate_kind == "dual_silu" else "ffn_gate_up"),
                SelectedCallIntent(op, down_kind, (down,), lanes_per_token=lanes_per_token,
                    input_rows_per_token=lanes_per_token, input_owner="ffn_intermediate", selected_owner="moe_selected_experts",
                    routing_owner="moe_routing_weights", output_owner="moe_down_out"),
            ))
    return tuple(calls)
