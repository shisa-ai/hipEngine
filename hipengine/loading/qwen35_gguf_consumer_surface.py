"""Shared CPU-safe GGUF invocation metadata (UD-U1 F3/F4).

Owns the base linear rows, row rewrites and weight-operand order used by
runtime/gguf_linear, embedding resolution, and the direct RMSNorm/conv/GDN
wrapper ABIs. Admission qualifies role participation separately and binds
these concrete contracts to resident bytes. No backend package import,
HIP library load, device query, allocation, or numerical certification.

Availability declarations are read from backend source literals. Availability
is not profile authorization. These descriptors name wrapper boundaries;
internal optimized/profile variant selection and runtime-entry certificate
consumption are not implemented here. Selected-expert direct-wrapper keys
remain an explicitly bounded interface, not a claim of registry mediation.

An F32-input declaration means an actually supplied F32 activation. There is
no automatic BF16 input fallback. Adapters name real caller conversions,
such as the verifier projection's F32-output to BF16 cast, rather than a
permission to reinterpret pointers. Native head scratch.norm remains BF16.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
import ctypes
import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from types import MappingProxyType

from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
)

__all__ = [
    "GGUF_ACTIVATION_BF16",
    "GGUF_ACTIVATION_F32",
    "GGUF_CONSUMER_LAYER_DECLARATION_NAME",
    "GGUF_LINEAR_DISPATCH_SURFACE",
    "GGUF_OUTPUT_BF16",
    "GGUF_OUTPUT_F32",
    "GGUF_OUTPUT_FP16",
    "GgufLinearDispatchSurfaceRow",
    "RAW_LINEAR_SOURCE_QUANT_KEYS",
    "backend_gguf_consumer_layers",
    "gguf_linear_dispatch_row",
    "read_backend_gguf_consumer_layers",
    "source_linear_dispatch_row",
]


GGUF_ACTIVATION_BF16 = "bf16"
GGUF_ACTIVATION_F32 = "f32"
GGUF_OUTPUT_BF16 = "bf16"
GGUF_OUTPUT_FP16 = "fp16"
GGUF_OUTPUT_F32 = "f32"

# Quant token the production table uses for per-quant raw-layout rows: the
# concrete quant key comes from the resident weight's spec.
FROM_WEIGHT_QUANT_TOKEN = "<from-weight>"

# Source types whose rank-2 linear slots plan to the raw layout, with the
# concrete per-type quant key the dispatcher resolves from the weight
# (planner truth: rank-2 Q6_K/Q8_0 stay raw; rank-2 Q4_K goes pack8 and
# rank-2 Q5_K expands dense-BF16, so they are deliberately absent).
RAW_LINEAR_SOURCE_QUANT_KEYS: Mapping[str, str] = MappingProxyType({
    "Q6_K": "gguf_q6_k",
    "Q8_0": "gguf_q8_0",
})


@dataclass(frozen=True)
class GgufLinearDispatchSurfaceRow:
    """One concrete ``(layout, activation, output)`` linear consumer row.

    ``variant`` is the rows=1 form exactly as the production dispatch table
    records it; ``variant_for_rows(rows)`` reproduces the production
    ``_variant_for_rows`` rewrite (including the Q4-T16 multirow rewrite the
    dispatcher applies in ``resolve_gguf_linear_dispatch``).
    """

    layout: str
    activation: str
    output: str
    layer: str
    quant: str
    variant: str
    abi: str
    pointer_activation: str | None = None

    def variant_for_rows(self, rows: int) -> str:
        if int(rows) <= 0:
            raise ValueError("rows must be positive")
        if int(rows) == 1:
            return self.variant
        if self.layout in (LAYOUT_GGUF_Q4_K_T16, LAYOUT_GGUF_Q4_K_QMICRO_T16):
            # resolve_gguf_linear_dispatch rewrites Q4 T16 multirow launches
            # to the WMMA prefill leaf regardless of the table variant.
            return "t16_wmma_prefill_bf16_bf16_out"
        return linear_variant_for_rows(self.variant, rows=rows)


def linear_variant_for_rows(variant: str, *, rows: int) -> str:
    """Shared base linear row rewrite (not a profile/optimization selector)."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows == 1:
        return variant
    if variant.startswith("pack8_"):
        return f"pack8_prefill_{variant[len('pack8_') :]}"
    if variant.startswith("gemv_"):
        return f"prefill_{variant[len('gemv_') :]}"
    return "prefill_out" if variant == "out" else variant


@dataclass(frozen=True)
class ConsumerContract:
    """Pure concrete boundary metadata, NOT numerical/profile certification.

    Operands are ordered (wrapper argument name, element dtype, access).
    Scalars include row/geometry arguments, not just device pointers. Direct
    ctypes wrappers use ``launch`` so pointer order and scalar ABI have one
    owner. Pointers cannot prove buffer contents; caller binding is required.
    """

    layer: str
    quant: str
    variant: str
    abi: str
    operands: tuple[tuple[str, str, str], ...]
    scalars: tuple[tuple[str, str], ...] = ()
    variant_rows_many: str | None = None
    module: str | None = None
    symbol: str | None = None
    boundary: str | None = None

    def key(self, backend: str):
        from hipengine.kernels.registry import KernelKey
        return KernelKey(backend, self.layer, self.quant, self.variant)

    def call(self, fn, values: Mapping[str, object], **kwargs):
        """Invoke a Python launch wrapper using this owner's argument order."""
        names = tuple(name for name, _, _ in self.operands) + tuple(name for name, dtype in self.scalars if dtype != "stream")
        return fn(*(values[name] for name in names), **kwargs)

    def launch(self, library, values: Mapping[str, object]) -> int:
        """Marshal an existing direct C wrapper; no allocation or conversion."""
        types = {"i64": ctypes.c_int64, "f32": ctypes.c_float, "stream": ctypes.c_void_p}
        signature = tuple((name, ctypes.c_void_p) for name, _, _ in self.operands)
        signature += tuple((name, types[dtype]) for name, dtype in self.scalars)
        fn = getattr(library, self.symbol)
        fn.argtypes = [ctype for _, ctype in signature]
        fn.restype = ctypes.c_int
        return fn(*(ctype(values[name]) for name, ctype in signature))


@dataclass(frozen=True)
class InvocationContract:
    """Canonical slot/operation intent bound alongside actual resident bytes.

    ``adapters`` names conversions actually executed at the declared caller
    boundary, never a permission to reinterpret a supplied pointer. Shape
    includes model geometry for composite consumers; rows_scope is the
    admitted row-mode domain, not a claim about a concrete graph bucket.
    """

    slot: str
    operation: str
    backend: str
    shape: tuple[int, ...]
    rows_scope: str
    consumer: ConsumerContract
    adapters: tuple[str, ...] = ()
    parameters: tuple[tuple[str, int | float | str], ...] = ()
    row_limits: tuple[int, int | None] = (1, None)

    def canonical_record(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


RMSNORM = ConsumerContract(
    "rmsnorm", "gguf_f32_weight", "bf16_out", "rmsnorm",
    (("x_ptr", "bf16", "read"), ("weight_ptr", "f32", "read"), ("out_ptr", "bf16", "write")),
    (("rows", "i64"), ("hidden_size", "i64"), ("eps", "f32"), ("threads", "i64"), ("stream", "stream")),
    symbol="hipengine_gguf_rmsnorm_bf16_f32_weight",
)

# Concrete direct wrapper ABIs: conv.hip:954/1352, gdn.hip:7801.
# The segmented GDN consumer produces F32; native rows cast afterward.
CONV_DECODE = ConsumerContract(
    "linear_attn_conv_decode", "gguf_qwen35", "bf16_indexed", "indexed_conv",
    (("hidden_states_ptr", "bf16", "read"), ("conv_state_ptr", "f32", "read_write"),
     ("conv_weight_ptr", "f32", "read"), ("out_ptr", "f32", "write"),
     ("state_indices_ptr", "i64", "read")),
    (("rows", "i64"), ("channels", "i64"), ("kernel_size", "i64"), ("stream", "stream")),
    symbol="hipengine_qwen35_linear_attn_conv_decode_indexed_bf16",
)
CONV_PREFILL = ConsumerContract(
    "linear_attn_conv_prefill", "gguf_qwen35", "f32_baseline", "prefill_conv",
    (("hidden_states_ptr", "f32", "read"),) + CONV_DECODE.operands[1:4],
    (("tokens", "i64"), ("channels", "i64"), ("kernel_size", "i64"), ("stream", "stream")),
    symbol="hipengine_qwen35_linear_attn_conv_prefill_f32",
)
GDN_SEGMENTS = ConsumerContract(
    "gdn_recurrent_rmsnorm_gate", "gguf_qwen35", "bf16_segments", "segmented_gdn",
    (("conv_out_ptr", "f32", "read"), ("gate_ptr", "bf16", "read"),
     ("a_ptr", "bf16", "read"), ("b_ptr", "bf16", "read"),
     ("dt_bias_ptr", "f32", "read"), ("a_log_ptr", "f32", "read"),
     ("norm_weight_ptr", "f32", "read"), ("recurrent_state_ptr", "f32", "read_write"),
     ("out_ptr", "f32", "write"), ("cu_seqlens_ptr", "i32", "read"),
     ("state_indices_ptr", "i64", "read")),
    (("total_tokens", "i64"), ("segments", "i64"), ("eps", "f32"),
     ("num_k_heads", "i64"), ("num_v_heads", "i64"), ("head_k_dim", "i64"),
     ("head_v_dim", "i64"), ("stream", "stream")),
    symbol="hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16",
)


CONV_SINGLE = replace(CONV_DECODE, variant="bf16", abi="single_conv",
    operands=CONV_DECODE.operands[:4], scalars=CONV_DECODE.scalars[1:],
    # The raw c1 wrapper is registered on the w4_paro compatibility key.
    quant="w4_paro", symbol="hipengine_qwen35_linear_attn_conv_decode_bf16")
GDN_SINGLE = replace(GDN_SEGMENTS, quant="w4_paro", variant="bf16_lowp", abi="single_gdn",
    operands=GDN_SEGMENTS.operands[:9], scalars=GDN_SEGMENTS.scalars[2:],
    symbol="hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16")
GDN_PREFILL = replace(GDN_SINGLE, layer="gdn_prefill_recurrent", quant="gguf_qwen35", variant="decode_order_bf16",
    abi="prefill_gdn", operands=GDN_SINGLE.operands[:-1] + (("out_ptr", "bf16", "write"),),
    scalars=(("eps", "f32"), ("tokens", "i64")) + GDN_SINGLE.scalars[1:],
    symbol="hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order",
    boundary="hipengine.runtime.qwen35_gguf_runner:Qwen35GGUFFullStackRunner._run_gdn_prefill")


@lru_cache(maxsize=32)
def resolve_gdn_operation_contract(operation: str, state_dtype: str = "f32") -> ConsumerContract:
    if operation not in {"ar_decode_c1", "ar_decode_rows", "ar_prefill", "ar_decode_native_rows"}:
        raise ValueError(f"unsupported GDN operation {operation!r}")
    if operation == "ar_decode_native_rows":
        return resolve_gdn_segments_contract(state_dtype)
    if operation == "ar_prefill":
        # The prefill caller owns fused/strict-chain selection and always
        # publishes BF16 recurrent output. State storage is an explicit intent.
        if state_dtype != "f32":
            raise ValueError("baseline prefill GDN requires F32 state; FP16 needs its explicit prefill plan")
        return GDN_PREFILL
    if state_dtype == "f32":
        return GDN_SINGLE
    resolve_gdn_segments_contract(state_dtype)
    return replace(GDN_SINGLE, quant="gguf_qwen35", variant="bf16_fp16state",
                   operands=tuple((name, "fp16" if name == "recurrent_state_ptr" else dtype, access)
                                  for name, dtype, access in GDN_SINGLE.operands),
                   symbol=GDN_SINGLE.symbol + "_fp16state")


@lru_cache(maxsize=16)
def conv_operation_contract(operation: str) -> ConsumerContract:
    if operation not in {"ar_decode_c1", "ar_decode_rows", "ar_prefill", "ar_decode_native_rows"}:
        raise ValueError(f"unsupported convolution operation {operation!r}")
    if operation == "ar_prefill":
        return CONV_PREFILL
    return CONV_DECODE if operation == "ar_decode_native_rows" else CONV_SINGLE


@lru_cache(maxsize=4)
def resolve_gdn_segments_contract(state_dtype: str = "f32") -> ConsumerContract:
    """Existing mixed GDN ABI, including the actual recurrent-state storage."""
    if state_dtype == "f32":
        return GDN_SEGMENTS
    if state_dtype != "fp16":
        raise ValueError(f"unsupported recurrent state dtype {state_dtype!r}")
    return replace(GDN_SEGMENTS, variant="bf16_segments_fp16state",
                   operands=tuple((name, "fp16" if name == "recurrent_state_ptr" else dtype, access)
                                  for name, dtype, access in GDN_SEGMENTS.operands),
                   symbol=GDN_SEGMENTS.symbol + "_fp16state")


@lru_cache(maxsize=1)
def native_alpha_beta_consumer_contract() -> ConsumerContract:
    """Native rows call the dense BF16 'out' wrapper directly, not prefill."""
    row = gguf_linear_dispatch_row(LAYOUT_DENSE_BF16, "bf16", "bf16")
    return replace(linear_consumer_contract(row), variant_rows_many=None,
                   module="hipengine.kernels.hip_gfx1100.linear.dense_gemv",
                   symbol="dense_gemv_out_bf16",
                   boundary="hipengine.kernels.hip_gfx1100.linear.dense_gemv:dense_gemv_out_bf16")


@dataclass(frozen=True)
class GDNOutputHandoff:
    activation: str
    adapters: tuple[str, ...]


@lru_cache(maxsize=32)
def resolve_gdn_output_handoff(layout: str, *, force_bf16: bool = False) -> GDNOutputHandoff:
    """Actual c1/row-local GDN->ssm_out conversion decision.

    Native rows always execute their explicit cast; prefill already produces
    BF16. This handoff is for the row-local F32 recurrent output caller only.
    """
    if not force_bf16 and gguf_linear_dispatch_row(layout, "f32", "bf16") is not None:
        return GDNOutputHandoff("f32", ())
    return GDNOutputHandoff("bf16", ("input:f32_to_bf16",))


# The runtime launch adapters consume this same operand order. Dense weight
# element dtype is resolved from the resident layout, not the adapter's legacy
# name ('dense_bf16' also launches dense-F32 weights).
LINEAR_WEIGHT_OPERANDS = MappingProxyType({
    "pack8": (("qweight", "i32", "read"), ("scales", "f32", "read"), ("mins", "f32", "read")),
    "raw": (("raw", "i8", "read"),),
    "t16": (("tiles", "i8", "read"),),
    "dense_bf16": (("raw", "resident_dense", "read"),),
})


OPERATION_ROW_LIMITS = MappingProxyType({
    "ar_decode_c1": (1, 1), "ar_decode_rows": (1, 8),
    "ar_decode_native_rows": (2, 8), "ar_prefill": (1, None),
    "lm_head_f32_logits": (1, 8), "embedding_lookup": (1, None),
})


def linear_weight_pointers(abi: str, weight) -> tuple[int, ...]:
    return tuple(weight.allocation(name).tensor.ptr for name, _, _ in LINEAR_WEIGHT_OPERANDS[abi])


@lru_cache(maxsize=64)
def linear_consumer_contract(row: GgufLinearDispatchSurfaceRow) -> ConsumerContract:
    weights = tuple((name, ({LAYOUT_DENSE_BF16: "bf16", LAYOUT_DENSE_F32: "f32"}[row.layout]
                            if dtype == "resident_dense" else dtype), access)
                    for name, dtype, access in LINEAR_WEIGHT_OPERANDS[row.abi])
    many = row.variant_for_rows(2)
    return ConsumerContract(
        row.layer, row.quant, row.variant, row.abi,
        (("x_ptr", row.pointer_activation or row.activation, "read"),) + weights + (("out_ptr", row.output, "write"),),
        (("rows", "i64"), ("in_features", "i64"), ("out_features", "i64")),
        variant_rows_many=None if many == row.variant else many,
        boundary="hipengine.runtime.gguf_linear:launch_gguf_linear",
    )


def resolve_linear_consumer_contract(
    layout: str, activation: str, output: str, *, quant_key: str, rows: int = 1,
) -> ConsumerContract:
    row = gguf_linear_dispatch_row(layout, activation, output)
    if row is None:
        raise ValueError("unsupported GGUF linear dispatch: "
                         f"layout={layout!r}, activation={activation!r}, output={output!r}")
    contract = linear_consumer_contract(row)
    return replace(contract, variant=row.variant_for_rows(rows),
                   quant=quant_key if row.quant == FROM_WEIGHT_QUANT_TOKEN else row.quant)


RAW_EMBEDDING_QUANTS = frozenset({"gguf_q4_k", "gguf_q5_k", "gguf_q6_k", "gguf_q8_0"})


def resolve_embedding_consumer_contract(layout: str, quant: str, output: str = "bf16") -> ConsumerContract:
    if output != "bf16":
        raise ValueError(f"unsupported GGUF embedding output dtype {output!r}")
    if layout == LAYOUT_RAW_GGUF and quant in RAW_EMBEDDING_QUANTS:
        contract = auxiliary_consumer_contract("token_embedding", layout=layout,
            layer="embedding", quant=quant, variant="lookup_bf16_out")
        return replace(contract, abi="raw")
    if layout == LAYOUT_DENSE_BF16:
        # Existing direct wrapper consumes only one token: admission must not
        # certify rows-any on this contract. Preserve runtime behavior.
        return ConsumerContract("embedding", "bf16", "lookup_bf16_out", "dense_bf16",
            (("raw", "bf16", "read"), ("token_ids_ptr", "i64", "read"), ("out_ptr", "bf16", "write")),
            (("hidden_size", "i64"), ("vocab_size", "i64")))
    raise ValueError("unsupported GGUF embedding dispatch: "
                     f"layout={layout!r}, quant={quant!r}, output={output!r}")


@lru_cache(maxsize=16)
def resolve_router_consumer_contract(layout: str, quant: str, activation: str = "bf16") -> ConsumerContract:
    """Actual router-logits caller boundary; no input conversion is implied.

    These are the three concrete registry ABIs called by the BF16/F32-hidden
    router adapters. F32 hidden plus BF16 weights has no registered consumer.
    """
    dtypes = {LAYOUT_DENSE_F32: "f32", LAYOUT_DENSE_BF16: "bf16"}
    weight_dtype = dtypes.get(layout)
    if weight_dtype != quant or (activation, weight_dtype) not in {
        ("bf16", "bf16"), ("bf16", "f32"), ("f32", "f32"),
    }:
        raise ValueError(f"unsupported router invocation: {layout=}, {quant=}, {activation=}")
    return ConsumerContract(
        "router_logits", quant, f"{activation}_hidden", "router_logits",
        (("hidden_ptr", activation, "read"), ("weight_ptr", weight_dtype, "read"), ("logits_ptr", "f32", "write")),
        (("tokens", "i64"), ("hidden_size", "i64"), ("num_rows", "i64")),
    )


def auxiliary_consumer_contract(
    role: str, *, layout: str, layer: str, quant: str, variant: str,
    module: str | None = None, symbol: str | None = None,
) -> ConsumerContract:
    """Describe the named non-linear boundary, not an entire runner pipeline.

    Qualification remains an admission concern. In particular this function
    cannot establish that an optimized/profile replacement is authorized.
    Selected calls are resolved separately by gguf_selected_contract, never
    inferred here from a primitive variant name.
    """
    if role == "norm":
        return RMSNORM
    elif role == "moe_router":
        return resolve_router_consumer_contract(layout, quant)
    elif role == "token_embedding":
        operands = (("token_ids_ptr", "i64", "read"), ("raw", "i8", "read"), ("out_ptr", "bf16", "write"))
        scalars = (("rows", "i64"), ("hidden_size", "i64"), ("vocab_size", "i64"))
    elif role == "moe_experts":
        raise ValueError("selected experts require an explicit SelectedCallIntent")
    else:
        raise ValueError(f"no invocation boundary for {role!r}")
    return ConsumerContract(layer, quant, variant, role, operands, scalars, module=module, symbol=symbol)


def _row(
    layout: str,
    activation: str,
    output: str,
    layer: str,
    quant: str,
    variant: str,
    abi: str,
    *, pointer_activation: str | None = None,
) -> GgufLinearDispatchSurfaceRow:
    return GgufLinearDispatchSurfaceRow(
        layout=layout,
        activation=activation,
        output=output,
        layer=layer,
        quant=quant,
        variant=variant,
        abi=abi,
        pointer_activation=pointer_activation,
    )


# Production owner. runtime/gguf_linear._DISPATCH_TABLE is a compatibility
# view generated from these rows; runtime and admission resolve here.
GGUF_LINEAR_DISPATCH_SURFACE: tuple[GgufLinearDispatchSurfaceRow, ...] = (
    _row(LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "linear", "gguf_q4_k", "pack8_bf16_bf16_out", "pack8"),
    _row(LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16, "linear", "gguf_q4_k", "pack8_bf16_fp16_out", "pack8"),
    _row(LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32, "linear", "gguf_q4_k", "pack8_bf16_f32_out", "pack8"),
    _row(LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "linear", FROM_WEIGHT_QUANT_TOKEN, "gemv_bf16_bf16_out", "raw"),
    _row(LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16, "linear", FROM_WEIGHT_QUANT_TOKEN, "gemv_bf16_fp16_out", "raw"),
    _row(LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32, "linear", FROM_WEIGHT_QUANT_TOKEN, "gemv_bf16_f32_out", "raw"),
    _row(LAYOUT_DENSE_BF16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "dense_gemv", "bf16", "out", "dense_bf16"),
    _row(LAYOUT_DENSE_BF16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32, "dense_gemv", "bf16", "f32_out", "dense_bf16"),
    _row(LAYOUT_DENSE_F32, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "dense_gemv", "f32", "bf16_hidden_bf16_out", "dense_bf16"),
    _row(LAYOUT_DENSE_F32, GGUF_ACTIVATION_F32, GGUF_OUTPUT_F32, "dense_gemv", "f32", "f32_hidden_f32_out", "dense_bf16"),
    _row(LAYOUT_GGUF_Q4_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "linear", "gguf_q4_k_t16_v1", "dense_single_local32_bf16_bf16_out", "t16"),
    _row(LAYOUT_GGUF_Q4_K_QMICRO_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "linear", "gguf_q4_k_qmicro_t16_v1", "dense_single_local32_bf16_bf16_out", "t16"),
    _row(LAYOUT_GGUF_Q5_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "linear", "gguf_q5_k_t16_v1", "t16_gemv_decode_bf16_bf16_out", "t16"),
    _row(LAYOUT_GGUF_Q6_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_bf16_out", "t16"),
    _row(LAYOUT_GGUF_Q6_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32, "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_f32_out", "t16"),
    _row(LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "linear", "gguf_q6_k_t16_qmicro_planar_v1", "t16_gemv_decode_bf16_bf16_out", "t16"),
    _row(LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32, "linear", "gguf_q6_k_t16_qmicro_planar_v1", "t16_gemv_decode_bf16_f32_out", "t16"),
    _row(LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16, "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out", "t16"),
    # Historical selector token, but the actual C ABI is half_t input/output.
    # No BF16->FP16 conversion is performed here; admission must not certify
    # a supplied BF16 pointer against this row.
    _row(LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16, "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_fp16_fp16_out", "t16", pointer_activation="fp16"),
    _row(LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_F32, GGUF_OUTPUT_BF16, "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_f32_bf16_out", "t16"),
)


def gguf_linear_dispatch_row(
    layout: str,
    activation: str,
    output: str,
) -> GgufLinearDispatchSurfaceRow | None:
    """The production dispatch row for one concrete dtype contract, or ``None``.

    ``None`` is the F3 fail-closed answer: the layout has no registered
    consumer for this activation/output combination (for example
    ``(dense_f32, bf16, f32)`` — the F32-logits caller dtype for a dense-F32
    resident), so admission must refuse the (slot, operation) instead of
    certifying a convenient nearby row.
    """

    for row in GGUF_LINEAR_DISPATCH_SURFACE:
        if row.layout == layout and row.activation == activation and row.output == output:
            return row
    return None


def source_linear_dispatch_row(
    source_ggml_type: str,
    layout: str,
    activation: str,
    output: str,
) -> GgufLinearDispatchSurfaceRow | None:
    """Dispatch row with the raw-layout quant resolved to a concrete key.

    For raw-layout rows the production table defers the quant to the resident
    weight; this resolves it per source type via
    :data:`RAW_LINEAR_SOURCE_QUANT_KEYS` so every certified record carries a
    concrete four-axis identity.  A raw source type with no concrete quant
    mapping has no certified consumer here (fail closed).
    """

    row = gguf_linear_dispatch_row(layout, activation, output)
    if row is None:
        return None
    if row.quant != FROM_WEIGHT_QUANT_TOKEN:
        return row
    quant = RAW_LINEAR_SOURCE_QUANT_KEYS.get(str(source_ggml_type))
    if quant is None:
        return None
    return GgufLinearDispatchSurfaceRow(
        layout=row.layout,
        activation=row.activation,
        output=row.output,
        layer=row.layer,
        quant=quant,
        variant=row.variant,
        abi=row.abi,
        pointer_activation=row.pointer_activation,
    )


# ---------------------------------------------------------------------------
# Per-backend GGUF consumer availability declarations
# ---------------------------------------------------------------------------

GGUF_CONSUMER_LAYER_DECLARATION_NAME = "GGUF_CONSUMER_LAYERS"

# Bounded literal wrappers the backend packages use for module constants.
_LITERAL_CALL_WRAPPERS = {
    "frozenset": frozenset,
    "set": set,
    "tuple": tuple,
    "list": list,
}


def _literal_value(node: ast.expr) -> object | None:
    """Resolve a module-level assignment expression, bounded to literals.

    Same convention as the quant-route audit's capability reader: literal
    forms plus the container-wrapper calls around a single literal argument.
    Anything else is unreadable and resolves to ``None`` — never guessed,
    never evaluated.
    """

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _LITERAL_CALL_WRAPPERS
        and len(node.args) == 1
        and not node.keywords
    ):
        try:
            inner = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            return None
        return _LITERAL_CALL_WRAPPERS[node.func.id](inner)
    return None


def _backend_source_path(backend: str) -> Path:
    # <pkg>/hipengine/loading/this.py -> <pkg>/hipengine/kernels/<backend>/
    return Path(__file__).resolve().parents[1] / "kernels" / str(backend) / "__init__.py"


def _iter_module_assignments(tree: ast.Module):
    """Yield ``(target name, value node)`` for module-level assignments.

    Covers both plain ``NAME = ...`` and annotated ``NAME: frozenset[str] = ...``
    forms (the backend packages use the annotated form for declarations).
    """

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield target.id, node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                yield node.target.id, node.value


def _read_declared_consumer_layers(backend: str) -> frozenset[str]:
    """Read one backend's ``GGUF_CONSUMER_LAYERS`` declaration from source.

    Never imports the backend package (kernel registration needs a GPU build
    toolchain at launch time and the audit/startup-isolation contract forbids
    GPU package imports on cold paths).  A missing file, missing constant, or
    nonliteral value reads as "declares none" — admission then fails closed
    for that backend instead of certifying unverifiable paths.
    """

    path = _backend_source_path(backend)
    try:
        source = path.read_text()
    except OSError:
        return frozenset()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    for name, value_node in _iter_module_assignments(tree):
        if name != GGUF_CONSUMER_LAYER_DECLARATION_NAME:
            continue
        value = _literal_value(value_node)
        if isinstance(value, frozenset) and all(isinstance(item, str) for item in value):
            return frozenset(value)
        return frozenset()
    return frozenset()


def read_backend_gguf_consumer_layers() -> dict[str, frozenset[str]]:
    """GGUF consumer-layer declarations for every hardware backend key.

    Backends without a source-declared ``GGUF_CONSUMER_LAYERS`` literal
    (scaffolds, future peers) read as declaring none.
    """

    from hipengine.kernels.backends import (
        CUDA_BACKEND_TARGET_ARCH,
        HIP_BACKEND_TARGET_ARCH,
    )

    return {
        backend: _read_declared_consumer_layers(backend)
        for backend in (*HIP_BACKEND_TARGET_ARCH, *CUDA_BACKEND_TARGET_ARCH)
    }


def backend_gguf_consumer_layers(backend: str) -> frozenset[str]:
    """One backend's declared GGUF consumer layers (empty = none declared)."""

    return read_backend_gguf_consumer_layers().get(str(backend), frozenset())
