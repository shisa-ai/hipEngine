"""CPU-safe concrete GGUF consumer-surface metadata (UD-U1 F3).

Cold-path policy metadata only: no HIP runtime, no kernel-package import, no
torch, no device query.  Two kinds of production truth are mirrored here and
bound by explicit parity tests
(``tests/test_qwen35_gguf_consumer_surface_parity.py``) instead of being
imported at admission time:

1. **The layout-aware linear dispatch surface.**
   ``runtime/gguf_linear._DISPATCH_TABLE`` maps
   ``(resident layout, activation dtype, output dtype)`` to a four-axis
   consumer family ``(layer, quant, variant)`` plus a launch ABI, and
   ``_variant_for_rows`` rewrites the variant for multirow launches (with a
   Q4-T16 special case in ``resolve_gguf_linear_dispatch``).  The production
   module imports GPU kernel packages, so the admission preflight cannot
   import it; :data:`GGUF_LINEAR_DISPATCH_SURFACE` mirrors every row — the
   same keys, layers, quant tokens, variants, ABIs, and rows>1 variant
   mapping — and the parity test asserts byte-for-byte equality with the
   production table.  A mirror that drifts fails the parity test instead of
   silently mis-certifying.

2. **Per-backend GGUF consumer availability.**
   Each hardware backend package declares the concrete GGUF consumer *layers*
   it registers as a module-level ``GGUF_CONSUMER_LAYERS`` frozenset literal
   in its ``__init__.py``.  :func:`read_backend_gguf_consumer_layers` reads
   those declarations with the same bounded AST literal-reader convention the
   quant-route audit uses for capability constants (never an import, never
   expression evaluation); a missing or nonliteral declaration reads as
   "declares none" and admission fails closed.  The parity test proves the
   declarations are neither lies nor stale: every concrete consumer key named
   by the certified coverage records is actually registered on every
   declaring backend, and the ``cuda_sm120a`` scaffold (which registers no
   GGUF consumer) declares none.

F3's concrete dtype rule lives here too: the *actual* caller activation dtype
of the certified operations is BF16 (the production lm-head / projection
callers pass ``scratch.norm`` BF16 without an input override).  The
``(dense_f32, bf16, f32)`` combination has NO dispatch row — a dense-F32
resident can produce F32 logits only through the registered F32-activation
row (``dense_gemv/f32/f32_hidden_f32_out``), which is a valid route only when
the caller declares an F32 input override.  Admission models this exactly:
the default record set certifies the actual default caller dtype, and the
F32-input records are consulted only for operations the caller explicitly
declares (``preflight_qwen35_gguf_artifact(f32_input_operations=...)``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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
RAW_LINEAR_SOURCE_QUANT_KEYS: Mapping[str, str] = {
    "Q6_K": "gguf_q6_k",
    "Q8_0": "gguf_q8_0",
}


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

    def variant_for_rows(self, rows: int) -> str:
        if int(rows) <= 0:
            raise ValueError("rows must be positive")
        if int(rows) == 1:
            return self.variant
        if self.layout in (LAYOUT_GGUF_Q4_K_T16, LAYOUT_GGUF_Q4_K_QMICRO_T16):
            # resolve_gguf_linear_dispatch rewrites Q4 T16 multirow launches
            # to the WMMA prefill leaf regardless of the table variant.
            return "t16_wmma_prefill_bf16_bf16_out"
        variant = self.variant
        if variant.startswith("pack8_"):
            return f"pack8_prefill_{variant[len('pack8_') :]}"
        if variant.startswith("gemv_"):
            return f"prefill_{variant[len('gemv_') :]}"
        if variant == "out":
            return "prefill_out"
        return variant


def _row(
    layout: str,
    activation: str,
    output: str,
    layer: str,
    quant: str,
    variant: str,
    abi: str,
) -> GgufLinearDispatchSurfaceRow:
    return GgufLinearDispatchSurfaceRow(
        layout=layout,
        activation=activation,
        output=output,
        layer=layer,
        quant=quant,
        variant=variant,
        abi=abi,
    )


# Exact mirror of runtime/gguf_linear._DISPATCH_TABLE (parity-tested).  Keep
# the same row order as the production table so drift review is a diff.
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
    _row(LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16, "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_fp16_fp16_out", "t16"),
    _row(LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_F32, GGUF_OUTPUT_BF16, "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_f32_bf16_out", "t16"),
)


def gguf_linear_dispatch_row(
    layout: str,
    activation: str,
    output: str,
) -> GgufLinearDispatchSurfaceRow | None:
    """The mirrored dispatch row for one concrete dtype contract, or ``None``.

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
