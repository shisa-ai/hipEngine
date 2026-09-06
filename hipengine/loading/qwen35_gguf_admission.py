"""Role-safe artifact admission and operation preflight for Qwen3.5 GGUF files.

UD-U1 of ``docs/UD-QUANTS.md``.  Cold-path and CPU-only: no device work, no
HIP runtime, no kernel-package import, no torch, and no
``backend == ...``/``quant == ...`` branches in engine, model, or dispatch
code.  This module is *policy metadata* next to the shared dense-weight policy
in :mod:`hipengine.loading.qwen35_gguf_policy`.

What UD-U1 changes, and why a stamp alone is unsafe
---------------------------------------------------

A GGUF ``general.file_type`` stamp (``MOSTLY_Q4_K_M``) is shared by the plain
Qwen3.8-27B ``Q4_K_M`` control and the Unsloth Dynamic ``UD-Q4_K_M`` artifact,
and a dtype histogram is preserved when the same types are swapped between
sensitive roles.  Neither identifies what the weights *are*.  Admission here
binds to the actual role manifest instead:

- every slot's ``(role, layer scope, logical shape, GGML storage type)`` is
  canonicalized into a manifest fingerprint (UD K_M, UD K_S, and both plain
  controls all have distinct fingerprints despite sharing stamps);
- the two UD artifacts resolve to explicit artifact preset identities
  (``gguf_ud_q4_k_m`` / ``gguf_ud_q4_k_s``) only through their pinned
  fingerprints — never through the stamp or a type histogram;
- per-tensor kernel quant keys (``gguf_q4_k``, ``gguf_q3_k``, ...) stay the
  storage/kernel identities; the preset key is a model/session admission
  identity, not a fifth registry axis;
- requested *operations* (decode rows, prefill, embedding, logits, native
  multirow, NextN draft) are qualified per slot against the certified
  cold-path coverage records below, which name the existing four-axis
  ``(backend, layer, quant, variant)`` consumer families;
- the preflight plans every slot, aggregates *every* unsupported slot/mode,
  and fails before the loader performs any device allocation.

UD dense consumers for Q3_K / IQ4_NL / IQ3_S / IQ3_XXS / IQ2_S and raw dense
IQ4_XS do not exist until UD-U2..U5, so both published UD artifacts are
rejected honestly here, with the complete per-slot refusal list.  Certified
scopes are AR-only for U1: ``mtp_nextn_draft`` is refused for UD presets until
U6 resolves the draft operation set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable, Mapping

from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35GGUFModelMap,
    build_qwen35_gguf_tensor_map,
)
from hipengine.core.dtype import DType
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q4_K_X8,
    LAYOUT_GGUF_Q5_K_QMICRO_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q5_K_X8,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_GGUF_Q6_K_X8,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    Qwen35GGUFWeightSpec,
    gguf_decode_repack_enabled,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.loading.qwen35_gguf_nextn import Qwen35GGUFNextNMap
from hipengine.loading.qwen35_gguf_policy import (
    gguf_ar_decode_repack_veto,
    gguf_ar_f32_linear_contraction,
)
from hipengine.quant.gguf import GGMLQuantizationType

__all__ = [
    "CERTIFIED_OPERATION_COVERAGE",
    "DEFAULT_AR_OPERATIONS",
    "GGUF_UD_Q4_K_M_PRESET",
    "GGUF_UD_Q4_K_S_PRESET",
    "GGUF_PRESET_SCOPE_AR",
    "GGUF_PRESET_SCOPE_MTP",
    "QWEN35_GGUF_OP_AR_DECODE_C1",
    "QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS",
    "QWEN35_GGUF_OP_AR_DECODE_ROWS",
    "QWEN35_GGUF_OP_AR_PREFILL",
    "QWEN35_GGUF_OP_EMBEDDING_LOOKUP",
    "QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS",
    "QWEN35_GGUF_OP_MTP_NEXTN_DRAFT",
    "Qwen35GGUFAdmissionCertificate",
    "Qwen35GGUFAdmissionError",
    "Qwen35GGUFAdmissionReport",
    "Qwen35GGUFArtifactPreset",
    "Qwen35GGUFOperationCoverage",
    "Qwen35GGUFRoleManifest",
    "Qwen35GGUFUnsupportedOperation",
    "build_qwen35_gguf_role_manifest",
    "certificate_covers_artifact",
    "preflight_qwen35_gguf_artifact",
    "qwen35_gguf_native_row_binding_errors",
    "resolve_qwen35_gguf_artifact_preset",
]


# ---------------------------------------------------------------------------
# Artifact preset identities (model/session admission keys)
# ---------------------------------------------------------------------------

GGUF_UD_Q4_K_M_PRESET = "gguf_ud_q4_k_m"
GGUF_UD_Q4_K_S_PRESET = "gguf_ud_q4_k_s"

GGUF_PRESET_SCOPE_AR = "ar"
GGUF_PRESET_SCOPE_MTP = "mtp"


@dataclass(frozen=True)
class Qwen35GGUFArtifactPreset:
    """One artifact-qualified preset identity bound to a role manifest.

    ``preset_key`` is the model/session admission identity (the planned
    ``gguf_ud_q4_k_m`` key from the K_M campaign plus the equally explicit
    K_S identity).  ``scopes`` states what the preset is certified for;
    UD presets are AR-only until U6 resolves draft/serving operations.
    ``manifest_fingerprint`` is the binding: a file with the same stamp but a
    different role/shape/type manifest never resolves to this preset.
    """

    preset_key: str
    scopes: tuple[str, ...]
    manifest_fingerprint: str
    file_type_stamp: str | None
    note: str = ""

    def scope_certified(self, scope: str) -> bool:
        return scope in self.scopes


# Pinned UD role-manifest fingerprints (UD-U0/U1 evidence).  Computed from the
# pinned local artifacts via build_qwen35_gguf_role_manifest over the actual AR
# map plus the trailing NextN block map (structural records; the UD NextN map
# is deliberately not dtype-validated here because its draft admission is
# refused separately).  The plain controls do not appear: unknown manifests
# stay on the plain stamp-based lane with unchanged behavior.
_UD_PRESET_FINGERPRINTS: Mapping[str, tuple[str, tuple[str, ...], str]] = {
    # Qwen3.8-27B-UD-Q4_K_M.gguf: payload sha256 322e194f..., header identity
    # ab826936... (docs/UD-QUANTS-U0-IDENTITY.json), 866 tensors / 851 AR.
    "5535c5bd7a3e84c6381de70bf8ca5c6f4bcd804dabf8435b85c8418038c8619f": (
        GGUF_UD_Q4_K_M_PRESET,
        (GGUF_PRESET_SCOPE_AR,),
        "Pinned Unsloth Dynamic Qwen3.8-27B UD-Q4_K_M role manifest.",
    ),
    # Qwen3.8-27B-UD-Q4_K_S.gguf: payload sha256 75bc9c8a..., header identity
    # d2568a4b... (docs/UD-QUANTS-U0-IDENTITY.json), 866 tensors / 851 AR.
    "91130e1698bb7fc24c89f94e8b1043dd788769682b56c6514d70b6d2cdea068c": (
        GGUF_UD_Q4_K_S_PRESET,
        (GGUF_PRESET_SCOPE_AR,),
        "Pinned Unsloth Dynamic Qwen3.8-27B UD-Q4_K_S role manifest.",
    ),
}


# ---------------------------------------------------------------------------
# Role manifest and fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen35GGUFRoleManifest:
    """Canonical ``(role, scope, shape, storage type)`` records for one file.

    The fingerprint covers role names and layer scopes, so a manifest with an
    identical dtype histogram but swapped recurrent/FFN types — or a different
    tensor inventory under the same file-type stamp — produces a different
    fingerprint and cannot inherit another manifest's certification.
    """

    records: tuple[tuple[str, str, tuple[int, ...], str], ...]
    fingerprint: str

    def has_type(self, ggml_type_name: str) -> bool:
        return any(record[3] == ggml_type_name for record in self.records)


def build_qwen35_gguf_role_manifest(
    model_map: Qwen35GGUFModelMap,
    *,
    nextn_map: Qwen35GGUFNextNMap | None = None,
) -> Qwen35GGUFRoleManifest:
    """Canonicalize the actual AR (and optional NextN) maps into a fingerprint.

    NextN records use the structural (non-dtype-validated) block map so a UD
    artifact whose draft dtypes would be refused still yields its identity
    fingerprint; draft *admission* is a separate, scope-gated decision.
    """

    records: list[tuple[str, str, tuple[int, ...], str]] = []
    for slot, tensor in model_map.root_tensors.items():
        records.append(
            ("root." + slot, "root", tuple(int(d) for d in tensor.shape), tensor.ggml_type_name)
        )
    for layer in model_map.layers:
        for slot, tensor in layer.tensors.items():
            records.append(
                (
                    f"layers.{layer.layer_id}.{slot}",
                    layer.layer_type,
                    tuple(int(d) for d in tensor.shape),
                    tensor.ggml_type_name,
                )
            )
    if nextn_map is not None:
        block_id = int(nextn_map.block_id)
        for slot, tensor in nextn_map.layer_tensors.items():
            records.append(
                (
                    f"nextn_block.{block_id}.{slot}",
                    "nextn_layer",
                    tuple(int(d) for d in tensor.shape),
                    tensor.ggml_type_name,
                )
            )
        for slot, tensor in nextn_map.nextn_tensors.items():
            records.append(
                (
                    f"nextn_block.{block_id}.{slot}",
                    "nextn",
                    tuple(int(d) for d in tensor.shape),
                    tensor.ggml_type_name,
                )
            )
        for slot, tensor in nextn_map.fallback_tensors.items():
            records.append(
                (
                    f"nextn_block.{block_id}.fallback:{slot}",
                    "nextn_fallback",
                    tuple(int(d) for d in tensor.shape),
                    tensor.ggml_type_name,
                )
            )
    ordered = tuple(sorted(records))
    lines = [
        f"{role}\t{scope}\t{','.join(str(dim) for dim in shape)}\t{ggml_type}"
        for role, scope, shape, ggml_type in ordered
    ]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return Qwen35GGUFRoleManifest(records=ordered, fingerprint=digest)


def resolve_qwen35_gguf_artifact_preset(
    model_map: Qwen35GGUFModelMap,
    *,
    nextn_map: Qwen35GGUFNextNMap | None = None,
    file_type_stamp: str | None = None,
) -> Qwen35GGUFArtifactPreset | None:
    """Resolve the artifact preset from the actual manifest, or ``None``.

    ``None`` means "no pinned UD manifest": the caller stays on the plain
    stamp-based lane with unchanged behavior.  A matching fingerprint yields
    the explicit UD preset identity; the stamp is telemetry only.
    """

    manifest = build_qwen35_gguf_role_manifest(model_map, nextn_map=nextn_map)
    pinned = _UD_PRESET_FINGERPRINTS.get(manifest.fingerprint)
    if pinned is None:
        return None
    preset_key, scopes, note = pinned
    return Qwen35GGUFArtifactPreset(
        preset_key=preset_key,
        scopes=tuple(scopes),
        manifest_fingerprint=manifest.fingerprint,
        file_type_stamp=None if file_type_stamp is None else str(file_type_stamp),
        note=note,
    )


# ---------------------------------------------------------------------------
# Requested operations
# ---------------------------------------------------------------------------

QWEN35_GGUF_OP_AR_DECODE_C1 = "ar_decode_c1"
QWEN35_GGUF_OP_AR_DECODE_ROWS = "ar_decode_rows"
QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS = "ar_decode_native_rows"
QWEN35_GGUF_OP_AR_PREFILL = "ar_prefill"
QWEN35_GGUF_OP_EMBEDDING_LOOKUP = "embedding_lookup"
QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS = "lm_head_f32_logits"
QWEN35_GGUF_OP_MTP_NEXTN_DRAFT = "mtp_nextn_draft"

# The loader's resident contract for AR execution: c1 decode, multirow decode
# (rows 2..8, row-local layout-aware lanes), bulk prefill, embedding gather,
# and F32 full-vocabulary logits.  The native multirow route is a distinct
# operation with a stricter alpha/beta consumer contract (BF16-pointer owner);
# it is requested explicitly by callers that engage that route, and the MTP
# draft operation is separately scope-gated.
DEFAULT_AR_OPERATIONS = (
    QWEN35_GGUF_OP_AR_DECODE_C1,
    QWEN35_GGUF_OP_AR_DECODE_ROWS,
    QWEN35_GGUF_OP_AR_PREFILL,
    QWEN35_GGUF_OP_EMBEDDING_LOOKUP,
    QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,
)

_KNOWN_OPERATIONS = frozenset(
    (
        QWEN35_GGUF_OP_AR_DECODE_C1,
        QWEN35_GGUF_OP_AR_DECODE_ROWS,
        QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,
        QWEN35_GGUF_OP_AR_PREFILL,
        QWEN35_GGUF_OP_EMBEDDING_LOOKUP,
        QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,
        QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,
    )
)


# ---------------------------------------------------------------------------
# Certified cold-path operation coverage records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen35GGUFOperationCoverage:
    """One certified ``(operation, role class, layout, rows, dtype)`` consumer.

    ``kernel`` names the four-axis registry family
    ``(layer, quant, variant)``; the backend axis comes from the admission
    request and ``None`` quant/variant components mean "resolved from the
    resident weight / row count by the existing runtime dispatcher"
    (``hipengine.runtime.gguf_linear`` and the embedding dispatch).  A record
    is cold-path policy metadata, not a launch; it certifies that the named
    consumer family exists and owns this slot shape today.
    """

    operation: str
    role_class: str
    resident_layout: str
    source_ggml_types: frozenset[str]
    rows_scope: str
    input_dtype: str
    output_dtype: str
    kernel_layer: str
    kernel_quant: str | None = None
    kernel_variant: str | None = None
    strict_fallback: str | None = None
    note: str = ""


# Raw-GGUF storage whose dense linear consumers are registered per quant key
# (k-gemv / q4-k-gemv families).  Q3_K/IQ* raw dense consumers do not exist
# until UD-U2..U5 and are intentionally absent.
_RAW_LINEAR_QUANT_TYPES = frozenset({"Q4_K", "Q5_K", "Q6_K", "Q8_0"})

# Resident layouts the layout-aware runtime linear dispatcher supports for
# decode rows and prefill (hipengine.runtime.gguf_linear._DISPATCH_TABLE
# families plus the per-quant raw gemv registrations).
_ROW_LOCAL_LINEAR_LAYOUTS: tuple[str, ...] = (
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_GGUF_Q8_0_T16,
)

# Layouts with a registered F32-output linear dispatch (full-vocabulary
# logits).  Q4/Q5/Q8 T16 residents have no F32-output consumer.  The dense_f32
# row consumes F32 activations (dense_gemv/f32/f32_hidden_f32_out).
_F32_OUTPUT_LINEAR_LAYOUTS: tuple[str, ...] = (
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
)

# Rank-3 selected-expert residents the MoE consumers own today (raw plus the
# decode-repack expert families).  Unchanged MoE manifests keep exactly these.
_SELECTED_EXPERT_LAYOUTS: tuple[str, ...] = (
    LAYOUT_RAW_GGUF,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_X8,
    LAYOUT_GGUF_Q5_K_QMICRO_T16,
    LAYOUT_GGUF_Q5_K_X8,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_X8,
)

_RAW_EMBEDDING_TYPES = frozenset({"Q4_K", "Q5_K", "Q6_K", "Q8_0"})

_LINEAR_SOURCE_TYPES = frozenset(
    {
        "Q4_K",
        "Q5_K",
        "Q6_K",
        "Q8_0",
        "Q4_1",
        "F16",
        "BF16",
        "F32",
        "IQ2_XS",
        "IQ4_XS",
    }
)


def _linear_records(
    operations: tuple[str, ...],
    role_class: str,
    *,
    layouts: Iterable[str] = _ROW_LOCAL_LINEAR_LAYOUTS,
    source_types: frozenset[str] = _LINEAR_SOURCE_TYPES,
    kernel_layer: str = "linear",
    strict_fallback: str | None = None,
    note: str = "",
) -> list[Qwen35GGUFOperationCoverage]:
    return [
        Qwen35GGUFOperationCoverage(
            operation=operation,
            role_class=role_class,
            resident_layout=layout,
            source_ggml_types=source_types,
            rows_scope="rows_1_8_row_local" if operation != QWEN35_GGUF_OP_AR_PREFILL else "prefill_rows",
            input_dtype="bf16",
            output_dtype="bf16",
            kernel_layer=kernel_layer,
            kernel_variant=None,
            strict_fallback=strict_fallback,
            note=note,
        )
        for operation in operations
        for layout in layouts
    ]


def _certified_coverage() -> tuple[Qwen35GGUFOperationCoverage, ...]:
    row_ops = (QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS)
    records: list[Qwen35GGUFOperationCoverage] = []
    records.extend(
        _linear_records(
            (*row_ops, QWEN35_GGUF_OP_AR_PREFILL),
            "projection",
            strict_fallback="layout-aware runtime linear dispatch (launch_gguf_linear)",
            note="Dense/recurrent/shared projections; Q3_K/IQ* dense layouts are absent until UD-U2..U5.",
        )
    )
    records.extend(
        _linear_records(
            (*row_ops, QWEN35_GGUF_OP_AR_PREFILL),
            "recurrent_alpha_beta",
            strict_fallback="layout-aware runtime linear dispatch (launch_gguf_linear)",
            note="Row-local alpha/beta lanes; the native multirow BF16-pointer owner is a separate operation.",
        )
    )
    # Native multirow: identical projection coverage (the native route launches
    # the same layout-aware linear/pair consumers), but alpha/beta must be a
    # dense BF16 owner because the route passes allocation("raw") directly to
    # dense_gemv_out_bf16 (uint16_t* weight ABI).
    records.extend(
        _linear_records(
            (QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
            "projection",
            strict_fallback="layout-aware runtime linear dispatch (launch_gguf_linear)",
        )
    )
    records.append(
        Qwen35GGUFOperationCoverage(
            operation=QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,
            role_class="recurrent_alpha_beta",
            resident_layout=LAYOUT_DENSE_BF16,
            source_ggml_types=frozenset({"F32", "BF16"}),
            rows_scope="rows_2_8_native_bf16_ptr",
            input_dtype="bf16",
            output_dtype="bf16",
            kernel_layer="dense_gemv",
            kernel_quant="bf16",
            kernel_variant="out",
            strict_fallback="layout-aware runtime linear dispatch (launch_gguf_linear)",
            note=(
                "The native multirow owner passes allocation('raw') to "
                "dense_gemv_out_bf16; only a dense BF16 resident is a valid "
                "BF16-pointer owner. Raw Q8_0 bytes and sole-T16 (no raw "
                "allocation) are refused."
            ),
        )
    )
    # Norms / GDN scalars / conv1d are consumed as F32 residents. The native
    # multirow route uses the same consumers (rmsnorm, indexed conv, GDN
    # scalar ABI), so those role classes qualify for it too; alpha/beta are
    # the route-specific delta (BF16-pointer owner).
    for role_class in ("norm", "gdn_scalar", "conv1d"):
        records.extend(
            _linear_records(
                (*row_ops, QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS, QWEN35_GGUF_OP_AR_PREFILL),
                role_class,
                layouts=(LAYOUT_DENSE_F32,),
                source_types=frozenset({"F32"}),
                kernel_layer="rmsnorm" if role_class == "norm" else "gdn_chain",
                strict_fallback="f32 resident consumers (rmsnorm / GDN scalar ABI)",
            )
        )
    # MoE router and selected experts: unchanged manifests keep today's routes.
    records.extend(
        _linear_records(
            (*row_ops, QWEN35_GGUF_OP_AR_PREFILL),
            "moe_router",
            layouts=(LAYOUT_DENSE_F32, LAYOUT_DENSE_BF16),
            source_types=frozenset({"F32", "BF16"}),
            kernel_layer="dense_gemv",
            strict_fallback="dense router gemv consumers",
        )
    )
    records.extend(
        _linear_records(
            (*row_ops, QWEN35_GGUF_OP_AR_PREFILL),
            "moe_experts",
            layouts=_SELECTED_EXPERT_LAYOUTS,
            source_types=frozenset({"Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XS", "IQ4_XS"}),
            kernel_layer="moe_selected",
            strict_fallback="rank-3 raw selected-expert consumers (gguf_q*_k raw)",
            note="Selected-expert consumers require expert IDs and rank-3 metadata.",
        )
    )
    # Rank-3 IQ3_XXS selected experts: the materializer keeps them raw GGUF and
    # the registered gguf_iq3_xxs selected moe_linear families (hip_gfx1100 and
    # hip_gfx1151 gguf_iq_gemv wrappers) consume them. There is no IQ3_XXS T16
    # or X8 repack, so only the raw layout is certified; rank-2 dense IQ3_XXS
    # remains unsupported until UD-U2..U5 (the planner refuses it).
    records.extend(
        _linear_records(
            (*row_ops, QWEN35_GGUF_OP_AR_PREFILL),
            "moe_experts",
            layouts=(LAYOUT_RAW_GGUF,),
            source_types=frozenset({"IQ3_XXS"}),
            kernel_layer="moe_selected",
            strict_fallback=(
                "rank-3 raw selected-expert consumers (gguf_iq3_xxs selected "
                "gemv / dual-SiLU / weighted-down moe_linear registrations)"
            ),
            note=(
                "Rank-3 selected IQ3_XXS experts stay raw GGUF; consumed by the "
                "registered gguf_iq3_xxs selected moe_linear consumers. Rank-2 "
                "dense IQ3_XXS has no consumer until UD-U2..U5."
            ),
        )
    )
    # Embedding gather: raw compressed lookup forwards every row.  The dense
    # BF16 consumer resolves a singleton lookup (rows are dropped before the
    # launch), so there is deliberately NO certified dense-BF16 embedding
    # record: a multirow gather would silently read one token.
    records.append(
        Qwen35GGUFOperationCoverage(
            operation=QWEN35_GGUF_OP_EMBEDDING_LOOKUP,
            role_class="token_embedding",
            resident_layout=LAYOUT_RAW_GGUF,
            source_ggml_types=_RAW_EMBEDDING_TYPES,
            rows_scope="rows_any",
            input_dtype="token_ids",
            output_dtype="bf16",
            kernel_layer="embedding",
            kernel_variant="lookup_bf16_out",
            strict_fallback=None,
            note="Raw Q4_K/Q5_K/Q6_K/Q8_0 lookup forwards rows to the kernel.",
        )
    )
    # F32 full-vocabulary logits.
    for layout in _F32_OUTPUT_LINEAR_LAYOUTS:
        records.append(
            Qwen35GGUFOperationCoverage(
                operation=QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,
                role_class="lm_head",
                resident_layout=layout,
                source_ggml_types=_LINEAR_SOURCE_TYPES | _RAW_EMBEDDING_TYPES,
                rows_scope="rows_1_8_row_local",
                input_dtype="f32" if layout == LAYOUT_DENSE_F32 else "bf16",
                output_dtype="f32",
                kernel_layer="linear",
                kernel_variant=None,
                strict_fallback="linear F32-output dispatch rows",
                note=(
                    "dense_f32 weights consume F32 activations "
                    "(dense_gemv/f32/f32_hidden_f32_out)"
                    if layout == LAYOUT_DENSE_F32
                    else ""
                ),
            )
        )
    return tuple(records)


CERTIFIED_OPERATION_COVERAGE: tuple[Qwen35GGUFOperationCoverage, ...] = _certified_coverage()

_COVERAGE_INDEX: Mapping[tuple[str, str, str], Qwen35GGUFOperationCoverage] = {
    (record.operation, record.role_class, record.resident_layout): record
    for record in CERTIFIED_OPERATION_COVERAGE
}

# Slot-suffix -> role class.  Root slots keep their names; layer slots are
# matched by suffix so AR and NextN block slots share classification.
_ROLE_CLASS_BY_SLOT: Mapping[str, str] = {
    "token_embedding": "token_embedding",
    "embed_tokens": "token_embedding",
    "lm_head": "lm_head",
    "shared_head_head": "lm_head",
    "output_norm": "norm",
    "shared_head_norm": "norm",
    "attn_norm": "norm",
    "post_attention_norm": "norm",
    "attn_q_norm": "norm",
    "attn_k_norm": "norm",
    "ssm_norm": "norm",
    "enorm": "norm",
    "hnorm": "norm",
    "ssm_a": "gdn_scalar",
    "ssm_dt_bias": "gdn_scalar",
    "ssm_conv1d": "conv1d",
    "ssm_alpha": "recurrent_alpha_beta",
    "ssm_beta": "recurrent_alpha_beta",
    "ssm_out": "projection",
    "attn_q": "projection",
    "attn_k": "projection",
    "attn_v": "projection",
    "attn_output": "projection",
    "attn_qkv": "projection",
    "attn_gate": "projection",
    "ffn_gate": "projection",
    "ffn_up": "projection",
    "ffn_down": "projection",
    "ffn_gate_shexp": "projection",
    "ffn_up_shexp": "projection",
    "ffn_down_shexp": "projection",
    "eh_proj": "projection",
    "ffn_gate_inp": "moe_router",
    "ffn_gate_inp_shexp": "moe_router",
    "ffn_gate_exps": "moe_experts",
    "ffn_up_exps": "moe_experts",
    "ffn_down_exps": "moe_experts",
}


def _role_class_for_slot(slot: str) -> str:
    return _ROLE_CLASS_BY_SLOT.get(slot.rsplit(".", 1)[-1], "")


# Which role classes each operation touches.  An operation is only qualified
# or refused against slots whose role participates in it; a slot outside the
# operation's scope is neither covered nor refused by that operation.
_OPERATION_ROLE_CLASSES: Mapping[str, frozenset[str]] = {
    QWEN35_GGUF_OP_AR_DECODE_C1: frozenset(
        {"projection", "recurrent_alpha_beta", "norm", "gdn_scalar", "conv1d", "moe_router", "moe_experts"}
    ),
    QWEN35_GGUF_OP_AR_DECODE_ROWS: frozenset(
        {"projection", "recurrent_alpha_beta", "norm", "gdn_scalar", "conv1d", "moe_router", "moe_experts"}
    ),
    QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS: frozenset(
        {"projection", "recurrent_alpha_beta", "norm", "gdn_scalar", "conv1d"}
    ),
    QWEN35_GGUF_OP_AR_PREFILL: frozenset(
        {"projection", "recurrent_alpha_beta", "norm", "gdn_scalar", "conv1d", "moe_router", "moe_experts"}
    ),
    QWEN35_GGUF_OP_EMBEDDING_LOOKUP: frozenset({"token_embedding"}),
    QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS: frozenset({"lm_head"}),
    QWEN35_GGUF_OP_MTP_NEXTN_DRAFT: frozenset(
        {"projection", "recurrent_alpha_beta", "norm", "gdn_scalar", "conv1d", "token_embedding", "lm_head"}
    ),
}


def _coverage_for(
    operation: str, role_class: str, layout: str
) -> Qwen35GGUFOperationCoverage | None:
    return _COVERAGE_INDEX.get((operation, role_class, layout))


# ---------------------------------------------------------------------------
# Preflight report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen35GGUFUnsupportedOperation:
    """One refused (slot, operation) with its concrete reason."""

    slot_path: str
    role_class: str
    operation: str
    source_ggml_type: str
    resident_layout: str | None
    stage: str  # "planner_refused" | "consumer_unqualified" | "scope_refused"
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "slot_path": self.slot_path,
            "role_class": self.role_class,
            "operation": self.operation,
            "source_ggml_type": self.source_ggml_type,
            "resident_layout": self.resident_layout,
            "stage": self.stage,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Qwen35GGUFAdmissionCertificate:
    """Immutable positive admission result for one artifact/operation set.

    Retain and re-check this object instead of re-deriving support from a
    stamp, a histogram, or a non-throwing planner.
    """

    preset_key: str | None
    manifest_fingerprint: str
    backend: str
    operations: tuple[str, ...]
    file_type_stamp: str | None
    covered_slots: int

    def as_dict(self) -> dict[str, object]:
        return {
            "preset_key": self.preset_key,
            "manifest_fingerprint": self.manifest_fingerprint,
            "backend": self.backend,
            "operations": list(self.operations),
            "file_type_stamp": self.file_type_stamp,
            "covered_slots": self.covered_slots,
        }


def certificate_covers_artifact(
    certificate: Qwen35GGUFAdmissionCertificate,
    *,
    manifest_fingerprint: str,
    backend: str | None = None,
    operations: Iterable[str] | None = None,
) -> bool:
    """Return whether a certificate still binds to the given artifact identity.

    A plain-artifact certificate never covers a UD manifest (different
    fingerprint) and vice versa, even when the file-type stamps match.
    """

    if certificate.manifest_fingerprint != str(manifest_fingerprint):
        return False
    if backend is not None and certificate.backend != str(backend):
        return False
    if operations is not None:
        requested = tuple(str(operation) for operation in operations)
        if any(operation not in certificate.operations for operation in requested):
            return False
    return True


@dataclass(frozen=True)
class Qwen35GGUFAdmissionReport:
    """Aggregated preflight result: every unsupported slot and mode."""

    backend: str
    file_type_stamp: str | None
    manifest_fingerprint: str
    preset: Qwen35GGUFArtifactPreset | None
    requested_operations: tuple[str, ...]
    unsupported: tuple[Qwen35GGUFUnsupportedOperation, ...]
    covered_slots: int
    qualified_records: tuple[Qwen35GGUFOperationCoverage, ...] = field(default=())

    @property
    def supported(self) -> bool:
        return not self.unsupported

    def certificate(self) -> Qwen35GGUFAdmissionCertificate:
        if not self.supported:
            raise Qwen35GGUFAdmissionError(
                "cannot certify an artifact with unsupported operations; see "
                "Qwen35GGUFAdmissionReport.unsupported"
            )
        return Qwen35GGUFAdmissionCertificate(
            preset_key=None if self.preset is None else self.preset.preset_key,
            manifest_fingerprint=self.manifest_fingerprint,
            backend=self.backend,
            operations=self.requested_operations,
            file_type_stamp=self.file_type_stamp,
            covered_slots=self.covered_slots,
        )

    def raise_for_errors(self) -> None:
        if self.supported:
            return
        # Cold-path failure: name every refused slot, not a truncated sample.
        raise Qwen35GGUFAdmissionError(self.render_refusals(max_slots=1024))

    def render_refusals(self, *, max_slots: int = 24) -> str:
        counts: dict[str, int] = {}
        for item in self.unsupported:
            counts[item.stage] = counts.get(item.stage, 0) + 1
        summary = ", ".join(f"{stage}={count}" for stage, count in sorted(counts.items()))
        lines = [
            "GGUF artifact admission preflight refused the requested operations:",
            f"  stamp={self.file_type_stamp!r} preset="
            f"{None if self.preset is None else self.preset.preset_key!r} "
            f"manifest_fingerprint={self.manifest_fingerprint}",
            f"  backend={self.backend!r} operations={list(self.requested_operations)}",
            f"  unsupported slots/modes: {len(self.unsupported)} ({summary})",
        ]
        for item in self.unsupported[:max_slots]:
            layout = "n/a" if item.resident_layout is None else item.resident_layout
            lines.append(
                f"  - [{item.stage}] {item.slot_path} op={item.operation} "
                f"role={item.role_class or '<unknown>'} type={item.source_ggml_type} "
                f"layout={layout}: {item.reason}"
            )
        remaining = len(self.unsupported) - max_slots
        if remaining > 0:
            lines.append(f"  ... and {remaining} more refused slot(s)")
        return "\n".join(lines)


class Qwen35GGUFAdmissionError(ValueError):
    """Raised when requested GGUF operations are not certified for an artifact."""


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_qwen35_gguf_artifact(
    model_map: Qwen35GGUFModelMap,
    *,
    backend: str,
    file_type_stamp: str | None = None,
    operations: Iterable[str] = DEFAULT_AR_OPERATIONS,
    nextn_map: Qwen35GGUFNextNMap | None = None,
    decode_repack: bool | None = None,
    repack_veto: bool | None = None,
    contract_f32_linear: bool | None = None,
    slot_filter: Iterable[str] | None = None,
    dense_q4_t16: bool = False,
    dense_q4_qmicro_t16_gate_up: bool = False,
    dense_q4_t16_attn_q_08b: bool = False,
    dense_q5_t16_ssm_out: bool = False,
    dense_q5_raw_mmq_ssm_out: bool = False,
    dense_q5_qmicro_planar_ssm_out: bool = False,
    dense_q5_t16_ssm_out_08b: bool = False,
    dense_q5_t16_qkv: bool = False,
    dense_q5_t16_h5120: bool = False,
    dense_q6_qmicro_planar: bool = False,
    dense_q6_qmicro_planar_excluded_slots: Iterable[str] = (),
) -> Qwen35GGUFAdmissionReport:
    """Preflight every requested operation over every planned slot.

    Plans each slot with the production per-slot planner (capturing refusals
    instead of raising on the first one), qualifies each planned resident
    against the certified coverage records for the requested operations, and
    aggregates every unsupported slot/mode into one report.  Pure metadata:
    no device allocation, no backend import, no kernel launch.

    ``decode_repack=None`` resolves through the loader's shared environment
    gate (:func:`gguf_decode_repack_enabled`) so the preflight sees the same
    plan the loader would materialize.  ``slot_filter`` restricts the
    preflight to a subset of canonical slot paths (the loader's test/debug
    ``selected_slots`` hook); unlisted slots are neither covered nor refused.
    """

    if decode_repack is None:
        decode_repack = gguf_decode_repack_enabled(None)
    else:
        decode_repack = bool(decode_repack)

    requested = tuple(dict.fromkeys(str(operation) for operation in operations))
    unknown_ops = tuple(op for op in requested if op not in _KNOWN_OPERATIONS)
    if unknown_ops:
        raise Qwen35GGUFAdmissionError(
            "unknown requested GGUF operations (fail closed; certified "
            f"operations: {sorted(_KNOWN_OPERATIONS)}): {list(unknown_ops)}"
        )
    manifest = build_qwen35_gguf_role_manifest(model_map, nextn_map=nextn_map)
    preset = resolve_qwen35_gguf_artifact_preset(
        model_map,
        nextn_map=nextn_map,
        file_type_stamp=file_type_stamp,
    )
    unsupported: list[Qwen35GGUFUnsupportedOperation] = []
    qualified: dict[tuple[str, str, str], Qwen35GGUFOperationCoverage] = {}
    covered_slots = 0

    # Scope gate: MTP draft operations require an explicitly certified MTP
    # scope.  No preset (plain lane) or an AR-only UD preset refuses the whole
    # operation; automatic MTP inheritance from plain-file certification is
    # impossible because the scope binds to the manifest fingerprint.
    if QWEN35_GGUF_OP_MTP_NEXTN_DRAFT in requested:
        if preset is None or not preset.scope_certified(GGUF_PRESET_SCOPE_MTP):
            unsupported.append(
                Qwen35GGUFUnsupportedOperation(
                    slot_path=(
                        None
                        if nextn_map is None
                        else f"nextn_block.{int(nextn_map.block_id)}"
                    ),
                    role_class="nextn_draft",
                    operation=QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,
                    source_ggml_type="-",
                    resident_layout=None,
                    stage="scope_refused",
                    reason=(
                        "artifact preset "
                        f"{None if preset is None else preset.preset_key!r} is not "
                        "certified for MTP/NextN draft operations; AR-only and "
                        "AR+MTP qualification are distinct"
                    ),
                )
            )

    # Per-slot planning with aggregated refusals (never first-exception).
    plan_flags = dict(
        decode_repack=bool(decode_repack),
        dense_q4_t16=bool(dense_q4_t16),
        dense_q4_qmicro_t16_gate_up=bool(dense_q4_qmicro_t16_gate_up),
        dense_q4_t16_attn_q_08b=bool(dense_q4_t16_attn_q_08b),
        dense_q5_t16_ssm_out=bool(dense_q5_t16_ssm_out),
        dense_q5_raw_mmq_ssm_out=bool(dense_q5_raw_mmq_ssm_out),
        dense_q5_qmicro_planar_ssm_out=bool(dense_q5_qmicro_planar_ssm_out),
        dense_q5_t16_ssm_out_08b=bool(dense_q5_t16_ssm_out_08b),
        dense_q5_t16_qkv=bool(dense_q5_t16_qkv),
        dense_q5_t16_h5120=bool(dense_q5_t16_h5120),
        dense_q6_qmicro_planar=bool(dense_q6_qmicro_planar),
        dense_q6_qmicro_planar_excluded_slots=tuple(
            str(slot) for slot in dense_q6_qmicro_planar_excluded_slots
        ),
    )
    if repack_veto is None:
        ar_types = (
            tensor.ggml_type
            for layer in model_map.layers
            for tensor in layer.tensors.values()
        )
        plan_flags["decode_repack"] = bool(decode_repack) and not gguf_ar_decode_repack_veto(
            ar_types
        )
    else:
        plan_flags["decode_repack"] = bool(decode_repack) and not bool(repack_veto)
    if contract_f32_linear is None:
        ar_types = (
            tensor.ggml_type
            for layer in model_map.layers
            for tensor in layer.tensors.values()
        )
        contraction = gguf_ar_f32_linear_contraction(ar_types)
    else:
        contraction = bool(contract_f32_linear)

    slot_tensors: list[tuple[str, object]] = [
        *(("root." + slot, tensor) for slot, tensor in model_map.root_tensors.items())
    ]
    for layer in model_map.layers:
        slot_tensors.extend(
            (f"layers.{layer.layer_id}.{slot}", tensor)
            for slot, tensor in layer.tensors.items()
        )
    if nextn_map is not None and QWEN35_GGUF_OP_MTP_NEXTN_DRAFT in requested:
        block_id = int(nextn_map.block_id)
        slot_tensors.extend(
            (f"nextn_block.{block_id}.{slot}", tensor)
            for slot, tensor in nextn_map.layer_tensors.items()
        )
        slot_tensors.extend(
            (f"nextn_block.{block_id}.{slot}", tensor)
            for slot, tensor in nextn_map.nextn_tensors.items()
        )
        slot_tensors.extend(
            (f"nextn_block.{block_id}.fallback:{slot}", tensor)
            for slot, tensor in nextn_map.fallback_tensors.items()
        )

    checked_ops = tuple(op for op in requested if op != QWEN35_GGUF_OP_MTP_NEXTN_DRAFT)
    allowed_slots = None if slot_filter is None else {str(slot) for slot in slot_filter}
    for slot_path, tensor in slot_tensors:
        if allowed_slots is not None and str(slot_path) not in allowed_slots:
            continue
        role_class = _role_class_for_slot(str(slot_path).rsplit(".", 1)[-1])
        if not role_class:
            unsupported.append(
                Qwen35GGUFUnsupportedOperation(
                    slot_path=str(slot_path),
                    role_class="",
                    operation=checked_ops[0] if checked_ops else "-",
                    source_ggml_type=tensor.ggml_type_name,
                    resident_layout=None,
                    stage="consumer_unqualified",
                    reason="slot has no certified role class",
                )
            )
            continue
        applicable_ops = tuple(
            op for op in checked_ops if role_class in _OPERATION_ROLE_CLASSES.get(op, frozenset())
        )
        if not applicable_ops:
            continue
        try:
            spec = plan_qwen35_gguf_weight_spec(
                str(slot_path),
                tensor,
                contract_f32_linear=contraction,
                **plan_flags,
            )
        except ValueError as error:
            for operation in applicable_ops:
                unsupported.append(
                    Qwen35GGUFUnsupportedOperation(
                        slot_path=str(slot_path),
                        role_class=role_class,
                        operation=operation,
                        source_ggml_type=tensor.ggml_type_name,
                        resident_layout=None,
                        stage="planner_refused",
                        reason=str(error),
                    )
                )
            continue
        slot_supported = True
        for operation in applicable_ops:
            record = _coverage_for(operation, role_class, spec.layout)
            if record is None or tensor.ggml_type_name not in record.source_ggml_types:
                slot_supported = False
                if role_class == "token_embedding" and spec.layout == LAYOUT_DENSE_BF16:
                    reason = (
                        "dense-BF16 embedding consumer (_launch_dense_bf16) does "
                        "not forward rows: a multirow gather would silently "
                        "resolve a singleton token"
                    )
                elif (
                    role_class == "recurrent_alpha_beta"
                    and operation == QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS
                ):
                    reason = (
                        "the native multirow owner passes allocation('raw') to "
                        "dense_gemv_out_bf16 (uint16_t* weight ABI); only a dense "
                        f"BF16 resident is a valid BF16-pointer owner, got "
                        f"source={tensor.ggml_type_name} layout={spec.layout!r}"
                    )
                else:
                    reason = (
                        "no certified consumer for role_class="
                        f"{role_class!r} source={tensor.ggml_type_name} "
                        f"layout={spec.layout!r} operation={operation!r}"
                    )
                unsupported.append(
                    Qwen35GGUFUnsupportedOperation(
                        slot_path=str(slot_path),
                        role_class=role_class,
                        operation=operation,
                        source_ggml_type=tensor.ggml_type_name,
                        resident_layout=spec.layout,
                        stage="consumer_unqualified",
                        reason=reason,
                    )
                )
                continue
            qualified[(operation, role_class, spec.layout)] = record
        if slot_supported:
            covered_slots += 1

    return Qwen35GGUFAdmissionReport(
        backend=str(backend),
        file_type_stamp=None if file_type_stamp is None else str(file_type_stamp),
        manifest_fingerprint=manifest.fingerprint,
        preset=preset,
        requested_operations=requested,
        unsupported=tuple(unsupported),
        covered_slots=covered_slots,
        qualified_records=tuple(qualified.values()),
    )


def qwen35_gguf_native_row_binding_errors(resident_weights: object) -> tuple[str, ...]:
    """Return every linear-attention alpha/beta resident that is not a valid
    BF16-pointer owner for ``ar_decode_native_rows``.

    The native multirow route (``Qwen35GGUFResidentSession.step_rows_native``
    and ``capture_native_rows_graph``) passes ``allocation('raw')`` for
    ``ssm_alpha``/``ssm_beta`` straight into ``dense_gemv_out_bf16``, whose
    weight ABI is a ``uint16_t*`` BF16 pointer.  Only a dense-BF16 resident
    with a real BF16 ``raw`` allocation satisfies that contract.  Raw-GGUF
    bytes (raw Q8_0), sole-T16 residents without any raw allocation, and
    dense-F32 residents (uncontracted plain alpha/beta) would be read as the
    wrong byte stream.  This check is pure host metadata over the actual
    resident records - no device work - so the runner can enforce the actual
    binding at native execution entry, before any state mutation or device
    call.  Admission-side, callers that know they will run the native route
    bind it before allocation via
    ``materialize_qwen35_gguf_weights(requested_operations=...)``.
    """

    config = getattr(resident_weights, "config", None)
    layers = tuple(getattr(resident_weights, "layers", ()) or ())
    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    errors: list[str] = []
    for layer_id, layer_type in enumerate(layer_types):
        if layer_type != LINEAR_ATTENTION or layer_id >= len(layers):
            continue
        layer = layers[layer_id]
        for slot in ("ssm_alpha", "ssm_beta"):
            weight = getattr(layer, "weights", {}).get(slot)
            if weight is None:
                errors.append(
                    f"layers.{layer_id}.{slot}: resident layer has no {slot!r} weight record"
                )
                continue
            layout = getattr(getattr(weight, "spec", None), "layout", None)
            try:
                allocation = weight.allocation("raw")
            except KeyError:
                allocation = None
            tensor_dtype = getattr(getattr(allocation, "tensor", None), "dtype", None)
            if layout == LAYOUT_DENSE_BF16 and allocation is not None and tensor_dtype == DType.BF16:
                continue
            if allocation is None:
                detail = "no 'raw' allocation exists (a sole-T16 resident cannot resolve the BF16-pointer owner at all)"
            elif layout != LAYOUT_DENSE_BF16:
                detail = (
                    f"resident layout {layout!r} is not a dense-BF16 owner; its 'raw' "
                    "bytes would be read as BF16 bits by dense_gemv_out_bf16"
                )
            else:
                detail = f"'raw' allocation dtype {tensor_dtype!r} is not BF16"
            errors.append(
                f"layers.{layer_id}.{slot}: ar_decode_native_rows requires a dense-BF16 "
                f"alpha/beta resident (BF16-pointer owner); {detail}"
            )
    return tuple(errors)


def _model_map_from_info(info) -> Qwen35GGUFModelMap:
    return build_qwen35_gguf_tensor_map(info)
