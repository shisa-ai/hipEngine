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
  and fails before the loader performs any device allocation;
- a positive certificate binds to the ACTUAL planned residents — a canonical
  per-slot record (logical slot, source identity, resident layout, quant
  key, allocation names, planned allocation bytes, sidecars) and its digest
  recorded in :class:`Qwen35GGUFPlanContract` — so an env-resolved layout
  switch (for example the selected gate/up X8 repack) changes the contract
  even when every caller kwarg matches, and operation-coverage approval
  requires verifying the intended plan contract, never source identity
  alone;
- authorization additionally requires COMPLETE qualification: the contract
  accounts every slot the preflight tried to qualify
  (``required_plan_slots``) plus operation-scope refusals, and is
  authorization-capable (:meth:`Qwen35GGUFPlanContract.is_complete`) only
  when the qualified records cover exactly that accounting — so a refused
  preflight report (which records only its successful slots) can never mint
  or verify as authorization. The intended operation set — explicit, or the
  intended contract's own checked operations when omitted — must be
  certified on both sides; identical planned residents never confer
  row-operation or dtype qualification;
- F3/F4 invocation binding: production and admission share CPU-safe
  descriptors, row resolution and caller/adapter metadata in
  :mod:`hipengine.loading.qwen35_gguf_consumer_surface` and
  :mod:`hipengine.loading.gguf_selected_contract`. Certificates bind actual
  operands, states, row domains, ordered selected partners and source/resident
  records. A filter cannot implicitly change a paired intent to a singleton;
  incomplete call dependencies cannot authorize even after narrowing;
- the requested backend must concretely register GGUF consumers: each
  backend package declares its ``GGUF_CONSUMER_LAYERS`` in source (read by
  a bounded AST literal reader, never an import), and a known target-arch
  name alone never qualifies (the cuda_sm120a scaffold declares none and
  is refused);
- coverage resolves ACTUAL caller operands, not convenient nearby rows.
  Most projections and the native head supply BF16. The GDN ssm_out handoff
  resolves F32 or its executed BF16 adapter explicitly. A dense-F32 lm-head
  has no default F32-logits row for a supplied BF16 activation; an explicit
  F32 declaration must describe a real F32 input, not permission to fall back
  to BF16. Availability/ABI certificates do not grant profile authorization;
  native runtime-entry consumption is implemented in F1. F5 profile plans use
  the header identity adapter here through their own qualifier; that repair
  awaits independent review and does not confer new numerical certification.

UD dense consumers for Q3_K / IQ4_NL / IQ3_S / IQ3_XXS / IQ2_S and raw dense
IQ4_XS do not exist until UD-U2..U5, so both published UD artifacts are
rejected honestly here, with the complete per-slot refusal list.  Certified
scopes are AR-only for U1: ``mtp_nextn_draft`` is refused for UD presets until
U6 resolves the draft operation set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
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
    planned_qwen35_gguf_weight_allocation_nbytes,
    validate_qwen35_gguf_resident_prerequisites,
)
from hipengine.kernels.backends import (
    CUDA_BACKEND_TARGET_ARCH,
    HIP_BACKEND_TARGET_ARCH,
)
from hipengine.loading.qwen35_gguf_consumer_surface import (
    CONV_DECODE, CONV_PREFILL, GDN_SEGMENTS, RMSNORM,
    InvocationContract, linear_consumer_contract, auxiliary_consumer_contract,
    OPERATION_ROW_LIMITS, resolve_embedding_consumer_contract,
    resolve_router_consumer_contract, resolve_gdn_segments_contract,
    native_alpha_beta_consumer_contract, resolve_gdn_operation_contract, conv_operation_contract,
    resolve_gdn_output_handoff, validate_gdn_geometry, gdn_value_head_dim,
    validate_conv_geometry,
    GGUF_ACTIVATION_BF16,
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_BF16,
    GGUF_OUTPUT_F32,
    RAW_LINEAR_SOURCE_QUANT_KEYS,
    backend_gguf_consumer_layers,
    source_linear_dispatch_row,
)
from hipengine.loading.qwen35_gguf_execution import NATIVE_EXECUTION_ROLE_CLASSES
from hipengine.loading.gguf_selected_contract import (
    RAW_SELECTED_CONSUMERS as _RAW_SELECTED_CONSUMERS,
    REPACKED_SELECTED_CONSUMERS as _REPACKED_SELECTED_CONSUMERS,
    SelectedCallIntent, BoundSelectedCall, bind_selected_call,
    default_selected_call_intents,
)
from hipengine.loading.qwen35_gguf_nextn import Qwen35GGUFNextNMap
from hipengine.loading.qwen35_gguf_policy import (
    gguf_ar_decode_repack_veto,
    gguf_ar_f32_linear_contraction,
)
from hipengine.quant.gguf import GGMLQuantizationType

__all__ = [
    "CERTIFIED_OPERATION_COVERAGE",
    "CERTIFIED_F32_INPUT_OPERATION_COVERAGE",
    "DEFAULT_AR_OPERATIONS",
    "GGUF_UD_Q4_K_M_PRESET",
    "GGUF_UNQUALIFIED_MANIFEST_PRESET",
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
    "Qwen35GGUFPlanContract",
    "Qwen35GGUFAdmissionError",
    "Qwen35GGUFAdmissionReport",
    "qwen35_gguf_artifact_preset_key",
    "qwen35_gguf_artifact_identity_from_info",
    "qwen35_gguf_artifact_preset_key_for_report",
    "Qwen35GGUFArtifactPreset",
    "Qwen35GGUFOperationCoverage",
    "Qwen35GGUFRoleManifest",
    "Qwen35GGUFUnsupportedOperation",
    "build_qwen35_gguf_role_manifest",
    "certificate_covers_artifact",
    "certificate_matches_artifact_identity",
    "preflight_qwen35_gguf_artifact",
    "qwen35_gguf_native_row_binding_errors",
    "qwen35_gguf_planned_weight_digest",
    "qwen35_gguf_planned_weight_record",
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
# refused separately).  The plain controls do not appear here; they are pinned
# separately below.
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

# Artifact preset-key sentinel bound to manifests that carry no UD admission
# preset and do not match a pinned qualified plain control.  It never enters
# preset tables; its only job is to extend the policy identity to a 3-tuple so
# unknown manifests miss every plain-stamp-keyed policy row and the packaged
# hot-vocabulary identity, and resolve through the generic strict fallback
# instead.
GGUF_UNQUALIFIED_MANIFEST_PRESET = "gguf-unqualified-manifest"

# Pinned qualified plain-control role-manifest fingerprints (UD-U0/U1
# evidence; computed on the physical benchmark host from the local artifacts
# with build_qwen35_gguf_role_manifest over the AR map plus the structural
# NextN block map when present).  A plain artifact whose manifest matches one
# of these keeps the historical (geometry, stamp) plain policy identity and
# the packaged hot-vocabulary identity.  Every OTHER preset-less manifest is
# an unknown manifest and gets GGUF_UNQUALIFIED_MANIFEST_PRESET: plain-certified
# arithmetic rows were tuned on these controls, so an unverified file sharing
# only a stamp must not silently inherit them.
_PINNED_PLAIN_CONTROL_FINGERPRINTS: frozenset[str] = frozenset(
    {
        # /models/gguf/Qwen3.8-27B-Q4_K_M.gguf (hidden 5120, MOSTLY_Q4_K_M,
        # NextN structural block present).
        "0b70a3061c99df35c5f1fd7e600fbe2e428a88710f615f246d1407f0e7f62a88",
        # /models/gguf/Qwen3.8-27B-Q4_K_S.gguf (hidden 5120, MOSTLY_Q4_K_S,
        # NextN structural block present).
        "a885f2c48acefe4de83113f1d1511db2fc3f01d9140235cd99a343990f8d8647",
        # /models/gguf/Qwen3.5-0.8B-Q8_0.gguf (hidden 1024, MOSTLY_Q8_0).
        "d21530dec88af620099bbcc61bd115762db07b3433913ed6ffd0382d390dc03d",
        # /models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf (hidden 1024, MOSTLY_Q4_K_M).
        "84f294df4b948d691d7be0da707721a62b7bbbc9e8e5727feb81d7380297b9b1",
        # /models/gguf/Qwen3.6-27B-Q4_K_M.gguf (hidden 5120, MOSTLY_Q4_K_M;
        # shares the Qwen3.8 dense geometry+stamp, so it must stay pinned or
        # it would silently lose the plain dense policy rows).
        "a0b9749e2d7bde4b5f633b7b2b92356db746ae2f82c582fefe2ea0778c4a97b3",
        # /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (MoE hidden 2048,
        # MOSTLY_Q4_K_M, NextN structural block present; not a pinned UD
        # preset — a qualified plain MoE control).
        "a4d6fd0a9468fc42250c1e459f1913ce1d6e7ad340974277716e2e49252fb0ad",
        # /models/gguf/Ornith-1.5-35B-A3B-Q4_K_M.gguf (MoE hidden 2048,
        # MOSTLY_Q4_K_M).
        "350b6afc6cb4a21eabc600149169cbaab02c0a94f819ba8d2522c62f288acbf8",
    }
)


def qwen35_gguf_artifact_preset_key_for_report(
    report: Qwen35GGUFAdmissionReport,
) -> str | None:
    """Artifact preset key for one admission report.

    ``None`` means the artifact is a pinned qualified plain control and keeps
    the historical plain (geometry, stamp) policy identity.  A UD preset key
    binds when the manifest matches a pinned UD fingerprint.  Everything else
    is an unknown manifest and receives ``GGUF_UNQUALIFIED_MANIFEST_PRESET``
    so policy-table consumers and the packaged hot-vocabulary resolver treat
    it as unqualified instead of inheriting plain-certified behavior.
    """

    if report.preset is not None:
        return report.preset.preset_key
    if report.manifest_fingerprint in _PINNED_PLAIN_CONTROL_FINGERPRINTS:
        return None
    return GGUF_UNQUALIFIED_MANIFEST_PRESET


def qwen35_gguf_artifact_preset_key(
    model_map: Qwen35GGUFModelMap,
    *,
    nextn_map: Qwen35GGUFNextNMap | None = None,
    file_type_stamp: str | None = None,
    preset: Qwen35GGUFArtifactPreset | None = None,
) -> str | None:
    """Artifact preset key for one artifact map (map-based form).

    Same contract as :func:`qwen35_gguf_artifact_preset_key_for_report` for
    callers that hold the maps instead of an admission report (for example
    the NextN/MTP draft materializer, which resolves the preset for scope
    checking and needs the same qualification for the hot-vocabulary
    identity).  ``preset`` may carry a pre-resolved preset to avoid a second
    resolution pass.
    """

    if preset is None:
        preset = resolve_qwen35_gguf_artifact_preset(
            model_map,
            nextn_map=nextn_map,
            file_type_stamp=file_type_stamp,
        )
    if preset is not None:
        return preset.preset_key
    manifest = build_qwen35_gguf_role_manifest(model_map, nextn_map=nextn_map)
    if manifest.fingerprint in _PINNED_PLAIN_CONTROL_FINGERPRINTS:
        return None
    return GGUF_UNQUALIFIED_MANIFEST_PRESET


def qwen35_gguf_artifact_identity_from_info(model_info) -> tuple[str, str | None]:
    """Header-only identity shared by profile and runtime policy callers.

    The structural NextN map participates exactly as in loader admission.
    ``None`` is returned only for an actually pinned plain manifest, never
    because qualification context was absent.
    """
    model_map = build_qwen35_gguf_tensor_map(model_info)
    nextn_map = None
    if model_map.config.ignored_block_ids:
        from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map

        nextn_map = build_qwen35_gguf_nextn_tensor_map(model_info, strict=False)
    manifest = build_qwen35_gguf_role_manifest(model_map, nextn_map=nextn_map)
    return manifest.fingerprint, qwen35_gguf_artifact_preset_key(
        model_map, nextn_map=nextn_map,
        file_type_stamp=getattr(model_info, "file_type_name", None),
    )


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

    ``None`` means only "no pinned UD manifest", not qualified plain. Callers
    use ``qwen35_gguf_artifact_preset_key`` to distinguish pinned plain controls
    from unknown manifests. A matching fingerprint yields the explicit UD
    preset identity; the stamp is telemetry only.
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

    F3: every record binds a CONCRETE four-axis consumer identity —
    ``kernel_layer``/``kernel_quant``/``kernel_variant`` name the exact
    registry key (rows=1 form; ``kernel_variant_rows_many`` additionally
    names the multirow variant when the dispatcher rewrites it), taken from
    the shared production dispatch owner or concrete auxiliary wrapper
    descriptors.  Placeholder quant/variant components (``<from-weight>``,
    ``None``) are not certification and no longer appear in the certified
    sets.  The single exception to registry mediation is documented
    explicitly: when the production runtime consumes a weight through a
    direct module wrapper that is not registered under its own key (raw
    rank-3 Q4_K selected experts), ``consumer_module``/``consumer_symbol``
    name that wrapper and the parity test proves the symbol exists on every
    declaring backend.

    These records describe the qualified primitive/resident surface. Actual
    authorization uses resolved ``InvocationContract`` and ``BoundSelectedCall``
    objects, not this index: selected calls have explicit partner/input/output
    intents, and the GDN handoff can resolve F32 input without an override.
    The index's selected-expert dtype fields describe its BF16 baseline only.
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
    kernel_variant_rows_many: str | None = None
    consumer_module: str | None = None
    consumer_symbol: str | None = None
    strict_fallback: str | None = None
    note: str = ""

    def invocation(self, spec, *, backend: str, config, recurrent_state_dtype: str = "f32", gdn_force_bf16: bool = False) -> InvocationContract:
        """Bind a qualified role to its concrete consumer and geometry."""
        if self.role_class in {"gdn_scalar", "gdn_norm"}:
            consumer = resolve_gdn_operation_contract(self.operation, recurrent_state_dtype)
            validate_gdn_geometry(
                config.ssm_group_count, config.ssm_time_step_rank, config.ssm_state_size,
                gdn_value_head_dim(config.ssm_inner_size, config.ssm_time_step_rank),
                prefill=consumer.abi == "prefill_gdn",
            )
        elif self.role_class == "recurrent_alpha_beta" and self.operation == QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS:
            consumer = native_alpha_beta_consumer_contract()
        elif self.role_class == "conv1d":
            consumer = conv_operation_contract(self.operation)
            validate_conv_geometry(
                2 * config.ssm_group_count * config.ssm_state_size + config.ssm_inner_size,
                config.ssm_conv_kernel,
            )
        elif self.role_class == "token_embedding":
            consumer = resolve_embedding_consumer_contract(
                spec.layout, spec.quant_key, rows=OPERATION_ROW_LIMITS[self.operation][0])
        elif self.role_class == "moe_router":
            consumer = resolve_router_consumer_contract(spec.layout, spec.quant_key, self.input_dtype)
        else:
            row = source_linear_dispatch_row(
                spec.source.ggml_type_name, spec.layout, self.input_dtype, self.output_dtype,
            ) if self.role_class in {"projection", "recurrent_alpha_beta", "lm_head"} else None
            consumer = linear_consumer_contract(row) if row is not None else auxiliary_consumer_contract(
                self.role_class, layout=spec.layout, layer=self.kernel_layer,
                quant=self.kernel_quant, variant=self.kernel_variant,
                module=self.consumer_module, symbol=self.consumer_symbol,
            )
        adapters = ()
        if self.input_dtype == "f32" and self.output_dtype == "f32" and self.role_class in {"projection", "recurrent_alpha_beta"}:
            # Runner's F32 verifier projections explicitly cast their output
            # to the BF16 activation buffers (not an input fallback).
            adapters = ("output:f32_to_bf16",)
        if self.role_class in {"gdn_scalar", "gdn_norm"} and self.operation == QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS:
            adapters = ("output:f32_to_bf16",)
        if spec.slot_path.endswith(".ssm_out") and self.operation in {QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS}:
            adapters = resolve_gdn_output_handoff(spec.layout, force_bf16=gdn_force_bf16).adapters
        shape = tuple(int(d) for d in spec.source.shape)
        if self.role_class in {"gdn_scalar", "gdn_norm", "conv1d"}:
            shape += tuple(int(getattr(config, name)) for name in (
                "ssm_group_count", "ssm_time_step_rank", "ssm_state_size", "ssm_inner_size", "ssm_conv_kernel",
            ))
        parameters = (("rms_norm_eps", float(config.rms_norm_eps)),) if self.role_class in {"norm", "gdn_scalar", "gdn_norm"} else ()
        return InvocationContract(spec.slot_path, self.operation, backend, shape,
                                  self.rows_scope, consumer, adapters, parameters,
                                  OPERATION_ROW_LIMITS[self.operation])


# Source types that can plan to each certified linear layout (planner truth:
# rank-2 Q4_K -> pack8; rank-2 Q5_K/Q6_K layer slots and F16/BF16/Q4_1/
# IQ2_XS/IQ4_XS expand dense-BF16; F32 stays dense-F32; Q6_K/Q8_0 heads stay
# raw without repack).  The admission lookup key includes the source type, so
# a record can only fire for a source the planner would actually route here.
_DENSE_BF16_LINEAR_SOURCE_TYPES = frozenset(
    {"Q4_1", "Q5_K", "Q6_K", "F16", "BF16", "IQ2_XS", "IQ4_XS"}
)
_DENSE_F32_LINEAR_SOURCE_TYPES = frozenset({"F32"})
_Q4_PACK8_SOURCE_TYPES = frozenset({"Q4_K"})
_Q4_T16_SOURCE_TYPES = frozenset({"Q4_K"})
_Q5_T16_SOURCE_TYPES = frozenset({"Q5_K"})
_Q6_T16_SOURCE_TYPES = frozenset({"Q6_K"})
_Q8_T16_SOURCE_TYPES = frozenset({"Q8_0"})


def _surface_records(
    operations: tuple[str, ...],
    role_class: str,
    layout: str,
    source_types: frozenset[str],
    *,
    activation: str,
    output: str,
    rows_scope: str | None = None,
    strict_fallback: str | None = None,
    note: str = "",
) -> list[Qwen35GGUFOperationCoverage]:
    """Emit qualification records bound to one shared production dispatch row.

    Raw-layout rows resolve the concrete per-source quant key; a source type
    without one has no certified consumer and emits nothing (fail closed).
    """

    records: list[Qwen35GGUFOperationCoverage] = []
    for source_type in sorted(source_types):
        row = source_linear_dispatch_row(source_type, layout, activation, output)
        if row is None or (row.pointer_activation is not None and row.pointer_activation != activation):
            continue
        variant_many = row.variant_for_rows(2)
        for operation in operations:
            records.append(
                Qwen35GGUFOperationCoverage(
                    operation=operation,
                    role_class=role_class,
                    resident_layout=layout,
                    source_ggml_types=frozenset({source_type}),
                    rows_scope=rows_scope
                    or (
                        "prefill_rows"
                        if operation == QWEN35_GGUF_OP_AR_PREFILL
                        else "rows_1_8_row_local"
                    ),
                    input_dtype=activation,
                    output_dtype=output,
                    kernel_layer=row.layer,
                    kernel_quant=row.quant,
                    kernel_variant=row.variant,
                    kernel_variant_rows_many=(
                        None if variant_many == row.variant else variant_many
                    ),
                    strict_fallback=strict_fallback,
                    note=note,
                )
            )
    return records


def _bf16_linear_records(
    operations: tuple[str, ...],
    role_class: str,
    *,
    layouts: tuple[str, ...] | None = None,
    rows_scope: str | None = None,
    strict_fallback: str | None = None,
    note: str = "",
) -> list[Qwen35GGUFOperationCoverage]:
    """Default-caller (BF16-activation) linear records for every layout that
    has a ``(layout, bf16, bf16)`` dispatch row (optionally restricted to
    ``layouts`` — for example the native-rows alpha/beta BF16-pointer owner
    admits only the dense-BF16 resident)."""

    all_layouts = (
        (LAYOUT_Q4_K_PACK8, _Q4_PACK8_SOURCE_TYPES),
        (LAYOUT_RAW_GGUF, frozenset(RAW_LINEAR_SOURCE_QUANT_KEYS)),
        (LAYOUT_DENSE_BF16, _DENSE_BF16_LINEAR_SOURCE_TYPES),
        (LAYOUT_DENSE_F32, _DENSE_F32_LINEAR_SOURCE_TYPES),
        (LAYOUT_GGUF_Q4_K_T16, _Q4_T16_SOURCE_TYPES),
        (LAYOUT_GGUF_Q4_K_QMICRO_T16, _Q4_T16_SOURCE_TYPES),
        (LAYOUT_GGUF_Q5_K_T16, _Q5_T16_SOURCE_TYPES),
        (LAYOUT_GGUF_Q6_K_T16, _Q6_T16_SOURCE_TYPES),
        (LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, _Q6_T16_SOURCE_TYPES),
        (LAYOUT_GGUF_Q8_0_T16, _Q8_T16_SOURCE_TYPES),
    )
    selected = all_layouts if layouts is None else tuple(
        (layout, types) for layout, types in all_layouts if layout in set(layouts)
    )
    records: list[Qwen35GGUFOperationCoverage] = []
    fallback = strict_fallback or "layout-aware runtime linear dispatch (launch_gguf_linear)"
    for layout, source_types in selected:
        records.extend(
            _surface_records(
                operations,
                role_class,
                layout,
                source_types,
                activation=GGUF_ACTIVATION_BF16,
                output=GGUF_OUTPUT_BF16,
                rows_scope=rows_scope,
                strict_fallback=fallback,
                note=note,
            )
        )
    return records


# Selected-expert consumers per concrete source format and resident layout
# (rank-3 experts).  Every entry names the concrete registered key (or, for
# raw rank-3 Q4_K, the production module wrapper the runtime calls).
_SELECTED_EXPERT_STRICT_FALLBACK = (
    "rank-3 raw selected-expert consumers (gguf_q*_k raw / gguf_iq* selected "
    "moe_linear registrations)"
)


def _selected_expert_records(
    operations: tuple[str, ...],
    *,
    raw: bool,
) -> list[Qwen35GGUFOperationCoverage]:
    records: list[Qwen35GGUFOperationCoverage] = []
    entries: list[tuple[str, str, str, str, str, str | None, str | None]] = []
    if raw:
        for source_type, layer, quant, variant, module, symbol in _RAW_SELECTED_CONSUMERS:
            entries.append((LAYOUT_RAW_GGUF, source_type, layer, quant, variant, module, symbol))
    else:
        for layout, source_type, layer, quant, variant in _REPACKED_SELECTED_CONSUMERS:
            entries.append((layout, source_type, layer, quant, variant, None, None))
    for layout, source_type, layer, quant, variant, module, symbol in entries:
        for operation in operations:
            records.append(
                Qwen35GGUFOperationCoverage(
                    operation=operation,
                    role_class="moe_experts",
                    resident_layout=layout,
                    source_ggml_types=frozenset({source_type}),
                    rows_scope=(
                        "rows_1_8_row_local"
                        if operation != QWEN35_GGUF_OP_AR_PREFILL
                        else "prefill_rows"
                    ),
                    input_dtype="bf16",
                    output_dtype="bf16",
                    kernel_layer=layer,
                    kernel_quant=quant,
                    kernel_variant=variant,
                    consumer_module=module,
                    consumer_symbol=symbol,
                    strict_fallback=_SELECTED_EXPERT_STRICT_FALLBACK,
                    note="Selected-expert consumers require expert IDs and rank-3 metadata.",
                )
            )
    return records


def _embedding_records(
    operation: str,
) -> list[Qwen35GGUFOperationCoverage]:
    records = []
    for source_type, quant in (
        ("Q3_K", "gguf_q3_k"),
        ("Q4_K", "gguf_q4_k"),
        ("Q5_K", "gguf_q5_k"),
        ("Q6_K", "gguf_q6_k"),
        ("Q8_0", "gguf_q8_0"),
    ):
        records.append(
            Qwen35GGUFOperationCoverage(
                operation=operation,
                role_class="token_embedding",
                resident_layout=LAYOUT_RAW_GGUF,
                source_ggml_types=frozenset({source_type}),
                rows_scope="rows_any",
                input_dtype="token_ids",
                output_dtype="bf16",
                kernel_layer="embedding",
                kernel_quant=quant,
                kernel_variant="lookup_bf16_out",
                strict_fallback=None,
                note="Raw Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 lookup forwards rows to the kernel.",
            )
        )
    return records


def _certified_coverage() -> tuple[Qwen35GGUFOperationCoverage, ...]:
    row_ops = (QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS)
    records: list[Qwen35GGUFOperationCoverage] = []
    records.extend(
        _bf16_linear_records(
            (*row_ops, QWEN35_GGUF_OP_AR_PREFILL),
            "projection",
            note="Dense/recurrent/shared projections; Q3_K/IQ* dense layouts are absent until UD-U2..U5.",
        )
    )
    records.extend(
        _bf16_linear_records(
            (*row_ops, QWEN35_GGUF_OP_AR_PREFILL),
            "recurrent_alpha_beta",
            note="Row-local alpha/beta lanes; the native multirow BF16-pointer owner is a separate operation.",
        )
    )
    # The F32->BF16 contraction (raw-IQ manifests) lands alpha/beta on the
    # dense-BF16 resident for the regular row/prefill operations too.
    records.extend(
        _surface_records(
            (*row_ops, QWEN35_GGUF_OP_AR_PREFILL),
            "recurrent_alpha_beta",
            LAYOUT_DENSE_BF16,
            frozenset({"F32"}),
            activation=GGUF_ACTIVATION_BF16,
            output=GGUF_OUTPUT_BF16,
            note="Contracted F32 alpha/beta resident (dense BF16).",
        )
    )
    # Native multirow: identical projection coverage (the native route launches
    # the same layout-aware linear/pair consumers), but alpha/beta must be a
    # dense BF16 owner because the route passes allocation("raw") directly to
    # dense_gemv_out_bf16 (uint16_t* weight ABI).
    records.extend(
        _bf16_linear_records(
            (QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
            "projection",
        )
    )
    records.extend(
        _bf16_linear_records(
            (QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
            "recurrent_alpha_beta",
            layouts=(LAYOUT_DENSE_BF16,),
            rows_scope="rows_2_8_native_bf16_ptr",
            strict_fallback="layout-aware runtime linear dispatch (launch_gguf_linear)",
            note=(
                "The native multirow owner passes allocation('raw') to "
                "dense_gemv_out_bf16; only a dense BF16 resident is a valid "
                "BF16-pointer owner. Raw Q8_0 bytes and sole-T16 (no raw "
                "allocation) are refused."
            ),
        )
    )
    # The F32->BF16 contraction (raw-IQ manifests) also lands alpha/beta on
    # the dense-BF16 native owner; record that source mapping explicitly.
    records.extend(
        _surface_records(
            (QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
            "recurrent_alpha_beta",
            LAYOUT_DENSE_BF16,
            frozenset({"F32"}),
            activation=GGUF_ACTIVATION_BF16,
            output=GGUF_OUTPUT_BF16,
            rows_scope="rows_2_8_native_bf16_ptr",
            strict_fallback="layout-aware runtime linear dispatch (launch_gguf_linear)",
            note="Contracted F32 alpha/beta resident (dense BF16 native owner).",
        )
    )
    # Role participation is model metadata; pointer/scalar ABIs belong to
    # the shared concrete wrapper contracts, including the SSM norm composite.
    for operation in (*row_ops, QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS, QWEN35_GGUF_OP_AR_PREFILL):
        conv = conv_operation_contract(operation)
        gdn = resolve_gdn_operation_contract(operation)
        for role_class, consumer in (("norm", RMSNORM), ("gdn_scalar", gdn),
                                     ("gdn_norm", gdn), ("conv1d", conv)):
            records.append(Qwen35GGUFOperationCoverage(
                operation=operation, role_class=role_class,
                resident_layout=LAYOUT_DENSE_F32, source_ggml_types=frozenset({"F32"}),
                rows_scope="prefill_rows" if operation == QWEN35_GGUF_OP_AR_PREFILL else "rows_1_8_row_local",
                input_dtype=consumer.operands[0][1],
                output_dtype=next(dtype for _, dtype, access in consumer.operands if access == "write"),
                kernel_layer=consumer.layer, kernel_quant=consumer.quant, kernel_variant=consumer.variant,
                strict_fallback="registered F32-resident auxiliary wrapper",
            ))
    # MoE router: F32/BF16 residents consumed by the registered router
    # logits family (variant per resident quant).
    for operations, layout, source_types, quant in (
        ((*row_ops, QWEN35_GGUF_OP_AR_PREFILL), LAYOUT_DENSE_F32, frozenset({"F32"}), "f32"),
        ((*row_ops, QWEN35_GGUF_OP_AR_PREFILL), LAYOUT_DENSE_BF16, frozenset({"F32", "BF16"}), "bf16"),
    ):
        router = resolve_router_consumer_contract(layout, quant)
        for operation in operations:
            records.append(
                Qwen35GGUFOperationCoverage(
                    operation=operation,
                    role_class="moe_router",
                    resident_layout=layout,
                    source_ggml_types=source_types,
                    rows_scope=(
                        "rows_1_8_row_local"
                        if operation != QWEN35_GGUF_OP_AR_PREFILL
                        else "prefill_rows"
                    ),
                    input_dtype=router.operands[0][1],
                    output_dtype=router.operands[-1][1],
                    kernel_layer=router.layer,
                    kernel_quant=router.quant,
                    kernel_variant=router.variant,
                    strict_fallback="dense router gemv consumers",
                )
            )
    # MoE selected experts: unchanged manifests keep today's routes (raw and
    # decode-repack families), each format with its concrete consumer.
    moe_ops = (*row_ops, QWEN35_GGUF_OP_AR_PREFILL)
    records.extend(_selected_expert_records(moe_ops, raw=True))
    records.extend(_selected_expert_records(moe_ops, raw=False))
    # Embedding gather: raw compressed lookup forwards every row.  The dense
    # BF16 consumer resolves a singleton lookup (rows are dropped before the
    # launch), so there is deliberately NO certified dense-BF16 embedding
    # record: a multirow gather would silently read one token.
    records.extend(_embedding_records(QWEN35_GGUF_OP_EMBEDDING_LOOKUP))
    # F32 full-vocabulary logits: the ACTUAL default caller dtype is BF16
    # (the production lm-head callers pass scratch.norm BF16 without an
    # input override), so the certified rows are the (layout, bf16, f32)
    # dispatch rows.  dense_f32 deliberately has NO default record: the
    # dispatch table has no (dense_f32, bf16, f32) row — that combination is
    # certifiable only through the declared F32-input route below.
    lm_head_layouts = (
        (LAYOUT_Q4_K_PACK8, _Q4_PACK8_SOURCE_TYPES),
        (LAYOUT_RAW_GGUF, frozenset(RAW_LINEAR_SOURCE_QUANT_KEYS)),
        (LAYOUT_DENSE_BF16, _DENSE_BF16_LINEAR_SOURCE_TYPES),
        (LAYOUT_GGUF_Q6_K_T16, _Q6_T16_SOURCE_TYPES),
        (LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, _Q6_T16_SOURCE_TYPES),
    )
    for layout, source_types in lm_head_layouts:
        records.extend(
            _surface_records(
                (QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
                "lm_head",
                layout,
                source_types,
                activation=GGUF_ACTIVATION_BF16,
                output=GGUF_OUTPUT_F32,
                rows_scope="rows_1_8_row_local",
                strict_fallback="linear F32-output dispatch rows",
            )
        )
    # Native enqueue uses the same embedding, BF16-input full-logit head,
    # and selected/router boundaries. Bind them under the native operation,
    # with native row limits; do not borrow a caller-declared F32 head.
    records.extend(replace(record, operation=QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,
                           rows_scope="rows_2_8_native")
                   for record in tuple(records)
                   if record.operation in {QWEN35_GGUF_OP_EMBEDDING_LOOKUP, QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS}
                   or (record.operation == QWEN35_GGUF_OP_AR_DECODE_ROWS
                       and record.role_class in {"moe_router", "moe_experts"}))
    return tuple(records)


CERTIFIED_OPERATION_COVERAGE: tuple[Qwen35GGUFOperationCoverage, ...] = _certified_coverage()


def _f32_input_coverage() -> tuple[Qwen35GGUFOperationCoverage, ...]:
    """Records for operations the caller DECLARES will receive F32
    activations (input override present — the c1/verifier F32 routes).

    These records replace the BF16 input contract for the declared linear
    operands. A missing F32 row refuses; no caller conversion is invented.
    Projection outputs that the verifier explicitly casts back to BF16 name
    that adapter in the invocation, unlike the standalone F32-logits head.
    """

    records: list[Qwen35GGUFOperationCoverage] = []
    # (dense_f32, f32, f32): the registered F32-activation consumer used by
    # the verifier F32 linear-projection route and the F32-input lm-head.
    for operations, role_class in (
        ((QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,), "lm_head"),
        ((QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS, QWEN35_GGUF_OP_AR_PREFILL), "projection"),
        ((QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS), "recurrent_alpha_beta"),
    ):
        records.extend(_surface_records(
            operations, role_class, LAYOUT_DENSE_F32, frozenset({"F32"}),
            activation=GGUF_ACTIVATION_F32, output=GGUF_OUTPUT_F32,
            strict_fallback="dense_gemv/f32/f32_hidden_f32_out (F32-input route)",
            note="Supplied F32 input; verifier projections explicitly cast the output to BF16.",
        ))
    router = resolve_router_consumer_contract(LAYOUT_DENSE_F32, "f32", "f32")
    for operation in (QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS, QWEN35_GGUF_OP_AR_PREFILL):
        records.append(Qwen35GGUFOperationCoverage(
            operation, "moe_router", LAYOUT_DENSE_F32, frozenset({"F32"}),
            "prefill_rows" if operation == QWEN35_GGUF_OP_AR_PREFILL else "rows_1_8_row_local",
            router.operands[0][1], router.operands[-1][1],
            router.layer, router.quant, router.variant,
        ))
    # (q8_0_t16, f32, bf16): the GDN decode-output handoff route for a T16
    # ssm_out resident (policy-selected F32 activations).
    records.extend(_surface_records(
        (QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS, QWEN35_GGUF_OP_AR_PREFILL),
        "projection", LAYOUT_GGUF_Q8_0_T16, frozenset({"Q8_0"}),
        activation=GGUF_ACTIVATION_F32, output=GGUF_OUTPUT_BF16,
        strict_fallback="t16_gemv_decode_bf16_bf16_out (BF16-activation row)",
        note="GDN decode-output handoff with supplied F32 activations.",
    ))
    return tuple(records)


CERTIFIED_F32_INPUT_OPERATION_COVERAGE: tuple[Qwen35GGUFOperationCoverage, ...] = (
    _f32_input_coverage()
)


def _build_coverage_index(
    records: Iterable[Qwen35GGUFOperationCoverage],
) -> Mapping[tuple[str, str, str, str], Qwen35GGUFOperationCoverage]:
    """Index coverage records per concrete GGML source type.

    Two records may share ``(operation, role_class, resident_layout)`` when
    they certify different source formats with different registered consumers
    (for example raw rank-3 MoE experts: the ``gguf_q*_k`` raw family for
    Q3_K/Q5_K/Q6_K/IQ2_XS/IQ4_XS versus the ``gguf_iq3_xxs`` family for
    IQ3_XXS).  The lookup key therefore includes the source type; two
    DIFFERENT records claiming the same source type for the same
    (operation, role_class, layout) would be an ambiguous certification and
    are rejected here instead of silently last-write-wins.
    """

    index: dict[tuple[str, str, str, str], Qwen35GGUFOperationCoverage] = {}
    for record in records:
        for source_type in sorted(record.source_ggml_types):
            key = (
                record.operation,
                record.role_class,
                record.resident_layout,
                source_type,
            )
            existing = index.get(key)
            if existing is not None:
                if existing != record:
                    raise ValueError(
                        "ambiguous certified coverage records for "
                        f"(operation={key[0]!r}, role_class={key[1]!r}, "
                        f"resident_layout={key[2]!r}, source={key[3]!r}): "
                        f"{existing!r} conflicts with {record!r}"
                    )
                continue
            index[key] = record
    return index


_COVERAGE_INDEX: Mapping[tuple[str, str, str, str], Qwen35GGUFOperationCoverage] = (
    _build_coverage_index(CERTIFIED_OPERATION_COVERAGE)
)

_COVERAGE_F32_INPUT_INDEX: Mapping[tuple[str, str, str, str], Qwen35GGUFOperationCoverage] = (
    _build_coverage_index(CERTIFIED_F32_INPUT_OPERATION_COVERAGE)
)

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
    "ssm_norm": "gdn_norm",
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
        {"projection", "recurrent_alpha_beta", "norm", "gdn_norm", "gdn_scalar", "conv1d", "moe_router", "moe_experts"}
    ),
    QWEN35_GGUF_OP_AR_DECODE_ROWS: frozenset(
        {"projection", "recurrent_alpha_beta", "norm", "gdn_norm", "gdn_scalar", "conv1d", "moe_router", "moe_experts"}
    ),
    QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS: NATIVE_EXECUTION_ROLE_CLASSES,
    QWEN35_GGUF_OP_AR_PREFILL: frozenset(
        {"projection", "recurrent_alpha_beta", "norm", "gdn_norm", "gdn_scalar", "conv1d", "moe_router", "moe_experts"}
    ),
    QWEN35_GGUF_OP_EMBEDDING_LOOKUP: frozenset({"token_embedding"}),
    QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS: frozenset({"lm_head"}),
    QWEN35_GGUF_OP_MTP_NEXTN_DRAFT: frozenset(
        {"projection", "recurrent_alpha_beta", "norm", "gdn_scalar", "conv1d", "token_embedding", "lm_head"}
    ),
}


def _coverage_for(
    operation: str,
    role_class: str,
    layout: str,
    source_type: str,
    *,
    f32_input: bool = False,
) -> Qwen35GGUFOperationCoverage | None:
    """The certified record for one concrete slot contract, or ``None``.

    ``f32_input`` selects the supplied F32 linear operand contract, never
    the default BF16 record. Auxiliary/composite operands retain their named
    mixed ABI; this declaration does not turn weights or state into F32.
    """

    key = (operation, role_class, layout, source_type)
    # A supplied F32 activation is not a preference. No cast is performed
    # by launch_gguf_linear; never certify the nearby BF16 row.
    if f32_input and role_class in {"projection", "recurrent_alpha_beta", "lm_head", "moe_router", "moe_experts"}:
        return _COVERAGE_F32_INPUT_INDEX.get(key)
    return _COVERAGE_INDEX.get(key)


# Concrete hardware backend keys that own GGUF consumer metadata. This is the
# same registry surface ``backend_package_capability`` /
# ``load_backend_kernel_package`` reject unknown backends against; checking it
# here is pure metadata (no backend-package import, no device query).
_KNOWN_HARDWARE_BACKEND_KEYS: frozenset[str] = frozenset(
    (*HIP_BACKEND_TARGET_ARCH, *CUDA_BACKEND_TARGET_ARCH)
)


# ---------------------------------------------------------------------------
# Canonical planned-resident identity (records + digest)
# ---------------------------------------------------------------------------

# Digest tag: a change to the canonical record format invalidates every
# previously recorded contract (fail closed), never silently compares.
_RESIDENT_PLAN_DIGEST_TAG = "qwen35-gguf-resident-plan-v1"
_RECORD_SLOT_PREFIX = "slot="


def qwen35_gguf_planned_weight_record(spec: Qwen35GGUFWeightSpec) -> str:
    """Canonical single-line identity of one planned resident weight spec.

    The line is a pure function of the planned spec and names everything the
    actual consumers bind to: the logical slot path, the source tensor
    identity (name, logical shape, GGML storage type), the resident layout,
    the per-tensor kernel quant key, the allocation names, the planned
    per-allocation byte counts (the exact metadata formula the loader
    allocates with, which also pins resident element sizing), and the
    sidecar layouts. It deliberately contains no environment-variable names,
    no caller kwargs, no reprs, no pointers, and no timestamps: two plans
    with identical residents produce identical lines regardless of how the
    options were spelled or ordered.
    """

    source = spec.source
    planned_nbytes = planned_qwen35_gguf_weight_allocation_nbytes(spec)
    fields = (
        f"slot={spec.slot_path}",
        f"source={source.name}",
        f"shape={','.join(str(int(dim)) for dim in source.shape)}",
        f"source_type={source.ggml_type_name}",
        f"layout={spec.layout}",
        f"quant_key={spec.quant_key}",
        f"allocations={','.join(spec.allocation_names)}",
        f"planned_nbytes={','.join(f'{name}:{int(nbytes)}' for name, nbytes in planned_nbytes)}",
        f"sidecars={','.join(spec.sidecar_layouts)}",
    )
    return "\t".join(fields)


def qwen35_gguf_planned_weight_digest(records: Iterable[str]) -> str:
    """Deterministic sha256 over a canonical planned-resident record set.

    Records are deduplicated and sorted before hashing, and the record count
    is length-prefixed, so the digest is stable across processes and input
    ordering and cannot collide across different-sized record sets.
    """

    ordered = tuple(sorted(dict.fromkeys(str(record) for record in records)))
    digest = hashlib.sha256()
    digest.update(_RESIDENT_PLAN_DIGEST_TAG.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(len(ordered)).encode("ascii"))
    digest.update(b"\x00")
    for record in ordered:
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _planned_weight_record_slot(record: str) -> str:
    """Logical slot path of one canonical record; empty for malformed lines."""

    if not record.startswith(_RECORD_SLOT_PREFIX):
        return ""
    return record[len(_RECORD_SLOT_PREFIX) :].split("\t", 1)[0]


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
class Qwen35GGUFPlanContract:
    """The exact effective plan/operation contract a preflight checked.

    This is the certificate's scope statement: which operations, which slots,
    and which effective plan flags (after environment/veto resolution and
    contraction inference) the admission verdict was computed under. Beyond
    the flags, the contract binds the ACTUAL planned residents: a canonical
    sorted record per checked slot (logical slot, source identity, resident
    layout, quant key, allocation names, planned per-allocation byte counts,
    sidecar layouts) plus a deterministic digest over those records. The
    digest is always re-derived from the stored records, so a contract can
    never carry a digest that disagrees with its own record set, and two
    plans whose env-resolved layouts differ can never compare equal even
    when every caller kwarg matches. Two contracts are interchangeable only
    when their recorded plans are equal — a contraction-enabled certificate
    is not reusable on an uncontracted plan, a repack-vetoed plan is not the
    certified plan, and an X8-env certificate is not reusable on a T16 plan.

    The contract also binds qualification COMPLETENESS:
    ``required_plan_slots`` enumerates every slot the preflight tried to
    qualify for its (operations, slot scope) and ``operation_scope_refusals``
    records operation-level scope refusals.  A contract is
    authorization-capable (:meth:`is_complete`) only when its records cover
    exactly the required slots and nothing was refused — a preflight report
    that refused anything (planner, consumer, unknown role, unknown filter
    entry, operation scope) carries a structurally incomplete contract and
    can never mint or verify as authorization, no matter how many slots
    qualified.  Partial records stay on refused reports for debugging; they
    just cannot authorize.
    """

    operations: tuple[str, ...]
    slot_filter: tuple[str, ...] | None
    decode_repack: bool
    repack_veto: bool
    contract_f32_linear: bool
    dense_q4_t16: bool
    dense_q4_qmicro_t16_gate_up: bool
    dense_q4_t16_attn_q_08b: bool
    dense_q5_t16_ssm_out: bool
    dense_q5_raw_mmq_ssm_out: bool
    dense_q5_qmicro_planar_ssm_out: bool
    dense_q5_t16_ssm_out_08b: bool
    dense_q5_t16_qkv: bool
    dense_q5_t16_h5120: bool
    dense_q6_qmicro_planar: bool
    dense_q6_qmicro_planar_excluded_slots: tuple[str, ...]
    backend: str = ""
    invocations: tuple[InvocationContract, ...] = ()
    required_invocations: tuple[tuple[str, str], ...] = ()
    selected_invocations: tuple[BoundSelectedCall, ...] = ()
    required_selected_intents: tuple[SelectedCallIntent, ...] = ()
    f32_input_operations: tuple[str, ...] = ()
    resident_plan_records: tuple[str, ...] = ()
    required_plan_slots: tuple[str, ...] = ()
    operation_scope_refusals: tuple[str, ...] = ()
    resident_plan_digest: str = ""

    def __post_init__(self) -> None:
        # Canonicalize: deduplicate + sort records and the accounting sets;
        # the digest is always derived from the stored records (never
        # caller-asserted).
        object.__setattr__(self, "invocations", tuple(sorted(self.invocations, key=lambda item: item.canonical_record())))
        object.__setattr__(self, "required_invocations", tuple(sorted(set(self.required_invocations))))
        records = tuple(sorted(dict.fromkeys(str(record) for record in self.resident_plan_records)))
        object.__setattr__(self, "resident_plan_records", records)
        object.__setattr__(
            self,
            "resident_plan_digest",
            qwen35_gguf_planned_weight_digest(records),
        )
        object.__setattr__(
            self,
            "f32_input_operations",
            tuple(
                sorted(dict.fromkeys(str(operation) for operation in self.f32_input_operations))
            ),
        )
        object.__setattr__(
            self,
            "required_plan_slots",
            tuple(sorted(dict.fromkeys(str(slot) for slot in self.required_plan_slots))),
        )
        object.__setattr__(
            self,
            "operation_scope_refusals",
            tuple(
                sorted(dict.fromkeys(str(operation) for operation in self.operation_scope_refusals))
            ),
        )

    def is_complete(self) -> bool:
        """Whether this contract proves COMPLETE successful qualification.

        This is the authorization-capability statement.  True only when the
        contract binds a qualification attempt that fully succeeded over its
        claimed scope:

        - records are well-formed, slot-unique, and digested from
          themselves;
        - the recorded residents cover EXACTLY ``required_plan_slots`` —
          every slot the preflight tried to qualify for its (operations,
          slot scope) is accounted: planner-refused, consumer-unqualified,
          unknown-role, and unknown-filter-entry slots are required but
          never recorded, so any refused preflight yields an incomplete
          contract and can never authorize; slots no requested operation
          touches are in neither set;
        - no operation was refused at the operation scope gate (for example
          MTP draft on an AR-only preset);
        - a filtered scope accounts only for slots inside the filter.

        Hand-built, legacy, or hollow contracts default to incomplete
        (``required_plan_slots=()`` with records, or records without their
        required slots) and fail closed.  The accounting sets are preflight
        enumeration inputs, not values derived from the successful records,
        so the completeness proof is not circular.
        """

        slots = self.resident_plan_slots
        identities = [(item.slot, item.operation) for item in self.invocations]
        if not self.backend or any(item.backend != self.backend for item in self.invocations):
            return False
        if any(item.operation in self.f32_input_operations
               and item.consumer.layer in {"linear", "dense_gemv", "router_logits"}
               and item.consumer.operands[0][1] != "f32" for item in self.invocations):
            return False
        if len(set(identities)) != len(identities) or set(identities) != set(self.required_invocations):
            return False
        if any(slot not in slots or op not in self.operations for slot, op in identities):
            return False
        selected_slots = set()
        if len(self.selected_invocations) != len(self.required_selected_intents):
            return False
        if {item.intent for item in self.selected_invocations} != set(self.required_selected_intents):
            return False
        by_slot = dict(zip(slots, self.resident_plan_records))
        for call in self.selected_invocations:
            if call.backend != self.backend or call.intent.operation not in self.operations:
                return False
            if tuple(slot for slot, _ in call.weight_bindings) != call.intent.weight_slots:
                return False
            if any(by_slot.get(slot) != record for slot, record in call.weight_bindings):
                return False
            selected_slots.update(call.intent.weight_slots)
        if set(slots) != {slot for slot, _ in identities} | selected_slots:
            return False
        if any(not slot for slot in slots):
            return False
        if len(set(slots)) != len(slots):
            return False
        if (
            self.resident_plan_digest
            != qwen35_gguf_planned_weight_digest(self.resident_plan_records)
        ):
            return False
        required = tuple(self.required_plan_slots)
        if any(not slot for slot in required):
            return False
        if set(slots) != set(required):
            return False
        if self.operation_scope_refusals:
            return False
        if self.slot_filter is not None and not set(required) <= {
            str(slot) for slot in self.slot_filter
        }:
            return False
        return True

    @property
    def invocation_digest(self) -> str:
        records = [item.canonical_record() for item in (*self.invocations, *self.selected_invocations)]
        payload = "qwen35-gguf-invocations-v1\n" + "\n".join(sorted(records))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def resident_plan_slots(self) -> tuple[str, ...]:
        """Logical slot paths of the recorded residents, in record order."""

        return tuple(
            _planned_weight_record_slot(record)
            for record in self.resident_plan_records
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "operations": list(self.operations),
            "slot_filter": None if self.slot_filter is None else list(self.slot_filter),
            "decode_repack": self.decode_repack,
            "repack_veto": self.repack_veto,
            "contract_f32_linear": self.contract_f32_linear,
            "dense_q4_t16": self.dense_q4_t16,
            "dense_q4_qmicro_t16_gate_up": self.dense_q4_qmicro_t16_gate_up,
            "dense_q4_t16_attn_q_08b": self.dense_q4_t16_attn_q_08b,
            "dense_q5_t16_ssm_out": self.dense_q5_t16_ssm_out,
            "dense_q5_raw_mmq_ssm_out": self.dense_q5_raw_mmq_ssm_out,
            "dense_q5_qmicro_planar_ssm_out": self.dense_q5_qmicro_planar_ssm_out,
            "dense_q5_t16_ssm_out_08b": self.dense_q5_t16_ssm_out_08b,
            "dense_q5_t16_qkv": self.dense_q5_t16_qkv,
            "dense_q5_t16_h5120": self.dense_q5_t16_h5120,
            "dense_q6_qmicro_planar": self.dense_q6_qmicro_planar,
            "dense_q6_qmicro_planar_excluded_slots": list(
                self.dense_q6_qmicro_planar_excluded_slots
            ),
            "backend": self.backend,
            "invocation_digest": self.invocation_digest,
            "invocations": [item.canonical_record() for item in self.invocations],
            "required_invocations": [list(item) for item in self.required_invocations],
            "selected_invocations": [item.canonical_record() for item in self.selected_invocations],
            "required_selected_intents": [asdict(item) for item in self.required_selected_intents],
            "f32_input_operations": list(self.f32_input_operations),
            "resident_plan_digest": self.resident_plan_digest,
            "resident_plan_record_count": len(self.resident_plan_records),
            "required_plan_slots": list(self.required_plan_slots),
            "operation_scope_refusals": list(self.operation_scope_refusals),
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
    slot_filter: tuple[str, ...] | None = None
    plan_contract: Qwen35GGUFPlanContract | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "preset_key": self.preset_key,
            "manifest_fingerprint": self.manifest_fingerprint,
            "backend": self.backend,
            "operations": list(self.operations),
            "file_type_stamp": self.file_type_stamp,
            "covered_slots": self.covered_slots,
            "slot_filter": None if self.slot_filter is None else list(self.slot_filter),
            "plan_contract": (
                None if self.plan_contract is None else self.plan_contract.as_dict()
            ),
        }


def certificate_matches_artifact_identity(
    certificate: Qwen35GGUFAdmissionCertificate,
    *,
    manifest_fingerprint: str,
    backend: str | None = None,
) -> bool:
    """Source-identity-only binding check. This does NOT authorize operations.

    True when the certificate was minted for exactly this role-manifest
    fingerprint (and backend, when given). This says "the certificate and the
    artifact are the same source identity" and nothing more: it deliberately
    inspects no resident plan, no requested operations, and no slot scope, so
    a True result can never be read as permission to run anything. Operation
    approval must go through :func:`certificate_covers_artifact`, which
    verifies the intended plan contract against the certified plan.
    """

    if certificate.manifest_fingerprint != str(manifest_fingerprint):
        return False
    if backend is not None and certificate.backend != str(backend):
        return False
    return True


def certificate_covers_artifact(
    certificate: Qwen35GGUFAdmissionCertificate,
    *,
    manifest_fingerprint: str,
    plan_contract: Qwen35GGUFPlanContract,
    backend: str | None = None,
    operations: Iterable[str] | None = None,
    slot_filter: Iterable[str] | None = None,
) -> bool:
    """Return whether a certificate authorizes the intended use of the artifact.

    ``plan_contract`` is REQUIRED: the caller's intended effective plan
    contract, taken from a fresh :func:`preflight_qwen35_gguf_artifact` call
    over the artifact/plan about to run. Both the certificate's recorded
    contract and the intended contract must be COMPLETE
    (:meth:`Qwen35GGUFPlanContract.is_complete`): a preflight that refused
    anything records only the successfully qualified slots, and that partial
    record set must never verify as authorization — so a refused report's
    contract fails closed here no matter how many of its records match the
    certified plan. A certificate whose own plan metadata is missing (legacy
    or hand-built) fails closed, as does any hollow or hand-built contract.

    A plain-artifact certificate never covers a UD manifest (different
    fingerprint) and vice versa, even when the file-type stamps match.

    ``slot_filter`` names the slots the caller intends to use now; the
    default ``None`` means the full artifact is intended. It must agree with
    the intended contract's own recorded scope, or the request fails closed.
    A full-artifact certificate (``slot_filter=None`` recorded at preflight
    time) covers any subset whose per-slot resident records verify against
    the certified plan; a certificate produced under a slot filter only
    covers uses within that exact checked subset — a one-slot debug
    certificate never reads as full-artifact coverage, including under an
    unspecified check. Narrowing is preserved; enlargement beyond the
    certified subset is refused. A named nonempty scope whose intended
    contract verified no residents for it authorizes nothing.

    ``operations`` names the operations the caller intends to run. When
    omitted, the intended operation set defaults to the intended contract's
    own checked operations — the WHOLE intended contract must be certified,
    so a c1-only certificate never covers a prefill or native-rows intent
    even when the planned residents are identical. When given explicitly,
    the set must be non-empty (an unnamed operation set authorizes nothing)
    and every requested operation must be in the certificate's certified
    operation set, in the certificate's recorded contract's checked set, AND
    in the intended contract's own checked set — explicit narrowing must
    belong to both qualifications. Planned resident bytes establish
    allocations, never activation/output dtype or row-operation
    qualification: identical residents cannot upgrade a c1/prefill
    certificate to the native multirow operation.
    """

    if not certificate_matches_artifact_identity(
        certificate,
        manifest_fingerprint=manifest_fingerprint,
        backend=backend,
    ):
        return False
    recorded = certificate.plan_contract
    if recorded is None:
        # A certificate without resident-plan identity cannot verify any
        # intended plan: fail closed instead of authorizing operations.
        return False
    if not plan_contract.is_complete():
        # A refused (or hollow, or hand-built) intended preflight can never
        # supply an authorization-capable contract, even when its partial
        # records are a subset of the certified ones.
        return False
    if not recorded.is_complete():
        return False
    intended = plan_contract
    if certificate.backend != recorded.backend or recorded.backend != intended.backend:
        return False
    if backend is not None and backend != intended.backend:
        return False
    requested_operations = (
        tuple(intended.operations)
        if operations is None
        else tuple(dict.fromkeys(str(operation) for operation in operations))
    )
    if not requested_operations:
        # An explicitly empty operation set names nothing and therefore
        # authorizes nothing (fail closed, never a blanket yes).
        return False
    if any(
        operation not in certificate.operations for operation in requested_operations
    ):
        return False
    if any(operation not in recorded.operations for operation in requested_operations):
        return False
    if any(operation not in intended.operations for operation in requested_operations):
        return False
    # Match entire effective invocation records for the narrowed operation
    # scope. Completeness was checked BEFORE narrowing, including failures.
    certified_invocations = {item.canonical_record() for item in recorded.invocations}
    if any(item.canonical_record() not in certified_invocations
           for item in intended.invocations if item.operation in requested_operations):
        return False
    selected_records = {item.canonical_record() for item in recorded.selected_invocations}
    if any(item.canonical_record() not in selected_records
           for item in intended.selected_invocations if item.intent.operation in requested_operations):
        return False
    requested_slots = (
        None if slot_filter is None else tuple(sorted({str(slot) for slot in slot_filter}))
    )
    if requested_slots != intended.slot_filter:
        # The call's slot scope and the intended contract's recorded scope
        # disagree: refuse instead of guessing which one was meant.
        return False
    intended_slots = intended.slot_filter
    certified_slots = recorded.slot_filter
    if intended_slots is None:
        # The full artifact is intended: a slot-filtered certificate never
        # covers it, and every resident the intended contract verified must
        # have been verified identically by the certificate's plan.
        if certified_slots is not None:
            return False
        if not intended.resident_plan_records:
            return False
        certified_records = set(recorded.resident_plan_records)
        return all(
            record in certified_records for record in intended.resident_plan_records
        )
    intended_slot_set = set(intended_slots)
    if certified_slots is not None and not intended_slot_set <= set(certified_slots):
        # Enlargement beyond the certified subset is never covered.
        return False
    if intended_slot_set and not intended.resident_plan_records:
        # A named nonempty scope the intended contract verified no residents
        # for authorizes nothing (vacuous coverage is not coverage).
        return False
    certified_by_slot = {
        _planned_weight_record_slot(record): record
        for record in recorded.resident_plan_records
    }
    for record in intended.resident_plan_records:
        slot = _planned_weight_record_slot(record)
        if certified_by_slot.get(slot) != record:
            # The intended resident for this slot is not the certified one.
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
    slot_filter: tuple[str, ...] | None = None
    plan_contract: Qwen35GGUFPlanContract | None = None

    @property
    def supported(self) -> bool:
        return not self.unsupported

    def certificate(self) -> Qwen35GGUFAdmissionCertificate:
        if not self.supported:
            raise Qwen35GGUFAdmissionError(
                "cannot certify an artifact with unsupported operations; see "
                "Qwen35GGUFAdmissionReport.unsupported"
            )
        if (self.plan_contract is None or not self.plan_contract.is_complete()
                or self.backend != self.plan_contract.backend
                or set(self.requested_operations) != set(self.plan_contract.operations)):
            # Mint-boundary invariant: only a COMPLETE qualification contract
            # can be certified. For real preflight reports this is equivalent
            # to ``supported`` (every refusal path also marks the contract
            # incomplete); the guard exists so the two can never drift apart.
            raise Qwen35GGUFAdmissionError(
                "cannot certify an artifact whose admission plan contract is "
                "not complete (required-slot accounting or operation-scope "
                "refusals failed); see Qwen35GGUFAdmissionReport.unsupported"
            )
        return Qwen35GGUFAdmissionCertificate(
            preset_key=None if self.preset is None else self.preset.preset_key,
            manifest_fingerprint=self.manifest_fingerprint,
            backend=self.backend,
            operations=self.requested_operations,
            file_type_stamp=self.file_type_stamp,
            covered_slots=self.covered_slots,
            slot_filter=self.slot_filter,
            plan_contract=self.plan_contract,
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
    f32_input_operations: Iterable[str] = (),
    selected_call_intents: Iterable[SelectedCallIntent] | None = None,
    recurrent_state_dtype: str = "f32",
    gdn_force_bf16: bool = False,
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
    preflight's selected-load scope (the loader's test/debug ``selected_slots``
    hook). Unknown filter entries refuse. Selected calls are independent:
    ``selected_call_intents=None`` uses the shared full-model caller plan and
    a missing filtered partner refuses, never turns a pair into a singleton.
    Explicit single-call intents support standalone diagnostics; an explicit
    empty tuple is appropriate for a non-expert diagnostic, not permission to
    omit dependencies of an expert call. ``f32_input_operations`` declares
    actually supplied linear/router activations; no BF16 fallback is invented.
    ``recurrent_state_dtype`` and ``gdn_force_bf16`` describe existing caller
    state/registered-cast choices, not permission to select numerical profiles.
    The row-local ssm_out handoff resolves its real F32 input or executed cast
    through the same resolver as production. The returned report always
    carries a plan contract; it is authorization-capable
    (:meth:`Qwen35GGUFPlanContract.is_complete`) exactly when nothing was
    refused, and :meth:`Qwen35GGUFAdmissionReport.certificate` refuses to
    mint otherwise.
    """

    resolve_gdn_segments_contract(recurrent_state_dtype)  # storage declaration, not profile permission
    if decode_repack is None:
        decode_repack = gguf_decode_repack_enabled(None)
    else:
        decode_repack = bool(decode_repack)

    backend_key = str(backend)
    if backend_key not in _KNOWN_HARDWARE_BACKEND_KEYS:
        # A syntactically valid request is not registration evidence: only
        # concrete registered hardware backend keys own GGUF consumer
        # metadata, so an unknown backend can never yield a positive
        # certificate. Fail closed before any per-slot planning.
        raise Qwen35GGUFAdmissionError(
            "unknown GGUF admission backend "
            f"{backend_key!r}; expected one of: "
            f"{', '.join(sorted(_KNOWN_HARDWARE_BACKEND_KEYS))}. No certified "
            "consumer metadata exists for unregistered backends."
        )

    requested = tuple(dict.fromkeys(str(operation) for operation in operations))
    unknown_ops = tuple(op for op in requested if op not in _KNOWN_OPERATIONS)
    if unknown_ops:
        raise Qwen35GGUFAdmissionError(
            "unknown requested GGUF operations (fail closed; certified "
            f"operations: {sorted(_KNOWN_OPERATIONS)}): {list(unknown_ops)}"
        )
    f32_input = tuple(dict.fromkeys(str(operation) for operation in f32_input_operations))
    invalid_f32_input = tuple(
        operation
        for operation in f32_input
        if operation not in _KNOWN_OPERATIONS or operation not in requested
    )
    if invalid_f32_input:
        # An F32-input declaration must name an actually requested, known
        # operation: a stray declaration is a caller error, not a silent
        # no-op that would widen someone else's coverage.
        raise Qwen35GGUFAdmissionError(
            "f32_input_operations must name requested known GGUF operations "
            f"(requested: {list(requested)}): {list(invalid_f32_input)}"
        )
    # F3: a known target-arch name is not consumer registration.  The
    # backend's source-declared GGUF consumer layers (parity-tested against
    # the real registration surface) gate every positive record; a backend
    # that declares none (for example the cuda_sm120a scaffold) refuses
    # every GGUF operation slot with an explicit reason.
    declared_consumer_layers = backend_gguf_consumer_layers(backend_key)
    manifest = build_qwen35_gguf_role_manifest(model_map, nextn_map=nextn_map)
    preset = resolve_qwen35_gguf_artifact_preset(
        model_map,
        nextn_map=nextn_map,
        file_type_stamp=file_type_stamp,
    )
    unsupported: list[Qwen35GGUFUnsupportedOperation] = []
    qualified: dict[
        tuple[str, str, str, str], Qwen35GGUFOperationCoverage
    ] = {}
    covered_slots = 0
    # Canonical records of the ACTUAL planned residents for every checked and
    # qualified slot. This — not the caller's kwargs or env names — is what
    # the plan contract (and therefore the certificate) binds to.
    resident_plan_records: list[str] = []
    invocations: list[InvocationContract] = []
    required_invocations: set[tuple[str, str]] = set()
    selected_specs: dict[str, Qwen35GGUFWeightSpec] = {}
    selected_required: set[tuple[str, str]] = set()
    # Authorization accounting: every slot the preflight TRIES to qualify
    # (participating or refused) lands in required_plan_slots; only fully
    # qualified slots earn a record. is_complete() requires the two sets to
    # agree, so a refused preflight can never produce an authorization-
    # capable contract. Operation-level scope refusals (MTP draft gating) are
    # recorded separately because they attach to no slot.
    required_plan_slots: set[str] = set()
    operation_scope_refusals: list[str] = []

    # Scope gate: MTP draft operations require an explicitly certified MTP
    # scope.  No preset (plain lane) or an AR-only UD preset refuses the whole
    # operation; automatic MTP inheritance from plain-file certification is
    # impossible because the scope binds to the manifest fingerprint.
    if QWEN35_GGUF_OP_MTP_NEXTN_DRAFT in requested:
        if preset is None or not preset.scope_certified(GGUF_PRESET_SCOPE_MTP):
            operation_scope_refusals.append(QWEN35_GGUF_OP_MTP_NEXTN_DRAFT)
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
    f32_input_set = frozenset(f32_input)
    allowed_slots = None if slot_filter is None else {str(slot) for slot in slot_filter}
    effective_slot_filter = (
        None if allowed_slots is None else tuple(sorted(allowed_slots))
    )
    if allowed_slots is not None:
        # A filter entry naming no slot in the (operation-relevant) map is a
        # caller error, not a silent no-op: refuse it and account it as a
        # required-but-unrecorded scope entry so the contract stays
        # incomplete and can never authorize that scope.
        known_slot_paths = {str(slot_path) for slot_path, _ in slot_tensors}
        for entry in sorted(allowed_slots - known_slot_paths):
            required_plan_slots.add(entry)
            unsupported.append(
                Qwen35GGUFUnsupportedOperation(
                    slot_path=entry,
                    role_class="",
                    operation="-",
                    source_ggml_type="-",
                    resident_layout=None,
                    stage="scope_refused",
                    reason=(
                        "slot_filter entry names no slot in the artifact map "
                        "for the requested operations"
                    ),
                )
            )
    for slot_path, tensor in slot_tensors:
        if allowed_slots is not None and str(slot_path) not in allowed_slots:
            continue
        role_class = _role_class_for_slot(str(slot_path).rsplit(".", 1)[-1])
        if not role_class:
            required_plan_slots.add(str(slot_path))
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
        # Every selected resident will be materialized, even when none of
        # the requested operations consumes it. Check resident prerequisites
        # independently of operation qualification (including debug subsets).
        if applicable_ops:
            required_plan_slots.add(str(slot_path))
            target = selected_required if role_class == "moe_experts" else required_invocations
            target.update((str(slot_path), op) for op in applicable_ops)
        try:
            spec = plan_qwen35_gguf_weight_spec(
                str(slot_path),
                tensor,
                contract_f32_linear=contraction,
                **plan_flags,
            )
            # Shape validity is NOT a consequence of computable bytes:
            # byte-neutral T16/X8 repacks still impose rank/tile constraints.
            validate_qwen35_gguf_resident_prerequisites(spec)
            planned_qwen35_gguf_weight_allocation_nbytes(spec)
        except ValueError as error:
            required_plan_slots.add(str(slot_path))
            for operation in applicable_ops or ("-",):
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
        if not applicable_ops:
            continue
        slot_supported = True
        for operation in applicable_ops:
            # Source-type-aware lookup: different expert/projection formats on
            # the same resident layout certify different registered consumers,
            # so the certified record must match this slot's concrete GGML
            # storage type, not whichever record was written last.
            supplied_f32 = operation in f32_input_set
            incompatible_handoff = False
            if str(slot_path).endswith(".ssm_out"):
                activation = (resolve_gdn_output_handoff(spec.layout, force_bf16=gdn_force_bf16).activation
                              if operation in {QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS} else "bf16")
                incompatible_handoff = supplied_f32 and activation != "f32"
                supplied_f32 = supplied_f32 or activation == "f32"
            record = None if incompatible_handoff else _coverage_for(
                operation, role_class, spec.layout, tensor.ggml_type_name,
                f32_input=supplied_f32,
            )
            if (
                record is not None
                and record.kernel_layer not in declared_consumer_layers
            ):
                # F3: the certified consumer family exists, but THIS backend
                # does not register it (a known target-arch name alone is
                # not registration).  Refuse with the concrete missing
                # consumer named instead of certifying a foreign backend's
                # consumer.
                slot_supported = False
                unsupported.append(
                    Qwen35GGUFUnsupportedOperation(
                        slot_path=str(slot_path),
                        role_class=role_class,
                        operation=operation,
                        source_ggml_type=tensor.ggml_type_name,
                        resident_layout=spec.layout,
                        stage="consumer_unqualified",
                        reason=(
                            f"backend {backend_key!r} registers no GGUF consumer "
                            f"for layer {record.kernel_layer!r} (declared GGUF "
                            f"consumer layers: "
                            f"{sorted(declared_consumer_layers)}); a known "
                            "target-arch name alone is not consumer "
                            "registration"
                        ),
                    )
                )
                continue
            if record is None:
                slot_supported = False
                if incompatible_handoff:
                    reason = ("ssm_out caller supplies BF16 after its GDN handoff, not the declared F32 input; "
                              "no alternative operand or conversion is inferred")
                elif role_class == "token_embedding" and spec.layout == LAYOUT_DENSE_BF16:
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
                elif (
                    role_class == "lm_head"
                    and operation == QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS
                    and spec.layout == LAYOUT_DENSE_F32
                ):
                    reason = (
                        "the actual lm-head F32-logits caller supplies BF16 "
                        "activations without an input override, and the runtime "
                        "linear dispatch has no (dense_f32, activation=bf16, "
                        "output=f32) row; the registered dense-F32 consumer is "
                        "the F32-activation row (dense_gemv/f32/"
                        "f32_hidden_f32_out), valid only with a declared F32 "
                        "input override (f32_input_operations)"
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
            if role_class == "moe_experts":
                selected_specs[str(slot_path)] = spec
            else:
                try:
                    invocations.append(record.invocation(spec, backend=backend_key, config=model_map.config,
                                                         recurrent_state_dtype=recurrent_state_dtype,
                                                         gdn_force_bf16=gdn_force_bf16))
                except ValueError as error:
                    slot_supported = False
                    unsupported.append(Qwen35GGUFUnsupportedOperation(
                        str(slot_path), role_class, operation, tensor.ggml_type_name,
                        spec.layout, "consumer_unqualified", str(error)))
                    continue
            qualified[
                (operation, role_class, spec.layout, tensor.ggml_type_name)
            ] = record
        if slot_supported:
            covered_slots += 1
            resident_plan_records.append(qwen35_gguf_planned_weight_record(spec))

    # Resolve the full-model selected topology independently of selected-load
    # filtering. Missing partner owners are refusals, never singleton fallbacks.
    if selected_call_intents is None:
        all_expert_quants = {}
        for path, tensor in slot_tensors:
            if _role_class_for_slot(path) != "moe_experts":
                continue
            try:
                planned = plan_qwen35_gguf_weight_spec(path, tensor, contract_f32_linear=contraction, **plan_flags)
                all_expert_quants[path] = planned.quant_key
            except ValueError:
                all_expert_quants[path] = "unqualified"
        lanes = int(model_map.config.expert_used_count)
        if all_expert_quants and any(op in {QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS, QWEN35_GGUF_OP_AR_PREFILL, QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS} for op in checked_ops) and lanes <= 0:
            operation_scope_refusals.extend(checked_ops)
            unsupported.append(Qwen35GGUFUnsupportedOperation(
                "model", "moe_experts", checked_ops[0], "-", None, "scope_refused",
                "full-model selected caller requires positive expert_used_count"))
        selected_intents = default_selected_call_intents(all_expert_quants, checked_ops, lanes_per_token=max(1, lanes))
    else:
        selected_intents = tuple(selected_call_intents)
    selected_bound = []
    selected_covered = set()
    resident_by_slot = {_planned_weight_record_slot(record): record for record in resident_plan_records}
    for call in selected_intents:
        try:
            if call.operation not in checked_ops:
                raise ValueError("selected call intent must name a requested operation")
            uses = {(slot, call.operation) for slot in call.weight_slots}
            if selected_covered & uses:
                raise ValueError("overlapping selected call intents are not a single qualification")
            bound = bind_selected_call(call, selected_specs, resident_by_slot, backend=backend_key,
                                       row_limits=OPERATION_ROW_LIMITS[call.operation])
            selected_bound.append(bound)
            selected_covered.update(uses)
        except ValueError as error:
            operation_scope_refusals.append(call.operation)
            unsupported.append(Qwen35GGUFUnsupportedOperation(
                call.weight_slots[0], "moe_experts", call.operation, "-", None,
                "scope_refused", str(error)))
    for slot, op in sorted(selected_required - selected_covered):
        operation_scope_refusals.append(op)
        unsupported.append(Qwen35GGUFUnsupportedOperation(
            slot, "moe_experts", op, "-", None, "scope_refused",
            "selected resident has no complete explicit call intent (including every partner)"))

    return Qwen35GGUFAdmissionReport(
        backend=str(backend),
        file_type_stamp=None if file_type_stamp is None else str(file_type_stamp),
        manifest_fingerprint=manifest.fingerprint,
        preset=preset,
        requested_operations=requested,
        unsupported=tuple(unsupported),
        covered_slots=covered_slots,
        qualified_records=tuple(dict.fromkeys(qualified.values())),
        slot_filter=effective_slot_filter,
        plan_contract=Qwen35GGUFPlanContract(
            operations=requested,
            slot_filter=effective_slot_filter,
            decode_repack=bool(plan_flags["decode_repack"]),
            repack_veto=bool(repack_veto) if repack_veto is not None else False,
            contract_f32_linear=bool(contraction),
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
            backend=backend_key,
            invocations=tuple(sorted(invocations, key=lambda item: (item.slot, item.operation))),
            required_invocations=tuple(sorted(required_invocations)),
            selected_invocations=tuple(selected_bound),
            required_selected_intents=tuple(selected_intents),
            f32_input_operations=f32_input,
            resident_plan_records=tuple(resident_plan_records),
            required_plan_slots=tuple(required_plan_slots),
            operation_scope_refusals=tuple(operation_scope_refusals),
        ),
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
    resident records, with no device work. This legacy diagnostic is NOT
    execution authorization: native entries consume the complete F4 contract
    through ``authorize_native_execution``, including embedding/head/MoE and
    actual physical ownership. Known native routes are requested before
    allocation through the session's ``execution_routes``.
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
