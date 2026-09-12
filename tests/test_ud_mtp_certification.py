"""U6: UD MTP is a separate certification unit from UD AR.

UD AR certificates are pinned per role-manifest fingerprint with scope
``("ar",)``.  MTP admission is a distinct table keyed by the same fingerprint,
so a new AR certificate can never widen MTP admission and an MTP certificate can
be revoked without touching AR.

These tests cover the enabling conditions and the identity binding, not the
performance qualification:

* the quant-agnostic speculative accept chain resolves for the UD session
  identities (it is registered per session quant identity, the same way the
  pre-existing ``gguf_ud_q3_k_m`` entry is);
* the UD NextN draft dtype manifest is resolved from the admitted preset
  identity, never from file metadata or a caller-claimed variant;
* that manifest must cover the validated draft slots exactly;
* UD presets stay AR-only until ``_UD_MTP_PRESET_FINGERPRINTS`` carries an
  entry, and populating it composes the MTP scope correctly.

The end-to-end MTP qualification (exactness, category suite, true no-MTP AR
denominator) is driven by ``scripts/ud_mtp_certification.py``; it is too long
for the default suite.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

# Import-time kernel registration: the accept chain registers itself when its
# module is imported, so the registry assertions below need this import.
import hipengine.kernels.hip_gfx1100.speculative.dflash_accept  # noqa: F401
from hipengine.kernels.backends import HIP_BACKEND_TARGET_ARCH
from hipengine.kernels.registry import KernelKey, registered_keys, resolve
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_admission import (
    GGUF_UD_Q4_K_M_PRESET,
    GGUF_UD_Q4_K_S_PRESET,
    GGUF_PRESET_SCOPE_AR,
    GGUF_PRESET_SCOPE_MTP,
    Qwen35GGUFArtifactPreset,
    resolve_qwen35_gguf_artifact_preset,
    resolve_qwen35_gguf_nextn_draft_qtypes,
)
from hipengine.loading.qwen35_gguf_nextn import (
    build_qwen35_gguf_nextn_tensor_map,
    validate_qwen35_gguf_nextn_tensor_map,
)
from hipengine.quant.gguf import GGMLQuantizationType

UD_Q4_K_M = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")
UD_Q4_K_S = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf")
PLAIN_Q4_K_M = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")

# The draft block's actual per-slot quantization, read from the pinned
# artifacts.  Both UD artifacts share this signature; the plain control's draft
# block differs (Q4_K attention/FFN with a Q8_0 eh_proj).
UD_DRAFT_QTYPES = {
    "attn_norm": GGMLQuantizationType.F32,
    "post_attention_norm": GGMLQuantizationType.F32,
    "attn_q_norm": GGMLQuantizationType.F32,
    "attn_k_norm": GGMLQuantizationType.F32,
    "enorm": GGMLQuantizationType.F32,
    "hnorm": GGMLQuantizationType.F32,
    "shared_head_norm": GGMLQuantizationType.F32,
    "eh_proj": GGMLQuantizationType.Q6_K,
    "attn_q": GGMLQuantizationType.Q6_K,
    "attn_k": GGMLQuantizationType.Q8_0,
    "attn_v": GGMLQuantizationType.Q8_0,
    "attn_output": GGMLQuantizationType.Q6_K,
    "ffn_gate": GGMLQuantizationType.Q6_K,
    "ffn_up": GGMLQuantizationType.Q6_K,
    "ffn_down": GGMLQuantizationType.Q6_K,
}

UD_SESSION_QUANTS = ("gguf_ud_q4_k_m", "gguf_ud_q4_k_s")


def _real_map(path: Path):
    from hipengine.loading.gguf import GGUFReader

    reader = GGUFReader(path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    nextn_map = None
    if model_map.config.ignored_block_ids:
        nextn_map = build_qwen35_gguf_nextn_tensor_map(reader.info, strict=False)
    return reader, model_map, nextn_map


@pytest.mark.parametrize("quant", UD_SESSION_QUANTS)
def test_ud_accept_chain_resolves_for_the_ud_session_identity(quant: str):
    """The accept chain is quant-agnostic (int32 buffers), so it is registered
    once per session quant identity that reaches it."""

    key = KernelKey("hip_gfx1100", "dflash_accept_chain", quant, "i32")
    assert key in registered_keys()
    kernel = resolve(
        backend="hip_gfx1100",
        layer="dflash_accept_chain",
        quant=quant,
        variant="i32",
    )
    assert kernel is not None


def test_accept_chain_registration_covers_every_supported_session_identity():
    """A lane whose session quant identity is missing here cannot run MTP."""

    for quant in UD_SESSION_QUANTS + ("w4_paro", "gguf_ud_q3_k_m", "gguf_q4_k_m", "gguf_q4_k_s"):
        assert KernelKey("hip_gfx1100", "dflash_accept_chain", quant, "i32") in registered_keys()


def test_draft_qtypes_resolve_only_through_the_preset_identity():
    """No preset, no pin: the manifest is not reachable from a variant claim."""

    assert resolve_qwen35_gguf_nextn_draft_qtypes(None) is None
    # A preset with a non-UD key carries no pinned draft manifest.
    foreign = Qwen35GGUFArtifactPreset(
        preset_key="gguf-unqualified-manifest",
        scopes=(GGUF_PRESET_SCOPE_AR,),
        manifest_fingerprint="0" * 64,
        file_type_stamp="MOSTLY_Q4_K_M",
    )
    assert resolve_qwen35_gguf_nextn_draft_qtypes(foreign) is None


@pytest.mark.parametrize(
    "path,preset_key",
    [(UD_Q4_K_M, GGUF_UD_Q4_K_M_PRESET), (UD_Q4_K_S, GGUF_UD_Q4_K_S_PRESET)],
)
def test_pinned_ud_draft_manifest_matches_the_artifact(path: Path, preset_key: str):
    if not path.exists():
        pytest.skip(f"pinned artifact missing: {path}")
    reader, model_map, nextn_map = _real_map(path)
    preset = resolve_qwen35_gguf_artifact_preset(
        model_map, nextn_map=nextn_map, file_type_stamp=reader.info.file_type_name
    )
    assert preset is not None and preset.preset_key == preset_key
    pin = resolve_qwen35_gguf_nextn_draft_qtypes(preset)
    assert pin is not None
    # The pin is the artifact's real per-slot quantization, not a relaxation.
    assert dict(pin) == UD_DRAFT_QTYPES
    # ... and it makes the strict draft validation pass where the plain
    # expectation fails.
    validation = validate_qwen35_gguf_nextn_tensor_map(reader.info, pinned_qtypes=pin)
    assert validation.dtype_errors == ()
    build_qwen35_gguf_nextn_tensor_map(reader.info, pinned_qtypes=pin)
    unpinned = validate_qwen35_gguf_nextn_tensor_map(reader.info)
    assert unpinned.dtype_errors, "plain expectation must not silently accept the UD draft"


def test_plain_control_gets_no_pinned_draft_manifest():
    if not PLAIN_Q4_K_M.exists():
        pytest.skip(f"pinned artifact missing: {PLAIN_Q4_K_M}")
    reader, model_map, nextn_map = _real_map(PLAIN_Q4_K_M)
    preset = resolve_qwen35_gguf_artifact_preset(
        model_map, nextn_map=nextn_map, file_type_stamp=reader.info.file_type_name
    )
    assert preset is None
    assert resolve_qwen35_gguf_nextn_draft_qtypes(preset) is None


def test_partial_pinned_manifest_is_refused():
    """A partial pin would silently fall back to the plain expectation."""

    if not UD_Q4_K_M.exists():
        pytest.skip(f"pinned artifact missing: {UD_Q4_K_M}")
    from hipengine.loading.gguf import GGUFReader

    reader = GGUFReader(UD_Q4_K_M)
    partial = {k: v for k, v in UD_DRAFT_QTYPES.items() if k != "ffn_down"}
    validation = validate_qwen35_gguf_nextn_tensor_map(reader.info, pinned_qtypes=partial)
    assert any("must cover exactly the validated slots" in e for e in validation.dtype_errors)

    unknown = {**UD_DRAFT_QTYPES, "not_a_draft_slot": GGMLQuantizationType.F32}
    validation = validate_qwen35_gguf_nextn_tensor_map(reader.info, pinned_qtypes=unknown)
    assert any("must cover exactly the validated slots" in e for e in validation.dtype_errors)


# ---------------------------------------------------------------------------
# U6 unit: the six gated elements
# ---------------------------------------------------------------------------


def _grant_mtp_scope(monkeypatch) -> None:
    """Grant MTP scope in-process for the pinned UD presets (candidate mode)."""

    from hipengine.loading import qwen35_gguf_admission as admission

    original = admission.resolve_qwen35_gguf_artifact_preset

    def patched(*args, **kwargs):
        preset = original(*args, **kwargs)
        if preset is not None and preset.preset_key in UD_SESSION_QUANTS:
            return admission.replace(
                preset, scopes=(*preset.scopes, GGUF_PRESET_SCOPE_MTP)
            )
        return preset

    monkeypatch.setattr(admission, "resolve_qwen35_gguf_artifact_preset", patched)


def test_u6_certification_records_define_all_six_items():
    """U6 defines every required element, and an incomplete unit mints no pin."""

    from hipengine.loading import qwen35_gguf_admission as admission

    required = {
        "nextn_residency_and_eh_proj_head_ownership",
        "blk64_draft_operation_set",
        "draft_and_verifier_state_ownership",
        "ud_specific_journal_requirements",
        "exact_ar_mtp_control_behavior",
        "supported_backend_quant_profile_context_width_scope",
    }
    assert set(admission._UD_MTP_CERTIFICATIONS) == set(admission._UD_PRESET_FINGERPRINTS)
    for fingerprint, certification in admission._UD_MTP_CERTIFICATIONS.items():
        assert certification.fingerprint == fingerprint
        # The declared envelope names a real backend and profile.
        assert certification.backend in HIP_BACKEND_TARGET_ARCH
        assert certification.execution_profile in {"strict", "production"}
        assert certification.widths and all(w >= 1 for w in certification.widths)
        assert {item.item for item in certification.items} == required
        # Every open item names concrete missing evidence, never a plan.
        for item in certification.items:
            assert item.contract and item.evidence
            assert bool(item.blocker) != bool(item.qualified), item.item
            assert item.phase in {"pre_measurement", "paired_run"}, item.item
        # Phase split: the structural/control items gate the paired run, the
        # items the run establishes do not (gating on them would be circular).
        assert certification.measurement_ready() == all(
            item.qualified for item in certification.items if item.phase == "pre_measurement"
        )
        assert certification.measurement_blockers == tuple(
            item.item
            for item in certification.items
            if item.phase == "pre_measurement" and not item.qualified
        )
        # The measurement gate is strictly weaker than the pin.
        if not certification.measurement_ready():
            assert not certification.is_complete()
        assert {
            item.item for item in certification.items if item.phase == "paired_run"
        } == {
            "exact_ar_mtp_control_behavior",
            "supported_backend_quant_profile_context_width_scope",
        }
        # The derived pin can only contain complete units.
        if certification.is_complete():
            assert fingerprint in admission._UD_MTP_PRESET_FINGERPRINTS
        else:
            assert fingerprint not in admission._UD_MTP_PRESET_FINGERPRINTS
            assert certification.blocked_items
    # Both pinned UD artifacts are defined but not yet certified for MTP.
    assert admission._UD_MTP_PRESET_FINGERPRINTS == {}


@pytest.mark.parametrize("path", [UD_Q4_K_M, UD_Q4_K_S])
def test_nextn_draft_owns_block_and_borrows_embedding_and_head(path: Path):
    """U6 item 1: draft residency and eh_proj/head ownership."""

    if not path.exists():
        pytest.skip(f"pinned artifact missing: {path}")
    reader, model_map, nextn_map = _real_map(path)
    assert nextn_map is not None

    owned = {tensor.name for tensor in (*nextn_map.layer_tensors.values(), *nextn_map.nextn_tensors.values())}
    root_names = {tensor.name for tensor in model_map.root_tensors.values()}
    # eh_proj is draft-owned, never borrowed.
    assert "eh_proj" in nextn_map.nextn_tensors
    # Every fallback resolves to a resident this artifact already plans, with
    # the identical source tensor: the target's root embedding/head, or a
    # draft-owned NextN slot (the shared head norm doubles as output norm).
    for slot, tensor in nextn_map.fallback_tensors.items():
        assert tensor.name in owned or tensor.name in root_names, (slot, tensor.name)
    assert nextn_map.fallback_tensors["token_embedding"].name == model_map.root_tensors["token_embedding"].name
    assert nextn_map.fallback_tensors["lm_head"].name == model_map.root_tensors["lm_head"].name
    assert nextn_map.fallback_tensors["output_norm"].name == nextn_map.nextn_tensors["shared_head_norm"].name
    # The draft does NOT own an embedding table or a head.
    assert not any(name == "token_embd.weight" for name in owned)
    assert not any(name == "output.weight" for name in owned)


def test_unplanned_nextn_fallback_is_refused(monkeypatch):
    """U6 item 1: a fallback naming no planned resident refuses."""

    from hipengine.loading import qwen35_gguf_admission as admission
    from hipengine.loading.qwen35_gguf_nextn import Qwen35GGUFNextNMap

    if not UD_Q4_K_M.exists():
        pytest.skip(f"pinned artifact missing: {UD_Q4_K_M}")
    _grant_mtp_scope(monkeypatch)
    reader, model_map, nextn_map = _real_map(UD_Q4_K_M)
    assert nextn_map is not None
    original = nextn_map.fallback_tensors["lm_head"]
    foreign = replace(original, name="blk.0.not_a_resident.weight")
    forged = replace(
        nextn_map,
        fallback_tensors={**dict(nextn_map.fallback_tensors), "lm_head": foreign},
    )
    assert isinstance(forged, Qwen35GGUFNextNMap)
    report = admission.preflight_qwen35_gguf_artifact(
        model_map,
        backend="hip_gfx1100",
        operations=(admission.QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,),
        nextn_map=forged,
    )
    # The MTP scope is granted, so the borrow check is the only refusal: the
    # forged fallback must be what makes the contract incomplete.
    assert [item.stage for item in report.unsupported] == ["fallback_unowned"]
    assert not report.plan_contract.is_complete()
    assert "nextn_block.64.fallback:lm_head" in report.plan_contract.required_plan_slots


@pytest.mark.parametrize("path", [UD_Q4_K_M, UD_Q4_K_S])
def test_draft_operation_set_is_slot_scoped_and_certified(path: Path, monkeypatch):
    """U6 item 2: the blk.64 draft operation set is certified and slot-scoped."""

    from hipengine.loading import qwen35_gguf_admission as admission

    if not path.exists():
        pytest.skip(f"pinned artifact missing: {path}")
    reader, model_map, nextn_map = _real_map(path)
    draft = admission.QWEN35_GGUF_OP_MTP_NEXTN_DRAFT

    # Without the MTP scope the whole operation refuses.
    scoped = admission.preflight_qwen35_gguf_artifact(
        model_map, backend="hip_gfx1100", operations=(draft,), nextn_map=nextn_map
    )
    assert any(item.stage == "scope_refused" for item in scoped.unsupported)

    _grant_mtp_scope(monkeypatch)
    report = admission.preflight_qwen35_gguf_artifact(
        model_map, backend="hip_gfx1100", operations=(draft,), nextn_map=nextn_map
    )
    assert report.supported, report.render_refusals()
    assert report.plan_contract.is_complete()
    # Exactly the draft block's owned slots are covered; the borrowed root
    # residents stay outside the draft scope.
    expected = len(nextn_map.layer_tensors) + len(nextn_map.nextn_tensors)
    assert report.covered_slots == expected
    assert all(record.operation == draft for record in report.qualified_records)
    assert all(record.kernel_layer for record in report.qualified_records)


def test_foreign_draft_signature_is_not_silently_certified():
    """U6 item 2: another artifact's draft layout is not certified by accident."""

    from hipengine.loading import qwen35_gguf_admission as admission

    if not PLAIN_Q4_K_M.exists():
        pytest.skip(f"pinned artifact missing: {PLAIN_Q4_K_M}")
    reader, model_map, nextn_map = _real_map(PLAIN_Q4_K_M)
    draft = admission.QWEN35_GGUF_OP_MTP_NEXTN_DRAFT
    report = admission.preflight_qwen35_gguf_artifact(
        model_map, backend="hip_gfx1100", operations=(draft,), nextn_map=nextn_map
    )
    assert not report.supported
    assert not report.plan_contract.is_complete()


def test_ud_journal_is_row_capable_across_the_declared_context():
    """U6 item 4: the UD journal must own rows whenever serial is reachable."""

    from hipengine.kernels.backends import backend_package_capability
    from hipengine.runtime.qwen35_gguf_mtp import _verify_journal_plan

    backend = "hip_gfx1100"
    native_limit = int(
        backend_package_capability(backend, "GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT", 0)
    )
    assert native_limit > 0
    # The serial route is reachable beyond the backend's native row context, so
    # a row-capable journal is mandatory: producer capture may not be paired
    # with a bounded journal.
    initial_state_only, producer_capture = _verify_journal_plan(
        "native",
        max_candidate_budget=2,
        backend=backend,
        max_end_position=native_limit + 1,
    )
    assert initial_state_only is False
    assert producer_capture is False
    # Within the native row context the initial-state-only journal is legal.
    initial_state_only, producer_capture = _verify_journal_plan(
        "native",
        max_candidate_budget=2,
        backend=backend,
        max_end_position=native_limit,
    )
    assert initial_state_only is True
    assert producer_capture is True


@pytest.mark.parametrize("path", [UD_Q4_K_M, UD_Q4_K_S])
def test_ud_presets_stay_ar_only_until_the_mtp_pin_lands(path: Path):
    """U6 gate: AR certificates never imply MTP admission."""

    if not path.exists():
        pytest.skip(f"pinned artifact missing: {path}")
    reader, model_map, nextn_map = _real_map(path)
    preset = resolve_qwen35_gguf_artifact_preset(
        model_map, nextn_map=nextn_map, file_type_stamp=reader.info.file_type_name
    )
    assert preset is not None
    assert preset.scope_certified(GGUF_PRESET_SCOPE_AR)
    from hipengine.loading.qwen35_gguf_admission import _UD_MTP_PRESET_FINGERPRINTS

    if preset.manifest_fingerprint in _UD_MTP_PRESET_FINGERPRINTS:
        assert preset.scope_certified(GGUF_PRESET_SCOPE_MTP)
    else:
        assert not preset.scope_certified(GGUF_PRESET_SCOPE_MTP)


def test_mtp_pin_composes_the_scope_and_evidence(monkeypatch):
    """Populating the separate table grants MTP scope without touching AR."""

    from hipengine.loading import qwen35_gguf_admission as admission

    if not UD_Q4_K_M.exists():
        pytest.skip(f"pinned artifact missing: {UD_Q4_K_M}")
    reader, model_map, nextn_map = _real_map(UD_Q4_K_M)
    base = resolve_qwen35_gguf_artifact_preset(
        model_map, nextn_map=nextn_map, file_type_stamp=reader.info.file_type_name
    )
    assert base is not None and base.scopes == (GGUF_PRESET_SCOPE_AR,)

    monkeypatch.setattr(
        admission,
        "_UD_MTP_PRESET_FINGERPRINTS",
        {base.manifest_fingerprint: "Test-only MTP qualification evidence."},
    )
    certified = resolve_qwen35_gguf_artifact_preset(
        model_map, nextn_map=nextn_map, file_type_stamp=reader.info.file_type_name
    )
    assert certified is not None
    assert certified.scopes == (GGUF_PRESET_SCOPE_AR, GGUF_PRESET_SCOPE_MTP)
    assert certified.scope_certified(GGUF_PRESET_SCOPE_MTP)
    assert certified.scope_certified(GGUF_PRESET_SCOPE_AR)
    assert "Test-only MTP qualification evidence." in certified.note
    # The AR pin itself is untouched.
    assert admission._UD_PRESET_FINGERPRINTS[base.manifest_fingerprint][1] == (
        GGUF_PRESET_SCOPE_AR,
    )


def _u6_records():
    from hipengine.loading.qwen35_gguf_admission import _UD_MTP_CERTIFICATIONS

    assert _UD_MTP_CERTIFICATIONS, "the U6 table must not be empty"
    return _UD_MTP_CERTIFICATIONS


def test_u6_open_items_name_a_blocker_and_qualified_items_do_not():
    """The ``blocker`` field is the record's own completeness proof.

    An open item must name the concrete missing evidence, and a qualified item
    must not keep a stale one: a unit that is finished but still advertises a
    blocker understates what is qualified, and the paired gate prints these
    strings verbatim into every artifact's ``u6_gate`` block.
    """

    for fingerprint, certification in _u6_records().items():
        for item in certification.items:
            if item.qualified:
                assert item.blocker == "", (fingerprint, item.item, item.blocker)
            else:
                assert item.blocker.strip(), (fingerprint, item.item)


def test_u6_evidence_names_existing_artifacts():
    """A cited benchmark artifact must exist in the tree.

    Evidence strings are the only trace from a qualified item back to the
    measurement that qualified it, so a dangling path silently turns a claim
    into an assertion.
    """

    import re

    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"benchmarks/results/[A-Za-z0-9._-]+\.json")
    missing: list[str] = []
    for fingerprint, certification in _u6_records().items():
        for item in certification.items:
            for reference in pattern.findall(item.evidence):
                if not (root / reference).is_file():
                    missing.append(f"{fingerprint[:8]}/{item.item}: {reference}")
    assert not missing, missing


def test_u6_certificate_is_incomplete_while_the_scope_item_is_open():
    """The pin stays unminted until every item, in both phases, qualifies."""

    for certification in _u6_records().values():
        assert certification.context_max is None
        assert certification.widths == (1,)
        assert not certification.is_complete()
        assert (
            "supported_backend_quant_profile_context_width_scope"
            in certification.blocked_items
        )
        assert certification.measurement_ready()
