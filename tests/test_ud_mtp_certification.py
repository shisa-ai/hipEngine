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

from pathlib import Path

import pytest

# Import-time kernel registration: the accept chain registers itself when its
# module is imported, so the registry assertions below need this import.
import hipengine.kernels.hip_gfx1100.speculative.dflash_accept  # noqa: F401
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

    monkeypatch.setitem(
        admission._UD_MTP_PRESET_FINGERPRINTS,
        base.manifest_fingerprint,
        "Test-only MTP qualification evidence.",
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
