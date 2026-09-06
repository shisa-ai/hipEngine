"""UD-U1 role-safe admission and policy tests (docs/UD-QUANTS.md section 8).

CPU-only: metadata maps, the shared pure policy API, and the cold-path
admission preflight.  No device allocation, no backend launch, no torch.
Real-artifact tests skip when the pinned local files are absent so no-ROCm /
no-model CI stays green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen35_gguf import (
    LINEAR_ATTENTION,
    Qwen35GGUFConfig,
    Qwen35GGUFLayerMap,
    Qwen35GGUFModelMap,
    build_qwen35_gguf_tensor_map,
)
from hipengine.loading.qwen35_gguf_admission import (
    CERTIFIED_OPERATION_COVERAGE,
    DEFAULT_AR_OPERATIONS,
    GGUF_UD_Q4_K_M_PRESET,
    GGUF_UD_Q4_K_S_PRESET,
    GGUF_PRESET_SCOPE_AR,
    GGUF_PRESET_SCOPE_MTP,
    QWEN35_GGUF_OP_AR_DECODE_C1,
    QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,
    QWEN35_GGUF_OP_AR_DECODE_ROWS,
    QWEN35_GGUF_OP_AR_PREFILL,
    QWEN35_GGUF_OP_EMBEDDING_LOOKUP,
    QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,
    QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,
    Qwen35GGUFAdmissionError,
    build_qwen35_gguf_role_manifest,
    certificate_covers_artifact,
    preflight_qwen35_gguf_artifact,
    resolve_qwen35_gguf_artifact_preset,
)
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.quant.gguf import GGMLQuantizationType

UD_Q4_K_M = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")
UD_Q4_K_S = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf")
PLAIN_Q4_K_M = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
PLAIN_Q4_K_S = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
SMALL_Q8_0 = Path("/models/gguf/Qwen3.5-0.8B-Q8_0.gguf")

# The 18 unsupported AR tensors of the pinned UD K_M artifact (docs/UD-QUANTS.md
# section 3 / the K_M campaign's exact unsupported tensor map).
UD_K_M_REFUSED_SLOTS = frozenset(
    {
        "layers.0.ffn_up",
        "layers.1.ffn_down",
        "layers.2.ffn_down",
        "layers.3.ffn_down",
        "layers.11.ffn_gate",
        "layers.13.ffn_down",
        "layers.13.ffn_gate",
        "layers.14.ffn_down",
        "layers.14.ffn_gate",
        "layers.14.ffn_up",
        "layers.15.ffn_down",
        "layers.15.ffn_gate",
        "layers.16.ffn_gate",
        "layers.17.ffn_down",
        "layers.21.attn_qkv",
        "layers.27.ffn_gate",
        "layers.27.ffn_up",
        "layers.50.ffn_gate",
    }
)


def _gguf_backend_capability():
    from hipengine.kernels.backends import backend_package_capability

    return backend_package_capability


def _dense_flags(backend: str, stamp: str | None):
    from hipengine.loading.qwen35_gguf_policy import resolve_gguf_dense_flags

    return resolve_gguf_dense_flags(
        backend,
        stamp,
        capability_reader=_gguf_backend_capability(),
        environ=os.environ,
    )


# ---------------------------------------------------------------------------
# Synthetic map helpers (metadata only)
# ---------------------------------------------------------------------------


def _tensor(
    name: str,
    shape: tuple[int, ...],
    qtype: GGMLQuantizationType = GGMLQuantizationType.F32,
) -> GGUFTensorInfo:
    n_elements = 1
    for dim in shape:
        n_elements *= int(dim)
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=n_elements,
        nbytes=n_elements * (4 if qtype == GGMLQuantizationType.F32 else 2),
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )


def _config(layer_types: tuple[str, ...]) -> Qwen35GGUFConfig:
    return Qwen35GGUFConfig(
        architecture="qwen35",
        block_count=len(layer_types),
        hidden_size=8,
        vocab_size=11,
        feed_forward_length=5,
        context_length=64,
        head_count=2,
        head_count_kv=1,
        key_length=4,
        value_length=4,
        full_attention_interval=0,
        layer_types=layer_types,
        rms_norm_eps=1.0e-6,
        rope_dimension_count=4,
        rope_dimension_sections=(),
        rope_freq_base=10000.0,
        ssm_inner_size=16,
        ssm_group_count=2,
        ssm_state_size=4,
        ssm_conv_kernel=2,
        ssm_time_step_rank=2,
        lm_head_tensor_name="token_embd.weight",
    )


def _linear_layer_tensors(
    layer_id: int,
    *,
    attn_qkv_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    alpha_beta_type: GGMLQuantizationType = GGMLQuantizationType.F32,
    ffn_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    gate_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
) -> dict:
    prefix = f"blk.{layer_id}"
    return {
        "attn_norm": _tensor(f"{prefix}.attn_norm.weight", (8,)),
        "post_attention_norm": _tensor(f"{prefix}.post_attention_norm.weight", (8,)),
        "attn_gate": _tensor(f"{prefix}.attn_gate.weight", (16, 8), gate_type),
        "attn_qkv": _tensor(f"{prefix}.attn_qkv.weight", (28, 8), attn_qkv_type),
        "ssm_a": _tensor(f"{prefix}.ssm_a", (2,)),
        "ssm_alpha": _tensor(f"{prefix}.ssm_alpha.weight", (2, 8), alpha_beta_type),
        "ssm_beta": _tensor(f"{prefix}.ssm_beta.weight", (2, 8), alpha_beta_type),
        "ssm_conv1d": _tensor(f"{prefix}.ssm_conv1d.weight", (28, 2)),
        "ssm_dt_bias": _tensor(f"{prefix}.ssm_dt.bias", (2,)),
        "ssm_norm": _tensor(f"{prefix}.ssm_norm.weight", (3,)),
        "ssm_out": _tensor(f"{prefix}.ssm_out.weight", (8, 16), gate_type),
        "ffn_gate": _tensor(f"{prefix}.ffn_gate.weight", (5, 8), ffn_type),
        "ffn_up": _tensor(f"{prefix}.ffn_up.weight", (5, 8), ffn_type),
        "ffn_down": _tensor(f"{prefix}.ffn_down.weight", (8, 5), ffn_type),
    }


def _synthetic_model_map(
    *,
    layer_types: tuple[str, ...] = (LINEAR_ATTENTION,),
    attn_qkv_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    alpha_beta_type: GGMLQuantizationType = GGMLQuantizationType.F32,
    ffn_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    gate_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    embedding_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    lm_head_type: GGMLQuantizationType = GGMLQuantizationType.Q6_K,
) -> Qwen35GGUFModelMap:
    from types import MappingProxyType

    root = {
        "token_embedding": _tensor("token_embd.weight", (11, 8), embedding_type),
        "output_norm": _tensor("output_norm.weight", (8,)),
        "lm_head": _tensor("token_embd.weight", (11, 8), lm_head_type),
    }
    layers = tuple(
        Qwen35GGUFLayerMap(
            layer_id=layer_id,
            layer_type=layer_type,
            tensors=MappingProxyType(
                _linear_layer_tensors(
                    layer_id,
                    attn_qkv_type=attn_qkv_type,
                    alpha_beta_type=alpha_beta_type,
                    ffn_type=ffn_type,
                    gate_type=gate_type,
                )
            ),
        )
        for layer_id, layer_type in enumerate(layer_types)
    )
    return Qwen35GGUFModelMap(
        config=_config(layer_types),
        root_tensors=MappingProxyType(root),
        layers=layers,
        validation=None,
    )


# ---------------------------------------------------------------------------
# Role-manifest fingerprint: same stamp, different maps
# ---------------------------------------------------------------------------


def test_role_manifest_fingerprint_distinguishes_same_stamp_different_maps():
    plain_map = _synthetic_model_map()
    ud_like_map = _synthetic_model_map(
        attn_qkv_type=GGMLQuantizationType.IQ4_XS,
        ffn_type=GGMLQuantizationType.Q4_K,
    )
    plain = build_qwen35_gguf_role_manifest(plain_map)
    ud_like = build_qwen35_gguf_role_manifest(ud_like_map)

    # Identical file-type stamp, different role/shape/type manifests.
    assert plain.fingerprint != ud_like.fingerprint
    # The stamp never enters the fingerprint input; only the manifest does.
    stamp_a = resolve_qwen35_gguf_artifact_preset(
        plain_map, file_type_stamp="MOSTLY_Q4_K_M"
    )
    stamp_b = resolve_qwen35_gguf_artifact_preset(
        ud_like_map, file_type_stamp="MOSTLY_Q4_K_M"
    )
    # Neither synthetic manifest is a pinned UD artifact, so neither resolves
    # to a UD preset -- and they are never conflated with each other.
    assert stamp_a is None and stamp_b is None
    assert certificate_covers_artifact(
        preflight_qwen35_gguf_artifact(
            plain_map, backend="hip_gfx1100", file_type_stamp="MOSTLY_Q4_K_M"
        ).certificate(),
        manifest_fingerprint=ud_like.fingerprint,
    ) is False


def test_role_manifest_fingerprint_distinguishes_swapped_recurrent_ffn_types():
    # Same dtype histogram (two Q8_0, rest Q4_K/F32) with the recurrent pair
    # and the attention-gate pair swapped.  A histogram cannot see this; the
    # role manifest must.
    recurrent_q8 = _synthetic_model_map(
        alpha_beta_type=GGMLQuantizationType.Q8_0,
        gate_type=GGMLQuantizationType.Q4_K,
    )
    ffn_q8 = _synthetic_model_map(
        alpha_beta_type=GGMLQuantizationType.Q4_K,
        gate_type=GGMLQuantizationType.Q8_0,
    )

    def histogram(records):
        from collections import Counter

        return Counter(record[3] for record in records)

    manifest_a = build_qwen35_gguf_role_manifest(recurrent_q8)
    manifest_b = build_qwen35_gguf_role_manifest(ffn_q8)
    assert histogram(manifest_a.records) == histogram(manifest_b.records)
    assert manifest_a.fingerprint != manifest_b.fingerprint

    certificate_a = preflight_qwen35_gguf_artifact(
        recurrent_q8, backend="hip_gfx1100", file_type_stamp="MOSTLY_Q4_K_M"
    ).certificate()
    assert (
        certificate_covers_artifact(
            certificate_a, manifest_fingerprint=manifest_b.fingerprint
        )
        is False
    )
    # The admission verdicts bind to roles, not the shared histogram: both
    # manifests refuse the native BF16-pointer owner, but for different
    # concrete source types (recurrent Q8_0 vs attention-gate Q8_0).
    report_a = preflight_qwen35_gguf_artifact(
        recurrent_q8,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
    )
    report_b = preflight_qwen35_gguf_artifact(
        ffn_q8,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
    )
    a_alpha = {
        (u.slot_path, u.source_ggml_type)
        for u in report_a.unsupported
        if u.role_class == "recurrent_alpha_beta"
    }
    b_alpha = {
        (u.slot_path, u.source_ggml_type)
        for u in report_b.unsupported
        if u.role_class == "recurrent_alpha_beta"
    }
    assert a_alpha and b_alpha
    assert {slot for slot, _ in a_alpha} == {slot for slot, _ in b_alpha}
    assert {ggml_type for _, ggml_type in a_alpha} == {"Q8_0"}
    assert {ggml_type for _, ggml_type in b_alpha} == {"Q4_K"}


def test_fingerprint_covers_nextn_block_and_fallback_binding():
    from hipengine.loading.qwen35_gguf_nextn import Qwen35GGUFNextNMap

    base = _synthetic_model_map()
    without_nextn = build_qwen35_gguf_role_manifest(base)
    tensors = dict(base.layers[0].tensors)
    nextn_map = Qwen35GGUFNextNMap(
        config=base.config,
        block_id=8,
        layer_tensors={
            "attn_norm": _tensor("blk.8.attn_norm.weight", (8,)),
            "attn_q": _tensor("blk.8.attn_q.weight", (16, 8), GGMLQuantizationType.Q4_K),
        },
        nextn_tensors={
            "eh_proj": _tensor("blk.8.nextn.eh_proj.weight", (8, 16), GGMLQuantizationType.Q8_0),
            "enorm": _tensor("blk.8.nextn.enorm.weight", (8,)),
            "hnorm": _tensor("blk.8.nextn.hnorm.weight", (8,)),
            "shared_head_norm": _tensor("blk.8.nextn.shared_head_norm.weight", (8,)),
        },
        fallback_tensors={
            "token_embedding": base.root_tensors["token_embedding"],
            "lm_head": base.root_tensors["lm_head"],
            "output_norm": base.root_tensors["output_norm"],
        },
        validation=None,
    )
    with_nextn = build_qwen35_gguf_role_manifest(base, nextn_map=nextn_map)
    assert with_nextn.fingerprint != without_nextn.fingerprint
    # The NextN block's own tensors and its fallback provenance are bound.
    roles = {record[0] for record in with_nextn.records}
    assert "nextn_block.8.eh_proj" in roles
    assert "nextn_block.8.fallback:lm_head" in roles


# ---------------------------------------------------------------------------
# Pinned UD preset identities (real artifacts, header-only reads)
# ---------------------------------------------------------------------------


def _real_map(path: Path, *, strict_nextn: bool = False):
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map

    reader = GGUFReader(path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    nextn_map = None
    if model_map.config.ignored_block_ids:
        nextn_map = build_qwen35_gguf_nextn_tensor_map(reader.info, strict=strict_nextn)
    return reader, model_map, nextn_map


@pytest.mark.skipif(not UD_Q4_K_M.exists(), reason=f"pinned artifact missing: {UD_Q4_K_M}")
def test_pinned_ud_q4_k_m_resolves_through_manifest_not_stamp():
    stamp = "MOSTLY_Q4_K_M"
    _reader, model_map, nextn_map = _real_map(UD_Q4_K_M)
    preset = resolve_qwen35_gguf_artifact_preset(
        model_map, nextn_map=nextn_map, file_type_stamp=stamp
    )
    assert preset is not None
    assert preset.preset_key == GGUF_UD_Q4_K_M_PRESET
    # AR-only until U6 resolves draft/serving scopes.
    assert preset.scopes == (GGUF_PRESET_SCOPE_AR,)
    assert preset.scope_certified(GGUF_PRESET_SCOPE_AR)
    assert not preset.scope_certified(GGUF_PRESET_SCOPE_MTP)
    assert preset.file_type_stamp == stamp
    # A plain control sharing the stamp does not resolve to the UD preset...
    _r2, plain_map, plain_nextn = _real_map(PLAIN_Q4_K_M)
    assert (
        resolve_qwen35_gguf_artifact_preset(
            plain_map, nextn_map=plain_nextn, file_type_stamp=stamp
        )
        is None
    )


@pytest.mark.skipif(not UD_Q4_K_S.exists(), reason=f"pinned artifact missing: {UD_Q4_K_S}")
def test_pinned_ud_q4_k_s_has_an_equally_explicit_preset_identity():
    stamp = "MOSTLY_Q4_K_S"
    _reader, model_map, nextn_map = _real_map(UD_Q4_K_S)
    preset = resolve_qwen35_gguf_artifact_preset(
        model_map, nextn_map=nextn_map, file_type_stamp=stamp
    )
    assert preset is not None
    assert preset.preset_key == GGUF_UD_Q4_K_S_PRESET
    assert preset.preset_key != GGUF_UD_Q4_K_M_PRESET
    assert preset.scopes == (GGUF_PRESET_SCOPE_AR,)
    # The plain K_S control with the same stamp stays on the plain lane.
    _r2, plain_map, plain_nextn = _real_map(PLAIN_Q4_K_S)
    assert (
        resolve_qwen35_gguf_artifact_preset(
            plain_map, nextn_map=plain_nextn, file_type_stamp=stamp
        )
        is None
    )


def test_preset_keys_are_session_identities_not_registry_axes():
    # The preset key is a plain admission string. Per-tensor kernel quant keys
    # remain the concrete storage/kernel identities in every planned spec.
    from hipengine.kernels.registry import KernelKey

    key = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_bf16_out")
    assert key.quant == "gguf_q4_k"
    assert key.quant != GGUF_UD_Q4_K_M_PRESET and key.layer != GGUF_UD_Q4_K_M_PRESET
    map_small = _synthetic_model_map()
    spec = plan_qwen35_gguf_weight_spec(
        "layers.0.ffn_up",
        map_small.layers[0].tensors["ffn_up"],
        decode_repack=False,
    )
    assert spec.quant_key == "gguf_q4_k"
    assert spec.layout == LAYOUT_Q4_K_PACK8


# ---------------------------------------------------------------------------
# Preflight aggregation on the real UD artifacts
# ---------------------------------------------------------------------------


def _preflight_real(path: Path, backend: str, operations=DEFAULT_AR_OPERATIONS):
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map

    reader = GGUFReader(path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    nextn_map = None
    if model_map.config.ignored_block_ids:
        nextn_map = build_qwen35_gguf_nextn_tensor_map(reader.info, strict=False)
    stamp = reader.info.file_type_name
    flags = _dense_flags(backend, None if stamp is None else str(stamp))
    return preflight_qwen35_gguf_artifact(
        model_map,
        backend=backend,
        file_type_stamp=None if stamp is None else str(stamp),
        operations=operations,
        nextn_map=nextn_map,
        **flags,
    )


def _real_stamp(path: Path) -> str | None:
    from hipengine.loading.gguf import GGUFReader

    reader = GGUFReader(path)
    stamp = reader.info.file_type_name
    return None if stamp is None else str(stamp)


@pytest.mark.skipif(not UD_Q4_K_M.exists(), reason=f"pinned artifact missing: {UD_Q4_K_M}")
def test_preflight_aggregates_every_unsupported_slot_on_ud_q4_k_m():
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        report = _preflight_real(UD_Q4_K_M, backend)
        assert report.supported is False
        assert report.preset is not None
        assert report.preset.preset_key == GGUF_UD_Q4_K_M_PRESET
        refused = {
            u.slot_path for u in report.unsupported if u.stage == "planner_refused"
        }
        # Every one of the 18 unsupported slots is reported, not just the
        # first exception, and no other slot is planner-refused.
        assert refused == UD_K_M_REFUSED_SLOTS
        with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
            report.raise_for_errors()
        message = str(excinfo.value)
        for slot in sorted(UD_K_M_REFUSED_SLOTS)[:8]:
            assert slot in message
        assert "unsupported slots/modes: 54" in message


@pytest.mark.skipif(not UD_Q4_K_S.exists(), reason=f"pinned artifact missing: {UD_Q4_K_S}")
def test_preflight_aggregates_every_unsupported_slot_on_ud_q4_k_s():
    report = _preflight_real(UD_Q4_K_S, "hip_gfx1100")
    assert report.supported is False
    assert report.preset is not None
    assert report.preset.preset_key == GGUF_UD_Q4_K_S_PRESET
    refused = {u.slot_path for u in report.unsupported if u.stage == "planner_refused"}
    # 40 projection refusals plus the Q3_K token embedding = the 41 pinned
    # AR refusals of docs/UD-QUANTS.md section 3.1.
    assert len(refused) == 41
    assert "root.token_embedding" in refused
    refused_types = {
        u.slot_path: u.source_ggml_type for u in report.unsupported if u.stage == "planner_refused"
    }
    type_values = set(refused_types.values())
    assert {"Q3_K", "IQ4_NL", "IQ3_S", "IQ3_XXS", "IQ2_S"} <= type_values


@pytest.mark.skipif(not PLAIN_Q4_K_M.exists(), reason=f"pinned artifact missing: {PLAIN_Q4_K_M}")
def test_plain_controls_pass_preflight_with_expected_coverage():
    for path, stamp in ((PLAIN_Q4_K_M, "MOSTLY_Q4_K_M"), (PLAIN_Q4_K_S, "MOSTLY_Q4_K_S")):
        for backend in ("hip_gfx1100", "hip_gfx1151"):
            report = _preflight_real(path, backend)
            assert report.supported is True, report.render_refusals()
            assert report.preset is None
            assert report.covered_slots == 851
            assert report.unsupported == ()
            certificate = report.certificate()
            assert certificate.preset_key is None
            assert certificate_covers_artifact(
                certificate, manifest_fingerprint=report.manifest_fingerprint
            )


@pytest.mark.skipif(not SMALL_Q8_0.exists(), reason=f"pinned artifact missing: {SMALL_Q8_0}")
def test_small_plain_q8_0_control_passes_default_operations():
    report = _preflight_real(SMALL_Q8_0, "hip_gfx1100")
    assert report.supported is True, report.render_refusals()
    assert report.covered_slots > 0


# ---------------------------------------------------------------------------
# Role/layout-specific hazards
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not UD_Q4_K_M.exists(), reason=f"pinned artifact missing: {UD_Q4_K_M}")
def test_native_multirow_alpha_beta_requires_a_bf16_pointer_owner():
    # UD stores alpha/beta as Q8_0; with the repack veto they stay raw GGUF
    # bytes.  The native multirow owner passes allocation("raw") straight to
    # dense_gemv_out_bf16 (uint16_t* weight ABI), so raw Q8_0 must be refused.
    report = _preflight_real(
        UD_Q4_K_M, "hip_gfx1100", operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,)
    )
    alpha_refusals = [
        u for u in report.unsupported if u.role_class == "recurrent_alpha_beta"
    ]
    _reader, model_map, _nextn = _real_map(UD_Q4_K_M)
    linear_layers = [
        layer.layer_id
        for layer in model_map.layers
        if layer.layer_type != "full_attention"
    ]
    assert {u.slot_path for u in alpha_refusals} == {
        f"layers.{layer}.{slot}"
        for layer in linear_layers
        for slot in ("ssm_alpha", "ssm_beta")
    }
    assert all(u.resident_layout == LAYOUT_RAW_GGUF for u in alpha_refusals)
    assert "BF16" in alpha_refusals[0].reason


@pytest.mark.skipif(not SMALL_Q8_0.exists(), reason=f"pinned artifact missing: {SMALL_Q8_0}")
def test_sole_t16_alpha_beta_cannot_enter_the_native_bf16_pointer_owner():
    # A repacked plain Q8_0 file plans alpha/beta as sole T16 residents with
    # no "raw" allocation at all: the BF16-pointer owner cannot even resolve.
    report = _preflight_real(
        SMALL_Q8_0, "hip_gfx1100", operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,)
    )
    alpha_refusals = [
        u for u in report.unsupported if u.role_class == "recurrent_alpha_beta"
    ]
    assert alpha_refusals
    assert {u.resident_layout for u in alpha_refusals} == {LAYOUT_GGUF_Q8_0_T16}
    _reader, model_map, _nextn = _real_map(SMALL_Q8_0)
    flags = _dense_flags("hip_gfx1100", _real_stamp(SMALL_Q8_0))
    spec = plan_qwen35_gguf_weight_spec(
        "layers.0.ssm_alpha",
        model_map.layers[0].tensors["ssm_alpha"],
        contract_f32_linear=False,
        **{**flags, "decode_repack": True},
    )
    assert spec.layout == LAYOUT_GGUF_Q8_0_T16
    assert spec.allocation_names == ("tiles",)
    assert "raw" not in spec.allocation_names


def test_contracted_bf16_alpha_beta_qualified_for_native_rows():
    # The raw-IQ contraction produces exactly the one qualified native-row
    # owner: a dense BF16 resident for alpha/beta.
    raw_iq_map = _synthetic_model_map(
        attn_qkv_type=GGMLQuantizationType.IQ4_XS,
    )
    report = preflight_qwen35_gguf_artifact(
        raw_iq_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
        decode_repack=True,
        dense_q4_t16=True,
    )
    assert report.unsupported == ()
    assert report.supported
    alpha_records = [
        record
        for record in report.qualified_records
        if record.role_class == "recurrent_alpha_beta"
    ]
    assert alpha_records
    assert {record.resident_layout for record in alpha_records} == {LAYOUT_DENSE_BF16}
    assert alpha_records[0].kernel_layer == "dense_gemv"
    assert alpha_records[0].kernel_quant == "bf16"
    assert alpha_records[0].rows_scope == "rows_2_8_native_bf16_ptr"


def test_dense_bf16_embedding_multirow_is_refused_not_silently_singleton():
    bf16_embedding_map = _synthetic_model_map(
        embedding_type=GGMLQuantizationType.Q4_1,
    )
    report = preflight_qwen35_gguf_artifact(
        bf16_embedding_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_EMBEDDING_LOOKUP,),
    )
    refusals = [u for u in report.unsupported if u.role_class == "token_embedding"]
    assert refusals
    assert refusals[0].resident_layout == LAYOUT_DENSE_BF16
    assert "singleton" in refusals[0].reason

    raw_embedding_map = _synthetic_model_map(
        embedding_type=GGMLQuantizationType.Q8_0,
    )
    raw_report = preflight_qwen35_gguf_artifact(
        raw_embedding_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_EMBEDDING_LOOKUP,),
    )
    assert raw_report.unsupported == ()
    raw_records = [
        record
        for record in raw_report.qualified_records
        if record.role_class == "token_embedding"
    ]
    assert raw_records and raw_records[0].rows_scope == "rows_any"
    assert raw_records[0].resident_layout == LAYOUT_RAW_GGUF


def test_q3_k_embedding_has_no_consumer_until_ud_u5():
    q3_embedding_map = _synthetic_model_map(
        embedding_type=GGMLQuantizationType.Q3_K,
    )
    report = preflight_qwen35_gguf_artifact(
        q3_embedding_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_EMBEDDING_LOOKUP,),
    )
    assert report.supported is False
    refusals = [u for u in report.unsupported if u.role_class == "token_embedding"]
    assert refusals and refusals[0].stage == "planner_refused"
    assert "Q3_K" in refusals[0].reason


# ---------------------------------------------------------------------------
# AR-only vs AR+MTP scope separation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not UD_Q4_K_M.exists(), reason=f"pinned artifact missing: {UD_Q4_K_M}")
def test_ud_preset_refuses_mtp_draft_scope():
    report = _preflight_real(
        UD_Q4_K_M, "hip_gfx1100", operations=(QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,)
    )
    assert report.supported is False
    scope_refusals = [u for u in report.unsupported if u.stage == "scope_refused"]
    assert len(scope_refusals) == 1
    assert scope_refusals[0].operation == QWEN35_GGUF_OP_MTP_NEXTN_DRAFT
    assert "AR-only" in scope_refusals[0].reason
    # AR decode remains the separately-requested scope.
    ar_report = _preflight_real(
        UD_Q4_K_M,
        "hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_MTP_NEXTN_DRAFT),
    )
    ar_scope = [u for u in ar_report.unsupported if u.stage == "scope_refused"]
    assert ar_scope and ar_scope[0].operation == QWEN35_GGUF_OP_MTP_NEXTN_DRAFT


def test_mtp_scope_requires_an_explicitly_certified_preset():
    plain_map = _synthetic_model_map()
    report = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,),
        file_type_stamp="MOSTLY_Q4_K_M",
    )
    assert report.supported is False
    assert all(u.stage == "scope_refused" for u in report.unsupported)


def test_ud_quant_key_does_not_inherit_execution_profiles():
    from hipengine.execution_profiles import (
        ExecutionProfile,
        MissingRuntimeProfilePlanError,
        resolve_runtime_profile,
    )

    with pytest.raises(MissingRuntimeProfilePlanError):
        resolve_runtime_profile(
            model="qwen3_5_gguf",
            backend="hip_gfx1100",
            quant=GGUF_UD_Q4_K_M_PRESET,
            profile=ExecutionProfile.STRICT,
        )
    with pytest.raises(MissingRuntimeProfilePlanError):
        resolve_runtime_profile(
            model="qwen3_5_gguf",
            backend="hip_gfx1100",
            quant=GGUF_UD_Q4_K_S_PRESET,
            profile=ExecutionProfile.PRODUCTION,
        )


def test_unknown_requested_operations_fail_closed():
    plain_map = _synthetic_model_map()
    with pytest.raises(Qwen35GGUFAdmissionError):
        preflight_qwen35_gguf_artifact(
            plain_map,
            backend="hip_gfx1100",
            operations=("totally_unknown_operation",),
        ).raise_for_errors()


# ---------------------------------------------------------------------------
# Coverage record conventions
# ---------------------------------------------------------------------------


def test_coverage_records_use_existing_registry_layer_names():
    from hipengine.kernels.registry import KernelKey

    known_layers = {"linear", "dense_gemv", "embedding", "rmsnorm", "gdn_chain", "moe_selected"}
    for record in CERTIFIED_OPERATION_COVERAGE:
        assert record.operation in {
            QWEN35_GGUF_OP_AR_DECODE_C1,
            QWEN35_GGUF_OP_AR_DECODE_ROWS,
            QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,
            QWEN35_GGUF_OP_AR_PREFILL,
            QWEN35_GGUF_OP_EMBEDDING_LOOKUP,
            QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,
        }
        # Every record names a real registry layer axis and resolves to a
        # well-formed four-axis key once the backend is known.
        assert record.kernel_layer in known_layers
        key = KernelKey(
            "hip_gfx1100",
            record.kernel_layer,
            record.kernel_quant or "<from-weight>",
            record.kernel_variant or "resolved_by_rows",
        )
        assert key.backend == "hip_gfx1100" and key.layer == record.kernel_layer


def test_lm_head_logits_require_an_f32_output_consumer():
    # Every lm_head layout the planner can emit has a registered F32-output
    # dispatch row (pack8/raw per quant, dense-BF16, dense-F32 with F32
    # activations, Q6_K T16); assert the positive contract on representative
    # types.
    for lm_head_type in (
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q6_K,
        GGMLQuantizationType.Q4_1,
        GGMLQuantizationType.F32,
    ):
        report = preflight_qwen35_gguf_artifact(
            _synthetic_model_map(lm_head_type=lm_head_type),
            backend="hip_gfx1100",
            operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
        )
        assert report.unsupported == (), (
            lm_head_type,
            report.render_refusals(),
        )
    # And the coverage table records the F32 output dtype for every lm_head
    # consumer.
    for record in CERTIFIED_OPERATION_COVERAGE:
        if record.operation == QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS:
            assert record.output_dtype == "f32"
