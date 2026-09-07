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

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35GGUFConfig,
    Qwen35GGUFLayerMap,
    Qwen35GGUFModelMap,
    build_qwen35_gguf_tensor_map,
)
from hipengine.loading.qwen35_gguf_admission import (
    CERTIFIED_F32_INPUT_OPERATION_COVERAGE,
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
    qwen35_gguf_native_row_binding_errors,
    resolve_qwen35_gguf_artifact_preset,
)
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q5_K_QMICRO_T16,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    Qwen35GGUFResidentWeights,
    Qwen35GGUFWeightSpec,
    materialize_qwen35_gguf_weights,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import (
    GGUF_Q8_0_BLOCK_BYTES,
    GGUF_Q8_0_T16_BLOCK_BYTES,
)

UD_Q4_K_M = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf")
UD_Q4_K_S = Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf")
PLAIN_Q4_K_M = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
PLAIN_Q4_K_S = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
SMALL_Q8_0 = Path("/models/gguf/Qwen3.5-0.8B-Q8_0.gguf")
SMALL_Q4_K_M = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
QWEN36_27B_Q4_K_M = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
QWEN36_35B_A3B_Q4_K_M = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
ORNITH_35B_A3B_Q4_K_M = Path("/models/gguf/Ornith-1.5-35B-A3B-Q4_K_M.gguf")

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
    """A block-truthful tensor-info fixture.

    ``byte_shape``/``nbytes`` come from the real GGUF quant layout so the
    synthetic metadata has the same geometry a real ``scan_gguf`` would
    produce; block-quant shapes must therefore be realizable (K a block
    multiple), exactly like a real file.
    """

    from hipengine.quant.gguf import nbytes_for_shape, quant_shape_to_byte_shape

    n_elements = 1
    for dim in shape:
        n_elements *= int(dim)
    if qtype == GGMLQuantizationType.F32:
        byte_shape = tuple(int(dim) for dim in shape)
        nbytes = n_elements * 4
    elif qtype in (GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
        byte_shape = tuple(int(dim) for dim in shape)
        nbytes = n_elements * 2
    else:
        byte_shape = quant_shape_to_byte_shape(shape, qtype)
        nbytes = nbytes_for_shape(shape, qtype)
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=n_elements,
        nbytes=nbytes,
        offset=0,
        data_offset=0,
        byte_shape=byte_shape,
    )


def _config(layer_types: tuple[str, ...]) -> Qwen35GGUFConfig:
    # Geometry is block-truthful: hidden/ffn/ssm-inner are Q4_K/Q8_0 block
    # multiples, the time-step rank is T16-aligned, and the vocabulary is
    # pack8/T16-aligned, so every planned resident is materializable.
    return Qwen35GGUFConfig(
        architecture="qwen35",
        block_count=len(layer_types),
        hidden_size=256,
        vocab_size=32,
        feed_forward_length=256,
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
        ssm_inner_size=256,
        ssm_group_count=2,
        ssm_state_size=4,
        ssm_conv_kernel=2,
        ssm_time_step_rank=16,
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
        "attn_norm": _tensor(f"{prefix}.attn_norm.weight", (256,)),
        "post_attention_norm": _tensor(f"{prefix}.post_attention_norm.weight", (256,)),
        "attn_gate": _tensor(f"{prefix}.attn_gate.weight", (256, 256), gate_type),
        "attn_qkv": _tensor(f"{prefix}.attn_qkv.weight", (272, 256), attn_qkv_type),
        "ssm_a": _tensor(f"{prefix}.ssm_a", (16,)),
        "ssm_alpha": _tensor(f"{prefix}.ssm_alpha.weight", (16, 256), alpha_beta_type),
        "ssm_beta": _tensor(f"{prefix}.ssm_beta.weight", (16, 256), alpha_beta_type),
        "ssm_conv1d": _tensor(f"{prefix}.ssm_conv1d.weight", (272, 2)),
        "ssm_dt_bias": _tensor(f"{prefix}.ssm_dt.bias", (16,)),
        "ssm_norm": _tensor(f"{prefix}.ssm_norm.weight", (4,)),
        "ssm_out": _tensor(f"{prefix}.ssm_out.weight", (256, 256), gate_type),
        "ffn_gate": _tensor(f"{prefix}.ffn_gate.weight", (256, 256), ffn_type),
        "ffn_up": _tensor(f"{prefix}.ffn_up.weight", (256, 256), ffn_type),
        "ffn_down": _tensor(f"{prefix}.ffn_down.weight", (256, 256), ffn_type),
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
        "token_embedding": _tensor("token_embd.weight", (32, 256), embedding_type),
        "output_norm": _tensor("output_norm.weight", (256,)),
        "lm_head": _tensor("token_embd.weight", (32, 256), lm_head_type),
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
    plain_report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", file_type_stamp="MOSTLY_Q4_K_M"
    )
    assert (
        certificate_covers_artifact(
            plain_report.certificate(),
            manifest_fingerprint=ud_like.fingerprint,
            plan_contract=plain_report.plan_contract,
        )
        is False
    )
    # Same intended plan on the matching manifest: covered.
    assert (
        certificate_covers_artifact(
            plain_report.certificate(),
            manifest_fingerprint=plain_report.manifest_fingerprint,
            plan_contract=plain_report.plan_contract,
        )
        is True
    )


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

    certificate_report = preflight_qwen35_gguf_artifact(
        recurrent_q8, backend="hip_gfx1100", file_type_stamp="MOSTLY_Q4_K_M"
    )
    certificate_a = certificate_report.certificate()
    assert (
        certificate_covers_artifact(
            certificate_a,
            manifest_fingerprint=manifest_b.fingerprint,
            plan_contract=certificate_report.plan_contract,
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
            "attn_q": _tensor("blk.8.attn_q.weight", (256, 256), GGMLQuantizationType.Q4_K),
        },
        nextn_tensors={
            "eh_proj": _tensor("blk.8.nextn.eh_proj.weight", (256, 256), GGMLQuantizationType.Q8_0),
            "enorm": _tensor("blk.8.nextn.enorm.weight", (256,)),
            "hnorm": _tensor("blk.8.nextn.hnorm.weight", (256,)),
            "shared_head_norm": _tensor("blk.8.nextn.shared_head_norm.weight", (256,)),
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
                certificate,
                manifest_fingerprint=report.manifest_fingerprint,
                plan_contract=report.plan_contract,
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
# NextN draft admission: native-XL manifest binding + MTP scope gate
# ---------------------------------------------------------------------------


def _native_xl_spoof_info(metadata_extra: dict):
    """Build a minimal info whose draft qtypes EXACTLY match the certified
    native-XL expected map, while the claimed metadata carries the variant and
    pinned digest. The tensor inventory differs from the certified artifact."""

    from types import SimpleNamespace

    from hipengine.loading.qwen35_gguf_nextn import _EXPECTED_QWEN38_NATIVE_XL_QTYPES

    block_id = 4
    config = _config((FULL_ATTENTION, FULL_ATTENTION, FULL_ATTENTION, FULL_ATTENTION))
    tensors = []
    # Draft block tensors typed exactly as the certified map expects.
    suffix_types = {
        "nextn.eh_proj.weight": "Q6_K",
        "attn_q.weight": "Q6_K",
        "attn_k.weight": "Q8_0",
        "attn_v.weight": "Q8_0",
        "attn_output.weight": "Q6_K",
        "ffn_gate.weight": "Q6_K",
        "ffn_up.weight": "Q6_K",
        "ffn_down.weight": "Q6_K",
        "attn_norm.weight": "F32",
        "post_attention_norm.weight": "F32",
        "nextn.enorm.weight": "F32",
        "nextn.hnorm.weight": "F32",
        "nextn.shared_head_norm.weight": "F32",
    }
    for slot_suffix, type_name in suffix_types.items():
        qtype = GGMLQuantizationType[type_name]
        tensors.append(
            _tensor(f"blk.{block_id}.{slot_suffix}", (256, 256), qtype)
        )
    tensors.append(_tensor("token_embd.weight", (32, 256), GGMLQuantizationType.Q4_K))
    tensors.append(_tensor("output_norm.weight", (256,)))
    by_name = {tensor.name: tensor for tensor in tensors}
    full_metadata = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 5,
        "qwen35.embedding_length": 8,
        "qwen35.feed_forward_length": 5,
        "qwen35.context_length": 64,
        "qwen35.attention.head_count": 2,
        "qwen35.attention.head_count_kv": 1,
        "qwen35.attention.key_length": 4,
        "qwen35.attention.value_length": 4,
        "qwen35.rope.dimension_count": 4,
        "qwen35.ssm.inner_size": 16,
        "qwen35.ssm.group_count": 2,
        "qwen35.ssm.state_size": 4,
        "qwen35.ssm.conv_kernel": 2,
        "qwen35.ssm.time_step_rank": 2,
    }
    full_metadata.update(metadata_extra)
    return SimpleNamespace(
        metadata=full_metadata,
        file_type_name="MOSTLY_Q4_K_M",
        tensors=tensors,
        tensor=lambda name: by_name[name],
    ), config


def test_native_xl_variant_manifest_is_recomputed_from_actual_tensors():
    """UD-U1 RED: a foreign artifact that stamps the native-XL variant AND the
    pinned digest is refused because the recomputed manifest of its actual
    tensors does not match — even when every draft qtype coincides with the
    expected map."""

    from hipengine.loading.qwen35_gguf_nextn import (
        QWEN38_NATIVE_XL_OUTPUT_TYPE_MANIFEST_SHA256,
        QWEN38_NATIVE_XL_QUANT_VARIANT,
        validate_qwen35_gguf_nextn_tensor_map,
    )

    info, _config_stub = _native_xl_spoof_info(
        {
            "hipengine.quant.variant": QWEN38_NATIVE_XL_QUANT_VARIANT,
            "hipengine.quant.output_type_manifest_sha256": (
                QWEN38_NATIVE_XL_OUTPUT_TYPE_MANIFEST_SHA256
            ),
        }
    )
    validation = validate_qwen35_gguf_nextn_tensor_map(info)
    joined = "\n".join(validation.dtype_errors)
    assert "does not match the actual tensor type manifest" in joined
    assert "is not the certified native-XL manifest" in joined
    from hipengine.loading.gguf import MissingGGUFTensorError

    with pytest.raises(MissingGGUFTensorError, match="dtype"):
        validation.raise_for_errors()

    # A truthful claim about a foreign manifest is still refused, with the
    # recomputed digest named.
    import hashlib

    actual = hashlib.sha256(
        "\n".join(
            f"{t.name}={t.ggml_type_name}"
            for t in sorted(info.tensors, key=lambda t: t.name)
        ).encode("utf-8")
    ).hexdigest()
    info2, _ = _native_xl_spoof_info(
        {
            "hipengine.quant.variant": QWEN38_NATIVE_XL_QUANT_VARIANT,
            "hipengine.quant.output_type_manifest_sha256": actual,
        }
    )
    validation2 = validate_qwen35_gguf_nextn_tensor_map(info2)
    joined2 = "\n".join(validation2.dtype_errors)
    assert "does not match the actual tensor type manifest" not in joined2
    assert "is not the certified native-XL manifest" in joined2


def test_non_native_xl_variant_validation_unchanged():
    """Without the native-XL variant metadata, validation keeps its per-slot
    dtype contract (no manifest recomputation gate)."""

    from hipengine.loading.qwen35_gguf_nextn import validate_qwen35_gguf_nextn_tensor_map

    info, _config_stub = _native_xl_spoof_info({})
    validation = validate_qwen35_gguf_nextn_tensor_map(info)
    assert not any(
        "output_type_manifest" in error for error in validation.dtype_errors
    )


@pytest.mark.skipif(not UD_Q4_K_M.exists(), reason=f"pinned artifact missing: {UD_Q4_K_M}")
def test_ud_artifact_nextn_draft_materialization_scope_refused(monkeypatch):
    """The UD preset is AR-only: draft materialization is refused before any
    planning or allocation, with the distinct-scope reason."""

    from hipengine.loading.qwen35_gguf_nextn_materialize import (
        materialize_qwen35_gguf_nextn_weights,
    )
    from hipengine.loading import qwen35_gguf_nextn_materialize as nextn_loader
    from hipengine.loading import materialize as host_materialize

    sentinel = _AllocationSentinel(
        "allocator invoked before NextN draft scope refusal"
    )
    monkeypatch.setattr(nextn_loader, "load_host_array_to_device_as_dtype", sentinel)
    monkeypatch.setattr(nextn_loader, "materialize_qwen35_gguf_weight_spec", sentinel)
    monkeypatch.setattr(host_materialize, "malloc", sentinel)
    with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
        materialize_qwen35_gguf_nextn_weights(str(UD_Q4_K_M), backend="hip_gfx1100")
    assert sentinel.calls == []
    message = str(excinfo.value)
    assert "gguf_ud_q4_k_m" in message
    assert "AR-only" in message or "'ar'" in message
    assert "MTP" in message


def test_ud_scope_refusal_is_distinct_from_plain_nextn_path():
    """A plain (unresolved-preset) artifact keeps the historical NextN path:
    no scope refusal is raised by admission for it."""

    from types import SimpleNamespace

    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_PRESET_SCOPE_MTP,
        preflight_qwen35_gguf_artifact,
        resolve_qwen35_gguf_artifact_preset,
    )

    plain_map = _synthetic_model_map()
    assert resolve_qwen35_gguf_artifact_preset(plain_map, file_type_stamp="MOSTLY_Q4_K_M") is None
    # The synthetic UD-shaped manifest never matches a pinned fingerprint, so
    # it stays on the plain lane too — the refusal binds to pinned manifests,
    # not to "looks like UD".
    ud_like_map = _synthetic_model_map(
        attn_qkv_type=GGMLQuantizationType.IQ4_XS,
        ffn_type=GGMLQuantizationType.Q3_K,
    )
    assert resolve_qwen35_gguf_artifact_preset(ud_like_map, file_type_stamp="MOSTLY_Q4_K_M") is None
    report = preflight_qwen35_gguf_artifact(
        ud_like_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,),
        file_type_stamp="MOSTLY_Q4_K_M",
    )
    scope_refusals = [u for u in report.unsupported if u.stage == "scope_refused"]
    assert scope_refusals and scope_refusals[0].operation == QWEN35_GGUF_OP_MTP_NEXTN_DRAFT
    assert not report.preset
    del GGUF_PRESET_SCOPE_MTP


# ---------------------------------------------------------------------------
# Coverage record conventions
# ---------------------------------------------------------------------------


def test_coverage_records_use_existing_registry_layer_names():
    # Every certified record names a concrete registry layer that actually
    # exists in the production registration surface (F3: the previous
    # placeholder layers "moe_selected"/"gdn_chain" never existed; the real
    # selected-expert consumers are registered under moe_linear and the GDN
    # chain under gdn_recurrent_rmsnorm_gate).  The full registration parity
    # is proven in tests/test_qwen35_gguf_consumer_surface_parity.py.
    known_layers = {
        "linear",
        "dense_gemv",
        "embedding",
        "rmsnorm",
        "router_logits",
        "gdn_recurrent_rmsnorm_gate",
        "gdn_prefill_recurrent",
        "linear_attn_conv_decode",
        "linear_attn_conv_prefill",
        "moe_linear",
    }
    for record in (*CERTIFIED_OPERATION_COVERAGE, *CERTIFIED_F32_INPUT_OPERATION_COVERAGE):
        assert record.operation in {
            QWEN35_GGUF_OP_AR_DECODE_C1,
            QWEN35_GGUF_OP_AR_DECODE_ROWS,
            QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,
            QWEN35_GGUF_OP_AR_PREFILL,
            QWEN35_GGUF_OP_EMBEDDING_LOOKUP,
            QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,
        }
        # F3: every record names a REAL registry layer and a concrete
        # (quant, variant) consumer identity — placeholders are not
        # certification.
        assert record.kernel_layer in known_layers
        assert record.kernel_variant
        if record.consumer_module is None:
            assert record.kernel_quant


def test_lm_head_logits_require_an_f32_output_consumer():
    # Every lm_head layout the planner can emit EXCEPT dense_f32 has a
    # registered BF16-activation/F32-output dispatch row (pack8/raw per
    # quant, dense-BF16, Q6_K T16); assert the positive contract on
    # representative types.  dense_f32 is the F3 counterexample: the actual
    # caller supplies BF16 activations and no (dense_f32, bf16, f32) row
    # exists — it is refused by default and certifiable only with a declared
    # F32 input override.
    for lm_head_type in (
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q6_K,
        GGMLQuantizationType.Q4_1,
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
    f32_report = preflight_qwen35_gguf_artifact(
        _synthetic_model_map(lm_head_type=GGMLQuantizationType.F32),
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
    )
    assert f32_report.supported is False
    # And the coverage table records the F32 output dtype for every lm_head
    # consumer.
    for record in CERTIFIED_OPERATION_COVERAGE:
        if record.operation == QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS:
            assert record.output_dtype == "f32"


# ---------------------------------------------------------------------------
# Loader integration: preflight before any device allocation
# ---------------------------------------------------------------------------


class _AllocationSentinel:
    """Recorder that fails loudly if any device allocation is attempted."""

    def __init__(self, message: str = "device allocation attempted"):
        self.message = message
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        raise AssertionError(self.message)


@pytest.mark.skipif(not UD_Q4_K_M.exists(), reason=f"pinned artifact missing: {UD_Q4_K_M}")
def test_ud_materialization_refused_before_any_device_allocation(monkeypatch):
    """UD-U1: the loader aggregates refusals before the first malloc call."""

    from hipengine.loading import materialize as host_materialize
    from hipengine.loading import qwen35_gguf_materialize as loader

    sentinel = _AllocationSentinel(
        "allocator invoked before GGUF admission preflight completed"
    )
    monkeypatch.setattr(loader, "malloc", sentinel)
    monkeypatch.setattr(host_materialize, "malloc", sentinel)
    with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
        materialize_qwen35_gguf_weights(str(UD_Q4_K_M), backend="hip_gfx1100")
    assert sentinel.calls == []
    message = str(excinfo.value)
    assert "gguf_ud_q4_k_m" in message
    for slot in sorted(UD_K_M_REFUSED_SLOTS):
        assert slot in message


@pytest.mark.skipif(not SMALL_Q8_0.exists(), reason=f"pinned artifact missing: {SMALL_Q8_0}")
def test_plain_control_passes_preflight_and_reaches_allocation(monkeypatch):
    """Qualified plain controls keep their rollback: preflight is not a gate
    that blocks them; the first allocation is reached (and fails here only
    because this host has no HIP device)."""

    from hipengine.loading import materialize as host_materialize
    from hipengine.loading import qwen35_gguf_materialize as loader

    sentinel = _AllocationSentinel("allocation-sentinel")
    monkeypatch.setattr(loader, "malloc", sentinel)
    monkeypatch.setattr(host_materialize, "malloc", sentinel)
    with pytest.raises(AssertionError, match="allocation-sentinel"):
        materialize_qwen35_gguf_weights(str(SMALL_Q8_0), backend="hip_gfx1100")
    assert len(sentinel.calls) >= 1


def test_resident_weights_carry_the_artifact_preset_key():
    resident = Qwen35GGUFResidentWeights(
        config=_config((LINEAR_ATTENTION,)),
        root_weights={},
        layers=(),
        backend="hip_gfx1100",
        artifact_preset_key=GGUF_UD_Q4_K_M_PRESET,
    )
    assert resident.artifact_preset_key == GGUF_UD_Q4_K_M_PRESET
    assert Qwen35GGUFResidentWeights(
        config=_config((LINEAR_ATTENTION,)), root_weights={}, layers=[], backend="cpu"
    ).artifact_preset_key is None


def test_runner_policy_identity_binds_artifact_preset(monkeypatch):
    """Same geometry + same stamp + different manifest preset => different
    policy-table identity, so plain-certified policy rows never apply to UD."""

    from types import SimpleNamespace

    from hipengine.kernels.policy import GGUFModelGeometry
    from hipengine.runtime.qwen35_gguf_runner import _gguf_policy_identity

    geometry = GGUFModelGeometry.try_from_config(_config((LINEAR_ATTENTION,)))
    assert geometry is not None
    plain = SimpleNamespace(
        geometry=geometry,
        file_type_name="MOSTLY_Q4_K_M",
        artifact_preset_key=None,
    )
    ud_same_stamp = SimpleNamespace(
        geometry=geometry,
        file_type_name="MOSTLY_Q4_K_M",
        artifact_preset_key=GGUF_UD_Q4_K_M_PRESET,
    )
    ud_ks_same_stamp = SimpleNamespace(
        geometry=geometry,
        file_type_name="MOSTLY_Q4_K_S",
        artifact_preset_key=GGUF_UD_Q4_K_S_PRESET,
    )
    identity_plain = _gguf_policy_identity(plain)
    identity_ud = _gguf_policy_identity(ud_same_stamp)
    assert identity_plain == (geometry, "MOSTLY_Q4_K_M")
    assert identity_ud == (geometry, "MOSTLY_Q4_K_M", GGUF_UD_Q4_K_M_PRESET)
    assert identity_plain != identity_ud
    assert _gguf_policy_identity(ud_ks_same_stamp) == (
        geometry,
        "MOSTLY_Q4_K_S",
        GGUF_UD_Q4_K_S_PRESET,
    )
    # A plain-certified policy table keyed by the historical identity admits
    # the plain control and does NOT admit the same-stamp UD preset.
    table = {(geometry, "MOSTLY_Q4_K_M"): "plain-row"}
    assert table.get(identity_plain) == "plain-row"
    assert table.get(identity_ud) is None


def test_hot_vocab_identity_binds_artifact_preset(monkeypatch, tmp_path):
    """A UD preset never reuses the plain artifact's packaged hot-vocab map."""

    from types import SimpleNamespace

    from hipengine.loading import gguf_mtp_hot_vocab as hot_vocab

    tokenizer_hash = "f" * 64
    monkeypatch.setattr(hot_vocab, "gguf_tokenizer_tokens_sha256", lambda info: tokenizer_hash)
    packaged_name = "qwen38-27b-hot131072-cjk-v1.json"  # real packaged artifact
    packaged = {
        ("qwen35", "Qwen3.8-27B", 65, "MOSTLY_Q4_K_M", tokenizer_hash): packaged_name,
        (
            "qwen35",
            "Qwen3.8-27B",
            65,
            "MOSTLY_Q4_K_M",
            tokenizer_hash,
            GGUF_UD_Q4_K_M_PRESET,
        ): packaged_name,
    }
    monkeypatch.setattr(hot_vocab, "_DEFAULT_HOT_VOCAB_IDENTITIES", packaged)
    info = SimpleNamespace(
        metadata={
            "general.architecture": "qwen35",
            "general.basename": "Qwen3.8-27B",
            "qwen35.block_count": 65,
        },
        file_type_name="MOSTLY_Q4_K_M",
    )
    # Plain artifact: unchanged 5-part identity resolves its packaged map.
    resolved = hot_vocab.default_gguf_hot_vocab_path(info)
    assert resolved is not None and resolved.name == packaged_name
    # Same stamp + tokenizer, UD preset: the plain row's key no longer matches;
    # the preset has its own distinct key, and unknown presets resolve to None.
    ud_resolved = hot_vocab.default_gguf_hot_vocab_path(
        info, artifact_preset_key=GGUF_UD_Q4_K_M_PRESET
    )
    assert ud_resolved is not None and ud_resolved.name == packaged_name
    assert hot_vocab.default_gguf_hot_vocab_path(
        info, artifact_preset_key="gguf_ud_q4_k_s"
    ) is None
    # And the plain artifact does not resolve through the preset-qualified key
    # either (the plain identity has no preset component).
    assert hot_vocab.default_gguf_hot_vocab_path(info) == resolved


def test_preflight_slot_filter_scopes_to_selected_slots():
    map_small = _synthetic_model_map()
    report = preflight_qwen35_gguf_artifact(
        map_small,
        backend="hip_gfx1100",
        slot_filter=("root.output_norm", "layers.0.attn_qkv"),
    )
    assert report.supported
    assert report.covered_slots == 2


@pytest.mark.skipif(not UD_Q4_K_M.exists(), reason=f"pinned artifact missing: {UD_Q4_K_M}")
def test_loader_selected_subset_preflight_passes_on_qualified_subset(monkeypatch):
    """The loader's test/debug selected_slots hook scopes the preflight, so a
    subset that only needs certified slots still materializes on a UD file."""

    from hipengine.loading import materialize as host_materialize
    from hipengine.loading import qwen35_gguf_materialize as loader

    sentinel = _AllocationSentinel("allocation-sentinel")
    monkeypatch.setattr(loader, "malloc", sentinel)
    monkeypatch.setattr(host_materialize, "malloc", sentinel)
    with pytest.raises(AssertionError, match="allocation-sentinel"):
        materialize_qwen35_gguf_weights(
            str(UD_Q4_K_M),
            backend="hip_gfx1100",
            selected_slots=("root.output_norm",),
        )


# ---------------------------------------------------------------------------
# U1 review repair 1: the native-row owner contract is enforced at the real
# loader and runner entries, not only in isolated preflight records
# ---------------------------------------------------------------------------


def _fake_allocation(shape: tuple[int, ...], dtype: DType):
    """A CPU allocation stand-in exposing a real-dtype Tensor like the loader."""

    from types import SimpleNamespace

    return SimpleNamespace(
        tensor=Tensor.from_handle(0x5000, shape, dtype, Device("hip", 0)),
    )


def _resident_with_alpha_beta(layout: str, *, raw_dtype: DType | None):
    """One-layer resident whose alpha/beta carry the given layout/allocations."""

    from types import MappingProxyType

    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_DENSE_BF16,
        Qwen35GGUFDeviceWeight,
        Qwen35GGUFResidentLayerWeights,
        Qwen35GGUFResidentWeights,
        Qwen35GGUFWeightSpec,
    )

    def spec_for(slot: str) -> Qwen35GGUFWeightSpec:
        if slot in ("ssm_alpha", "ssm_beta"):
            return Qwen35GGUFWeightSpec(
                slot_path=f"layers.0.{slot}",
                source=_tensor(f"blk.0.{slot}.weight", (2, 8)),
                quant_key="bf16" if layout == LAYOUT_DENSE_BF16 else layout,
                layout=layout,
                allocation_names=("raw",) if raw_dtype is not None else ("tiles",),
            )
        raise AssertionError(slot)

    allocations = {}
    if raw_dtype is not None:
        allocations["raw"] = _fake_allocation((2, 8), raw_dtype)
    else:
        allocations["tiles"] = _fake_allocation((2, 1, 8), DType.INT8)
    weight = Qwen35GGUFDeviceWeight(
        spec=spec_for("ssm_alpha"),
        allocations=MappingProxyType(allocations),
        backend="hip_gfx1100",
    )
    beta = Qwen35GGUFDeviceWeight(
        spec=spec_for("ssm_beta"),
        allocations=MappingProxyType(dict(allocations)),
        backend="hip_gfx1100",
    )
    return Qwen35GGUFResidentWeights(
        config=_config((LINEAR_ATTENTION,)),
        root_weights={},
        layers=(
            Qwen35GGUFResidentLayerWeights(
                layer_id=0,
                layer_type=LINEAR_ATTENTION,
                weights=MappingProxyType({"ssm_alpha": weight, "ssm_beta": beta}),
            ),
        ),
        backend="hip_gfx1100",
    )


def test_native_row_binding_errors_name_every_invalid_alpha_beta_owner():
    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_DENSE_BF16,
        LAYOUT_DENSE_F32,
        LAYOUT_GGUF_Q8_0_T16,
        LAYOUT_RAW_GGUF,
    )

    valid = _resident_with_alpha_beta(LAYOUT_DENSE_BF16, raw_dtype=DType.BF16)
    assert qwen35_gguf_native_row_binding_errors(valid) == ()

    raw_q8 = _resident_with_alpha_beta(LAYOUT_RAW_GGUF, raw_dtype=DType.INT8)
    raw_errors = qwen35_gguf_native_row_binding_errors(raw_q8)
    assert len(raw_errors) == 2
    assert all("layers.0.ssm_alpha" in e or "layers.0.ssm_beta" in e for e in raw_errors)
    assert all("ar_decode_native_rows" in e for e in raw_errors)

    sole_t16 = _resident_with_alpha_beta(LAYOUT_GGUF_Q8_0_T16, raw_dtype=None)
    t16_errors = qwen35_gguf_native_row_binding_errors(sole_t16)
    assert len(t16_errors) == 2
    assert all("raw" in e for e in t16_errors)

    dense_f32 = _resident_with_alpha_beta(LAYOUT_DENSE_F32, raw_dtype=DType.FP32)
    f32_errors = qwen35_gguf_native_row_binding_errors(dense_f32)
    assert len(f32_errors) == 2
    assert all("dense_f32" in e for e in f32_errors)


def _cpu_allocation_fakes(monkeypatch):
    """Let the real loader materialize real allocation records on CPU."""

    from hipengine.loading import materialize as host_materialize
    from hipengine.loading import qwen35_gguf_materialize as loader

    copied: list[tuple[str, int]] = []

    class _FakeBuffer:
        def __init__(self, nbytes: int, ptr: int):
            self.nbytes = int(nbytes)
            self.ptr = int(ptr)

    counter = {"ptr": 0x10000}

    def fake_malloc(nbytes, runtime=None):
        counter["ptr"] += 0x1000
        return _FakeBuffer(nbytes, counter["ptr"])

    def fake_copy(buffer, host_array, nbytes, runtime=None):
        copied.append((getattr(host_array, "name", "?"), int(nbytes)))

    monkeypatch.setattr(host_materialize, "malloc", fake_malloc)
    monkeypatch.setattr(host_materialize, "copy_host_to_device", fake_copy)
    monkeypatch.setattr(loader, "malloc", fake_malloc)
    return copied


def _materialize_fixture_on_cpu(path: Path, monkeypatch, **kwargs):
    from hipengine.loading.qwen35_gguf_materialize import materialize_qwen35_gguf_weights

    _cpu_allocation_fakes(monkeypatch)
    return materialize_qwen35_gguf_weights(str(path), backend="hip_gfx1100", **kwargs)


def _native_entry_session(resident, *, scratch_owner):
    """A real Qwen35GGUFResidentSession frame without device construction."""

    from types import SimpleNamespace

    import numpy as np

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    session = object.__new__(Qwen35GGUFResidentSession)
    session.model_path = "<fixture>"
    session.runtime = object()
    session.backend = "hip_gfx1100"
    session.max_batch_size = 8
    session._position = 0
    session.runner = SimpleNamespace(
        weights=resident,
        backend="hip_gfx1100",
        hidden_size=256,
        vocab_size=64,
        fp16_recurrent_state=False,
    )
    session._target_scratch_owner = scratch_owner
    # Truthful raw-buffer geometry for the mocked native allocation owner.
    cfg = resident.config
    cursor = 0x1000000
    def buffer(nbytes):
        nonlocal cursor
        ptr = cursor
        cursor += int(nbytes) + 256
        return SimpleNamespace(ptr=ptr, nbytes=int(nbytes))
    qkv = 2 * cfg.ssm_group_count * cfg.ssm_state_size + cfg.ssm_inner_size
    scratch_owner.slot_count = 8
    scratch_owner.recurrent_zero = np.zeros(1, dtype=np.float32)
    for name, width in (("norm", cfg.hidden_size * 2), ("post_norm", cfg.hidden_size * 2),
                        ("linear_qkv", qkv * 2), ("linear_z", cfg.ssm_inner_size * 2),
                        ("linear_alpha", cfg.ssm_time_step_rank * 2), ("linear_beta", cfg.ssm_time_step_rank * 2),
                        ("conv_out", qkv * 4), ("recurrent_out", cfg.ssm_inner_size * 4),
                        ("recurrent_bf16", cfg.ssm_inner_size * 2)):
        setattr(scratch_owner, name, buffer(8 * width))
    scratch_owner.layer_conv_states = tuple(buffer(8 * qkv * cfg.ssm_conv_kernel * 4) for _ in resident.layers)
    scratch_owner.layer_recurrent_states = tuple(buffer(8 * cfg.ssm_inner_size * cfg.ssm_state_size * 4) for _ in resident.layers)
    session._token_buf = buffer(8 * 8)
    session._hidden_a = buffer(8 * cfg.hidden_size * 2)
    session._hidden_b = buffer(8 * cfg.hidden_size * 2)
    session._logits_buf = buffer(8 * cfg.vocab_size * 4)
    session._native_cu_seqlens_buf = buffer(9 * 4)
    session._native_state_indices_buf = buffer(8 * 8)
    session._native_token_ids_host = object()
    return session


class _PositionOwnerSentinel:
    """Records set_full_attention_positions calls; device work must not happen."""

    def __init__(self, rows: int = 2):
        import numpy as np

        self.position_host = np.zeros(rows, dtype=np.int64)
        self.calls: list[tuple] = []

    def set_full_attention_positions(self, positions, runtime):
        self.calls.append(tuple(positions))


@pytest.mark.skipif(not SMALL_Q8_0.exists(), reason=f"pinned artifact missing: {SMALL_Q8_0}")
def test_materializer_requested_operations_bind_native_rows_before_allocation(monkeypatch):
    """Requesting ar_decode_native_rows at load refuses an unqualified artifact
    through the aggregated preflight BEFORE any allocation (the requested mode
    is bound to pre-allocation admission, not only to the runtime entry)."""

    from hipengine.loading import materialize as host_materialize
    from hipengine.loading import qwen35_gguf_materialize as loader

    sentinel = _AllocationSentinel("allocator invoked before native-row admission")
    monkeypatch.setattr(loader, "malloc", sentinel)
    monkeypatch.setattr(host_materialize, "malloc", sentinel)
    with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
        materialize_qwen35_gguf_weights(
            str(SMALL_Q8_0),
            backend="hip_gfx1100",
            requested_operations=(*DEFAULT_AR_OPERATIONS, QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS),
        )
    assert sentinel.calls == []
    message = str(excinfo.value)
    assert "ssm_alpha" in message and "ssm_beta" in message
    assert "native" in message.lower()


def test_step_rows_native_refuses_unbound_native_owner_before_state_or_device(monkeypatch, tmp_path):
    """RED (U1 review repair 1): a real loader-materialized resident whose
    alpha/beta are dense F32 (uncontracted plain lane) must be refused at the
    real step_rows_native entry BEFORE positions mutate or any device call."""

    from tests._qwen35_gguf_fixture import (
        default_fixture_tensors,
        fixture_metadata,
        write_qwen35_gguf,
    )

    path = tmp_path / "f32-alpha-beta.gguf"
    write_qwen35_gguf(path, default_fixture_tensors(1), fixture_metadata(1))
    resident = _materialize_fixture_on_cpu(path, monkeypatch)
    owner = _PositionOwnerSentinel()
    session = _native_entry_session(resident, scratch_owner=owner)

    with pytest.raises(ValueError) as excinfo:
        session.step_rows_native((11, 22))
    message = str(excinfo.value)
    assert "ar_decode_native_rows" in message
    assert "partial certificate" in message
    assert owner.calls == [], "positions mutated before the native-row binding check"
    assert int(owner.position_host[0]) == 0


def test_capture_native_rows_graph_refuses_unbound_native_owner_before_device(monkeypatch, tmp_path):
    from tests._qwen35_gguf_fixture import (
        default_fixture_tensors,
        fixture_metadata,
        write_qwen35_gguf,
    )

    path = tmp_path / "f32-alpha-beta-capture.gguf"
    write_qwen35_gguf(path, default_fixture_tensors(1), fixture_metadata(1))
    resident = _materialize_fixture_on_cpu(path, monkeypatch)
    owner = _PositionOwnerSentinel()
    session = _native_entry_session(resident, scratch_owner=owner)

    with pytest.raises(ValueError) as excinfo:
        session.capture_native_rows_graph(rows=2, max_context_len=64)
    assert "ar_decode_native_rows" in str(excinfo.value)
    assert owner.calls == []


def test_step_rows_native_admits_contracted_bf16_owner_and_reaches_device_entry(monkeypatch, tmp_path):
    """The valid owner (raw-IQ manifest contracts F32 alpha/beta to dense BF16)
    passes the entry check and proceeds into device staging."""

    from tests._qwen35_gguf_fixture import (
        fixture_metadata,
        linear_attention_layer_slots,
        write_qwen35_gguf,
    )
    from hipengine.quant.gguf import GGMLQuantizationType

    tensors = [
        ("token_embd.weight", (64, 256), GGMLQuantizationType.Q8_0),
        ("output_norm.weight", (256,), GGMLQuantizationType.F32),
    ]
    tensors.extend(
        linear_attention_layer_slots(
            0,
            projection_type=GGMLQuantizationType.Q4_K,
            alpha_beta_type=GGMLQuantizationType.F32,
            attn_qkv_type=GGMLQuantizationType.IQ4_XS,
        )
    )
    path = tmp_path / "raw-iq-contracted.gguf"
    write_qwen35_gguf(path, tensors, fixture_metadata(1))
    resident = _materialize_fixture_on_cpu(path, monkeypatch,
        requested_operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,))
    assert qwen35_gguf_native_row_binding_errors(resident) == ()

    owner = _PositionOwnerSentinel()
    session = _native_entry_session(resident, scratch_owner=owner)

    def _fail_device_entry(*args, **kwargs):
        raise RuntimeError("reached-device-entry")

    session._native_compact_scratch = _fail_device_entry
    with pytest.raises(RuntimeError, match="reached-device-entry"):
        session.step_rows_native((11, 22))
    # The position owner was staged before the (mocked) device scratch step,
    # proving the entry gate admitted a genuinely contracted resident.
    assert owner.calls == [(0, 0)]


# ---------------------------------------------------------------------------
# U1 review repair 2: rank-3 IQ3_XXS selected experts keep their registered
# MoE consumers; rank-2 dense IQ3_XXS stays refused
# ---------------------------------------------------------------------------


def _moe_layer_tensors(layer_id: int, *, expert_type: GGMLQuantizationType) -> dict:
    prefix = f"blk.{layer_id}"
    tensors = dict(_linear_layer_tensors(layer_id))
    tensors.pop("ffn_gate")
    tensors.pop("ffn_up")
    tensors.pop("ffn_down")
    tensors.update(
        {
            "ffn_gate_inp": _tensor(f"{prefix}.ffn_gate_inp.weight", (4, 256)),
            "ffn_gate_inp_shexp": _tensor(f"{prefix}.ffn_gate_inp_shexp.weight", (256, 256)),
            "ffn_gate_exps": _tensor(f"{prefix}.ffn_gate_exps.weight", (4, 256, 256), expert_type),
            "ffn_up_exps": _tensor(f"{prefix}.ffn_up_exps.weight", (4, 256, 256), expert_type),
            "ffn_down_exps": _tensor(f"{prefix}.ffn_down_exps.weight", (4, 256, 256), expert_type),
            "ffn_gate_shexp": _tensor(f"{prefix}.ffn_gate_shexp.weight", (256, 256), GGMLQuantizationType.Q8_0),
            "ffn_up_shexp": _tensor(f"{prefix}.ffn_up_shexp.weight", (256, 256), GGMLQuantizationType.Q8_0),
            "ffn_down_shexp": _tensor(f"{prefix}.ffn_down_shexp.weight", (256, 256), GGMLQuantizationType.Q8_0),
        }
    )
    return tensors


def _synthetic_moe_model_map(
    *,
    expert_type: GGMLQuantizationType = GGMLQuantizationType.IQ3_XXS,
) -> Qwen35GGUFModelMap:
    from types import MappingProxyType

    config = _config((LINEAR_ATTENTION,))
    object.__setattr__(config, "architecture", "qwen35moe")
    object.__setattr__(config, "expert_count", 4)
    object.__setattr__(config, "expert_used_count", 2)
    object.__setattr__(config, "expert_feed_forward_length", 256)
    object.__setattr__(config, "expert_shared_feed_forward_length", 256)
    root = {
        "token_embedding": _tensor("token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0),
        "output_norm": _tensor("output_norm.weight", (256,)),
        "lm_head": _tensor("token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0),
    }
    layers = tuple(
        Qwen35GGUFLayerMap(
            layer_id=layer_id,
            layer_type=LINEAR_ATTENTION,
            tensors=MappingProxyType(_moe_layer_tensors(layer_id, expert_type=expert_type)),
        )
        for layer_id in range(1)
    )
    return Qwen35GGUFModelMap(
        config=config,
        root_tensors=MappingProxyType(root),
        layers=layers,
        validation=None,
    )


def test_rank3_iq3_xxs_selected_experts_keep_their_registered_moe_consumers():
    """U1 regression: the moe_experts whitelist omitted rank-3 IQ3_XXS even
    though the materializer keeps it raw and gguf_iq_gemv registers selected
    gguf_iq3_xxs moe_linear consumers on both HIP backends."""

    report = preflight_qwen35_gguf_artifact(
        _synthetic_moe_model_map(),
        backend="hip_gfx1100",
    )
    assert report.unsupported == (), report.render_refusals()
    assert report.supported
    expert_records = [
        record
        for record in report.qualified_records
        if record.role_class == "moe_experts" and "IQ3_XXS" in record.source_ggml_types
    ]
    assert expert_records, "no certified IQ3_XXS expert record"
    # The contract is the raw rank-3 selected consumer family, not a dense
    # rank-2 T16/X8 repack: only the raw layout carries IQ3_XXS.
    iq3_layouts = {record.resident_layout for record in expert_records}
    assert iq3_layouts == {LAYOUT_RAW_GGUF}
    assert {record.kernel_layer for record in expert_records} == {"moe_linear"}
    assert all(record.input_dtype == "bf16" and record.output_dtype == "bf16" for record in expert_records)
    assert {record.rows_scope for record in expert_records} == {
        "rows_1_8_row_local",
        "prefill_rows",
    }
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        backend_report = preflight_qwen35_gguf_artifact(
            _synthetic_moe_model_map(), backend=backend
        )
        assert backend_report.supported, backend_report.render_refusals()


def test_rank2_dense_iq3_xxs_is_refused_not_silently_supported():
    from types import MappingProxyType

    config = _config((LINEAR_ATTENTION,))
    root = {
        "token_embedding": _tensor("token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0),
        "output_norm": _tensor("output_norm.weight", (256,)),
        "lm_head": _tensor("token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0),
    }
    tensors = dict(_linear_layer_tensors(0))
    tensors["ffn_gate"] = _tensor(
        "blk.0.ffn_gate.weight", (256, 256), GGMLQuantizationType.IQ3_XXS
    )
    dense_iq3_map = Qwen35GGUFModelMap(
        config=config,
        root_tensors=MappingProxyType(root),
        layers=(
            Qwen35GGUFLayerMap(
                layer_id=0,
                layer_type=LINEAR_ATTENTION,
                tensors=MappingProxyType(tensors),
            ),
        ),
        validation=None,
    )
    report = preflight_qwen35_gguf_artifact(dense_iq3_map, backend="hip_gfx1100")
    assert report.supported is False
    refusals = [u for u in report.unsupported if u.slot_path == "layers.0.ffn_gate"]
    assert refusals and refusals[0].stage == "planner_refused"
    assert "rank-3" in refusals[0].reason
    with pytest.raises(Qwen35GGUFAdmissionError):
        report.raise_for_errors()


# ---------------------------------------------------------------------------
# U1 review repair 3: admission binds the concrete backend and shape/repack
# materializability prerequisites; all failures aggregate before allocation
# ---------------------------------------------------------------------------


def test_unknown_backend_gets_no_certificate():
    """A syntactically free-form backend string is not registration evidence:
    admission only certifies concrete registered hardware backend keys."""

    report_or_error = None
    with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
        report_or_error = preflight_qwen35_gguf_artifact(
            _synthetic_model_map(), backend="nonexistent"
        )
    assert report_or_error is None
    assert "nonexistent" in str(excinfo.value)
    assert "backend" in str(excinfo.value).lower()
    # The cpu_reference package is not a GGUF device materialization backend.
    with pytest.raises(Qwen35GGUFAdmissionError):
        preflight_qwen35_gguf_artifact(_synthetic_model_map(), backend="cpu_reference")


def test_registered_backend_keys_are_the_concrete_admission_surface():
    from hipengine.kernels.backends import CUDA_BACKEND_TARGET_ARCH, HIP_BACKEND_TARGET_ARCH

    # Only backends whose packages declare concrete GGUF consumer layers can
    # earn a positive certificate; both HIP peers declare the full surface.
    # cuda_sm120a is the scaffold counterexample (see
    # test_scaffold_cuda_backend_gets_no_gguf_consumer_certificate).
    for backend in tuple(HIP_BACKEND_TARGET_ARCH):
        report = preflight_qwen35_gguf_artifact(
            _synthetic_model_map(),
            backend=backend,
            operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
        )
        assert report.backend == backend
        assert report.supported, report.render_refusals()
    for backend in tuple(CUDA_BACKEND_TARGET_ARCH):
        report = preflight_qwen35_gguf_artifact(
            _synthetic_model_map(),
            backend=backend,
            operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
        )
        assert report.supported is False


def test_unmaterializable_q6_head_shape_is_refused_before_allocation(monkeypatch, tmp_path):
    """A Q6_K head of shape (257, 256) with decode_repack=True plans a T16
    resident that repack_gguf_q6_k_tile16 rejects (N % 16 != 0). The preflight
    must aggregate that refusal BEFORE the loader allocates anything."""

    from tests._qwen35_gguf_fixture import (
        fixture_metadata,
        linear_attention_layer_slots,
        write_qwen35_gguf,
    )
    from hipengine.loading import materialize as host_materialize
    from hipengine.loading import qwen35_gguf_materialize as loader
    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf import GGMLQuantizationType

    tensors = [
        ("token_embd.weight", (257, 256), GGMLQuantizationType.Q6_K),
        ("output_norm.weight", (256,), GGMLQuantizationType.F32),
    ]
    tensors.extend(linear_attention_layer_slots(0, projection_type=GGMLQuantizationType.Q4_K))
    path = tmp_path / "q6-head-257.gguf"
    write_qwen35_gguf(path, tensors, fixture_metadata(1))

    reader = GGUFReader(path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    report = preflight_qwen35_gguf_artifact(
        model_map,
        backend="hip_gfx1100",
        file_type_stamp=reader.info.file_type_name,
        decode_repack=True,
    )
    assert report.supported is False
    refusals = [u for u in report.unsupported if u.slot_path == "root.lm_head"]
    assert refusals, "misaligned T16 head plan was not refused"
    assert all(u.stage == "planner_refused" for u in refusals)
    assert "out_features must be positive and divisible by 16" in refusals[0].reason
    with pytest.raises(Qwen35GGUFAdmissionError):
        report.raise_for_errors()

    # Loader-level: the refusal happens before ANY device allocation (the old
    # behavior failed inside repack after earlier root allocations).
    sentinel = _AllocationSentinel("allocator invoked before T16 alignment refusal")
    monkeypatch.setattr(loader, "malloc", sentinel)
    monkeypatch.setattr(host_materialize, "malloc", sentinel)
    with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
        materialize_qwen35_gguf_weights(str(path), backend="hip_gfx1100", decode_repack=True)
    assert sentinel.calls == []
    assert "root.lm_head" in str(excinfo.value)


def test_misaligned_pack8_projection_is_refused_before_allocation():
    """Same aggregate-before-allocation contract for the Q4 pack8 tile check:
    a hand-built map whose ffn_up output width breaks pack8 alignment is
    refused by the resident prerequisite validator before byte accounting."""

    from types import MappingProxyType

    config = _config((LINEAR_ATTENTION,))
    root = {
        "token_embedding": _tensor("token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0),
        "output_norm": _tensor("output_norm.weight", (256,)),
        "lm_head": _tensor("token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0),
    }
    tensors = dict(_linear_layer_tensors(0))
    # out=300 breaks pack8 tile alignment (out % 8 != 0); the payload bytes are
    # irrelevant because the refusal fires in the metadata accounting stage.
    tensors["ffn_up"] = _tensor(
        "blk.0.ffn_up.weight", (300, 256), GGMLQuantizationType.Q4_K
    )
    misaligned_map = Qwen35GGUFModelMap(
        config=config,
        root_tensors=MappingProxyType(root),
        layers=(
            Qwen35GGUFLayerMap(
                layer_id=0,
                layer_type=LINEAR_ATTENTION,
                tensors=MappingProxyType(tensors),
            ),
        ),
        validation=None,
    )
    report = preflight_qwen35_gguf_artifact(
        misaligned_map,
        backend="hip_gfx1100",
        decode_repack=False,
    )
    assert report.supported is False
    refusals = [u for u in report.unsupported if u.slot_path == "layers.0.ffn_up"]
    assert refusals and all(u.stage == "planner_refused" for u in refusals)
    assert "out_features must be positive and divisible by 8" in refusals[0].reason
    with pytest.raises(Qwen35GGUFAdmissionError):
        report.raise_for_errors()


def test_coverage_families_are_registered_consumers_not_just_valid_keys():
    """Syntactically valid four-axis keys are not registration evidence: the
    concrete consumer keys named by the certified coverage records must be
    registrable through the production registrars (the same functions the
    backend package and the runtime dispatcher call), and land in the
    registry under the exact four-axis keys the records name.  F3: no skip
    for placeholder records — every record is concrete and is checked on
    BOTH GGUF backends (the declaration-declared consumer surface), plus
    the direct-wrapper consumer (raw rank-3 Q4_K selected experts) exists
    as a module symbol.  The complete no-skip parity (all records, both
    row-mode variants, both backends) lives in
    tests/test_qwen35_gguf_consumer_surface_parity.py; this test keeps the
    in-file invariant independently of that module."""

    import importlib

    iq_gemv = pytest.importorskip(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv",
        reason="hip_gfx1100 IQ GEMV package not importable on this host",
    )
    dense_gemv = pytest.importorskip(
        "hipengine.kernels.hip_gfx1100.linear.dense_gemv",
        reason="hip_gfx1100 dense GEMV package not importable on this host",
    )
    # Test isolation restores the collection-time registry baseline after each
    # test, so re-run the production registrars (idempotently: skip keys that
    # are already registered) exactly as the backend package / runtime
    # dispatcher would — the same registrar set the parity test uses.
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import register_gguf_ops
    from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
        register_qwen35_linear_attn_conv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
        register_qwen35_linear_attn_gdn_kernels,
    )
    from hipengine.kernels.hip_gfx1100.moe.router import register_qwen35_router_kernels
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        register_gguf_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        register_gguf_k_t16_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q3_k_gemv import (
        register_gguf_q3_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        register_gguf_q4_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
        register_gguf_q6_k_embedding_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        register_gguf_q6_k_t16_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
        register_gguf_q8_0_t16_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        register_gguf_t16_selected_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
        register_gguf_x8_selected_gemv_kernels,
    )
    from hipengine.kernels.registry import DuplicateKernelError

    for registrar in (
        iq_gemv.register_gguf_iq_gemv_kernels,
        dense_gemv.register_dense_gemv_kernels,
        register_gguf_k_gemv_kernels,
        register_gguf_q4_k_gemv_kernels,
        register_gguf_q3_k_gemv_kernels,
        register_gguf_x8_selected_gemv_kernels,
        register_gguf_q8_0_t16_gemv_kernels,
        register_gguf_t16_selected_gemv_kernels,
        register_gguf_q6_k_t16_gemv_kernels,
        register_gguf_k_t16_selected_prefill_kernels,
        register_gguf_q6_k_embedding_kernels,
        register_gguf_ops,
        register_qwen35_linear_attn_conv_kernels,
        register_qwen35_linear_attn_gdn_kernels,
        register_qwen35_router_kernels,
    ):
        try:
            registrar(replace=False)
        except DuplicateKernelError:
            pass
    from hipengine.kernels.registry import KernelKey, is_registered, registered_keys

    registered = set(registered_keys())
    checked = 0
    for record in CERTIFIED_OPERATION_COVERAGE:
        if record.consumer_module is not None:
            module = importlib.import_module(record.consumer_module)
            assert hasattr(module, str(record.consumer_symbol)), record
            checked += 1
            continue
        for variant in (
            record.kernel_variant,
            *( [record.kernel_variant_rows_many] if record.kernel_variant_rows_many else [] ),
        ):
            key = KernelKey(
                "hip_gfx1100",
                record.kernel_layer,
                record.kernel_quant,
                variant,
            )
            assert key in registered or is_registered(key), (
                f"coverage record names an unregistered consumer: {key} "
                f"(operation={record.operation} role={record.role_class})"
            )
            checked += 1
    assert checked > len(CERTIFIED_OPERATION_COVERAGE) // 2, (
        "the no-placeholder invariant regressed: most records were skipped"
    )
    # And the IQ3_XXS selected-expert family named by repair 2 is genuinely
    # registered under moe_linear.
    assert any(
        key.layer == "moe_linear" and key.quant == "gguf_iq3_xxs"
        for key in registered
    )


def test_partial_preflight_certificate_does_not_cover_full_artifact():
    """U1 repair 4: a slot-filtered preflight certifies exactly the slots it
    checked; the certificate must not read as full-artifact coverage."""

    plain_map = _synthetic_model_map()
    report = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        slot_filter=("root.output_norm",),
    )
    assert report.supported is True
    assert report.covered_slots == 1
    certificate = report.certificate()
    assert certificate.slot_filter == ("root.output_norm",)
    manifest = build_qwen35_gguf_role_manifest(plain_map)
    full_report = preflight_qwen35_gguf_artifact(plain_map, backend="hip_gfx1100")

    # Same manifest identity, but the caller now intends the FULL artifact:
    # the one-slot certificate does not cover it (the intended plan here is
    # the full artifact's own contract).
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=None,
            plan_contract=full_report.plan_contract,
        )
        is False
    )
    # A different partial use is not covered either (intended contract built
    # by a real preflight over that other subset).
    other_report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", slot_filter=("root.lm_head",)
    )
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=("root.lm_head",),
            plan_contract=other_report.plan_contract,
        )
        is False
    )
    # The exact checked subset is covered (debug/subset loading preserved).
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=("root.output_norm",),
            plan_contract=report.plan_contract,
        )
        is True
    )

    # Conversely, a full-artifact certificate covers a one-slot debug load:
    # the intended subset contract's records verify per slot.
    full_certificate = full_report.certificate()
    assert full_certificate.slot_filter is None
    assert (
        certificate_covers_artifact(
            full_certificate,
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=("root.output_norm",),
            plan_contract=report.plan_contract,
        )
        is True
    )


def test_empty_slot_filter_certificate_covers_only_the_empty_use():
    plain_map = _synthetic_model_map()
    manifest = build_qwen35_gguf_role_manifest(plain_map)
    report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", slot_filter=()
    )
    assert report.supported is True
    assert report.covered_slots == 0
    certificate = report.certificate()
    assert certificate.slot_filter == ()
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=(),
            plan_contract=report.plan_contract,
        )
        is True
    )
    # A real one-slot use (with its own preflight contract) is not covered by
    # the empty certificate.
    one_report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", slot_filter=("root.output_norm",)
    )
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=("root.output_norm",),
            plan_contract=one_report.plan_contract,
        )
        is False
    )


def test_certificate_binds_the_effective_plan_contract():
    """A contraction-enabled certificate must not be reusable on an
    uncontracted plan (and vice versa); the contract is the certificate's
    effective resident/operation contract, not a caller claim."""

    from hipengine.loading.qwen35_gguf_admission import Qwen35GGUFPlanContract

    plain_map = _synthetic_model_map()
    manifest = build_qwen35_gguf_role_manifest(plain_map)

    contracted = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", contract_f32_linear=True
    )
    uncontracted = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", contract_f32_linear=False
    )
    cert_contracted = contracted.certificate()
    assert cert_contracted.plan_contract is not None
    assert cert_contracted.plan_contract.contract_f32_linear is True
    assert uncontracted.plan_contract.contract_f32_linear is False

    # Same manifest identity, same backend, but the intended plan is the
    # uncontracted one: the contraction-enabled certificate must refuse it.
    assert (
        certificate_covers_artifact(
            cert_contracted,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=uncontracted.plan_contract,
        )
        is False
    )
    # And the exact contract is covered.
    assert (
        certificate_covers_artifact(
            cert_contracted,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=cert_contracted.plan_contract,
        )
        is True
    )

    # decode_repack is part of the contract: a repack-vetoed plan is not the
    # certified plan even when the veto was implicit.
    vetoed = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", repack_veto=True
    )
    assert vetoed.plan_contract.decode_repack is False
    assert (
        certificate_covers_artifact(
            cert_contracted,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=vetoed.plan_contract,
        )
        is False
    )

    # A legacy certificate without contract metadata cannot verify a plan
    # contract at all (fail closed).
    from dataclasses import replace as dataclass_replace

    legacy = dataclass_replace(cert_contracted, plan_contract=None)
    assert (
        certificate_covers_artifact(
            legacy,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=cert_contracted.plan_contract,
        )
        is False
    )
    assert Qwen35GGUFPlanContract is not None


def test_plan_contract_records_effective_operations_and_slot_scope():
    plain_map = _synthetic_model_map()
    report = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_PREFILL, QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS),
        slot_filter=("root.lm_head", "layers.0.attn_qkv"),
    )
    contract = report.plan_contract
    assert contract is not None
    assert contract.operations == (
        QWEN35_GGUF_OP_AR_PREFILL,
        QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,
    )
    assert contract.slot_filter == ("layers.0.attn_qkv", "root.lm_head")
    certificate = report.certificate()
    assert certificate.plan_contract == contract
    # as_dict exports the scoped contract for audit trails.
    exported = certificate.as_dict()
    assert exported["slot_filter"] == ["layers.0.attn_qkv", "root.lm_head"]
    assert exported["plan_contract"]["contract_f32_linear"] in (True, False)


# ---------------------------------------------------------------------------
# U1 review repair round 3 (F4): certificates bind the ACTUAL planned resident
# contract (canonical per-slot records + digest), not caller kwargs or
# env-name lists; coverage approval requires the intended plan contract.
# ---------------------------------------------------------------------------


def _q4_k_moe_map():
    return _synthetic_moe_model_map(expert_type=GGMLQuantizationType.Q4_K)


def test_certificate_coverage_requires_the_intended_plan_contract():
    """RED (F4-a): operation coverage must REQUIRE the intended plan
    contract; a legacy/hand-built certificate without plan metadata fails
    closed, and the source-identity-only check is distinct and can never
    authorize operations."""

    from dataclasses import replace as dataclass_replace

    from hipengine.loading.qwen35_gguf_admission import (
        certificate_matches_artifact_identity,
    )

    plain_map = _synthetic_model_map()
    report = preflight_qwen35_gguf_artifact(plain_map, backend="hip_gfx1100")
    certificate = report.certificate()
    manifest = build_qwen35_gguf_role_manifest(plain_map)

    # Omitting the intended plan cannot authorize operations.
    with pytest.raises(TypeError):
        certificate_covers_artifact(
            certificate, manifest_fingerprint=manifest.fingerprint
        )

    # A certificate without recorded plan metadata fails closed even when an
    # intended contract is supplied.
    legacy = dataclass_replace(certificate, plan_contract=None)
    assert (
        certificate_covers_artifact(
            legacy,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=report.plan_contract,
        )
        is False
    )

    # The identity helper binds the source but is NOT operation approval: it
    # returns True for a certificate that covers no verifiable plan.
    assert (
        certificate_matches_artifact_identity(
            legacy, manifest_fingerprint=manifest.fingerprint
        )
        is True
    )
    assert (
        certificate_matches_artifact_identity(
            legacy, manifest_fingerprint="0" * 64
        )
        is False
    )


def test_env_selected_gate_up_x8_changes_the_recorded_resident_contract(monkeypatch):
    """RED (F4-b): with decode_repack on, HIPENGINE_GGUF_SELECTED_GATE_UP_X8
    flips the actual rank-3 Q4_K gate/up residents between T16 and X8 while
    the source shape/type/hash stay identical. The recorded contract must
    follow the ACTUAL planned residents (records + digest differ), and a
    certificate minted under one env must refuse the other env's intended
    plan."""

    from hipengine.loading.qwen35_gguf_materialize import (
        HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV,
        LAYOUT_GGUF_Q4_K_X8,
    )

    moe_map = _q4_k_moe_map()
    manifest = build_qwen35_gguf_role_manifest(moe_map)
    monkeypatch.delenv(HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV, raising=False)
    off_report = preflight_qwen35_gguf_artifact(
        moe_map, backend="hip_gfx1100", decode_repack=True
    )
    monkeypatch.setenv(HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV, "1")
    on_report = preflight_qwen35_gguf_artifact(
        moe_map, backend="hip_gfx1100", decode_repack=True
    )
    assert off_report.supported, off_report.render_refusals()
    assert on_report.supported, on_report.render_refusals()

    # The actual planned residents switched: expert gate/up layouts differ.
    def _expert_layouts(report):
        return {
            record.resident_layout
            for record in report.qualified_records
            if record.role_class == "moe_experts"
        }

    assert LAYOUT_GGUF_Q4_K_X8 not in _expert_layouts(off_report)
    assert LAYOUT_GGUF_Q4_K_X8 in _expert_layouts(on_report)

    # The recorded contracts are NOT equal: the canonical resident records
    # and digest capture the env-resolved layout the kwargs never named.
    assert off_report.plan_contract != on_report.plan_contract
    assert (
        off_report.plan_contract.resident_plan_records
        != on_report.plan_contract.resident_plan_records
    )
    assert (
        off_report.plan_contract.resident_plan_digest
        != on_report.plan_contract.resident_plan_digest
    )

    # Cross-env coverage is refused in BOTH directions; same-env matches.
    off_certificate = off_report.certificate()
    on_certificate = on_report.certificate()
    assert (
        certificate_covers_artifact(
            off_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=on_report.plan_contract,
        )
        is False
    )
    assert (
        certificate_covers_artifact(
            on_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=off_report.plan_contract,
        )
        is False
    )
    assert (
        certificate_covers_artifact(
            off_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=off_report.plan_contract,
        )
        is True
    )
    assert (
        certificate_covers_artifact(
            on_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=on_report.plan_contract,
        )
        is True
    )


def test_explicit_plan_overrides_change_the_recorded_resident_contract():
    """An explicit planner override that moves a resident must move the
    canonical records and digest with it; cross-plan coverage is refused."""

    planar_map = _q5_planar_ssm_out_map()
    manifest = build_qwen35_gguf_role_manifest(planar_map)
    off = preflight_qwen35_gguf_artifact(
        planar_map,
        backend="hip_gfx1100",
        decode_repack=True,
        dense_q5_t16_ssm_out=True,
        dense_q5_raw_mmq_ssm_out=True,
    )
    on = preflight_qwen35_gguf_artifact(
        planar_map,
        backend="hip_gfx1100",
        decode_repack=True,
        dense_q5_t16_ssm_out=True,
        dense_q5_raw_mmq_ssm_out=True,
        dense_q5_qmicro_planar_ssm_out=True,
    )
    assert off.supported, off.render_refusals()
    assert on.supported, on.render_refusals()
    assert off.plan_contract != on.plan_contract
    assert (
        off.plan_contract.resident_plan_records
        != on.plan_contract.resident_plan_records
    )
    assert (
        off.plan_contract.resident_plan_digest
        != on.plan_contract.resident_plan_digest
    )
    # Records name the sidecar allocation on the planar variant only.
    assert any(
        "qmicro_planar" in record for record in on.plan_contract.resident_plan_records
    )
    assert not any(
        "qmicro_planar" in record for record in off.plan_contract.resident_plan_records
    )
    assert (
        certificate_covers_artifact(
            off.certificate(),
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=on.plan_contract,
        )
        is False
    )


def test_identical_effective_plans_compare_stably_across_input_spelling(monkeypatch):
    """Same effective plan, different input spelling/order: identical
    contracts, identical stable digests, mutually covering certificates."""

    plain_map = _synthetic_model_map()
    manifest = build_qwen35_gguf_role_manifest(plain_map)
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_SELECTED_GATE_UP_X8", raising=False)

    env_default = preflight_qwen35_gguf_artifact(plain_map, backend="hip_gfx1100")
    explicit = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", decode_repack=True, repack_veto=False
    )
    assert env_default.plan_contract == explicit.plan_contract
    assert (
        env_default.plan_contract.resident_plan_digest
        == explicit.plan_contract.resident_plan_digest
    )
    assert len(env_default.plan_contract.resident_plan_digest) == 64
    assert env_default.plan_contract.resident_plan_digest.isalnum()
    assert (
        certificate_covers_artifact(
            env_default.certificate(),
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=explicit.plan_contract,
        )
        is True
    )

    shuffled = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        slot_filter=("layers.0.attn_qkv", "root.output_norm"),
    )
    ordered = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        slot_filter=("root.output_norm", "layers.0.attn_qkv"),
    )
    assert shuffled.plan_contract == ordered.plan_contract
    assert (
        shuffled.plan_contract.resident_plan_records
        == ordered.plan_contract.resident_plan_records
    )
    assert (
        certificate_covers_artifact(
            shuffled.certificate(),
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=("root.output_norm", "layers.0.attn_qkv"),
            plan_contract=ordered.plan_contract,
        )
        is True
    )


def test_subset_narrowing_verifies_each_recorded_resident_slot():
    """Full-artifact certificates cover narrowed uses only when the intended
    contract's per-slot records verify against the certified plan;
    enlargement and partial contracts fail closed."""

    from dataclasses import replace as dataclass_replace

    plain_map = _synthetic_model_map()
    manifest = build_qwen35_gguf_role_manifest(plain_map)
    full_report = preflight_qwen35_gguf_artifact(plain_map, backend="hip_gfx1100")
    full_certificate = full_report.certificate()
    subset = ("root.output_norm", "layers.0.attn_qkv")
    subset_report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", slot_filter=subset
    )

    # Narrowed use against the full certificate: the intended contract's
    # records are verified slot-by-slot against the certified plan.
    assert (
        certificate_covers_artifact(
            full_certificate,
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=subset,
            plan_contract=subset_report.plan_contract,
        )
        is True
    )

    # Enlargement beyond a certified subset is refused.
    one_report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", slot_filter=("root.output_norm",)
    )
    assert (
        certificate_covers_artifact(
            one_report.certificate(),
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=subset,
            plan_contract=subset_report.plan_contract,
        )
        is False
    )

    # A partial intended contract (claims a slot it never recorded) fails
    # closed even against the full certificate.
    partial = dataclass_replace(one_report.plan_contract, slot_filter=subset)
    assert (
        certificate_covers_artifact(
            full_certificate,
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=subset,
            plan_contract=partial,
        )
        is False
    )

    # A narrowed contract whose resident plan differs inside the certified
    # subset is refused: the contraction moves the alpha resident record.
    contracted_full = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", contract_f32_linear=True
    )
    plain_alpha = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", slot_filter=("layers.0.ssm_alpha",)
    )
    assert (
        certificate_covers_artifact(
            contracted_full.certificate(),
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=("layers.0.ssm_alpha",),
            plan_contract=plain_alpha.plan_contract,
        )
        is False
    )
    # ...while the matching contracted subset record verifies.
    contracted_alpha = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        contract_f32_linear=True,
        slot_filter=("layers.0.ssm_alpha",),
    )
    assert (
        certificate_covers_artifact(
            contracted_full.certificate(),
            manifest_fingerprint=manifest.fingerprint,
            slot_filter=("layers.0.ssm_alpha",),
            plan_contract=contracted_alpha.plan_contract,
        )
        is True
    )


def test_plan_contract_records_cover_exactly_the_claimed_slot_scope():
    """The canonical records cover exactly the contract's slot scope (full
    scope: every checked slot; filtered scope: exactly the filter), and the
    digest is always derived from the stored records."""

    from hipengine.loading.qwen35_gguf_admission import (
        qwen35_gguf_planned_weight_digest,
    )

    plain_map = _synthetic_model_map()
    full_contract = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100"
    ).plan_contract
    assert full_contract.slot_filter is None
    assert full_contract.resident_plan_records
    slots = [
        record.split("\t", 1)[0][len("slot=") :]
        for record in full_contract.resident_plan_records
    ]
    assert len(set(slots)) == len(slots)
    assert slots == sorted(slots)
    assert full_contract.resident_plan_digest == qwen35_gguf_planned_weight_digest(
        full_contract.resident_plan_records
    )
    # Records pin layout, quant key, allocation names, planned bytes, and
    # source identity per slot.
    alpha_record = next(
        record for record in full_contract.resident_plan_records if "ssm_alpha" in record
    )
    assert "layout=dense_f32" in alpha_record
    assert "quant_key=f32" in alpha_record
    assert "allocations=raw" in alpha_record
    assert "planned_nbytes=raw:" in alpha_record
    assert "source_type=F32" in alpha_record

    subset_contract = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", slot_filter=("root.output_norm",)
    ).plan_contract
    subset_slots = [
        record.split("\t", 1)[0][len("slot=") :]
        for record in subset_contract.resident_plan_records
    ]
    assert subset_slots == ["root.output_norm"]
    assert subset_contract.resident_plan_digest != full_contract.resident_plan_digest


# ---------------------------------------------------------------------------
# U1 review repair round 4 (F4): intended-operation defaults + complete
# qualification accounting.  A certificate authorizes an intended use only
# when (a) the intended operation set -- explicit, or the intended contract's
# own checked set when omitted -- is certified on BOTH sides (identical
# residents never confer row-operation qualification), and (b) the intended
# contract proves COMPLETE successful qualification over its claimed scope:
# every slot the preflight tried to qualify is recorded, so a refused
# preflight report can never supply an authorizing contract.
# ---------------------------------------------------------------------------


def test_coverage_without_operations_defaults_to_the_intended_contracts_ops():
    """RED (F4 round 4, reviewer repro 1): a c1-only certificate must not
    cover a prefill intended plan just because the planned residents are
    identical; the default intended operation set is the intended contract's
    own checked operations."""

    plain_map = _synthetic_model_map()
    manifest = build_qwen35_gguf_role_manifest(plain_map)
    c1_report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", operations=(QWEN35_GGUF_OP_AR_DECODE_C1,)
    )
    assert c1_report.supported, c1_report.render_refusals()
    c1_certificate = c1_report.certificate()
    prefill_report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", operations=(QWEN35_GGUF_OP_AR_PREFILL,)
    )
    assert prefill_report.supported, prefill_report.render_refusals()
    # Same residents: the two contracts' planned records agree per slot, so
    # ONLY the certified operation sets distinguish the intended uses.
    assert (
        c1_report.plan_contract.resident_plan_records
        == prefill_report.plan_contract.resident_plan_records
    )

    # Omitted operations: the default intended set is the intended
    # contract's ops, which the c1-only certificate never certified.
    assert (
        certificate_covers_artifact(
            c1_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=prefill_report.plan_contract,
        )
        is False
    )
    # Explicitly naming the uncertified operation is refused as well.
    assert (
        certificate_covers_artifact(
            c1_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=prefill_report.plan_contract,
            operations=(QWEN35_GGUF_OP_AR_PREFILL,),
        )
        is False
    )
    # Explicit narrowing must belong to BOTH qualifications: the certified
    # op (c1) is still refused because the intended prefill contract never
    # checked c1.
    assert (
        certificate_covers_artifact(
            c1_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=prefill_report.plan_contract,
            operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
        )
        is False
    )

    # An intended contract that checked both operations narrows cleanly to
    # the certified one, and an explicitly EMPTY operation set authorizes
    # nothing (fail closed).
    both_report = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_PREFILL),
    )
    assert (
        certificate_covers_artifact(
            c1_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=both_report.plan_contract,
            operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
        )
        is True
    )
    assert (
        certificate_covers_artifact(
            c1_certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=both_report.plan_contract,
            operations=(),
        )
        is False
    )


def test_same_residents_do_not_upgrade_row_operations():
    """RED (F4 round 4, neighboring counterexample): identical planned
    residents must not upgrade c1/prefill coverage to the native multirow
    operation; row-operation qualification is an operation-set fact, not a
    resident-plan fact."""

    raw_iq_map = _synthetic_model_map(attn_qkv_type=GGMLQuantizationType.IQ4_XS)
    manifest = build_qwen35_gguf_role_manifest(raw_iq_map)
    plan_kwargs = dict(decode_repack=True, dense_q4_t16=True, contract_f32_linear=True)
    ar_report = preflight_qwen35_gguf_artifact(
        raw_iq_map,
        backend="hip_gfx1100",
        operations=DEFAULT_AR_OPERATIONS,
        **plan_kwargs,
    )
    assert ar_report.supported, ar_report.render_refusals()
    native_report = preflight_qwen35_gguf_artifact(
        raw_iq_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
        **plan_kwargs,
    )
    assert native_report.supported, native_report.render_refusals()
    # Identical effective residents (contraction produced the BF16 owner).
    assert (
        ar_report.plan_contract.resident_plan_records
        == native_report.plan_contract.resident_plan_records
    )
    certificate = ar_report.certificate()
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=native_report.plan_contract,
        )
        is False
    )
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=native_report.plan_contract,
            operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
        )
        is False
    )


def test_refused_preflight_contracts_never_authorize():
    """RED (F4 round 4, reviewer repro 2): a refused preflight's plan
    contract records only the successfully qualified slots; it must never
    verify as the intended plan of a full-scope certificate, with or without
    an explicit operation narrowing."""

    raw_iq_map = _synthetic_model_map(attn_qkv_type=GGMLQuantizationType.IQ4_XS)
    manifest = build_qwen35_gguf_role_manifest(raw_iq_map)
    plan_kwargs = dict(decode_repack=True, dense_q4_t16=True)
    ok_report = preflight_qwen35_gguf_artifact(
        raw_iq_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
        contract_f32_linear=True,
        **plan_kwargs,
    )
    assert ok_report.supported, ok_report.render_refusals()
    refused_report = preflight_qwen35_gguf_artifact(
        raw_iq_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
        contract_f32_linear=False,
        **plan_kwargs,
    )
    # alpha/beta stay dense F32 and are refused for the BF16-pointer owner.
    assert not refused_report.supported
    assert {
        u.slot_path
        for u in refused_report.unsupported
        if u.role_class == "recurrent_alpha_beta"
    } == {"layers.0.ssm_alpha", "layers.0.ssm_beta"}
    # The refused report's records are a strict subset of the certified ones
    # (alpha/beta omitted): exactly the shape that used to compare True.
    ok_records = set(ok_report.plan_contract.resident_plan_records)
    refused_records = set(refused_report.plan_contract.resident_plan_records)
    assert refused_records < ok_records

    certificate = ok_report.certificate()
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=refused_report.plan_contract,
            operations=(QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS,),
        )
        is False
    )
    assert (
        certificate_covers_artifact(
            certificate,
            manifest_fingerprint=manifest.fingerprint,
            plan_contract=refused_report.plan_contract,
        )
        is False
    )

    # A refused report cannot mint, and its contract stays available for
    # debugging while being provably incomplete.
    with pytest.raises(Qwen35GGUFAdmissionError):
        refused_report.certificate()
    assert refused_report.plan_contract is not None
    assert refused_report.plan_contract.resident_plan_records
    assert refused_report.plan_contract.is_complete() is False
    assert ok_report.plan_contract.is_complete() is True


def test_plan_contract_accounts_for_every_required_slot():
    """The contract binds EXPECTED coverage vs ACTUAL qualification:
    required_plan_slots names every slot the preflight tried to qualify
    (participating or refused), records carry exactly the qualified ones,
    and a hollow contract (required slot with no record) fails closed."""

    from dataclasses import replace as dataclass_replace

    plain_map = _synthetic_model_map()
    report = preflight_qwen35_gguf_artifact(plain_map, backend="hip_gfx1100")
    contract = report.plan_contract
    assert contract.is_complete() is True
    assert set(contract.resident_plan_slots) == set(contract.required_plan_slots)
    assert contract.required_plan_slots

    # Dropping one record (the digest re-derives) leaves the contract
    # claiming a required scope it never recorded: incomplete, and coverage
    # refuses it even against the full certificate it came from.
    hollow = dataclass_replace(
        contract, resident_plan_records=contract.resident_plan_records[1:]
    )
    assert hollow.is_complete() is False
    assert (
        certificate_covers_artifact(
            report.certificate(),
            manifest_fingerprint=report.manifest_fingerprint,
            plan_contract=hollow,
        )
        is False
    )

    # as_dict exports the accounting for audit trails.
    exported = report.certificate().as_dict()
    assert exported["plan_contract"]["required_plan_slots"] == list(
        contract.required_plan_slots
    )
    assert exported["plan_contract"]["operation_scope_refusals"] == []


def _map_with_extra_root_slot(extra_slot: str) -> Qwen35GGUFModelMap:
    from types import MappingProxyType

    base = _synthetic_model_map()
    root = dict(base.root_tensors)
    root[extra_slot] = _tensor(f"{extra_slot}.weight", (256,))
    return Qwen35GGUFModelMap(
        config=base.config,
        root_tensors=MappingProxyType(root),
        layers=base.layers,
        validation=None,
    )


def test_unknown_role_slot_incompletes_the_contract():
    """A slot with no certified role class is a required-but-refused slot:
    the report is refused AND the contract is structurally incomplete."""

    mystery_map = _map_with_extra_root_slot("mystery")
    report = preflight_qwen35_gguf_artifact(mystery_map, backend="hip_gfx1100")
    assert not report.supported
    refusals = [u for u in report.unsupported if u.slot_path == "root.mystery"]
    assert refusals and refusals[0].stage == "consumer_unqualified"
    assert "root.mystery" in report.plan_contract.required_plan_slots
    assert "root.mystery" not in report.plan_contract.resident_plan_slots
    assert report.plan_contract.is_complete() is False
    with pytest.raises(Qwen35GGUFAdmissionError):
        report.certificate()


def test_mtp_scope_refusal_incompletes_the_contract():
    """Operation-level scope refusals (MTP draft on a non-MTP preset) are
    recorded on the contract and make it incomplete even when every AR slot
    qualified -- an explicit narrowing to the AR operation must still be
    refused because the intended contract is not authorization-capable."""

    plain_map = _synthetic_model_map()
    report = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        operations=(
            QWEN35_GGUF_OP_AR_DECODE_C1,
            QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,
        ),
    )
    assert not report.supported
    contract = report.plan_contract
    assert contract.operation_scope_refusals == (QWEN35_GGUF_OP_MTP_NEXTN_DRAFT,)
    assert contract.is_complete() is False

    ar_certificate = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", operations=(QWEN35_GGUF_OP_AR_DECODE_C1,)
    ).certificate()
    # c1 IS certified on both sides; only the completeness gate refuses.
    assert (
        certificate_covers_artifact(
            ar_certificate,
            manifest_fingerprint=report.manifest_fingerprint,
            plan_contract=contract,
            operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
        )
        is False
    )


def test_unknown_slot_filter_entries_are_refused_not_silently_ignored():
    """A slot_filter entry naming no slot in the map is a caller error: the
    preflight refuses it, accounts it as required-but-unrecorded, and the
    resulting contract can never authorize that scope."""

    plain_map = _synthetic_model_map()
    bogus = ("root.output_norm", "layers.0.nonexistent")
    report = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100", slot_filter=bogus
    )
    assert not report.supported
    refusal = [u for u in report.unsupported if u.slot_path == "layers.0.nonexistent"]
    assert refusal and refusal[0].stage == "scope_refused"
    assert "layers.0.nonexistent" in report.plan_contract.required_plan_slots
    assert report.plan_contract.is_complete() is False

    # Without the completeness gate this subset intent would compare True
    # against the full certificate (the one real record verifies per slot).
    full_certificate = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100"
    ).certificate()
    assert (
        certificate_covers_artifact(
            full_certificate,
            manifest_fingerprint=report.manifest_fingerprint,
            plan_contract=report.plan_contract,
            slot_filter=bogus,
        )
        is False
    )


def test_named_scope_with_no_verified_records_fails_closed():
    """A subset intent that names a nonempty scope but whose intended
    contract verified no residents for it authorizes nothing: the certified
    records cannot vouch for slots the intended preflight never checked."""

    plain_map = _synthetic_model_map()
    full_certificate = preflight_qwen35_gguf_artifact(
        plain_map, backend="hip_gfx1100"
    ).certificate()
    # token_embedding participates in no decode-only operation: the
    # intended contract records nothing for the named scope.
    decode_only = preflight_qwen35_gguf_artifact(
        plain_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
        slot_filter=("root.token_embedding",),
    )
    assert decode_only.supported
    assert decode_only.plan_contract.resident_plan_records == ()
    assert (
        certificate_covers_artifact(
            full_certificate,
            manifest_fingerprint=decode_only.manifest_fingerprint,
            plan_contract=decode_only.plan_contract,
            slot_filter=("root.token_embedding",),
        )
        is False
    )


@pytest.mark.skipif(not SMALL_Q8_0.exists(), reason=f"pinned artifact missing: {SMALL_Q8_0}")
def test_loader_reverifies_a_supplied_admission_certificate(monkeypatch, tmp_path):
    """The loader is a real certificate consumer: a caller-supplied
    certificate must cover the plan THIS load would materialize (env-resolved
    layouts included) before any allocation; the minted certificate rides on
    the resident for downstream re-verification."""

    from hipengine.loading import materialize as host_materialize
    from hipengine.loading import qwen35_gguf_materialize as loader
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_materialize import (
        HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL_ENV,
        HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV,
    )
    from tests._qwen35_gguf_fixture import (
        default_fixture_tensors,
        fixture_metadata,
        write_qwen35_gguf,
    )

    path = tmp_path / "cert-consumer.gguf"
    write_qwen35_gguf(path, default_fixture_tensors(1), fixture_metadata(1))

    def _mint():
        report = preflight_qwen35_gguf_artifact(
            build_qwen35_gguf_tensor_map(GGUFReader(str(path)).info),
            backend="hip_gfx1100",
            file_type_stamp="MOSTLY_Q4_K_M",
            **_dense_flags("hip_gfx1100", "MOSTLY_Q4_K_M"),
        )
        assert report.supported, report.render_refusals()
        return report.certificate()

    # Same plan: the certificate is honored and the load completes on CPU.
    monkeypatch.delenv(HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV, raising=False)
    monkeypatch.delenv(HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL_ENV, raising=False)
    matching = _mint()
    resident = _materialize_fixture_on_cpu(
        path, monkeypatch, admission_certificate=matching
    )
    assert resident.admission_certificate is not None
    assert resident.admission_certificate == matching

    # Env-resolved plan drift (Q8_0 raw sidecar on T16 residents): the stale
    # certificate is refused BEFORE allocation, source identity unchanged.
    monkeypatch.setenv(HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV, "1")
    monkeypatch.setenv(HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL_ENV, "1")
    sentinel = _AllocationSentinel("allocation before certificate reverification")
    monkeypatch.setattr(loader, "malloc", sentinel)
    monkeypatch.setattr(host_materialize, "malloc", sentinel)
    with pytest.raises(Qwen35GGUFAdmissionError, match="certificate"):
        materialize_qwen35_gguf_weights(
            str(path), backend="hip_gfx1100", admission_certificate=matching
        )
    assert sentinel.calls == []

    # A certificate minted under the current env covers the load.
    fresh = _mint()
    resident_fresh = _materialize_fixture_on_cpu(
        path, monkeypatch, admission_certificate=fresh
    )
    assert resident_fresh.admission_certificate == fresh
    assert (
        certificate_covers_artifact(
            resident_fresh.admission_certificate,
            manifest_fingerprint=(
                resident_fresh.admission_certificate.manifest_fingerprint
            ),
            plan_contract=resident_fresh.admission_certificate.plan_contract,
        )
        is True
    )


def test_packed_decode_graph_min_replay_steps_survives_preset_bound_identity(
    monkeypatch,
):
    """U1 repair 6 regression: ``packed_decode_graph_min_replay_steps`` used a
    2-element unpack of a policy identity that now carries an optional preset
    key.  With a preset-bound resident it must (a) not raise and (b) not read
    plain-stamp-keyed policy rows (the qualification boundary)."""

    import hipengine.runtime.qwen35_gguf_runner as runner_module
    from types import SimpleNamespace

    from hipengine.kernels.policy import GGUFModelGeometry
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    geometry = GGUFModelGeometry.try_from_config(_config((LINEAR_ATTENTION,)))
    assert geometry is not None

    def fake_capability(backend, name, default):
        assert name == "GGUF_PACKED_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_POLICY"
        return {
            # Plain-stamp-keyed row: must apply to plain residents only.
            (geometry, "MOSTLY_Q4_K_M"): {24: 9},
        }

    monkeypatch.setattr(
        runner_module, "backend_package_capability", fake_capability
    )

    def session_with(weights):
        fake_session = SimpleNamespace(
            runner=SimpleNamespace(backend="hip_gfx1151", weights=weights),
            _decode_graph_min_replay_steps_cache=5,
        )
        # The real method, called with a mocked resident session.
        return lambda rows: Qwen35GGUFResidentSession.packed_decode_graph_min_replay_steps(
            fake_session, rows
        )

    plain_weights = SimpleNamespace(
        geometry=geometry,
        file_type_name="MOSTLY_Q4_K_M",
        artifact_preset_key=None,
    )
    # Plain identity reads the plain policy row: rows=24 -> minimum 9.
    assert session_with(plain_weights)(24) == 9

    ud_weights = SimpleNamespace(
        geometry=geometry,
        file_type_name="MOSTLY_Q4_K_M",
        artifact_preset_key=GGUF_UD_Q4_K_M_PRESET,
    )
    # Preset-bound identity must not raise (the old unpack ValueError) and
    # must not inherit the plain row: generic fallback ceil(5/24) = 1.
    assert session_with(ud_weights)(24) == 1

    unknown_weights = SimpleNamespace(
        geometry=geometry,
        file_type_name=None,
        artifact_preset_key=None,
    )
    assert session_with(unknown_weights)(24) == 1


def _real_map_and_nextn(path: Path):
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map

    reader = GGUFReader(path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    nextn_map = None
    if model_map.config.ignored_block_ids:
        nextn_map = build_qwen35_gguf_nextn_tensor_map(reader.info, strict=False)
    return model_map, nextn_map, (
        None if reader.info.file_type_name is None else str(reader.info.file_type_name)
    )


@pytest.mark.skipif(not PLAIN_Q4_K_M.exists(), reason=f"pinned artifact missing: {PLAIN_Q4_K_M}")
def test_qualified_plain_controls_keep_plain_identity_and_exact_ud_binds_preset():
    """U1 repair 5: established plain controls must keep the historical
    (geometry, stamp) plain policy identity; exact-UD manifests keep their
    bound preset key; everything else is an unknown manifest."""

    from hipengine.loading.qwen35_gguf_admission import (
        qwen35_gguf_artifact_preset_key,
    )

    for path in (
        PLAIN_Q4_K_M,
        PLAIN_Q4_K_S,
        SMALL_Q8_0,
        SMALL_Q4_K_M,
        QWEN36_27B_Q4_K_M,
        QWEN36_35B_A3B_Q4_K_M,
        ORNITH_35B_A3B_Q4_K_M,
    ):
        if not path.exists():
            continue
        model_map, nextn_map, stamp = _real_map_and_nextn(path)
        assert (
            qwen35_gguf_artifact_preset_key(
                model_map, nextn_map=nextn_map, file_type_stamp=stamp
            )
            is None
        ), f"qualified plain control lost its plain identity: {path.name}"

    model_map, nextn_map, stamp = _real_map_and_nextn(UD_Q4_K_M)
    assert qwen35_gguf_artifact_preset_key(
        model_map, nextn_map=nextn_map, file_type_stamp=stamp
    ) == GGUF_UD_Q4_K_M_PRESET
    model_map, nextn_map, stamp = _real_map_and_nextn(UD_Q4_K_S)
    assert qwen35_gguf_artifact_preset_key(
        model_map, nextn_map=nextn_map, file_type_stamp=stamp
    ) == GGUF_UD_Q4_K_S_PRESET


@pytest.mark.skipif(not PLAIN_Q4_K_M.exists(), reason=f"pinned artifact missing: {PLAIN_Q4_K_M}")
def test_mutated_plain_manifest_with_same_histogram_is_an_unknown_manifest():
    """U1 repair 5: a mutated plain K_M whose quant histogram is unchanged
    (ffn_up/ffn_down types swapped) has a different role-manifest fingerprint,
    so it must not inherit the plain-certified policy identity, the packaged
    hot-vocabulary selection, or the plain 2-part policy-table key."""

    import dataclasses
    from types import MappingProxyType, SimpleNamespace

    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_UNQUALIFIED_MANIFEST_PRESET,
        build_qwen35_gguf_role_manifest,
        qwen35_gguf_artifact_preset_key,
    )
    from hipengine.kernels.policy import GGUFModelGeometry
    from hipengine.runtime.qwen35_gguf_runner import _gguf_policy_identity

    model_map, nextn_map, stamp = _real_map_and_nextn(PLAIN_Q4_K_M)
    base_fingerprint = build_qwen35_gguf_role_manifest(
        model_map, nextn_map=nextn_map
    ).fingerprint

    def swapped(map_like):
        layers = []
        for layer in map_like.layers:
            tensors = dict(layer.tensors)
            up = tensors["ffn_up"]
            down = tensors["ffn_down"]
            tensors["ffn_up"] = dataclasses.replace(up, ggml_type=down.ggml_type, ggml_type_name=down.ggml_type_name)
            tensors["ffn_down"] = dataclasses.replace(down, ggml_type=up.ggml_type, ggml_type_name=up.ggml_type_name)
            layers.append(dataclasses.replace(layer, tensors=MappingProxyType(tensors)))
        return dataclasses.replace(map_like, layers=tuple(layers))

    mutated_map = swapped(model_map)
    mutated_manifest = build_qwen35_gguf_role_manifest(mutated_map, nextn_map=nextn_map)
    mutated_histogram = sorted(record[3] for record in mutated_manifest.records)
    base_histogram = sorted(record[3] for record in
                            build_qwen35_gguf_role_manifest(model_map, nextn_map=nextn_map).records)
    assert mutated_histogram == base_histogram
    assert mutated_manifest.fingerprint != base_fingerprint

    # Unknown manifest: not a qualified plain control, not a UD preset.
    assert qwen35_gguf_artifact_preset_key(
        mutated_map, nextn_map=nextn_map, file_type_stamp=stamp
    ) == GGUF_UNQUALIFIED_MANIFEST_PRESET

    geometry = GGUFModelGeometry.try_from_config(mutated_map.config)
    assert geometry is not None
    sentinel_resident = SimpleNamespace(
        geometry=geometry,
        file_type_name=stamp,
        artifact_preset_key=GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )
    identity = _gguf_policy_identity(sentinel_resident)
    assert identity == (geometry, stamp, GGUF_UNQUALIFIED_MANIFEST_PRESET)
    # The sentinel identity misses plain 2-part policy-table keys by
    # construction (3-tuple != (geometry, stamp)).


@pytest.mark.skipif(not PLAIN_Q4_K_M.exists(), reason=f"pinned artifact missing: {PLAIN_Q4_K_M}")
def test_unknown_manifest_cannot_reuse_the_packaged_hot_vocabulary():
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.gguf_mtp_hot_vocab import default_gguf_hot_vocab_path
    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )

    info = GGUFReader(PLAIN_Q4_K_M).info
    # The qualified plain control resolves the packaged selection...
    assert default_gguf_hot_vocab_path(info) is not None
    # ...an unknown manifest with the same metadata never does.
    assert (
        default_gguf_hot_vocab_path(
            info, artifact_preset_key=GGUF_UNQUALIFIED_MANIFEST_PRESET
        )
        is None
    )


def test_loader_binds_the_unqualified_manifest_sentinel_for_unknown_manifests(
    monkeypatch, tmp_path
):
    """The real loader path derives the artifact preset key from the admission
    report: a fixture manifest (not a pinned control) gets the sentinel."""

    from tests._qwen35_gguf_fixture import (
        fixture_metadata,
        linear_attention_layer_slots,
        write_qwen35_gguf,
    )
    from hipengine.quant.gguf import GGMLQuantizationType

    tensors = [
        ("token_embd.weight", (64, 256), GGMLQuantizationType.Q8_0),
        ("output_norm.weight", (256,), GGMLQuantizationType.F32),
    ]
    tensors.extend(linear_attention_layer_slots(0, projection_type=GGMLQuantizationType.Q8_0))
    path = tmp_path / "unknown-manifest.gguf"
    write_qwen35_gguf(path, tensors, fixture_metadata(1))

    resident = _materialize_fixture_on_cpu(path, monkeypatch)
    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )

    assert resident.artifact_preset_key == GGUF_UNQUALIFIED_MANIFEST_PRESET


# ---------------------------------------------------------------------------
# U1 repair round 2 (F2): raw moe_experts coverage must be source-type aware.
# The repair-2 IQ3_XXS records reused the earlier raw records' (operation,
# role_class, layout) index key, so the last-write-wins index silently
# displaced Q3_K/Q4_K/Q5_K/Q6_K/IQ2_XS/IQ4_XS raw experts.
# ---------------------------------------------------------------------------

_RAW_MOE_EXPERT_TYPE_NAMES = (
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "IQ2_XS",
    "IQ4_XS",
    "IQ3_XXS",
)


@pytest.mark.parametrize("expert_type_name", _RAW_MOE_EXPERT_TYPE_NAMES)
def test_rank3_raw_expert_coverage_is_type_aware_per_format(expert_type_name):
    """Every already-supported raw expert type keeps its certified selected
    MoE consumers on the raw layout with repacking off (decode_repack=False
    keeps rank-3 experts raw). RED: all types except IQ3_XXS were silently
    displaced by the repair-2 IQ3_XXS records sharing one index key."""

    expert_type = GGMLQuantizationType[expert_type_name]
    report = preflight_qwen35_gguf_artifact(
        _synthetic_moe_model_map(expert_type=expert_type),
        backend="hip_gfx1100",
        decode_repack=False,
    )
    assert report.unsupported == (), report.render_refusals()
    assert report.supported
    raw_records = [
        record
        for record in report.qualified_records
        if record.role_class == "moe_experts"
        and record.resident_layout == LAYOUT_RAW_GGUF
        and expert_type_name in record.source_ggml_types
    ]
    assert raw_records, f"no raw coverage record claims {expert_type_name}"
    # Per-format concrete consumer metadata: IQ3_XXS is owned by the
    # registered gguf_iq3_xxs selected consumers (moe_linear); every other
    # raw expert type keeps its concrete selected-expert consumer (the
    # gguf_q*_k raw family under linear/moe_linear, with raw rank-3 Q4_K
    # consumed by the direct gguf_q4_k_gemv wrapper).  Different formats keep
    # different records — never one merged blob with a wrong kernel key.
    if expert_type_name == "IQ3_XXS":
        assert all(record.kernel_quant == "gguf_iq3_xxs" for record in raw_records)
    else:
        assert all(record.kernel_quant and record.kernel_variant for record in raw_records)
    for record in raw_records:
        assert record.kernel_layer in {"linear", "moe_linear"}
        assert record.input_dtype == "bf16" and record.output_dtype == "bf16"
        assert record.rows_scope in ("rows_1_8_row_local", "prefill_rows")


def test_mixed_expert_type_map_keeps_every_formats_consumer():
    """Real MoE manifests mix expert formats across expert slots; each slot's
    source type must resolve ITS own certified consumer, not the last one
    written into a shared (operation, role, layout) index bucket."""

    from types import MappingProxyType

    mixed_types = {
        "ffn_gate_exps": GGMLQuantizationType.IQ2_XS,
        "ffn_up_exps": GGMLQuantizationType.Q4_K,
        "ffn_down_exps": GGMLQuantizationType.IQ4_XS,
    }
    tensors = _moe_layer_tensors(0, expert_type=GGMLQuantizationType.Q4_K)
    for slot, expert_type in mixed_types.items():
        tensors[slot] = _tensor(
            f"blk.0.{slot}.weight", (4, 256, 256), expert_type
        )
    mixed_map = Qwen35GGUFModelMap(
        config=_synthetic_moe_model_map().config,
        root_tensors=MappingProxyType(
            {
                "token_embedding": _tensor(
                    "token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0
                ),
                "output_norm": _tensor("output_norm.weight", (256,)),
                "lm_head": _tensor(
                    "token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0
                ),
            }
        ),
        layers=(
            Qwen35GGUFLayerMap(
                layer_id=0,
                layer_type=LINEAR_ATTENTION,
                tensors=MappingProxyType(tensors),
            ),
        ),
        validation=None,
    )
    report = preflight_qwen35_gguf_artifact(
        mixed_map, backend="hip_gfx1100", decode_repack=False
    )
    assert report.unsupported == (), report.render_refusals()
    assert report.supported
    qualified = {
        (record.operation, record.role_class, record.resident_layout)
        for record in report.qualified_records
        if record.role_class == "moe_experts"
    }
    for operation in (QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS, QWEN35_GGUF_OP_AR_PREFILL):
        assert (operation, "moe_experts", LAYOUT_RAW_GGUF) in qualified


def test_iq3_xxs_moe_contract_stays_raw_only_and_repacked_contracts_unchanged():
    """Per-format layout requirements: IQ3_XXS is claimed ONLY on the raw
    layout (no T16/X8 repack exists), the six pre-existing raw types are
    still claimed, and repacked expert layouts never claim IQ3_XXS or Q3_K."""

    records_by_key: dict[tuple[str, str], list] = {}
    for record in CERTIFIED_OPERATION_COVERAGE:
        if record.role_class != "moe_experts":
            continue
        records_by_key.setdefault(
            (record.operation, record.resident_layout), []
        ).append(record)
    for operation in (QWEN35_GGUF_OP_AR_DECODE_C1, QWEN35_GGUF_OP_AR_DECODE_ROWS, QWEN35_GGUF_OP_AR_PREFILL):
        raw = records_by_key[(operation, LAYOUT_RAW_GGUF)]
        claimed_raw_types = frozenset().union(
            *(record.source_ggml_types for record in raw)
        )
        assert set(_RAW_MOE_EXPERT_TYPE_NAMES) <= claimed_raw_types
        iq3_raw = [r for r in raw if "IQ3_XXS" in r.source_ggml_types]
        assert iq3_raw and all(r.source_ggml_types == frozenset({"IQ3_XXS"}) for r in iq3_raw)
        for layout in (LAYOUT_GGUF_Q4_K_T16, LAYOUT_GGUF_Q4_K_QMICRO_T16, LAYOUT_GGUF_Q5_K_QMICRO_T16):
            repacked = records_by_key[(operation, layout)]
            claimed = frozenset().union(*(r.source_ggml_types for r in repacked))
            # No IQ3_XXS repack exists, so repacked records must never claim it.
            assert "IQ3_XXS" not in claimed


def test_coverage_index_is_unambiguous_per_source_type():
    """Invariant: two DIFFERENT coverage records must never claim the same
    (operation, role_class, resident_layout, source_ggml_type) index key. The
    type-aware index rejects that conflict at construction instead of the old
    last-write-wins displacement."""

    from hipengine.loading.qwen35_gguf_admission import (
        Qwen35GGUFOperationCoverage,
        _build_coverage_index,
    )

    index = _build_coverage_index(CERTIFIED_OPERATION_COVERAGE)
    for record in CERTIFIED_OPERATION_COVERAGE:
        for source_type in sorted(record.source_ggml_types):
            key = (
                record.operation,
                record.role_class,
                record.resident_layout,
                source_type,
            )
            assert index[key] is record

    raw_experts = next(
        record
        for record in CERTIFIED_OPERATION_COVERAGE
        if record.role_class == "moe_experts"
        and record.resident_layout == LAYOUT_RAW_GGUF
        and "Q4_K" in record.source_ggml_types
    )
    impostor = Qwen35GGUFOperationCoverage(
        operation=raw_experts.operation,
        role_class=raw_experts.role_class,
        resident_layout=raw_experts.resident_layout,
        source_ggml_types=frozenset({"Q4_K"}),
        rows_scope=raw_experts.rows_scope,
        input_dtype="bf16",
        output_dtype="bf16",
        kernel_layer="moe_linear",
        note="impostor record re-claiming an indexed source type",
    )
    with pytest.raises(ValueError, match="Q4_K"):
        _build_coverage_index((*CERTIFIED_OPERATION_COVERAGE, impostor))


def test_repacked_moe_experts_still_qualify_with_decode_repack():
    """Guard against overcorrection: the repacked expert families keep their
    own (operation, role, layout) coverage with repacking enabled."""

    report = preflight_qwen35_gguf_artifact(
        _synthetic_moe_model_map(expert_type=GGMLQuantizationType.Q4_K),
        backend="hip_gfx1100",
        decode_repack=True,
    )
    assert report.unsupported == (), report.render_refusals()
    repacked = {
        record.resident_layout
        for record in report.qualified_records
        if record.role_class == "moe_experts"
    }
    assert LAYOUT_GGUF_Q4_K_T16 in repacked


# ---------------------------------------------------------------------------
# U1 repair round 2 (F3): the mandatory allocation-formula validation must
# keep the previously supported env-gated Q5 planar sidecar loadable.
# ---------------------------------------------------------------------------

_Q5_PLANAR_SSM_OUT_SHAPE = (5_120, 6_144)

_Q5_PLANAR_FLAGS = dict(
    dense_q5_t16_ssm_out=True,
    dense_q5_raw_mmq_ssm_out=True,
    dense_q5_qmicro_planar_ssm_out=True,
)


def _q5_planar_ssm_out_map() -> Qwen35GGUFModelMap:
    from types import MappingProxyType

    config = _config((LINEAR_ATTENTION,))
    root = {
        "token_embedding": _tensor(
            "token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0
        ),
        "output_norm": _tensor("output_norm.weight", (256,)),
        "lm_head": _tensor("token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0),
    }
    tensors = dict(_linear_layer_tensors(0))
    tensors["ssm_out"] = _tensor(
        "blk.0.ssm_out.weight",
        _Q5_PLANAR_SSM_OUT_SHAPE,
        GGMLQuantizationType.Q5_K,
    )
    return Qwen35GGUFModelMap(
        config=config,
        root_tensors=MappingProxyType(root),
        layers=(
            Qwen35GGUFLayerMap(
                layer_id=0,
                layer_type=LINEAR_ATTENTION,
                tensors=MappingProxyType(tensors),
            ),
        ),
        validation=None,
    )


def _expected_q5_planar_nbytes() -> tuple[dict[str, int], object]:
    """Exact per-allocation bytes for the planar Q5 T16 ssm_out resident.

    The planar sidecar payload is the INT8 ``planar.tiles`` array of the real
    converter chain ``convert_gguf_q5_k_qmicro_tile16_to_planar(
    repack_gguf_q5_k_qmicro_tile16(raw[None, ...]))``: shape
    ``[1, out/16, bytes_per_row/176, GGUF_Q5_K_QMICRO_PLANAR_T16_BLOCK_BYTES]``.
    """

    import numpy as np
    from hipengine.quant.gguf_t16 import (
        GGUF_Q5_K_BLOCK_BYTES,
        GGUF_Q5_K_T16_BLOCK_BYTES,
        GGUF_Q5_K_QMICRO_PLANAR_T16_BLOCK_BYTES,
        convert_gguf_q5_k_qmicro_tile16_to_planar,
        repack_gguf_q5_k_qmicro_tile16,
    )

    model_map = _q5_planar_ssm_out_map()
    source = model_map.layers[0].tensors["ssm_out"]
    out_features, bytes_per_row = (int(dim) for dim in source.byte_shape)
    blocks_per_row = bytes_per_row // GGUF_Q5_K_BLOCK_BYTES
    raw = np.zeros(
        (out_features, bytes_per_row), dtype=np.uint8
    )
    planar = convert_gguf_q5_k_qmicro_tile16_to_planar(
        repack_gguf_q5_k_qmicro_tile16(raw[None, ...])
    )
    expected = {
        "tiles": (out_features // 16)
        * blocks_per_row
        * GGUF_Q5_K_T16_BLOCK_BYTES,
        "raw": int(source.nbytes),
        "qmicro_planar": int(planar.tiles.nbytes),
    }
    return expected, source


def test_q5_planar_sidecar_resident_plans_exact_converter_bytes():
    """RED (F3 regression): the allocation-formula validation added with U1
    repair 3 hard-refused the previously supported qmicro_planar sidecar
    (``unsupported resident allocation 'qmicro_planar'``). The planner must
    size the sidecar exactly like the host-side converter/allocation ABI."""

    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_GGUF_Q5_K_T16,
        plan_qwen35_gguf_weight_spec,
        planned_qwen35_gguf_weight_allocation_nbytes,
    )

    model_map = _q5_planar_ssm_out_map()
    spec = plan_qwen35_gguf_weight_spec(
        "layers.0.ssm_out",
        model_map.layers[0].tensors["ssm_out"],
        decode_repack=True,
        **_Q5_PLANAR_FLAGS,
    )
    assert spec.layout == LAYOUT_GGUF_Q5_K_T16
    assert spec.allocation_names == ("tiles", "raw", "qmicro_planar")
    expected, _source = _expected_q5_planar_nbytes()
    planned = dict(planned_qwen35_gguf_weight_allocation_nbytes(spec))
    assert planned == expected
    assert planned["qmicro_planar"] > 0


def test_q5_planar_sidecar_resident_is_admitted_with_planar_flags():
    """The previously supported env-gated planar resident passes admission
    again (all three flags on), alongside the default-off control."""

    planar_report = preflight_qwen35_gguf_artifact(
        _q5_planar_ssm_out_map(),
        backend="hip_gfx1100",
        decode_repack=True,
        **_Q5_PLANAR_FLAGS,
    )
    assert planar_report.unsupported == (), planar_report.render_refusals()
    assert planar_report.supported

    # Default-off control: raw sidecar without the planar sidecar stays exact.
    off_report = preflight_qwen35_gguf_artifact(
        _q5_planar_ssm_out_map(),
        backend="hip_gfx1100",
        decode_repack=True,
        dense_q5_t16_ssm_out=True,
        dense_q5_raw_mmq_ssm_out=True,
    )
    assert off_report.unsupported == (), off_report.render_refusals()


def test_q5_planar_env_gate_resolves_off_by_default_and_on_with_env(monkeypatch):
    """End-to-end through the shared policy API: HIPENGINE_C8_Q5_PLANAR_DP4A
    defaults off (tiles+raw only) and resolves the planar sidecar on with the
    env set — both variants must size exactly, on with no refusal."""

    import os

    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_GGUF_Q5_K_T16,
        plan_qwen35_gguf_weight_spec,
        planned_qwen35_gguf_weight_allocation_nbytes,
    )
    from hipengine.loading.qwen35_gguf_policy import resolve_gguf_dense_flags

    monkeypatch.delenv("HIPENGINE_C8_Q5_PLANAR_DP4A", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_C8_Q5_RAW_MMQ", raising=False)
    flags = resolve_gguf_dense_flags(
        "hip_gfx1100",
        "MOSTLY_Q4_K_M",
        capability_reader=_gguf_backend_capability(),
        environ=os.environ,
    )
    assert flags["dense_q5_t16_ssm_out"] is True
    assert flags["dense_q5_raw_mmq_ssm_out"] is True
    assert flags["dense_q5_qmicro_planar_ssm_out"] is False
    source = _q5_planar_ssm_out_map().layers[0].tensors["ssm_out"]
    off_spec = plan_qwen35_gguf_weight_spec(
        "layers.0.ssm_out", source, decode_repack=True, **flags
    )
    assert off_spec.allocation_names == ("tiles", "raw")
    assert dict(planned_qwen35_gguf_weight_allocation_nbytes(off_spec)) == {
        "tiles": _expected_q5_planar_nbytes()[0]["tiles"],
        "raw": _expected_q5_planar_nbytes()[0]["raw"],
    }

    monkeypatch.setenv("HIPENGINE_C8_Q5_PLANAR_DP4A", "1")
    flags_on = resolve_gguf_dense_flags(
        "hip_gfx1100",
        "MOSTLY_Q4_K_M",
        capability_reader=_gguf_backend_capability(),
        environ=os.environ,
    )
    assert flags_on["dense_q5_qmicro_planar_ssm_out"] is True
    on_spec = plan_qwen35_gguf_weight_spec(
        "layers.0.ssm_out", source, decode_repack=True, **flags_on
    )
    assert on_spec.allocation_names == ("tiles", "raw", "qmicro_planar")
    planned_on = dict(planned_qwen35_gguf_weight_allocation_nbytes(on_spec))
    assert planned_on == _expected_q5_planar_nbytes()[0]


def test_unaffected_sidecar_and_t16_controls_keep_exact_accounting():
    """Controls that never involved the planar formula: raw Q8_0 sidecar,
    Q5 08b T16, and the Q6 qmicro-planar resident layout all keep sizing"""

    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        LAYOUT_GGUF_Q8_0_T16,
        plan_qwen35_gguf_weight_spec,
        planned_qwen35_gguf_weight_allocation_nbytes,
    )

    # Raw-sidecar control: Q8_0 alpha/beta with decode repack plans sole T16
    # tiles (the raw sidecar needs its own env); the tiles accounting is exact.
    q8_map = _synthetic_model_map(
        embedding_type=GGMLQuantizationType.Q8_0,
        alpha_beta_type=GGMLQuantizationType.Q8_0,
    )
    alpha = q8_map.layers[0].tensors["ssm_alpha"]
    q8_spec = plan_qwen35_gguf_weight_spec(
        "layers.0.ssm_alpha", alpha, decode_repack=True
    )
    assert q8_spec.layout == LAYOUT_GGUF_Q8_0_T16
    q8_planned = dict(planned_qwen35_gguf_weight_allocation_nbytes(q8_spec))
    assert q8_planned["tiles"] == (int(alpha.byte_shape[0]) // 16) * (
        int(alpha.byte_shape[1]) // GGUF_Q8_0_BLOCK_BYTES
    ) * GGUF_Q8_0_T16_BLOCK_BYTES

    # Q6 qmicro-planar resident layout control (already formula-supported).
    config = _config((LINEAR_ATTENTION,))
    tensors = dict(_linear_layer_tensors(0))
    tensors["attn_v"] = _tensor(
        "blk.0.attn_v.weight", (1_024, 5_120), GGMLQuantizationType.Q6_K
    )
    from types import MappingProxyType

    q6_map = Qwen35GGUFModelMap(
        config=config,
        root_tensors=MappingProxyType(
            {
                "token_embedding": _tensor(
                    "token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0
                ),
                "output_norm": _tensor("output_norm.weight", (256,)),
                "lm_head": _tensor(
                    "token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0
                ),
            }
        ),
        layers=(
            Qwen35GGUFLayerMap(
                layer_id=0,
                layer_type=LINEAR_ATTENTION,
                tensors=MappingProxyType(tensors),
            ),
        ),
        validation=None,
    )
    q6_report = preflight_qwen35_gguf_artifact(
        q6_map,
        backend="hip_gfx1100",
        decode_repack=True,
        dense_q6_qmicro_planar=True,
    )
    assert q6_report.unsupported == (), q6_report.render_refusals()
    q6_spec = plan_qwen35_gguf_weight_spec(
        "layers.0.attn_v",
        tensors["attn_v"],
        decode_repack=True,
        dense_q6_qmicro_planar=True,
    )
    assert q6_spec.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
    assert dict(planned_qwen35_gguf_weight_allocation_nbytes(q6_spec)) == {
        "tiles": int(tensors["attn_v"].nbytes)
    }


def test_invalid_planar_layouts_and_shapes_still_fail_closed():
    """Supporting the planar formula must not waive the mandatory shape or
    layout checks: the never-produced Q5 planar resident layout, a planar
    allocation on a non-Q5-T16 layout, and a non-tile-aligned planar shape
    are all still refused by the allocation formula."""

    from hipengine.loading.gguf import GGUFTensorInfo
    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_GGUF_Q5_K_QMICRO_PLANAR,
        LAYOUT_GGUF_Q5_K_QMICRO_T16,
        LAYOUT_GGUF_Q5_K_T16,
        planned_qwen35_gguf_weight_allocation_nbytes,
    )

    good_source = _q5_planar_ssm_out_map().layers[0].tensors["ssm_out"]

    def spec_with(layout: str, allocation_names: tuple[str, ...], source=None):
        return Qwen35GGUFWeightSpec(
            slot_path="layers.0.ssm_out",
            source=good_source if source is None else source,
            quant_key="gguf_q5_k_t16_v1",
            layout=layout,
            allocation_names=allocation_names,
        )

    # The never-produced standalone Q5 planar resident layout stays refused.
    with pytest.raises(ValueError, match="unsupported resident layout"):
        planned_qwen35_gguf_weight_allocation_nbytes(
            spec_with(LAYOUT_GGUF_Q5_K_QMICRO_PLANAR, ("tiles",))
        )
    # A planar allocation outside the Q5 T16 resident layout is refused
    # (mirrors the materializer's own sidecar gate).
    with pytest.raises(ValueError, match="qmicro_planar"):
        planned_qwen35_gguf_weight_allocation_nbytes(
            spec_with(LAYOUT_GGUF_Q5_K_QMICRO_T16, ("tiles", "qmicro_planar"))
        )
    # A non-tile-aligned planar shape is refused by the same T16 tile check
    # the tiles allocation enforces (out_features not a multiple of 16).
    import dataclasses

    misaligned = dataclasses.replace(
        good_source,
        shape=(5_122, 6_144),
        n_elements=5_122 * 6_144,
        byte_shape=(5_122, int(good_source.byte_shape[1])),
        nbytes=5_122 * int(good_source.byte_shape[1]),
    )
    with pytest.raises(ValueError, match="tile-aligned"):
        planned_qwen35_gguf_weight_allocation_nbytes(
            spec_with(LAYOUT_GGUF_Q5_K_T16, ("tiles", "qmicro_planar"), misaligned)
        )


# ---------------------------------------------------------------------------
# U1 review repair round 5 (F5): actual stamp-only policy callers must bind the
# artifact qualification (loader-resolved preset key / unqualified-manifest
# sentinel). The sentinel and the 3-tuple policy identity already exist; these
# tests cover the production callers that still passed only (geometry, stamp)
# or the bare stamp into artifact-qualified policy selection.
# ---------------------------------------------------------------------------


def test_runner_default_fp16_state_binds_artifact_qualification(monkeypatch):
    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )


    """The real ``Qwen35GGUFFullStackRunner`` initializer freezes the FP16
    recurrent-state default from the resident artifact: only a qualified plain
    control (loader-resolved ``artifact_preset_key=None``) may inherit the
    stamp-certified default; unknown manifests and UD presets fall back to the
    generic strict FP32 storage. The env var stays the explicit developer
    opt-out for every identity."""

    import hipengine.runtime.qwen35_gguf_runner as runner_module
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner
    from types import SimpleNamespace

    monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", raising=False)
    monkeypatch.setattr(runner_module, "resolve_backend", lambda backend: backend)
    monkeypatch.setattr(
        runner_module, "hip_target_arch_for_backend", lambda backend: "gfx1151"
    )
    monkeypatch.setattr(
        runner_module, "load_backend_kernel_package", lambda backend: None
    )
    monkeypatch.setattr(runner_module, "resolve", lambda **kwargs: object())
    monkeypatch.setattr(
        runner_module,
        "backend_package_capability",
        lambda backend, name, default=None: (
            frozenset({"mostly_q4_k_s"})
            if name == "GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES"
            else default
        ),
    )

    def initialized_runner(artifact_preset_key):
        runner = object.__new__(Qwen35GGUFFullStackRunner)
        runner.backend = "hip_gfx1151"
        runner.runtime = object()
        runner.compiler_version = None
        runner.require_cached_build = False
        runner.token_embedding_placement = "device"
        runner.resident_weights = SimpleNamespace(
            backend="hip_gfx1151",
            file_type_name="MOSTLY_Q4_K_S",
            artifact_preset_key=artifact_preset_key,
            config=None,
        )
        Qwen35GGUFFullStackRunner.__post_init__(runner)
        return runner

    # Qualified plain control keeps the certified stamp default.
    assert initialized_runner(None).fp16_recurrent_state is True
    # Unknown manifest / UD preset sharing the exact stamp must not inherit it.
    assert (
        initialized_runner(GGUF_UNQUALIFIED_MANIFEST_PRESET).fp16_recurrent_state
        is False
    )
    assert initialized_runner(GGUF_UD_Q4_K_S_PRESET).fp16_recurrent_state is False

    # Explicit env override remains the developer opt-out seam for every
    # identity (forces FP16 on for unknown, off for the plain control).
    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")
    assert (
        initialized_runner(GGUF_UNQUALIFIED_MANIFEST_PRESET).fp16_recurrent_state
        is True
    )
    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "0")
    assert initialized_runner(None).fp16_recurrent_state is False


def test_decode_graph_submission_policy_binds_artifact_qualification(monkeypatch):
    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )


    """The graph submission-transport policy is certified per artifact
    (geometry, stamp): preset-bound and unknown-manifest identities resolve
    only exact preset-keyed rows (none ship today) and otherwise the generic
    hipgraph fallback; plain controls keep the certified rows."""

    import hipengine.runtime.qwen35_gguf_runner as runner_module
    from hipengine.kernels.policy import GGUFModelGeometry
    from hipengine.runtime.qwen35_gguf_runner import (
        _resolve_gguf_decode_graph_submission_transport,
    )

    geometry = GGUFModelGeometry.try_from_config(_config((LINEAR_ATTENTION,)))
    assert geometry is not None

    monkeypatch.setattr(
        runner_module,
        "backend_package_capability",
        lambda backend, name, default=None: (
            {
                (geometry, "MOSTLY_Q4_K_S"): {
                    "transport": "pm4",
                    "min_replay_steps_by_physical_rows": {1: 8},
                },
            }
            if name == "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES"
            else default
        ),
    )

    # Plain identity keeps the certified row.
    assert (
        _resolve_gguf_decode_graph_submission_transport(
            "hip_gfx1151",
            geometry=geometry,
            file_type_name="MOSTLY_Q4_K_S",
            physical_rows=1,
            replay_steps=8,
            env={},
        )
        == "pm4"
    )
    # Unknown/preset-bound identities never inherit the plain row.
    for preset_key in (GGUF_UNQUALIFIED_MANIFEST_PRESET, GGUF_UD_Q4_K_S_PRESET):
        assert (
            _resolve_gguf_decode_graph_submission_transport(
                "hip_gfx1151",
                geometry=geometry,
                file_type_name="MOSTLY_Q4_K_S",
                artifact_preset_key=preset_key,
                physical_rows=1,
                replay_steps=8,
                env={},
            )
            == "hipgraph"
        )
    # Explicit requests keep overriding for any identity.
    assert (
        _resolve_gguf_decode_graph_submission_transport(
            "hip_gfx1151",
            geometry=geometry,
            file_type_name="MOSTLY_Q4_K_S",
            artifact_preset_key=GGUF_UNQUALIFIED_MANIFEST_PRESET,
            requested="pm4",
            env={},
        )
        == "pm4"
    )


def test_capture_decode_graph_callers_thread_resident_qualification(monkeypatch):
    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )


    """The real capture entry points (single-slot and packed) must pass the
    resident artifact preset key into transport selection; a stamp-only
    identity would let an unknown manifest inherit the plain-certified PM4
    transport row."""

    from types import SimpleNamespace

    import hipengine.core.pm4.transport as pm4_transport
    import hipengine.runtime.gguf_decode_graph as decode_graph_module
    import hipengine.runtime.gguf_packed_decode_graph as packed_graph_module
    import hipengine.runtime.qwen35_gguf_runner as runner_module
    from hipengine.kernels.policy import GGUFModelGeometry
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    geometry = GGUFModelGeometry.try_from_config(_config((LINEAR_ATTENTION,)))
    assert geometry is not None

    captured: dict[str, object] = {}
    real_resolve = runner_module._resolve_gguf_decode_graph_submission_transport

    def spy_resolve(backend, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return real_resolve(backend, **kwargs)

    monkeypatch.setattr(
        runner_module, "_resolve_gguf_decode_graph_submission_transport", spy_resolve
    )
    monkeypatch.setattr(
        pm4_transport, "create_graph_submission_context", lambda **kwargs: None
    )
    monkeypatch.setattr(
        decode_graph_module, "capture_qwen35_gguf_decode_graph", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        packed_graph_module,
        "capture_qwen35_gguf_packed_decode_graph",
        lambda *a, **k: object(),
    )

    def fake_session(artifact_preset_key):
        return SimpleNamespace(
            runner=SimpleNamespace(
                backend="hip_gfx1151",
                target_arch="gfx1151",
                weights=SimpleNamespace(
                    geometry=geometry,
                    file_type_name="MOSTLY_Q4_K_S",
                    artifact_preset_key=artifact_preset_key,
                ),
            ),
            runtime=object(),
            _decode_graph_submission_contexts={},
            _pin_device_kv_graph=lambda graph: None,
        )

    for preset_key in (GGUF_UNQUALIFIED_MANIFEST_PRESET, GGUF_UD_Q4_K_S_PRESET):
        Qwen35GGUFResidentSession.capture_decode_graph(
            fake_session(preset_key), position=0
        )
        assert captured.get("artifact_preset_key") == preset_key, (
            "capture_decode_graph dropped the resident artifact qualification"
        )
        Qwen35GGUFResidentSession.capture_packed_decode_graph(
            fake_session(preset_key), [1, 2, 3, 4]
        )
        assert captured.get("artifact_preset_key") == preset_key, (
            "capture_packed_decode_graph dropped the resident artifact qualification"
        )

    # Plain controls keep their None identity threaded through unchanged.
    Qwen35GGUFResidentSession.capture_decode_graph(fake_session(None), position=0)
    assert captured.get("artifact_preset_key") is None


def test_private_c1_arena_policies_bind_artifact_qualification(monkeypatch):
    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )


    """The private-c1 arena admissions are (geometry, stamp)-certified rows:
    preset-bound and unknown-manifest identities must resolve only exact
    preset-keyed rows (none ship today), never the plain rows."""

    import hipengine.runtime.qwen35_gguf_runner as runner_module
    from hipengine.kernels.policy import GGUFModelGeometry
    from hipengine.runtime.qwen35_gguf_runner import (
        _resolve_gguf_private_c1_decode_scratch_arena,
        _resolve_gguf_private_c1_small_weight_arena,
        _resolve_gguf_private_c1_weight_arena_max_allocation_bytes,
    )

    geometry = GGUFModelGeometry.try_from_config(_config((LINEAR_ATTENTION,)))
    assert geometry is not None
    default_bytes = runner_module.GGUF_SELECTIVE_WEIGHT_ARENA_MAX_ALLOCATION_BYTES

    def fake_capability(backend, name, default=None):
        if name == "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA":
            return False
        if name == "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES":
            return {
                (geometry, "MOSTLY_Q4_K_S"): {
                    "enabled": True,
                    "max_allocation_bytes": 4096,
                }
            }
        if name == "GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES":
            return {(geometry, "MOSTLY_Q4_K_S"): {"enabled": True}}
        return default

    monkeypatch.setattr(runner_module, "backend_package_capability", fake_capability)

    common = {
        "backend": "hip_gfx1151",
        "max_batch_size": 1,
        "has_shared_runner": False,
        "geometry": geometry,
        "file_type_name": "MOSTLY_Q4_K_S",
    }
    # Plain identity keeps the certified rows.
    assert _resolve_gguf_private_c1_small_weight_arena(**common) == (
        True,
        "private_c1_selective",
    )
    assert _resolve_gguf_private_c1_decode_scratch_arena(**common) == (
        True,
        "private_c1_geometry_policy",
    )
    assert (
        _resolve_gguf_private_c1_weight_arena_max_allocation_bytes(
            backend="hip_gfx1151",
            geometry=geometry,
            file_type_name="MOSTLY_Q4_K_S",
        )
        == 4096
    )
    # Unknown/preset-bound identities never inherit the plain rows.
    for preset_key in (GGUF_UNQUALIFIED_MANIFEST_PRESET, GGUF_UD_Q4_K_S_PRESET):
        unqualified = {**common, "artifact_preset_key": preset_key}
        assert _resolve_gguf_private_c1_small_weight_arena(**unqualified) == (
            False,
            "backend_capability_fallback",
        )
        assert _resolve_gguf_private_c1_decode_scratch_arena(**unqualified) == (
            False,
            "backend_capability_fallback",
        )
        assert (
            _resolve_gguf_private_c1_weight_arena_max_allocation_bytes(
                backend="hip_gfx1151",
                geometry=geometry,
                file_type_name="MOSTLY_Q4_K_S",
                artifact_preset_key=preset_key,
            )
            == default_bytes
        )
    # An explicit ``requested=True`` is only an opt-OUT seam: unlike a forced
    # override it does not bypass the artifact-qualified admission, so an
    # unknown identity is still refused the plain-certified rows.
    assert _resolve_gguf_private_c1_small_weight_arena(
        **{**common, "artifact_preset_key": GGUF_UNQUALIFIED_MANIFEST_PRESET,
           "requested": True}
    ) == (False, "backend_capability_fallback")
    assert _resolve_gguf_private_c1_small_weight_arena(
        **{**common, "requested": True}
    ) == (True, "private_c1_selective")


def test_session_private_c1_arena_admission_binds_actual_manifest(
    monkeypatch, tmp_path
):
    """The real ``Qwen35GGUFResidentSession`` initializer resolves the private
    -c1 arena policy from the GGUF header before materialization: the
    admission must bind the actual artifact qualification derived from the
    manifest (unknown fixture manifests get the sentinel and must not inherit
    the stamp-certified rows), while pinning the manifest as a qualified plain
    control restores the historical admission."""

    from types import SimpleNamespace

    import hipengine.loading.qwen35_gguf_admission as admission_module
    import hipengine.runtime.qwen35_gguf_runner as runner_module
    from hipengine.kernels.policy import GGUFModelGeometry
    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf import GGMLQuantizationType
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from tests._qwen35_gguf_fixture import (
        fixture_metadata,
        linear_attention_layer_slots,
        write_qwen35_gguf,
    )

    tensors = [
        ("token_embd.weight", (64, 256), GGMLQuantizationType.Q8_0),
        ("output_norm.weight", (256,), GGMLQuantizationType.F32),
    ]
    tensors.extend(
        linear_attention_layer_slots(0, projection_type=GGMLQuantizationType.Q8_0)
    )
    path = tmp_path / "unknown-ks-manifest.gguf"
    write_qwen35_gguf(path, tensors, fixture_metadata(1, file_type=14))

    info = GGUFReader(path).info
    geometry = GGUFModelGeometry.from_config(
        runner_module.qwen35_gguf_config_from_metadata(info)
    )
    stamp = info.file_type_name

    class _Abort(Exception):
        pass

    def aborting_runner_ctor(*args, **kwargs):
        raise _Abort()

    monkeypatch.setattr(
        runner_module, "Qwen35GGUFFullStackRunner", aborting_runner_ctor
    )
    monkeypatch.setattr(
        runner_module,
        "backend_package_capability",
        lambda backend, name, default=None: (
            {
                (geometry, stamp): {
                    "enabled": True,
                    "max_allocation_bytes": 4096,
                }
            }
            if name == "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES"
            else {(geometry, stamp): {"enabled": True}}
            if name == "GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES"
            else False
            if name == "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA"
            else default
        ),
    )

    def initialized_session():
        session = object.__new__(Qwen35GGUFResidentSession)
        session.dms_metadata_path = None
        session.dms_decision_mode = "sidecar"
        session.max_batch_size = 1
        session.prefill_queue_drain = "none"
        session.runtime = object()
        session.backend = "hip_gfx1151"
        session.model_path = str(path)
        session.shared_runner = None
        session.token_embedding_placement = "auto"
        session.use_small_weight_arena = None
        session.use_decode_scratch_arena = None
        with pytest.raises(_Abort):
            Qwen35GGUFResidentSession.__post_init__(session)
        return session

    # Unknown fixture manifest: sentinel identity, no plain-row inheritance.
    session = initialized_session()
    assert session.small_weight_arena_enabled is False
    assert session.small_weight_arena_reason == "backend_capability_fallback"
    assert session.decode_scratch_arena_enabled is False
    assert session.decode_scratch_arena_reason == "backend_capability_fallback"

    # Pin the exact fixture manifest fingerprint: the qualified plain control
    # lane restores the historical stamp-keyed admission.
    model_map = runner_module.build_qwen35_gguf_tensor_map(info)
    manifest = build_qwen35_gguf_role_manifest(model_map)
    monkeypatch.setattr(
        admission_module,
        "_PINNED_PLAIN_CONTROL_FINGERPRINTS",
        frozenset({manifest.fingerprint}),
    )
    session = initialized_session()
    assert session.small_weight_arena_enabled is True
    assert session.small_weight_arena_reason == "private_c1_selective"
    assert session.small_weight_arena_max_allocation_bytes == 4096
    assert session.decode_scratch_arena_enabled is True
    assert session.decode_scratch_arena_reason == "private_c1_geometry_policy"


def test_fp16_default_policy_mirror_binds_artifact_qualification():
    from hipengine.loading.qwen35_gguf_admission import (
        GGUF_UNQUALIFIED_MANIFEST_PRESET,
    )


    """The shared pure policy mirror (used by the CPU-only quant-route audit)
    must mirror the runner's artifact-qualified FP16-recurrent-state default:
    preset-bound and unknown identities resolve the generic False default."""

    from hipengine.loading.qwen35_gguf_policy import gguf_fp16_recurrent_state_default

    def reader(backend, name, default):
        if name == "GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES":
            return frozenset({"mostly_q4_k_s"})
        return default

    assert (
        gguf_fp16_recurrent_state_default(
            "b", "MOSTLY_Q4_K_S", capability_reader=reader
        )
        is True
    )
    for preset_key in (GGUF_UNQUALIFIED_MANIFEST_PRESET, GGUF_UD_Q4_K_S_PRESET):
        assert (
            gguf_fp16_recurrent_state_default(
                "b",
                "MOSTLY_Q4_K_S",
                capability_reader=reader,
                artifact_preset_key=preset_key,
            )
            is False
        )
    # Without a stamp there is no certified default for any identity.
    assert (
        gguf_fp16_recurrent_state_default(
            "b",
            None,
            capability_reader=reader,
            artifact_preset_key=GGUF_UNQUALIFIED_MANIFEST_PRESET,
        )
        is False
    )


# ---------------------------------------------------------------------------
# U1 review repair F3: concrete backend consumer + actual caller dtype
# ---------------------------------------------------------------------------


def test_scaffold_cuda_backend_gets_no_gguf_consumer_certificate():
    """F3 counterexample: ``cuda_sm120a`` is a registered hardware backend
    key, but its package registers no GGUF consumer families (no ``gguf_*``
    linear/embedding/dense consumers; moonshine/maple/PARO keys only).  A
    known target-arch name alone must never yield a positive GGUF admission
    certificate; every slot is refused with the missing concrete consumer
    named, before any allocation."""

    report = preflight_qwen35_gguf_artifact(
        _synthetic_model_map(),
        backend="cuda_sm120a",
        operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
    )
    assert report.supported is False, report.render_refusals()
    assert report.unsupported, "scaffold backend earned GGUF coverage"
    assert all(u.stage == "consumer_unqualified" for u in report.unsupported)
    assert all("cuda_sm120a" in u.reason for u in report.unsupported)
    with pytest.raises(Qwen35GGUFAdmissionError):
        report.raise_for_errors()


def test_dense_f32_lm_head_refuses_default_f32_logits_caller_dtype(monkeypatch, tmp_path):
    """F3 concrete dtype repro: the production lm-head callers
    (``logits_from_hidden_bits`` / native rows / packed verify) supply BF16
    hidden state without an input override, and the runtime linear dispatch
    has no ``(dense_f32, bf16 activation, f32 output)`` row.  Certifying the
    F32-input row as if it were the default caller contract would let the
    artifact load and then fail at the first decode; admission must refuse
    the (slot, operation) before any allocation instead."""

    f32_head_map = _synthetic_model_map(lm_head_type=GGMLQuantizationType.F32)
    report = preflight_qwen35_gguf_artifact(
        f32_head_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
    )
    assert report.supported is False, report.render_refusals()
    refusals = [u for u in report.unsupported if u.slot_path == "root.lm_head"]
    assert refusals and refusals[0].stage == "consumer_unqualified"
    reason = refusals[0].reason
    assert "dense_f32" in reason and "bf16" in reason and "f32" in reason
    with pytest.raises(Qwen35GGUFAdmissionError):
        report.raise_for_errors()

    # Loader-level: the refusal happens before ANY device allocation.
    from tests._qwen35_gguf_fixture import (
        fixture_metadata,
        linear_attention_layer_slots,
        write_qwen35_gguf,
    )
    from hipengine.loading import materialize as host_materialize
    from hipengine.loading import qwen35_gguf_materialize as loader
    from hipengine.loading.gguf import GGUFReader

    tensors = [
        ("token_embd.weight", (32, 256), GGMLQuantizationType.Q8_0),
        ("output_norm.weight", (256,), GGMLQuantizationType.F32),
        ("output.weight", (32, 256), GGMLQuantizationType.F32),
    ]
    tensors.extend(linear_attention_layer_slots(0, projection_type=GGMLQuantizationType.Q4_K))
    path = tmp_path / "f32-head.gguf"
    write_qwen35_gguf(path, tensors, fixture_metadata(1))
    reader = GGUFReader(path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    dense_head_report = preflight_qwen35_gguf_artifact(
        model_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
    )
    assert dense_head_report.supported is False
    assert any(u.slot_path == "root.lm_head" for u in dense_head_report.unsupported)
    sentinel = _AllocationSentinel("allocator invoked before dense-F32 head refusal")
    monkeypatch.setattr(loader, "malloc", sentinel)
    monkeypatch.setattr(host_materialize, "malloc", sentinel)
    with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
        materialize_qwen35_gguf_weights(
            str(path),
            backend="hip_gfx1100",
            requested_operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
        )
    assert sentinel.calls == []
    assert "root.lm_head" in str(excinfo.value)


def test_dense_f32_lm_head_qualifies_with_declared_f32_input_override():
    """The registered ``(dense_f32, f32 activation, f32 output)`` consumer
    (``dense_gemv/f32/f32_hidden_f32_out``) stays a valid certified route
    when the caller declares it will actually supply F32 activations (the
    c1/verifier F32-input route with an input override)."""

    f32_head_map = _synthetic_model_map(lm_head_type=GGMLQuantizationType.F32)
    report = preflight_qwen35_gguf_artifact(
        f32_head_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
        f32_input_operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
    )
    assert report.unsupported == (), report.render_refusals()
    head_records = [
        record
        for record in report.qualified_records
        if record.role_class == "lm_head"
    ]
    assert head_records
    assert all(record.input_dtype == "f32" for record in head_records)
    # The plan contract records the declared activation override so
    # certificates do not silently transfer across caller contracts.
    assert report.plan_contract is not None
    assert report.plan_contract.f32_input_operations == (
        QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,
    )
    # Same-manifest verification: the f32-declared certificate covers the
    # f32-declared intent (the c1 F32 valid route is certifiable), while a
    # certificate minted from a different artifact's plan never covers it.
    certificate = report.certificate()
    assert certificate_covers_artifact(
        certificate,
        manifest_fingerprint=f32_head_map_fingerprint(f32_head_map),
        plan_contract=report.plan_contract,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
    )
    bf16_report = preflight_qwen35_gguf_artifact(
        _synthetic_model_map(lm_head_type=GGMLQuantizationType.Q8_0),
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
    )
    assert bf16_report.supported
    assert not certificate_covers_artifact(
        bf16_report.certificate(),
        manifest_fingerprint=f32_head_map_fingerprint(f32_head_map),
        plan_contract=report.plan_contract,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
    )


def f32_head_map_fingerprint(model_map):
    return build_qwen35_gguf_role_manifest(model_map).fingerprint


def test_f32_input_declaration_is_validated_fail_closed():
    """Declaring an F32-input override for an operation that was not
    requested, or an unknown operation name, is a caller error."""

    with pytest.raises(Qwen35GGUFAdmissionError):
        preflight_qwen35_gguf_artifact(
            _synthetic_model_map(),
            backend="hip_gfx1100",
            operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
            f32_input_operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
        )
    with pytest.raises(Qwen35GGUFAdmissionError):
        preflight_qwen35_gguf_artifact(
            _synthetic_model_map(),
            backend="hip_gfx1100",
            operations=(QWEN35_GGUF_OP_AR_DECODE_C1,),
            f32_input_operations=("not_an_operation",),
        )


def test_declared_f32_input_does_not_widen_unsupported_layouts():
    """A supplied F32 pointer cannot use the pack8 BF16-input row.

    The head has no caller-side conversion adapter, so refuse before load.
    """

    pack8_map = _synthetic_model_map(lm_head_type=GGMLQuantizationType.Q4_K)
    report = preflight_qwen35_gguf_artifact(
        pack8_map,
        backend="hip_gfx1100",
        operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
        f32_input_operations=(QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS,),
    )
    assert not report.supported
    assert any(item.slot_path == "root.lm_head" for item in report.unsupported)
    assert not report.plan_contract.is_complete()
    assert not any(record.role_class == "lm_head" for record in report.qualified_records)
