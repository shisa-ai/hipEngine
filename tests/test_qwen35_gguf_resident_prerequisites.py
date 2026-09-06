"""CPU-only resident/repack shape contracts; no device or model payloads.

Converter arrays are small synthetic byte streams. Admission fixtures use the
real GGUF block geometry, not convenient guessed byte counts.
"""
from dataclasses import replace

import numpy as np
import pytest

from hipengine.loading import qwen35_gguf_materialize as materialize
from hipengine.loading.qwen35_gguf_admission import preflight_qwen35_gguf_artifact
from hipengine.quant.gguf import GGMLQuantizationType as Q
from hipengine.quant.gguf_q4_k import (
    repack_gguf_q4_k_pack8, repack_gguf_q4_k_tile16, repack_gguf_q4_k_tile16_qmicro,
)
from hipengine.quant.gguf_t16 import (
    repack_gguf_q5_k_qmicro_tile16, repack_gguf_q5_k_tile16,
    repack_gguf_q6_k_tile16, repack_gguf_q6_k_tile16_qmicro_planar,
    repack_gguf_q8_0_tile16,
)
from hipengine.quant.gguf_x8 import (
    repack_gguf_q4_k_x8,
    repack_gguf_q5_k_x8,
    repack_gguf_q6_k_x8,
)
from tests.test_gguf_ud_admission import _synthetic_model_map, _tensor


@pytest.mark.parametrize("qtype,slot,shape,flags,env,converter", [
    (Q.Q6_K, "ffn_down", (5121, 256), {"dense_q6_qmicro_planar": True}, {},
     repack_gguf_q6_k_tile16_qmicro_planar),
    (Q.Q5_K, "ffn_gate_exps", (2, 257, 256), {}, {}, repack_gguf_q5_k_qmicro_tile16),
    (Q.Q4_K, "ffn_gate_exps", (2, 257, 256), {},
     {"HIPENGINE_GGUF_SELECTED_GATE_UP_X8": "1"}, repack_gguf_q4_k_x8),
    (Q.Q5_K, "ffn_down_exps", (2, 257, 256), {},
     {"HIPENGINE_GGUF_SELECTED_X8_REPACK": "1"}, repack_gguf_q5_k_x8),
    (Q.Q6_K, "ffn_down_exps", (2, 257, 256), {},
     {"HIPENGINE_GGUF_SELECTED_X8_REPACK": "1"}, repack_gguf_q6_k_x8),
])
def test_byte_neutral_repack_refuses_before_byte_accounting(
    monkeypatch, qtype, slot, shape, flags, env, converter,
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    tensor = _tensor(f"blk.0.{slot}.weight", shape, qtype)
    model = _synthetic_model_map()
    model = replace(model, layers=(replace(model.layers[0], tensors={slot: tensor}),))
    raw = np.zeros(tensor.byte_shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="out_features"):
        converter(raw[None, ...] if raw.ndim == 2 else raw)

    # Neighboring legal N must still qualify and agree with the actual
    # converter's allocation, including every byte-neutral layout.
    legal_shape = (*shape[:-2], shape[-2] - 1, shape[-1])
    legal = _tensor(tensor.name, legal_shape, qtype)
    legal_model = replace(model, layers=(replace(model.layers[0], tensors={slot: legal}),))
    legal_report = preflight_qwen35_gguf_artifact(
        legal_model, backend="hip_gfx1100", operations=("ar_decode_c1", "ar_prefill"),
        decode_repack=True, slot_filter=(f"layers.0.{slot}",), **flags,
    )
    assert legal_report.supported, legal_report.render_refusals()
    legal_raw = np.zeros(legal.byte_shape, dtype=np.uint8)
    packed = converter(legal_raw[None, ...] if legal_raw.ndim == 2 else legal_raw)
    legal_spec = materialize.plan_qwen35_gguf_weight_spec(
        f"layers.0.{slot}", legal, decode_repack=True, **flags,
    )
    assert dict(materialize.planned_qwen35_gguf_weight_allocation_nbytes(legal_spec))["tiles"] == packed.tiles.nbytes

    def no_bytes(_spec):
        pytest.fail("invalid repack reached allocation-byte accounting")

    monkeypatch.setattr(
        "hipengine.loading.qwen35_gguf_admission.planned_qwen35_gguf_weight_allocation_nbytes",
        no_bytes,
    )
    report = preflight_qwen35_gguf_artifact(
        model, backend="hip_gfx1100", operations=("ar_decode_c1", "ar_prefill"),
        decode_repack=True, slot_filter=(f"layers.0.{slot}",), **flags,
    )
    assert not report.supported
    assert not report.plan_contract.is_complete()
    assert len(report.unsupported) == 2
    assert all(item.stage == "planner_refused" for item in report.unsupported)
    assert all("out_features" in item.reason for item in report.unsupported)


def test_real_loader_refuses_selected_resident_before_payload_or_allocation(monkeypatch, tmp_path):
    from tests._qwen35_gguf_fixture import (
        default_fixture_tensors, fixture_metadata, write_qwen35_gguf,
    )
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf_admission import Qwen35GGUFAdmissionError
    from hipengine.loading import materialize as host

    # Valid model geometry, but N65 cannot be repacked to the selected head T16.
    tensors = default_fixture_tensors(1)
    tensors[0] = ("token_embd.weight", (65, 256), Q.Q8_0)
    tensors.append(("output.weight", (65, 256), Q.Q6_K))
    path = tmp_path / "head-n65.gguf"
    write_qwen35_gguf(path, tensors, fixture_metadata(1))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        pytest.fail("payload/allocation before resident prerequisite refusal")

    monkeypatch.setattr(GGUFReader, "tensor_data", forbidden)
    monkeypatch.setattr(materialize, "malloc", forbidden)
    monkeypatch.setattr(host, "malloc", forbidden)
    # Even an operation that doesn't consume the head must not skip validation
    # of that selected resident: the loader still materializes it.
    for selected in (None, ("root.lm_head",)):
        with pytest.raises(Qwen35GGUFAdmissionError, match="out_features"):
            materialize.materialize_qwen35_gguf_weights(
                path, backend="hip_gfx1100", decode_repack=True,
                selected_slots=selected, requested_operations=("ar_decode_c1",),
            )
    assert calls == []


# Explicit expected converter/rank contracts, independent of the materializer
# route table. These are the 11 current primary repacks (not hypothetical quants).
REPACKS = [
    (materialize.LAYOUT_Q4_K_PACK8, Q.Q4_K, 2, 8, False, repack_gguf_q4_k_pack8),
    (materialize.LAYOUT_GGUF_Q4_K_T16, Q.Q4_K, 3, 16, True, repack_gguf_q4_k_tile16),
    (materialize.LAYOUT_GGUF_Q4_K_QMICRO_T16, Q.Q4_K, 3, 16, True, repack_gguf_q4_k_tile16_qmicro),
    (materialize.LAYOUT_GGUF_Q5_K_T16, Q.Q5_K, 3, 16, True, repack_gguf_q5_k_tile16),
    (materialize.LAYOUT_GGUF_Q5_K_QMICRO_T16, Q.Q5_K, 3, 16, False, repack_gguf_q5_k_qmicro_tile16),
    (materialize.LAYOUT_GGUF_Q6_K_T16, Q.Q6_K, 3, 16, True, repack_gguf_q6_k_tile16),
    (materialize.LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, Q.Q6_K, 3, 16, True, repack_gguf_q6_k_tile16_qmicro_planar),
    (materialize.LAYOUT_GGUF_Q8_0_T16, Q.Q8_0, 2, 16, False, repack_gguf_q8_0_tile16),
    (materialize.LAYOUT_GGUF_Q4_K_X8, Q.Q4_K, 3, 8, False, repack_gguf_q4_k_x8),
    (materialize.LAYOUT_GGUF_Q5_K_X8, Q.Q5_K, 3, 8, False, repack_gguf_q5_k_x8),
    (materialize.LAYOUT_GGUF_Q6_K_X8, Q.Q6_K, 3, 8, False, repack_gguf_q6_k_x8),
]


def _spec(layout, qtype, shape, names=None):
    if names is None:
        names = ("qweight", "scales", "mins") if layout == materialize.LAYOUT_Q4_K_PACK8 else ("tiles",)
    return materialize.Qwen35GGUFWeightSpec(
        "layers.0.ffn_down_exps", _tensor("weight", shape, qtype),
        layout, layout, names,
    )


def test_multiple_invalid_residents_aggregate_in_full_and_filtered_preflight(monkeypatch):
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_GATE_UP_X8", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_X8_REPACK", "1")
    tensors = {
        "ffn_down": _tensor("down", (5121, 256), Q.Q6_K),
        "ffn_gate_exps": _tensor("gate", (2, 257, 256), Q.Q4_K),
        "ffn_up_exps": _tensor("up", (2, 257, 256), Q.Q5_K),
        "ffn_down_exps": _tensor("experts_down", (2, 257, 256), Q.Q6_K),
    }
    model = _synthetic_model_map()
    model = replace(model, layers=(replace(model.layers[0], tensors=tensors),))
    for selected in (None, ("layers.0.ffn_down", "layers.0.ffn_up_exps")):
        report = preflight_qwen35_gguf_artifact(
            model, backend="hip_gfx1100", operations=("ar_decode_c1", "ar_prefill"),
            decode_repack=True, dense_q6_qmicro_planar=True, slot_filter=selected,
        )
        expected = set(selected) if selected is not None else {"layers.0." + slot for slot in tensors}
        assert {item.slot_path for item in report.unsupported} == expected
        assert len(report.unsupported) == 2 * len(expected)
        assert all(item.stage == "planner_refused" for item in report.unsupported)
        assert not report.plan_contract.is_complete()


def test_converter_dispatch_has_no_unchecked_primary_routes():
    assert set(materialize._RESIDENT_REPACKS) == {row[0] for row in REPACKS}
    for layout, _qtype, _rank, _columns, promote, converter in REPACKS:
        route = materialize._RESIDENT_REPACKS[layout]
        assert route.converter is converter
        assert route.promote_dense == promote


@pytest.mark.parametrize("layout,qtype,rank,columns,promote,converter", REPACKS)
def test_all_primary_repack_boundaries_match_real_converters(
    monkeypatch, layout, qtype, rank, columns, promote, converter,
):
    from hipengine.quant.gguf_repack import GGUFRepackShape

    validate = materialize.validate_qwen35_gguf_resident_prerequisites
    # Observe identity of the shared contract used inside the REAL converter;
    # this assertion prevents metadata and converters growing separate rules.
    seen = []
    original_validate = GGUFRepackShape.validate

    def observe(self, byte_shape):
        seen.append(self)
        return original_validate(self, byte_shape)

    monkeypatch.setattr(GGUFRepackShape, "validate", observe)
    k = 32 if qtype == Q.Q8_0 else 256
    prefix = (2,) if rank == 3 else ()
    shapes = [
        (*prefix, columns, k), (*prefix, columns * 2, k * 2),
        (*prefix, columns - 1, k), (*prefix, columns + 1, k),
        (*prefix, 0, k), (*prefix, columns, 0), (k,),
        (1, 1, columns, k),
    ]
    if rank == 3:
        shapes += [(0, columns, k), (columns, k)]
    else:
        shapes += [(1, columns, k)]
    for shape in shapes:
        spec = _spec(layout, qtype, shape)
        raw = np.zeros(spec.source.byte_shape, dtype=np.uint8)
        if promote and raw.ndim == 2:
            raw = raw[None, ...]
        seen.clear()
        try:
            packed = converter(raw)
        except ValueError:
            assert materialize._RESIDENT_REPACKS[layout].shape in seen
            with pytest.raises(ValueError):
                validate(spec)
        else:
            assert materialize._RESIDENT_REPACKS[layout].shape in seen
            validate(spec)
            sizes = dict(materialize.planned_qwen35_gguf_weight_allocation_nbytes(spec))
            if layout == materialize.LAYOUT_Q4_K_PACK8:
                assert sizes == {name: getattr(packed, name).nbytes for name in sizes}
            else:
                assert sizes["tiles"] == packed.tiles.nbytes
    assert materialize._RESIDENT_REPACKS[layout].shape in seen

    # Illegal source row byte length: compare the converter's block boundary
    # with the admission source-consistency boundary (no payload conversion).
    valid = _spec(layout, qtype, (*prefix, columns, k))
    byte_shape = (*valid.source.byte_shape[:-1], valid.source.byte_shape[-1] - 1)
    with pytest.raises(ValueError, match="bytes_per_row"):
        converter(np.zeros(byte_shape, dtype=np.uint8))
    with pytest.raises(ValueError, match="byte_shape"):
        validate(replace(valid, source=replace(valid.source, byte_shape=byte_shape)))
    with pytest.raises(ValueError, match="row size"):
        validate(replace(valid, source=replace(valid.source, shape=(*valid.source.shape[:-1], k + 1))))
    with pytest.raises(ValueError, match="positive"):
        validate(replace(valid, source=replace(valid.source, shape=(*valid.source.shape[:-2], -columns, k))))


@pytest.mark.parametrize("layout,qtype,rank,columns,promote,converter", REPACKS)
def test_actual_materializer_uses_the_validated_converter_route(
    monkeypatch, layout, qtype, rank, columns, promote, converter,
):
    from types import SimpleNamespace

    shape = (2, columns, 256) if rank == 3 else (columns, 256)
    spec = _spec(layout, qtype, shape)
    raw = np.zeros(spec.source.byte_shape, dtype=np.uint8)
    reader = SimpleNamespace(tensor_data=lambda name: raw)
    uploads = []

    def upload(name, array, dtype, **kwargs):
        uploads.append(array.copy())
        return SimpleNamespace(tensor=SimpleNamespace(dtype=dtype))

    monkeypatch.setattr(materialize, "_load_host_array_to_device_as_dtype", upload)
    resident = materialize.materialize_qwen35_gguf_weight_spec(spec, reader)
    expected = converter(raw[None, ...] if promote and raw.ndim == 2 else raw)
    fields = ("qweight", "scales", "mins") if layout == materialize.LAYOUT_Q4_K_PACK8 else ("tiles",)
    assert set(resident.allocations) == set(fields)
    for actual, field in zip(uploads, fields, strict=True):
        np.testing.assert_array_equal(actual, getattr(expected, field))


def test_allocated_sidecars_validate_independently_before_payload():
    validate = materialize.validate_qwen35_gguf_resident_prerequisites
    # pack8 itself accepts N8; its selected T16 sidecar requires N16.
    for name in (materialize.Q4_T16_DECODE_TILES, materialize.Q4_T16_DECODE_TILES_R3PLUS):
        spec = _spec(materialize.LAYOUT_Q4_K_PACK8, Q.Q4_K, (8, 256))
        validate(spec)
        invalid = replace(spec, allocation_names=(*spec.allocation_names, name))
        with pytest.raises(ValueError, match="divisible by 16"):
            validate(invalid)
        with pytest.raises(ValueError, match="divisible by 16"):
            materialize.materialize_qwen35_gguf_weight_spec(invalid, None)
        legal = replace(invalid, source=_tensor("weight", (16, 256), Q.Q4_K))
        validate(legal)
        sidecar = repack_gguf_q4_k_tile16(np.zeros((1, 16, 144), dtype=np.uint8))
        assert dict(materialize.planned_qwen35_gguf_weight_allocation_nbytes(legal))[name] == sidecar.tiles.nbytes

    for layout, qtype, name, converter in (
        (materialize.LAYOUT_GGUF_Q6_K_T16, Q.Q6_K, "x8", repack_gguf_q6_k_x8),
        (materialize.LAYOUT_GGUF_Q5_K_T16, Q.Q5_K, "qmicro_planar", repack_gguf_q5_k_qmicro_tile16),
        (materialize.LAYOUT_GGUF_Q8_0_T16, Q.Q8_0, "raw", None),
    ):
        spec = _spec(layout, qtype, (16, 256), ("tiles", name))
        validate(spec)
        raw = np.zeros(spec.source.byte_shape, dtype=np.uint8)
        if name == "qmicro_planar":
            from hipengine.quant.gguf_t16 import convert_gguf_q5_k_qmicro_tile16_to_planar
            result = convert_gguf_q5_k_qmicro_tile16_to_planar(converter(raw[None, ...]))
            nbytes = result.tiles.nbytes
        elif converter:
            nbytes = converter(raw[None, ...]).tiles.nbytes
        else:
            nbytes = raw.nbytes
        assert dict(materialize.planned_qwen35_gguf_weight_allocation_nbytes(spec))[name] == nbytes
    for layout, qtype, name in (
        (materialize.LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR, Q.Q6_K, "x8"),
        (materialize.LAYOUT_GGUF_Q6_K_T16, Q.Q6_K, "qmicro_planar"),
        (materialize.LAYOUT_GGUF_Q5_K_T16, Q.Q5_K, "unknown"),
    ):
        with pytest.raises(ValueError, match="allocations"):
            validate(_spec(layout, qtype, (16, 256), ("tiles", name)))
    with pytest.raises(ValueError, match="unsupported resident layout"):
        validate(_spec(materialize.LAYOUT_GGUF_Q5_K_QMICRO_PLANAR, Q.Q5_K, (2, 16, 256)))


@pytest.mark.parametrize("qtype", [Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K, Q.IQ2_XS, Q.IQ3_XXS, Q.IQ4_XS])
def test_raw_experts_do_not_inherit_optional_sidecar_alignment(qtype):
    # No allocated repack: N257 is legal raw storage. A potential, separately
    # selected expert sidecar is not permission to impose its N8 alignment now.
    spec = _spec(materialize.LAYOUT_RAW_GGUF, qtype, (2, 257, 256), ("raw",))
    spec = replace(spec, sidecar_layouts=(materialize.LAYOUT_GGUF_EXPERT_PACK8_SIDECAR,))
    materialize.validate_qwen35_gguf_resident_prerequisites(spec)


@pytest.mark.parametrize("layout,qtype,shape", [
    (materialize.LAYOUT_DENSE_F32, Q.F32, (7,)),
    (materialize.LAYOUT_DENSE_BF16, Q.F32, (7, 3)),
    (materialize.LAYOUT_DENSE_BF16, Q.BF16, (7, 3)),
    (materialize.LAYOUT_DENSE_BF16, Q.Q5_K, (7, 256)),
    (materialize.LAYOUT_RAW_GGUF, Q.Q8_0, (7, 32)),
])
def test_nonrepacked_residents_use_source_geometry_not_tile_alignment(layout, qtype, shape):
    spec = _spec(layout, qtype, shape, ("raw",))
    materialize.validate_qwen35_gguf_resident_prerequisites(spec)
    for field, value in (("nbytes", spec.source.nbytes + 1), ("n_elements", spec.source.n_elements + 1)):
        with pytest.raises(ValueError, match="counts"):
            materialize.validate_qwen35_gguf_resident_prerequisites(
                replace(spec, source=replace(spec.source, **{field: value})),
            )
