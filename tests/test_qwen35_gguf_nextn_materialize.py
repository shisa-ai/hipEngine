from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo
from hipengine.loading.gguf_mtp_hot_vocab import GGUFHotVocabSelection
from hipengine.loading import qwen35_gguf_nextn_materialize as nextn_materialize
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    Qwen35GGUFWeightSpec,
)
from hipengine.loading.qwen35_gguf_nextn_materialize import (
    Qwen35GGUFNextNMaterializationPlan,
    materialize_qwen35_gguf_nextn_weights,
    plan_qwen35_gguf_nextn_materialization,
)
from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map
from hipengine.quant.gguf import GGMLQuantizationType


class _FakeWeight:
    def __init__(self, spec: Qwen35GGUFWeightSpec, *, backend: str = "hip_gfx1100") -> None:
        self.spec = spec
        self.backend = backend
        self.free_calls = 0

    def free(self, *, runtime=None) -> None:
        del runtime
        self.free_calls += 1


def _tensor(name: str, qtype: GGMLQuantizationType, shape: tuple[int, ...] = (2, 3)) -> GGUFTensorInfo:
    elements = 1
    for value in shape:
        elements *= value
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=elements,
        nbytes=max(elements, 1),
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )


def _spec(
    slot_path: str,
    source: GGUFTensorInfo,
    *,
    quant_key: str,
    layout: str,
    allocation: str,
) -> Qwen35GGUFWeightSpec:
    return Qwen35GGUFWeightSpec(
        slot_path=slot_path,
        source=source,
        quant_key=quant_key,
        layout=layout,
        allocation_names=(allocation,),
    )


@pytest.mark.parametrize(
    ("model", "expected_q4_count"),
    (
        (Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf"), 5),
        (Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"), 5),
        (Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf"), 7),
    ),
)
def test_dense_nextn_plan_uses_sole_t16_for_all_q4_draft_weights(
    model: Path,
    expected_q4_count: int,
) -> None:
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    info = GGUFReader(model).info
    if not any(".nextn." in tensor.name for tensor in info.tensors):
        pytest.skip(f"local GGUF fixture has no trailing NextN block: {model}")
    model_map = build_qwen35_gguf_nextn_tensor_map(info)

    plan = plan_qwen35_gguf_nextn_materialization(
        model_map,
        decode_repack=True,
        dense_q4_t16=True,
        dense_q6_qmicro_planar=True,
    )
    q4_specs = tuple(
        spec for spec in plan.draft_specs if spec.source.ggml_type_name == "Q4_K"
    )

    assert len(q4_specs) == expected_q4_count
    assert all(spec.layout == LAYOUT_GGUF_Q4_K_T16 for spec in q4_specs)
    assert all(spec.quant_key == "gguf_q4_k_t16_v1" for spec in q4_specs)
    assert all(spec.allocation_names == ("tiles",) for spec in q4_specs)
    expected_wide_layout = (
        LAYOUT_GGUF_Q4_K_T16 if info.file_type_name == "MOSTLY_Q4_K_S" else "raw_gguf"
    )
    assert plan.layer_specs["attn_v"].layout == expected_wide_layout
    assert plan.layer_specs["ffn_down"].layout == expected_wide_layout
    lm_head = plan.fallback_specs["lm_head"]
    assert lm_head.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
    assert lm_head.quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
    assert lm_head.allocation_names == ("tiles",)


def test_moe_q4km_nextn_map_accepts_actual_expert_qtypes() -> None:
    model = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")

    model_map = build_qwen35_gguf_nextn_tensor_map(GGUFReader(model).info)

    assert model_map.validation.passed
    assert model_map.tensor("ffn_gate_exps").ggml_type_name == "Q4_K"
    assert model_map.tensor("ffn_up_exps").ggml_type_name == "Q4_K"
    assert model_map.tensor("ffn_down_exps").ggml_type_name == "Q5_K"


def test_hot_vocab_materializer_repacks_individually_selected_q6_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _tensor("output.weight", GGMLQuantizationType.Q6_K, (32, 256))
    spec = _spec(
        "root.lm_head",
        source,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        allocation="tiles",
    )
    selection = GGUFHotVocabSelection(
        token_ids=tuple(range(0, 32, 2)),
        tokenizer_tokens_sha256="fixture",
        source_path=tmp_path / "hot.json",
        metadata={},
    )
    raw = np.arange(32 * 210, dtype=np.uint8).reshape(32, 210)
    reader = SimpleNamespace(
        info=SimpleNamespace(metadata={"tokenizer.ggml.tokens": [str(i) for i in range(32)]}),
        tensor_data=lambda name: raw,
    )
    repacked_rows = []
    allocations = []

    def fake_repack(selected):
        repacked_rows.append(selected.copy())
        return SimpleNamespace(tiles=np.zeros((1, 1, 1, 3360), dtype=np.uint8))

    def fake_load(name, array, dtype, **kwargs):
        allocation = SimpleNamespace(
            name=name,
            array=np.asarray(array).copy(),
            tensor=SimpleNamespace(ptr=0x1000 + len(allocations) * 0x100),
            free=lambda **free_kwargs: None,
        )
        allocations.append(allocation)
        return allocation

    monkeypatch.setattr(nextn_materialize, "repack_gguf_q6_k_tile16_qmicro_planar", fake_repack)
    monkeypatch.setattr(nextn_materialize, "load_host_array_to_device_as_dtype", fake_load)

    hot = nextn_materialize._materialize_hot_vocab(
        reader,
        spec,
        selection,
        device=None,
        runtime=None,
        backend="hip_gfx1100",
    )

    np.testing.assert_array_equal(repacked_rows[0][0], raw[list(selection.token_ids)])
    assert hot.size == 16
    assert hot.lm_head.spec.slot_path == "draft.hot_lm_head"
    assert hot.lm_head.allocation("tiles") is allocations[0]
    np.testing.assert_array_equal(allocations[1].array, np.asarray(selection.token_ids, dtype=np.int32))


def test_nextn_materialization_borrows_compatible_target_fallbacks_without_owning_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = _tensor("token_embd.weight", GGMLQuantizationType.Q4_K)
    output_norm = _tensor(
        "blk.64.nextn.shared_head_norm.weight",
        GGMLQuantizationType.F32,
        (3,),
    )
    target_output_norm = _tensor("output_norm.weight", GGMLQuantizationType.F32, (3,))
    lm_head = _tensor("output.weight", GGMLQuantizationType.Q6_K)
    draft_layer = _tensor("blk.64.attn_q.weight", GGMLQuantizationType.Q8_0)
    draft_nextn = _tensor("blk.64.nextn.eh_proj.weight", GGMLQuantizationType.Q8_0)
    fallback_specs = MappingProxyType(
        {
            "token_embedding": _spec(
                "root.token_embedding",
                embedding,
                quant_key="gguf_q4_k",
                layout="raw_gguf",
                allocation="raw",
            ),
            "output_norm": _spec(
                "root.output_norm",
                output_norm,
                quant_key="f32",
                layout="dense_f32",
                allocation="raw",
            ),
            "lm_head": _spec(
                "root.lm_head",
                lm_head,
                quant_key="gguf_q6_k",
                layout="raw_gguf",
                allocation="raw",
            ),
        }
    )
    plan = Qwen35GGUFNextNMaterializationPlan(
        model_map=SimpleNamespace(config=object(), block_id=64),
        layer_specs=MappingProxyType(
            {
                "attn_q": _spec(
                    "draft.layer.attn_q",
                    draft_layer,
                    quant_key="gguf_q8_0",
                    layout="raw_gguf",
                    allocation="raw",
                )
            }
        ),
        nextn_specs=MappingProxyType(
            {
                "eh_proj": _spec(
                    "draft.nextn.eh_proj",
                    draft_nextn,
                    quant_key="gguf_q8_0",
                    layout="raw_gguf",
                    allocation="raw",
                )
            }
        ),
        fallback_specs=fallback_specs,
    )
    borrowed = {
        "token_embedding": _FakeWeight(fallback_specs["token_embedding"]),
        "lm_head": _FakeWeight(
            _spec(
                "root.lm_head",
                lm_head,
                quant_key="gguf_q6_k_t16_v1",
                layout="gguf_q6_k_t16_v1",
                allocation="tiles",
            )
        ),
    }
    materialized: list[_FakeWeight] = []

    class _FakeReader:
        def __init__(self, path) -> None:
            self.path = path
            self.info = object()

    def fake_materialize(spec, reader, *, device, runtime, backend):
        del reader, device, runtime
        weight = _FakeWeight(spec, backend=backend)
        materialized.append(weight)
        return weight

    monkeypatch.setattr(nextn_materialize, "GGUFReader", _FakeReader)
    monkeypatch.setattr(nextn_materialize, "build_qwen35_gguf_nextn_tensor_map", lambda info: object())
    monkeypatch.setattr(
        nextn_materialize,
        "plan_qwen35_gguf_nextn_materialization",
        lambda model_map, **kwargs: plan,
    )
    monkeypatch.setattr(nextn_materialize, "materialize_qwen35_gguf_weight_spec", fake_materialize)

    resident = materialize_qwen35_gguf_nextn_weights(
        "/tmp/fake.gguf",
        borrowed_fallback_weights=borrowed,
    )

    assert [weight.spec.slot_path for weight in materialized] == [
        "draft.layer.attn_q",
        "draft.nextn.eh_proj",
        "root.output_norm",
    ]
    assert resident.fallback("token_embedding") is borrowed["token_embedding"]
    assert resident.fallback("lm_head") is borrowed["lm_head"]
    assert resident.fallback("output_norm") is materialized[-1]
    assert resident.plan.fallback_specs["lm_head"] is borrowed["lm_head"].spec
    assert resident.plan.fallback_specs["output_norm"] is fallback_specs["output_norm"]
    assert resident.owned_weights == tuple(materialized)

    resident.free()

    assert [weight.free_calls for weight in materialized] == [1, 1, 1]
    assert [weight.free_calls for weight in borrowed.values()] == [0, 0]

    incompatible = {
        **borrowed,
        "output_norm": _FakeWeight(
            _spec(
                "root.output_norm",
                target_output_norm,
                quant_key="f32",
                layout="dense_f32",
                allocation="raw",
            )
        ),
    }
    with pytest.raises(ValueError, match="does not match the NextN source tensor"):
        materialize_qwen35_gguf_nextn_weights(
            "/tmp/fake.gguf",
            borrowed_fallback_weights=incompatible,
        )
