from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading import qwen35_gguf_nextn_materialize as nextn_materialize
from hipengine.loading.qwen35_gguf_materialize import Qwen35GGUFWeightSpec
from hipengine.loading.qwen35_gguf_nextn_materialize import (
    Qwen35GGUFNextNMaterializationPlan,
    materialize_qwen35_gguf_nextn_weights,
)
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
    monkeypatch.setattr(nextn_materialize, "plan_qwen35_gguf_nextn_materialization", lambda model_map: plan)
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
