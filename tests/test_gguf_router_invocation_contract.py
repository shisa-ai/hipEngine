"""CPU contracts at the production router caller (not just registry keys)."""
from types import SimpleNamespace as NS

import pytest

from hipengine.loading.qwen35_gguf_consumer_surface import resolve_router_consumer_contract
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_DENSE_BF16, LAYOUT_DENSE_F32
from hipengine.loading.qwen35_gguf_admission import (
    QWEN35_GGUF_OP_AR_DECODE_C1 as C1,
    certificate_covers_artifact,
    preflight_qwen35_gguf_artifact,
)
from test_gguf_ud_admission import _synthetic_moe_model_map


@pytest.mark.parametrize("backend", ("hip_gfx1100", "hip_gfx1151"))
@pytest.mark.parametrize("layout,quant,activation", (
    (LAYOUT_DENSE_BF16, "bf16", "bf16"),
    (LAYOUT_DENSE_F32, "f32", "bf16"),
    (LAYOUT_DENSE_F32, "f32", "f32"),
))
def test_production_router_uses_shared_key_and_operands(monkeypatch, backend, layout, quant, activation):
    import hipengine.runtime.qwen35_gguf_runner as runner
    contract = resolve_router_consumer_contract(layout, quant, activation)
    resolutions, launches = [], []
    def resolve(**key):
        resolutions.append(key)
        return lambda *a, **kw: launches.append((a, kw))
    monkeypatch.setattr(runner, "resolve", resolve)
    weight = NS(backend=backend, spec=NS(layout=layout, quant_key=quant),
                allocation=lambda: NS(tensor=NS(ptr=102)))
    caller = getattr(runner, f"_launch_qwen35_router_logits_{activation}_hidden")
    caller(101, weight, 103, 2, 256, 4, stream=7, runtime="fake")
    assert resolutions == [dict(backend=backend, layer="router_logits", quant=quant, variant=f"{activation}_hidden")]
    assert launches == [((101, 102, 103, 2, 256, 4), dict(stream=7, runtime="fake"))]
    assert contract.operands == (("hidden_ptr", activation, "read"), ("weight_ptr", quant, "read"), ("logits_ptr", "f32", "write"))


def test_f32_pointer_cannot_fall_back_to_bf16_router(monkeypatch):
    import hipengine.runtime.qwen35_gguf_runner as runner
    monkeypatch.setattr(runner, "resolve", lambda **kw: pytest.fail("resolved an unsupported router ABI"))
    weight = NS(backend="hip_gfx1100", spec=NS(layout=LAYOUT_DENSE_BF16, quant_key="bf16"))
    with pytest.raises(ValueError, match="unsupported router invocation"):
        runner._launch_qwen35_router_logits_f32_hidden(1, weight, 3, 2, 256, 4, runtime="fake")


def test_two_successful_router_operand_contracts_do_not_transfer():
    model = _synthetic_moe_model_map()
    slots = ("layers.0.ffn_gate_inp",)
    reports = [preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100",
               operations=(C1,), slot_filter=slots, contract_f32_linear=False,
               f32_input_operations=override, selected_call_intents=())
               for override in ((), (C1,))]
    bf16, f32 = reports
    assert bf16.supported and f32.supported
    assert bf16.plan_contract.resident_plan_records == f32.plan_contract.resident_plan_records
    left, = bf16.plan_contract.invocations
    right, = f32.plan_contract.invocations
    assert left.consumer.operands[0][1] == "bf16"
    assert right.consumer.operands[0][1] == "f32"
    for source, intended in ((bf16, f32), (f32, bf16)):
        assert not certificate_covers_artifact(source.certificate(),
            manifest_fingerprint=intended.manifest_fingerprint,
            plan_contract=intended.plan_contract, slot_filter=slots)
