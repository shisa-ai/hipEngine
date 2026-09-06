"""Explicit selected-call scope and ordered partner ownership (CPU only)."""
from dataclasses import replace
from types import MappingProxyType

import pytest

from test_gguf_ud_admission import _synthetic_moe_model_map
from hipengine.loading.qwen35_gguf_admission import (
    preflight_qwen35_gguf_artifact, certificate_covers_artifact,
    QWEN35_GGUF_OP_AR_DECODE_C1 as C1,
)

GATE = "layers.0.ffn_gate_exps"
UP = "layers.0.ffn_up_exps"
DOWN = "layers.0.ffn_down_exps"


def intent(kind, slots, **kwargs):
    from hipengine.loading.gguf_selected_contract import SelectedCallIntent
    return SelectedCallIntent(C1, kind, tuple(slots), **kwargs)


def preflight(model=None, *, calls=None, slots=None, backend="hip_gfx1100"):
    return preflight_qwen35_gguf_artifact(model or _synthetic_moe_model_map(),
        backend=backend, operations=(C1,), decode_repack=False,
        selected_call_intents=calls, slot_filter=slots)


def covers(left, right, **kwargs):
    return certificate_covers_artifact(left.certificate(),
        manifest_fingerprint=right.manifest_fingerprint, plan_contract=right.plan_contract,
        slot_filter=right.slot_filter, **kwargs)


def test_default_pair_does_not_become_single_under_filter():
    whole = preflight()
    assert whole.supported
    assert whole.plan_contract.selected_invocations[0:]
    partial = preflight(slots=(GATE,))
    assert not partial.supported
    assert not partial.plan_contract.is_complete()
    assert any(UP in item.reason for item in partial.unsupported)
    assert not covers(whole, partial)


def test_explicit_single_diagnostic_cannot_authorize_pair():
    single = preflight(calls=(intent("single", (GATE,)),), slots=(GATE,))
    pair = preflight(calls=(intent("dual_silu", (GATE, UP)),), slots=(GATE, UP))
    assert single.supported and pair.supported
    assert covers(single, single) and covers(pair, pair)
    assert not covers(single, pair)
    assert not covers(pair, single)


def test_partner_source_and_order_bind_even_with_same_manifest():
    left_model = _synthetic_moe_model_map()
    tensors = dict(left_model.layers[0].tensors)
    tensors["ffn_up_exps"] = replace(tensors["ffn_up_exps"], name="different_up.weight")
    right_model = replace(left_model, layers=(replace(left_model.layers[0], tensors=MappingProxyType(tensors)),))
    call = intent("dual_silu", (GATE, UP))
    left = preflight(left_model, calls=(call,), slots=(GATE, UP))
    right = preflight(right_model, calls=(call,), slots=(GATE, UP))
    reversed_pair = preflight(left_model, calls=(intent("dual_silu", (UP, GATE)),), slots=(GATE, UP))
    assert left.supported and right.supported and reversed_pair.supported
    assert left.manifest_fingerprint == right.manifest_fingerprint
    assert not covers(left, right)
    assert not covers(left, reversed_pair)
    # A genuinely explicit singleton does not consume the up weight.
    single_call = intent("single", (GATE,))
    single_left = preflight(left_model, calls=(single_call,), slots=(GATE,))
    single_right = preflight(right_model, calls=(single_call,), slots=(GATE,))
    assert covers(single_left, single_right)
    binding = left.plan_contract.selected_invocations[0]
    assert tuple(slot for slot, _ in binding.weight_bindings) == (GATE, UP)


@pytest.mark.parametrize("slots", [(GATE,), (), (UP,)])
def test_explicit_pair_cannot_drop_dependencies(slots):
    report = preflight(calls=(intent("dual_silu", (GATE, UP)),), slots=slots)
    assert not report.supported
    assert not report.plan_contract.is_complete()
    with pytest.raises(ValueError):
        report.certificate()


def test_weighted_down_route_owner_and_backend_are_binding():
    call = intent("weighted_down", (DOWN,))
    left = preflight(calls=(call,), slots=(DOWN,))
    right = preflight(calls=(replace(call, routing_owner="other-route-weights"),), slots=(DOWN,))
    other_backend = preflight(calls=(call,), slots=(DOWN,), backend="hip_gfx1151")
    assert left.supported and right.supported and other_backend.supported
    assert left.plan_contract.resident_plan_records == right.plan_contract.resident_plan_records
    assert not covers(left, right)
    assert not covers(left, other_backend)
    operands = left.plan_contract.selected_invocations[0].operands
    assert ("routing_weights_ptr", "f32", "read") in operands


def test_empty_intents_do_not_qualify_selected_residents():
    report = preflight(calls=(), slots=(GATE, UP))
    assert not report.supported
    assert not report.plan_contract.is_complete()


def test_real_loader_refuses_missing_partner_before_payload_or_allocation(tmp_path, monkeypatch):
    from tests._qwen35_gguf_fixture import default_fixture_tensors, fixture_metadata, write_qwen35_gguf
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    from hipengine.loading.gguf import GGUFReader
    import hipengine.loading.qwen35_gguf_materialize as loader
    tensors = [t for t in default_fixture_tensors() if not t[0].startswith("blk.0.ffn_")]
    tensors += [("output.weight", (64, 256), Q.Q8_0),
                ("blk.0.ffn_gate_inp.weight", (4, 256), Q.F32),
                ("blk.0.ffn_gate_inp_shexp.weight", (256,), Q.F32)]
    for suffix in ("gate", "up", "down"):
        tensors.extend([(f"blk.0.ffn_{suffix}_exps.weight", (4, 256, 256), Q.Q4_K),
                        (f"blk.0.ffn_{suffix}_shexp.weight", (256, 256), Q.Q4_K)])
    metadata = [(name.replace("qwen35.", "qwen35moe."), typ,
                 "qwen35moe" if name == "general.architecture" else value)
                for name, typ, value in fixture_metadata(1)]
    metadata += [("qwen35moe." + name, 4, value) for name, value in (
        ("expert_count", 4), ("expert_used_count", 2),
        ("expert_feed_forward_length", 256), ("expert_shared_feed_forward_length", 256))]
    path = write_qwen35_gguf(tmp_path / "moe.gguf", tensors, metadata)
    def forbidden(*args, **kwargs):
        pytest.fail("payload/allocation before selected partner refusal")
    monkeypatch.setattr(GGUFReader, "tensor_data", forbidden)
    monkeypatch.setattr(loader, "malloc", forbidden)
    with pytest.raises(ValueError, match="requires resident partner"):
        loader.materialize_qwen35_gguf_weights(path, selected_slots=(GATE,),
            requested_operations=(C1,), decode_repack=False)


def test_actual_selected_callers_use_ordered_operands(monkeypatch):
    from types import SimpleNamespace as NS
    from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_weight_spec
    import hipengine.runtime.qwen35_gguf_runner as runner
    model = _synthetic_moe_model_map()
    def weight(slot, ptr):
        spec = plan_qwen35_gguf_weight_spec(slot, model.layers[0].tensors[slot.rsplit(".", 1)[1]], decode_repack=False)
        return NS(spec=spec, backend="hip_gfx1100", allocation=lambda name: NS(tensor=NS(ptr=ptr)))
    gate, up, down = weight(GATE, 11), weight(UP, 12), weight(DOWN, 13)
    calls = []
    def resolve(quant, variant):
        return lambda *args, **kw: calls.append((quant, variant, args, kw))
    monkeypatch.setattr(runner, "_resolve_exact_selected_moe_kernel", resolve)
    geometry = dict(x_rows=1, rows=2, num_experts=4, in_features=256, out_features=256, stream=0, runtime="fake")
    assert runner._launch_selected_raw_gguf_moe_pair_silu(gate, up, 21, 22, 23, **geometry)
    assert runner._launch_selected_raw_gguf_moe_pair_silu(up, gate, 21, 22, 23, **geometry)
    runner._launch_selected_raw_gguf_moe_linear(gate, 21, 22, 23, **geometry)
    assert runner._launch_weighted_selected_raw_gguf_moe_linear(down, 21, 22, 24, 23,
        tokens=1, top_k=2, num_experts=4, in_features=256, out_features=256, stream=0, runtime="fake")
    assert [item[2] for item in calls] == [(21, 22, 11, 12, 23), (21, 22, 12, 11, 23),
                                         (21, 22, 11, 23), (21, 22, 24, 13, 23)]
    assert [item[1] for item in calls] == [runner._SELECTED_MOE_DUAL_SILU_VARIANT] * 2 + [runner._SELECTED_MOE_SINGLE_VARIANT, runner._SELECTED_MOE_WEIGHTED_DOWN_VARIANT]


def test_dual_boundary_keeps_order_and_exact_two_single_fallback(monkeypatch):
    import hipengine.runtime.qwen35_gguf_runner as runner
    calls = []
    monkeypatch.setattr(runner, "_launch_selected_raw_gguf_moe_pair", lambda *a, **kw: False)
    monkeypatch.setattr(runner, "_launch_selected_raw_gguf_moe_linear", lambda *a, **kw: calls.append((a, kw)))
    runner._launch_selected_raw_gguf_moe_dual("gate", "up", 1, 2, 3, 4,
        x_rows=1, rows=2, num_experts=4, in_features=256, out_features=256,
        q8_1_workspace_ptr=99, stream=7, runtime="fake")
    assert [item[0] for item in calls] == [("gate", 1, 2, 3), ("up", 1, 2, 4)]
    assert all("q8_1_workspace_ptr" not in kwargs for _, kwargs in calls)


def test_x8_actual_input_adapters_and_output_contracts_bind(monkeypatch):
    from types import SimpleNamespace as NS
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_weight_spec
    import hipengine.runtime.qwen35_gguf_runner as runner
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_X8_REPACK", "q5")
    model = _synthetic_moe_model_map(expert_type=Q.Q5_K)
    reports = []
    for dtype in ("bf16", "f32"):
        call = intent("single", (DOWN,), input_dtype=dtype, output_dtype=dtype)
        reports.append(preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100",
            operations=(C1,), decode_repack=True, slot_filter=(DOWN,), selected_call_intents=(call,)))
    left, right = reports
    assert left.supported and right.supported
    assert left.plan_contract.resident_plan_records == right.plan_contract.resident_plan_records
    assert not covers(left, right)
    spec = plan_qwen35_gguf_weight_spec(DOWN, model.layers[0].tensors["ffn_down_exps"], decode_repack=True)
    weight = NS(spec=spec, backend="hip_gfx1100", allocation=lambda name: NS(tensor=NS(ptr=11)))
    launches, adapters = [], []
    monkeypatch.setattr(runner, "_resolve_exact_selected_moe_kernel", lambda *a: None)
    monkeypatch.setattr(runner, "gguf_q4_k_quantize_bf16_q8_1", lambda *a, **kw: adapters.append(("bf16", a)))
    monkeypatch.setattr(runner, "gguf_q4_k_quantize_f32_q8_1", lambda *a, **kw: adapters.append(("f32", a)))
    for dtype, report in zip(("bf16", "f32"), reports):
        symbol = f"gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_{dtype}_out"
        monkeypatch.setattr(runner, symbol, lambda *a, **kw: launches.append(a))
        runner._launch_selected_raw_gguf_moe_linear(weight, 31, 61, 51,
            x_rows=1, rows=1, num_experts=4, in_features=256, out_features=256,
            q8_1_workspace_ptr=41, x_f32_ptr=32 if dtype == "f32" else None,
            prefer_f32_out=dtype == "f32", stream=0, runtime="fake")
        bound, = report.plan_contract.selected_invocations
        assert bound.adapters == (f"gguf_q4_k_quantize_{dtype}_q8_1",)
        assert ("q8_1_workspace_ptr", "q8_1", "read_write") in bound.operands
    assert adapters == [("bf16", (31, 41, 1, 256)), ("f32", (32, 41, 1, 256))]
    assert [a[:4] for a in launches] == [(41, 61, 11, 51)] * 2


def test_two_successful_partner_layouts_bind_with_identical_gate():
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    from test_gguf_ud_admission import _tensor
    model = _synthetic_moe_model_map(expert_type=Q.Q3_K)
    tensors = dict(model.layers[0].tensors)
    tensors["ffn_up_exps"] = _tensor("blk.0.ffn_up_exps.weight", (4, 256, 256), Q.Q5_K)
    model = replace(model, layers=(replace(model.layers[0], tensors=MappingProxyType(tensors)),))
    call = intent("dual", (GATE, UP))
    reports = [preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1,),
               selected_call_intents=(call,), slot_filter=(GATE, UP),
               decode_repack=repack, repack_veto=False) for repack in (False, True)]
    left, right = reports
    assert left.supported and right.supported
    a, = left.plan_contract.selected_invocations
    b, = right.plan_contract.selected_invocations
    assert a.weight_bindings[0] == b.weight_bindings[0]
    assert a.weight_bindings[1] != b.weight_bindings[1]
    assert not covers(left, right)


def test_input_row_broadcast_and_per_lane_contracts_do_not_transfer():
    call = intent("single", (DOWN,), lanes_per_token=2, input_rows_per_token=1)
    left = preflight(calls=(call,), slots=(DOWN,))
    right = preflight(calls=(replace(call, input_rows_per_token=2),), slots=(DOWN,))
    assert left.supported and right.supported
    assert left.plan_contract.resident_plan_records == right.plan_contract.resident_plan_records
    assert not covers(left, right)
    with pytest.raises(ValueError, match="one input row"):
        intent("weighted_down", (DOWN,), lanes_per_token=2, input_rows_per_token=1)


def test_selected_row_domain_and_incomplete_intents_cannot_promote():
    call = intent("dual_silu", (GATE, UP))
    left = preflight(calls=(call,), slots=(GATE, UP))
    right = preflight(calls=(replace(call, lanes_per_token=2),), slots=(GATE, UP))
    assert left.supported and right.supported
    assert not covers(left, right)
    partial = replace(right.plan_contract, selected_invocations=())
    assert not partial.is_complete()
    assert not certificate_covers_artifact(left.certificate(), manifest_fingerprint=right.manifest_fingerprint,
        plan_contract=partial, operations=(C1,), slot_filter=(GATE, UP))
    with pytest.raises(ValueError):
        replace(right, plan_contract=partial).certificate()


@pytest.mark.parametrize("flag,kind,quant", [
    ("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "single", "Q5_K"),
    ("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "dual", "Q4_K"),
    ("HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A", "dual", "Q4_K"),
    ("HIPENGINE_GGUF_T16_SELECTED_DP4A", "dual", "Q4_K"),
])
def test_unrepresented_optional_adapter_refuses_certificate(monkeypatch, flag, kind, quant):
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    model = _synthetic_moe_model_map(expert_type=getattr(Q, quant))
    slots = (GATE, UP) if kind == "dual" else (DOWN,)
    kwargs = dict(backend="hip_gfx1100", operations=(C1,), slot_filter=slots,
                  selected_call_intents=(intent(kind, slots),),
                  decode_repack=flag == "HIPENGINE_GGUF_T16_SELECTED_DP4A",
                  repack_veto=False)
    monkeypatch.delenv(flag, raising=False)
    baseline = preflight_qwen35_gguf_artifact(model, **kwargs)
    assert baseline.supported
    monkeypatch.setenv(flag, "1")
    refused = preflight_qwen35_gguf_artifact(model, **kwargs)
    assert not refused.supported and not refused.plan_contract.is_complete()
    assert any("optional selected DP4A adapter" in item.reason for item in refused.unsupported)
    assert not covers(baseline, refused)
    with pytest.raises(ValueError):
        refused.certificate()


def test_shared_full_model_default_is_the_actual_c1_caller_plan(monkeypatch):
    from types import SimpleNamespace as NS
    import hipengine.runtime.qwen35_gguf_runner as runner
    from hipengine.loading.gguf_selected_contract import default_selected_call_intents
    quants = {GATE: "gguf_iq3_xxs", UP: "gguf_iq3_xxs", DOWN: "gguf_iq3_xxs"}
    plan = default_selected_call_intents(quants, (C1,), lanes_per_token=2)
    weights = [NS(spec=NS(quant_key=quants[slot], slot_path=slot)) for slot in (GATE, UP, DOWN)]
    scratch = NS(**{name: NS(ptr=idx + 1) for idx, name in enumerate((
        "post_norm", "moe_selected_experts", "ffn_intermediate", "moe_routing_weights", "moe_down_out"))})
    calls = []
    def paired(a, b, *args, **kw):
        calls.append(("dual_silu", (a.spec.slot_path, b.spec.slot_path)))
        return True
    def weighted(a, *args, **kw):
        calls.append(("weighted_down", (a.spec.slot_path,)))
        return True
    monkeypatch.setattr(runner, "_launch_selected_raw_gguf_moe_pair_silu", paired)
    monkeypatch.setattr(runner, "_launch_weighted_selected_raw_gguf_moe_linear", weighted)
    monkeypatch.setattr(runner, "_gguf_use_f32_selected_intermediate", lambda *a: False)
    self = NS(weights=NS(config=_synthetic_moe_model_map().config), hidden_size=256)
    assert runner.Qwen35GGUFFullStackRunner._run_post_attention_moe_c1_unfused_selected_ffn(
        self, *weights, scratch, selected_rows=2, stream=0, runtime="fake") == (False, True)
    assert calls == [(item.kind, item.weight_slots) for item in plan]
