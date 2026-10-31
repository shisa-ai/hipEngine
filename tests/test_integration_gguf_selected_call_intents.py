"""Explicit selected-call scope and ordered partner ownership (CPU only)."""
from dataclasses import replace
from types import MappingProxyType

import pytest

from test_live_gguf_ud_admission import _synthetic_moe_model_map
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
    from test_live_gguf_ud_admission import _tensor
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


OPTIONAL_ROUTES = (
    # selector, kind, expert type, decode_repack, slots, launched kernel
    ("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "single", "Q5_K", False, (DOWN,),
     "gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out"),
    ("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "single", "Q6_K", False, (DOWN,),
     "gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out"),
    ("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "dual", "Q4_K", False, (GATE, UP),
     "gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out"),
    ("HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A", "dual", "Q4_K", False, (GATE, UP),
     "gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out"),
    ("HIPENGINE_GGUF_T16_SELECTED_DP4A", "dual", "Q4_K", True, (GATE, UP),
     "gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out"),
)


def optional_intent(kind, slots, **kwargs):
    from hipengine.loading.gguf_selected_contract import (
        SELECTED_Q8_1_DP4A, SELECTED_Q8_1_WORKSPACE_OWNER)
    return intent(kind, slots, optional_adapter=SELECTED_Q8_1_DP4A,
                  workspace_owner=SELECTED_Q8_1_WORKSPACE_OWNER, **kwargs)


def optional_preflight(model, *, flag, kind, repack, slots):
    return preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1,),
        decode_repack=repack, repack_veto=False, slot_filter=slots,
        selected_call_intents=(optional_intent(kind, slots),))


@pytest.mark.parametrize("flag,kind,quant,repack,slots,kernel", OPTIONAL_ROUTES)
def test_optional_q8_1_dp4a_adapter_binds_kernel_operands_and_workspace(
        monkeypatch, flag, kind, quant, repack, slots, kernel):
    """An engaged optional adapter binds the conversion it executes, the
    Q8_1 operand it consumes, and the plan-owned workspace that owns it."""
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    from hipengine.loading.gguf_selected_contract import (
        SELECTED_WORKSPACE_LIFETIME, selected_input_adapter)
    model = _synthetic_moe_model_map(expert_type=getattr(Q, quant))
    monkeypatch.setenv(flag, "1")
    report = optional_preflight(model, flag=flag, kind=kind, repack=repack, slots=slots)
    assert report.supported, report.render_refusals()
    bound, = report.plan_contract.selected_invocations
    assert bound.adapter_kernels == (kernel,)
    assert bound.adapters == (selected_input_adapter("bf16"),)
    assert bound.operands[-1] == ("q8_1_workspace_ptr", "q8_1", "read_write")
    assert bound.workspace.owner == "moe_q8_1"
    assert bound.workspace.lifetime == SELECTED_WORKSPACE_LIFETIME
    assert bound.workspace.rows_per_token == bound.intent.input_rows_per_token
    assert bound.workspace.in_features == bound.shape[2]
    assert [slot for slot, _ in bound.weight_bindings] == list(slots)
    assert bound.row_limits == (1, 1)
    assert bound.entry.endswith("_launch_selected_raw_gguf_moe_dual" if kind == "dual"
                                else "_launch_selected_raw_gguf_moe_linear")


@pytest.mark.parametrize("flag,kind,quant,repack,slots,kernel", OPTIONAL_ROUTES)
def test_optional_adapter_engagement_is_binding_not_declared(
        monkeypatch, flag, kind, quant, repack, slots, kernel):
    """The same declared intent binds two different arithmetic contracts: the
    selector decides whether the conversion actually runs."""
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    model = _synthetic_moe_model_map(expert_type=getattr(Q, quant))
    monkeypatch.setenv(flag, "1")
    engaged = optional_preflight(model, flag=flag, kind=kind, repack=repack, slots=slots)
    monkeypatch.delenv(flag)
    plain = optional_preflight(model, flag=flag, kind=kind, repack=repack, slots=slots)
    assert engaged.supported and plain.supported
    adapted, unconverted = (report.plan_contract.selected_invocations[0]
                            for report in (engaged, plain))
    assert adapted.intent == unconverted.intent  # one shared declaration
    assert adapted.adapter_kernels == (kernel,) and unconverted.adapter_kernels == ()
    assert adapted.workspace is not None and unconverted.workspace is None
    assert adapted.adapters and not unconverted.adapters
    assert adapted.operands != unconverted.operands
    assert engaged.plan_contract.resident_plan_records == plain.plan_contract.resident_plan_records
    assert not covers(engaged, plain) and not covers(plain, engaged)


@pytest.mark.parametrize("flag,kind,quant,repack,slots,kernel", OPTIONAL_ROUTES)
def test_optional_adapter_binding_names_the_kernel_the_caller_launches(
        monkeypatch, flag, kind, quant, repack, slots, kernel):
    """The bound record must name the symbol, operand order and per-token
    workspace geometry the actual selected launcher executes."""
    from types import SimpleNamespace as NS
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_weight_spec
    from hipengine.loading.gguf_selected_contract import selected_allocation
    import hipengine.runtime.qwen35_gguf_runner as runner
    model = _synthetic_moe_model_map(expert_type=getattr(Q, quant))
    monkeypatch.setenv(flag, "1")
    report = optional_preflight(model, flag=flag, kind=kind, repack=repack, slots=slots)
    bound, = report.plan_contract.selected_invocations

    def weight(slot, ptr):
        spec = plan_qwen35_gguf_weight_spec(slot, model.layers[0].tensors[slot.rsplit(".", 1)[1]],
                                           decode_repack=repack)
        return NS(spec=spec, backend="hip_gfx1100",
                  allocation=lambda name: NS(tensor=NS(ptr=ptr)))

    from hipengine.loading.gguf_selected_contract import SELECTED_Q8_1_WORKSPACE_OWNER
    weights = [weight(slot, 11 + index) for index, slot in enumerate(slots)]
    pointers = {w.spec.slot_path: w.allocation(selected_allocation(w.spec.quant_key)).tensor.ptr
                for w in weights}
    assert [pointers[slot] for slot in slots] == [11 + index for index in range(len(slots))]
    launched, converted = [], []
    monkeypatch.setattr(runner, "_resolve_exact_selected_moe_kernel", lambda *a: None)
    monkeypatch.setattr(runner, "gguf_q4_k_quantize_bf16_q8_1",
                        lambda *a, **kw: converted.append(a))
    assert getattr(runner, kernel)  # the bound symbol is the launcher's own dispatch target
    monkeypatch.setattr(runner, kernel, lambda *a, **kw: launched.append(a))
    geometry = dict(x_rows=1, rows=1, num_experts=4, in_features=256, out_features=256)
    if kind == "dual":
        runner._launch_selected_raw_gguf_moe_dual(
            *weights, 21, 22, 23, 24, q8_1_workspace_ptr=41, stream=0, runtime="fake", **geometry)
        assert len(launched) == 1, "the pair must not fall back to two singles"
    else:
        runner._launch_selected_raw_gguf_moe_linear(
            weights[0], 21, 22, 23, q8_1_workspace_ptr=41, stream=0, runtime="fake", **geometry)
    # The dp4a kernel consumes the converted activation, so the bound workspace
    # leads the call and every other declared operand follows in record order.
    actual = {"x_ptr": 21, "x_f32_ptr": 21, "selected_ptr": 22,
              "q8_1_workspace_ptr": 41, "out_ptr": 23, "out_a_ptr": 23, "out_b_ptr": 24,
              "weight_ptr": pointers[slots[0]], "weight_a_ptr": pointers[slots[0]]}
    if len(slots) == 2:
        actual["weight_b_ptr"] = pointers[slots[1]]
    declared = [name for name, _, _ in bound.operands]
    assert declared[-1] == "q8_1_workspace_ptr"
    expected = (actual["q8_1_workspace_ptr"],) + tuple(
        actual[name] for name in declared
        if name not in {"x_ptr", "x_f32_ptr", "q8_1_workspace_ptr"})
    assert launched[0][:len(expected)] == expected
    assert bound.workspace.owner == SELECTED_Q8_1_WORKSPACE_OWNER
    # One activation row per token, the resident's K, into the bound workspace.
    assert converted == [(21, 41, bound.workspace.rows_per_token, bound.workspace.in_features)]


def test_optional_adapter_f32_input_launches_the_f32_converter(monkeypatch):
    """The F32-input leg converts from the declared x_f32 operand, not from x."""
    from types import SimpleNamespace as NS
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_weight_spec
    from hipengine.loading.gguf_selected_contract import selected_allocation
    import hipengine.runtime.qwen35_gguf_runner as runner
    model = _synthetic_moe_model_map(expert_type=Q.Q6_K)
    monkeypatch.setenv("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "1")
    report = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1,),
        decode_repack=False, slot_filter=(DOWN,), selected_call_intents=(
            optional_intent("single", (DOWN,), input_dtype="f32"),))
    assert report.supported
    bound, = report.plan_contract.selected_invocations
    spec = plan_qwen35_gguf_weight_spec(DOWN, model.layers[0].tensors["ffn_down_exps"],
                                       decode_repack=False)
    weight = NS(spec=spec, backend="hip_gfx1100",
                allocation=lambda name: NS(tensor=NS(ptr=11)))
    assert weight.allocation(selected_allocation(spec.quant_key)).tensor.ptr == 11
    launched, converted = [], []
    monkeypatch.setattr(runner, "_resolve_exact_selected_moe_kernel", lambda *a: None)
    monkeypatch.setattr(runner, "gguf_q4_k_quantize_f32_q8_1",
                        lambda *a, **kw: converted.append(a))
    monkeypatch.setattr(runner, "gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out",
                        lambda *a, **kw: launched.append(a))
    runner._launch_selected_raw_gguf_moe_linear(
        weight, 31, 61, 51, x_rows=1, rows=1, num_experts=4, in_features=256,
        out_features=256, q8_1_workspace_ptr=41, x_f32_ptr=32, stream=0, runtime="fake")
    assert converted == [(32, 41, bound.workspace.rows_per_token, bound.workspace.in_features)]
    assert launched[0][:4] == (41, 61, 11, 51)
    assert bound.operands[0] == ("x_f32_ptr", "f32", "read")
    assert bound.adapters == ("gguf_q4_k_quantize_f32_q8_1",)


def test_optional_adapter_intent_cannot_be_partial_or_invented(monkeypatch):
    from hipengine.loading.gguf_selected_contract import (
        SELECTED_Q8_1_DP4A, SELECTED_Q8_1_WORKSPACE_OWNER, SelectedCallIntent)
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    with pytest.raises(ValueError, match="requires an optional adapter"):
        SelectedCallIntent(C1, "single", (DOWN,), workspace_owner=SELECTED_Q8_1_WORKSPACE_OWNER)
    for alias, owner in ((DOWN, "weight_slot"), ("post_norm", "input_owner"),
                         ("moe_down_out", "output_owner")):
        declared = dict(input_owner="post_norm", output_owner="moe_down_out")
        if owner == "weight_slot":
            declared = {}
        with pytest.raises(ValueError, match="must not alias"):
            SelectedCallIntent(C1, "single", (DOWN,), optional_adapter=SELECTED_Q8_1_DP4A,
                               workspace_owner=alias, **declared)
    with pytest.raises(ValueError, match="unknown optional selected adapter"):
        SelectedCallIntent(C1, "single", (DOWN,), optional_adapter="dp4a_unqualified",
                           workspace_owner=SELECTED_Q8_1_WORKSPACE_OWNER)
    # The runtime consumes exactly the plan-owned slot: a certificate may not
    # name a different owner than the one the adapter writes.
    for unexecuted in ("selected_q8_1", "moe_q8_1_f32", "workspace"):
        with pytest.raises(ValueError, match="is not an executed Q8_1 workspace"):
            SelectedCallIntent(C1, "single", (DOWN,), optional_adapter=SELECTED_Q8_1_DP4A,
                               workspace_owner=unexecuted)
    # The engaged conversion needs a named workspace before it can bind.
    model = _synthetic_moe_model_map(expert_type=Q.Q5_K)
    monkeypatch.setenv("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "1")
    refused = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1,),
        decode_repack=False, slot_filter=(DOWN,), selected_call_intents=(
            intent("single", (DOWN,), optional_adapter=SELECTED_Q8_1_DP4A),))
    assert not refused.supported and not refused.plan_contract.is_complete()
    assert any("named q8_1 workspace owner" in item.reason for item in refused.unsupported)
    # A declared route this call cannot take is refused, never ignored.
    no_route = preflight_qwen35_gguf_artifact(_synthetic_moe_model_map(), backend="hip_gfx1100",
        operations=(C1,), slot_filter=(DOWN,), selected_call_intents=(
            intent("weighted_down", (DOWN,), optional_adapter=SELECTED_Q8_1_DP4A,
                   workspace_owner=SELECTED_Q8_1_WORKSPACE_OWNER),))
    assert not no_route.supported
    assert any("no route for this selected call" in item.reason for item in no_route.unsupported)


def test_optional_dual_adapter_keeps_ordered_partners_and_requires_one_layout(monkeypatch):
    from types import MappingProxyType
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    from test_live_gguf_ud_admission import _tensor
    model = _synthetic_moe_model_map(expert_type=Q.Q4_K)
    monkeypatch.setenv("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "1")
    forward = optional_preflight(model, flag="HIPENGINE_GGUF_RAW_SELECTED_DP4A",
                                 kind="dual", repack=False, slots=(GATE, UP))
    reversed_pair = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1,),
        decode_repack=False, repack_veto=False, slot_filter=(GATE, UP),
        selected_call_intents=(optional_intent("dual", (UP, GATE)),))
    assert forward.supported and reversed_pair.supported
    assert not covers(forward, reversed_pair)
    assert [slot for slot, _ in reversed_pair.plan_contract.selected_invocations[0].weight_bindings] \
        == [UP, GATE]
    # The dp4a pair reads one resident layout; a mixed pair has no route.
    tensors = dict(model.layers[0].tensors)
    tensors["ffn_up_exps"] = _tensor("blk.0.ffn_up_exps.weight", (4, 256, 256), Q.Q5_K)
    mixed = replace(model, layers=(replace(model.layers[0], tensors=MappingProxyType(tensors)),))
    refused = preflight_qwen35_gguf_artifact(mixed, backend="hip_gfx1100", operations=(C1,),
        decode_repack=False, repack_veto=False, slot_filter=(GATE, UP),
        selected_call_intents=(optional_intent("dual", (GATE, UP)),))
    assert not refused.supported
    assert any("no route for this selected call" in item.reason for item in refused.unsupported)


def test_optional_adapter_keeps_f32_input_and_refuses_f32_output(monkeypatch):
    from hipengine.quant.gguf import GGMLQuantizationType as Q
    from hipengine.loading.gguf_selected_contract import selected_input_adapter
    model = _synthetic_moe_model_map(expert_type=Q.Q6_K)
    monkeypatch.setenv("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "1")
    f32 = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1,),
        decode_repack=False, slot_filter=(DOWN,), selected_call_intents=(
            optional_intent("single", (DOWN,), input_dtype="f32"),))
    assert f32.supported
    bound, = f32.plan_contract.selected_invocations
    assert bound.adapters == (selected_input_adapter("f32"),)
    assert bound.operands[0] == ("x_f32_ptr", "f32", "read")
    refused = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1,),
        decode_repack=False, slot_filter=(DOWN,), selected_call_intents=(
            optional_intent("single", (DOWN,), output_dtype="f32"),))
    assert not refused.supported
    assert any("no F32 output" in item.reason for item in refused.unsupported)


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
    self = NS(weights=NS(config=_synthetic_moe_model_map().config), hidden_size=256,
              backend="hip_gfx1100")
    assert runner.Qwen35GGUFFullStackRunner._run_post_attention_moe_c1_unfused_selected_ffn(
        self, *weights, scratch, selected_rows=2, stream=0, runtime="fake") == (False, True)
    assert calls == [(item.kind, item.weight_slots) for item in plan]
