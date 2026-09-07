"""CPU regressions for invocation ownership, not numerical certification."""
from dataclasses import replace

import pytest

from test_gguf_ud_admission import _synthetic_model_map
from hipengine.loading.qwen35_gguf_admission import (
    QWEN35_GGUF_OP_AR_DECODE_C1 as C1,
    QWEN35_GGUF_OP_AR_PREFILL as PREFILL,
    QWEN35_GGUF_OP_LM_HEAD_F32_LOGITS as HEAD,
    certificate_covers_artifact,
    preflight_qwen35_gguf_artifact,
)
from hipengine.loading.gguf import GGMLQuantizationType


def test_supplied_f32_raw_head_has_no_bf16_fallback():
    report = preflight_qwen35_gguf_artifact(
        _synthetic_model_map(lm_head_type=GGMLQuantizationType.Q8_0),
        backend="hip_gfx1100", operations=(HEAD,), f32_input_operations=(HEAD,),
    )
    assert not report.supported
    assert not report.plan_contract.is_complete()


def test_successful_backend_contracts_do_not_transfer_when_backend_omitted():
    model = _synthetic_model_map()
    left, right = [preflight_qwen35_gguf_artifact(model, backend=b, operations=(C1,))
                   for b in ("hip_gfx1100", "hip_gfx1151")]
    assert left.supported and right.supported
    assert left.plan_contract.resident_plan_records == right.plan_contract.resident_plan_records
    assert not certificate_covers_artifact(
        left.certificate(), manifest_fingerprint=right.manifest_fingerprint,
        plan_contract=right.plan_contract,
    )


@pytest.mark.parametrize("operation,input_dtype", [(C1, "bf16"), (PREFILL, "f32")])
def test_conv_contract_matches_actual_wrapper_abi(operation, input_dtype):
    report = preflight_qwen35_gguf_artifact(
        _synthetic_model_map(), backend="hip_gfx1100", operations=(operation,),
    )
    conv = next(r for r in report.qualified_records if r.role_class == "conv1d")
    assert (conv.input_dtype, conv.output_dtype) == (input_dtype, "f32")


def test_gdn_boundary_is_mixed_and_ssm_norm_is_composite():
    report = preflight_qwen35_gguf_artifact(
        _synthetic_model_map(), backend="hip_gfx1100", operations=(C1,),
    )
    gdn = next(r for r in report.qualified_records if r.role_class == "gdn_scalar")
    assert (gdn.input_dtype, gdn.output_dtype) == ("f32", "f32")
    assert any(r.role_class == "gdn_norm" and r.kernel_layer == "gdn_recurrent_rmsnorm_gate"
               for r in report.qualified_records)


def test_two_successful_actual_activation_contracts_do_not_transfer():
    model = _synthetic_model_map(attn_qkv_type=GGMLQuantizationType.F32)
    kwargs = dict(backend="hip_gfx1100", operations=(C1,), slot_filter=("layers.0.attn_qkv",))
    bf16 = preflight_qwen35_gguf_artifact(model, **kwargs)
    f32 = preflight_qwen35_gguf_artifact(model, f32_input_operations=(C1,), **kwargs)
    assert bf16.supported and f32.supported
    assert bf16.plan_contract.resident_plan_records == f32.plan_contract.resident_plan_records
    assert bf16.plan_contract.invocations != f32.plan_contract.invocations
    invocation, = f32.plan_contract.invocations
    assert invocation.consumer.operands == (("x_ptr", "f32", "read"), ("raw", "f32", "read"), ("out_ptr", "f32", "write"))
    assert invocation.adapters == ("output:f32_to_bf16",)
    for source, target in ((bf16, f32), (f32, bf16)):
        assert not certificate_covers_artifact(source.certificate(),
            manifest_fingerprint=target.manifest_fingerprint,
            plan_contract=target.plan_contract, slot_filter=kwargs["slot_filter"])
        assert certificate_covers_artifact(source.certificate(),
            manifest_fingerprint=source.manifest_fingerprint,
            plan_contract=source.plan_contract, slot_filter=kwargs["slot_filter"])


@pytest.mark.parametrize("field,value", [
    ("shape", (512, 256)), ("rows_scope", "rows_2_8_native_bf16_ptr"),
    ("adapters", ("input:f32_to_bf16",)),
])
def test_effective_invocation_fields_are_binding(field, value):
    report = preflight_qwen35_gguf_artifact(_synthetic_model_map(), backend="hip_gfx1100", operations=(HEAD,))
    invocation, = report.plan_contract.invocations
    intended = replace(report.plan_contract, invocations=(replace(invocation, **{field: value}),))
    assert intended.is_complete()  # ordinary immutable internal metadata, not a signature
    assert not certificate_covers_artifact(report.certificate(),
        manifest_fingerprint=report.manifest_fingerprint, plan_contract=intended)


def test_equivalent_contracts_and_missing_invocations():
    model = _synthetic_model_map()
    left = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1, PREFILL))
    right = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(PREFILL, C1, C1))
    assert left.supported and right.supported
    assert certificate_covers_artifact(left.certificate(), manifest_fingerprint=right.manifest_fingerprint,
                                      plan_contract=right.plan_contract)
    partial = replace(right.plan_contract, invocations=right.plan_contract.invocations[:-1])
    assert not partial.is_complete()
    assert not certificate_covers_artifact(left.certificate(), manifest_fingerprint=right.manifest_fingerprint,
        plan_contract=partial, operations=(C1,))


def test_aux_wrappers_use_shared_abi_without_loading_hip():
    from types import SimpleNamespace
    from hipengine.loading.qwen35_gguf_consumer_surface import CONV_DECODE, CONV_PREFILL, GDN_SEGMENTS
    from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
        qwen35_linear_attn_conv_decode_indexed_bf16 as decode,
        qwen35_linear_attn_conv_prefill_f32 as prefill,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn.gdn import qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16 as gdn
    calls = []
    def native(*args):
        calls.append(tuple(arg.value for arg in args))
        return 0
    library = SimpleNamespace(**{abi.symbol: native for abi in (CONV_DECODE, CONV_PREFILL, GDN_SEGMENTS)})
    runtime = SimpleNamespace(check=lambda err: pytest.fail(str(err)))
    decode(11, 12, 13, 14, 15, 2, 256, 4, stream=99, library=library, runtime=runtime)
    prefill(21, 22, 23, 24, 3, 256, 4, stream=99, library=library, runtime=runtime)
    gdn(31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        2, 2, .125, 1, 2, 128, 128, stream=99, library=library, runtime=runtime)
    assert calls == [(11, 12, 13, 14, 15, 2, 256, 4, 99),
                     (21, 22, 23, 24, 3, 256, 4, 99),
                     (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 2, 2, .125, 1, 2, 128, 128, 99)]


def test_aux_operand_elements_match_native_c_signatures():
    """Independent native ABI oracle, including mutability and mixed inputs."""
    import re
    from pathlib import Path
    from hipengine.loading.qwen35_gguf_consumer_surface import (
        CONV_DECODE, CONV_PREFILL, CONV_SINGLE, GDN_SEGMENTS, GDN_SINGLE, GDN_PREFILL,
        resolve_gdn_segments_contract,
    )
    for abi, filename in ((CONV_DECODE, "conv"), (CONV_PREFILL, "conv"), (CONV_SINGLE, "conv"),
                          (GDN_SEGMENTS, "gdn"), (GDN_SINGLE, "gdn"), (GDN_PREFILL, "gdn"),
                          (resolve_gdn_segments_contract("fp16"), "gdn")):
        source = Path(f"hipengine/kernels/hip_gfx1100/linear_attn/{filename}.hip").read_text()
        signature = re.search(r'extern "C" int ' + abi.symbol + r'\((.*?)\) \{', source, re.S).group(1)
        native_operands = re.findall(r'(const )?(float|uint16_t|int32_t|int64_t|half_t)\* (\w+)', signature)
        assert len(native_operands) == len(abi.operands)
        for (const, ctype, name), (argument, dtype, access) in zip(native_operands, abi.operands):
            assert argument == name + "_ptr"
            assert dtype == {"float": "f32", "uint16_t": "bf16", "int32_t": "i32", "int64_t": "i64", "half_t": "fp16"}[ctype]
            assert (access == "read") == bool(const)


def test_native_caller_supplies_mixed_gdn_operands_and_executes_cast(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS
    import hipengine.runtime.qwen35_gguf_runner as runner
    from hipengine.loading.qwen35_gguf_consumer_surface import CONV_DECODE, GDN_SEGMENTS
    from hipengine.loading.qwen35_gguf_admission import QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS as NATIVE
    model = _synthetic_model_map(alpha_beta_type=GGMLQuantizationType.BF16)
    report = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(NATIVE,))
    assert report.supported
    dtypes = {}
    def buffer(dtype):
        ptr = (len(dtypes) + 1) * 0x1000000
        dtypes[ptr] = dtype
        return NS(ptr=ptr, nbytes=0x100000)
    buffers = {name: buffer(dtype) for name, dtype in (
        ("norm", "bf16"), ("linear_qkv", "bf16"), ("linear_z", "bf16"),
        ("linear_alpha", "bf16"), ("linear_beta", "bf16"), ("conv_out", "f32"),
        ("recurrent_out", "f32"), ("recurrent_bf16", "bf16"), ("attn_out", "bf16"))}
    import numpy as np
    scratch = NS(**buffers, post_norm=buffer("bf16"), slot_count=2,
                 recurrent_zero=np.zeros(1, dtype=np.float32),
                 layer_conv_states=[buffer("f32")], layer_recurrent_states=[buffer("f32")])
    from tests.test_gguf_execution_authorization import native_resident
    from hipengine.core.dtype import DType
    resident = native_resident(monkeypatch, tmp_path)
    model = NS(config=resident.config)
    layer = resident.layer(0)
    weights = layer.weights
    for weight in resident.weights:
        for allocation in weight.allocations.values():
            dtypes[allocation.tensor.ptr] = {DType.BF16: "bf16", DType.FP32: "f32"}.get(allocation.tensor.dtype, "packed")
    calls = []
    def capture(abi):
        def call(*args, **kwargs):
            assert tuple(dtypes[ptr] for ptr in args[:len(abi.operands)]) == tuple(dtype for _, dtype, _ in abi.operands)
            calls.append(abi.abi)
        return call
    monkeypatch.setattr(runner, "gguf_rmsnorm_bf16_f32_weight", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "dense_gemv_out_bf16", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "launch_gguf_linear_pair", lambda *a, **kw: True)
    monkeypatch.setattr(runner, "qwen35_linear_attn_conv_decode_indexed_bf16", capture(CONV_DECODE))
    monkeypatch.setattr(runner, "_gdn_decode_segments_kernel", lambda fp16: capture(GDN_SEGMENTS))
    def cast(src, dst, count, **kw):
        assert (dtypes[src], dtypes[dst]) == ("f32", "bf16")
        assert count == 2 * model.config.ssm_inner_size
        calls.append("cast")
    monkeypatch.setattr(runner, "f32_to_bf16", cast)
    def linear(weight, x, out, **kw):
        assert weight is weights["ssm_out"]
        assert dtypes[x] == kw.get("activation_dtype", "bf16") == "bf16"
        assert calls[-1] == "cast"
    monkeypatch.setattr(runner, "launch_gguf_linear", linear)
    session = NS(weights=resident, runtime=object(), backend="hip_gfx1100",
                 fp16_recurrent_state=False, hidden_size=256, vocab_size=64, linear_qkv_width=192,
                 ssm_value_dim=32, _cast_library=lambda: object(),
                 _run_post_attention_ffn_rows=lambda *a, **kw: None)
    from tests.test_gguf_ud_admission import _native_entry_session
    scratch.calls = []
    owner_session = _native_entry_session(resident, scratch_owner=scratch)
    owner_session.runner = session
    owner_session.runtime = session.runtime
    owner_session._native_compact_scratch = lambda *a, **kw: scratch
    for name, dtype in (("norm", "bf16"), ("linear_qkv", "bf16"), ("linear_z", "bf16"),
                        ("linear_alpha", "bf16"), ("linear_beta", "bf16"), ("conv_out", "f32"),
                        ("recurrent_out", "f32"), ("recurrent_bf16", "bf16")):
        dtypes[getattr(scratch, name).ptr] = dtype
    dtypes[scratch.layer_conv_states[0].ptr] = "f32"
    dtypes[scratch.layer_recurrent_states[0].ptr] = "f32"
    dtypes[owner_session._native_cu_seqlens_buf.ptr] = "i32"
    dtypes[owner_session._native_state_indices_buf.ptr] = "i64"
    context = owner_session._native_invocation_context(2, scratch)
    result = runner.Qwen35GGUFFullStackRunner._run_linear_attention_decode_rows_native(
        session, 0, owner_session._hidden_a.ptr, owner_session._hidden_b.ptr, scratch, rows=2,
        cu_seqlens_ptr=owner_session._native_cu_seqlens_buf.ptr,
        state_indices_ptr=owner_session._native_state_indices_buf.ptr, invocation_context=context)
    assert result == "indexed_conv_gdn"
    assert calls == ["indexed_conv", "segmented_gdn", "cast"]
    ssm_norm = next(item for item in report.plan_contract.invocations if item.slot.endswith("ssm_norm"))
    assert ssm_norm.consumer == GDN_SEGMENTS
    assert ssm_norm.adapters == ("output:f32_to_bf16",)


def test_prefill_gdn_caller_publishes_bf16_composite_output():
    from types import SimpleNamespace as NS
    import hipengine.runtime.qwen35_gguf_runner as runner
    from hipengine.loading.qwen35_gguf_consumer_surface import GDN_PREFILL
    class Plan(NS):
        def __getattr__(self, name):
            return None
    calls = []
    plan = Plan(has_fused=True, has_chain=False,
                fused_decode_order=lambda *a, **kw: calls.append((a, kw)))
    model = _synthetic_model_map()
    scratch = NS(gdn_effective_mode="fused", conv_out=NS(ptr=101), linear_z=NS(ptr=102),
                 linear_alpha=NS(ptr=103), linear_beta=NS(ptr=104), recurrent_bf16=NS(ptr=109))
    pointers = {"ssm_dt_bias": 105, "ssm_a": 106, "ssm_norm": 107}
    layer = NS(weight=lambda slot: NS(allocation=lambda: NS(tensor=NS(ptr=pointers[slot]))))
    self = NS(_gdn_prefill_plan=lambda: plan, backend="hip_gfx1100", ssm_value_dim=32,
              fp16_recurrent_state=False)
    runner.Qwen35GGUFFullStackRunner._run_gdn_prefill(self, layer=layer, scratch=scratch,
        cfg=model.config, rows=3, recurrent_state=NS(ptr=108), stream=7, runtime="fake")
    assert len(calls) == 1
    assert calls[0][0][:9] == tuple(range(101, 110))
    report = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(PREFILL,))
    assert report.supported
    norm = next(item for item in report.plan_contract.invocations if item.slot.endswith("ssm_norm"))
    assert norm.consumer == GDN_PREFILL
    assert norm.consumer.operands[-1] == ("out_ptr", "bf16", "write")


def test_q8_fp16_legacy_selector_does_not_claim_bf16_operand():
    from hipengine.loading.qwen35_gguf_consumer_surface import gguf_linear_dispatch_row, linear_consumer_contract
    row = gguf_linear_dispatch_row("gguf_q8_0_t16_v1", "bf16", "fp16")
    # Preserve the historical selector key, but the C instantiation is
    # launch_single<half_t, half_t>, not BF16 input or an executed cast.
    assert row.activation == "bf16"
    assert linear_consumer_contract(row).operands[0][1] == "fp16"


def test_recurrent_state_intent_binds_actual_production_consumer():
    import hipengine.runtime.qwen35_gguf_runner as runner
    from hipengine.loading.qwen35_gguf_admission import QWEN35_GGUF_OP_AR_DECODE_NATIVE_ROWS as NATIVE
    from hipengine.loading.qwen35_gguf_consumer_surface import resolve_gdn_segments_contract
    model = _synthetic_model_map(alpha_beta_type=GGMLQuantizationType.BF16)
    reports = [preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(NATIVE,),
                                          recurrent_state_dtype=dtype) for dtype in ("f32", "fp16")]
    left, right = reports
    assert left.supported and right.supported
    assert left.plan_contract.resident_plan_records == right.plan_contract.resident_plan_records
    for state, report in zip(("f32", "fp16"), reports):
        consumer = resolve_gdn_segments_contract(state)
        assert runner._gdn_decode_segments_kernel(state == "fp16").__name__ == consumer.symbol.removeprefix("hipengine_")
        assert any(item.consumer == consumer for item in report.plan_contract.invocations)
        assert ("recurrent_state_ptr", state, "read_write") in consumer.operands
    assert not certificate_covers_artifact(left.certificate(), manifest_fingerprint=right.manifest_fingerprint,
                                          plan_contract=right.plan_contract)


def test_real_f32_alpha_beta_caller_executes_recorded_output_adapter(monkeypatch):
    from types import SimpleNamespace as NS
    import hipengine.runtime.qwen35_gguf_runner as runner
    from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_weight_spec
    model = _synthetic_model_map()
    slots = ("layers.0.ssm_alpha", "layers.0.ssm_beta")
    report = preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100", operations=(C1,),
        slot_filter=slots, f32_input_operations=(C1,), contract_f32_linear=False)
    assert report.supported
    weights = {slot: NS(spec=plan_qwen35_gguf_weight_spec("layers.0." + slot, model.layers[0].tensors[slot]),
                       backend="hip_gfx1100") for slot in ("ssm_alpha", "ssm_beta")}
    scratch = NS(linear_alpha=NS(ptr=11), linear_beta=NS(ptr=12), linear_alpha_f32=NS(ptr=21), linear_beta_f32=NS(ptr=22))
    launches, casts = [], []
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS", "1")
    monkeypatch.setattr(runner, "launch_gguf_linear", lambda *a, **kw: launches.append((a, kw)))
    monkeypatch.setattr(runner, "f32_to_bf16", lambda *a, **kw: casts.append(a))
    self = NS(weights=NS(config=model.config), hidden_size=256, _cast_library=lambda: "fake")
    result = runner.Qwen35GGUFFullStackRunner._run_linear_attention_alpha_beta_rows(
        self, NS(weight=lambda slot: weights[slot]), 31, 32, scratch, rows=1, stream=0, runtime="fake")
    assert result == "f32_singletons_f32_out"
    assert [(a[1], a[2], kw["activation_dtype"], kw["output_dtype"]) for a, kw in launches] == [(32, 21, "f32", "f32"), (32, 22, "f32", "f32")]
    assert casts == [(21, 11, model.config.ssm_time_step_rank), (22, 12, model.config.ssm_time_step_rank)]
    assert all(item.adapters == ("output:f32_to_bf16",) for item in report.plan_contract.invocations)


def test_gdn_handoff_uses_actual_f32_or_executed_bf16_adapter():
    from types import SimpleNamespace as NS
    from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_weight_spec
    import hipengine.runtime.qwen35_gguf_runner as runner
    model = _synthetic_model_map(gate_type=GGMLQuantizationType.Q8_0)
    slot = "layers.0.ssm_out"
    kwargs = dict(backend="hip_gfx1100", operations=(C1,), slot_filter=(slot,), decode_repack=True)
    natural = preflight_qwen35_gguf_artifact(model, **kwargs)
    converted = preflight_qwen35_gguf_artifact(model, gdn_force_bf16=True, **kwargs)
    declared = preflight_qwen35_gguf_artifact(model, f32_input_operations=(C1,), **kwargs)
    assert natural.supported and converted.supported and declared.supported
    first, = natural.plan_contract.invocations
    second, = converted.plan_contract.invocations
    assert first.consumer.operands[0][1] == "f32" and first.adapters == ()
    assert second.consumer.operands[0][1] == "bf16" and second.adapters == ("input:f32_to_bf16",)
    assert natural.plan_contract.resident_plan_records == converted.plan_contract.resident_plan_records
    assert not certificate_covers_artifact(natural.certificate(), manifest_fingerprint=converted.manifest_fingerprint,
        plan_contract=converted.plan_contract, slot_filter=(slot,))
    assert certificate_covers_artifact(natural.certificate(), manifest_fingerprint=declared.manifest_fingerprint,
        plan_contract=declared.plan_contract, slot_filter=(slot,))
    spec = plan_qwen35_gguf_weight_spec(slot, model.layers[0].tensors["ssm_out"], decode_repack=True)
    weight = NS(spec=spec, backend="hip_gfx1100")
    for cast in (None, runner.f32_to_bf16):
        self = NS(backend="hip_gfx1100", _gdn_decode_output_cast_fn=lambda cast=cast: cast)
        assert runner.Qwen35GGUFFullStackRunner._gdn_decode_output_cast_for_weight(self, weight) is cast
    refused = preflight_qwen35_gguf_artifact(model, gdn_force_bf16=True, f32_input_operations=(C1,), **kwargs)
    assert not refused.supported and not refused.plan_contract.is_complete()


def test_prefill_fp16_state_cannot_borrow_baseline_f32_certificate():
    model = _synthetic_model_map()
    kwargs = dict(backend="hip_gfx1100", operations=(PREFILL,))
    baseline = preflight_qwen35_gguf_artifact(model, **kwargs)
    refused = preflight_qwen35_gguf_artifact(model, recurrent_state_dtype="fp16", **kwargs)
    assert baseline.supported
    assert not refused.supported and not refused.plan_contract.is_complete()
    assert any("baseline prefill GDN requires F32 state" in item.reason for item in refused.unsupported)
    assert not certificate_covers_artifact(baseline.certificate(),
        manifest_fingerprint=refused.manifest_fingerprint, plan_contract=refused.plan_contract)
    with pytest.raises(ValueError):
        refused.certificate()


def test_all_twenty_rows_resolve_and_marshal_production_operands():
    from types import SimpleNamespace
    import hipengine.runtime.gguf_linear as runtime
    from hipengine.loading.qwen35_gguf_consumer_surface import GGUF_LINEAR_DISPATCH_SURFACE, linear_consumer_contract
    addresses = {"raw": 101, "tiles": 102, "qweight": 103, "scales": 104, "mins": 105}
    assert len(GGUF_LINEAR_DISPATCH_SURFACE) == 20
    for row in GGUF_LINEAR_DISPATCH_SURFACE:
        weight = SimpleNamespace(backend="hip_gfx1151", spec=SimpleNamespace(layout=row.layout, quant_key="gguf_q8_0"),
            allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=addresses[name])))
        for rows in (1, 2, 8, 16):
            dispatch = runtime.resolve_gguf_linear_dispatch(weight, activation_dtype=row.activation,
                                                          output_dtype=row.output, rows=rows)
            assert dispatch.key.backend == "hip_gfx1151"
            assert dispatch.key.variant == row.variant_for_rows(rows)
            calls = []
            runtime._LAUNCH_ABI[dispatch.abi](lambda *a, **kw: calls.append((a, kw)), weight,
                                            201, 202, rows, 256, 128, {"stream": 7})
            expected_weights = (103, 104, 105) if dispatch.abi == "pack8" else ((102,) if dispatch.abi == "t16" else (101,))
            assert calls == [((201, *expected_weights, 202, rows, 256, 128), {"stream": 7})]
            operands = linear_consumer_contract(row).operands
            assert tuple(addresses[name] for name, _, _ in operands[1:-1]) == expected_weights
    with pytest.raises(ValueError, match="rows must be positive"):
        runtime.resolve_gguf_linear_dispatch(weight, activation_dtype=row.activation, output_dtype=row.output, rows=0)
