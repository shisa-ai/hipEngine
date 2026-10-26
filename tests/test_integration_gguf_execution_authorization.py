"""Device-free F1 execution closure/entry tests, not numerical certification."""
import pytest

from hipengine.loading.gguf import GGUFReader, MissingGGUFTensorError
from hipengine.loading import qwen35_gguf_materialize as loader
from hipengine.loading.qwen35_gguf_admission import Qwen35GGUFAdmissionError
from hipengine.quant.gguf import GGMLQuantizationType as Q
from tests._qwen35_gguf_fixture import (
    default_fixture_tensors,
    fixture_metadata,
    linear_attention_layer_slots,
    write_qwen35_gguf,
)

NATIVE = "ar_decode_native_rows"


def native_resident(monkeypatch, tmp_path):
    from tests.test_live_gguf_ud_admission import _materialize_fixture_on_cpu
    return _materialize_fixture_on_cpu(tiny_native(tmp_path), monkeypatch,
                                       decode_repack=False, requested_operations=(NATIVE,))


def tiny_native(tmp_path, *, embedding=Q.Q8_0, head=Q.Q8_0):
    tensors = default_fixture_tensors(1, alpha_beta_type=Q.BF16)
    tensors[0] = ("token_embd.weight", (64, 256), embedding)
    tensors.append(("output.weight", (64, 256), head))
    path = tmp_path / "native.gguf"
    write_qwen35_gguf(path, tensors, fixture_metadata(1))
    return path


def native_fixture(tmp_path, *, rank=2, alpha_beta=Q.BF16, omit=()):
    """One tiny native-ready GGUF whose SSM timestep rank is overridable.

    The rank is metadata-bound as well as shape-bound so a Q8_0 alpha/beta
    pair can satisfy the T16 repack prerequisite (rank divisible by 16) and
    plan the sole-T16 resident that has no ``raw`` allocation at all.
    """
    tensors = [
        ("token_embd.weight", (64, 256), Q.Q8_0),
        ("output_norm.weight", (256,), Q.F32),
    ]
    for name, shape, qtype in linear_attention_layer_slots(0, alpha_beta_type=alpha_beta):
        if name.endswith((".ssm_a", ".ssm_dt.bias")):
            shape = (rank,)
        elif name.endswith((".ssm_alpha.weight", ".ssm_beta.weight")):
            shape = (rank, 256)
        if name not in omit:
            tensors.append((name, shape, qtype))
    tensors.append(("output.weight", (64, 256), Q.Q8_0))
    metadata = [
        (key, kind, rank if key == "qwen35.ssm.time_step_rank" else value)
        for key, kind, value in fixture_metadata(1)
    ]
    path = tmp_path / f"native-rank{rank}-{alpha_beta.name}.gguf"
    write_qwen35_gguf(path, tensors, metadata)
    return path


@pytest.mark.parametrize("embedding,head,slot", [
    (Q.BF16, Q.Q8_0, "root.token_embedding"),
    (Q.Q8_0, Q.F32, "root.lm_head"),
])
def test_native_only_real_loader_refuses_before_payload_or_allocation(monkeypatch, tmp_path, embedding, head, slot):
    path = tiny_native(tmp_path, embedding=embedding, head=head)
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported native request reached payload/allocation")
    monkeypatch.setattr(GGUFReader, "tensor_data", forbidden)
    monkeypatch.setattr(loader, "malloc", forbidden)
    with pytest.raises(Qwen35GGUFAdmissionError, match=slot):
        loader.materialize_qwen35_gguf_weights(path, decode_repack=False, requested_operations=(NATIVE,))


def test_dense_embedding_multirow_refuses_before_leaf(monkeypatch):
    from types import SimpleNamespace
    from hipengine.runtime import gguf_embedding as embedding
    weight = SimpleNamespace(spec=SimpleNamespace(layout=loader.LAYOUT_DENSE_BF16, quant_key="bf16"), backend="hip_gfx1100")
    monkeypatch.setattr(embedding, "_ensure_embedding_kernel_registered", lambda *_: pytest.fail("resolution before row refusal"))
    with pytest.raises(ValueError, match="rows"):
        embedding.launch_gguf_embedding(weight, 10, 20, rows=2, hidden_size=256, vocab_size=64)


def test_real_loader_native_closure_and_positive_rows(monkeypatch, tmp_path):
    from hipengine.loading.qwen35_gguf_execution import authorize_native_execution, execution_operations
    resident = native_resident(monkeypatch, tmp_path)
    contract = resident.admission_certificate.plan_contract
    assert set(contract.operations) == set(execution_operations(("native_rows",)))
    native_slots = {i.slot for i in contract.invocations if i.operation == NATIVE}
    assert native_slots == set(contract.required_plan_slots)
    head = next(i for i in contract.invocations if i.operation == NATIVE and i.slot == "root.lm_head")
    assert head.consumer.operands[0][1] == "bf16"  # actual scratch.norm, not invented F32
    for rows in (2, 4, 8):
        assert authorize_native_execution(resident, backend="hip_gfx1100", rows=rows)
    with pytest.raises(ValueError, match="rows"):
        authorize_native_execution(resident, backend="hip_gfx1100", rows=9)


@pytest.mark.parametrize("alpha_beta,layout", [
    (Q.Q8_0, "raw_gguf"),
    (Q.F32, "dense_f32"),
])
def test_native_alpha_beta_owner_refusal_is_the_execution_contract_report(
        monkeypatch, tmp_path, alpha_beta, layout):
    """The native multirow route passes alpha/beta ``allocation('raw')`` to
    ``dense_gemv_out_bf16`` (``uint16_t*`` weight ABI), so only a dense-BF16
    resident is a valid BF16-pointer owner.  Raw Q8_0 bytes and an
    uncontracted dense-F32 lane must be refused by the pre-allocation
    execution-contract report, which names the slot, operation, and layout."""
    tensors = default_fixture_tensors(1, alpha_beta_type=alpha_beta)
    tensors[0] = ("token_embd.weight", (64, 256), Q.Q8_0)
    tensors.append(("output.weight", (64, 256), Q.Q8_0))
    path = tmp_path / f"native-ab-{layout}.gguf"
    write_qwen35_gguf(path, tensors, fixture_metadata(1))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid native alpha/beta owner reached payload/allocation")
    monkeypatch.setattr(GGUFReader, "tensor_data", forbidden)
    monkeypatch.setattr(loader, "malloc", forbidden)
    with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
        loader.materialize_qwen35_gguf_weights(
            path, decode_repack=False, requested_operations=(NATIVE,))
    message = str(excinfo.value)
    for slot in ("layers.0.ssm_alpha", "layers.0.ssm_beta"):
        assert f"{slot} op={NATIVE}" in message
        assert f"role=recurrent_alpha_beta type={alpha_beta.name} layout={layout}" in message
    assert "dense BF16" in message


def test_sole_t16_alpha_beta_owner_is_refused_by_the_execution_contract(monkeypatch, tmp_path):
    """A repacked Q8_0 file plans alpha/beta as sole-T16 residents with only a
    ``tiles`` allocation, so the BF16-pointer owner cannot be resolved at all.
    The execution-contract report, not a private resident diagnostic, must
    refuse it before any payload read or allocation."""
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_admission import preflight_qwen35_gguf_artifact
    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_GGUF_Q8_0_T16, plan_qwen35_gguf_weight_spec,
    )

    path = native_fixture(tmp_path, rank=16, alpha_beta=Q.Q8_0)
    model_map = build_qwen35_gguf_tensor_map(GGUFReader(path).info)
    for slot in ("ssm_alpha", "ssm_beta"):
        spec = plan_qwen35_gguf_weight_spec(
            f"layers.0.{slot}", model_map.layers[0].tensors[slot],
            contract_f32_linear=False, decode_repack=True,
        )
        assert spec.layout == LAYOUT_GGUF_Q8_0_T16
        assert spec.allocation_names == ("tiles",), "sole T16: no 'raw' owner exists"

    # Control: the identical artifact qualifies those slots for the eager
    # row/prefill routes, so the refusal below is the native BF16-pointer
    # owner contract and not a general sole-T16 defect.
    for operation in ("ar_decode_c1", "ar_decode_rows", "ar_prefill"):
        report = preflight_qwen35_gguf_artifact(
            model_map, backend="hip_gfx1100", operations=(operation,), decode_repack=True,
        )
        assert not [u for u in report.unsupported if u.role_class == "recurrent_alpha_beta"]

    def forbidden(*args, **kwargs):
        pytest.fail("sole-T16 native alpha/beta owner reached payload/allocation")
    monkeypatch.setattr(GGUFReader, "tensor_data", forbidden)
    monkeypatch.setattr(loader, "malloc", forbidden)
    with pytest.raises(Qwen35GGUFAdmissionError) as excinfo:
        loader.materialize_qwen35_gguf_weights(
            path, decode_repack=True, requested_operations=(NATIVE,))
    message = str(excinfo.value)
    for slot in ("layers.0.ssm_alpha", "layers.0.ssm_beta"):
        assert f"{slot} op={NATIVE}" in message
        assert f"layout={LAYOUT_GGUF_Q8_0_T16}" in message
    assert "dense BF16" in message


def test_missing_alpha_beta_owner_is_refused_before_payload(monkeypatch, tmp_path):
    """An artifact with no ``ssm_alpha`` record has no alpha/beta owner to bind;
    the artifact-level contract must name the missing tensor before any
    payload read or allocation, for native and eager routes alike."""
    path = native_fixture(tmp_path, omit=("blk.0.ssm_alpha.weight",))

    def forbidden(*args, **kwargs):
        pytest.fail("missing alpha/beta owner reached payload/allocation")
    monkeypatch.setattr(GGUFReader, "tensor_data", forbidden)
    monkeypatch.setattr(loader, "malloc", forbidden)
    with pytest.raises(MissingGGUFTensorError, match=r"blk\.0\.ssm_alpha\.weight"):
        loader.materialize_qwen35_gguf_weights(
            path, decode_repack=False, requested_operations=(NATIVE,))


def test_non_bf16_raw_alpha_beta_owner_cannot_authorize_native_execution(monkeypatch, tmp_path):
    """Physical allocation dtype is part of the published resident ownership:
    a real materialized dense-BF16 resident whose alpha/beta ``raw`` tensor is
    FP16 (same slot, names, pointer, shape) cannot authorize the native route,
    while the untouched resident still does."""
    from dataclasses import replace
    from types import MappingProxyType

    from hipengine.core.dtype import DType
    from hipengine.loading.qwen35_gguf_execution import authorize_native_execution

    resident = native_resident(monkeypatch, tmp_path)
    assert authorize_native_execution(resident, backend="hip_gfx1100", rows=2)
    alpha = resident.layers[0].weight("ssm_alpha")
    raw = alpha.allocation("raw")
    assert raw.tensor.dtype == DType.BF16
    tampered = replace(alpha, allocations=MappingProxyType({
        **alpha.allocations,
        "raw": replace(raw, tensor=replace(raw.tensor, dtype=DType.FP16)),
    }))
    layer = replace(resident.layers[0], weights=MappingProxyType({
        **resident.layers[0].weights, "ssm_alpha": tampered,
    }))
    with pytest.raises(Qwen35GGUFAdmissionError, match="ownership changed"):
        authorize_native_execution(
            replace(resident, layers=(layer, *resident.layers[1:])),
            backend="hip_gfx1100", rows=2,
        )


@pytest.mark.parametrize("routes,native", [(("eager",), False), (("native_rows", "native_graph"), True)])
def test_real_session_forwards_route_scope_to_real_loader(monkeypatch, tmp_path, routes, native):
    from hipengine.runtime import qwen35_gguf_runner as runner
    from hipengine.loading.qwen35_gguf_execution import execution_operations
    path = tmp_path / "session.gguf"
    write_qwen35_gguf(path, default_fixture_tensors(1), fixture_metadata(1))
    monkeypatch.setattr(runner, "resolve_backend", lambda backend: "hip_gfx1100")
    monkeypatch.setattr(runner, "load_backend_kernel_package", lambda *_: None)
    monkeypatch.setattr(runner, "resolve", lambda **_: object())
    monkeypatch.setattr(runner, "backend_package_capability", lambda backend, name, default=None: default)
    calls = []
    actual = runner.materialize_qwen35_gguf_weights
    def load(*args, **kwargs):
        calls.append(kwargs["requested_operations"])
        return actual(*args, **kwargs)
    monkeypatch.setattr(runner, "materialize_qwen35_gguf_weights", load)
    class PayloadReached(RuntimeError):
        pass
    def payload(*args):
        raise PayloadReached("eager reached payload")
    monkeypatch.setattr(GGUFReader, "tensor_data", payload)
    with pytest.raises(Qwen35GGUFAdmissionError if native else PayloadReached):
        runner.Qwen35GGUFResidentSession(path, runtime=object(), backend="hip_gfx1100",
                                        max_batch_size=2, execution_routes=routes)
    assert calls == [execution_operations(routes)]


@pytest.mark.parametrize("kwargs", [{"max_batch_size": 9}, {"max_batch_size": 2, "use_expert_sidecar": True},
                                  {"max_batch_size": 2, "token_embedding_placement": "host"}])
def test_known_unsupported_session_modes_refuse_before_device_or_loader(monkeypatch, kwargs):
    from hipengine.runtime import qwen35_gguf_runner as runner
    monkeypatch.setattr(runner, "get_hip_runtime", lambda: pytest.fail("device access before route refusal"))
    with pytest.raises(ValueError, match="ar_decode_native_rows"):
        runner.Qwen35GGUFResidentSession("unused.gguf", execution_routes=("native_rows",), **kwargs)


def test_qualified_capture_reaches_stream_only_after_entry_contract(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tests.test_live_gguf_ud_admission import _native_entry_session, _PositionOwnerSentinel
    resident = native_resident(monkeypatch, tmp_path)
    owner = _PositionOwnerSentinel()
    session = _native_entry_session(resident, scratch_owner=owner)
    session._native_compact_scratch = lambda *a, **kw: owner
    class ReachedStream(RuntimeError):
        pass
    def stream():
        raise ReachedStream("certified graph entry")
    session.runtime = SimpleNamespace(stream_create=stream)
    session.runner.runtime = session.runtime
    with pytest.raises(ReachedStream):
        session.capture_native_rows_graph(rows=2, max_context_len=64)
    assert owner.calls == [(0, 0)]


@pytest.mark.parametrize("deferred", [False, True])
def test_real_loader_preserves_deferred_embedding_and_arena_denial(monkeypatch, tmp_path, deferred):
    from tests.test_live_gguf_ud_admission import _cpu_allocation_fakes
    _cpu_allocation_fakes(monkeypatch)
    monkeypatch.setattr(loader.DeviceMemoryArena, "create",
                        lambda *a, **kw: (_ for _ in ()).throw(MemoryError("owner denied")))
    tensors = [(name, shape, Q.BF16 if quant == Q.Q4_K else quant)
               for name, shape, quant in default_fixture_tensors(1, alpha_beta_type=Q.BF16)]
    path = write_qwen35_gguf(tmp_path / "arena.gguf", tensors, fixture_metadata(1))
    resident = loader.materialize_qwen35_gguf_weights(
        path, decode_repack=False, use_selective_weight_arena=True,
        deferred_device_slots=("root.token_embedding",) if deferred else (),
    )
    assert resident.admission_certificate.plan_contract.is_complete()
    assert resident.allocation_mode == "dedicated_selective_arena_denied"
    assert resident.allocation_arena is None
    assert resident.allocation_arena_reason == "owner denied"
    assert bool(resident.root("token_embedding").allocations) != deferred
    assert resident.root("output_norm").allocations


def test_native_moe_full_closure_and_gate_only_certificate_refusal(monkeypatch, tmp_path):
    from hipengine.loading.gguf_selected_contract import SelectedCallIntent
    from hipengine.loading.qwen35_gguf_execution import authorize_native_execution
    from tests.test_live_gguf_ud_admission import _cpu_allocation_fakes
    tensors = [t for t in default_fixture_tensors(1, alpha_beta_type=Q.BF16)
               if not t[0].startswith("blk.0.ffn_")]
    tensors += [("output.weight", (64, 256), Q.Q8_0),
                ("blk.0.ffn_gate_inp.weight", (4, 256), Q.F32),
                ("blk.0.ffn_gate_inp_shexp.weight", (256,), Q.F32)]
    for suffix in ("gate", "up", "down"):
        tensors += [(f"blk.0.ffn_{suffix}_exps.weight", (4, 256, 256), Q.Q4_K),
                    (f"blk.0.ffn_{suffix}_shexp.weight", (256, 256), Q.Q4_K)]
    metadata = [(name.replace("qwen35.", "qwen35moe."), typ,
                 "qwen35moe" if name == "general.architecture" else value)
                for name, typ, value in fixture_metadata(1)]
    metadata += [("qwen35moe." + name, 4, value) for name, value in (
        ("expert_count", 4), ("expert_used_count", 2),
        ("expert_feed_forward_length", 256), ("expert_shared_feed_forward_length", 256))]
    path = write_qwen35_gguf(tmp_path / "moe.gguf", tensors, metadata)
    _cpu_allocation_fakes(monkeypatch)
    resident = loader.materialize_qwen35_gguf_weights(path, decode_repack=False, requested_operations=(NATIVE,))
    contract = resident.admission_certificate.plan_contract
    calls = [i for i in contract.selected_invocations if i.intent.operation == NATIVE]
    assert [i.intent.weight_slots for i in calls] == [
        ("layers.0.ffn_gate_exps", "layers.0.ffn_up_exps"), ("layers.0.ffn_down_exps",)]
    assert any(i.slot == "layers.0.ffn_gate_inp" and i.operation == NATIVE for i in contract.invocations)
    assert authorize_native_execution(resident, backend="hip_gfx1100", rows=2)
    gate = "layers.0.ffn_gate_exps"
    partial = loader.materialize_qwen35_gguf_weights(
        path, decode_repack=False, selected_slots=(gate,), requested_operations=("ar_decode_c1",),
        selected_call_intents=(SelectedCallIntent("ar_decode_c1", "single", (gate,)),))
    assert partial.admission_certificate.plan_contract.is_complete()
    with pytest.raises(ValueError, match="partial certificate"):
        authorize_native_execution(partial, backend="hip_gfx1100", rows=2)
